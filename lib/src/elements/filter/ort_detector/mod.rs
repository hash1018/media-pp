//! Object detection with ONNX Runtime, as filters that hand each picture on
//! with what was found in it.
//!
//! One element per place a picture lives, as the scalers are:
//! [`SwOrtDetector`] reads pictures in system memory and infers on the CPU,
//! and `CudaOrtDetector` reads CUDA pictures and infers through TensorRT.
//! What they share is here: the [`Detections`] they put on each picture as
//! its [`Metadata`](crate::buffer::Metadata), how a model's output is read
//! into them, and how a picture is fitted to a model's input.
//!
//! The models they read are described on [`Detections`], since this
//! module is not public.

use std::sync::Arc;

use ndarray::{ArrayViewD, Axis};
use thiserror::Error as ThisError;

use crate::buffer::MediaBuffer;
use crate::ffmpeg;

#[cfg(feature = "ort-cuda")]
mod cuda;
mod sw_ort_detector;

#[cfg(feature = "ort-cuda")]
pub use cuda::{
    CudaOrtDetector, CudaOrtDetectorOptions, CudaRuntime, LibraryVersion, RuntimeShortfall,
    UseTensorRtPolicy,
};
pub use sw_ort_detector::SwOrtDetector;

/// One object a detector found, placed as fractions of the picture it is
/// attached to: `x` and `y` are its top-left corner, `width` and `height`
/// its size, each from 0 to 1. Multiply by the picture's own width and
/// height for pixels — which holds whatever the picture is scaled to
/// afterwards.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    /// Index into the label set the model was trained on — see
    /// [`COCO_CLASS_LABELS`] for stock Ultralytics weights.
    pub class_id: usize,
    /// The model's confidence, after thresholding.
    pub score: f32,
    /// Left edge, as a fraction of the picture's width.
    pub x: f32,
    /// Top edge, as a fraction of the picture's height.
    pub y: f32,
    /// Width, as a fraction of the picture's width.
    pub width: f32,
    /// Height, as a fraction of the picture's height.
    pub height: f32,
}

/// What a detector found in one picture: the
/// [`Metadata`](crate::buffer::Metadata) it puts on the picture it hands on.
/// Read it downstream with `buffer.metadata()?.get::<Detections>()`.
///
/// An empty list is an answer — the detector looked and found nothing —
/// where a picture carrying no `Detections` was not looked at.
///
/// # Models
///
/// A YOLO detector exported to ONNX with one image input, `[1, 3, height,
/// width]`, RGB scaled to 0–1, and one output in either of the two layouts
/// Ultralytics exports:
///
/// - `[1, 4 + classes, boxes]` — YOLOv8 and YOLO11: a box's centre, width
///   and height, then a score per class. Boxes are thresholded and put
///   through non-maximum suppression by the detector.
/// - `[1, boxes, 6]` — YOLOv10 and YOLO26, which suppress duplicates
///   themselves: each row a box's corners, its score and its class.
///
/// A picture is fitted to the model's input the way Ultralytics trains it:
/// scaled to fit inside, keeping its proportions, and the rest filled with
/// grey (114). The boxes are mapped back, so they describe the picture the
/// detector was handed, whatever its size.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Detections {
    /// The name of the element that looked.
    pub detector: Arc<str>,
    /// The model's class names, indexed by [`Detection::class_id`] — shared
    /// by every picture the detector looks at, not copied onto each.
    pub labels: Arc<[Arc<str>]>,
    /// What it found, most confident first.
    pub items: Vec<Detection>,
}

impl Detections {
    /// What `detection`'s class is called, where the model or the caller
    /// named its classes.
    pub fn label(&self, detection: &Detection) -> Option<&str> {
        self.labels.get(detection.class_id).map(|label| &**label)
    }
}

/// The 80 COCO classes stock Ultralytics weights are trained on, in their
/// order. Meaningless for a model trained on another set: index
/// [`Detection::class_id`] into its own labels instead.
#[rustfmt::skip]
pub const COCO_CLASS_LABELS: [&str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
];

/// How a detector decides what counts as found, and what it calls it.
#[derive(Debug, Clone, PartialEq)]
pub struct OrtDetectorOptions {
    /// The lowest score a box is kept at.
    pub conf_threshold: f32,
    /// How much two boxes of one class may overlap, as intersection over
    /// union, before the less confident is dropped as the same object. Used
    /// only for the `[1, 4 + classes, boxes]` layout; the other suppresses
    /// duplicates itself.
    pub iou_threshold: f32,
    /// The model's class names, in its class order. `None` reads them from
    /// the model, which an Ultralytics export carries in its `names`
    /// metadata; a model that names none — or names each class only by its
    /// own number, as an export that lost the real names does — leaves
    /// [`Detections::labels`] empty, and its detections only their class
    /// numbers.
    /// [`COCO_CLASS_LABELS`] are stock Ultralytics weights' classes.
    pub labels: Option<Vec<String>>,
}

impl Default for OrtDetectorOptions {
    /// Ultralytics' own thresholds, and the model's own labels.
    fn default() -> Self {
        Self {
            conf_threshold: 0.25,
            iou_threshold: 0.45,
            labels: None,
        }
    }
}

/// The class names a detector puts on what it finds: `given`, or else what
/// the model's `names` metadata says, or else none.
pub(crate) fn labels(given: Option<&[String]>, session: &ort::session::Session) -> Arc<[Arc<str>]> {
    if let Some(given) = given {
        return given
            .iter()
            .map(|label| Arc::from(label.as_str()))
            .collect();
    }
    let names = session
        .metadata()
        .ok()
        .and_then(|metadata| metadata.custom("names"))
        .map(|names| parse_names(&names))
        .unwrap_or_default();
    named(names)
}

/// `names` as labels — none where each is only its own number, which names
/// nothing: an export that lost the real ones writes `{0: '0', 1: '1',
/// ...}`.
fn named(names: Vec<String>) -> Arc<[Arc<str>]> {
    if names
        .iter()
        .enumerate()
        .all(|(index, name)| *name == index.to_string())
    {
        return Arc::from([]);
    }
    names.into_iter().map(Arc::from).collect()
}

/// Reads Ultralytics' `names` metadata — a Python dict, `{0: 'person', 1:
/// 'bicycle', ...}` — into a list by class number. A number left out is an
/// empty name, so the rest keep their places.
fn parse_names(names: &str) -> Vec<String> {
    let mut found: Vec<(usize, String)> = Vec::new();
    let mut rest = names.trim().trim_start_matches('{').trim_end_matches('}');
    while let Some(colon) = rest.find(':') {
        let Ok(index) = rest[..colon]
            .trim()
            .trim_start_matches(',')
            .trim()
            .parse::<usize>()
        else {
            break;
        };
        let value = rest[colon + 1..].trim_start();
        let Some(quote) = value.chars().next().filter(|c| *c == '\'' || *c == '"') else {
            break;
        };
        let Some(end) = value[1..].find(quote) else {
            break;
        };
        found.push((index, value[1..1 + end].to_owned()));
        rest = &value[end + 2..];
    }
    let mut labels =
        vec![String::new(); found.iter().map(|(index, _)| index + 1).max().unwrap_or(0)];
    for (index, name) in found {
        labels[index] = name;
    }
    labels
}

/// Why a detector could not be made or could not look at a picture.
#[derive(Debug, ThisError)]
#[non_exhaustive]
pub enum OrtDetectorError {
    /// ONNX Runtime refused the model, the execution provider, or a run.
    #[error("onnxruntime error: {0}")]
    Ort(#[from] ort::Error),
    /// The model's input or output is not a shape this reads.
    #[error("unsupported model: {0}")]
    UnsupportedModel(String),
    /// Not a decoded picture this detector reads.
    #[error("{detector} takes {wanted}, got {got}")]
    UnsupportedBuffer {
        /// Which detector refused it.
        detector: &'static str,
        /// What it takes.
        wanted: &'static str,
        /// What it was handed.
        got: &'static str,
    },
    /// Fitting a picture to the model's input failed.
    #[error("could not convert a picture for the model: {0}")]
    Convert(#[from] ffmpeg::Error),
    /// The CUDA driver refused a kernel, an allocation or a wait.
    #[cfg(feature = "ort-cuda")]
    #[error(transparent)]
    Cuda(#[from] crate::platform::cuda::CudaDriverError),
    /// A picture is not a CUDA surface this detector reads.
    #[cfg(feature = "ort-cuda")]
    #[error(transparent)]
    CudaFrame(#[from] crate::platform::cuda::CudaFrameError),
    /// A CUDA picture with no device pointer in it.
    #[cfg(feature = "ort-cuda")]
    #[error("a CUDA picture has no surface")]
    MissingSurface,
    /// The driver, the CUDA runtime or cuDNN that the CUDA provider needs
    /// is missing or too old: CUDA 13 with cuBLAS and cuRAND, and cuDNN 9.
    #[cfg(feature = "ort-cuda")]
    #[error("the CUDA runtime cannot be used: {0}")]
    CudaRuntimeUnavailable(RuntimeShortfall),
    /// TensorRT was required and is missing or too old, so the detector
    /// would have run on CUDA alone.
    #[cfg(feature = "ort-cuda")]
    #[error("TensorRT is required and cannot be used: {0}")]
    TensorRtUnavailable(RuntimeShortfall),
}

/// The model's input size, from its first input's `[1, 3, height, width]`;
/// 640 for a side the model leaves open.
pub(crate) fn model_input(session: &ort::session::Session) -> Result<(u32, u32), OrtDetectorError> {
    let input = session
        .inputs()
        .first()
        .ok_or_else(|| OrtDetectorError::UnsupportedModel("the model has no input".into()))?;
    let shape = input
        .dtype()
        .tensor_shape()
        .ok_or_else(|| OrtDetectorError::UnsupportedModel("its input is not a tensor".into()))?;
    let side = |value: i64| if value > 0 { value as u32 } else { 640 };
    match shape.iter().as_slice() {
        [_, 3, height, width] => Ok((side(*width), side(*height))),
        other => Err(OrtDetectorError::UnsupportedModel(format!(
            "its input is {other:?}, not [1, 3, height, width]"
        ))),
    }
}

/// How a picture is fitted inside the model's input: scaled to fit,
/// proportions kept, centred, the rest grey.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Letterbox {
    /// The picture's size.
    pub(crate) frame: (u32, u32),
    /// The model's input size.
    pub(crate) model: (u32, u32),
    /// The size the picture is scaled to inside the input.
    pub(crate) scaled: (u32, u32),
    /// Where the scaled picture's top-left corner sits in the input.
    pub(crate) offset: (u32, u32),
}

impl Letterbox {
    pub(crate) fn new(frame: (u32, u32), model: (u32, u32)) -> Self {
        let scale =
            (model.0 as f32 / frame.0.max(1) as f32).min(model.1 as f32 / frame.1.max(1) as f32);
        let scaled = (
            ((frame.0 as f32 * scale).round() as u32).clamp(1, model.0),
            ((frame.1 as f32 * scale).round() as u32).clamp(1, model.1),
        );
        Self {
            frame,
            model,
            scaled,
            offset: ((model.0 - scaled.0) / 2, (model.1 - scaled.1) / 2),
        }
    }

    /// A point of the model's input, as fractions of the picture, clamped
    /// to it.
    fn onto_frame(&self, x: f32, y: f32) -> (f32, f32) {
        (
            ((x - self.offset.0 as f32) / self.scaled.0 as f32).clamp(0.0, 1.0),
            ((y - self.offset.1 as f32) / self.scaled.1 as f32).clamp(0.0, 1.0),
        )
    }

    /// A box of the model's input by its corners, as a [`Detection`] on the
    /// picture.
    fn detection(&self, class_id: usize, score: f32, corners: [f32; 4]) -> Detection {
        let (x1, y1) = self.onto_frame(corners[0], corners[1]);
        let (x2, y2) = self.onto_frame(corners[2], corners[3]);
        Detection {
            class_id,
            score,
            x: x1,
            y: y1,
            width: x2 - x1,
            height: y2 - y1,
        }
    }
}

/// Reads a model's output into what it found in the picture `letterbox`
/// fitted, most confident first — see the module docs for the two layouts.
pub(crate) fn decode(
    output: ArrayViewD<'_, f32>,
    letterbox: &Letterbox,
    options: &OrtDetectorOptions,
) -> Result<Vec<Detection>, OrtDetectorError> {
    let shape = output.shape().to_vec();
    let output = match shape.as_slice() {
        [1, _, _] => output.index_axis(Axis(0), 0),
        other => {
            return Err(OrtDetectorError::UnsupportedModel(format!(
                "its output is {other:?}, not [1, rows, columns]"
            )));
        }
    };
    let mut found: Vec<Detection> = if shape[2] == 6 {
        // `[boxes, 6]`: corners, score, class — already suppressed.
        output
            .axis_iter(Axis(0))
            .filter(|row| row[4] >= options.conf_threshold)
            .map(|row| {
                letterbox.detection(
                    row[5].max(0.0) as usize,
                    row[4],
                    [row[0], row[1], row[2], row[3]],
                )
            })
            .collect()
    } else if shape[1] > 4 {
        // `[4 + classes, boxes]`: a column per box.
        let mut candidates = Vec::new();
        for column in output.axis_iter(Axis(1)) {
            let (class_id, score) = column
                .iter()
                .skip(4)
                .copied()
                .enumerate()
                .reduce(|best, next| if next.1 > best.1 { next } else { best })
                .expect("more than four rows");
            if score < options.conf_threshold {
                continue;
            }
            let (cx, cy, w, h) = (column[0], column[1], column[2], column[3]);
            candidates.push((
                class_id,
                score,
                [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
            ));
        }
        non_max_suppression(candidates, options.iou_threshold)
            .into_iter()
            .map(|(class_id, score, corners)| letterbox.detection(class_id, score, corners))
            .collect()
    } else {
        return Err(OrtDetectorError::UnsupportedModel(format!(
            "its output is {shape:?}, neither [1, 4 + classes, boxes] nor [1, boxes, 6]"
        )));
    };
    found.sort_by(|a, b| b.score.total_cmp(&a.score));
    Ok(found)
}

/// One candidate: class, score, corners in the model's input.
type Candidate = (usize, f32, [f32; 4]);

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let overlap_w = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let overlap_h = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let intersection = overlap_w * overlap_h;
    let union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection;
    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

/// Highest score first, keeping each box that overlaps no kept box of its
/// own class past `iou_threshold` — Ultralytics' default, per-class NMS.
fn non_max_suppression(mut candidates: Vec<Candidate>, iou_threshold: f32) -> Vec<Candidate> {
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut kept: Vec<Candidate> = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let duplicate = kept
            .iter()
            .any(|kept| kept.0 == candidate.0 && iou(&kept.2, &candidate.2) > iou_threshold);
        if !duplicate {
            kept.push(candidate);
        }
    }
    kept
}

/// `buf`, carrying `detections` beside whatever it carried already.
pub(crate) fn attach(buf: MediaBuffer, detections: Detections) -> MediaBuffer {
    let metadata = buf.metadata().cloned().unwrap_or_default().with(detections);
    buf.with_metadata(metadata)
}

/// The metadata `buf` would carry once `detections` were attached — for a
/// test to compare against.
#[cfg(test)]
pub(crate) fn detections_of(buf: &MediaBuffer) -> Option<&Detections> {
    buf.metadata()?.get::<Detections>()
}

#[cfg(test)]
mod tests {
    use ndarray::Array3;

    use super::*;
    use crate::buffer::Metadata;

    #[test]
    fn a_wide_picture_is_fitted_with_grey_above_and_below() {
        let letterbox = Letterbox::new((1920, 1080), (640, 640));
        assert_eq!(letterbox.scaled, (640, 360));
        assert_eq!(letterbox.offset, (0, 140));
    }

    /// A box drawn on the fitted input lands on the same place of the
    /// picture, as fractions of it.
    #[test]
    fn an_end_to_end_row_is_mapped_back_onto_the_picture() {
        let letterbox = Letterbox::new((1920, 1080), (640, 640));
        // The right half of the picture, top to bottom, in input pixels.
        let mut output = Array3::<f32>::zeros((1, 2, 6));
        output
            .slice_mut(ndarray::s![0, 0, ..])
            .assign(&ndarray::arr1(&[320.0, 140.0, 640.0, 500.0, 0.9, 2.0]));
        output
            .slice_mut(ndarray::s![0, 1, ..])
            .assign(&ndarray::arr1(&[0.0, 0.0, 10.0, 10.0, 0.1, 0.0]));
        let found = decode(
            output.view().into_dyn(),
            &letterbox,
            &OrtDetectorOptions::default(),
        )
        .unwrap();
        assert_eq!(found.len(), 1, "the 0.1 row is below the threshold");
        let car = found[0];
        assert_eq!(car.class_id, 2);
        assert!((car.x - 0.5).abs() < 1e-6 && car.y.abs() < 1e-6, "{car:?}");
        assert!(
            (car.width - 0.5).abs() < 1e-6 && (car.height - 1.0).abs() < 1e-6,
            "{car:?}"
        );
    }

    /// The v8 layout: two overlapping boxes of one class are one object,
    /// and a box of another class beside them is not suppressed by them.
    #[test]
    fn overlapping_boxes_of_one_class_are_one_object() {
        let letterbox = Letterbox::new((640, 640), (640, 640));
        // Rows: cx, cy, w, h, then a score for each of two classes.
        let columns = [
            [100.0, 100.0, 50.0, 50.0, 0.9, 0.0],
            [102.0, 101.0, 50.0, 50.0, 0.8, 0.0],
            [101.0, 100.0, 50.0, 50.0, 0.0, 0.7],
        ];
        let mut output = Array3::<f32>::zeros((1, 6, columns.len()));
        for (box_index, column) in columns.iter().enumerate() {
            for (row, value) in column.iter().enumerate() {
                output[[0, row, box_index]] = *value;
            }
        }
        let found = decode(
            output.view().into_dyn(),
            &letterbox,
            &OrtDetectorOptions::default(),
        )
        .unwrap();
        let classes: Vec<_> = found
            .iter()
            .map(|found| (found.class_id, found.score))
            .collect();
        assert_eq!(classes, vec![(0, 0.9), (1, 0.7)]);
    }

    #[test]
    fn ultralytics_names_are_read_by_class_number() {
        assert_eq!(
            parse_names("{0: 'person', 1: \"bicycle\", 3: 'motor, cycle'}"),
            vec!["person", "bicycle", "", "motor, cycle"]
        );
        assert!(parse_names("not a dict").is_empty());
    }

    #[test]
    fn classes_named_only_by_their_numbers_are_unnamed() {
        assert!(named(parse_names("{0: '0', 1: '1', 2: '2'}")).is_empty());
        assert_eq!(named(vec!["person".into(), "1".into()]).len(), 2);
    }

    #[test]
    fn a_detection_is_named_by_its_class() {
        let detections = Detections {
            detector: "d".into(),
            labels: Arc::from([Arc::from("person"), Arc::from("bicycle")]),
            items: Vec::new(),
        };
        let bicycle = Detection {
            class_id: 1,
            score: 1.0,
            x: 0.0,
            y: 0.0,
            width: 1.0,
            height: 1.0,
        };
        assert_eq!(detections.label(&bicycle), Some("bicycle"));
        let unnamed = Detection {
            class_id: 7,
            ..bicycle
        };
        assert_eq!(detections.label(&unnamed), None);
    }

    #[test]
    fn an_output_of_another_shape_is_refused() {
        let output = Array3::<f32>::zeros((2, 3, 6));
        let letterbox = Letterbox::new((640, 640), (640, 640));
        assert!(
            decode(
                output.view().into_dyn(),
                &letterbox,
                &OrtDetectorOptions::default()
            )
            .is_err()
        );
    }

    #[test]
    fn detections_go_beside_what_a_buffer_carried() {
        #[derive(Debug, PartialEq)]
        struct Earlier;
        let buf = MediaBuffer::video(ffmpeg::frame::Video::new(
            ffmpeg::format::Pixel::RGB24,
            2,
            2,
        ))
        .with_metadata(Metadata::new().with(Earlier));
        let detections = Detections {
            detector: "d".into(),
            labels: Arc::from([Arc::from("person")]),
            items: Vec::new(),
        };
        let buf = attach(buf, detections.clone());
        assert_eq!(detections_of(&buf), Some(&detections));
        assert!(buf.metadata().unwrap().contains::<Earlier>());
    }
}
