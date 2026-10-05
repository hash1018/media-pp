//! Object detection with ONNX Runtime, as filters that hand each picture on
//! with what it found as [`Detections`](crate::elements::Detections) on it.
//!
//! One element per place a picture lives, as the scalers are:
//! [`SwOrtDetector`] reads pictures in system memory and infers on the CPU,
//! `CudaOrtDetector` reads CUDA pictures and infers on CUDA or through
//! TensorRT, and `MetalOrtDetector` reads VideoToolbox pictures and infers
//! through Core ML. What they share is here: their options and errors, how a
//! model's class names are read, and in `yolo` how its output is read and a
//! picture fitted to its input.

use std::sync::Arc;

use thiserror::Error as ThisError;

use crate::ffmpeg;

mod classify;
#[cfg(feature = "ort-cuda")]
mod cuda;
#[cfg(all(target_os = "macos", feature = "ort-coreml"))]
mod metal;
mod sw_ort_classifier;
mod sw_ort_detector;
mod yolo;

pub use classify::{InputScale, OrtClassifierOptions};
#[cfg(feature = "ort-tensorrt")]
pub use cuda::UseTensorRtPolicy;
#[cfg(feature = "ort-cuda")]
pub use cuda::{
    CudaOrtClassifier, CudaOrtDetector, CudaOrtDetectorOptions, CudaRuntime, LibraryVersion,
    RuntimeShortfall,
};
#[cfg(all(target_os = "macos", feature = "ort-coreml"))]
pub use metal::MetalOrtDetector;
pub use sw_ort_classifier::SwOrtClassifier;
pub use sw_ort_detector::SwOrtDetector;
use yolo::{Letterbox, decode};

/// How a detector decides what counts as found, and what it calls it.
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
    /// [`Detections::labels`](crate::elements::Detections::labels) empty, and its detections only their class
    /// numbers.
    /// [`COCO_CLASS_LABELS`](crate::elements::COCO_CLASS_LABELS) are stock
    /// Ultralytics weights' classes.
    pub labels: Option<Vec<String>>,
    /// How many pictures to let by unlooked-at between two it looks at: 0
    /// looks at every one, 2 at every third. A picture let by carries no
    /// [`Detections`](crate::elements::Detections) — which says it was not
    /// looked at, where an empty list says nothing was found — and an
    /// [`ObjectTracker`](crate::elements::ObjectTracker) after the detector
    /// puts where it expects the objects to be on it instead: DeepStream's
    /// `interval`, for a model too slow to look at every picture.
    pub interval: u32,
}

impl Default for OrtDetectorOptions {
    /// Ultralytics' own thresholds, the model's own labels, and every
    /// picture looked at.
    fn default() -> Self {
        Self {
            conf_threshold: 0.25,
            iou_threshold: 0.45,
            labels: None,
            interval: 0,
        }
    }
}

/// Which pictures a detector looks at, by [`OrtDetectorOptions::interval`]:
/// the first, then every `interval + 1`th after it.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Interval {
    interval: u32,
    /// Pictures let by since the last one looked at; `None` before the
    /// first.
    since: Option<u32>,
}

impl Interval {
    pub(crate) fn new(interval: u32) -> Self {
        Self {
            interval,
            since: None,
        }
    }

    /// Whether to look at the picture in hand.
    pub(crate) fn look(&mut self) -> bool {
        match self.since {
            Some(since) if since < self.interval => {
                self.since = Some(since + 1);
                false
            }
            _ => {
                self.since = Some(0);
                true
            }
        }
    }

    /// Starts over, looking at the next picture: after a seek or a flush,
    /// the first picture of what follows is the one to look at.
    pub(crate) fn restart(&mut self) {
        self.since = None;
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

/// Why an ONNX Runtime element — a detector, a classifier — could not be
/// made or could not look at a picture.
#[derive(Debug, ThisError)]
#[non_exhaustive]
pub enum OrtError {
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
    /// The driver is missing, or it, the CUDA runtime or cuDNN is too old
    /// for the CUDA provider.
    #[cfg(feature = "ort-cuda")]
    #[error("the CUDA runtime cannot be used: {0}")]
    CudaRuntimeUnavailable(RuntimeShortfall),
    /// TensorRT was required and is too old, so the detector would have
    /// run on CUDA alone.
    #[cfg(feature = "ort-tensorrt")]
    #[error("TensorRT is required and cannot be used: {0}")]
    TensorRtUnavailable(RuntimeShortfall),
    /// Metal refused a kernel, a buffer or a pass.
    #[cfg(all(target_os = "macos", feature = "ort-coreml"))]
    #[error(transparent)]
    Metal(#[from] crate::platform::macos::metal::MetalError),
    /// Not an NV12 or BGRA VideoToolbox picture: what it is, or for a
    /// VideoToolbox picture what it holds.
    #[cfg(all(target_os = "macos", feature = "ort-coreml"))]
    #[error("MetalOrtDetector takes NV12 or BGRA VideoToolbox pictures, got {0:?}")]
    UnsupportedPicture(ffmpeg::format::Pixel),
    /// A VideoToolbox picture with no frames context to say what it holds,
    /// or no pixel buffer in it.
    #[cfg(all(target_os = "macos", feature = "ort-coreml"))]
    #[error("a VideoToolbox picture has no {0}")]
    MissingPixelBuffer(&'static str),
    /// A VideoToolbox picture larger than the pixel buffer behind it.
    #[cfg(all(target_os = "macos", feature = "ort-coreml"))]
    #[error("a {picture:?} picture is larger than its {surface:?} pixel buffer")]
    PictureOutsideSurface {
        /// The picture's size.
        picture: (u32, u32),
        /// The pixel buffer's.
        surface: (u32, u32),
    },
}

/// [`OrtError`]'s name from when the detectors were its only elements.
pub type OrtDetectorError = OrtError;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_picture_is_looked_at_then_every_interval_plus_oneth() {
        let mut every_third = Interval::new(2);
        let looked: Vec<bool> = (0..7).map(|_| every_third.look()).collect();
        assert_eq!(looked, [true, false, false, true, false, false, true]);
        every_third.restart();
        assert!(every_third.look(), "after a restart, the next one");
        let mut every = Interval::new(0);
        assert!((0..4).all(|_| every.look()));
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
}
