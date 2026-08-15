use std::{path::Path, sync::Arc};

use crate::pp_log::{PpLog, pp_error, pp_info};
use ffmpeg_next as ffmpeg;
use ndarray::{Array4, Axis, s};
use ort::{inputs, session::Session, value::TensorRef};
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Element, ElementType, Sink, element_pp_log},
    error::Result,
};

/// One detected object, in the pixel space of the frame [`OrtDetector`]
/// was handed — see its own doc comment for why no further rescaling is
/// needed to place this on top of that same frame.
#[derive(Debug, Clone, Copy)]
pub struct Detection {
    /// Index into whatever label set the model was trained on — see
    /// [`COCO_CLASS_LABELS`] for stock Ultralytics YOLOv8/v11 weights.
    pub class_id: usize,
    pub score: f32,
    /// Top-left corner (not center — already converted from the model's
    /// own center/width/height encoding).
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

/// Convenience label table for the 80 COCO classes stock Ultralytics
/// YOLOv8/v11 weights are trained on. Meaningless for a custom-trained
/// model with a different class set — index [`Detection::class_id`] into
/// your own labels in that case instead.
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

/// Errors specific to `OrtDetector`. Converts into the crate-wide
/// `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum OrtDetectorError {
    #[error("onnxruntime error: {0}")]
    Ort(#[from] ort::Error),

    #[error(
        "OrtDetector only accepts RGB24 Video frames, got {0:?}; \
         link it straight after a Scaler configured with Pixel::RGB24"
    )]
    UnsupportedFormat(ffmpeg::format::Pixel),

    #[error(
        "OrtDetector only accepts decoded Video frames, got a {0}; \
         link it straight after a Scaler"
    )]
    UnsupportedBuffer(&'static str),
}

/// Terminal sink that runs a YOLOv8/v11-style ONNX object-detection model
/// (an Ultralytics export: one image input, one `[1, 4 + num_classes,
/// num_boxes]` output, box coordinates as center/width/height) on every
/// incoming frame via `ort`, then hands the decoded, NMS-filtered
/// detections to a plain closure — same "bring your own closure" shape as
/// [`crate::elements::AppSink`], except the closure gets structured
/// [`Detection`]s instead of a raw [`MediaBuffer`].
///
/// Expects every frame's pixel dimensions to already match the model's own
/// input resolution (e.g. 640x640 for stock YOLOv8/11 weights) and its
/// format to be `Pixel::RGB24` — put a [`crate::elements::Scaler`]
/// configured that way directly upstream. Because of that, a detection's
/// box coordinates need no rescaling back to some "original" resolution:
/// they come straight out of the model in the exact same pixel space as
/// the frame handed to the closure.
///
/// Input/output tensors are bound by position, not by name (`images` /
/// `output0` aren't assumed) — whatever the export happens to call its
/// single input and single output, this binds to index `0` of each.
///
/// NMS is per-class (a box only suppresses another box of the *same*
/// `class_id`), matching Ultralytics' own default (non-agnostic) NMS.
pub struct OrtDetector<F> {
    pp_log: PpLog,
    name: Arc<str>,
    session: Session,
    conf_threshold: f32,
    iou_threshold: f32,
    on_detections: F,
}

impl<F> OrtDetector<F>
where
    F: FnMut(&ffmpeg::frame::Video, &[Detection]) -> Result<()> + Send + 'static,
{
    /// `conf_threshold` drops candidate boxes below that class score before
    /// NMS ever sees them; `iou_threshold` is how much two same-class boxes
    /// may overlap before the lower-scoring one is suppressed as a
    /// duplicate of the other.
    pub fn new(
        name: impl Into<String>,
        model_path: impl AsRef<Path>,
        conf_threshold: f32,
        iou_threshold: f32,
        on_detections: F,
    ) -> Result<Self> {
        let model_path_display = model_path.as_ref().display().to_string();
        let session = Session::builder()
            .map_err(OrtDetectorError::from)?
            .commit_from_file(model_path)
            .map_err(OrtDetectorError::from)?;
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::OrtDetector, &name, None);
        pp_info!(
            pp_log: &pp_log,
            "model loaded: path={model_path_display}, conf_threshold={conf_threshold}, iou_threshold={iou_threshold}"
        );
        Ok(Self {
            name,
            pp_log,
            session,
            conf_threshold,
            iou_threshold,
            on_detections,
        })
    }

    /// Builds the `[1, 3, height, width]` normalized input tensor from
    /// `frame`'s packed RGB24 bytes (skipping over `stride`'s per-row
    /// padding, which is usually wider than `width * 3`), runs inference,
    /// then decodes + NMS-filters the raw output into [`Detection`]s.
    fn detect(&mut self, frame: &ffmpeg::frame::Video) -> Result<Vec<Detection>> {
        let width = frame.width() as usize;
        let height = frame.height() as usize;
        let stride = frame.stride(0);
        let data = frame.data(0);

        let mut input = Array4::<f32>::zeros((1, 3, height, width));
        for y in 0..height {
            let row = &data[y * stride..y * stride + width * 3];
            for x in 0..width {
                let pixel = &row[x * 3..x * 3 + 3];
                input[[0, 0, y, x]] = pixel[0] as f32 / 255.0;
                input[[0, 1, y, x]] = pixel[1] as f32 / 255.0;
                input[[0, 2, y, x]] = pixel[2] as f32 / 255.0;
            }
        }

        let outputs = self
            .session
            .run(inputs![
                TensorRef::from_array_view(&input).map_err(OrtDetectorError::from)?
            ])
            .map_err(OrtDetectorError::from)?;
        // `[1, 4 + num_classes, num_boxes]` -> transpose -> `[num_boxes, 4 +
        // num_classes, 1]` -> drop the now-trailing batch axis -> `[num_boxes,
        // 4 + num_classes]`, one row per candidate box.
        let output = outputs[0]
            .try_extract_array::<f32>()
            .map_err(OrtDetectorError::from)?
            .t()
            .into_owned();
        let output = output.slice(s![.., .., 0]);

        let mut candidates = Vec::new();
        for row in output.axis_iter(Axis(0)) {
            let (class_id, score) = row
                .iter()
                // first 4 columns are the box, not a class score
                .skip(4)
                .enumerate()
                .map(|(index, value)| (index, *value))
                .reduce(|best, next| if next.1 > best.1 { next } else { best })
                .expect("model output has at least one class column");
            if score < self.conf_threshold {
                continue;
            }
            let (cx, cy, w, h) = (row[0usize], row[1usize], row[2usize], row[3usize]);
            candidates.push(Detection {
                class_id,
                score,
                x: cx - w / 2.0,
                y: cy - h / 2.0,
                width: w,
                height: h,
            });
        }

        Ok(non_max_suppression(candidates, self.iou_threshold))
    }
}

fn iou(a: &Detection, b: &Detection) -> f32 {
    let (ax2, ay2) = (a.x + a.width, a.y + a.height);
    let (bx2, by2) = (b.x + b.width, b.y + b.height);
    let overlap_w = (ax2.min(bx2) - a.x.max(b.x)).max(0.0);
    let overlap_h = (ay2.min(by2) - a.y.max(b.y)).max(0.0);
    let intersection = overlap_w * overlap_h;
    let union = a.width * a.height + b.width * b.height - intersection;
    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

/// Highest score first, then greedily keeps each box that doesn't overlap
/// (past `iou_threshold`) an already-kept box of the same `class_id`.
fn non_max_suppression(mut candidates: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));

    let mut kept: Vec<Detection> = Vec::with_capacity(candidates.len());
    'candidates: for candidate in candidates {
        for already_kept in &kept {
            if already_kept.class_id == candidate.class_id
                && iou(already_kept, &candidate) > iou_threshold
            {
                continue 'candidates;
            }
        }
        kept.push(candidate);
    }
    kept
}

impl<F> Element for OrtDetector<F>
where
    F: FnMut(&ffmpeg::frame::Video, &[Detection]) -> Result<()> + Send + 'static,
{
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::OrtDetector
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl<F> Sink for OrtDetector<F>
where
    F: FnMut(&ffmpeg::frame::Video, &[Detection]) -> Result<()> + Send + 'static,
{
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Video(frame) => {
                if frame.format() != ffmpeg::format::Pixel::RGB24 {
                    pp_error!(self, "unsupported pixel format: {:?}", frame.format());
                    return Err(OrtDetectorError::UnsupportedFormat(frame.format()).into());
                }
                let detections = self
                    .detect(&frame)
                    .inspect_err(|error| pp_error!(self, "detect failed: {error}"))?;
                (self.on_detections)(&frame, &detections)
            }
            MediaBuffer::Eos => Ok(()),
            MediaBuffer::Packet(_) => {
                pp_error!(self, "unsupported buffer: Packet");
                Err(OrtDetectorError::UnsupportedBuffer("Packet").into())
            }
            MediaBuffer::Audio(_) => {
                pp_error!(self, "unsupported buffer: Audio");
                Err(OrtDetectorError::UnsupportedBuffer("Audio").into())
            }
        }
    }

    fn control(&mut self, _msg: ControlMsg) -> Result<()> {
        // Terminal, same as AppSink/D3d12Renderer: nothing buffered or
        // downstream to flush/forward for any ControlMsg.
        Ok(())
    }
}
