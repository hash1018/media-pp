//! Video analysis — what DeepStream does for NVIDIA pipelines, as elements
//! of this crate's own: finding objects in pictures, and doing something
//! with what was found.
//!
//! What every one of them shares is [`Detections`],
//! the [`Metadata`](crate::buffer::Metadata) a detector puts on each
//! picture and every later element reads. Then a family of elements per
//! stage, each with its software and GPU members:
//!
//! - `infer` — models that look at pictures: ONNX Runtime's, with the
//!   `ort` features, as `SwOrtDetector`, `CudaOrtDetector` and
//!   `MetalOrtDetector`.
//! - `cut` — where one shot of an edited video ends and the next begins:
//!   [`SwCutDetector`] and `CudaCutDetector`, which put a [`SceneCut`] on
//!   the first picture of each shot.
//! - `track` — following what was found from picture to picture:
//!   [`ObjectTracker`], which numbers each object and fills in the pictures
//!   a detector let by.
//! - `analytics` — making sense of it: [`ObjectAnalytics`], which counts
//!   the objects in zones of the picture and across lines.
//! - `overlay` — drawing what was found onto the pictures:
//!   [`SwDetectionOverlay`], `CudaDetectionOverlay` and
//!   `MetalDetectionOverlay`.
//!
//! The data and the overlays need no feature of their own; only inference
//! brings a runtime with it.

mod analytics;
mod batch;
mod cut;
#[cfg(feature = "ort")]
mod infer;
mod meta;
mod overlay;
mod track;

pub use analytics::{
    AnalyticsOptions, Line, ObjectAnalytics, ObjectAnalyticsError, StreamAnalyticsOptions, Zone,
};
pub use batch::{
    BatchSlot, StreamDemuxHandle, StreamId, StreamMux, StreamMuxError, StreamMuxHandle,
    StreamMuxInput, StreamMuxOptions, StreamOrigin,
};
#[cfg(feature = "cuda")]
pub use cut::{CudaCutDetector, CudaCutDetectorError};
pub use cut::{CutDetectorOptions, CutDetectorOptionsError, SwCutDetector, SwCutDetectorError};
#[cfg(all(target_os = "macos", feature = "metal"))]
pub use cut::{MetalCutDetector, MetalCutDetectorError};
#[cfg(feature = "ort")]
pub use infer::*;
pub use meta::{
    Analytics, COCO_CLASS_LABELS, Classification, Crossing, Cutout, Detection, Detections,
    Embedding, LineCount, SceneCut, ZoneCount,
};
pub use overlay::{
    BoxColors, BoxStyle, ClassId, ClassRule, DetectionOverlayOptions, DetectionOverlayOptionsError,
    HideShape, Hiding, LabelStyle, OverlayParts, RedactStyle, SwDetectionOverlay,
    SwDetectionOverlayError, Treatment,
};
#[cfg(feature = "cuda")]
pub use overlay::{CudaDetectionOverlay, CudaDetectionOverlayError};
#[cfg(all(target_os = "macos", feature = "metal"))]
pub use overlay::{MetalDetectionOverlay, MetalDetectionOverlayError};
pub use track::{ObjectTracker, TrackerOptions};

/// Whether `format` is a hardware frame's, whose pixels are not in it.
pub(crate) fn is_hardware(format: crate::ffmpeg::format::Pixel) -> bool {
    // SAFETY: a lookup in libavutil's static table of descriptors.
    unsafe {
        let descriptor = crate::ffmpeg::ffi::av_pix_fmt_desc_get(format.into());
        !descriptor.is_null()
            && (*descriptor).flags & (crate::ffmpeg::ffi::AV_PIX_FMT_FLAG_HWACCEL as u64) != 0
    }
}
