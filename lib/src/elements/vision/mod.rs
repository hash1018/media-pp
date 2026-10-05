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
#[cfg(feature = "ort")]
mod infer;
mod meta;
mod overlay;
mod track;

pub use analytics::{AnalyticsOptions, Line, ObjectAnalytics, ObjectAnalyticsError, Zone};
pub use batch::{
    BatchSlot, StreamId, StreamMux, StreamMuxError, StreamMuxHandle, StreamMuxInput,
    StreamMuxOptions, StreamOrigin,
};
#[cfg(feature = "ort")]
pub use infer::*;
pub use meta::{
    Analytics, COCO_CLASS_LABELS, Classification, Crossing, Detection, Detections, LineCount,
    ZoneCount,
};
pub use overlay::{
    BoxColors, DetectionOverlayOptions, LabelStyle, SwDetectionOverlay, SwDetectionOverlayError,
};
#[cfg(feature = "cuda")]
pub use overlay::{CudaDetectionOverlay, CudaDetectionOverlayError};
#[cfg(all(target_os = "macos", feature = "metal"))]
pub use overlay::{MetalDetectionOverlay, MetalDetectionOverlayError};
pub use track::{ObjectTracker, TrackerOptions};
