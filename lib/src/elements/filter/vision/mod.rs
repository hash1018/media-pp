//! Video analysis — what DeepStream does for NVIDIA pipelines, as elements
//! of this crate's own: finding objects in pictures, and doing something
//! with what was found.
//!
//! What every one of them shares is the data in [`meta`]: [`Detections`],
//! the [`Metadata`](crate::buffer::Metadata) a detector puts on each
//! picture and every later element reads. Then a family of elements per
//! stage, each with its software and GPU members:
//!
//! - [`infer`] — models that look at pictures: ONNX Runtime's, with the
//!   `ort` features, as `SwOrtDetector`, `CudaOrtDetector` and
//!   `MetalOrtDetector`.
//! - [`overlay`] — drawing what was found onto the pictures:
//!   [`SwDetectionOverlay`] and `CudaDetectionOverlay`.
//!
//! The data and the overlays need no feature of their own; only inference
//! brings a runtime with it.

#[cfg(feature = "ort")]
mod infer;
mod meta;
mod overlay;

#[cfg(feature = "ort")]
pub use infer::*;
pub use meta::{COCO_CLASS_LABELS, Detection, Detections};
pub use overlay::{
    BoxColors, DetectionOverlayOptions, LabelStyle, SwDetectionOverlay, SwDetectionOverlayError,
};
#[cfg(feature = "cuda")]
pub use overlay::{CudaDetectionOverlay, CudaDetectionOverlayError};
