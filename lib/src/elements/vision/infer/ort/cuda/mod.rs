//! [`CudaOrtDetector`]: detection on CUDA pictures, on CUDA or through
//! TensorRT.

mod cuda_ort_classifier;
mod cuda_ort_detector;
mod runtime;

pub use cuda_ort_classifier::CudaOrtClassifier;
#[cfg(feature = "ort-tensorrt")]
pub use cuda_ort_detector::UseTensorRtPolicy;
pub use cuda_ort_detector::{CudaOrtDetector, CudaOrtDetectorOptions};
pub use runtime::{CudaRuntime, LibraryVersion, RuntimeShortfall};
