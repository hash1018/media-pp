//! [`CudaOrtDetector`]: detection on CUDA pictures, through TensorRT.

mod cuda_ort_detector;
mod runtime;

pub use cuda_ort_detector::{CudaOrtDetector, CudaOrtDetectorOptions, UseTensorRtPolicy};
pub use runtime::{CudaRuntime, LibraryVersion, RuntimeShortfall};
