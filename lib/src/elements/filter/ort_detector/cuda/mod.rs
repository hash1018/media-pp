//! [`CudaOrtDetector`]: detection on CUDA pictures, through TensorRT.

mod cuda_ort_detector;

pub use cuda_ort_detector::{
    CudaOrtDetector, CudaOrtDetectorOptions, CudaRuntime, UseTensorRtPolicy,
};
