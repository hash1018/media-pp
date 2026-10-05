//! Models that look at pictures and say what they found, as filters that
//! hand each picture on carrying it — DeepStream's `nvinfer`. One directory
//! per runtime: [`ort`], ONNX Runtime's.

mod ort;

#[cfg(feature = "ort-tensorrt")]
pub use self::ort::UseTensorRtPolicy;
#[cfg(feature = "ort-cuda")]
pub use self::ort::{
    CudaOrtDetector, CudaOrtDetectorOptions, CudaRuntime, LibraryVersion, RuntimeShortfall,
};
pub use self::ort::{OrtDetectorError, OrtDetectorOptions, SwOrtDetector};
