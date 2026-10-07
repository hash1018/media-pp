//! Models that look at pictures and say what they found, as filters that
//! hand each picture on carrying it — DeepStream's `nvinfer`. One directory
//! per runtime: [`ort`], ONNX Runtime's.

mod ort;

#[cfg(feature = "ort-tensorrt")]
pub use self::ort::UseTensorRtPolicy;
pub use self::ort::{
    ChannelOrder, DetectorDecoder, DetectorModel, InputScale, ModelBox, ModelInput, ModelOutput,
    OrtClassifierOptions, OrtDetectorError, OrtDetectorOptions, OrtError, SwOrtClassifier,
    SwOrtDetector, non_max_suppression,
};
#[cfg(all(target_os = "macos", feature = "ort-coreml"))]
pub use self::ort::{
    CoreMlComputeUnits, MetalOrtClassifier, MetalOrtDetector, MetalOrtDetectorOptions,
};
#[cfg(feature = "ort-cuda")]
pub use self::ort::{
    CudaOrtClassifier, CudaOrtDetector, CudaOrtDetectorOptions, CudaRuntime, LibraryVersion,
    RuntimeShortfall,
};
