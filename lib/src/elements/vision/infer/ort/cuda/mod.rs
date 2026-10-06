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

/// Where TensorRT keeps the engines it builds, unless told otherwise:
/// `$XDG_CACHE_HOME/media-pp/tensorrt`, or the platform's own cache
/// directory, or the temporary one where there is neither. ONNX Runtime
/// names an engine after its model's graph, so a detector's and a
/// classifier's sit side by side.
#[cfg(feature = "ort-tensorrt")]
fn default_engine_cache() -> std::path::PathBuf {
    use std::path::PathBuf;
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("LOCALAPPDATA").map(PathBuf::from))
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
        .unwrap_or_else(std::env::temp_dir);
    base.join("media-pp").join("tensorrt")
}
