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

/// The batches a TensorRT engine is built for: the fewest, the one it is
/// fastest at, and the most.
#[cfg(feature = "ort-tensorrt")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BatchProfile {
    min: usize,
    opt: usize,
    max: usize,
}

/// Where in the cache directory `root` the engine for `profile` is kept.
///
/// ONNX Runtime names an engine after its model's graph and precision, not
/// after the batches it was built for: two detectors of one model, one
/// picture at a time and four batched, would find each other's engine in
/// one directory, find it built for other shapes, and each build its own
/// again on every start — minutes. An engine built for a range of batches
/// is kept in a directory of that range's own, `batch-1-4-4`; one built
/// for no range, a model's own batch, in `root` itself. The timing cache
/// stays in `root`, shared: it makes any engine's build quicker, whatever
/// its shapes.
#[cfg(feature = "ort-tensorrt")]
fn engine_directory(root: &std::path::Path, profile: Option<BatchProfile>) -> std::path::PathBuf {
    match profile {
        None => root.to_path_buf(),
        Some(BatchProfile { min, opt, max }) => root.join(format!("batch-{min}-{opt}-{max}")),
    }
}

#[cfg(all(test, feature = "ort-tensorrt"))]
mod tests {
    use std::path::Path;

    use super::{BatchProfile, engine_directory};

    /// Each range of batches has a directory of its own, so that engines
    /// built for two do not take each other's place; no range is the cache
    /// itself, where engines built before ranges were kept apart still are.
    #[test]
    fn each_range_of_batches_keeps_its_engine_apart() {
        let root = Path::new("/cache/tensorrt");
        let range = |min, opt, max| engine_directory(root, Some(BatchProfile { min, opt, max }));
        assert_eq!(engine_directory(root, None), root);
        assert_eq!(range(1, 4, 4), root.join("batch-1-4-4"));
        assert_ne!(range(1, 4, 4), range(1, 8, 8));
        assert_ne!(range(1, 8, 32), range(1, 32, 32));
    }
}
