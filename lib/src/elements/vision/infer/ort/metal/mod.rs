//! Inference on VideoToolbox pictures through Core ML: [`MetalOrtDetector`],
//! [`MetalOrtClassifier`] and [`MetalOrtEmbedder`], each fitting what it
//! reads into its model's input with Metal.

mod best_class;
mod fitting;
mod metal_ort_classifier;
mod metal_ort_detector;
mod metal_ort_embedder;

use std::path::{Path, PathBuf};

use ort::{
    ep::CoreML,
    ep::coreml::{ComputeUnits, ModelFormat},
    session::Session,
};

use ort::value::ValueType;

use super::OrtError;
use super::classify::{Input, image_input};
use crate::pp_log::{PpLog, pp_warn};

pub use metal_ort_classifier::MetalOrtClassifier;
pub use metal_ort_detector::{MetalOrtDetector, MetalOrtDetectorOptions};
pub use metal_ort_embedder::{MetalOrtEmbedder, MetalOrtEmbedderOptions};

/// Where Core ML may run a model: what its `MLComputeUnits` allows. The
/// CPU is allowed in each, for the layers the others do not take — a
/// detector that is to run on the CPU alone is
/// [`SwOrtDetector`](crate::elements::SwOrtDetector).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoreMlComputeUnits {
    /// The GPU, the Neural Engine and the CPU, as Core ML chooses layer by
    /// layer.
    #[default]
    All,
    /// The GPU and the CPU.
    CpuAndGpu,
    /// The Neural Engine and the CPU.
    CpuAndNeuralEngine,
}

/// A session on the model at `model_path` through Core ML, or none: ONNX
/// Runtime passes over a provider that fails to register and runs on the
/// CPU in its place, which this refuses. What Core ML takes of the model it
/// runs on the Neural Engine, the GPU or the CPU as it chooses, and what it
/// does not take ONNX Runtime runs on the CPU — both allowed.
///
/// `fixed`, the dimensions the model leaves open by their names, and the
/// size each is fixed at for this session; `units`, where Core ML may run
/// it; `cache`, where the model Core ML compiles is kept — see
/// [`model_cache`]. A cache that cannot be made is gone without, with a
/// warning: the model is compiled anew, as it is every time without one.
fn core_ml_session(
    model_path: impl AsRef<Path>,
    fixed: &[(&str, i64)],
    units: CoreMlComputeUnits,
    cache: Option<&Path>,
    pp_log: &PpLog,
) -> Result<Session, OrtError> {
    let mut provider = CoreML::default()
        .with_model_format(ModelFormat::MLProgram)
        .with_compute_units(match units {
            CoreMlComputeUnits::All => ComputeUnits::All,
            CoreMlComputeUnits::CpuAndGpu => ComputeUnits::CPUAndGPU,
            CoreMlComputeUnits::CpuAndNeuralEngine => ComputeUnits::CPUAndNeuralEngine,
        });
    let cache = model_cache(cache, fixed);
    match (std::fs::create_dir_all(&cache), cache.to_str()) {
        (Ok(()), Some(dir)) => provider = provider.with_model_cache_dir(dir),
        (Err(error), _) => pp_warn!(
            pp_log: pp_log,
            "no model cache at {}: {error}; Core ML compiles the model anew",
            cache.display()
        ),
        (Ok(()), None) => pp_warn!(
            pp_log: pp_log,
            "the model cache {} is no UTF-8 path, which ONNX Runtime takes; \
             Core ML compiles the model anew",
            cache.display()
        ),
    }
    let provider = provider.build().error_on_failure();
    let mut builder = Session::builder()?;
    for &(name, size) in fixed {
        builder = builder
            .with_dimension_override(name, size)
            .map_err(|error| OrtError::Ort(error.into()))?;
    }
    builder
        .with_execution_providers([provider])
        .map_err(|error| OrtError::Ort(error.into()))?
        .commit_from_file(model_path)
        .map_err(OrtError::from)
}

/// Where the model Core ML compiles for a session with `fixed` dimensions is
/// kept: in `given`, or where none is, the user's cache directory under
/// `media-pp/coreml` — `~/Library/Caches/media-pp/coreml` — each in a
/// directory of its fixed dimensions' own (`batch-4`, `as-made`).
///
/// Without one, ONNX Runtime converts the model and Core ML compiles it on
/// every session, each time a pipeline is made. With one, the first
/// session's is kept and every later one loads it: on an M5, a RetinaFace
/// ResNet-50 fixed to a batch of 7, YOLO11n the same, SFace and OSNet
/// opened in 2.0 seconds compiled and in 0.4 from the cache.
/// ONNX Runtime keys a model by its graph and metadata, not by its weights,
/// so a model whose weights alone change under the same name is taken for
/// the one it replaced: give it a name of its own, or another cache. The
/// fixed dimensions are kept apart here, so one model fixed to a batch of 1
/// and to one of 7 — sessions the file alone does not tell apart — can
/// never be given each other's compiled model.
fn model_cache(given: Option<&Path>, fixed: &[(&str, i64)]) -> PathBuf {
    let root = given.map(Path::to_path_buf).unwrap_or_else(|| {
        std::env::var_os("HOME")
            .map(|home| PathBuf::from(home).join("Library").join("Caches"))
            .unwrap_or_else(std::env::temp_dir)
            .join("media-pp")
            .join("coreml")
    });
    let shapes: Vec<String> = fixed
        .iter()
        .map(|(name, size)| {
            let name: String = name
                .chars()
                .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
                .collect();
            format!("{name}-{size}")
        })
        .collect();
    root.join(if shapes.is_empty() {
        "as-made".to_string()
    } else {
        shapes.join("_")
    })
}

/// How many objects a model that takes any number at once is fixed to take
/// through Core ML, the inputs past those a picture has left as they were.
///
/// Core ML compiles a model of open shape again for each new batch — 0.6
/// seconds a time for MobileNetV2 — and runs it no faster than the CPU, a
/// batch of one through an error it then recovers from; of fixed shape it
/// compiles once and runs on the Neural Engine. Which size is a trade: after
/// a tracker most pictures have an object or two to classify, which a
/// smaller batch runs sooner, and a crowd first seen comes in a run per
/// batch. On an M5 with MobileNetV2 behind YOLOv10n and a tracker, 4 kept
/// Intel's walking people at 195 pictures a second against 199 for 1 and
/// 112 for 8, and a clip with an object classified on every picture at 137
/// against 166 for 1, while taking a crowd in a quarter of the runs.
const CORE_ML_BATCH: usize = 4;

/// A session through Core ML for a model run on the objects a detector
/// found — a classifier's, an embedder's — and its input, `open` for a side
/// it leaves open. A batch it leaves open is fixed to [`CORE_ML_BATCH`] by
/// its name; one with no name to fix it by is left open, with a warning.
fn object_model_session(
    model_path: &Path,
    open: u32,
    cache: Option<&Path>,
    pp_log: &PpLog,
) -> Result<(Session, Input), OrtError> {
    // Read on the CPU first, which compiles nothing: what the model calls a
    // batch it leaves open, to fix it for Core ML.
    let probe = Session::builder()?.commit_from_file(model_path)?;
    let batch = match (image_input(&probe, open)?.batch, probe.inputs()[0].dtype()) {
        (
            None,
            ValueType::Tensor {
                dimension_symbols, ..
            },
        ) if !dimension_symbols[0].is_empty() => Some(dimension_symbols[0].clone()),
        (None, _) => {
            pp_warn!(
                pp_log: pp_log,
                "the model leaves its batch open without a name to fix it by: \
                 Core ML compiles it again for each number of objects"
            );
            None
        }
        (Some(_), _) => None,
    };
    drop(probe);
    let session = core_ml_session(
        model_path,
        batch
            .as_deref()
            .map(|name| (name, CORE_ML_BATCH as i64))
            .as_slice(),
        CoreMlComputeUnits::All,
        cache,
        pp_log,
    )?;
    let input = image_input(&session, open)?;
    Ok((session, input))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Where a model fixed to a batch is kept: under the directory given,
    /// or media-pp's own under the user's caches, in one of its fixed
    /// dimensions' own, so one model fixed two ways is kept twice.
    #[test]
    fn a_model_is_cached_apart_for_each_shape_it_is_fixed_to() {
        let given = Path::new("/somewhere/cache");
        assert_eq!(model_cache(Some(given), &[]), given.join("as-made"));
        assert_eq!(
            model_cache(Some(given), &[("batch", 7)]),
            given.join("batch-7")
        );
        assert_eq!(
            model_cache(Some(given), &[("height", 640), ("batch:0", 1)]),
            given.join("height-640_batch_0-1")
        );
        assert_ne!(
            model_cache(Some(given), &[("batch", 1)]),
            model_cache(Some(given), &[("batch", 7)])
        );
        let own = model_cache(None, &[("batch", 4)]);
        assert!(
            own.ends_with("media-pp/coreml/batch-4"),
            "{}",
            own.display()
        );
    }
}
