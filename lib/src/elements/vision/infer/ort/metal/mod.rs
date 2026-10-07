//! Inference on VideoToolbox pictures through Core ML: [`MetalOrtDetector`],
//! [`MetalOrtClassifier`] and [`MetalOrtEmbedder`], each fitting what it
//! reads into its model's input with Metal.

mod best_class;
mod fitting;
mod metal_ort_classifier;
mod metal_ort_detector;
mod metal_ort_embedder;

use std::path::Path;

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
pub use metal_ort_embedder::MetalOrtEmbedder;

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
/// it.
fn core_ml_session(
    model_path: impl AsRef<Path>,
    fixed: &[(&str, i64)],
    units: CoreMlComputeUnits,
) -> Result<Session, OrtError> {
    let provider = CoreML::default()
        .with_model_format(ModelFormat::MLProgram)
        .with_compute_units(match units {
            CoreMlComputeUnits::All => ComputeUnits::All,
            CoreMlComputeUnits::CpuAndGpu => ComputeUnits::CPUAndGPU,
            CoreMlComputeUnits::CpuAndNeuralEngine => ComputeUnits::CPUAndNeuralEngine,
        })
        .build()
        .error_on_failure();
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
    )?;
    let input = image_input(&session, open)?;
    Ok((session, input))
}
