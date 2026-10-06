//! Inference on VideoToolbox pictures through Core ML: [`MetalOrtDetector`]
//! and [`MetalOrtClassifier`], each fitting what it reads into its model's
//! input with Metal.

mod best_class;
mod fitting;
mod metal_ort_classifier;
mod metal_ort_detector;

use std::path::Path;

use ort::{
    ep::CoreML,
    ep::coreml::{ComputeUnits, ModelFormat},
    session::Session,
};

use super::OrtError;

pub use metal_ort_classifier::MetalOrtClassifier;
pub use metal_ort_detector::{MetalOrtDetector, MetalOrtDetectorOptions};

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
