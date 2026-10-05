//! Inference on VideoToolbox pictures through Core ML: [`MetalOrtDetector`]
//! and [`MetalOrtClassifier`], each fitting what it reads into its model's
//! input with Metal.

mod fitting;
mod metal_ort_classifier;
mod metal_ort_detector;

use std::path::Path;

use ort::{ep::CoreML, ep::coreml::ModelFormat, session::Session};

use super::OrtError;

pub use metal_ort_classifier::MetalOrtClassifier;
pub use metal_ort_detector::{MetalOrtDetector, MetalOrtDetectorOptions};

/// A session on the model at `model_path` through Core ML, or none: ONNX
/// Runtime passes over a provider that fails to register and runs on the
/// CPU in its place, which this refuses. What Core ML takes of the model it
/// runs on the Neural Engine, the GPU or the CPU as it chooses, and what it
/// does not take ONNX Runtime runs on the CPU — both allowed.
///
/// `fixed`, a dimension the model leaves open by its name, and the size it
/// is fixed at for this session.
fn core_ml_session(
    model_path: impl AsRef<Path>,
    fixed: Option<(&str, i64)>,
) -> Result<Session, OrtError> {
    let provider = CoreML::default()
        .with_model_format(ModelFormat::MLProgram)
        .build()
        .error_on_failure();
    let mut builder = Session::builder()?;
    if let Some((name, size)) = fixed {
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
