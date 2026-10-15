//! Inference on D3D12 pictures through ONNX Runtime's DirectML provider:
//! the picture fitted to the model by a compute shader into a buffer on the
//! GPU, which DirectML reads where it is — on the [`D3d12Gpu`]'s own device
//! and queue, so the fit and the model are ordered by the queue alone and
//! nothing goes through host memory on the way in. See
//! `docs/d3d12-inference.md` for the measurements this rests on.

mod d3d12_ort_classifier;
mod d3d12_ort_detector;
mod d3d12_ort_embedder;
mod fitting;

use std::path::Path;

use ort::{
    AsPointer,
    memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType},
    session::Session,
    value::ValueType,
};
use windows::{
    Win32::AI::MachineLearning::DirectML::{
        DML_CREATE_DEVICE_FLAG_NONE, DMLCreateDevice, IDMLDevice,
    },
    core::Interface,
};

use super::OrtError;
use super::classify::{Input, image_input};
use crate::platform::windows::d3d12_gpu::D3d12Gpu;
use crate::pp_log::{PpLog, pp_warn};

pub use d3d12_ort_classifier::D3d12OrtClassifier;
pub use d3d12_ort_detector::{D3d12OrtDetector, D3d12OrtDetectorOptions};
pub use d3d12_ort_embedder::D3d12OrtEmbedder;

/// What DirectML runs on: a DirectML device made on a [`D3d12Gpu`]'s
/// device, and the GPU's queue it is handed beside it.
struct DirectMl {
    device: IDMLDevice,
    gpu: D3d12Gpu,
}

// SAFETY: a DirectML device is free-threaded, as the D3D12 objects are.
unsafe impl Send for DirectMl {}

impl DirectMl {
    /// DirectML on `gpu`.
    fn new(gpu: &D3d12Gpu) -> Result<Self, OrtError> {
        if ort::ep::directml::api().is_none() {
            return Err(OrtError::DirectMlUnavailable);
        }
        let mut device: Option<IDMLDevice> = None;
        // SAFETY: a live D3D12 device and a correctly typed out-parameter.
        unsafe { DMLCreateDevice(gpu.device(), DML_CREATE_DEVICE_FLAG_NONE, &mut device) }?;
        Ok(Self {
            device: device.expect("DMLCreateDevice succeeded without a device"),
            gpu: gpu.clone(),
        })
    }

    /// A session on the model at `model_path`, run by DirectML on this
    /// device and queue, the dimensions `fixed` names fixed at their sizes.
    ///
    /// DirectML wants every dimension known when the session is made, and
    /// neither ONNX Runtime's memory patterns nor its parallel execution.
    fn session(&self, model_path: &Path, fixed: &[(&str, i64)]) -> Result<Session, OrtError> {
        let api = ort::ep::directml::api().ok_or(OrtError::DirectMlUnavailable)?;
        let mut builder = Session::builder()?
            .with_memory_pattern(false)
            .map_err(|error| OrtError::Ort(error.into()))?
            .with_parallel_execution(false)
            .map_err(|error| OrtError::Ort(error.into()))?;
        for &(name, size) in fixed {
            builder = builder
                .with_dimension_override(name, size)
                .map_err(|error| OrtError::Ort(error.into()))?;
        }
        // `ort`'s own DirectML provider takes only an adapter's index and
        // makes a device of its own; this hands it the GPU's, so a buffer
        // of that device is a tensor it can read.
        // SAFETY: the options are the builder's own and live; the DirectML
        // device and queue are live COM objects, which ONNX Runtime takes
        // references of.
        unsafe {
            ort::Error::result_from_status((api.SessionOptionsAppendExecutionProvider_DML1)(
                builder.ptr_mut().cast(),
                self.device.as_raw(),
                self.gpu.queue().as_raw(),
            ))?;
        }
        Ok(builder.commit_from_file(model_path)?)
    }
}

/// How many objects a model that takes any number at once is fixed to take
/// for DirectML, which is fastest with every dimension fixed: a picture's
/// objects go in eights, the inputs past them left as they were.
const DIRECTML_BATCH: usize = 8;

/// A session on the object model at `model_path` — an embedder's or a
/// classifier's, of one input `[batch, 3, height, width]`, `open_side` for a side
/// it leaves open — run by DirectML on `gpu`, with its input as fixed.
///
/// The model is read on the CPU first for what it calls the dimensions it
/// leaves open, which are fixed to those sizes, its batch to
/// [`DIRECTML_BATCH`].
fn object_model_session(
    gpu: &D3d12Gpu,
    model_path: &Path,
    open_side: u32,
    pp_log: &PpLog,
) -> Result<(Session, Input), OrtError> {
    let probe = Session::builder()
        .map_err(OrtError::from)?
        .commit_from_file(model_path)
        .map_err(OrtError::from)?;
    let open = image_input(&probe, open_side)?;
    let mut fixed: Vec<(String, i64)> = Vec::new();
    if let Some(ValueType::Tensor {
        shape,
        dimension_symbols,
        ..
    }) = probe.inputs().first().map(|input| input.dtype())
    {
        let named = |axis: usize| {
            dimension_symbols
                .get(axis)
                .filter(|symbol| !symbol.is_empty() && shape[axis] <= 0)
                .cloned()
        };
        for (axis, size) in [(2, open.size.1), (3, open.size.0)] {
            if let Some(symbol) = named(axis) {
                fixed.push((symbol, i64::from(size)));
            }
        }
        match (open.batch, named(0)) {
            (None, Some(symbol)) => fixed.push((symbol, DIRECTML_BATCH as i64)),
            (None, None) => pp_warn!(
                pp_log: pp_log,
                "the model leaves its batch open without a name to fix it by: \
                 DirectML runs it at each number of objects as it comes"
            ),
            (Some(_), _) => {}
        }
    }
    drop(probe);
    let fixed: Vec<(&str, i64)> = fixed
        .iter()
        .map(|(symbol, size)| (symbol.as_str(), *size))
        .collect();
    let session = DirectMl::new(gpu)?.session(model_path, &fixed)?;
    let input = image_input(&session, open_side)?;
    Ok((session, input))
}

/// Where a tensor DirectML reads is: a D3D12 buffer on its device.
fn directml_memory() -> Result<MemoryInfo<'static>, OrtError> {
    Ok(MemoryInfo::new(
        AllocationDevice::DIRECTML,
        0,
        AllocatorType::Device,
        MemoryType::Default,
    )?)
}

/// Where outputs are copied for the CPU to read.
fn host_memory() -> Result<MemoryInfo<'static>, OrtError> {
    Ok(MemoryInfo::new(
        AllocationDevice::CPU,
        0,
        AllocatorType::Device,
        MemoryType::CPUOutput,
    )?)
}
