//! Inference on D3D12 pictures through ONNX Runtime's DirectML provider:
//! the picture fitted to the model by a compute shader into a buffer on the
//! GPU, which DirectML reads where it is — on the [`D3d12Gpu`]'s own device
//! and queue, so the fit and the model are ordered by the queue alone and
//! nothing goes through host memory on the way in. See
//! `docs/d3d12-inference.md` for the measurements this rests on.

mod d3d12_ort_detector;
mod d3d12_ort_embedder;
mod fitting;

use std::path::Path;

use ort::{
    AsPointer,
    memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType},
    session::Session,
};
use windows::{
    Win32::AI::MachineLearning::DirectML::{
        DML_CREATE_DEVICE_FLAG_NONE, DMLCreateDevice, IDMLDevice,
    },
    core::Interface,
};

use super::OrtError;
use crate::platform::windows::d3d12_gpu::D3d12Gpu;

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
