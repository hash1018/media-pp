//! The one D3D12 device a pipeline's D3D12 elements share, and the queue
//! windows present through.

use thiserror::Error as ThisError;
use windows::Win32::Graphics::{
    Direct3D::D3D_FEATURE_LEVEL_11_0,
    Direct3D12::{
        D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC, D3D12CreateDevice,
        ID3D12CommandQueue, ID3D12Device,
    },
};

/// Why a [`D3d12Gpu`] could not be set up.
#[derive(Debug, ThisError)]
pub enum D3d12GpuError {
    /// Direct3D 12 would not create a hardware device.
    #[error("could not create a Direct3D 12 device: {0}")]
    Create(windows::core::Error),

    /// The device would not create the queue windows present through.
    #[error("could not create a Direct3D 12 command queue: {0}")]
    Queue(windows::core::Error),
}

/// The D3D12 device every D3D12 element of a pipeline shares — the `device`
/// their constructors ask for — and the one direct command queue a
/// [`D3d12WindowRenderer`](crate::elements::D3d12WindowRenderer) presents
/// through.
///
/// Unlike D3D11 there is no context to share: a D3D12 element records its
/// own commands and synchronizes through fences, so what has to agree across
/// the pipeline is the device alone. The queue is here so every window on
/// the device presents through one rather than each making its own.
///
/// Cloning is cheap — COM references — and every clone is the same device
/// and queue. It keeps them alive; nothing else.
#[derive(Clone)]
pub struct D3d12Gpu {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
}

impl std::fmt::Debug for D3d12Gpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("D3d12Gpu").finish_non_exhaustive()
    }
}

// SAFETY: `ID3D12Device` and `ID3D12CommandQueue` are free-threaded.
unsafe impl Send for D3d12Gpu {}
// SAFETY: as for `Send`.
unsafe impl Sync for D3d12Gpu {}

impl D3d12Gpu {
    /// Creates a hardware device on the default adapter, at feature level
    /// 11.0 — what D3D12VA decoding and this crate's D3D12 elements need.
    pub fn new() -> Result<Self, D3d12GpuError> {
        let mut device: Option<ID3D12Device> = None;
        // SAFETY: a null adapter asks for the default one; `device` is a live
        // out-parameter of the requested interface.
        unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut device) }
            .map_err(D3d12GpuError::Create)?;
        let device = device.ok_or_else(|| {
            D3d12GpuError::Create(windows::core::Error::from(
                windows::Win32::Foundation::E_POINTER,
            ))
        })?;
        Self::from_device(device)
    }

    /// Shares a device made elsewhere, making the queue windows present
    /// through on it.
    pub fn from_device(device: ID3D12Device) -> Result<Self, D3d12GpuError> {
        // SAFETY: creates a queue on a live device from a local description.
        let queue = unsafe {
            device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })
        }
        .map_err(D3d12GpuError::Queue)?;
        Ok(Self { device, queue })
    }

    /// The device — for every D3D12 element's `device` parameter.
    pub fn device(&self) -> &ID3D12Device {
        &self.device
    }

    /// The direct queue windows on this device present through.
    pub(crate) fn queue(&self) -> &ID3D12CommandQueue {
        &self.queue
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_gpu_is_one_device_and_one_queue() {
        let gpu = match D3d12Gpu::new() {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!("skipping: no Direct3D 12 hardware device here ({error})");
                return;
            }
        };
        let clone = gpu.clone();
        assert_eq!(gpu.device(), clone.device());
        assert_eq!(gpu.queue(), clone.queue());
        let shared = D3d12Gpu::from_device(gpu.device().clone()).expect("share it again");
        assert_eq!(shared.device(), gpu.device());
    }
}
