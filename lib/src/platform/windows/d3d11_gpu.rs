//! The one D3D11 device a pipeline's D3D11 elements share, and its one
//! immediate context.

use std::sync::{Arc, Mutex};

use thiserror::Error as ThisError;
use windows::Win32::Graphics::{
    Direct3D::D3D_DRIVER_TYPE_HARDWARE,
    Direct3D11::{
        D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_CREATE_DEVICE_VIDEO_SUPPORT, D3D11_SDK_VERSION,
        D3D11CreateDevice, ID3D11Device, ID3D11DeviceContext,
    },
};

use super::d3d11::protect_shared_device;
use crate::error::D3d11SharedDeviceError;

/// Why a [`D3d11Gpu`] could not be set up.
#[derive(Debug, ThisError)]
pub enum D3d11GpuError {
    /// Direct3D 11 would not create a hardware device.
    #[error("could not create a Direct3D 11 device: {0}")]
    Create(windows::core::Error),

    /// The device cannot be shared across a pipeline's threads.
    #[error(transparent)]
    SharedDevice(#[from] D3d11SharedDeviceError),
}

/// The D3D11 device every D3D11 element of a pipeline shares, and its one
/// immediate context behind the lock they all take — the `device` and
/// `context` their constructors ask for, made once, correctly.
///
/// Every D3D11 element here has to be on the same device, and every one that
/// draws or copies has to go through the same `Arc<Mutex<_>>` around that
/// device's immediate context: the runtime serializes single calls once
/// multithread protection is on, but not a bind-draw-present *sequence*, and
/// the lock is what keeps two elements' sequences from interleaving on the one
/// context. That is why the context is handed out as the one shared
/// `Arc<Mutex<_>>` rather than fetched per element, and why this type exists
/// rather than every program writing its own `D3D11CreateDevice` call.
///
/// Cloning is cheap — a COM reference and an `Arc` — and every clone is the
/// same device and the same lock. It keeps the device alive; nothing else.
#[derive(Clone)]
pub struct D3d11Gpu {
    device: ID3D11Device,
    context: Arc<Mutex<ID3D11DeviceContext>>,
}

impl std::fmt::Debug for D3d11Gpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("D3d11Gpu").finish_non_exhaustive()
    }
}

// SAFETY: `ID3D11Device` is free-threaded; the immediate context is only
// reached through the `Mutex`, with the runtime's multithread protection on
// (`protect_shared_device`) for the single calls elements make under their
// own lock scopes.
unsafe impl Send for D3d11Gpu {}
// SAFETY: as for `Send`; shared use only reaches the context through its lock.
unsafe impl Sync for D3d11Gpu {}

impl D3d11Gpu {
    /// Creates a hardware device on the default adapter, with what this
    /// crate's elements need of it: BGRA surfaces (screen and window capture,
    /// window rendering) and video support (D3D11VA decode, the video
    /// processor behind `D3d11Scaler`).
    pub fn new() -> Result<Self, D3d11GpuError> {
        let mut device = None;
        // SAFETY: out-pointers to locals; no caller-owned storage is kept.
        unsafe {
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                Default::default(),
                D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
                None,
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                None,
            )
        }
        .map_err(D3d11GpuError::Create)?;
        let device = device.ok_or_else(|| {
            D3d11GpuError::Create(windows::core::Error::from(
                windows::Win32::Foundation::E_POINTER,
            ))
        })?;
        Self::from_device(device)
    }

    /// Shares a device made elsewhere — one pinned to a particular adapter,
    /// such as the one `DxgiCaptureSource::open` returns for the monitor it
    /// captures. Fails for a device created single-threaded, which cannot be
    /// shared across a pipeline's threads.
    pub fn from_device(device: ID3D11Device) -> Result<Self, D3d11GpuError> {
        let context = protect_shared_device(&device)?;
        Ok(Self {
            device,
            context: Arc::new(Mutex::new(context)),
        })
    }

    /// The device — for every D3D11 element's `device` parameter.
    pub fn device(&self) -> &ID3D11Device {
        &self.device
    }

    /// The immediate context behind the one lock every D3D11 element shares
    /// — for every D3D11 element's `context` parameter. Clones of this are the
    /// same lock.
    pub fn context(&self) -> Arc<Mutex<ID3D11DeviceContext>> {
        Arc::clone(&self.context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A device made here is protected for sharing, and every clone and
    /// every `context()` is the same lock — the invariant every D3D11
    /// element's own constructor relies on.
    #[test]
    fn a_new_gpu_is_one_device_and_one_shared_lock() {
        let gpu = match D3d11Gpu::new() {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!("skipping: no Direct3D 11 hardware device here ({error})");
                return;
            }
        };
        let clone = gpu.clone();
        assert!(Arc::ptr_eq(&gpu.context(), &clone.context()));
        assert_eq!(gpu.device(), clone.device());
        assert!(
            protect_shared_device(gpu.device()).is_ok(),
            "protected, so any element may take it"
        );

        let shared = D3d11Gpu::from_device(gpu.device().clone()).expect("share it again");
        assert_eq!(shared.device(), gpu.device());
    }
}
