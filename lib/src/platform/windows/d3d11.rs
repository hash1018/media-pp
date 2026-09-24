//! D3D11 helpers shared by more than one element. Nothing pipeline-shaped
//! lives here — just the small pieces of the Win32 API that every D3D11
//! element that draws through its own shaders needs identically.

use windows::{
    Win32::Graphics::Direct3D11::{
        D3D11_CREATE_DEVICE_SINGLETHREADED, ID3D11Device, ID3D11DeviceContext, ID3D11Multithread,
    },
    core::Interface,
};

use crate::error::D3d11SharedDeviceError;

/// Prepares `device` to be shared by the elements of one pipeline and returns
/// the immediate context they all have to funnel their GPU commands through.
///
/// D3D11 device methods are free-threaded; the single immediate context a
/// device owns is not. A `Queue` deliberately puts the elements on either side
/// of it on different threads, so every entry point in this crate that accepts
/// a caller-owned device routes it through here first — the whole fence-free
/// design of this D3D11 stack rests on the runtime serializing those calls, and
/// that serialization is off by default.
///
/// Enabling is idempotent, so it does not matter which element gets there
/// first, and the result is read back rather than assumed: a device that ends
/// up unprotected must fail loudly at construction instead of racing later.
pub(crate) fn protect_shared_device(
    device: &ID3D11Device,
) -> Result<ID3D11DeviceContext, D3d11SharedDeviceError> {
    // SAFETY: reads immutable creation metadata from a live device.
    let flags = unsafe { device.GetCreationFlags() };
    if flags & D3D11_CREATE_DEVICE_SINGLETHREADED.0 != 0 {
        return Err(D3d11SharedDeviceError::SingleThreaded);
    }
    // SAFETY: returns the one immediate context owned by this live device.
    let context = unsafe { device.GetImmediateContext()? };
    let multithread: ID3D11Multithread = context.cast()?;
    // SAFETY: `multithread` is the live immediate context's standard runtime
    // synchronization interface. Both calls are process-local and borrow
    // nothing; `SetMultithreadProtected` returns the previous setting, which
    // says nothing about whether the new one took, hence the read back.
    unsafe {
        let _ = multithread.SetMultithreadProtected(true);
        if !multithread.GetMultithreadProtected().as_bool() {
            return Err(D3d11SharedDeviceError::ProtectionRefused);
        }
    }
    Ok(context)
}

pub(crate) use super::hlsl::compile_shader;

#[cfg(test)]
mod tests {
    use windows::Win32::Graphics::Direct3D11::ID3D11Multithread;

    use super::*;
    use crate::test_support::{try_d3d11_device, try_single_threaded_d3d11_device};

    #[test]
    fn enables_runtime_protection_on_a_multithread_capable_device() {
        let Some((device, _context)) = try_d3d11_device() else {
            return;
        };
        let context = protect_shared_device(&device).expect("protect the shared device");
        let multithread: ID3D11Multithread = context.cast().expect("multithread interface");
        // SAFETY: reads one boolean property from the live context interface.
        assert!(unsafe { multithread.GetMultithreadProtected() }.as_bool());

        // Idempotent: whichever element reaches the device second must not be
        // told the device is unusable.
        protect_shared_device(&device).expect("protect an already protected device");
    }

    #[test]
    fn rejects_a_device_that_promised_single_threaded_use() {
        let Some(device) = try_single_threaded_d3d11_device() else {
            return;
        };
        assert!(matches!(
            protect_shared_device(&device),
            Err(D3d11SharedDeviceError::SingleThreaded)
        ));
    }
}
