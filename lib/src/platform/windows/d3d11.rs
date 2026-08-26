//! D3D11 helpers shared by more than one element. Nothing pipeline-shaped
//! lives here — just the small pieces of the Win32 API that every D3D11
//! element that draws through its own shaders needs identically.

use std::ffi::c_void;

use windows::{
    Win32::Graphics::{
        Direct3D::{Fxc::*, ID3DBlob, ID3DInclude},
        Direct3D11::{
            D3D11_CREATE_DEVICE_SINGLETHREADED, ID3D11Device, ID3D11DeviceContext,
            ID3D11Multithread,
        },
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

/// Compiles one HLSL entry point at runtime, turning a compiler diagnostic
/// into the returned error's message instead of an opaque `E_FAIL` — a
/// shader that fails to compile is a build-time mistake in this crate's own
/// `shaders/` sources, and the compiler's text is the only thing that says
/// which line.
///
/// `file_name` is what the compiler prints in those diagnostics; pass the
/// shader's own file name so an error names the file a reader can open.
pub(crate) unsafe fn compile_shader(
    source: &[u8],
    file_name: windows::core::PCSTR,
    entry: windows::core::PCSTR,
    target: windows::core::PCSTR,
) -> windows::core::Result<ID3DBlob> {
    let mut shader = None;
    let mut errors = None;
    let flags = if cfg!(debug_assertions) {
        D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION
    } else {
        D3DCOMPILE_OPTIMIZATION_LEVEL3
    };
    // SAFETY: `source` is readable for its exact length, `file_name`, `entry`,
    // and `target` are static NUL-terminated strings supplied by the callers,
    // and both blob slots are live out-parameters.
    let result = unsafe {
        D3DCompile(
            source.as_ptr().cast::<c_void>(),
            source.len(),
            file_name,
            None,
            None::<&ID3DInclude>,
            entry,
            target,
            flags,
            0,
            &mut shader,
            Some(&mut errors),
        )
    };
    if let Err(error) = result {
        let message = errors
            // SAFETY: a compiler error blob owns `GetBufferSize()` readable
            // bytes at `GetBufferPointer()` for the lifetime of the blob.
            .map(|blob| unsafe {
                let bytes = std::slice::from_raw_parts(
                    blob.GetBufferPointer().cast::<u8>(),
                    blob.GetBufferSize(),
                );
                String::from_utf8_lossy(bytes).into_owned()
            })
            .unwrap_or_else(|| error.message());
        return Err(windows::core::Error::new(
            windows::Win32::Foundation::E_FAIL,
            message,
        ));
    }
    Ok(shader.expect("D3DCompile succeeded without producing a blob"))
}

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
