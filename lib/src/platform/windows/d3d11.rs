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

/// Why an immediate context could not be made safe for cross-thread use.
#[derive(Debug)]
pub(crate) enum MultithreadProtectionError {
    /// The device explicitly opted out of all D3D11 cross-thread use.
    SingleThreadedDevice,
    /// The immediate context did not expose the standard protection interface.
    Windows(windows::core::Error),
}

/// Enables the runtime critical section around every immediate-context call.
///
/// D3D11 device methods are free-threaded, but the one immediate context shared
/// by a device is not. Pipeline queues deliberately move downstream work to
/// another thread, so a device crossing an element boundary must have this
/// protection before either side starts issuing context commands.
pub(crate) fn enable_multithread_protection(
    device: &ID3D11Device,
    context: &ID3D11DeviceContext,
) -> Result<(), MultithreadProtectionError> {
    // SAFETY: reads immutable creation metadata from a live device.
    let flags = unsafe { device.GetCreationFlags() };
    if flags & D3D11_CREATE_DEVICE_SINGLETHREADED.0 != 0 {
        return Err(MultithreadProtectionError::SingleThreadedDevice);
    }
    let multithread: ID3D11Multithread = context
        .cast()
        .map_err(MultithreadProtectionError::Windows)?;
    // SAFETY: `multithread` is the live immediate context's standard runtime
    // synchronization interface. Enabling protection is process-local and
    // does not borrow caller storage.
    let _ = unsafe { multithread.SetMultithreadProtected(true) };
    Ok(())
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
    use windows::Win32::Graphics::{
        Direct3D::D3D_DRIVER_TYPE_HARDWARE,
        Direct3D11::{
            D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_CREATE_DEVICE_FLAG, D3D11_SDK_VERSION,
            D3D11CreateDevice, ID3D11Device, ID3D11DeviceContext, ID3D11Multithread,
        },
    };

    use super::*;

    fn try_device(single_threaded: bool) -> Option<(ID3D11Device, ID3D11DeviceContext)> {
        let mut device = None;
        let mut context = None;
        let flags = D3D11_CREATE_DEVICE_FLAG(
            D3D11_CREATE_DEVICE_BGRA_SUPPORT.0
                | if single_threaded {
                    D3D11_CREATE_DEVICE_SINGLETHREADED.0
                } else {
                    0
                },
        );
        // SAFETY: the default hardware adapter and feature levels are used;
        // both output slots are correctly typed and live for the call.
        let result = unsafe {
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                Default::default(),
                flags,
                None,
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut context),
            )
        };
        if let Err(error) = result {
            eprintln!("skipping: D3D11CreateDevice failed on this machine: {error}");
            return None;
        }
        Some((device.unwrap(), context.unwrap()))
    }

    #[test]
    fn enables_runtime_protection_on_a_multithread_capable_device() {
        let Some((device, context)) = try_device(false) else {
            return;
        };
        enable_multithread_protection(&device, &context).expect("enable protection");
        let multithread: ID3D11Multithread = context.cast().expect("multithread interface");
        // SAFETY: reads one boolean property from the live context interface.
        assert!(unsafe { multithread.GetMultithreadProtected() }.as_bool());
    }

    #[test]
    fn rejects_a_device_that_promised_single_threaded_use() {
        let Some((device, context)) = try_device(true) else {
            return;
        };
        assert!(matches!(
            enable_multithread_protection(&device, &context),
            Err(MultithreadProtectionError::SingleThreadedDevice)
        ));
    }
}
