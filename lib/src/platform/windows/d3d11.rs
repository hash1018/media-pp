//! D3D11 helpers shared by more than one element. Nothing pipeline-shaped
//! lives here — just the small pieces of the Win32 API that every D3D11
//! element that draws through its own shaders needs identically.

use std::ffi::c_void;

use windows::Win32::Graphics::Direct3D::{Fxc::*, ID3DBlob, ID3DInclude};

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
