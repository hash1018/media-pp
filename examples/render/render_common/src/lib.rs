//! Owns the D3D11/D3D12 rendering this project's render examples share —
//! no external `renderer-engine` dependency. [`D3d12GpuContext`]/
//! [`D3d11GpuContext`] are the process-wide device/queue/shader-pipeline
//! owners (create one per stack, share it across every window);
//! [`d3d12_window_renderer`]/[`d3d11_window_renderer`] open one window's
//! [`D3d12WindowRenderer`]/[`D3d11WindowRenderer`], already wrapped as a
//! `media_pp::elements::D3d12Renderer`/`D3d11Renderer`. The two stacks are
//! independent — separate device, separate shader set, nothing shared
//! between them.

mod d3d11_gpu_context;
mod d3d11_window_renderer;
mod d3d12_gpu_context;
mod d3d12_window_renderer;

pub use d3d11_gpu_context::D3d11GpuContext;
pub use d3d11_window_renderer::D3d11WindowRenderer;
pub use d3d12_gpu_context::D3d12GpuContext;
pub use d3d12_window_renderer::D3d12WindowRenderer;
use media_pp::elements::{D3d11Renderer, D3d12Renderer, SubmitError};

/// Opens a window renderer for `hwnd` and wraps it as a `D3d12Renderer` —
/// the whole point of this crate, so callers don't write the wrapper
/// themselves.
pub fn d3d12_window_renderer(
    name: impl Into<String>,
    gpu: &D3d12GpuContext,
    hwnd: isize,
    width: u32,
    height: u32,
) -> Result<D3d12Renderer, SubmitError> {
    let renderer = D3d12WindowRenderer::new(gpu, hwnd, width, height)?;
    Ok(D3d12Renderer::new(name, Box::new(renderer)))
}

/// The D3D11 sibling of [`d3d12_window_renderer`] — opens a window
/// renderer for `hwnd` and wraps it as a `D3d11Renderer`.
pub fn d3d11_window_renderer(
    name: impl Into<String>,
    gpu: &D3d11GpuContext,
    hwnd: isize,
    width: u32,
    height: u32,
) -> Result<D3d11Renderer, SubmitError> {
    let renderer = D3d11WindowRenderer::new(gpu, hwnd, width, height)?;
    Ok(D3d11Renderer::new(name, Box::new(renderer)))
}
