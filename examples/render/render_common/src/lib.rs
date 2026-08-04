//! Owns the D3D12 rendering this project's render examples share — no
//! external `renderer-engine` dependency. [`GpuContext`] is the
//! process-wide device/queue/shader-pipeline owner (create one, share it
//! across every window); [`window_renderer`] opens one window's
//! [`D3d12WindowRenderer`], already wrapped as a `media_pp::elements::D3d12Renderer`.

mod gpu_context;
mod window_renderer;

pub use gpu_context::GpuContext;
use media_pp::elements::{D3d12Renderer, SubmitError};
pub use window_renderer::D3d12WindowRenderer;

/// Opens a window renderer for `hwnd` and wraps it as a `D3d12Renderer` —
/// the whole point of this crate, so callers don't write the wrapper
/// themselves.
pub fn window_renderer(
    name: impl Into<String>,
    gpu: &GpuContext,
    hwnd: isize,
    width: u32,
    height: u32,
) -> Result<D3d12Renderer, SubmitError> {
    let renderer = D3d12WindowRenderer::new(gpu, hwnd, width, height)?;
    Ok(D3d12Renderer::new(name, Box::new(renderer)))
}
