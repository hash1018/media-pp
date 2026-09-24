//! Owns the D3D11/D3D12 rendering this project's render examples share —
//! no external `renderer-engine` dependency. `D3d12GpuContext`/
//! `D3d11GpuContext` are the process-wide device/queue/shader-pipeline
//! owners (create one per stack, share it across every window);
//! `d3d12_window_renderer`/`d3d11_window_renderer` open one window's
//! `D3d12WindowRenderer`/`D3d11WindowRenderer`, already wrapped as a
//! `media_pp::elements::D3d12Renderer`/`D3d11Renderer`.
//!
//! Named rather than linked, all of them: they are `#[cfg(windows)]`, so on
//! any other host the links this file is read on would be broken ones. The
//! same reason the Linux modules name their D3D siblings in plain text. The two stacks are
//! independent — separate device, separate shader set, nothing shared
//! between them.
//!
//! Both present from the pipeline's own thread into a window the main thread
//! owns, so they share `run_window`: the winit shell that opens that window,
//! runs the work beside it, and — the part that is easy to get wrong and
//! fatal to get wrong — stops and joins before the window is dropped. See
//! that function and [`Shutdown`] for the orderings it exists to get right.
//!
//! On Linux there is no renderer here: the library's `VulkanWindowRenderer`
//! opens its own window and draws into it, from system memory or CUDA, and
//! the examples use it directly. What they share is [`stop_on_close`], which
//! turns closing that window into the same [`Shutdown`] the Windows shell
//! uses.

mod shutdown;

pub use shutdown::Shutdown;

#[cfg(target_os = "windows")]
mod window_shell;

#[cfg(target_os = "windows")]
pub use window_shell::{WindowTarget, run_window, run_windows};

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
pub use linux::{stop_on_close, to_drawable};

#[cfg(target_os = "windows")]
mod d3d11_gpu_context;
#[cfg(target_os = "windows")]
mod d3d11_window_renderer;
#[cfg(target_os = "windows")]
mod d3d12_gpu_context;
#[cfg(target_os = "windows")]
mod d3d12_window_renderer;

#[cfg(target_os = "windows")]
pub use d3d11_gpu_context::D3d11GpuContext;
#[cfg(target_os = "windows")]
pub use d3d11_window_renderer::D3d11WindowRenderer;
#[cfg(target_os = "windows")]
pub use d3d12_gpu_context::D3d12GpuContext;
#[cfg(target_os = "windows")]
pub use d3d12_window_renderer::D3d12WindowRenderer;
#[cfg(target_os = "windows")]
use media_pp::elements::{D3d11Renderer, D3d12Renderer, SubmitError};

#[cfg(target_os = "windows")]
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

#[cfg(target_os = "windows")]
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
