//! The Linux rendering stack: a Vulkan swapchain that presents
//! `media_pp::elements::CudaDecoder` output.

mod cuda_ffi;
mod cuda_window_renderer;
mod vulkan_context;

use media_pp::elements::{CudaDevice, CudaRenderer};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

pub use cuda_window_renderer::CudaWindowRenderer;
pub use vulkan_context::VulkanGpuContext;

/// Opens a window renderer for `window` and wraps it as a `CudaRenderer` —
/// the Linux counterpart of `d3d11_window_renderer`, so callers
/// don't write the wrapper themselves.
pub fn cuda_window_renderer(
    name: impl Into<String>,
    gpu: &VulkanGpuContext,
    device: &CudaDevice,
    display: RawDisplayHandle,
    window: RawWindowHandle,
    width: u32,
    height: u32,
) -> Result<CudaRenderer, String> {
    let renderer = CudaWindowRenderer::new(gpu, display, window, width, height)?;
    Ok(CudaRenderer::new(name, device, Box::new(renderer)))
}
