//! The one place that adapts `renderer_engine::window_renderer::WindowRenderer`
//! to `media_pp`'s `FrameRenderer` trait, so every `examples/*_render` crate
//! doesn't hand-copy the same ~40 lines. `media-pp` itself still has no
//! dependency on `renderer-engine` at all — that dependency lives here (and
//! in whichever example needs a `RendererEngine` to open one, e.g. to pass
//! `engine.device()` into `D3d12vaDecoder`).

use media_pp::elements::{Dx12Renderer, FrameRenderer};
use renderer_engine::{
    engine::RendererEngine,
    window_renderer::{RawPlane, SubmitError, WindowRenderer},
};
use windows::Win32::Graphics::Direct3D12::{ID3D12Fence, ID3D12Resource};

struct RealFrameRenderer(WindowRenderer);

impl FrameRenderer for RealFrameRenderer {
    unsafe fn submit_yuv420p(
        &self,
        y: media_pp::elements::RawPlane,
        u: media_pp::elements::RawPlane,
        v: media_pp::elements::RawPlane,
        width: u32,
        height: u32,
    ) -> Result<(), media_pp::elements::SubmitError> {
        let plane = |p: media_pp::elements::RawPlane| RawPlane {
            data: p.data,
            len: p.len,
            stride: p.stride,
        };
        unsafe {
            self.0
                .submit_yuv420p(plane(y), plane(u), plane(v), width, height)
        }
        .map_err(convert_error)
    }

    unsafe fn submit_nv12_texture(
        &self,
        texture: ID3D12Resource,
        fence: ID3D12Fence,
        fence_value: u64,
        width: u32,
        height: u32,
        keep_alive: Box<dyn std::any::Any + Send>,
    ) -> Result<(), media_pp::elements::SubmitError> {
        unsafe {
            self.0
                .submit_nv12_texture(texture, fence, fence_value, width, height, keep_alive)
        }
        .map_err(convert_error)
    }

    fn resize(&self, width: u32, height: u32) -> Result<(), media_pp::elements::SubmitError> {
        self.0.resize(width, height).map_err(convert_error)
    }
}

fn convert_error(error: SubmitError) -> media_pp::elements::SubmitError {
    match error {
        SubmitError::NullBuffer => media_pp::elements::SubmitError::NullBuffer,
        SubmitError::InvalidFrame => media_pp::elements::SubmitError::InvalidFrame,
        SubmitError::NoFreeSlot => media_pp::elements::SubmitError::NoFreeSlot,
        SubmitError::RendererStopped => media_pp::elements::SubmitError::RendererStopped,
        SubmitError::RenderFailed => media_pp::elements::SubmitError::RenderFailed,
        SubmitError::DeviceRemoved => media_pp::elements::SubmitError::DeviceRemoved,
    }
}

/// Opens a `WindowRenderer` for `hwnd` and wraps it as a `Dx12Renderer` —
/// the whole point of this crate, so callers don't write the wrapper
/// themselves. `Err` only on `WindowRenderer::new` failing (e.g. a bad
/// `hwnd` or zero size); the wrap itself is infallible.
pub fn window_renderer(
    name: impl Into<String>,
    engine: &RendererEngine,
    hwnd: isize,
    width: u32,
    height: u32,
) -> Result<Dx12Renderer, SubmitError> {
    let window_renderer = WindowRenderer::new(engine, hwnd, width, height)?;
    Ok(Dx12Renderer::new(
        name,
        Box::new(RealFrameRenderer(window_renderer)),
    ))
}
