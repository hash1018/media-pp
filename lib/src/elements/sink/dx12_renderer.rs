use std::sync::Arc;

use ffmpeg_next as ffmpeg;
use renderer_engine::{
    engine::RendererEngine,
    window_renderer::{RawPlane, SubmitError, WindowRenderer},
};
use thiserror::Error as ThisError;
use windows::{
    Win32::Graphics::Direct3D12::{ID3D12Fence, ID3D12Resource},
    core::Interface,
};

use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Element, Sink},
    elements::filter::decoder::d3d12va_decoder::d3d12va_texture,
    error::Result,
};

/// Errors specific to `Dx12Renderer`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum Dx12RendererError {
    #[error("failed to create window renderer: {0:?}")]
    Create(SubmitError),

    #[error("failed to submit frame: {0:?}")]
    Submit(SubmitError),

    #[error("failed to resize: {0:?}")]
    Resize(SubmitError),

    #[error(
        "Dx12Renderer only handles YUV420P frames (CPU) or D3D12 frames \
         (from D3d12vaDecoder), got {0:?}"
    )]
    UnsupportedFormat(ffmpeg::format::Pixel),

    #[error(
        "frame claimed the D3D12 pixel format but has no AVD3D12VAFrame \
         payload — must come from D3d12vaDecoder"
    )]
    InvalidD3d12Frame,
}

/// Terminal sink that submits decoded video frames to a native window via
/// [`renderer_engine`]'s DX12 `WindowRenderer`. Only built with the
/// `dx12-renderer` feature — every consumer that doesn't need to render to
/// a window pulls in neither this nor the DX12 dependency it wraps.
///
/// Handles two kinds of input, dispatched on `frame.format()`:
///   - `Pixel::YUV420P`: CPU-decoded (e.g. from `SwDecoder`) — copies
///     pixel bytes to the GPU via `submit_yuv420p`.
///   - `Pixel::D3D12`: GPU-decoded (from `D3d12vaDecoder`) — zero-copy,
///     draws straight from the decoder's own texture via
///     `submit_nv12_texture`.
pub struct Dx12Renderer {
    name: String,
    inner: WindowRenderer,
}

impl Dx12Renderer {
    /// `hwnd` is the target window's handle (`HWND` cast to `isize`, same
    /// as `renderer_engine::window_renderer::WindowRenderer::new` takes).
    /// `engine` is shared across every `Dx12Renderer` in the process —
    /// create one `RendererEngine` up front and pass it to each.
    pub fn new(
        name: impl Into<String>,
        engine: &RendererEngine,
        hwnd: isize,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let inner =
            WindowRenderer::new(engine, hwnd, width, height).map_err(Dx12RendererError::Create)?;
        Ok(Self {
            name: name.into(),
            inner,
        })
    }

    /// Call when the target window resizes.
    pub fn resize(&self, width: u32, height: u32) -> Result<()> {
        self.inner
            .resize(width, height)
            .map_err(Dx12RendererError::Resize)?;
        Ok(())
    }

    fn submit_yuv420p_frame(&self, frame: &ffmpeg::frame::Video) -> Result<()> {
        let plane = |index: usize| RawPlane {
            data: frame.data(index).as_ptr(),
            len: frame.data(index).len(),
            stride: frame.stride(index),
        };

        // Safety: `plane(0..3)` point into `frame`'s own buffers, which
        // outlive this call — `submit_yuv420p` only reads them before
        // returning.
        unsafe {
            self.inner
                .submit_yuv420p(plane(0), plane(1), plane(2), frame.width(), frame.height())
                .map_err(Dx12RendererError::Submit)?;
        }
        Ok(())
    }

    fn submit_d3d12_frame(&self, frame: Arc<ffmpeg::frame::Video>) -> Result<()> {
        let (texture_raw, fence_raw, fence_value) =
            d3d12va_texture(&frame).ok_or(Dx12RendererError::InvalidD3d12Frame)?;
        let width = frame.width();
        let height = frame.height();

        // Safety: `texture_raw`/`fence_raw` are borrowed raw COM pointers
        // — still owned by `frame`'s own hw frame pool reference, not by
        // us. `.clone()` (`AddRef`) gives us our own independently
        // ref-counted handle, valid for as long as *we* hold it,
        // regardless of what `frame`/ffmpeg later does with its copy.
        let (texture, fence) = unsafe {
            let texture = ID3D12Resource::from_raw_borrowed(&texture_raw)
                .expect("AVD3D12VAFrame.texture must not be null")
                .clone();
            let fence = ID3D12Fence::from_raw_borrowed(&fence_raw)
                .expect("AVD3D12VAFrame.sync_ctx.fence must not be null")
                .clone();
            (texture, fence)
        };

        // `frame` (an `Arc`) is what keeps the underlying D3D12 texture
        // memory from being recycled by the decoder's frame pool while
        // `renderer-engine` still has it queued to draw — independent of,
        // and in addition to, the `texture`/`fence` COM references above.
        unsafe {
            self.inner
                .submit_nv12_texture(texture, fence, fence_value, width, height, Box::new(frame))
                .map_err(Dx12RendererError::Submit)?;
        }
        Ok(())
    }
}

impl Element for Dx12Renderer {
    fn name(&self) -> &str {
        &self.name
    }
}

impl Sink for Dx12Renderer {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        let MediaBuffer::Video(frame) = buf else {
            return Ok(());
        };

        match frame.format() {
            ffmpeg::format::Pixel::YUV420P => self.submit_yuv420p_frame(&frame),
            ffmpeg::format::Pixel::D3D12 => self.submit_d3d12_frame(frame),
            other => Err(Dx12RendererError::UnsupportedFormat(other).into()),
        }
    }

    fn control(&mut self, _msg: ControlMsg) -> Result<()> {
        // Terminal, nothing to flush or forward — a paused/stopped window
        // just stops receiving new frames (see `Queue`'s worker loop) and
        // keeps showing whatever was submitted last.
        Ok(())
    }
}
