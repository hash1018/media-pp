use ffmpeg_next as ffmpeg;
use renderer_engine::{
    engine::RendererEngine,
    window_renderer::{RawPlane, SubmitError, WindowRenderer},
};
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    element::{Element, Sink},
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

    #[error("Dx12Renderer only handles YUV420P frames, got {0:?} — decode into that format first")]
    UnsupportedFormat(ffmpeg::format::Pixel),
}

/// Terminal sink that submits decoded video frames to a native window via
/// [`renderer_engine`]'s DX12 `WindowRenderer`. Only built with the
/// `dx12-renderer` feature — every consumer that doesn't need to render to
/// a window pulls in neither this nor the DX12 dependency it wraps.
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
        if frame.format() != ffmpeg::format::Pixel::YUV420P {
            return Err(Dx12RendererError::UnsupportedFormat(frame.format()).into());
        }

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
}
