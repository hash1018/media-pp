use std::{any::Any, sync::Arc};

use ffmpeg_next as ffmpeg;
use rust_hlog::{HLog, herror, hinfo};
use thiserror::Error as ThisError;
use windows::{
    Win32::Graphics::Direct3D12::{ID3D12Fence, ID3D12Resource},
    core::Interface,
};

use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Element, ElementType, Sink, element_hlog},
    elements::filter::decoder::d3d12va_decoder::d3d12va_texture,
    error::Result,
    pool::UnboundObjectPoolRef,
};

/// One CPU-resident image plane — data pointer, byte length, and row
/// stride. Deliberately a plain, GPU-vendor-agnostic struct (no
/// dependency on any particular rendering crate's own type) so
/// [`FrameRenderer`] implementors on the caller's side don't need this
/// crate to know anything about their concrete rendering setup.
#[derive(Clone, Copy)]
pub struct RawPlane {
    pub data: *const u8,
    pub len: usize,
    pub stride: usize,
}

/// Errors a [`FrameRenderer`] implementation can report. Mirrors the
/// shape of `renderer_engine::window_renderer::SubmitError` (the crate
/// `Dx12Renderer` was originally built directly against) without this
/// crate depending on that one — see [`FrameRenderer`]'s own docs on why
/// that dependency was pushed out to the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitError {
    NullBuffer,
    InvalidFrame,
    NoFreeSlot,
    RendererStopped,
    RenderFailed,
    /// The GPU device is no longer valid (driver reset/removal). Recovery
    /// requires recreating the whole rendering setup, not just retrying —
    /// same meaning as upstream `renderer_engine`'s own variant.
    DeviceRemoved,
}

/// What [`Dx12Renderer`] needs from an actual DX12 window/rendering
/// implementation — deliberately the *only* thing this crate knows about
/// GPU-vendor-specific rendering. `Dx12Renderer` itself only depends on
/// this trait (plus the `windows` COM types the zero-copy path needs to
/// pass through) — not on `renderer_engine` or any other concrete
/// rendering crate. A caller wanting to actually render implements this
/// for whatever they're using (e.g. a small newtype wrapping
/// `renderer_engine::window_renderer::WindowRenderer`) in their own
/// example/application code, not in this crate.
pub trait FrameRenderer: Send {
    /// # Safety
    /// All plane pointers must be readable for the given length and
    /// remain valid until this call returns.
    unsafe fn submit_yuv420p(
        &self,
        y: RawPlane,
        u: RawPlane,
        v: RawPlane,
        width: u32,
        height: u32,
    ) -> std::result::Result<(), SubmitError>;

    /// # Safety
    /// `texture` must be a valid `ID3D12Resource` on the same
    /// `ID3D12Device` this renderer was created with, laid out as NV12.
    /// `fence` must only reach `fence_value` once the GPU work that
    /// produced `texture`'s contents has completed.
    unsafe fn submit_nv12_texture(
        &self,
        texture: ID3D12Resource,
        fence: ID3D12Fence,
        fence_value: u64,
        width: u32,
        height: u32,
        keep_alive: Box<dyn Any + Send>,
    ) -> std::result::Result<(), SubmitError>;

    fn resize(&self, width: u32, height: u32) -> std::result::Result<(), SubmitError>;
}

/// Errors specific to `Dx12Renderer`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum Dx12RendererError {
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

/// Terminal sink that submits decoded video frames to a caller-supplied
/// [`FrameRenderer`]. Only built with the `dx12-renderer` feature — every
/// consumer that doesn't need to render to a window pulls in neither this
/// nor the `windows` dependency it needs for the zero-copy path.
///
/// Handles two kinds of input, dispatched on `frame.format()`:
///   - `Pixel::YUV420P`: CPU-decoded (e.g. from `SwDecoder`) — copies
///     pixel bytes to the GPU via `FrameRenderer::submit_yuv420p`.
///   - `Pixel::D3D12`: GPU-decoded (from `D3d12vaDecoder`) — zero-copy,
///     draws straight from the decoder's own texture via
///     `FrameRenderer::submit_nv12_texture`.
#[rust_hlog::hlog]
pub struct Dx12Renderer {
    name: Arc<str>,
    inner: Box<dyn FrameRenderer>,
}

impl Dx12Renderer {
    /// `renderer` is whatever the caller's own [`FrameRenderer`]
    /// implementation is — already constructed and pointed at a real
    /// window/device by the time it gets here. This element doesn't
    /// create or own a window itself.
    pub fn new(name: impl Into<String>, renderer: Box<dyn FrameRenderer>) -> Self {
        let name: Arc<str> = name.into().into();
        let hlog = element_hlog(ElementType::Dx12Renderer, &name, None);
        hinfo!(hlog: &hlog, "created");
        Self {
            name,
            hlog,
            inner: renderer,
        }
    }

    /// Call when the target window resizes.
    pub fn resize(&self, width: u32, height: u32) -> Result<()> {
        self.inner
            .resize(width, height)
            .inspect_err(|error| herror!(self, "resize failed: {error:?}"))
            .map_err(Dx12RendererError::Resize)?;
        hinfo!(self, "resized: {width}x{height}");
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

    fn submit_d3d12_frame(
        &self,
        frame: Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    ) -> Result<()> {
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
        // the renderer still has it queued to draw — independent of, and
        // in addition to, the `texture`/`fence` COM references above.
        unsafe {
            self.inner
                .submit_nv12_texture(texture, fence, fence_value, width, height, Box::new(frame))
                .map_err(Dx12RendererError::Submit)?;
        }
        Ok(())
    }
}

impl Element for Dx12Renderer {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::Dx12Renderer
    }

    fn hlog(&self) -> &HLog {
        &self.hlog
    }

    fn hlog_mut(&mut self) -> &mut HLog {
        &mut self.hlog
    }
}

impl Sink for Dx12Renderer {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        let MediaBuffer::Video(frame) = buf else {
            return Ok(());
        };

        match frame.format() {
            ffmpeg::format::Pixel::YUV420P => self
                .submit_yuv420p_frame(&frame)
                .inspect_err(|error| herror!(self, "submit_yuv420p_frame failed: {error}")),
            ffmpeg::format::Pixel::D3D12 => self
                .submit_d3d12_frame(frame)
                .inspect_err(|error| herror!(self, "submit_d3d12_frame failed: {error}")),
            other => {
                herror!(self, "unsupported pixel format: {other:?}");
                Err(Dx12RendererError::UnsupportedFormat(other).into())
            }
        }
    }

    fn control(&mut self, _msg: ControlMsg) -> Result<()> {
        // Terminal, nothing to flush or forward — a paused/stopped window
        // just stops receiving new frames (see `Queue`'s worker loop) and
        // keeps showing whatever was submitted last.
        Ok(())
    }
}
