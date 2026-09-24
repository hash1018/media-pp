use std::{any::Any, sync::Arc};

use crate::pp_log::{PpLog, pp_error, pp_info};
use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;
use windows::{
    Win32::Graphics::Direct3D12::{ID3D12Device, ID3D12Fence, ID3D12Resource},
    core::Interface,
};

use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, PortContract},
    control::ControlMsg,
    element::{Element, ElementType, Sink, element_pp_log},
    elements::SubmitError,
    error::Result,
    platform::windows::d3d12va::d3d12va_texture,
    pool::UnboundObjectPoolRef,
};

/// What [`D3d12Renderer`] needs from an actual DX12 window/rendering
/// implementation — deliberately the *only* thing this crate knows about
/// D3D12 rendering. `submit_nv12_texture` takes
/// `ID3D12Resource`/`ID3D12Fence` directly, so this trait (and any
/// element built on it) is inherently D3D12-only — a Vulkan/CUDA renderer
/// would need its own trait, not an impl of this one. `D3d12Renderer`
/// itself only depends on this trait (plus the `windows` COM types the
/// zero-copy path needs to pass through) — not on `renderer_engine` or
/// any other concrete rendering crate. A caller wanting to actually
/// render implements this for its own window/rendering stack; this
/// repository's examples use `examples/render/render_common` for that
/// implementation, outside the `media-pp` crate itself — unpublished, so
/// a program of its own copies what it needs from there rather than
/// depending on it.
///
/// A successful submit must install the frame as the current presentation
/// content or enqueue its swap-chain presentation before returning. Pipeline
/// preroll treats that return as the terminal's presentation commitment; it
/// does not require the implementation to wait for physical scanout.
pub trait D3d12FrameRenderer: Send {
    /// The `ID3D12Device` this implementation actually renders/submits
    /// with. [`D3d12Renderer`] reads this once at construction to guard
    /// [`D3d12FrameRenderer::submit_nv12_texture`]'s zero-copy path: a
    /// texture from a different device is invalid to draw from at all
    /// (not just wrong-looking), so it's checked against this rather than
    /// trusted.
    fn device(&self) -> ID3D12Device;

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

    /// Updates the presentation target dimensions.
    fn resize(&self, width: u32, height: u32) -> std::result::Result<(), SubmitError>;
}

/// Errors specific to `D3d12Renderer`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum D3d12RendererError {
    /// The caller-provided renderer rejected frame submission.
    #[error("failed to submit frame: {0:?}")]
    Submit(SubmitError),
    /// The caller-provided renderer rejected a size change.

    #[error("failed to resize: {0:?}")]
    Resize(SubmitError),
    /// The input is neither CPU YUV420P nor a D3D12 hardware frame.

    #[error(
        "D3d12Renderer only handles YUV420P frames (CPU) or D3D12 frames \
         (from D3d12Decoder), got {0:?}"
    )]
    UnsupportedFormat(ffmpeg::format::Pixel),
    /// A frame tagged as D3D12 lacks its texture or synchronization payload.

    #[error(
        "frame claimed the D3D12 pixel format but has no AVD3D12VAFrame \
         payload — must come from D3d12Decoder"
    )]
    InvalidD3d12Frame,
    /// The input texture belongs to another D3D12 device.

    #[error(
        "a Pixel::D3D12 frame's texture lives on a different ID3D12Device \
         than this D3d12Renderer was created with — the producer \
         (D3d12Decoder/D3d12Upload) and the D3d12FrameRenderer impl \
         must share the same device for zero-copy to be valid"
    )]
    DeviceMismatch,
}

/// Terminal sink that submits decoded video frames to a caller-supplied
/// [`D3d12FrameRenderer`]. Only built with the `d3d12` feature —
/// every consumer that doesn't need to render to a window pulls in
/// neither this nor the `windows` dependency it needs for the zero-copy
/// path.
///
/// Takes `Pixel::D3D12` frames only, drawn zero-copy straight from
/// whatever produced the texture via
/// `D3d12FrameRenderer::submit_nv12_texture` — a `D3d12Decoder`'s output,
/// or a [`crate::elements::D3d12Upload`] for a CPU-decoded stream. That
/// upload is the one place a system-memory frame reaches the GPU, rather
/// than this sink keeping a second, separate CPU path of its own; see
/// [`crate::contract`] for how a chain missing it is refused when the
/// branch is built.
pub struct D3d12Renderer {
    pp_log: PpLog,
    name: Arc<str>,
    inner: Box<dyn D3d12FrameRenderer>,
    /// Captured once from `inner.device()` at construction — the
    /// reference `submit_d3d12_frame` checks every zero-copy frame's
    /// actual device against. Fetched from `inner` itself rather than
    /// taken as a separate constructor parameter: a second
    /// independently-supplied device would just be another value the
    /// caller could get wrong, proving nothing about what `inner` really
    /// renders with.
    device: ID3D12Device,
}

impl D3d12Renderer {
    /// `renderer` is whatever the caller's own [`D3d12FrameRenderer`]
    /// implementation is — already constructed and pointed at a real
    /// window/device by the time it gets here. This element doesn't
    /// create or own a window itself.
    pub fn new(name: impl Into<String>, renderer: Box<dyn D3d12FrameRenderer>) -> Self {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::D3d12Renderer, &name, None);
        pp_info!(pp_log: &pp_log, "created");
        let device = renderer.device();
        Self {
            name,
            pp_log,
            inner: renderer,
            device,
        }
    }

    /// Call when the target window resizes.
    pub fn resize(&self, width: u32, height: u32) -> Result<()> {
        self.inner
            .resize(width, height)
            .inspect_err(|error| pp_error!(self, "resize failed: {error:?}"))
            .map_err(D3d12RendererError::Resize)?;
        pp_info!(self, "resized: {width}x{height}");
        Ok(())
    }

    fn submit_d3d12_frame(
        &self,
        frame: Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    ) -> Result<()> {
        let picture = check_d3d12_frame(&frame, &self.device)?;
        let (width, height) = (frame.width(), frame.height());
        // `frame` (an `Arc`) is what keeps the underlying D3D12 texture
        // memory from being recycled by the decoder's frame pool while
        // the renderer still has it queued to draw — independent of, and
        // in addition to, the `texture`/`fence` COM references.
        // SAFETY: `check_d3d12_frame` established the resource device, and
        // the fence is the producer's own. The boxed frame keeps the pooled
        // allocation alive until the renderer releases it.
        unsafe {
            self.inner
                .submit_nv12_texture(
                    picture.texture,
                    picture.fence,
                    picture.fence_value,
                    width,
                    height,
                    Box::new(frame),
                )
                .map_err(D3d12RendererError::Submit)?;
        }
        Ok(())
    }
}

/// A D3D12 frame found fit to draw on a device: its texture, and the fence
/// value its producer signals once the texture is written.
pub(crate) struct D3d12Picture {
    pub(crate) texture: ID3D12Resource,
    pub(crate) fence: ID3D12Fence,
    pub(crate) fence_value: u64,
}

/// Everything a D3D12 renderer checks of a frame before drawing it: that it
/// is a D3D12 frame at all, and that its texture was made on `device` —
/// the producer and the renderer are separate constructions that only
/// *should* share one, and drawing another device's texture is invalid, not
/// just wrong-looking. Shared by [`D3d12Renderer`] and `D3d12WindowRenderer`.
pub(crate) fn check_d3d12_frame(
    frame: &ffmpeg::frame::Video,
    device: &ID3D12Device,
) -> std::result::Result<D3d12Picture, D3d12RendererError> {
    if frame.format() != ffmpeg::format::Pixel::D3D12 {
        return Err(D3d12RendererError::UnsupportedFormat(frame.format()));
    }
    let (texture_raw, fence_raw, fence_value) =
        d3d12va_texture(frame).ok_or(D3d12RendererError::InvalidD3d12Frame)?;
    // SAFETY: `texture_raw`/`fence_raw` are borrowed raw COM pointers —
    // still owned by `frame`'s own hw frame pool reference, not by us.
    // `.clone()` (`AddRef`) gives our own independently ref-counted handles,
    // valid for as long as they are held.
    let (texture, fence) = unsafe {
        (
            ID3D12Resource::from_raw_borrowed(&texture_raw)
                .ok_or(D3d12RendererError::InvalidD3d12Frame)?
                .clone(),
            ID3D12Fence::from_raw_borrowed(&fence_raw)
                .ok_or(D3d12RendererError::InvalidD3d12Frame)?
                .clone(),
        )
    };
    let mut texture_device: Option<ID3D12Device> = None;
    // SAFETY: `texture` is live and `texture_device` is a correctly typed
    // out-parameter for the resource's creating device.
    unsafe { texture.GetDevice(&mut texture_device) }
        .map_err(|_| D3d12RendererError::DeviceMismatch)?;
    let texture_device = texture_device.ok_or(D3d12RendererError::DeviceMismatch)?;
    if texture_device.as_raw() != device.as_raw() {
        return Err(D3d12RendererError::DeviceMismatch);
    }
    Ok(D3d12Picture {
        texture,
        fence,
        fence_value,
    })
}

impl Element for D3d12Renderer {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::D3d12Renderer
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Sink for D3d12Renderer {
    /// Presents a device resource; a system-memory frame reaches the GPU
    /// through [`crate::elements::D3d12Upload`], not through a second
    /// path inside this sink.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::D3d12)
                .with_layouts(crate::contract::PixelLayoutSet::NV12),
        )
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        let MediaBuffer::Video(frame) = buf else {
            return Ok(());
        };

        self.submit_d3d12_frame(frame)
            .inspect_err(|error| pp_error!(self, "submit_d3d12_frame failed: {error}"))
    }

    fn control(&mut self, _msg: ControlMsg) -> Result<()> {
        // Terminal, nothing to flush or forward — a paused/stopped window
        // just stops receiving new frames (see `Queue`'s worker loop) and
        // keeps showing whatever was submitted last.
        Ok(())
    }
}
