use std::sync::Arc;

use crate::pp_log::{PpLog, pp_error, pp_info};
use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;
use windows::{
    Win32::Graphics::{
        Direct3D11::{ID3D11Device, ID3D11Texture2D},
        Dxgi::Common::{DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_NV12},
    },
    core::Interface,
};

use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Element, ElementType, Sink, element_pp_log},
    elements::SubmitError,
    error::Result,
    platform::windows::d3d11va::d3d11va_texture,
    pool::UnboundObjectPoolRef,
};

/// What [`D3d11Renderer`] needs from an actual DX11 window/rendering
/// implementation — the D3D11 sibling of
/// `D3d12FrameRenderer`, deliberately **not** an impl of
/// that trait (it's documented as inherently D3D12-only). Unlike the D3D12
/// trait, neither submit method here takes a fence *or* a `keep_alive` —
/// see [`crate::elements::D3d11Renderer`]'s own docs on why a single
/// shared `ID3D11Device` needs no explicit GPU-side synchronization at
/// all, and this doc comment's own note below on why lifetime-keeping is
/// unnecessary too. Both paths here are zero-copy (no CPU-upload method
/// the way `D3d12FrameRenderer::submit_yuv420p` is one): everything in
/// this crate's D3D11 stack already produces GPU-resident `Pixel::D3D11`
/// textures (see [`crate::elements::D3d11Upload`]/
/// `DxgiCaptureSource`'s GPU capture mode/
/// [`crate::elements::D3d11Decoder`]), so there's no CPU-side pixel data
/// left to upload by the time a frame reaches here.
///
/// No `keep_alive` parameter (unlike `D3d12FrameRenderer::submit_nv12_texture`):
/// D3D11's own COM+driver contract already defers actually freeing a
/// resource's GPU memory until the GPU has finished any outstanding work
/// that reads it, *regardless* of when the app-level reference count hits
/// zero — this is precisely the abstraction D3D12 (deliberately) doesn't
/// provide, which is why that side needs the caller to keep the source
/// frame alive by hand via an explicit fence. Here, once
/// `D3d11Renderer::submit_d3d11_frame`'s local `texture` clone (and
/// whatever `Arc<UnboundObjectPoolRef<..>>` produced it) drops, the
/// runtime — not this crate — is what keeps the actual texture memory
/// valid for as long as the GPU still needs it.
pub trait D3d11FrameRenderer: Send {
    /// The `ID3D11Device` this implementation actually renders/submits
    /// with. [`D3d11Renderer`] reads this once at construction to guard
    /// every submit against a texture from a different device — same
    /// reasoning as `D3d12Renderer`'s own device-mismatch guard.
    fn device(&self) -> ID3D11Device;

    /// `texture` is a plain packed-BGRA surface — from
    /// `DxgiCaptureSource`'s GPU capture mode or
    /// [`crate::elements::D3d11Upload`] fed a BGRA source. `array_index` is
    /// always `0` for these producers (neither ever builds an array
    /// texture) — see `submit_nv12_texture`'s own docs on why it's a
    /// parameter here at all.
    ///
    /// # Safety
    /// `texture` must be a valid `ID3D11Texture2D` on the same
    /// `ID3D11Device` this renderer was created with, `DXGI_FORMAT_B8G8R8A8_UNORM`,
    /// with `array_index < ` its `ArraySize`.
    unsafe fn submit_bgra_texture(
        &self,
        texture: ID3D11Texture2D,
        array_index: u32,
        width: u32,
        height: u32,
    ) -> std::result::Result<(), SubmitError>;

    /// `texture` is an NV12 surface — from [`crate::elements::D3d11Decoder`]
    /// or [`crate::elements::D3d11Upload`] fed an NV12 source. `array_index`
    /// is which slice of `texture` this frame actually is: libavcodec's own
    /// D3D11VA hwaccel decode pools frames as slices of one shared **array**
    /// texture (unlike `D3d11Upload`, which always builds a fresh
    /// non-array, single-slice texture per frame — `array_index` is always
    /// `0` there) — see `d3d11va_texture`'s own docs.
    ///
    /// # Safety
    /// `texture` must be a valid `ID3D11Texture2D` on the same
    /// `ID3D11Device` this renderer was created with, `DXGI_FORMAT_NV12`,
    /// with `array_index < ` its `ArraySize`.
    unsafe fn submit_nv12_texture(
        &self,
        texture: ID3D11Texture2D,
        array_index: u32,
        width: u32,
        height: u32,
    ) -> std::result::Result<(), SubmitError>;

    fn resize(&self, width: u32, height: u32) -> std::result::Result<(), SubmitError>;
}

/// Errors specific to `D3d11Renderer`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum D3d11RendererError {
    #[error("failed to submit frame: {0:?}")]
    Submit(SubmitError),

    #[error("failed to resize: {0:?}")]
    Resize(SubmitError),

    #[error("D3d11Renderer only handles Pixel::D3D11 frames, got {0:?}")]
    UnsupportedFormat(ffmpeg::format::Pixel),

    #[error(
        "frame claimed the D3D11 pixel format but carries no texture — must \
         come from D3d11Upload/D3d11Decoder/DxgiCaptureSource's GPU mode"
    )]
    InvalidD3d11Frame,

    #[error(
        "D3d11Renderer only draws DXGI_FORMAT_B8G8R8A8_UNORM or DXGI_FORMAT_NV12 textures, got {0:?}"
    )]
    UnsupportedTextureFormat(DXGI_FORMAT),

    #[error(
        "a Pixel::D3D11 frame's texture lives on a different ID3D11Device \
         than this D3d11Renderer was created with — every D3D11 element in \
         one pipeline must share exactly one device for zero-copy to be \
         valid"
    )]
    DeviceMismatch,

    #[error("D3D11 texture array index {index} is outside ArraySize {array_size}")]
    InvalidArrayIndex { index: isize, array_size: u32 },

    #[error("windows error: {0}")]
    Windows(#[from] windows::core::Error),
}

/// Terminal sink that submits `Pixel::D3D11` video frames to a
/// caller-supplied [`D3d11FrameRenderer`] — the D3D11 sibling of
/// `D3d12Renderer`. Only built with the
/// `d3d11` feature.
///
/// Every producer in this crate's D3D11 stack
/// ([`crate::elements::D3d11Upload`], [`crate::elements::D3d11Decoder`],
/// `DxgiCaptureSource`'s GPU capture mode) is meant to
/// share **one** `ID3D11Device` (and its one immediate context) with
/// whatever [`D3d11FrameRenderer`] impl this wraps. That single-context
/// requirement is what makes zero-copy here need **no explicit fence**,
/// unlike `D3d12Renderer`'s `submit_nv12_texture`
/// (which needs one because the D3D12 decoder and renderer are genuinely
/// different devices/queues with nothing else to serialize them): an
/// `ID3D11Device` created without `D3D11_CREATE_DEVICE_SINGLETHREADED` has
/// its immediate context auto-serialized by the runtime across threads,
/// and as long as every element funnels its GPU commands through that one
/// context, the driver executes them in submission order — no separate
/// sync object needed. This only holds because everything shares the
/// *same* context; a second `ID3D11Device` in the mix would need its own
/// explicit synchronization, same as the D3D12 case.
///
/// Dispatches on the *texture's own* `DXGI_FORMAT` (via `GetDesc`), not on
/// any extra tag carried by the frame. `D3d11Upload` and GPU screen capture
/// wrap manually-created textures, while `D3d11Decoder` receives textures
/// from FFmpeg's D3D11VA frame pool; reading the actual texture description
/// gives all of those producer paths one reliable source of truth for the
/// pixel layout.
pub struct D3d11Renderer {
    pp_log: PpLog,
    name: Arc<str>,
    inner: Box<dyn D3d11FrameRenderer>,
    /// Captured once from `inner.device()` at construction — see
    /// `D3d12Renderer`'s own `device` field docs for why (fetched from
    /// `inner` itself rather than a separate constructor parameter).
    device: ID3D11Device,
}

impl D3d11Renderer {
    /// `renderer` is whatever the caller's own [`D3d11FrameRenderer`]
    /// implementation is — already constructed and pointed at a real
    /// window/device by the time it gets here.
    pub fn new(name: impl Into<String>, renderer: Box<dyn D3d11FrameRenderer>) -> Self {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::D3d11Renderer, &name, None);
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
            .map_err(D3d11RendererError::Resize)?;
        pp_info!(self, "resized: {width}x{height}");
        Ok(())
    }

    fn submit_d3d11_frame(
        &self,
        frame: Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    ) -> Result<()> {
        let (texture_raw, index) =
            d3d11va_texture(&frame).ok_or(D3d11RendererError::InvalidD3d11Frame)?;
        let width = frame.width();
        let height = frame.height();

        // SAFETY: `texture_raw` is a borrowed raw `ID3D11Texture2D*` —
        // still owned by `frame`'s own buffer reference, not by us.
        // `.clone()` (`AddRef`) gives us an independently ref-counted
        // handle, valid for as long as we hold it.
        let texture = unsafe {
            ID3D11Texture2D::from_raw_borrowed(&texture_raw)
                .expect("D3d11 frame's texture pointer must not be null")
                .clone()
        };

        // Same reasoning as `D3d12Renderer::submit_d3d12_frame`'s own
        // device check: the producer and `self.inner` are independent
        // constructions that only *should* share a device by convention —
        // verify it.
        // SAFETY: `texture` is a live cloned COM interface; `GetDevice`
        // returns an owned reference to the device that created it.
        let texture_device = unsafe { texture.GetDevice() }.map_err(D3d11RendererError::from)?;
        if texture_device.as_raw() != self.device.as_raw() {
            return Err(D3d11RendererError::DeviceMismatch.into());
        }

        let mut desc = Default::default();
        // SAFETY: `desc` is a live, correctly typed out-parameter for the
        // live texture.
        unsafe { texture.GetDesc(&mut desc) };
        if index < 0 || index as u64 >= u64::from(desc.ArraySize) {
            let error = D3d11RendererError::InvalidArrayIndex {
                index,
                array_size: desc.ArraySize,
            };
            pp_error!(self, "{error}");
            return Err(error.into());
        }
        let array_index = index as u32;

        // No `keep_alive` to pass through here — see `D3d11FrameRenderer`'s
        // own docs on why D3D11's driver-deferred resource destruction
        // makes that unnecessary, unlike `D3d12Renderer`. `frame` itself
        // just drops normally at the end of this function.
        match desc.Format {
            // SAFETY: device, format, dimensions, and array index were all
            // validated above; the cloned texture remains live for the call.
            DXGI_FORMAT_B8G8R8A8_UNORM => unsafe {
                self.inner
                    .submit_bgra_texture(texture, array_index, width, height)
                    .map_err(D3d11RendererError::Submit)?;
            },
            // SAFETY: the same validation applies to the NV12-specific submit
            // contract selected by this format arm.
            DXGI_FORMAT_NV12 => unsafe {
                self.inner
                    .submit_nv12_texture(texture, array_index, width, height)
                    .map_err(D3d11RendererError::Submit)?;
            },
            other => return Err(D3d11RendererError::UnsupportedTextureFormat(other).into()),
        }
        Ok(())
    }
}

impl Element for D3d11Renderer {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::D3d11Renderer
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Sink for D3d11Renderer {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        let MediaBuffer::Video(frame) = buf else {
            return Ok(());
        };

        if frame.format() != ffmpeg::format::Pixel::D3D11 {
            let format = frame.format();
            pp_error!(self, "unsupported pixel format: {format:?}");
            return Err(D3d11RendererError::UnsupportedFormat(format).into());
        }
        self.submit_d3d11_frame(frame)
            .inspect_err(|error| pp_error!(self, "submit_d3d11_frame failed: {error}"))
    }

    fn control(&mut self, _msg: ControlMsg) -> Result<()> {
        // Terminal, nothing to flush or forward — same reasoning as
        // `D3d12Renderer::control`.
        Ok(())
    }
}
