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
    contract::{InputContract, MediaKind, MemoryDomain, PortContract},
    control::ControlMsg,
    element::{Element, ElementType, Sink, element_pp_log},
    elements::SubmitError,
    error::{D3d11SharedDeviceError, Result},
    platform::windows::{d3d11::protect_shared_device, d3d11va::d3d11va_texture},
};

/// What [`D3d11Renderer`] needs from an actual DX11 window/rendering
/// implementation — the D3D11 sibling of
/// `D3d12FrameRenderer`, deliberately **not** an impl of
/// that trait (it's documented as inherently D3D12-only). Unlike the D3D12
/// trait, neither submit method here takes a fence *or* a `keep_alive` —
/// see [`crate::elements::D3d11Renderer`]'s own docs on why a single
/// shared `ID3D11Device` needs no explicit GPU-side synchronization at
/// all, and this doc comment's own note below on why lifetime-keeping is
/// unnecessary too. Both paths here are zero-copy: everything in
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
///
/// It is handed a texture and nothing about its colour: an NV12 frame's
/// matrix and range stay with the frame. A window of the renderer's own,
/// where each frame is converted with its own, is what `D3d11WindowRenderer` is
/// for; this trait is for drawing into something else — a UI's own swap
/// chain, an offscreen target.
///
/// A successful submit must install the frame as the current presentation
/// content or enqueue its swap-chain presentation before returning. Pipeline
/// preroll treats that return as the terminal's presentation commitment; it
/// does not require the implementation to wait for physical scanout.
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
    /// `color` is what the frame says of its Y'CbCr, each part
    /// `Unspecified` where it says nothing; its
    /// [`yuv_to_rgb_rows`](crate::color::ColorDescription::yuv_to_rgb_rows)
    /// are the rows to convert it with, filling in what it leaves unsaid as
    /// this crate's own renderers do. A decoded HD stream is BT.709, and
    /// drawn with a fixed BT.601 matrix it comes out visibly wrong.
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
        color: crate::color::ColorDescription,
    ) -> std::result::Result<(), SubmitError>;

    /// Updates the presentation target dimensions.
    fn resize(&self, width: u32, height: u32) -> std::result::Result<(), SubmitError>;
}

/// Errors specific to `D3d11Renderer`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum D3d11RendererError {
    /// The caller-provided renderer rejected frame submission.
    #[error("failed to submit frame: {0:?}")]
    Submit(SubmitError),
    /// The caller-provided renderer rejected a size change.

    #[error("failed to resize: {0:?}")]
    Resize(SubmitError),
    /// The input frame is not backed by a D3D11 texture.

    #[error("D3d11Renderer only handles Pixel::D3D11 frames, got {0:?}")]
    UnsupportedFormat(ffmpeg::format::Pixel),
    /// A frame tagged as D3D11 contains no valid texture reference.

    #[error(
        "frame claimed the D3D11 pixel format but carries no texture — must \
         come from D3d11Upload/D3d11Decoder/DxgiCaptureSource's GPU mode"
    )]
    InvalidD3d11Frame,
    /// The texture uses a DXGI format unsupported by the renderer contract.

    #[error(
        "D3d11Renderer only draws DXGI_FORMAT_B8G8R8A8_UNORM or DXGI_FORMAT_NV12 textures, got {0:?}"
    )]
    UnsupportedTextureFormat(DXGI_FORMAT),
    /// The input texture belongs to another D3D11 device.

    #[error(
        "a Pixel::D3D11 frame's texture lives on a different ID3D11Device \
         than this D3d11Renderer was created with — every D3D11 element in \
         one pipeline must share exactly one device for zero-copy to be \
         valid"
    )]
    DeviceMismatch,
    /// The frame selects a texture-array slice outside the resource bounds.

    #[error("D3D11 texture array index {index} is outside ArraySize {array_size}")]
    InvalidArrayIndex {
        /// Invalid texture-array index.
        index: isize,
        /// Number of slices in the texture array.
        array_size: u32,
    },
    /// Inspecting the D3D11 texture or device failed.

    #[error("windows error: {0}")]
    Windows(#[from] windows::core::Error),

    /// The renderer's device cannot be shared across a pipeline's threads.
    /// Detected at construction, reported on the first frame — see
    /// [`D3d11Renderer::new`].
    #[error(transparent)]
    SharedDevice(#[from] D3d11SharedDeviceError),
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
/// immediate-context `ID3D11Multithread` protection serializes calls from
/// multiple threads. As long as every element funnels its GPU commands through
/// that one context, the driver executes them in submission order — no
/// separate sync object is needed. That protection is *off* by default, so
/// every element in this crate that is handed an `ID3D11Device` — this one
/// included — enables it and rejects `D3D11_CREATE_DEVICE_SINGLETHREADED`
/// rather than assuming some other element got there first. This only holds
/// because everything shares the *same* context; a second `ID3D11Device` in
/// the mix would need its own explicit synchronization, same as the D3D12
/// case.
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
    /// Why the renderer's device could not be made safe to share, if it
    /// could not. This constructor is infallible, and it is not worth
    /// breaking that for a condition every other element already reports —
    /// so the answer is kept and returned by the first `consume`, before a
    /// single frame has been submitted.
    shared_device_error: Option<D3d11SharedDeviceError>,
}

impl D3d11Renderer {
    /// `renderer` is whatever the caller's own [`D3d11FrameRenderer`]
    /// implementation is — already constructed and pointed at a real
    /// window/device by the time it gets here.
    ///
    /// A renderer is the far side of a `Queue` almost by definition, so its
    /// device gets the same multithread protection every other D3D11 element
    /// applies to the device it is handed. A device that refuses it fails the
    /// first `consume` rather than this call.
    pub fn new(name: impl Into<String>, renderer: Box<dyn D3d11FrameRenderer>) -> Self {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::D3d11Renderer, &name, None);
        let device = renderer.device();
        let shared_device_error = match protect_shared_device(&device) {
            Ok(_) => None,
            Err(error) => {
                pp_error!(pp_log: &pp_log, "device cannot be shared: {error}");
                Some(error)
            }
        };
        pp_info!(pp_log: &pp_log, "created");
        Self {
            name,
            pp_log,
            inner: renderer,
            device,
            shared_device_error,
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

    fn submit_d3d11_frame(&self, frame: &ffmpeg::frame::Video) -> Result<()> {
        let picture = check_d3d11_frame(frame, &self.device)?;
        let (width, height) = (frame.width(), frame.height());
        // No `keep_alive` to pass through here — see `D3d11FrameRenderer`'s
        // own docs on why D3D11's driver-deferred resource destruction
        // makes that unnecessary, unlike `D3d12Renderer`. `frame` itself
        // just drops normally once `consume` returns.
        match picture.format {
            // SAFETY: `check_d3d11_frame` validated the device, format and
            // array index; the cloned texture remains live for the call.
            DXGI_FORMAT_B8G8R8A8_UNORM => unsafe {
                self.inner
                    .submit_bgra_texture(picture.texture, picture.array_index, width, height)
                    .map_err(D3d11RendererError::Submit)?;
            },
            // SAFETY: the same validation applies to the NV12-specific submit
            // contract selected by this format arm.
            _ => unsafe {
                self.inner
                    .submit_nv12_texture(
                        picture.texture,
                        picture.array_index,
                        width,
                        height,
                        crate::color::ColorDescription::of(frame),
                    )
                    .map_err(D3d11RendererError::Submit)?;
            },
        }
        Ok(())
    }
}

/// A D3D11 frame found fit to draw on a device: its texture, the slice of
/// the texture array it is, and the texture's format — NV12 or BGRA, the two
/// a renderer here draws.
pub(crate) struct D3d11Picture {
    pub(crate) texture: ID3D11Texture2D,
    pub(crate) array_index: u32,
    pub(crate) format: DXGI_FORMAT,
}

/// Everything a D3D11 renderer checks of a frame before drawing it: that it
/// is a D3D11 frame at all, that its texture was made on `device` — drawing
/// another device's texture is invalid, not just wrong-looking — that its
/// slice is inside the texture array, and that its format is one a renderer
/// draws. Shared by [`D3d11Renderer`] and `D3d11WindowRenderer`.
pub(crate) fn check_d3d11_frame(
    frame: &ffmpeg::frame::Video,
    device: &ID3D11Device,
) -> std::result::Result<D3d11Picture, D3d11RendererError> {
    if frame.format() != ffmpeg::format::Pixel::D3D11 {
        return Err(D3d11RendererError::UnsupportedFormat(frame.format()));
    }
    let (texture_raw, index) =
        d3d11va_texture(frame).ok_or(D3d11RendererError::InvalidD3d11Frame)?;
    // SAFETY: `texture_raw` is a borrowed raw `ID3D11Texture2D*` — still
    // owned by `frame`'s own buffer reference, not by us. `.clone()`
    // (`AddRef`) gives an independently ref-counted handle, valid for as
    // long as it is held.
    let texture = unsafe { ID3D11Texture2D::from_raw_borrowed(&texture_raw) }
        .ok_or(D3d11RendererError::InvalidD3d11Frame)?
        .clone();

    // SAFETY: `texture` is a live cloned COM interface; `GetDevice` returns
    // an owned reference to the device that created it.
    let texture_device = unsafe { texture.GetDevice() }?;
    if texture_device.as_raw() != device.as_raw() {
        return Err(D3d11RendererError::DeviceMismatch);
    }

    let mut desc = Default::default();
    // SAFETY: `desc` is a live, correctly typed out-parameter for the live
    // texture.
    unsafe { texture.GetDesc(&mut desc) };
    if index < 0 || index as u64 >= u64::from(desc.ArraySize) {
        return Err(D3d11RendererError::InvalidArrayIndex {
            index,
            array_size: desc.ArraySize,
        });
    }
    match desc.Format {
        DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_NV12 => Ok(D3d11Picture {
            texture,
            array_index: index as u32,
            format: desc.Format,
        }),
        other => Err(D3d11RendererError::UnsupportedTextureFormat(other)),
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
    /// Presents a device texture; nothing else has a path to the swap chain.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::D3d11)
                .with_layouts(crate::contract::PixelLayoutSet::NV12_OR_BGRA),
        )
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        if let Some(error) = &self.shared_device_error {
            return Err(error.clone().into());
        }
        let MediaBuffer::Video(frame) = buf else {
            return Ok(());
        };

        self.submit_d3d11_frame(&frame)
            .inspect_err(|error| pp_error!(self, "submit_d3d11_frame failed: {error}"))
    }

    fn control(&mut self, _msg: ControlMsg) -> Result<()> {
        // Terminal, nothing to flush or forward — same reasoning as
        // `D3d12Renderer::control`.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::{color::ColorDescription, elements::D3d11Upload, repeat::PerFrameTransform};

    /// Takes what it is handed and keeps the colour an NV12 frame came with.
    struct Recording {
        device: ID3D11Device,
        color: Arc<Mutex<Option<ColorDescription>>>,
    }

    impl D3d11FrameRenderer for Recording {
        fn device(&self) -> ID3D11Device {
            self.device.clone()
        }

        unsafe fn submit_bgra_texture(
            &self,
            _texture: ID3D11Texture2D,
            _array_index: u32,
            _width: u32,
            _height: u32,
        ) -> std::result::Result<(), SubmitError> {
            Ok(())
        }

        unsafe fn submit_nv12_texture(
            &self,
            _texture: ID3D11Texture2D,
            _array_index: u32,
            _width: u32,
            _height: u32,
            color: ColorDescription,
        ) -> std::result::Result<(), SubmitError> {
            *self.color.lock().unwrap() = Some(color);
            Ok(())
        }

        fn resize(&self, _width: u32, _height: u32) -> std::result::Result<(), SubmitError> {
            Ok(())
        }
    }

    /// What an NV12 frame says of its colour reaches the presenter with it,
    /// for it to draw the frame in its own colours rather than a fixed
    /// matrix's.
    #[test]
    fn a_frames_colour_reaches_the_presenter() {
        let Some(gpu) = crate::test_support::try_d3d11_gpu() else {
            return;
        };
        let mut picture = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, 64, 64);
        ColorDescription::BT709_LIMITED.describe(&mut picture);
        let MediaBuffer::Video(picture) = MediaBuffer::video(picture) else {
            unreachable!()
        };
        let texture = D3d11Upload::new("upload", &gpu)
            .transform(&picture)
            .expect("the frame uploads");

        let color = Arc::new(Mutex::new(None));
        let mut renderer = D3d11Renderer::new(
            "renderer",
            Box::new(Recording {
                device: gpu.device().clone(),
                color: color.clone(),
            }),
        );
        renderer
            .consume(MediaBuffer::Video(texture))
            .expect("the frame is presented");
        assert_eq!(
            *color.lock().unwrap(),
            Some(ColorDescription::BT709_LIMITED)
        );
    }
}
