use std::{ffi::c_void, sync::Arc};

use crate::pp_log::{PpLog, pp_error, pp_info};
use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;
use windows::{Win32::Graphics::Direct3D12::ID3D12Device, core::Interface};

use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    pad::SrcPad,
    pool::UnboundObjectPool,
};

use crate::elements::filter::is_codec_drain_boundary;

/// Mirrors of the D3D12VA-specific structs from FFmpeg's
/// `libavutil/hwcontext_d3d12va.h` (as of FFmpeg n8.0), hand-written
/// because `ffmpeg-sys-next` doesn't bind that header — only the
/// type-agnostic `AVHWDeviceContext`/`AVHWFramesContext` are generated.
/// COM pointers are kept as `*mut c_void` rather than typed `windows-rs`
/// handles so this struct's layout depends only on pointer size, not on
/// `windows-rs`'s internal representation. If a future FFmpeg version
/// changes this header's layout, these silently go stale — there's no
/// compile-time check tying them together.
#[repr(C)]
struct AVD3D12VADeviceContext {
    device: *mut c_void,
    video_device: *mut c_void,
    lock: Option<unsafe extern "C" fn(lock_ctx: *mut c_void)>,
    unlock: Option<unsafe extern "C" fn(lock_ctx: *mut c_void)>,
    lock_ctx: *mut c_void,
}

#[repr(C)]
struct AVD3D12VASyncContext {
    fence: *mut c_void,
    event: *mut c_void,
    fence_value: u64,
}

#[repr(C)]
struct AVD3D12VAFrame {
    texture: *mut c_void,
    sync_ctx: AVD3D12VASyncContext,
}

/// Errors specific to `D3d12vaDecoder`. Converts into the crate-wide
/// `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum D3d12vaDecoderError {
    #[error("unsupported media type: {0:?} (D3D12VA decode is video-only)")]
    UnsupportedMediaType(ffmpeg::media::Type),

    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),

    #[error("failed to create D3D12VA hw device context (code {0})")]
    HwDeviceInit(i32),

    #[error(
        "decoder did not select the D3D12VA pixel format — hardware decode \
         unavailable for this stream/GPU/driver"
    )]
    HwAccelUnavailable,
}

/// Decodes one video stream's `Packet`s into GPU-resident `Video` frames
/// via D3D12VA hardware acceleration, instead of [`crate::elements::SwDecoder`]'s
/// plain libavcodec software path. A `Filter`, same shape as `SwDecoder`.
///
/// Frames this produces are still plain `MediaBuffer::Video` — nothing
/// downstream needs to change to receive them. `Pacer`/`Tee`/
/// `FrameCounter` only ever touch `.pts()` or match the enum variant, so
/// they work unmodified. Only [`crate::elements::D3d12Renderer`] cares:
/// it checks `frame.format()` and, for `Pixel::D3D12`, takes the
/// zero-copy path via `d3d12va_texture` instead of reading pixel bytes.
pub struct D3d12vaDecoder {
    pp_log: PpLog,
    name: Arc<str>,
    decoder: ffmpeg::decoder::Video,
    hw_device_ctx: *mut ffi::AVBufferRef,
    pad: SrcPad,
    /// Reused across every decoded frame — see [`UnboundObjectPool`]'s
    /// docs. The actual GPU texture behind a `Pixel::D3D12` frame is
    /// already pooled/recycled by ffmpeg's own hw frames context
    /// regardless of what this crate does, so the benefit here is
    /// smaller than for [`crate::elements::SwDecoder`]/
    /// [`crate::elements::SwScaler`] (just the small CPU-side `AVFrame`
    /// wrapper, not the texture) — but `MediaBuffer::Video` requires
    /// this either way, so there's no reason not to.
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
}

// SAFETY: `hw_device_ctx` is a heap-allocated FFmpeg buffer with no
// thread affinity; `decoder`'s own `Send` already covers the rest.
// `&mut self` on every method that touches it rules out concurrent
// access from multiple threads.
unsafe impl Send for D3d12vaDecoder {}

impl D3d12vaDecoder {
    /// The D3D12VA hardware context owns an independent COM reference to
    /// `device`, so the caller does not need to keep its handle alive. Pass
    /// the same underlying `ID3D12Device` your
    /// [`crate::elements::D3d12Renderer`]'s own
    /// [`crate::elements::D3d12FrameRenderer`] impl renders with (see
    /// that trait's own `device()`) so decoded frames land on the same
    /// device the renderer reads from — required for the zero-copy path
    /// to be valid at all; checked at render time, not just documented,
    /// via `D3d12Renderer`'s own device-mismatch guard.
    pub fn new(
        name: impl Into<String>,
        params: ffmpeg::codec::Parameters,
        device: &ID3D12Device,
    ) -> Result<Self, D3d12vaDecoderError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::D3d12vaDecoder, &name, None);

        let hw_device_ctx =
            unsafe { create_hw_device_ctx(device) }.map_err(D3d12vaDecoderError::HwDeviceInit)?;

        let mut context = match ffmpeg::codec::context::Context::from_parameters(params) {
            Ok(context) => context,
            Err(error) => {
                unsafe { free_buffer(hw_device_ctx) };
                return Err(error.into());
            }
        };
        if context.medium() != ffmpeg::media::Type::Video {
            unsafe { free_buffer(hw_device_ctx) };
            return Err(D3d12vaDecoderError::UnsupportedMediaType(context.medium()));
        }
        unsafe {
            let ctx_ptr = context.as_mut_ptr();
            (*ctx_ptr).hw_device_ctx = ffi::av_buffer_ref(hw_device_ctx);
            (*ctx_ptr).get_format = Some(get_format);
        }

        let decoder = match context.decoder().video() {
            Ok(decoder) => decoder,
            Err(error) => {
                unsafe { free_buffer(hw_device_ctx) };
                return Err(error.into());
            }
        };

        let pad = SrcPad::new(format!("{name}_src"));
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        pp_info!(pp_log: &pp_log, "opened: codec={:?}", decoder.id());
        Ok(Self {
            name,
            pp_log,
            decoder,
            hw_device_ctx,
            pad,
            pool,
        })
    }

    fn drain(&mut self) -> crate::error::Result<()> {
        let mut frame = self.pool.get();
        loop {
            match self.decoder.receive_frame(&mut frame) {
                Ok(()) => {
                    if frame.format() != ffmpeg::format::Pixel::D3D12 {
                        pp_error!(self, "decoder did not select the D3D12VA pixel format");
                        return Err(D3d12vaDecoderError::HwAccelUnavailable.into());
                    }
                    self.pad.push(MediaBuffer::Video(Arc::new(frame)))?;
                    frame = self.pool.get();
                }
                Err(error) if is_codec_drain_boundary(&error) => break,
                Err(error) => return Err(D3d12vaDecoderError::from(error).into()),
            }
        }
        Ok(())
    }
}

impl Element for D3d12vaDecoder {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::D3d12vaDecoder
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for D3d12vaDecoder {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for D3d12vaDecoder {
    fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
        match buf {
            MediaBuffer::Packet(packet) => {
                self.decoder
                    .send_packet(&*packet)
                    .inspect_err(|error| pp_error!(self, "send_packet failed: {error}"))
                    .map_err(D3d12vaDecoderError::from)?;
                self.drain()
            }
            MediaBuffer::Eos => {
                self.decoder
                    .send_eof()
                    .inspect_err(|error| pp_error!(self, "send_eof failed: {error}"))
                    .map_err(D3d12vaDecoderError::from)?;
                self.drain()?;
                self.pad.push(MediaBuffer::Eos)
            }
            other => {
                let _ = other;
                Ok(())
            }
        }
    }

    fn control(&mut self, msg: ControlMsg) -> crate::error::Result<()> {
        // `Stop`: no local reaction needed — see `SwDecoder::control`;
        // same reasoning applies to the hw device context, freed in
        // `Drop`.
        //
        // `Seek`: same reasoning as `SwDecoder::control` too — flush
        // leftover reference-frame state before decoding resumes from
        // the new position.
        if let ControlMsg::Seek(_) = msg {
            self.decoder.flush();
        }
        self.pad.control(msg)
    }
}

impl Drop for D3d12vaDecoder {
    fn drop(&mut self) {
        pp_info!(self, "dropped: freeing hw_device_ctx");
        unsafe { free_buffer(self.hw_device_ctx) };
    }
}

/// Shared with [`crate::elements::D3d12Upload`] and
/// [`crate::elements::D3d12Scaler`]. Returns a raw `AVERROR` code rather than
/// `D3d12vaDecoderError` so it doesn't tie that caller to this module's
/// own error type; each call site maps it into its own error variant.
pub(crate) unsafe fn create_hw_device_ctx(
    device: &ID3D12Device,
) -> Result<*mut ffi::AVBufferRef, i32> {
    unsafe {
        let buf = ffi::av_hwdevice_ctx_alloc(ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_D3D12VA);
        if buf.is_null() {
            return Err(-1);
        }

        let hw_device_ctx = (*buf).data as *mut ffi::AVHWDeviceContext;
        let d3d12_ctx = (*hw_device_ctx).hwctx as *mut AVD3D12VADeviceContext;
        // FFmpeg releases a caller-supplied device when this context is
        // freed, so transfer an independently owned COM reference.
        (*d3d12_ctx).device = device.clone().into_raw();

        let result = ffi::av_hwdevice_ctx_init(buf);
        if result < 0 {
            free_buffer(buf);
            return Err(result);
        }

        Ok(buf)
    }
}

/// Shared with the other D3D12 filters — see
/// [`create_hw_device_ctx`]'s own docs.
pub(crate) unsafe fn free_buffer(mut buf: *mut ffi::AVBufferRef) {
    unsafe { ffi::av_buffer_unref(&mut buf) };
}

/// Creates an NV12 D3D12VA frame pool tied to `hw_device_ctx`.
pub(crate) unsafe fn create_hw_frames_ctx(
    hw_device_ctx: *mut ffi::AVBufferRef,
    width: u32,
    height: u32,
    initial_pool_size: i32,
) -> Result<*mut ffi::AVBufferRef, i32> {
    unsafe {
        let buf = ffi::av_hwframe_ctx_alloc(hw_device_ctx);
        if buf.is_null() {
            return Err(-1);
        }
        let frames_ctx = (*buf).data as *mut ffi::AVHWFramesContext;
        (*frames_ctx).format = ffi::AVPixelFormat::AV_PIX_FMT_D3D12;
        (*frames_ctx).sw_format = ffi::AVPixelFormat::AV_PIX_FMT_NV12;
        (*frames_ctx).width = width as i32;
        (*frames_ctx).height = height as i32;
        (*frames_ctx).initial_pool_size = initial_pool_size;
        let result = ffi::av_hwframe_ctx_init(buf);
        if result < 0 {
            free_buffer(buf);
            return Err(result);
        }
        Ok(buf)
    }
}

unsafe extern "C" fn get_format(
    _ctx: *mut ffi::AVCodecContext,
    mut fmt: *const ffi::AVPixelFormat,
) -> ffi::AVPixelFormat {
    unsafe {
        while *fmt != ffi::AVPixelFormat::AV_PIX_FMT_NONE {
            if *fmt == ffi::AVPixelFormat::AV_PIX_FMT_D3D12 {
                return ffi::AVPixelFormat::AV_PIX_FMT_D3D12;
            }
            fmt = fmt.add(1);
        }
        ffi::AVPixelFormat::AV_PIX_FMT_NONE
    }
}

/// Extracts `(texture, fence, fence_value)` from a frame produced by
/// [`D3d12vaDecoder`] — `None` if `frame` isn't actually a D3D12VA frame.
/// `texture`/`fence` are borrowed raw `ID3D12Resource*`/`ID3D12Fence*`
/// pointers (still owned by `frame`'s own hw frame pool reference) — the
/// caller must `AddRef` (e.g. via `Interface::from_raw_borrowed(..).clone()`)
/// before holding onto them independently of `frame`.
pub(crate) fn d3d12va_texture(
    frame: &ffmpeg::frame::Video,
) -> Option<(*mut c_void, *mut c_void, u64)> {
    if frame.format() != ffmpeg::format::Pixel::D3D12 {
        return None;
    }
    unsafe {
        let ptr = frame.as_ptr();
        let d3d12_frame = &*((*ptr).data[0] as *const AVD3D12VAFrame);
        Some((
            d3d12_frame.texture,
            d3d12_frame.sync_ctx.fence,
            d3d12_frame.sync_ctx.fence_value,
        ))
    }
}

/// Updates the fence value belonging to a D3D12VA frame after a producer has
/// queued work and signalled the frame pool's own fence.
pub(crate) fn set_d3d12va_fence_value(frame: &mut ffmpeg::frame::Video, fence_value: u64) -> bool {
    if frame.format() != ffmpeg::format::Pixel::D3D12 {
        return false;
    }
    unsafe {
        let data = (*frame.as_mut_ptr()).data[0];
        if data.is_null() {
            return false;
        }
        (*(data as *mut AVD3D12VAFrame)).sync_ctx.fence_value = fence_value;
    }
    true
}
