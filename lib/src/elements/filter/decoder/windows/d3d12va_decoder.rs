use std::sync::Arc;

use crate::pp_log::{PpLog, pp_error, pp_info};
use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;
use windows::Win32::Graphics::Direct3D12::ID3D12Device;

use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    pad::SrcPad,
    platform::{ffmpeg::AvBufferRef, windows::d3d12va::create_hw_device_ctx},
    pool::UnboundObjectPool,
};

use crate::elements::filter::is_codec_drain_boundary;

/// Errors specific to `D3d12Decoder`. Converts into the crate-wide
/// `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum D3d12DecoderError {
    /// The selected stream is not video.
    #[error("unsupported media type: {0:?} (D3D12VA decode is video-only)")]
    UnsupportedMediaType(ffmpeg::media::Type),
    /// FFmpeg rejected decoder or packet/frame processing.

    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),
    /// FFmpeg could not wrap the supplied D3D12 device.

    #[error("failed to create D3D12VA hw device context (code {0})")]
    HwDeviceInit(i32),
    /// FFmpeg could not retain the D3D12 hardware device context.

    #[error("failed to reference the D3D12VA hw device context")]
    HwDeviceRef,
    /// FFmpeg could not negotiate D3D12VA hardware output for the stream.

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
/// zero-copy path through the frame's D3D12VA texture instead of reading
/// pixel bytes.
pub struct D3d12Decoder {
    pp_log: PpLog,
    name: Arc<str>,
    decoder: ffmpeg::decoder::Video,
    _hw_device_ctx: AvBufferRef,
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
unsafe impl Send for D3d12Decoder {}

impl D3d12Decoder {
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
    ) -> Result<Self, D3d12DecoderError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::D3d12Decoder, &name, None);

        // SAFETY: `device` is live and the helper clones its COM reference
        // into the returned FFmpeg hardware-device context.
        let hw_device_ctx =
            unsafe { create_hw_device_ctx(device) }.map_err(D3d12DecoderError::HwDeviceInit)?;

        let mut context = ffmpeg::codec::context::Context::from_parameters(params)?;
        if context.medium() != ffmpeg::media::Type::Video {
            return Err(D3d12DecoderError::UnsupportedMediaType(context.medium()));
        }
        let codec_device_ctx = hw_device_ctx
            .try_clone()
            .ok_or(D3d12DecoderError::HwDeviceRef)?;
        // SAFETY: `context` is exclusively owned and unopened. Ownership of
        // `codec_device_ctx` is transferred to FFmpeg and the callback is set
        // before decoder construction can inspect either field.
        unsafe {
            let ctx_ptr = context.as_mut_ptr();
            (*ctx_ptr).hw_device_ctx = codec_device_ctx.into_raw();
            (*ctx_ptr).get_format = Some(get_format);
        }

        let decoder = context.decoder().video()?;

        let pad = SrcPad::new(format!("{name}_src"));
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        pp_info!(pp_log: &pp_log, "opened: codec={:?}", decoder.id());
        Ok(Self {
            name,
            pp_log,
            decoder,
            _hw_device_ctx: hw_device_ctx,
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
                        return Err(D3d12DecoderError::HwAccelUnavailable.into());
                    }
                    self.pad.push(MediaBuffer::Video(Arc::new(frame)))?;
                    frame = self.pool.get();
                }
                Err(error) if is_codec_drain_boundary(&error) => break,
                Err(error) => return Err(D3d12DecoderError::from(error).into()),
            }
        }
        Ok(())
    }
}

impl Element for D3d12Decoder {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::D3d12Decoder
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for D3d12Decoder {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for D3d12Decoder {
    fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
        match buf {
            MediaBuffer::Packet(packet) => {
                self.decoder
                    .send_packet(&*packet)
                    .inspect_err(|error| pp_error!(self, "send_packet failed: {error}"))
                    .map_err(D3d12DecoderError::from)?;
                self.drain()
            }
            MediaBuffer::Eos => {
                self.decoder
                    .send_eof()
                    .inspect_err(|error| pp_error!(self, "send_eof failed: {error}"))
                    .map_err(D3d12DecoderError::from)?;
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

impl Drop for D3d12Decoder {
    fn drop(&mut self) {
        pp_info!(self, "dropped: freeing hw_device_ctx");
    }
}

unsafe extern "C" fn get_format(
    _ctx: *mut ffi::AVCodecContext,
    mut fmt: *const ffi::AVPixelFormat,
) -> ffi::AVPixelFormat {
    // SAFETY: FFmpeg supplies a readable `AV_PIX_FMT_NONE`-terminated array
    // for the duration of this callback.
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
