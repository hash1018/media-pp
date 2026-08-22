use std::sync::Arc;

use crate::pp_log::{PpLog, pp_error, pp_info};
use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;
use windows::Win32::Graphics::Direct3D11::{D3D11_BIND_SHADER_RESOURCE, ID3D11Device};

use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    pad::SrcPad,
    platform::{
        ffmpeg::AvBufferRef,
        windows::d3d11va::{create_hw_device_ctx, or_frames_bind_flags},
    },
    pool::UnboundObjectPool,
};

use crate::elements::filter::is_codec_drain_boundary;

/// Errors specific to `D3d11Decoder`. Converts into the crate-wide
/// `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum D3d11DecoderError {
    /// The selected stream is not video.
    #[error("unsupported media type: {0:?} (D3D11VA decode is video-only)")]
    UnsupportedMediaType(ffmpeg::media::Type),
    /// FFmpeg rejected decoder or packet/frame processing.

    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),
    /// FFmpeg could not wrap the supplied D3D11 device.

    #[error("failed to create D3D11VA hw device context (code {0})")]
    HwDeviceInit(i32),
    /// FFmpeg could not retain the D3D11 hardware device context.

    #[error("failed to reference the D3D11VA hw device context")]
    HwDeviceRef,
    /// FFmpeg could not negotiate D3D11VA hardware output for the stream.

    #[error(
        "decoder did not select the D3D11VA pixel format — hardware decode \
         unavailable for this stream/GPU/driver"
    )]
    HwAccelUnavailable,
}

/// Decodes one video stream's `Packet`s into GPU-resident `Video` frames
/// via D3D11VA hardware acceleration — the D3D11 sibling of
/// `D3d12Decoder`, for a pipeline built entirely on
/// one shared `ID3D11Device` (see [`crate::elements::D3d11Renderer`]'s own
/// docs on why that means no explicit fence/sync is needed anywhere in
/// this stack, unlike the D3D12 side). A `Filter`, same shape as
/// `SwDecoder`/`D3d12Decoder`.
pub struct D3d11Decoder {
    pp_log: PpLog,
    name: Arc<str>,
    decoder: ffmpeg::decoder::Video,
    _hw_device_ctx: AvBufferRef,
    pad: SrcPad,
    /// Reused across every decoded frame — see [`UnboundObjectPool`]'s
    /// docs; same reasoning as `D3d12Decoder`'s own `pool` field (the
    /// GPU texture itself is already pooled by ffmpeg's own hw frames
    /// context, this only reuses the small CPU-side `AVFrame` wrapper).
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
}

// SAFETY: `hw_device_ctx` is a heap-allocated FFmpeg buffer with no
// thread affinity; `decoder`'s own `Send` already covers the rest.
// `&mut self` on every method that touches it rules out concurrent
// access from multiple threads — same reasoning as `D3d12Decoder`.
unsafe impl Send for D3d11Decoder {}

impl D3d11Decoder {
    /// `device` must outlive this decoder (and, transitively, every frame
    /// it produces that's still alive downstream), and must be the same
    /// `ID3D11Device` every other D3D11 element in this pipeline shares —
    /// see [`crate::elements::D3d11Renderer`]'s own docs on why this whole
    /// stack requires exactly one shared device/context, not just a
    /// same-adapter one.
    ///
    /// `extra_hw_frames` sets `AVCodecContext.extra_hw_frames` — how many
    /// *additional* decode surfaces to allocate beyond what the codec's own
    /// reference-frame count strictly requires. Unlike
    /// `D3d12Decoder` (no equivalent parameter needed),
    /// D3D11VA's decode surface pool is a **fixed-size** texture array,
    /// sized once at `av_hwframe_ctx_init()` time and never grown — every
    /// decoded frame still alive downstream (sitting in a
    /// [`crate::queue::Queue`], held by a slow renderer, ...) keeps one pool
    /// slot occupied, and once the pool runs out, decode itself starts
    /// failing (`AVERROR(ENOMEM)`, "Static surface pool size exceeded" in
    /// the log) instead of just blocking. Pass at least as many extra frames
    /// as the deepest queue/buffer this decoder's output can pile up in
    /// (e.g. match a downstream `ChainBuilder::queue` depth) — too few
    /// reproduces exactly that failure under real playback, not just in
    /// theory.
    pub fn new(
        name: impl Into<String>,
        params: ffmpeg::codec::Parameters,
        device: &ID3D11Device,
        extra_hw_frames: i32,
    ) -> Result<Self, D3d11DecoderError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::D3d11Decoder, &name, None);

        // SAFETY: `device` is a live D3D11 device; the returned FFmpeg
        // context takes its own COM reference as documented by the helper.
        let hw_device_ctx =
            unsafe { create_hw_device_ctx(device) }.map_err(D3d11DecoderError::HwDeviceInit)?;

        let mut context = ffmpeg::codec::context::Context::from_parameters(params)?;
        if context.medium() != ffmpeg::media::Type::Video {
            return Err(D3d11DecoderError::UnsupportedMediaType(context.medium()));
        }
        let codec_device_ctx = hw_device_ctx
            .try_clone()
            .ok_or(D3d11DecoderError::HwDeviceRef)?;
        // SAFETY: `context` is exclusively owned and not opened yet. The raw
        // FFmpeg buffer reference is transferred into `hw_device_ctx`, and the
        // callback and frame count are set before the decoder can read them.
        unsafe {
            let ctx_ptr = context.as_mut_ptr();
            (*ctx_ptr).hw_device_ctx = codec_device_ctx.into_raw();
            (*ctx_ptr).get_format = Some(get_format);
            (*ctx_ptr).extra_hw_frames = extra_hw_frames;
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
                    if frame.format() != ffmpeg::format::Pixel::D3D11 {
                        pp_error!(self, "decoder did not select the D3D11VA pixel format");
                        return Err(D3d11DecoderError::HwAccelUnavailable.into());
                    }
                    self.pad.push(MediaBuffer::Video(Arc::new(frame)))?;
                    frame = self.pool.get();
                }
                Err(error) if is_codec_drain_boundary(&error) => break,
                Err(error) => return Err(D3d11DecoderError::from(error).into()),
            }
        }
        Ok(())
    }
}

impl Element for D3d11Decoder {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::D3d11Decoder
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for D3d11Decoder {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for D3d11Decoder {
    fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
        match buf {
            MediaBuffer::Packet(packet) => {
                self.decoder
                    .send_packet(&*packet)
                    .inspect_err(|error| pp_error!(self, "send_packet failed: {error}"))
                    .map_err(D3d11DecoderError::from)?;
                self.drain()
            }
            MediaBuffer::Eos => {
                self.decoder
                    .send_eof()
                    .inspect_err(|error| pp_error!(self, "send_eof failed: {error}"))
                    .map_err(D3d11DecoderError::from)?;
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
        // Same reasoning as `D3d12Decoder::control`: nothing to do on
        // `Stop` (the hw device context is freed in `Drop`), flush
        // reference-frame state on `Seek`.
        if let ControlMsg::Seek(_) = msg {
            self.decoder.flush();
        }
        self.pad.control(msg)
    }
}

impl Drop for D3d11Decoder {
    fn drop(&mut self) {
        pp_info!(self, "dropped: freeing hw_device_ctx");
    }
}

/// Builds and initializes `avctx->hw_frames_ctx` during D3D11VA format
/// negotiation. Decoder surfaces need `D3D11_BIND_SHADER_RESOURCE` in
/// addition to the decoder bind flags so downstream renderers and filters can
/// sample them. The shared ABI write is confined to
/// [`crate::platform::windows::d3d11va::or_frames_bind_flags`].
unsafe fn configure_hw_frames_ctx(ctx: *mut ffi::AVCodecContext) -> Result<(), i32> {
    // SAFETY: the callback receives a live codec context during format
    // negotiation. FFmpeg initializes the returned reference; it is wrapped
    // immediately so every error path releases it before returning.
    unsafe {
        let mut frames_ref: *mut ffi::AVBufferRef = std::ptr::null_mut();
        let result = ffi::avcodec_get_hw_frames_parameters(
            ctx,
            (*ctx).hw_device_ctx,
            ffi::AVPixelFormat::AV_PIX_FMT_D3D11,
            &mut frames_ref,
        );
        if result < 0 {
            return Err(result);
        }

        let frames_ctx = (*frames_ref).data as *mut ffi::AVHWFramesContext;
        or_frames_bind_flags(frames_ctx, D3D11_BIND_SHADER_RESOURCE.0 as u32);

        let result = ffi::av_hwframe_ctx_init(frames_ref);
        if result < 0 {
            ffi::av_buffer_unref(&mut frames_ref);
            return Err(result);
        }

        // Transfers ownership of `frames_ref` to `avctx` — matches
        // `avcodec_get_hw_frames_parameters`'s own documented contract
        // ("the user's responsibility to ... set
        // AVCodecContext.hw_frames_ctx to it").
        (*ctx).hw_frames_ctx = frames_ref;
        Ok(())
    }
}

unsafe extern "C" fn get_format(
    ctx: *mut ffi::AVCodecContext,
    mut fmt: *const ffi::AVPixelFormat,
) -> ffi::AVPixelFormat {
    // SAFETY: FFmpeg supplies a live `AV_PIX_FMT_NONE`-terminated format list
    // and a live codec context for the duration of this callback.
    unsafe {
        while *fmt != ffi::AVPixelFormat::AV_PIX_FMT_NONE {
            if *fmt == ffi::AVPixelFormat::AV_PIX_FMT_D3D11 {
                if configure_hw_frames_ctx(ctx).is_ok() {
                    return ffi::AVPixelFormat::AV_PIX_FMT_D3D11;
                }
                // Configuration failed (e.g. this GPU/driver doesn't
                // support sampling decoded D3D11VA surfaces) — fall
                // through to whatever other format the decoder offers,
                // same as if D3D11 had never been in `fmt`'s list at all.
                fmt = fmt.add(1);
                continue;
            }
            fmt = fmt.add(1);
        }
        ffi::AVPixelFormat::AV_PIX_FMT_NONE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::try_d3d11_device;

    /// Regression test for a real crash (`STATUS_ACCESS_VIOLATION`/
    /// `STATUS_BREAKPOINT`, depending on the run) found via
    /// `d3d11_decode_render`: `create_hw_device_ctx` used to hand FFmpeg a
    /// bare, non-`AddRef`'d `ID3D11Device*` — but
    /// `libavutil/hwcontext_d3d11va.c`'s `d3d11va_device_uninit`
    /// unconditionally `Release()`s whatever's in that field, unlike the
    /// D3D12 sibling (which never releases the caller-provided device).
    /// That one extra `Release()` didn't crash immediately — only later,
    /// once every *other* reference (including this test's own final
    /// `device` drop) had also released and the COM object was already
    /// gone. A real pipeline run masked exactly how late "later" was
    /// (looked like a clean exit until the process teardown itself);
    /// decoding a full file end-to-end and then dropping everything, with
    /// nothing downstream holding frames open, is what actually reproduces
    /// it deterministically.
    #[test]
    fn decodes_a_full_file_and_tears_down_cleanly() {
        let Some((device, _context)) = try_d3d11_device() else {
            return;
        };

        let Some(path) = crate::test_support::try_test_video() else {
            return;
        };
        let mut input = ffmpeg::format::input(&path).expect("failed to open test video");
        let video_stream = input
            .streams()
            .best(ffmpeg::media::Type::Video)
            .expect("no video stream");
        let video_index = video_stream.index();
        let params = video_stream.parameters();

        let mut decoder = D3d11Decoder::new("test-decoder", params, &device, 32)
            .expect("failed to open D3D11VA decoder");

        for (stream, packet) in input.packets() {
            if stream.index() != video_index {
                continue;
            }
            decoder
                .consume(MediaBuffer::Packet(Arc::new(packet)))
                .expect("consume(Packet) failed");
        }
        decoder
            .consume(MediaBuffer::Eos)
            .expect("consume(Eos) failed");

        // The crash this test guards against only ever showed up here,
        // on this final drop — see this test's own docs.
        drop(decoder);
        drop(input);
        drop(device);
    }
}
