use std::sync::Arc;

use crate::pp_log::{PpLog, pp_error, pp_info};
use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;
use windows::Win32::Graphics::Direct3D11::{D3D11_BIND_SHADER_RESOURCE, ID3D11Device};

use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    error::D3d11SharedDeviceError,
    pad::SrcPad,
    platform::{
        ffmpeg::AvBufferRef,
        windows::d3d11::protect_shared_device,
        windows::d3d11va::{create_hw_device_ctx, or_frames_bind_flags},
    },
    pool::UnboundObjectPool,
};

use super::super::hw_decoder::{NegotiationRefusal, capable_decoder};
use super::super::preroll_gate::{PrerollGate, hw_surface_budget};
use crate::elements::filter::is_codec_drain_boundary;

/// Errors specific to `D3d11Decoder`. Converts into the crate-wide
/// `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum D3d11DecoderError {
    /// The selected stream is not video.
    #[error("unsupported media type: {0:?} (D3D11VA decode is video-only)")]
    UnsupportedMediaType(ffmpeg::media::Type),
    /// No decoder this FFmpeg build has for the codec can decode through
    /// D3D11VA — see [`D3d11Decoder::supports`].
    #[error("no decoder in this FFmpeg build decodes {0:?} through D3D11VA")]
    UnsupportedCodec(ffmpeg::codec::Id),
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
    /// The caller supplied a negative downstream surface budget, or adding
    /// the internally retained accurate-seek candidate would overflow it.
    #[error("invalid downstream hardware-frame budget: {0}")]
    InvalidSurfaceBudget(i32),

    /// The device cannot be shared across a pipeline's threads.
    #[error(transparent)]
    SharedDevice(#[from] D3d11SharedDeviceError),
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
    /// Suppresses decoded samples before a seek target during preroll.
    preroll_gate: PrerollGate,
    /// Set by `get_format` when the GPU refuses the stream — see
    /// [`NegotiationRefusal`]. After `decoder`, so it outlives the codec
    /// context that points at it.
    refused: NegotiationRefusal,
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
    /// `downstream_hw_frames` is the deepest number of decoded frames that
    /// downstream queues and sinks may retain. The decoder adds its own one-
    /// frame accurate-seek candidate and writes the sum to
    /// `AVCodecContext.extra_hw_frames`. Unlike
    /// `D3d12Decoder` (no equivalent parameter needed),
    /// D3D11VA's decode surface pool is a **fixed-size** texture array,
    /// sized once at `av_hwframe_ctx_init()` time and never grown — every
    /// decoded frame still alive downstream (sitting in a
    /// [`crate::queue::Queue`], held by a slow renderer, ...) keeps one pool
    /// slot occupied, and once the pool runs out, decode itself starts
    /// failing (`AVERROR(ENOMEM)`, "Static surface pool size exceeded" in
    /// the log) instead of just blocking. Pass the deepest downstream
    /// queue/buffer depth; the internal candidate must not be counted again.
    /// Too small a downstream budget reproduces exactly that failure under
    /// real playback, not just in theory.
    ///
    /// The decoder opened is the first one FFmpeg has for the codec that can
    /// decode through D3D11VA, which need not be the one FFmpeg would pick by
    /// default — see [`Self::supports`]. A codec with none fails here with
    /// [`D3d11DecoderError::UnsupportedCodec`] rather than at the first frame.
    pub fn new(
        name: impl Into<String>,
        params: ffmpeg::codec::Parameters,
        device: &ID3D11Device,
        downstream_hw_frames: i32,
    ) -> Result<Self, D3d11DecoderError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::D3d11Decoder, &name, None);
        let extra_hw_frames = hw_surface_budget(downstream_hw_frames).ok_or(
            D3d11DecoderError::InvalidSurfaceBudget(downstream_hw_frames),
        )?;

        // What to open is settled before any device work, so a stream this
        // cannot decode is refused before it touches the device.
        let mut context = ffmpeg::codec::context::Context::from_parameters(params)?;
        if context.medium() != ffmpeg::media::Type::Video {
            return Err(D3d11DecoderError::UnsupportedMediaType(context.medium()));
        }
        let codec = d3d11va_capable_decoder(context.id())
            .ok_or(D3d11DecoderError::UnsupportedCodec(context.id()))?;

        // FFmpeg reaches this device's immediate context for the video
        // context it decodes through, and decoded frames are consumed on
        // another thread past the next `Queue`.
        protect_shared_device(device)?;

        // SAFETY: `device` is a live D3D11 device; the returned FFmpeg
        // context takes its own COM reference as documented by the helper.
        let hw_device_ctx =
            unsafe { create_hw_device_ctx(device) }.map_err(D3d11DecoderError::HwDeviceInit)?;

        let codec_device_ctx = hw_device_ctx
            .try_clone()
            .ok_or(D3d11DecoderError::HwDeviceRef)?;
        let refused = NegotiationRefusal::new();
        // SAFETY: `context` is exclusively owned and not opened yet. The raw
        // FFmpeg buffer reference is transferred into `hw_device_ctx`, and the
        // callback and frame count are set before the decoder can read them.
        // `opaque` points at `refused`, which the decoder keeps for as long
        // as the codec context — see that field.
        unsafe {
            let ctx_ptr = context.as_mut_ptr();
            (*ctx_ptr).opaque = refused.opaque();
            (*ctx_ptr).hw_device_ctx = codec_device_ctx.into_raw();
            (*ctx_ptr).get_format = Some(get_format);
            (*ctx_ptr).extra_hw_frames = extra_hw_frames;
        }

        let decoder = context.decoder().open_as(codec)?.video()?;

        let pad = SrcPad::with_contract(
            format!("{name}_src"),
            OutputContract::Fixed(PortContract::frame(
                MediaKind::VideoFrame,
                MemoryDomain::D3d11,
            )),
        );
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        pp_info!(pp_log: &pp_log, "opened: codec={:?}", decoder.id());
        Ok(Self {
            name,
            pp_log,
            decoder,
            _hw_device_ctx: hw_device_ctx,
            pad,
            pool,
            preroll_gate: PrerollGate::default(),
            refused,
        })
    }

    /// Whether this FFmpeg build has a decoder for `codec` that can decode
    /// through D3D11VA — whether [`Self::new`] gets past choosing one. Needs
    /// no device, so a caller can route a stream elsewhere before building
    /// anything.
    ///
    /// A question about the decoder, not the codec — see `CudaDecoder`'s
    /// `supports`, which answers the same for NVDEC: with `libdav1d` built
    /// in, FFmpeg's default for AV1 has no hardware path, while its own `av1`
    /// decoder does.
    ///
    /// `true` does not promise the GPU takes every stream of the codec. A
    /// profile the driver lacks, or a codec the card is older than, still
    /// fails at the first frame with [`D3d11DecoderError::HwAccelUnavailable`].
    pub fn supports(codec: ffmpeg::codec::Id) -> bool {
        d3d11va_capable_decoder(codec).is_some()
    }

    /// `error` as this decoder's own: the GPU refusing the stream where
    /// `get_format` said so, which FFmpeg's own error does not — see
    /// [`NegotiationRefusal`].
    fn decode_error(&self, error: ffmpeg::Error) -> D3d11DecoderError {
        if self.refused.happened() {
            D3d11DecoderError::HwAccelUnavailable
        } else {
            error.into()
        }
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
                    // Reassigning `frame` releases a suppressed one right here,
                    // returning its fixed-pool surface a whole branch earlier
                    // than dropping it downstream would.
                    self.preroll_gate
                        .push_admitted(MediaBuffer::Video(Arc::new(frame)), |frame| {
                            self.pad.push(frame)
                        })?;
                    frame = self.pool.get();
                }
                Err(error) if is_codec_drain_boundary(&error) => break,
                Err(error) => return Err(self.decode_error(error).into()),
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
    /// Decodes into D3D11VA surfaces, so what it accepts is the same encoded data any decoder takes.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::packet(MediaKind::VideoPacket))
    }

    fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
        match buf {
            MediaBuffer::Packet(packet) => {
                // Decoded frames carry a `pts` but not the unit it is in.
                self.preroll_gate.observe_packet(&packet);
                self.decoder
                    .send_packet(&*packet)
                    .inspect_err(|error| pp_error!(self, "send_packet failed: {error}"))
                    .map_err(|error| self.decode_error(error))?;
                self.drain()
            }
            MediaBuffer::Eos => {
                self.decoder
                    .send_eof()
                    .inspect_err(|error| pp_error!(self, "send_eof failed: {error}"))
                    .map_err(|error| self.decode_error(error))?;
                self.drain()?;
                self.preroll_gate
                    .push_eos_candidate(|candidate| self.pad.push(candidate))?;
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
        // reference-frame state on `Flush`.
        //
        // `Preroll` may carry a seek target; the samples decoded while
        // catching up to it exist only to warm the codec.
        match &msg {
            ControlMsg::Flush => {
                self.decoder.flush();
                self.preroll_gate.reset();
            }
            ControlMsg::Preroll(context) => self.preroll_gate.begin(context),
            ControlMsg::Pause | ControlMsg::Resume | ControlMsg::Stop => self.preroll_gate.clear(),
            ControlMsg::CheckSeek(_) | ControlMsg::Seek(_) => {}
        }
        self.pad.control(msg)
    }
}

impl Drop for D3d11Decoder {
    fn drop(&mut self) {
        pp_info!(self, "dropped: freeing hw_device_ctx");
    }
}

/// The first decoder for `id` that decodes through D3D11VA — see
/// [`capable_decoder`].
fn d3d11va_capable_decoder(id: ffmpeg::codec::Id) -> Option<ffmpeg::Codec> {
    capable_decoder(id, ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_D3D11VA)
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
        // Nothing here is hardware, which is the refusal `new` pointed
        // `opaque` at `refused` to hear about.
        NegotiationRefusal::mark(ctx);
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

        // A D3D11 device is not the same thing as a D3D11 *video* device.
        // `try_d3d11_device` gets one on any machine that can render at all,
        // including the Basic Render Driver a CI runner falls back to, and
        // wrapping that as a D3D11VA hardware device is what fails there —
        // `AVERROR_UNKNOWN`, arriving as `HwDeviceInit(-1313558101)`. That is
        // the absence of hardware rather than a regression, and this skips
        // for it the way every other hardware test skips for its own.
        let mut decoder = match D3d11Decoder::new("test-decoder", params, &device, 32) {
            Ok(decoder) => decoder,
            Err(D3d11DecoderError::HwDeviceInit(code)) => {
                eprintln!(
                    "skipping: this machine's D3D11 device does not do video decoding \
                     (hw device init failed with {code})"
                );
                return;
            }
            Err(error) => panic!("failed to open D3D11VA decoder: {error}"),
        };

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

    /// Video parameters for `codec` carrying nothing but its size, which is
    /// all choosing a decoder asks of them.
    fn video_parameters(codec: ffmpeg::codec::Id) -> ffmpeg::codec::Parameters {
        let mut params = ffmpeg::codec::Parameters::new();
        // SAFETY: `as_mut_ptr` on parameters this function just created and
        // still owns exclusively; every field set is a plain one.
        unsafe {
            let raw = params.as_mut_ptr();
            (*raw).codec_type = ffmpeg::media::Type::Video.into();
            (*raw).codec_id = codec.into();
            (*raw).width = 320;
            (*raw).height = 240;
        }
        params
    }

    /// D3D11VA has no ProRes, and no FFmpeg decoder offers a path for it.
    /// Answered from FFmpeg's own tables, so it holds on a machine with no GPU.
    #[test]
    fn supports_says_no_for_a_codec_d3d11va_lacks() {
        crate::init().unwrap();
        assert!(!D3d11Decoder::supports(ffmpeg::codec::Id::PRORES));
    }

    /// Refused while being built, as a typed error, and before the device is
    /// touched. It used to open, as the software decoder FFmpeg handed it,
    /// and fail only at the first frame.
    #[test]
    fn a_codec_with_no_d3d11va_decoder_is_refused_when_built() {
        let Some((device, _context)) = try_d3d11_device() else {
            return;
        };
        let codec = ffmpeg::codec::Id::PRORES;
        let error = D3d11Decoder::new("test-decoder", video_parameters(codec), &device, 0)
            .err()
            .expect("a codec with no D3D11VA decoder must not open");
        assert!(
            matches!(error, D3d11DecoderError::UnsupportedCodec(id) if id == codec),
            "expected UnsupportedCodec, got {error:?}"
        );
    }

    /// The case that broke, run: AV1, whose default decoder in a build with
    /// `libdav1d` has no D3D11VA path. What opens is FFmpeg's own `av1`,
    /// and every frame of a real AV1 stream comes out on the GPU. Without
    /// the choice, `libdav1d` opened and the first frame failed with
    /// `HwAccelUnavailable`.
    ///
    /// A GPU without AV1 decode fails that same way for a different reason,
    /// and is skipped — after the decoder chosen has been checked, which is
    /// the part a machine without the hardware can still vouch for.
    #[test]
    fn av1_opens_a_d3d11va_decoder_and_decodes_on_the_gpu() {
        use crate::elements::AppSink;

        let Some((device, _context)) = try_d3d11_device() else {
            return;
        };
        let Some((params, packets)) = crate::test_support::try_av1_packets() else {
            return;
        };
        let mut decoder = match D3d11Decoder::new("test-av1", params, &device, 4) {
            Ok(decoder) => decoder,
            Err(D3d11DecoderError::HwDeviceInit(code)) => {
                eprintln!("skipping: this device does not do video decoding (code {code})");
                return;
            }
            Err(error) => panic!("AV1 did not open: {error}"),
        };
        let opened = decoder
            .decoder
            .codec()
            .expect("an open decoder has a codec");
        // SAFETY: the codec an open decoder reports is FFmpeg's static one.
        let through_d3d11va = unsafe {
            super::super::super::hw_decoder::decodes_on(
                opened.as_ptr(),
                ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_D3D11VA,
            )
        };
        assert!(
            through_d3d11va,
            "AV1 opened {}, which has no D3D11VA path",
            opened.name()
        );

        let on_gpu = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counted = Arc::clone(&on_gpu);
        decoder.src_pads()[0].link(Box::new(AppSink::new("test-av1-frames", move |buffer| {
            if let MediaBuffer::Video(frame) = buffer
                && frame.format() == ffmpeg::format::Pixel::D3D11
            {
                counted.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            Ok(())
        })));
        let sent = packets.len();
        let decoded = (|| {
            for packet in packets {
                decoder.consume(MediaBuffer::Packet(Arc::new(packet)))?;
            }
            decoder.consume(MediaBuffer::Eos)
        })();
        if let Err(crate::error::Error::D3d11DecoderError(D3d11DecoderError::HwAccelUnavailable)) =
            decoded
        {
            eprintln!("skipping the decode: this GPU does not decode AV1 through D3D11VA");
            return;
        }
        decoded.expect("an AV1 stream decodes");
        assert_eq!(
            on_gpu.load(std::sync::atomic::Ordering::Relaxed),
            sent,
            "every AV1 frame comes out as a D3D11 texture"
        );
    }

    /// A GPU that lacks the stream's profile says so as
    /// [`D3d11DecoderError::HwAccelUnavailable`], not as the `EPERM` and
    /// `AVERROR_INVALIDDATA` FFmpeg answers with, which a damaged stream
    /// also answers with — see `NegotiationRefusal`. VP9 profile 1, 8-bit
    /// 4:4:4, is one no hardware decoder here has.
    #[test]
    fn a_profile_the_gpu_lacks_is_reported_as_refused() {
        let Some((device, _context)) = try_d3d11_device() else {
            return;
        };
        let Some((params, packets)) = crate::test_support::try_encoded_packets(
            "libvpx-vp9",
            ffmpeg::format::Pixel::YUV444P,
            (64, 64),
            90,
        ) else {
            return;
        };
        let mut decoder = D3d11Decoder::new("refused", params, &device, 4).unwrap();
        decoder.src_pads()[0].link(Box::new(crate::elements::AppSink::new(
            "refused-frames",
            |_| Ok(()),
        )));
        for packet in packets {
            let error = decoder
                .consume(MediaBuffer::Packet(Arc::new(packet)))
                .expect_err("the GPU has no VP9 profile 1");
            assert!(
                matches!(
                    error,
                    crate::error::Error::D3d11DecoderError(D3d11DecoderError::HwAccelUnavailable)
                ),
                "every packet after the refusal says why: {error}"
            );
        }
    }
}
