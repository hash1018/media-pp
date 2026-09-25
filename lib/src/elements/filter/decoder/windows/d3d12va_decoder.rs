use std::sync::Arc;

use crate::pp_log::{PpLog, pp_error, pp_info};
use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    control::ControlMsg,
    element::{Element, ElementType, ReversibleDecoder, Sink, Source, element_pp_log},
    pad::SrcPad,
    platform::{
        ffmpeg::AvBufferRef,
        windows::{d3d12_gpu::D3d12Gpu, d3d12va::create_hw_device_ctx},
    },
    pool::UnboundObjectPool,
};

use super::super::backwards::Stretch;
use super::super::hw_decoder::{CopyFrames, NegotiationRefusal, capable_decoder};
use super::super::preroll_gate::PrerollGate;
use super::d3d12_copier::D3d12Copier;
use crate::elements::filter::is_codec_drain_boundary;

/// Errors specific to `D3d12Decoder`. Converts into the crate-wide
/// `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum D3d12DecoderError {
    /// The selected stream is not video.
    #[error("unsupported media type: {0:?} (D3D12VA decode is video-only)")]
    UnsupportedMediaType(ffmpeg::media::Type),
    /// No decoder this FFmpeg build has for the codec can decode through
    /// D3D12VA — see [`D3d12Decoder::supports`].
    #[error("no decoder in this FFmpeg build decodes {0:?} through D3D12VA")]
    UnsupportedCodec(ffmpeg::codec::Id),
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
    /// Copying a decoded picture out of the decoder's pool to play it
    /// backwards failed on the device.
    #[error("copying a decoded picture to play it backwards failed: {0}")]
    Copy(windows::core::Error),
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
    /// Suppresses decoded samples before a seek target during preroll.
    preroll_gate: PrerollGate,
    /// Set by `get_format` when the GPU refuses the stream — see
    /// [`NegotiationRefusal`]. After `decoder`, so it outlives the codec
    /// context that points at it.
    refused: NegotiationRefusal,
    /// Holds a stretch's pictures to hand on last first, playing backwards
    /// — copies of them, since the surfaces they were decoded to are a pool
    /// the rest of the stretch is decoded into.
    stretch: Stretch,
    /// The device, for the copier.
    gpu: D3d12Gpu,
    /// Where those copies come from, on this decoder's device.
    copies: CopyFrames,
    /// What copies them, made on the first.
    copier: Option<D3d12Copier>,
    /// What the copies travel in.
    copy_wrappers: UnboundObjectPool<ffmpeg::frame::Video>,
}

// SAFETY: `hw_device_ctx` is a heap-allocated FFmpeg buffer with no
// thread affinity; `decoder`'s own `Send` already covers the rest.
// `&mut self` on every method that touches it rules out concurrent
// access from multiple threads.
unsafe impl Send for D3d12Decoder {}

impl D3d12Decoder {
    /// `gpu` must be the [`D3d12Gpu`] every other D3D12 element in this
    /// pipeline shares; the D3D12VA hardware context owns an independent COM
    /// reference to its device, so the caller does not need to keep `gpu`
    /// alive. A renderer has to render with the same device (see
    /// [`crate::elements::D3d12FrameRenderer`]'s own `device()`) so decoded
    /// frames land on the device it reads from — required for the zero-copy
    /// path to be valid at all; checked at render time, not just documented,
    /// via `D3d12Renderer`'s own device-mismatch guard.
    ///
    /// The decoder opened is the first one FFmpeg has for the codec that can
    /// decode through D3D12VA, which need not be the one FFmpeg would pick by
    /// default — see [`Self::supports`]. A codec with none fails here with
    /// [`D3d12DecoderError::UnsupportedCodec`] rather than at the first frame.
    pub fn new(
        name: impl Into<String>,
        params: ffmpeg::codec::Parameters,
        gpu: &D3d12Gpu,
    ) -> Result<Self, D3d12DecoderError> {
        let device = gpu.device();
        crate::ensure_ffmpeg();
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::D3d12Decoder, &name, None);

        // What to open is settled before any device work, so a stream this
        // cannot decode is refused before it touches the device.
        let mut context = ffmpeg::codec::context::Context::from_parameters(params)?;
        if context.medium() != ffmpeg::media::Type::Video {
            return Err(D3d12DecoderError::UnsupportedMediaType(context.medium()));
        }
        let codec = d3d12va_capable_decoder(context.id())
            .ok_or(D3d12DecoderError::UnsupportedCodec(context.id()))?;

        // SAFETY: `device` is live and the helper clones its COM reference
        // into the returned FFmpeg hardware-device context.
        let hw_device_ctx =
            unsafe { create_hw_device_ctx(device) }.map_err(D3d12DecoderError::HwDeviceInit)?;

        let codec_device_ctx = hw_device_ctx
            .try_clone()
            .ok_or(D3d12DecoderError::HwDeviceRef)?;
        let refused = NegotiationRefusal::new();
        // SAFETY: `context` is exclusively owned and unopened. Ownership of
        // `codec_device_ctx` is transferred to FFmpeg and the callback is set
        // before decoder construction can inspect either field.
        // `opaque` points at `refused`, which the decoder keeps for as long
        // as the codec context — see that field.
        unsafe {
            let ctx_ptr = context.as_mut_ptr();
            (*ctx_ptr).opaque = refused.opaque();
            (*ctx_ptr).hw_device_ctx = codec_device_ctx.into_raw();
            (*ctx_ptr).get_format = Some(get_format);
        }

        let decoder = context.decoder().open_as(codec)?.video()?;

        let pad = SrcPad::with_contract(
            format!("{name}_src"),
            OutputContract::Fixed(
                PortContract::frame(MediaKind::VideoFrame, MemoryDomain::D3d12).with_layouts(
                    crate::contract::PixelLayoutSet::decoded_from(decoder.format()),
                ),
            ),
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
            stretch: Stretch::default(),
            gpu: gpu.clone(),
            copies: CopyFrames::default(),
            copier: None,
            copy_wrappers: UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {}),
        })
    }

    /// Whether this FFmpeg build has a decoder for `codec` that can decode
    /// through D3D12VA — whether [`Self::new`] gets past choosing one. Needs
    /// no device. The same question `D3d11Decoder::supports` answers for
    /// D3D11VA, and with the same limit: `true` does not promise the GPU
    /// takes every profile, and one it lacks still fails at the first frame
    /// with [`D3d12DecoderError::HwAccelUnavailable`].
    pub fn supports(codec: ffmpeg::codec::Id) -> bool {
        d3d12va_capable_decoder(codec).is_some()
    }

    /// `error` as this decoder's own: the GPU refusing the stream where
    /// `get_format` said so, which FFmpeg's own error does not — see
    /// [`NegotiationRefusal`].
    fn decode_error(&self, error: ffmpeg::Error) -> D3d12DecoderError {
        if self.refused.happened() {
            D3d12DecoderError::HwAccelUnavailable
        } else {
            error.into()
        }
    }

    /// `frame`'s picture in a surface of the copies' own, with its timing
    /// and colour, the copy done on the device before this returns.
    fn copied(
        &mut self,
        frame: &ffmpeg::frame::Video,
    ) -> crate::error::Result<crate::pool::UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        let mut copy = self
            .copies
            .frame_like(frame, 0)
            .map_err(D3d12DecoderError::Ffmpeg)?;
        let copier = match &mut self.copier {
            Some(copier) => copier,
            copier => {
                copier.insert(D3d12Copier::new(self.gpu.device()).map_err(D3d12DecoderError::Copy)?)
            }
        };
        copier
            .copy(frame, &mut copy)
            .map_err(D3d12DecoderError::Copy)?;
        // SAFETY: both frames are live; this copies timing, colour and side
        // data, not buffers.
        unsafe {
            ffi::av_frame_copy_props(copy.as_mut_ptr(), frame.as_ptr());
        }
        let mut wrapped = self.copy_wrappers.get();
        *wrapped = copy;
        Ok(wrapped)
    }

    /// Hands on what was held of the stretch just over, last first.
    fn release(&mut self) -> crate::error::Result<()> {
        for buffer in self.stretch.end() {
            self.preroll_gate
                .push_admitted(buffer, |frame| self.pad.push(frame))?;
        }
        Ok(())
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
                    if self.stretch.holding() {
                        // Held as a copy; the surface goes back to the pool
                        // as `frame` is reused below.
                        let copy = self.copied(&frame)?;
                        self.stretch.hold(MediaBuffer::Video(Arc::new(copy)));
                        frame = self.pool.get();
                        continue;
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

    /// The pipeline says when a seek's preroll starts and ends — see
    /// `PrerollGate`.
    fn attach_context(&mut self, context: &Arc<crate::element::Context>) {
        self.preroll_gate.attach(&context.state);
    }
}

impl Source for D3d12Decoder {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for D3d12Decoder {
    /// Not while a preroll this has already given its sample to is still
    /// running — see `PrerollGate::holding`.
    fn ready_consume(&mut self) -> bool {
        !self.preroll_gate.holding()
    }

    /// Decodes into D3D12VA resources, so what it accepts is the same encoded data any decoder takes.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::packet(MediaKind::VideoPacket))
    }

    fn as_reversible(&mut self) -> Option<&mut dyn ReversibleDecoder> {
        Some(self)
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

    fn control(&mut self, msg: &ControlMsg) -> crate::error::Result<()> {
        // `Stop`: no local reaction needed — see `SwDecoder::control`;
        // same reasoning applies to the hw device context, freed in
        // `Drop`.
        //
        // `Flush`: same reasoning as `SwDecoder::control` too — discard
        // leftover reference-frame state before decoding resumes from
        // the new position.
        //
        // `Preroll` may carry a seek target; the samples decoded while
        // catching up to it exist only to warm the codec.
        if *msg == ControlMsg::Flush {
            self.decoder.flush();
            self.preroll_gate.reset();
            self.stretch.reset();
        }
        Ok(())
    }
}

impl ReversibleDecoder for D3d12Decoder {
    fn begin_stretch(&mut self) -> crate::error::Result<()> {
        self.stretch.begin();
        Ok(())
    }

    /// Drains the decoder of the stretch and leaves it fresh for the next
    /// one's keyframe, then hands the stretch on: copies of its pictures,
    /// since the decoder's surfaces are a pool.
    fn end_stretch(&mut self) -> crate::error::Result<()> {
        self.decoder
            .send_eof()
            .inspect_err(|error| pp_error!(self, "send_eof failed: {error}"))
            .map_err(|error| self.decode_error(error))?;
        self.drain()?;
        self.decoder.flush();
        self.release()
    }
}

impl Drop for D3d12Decoder {
    fn drop(&mut self) {
        pp_info!(self, "dropped: freeing hw_device_ctx");
    }
}

/// The first decoder for `id` that decodes through D3D12VA — see
/// [`capable_decoder`].
fn d3d12va_capable_decoder(id: ffmpeg::codec::Id) -> Option<ffmpeg::Codec> {
    capable_decoder(id, ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_D3D12VA)
}

unsafe extern "C" fn get_format(
    ctx: *mut ffi::AVCodecContext,
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
        // Nothing here is hardware, which is the refusal `new` pointed
        // `opaque` at `refused` to hear about.
        NegotiationRefusal::mark(ctx);
        ffi::AVPixelFormat::AV_PIX_FMT_NONE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::try_d3d12_gpu;

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

    /// No FFmpeg decoder offers a D3D12VA path for ProRes. Answered from
    /// FFmpeg's own tables, so it holds on a machine with no GPU.
    #[test]
    fn supports_says_no_for_a_codec_d3d12va_lacks() {
        assert!(!D3d12Decoder::supports(ffmpeg::codec::Id::PRORES));
    }

    /// Refused while being built, before the device is touched, rather than
    /// opening as software and failing at the first frame.
    #[test]
    fn a_codec_with_no_d3d12va_decoder_is_refused_when_built() {
        let Some(gpu) = try_d3d12_gpu() else {
            return;
        };
        let codec = ffmpeg::codec::Id::PRORES;
        let error = D3d12Decoder::new("test-decoder", video_parameters(codec), &gpu)
            .err()
            .expect("a codec with no D3D12VA decoder must not open");
        assert!(
            matches!(error, D3d12DecoderError::UnsupportedCodec(id) if id == codec),
            "expected UnsupportedCodec, got {error:?}"
        );
    }

    /// AV1, run: FFmpeg's default for it with `libdav1d` built in has no
    /// D3D12VA path, so what opens has to be its own `av1`, and every frame
    /// of a real AV1 stream comes out on the GPU. A GPU without AV1 decode is
    /// skipped once the choice of decoder has been checked — see
    /// `D3d11Decoder`'s test of the same.
    #[test]
    fn av1_opens_a_d3d12va_decoder_and_decodes_on_the_gpu() {
        use crate::elements::AppSink;

        let Some(gpu) = try_d3d12_gpu() else {
            return;
        };
        let Some((params, packets)) = crate::test_support::try_av1_packets() else {
            return;
        };
        let mut decoder = match D3d12Decoder::new("test-av1", params, &gpu) {
            Ok(decoder) => decoder,
            Err(D3d12DecoderError::HwDeviceInit(code)) => {
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
        let through_d3d12va = unsafe {
            super::super::super::hw_decoder::decodes_on(
                opened.as_ptr(),
                ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_D3D12VA,
            )
        };
        assert!(
            through_d3d12va,
            "AV1 opened {}, which has no D3D12VA path",
            opened.name()
        );

        let on_gpu = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counted = Arc::clone(&on_gpu);
        decoder.src_pads()[0].link(Box::new(AppSink::new("test-av1-frames", move |buffer| {
            if let MediaBuffer::Video(frame) = buffer
                && frame.format() == ffmpeg::format::Pixel::D3D12
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
        if let Err(crate::error::Error::D3d12DecoderError(D3d12DecoderError::HwAccelUnavailable)) =
            decoded
        {
            eprintln!("skipping the decode: this GPU does not decode AV1 through D3D12VA");
            return;
        }
        decoded.expect("an AV1 stream decodes");
        assert_eq!(
            on_gpu.load(std::sync::atomic::Ordering::Relaxed),
            sent,
            "every AV1 frame comes out as a D3D12 resource"
        );
    }

    /// A GPU that lacks the stream's profile says so as
    /// [`D3d12DecoderError::HwAccelUnavailable`], not as the `EPERM` and
    /// `AVERROR_INVALIDDATA` FFmpeg answers with, which a damaged stream
    /// also answers with — see `NegotiationRefusal`. VP9 profile 1, 8-bit
    /// 4:4:4, is one no hardware decoder here has.
    #[test]
    fn a_profile_the_gpu_lacks_is_reported_as_refused() {
        let Some(gpu) = try_d3d12_gpu() else {
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
        let mut decoder = D3d12Decoder::new("refused", params, &gpu).unwrap();
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
                    crate::error::Error::D3d12DecoderError(D3d12DecoderError::HwAccelUnavailable)
                ),
                "every packet after the refusal says why: {error}"
            );
        }
    }
}
