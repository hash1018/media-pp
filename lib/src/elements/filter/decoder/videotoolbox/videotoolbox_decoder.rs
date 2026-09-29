use std::sync::Arc;

use super::super::backwards::Stretch;
use super::super::hw_decoder::{NegotiationRefusal, capable_decoder};
use super::super::preroll_gate::PrerollGate;
use super::super::qos::Qos;
use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_error, pp_info};

use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    control::ControlMsg,
    element::{Element, ElementType, Flow, ReversibleDecoder, Sink, Source, element_pp_log},
    elements::{VideoToolboxDevice, filter::is_codec_drain_boundary},
    pad::SrcPad,
    platform::ffmpeg::AvBufferRef,
    pool::UnboundObjectPool,
};

/// Errors specific to [`VideoToolboxDecoder`]. Converts into the
/// crate-wide `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum VideoToolboxDecoderError {
    /// The selected stream is not video.
    #[error("unsupported media type: {0:?} (VideoToolbox decode is video-only)")]
    UnsupportedMediaType(ffmpeg::media::Type),

    /// No decoder this FFmpeg build has for the codec can decode through
    /// VideoToolbox — see [`VideoToolboxDecoder::supports`].
    #[error("no decoder in this FFmpeg build decodes {0:?} through VideoToolbox")]
    UnsupportedCodec(ffmpeg::codec::Id),

    /// FFmpeg rejected decoder or packet/frame processing.
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),

    /// FFmpeg could not take another reference to the device context.
    #[error("failed to reference the VideoToolbox device context")]
    HwDeviceRef,

    /// VideoToolbox will not decode this stream: not this codec's profile,
    /// or not this size, on this Mac.
    #[error(
        "decoder did not select the VideoToolbox pixel format — VideoToolbox \
         decode unavailable for this stream on this Mac"
    )]
    HwAccelUnavailable,
}

/// Decodes one video stream's `Packet`s into VideoToolbox frames through
/// FFmpeg's VideoToolbox hwaccel, on a [`VideoToolboxDevice`] — the Mac's
/// own media engine, for H.264, HEVC and whatever else this Mac decodes in
/// hardware. A `Filter`, same shape as [`crate::elements::SwDecoder`].
///
/// Frames this produces are still plain `MediaBuffer::Video`, tagged
/// [`ffmpeg::format::Pixel::VIDEOTOOLBOX`]: `Pacer`, `Tee` and `Queue` take
/// them as they take any frame, and
/// [`crate::elements::VideoToolboxDownload`] brings one back to system memory
/// for anything that reads pixel bytes.
///
/// Its pictures come from Core Video's own pool, which grows as they are
/// held, so unlike `CudaDecoder` it is given no budget of frames downstream
/// may hold; playing backwards, a stretch is held as it was decoded.
pub struct VideoToolboxDecoder {
    pp_log: PpLog,
    name: Arc<str>,
    decoder: ffmpeg::decoder::Video,
    _hw_device_ctx: Arc<AvBufferRef>,
    pad: SrcPad,
    /// Reused across every decoded frame; the image itself is pooled by
    /// FFmpeg's own frames context, so this only recycles the `AVFrame`
    /// wrapper.
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
    /// Suppresses decoded samples before a seek target during preroll.
    preroll_gate: PrerollGate,
    /// Set by `get_format` when the GPU refuses the stream — see
    /// [`NegotiationRefusal`]. After `decoder`, so it outlives the codec
    /// context that points at it.
    refused: NegotiationRefusal,
    /// Holds a stretch's pictures to hand on last first, playing backwards.
    stretch: Stretch,
    /// What it leaves undecoded while pictures come too late — see `Qos`.
    qos: Qos,
}

// SAFETY: `hw_device_ctx` is a heap-allocated FFmpeg buffer with no thread
// affinity of its own, and VideoToolbox sessions may be driven from any
// thread. `decoder`'s own `Send` covers the rest, and `&mut self` on every
// method that touches either rules out concurrent access. Same reasoning as
// `CudaDecoder`.
unsafe impl Send for VideoToolboxDecoder {}

impl VideoToolboxDecoder {
    /// `device` is the [`VideoToolboxDevice`] the pipeline's other
    /// VideoToolbox elements share. This decoder takes its own reference, so
    /// `device` itself need not outlive the call.
    ///
    /// The decoder opened is the first one FFmpeg has for the codec that can
    /// decode through VideoToolbox — see [`Self::supports`]. A codec with
    /// none fails here with [`VideoToolboxDecoderError::UnsupportedCodec`]
    /// rather than at the first frame.
    pub fn new(
        name: impl Into<String>,
        params: ffmpeg::codec::Parameters,
        device: &VideoToolboxDevice,
    ) -> Result<Self, VideoToolboxDecoderError> {
        crate::ensure_ffmpeg();
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::VideoToolboxDecoder, &name, None);

        let mut context = ffmpeg::codec::context::Context::from_parameters(params)?;
        if context.medium() != ffmpeg::media::Type::Video {
            return Err(VideoToolboxDecoderError::UnsupportedMediaType(
                context.medium(),
            ));
        }
        let codec = videotoolbox_capable_decoder(context.id())
            .ok_or(VideoToolboxDecoderError::UnsupportedCodec(context.id()))?;

        let hw_device_ctx = device.retain();
        let codec_device_ctx = hw_device_ctx
            .try_clone()
            .ok_or(VideoToolboxDecoderError::HwDeviceRef)?;
        let refused = NegotiationRefusal::new();
        // SAFETY: `ctx_ptr` is the codec context this owns and has not opened
        // yet, which is the only point at which these fields may be set. The
        // device reference is transferred with `into_raw`, so the codec frees
        // it and this no longer does. `opaque` points at `refused`, which the
        // decoder keeps for as long as the codec context.
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
                PortContract::frame(MediaKind::VideoFrame, MemoryDomain::VideoToolbox)
                    .with_layouts(crate::contract::PixelLayoutSet::decoded_from(
                        decoder.format(),
                    )),
            ),
        );
        pp_info!(pp_log: &pp_log, "opened: codec={:?} on VideoToolbox", decoder.id());
        Ok(Self {
            name,
            pp_log,
            decoder,
            _hw_device_ctx: hw_device_ctx,
            pad,
            pool: UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {}),
            preroll_gate: PrerollGate::default(),
            refused,
            stretch: Stretch::default(),
            qos: Qos::default(),
        })
    }

    /// Whether this FFmpeg build has a decoder for `codec` that can decode
    /// through VideoToolbox — whether [`Self::new`] gets past choosing one.
    /// Needs no device.
    ///
    /// `true` does not promise this Mac decodes every stream of the codec in
    /// hardware: one it will not fails at the first frame with
    /// [`VideoToolboxDecoderError::HwAccelUnavailable`].
    pub fn supports(codec: ffmpeg::codec::Id) -> bool {
        videotoolbox_capable_decoder(codec).is_some()
    }

    /// `error` as this decoder's own: the GPU refusing the stream where
    /// `get_format` said so — see [`NegotiationRefusal`].
    fn decode_error(&self, error: ffmpeg::Error) -> VideoToolboxDecoderError {
        if self.refused.happened() {
            VideoToolboxDecoderError::HwAccelUnavailable
        } else {
            error.into()
        }
    }

    /// Hands on what was held of the stretch just over, last first.
    fn release(&mut self) -> crate::error::Result<()> {
        for buffer in self.stretch.end() {
            self.preroll_gate.push_admitted(buffer, &mut self.pad)?;
        }
        Ok(())
    }

    fn drain(&mut self) -> crate::error::Result<()> {
        let mut frame = self.pool.get();
        loop {
            match self.decoder.receive_frame(&mut frame) {
                Ok(()) => {
                    if frame.format() != ffmpeg::format::Pixel::VIDEOTOOLBOX {
                        pp_error!(self, "decoder did not select the VideoToolbox pixel format");
                        return Err(VideoToolboxDecoderError::HwAccelUnavailable.into());
                    }
                    let buffer = MediaBuffer::Video(Arc::new(frame));
                    if self.stretch.holding() {
                        self.stretch.hold(buffer);
                    } else {
                        self.preroll_gate.push_admitted(buffer, &mut self.pad)?;
                    }
                    frame = self.pool.get();
                }
                Err(error) if is_codec_drain_boundary(&error) => break,
                Err(error) => return Err(self.decode_error(error).into()),
            }
        }
        Ok(())
    }
}

impl Element for VideoToolboxDecoder {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::VideoToolboxDecoder
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
        self.qos.attach(&context.state);
    }
}

impl Source for VideoToolboxDecoder {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for VideoToolboxDecoder {
    /// Not while a preroll this has already given its sample to is still
    /// running — see `PrerollGate::holding`.
    fn ready_consume(&mut self) -> bool {
        self.preroll_gate.ready(&mut self.pad)
    }

    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::packet(MediaKind::VideoPacket))
    }

    fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
        match buf {
            MediaBuffer::Packet(packet) => {
                // Decoded frames carry a `pts` but not the unit it is in.
                self.preroll_gate.observe_packet(&packet);
                self.qos.follow(&mut self.decoder, &self.pp_log);
                self.decoder
                    .send_packet(&*packet)
                    .inspect_err(|error| pp_error!(self, "send_packet failed: {error}"))
                    .map_err(|error| self.decode_error(error))?;
                self.drain()
            }
            _ => Ok(()),
        }
    }

    /// The segment the stream is in, which the preroll gate puts a seek's
    /// target on the samples' timeline by; and at the end, what the codec
    /// still holds, ahead of the end the graph passes on after it.
    fn stream_event(&mut self, event: &crate::stream::StreamEvent) -> crate::error::Result<()> {
        match event {
            crate::stream::StreamEvent::Segment(segment) => {
                self.preroll_gate.begin_segment(segment, &mut self.pad)
            }
            crate::stream::StreamEvent::Eos => {
                self.decoder
                    .send_eof()
                    .inspect_err(|error| pp_error!(self, "send_eof failed: {error}"))
                    .map_err(|error| self.decode_error(error))?;
                self.drain()?;
                self.preroll_gate.push_eos_candidate(&mut self.pad)
            }
        }
    }

    fn flow(&mut self, Flow(msg): Flow<'_>) -> crate::error::Result<()> {
        // Reference-frame state belongs to the timeline a seek leaves; the
        // samples decoded while a preroll catches up exist only to warm the
        // codec.
        if *msg == ControlMsg::Flush {
            self.decoder.flush();
            self.qos.reset(&mut self.decoder);
            self.preroll_gate.reset();
            self.stretch.reset();
        }
        Ok(())
    }

    fn as_reversible(&mut self) -> Option<&mut dyn ReversibleDecoder> {
        Some(self)
    }
}

impl ReversibleDecoder for VideoToolboxDecoder {
    fn begin_stretch(&mut self) -> crate::error::Result<()> {
        self.stretch.begin();
        Ok(())
    }

    /// Drains the decoder of the stretch and leaves it fresh for the next
    /// one's keyframe, then hands the stretch on.
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

impl Drop for VideoToolboxDecoder {
    fn drop(&mut self) {
        pp_info!(self, "dropped: releasing hw_device_ctx");
    }
}

/// The first decoder for `id` that decodes through VideoToolbox — see
/// [`capable_decoder`].
fn videotoolbox_capable_decoder(id: ffmpeg::codec::Id) -> Option<ffmpeg::Codec> {
    capable_decoder(id, ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VIDEOTOOLBOX)
}

/// Picks `AV_PIX_FMT_VIDEOTOOLBOX` out of whatever libavcodec offers;
/// libavcodec makes the frames context itself.
unsafe extern "C" fn get_format(
    ctx: *mut ffi::AVCodecContext,
    mut fmt: *const ffi::AVPixelFormat,
) -> ffi::AVPixelFormat {
    // SAFETY: `fmt` is FFmpeg's own `AV_PIX_FMT_NONE`-terminated list, which is
    // what `get_format` is defined to be handed, so the walk stops inside it.
    unsafe {
        while *fmt != ffi::AVPixelFormat::AV_PIX_FMT_NONE {
            if *fmt == ffi::AVPixelFormat::AV_PIX_FMT_VIDEOTOOLBOX {
                return ffi::AVPixelFormat::AV_PIX_FMT_VIDEOTOOLBOX;
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
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::{
        elements::VideoToolboxDownload,
        test_support::{CapturingSink, try_test_video, try_videotoolbox_device},
    };

    /// The fixture's picture stream: its parameters, and every packet of it.
    fn fixture() -> Option<(ffmpeg::codec::Parameters, Vec<ffmpeg::Packet>)> {
        let path = try_test_video()?;
        let mut input = ffmpeg::format::input(&path).expect("open the test video");
        let stream = input
            .streams()
            .best(ffmpeg::media::Type::Video)
            .expect("the test video has a picture");
        let (index, params) = (stream.index(), stream.parameters());
        let packets = input
            .packets()
            .filter(|(stream, _)| stream.index() == index)
            .map(|(_, packet)| packet)
            .collect();
        Some((params, packets))
    }

    fn capture(decoder: &mut VideoToolboxDecoder) -> Arc<Mutex<Vec<MediaBuffer>>> {
        let received = Arc::new(Mutex::new(Vec::new()));
        decoder.src_pads()[0].link(Box::new(CapturingSink {
            received: received.clone(),
            pp_log: element_pp_log(ElementType::Other, "capture", None),
        }));
        received
    }

    /// Frames come out as VideoToolbox frames with their timestamps, every
    /// one of the stream, and read back as a picture; the end of the stream
    /// is passed on. Every frame is kept until the end, which a decoder whose
    /// pool could not grow would fail at.
    #[test]
    fn decodes_the_whole_stream_into_videotoolbox_frames() {
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let Some((params, packets)) = fixture() else {
            return;
        };
        if !VideoToolboxDecoder::supports(params.id()) {
            eprintln!(
                "skipping: no VideoToolbox decoder for {:?} in this FFmpeg",
                params.id()
            );
            return;
        }
        let mut decoder = VideoToolboxDecoder::new("vt-decoder", params, &device).unwrap();
        let received = capture(&mut decoder);
        let sent = packets.len();
        for packet in packets {
            match decoder.consume(MediaBuffer::Packet(Arc::new(packet))) {
                Err(crate::error::Error::VideoToolboxDecoderError(
                    VideoToolboxDecoderError::HwAccelUnavailable,
                )) => {
                    eprintln!(
                        "skipping: this Mac does not decode the fixture through VideoToolbox"
                    );
                    return;
                }
                result => result.expect("decode"),
            }
        }
        crate::stream::deliver(&mut decoder, &crate::stream::StreamEvent::Eos).expect("eos");

        let received = std::mem::take(&mut *received.lock().unwrap());
        let frames: Vec<_> = received
            .iter()
            .filter_map(|buffer| match buffer {
                MediaBuffer::Video(frame) => Some(frame.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(frames.len(), sent, "one picture a packet, none lost");
        let mut pts: Vec<_> = frames.iter().map(|frame| frame.pts()).collect();
        assert!(pts.iter().all(Option::is_some), "every picture has its pts");
        pts.dedup();
        assert_eq!(pts.len(), frames.len(), "and each its own");
        assert!(
            frames
                .iter()
                .all(|frame| frame.format() == ffmpeg::format::Pixel::VIDEOTOOLBOX)
        );

        let mut download = VideoToolboxDownload::new("download");
        let back = Arc::new(Mutex::new(Vec::new()));
        download.src_pads()[0].link(Box::new(CapturingSink {
            received: back.clone(),
            pp_log: element_pp_log(ElementType::Other, "back", None),
        }));
        download
            .consume(MediaBuffer::Video(frames[0].clone()))
            .expect("read the first picture back");
        let back = back.lock().unwrap();
        let MediaBuffer::Video(picture) = &back[0] else {
            panic!("expected a picture");
        };
        assert_eq!(
            (picture.width(), picture.height()),
            (frames[0].width(), frames[0].height())
        );
        assert!(
            picture
                .data(0)
                .iter()
                .any(|&luma| luma != picture.data(0)[0]),
            "the picture read back is a picture, not one flat value"
        );
    }

    /// Played backwards, a stretch comes out last picture first, every one
    /// of it.
    #[test]
    fn a_stretch_comes_out_last_first() {
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let Some((params, packets)) = fixture() else {
            return;
        };
        if !VideoToolboxDecoder::supports(params.id()) {
            return;
        }
        let mut decoder = VideoToolboxDecoder::new("vt-decoder", params, &device).unwrap();
        let received = capture(&mut decoder);
        // The first stretch the fixture has: from its first keyframe to the
        // packet before the next.
        let next_key = packets
            .iter()
            .skip(1)
            .position(|packet| packet.is_key())
            .map_or(packets.len(), |at| at + 1);
        decoder.begin_stretch().unwrap();
        for packet in packets.into_iter().take(next_key) {
            match decoder.consume(MediaBuffer::Packet(Arc::new(packet))) {
                Err(crate::error::Error::VideoToolboxDecoderError(
                    VideoToolboxDecoderError::HwAccelUnavailable,
                )) => {
                    return;
                }
                result => result.expect("decode"),
            }
        }
        assert!(
            received.lock().unwrap().is_empty(),
            "held until the stretch ends"
        );
        decoder.end_stretch().unwrap();
        let pts: Vec<_> = received
            .lock()
            .unwrap()
            .iter()
            .filter_map(|buffer| match buffer {
                MediaBuffer::Video(frame) => frame.pts(),
                _ => None,
            })
            .collect();
        assert_eq!(pts.len(), next_key, "every picture of the stretch");
        assert!(
            pts.windows(2).all(|pair| pair[0] > pair[1]),
            "last first: {pts:?}"
        );
    }

    /// Theora has no VideoToolbox decoder, and H.264 has one in every
    /// FFmpeg built for a Mac. Answered from FFmpeg's own tables, so it holds
    /// with no device.
    #[test]
    fn supports_answers_from_ffmpegs_decoders() {
        assert!(!VideoToolboxDecoder::supports(ffmpeg::codec::Id::THEORA));
        assert!(VideoToolboxDecoder::supports(ffmpeg::codec::Id::H264));
    }
}
