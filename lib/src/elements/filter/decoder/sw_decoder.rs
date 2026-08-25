use std::sync::Arc;

use crate::pp_log::{PpLog, pp_error, pp_info};
use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    pad::SrcPad,
    pool::UnboundObjectPool,
};

use super::preroll_gate::PrerollGate;
use crate::elements::filter::is_codec_drain_boundary;

/// Errors specific to `SwDecoder`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum SwDecoderError {
    /// The selected stream is neither audio nor video.
    #[error("unsupported media type: {0:?}")]
    UnsupportedMediaType(ffmpeg::media::Type),

    /// FFmpeg rejected decoder creation or packet/frame processing.
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),
}

enum Kind {
    Video(ffmpeg::decoder::Video),
    Audio(ffmpeg::decoder::Audio),
}

/// Decodes one stream's `Packet`s into `Frame`s in software (plain
/// libavcodec, no hardware acceleration). A `Filter`: receives via `Sink`,
/// pushes what it produces into its own (single) src pad.
///
/// One packet can turn into zero, one, or several frames (B-frame
/// reordering, decoder buffering, ...) — `consume` just drains
/// `receive_frame` in a loop after every `send_packet`/`send_eof`, pushing
/// however many frames come out.
pub struct SwDecoder {
    pp_log: PpLog,
    name: Arc<str>,
    kind: Kind,
    pad: SrcPad,
    /// Reused across every decoded video frame instead of allocating a
    /// fresh one each time — see [`UnboundObjectPool`]'s docs. Starts
    /// empty: decoded format/dimensions aren't known until the first
    /// frame actually comes out of the decoder, so `init` just makes an
    /// empty frame (`avcodec_receive_frame` allocates it on first use,
    /// same as before this existed) and the pool fills organically as
    /// frames get returned. Unused (harmlessly) if this turns out to be
    /// an audio decoder — `MediaBuffer::Audio` isn't pooled.
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
    /// Suppresses decoded samples before a seek target during preroll.
    preroll_gate: PrerollGate,
}

impl SwDecoder {
    /// `params` should come from the stream you want to decode — see
    /// [`crate::elements::FileDemuxer::stream_parameters`].
    pub fn new(
        name: impl Into<String>,
        params: ffmpeg::codec::Parameters,
    ) -> Result<Self, SwDecoderError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::SwDecoder, &name, None);
        let context = ffmpeg::codec::context::Context::from_parameters(params)?;

        let kind = match context.medium() {
            ffmpeg::media::Type::Video => Kind::Video(context.decoder().video()?),
            ffmpeg::media::Type::Audio => Kind::Audio(context.decoder().audio()?),
            other => return Err(SwDecoderError::UnsupportedMediaType(other)),
        };

        // Which of the two this decoder emits is settled here, by the
        // stream parameters, even though the decoded format and size are
        // not known until the first frame comes back out. That split is
        // exactly what a link check can and cannot know at wiring time.
        let produced = PortContract::frame(
            match &kind {
                Kind::Video(_) => MediaKind::VideoFrame,
                Kind::Audio(_) => MediaKind::AudioFrame,
            },
            MemoryDomain::System,
        );
        let pad = SrcPad::with_contract(format!("{name}_src"), OutputContract::Fixed(produced));
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        pp_info!(
            pp_log: &pp_log,
            "opened: {}",
            match &kind {
                Kind::Video(d) => format!("video, codec={:?}", d.id()),
                Kind::Audio(d) => format!("audio, codec={:?}", d.id()),
            }
        );
        Ok(Self {
            name,
            pp_log,
            kind,
            pad,
            pool,
            preroll_gate: PrerollGate::default(),
        })
    }
}

impl Element for SwDecoder {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::SwDecoder
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for SwDecoder {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for SwDecoder {
    /// The medium, not just "a packet": an audio stream wired into a
    /// decoder opened for video is the mistake this rules out, and both
    /// sides of it are `MediaBuffer::Packet`.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::packet(match &self.kind {
            Kind::Video(_) => MediaKind::VideoPacket,
            Kind::Audio(_) => MediaKind::AudioPacket,
        }))
    }

    fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
        match buf {
            MediaBuffer::Packet(packet) => {
                // Decoded frames carry a `pts` but not the unit it is in, so
                // the gate learns that from the packets on the way in.
                self.preroll_gate.observe_packet(&packet);
                match &mut self.kind {
                    Kind::Video(decoder) => {
                        decoder
                            .send_packet(&*packet)
                            .inspect_err(|error| pp_error!(self, "send_packet failed: {error}"))
                            .map_err(SwDecoderError::from)?;
                        drain_video(decoder, &mut self.pad, &self.pool, &mut self.preroll_gate)
                    }
                    Kind::Audio(decoder) => {
                        decoder
                            .send_packet(&*packet)
                            .inspect_err(|error| pp_error!(self, "send_packet failed: {error}"))
                            .map_err(SwDecoderError::from)?;
                        drain_audio(decoder, &mut self.pad, &mut self.preroll_gate)
                    }
                }
            }
            MediaBuffer::Eos => {
                match &mut self.kind {
                    Kind::Video(decoder) => {
                        decoder
                            .send_eof()
                            .inspect_err(|error| pp_error!(self, "send_eof failed: {error}"))
                            .map_err(SwDecoderError::from)?;
                        drain_video(decoder, &mut self.pad, &self.pool, &mut self.preroll_gate)?;
                    }
                    Kind::Audio(decoder) => {
                        decoder
                            .send_eof()
                            .inspect_err(|error| pp_error!(self, "send_eof failed: {error}"))
                            .map_err(SwDecoderError::from)?;
                        drain_audio(decoder, &mut self.pad, &mut self.preroll_gate)?;
                    }
                }
                if let Some(candidate) = self.preroll_gate.finish_on_eos() {
                    self.pad.push(candidate)?;
                }
                self.pad.push(MediaBuffer::Eos)
            }
            other => {
                let _ = other;
                Ok(())
            }
        }
    }

    fn control(&mut self, msg: ControlMsg) -> crate::error::Result<()> {
        // `Stop`: no local reaction needed — abandon means there's
        // nothing to flush before this decoder's own `Drop` frees the
        // codec context.
        //
        // `Flush` discards leftover reference/reordering state before a new
        // timeline starts. `Seek` itself only announces the new position.
        //
        // `Preroll` may carry a seek target, in which case the decoded samples
        // this decoder produces while catching up to it exist only to warm the
        // codec and must not be forwarded.
        match &msg {
            ControlMsg::Flush => {
                match &mut self.kind {
                    Kind::Video(decoder) => decoder.flush(),
                    Kind::Audio(decoder) => decoder.flush(),
                }
                self.preroll_gate.reset();
            }
            ControlMsg::Preroll(context) => self.preroll_gate.begin(context),
            ControlMsg::Pause | ControlMsg::Resume | ControlMsg::Stop => self.preroll_gate.clear(),
            ControlMsg::CheckSeek(_) | ControlMsg::Seek(_) => {}
        }
        self.pad.control(msg)
    }
}

fn drain_video(
    decoder: &mut ffmpeg::decoder::Video,
    pad: &mut SrcPad,
    pool: &UnboundObjectPool<ffmpeg::frame::Video>,
    gate: &mut PrerollGate,
) -> crate::error::Result<()> {
    let mut frame = pool.get();
    loop {
        match decoder.receive_frame(&mut frame) {
            Ok(()) => {
                // Reassigning `frame` releases the suppressed one right here.
                // On a fixed hardware pool that returns its surface a whole
                // branch earlier than dropping it downstream would.
                if let Some(frame) = gate.admit(MediaBuffer::Video(Arc::new(frame))) {
                    pad.push(frame)?;
                }
                frame = pool.get();
            }
            Err(error) if is_codec_drain_boundary(&error) => break,
            Err(error) => return Err(SwDecoderError::from(error).into()),
        }
    }
    Ok(())
}

fn drain_audio(
    decoder: &mut ffmpeg::decoder::Audio,
    pad: &mut SrcPad,
    gate: &mut PrerollGate,
) -> crate::error::Result<()> {
    let mut frame = ffmpeg::frame::Audio::empty();
    loop {
        match decoder.receive_frame(&mut frame) {
            Ok(()) => {
                if let Some(frame) = gate.admit(MediaBuffer::Audio(Arc::new(frame))) {
                    pad.push(frame)?;
                }
                frame = ffmpeg::frame::Audio::empty();
            }
            Err(error) if is_codec_drain_boundary(&error) => break,
            Err(error) => return Err(SwDecoderError::from(error).into()),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::{element::ElementType, error::Result, test_support::try_test_video};

    #[derive(Default)]
    struct Received {
        buffers: Vec<MediaBuffer>,
        controls: Vec<ControlMsg>,
    }

    struct CapturingSink {
        pp_log: PpLog,
        received: Arc<Mutex<Received>>,
    }

    impl Element for CapturingSink {
        fn name(&self) -> Arc<str> {
            "capture".into()
        }

        fn element_type(&self) -> ElementType {
            ElementType::Other
        }

        fn pp_log(&self) -> &PpLog {
            &self.pp_log
        }

        fn pp_log_mut(&mut self) -> &mut PpLog {
            &mut self.pp_log
        }
    }

    impl Sink for CapturingSink {
        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            self.received.lock().unwrap().buffers.push(buf);
            Ok(())
        }

        fn control(&mut self, msg: ControlMsg) -> Result<()> {
            self.received.lock().unwrap().controls.push(msg);
            Ok(())
        }
    }

    fn link_capture(decoder: &mut SwDecoder) -> Arc<Mutex<Received>> {
        let received = Arc::new(Mutex::new(Received::default()));
        decoder.src_pads()[0].link(Box::new(CapturingSink {
            received: received.clone(),
            pp_log: element_pp_log(ElementType::Other, "capture", None),
        }));
        received
    }

    /// A bare H.264 decoder, built without any fixture: only `codec_type`/
    /// `codec_id` decide which decoder `SwDecoder::new` opens, and libavcodec
    /// takes an Annex B stream's parameters from the bitstream itself rather
    /// than from `extradata`. Lets the control/passthrough contracts below
    /// run on every machine instead of only where `MEDIA_PP_TEST_VIDEO` is set.
    fn video_decoder(name: &str) -> SwDecoder {
        let mut params = ffmpeg::codec::Parameters::new();
        // SAFETY: `as_mut_ptr` on parameters this test just created and still
        // owns exclusively; both are plain fields of `AVCodecParameters`.
        unsafe {
            (*params.as_mut_ptr()).codec_type = ffmpeg::media::Type::Video.into();
            (*params.as_mut_ptr()).codec_id = ffmpeg::codec::Id::H264.into();
        }
        SwDecoder::new(name, params).expect("failed to open the built-in H.264 decoder")
    }

    /// Neither audio nor video has to fail here with this element's own typed
    /// error, not somewhere inside libavcodec later. Freshly allocated
    /// parameters report `Unknown`, which is exactly that case.
    #[test]
    fn parameters_that_are_neither_audio_nor_video_are_rejected() {
        let error = SwDecoder::new("decoder", ffmpeg::codec::Parameters::new())
            .err()
            .expect("unknown-medium parameters must not open a decoder");

        assert!(
            matches!(
                error,
                SwDecoderError::UnsupportedMediaType(ffmpeg::media::Type::Unknown)
            ),
            "expected UnsupportedMediaType, got {error:?}"
        );
    }

    /// `Flush` is the one control this element reacts to locally — it resets
    /// the codec's reference/reordering state so packets from the new position
    /// don't decode against stale state. It must still reach the rest of the
    /// branch afterwards; swallowing it would silently strand every downstream
    /// element at the old position.
    #[test]
    fn flush_is_forwarded_downstream_after_resetting() {
        let mut decoder = video_decoder("decoder");
        let received = link_capture(&mut decoder);

        decoder
            .control(ControlMsg::Flush)
            .expect("flush must not fail");

        assert_eq!(received.lock().unwrap().controls, [ControlMsg::Flush]);
    }

    /// Every other control passes straight through — this element has no
    /// local reaction to them (see `SwDecoder::control`'s own comment).
    #[test]
    fn other_controls_are_forwarded_unchanged() {
        let mut decoder = video_decoder("decoder");
        let received = link_capture(&mut decoder);

        for msg in [
            ControlMsg::Pause,
            ControlMsg::Resume,
            ControlMsg::Stop,
            ControlMsg::Seek(std::time::Duration::from_secs(3)),
        ] {
            decoder.control(msg).expect("control must not fail");
        }

        assert_eq!(
            received.lock().unwrap().controls,
            [
                ControlMsg::Pause,
                ControlMsg::Resume,
                ControlMsg::Stop,
                ControlMsg::Seek(std::time::Duration::from_secs(3)),
            ]
        );
    }

    /// A buffer this decoder has nothing to do with is dropped, not forwarded:
    /// its src pad carries what *this* element decoded, so passing an
    /// already-decoded frame along would put a buffer on the pad that never
    /// came out of the codec.
    #[test]
    fn buffers_other_than_packets_and_eos_are_dropped() {
        let mut decoder = video_decoder("decoder");
        let received = link_capture(&mut decoder);

        decoder
            .consume(MediaBuffer::Audio(Arc::new(ffmpeg::frame::Audio::empty())))
            .expect("an unrelated buffer must not fail the decoder");

        assert!(
            received.lock().unwrap().buffers.is_empty(),
            "a buffer this decoder never produced was pushed downstream"
        );
    }

    /// The contract EOS draining exists for: frames the codec was still
    /// holding (B-frame reordering, decoder latency) must come out *before*
    /// the `Eos` that ends the branch, and each must keep its timestamp.
    /// Asserted against whatever fixture is configured — nothing here depends
    /// on its codec, size, or duration.
    #[test]
    fn eos_drains_delayed_frames_before_forwarding_it() {
        let Some(path) = try_test_video() else {
            return;
        };

        let mut input = ffmpeg::format::input(&path).expect("failed to open the test video");
        let stream = input
            .streams()
            .best(ffmpeg::media::Type::Video)
            .expect("the test video has no video stream");
        let stream_index = stream.index();
        let params = stream.parameters();

        let mut decoder = SwDecoder::new("decoder", params).expect("failed to open the decoder");
        let received = link_capture(&mut decoder);

        let mut sent = 0;
        for (stream, packet) in input.packets() {
            if stream.index() != stream_index {
                continue;
            }
            decoder
                .consume(MediaBuffer::Packet(Arc::new(packet)))
                .expect("decode failed");
            sent += 1;
            if sent >= 30 {
                break;
            }
        }
        decoder.consume(MediaBuffer::Eos).expect("eos failed");

        let received = received.lock().unwrap();
        let (last, frames) = received
            .buffers
            .split_last()
            .expect("the decoder pushed nothing at all");
        assert!(
            last.is_eos(),
            "Eos was not the last buffer — delayed frames escaped after it"
        );
        assert!(
            !frames.is_empty(),
            "no frames were decoded from the configured fixture"
        );
        for buf in frames {
            let MediaBuffer::Video(frame) = buf else {
                panic!("a video decoder pushed a buffer that was not a video frame");
            };
            assert!(frame.pts().is_some(), "a decoded frame lost its pts");
        }
    }
}
