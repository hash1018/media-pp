use std::sync::Arc;

use crate::pp_log::{PpLog, pp_error, pp_info};
use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    control::ControlMsg,
    element::{Element, ElementType, ReversibleDecoder, Sink, Source, element_pp_log},
    pad::SrcPad,
    pool::UnboundObjectPool,
};

use super::backwards::Stretch;
use super::preroll_gate::PrerollGate;
use super::qos::Qos;
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

/// How a software video decoder's threads share the work: on several
/// pictures at once, or on the pieces of one.
///
/// FFmpeg runs a decoder one way or the other, never both. Several pictures
/// at once is how H.264 and HEVC get most of theirs — about five and four
/// times one thread, measured at 1080p on twelve — and it hands each picture
/// out as many pictures later as there are threads: some four hundred
/// milliseconds at 30 a second on twelve. A file has no deadline and does not
/// notice; something live does. Codecs that split one picture into
/// independent pieces — ProRes's slices, VP9's tiles — gain most of it within
/// one picture, about six and two times, and hold nothing back — and for one
/// whose every picture stands alone, several at once gains nothing: at 1080p
/// ProRes decoded no faster that way, and took 335 MB against 93.
///
/// Which suits a stream depends on its codec, its size and what it is for,
/// so it is the caller's to say; nothing here chooses.
///
/// Audio decoders are left as FFmpeg opens them: none gains anything here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeThreadKind {
    /// Within one picture only, so nothing is held back that a single
    /// thread would not hold. For a live source, whose picture is wanted
    /// now. A codec that cannot split a picture decodes on one thread.
    Slice,
    /// Several pictures at once, a picture per thread later. A codec that
    /// cannot do that is split within each picture instead, as with
    /// [`Self::Slice`].
    Frame,
}

/// How many threads a software video decoder has, and how they share the
/// work.
///
/// Where several decode at once, or share the machine with an encoder or a
/// compositor, a count keeps them from each taking all of it; several
/// pictures at once also keeps a copy of the decoder's state per thread, so
/// the count bounds memory too.
///
/// Given to [`SwDecoder::with_threading`]. [`SwDecoder::new`] sets none of
/// this, and decodes on the one thread FFmpeg opens a decoder with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeThreading {
    /// How many threads; `None` for as many as the machine has, which is
    /// FFmpeg's own choice and capped by it.
    pub threads: Option<std::num::NonZeroU32>,
    /// How they share the work — see [`DecodeThreadKind`].
    pub kind: DecodeThreadKind,
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
    /// Holds a stretch's pictures to hand on last first, playing backwards
    /// — see [`ReversibleDecoder`].
    stretch: Stretch,
    /// What it leaves undecoded while pictures come too late — see `Qos`.
    qos: Qos,
}

impl SwDecoder {
    /// `params` should come from the stream you want to decode — see
    /// [`crate::elements::FileDemuxer::best`], whose `parameters` these are.
    ///
    /// Threading is left as FFmpeg opens a decoder, which is one thread;
    /// [`Self::with_threading`] is how to give it more.
    pub fn new(
        name: impl Into<String>,
        params: ffmpeg::codec::Parameters,
    ) -> Result<Self, SwDecoderError> {
        crate::ensure_ffmpeg();
        Self::open(name, params, None)
    }

    /// The same, with as many threads as `threading` says, spent as it says
    /// — see [`DecodeThreading`]. Ignored for audio.
    pub fn with_threading(
        name: impl Into<String>,
        params: ffmpeg::codec::Parameters,
        threading: DecodeThreading,
    ) -> Result<Self, SwDecoderError> {
        crate::ensure_ffmpeg();
        Self::open(name, params, Some(threading))
    }

    /// Both constructors: `None` leaves threading as FFmpeg has it.
    fn open(
        name: impl Into<String>,
        params: ffmpeg::codec::Parameters,
        threading: Option<DecodeThreading>,
    ) -> Result<Self, SwDecoderError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::SwDecoder, &name, None);
        let mut context = ffmpeg::codec::context::Context::from_parameters(params)?;
        if let Some(threading) = threading
            && context.medium() == ffmpeg::media::Type::Video
        {
            use ffmpeg::ffi::{FF_THREAD_FRAME, FF_THREAD_SLICE};
            // Both bits for Frame: FFmpeg takes frame threading where the
            // codec has it and falls back to slices where it does not.
            let kinds = match threading.kind {
                DecodeThreadKind::Slice => FF_THREAD_SLICE,
                DecodeThreadKind::Frame => FF_THREAD_FRAME | FF_THREAD_SLICE,
            };
            // Zero is FFmpeg's own "as many as the machine has".
            let count = threading.threads.map_or(0, |threads| {
                i32::try_from(threads.get()).unwrap_or(i32::MAX)
            });
            // SAFETY: `context` is exclusively owned and not opened yet,
            // which is the only time these two may be set.
            unsafe {
                let ctx = context.as_mut_ptr();
                (*ctx).thread_count = count;
                (*ctx).thread_type = kinds;
            }
        }

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
                Kind::Video(d) => {
                    let (threads, frames) = threads_in_use(d);
                    format!(
                        "video, codec={:?}, threads={threads}{}",
                        d.id(),
                        if frames { " (several pictures at once)" } else { "" }
                    )
                }
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
            stretch: Stretch::default(),
            qos: Qos::default(),
        })
    }
}

/// How many threads an opened decoder has, and whether they work on several
/// pictures at once — what FFmpeg settled on, which for a codec that cannot
/// do what was asked is less.
fn threads_in_use(decoder: &ffmpeg::decoder::Video) -> (i32, bool) {
    // SAFETY: reads two plain fields of the live context `decoder` owns.
    unsafe {
        let ctx = decoder.as_ptr();
        (
            (*ctx).thread_count,
            (*ctx).active_thread_type & ffmpeg::ffi::FF_THREAD_FRAME != 0,
        )
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

    /// The pipeline says when a seek's preroll starts and ends — see
    /// `PrerollGate`.
    fn attach_context(&mut self, context: &Arc<crate::element::Context>) {
        self.preroll_gate.attach(&context.state);
        self.qos.attach(&context.state);
    }
}

impl Source for SwDecoder {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for SwDecoder {
    /// Not while a preroll this has already given its sample to is still
    /// running — see `PrerollGate::holding`.
    fn ready_consume(&mut self) -> bool {
        !self.preroll_gate.holding()
    }

    /// The medium, not just "a packet": an audio stream wired into a
    /// decoder opened for video is the mistake this rules out, and both
    /// sides of it are `MediaBuffer::Packet`.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::packet(match &self.kind {
            Kind::Video(_) => MediaKind::VideoPacket,
            Kind::Audio(_) => MediaKind::AudioPacket,
        }))
    }

    /// A picture's decoder hands a stretch on backwards; a sound's is never
    /// given one.
    fn as_reversible(&mut self) -> Option<&mut dyn ReversibleDecoder> {
        if matches!(self.kind, Kind::Video(_)) {
            Some(self)
        } else {
            None
        }
    }

    fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
        match buf {
            MediaBuffer::Packet(packet) => {
                // Decoded frames carry a `pts` but not the unit it is in, so
                // the gate learns that from the packets on the way in.
                self.preroll_gate.observe_packet(&packet);
                match &mut self.kind {
                    Kind::Video(decoder) => {
                        self.qos.follow(decoder, &self.pp_log);
                        decoder
                            .send_packet(&*packet)
                            .inspect_err(|error| pp_error!(self, "send_packet failed: {error}"))
                            .map_err(SwDecoderError::from)?;
                        drain_video(
                            decoder,
                            &mut self.pad,
                            &self.pool,
                            &mut self.preroll_gate,
                            &mut self.stretch,
                        )
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
                        drain_video(
                            decoder,
                            &mut self.pad,
                            &self.pool,
                            &mut self.preroll_gate,
                            &mut self.stretch,
                        )?;
                    }
                    Kind::Audio(decoder) => {
                        decoder
                            .send_eof()
                            .inspect_err(|error| pp_error!(self, "send_eof failed: {error}"))
                            .map_err(SwDecoderError::from)?;
                        drain_audio(decoder, &mut self.pad, &mut self.preroll_gate)?;
                    }
                }
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
        if *msg == ControlMsg::Flush {
            match &mut self.kind {
                Kind::Video(decoder) => {
                    decoder.flush();
                    self.qos.reset(decoder);
                }
                Kind::Audio(decoder) => decoder.flush(),
            }
            self.preroll_gate.reset();
            self.stretch.reset();
        }
        Ok(())
    }
}

impl ReversibleDecoder for SwDecoder {
    fn begin_stretch(&mut self) -> crate::error::Result<()> {
        self.stretch.begin();
        Ok(())
    }

    /// Drains the decoder of the stretch and leaves it fresh for the next
    /// one's keyframe, then hands the stretch on.
    fn end_stretch(&mut self) -> crate::error::Result<()> {
        if let Kind::Video(decoder) = &mut self.kind {
            decoder
                .send_eof()
                .inspect_err(|error| pp_error!(self, "send_eof failed: {error}"))
                .map_err(SwDecoderError::from)?;
            drain_video(
                decoder,
                &mut self.pad,
                &self.pool,
                &mut self.preroll_gate,
                &mut self.stretch,
            )?;
            decoder.flush();
        }
        release(&mut self.stretch, &mut self.pad, &mut self.preroll_gate)
    }
}

fn drain_video(
    decoder: &mut ffmpeg::decoder::Video,
    pad: &mut SrcPad,
    pool: &UnboundObjectPool<ffmpeg::frame::Video>,
    gate: &mut PrerollGate,
    stretch: &mut Stretch,
) -> crate::error::Result<()> {
    let mut frame = pool.get();
    loop {
        match decoder.receive_frame(&mut frame) {
            Ok(()) => {
                let buffer = MediaBuffer::Video(Arc::new(frame));
                if stretch.holding() {
                    stretch.hold(buffer);
                } else {
                    // Reassigning `frame` releases the suppressed one right
                    // here. On a fixed hardware pool that returns its surface
                    // a whole branch earlier than dropping it downstream would.
                    gate.push_admitted(buffer, |frame| pad.push(frame))?;
                }
                frame = pool.get();
            }
            Err(error) if is_codec_drain_boundary(&error) => break,
            Err(error) => return Err(SwDecoderError::from(error).into()),
        }
    }
    Ok(())
}

/// Hands on what `stretch` held of the stretch just over, last first.
fn release(
    stretch: &mut Stretch,
    pad: &mut SrcPad,
    gate: &mut PrerollGate,
) -> crate::error::Result<()> {
    for buffer in stretch.end() {
        gate.push_admitted(buffer, |frame| pad.push(frame))?;
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
                gate.push_admitted(MediaBuffer::Audio(Arc::new(frame)), |frame| pad.push(frame))?;
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

        fn control(&mut self, msg: &ControlMsg) -> Result<()> {
            self.received.lock().unwrap().controls.push(msg.clone());
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
    /// run without waiting on a fixture to be built.
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

        crate::control::deliver(&mut decoder, &ControlMsg::Flush).expect("flush must not fail");

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
            crate::control::deliver(&mut decoder, &msg).expect("control must not fail");
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

    fn h264() -> (ffmpeg::codec::Parameters, Vec<ffmpeg::Packet>) {
        crate::test_support::try_encoded_packets(
            "libopenh264",
            ffmpeg::format::Pixel::YUV420P,
            (64, 64),
            90,
        )
        .expect("OpenH264 is always present")
    }

    fn threaded(kind: DecodeThreadKind, threads: Option<u32>) -> SwDecoder {
        let threading = DecodeThreading {
            threads: threads.and_then(std::num::NonZeroU32::new),
            kind,
        };
        SwDecoder::with_threading("threaded", h264().0, threading).expect("H.264 opens")
    }

    fn threads(decoder: &SwDecoder) -> (i32, bool) {
        let Kind::Video(video) = &decoder.kind else {
            panic!("an H.264 decoder decodes video");
        };
        threads_in_use(video)
    }

    /// Frame works on several pictures at once, and Slice does not — what
    /// FFmpeg settled on, not only what was asked of it.
    #[test]
    fn frame_decodes_several_pictures_at_once_and_slice_does_not() {
        let (count, frames) = threads(&threaded(DecodeThreadKind::Frame, Some(4)));
        assert_eq!(count, 4);
        assert!(frames, "H.264 decodes a picture per thread");

        let (count, frames) = threads(&threaded(DecodeThreadKind::Slice, Some(4)));
        assert_eq!(count, 4, "Slice keeps its threads");
        assert!(!frames, "and does not spend them on several pictures");
    }

    /// No count is FFmpeg's own, which is more than one on any machine that
    /// has more than one core.
    #[test]
    fn no_count_is_as_many_as_the_machine_has() {
        let cores = std::thread::available_parallelism().map_or(1, usize::from);
        let (count, _) = threads(&threaded(DecodeThreadKind::Frame, None));
        if cores > 1 {
            assert!(count > 1, "{count} threads on {cores} cores");
        }
    }

    /// What Slice is for: every picture a packet completes comes out with
    /// that packet, where Frame holds some back until the stream ends — and
    /// then hands every one of them over.
    #[test]
    fn slice_hands_each_picture_over_at_once_and_frame_catches_up_at_the_end() {
        let decoded_before_eos = |kind| {
            let (params, packets) = h264();
            let sent = packets.len();
            let threading = DecodeThreading {
                threads: std::num::NonZeroU32::new(4),
                kind,
            };
            let mut decoder =
                SwDecoder::with_threading("kind", params, threading).expect("H.264 opens");
            let received = link_capture(&mut decoder);
            for packet in packets {
                decoder
                    .consume(MediaBuffer::Packet(Arc::new(packet)))
                    .expect("decodes");
            }
            let before = received.lock().unwrap().buffers.len();
            decoder.consume(MediaBuffer::Eos).expect("drains");
            let pictures = received
                .lock()
                .unwrap()
                .buffers
                .iter()
                .filter(|buf| matches!(buf, MediaBuffer::Video(_)))
                .count();
            assert_eq!(pictures, sent, "{kind:?} decodes every picture");
            (before, sent)
        };

        let (before, sent) = decoded_before_eos(DecodeThreadKind::Slice);
        assert_eq!(before, sent, "Slice holds nothing back");
        let (before, sent) = decoded_before_eos(DecodeThreadKind::Frame);
        assert!(
            before < sent,
            "Frame on four threads holds pictures back ({before} of {sent})"
        );
    }

    /// `new` sets no threading at all: the decoder is what FFmpeg opens,
    /// one thread, as it was before there was a choice.
    #[test]
    fn new_leaves_threading_as_ffmpeg_has_it() {
        assert_eq!(threads(&video_decoder("plain")), (1, false));
    }
}
