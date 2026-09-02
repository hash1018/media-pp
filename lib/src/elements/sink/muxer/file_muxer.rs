use std::{
    path::Path,
    sync::{Arc, Mutex},
};

use crate::pp_log::{PpLog, pp_error};
use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, PortContract},
    control::{ControlMsg, SeekRejectReason},
    element::{Element, ElementType, Sink, element_pp_log},
    error::Result,
};

/// Errors specific to `FileMuxer`. Converts into the crate-wide `Error` via
/// `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum FileMuxerError {
    /// A stream sink received a buffer other than a packet or end-of-stream.
    #[error("FileMuxer stream sinks only accept Packet or Eos buffers, got {0}")]
    UnsupportedBuffer(&'static str),

    /// FFmpeg rejected muxer creation, packet writing, or finalization.
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),
}

/// One track registered via [`FileMuxer::add_stream`], waiting for
/// [`FileMuxer::open`] to turn it into a real [`FileMuxerStreamSink`] — its
/// `name` becomes that sink's own [`Element::name`]/`pp_log` identity, and
/// `input_time_base` is what every `Packet` it receives already carries
/// `pts`/`dts` in (the same one its upstream encoder was opened with).
struct PendingStream {
    name: Arc<str>,
    input_time_base: ffmpeg::Rational,
    /// Taken from the parameters this track was registered with, so its
    /// sink can refuse the other medium's packets at wiring time.
    kind: Option<MediaKind>,
}

/// Builds one container file with one or more tracks, then opens it into
/// one [`Sink`] per track.
///
/// Which container is the path's own: `format::output` asks FFmpeg to guess
/// a muxer from the file name, so `.mp4` gets MP4 and `.mkv` gets Matroska
/// out of the same type. Nothing here is MP4-specific.
///
/// Two-phase on purpose: a container's header has to describe
/// every stream's codec parameters up front — `avformat_write_header`
/// can't run until every [`FileMuxer::add_stream`] this file will ever hold
/// has already happened — so there's no way to make this a single
/// long-lived `Sink` that tracks attach to one at a time as their encoders
/// come online (contrast [`crate::elements::AudioMixer`], whose inputs
/// *can* attach at any time — it has no "known shape before the first
/// byte" constraint the way a container header does).
///
/// ```no_run
/// # use media_pp::ffmpeg;
/// # use media_pp::elements::{
/// #     AudioCodec, FileMuxer, SwAudioEncoder, SwAudioEncoderOptions, SwEncoder,
/// #     SwEncoderOptions, VideoCodec,
/// # };
/// # fn main() -> media_pp::Result<()> {
/// # let video_time_base = ffmpeg::Rational(1, 30);
/// # let audio_time_base = ffmpeg::Rational(1, 48_000);
/// # let video_encoder = SwEncoder::new("video", SwEncoderOptions {
/// #     codec: VideoCodec::H264,
/// #     width: 640,
/// #     height: 360,
/// #     time_base: video_time_base,
/// #     frame_rate: ffmpeg::Rational(30, 1),
/// #     bit_rate: 2_000_000,
/// #     gop_size: 30,
/// #     max_b_frames: None,
/// # })?;
/// # let audio_encoder = SwAudioEncoder::new("audio", SwAudioEncoderOptions {
/// #     codec: AudioCodec::Aac,
/// #     sample_rate: 48_000,
/// #     channels: 2,
/// #     time_base: audio_time_base,
/// #     bit_rate: 128_000,
/// # })?;
/// let mut muxer = FileMuxer::create("out.mp4")?;
/// muxer.add_stream("video", video_encoder.parameters(), video_time_base)?;
/// muxer.add_stream("audio", audio_encoder.parameters(), audio_time_base)?;
/// let mut sinks = muxer.open()?; // writes the header
/// let audio_sink = sinks.pop().unwrap();
/// let video_sink = sinks.pop().unwrap();
/// # Ok(())
/// # }
/// ```
pub struct FileMuxer {
    output: ffmpeg::format::context::Output,
    streams: Vec<PendingStream>,
}

impl FileMuxer {
    /// Allocates the output file. No header is written yet — nothing is on
    /// disk in a readable shape until [`FileMuxer::open`] runs.
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        let output = ffmpeg::format::output(&path).map_err(FileMuxerError::from)?;
        Ok(Self {
            output,
            streams: Vec::new(),
        })
    }

    /// Registers one more track this file will hold. `parameters`/
    /// `time_base` describe it — typically
    /// [`crate::elements::SwEncoder::parameters`]/the same `time_base`
    /// passed to its own `SwEncoderOptions` (or the
    /// [`crate::elements::SwAudioEncoder`] equivalents). `name` becomes
    /// this track's own [`Element::name`]/`pp_log` identity once
    /// [`FileMuxer::open`] turns it into a `Sink` — pick something that
    /// tells multiple tracks apart in logs/[`crate::bus::BusEvent`]s,
    /// e.g. `"video"`/`"audio"`.
    ///
    /// Add streams in the same order the caller will treat
    /// [`FileMuxer::open`]'s returned `Vec` — index 0 is whichever stream
    /// was added first, and so on.
    pub fn add_stream(
        &mut self,
        name: impl Into<String>,
        parameters: ffmpeg::codec::Parameters,
        time_base: ffmpeg::Rational,
    ) -> Result<()> {
        let mut stream = self
            .output
            .add_stream(parameters.id())
            .map_err(FileMuxerError::from)?;
        let kind = MediaKind::packet_for(parameters.medium());
        stream.set_time_base(time_base);
        stream.set_parameters(parameters);
        self.streams.push(PendingStream {
            name: name.into().into(),
            input_time_base: time_base,
            kind,
        });
        Ok(())
    }

    /// Writes the container header — every [`FileMuxer::add_stream`] call
    /// this file will ever get must already have happened — and returns
    /// one [`Sink`] per track, in the order [`FileMuxer::add_stream`] added
    /// them.
    ///
    /// All returned `Sink`s write into the same underlying file behind a
    /// shared lock: packets from independently-threaded branches (e.g. a
    /// video encode chain and an audio encode chain, each on their own
    /// [`crate::queue::Queue`]) can arrive concurrently, and neither
    /// `av_interleaved_write_frame` nor `av_write_trailer` is safe to call
    /// from multiple threads against the same file at once. They also
    /// share one trailer: it's written once every track has reported
    /// itself done — via `Eos` *or* [`ControlMsg::Stop`], either meaning
    /// "this track is finished" rather than "abandon the whole file" —
    /// not on whichever track finishes first, which would silently
    /// truncate whatever the other track(s) still had left to write. A
    /// single-track file (e.g. `screen_record_software`/`audio_record`) degenerates
    /// to finalizing on that one track's own `Eos`/`Stop`, same as before
    /// this type supported more than one.
    ///
    /// A caller driving multiple tracks from independent
    /// [`crate::pipeline::Pipeline`]s (today's architecture: one
    /// `SourceElement` per pipeline, so a live video capture and a live
    /// audio capture are necessarily two separate pipelines) is
    /// responsible for stopping all of them — the file's trailer only
    /// gets written once every track has actually reported done, so
    /// stopping only one pipeline while another keeps running leaves the
    /// file un-finalized (and unplayable) until the rest catch up too.
    pub fn open(mut self) -> Result<Vec<Box<dyn Sink>>> {
        self.output.write_header().map_err(FileMuxerError::from)?;
        let total = self.streams.len();
        let shared = Arc::new(FileMuxerShared {
            state: Mutex::new(MuxerState {
                output: self.output,
                done: 0,
                finished: false,
            }),
            total,
        });
        Ok(self
            .streams
            .into_iter()
            .enumerate()
            .map(|(index, stream)| -> Box<dyn Sink> {
                Box::new(FileMuxerStreamSink {
                    pp_log: element_pp_log(ElementType::FileMuxer, &stream.name, None),
                    name: stream.name,
                    shared: shared.clone(),
                    stream_index: index,
                    input_time_base: stream.input_time_base,
                    kind: stream.kind,
                    done: false,
                })
            })
            .collect())
    }
}

struct MuxerState {
    output: ffmpeg::format::context::Output,
    /// How many tracks have reported themselves finished (`Eos` or
    /// `Stop`) — the trailer is written once this reaches
    /// [`FileMuxerShared::total`], not on the first one (see
    /// [`FileMuxer::open`]'s own docs for why).
    done: usize,
    /// Set once the trailer has been written. Each
    /// [`FileMuxerStreamSink`]'s own `done` flag already prevents
    /// double-counting *that* track's contribution to `done`; this
    /// additionally guards [`FileMuxerShared::write_packet`] against
    /// writing into a file whose trailer has already closed it.
    finished: bool,
}

/// Shared between every [`FileMuxerStreamSink`] [`FileMuxer::open`] hands
/// out for the same file — one lock around the whole
/// [`ffmpeg::format::context::Output`] so concurrent tracks never
/// interleave two writes against it (see [`FileMuxer::open`]'s own docs).
struct FileMuxerShared {
    state: Mutex<MuxerState>,
    total: usize,
}

impl FileMuxerShared {
    fn write_packet(
        &self,
        stream_index: usize,
        input_time_base: ffmpeg::Rational,
        packet: &ffmpeg::Packet,
    ) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        if state.finished {
            return Ok(());
        }
        // Cloned, not mutated in place — `Arc<Packet>` may be shared with
        // another branch (e.g. a `PacketCounter` off the same `Tee`),
        // which must not see this stream's `set_stream`/rescaled
        // timestamps.
        let mut packet = packet.clone();
        let output_time_base = state
            .output
            .stream(stream_index)
            .expect("stream was added in FileMuxer::add_stream")
            .time_base();
        packet.rescale_ts(input_time_base, output_time_base);
        packet.set_stream(stream_index);
        packet.set_position(-1);
        packet
            .write_interleaved(&mut state.output)
            .map_err(FileMuxerError::from)?;
        Ok(())
    }

    /// One track reporting itself done (`Eos` or `Stop`) — writes the
    /// trailer exactly once, only once every track has called this.
    fn finish_track(&self) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        state.done += 1;
        if state.finished || state.done < self.total {
            return Ok(());
        }
        state.finished = true;
        state.output.write_trailer().map_err(FileMuxerError::from)?;
        Ok(())
    }
}

/// One track's own [`Sink`] — a lightweight handle sharing a
/// `FileMuxerShared` with every other track [`FileMuxer::open`] returned
/// alongside it. See [`FileMuxer::open`]'s own docs for the
/// finalize-once-every-track-is-done contract this relies on.
pub struct FileMuxerStreamSink {
    pp_log: PpLog,
    name: Arc<str>,
    shared: Arc<FileMuxerShared>,
    stream_index: usize,
    input_time_base: ffmpeg::Rational,
    /// The medium this track was registered for; `None` for one this
    /// crate does not model, which then declares nothing.
    kind: Option<MediaKind>,
    /// Set once this sink has contributed to
    /// [`FileMuxerShared::finish_track`] — guards against double-counting
    /// if both a natural `Eos` and a later `Stop` arrive for the same
    /// track.
    done: bool,
}

impl FileMuxerStreamSink {
    fn finish(&mut self) -> Result<()> {
        if self.done {
            return Ok(());
        }
        self.done = true;
        self.shared
            .finish_track()
            .inspect_err(|error| pp_error!(self, "write_trailer failed: {error}"))
    }
}

impl Element for FileMuxerStreamSink {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::FileMuxer
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Sink for FileMuxerStreamSink {
    /// A muxer interleaves already-encoded data; it has no encoder of
    /// its own, so a decoded frame has no route through it. The medium is
    /// this track's own, so a video encoder wired into the audio track is
    /// refused rather than writing a file no player can make sense of.
    fn input_contract(&self) -> InputContract {
        match self.kind {
            Some(kind) => InputContract::Fixed(PortContract::packet(kind)),
            None => InputContract::Unknown,
        }
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Packet(packet) => self
                .shared
                .write_packet(self.stream_index, self.input_time_base, &packet)
                .inspect_err(|error| pp_error!(self, "write_interleaved failed: {error}")),
            MediaBuffer::Eos => self.finish(),
            other => Err(FileMuxerError::UnsupportedBuffer(other.kind()).into()),
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        if let ControlMsg::CheckSeek(context) = &msg {
            context.reject(
                self.element_type(),
                self.name(),
                SeekRejectReason::ElementNotSeekable,
            );
        }
        // Terminal, nothing to forward. `Stop` still contributes to this
        // track's own "done" count — see `FileMuxer::open`'s own docs on
        // why the trailer waits for every track rather than finalizing on
        // whichever stops first.
        if msg == ControlMsg::Stop {
            self.finish()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::{SeekCheckContext, SeekRejectReason};
    use crate::element::Source;
    use crate::elements::{AudioCodec, SwAudioEncoder, SwAudioEncoderOptions};

    fn open_aac_encoder(sample_rate: u32, channels: u16) -> SwAudioEncoder {
        SwAudioEncoder::new(
            "encoder",
            SwAudioEncoderOptions {
                codec: AudioCodec::Aac,
                sample_rate,
                channels,
                time_base: ffmpeg::Rational::new(1, sample_rate as i32),
                bit_rate: 64_000,
            },
        )
        .expect("aac encoder must be available")
    }

    /// `None` where the build has no `libopus`, so a stripped FFmpeg skips
    /// rather than failing on something it was never going to have.
    fn open_opus_encoder(sample_rate: u32, channels: u16) -> Option<SwAudioEncoder> {
        SwAudioEncoder::new(
            "encoder",
            SwAudioEncoderOptions {
                codec: AudioCodec::Opus,
                sample_rate,
                channels,
                time_base: ffmpeg::Rational::new(1, sample_rate as i32),
                bit_rate: 64_000,
            },
        )
        .ok()
    }

    fn silent_frame(
        sample_rate: u32,
        channels: u16,
        samples: usize,
        pts: i64,
    ) -> ffmpeg::frame::Audio {
        let mut frame = ffmpeg::frame::Audio::new(
            ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed),
            samples,
            ffmpeg::ChannelLayout::default(channels as i32),
        );
        frame.set_rate(sample_rate);
        frame.set_pts(Some(pts));
        // `frame::Audio::new` doesn't zero its buffer — leaving it
        // uninitialized risks the encoder reading garbage bytes as NaN/Inf
        // floats (`avcodec_send_frame` then rejects the frame outright).
        frame.data_mut(0).fill(0);
        frame
    }

    #[test]
    fn seek_check_rejects_a_muxer_track() {
        let encoder = open_aac_encoder(48000, 1);
        let path = std::env::temp_dir().join(format!(
            "file_muxer_seek_check_test_{}.mp4",
            std::process::id()
        ));
        let mut muxer = FileMuxer::create(&path).expect("create muxer");
        muxer
            .add_stream(
                "audio",
                encoder.parameters(),
                ffmpeg::Rational::new(1, 48000),
            )
            .expect("add stream");
        let mut sink = muxer.open().expect("open muxer").pop().unwrap();
        let context = Arc::new(SeekCheckContext::new());

        sink.control(ControlMsg::CheckSeek(Arc::clone(&context)))
            .expect("check control");

        let error = context.result().expect_err("muxer must reject seek");
        assert_eq!(error.rejections().len(), 1);
        assert_eq!(
            error.rejections()[0].reason,
            SeekRejectReason::ElementNotSeekable
        );
        sink.control(ControlMsg::Stop).expect("finalize muxer");
        std::fs::remove_file(path).ok();
    }

    /// One track, driven end to end (encode -> mux -> write_trailer on
    /// `Eos`), still produces a real, playable file — the single-track
    /// case `FileMuxer` degenerates to.
    #[test]
    fn single_track_still_produces_a_playable_file() {
        let mut encoder = open_aac_encoder(48000, 1);

        let dir = std::env::temp_dir();
        let path = dir.join(format!("file_muxer_single_test_{}.mp4", std::process::id()));

        let mut muxer = FileMuxer::create(&path).expect("the muxer must open");
        muxer
            .add_stream(
                "audio",
                encoder.parameters(),
                ffmpeg::Rational::new(1, 48000),
            )
            .expect("add_stream must succeed");
        let mut sinks = muxer.open().expect("open must write the header");
        assert_eq!(sinks.len(), 1);
        encoder.src_pads()[0].link(sinks.pop().unwrap());

        for tick in 0..20i64 {
            encoder
                .consume(MediaBuffer::Audio(Arc::new(silent_frame(
                    48000,
                    1,
                    960,
                    tick * 960,
                ))))
                .expect("consume must succeed");
        }
        encoder
            .consume(MediaBuffer::Eos)
            .expect("eos must flush cleanly");
        drop(encoder);

        let input = ffmpeg::format::input(&path).expect("muxed file must be readable back");
        assert_eq!(input.streams().count(), 1);
        std::fs::remove_file(&path).ok();
    }

    /// Regression test against a leaked file handle: dropping every track
    /// `Sink` without ever sending `Eos`/`Stop` (simulating a `Pipeline`
    /// just getting dropped mid-recording, e.g. the process is tearing
    /// down) must still release the underlying file — no stray clone of
    /// the shared `Arc` (or the `ffmpeg::format::context::Output` it
    /// guards) left holding it open. Windows won't let an open file be
    /// deleted, so a successful `remove_file` here is direct proof
    /// nothing lingered; on a build where that isn't already guaranteed
    /// by construction, this would instead hang or fail with a sharing
    /// violation.
    #[test]
    fn dropping_every_sink_without_eos_or_stop_still_releases_the_file() {
        let encoder = open_aac_encoder(48000, 1);

        let dir = std::env::temp_dir();
        let path = dir.join(format!("file_muxer_drop_test_{}.mp4", std::process::id()));

        let mut muxer = FileMuxer::create(&path).expect("the muxer must open");
        muxer
            .add_stream(
                "audio",
                encoder.parameters(),
                ffmpeg::Rational::new(1, 48000),
            )
            .expect("add_stream must succeed");
        let sinks = muxer.open().expect("open must write the header");

        // No `Eos`/`Stop`, no trailer — just drop everything, on purpose.
        drop(sinks);
        drop(encoder);

        std::fs::remove_file(&path)
            .expect("file handle must be released once every sink is dropped");
    }

    /// Two independent tracks (standing in for a real video+audio pair —
    /// `FileMuxer` treats every stream as an opaque `codec::Parameters`, so
    /// two AAC tracks at different sample rates exercise the same
    /// stream-index/trailer-timing machinery a real video+audio pair
    /// would) muxed into one file. Track `a` reaches `Eos` well before
    /// track `b` does — proving the trailer isn't written until *both*
    /// report done, not on whichever finishes first (which would
    /// silently truncate whichever track was still running).
    #[test]
    fn muxes_two_independent_tracks_without_finalizing_early() {
        let mut encoder_a = open_aac_encoder(48000, 2);
        let mut encoder_b = open_aac_encoder(44100, 1);

        let dir = std::env::temp_dir();
        let path = dir.join(format!("file_muxer_multi_test_{}.mp4", std::process::id()));

        let mut muxer = FileMuxer::create(&path).expect("the muxer must open");
        muxer
            .add_stream("a", encoder_a.parameters(), ffmpeg::Rational::new(1, 48000))
            .expect("add_stream a");
        muxer
            .add_stream("b", encoder_b.parameters(), ffmpeg::Rational::new(1, 44100))
            .expect("add_stream b");
        let mut sinks = muxer.open().expect("open must write the header");
        assert_eq!(sinks.len(), 2);
        let sink_b = sinks.pop().unwrap();
        let sink_a = sinks.pop().unwrap();
        encoder_a.src_pads()[0].link(sink_a);
        encoder_b.src_pads()[0].link(sink_b);

        for tick in 0..10i64 {
            encoder_a
                .consume(MediaBuffer::Audio(Arc::new(silent_frame(
                    48000,
                    2,
                    960,
                    tick * 960,
                ))))
                .expect("consume must succeed");
        }
        // Track `a` finishes here — well before track `b` has written
        // anything at all.
        encoder_a
            .consume(MediaBuffer::Eos)
            .expect("eos must flush cleanly");

        for tick in 0..10i64 {
            encoder_b
                .consume(MediaBuffer::Audio(Arc::new(silent_frame(
                    44100,
                    1,
                    882,
                    tick * 882,
                ))))
                .expect("consume must succeed");
        }
        encoder_b
            .consume(MediaBuffer::Eos)
            .expect("eos must flush cleanly");

        drop(encoder_a);
        drop(encoder_b);

        let mut input = ffmpeg::format::input(&path).expect("muxed file must be readable back");
        assert_eq!(input.streams().count(), 2, "expected two tracks");

        let mut counts = [0usize; 2];
        let mut packet = ffmpeg::Packet::empty();
        while packet.read(&mut input).is_ok() {
            counts[packet.stream()] += 1;
            packet = ffmpeg::Packet::empty();
        }
        assert!(counts[0] > 0, "track a has no packets: {counts:?}");
        assert!(
            counts[1] > 0,
            "track b has no packets: {counts:?} — trailer was written before track b finished"
        );
        std::fs::remove_file(&path).ok();
    }

    /// The container comes from the path's own extension, not from this
    /// type's name.
    ///
    /// `format::output` asks FFmpeg to guess a muxer from the filename, so
    /// this writes Matroska for a `.mkv` as readily as MP4 for a `.mp4` — the
    /// name says what it was written for, not what it is limited to. Worth a
    /// test rather than a comment: it is the difference between "we would
    /// have to add a muxer" and "name the file .mkv", and nothing else here
    /// would have caught the day it stopped being true.
    ///
    /// Written with real packets rather than a header and a trailer alone: a
    /// Matroska file with no cluster in it is not something FFmpeg will read
    /// back, so an empty one would fail here for a reason that has nothing to
    /// do with which muxer was picked.
    #[test]
    fn the_container_follows_the_path_extension() {
        let mut encoder = open_aac_encoder(48000, 1);

        let dir = std::env::temp_dir();
        let path = dir.join(format!("muxer_container_test_{}.mkv", std::process::id()));
        let _ = std::fs::remove_file(&path);

        let mut muxer = FileMuxer::create(&path).expect("the muxer must open a .mkv path");
        muxer
            .add_stream(
                "audio",
                encoder.parameters(),
                ffmpeg::Rational::new(1, 48000),
            )
            .expect("add_stream must succeed");
        let mut sinks = muxer.open().expect("open must write the header");
        encoder.src_pads()[0].link(sinks.pop().expect("exactly one stream was added"));

        for tick in 0..20i64 {
            encoder
                .consume(MediaBuffer::Audio(Arc::new(silent_frame(
                    48000,
                    1,
                    960,
                    tick * 960,
                ))))
                .expect("consume must succeed");
        }
        encoder
            .consume(MediaBuffer::Eos)
            .expect("eos must flush cleanly");
        drop(encoder);

        let input = ffmpeg::format::input(&path).expect("the file must be readable");
        let format = input.format();
        assert!(
            format.name().contains("matroska"),
            "a .mkv path should have produced Matroska, got {:?}",
            format.name()
        );
        assert_eq!(input.streams().count(), 1);
        std::fs::remove_file(&path).ok();
    }

    /// A video track's codec extradata has to be in the container's header
    /// for Matroska, which writes `CodecPrivate` up front. MP4 does not need
    /// it there — `avcC` is written in the trailer, and the mov muxer will
    /// take the Annex-B SPS/PPS out of the packets themselves — so a
    /// video encoder opened without `AV_CODEC_FLAG_GLOBAL_HEADER` produces
    /// a working `.mp4` and an `avformat_write_header` that fails with
    /// `INVALIDDATA` for `.mkv`.
    ///
    /// Which is the whole reason this test exists at the muxer rather than at
    /// the encoder: the flag is the encoder's, and nothing but a non-MP4
    /// container ever notices it is missing.
    #[test]
    fn a_video_track_carries_its_extradata_into_a_matroska_header() {
        use crate::elements::{SwEncoder, SwEncoderOptions, VideoCodec};

        let time_base = ffmpeg::Rational::new(1, 30);
        let options = |codec| SwEncoderOptions {
            codec,
            width: 320,
            height: 180,
            time_base,
            frame_rate: ffmpeg::Rational::new(30, 1),
            bit_rate: 400_000,
            gop_size: 30,
            max_b_frames: None,
        };
        // Either software H.264 will do; a build carrying neither is one this
        // cannot be asked about, so it skips the way a hardware test does.
        let Some(encoder) = [VideoCodec::OpenH264, VideoCodec::H264]
            .into_iter()
            .find_map(|codec| SwEncoder::new("video", options(codec)).ok())
        else {
            eprintln!("skipping: this FFmpeg build has no libopenh264 or libx264");
            return;
        };

        let dir = std::env::temp_dir();
        let path = dir.join(format!("file_muxer_mkv_video_{}.mkv", std::process::id()));
        let _ = std::fs::remove_file(&path);

        let mut muxer = FileMuxer::create(&path).expect("the muxer must open");
        muxer
            .add_stream("video", encoder.parameters(), time_base)
            .expect("add_stream must succeed");
        // `open` is the whole assertion: it is `avformat_write_header`, and
        // that is what refuses a video track it has no `CodecPrivate` for.
        // Nothing is read back afterwards because a Matroska file with no
        // cluster in it is not readable at all — proving that would mean
        // encoding frames, which is a different element's contract.
        let sinks = muxer
            .open()
            .expect("Matroska must accept a video track's header");
        drop(sinks);
        std::fs::remove_file(&path).ok();
    }

    /// Opus in Matroska reads back as Opus.
    ///
    /// Not a guard on the global-header flag, which is what it was written to
    /// check: FFmpeg's `libopus` wrapper writes `OpusHead` into `extradata`
    /// whether or not the flag is set, so this passes with the flag reverted.
    /// Unlike the video track above, Opus was never at risk here.
    ///
    /// Kept because the pairing is worth an actual check rather than an
    /// assumption from the AAC one: `libopus` is an external library, the
    /// only audio codec here that a build can be missing, and Matroska is the
    /// container that writes a `CodecPrivate` for it up front.
    ///
    /// Skips where the build has no `libopus`, which is a real configuration
    /// — which is also why the application probes for it rather than
    /// offering it blind.
    #[test]
    fn opus_carries_its_own_header_into_matroska() {
        // 48 kHz because libopus takes nothing else.
        let Some(mut encoder) = open_opus_encoder(48_000, 2) else {
            eprintln!("skipping: this FFmpeg build has no libopus");
            return;
        };

        let dir = std::env::temp_dir();
        let path = dir.join(format!("file_muxer_opus_mkv_{}.mkv", std::process::id()));
        let _ = std::fs::remove_file(&path);

        let mut muxer = FileMuxer::create(&path).expect("the muxer must open");
        muxer
            .add_stream(
                "audio",
                encoder.parameters(),
                ffmpeg::Rational::new(1, 48_000),
            )
            .expect("add_stream must succeed");
        let mut sinks = muxer
            .open()
            .expect("Matroska must accept an Opus track's header");
        encoder.src_pads()[0].link(sinks.pop().expect("exactly one stream was added"));

        for tick in 0..20i64 {
            encoder
                .consume(MediaBuffer::Audio(Arc::new(silent_frame(
                    48_000,
                    2,
                    960,
                    tick * 960,
                ))))
                .expect("consume must succeed");
        }
        encoder
            .consume(MediaBuffer::Eos)
            .expect("eos must flush cleanly");
        drop(encoder);

        let input = ffmpeg::format::input(&path).expect("the file must be readable");
        assert!(
            input.format().name().contains("matroska"),
            "got {:?}",
            input.format().name()
        );
        let stream = input
            .streams()
            .best(ffmpeg::media::Type::Audio)
            .expect("the file must hold the audio track it was given");
        assert_eq!(
            stream.parameters().id(),
            ffmpeg::codec::Id::OPUS,
            "the track must read back as Opus, not as whatever the header defaulted to"
        );
        std::fs::remove_file(&path).ok();
    }

    /// Packets whose `dts` and `pts` differ, remuxed, still differ.
    ///
    /// A B-frame is coded from frames on both sides of it, so a container
    /// holding any carries its packets in decode order and their timestamps
    /// stop being the same number. A muxer that wrote `pts` into both — or
    /// that reordered on the way through — produces a file whose decode order
    /// no longer matches its timestamps, and every player of it either stalls
    /// or shows the frames in the wrong order.
    ///
    /// The ordinary fixture cannot show this: `libopenh264` emits no
    /// B-frames. This one is MPEG-4 Part 2 for that reason alone — see
    /// `test_support::synthesize_reordered`.
    #[test]
    fn a_reordered_stream_keeps_its_decode_order_through_the_muxer() {
        use crate::elements::FileDemuxer;
        use crate::pipeline::Pipeline;

        let fixture = crate::test_support::synthesize_reordered("reordered", 3.0);
        let source = fixture.path.to_string_lossy().into_owned();

        // The fixture has to actually be reordered, or the rest of this
        // asserts nothing. Measured rather than assumed: whether an encoder
        // honours `max_b_frames` is the encoder's business, not this crate's.
        let arriving = timestamps(&source);
        assert!(
            arriving.iter().any(|(dts, pts)| dts != pts),
            "the fixture carries no reordering to preserve: {:?}",
            &arriving[..arriving.len().min(8)]
        );

        let path = std::env::temp_dir().join("media-pp-reordered-remux.mp4");
        let _ = std::fs::remove_file(&path);
        let (demuxer, streams) = FileDemuxer::open("demuxer", &source).expect("open the fixture");
        let video = streams
            .iter()
            .find(|stream| stream.kind == ffmpeg::media::Type::Video)
            .expect("the fixture has video")
            .index;
        let parameters = demuxer.stream_parameters(video).expect("video parameters");
        let time_base = demuxer.stream_time_base(video).expect("video time base");

        let mut muxer = FileMuxer::create(&path).expect("create the remux");
        muxer
            .add_stream("video", parameters, time_base)
            .expect("add the video stream");
        let sink = muxer
            .open()
            .expect("write the header")
            .pop()
            .expect("one stream was added");

        let pipeline = Pipeline::new("remux", demuxer, move |source, context| {
            let branch = context.branch().to(sink)?;
            context.attach(source, video, branch)?;
            Ok(())
        })
        .expect("wire the remux");
        pipeline.run().expect("run the remux");
        for event in pipeline.bus().iter() {
            if matches!(event, crate::bus::BusEvent::Eos { .. }) {
                break;
            }
        }
        pipeline.stop();

        let written = timestamps(&path.to_string_lossy());
        assert_eq!(
            written.len(),
            arriving.len(),
            "every packet that came in has to come out"
        );
        assert!(
            written.iter().any(|(dts, pts)| dts != pts),
            "the reordering did not survive being written"
        );
        assert!(
            written.windows(2).all(|pair| pair[0].0 <= pair[1].0),
            "decode order is what `dts` is for and it has to keep rising: {:?}",
            &written[..written.len().min(8)]
        );
        assert_eq!(
            written, arriving,
            "a remux copies packets; it does not restamp them"
        );
        std::fs::remove_file(&path).ok();
    }

    /// Every video packet's `(dts, pts)`, in the order the container holds
    /// them.
    fn timestamps(path: &str) -> Vec<(i64, i64)> {
        let mut input = ffmpeg::format::input(path).expect("the file opens");
        let video = input
            .streams()
            .find(|stream| stream.parameters().medium() == ffmpeg::media::Type::Video)
            .expect("it has video")
            .index();
        input
            .packets()
            .filter(|(stream, _)| stream.index() == video)
            .filter_map(|(_, packet)| Some((packet.dts()?, packet.pts()?)))
            .collect()
    }

    /// What a written file's timeline looks like from outside it.
    #[derive(Debug, Clone, Copy)]
    struct Shape {
        video_frames: usize,
        video_seconds: f64,
        audio_seconds: f64,
        video_start: f64,
        audio_start: f64,
    }

    impl Shape {
        fn video_fps(self) -> f64 {
            self.video_frames as f64 / self.video_seconds
        }
    }

    /// Reads one back, the way anything that plays it would.
    fn shape_of(path: &str) -> Shape {
        let input = ffmpeg::format::input(path).expect("the file opens");
        let stream = |medium| {
            input
                .streams()
                .find(|stream| stream.parameters().medium() == medium)
                .unwrap_or_else(|| panic!("{path} carries no {medium:?}"))
        };
        let seconds = |stream: &ffmpeg::format::stream::Stream<'_>| {
            stream.duration() as f64 * f64::from(stream.time_base())
        };
        let start = |stream: &ffmpeg::format::stream::Stream<'_>| {
            stream.start_time() as f64 * f64::from(stream.time_base())
        };
        let video = stream(ffmpeg::media::Type::Video);
        let audio = stream(ffmpeg::media::Type::Audio);
        Shape {
            video_frames: video.frames() as usize,
            video_seconds: seconds(&video),
            audio_seconds: seconds(&audio),
            video_start: start(&video),
            audio_start: start(&audio),
        }
    }

    /// A file decoded, re-encoded and written back keeps the shape it had.
    ///
    /// The path a recording really takes, and the one every defect found in
    /// this crate's timeline handling has lived on. What makes it worth a
    /// test of its own is that each of those was invisible from inside: every
    /// element returned `Ok`, every buffer went where it was sent, and the
    /// file that came out was the wrong length, or its sound no longer sat
    /// against its picture. None of that can be seen without reading the
    /// result back.
    ///
    /// The tolerances are deliberately loose. What this is watching for is a
    /// stream that lost or gained *time* — an encoder dropping samples, a
    /// muxer restamping a track, a time base that means something different
    /// at each end — not the frame or two an encoder is entitled to hold.
    #[test]
    fn a_transcoded_file_keeps_the_shape_of_what_went_in() {
        use crate::elements::{FileDemuxer, SwDecoder, SwEncoder, SwEncoderOptions, VideoCodec};
        use crate::pipeline::Pipeline;

        let fixture = crate::test_support::synthesize("transcode-shape", 4.0, 44_100);
        let source_path = fixture.path.to_string_lossy().into_owned();
        let arriving = shape_of(&source_path);

        let path = std::env::temp_dir().join("media-pp-transcode-shape.mp4");
        let _ = std::fs::remove_file(&path);

        let (demuxer, streams) = FileDemuxer::open("demuxer", &source_path).expect("open");
        let index = |medium| {
            streams
                .iter()
                .find(|stream| stream.kind == medium)
                .unwrap_or_else(|| panic!("the fixture carries no {medium:?}"))
                .index
        };
        let video = index(ffmpeg::media::Type::Video);
        let audio = index(ffmpeg::media::Type::Audio);
        let video_decoder = SwDecoder::new(
            "video-decoder",
            demuxer.stream_parameters(video).expect("video parameters"),
        )
        .expect("open the video decoder");
        let audio_decoder = SwDecoder::new(
            "audio-decoder",
            demuxer.stream_parameters(audio).expect("audio parameters"),
        )
        .expect("open the audio decoder");

        // Re-encoded at the same rates it arrived with, so a difference in
        // the result is this crate's doing rather than a conversion's.
        let width = 320;
        let height = 240;
        // The container's own unit, not `1/frame_rate`: what the decoder
        // hands over is stamped in whatever the container counts in, and an
        // encoder told a different unit writes those same numbers meaning
        // something else. Which is not hypothetical — the first version of
        // this test said `1/30` and produced 121 frames across 34 minutes.
        let video_time_base = demuxer.stream_time_base(video).expect("video time base");
        let video_encoder = SwEncoder::new(
            "video-encoder",
            SwEncoderOptions {
                codec: VideoCodec::OpenH264,
                width,
                height,
                time_base: video_time_base,
                frame_rate: ffmpeg::Rational::new(30, 1),
                bit_rate: 800_000,
                gop_size: 30,
                max_b_frames: None,
            },
        )
        .expect("open the video encoder");
        // Written at 48kHz from a 44.1kHz source, which is what a recording
        // really does — a file's own rate is rarely the one everything else
        // in the graph runs at. It also puts a rate conversion inside what
        // this measures, and an audio path losing a fraction of every frame
        // to one is a defect this crate has actually had.
        let audio_encoder = open_aac_encoder(48_000, 2);

        let mut muxer = FileMuxer::create(&path).expect("create the output");
        muxer
            .add_stream("video", video_encoder.parameters(), video_time_base)
            .expect("add the video stream");
        muxer
            .add_stream(
                "audio",
                audio_encoder.parameters(),
                audio_encoder.time_base(),
            )
            .expect("add the audio stream");
        let mut sinks = muxer.open().expect("write the header");
        let audio_sink = sinks.pop().expect("audio was added second");
        let video_sink = sinks.pop().expect("video was added first");

        let scaler = crate::elements::SwScaler::new(
            "to-yuv",
            ffmpeg::format::Pixel::YUV420P,
            width,
            height,
            ffmpeg::software::scaling::Flags::BILINEAR,
        );

        let pipeline = Pipeline::new("transcode", demuxer, move |source, context| {
            let picture = context
                .branch()
                .pipe(video_decoder)
                .pipe(scaler)
                .pipe(video_encoder)
                .to(video_sink)?;
            context.attach(source, video, picture)?;
            let sound = context
                .branch()
                .pipe(audio_decoder)
                .pipe(audio_encoder)
                .to(audio_sink)?;
            context.attach(source, audio, sound)?;
            Ok(())
        })
        .expect("wire the transcode");
        pipeline.run().expect("run the transcode");
        for event in pipeline.bus().iter() {
            if matches!(event, crate::bus::BusEvent::Eos { .. }) {
                break;
            }
        }
        pipeline.stop();

        let written = shape_of(&path.to_string_lossy());
        eprintln!("SHAPE in : {arriving:?} fps={:.3}", arriving.video_fps());
        eprintln!("SHAPE out: {written:?} fps={:.3}", written.video_fps());

        assert!(
            (written.video_fps() - arriving.video_fps()).abs() < 0.5,
            "the picture came out at a different rate: {:.3} in, {:.3} out",
            arriving.video_fps(),
            written.video_fps()
        );
        assert!(
            (written.video_seconds - arriving.video_seconds).abs() < 0.25,
            "the picture came out a different length: {:.3}s in, {:.3}s out",
            arriving.video_seconds,
            written.video_seconds
        );
        assert!(
            (written.audio_seconds - written.video_seconds).abs() < 0.25,
            "the sound and the picture came out different lengths: \
             {:.3}s of sound against {:.3}s of picture",
            written.audio_seconds,
            written.video_seconds
        );
        assert!(
            (written.audio_start - written.video_start).abs()
                <= (arriving.audio_start - arriving.video_start).abs() + 0.05,
            "the two tracks no longer start where they did: \
             in {:.3}s/{:.3}s, out {:.3}s/{:.3}s",
            arriving.video_start,
            arriving.audio_start,
            written.video_start,
            written.audio_start
        );
        std::fs::remove_file(&path).ok();
    }
}
