use std::{
    path::Path,
    sync::{Arc, Mutex},
};

use ffmpeg_next as ffmpeg;
use rust_hlog::{HLog, herror};
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Element, ElementType, Sink, element_hlog},
    error::Result,
};

/// Errors specific to `Mp4Muxer`. Converts into the crate-wide `Error` via
/// `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum Mp4MuxerError {
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),
}

/// One track registered via [`Mp4Muxer::add_stream`], waiting for
/// [`Mp4Muxer::open`] to turn it into a real [`Mp4MuxerStreamSink`] — its
/// `name` becomes that sink's own [`Element::name`]/`hlog` identity, and
/// `input_time_base` is what every `Packet` it receives already carries
/// `pts`/`dts` in (the same one its upstream encoder was opened with).
struct PendingStream {
    name: Arc<str>,
    input_time_base: ffmpeg::Rational,
}

/// Builds an MP4 (or any other container ffmpeg infers from `path`'s
/// extension) with one or more tracks, then opens it into one [`Sink`] per
/// track. Two-phase on purpose: a container's header has to describe
/// every stream's codec parameters up front — `avformat_write_header`
/// can't run until every [`Mp4Muxer::add_stream`] this file will ever hold
/// has already happened — so there's no way to make this a single
/// long-lived `Sink` that tracks attach to one at a time as their encoders
/// come online (contrast [`crate::elements::AudioMixer`], whose inputs
/// *can* attach at any time — it has no "known shape before the first
/// byte" constraint the way a container header does).
///
/// ```ignore
/// let mut muxer = Mp4Muxer::create("out.mp4")?;
/// muxer.add_stream("video", video_encoder.parameters(), video_time_base)?;
/// muxer.add_stream("audio", audio_encoder.parameters(), audio_time_base)?;
/// let mut sinks = muxer.open()?; // writes the header
/// let audio_sink = sinks.pop().unwrap();
/// let video_sink = sinks.pop().unwrap();
/// ```
pub struct Mp4Muxer {
    output: ffmpeg::format::context::Output,
    streams: Vec<PendingStream>,
}

impl Mp4Muxer {
    /// Allocates the output file. No header is written yet — nothing is on
    /// disk in a readable shape until [`Mp4Muxer::open`] runs.
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        let output = ffmpeg::format::output(&path).map_err(Mp4MuxerError::from)?;
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
    /// this track's own [`Element::name`]/`hlog` identity once
    /// [`Mp4Muxer::open`] turns it into a `Sink` — pick something that
    /// tells multiple tracks apart in logs/[`crate::bus::BusEvent`]s,
    /// e.g. `"video"`/`"audio"`.
    ///
    /// Add streams in the same order the caller will treat
    /// [`Mp4Muxer::open`]'s returned `Vec` — index 0 is whichever stream
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
            .map_err(Mp4MuxerError::from)?;
        stream.set_time_base(time_base);
        stream.set_parameters(parameters);
        self.streams.push(PendingStream {
            name: name.into().into(),
            input_time_base: time_base,
        });
        Ok(())
    }

    /// Writes the container header — every [`Mp4Muxer::add_stream`] call
    /// this file will ever get must already have happened — and returns
    /// one [`Sink`] per track, in the order [`Mp4Muxer::add_stream`] added
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
    /// single-track file (e.g. `screen_record`/`audio_record`) degenerates
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
        self.output.write_header().map_err(Mp4MuxerError::from)?;
        let total = self.streams.len();
        let shared = Arc::new(Mp4MuxerShared {
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
                Box::new(Mp4MuxerStreamSink {
                    hlog: element_hlog(ElementType::Mp4Muxer, &stream.name, None),
                    name: stream.name,
                    shared: shared.clone(),
                    stream_index: index,
                    input_time_base: stream.input_time_base,
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
    /// [`Mp4MuxerShared::total`], not on the first one (see
    /// [`Mp4Muxer::open`]'s own docs for why).
    done: usize,
    /// Set once the trailer has been written. Each
    /// [`Mp4MuxerStreamSink`]'s own `done` flag already prevents
    /// double-counting *that* track's contribution to `done`; this
    /// additionally guards [`Mp4MuxerShared::write_packet`] against
    /// writing into a file whose trailer has already closed it.
    finished: bool,
}

/// Shared between every [`Mp4MuxerStreamSink`] [`Mp4Muxer::open`] hands
/// out for the same file — one lock around the whole
/// [`ffmpeg::format::context::Output`] so concurrent tracks never
/// interleave two writes against it (see [`Mp4Muxer::open`]'s own docs).
struct Mp4MuxerShared {
    state: Mutex<MuxerState>,
    total: usize,
}

impl Mp4MuxerShared {
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
            .expect("stream was added in Mp4Muxer::add_stream")
            .time_base();
        packet.rescale_ts(input_time_base, output_time_base);
        packet.set_stream(stream_index);
        packet.set_position(-1);
        packet
            .write_interleaved(&mut state.output)
            .map_err(Mp4MuxerError::from)?;
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
        state.output.write_trailer().map_err(Mp4MuxerError::from)?;
        Ok(())
    }
}

/// One track's own [`Sink`] — a lightweight handle sharing a
/// [`Mp4MuxerShared`] with every other track [`Mp4Muxer::open`] returned
/// alongside it. See [`Mp4Muxer::open`]'s own docs for the
/// finalize-once-every-track-is-done contract this relies on.
#[rust_hlog::hlog]
pub struct Mp4MuxerStreamSink {
    name: Arc<str>,
    shared: Arc<Mp4MuxerShared>,
    stream_index: usize,
    input_time_base: ffmpeg::Rational,
    /// Set once this sink has contributed to
    /// [`Mp4MuxerShared::finish_track`] — guards against double-counting
    /// if both a natural `Eos` and a later `Stop` arrive for the same
    /// track.
    done: bool,
}

impl Mp4MuxerStreamSink {
    fn finish(&mut self) -> Result<()> {
        if self.done {
            return Ok(());
        }
        self.done = true;
        self.shared
            .finish_track()
            .inspect_err(|error| herror!(self, "write_trailer failed: {error}"))
    }
}

impl Element for Mp4MuxerStreamSink {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::Mp4Muxer
    }

    fn hlog(&self) -> &HLog {
        &self.hlog
    }

    fn hlog_mut(&mut self) -> &mut HLog {
        &mut self.hlog
    }
}

impl Sink for Mp4MuxerStreamSink {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Packet(packet) => self
                .shared
                .write_packet(self.stream_index, self.input_time_base, &packet)
                .inspect_err(|error| herror!(self, "write_interleaved failed: {error}")),
            MediaBuffer::Eos => self.finish(),
            other => {
                let _ = other;
                Ok(())
            }
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        // Terminal, nothing to forward. `Stop` still contributes to this
        // track's own "done" count — see `Mp4Muxer::open`'s own docs on
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

    /// One track, driven end to end (encode -> mux -> write_trailer on
    /// `Eos`), still produces a real, playable file — the single-track
    /// case `Mp4Muxer` degenerates to.
    #[test]
    fn single_track_still_produces_a_playable_file() {
        let mut encoder = open_aac_encoder(48000, 1);

        let dir = std::env::temp_dir();
        let path = dir.join(format!("mp4_muxer_single_test_{}.mp4", std::process::id()));

        let mut muxer = Mp4Muxer::create(&path).expect("mp4 muxer must open");
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
        let path = dir.join(format!("mp4_muxer_drop_test_{}.mp4", std::process::id()));

        let mut muxer = Mp4Muxer::create(&path).expect("mp4 muxer must open");
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
    /// `Mp4Muxer` treats every stream as an opaque `codec::Parameters`, so
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
        let path = dir.join(format!("mp4_muxer_multi_test_{}.mp4", std::process::id()));

        let mut muxer = Mp4Muxer::create(&path).expect("mp4 muxer must open");
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
}
