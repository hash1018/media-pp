//! The half every muxer here shares: one FFmpeg output written through
//! several tracks' sinks, each on a thread of its own, and finalized once
//! all of them are done.
//!
//! `FileMuxer`, `HlsMuxer`, `RtmpMuxer` and `RtspMuxer` differ in how their
//! output is created and how its header is written. From the header on they
//! are the same: a track's packet is rescaled into the time base the muxer
//! settled on, stamped with the track's stream index, and interleaved into
//! the output under one lock, and the trailer is written by whichever track
//! finishes last. That part lives here once, as [`TrackSink`], so the rule
//! that one track finishing early must not truncate the others is written
//! and fixed in one place.
//!
//! What still differs is a [`Muxer`]'s to say — which errors it reports and
//! which `ElementType` its tracks carry — and a [`TrackOptions`]'s: whether
//! a track follows an upstream seek, and whether `Stop` finishes it.

use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use ffmpeg_next as ffmpeg;

use super::tracks::{MuxerId, MuxerSinks};
use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, PortContract},
    control::{ControlMsg, SeekRejectReason},
    element::{Element, ElementType, Sink, element_pp_log},
    error::{Error, Result},
    pp_log::{PpLog, pp_error, pp_info},
};

/// One track registered with a muxer's `add_stream`, waiting for its `open`.
///
/// `name` becomes the track's sink's own [`Element::name`]/`pp_log`
/// identity; `input_time_base` is what every packet it receives already
/// carries `pts`/`dts` in — the one its upstream encoder was opened with.
pub(super) struct PendingStream {
    pub(super) name: Arc<str>,
    pub(super) input_time_base: ffmpeg::Rational,
    /// Taken from the parameters the track was registered with, so its sink
    /// can refuse the other medium's packets at wiring time.
    pub(super) kind: Option<MediaKind>,
}

impl PendingStream {
    pub(super) fn new(
        name: impl Into<String>,
        parameters: &ffmpeg::codec::Parameters,
        input_time_base: ffmpeg::Rational,
    ) -> Self {
        Self {
            name: name.into().into(),
            input_time_base,
            kind: MediaKind::packet_for(parameters.medium()),
        }
    }
}

/// What a muxer's tracks report themselves as, and how their failures reach
/// the caller: as that muxer's own error type, as they always have.
pub(super) trait Muxer: 'static {
    const ELEMENT_TYPE: ElementType;

    fn ffmpeg_error(error: ffmpeg::Error) -> Error;

    /// A track handed something that is not a packet or `Eos`.
    fn unsupported_buffer(kind: &'static str) -> Error;
}

/// A track's own timestamps, where it follows an upstream seek rather than
/// refusing one — see `RtspMuxer`, the one muxer whose tracks do.
pub(super) trait TrackTimeline: Send {
    /// Rewrites `packet`'s timestamps, already in the output's time base,
    /// into the timeline this track has been publishing.
    fn stamp(&mut self, packet: &mut ffmpeg::Packet);

    /// The source moved; the next packet starts from wherever it landed.
    fn seeked(&mut self);
}

/// How a muxer's tracks behave where the muxers differ.
pub(super) struct TrackOptions {
    /// Whether `Stop` finishes a track as `Eos` does. It does for a file,
    /// where a stopped recording should still be a playable one; a live
    /// publish finishes on `Eos` alone — see `RtspMuxer`.
    pub(super) finish_on_stop: bool,
    /// Where the output goes, already stripped of anything secret, for a
    /// muxer that publishes: logged as each track opens and when the
    /// publish closes. `None` for a file, which says nothing of the sort.
    pub(super) destination: Option<Arc<str>>,
    /// Makes each track's own timeline, for a muxer whose tracks follow a
    /// seek. `None` for one whose tracks refuse it.
    pub(super) timeline: Option<fn() -> Box<dyn TrackTimeline>>,
}

/// Turns an output whose header has been written into one sink per track.
///
/// The time base each track is rescaled into is read here, after the
/// header, because writing it is where a muxer settles one — RTSP announces
/// its own in the SDP rather than keeping the one a track was registered
/// with.
pub(super) fn open_tracks<M: Muxer>(
    id: MuxerId,
    output: ffmpeg::format::context::Output,
    streams: Vec<PendingStream>,
    options: TrackOptions,
) -> MuxerSinks {
    let output_time_bases: Vec<_> = (0..streams.len())
        .map(|index| {
            output
                .stream(index)
                .expect("a stream was added for every PendingStream")
                .time_base()
        })
        .collect();
    let total = streams.len();
    let shared = Arc::new(SharedOutput::<M> {
        state: Mutex::new(OutputState {
            output,
            done: 0,
            finished: false,
        }),
        total,
        destination: options.destination,
        muxer: PhantomData,
    });
    id.sinks(
        streams
            .into_iter()
            .zip(output_time_bases)
            .enumerate()
            .map(|(index, (stream, output_time_base))| -> Box<dyn Sink> {
                let pp_log = element_pp_log(M::ELEMENT_TYPE, &stream.name, None);
                if let Some(destination) = &shared.destination {
                    pp_info!(pp_log: &pp_log, "publishing: url={destination}, tracks={total}");
                }
                Box::new(TrackSink {
                    pp_log,
                    name: stream.name,
                    shared: Arc::clone(&shared),
                    stream_index: index,
                    input_time_base: stream.input_time_base,
                    output_time_base,
                    kind: stream.kind,
                    finish_on_stop: options.finish_on_stop,
                    timeline: options.timeline.map(|make| make()),
                    done: false,
                })
            })
            .collect(),
    )
}

struct OutputState {
    output: ffmpeg::format::context::Output,
    /// How many tracks have reported themselves finished — the trailer is
    /// written once this reaches [`SharedOutput::total`], not on the first
    /// one, which would cut off whatever the others still had to write.
    done: usize,
    /// Set once the trailer has been written, so nothing is written into an
    /// output it has already closed.
    finished: bool,
}

/// Shared by every sink one `open` handed out: one lock around the whole
/// output, because packets from independently threaded branches arrive
/// concurrently and neither `av_interleaved_write_frame` nor
/// `av_write_trailer` may be called on one output from two threads at once.
struct SharedOutput<M> {
    state: Mutex<OutputState>,
    total: usize,
    destination: Option<Arc<str>>,
    muxer: PhantomData<fn() -> M>,
}

impl<M: Muxer> SharedOutput<M> {
    /// Writes a packet whose stream index and timestamps are settled.
    fn write(&self, packet: &mut ffmpeg::Packet) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        if state.finished {
            return Ok(());
        }
        packet
            .write_interleaved(&mut state.output)
            .map_err(M::ffmpeg_error)
    }

    /// One track reporting itself finished. Writes the trailer exactly once,
    /// when the last one does, and says whether this call was that one.
    fn finish_track(&self) -> Result<bool> {
        let mut state = self.state.lock().unwrap();
        state.done += 1;
        if state.finished || state.done < self.total {
            return Ok(false);
        }
        state.finished = true;
        state.output.write_trailer().map_err(M::ffmpeg_error)?;
        Ok(true)
    }
}

/// One track's own sink: a light handle on the output it shares with every
/// other track its muxer's `open` returned alongside it.
struct TrackSink<M> {
    pp_log: PpLog,
    name: Arc<str>,
    shared: Arc<SharedOutput<M>>,
    stream_index: usize,
    input_time_base: ffmpeg::Rational,
    /// What the muxer settled on for this track once its header was
    /// written, which is not necessarily what it was registered with.
    output_time_base: ffmpeg::Rational,
    /// The medium this track was registered for; `None` for one this crate
    /// does not model, which then declares nothing.
    kind: Option<MediaKind>,
    finish_on_stop: bool,
    /// `Some` for a track that follows a seek — see [`TrackTimeline`].
    timeline: Option<Box<dyn TrackTimeline>>,
    /// Set once this track has counted itself finished, so an `Eos` and a
    /// later `Stop` for the same track cannot count it twice.
    done: bool,
}

impl<M: Muxer> TrackSink<M> {
    fn finish(&mut self) -> Result<()> {
        if self.done {
            return Ok(());
        }
        self.done = true;
        let closed = self
            .shared
            .finish_track()
            .inspect_err(|error| pp_error!(self, "write_trailer failed: {error}"))?;
        if closed && let Some(destination) = &self.shared.destination {
            pp_info!(self, "publish closed: url={destination}");
        }
        Ok(())
    }
}

impl<M: Muxer> Element for TrackSink<M> {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        M::ELEMENT_TYPE
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl<M: Muxer> Sink for TrackSink<M> {
    /// A muxer interleaves already-encoded data and has no encoder of its
    /// own, so a decoded frame has no route through it. The medium is this
    /// track's own, so a video encoder wired into the audio track is refused
    /// rather than writing something no player can make sense of.
    fn input_contract(&self) -> InputContract {
        match self.kind {
            Some(kind) => InputContract::Fixed(PortContract::packet(kind)),
            None => InputContract::Unknown,
        }
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Packet(packet) => {
                // Cloned, not changed in place: the `Arc<Packet>` may be
                // shared with another branch off the same `Tee`, which must
                // not see this track's timestamps or stream index.
                let mut packet = (*packet).clone();
                packet.rescale_ts(self.input_time_base, self.output_time_base);
                if let Some(timeline) = &mut self.timeline {
                    timeline.stamp(&mut packet);
                }
                packet.set_stream(self.stream_index);
                packet.set_position(-1);
                self.shared
                    .write(&mut packet)
                    .inspect_err(|error| pp_error!(self, "write_interleaved failed: {error}"))
            }
            MediaBuffer::Eos => self.finish(),
            other => {
                pp_error!(self, "unsupported buffer: {}", other.kind());
                Err(M::unsupported_buffer(other.kind()))
            }
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        // Terminal, so nothing is forwarded.
        match msg {
            ControlMsg::CheckSeek(context) if self.timeline.is_none() => {
                context.reject(
                    self.element_type(),
                    self.name(),
                    SeekRejectReason::ElementNotSeekable,
                );
            }
            ControlMsg::Seek(_) => {
                if let Some(timeline) = &mut self.timeline {
                    timeline.seeked();
                }
            }
            ControlMsg::Stop if self.finish_on_stop => self.finish()?,
            _ => {}
        }
        Ok(())
    }
}

impl<M> Drop for TrackSink<M> {
    fn drop(&mut self) {
        if let Some(destination) = &self.shared.destination {
            pp_info!(
                pp_log: &self.pp_log,
                "dropped: releasing this track of the publish to {destination}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::SeekCheckContext;
    use crate::elements::{AudioCodec, FileMuxerError, SwAudioEncoder, SwAudioEncoderOptions};

    /// A file muxer in all but name, so the options can be varied freely.
    struct TestMuxer;

    impl Muxer for TestMuxer {
        const ELEMENT_TYPE: ElementType = ElementType::FileMuxer;

        fn ffmpeg_error(error: ffmpeg::Error) -> Error {
            FileMuxerError::Ffmpeg(error).into()
        }

        fn unsupported_buffer(kind: &'static str) -> Error {
            FileMuxerError::UnsupportedBuffer(kind).into()
        }
    }

    struct NoTimeline;

    impl TrackTimeline for NoTimeline {
        fn stamp(&mut self, _packet: &mut ffmpeg::Packet) {}
        fn seeked(&mut self) {}
    }

    fn path(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "track_sink_{label}_{}_{:?}.mp4",
            std::process::id(),
            std::thread::current().id()
        ))
    }

    /// One AAC track's sink over an MP4 whose header is already written.
    fn one_track(path: &std::path::Path, options: TrackOptions) -> Box<dyn Sink> {
        let time_base = ffmpeg::Rational::new(1, 48_000);
        let encoder = SwAudioEncoder::new(
            "encoder",
            SwAudioEncoderOptions {
                codec: AudioCodec::Aac,
                sample_rate: 48_000,
                channels: 1,
                time_base,
                bit_rate: 64_000,
            },
        )
        .expect("AAC is built into FFmpeg");
        let parameters = encoder.parameters();
        let mut output = ffmpeg::format::output(&path).expect("the output must open");
        let pending = PendingStream::new("audio", &parameters, time_base);
        let mut stream = output.add_stream(parameters.id()).expect("add the stream");
        stream.set_time_base(time_base);
        stream.set_parameters(parameters);
        output.write_header().expect("write the header");
        let id = MuxerId::next();
        let track = id.track(0);
        open_tracks::<TestMuxer>(id, output, vec![pending], options)
            .take(track)
            .expect("the muxer's own track")
    }

    fn seek_refused(sink: &mut Box<dyn Sink>) -> bool {
        let context = Arc::new(SeekCheckContext::new());
        sink.control(ControlMsg::CheckSeek(Arc::clone(&context)))
            .expect("a seek check is answered, not failed");
        context.result().is_err()
    }

    /// Whether the file has its trailer: an MP4 is unreadable until then.
    fn finalized(path: &std::path::Path) -> bool {
        ffmpeg::format::input(&path).is_ok()
    }

    /// A file's track: a seek is refused, and `Stop` finishes it, so a
    /// stopped recording is still a playable one.
    #[test]
    fn a_track_without_a_timeline_refuses_a_seek_and_finishes_on_stop() {
        let path = path("file");
        let mut sink = one_track(
            &path,
            TrackOptions {
                finish_on_stop: true,
                destination: None,
                timeline: None,
            },
        );
        assert!(seek_refused(&mut sink));
        sink.control(ControlMsg::Stop).expect("stop");
        assert!(finalized(&path), "Stop must write the trailer");
        drop(sink);
        std::fs::remove_file(&path).ok();
    }

    /// A publish's track, as `RtspMuxer` opens them: it follows a seek
    /// rather than refusing it, and only `Eos` finishes it.
    #[test]
    fn a_track_with_a_timeline_follows_a_seek_and_finishes_on_eos_alone() {
        let path = path("publish");
        let mut sink = one_track(
            &path,
            TrackOptions {
                finish_on_stop: false,
                destination: Some(Arc::from("rtsp://example/test")),
                timeline: Some(|| Box::new(NoTimeline)),
            },
        );
        assert!(!seek_refused(&mut sink));
        sink.control(ControlMsg::Stop).expect("stop");
        assert!(!finalized(&path), "Stop must not write the trailer here");
        sink.consume(MediaBuffer::Eos).expect("eos");
        assert!(finalized(&path), "Eos must write the trailer");
        drop(sink);
        std::fs::remove_file(&path).ok();
    }
}
