use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use crate::pp_log::{PpLog, pp_error, pp_info};
use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, PortContract},
    control::{ControlMsg, SeekRejectReason},
    element::{Element, ElementType, Sink, element_pp_log},
    error::Result,
};

/// How long one network operation may block before FFmpeg abandons it.
///
/// A publish has no equivalent of a file write that always returns: a server
/// that stops reading leaves the socket writable-then-not, and an unbounded
/// write parks the thread inside [`Sink::consume`] — which the
/// [`Queue`](crate::queue::Queue) in front of it cannot reclaim (see the
/// module docs on [`crate::elements`]'s sink module).
///
/// Ten seconds rather than something tighter: a live stream survives brief
/// congestion all the time, and a timeout short enough to catch a stall
/// quickly is also short enough to end a working broadcast over a hiccup.
/// It bounds the connect and each write as far as FFmpeg's RTMP protocol
/// honours `rw_timeout`, which is not a promise that every internal wait is
/// covered — a `Queue` upstream is still the recovery boundary.
const IO_TIMEOUT: Duration = Duration::from_secs(10);

/// URL schemes FFmpeg's RTMP protocol answers to.
const RTMP_SCHEMES: [&str; 6] = ["rtmp", "rtmps", "rtmpt", "rtmpe", "rtmpte", "rtmpts"];

/// Errors specific to `RtmpMuxer`. Converts into the crate-wide `Error` via
/// `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum RtmpMuxerError {
    /// A stream sink received a buffer other than a packet or end-of-stream.
    #[error("RtmpMuxer stream sinks only accept Packet or Eos buffers, got {0}")]
    UnsupportedBuffer(&'static str),

    /// FFmpeg rejected the connection, the FLV header, packet writing, or
    /// finalization. A codec FLV cannot carry arrives here, from
    /// [`RtmpMuxer::open`].
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),

    /// The URL contains an interior NUL byte rejected by FFmpeg's C API.
    ///
    /// Checked here rather than left to the conversion, which panics on one.
    #[error("RTMP URL contains a NUL byte")]
    InvalidUrl,

    /// The URL is not one FFmpeg's RTMP protocol handles. Reported without
    /// the URL itself, which may carry a stream key.
    #[error(
        "not an RTMP URL: expected one of {schemes} — \
         use FileMuxer with a .flv path to write FLV to a file",
        schemes = RTMP_SCHEMES.join(", ")
    )]
    NotAnRtmpUrl,
}

/// One track registered via [`RtmpMuxer::add_stream`], waiting for
/// [`RtmpMuxer::open`] to turn it into a real [`RtmpMuxerStreamSink`] — its
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

/// Publishes one FLV stream to an RTMP server — Twitch, YouTube, or a local
/// [MediaMTX] — and hands out one [`Sink`] per track.
///
/// This is the publishing half only. It does not run a server and does not
/// depend on a particular one: the address and the stream key come from
/// whoever is receiving. For local work, MediaMTX accepts RTMP publishing on
/// port 1935 out of the box.
///
/// # Muxer, not sink, and why it is shaped like [`crate::elements::FileMuxer`]
///
/// A broadcast is video *and* audio in one FLV container, so the header has
/// to describe both before the first packet goes out — the same two-phase
/// constraint `FileMuxer` has, and the reason this is a builder that returns
/// sinks rather than a `Sink` itself. Every muxer in this crate is shaped
/// that way, including [`RtspMuxer`](crate::elements::RtspMuxer).
///
/// It is a remuxer: incoming buffers are compressed
/// [`MediaBuffer::Packet`] values, and nothing here encodes. FLV carries a
/// limited set of codecs — H.264 and AAC are the pair every service accepts
/// — and one FFmpeg does not accept is refused by [`RtmpMuxer::open`] as an
/// [`RtmpMuxerError::Ffmpeg`], since which codecs a given FFmpeg build will
/// put in FLV is that build's answer to give, not this crate's. Place a
/// [`Pacer`](crate::elements::Pacer) upstream when publishing packets read
/// from a file, or the file goes out faster than real time.
///
/// # The connection opens in `create`, and the stream key never reaches a log
///
/// [`RtmpMuxer::create`] performs the RTMP handshake, so an unreachable
/// server or a rejected key fails there rather than at the first packet.
/// Both calls block for network I/O; neither belongs on a UI thread.
///
/// A publish URL ends in a stream key, which is a credential. Nothing in
/// this type logs the URL it was given: [`RtmpMuxer::redacted_url`] is what
/// goes to the log and what a caller should display, and it replaces the
/// last path segment and any query string.
///
/// # What it does not do
///
/// It does not reconnect. A connection lost mid-broadcast surfaces as a
/// write error on the affected track and the publish is over — recovering
/// means building a new `RtmpMuxer`, which is a new FLV header and so needs
/// a fresh keyframe from upstream. That decision belongs to the application
/// driving the encoders, not to a sink that only sees packets.
///
/// # One warning on a clean shutdown is expected
///
/// Ending a publish prints `Failed to update header with correct filesize`
/// from FFmpeg's FLV muxer. It is not a failure: the trailer tries to seek
/// back and patch the header's size and duration, which a socket cannot do
/// and a file can. Nothing downstream needs those fields for a live stream,
/// and the server has already received everything.
///
/// [MediaMTX]: https://github.com/bluenviron/mediamtx
///
/// ```no_run
/// # use media_pp::ffmpeg;
/// # use media_pp::elements::{
/// #     AudioCodec, RtmpMuxer, SwAudioEncoder, SwAudioEncoderOptions, SwEncoder,
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
/// let mut muxer = RtmpMuxer::create("rtmp://127.0.0.1:1935/live/stream")?;
/// muxer.add_stream("video", video_encoder.parameters(), video_time_base)?;
/// muxer.add_stream("audio", audio_encoder.parameters(), audio_time_base)?;
/// let mut sinks = muxer.open()?; // writes the FLV header
/// let audio_sink = sinks.pop().unwrap();
/// let video_sink = sinks.pop().unwrap();
/// # Ok(())
/// # }
/// ```
pub struct RtmpMuxer {
    output: ffmpeg::format::context::Output,
    streams: Vec<PendingStream>,
    redacted_url: Arc<str>,
}

impl RtmpMuxer {
    /// Validates `url` and performs the RTMP handshake. No FLV header is
    /// written yet — the server has a connection and nothing to play until
    /// [`RtmpMuxer::open`] runs.
    ///
    /// Blocks for the handshake, and for no more than ten seconds — the
    /// timeout this puts on every network operation the publish makes,
    /// since an unbounded one would park a pipeline thread.
    pub fn create(url: impl AsRef<str>) -> Result<Self> {
        let url = url.as_ref();
        if url.contains('\0') {
            return Err(RtmpMuxerError::InvalidUrl.into());
        }
        if !is_rtmp_url(url) {
            return Err(RtmpMuxerError::NotAnRtmpUrl.into());
        }

        // `rw_timeout` is the generic AVIO option, in microseconds, applied
        // to the protocol this opens. Handed to `avio_open2` rather than to
        // the header write, which is where the socket is actually created.
        let mut options = ffmpeg::Dictionary::new();
        let timeout = IO_TIMEOUT.as_micros().to_string();
        options.set("rw_timeout", &timeout);

        // The format is named rather than guessed: a publish URL has no
        // extension for `format::output` to infer FLV from.
        let output =
            ffmpeg::format::output_as_with(url, "flv", options).map_err(RtmpMuxerError::from)?;

        Ok(Self {
            output,
            streams: Vec::new(),
            redacted_url: redact(url).into(),
        })
    }

    /// Registers one more track this broadcast will carry — in practice one
    /// video and one audio, which is what FLV holds. `parameters`/
    /// `time_base` describe it, typically
    /// [`crate::elements::SwEncoder::parameters`] and the same `time_base`
    /// its `SwEncoderOptions` was given. `name` becomes this track's own
    /// [`Element::name`]/`pp_log` identity once [`RtmpMuxer::open`] turns it
    /// into a `Sink`.
    ///
    /// Add streams in the same order the caller will read
    /// [`RtmpMuxer::open`]'s returned `Vec` — index 0 is whichever was
    /// added first.
    pub fn add_stream(
        &mut self,
        name: impl Into<String>,
        parameters: ffmpeg::codec::Parameters,
        time_base: ffmpeg::Rational,
    ) -> Result<()> {
        let mut stream = self
            .output
            .add_stream(parameters.id())
            .map_err(RtmpMuxerError::from)?;
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

    /// Writes the FLV header — every [`RtmpMuxer::add_stream`] call this
    /// broadcast will get must already have happened — and returns one
    /// [`Sink`] per track, in the order they were added.
    ///
    /// This is where a codec FLV cannot carry is refused, and where a
    /// server that accepted the connection but rejects the stream says so.
    ///
    /// All returned `Sink`s write through the same connection behind a
    /// shared lock: a video encode chain and an audio encode chain sit on
    /// their own [`Queue`](crate::queue::Queue)s and arrive concurrently,
    /// and neither `av_interleaved_write_frame` nor `av_write_trailer` is
    /// safe to call from two threads against one output at once. They also
    /// share one trailer, written once every track has reported itself done
    /// — via `Eos` *or* [`ControlMsg::Stop`], either meaning "this track is
    /// finished" rather than "abandon the broadcast" — so ending only the
    /// video pipeline while audio keeps running leaves the publish open
    /// until audio catches up too.
    pub fn open(mut self) -> Result<Vec<Box<dyn Sink>>> {
        self.output.write_header().map_err(RtmpMuxerError::from)?;
        let total = self.streams.len();
        let redacted_url = self.redacted_url;
        let shared = Arc::new(RtmpMuxerShared {
            state: Mutex::new(MuxerState {
                output: self.output,
                done: 0,
                finished: false,
            }),
            total,
            redacted_url: redacted_url.clone(),
        });
        Ok(self
            .streams
            .into_iter()
            .enumerate()
            .map(|(index, stream)| -> Box<dyn Sink> {
                let pp_log = element_pp_log(ElementType::RtmpMuxer, &stream.name, None);
                pp_info!(pp_log: &pp_log, "publishing: url={redacted_url}, tracks={total}");
                Box::new(RtmpMuxerStreamSink {
                    pp_log,
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

    /// The publish address with its stream key removed — what to log, and
    /// what to show a user. See this type's own docs on why the URL itself
    /// is never handed back.
    pub fn redacted_url(&self) -> &str {
        &self.redacted_url
    }
}

/// Whether FFmpeg's RTMP protocol is the one that would open this URL.
///
/// The scheme decides it, case-insensitively, as FFmpeg's own protocol
/// lookup does.
fn is_rtmp_url(url: &str) -> bool {
    let Some((scheme, _)) = url.split_once("://") else {
        return false;
    };
    RTMP_SCHEMES
        .iter()
        .any(|candidate| scheme.eq_ignore_ascii_case(candidate))
}

/// Removes the credential from a publish URL, leaving enough to recognize
/// where a stream was going.
///
/// The last path segment is the stream key at every service this has been
/// pointed at — `rtmp://live.twitch.tv/app/<key>`,
/// `rtmp://a.rtmp.youtube.com/live2/<key>` — and a query string may carry a
/// token, so both go. A URL with a single path segment loses that segment
/// too: over-redacting a log line costs nothing, and guessing which
/// services put the key first would eventually be wrong about one.
fn redact(url: &str) -> String {
    let Some((scheme, rest)) = url.split_once("://") else {
        // Never reached from `create`, which rejects this first. A caller
        // that has not validated still gets nothing quotable.
        return "<url>".to_string();
    };
    let (rest, query) = match rest.split_once('?') {
        Some((before, _)) => (before, "?<query>"),
        None => (rest, ""),
    };
    match rest.rsplit_once('/') {
        Some((prefix, last)) if !last.is_empty() => format!("{scheme}://{prefix}/<key>{query}"),
        _ => format!("{scheme}://{rest}{query}"),
    }
}

struct MuxerState {
    output: ffmpeg::format::context::Output,
    /// How many tracks have reported themselves finished (`Eos` or `Stop`)
    /// — the trailer is written once this reaches
    /// [`RtmpMuxerShared::total`], not on the first one (see
    /// [`RtmpMuxer::open`]'s own docs for why).
    done: usize,
    /// Set once the trailer has been written. Each
    /// [`RtmpMuxerStreamSink`]'s own `done` flag already prevents
    /// double-counting *that* track; this additionally guards
    /// [`RtmpMuxerShared::write_packet`] against writing into a connection
    /// already closed by its trailer.
    finished: bool,
}

/// Shared between every [`RtmpMuxerStreamSink`] [`RtmpMuxer::open`] hands
/// out for the same broadcast — one lock around the whole
/// [`ffmpeg::format::context::Output`] so concurrent tracks never interleave
/// two writes against it (see [`RtmpMuxer::open`]'s own docs).
struct RtmpMuxerShared {
    state: Mutex<MuxerState>,
    total: usize,
    redacted_url: Arc<str>,
}

impl RtmpMuxerShared {
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
        // another branch off the same `Tee`, which must not see this
        // stream's `set_stream`/rescaled timestamps.
        let mut packet = packet.clone();
        let output_time_base = state
            .output
            .stream(stream_index)
            .expect("stream was added in RtmpMuxer::add_stream")
            .time_base();
        packet.rescale_ts(input_time_base, output_time_base);
        packet.set_stream(stream_index);
        packet.set_position(-1);
        packet
            .write_interleaved(&mut state.output)
            .map_err(RtmpMuxerError::from)?;
        Ok(())
    }

    /// One track reporting itself done (`Eos` or `Stop`) — writes the
    /// trailer exactly once, only after every track has called this.
    /// `true` when this call was the one that ended the broadcast.
    fn finish_track(&self) -> Result<bool> {
        let mut state = self.state.lock().unwrap();
        state.done += 1;
        if state.finished || state.done < self.total {
            return Ok(false);
        }
        state.finished = true;
        state.output.write_trailer().map_err(RtmpMuxerError::from)?;
        Ok(true)
    }
}

/// One track's own [`Sink`] — a lightweight handle sharing a
/// `RtmpMuxerShared` with every other track [`RtmpMuxer::open`] returned
/// alongside it. See [`RtmpMuxer::open`]'s own docs for the
/// finalize-once-every-track-is-done contract this relies on.
pub struct RtmpMuxerStreamSink {
    pp_log: PpLog,
    name: Arc<str>,
    shared: Arc<RtmpMuxerShared>,
    stream_index: usize,
    input_time_base: ffmpeg::Rational,
    /// The medium this track was registered for; `None` for one this crate
    /// does not model, which then declares nothing.
    kind: Option<MediaKind>,
    /// Set once this sink has contributed to
    /// [`RtmpMuxerShared::finish_track`] — guards against double-counting
    /// if both a natural `Eos` and a later `Stop` arrive for the same track.
    done: bool,
}

impl RtmpMuxerStreamSink {
    /// The publish address with its stream key removed — safe to log or
    /// show. See [`RtmpMuxer`]'s own docs.
    pub fn redacted_url(&self) -> &str {
        &self.shared.redacted_url
    }

    fn finish(&mut self) -> Result<()> {
        if self.done {
            return Ok(());
        }
        self.done = true;
        let closed = self
            .shared
            .finish_track()
            .inspect_err(|error| pp_error!(self, "write_trailer failed: {error}"))?;
        if closed {
            pp_info!(self, "publish closed: url={}", self.shared.redacted_url);
        }
        Ok(())
    }
}

impl Element for RtmpMuxerStreamSink {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::RtmpMuxer
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Sink for RtmpMuxerStreamSink {
    /// A muxer interleaves already-encoded data; it has no encoder of its
    /// own, so a decoded frame has no route through it. The medium is this
    /// track's own, so a video encoder wired into the audio track is
    /// refused rather than publishing something no player can follow.
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
            other => Err(RtmpMuxerError::UnsupportedBuffer(other.kind()).into()),
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        if let ControlMsg::CheckSeek(context) = &msg {
            // A live publish has no timeline to move within: what the
            // server already received cannot be taken back.
            context.reject(
                self.element_type(),
                self.name(),
                SeekRejectReason::ElementNotSeekable,
            );
        }
        // Terminal, nothing to forward. `Stop` still contributes to this
        // track's own "done" count — see `RtmpMuxer::open`'s own docs on
        // why the trailer waits for every track.
        if msg == ControlMsg::Stop {
            self.finish()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{RtmpMuxer, RtmpMuxerError, redact};
    use crate::error::Error;

    #[test]
    fn rejects_a_url_containing_a_nul_byte_before_connecting() {
        let result = RtmpMuxer::create("rtmp://127.0.0.1:1935/live/stream\0invalid");

        assert!(matches!(
            result,
            Err(Error::RtmpMuxerError(RtmpMuxerError::InvalidUrl))
        ));
    }

    /// A file path would otherwise reach `avio_open2` and create a file
    /// named after it, from a type whose whole contract is a live publish.
    #[test]
    fn rejects_a_url_that_is_not_rtmp() {
        for url in ["out.flv", "file:///tmp/out.flv", "rtsp://127.0.0.1:8554/s"] {
            assert!(
                matches!(
                    RtmpMuxer::create(url),
                    Err(Error::RtmpMuxerError(RtmpMuxerError::NotAnRtmpUrl))
                ),
                "{url} must not be accepted as an RTMP publish target"
            );
        }
    }

    #[test]
    fn accepts_every_scheme_ffmpegs_rtmp_protocol_answers_to() {
        for scheme in super::RTMP_SCHEMES {
            let url = format!("{scheme}://127.0.0.1:1935/live/stream");
            assert!(
                super::is_rtmp_url(&url),
                "{url} must be recognized as an RTMP publish target"
            );
        }
        assert!(
            super::is_rtmp_url("RTMPS://127.0.0.1/live/stream"),
            "the scheme is matched case-insensitively, as FFmpeg matches it"
        );
    }

    /// The stream key is a credential, and this is the only thing standing
    /// between it and a log file that gets attached to a bug report.
    #[test]
    fn redacts_the_stream_key() {
        let cases = [
            (
                "rtmp://live.twitch.tv/app/live_123456_abcdef",
                "rtmp://live.twitch.tv/app/<key>",
            ),
            (
                "rtmps://a.rtmp.youtube.com/live2/xxxx-yyyy-zzzz",
                "rtmps://a.rtmp.youtube.com/live2/<key>",
            ),
            (
                "rtmp://127.0.0.1:1935/live/stream?token=secret",
                "rtmp://127.0.0.1:1935/live/<key>?<query>",
            ),
            // One segment: redacted anyway rather than guessed about.
            ("rtmp://127.0.0.1/mystream", "rtmp://127.0.0.1/<key>"),
            // Nothing to remove, and nothing invented either.
            ("rtmp://127.0.0.1:1935", "rtmp://127.0.0.1:1935"),
            // A trailing slash is a path with no key in it.
            ("rtmp://127.0.0.1/live/", "rtmp://127.0.0.1/live/"),
        ];

        for (url, expected) in cases {
            assert_eq!(redact(url), expected, "redacting {url}");
        }
    }

    #[test]
    fn redacting_something_that_is_not_a_url_quotes_none_of_it() {
        assert_eq!(redact("live_123456_abcdef"), "<url>");
    }
}
