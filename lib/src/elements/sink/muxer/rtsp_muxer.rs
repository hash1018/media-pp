use std::{
    ffi::CString,
    ptr,
    sync::{Arc, Mutex},
};

use crate::pp_log::{PpLog, pp_error, pp_info};
use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, PortContract},
    control::ControlMsg,
    element::{Element, ElementType, Sink, element_pp_log},
    elements::RtspTransport,
    error::Result,
};

/// Errors produced while opening or writing an [`RtspMuxer`].
#[derive(Debug, ThisError)]
pub enum RtspMuxerError {
    /// FFmpeg rejected connection setup or packet writing.
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),

    /// A stream sink received a decoded frame instead of compressed packet
    /// data.
    #[error(
        "RtspMuxer only remuxes compressed Packets, got a decoded {0}; \
         connect an encoder or demuxer packet pad instead"
    )]
    UnsupportedBuffer(&'static str),

    /// The URL contains an interior NUL byte rejected by FFmpeg's C API.
    #[error("RTSP URL contains a NUL byte")]
    InvalidUrl,
}

/// One track registered via [`RtspMuxer::add_stream`], waiting for
/// [`RtspMuxer::open`] to turn it into a real [`RtspMuxerStreamSink`] — its
/// `name` becomes that sink's own [`Element::name`]/`pp_log` identity, and
/// `input_time_base` is what every `Packet` it receives already carries
/// `pts`/`dts` in.
struct PendingStream {
    name: Arc<str>,
    input_time_base: ffmpeg::Rational,
    /// Taken from the parameters this track was registered with, so its
    /// sink can refuse the other medium's packets at wiring time.
    kind: Option<MediaKind>,
}

/// Publishes one or more compressed packet streams to an already-running
/// RTSP server, and hands out one [`Sink`] per track.
///
/// The server must already be listening at `url` and must permit publishing
/// to that path. It can be [MediaMTX] or any other implementation that
/// accepts RTSP publishing; this element does not start, stop, or otherwise
/// depend on a particular server process.
///
/// This is a remuxing muxer, not an encoder. Incoming buffers must be
/// compressed [`MediaBuffer::Packet`] values whose codec parameters and time
/// base match what that track was registered with. Place a
/// [`Pacer`](crate::elements::Pacer) upstream when publishing packets from a
/// file, otherwise the file is sent faster than real time.
///
/// # The handshake is in `open`, not `create`
///
/// RTSP is a libavformat muxer rather than a generic AVIO protocol: nothing
/// is on the network until the header is written, and the
/// `ANNOUNCE`/`SETUP`/`RECORD` handshake happens inside
/// [`RtspMuxer::open`]. So an unreachable server, a refused path, or a
/// transport the server will not negotiate all fail there — not at
/// [`RtspMuxer::create`], which only allocates. This is the opposite of
/// [`RtmpMuxer`](crate::elements::RtmpMuxer), whose protocol *is* an AVIO
/// one and so connects when it is created.
///
/// # Credentials
///
/// An RTSP URL may carry them in its authority — `rtsp://user:pass@host/path`
/// is how most cameras and some servers expect to be addressed. Nothing here
/// logs the URL it was given: [`RtspMuxer::redacted_url`] is what reaches a
/// log and what a caller should display, and it replaces the userinfo. The
/// path is left alone, which is the opposite of `RtmpMuxer`'s redaction —
/// there the credential *is* the last path segment.
///
/// # Seeking, and finalizing
///
/// Unlike the other muxers this one does not refuse a seek: an upstream that
/// jumps keeps a monotonic published timeline, because each track rebases its
/// own output timestamps onto its own last one. The tracks rebase
/// independently, so a seek can shift them relative to each other by however
/// far apart their last outputs were.
///
/// It also finalizes on `Eos` alone, not on [`ControlMsg::Stop`]. A live
/// publish that is abandoned has nothing that needs a valid trailer to be
/// readable — unlike a file, which is why
/// [`FileMuxer`](crate::elements::FileMuxer) treats `Stop` as a track
/// finishing and this does not. Dropping the last sink tears the session
/// down either way.
///
/// [MediaMTX]: https://github.com/bluenviron/mediamtx
///
/// ```no_run
/// # use media_pp::ffmpeg;
/// # use media_pp::elements::{RtspMuxer, RtspTransport};
/// # fn main() -> media_pp::Result<()> {
/// # let video_params = ffmpeg::codec::Parameters::new();
/// # let audio_params = ffmpeg::codec::Parameters::new();
/// # let video_time_base = ffmpeg::Rational(1, 90_000);
/// # let audio_time_base = ffmpeg::Rational(1, 48_000);
/// let mut muxer = RtspMuxer::create("rtsp://127.0.0.1:8554/stream", RtspTransport::Tcp)?;
/// muxer.add_stream("video", video_params, video_time_base)?;
/// muxer.add_stream("audio", audio_params, audio_time_base)?;
/// let mut sinks = muxer.open()?; // performs the RTSP handshake
/// let audio_sink = sinks.pop().unwrap();
/// let video_sink = sinks.pop().unwrap();
/// # Ok(())
/// # }
/// ```
pub struct RtspMuxer {
    output: ffmpeg::format::context::Output,
    streams: Vec<PendingStream>,
    transport: RtspTransport,
    redacted_url: Arc<str>,
}

impl RtspMuxer {
    /// Allocates the RTSP muxer for `url`. Nothing reaches the network yet
    /// — see this type's own docs on why the handshake waits for
    /// [`RtspMuxer::open`].
    ///
    /// TCP is the most reliable transport for general networks; UDP is
    /// useful when the network path and server permit the negotiated
    /// RTP/RTCP ports.
    pub fn create(url: impl AsRef<str>, transport: RtspTransport) -> Result<Self> {
        let url = url.as_ref();
        let output = alloc_output(url)?;
        Ok(Self {
            output,
            streams: Vec::new(),
            transport,
            redacted_url: redact(url).into(),
        })
    }

    /// Registers one more track this session will publish. `parameters`/
    /// `time_base` must describe every packet subsequently passed to that
    /// track's [`Sink::consume`]. `name` becomes the track's own
    /// [`Element::name`]/`pp_log` identity — pick something that tells the
    /// tracks apart in logs and [`crate::bus::BusEvent`]s, such as
    /// `"video"`/`"audio"`.
    ///
    /// Add streams in the same order the caller will read
    /// [`RtspMuxer::open`]'s returned `Vec` — index 0 is whichever was
    /// added first.
    pub fn add_stream(
        &mut self,
        name: impl Into<String>,
        parameters: ffmpeg::codec::Parameters,
        time_base: ffmpeg::Rational,
    ) -> Result<()> {
        let kind = MediaKind::packet_for(parameters.medium());
        let mut stream = self
            .output
            .add_stream(ffmpeg::encoder::find(ffmpeg::codec::Id::None))
            .map_err(RtspMuxerError::from)?;
        stream.set_parameters(parameters);
        // Avoid codec-tag incompatibilities when the input packet came from
        // a container with a different tag convention.
        // SAFETY: `as_mut_ptr` on parameters this stream owns, written before the
        // stream is handed to the muxer — see the comment beside it for why the tag
        // is cleared at all.
        unsafe {
            (*stream.parameters().as_mut_ptr()).codec_tag = 0;
        }
        stream.set_time_base(time_base);

        self.streams.push(PendingStream {
            name: name.into().into(),
            input_time_base: time_base,
            kind,
        });
        Ok(())
    }

    /// Performs the `ANNOUNCE`/`SETUP`/`RECORD` handshake — every
    /// [`RtspMuxer::add_stream`] call this session will get must already
    /// have happened, since the SDP it announces describes them all — and
    /// returns one [`Sink`] per track, in the order they were added.
    ///
    /// All returned `Sink`s write through the same session behind a shared
    /// lock: independently-threaded branches arrive concurrently, and
    /// neither `av_interleaved_write_frame` nor `av_write_trailer` is safe
    /// to call from two threads against one output at once. They also share
    /// one trailer, written once every track has reported `Eos` — not on
    /// whichever finishes first, which would cut the others off mid-stream.
    pub fn open(mut self) -> Result<Vec<Box<dyn Sink>>> {
        let mut options = ffmpeg::Dictionary::new();
        options.set("rtsp_transport", self.transport.as_ffmpeg_option());
        self.output
            .write_header_with(options)
            .map_err(RtspMuxerError::from)?;

        // Read after the header, which is when the muxer has settled it —
        // RTSP does not keep the time base a track was registered with, it
        // announces its own in the SDP.
        let output_time_bases: Vec<_> = (0..self.streams.len())
            .map(|index| {
                self.output
                    .stream(index)
                    .expect("stream was added in RtspMuxer::add_stream")
                    .time_base()
            })
            .collect();

        let total = self.streams.len();
        let redacted_url = self.redacted_url;
        let transport = self.transport;
        let shared = Arc::new(RtspMuxerShared {
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
                let pp_log = element_pp_log(ElementType::RtspMuxer, &stream.name, None);
                pp_info!(
                    pp_log: &pp_log,
                    "publishing: url={redacted_url}, transport={transport:?}, tracks={total}"
                );
                Box::new(RtspMuxerStreamSink {
                    pp_log,
                    name: stream.name,
                    shared: shared.clone(),
                    stream_index: index,
                    input_time_base: stream.input_time_base,
                    output_time_base: output_time_bases[index],
                    kind: stream.kind,
                    timeline: Timeline::default(),
                    done: false,
                })
            })
            .collect())
    }

    /// The publish address with any credentials removed — what to log, and
    /// what to show a user. See this type's own docs.
    pub fn redacted_url(&self) -> &str {
        &self.redacted_url
    }
}

/// Allocates an RTSP muxer without opening a generic `AVIOContext`.
///
/// RTSP is a libavformat muxer, not a generic AVIO protocol. Its muxer owns
/// the control and RTP sockets internally during header/packet writes, while
/// `ffmpeg_next::format::output_as` attempts an incompatible generic
/// `avio_open2` first on FFmpeg builds where `rtsp` is not an AVIO protocol.
fn alloc_output(url: &str) -> Result<ffmpeg::format::context::Output> {
    let c_url = CString::new(url).map_err(|_| RtspMuxerError::InvalidUrl)?;
    let c_format = CString::new("rtsp").expect("static format name contains no NUL");

    // SAFETY: `c_format` and `c_url` are live NUL-terminated `CString`s, and
    // `context` is a live local. Every path below checks it before use, and the
    // failure paths free what was allocated.
    unsafe {
        let mut context: *mut ffi::AVFormatContext = ptr::null_mut();
        let result = ffi::avformat_alloc_output_context2(
            &mut context,
            ptr::null_mut(),
            c_format.as_ptr(),
            c_url.as_ptr(),
        );
        if result < 0 {
            return Err(RtspMuxerError::Ffmpeg(ffmpeg::Error::from(result)).into());
        }

        Ok(ffmpeg::format::context::Output::wrap(context))
    }
}

/// Removes credentials from a publish URL, leaving enough to recognize where
/// a stream was going.
///
/// Only the authority can carry userinfo, so a `@` later in the path is part
/// of the path and stays. Unlike RTMP, the path itself is not a secret here.
fn redact(url: &str) -> String {
    let Some((scheme, rest)) = url.split_once("://") else {
        // Not a shape this understands, so nothing about it is quotable.
        return "<url>".to_string();
    };
    let authority_end = rest.find('/').unwrap_or(rest.len());
    match rest[..authority_end].rfind('@') {
        Some(at) => format!("{scheme}://<credentials>@{}", &rest[at + 1..]),
        None => url.to_string(),
    }
}

/// One track's published timestamps, kept monotonic across an upstream seek.
///
/// Split out from the sink because it is the part worth testing on its own:
/// exercising it through a `Sink` would need a listening RTSP server, and
/// what it has to get right is arithmetic.
#[derive(Default)]
struct Timeline {
    last_output_dts: Option<i64>,
    last_output_pts: Option<i64>,
    pts_offset: i64,
    pending_seek: bool,
}

impl Timeline {
    /// Rewrites `packet`'s timestamps into this track's published timeline.
    ///
    /// A seek rebases onto what this track itself last sent, so the stream a
    /// receiver sees never jumps backwards where the source did. Each track
    /// answers only for its own timeline — see [`RtspMuxer`]'s own docs on
    /// what that means for two of them.
    ///
    /// DTS is the muxer's hard ordering requirement; PTS is the fallback for
    /// packets that carry no DTS, and is left free to reorder where one
    /// does, since B-frames legitimately present out of decode order.
    fn stamp(&mut self, packet: &mut ffmpeg::Packet) {
        let Some(raw_pts) = packet.pts() else {
            return;
        };
        let raw_dts = packet.dts();

        if self.pending_seek {
            self.pts_offset = match (self.last_output_dts, raw_dts) {
                (Some(last_dts), Some(raw_dts)) => last_dts + 1 - raw_dts,
                _ => match self.last_output_pts {
                    Some(last_pts) => last_pts + 1 - raw_pts,
                    None => 0,
                },
            };
            self.pending_seek = false;
        }

        let mut corrected_pts = raw_pts + self.pts_offset;
        let mut corrected_dts = raw_dts.map(|dts| dts + self.pts_offset);

        // The rebase above anchors on whichever packet arrived first, and a
        // seek is answered with a short out-of-order burst: the demuxer
        // lands near the seek point rather than exactly on it, so a
        // straggler can still carry a lower timestamp than the packet that
        // set the offset. Measured on a real file, that is about three AAC
        // frames — 72ms — of audio landing behind what was already sent.
        //
        // Shift such a packet forward instead of publishing it late. A
        // receiver drops a packet whose DTS went backwards, and a muxer
        // refuses to interleave one, so 72ms of skew is the cheaper of the
        // two answers. Both stamps move together, keeping the gap between
        // them the decoder's delay rather than an artifact of this.
        match (corrected_dts, self.last_output_dts) {
            (Some(dts), Some(last_dts)) if dts <= last_dts => {
                let shift = last_dts + 1 - dts;
                corrected_dts = Some(dts + shift);
                corrected_pts += shift;
            }
            // No DTS to order by, so PTS is what the muxer will use.
            (None, _) => {
                if let Some(last_pts) = self.last_output_pts
                    && corrected_pts <= last_pts
                {
                    corrected_pts = last_pts + 1;
                }
            }
            _ => {}
        }

        packet.set_pts(Some(corrected_pts));
        if let Some(dts) = corrected_dts {
            packet.set_dts(Some(dts));
            self.last_output_dts = Some(dts);
        }
        self.last_output_pts = Some(corrected_pts);
    }
}

struct MuxerState {
    output: ffmpeg::format::context::Output,
    /// How many tracks have reported `Eos` — the trailer is written once
    /// this reaches [`RtspMuxerShared::total`], not on the first one (see
    /// [`RtspMuxer::open`]'s own docs for why).
    done: usize,
    /// Set once the trailer has been written. Each
    /// [`RtspMuxerStreamSink`]'s own `done` flag already prevents
    /// double-counting *that* track; this additionally guards
    /// [`RtspMuxerShared::write_packet`] against writing into a session
    /// already torn down.
    finished: bool,
}

/// Shared between every [`RtspMuxerStreamSink`] [`RtspMuxer::open`] hands out
/// for the same session — one lock around the whole
/// [`ffmpeg::format::context::Output`] so concurrent tracks never interleave
/// two writes against it (see [`RtspMuxer::open`]'s own docs).
struct RtspMuxerShared {
    state: Mutex<MuxerState>,
    total: usize,
    redacted_url: Arc<str>,
}

impl RtspMuxerShared {
    /// Writes a packet whose stream index and timestamps its caller has
    /// already settled — the rebase is per track, so it happens in the sink
    /// rather than here (see [`RtspMuxerStreamSink::stamp`]).
    fn write_packet(&self, packet: &mut ffmpeg::Packet) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        if state.finished {
            return Ok(());
        }
        packet
            .write_interleaved(&mut state.output)
            .map_err(RtspMuxerError::from)?;
        Ok(())
    }

    /// One track reporting `Eos` — writes the trailer exactly once, only
    /// after every track has called this. `true` when this call was the one
    /// that ended the session.
    fn finish_track(&self) -> Result<bool> {
        let mut state = self.state.lock().unwrap();
        state.done += 1;
        if state.finished || state.done < self.total {
            return Ok(false);
        }
        state.finished = true;
        state.output.write_trailer().map_err(RtspMuxerError::from)?;
        Ok(true)
    }
}

/// One track's own [`Sink`] — a lightweight handle sharing a
/// `RtspMuxerShared` with every other track [`RtspMuxer::open`] returned
/// alongside it, plus the timestamp state that makes *this* track's
/// published timeline monotonic across an upstream seek.
pub struct RtspMuxerStreamSink {
    pp_log: PpLog,
    name: Arc<str>,
    shared: Arc<RtspMuxerShared>,
    stream_index: usize,
    input_time_base: ffmpeg::Rational,
    /// What the muxer announced for this track, which is not necessarily
    /// what it was registered with — captured in [`RtspMuxer::open`] once
    /// the header had settled it.
    output_time_base: ffmpeg::Rational,
    /// The medium this track was registered for; `None` for one this crate
    /// does not model, which then declares nothing.
    kind: Option<MediaKind>,
    /// This track's published timestamps. Per track on purpose: each
    /// rebases onto its own last output, not onto a shared one.
    timeline: Timeline,
    /// Set once this sink has contributed to
    /// [`RtspMuxerShared::finish_track`], so a second `Eos` cannot count
    /// this track twice.
    done: bool,
}

impl RtspMuxerStreamSink {
    /// The publish address with any credentials removed — safe to log or
    /// show. See [`RtspMuxer`]'s own docs.
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

impl Element for RtspMuxerStreamSink {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::RtspMuxer
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Sink for RtspMuxerStreamSink {
    /// Republishes encoded data as-is; it has no encoder of its own. The
    /// medium is this track's own, so a video packet pad wired into the
    /// audio track is refused rather than announced as something it is not.
    fn input_contract(&self) -> InputContract {
        match self.kind {
            Some(kind) => InputContract::Fixed(PortContract::packet(kind)),
            None => InputContract::Unknown,
        }
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Packet(packet) => {
                // Cloned, not mutated in place — `Arc<Packet>` may be shared
                // with another branch off the same `Tee`, which must not see
                // this track's rescaled timestamps or stream index.
                let mut packet = (*packet).clone();
                packet.rescale_ts(self.input_time_base, self.output_time_base);
                self.timeline.stamp(&mut packet);
                packet.set_stream(self.stream_index);
                packet.set_position(-1);
                self.shared
                    .write_packet(&mut packet)
                    .inspect_err(|error| pp_error!(self, "write_interleaved failed: {error}"))
            }
            MediaBuffer::Eos => self.finish(),
            MediaBuffer::Video(_) => {
                pp_error!(self, "unsupported buffer: Video");
                Err(RtspMuxerError::UnsupportedBuffer("Video").into())
            }
            MediaBuffer::Audio(_) => {
                pp_error!(self, "unsupported buffer: Audio");
                Err(RtspMuxerError::UnsupportedBuffer("Audio").into())
            }
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        match msg {
            ControlMsg::Seek(_) => self.timeline.pending_seek = true,
            // Deliberately not finalizing on `Stop` — see `RtspMuxer`'s own
            // docs on why a live publish differs from a file here.
            ControlMsg::Pause
            | ControlMsg::Resume
            | ControlMsg::Stop
            | ControlMsg::Flush
            | ControlMsg::CheckSeek(_)
            | ControlMsg::Preroll(_) => {}
        }
        Ok(())
    }
}

impl Drop for RtspMuxerStreamSink {
    fn drop(&mut self) {
        pp_info!(
            self,
            "dropped: releasing this track of the publisher session to {}",
            self.shared.redacted_url
        );
    }
}

#[cfg(test)]
mod tests {
    use ffmpeg_next as ffmpeg;

    use super::{RtspMuxer, RtspMuxerError, redact};
    use crate::{elements::RtspTransport, error::Error};

    #[test]
    fn rejects_a_url_containing_a_nul_byte_before_connecting() {
        let result = RtspMuxer::create("rtsp://127.0.0.1:8554/stream\0invalid", RtspTransport::Tcp);

        assert!(matches!(
            result,
            Err(Error::RtspMuxerError(RtspMuxerError::InvalidUrl))
        ));
    }

    /// Allocating does not reach the network, so this must succeed with
    /// nothing listening — the handshake is `open`'s to fail at.
    #[test]
    fn creating_does_not_need_a_server() {
        let mut muxer = RtspMuxer::create("rtsp://127.0.0.1:1/stream", RtspTransport::Tcp)
            .expect("allocating an RTSP muxer must not connect");
        muxer
            .add_stream(
                "video",
                ffmpeg::codec::Parameters::new(),
                ffmpeg::Rational(1, 90_000),
            )
            .expect("registering a track must not connect either");
    }

    /// Cameras and some servers are addressed with the password in the URL,
    /// and this is the only thing between it and a log file.
    #[test]
    fn redacts_credentials_from_the_authority() {
        let cases = [
            (
                "rtsp://admin:hunter2@192.168.0.10:554/stream1",
                "rtsp://<credentials>@192.168.0.10:554/stream1",
            ),
            ("rtsp://user@host/path", "rtsp://<credentials>@host/path"),
            // Nothing to remove: an RTSP path is not itself a secret, which
            // is where this differs from `RtmpMuxer`.
            (
                "rtsp://127.0.0.1:8554/stream",
                "rtsp://127.0.0.1:8554/stream",
            ),
            // A `@` in the path is part of the path, not userinfo.
            ("rtsp://127.0.0.1:8554/a@b", "rtsp://127.0.0.1:8554/a@b"),
        ];

        for (url, expected) in cases {
            assert_eq!(redact(url), expected, "redacting {url}");
        }
    }

    #[test]
    fn redacting_something_that_is_not_a_url_quotes_none_of_it() {
        assert_eq!(redact("admin:hunter2"), "<url>");
    }

    /// Feeds `timeline` packets stamped `pts == dts == t` and reports what
    /// it published for each.
    fn publish(timeline: &mut super::Timeline, timestamps: &[i64]) -> Vec<i64> {
        timestamps
            .iter()
            .map(|&t| {
                let mut packet = ffmpeg::Packet::empty();
                packet.set_pts(Some(t));
                packet.set_dts(Some(t));
                timeline.stamp(&mut packet);
                packet.dts().expect("a stamped packet keeps its dts")
            })
            .collect()
    }

    #[test]
    fn an_unsought_stream_is_published_exactly_as_it_arrived() {
        let mut timeline = super::Timeline::default();

        assert_eq!(
            publish(&mut timeline, &[1024, 2048, 3072]),
            [1024, 2048, 3072],
            "nothing may be rewritten while no seek has happened"
        );
    }

    #[test]
    fn a_seek_continues_from_what_this_track_last_sent() {
        let mut timeline = super::Timeline::default();
        publish(&mut timeline, &[1024, 2048]);

        timeline.pending_seek = true;
        assert_eq!(
            publish(&mut timeline, &[500_000, 501_024]),
            [2049, 3073],
            "a source that jumped must not take the published timeline with it"
        );
    }

    /// The measured failure: publishing a real file over RTSP, a seek was
    /// answered with three AAC frames that arrived after the one the rebase
    /// anchored on but carried lower timestamps — 3179 samples, 72ms, of
    /// audio landing behind what had already gone out. A receiver drops a
    /// packet whose DTS went backwards.
    #[test]
    fn a_straggler_after_a_seek_is_still_published_in_order() {
        let mut timeline = super::Timeline::default();
        publish(&mut timeline, &[440_321, 441_345, 442_369, 443_393]);

        timeline.pending_seek = true;
        // The burst as it actually arrived: the anchor first, then three
        // packets from before it.
        let published = publish(&mut timeline, &[2_443_393, 2_440_214, 2_441_238, 2_442_262]);

        assert_eq!(
            published[0], 443_394,
            "the anchor still continues from the last published packet"
        );
        assert!(
            published.windows(2).all(|pair| pair[1] > pair[0]),
            "every packet after it must still advance: {published:?}"
        );
    }

    #[test]
    fn a_packet_with_no_dts_is_ordered_by_its_pts() {
        let mut timeline = super::Timeline::default();
        let stamp = |timeline: &mut super::Timeline, pts: i64| {
            let mut packet = ffmpeg::Packet::empty();
            packet.set_pts(Some(pts));
            packet.set_dts(None);
            timeline.stamp(&mut packet);
            packet.pts().expect("a stamped packet keeps its pts")
        };

        assert_eq!(stamp(&mut timeline, 1_000), 1_000);
        assert_eq!(
            stamp(&mut timeline, 900),
            1_001,
            "with no dts to order by, pts is what the muxer interleaves on"
        );
    }
}
