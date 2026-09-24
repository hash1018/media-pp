use std::{ffi::CString, ptr, sync::Arc};

use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;

use super::track_sink::{Muxer, PendingStream, TrackOptions, TrackTimeline, open_tracks};
use super::tracks::{MuxerId, MuxerSinks, MuxerTrack, TrackFormat};

use crate::{
    element::ElementType,
    elements::{RtspOptions, rtsp::redact},
    error::{Error, Result},
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

/// Publishes one or more compressed packet streams to an already-running
/// RTSP server, and hands out one [`Sink`](crate::element::Sink) per track.
///
/// The server must already be listening at `url` and must permit publishing
/// to that path. It can be [MediaMTX] or any other implementation that
/// accepts RTSP publishing; this element does not start, stop, or otherwise
/// depend on a particular server process.
///
/// This is a remuxing muxer, not an encoder. Incoming buffers must be
/// compressed [`MediaBuffer::Packet`](crate::buffer::MediaBuffer::Packet) values whose codec parameters and time
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
/// It also finalizes on `Eos` alone, not on [`ControlMsg::Stop`](crate::control::ControlMsg::Stop). A live
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
/// # use media_pp::elements::{RtspMuxer, RtspOptions, TrackFormat};
/// # fn main() -> media_pp::Result<()> {
/// # // Usually `&encoder`, for the encoder feeding each track.
/// # let video_format = TrackFormat::new(ffmpeg::codec::Parameters::new(), ffmpeg::Rational(1, 90_000));
/// # let audio_format = TrackFormat::new(ffmpeg::codec::Parameters::new(), ffmpeg::Rational(1, 48_000));
/// let mut muxer = RtspMuxer::create("rtsp://127.0.0.1:8554/stream", RtspOptions::default())?;
/// let video = muxer.add_stream("video", video_format)?;
/// let audio = muxer.add_stream("audio", audio_format)?;
/// let mut sinks = muxer.open()?; // performs the RTSP handshake
/// let video_sink = sinks.take(video)?;
/// let audio_sink = sinks.take(audio)?;
/// # Ok(())
/// # }
/// ```
pub struct RtspMuxer {
    id: MuxerId,
    output: ffmpeg::format::context::Output,
    streams: Vec<PendingStream>,
    options: RtspOptions,
    redacted_url: Arc<str>,
}

impl RtspMuxer {
    /// Allocates the RTSP muxer for `url`. Nothing reaches the network yet
    /// — see this type's own docs on why the handshake waits for
    /// [`RtspMuxer::open`].
    ///
    /// TCP is the most reliable transport for general networks; UDP is
    /// useful when the network path and server permit the negotiated
    /// RTP/RTCP ports. The timeout bounds `open`'s handshake and every
    /// write after it, so a server that stops answering fails a write rather
    /// than holding the branch behind it.
    pub fn create(url: impl AsRef<str>, options: RtspOptions) -> Result<Self> {
        crate::ensure_ffmpeg();
        let url = url.as_ref();
        let output = alloc_output(url)?;
        Ok(Self {
            id: MuxerId::next(),
            output,
            streams: Vec::new(),
            options,
            redacted_url: redact(url).into(),
        })
    }

    /// Registers one more track this session will publish. `format` (see
    /// [`TrackFormat`]) must describe every packet subsequently passed to that
    /// track's [`Sink::consume`](crate::element::Sink::consume). `name` becomes the track's own
    /// [`Element::name`](crate::element::Element::name)/`pp_log` identity — pick something that tells the
    /// tracks apart in logs and [`crate::bus::BusEvent`]s, such as
    /// `"video"`/`"audio"`.
    ///
    /// The returned [`MuxerTrack`] is how this track's sink is taken out of
    /// what [`RtspMuxer::open`] returns.
    pub fn add_stream(
        &mut self,
        name: impl Into<String>,
        format: impl Into<TrackFormat>,
    ) -> Result<MuxerTrack> {
        let TrackFormat {
            parameters,
            time_base,
        } = format.into();
        let pending = PendingStream::new(name, &parameters, time_base);
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

        let track = self.id.track(self.streams.len());
        self.streams.push(pending);
        Ok(track)
    }

    /// Performs the `ANNOUNCE`/`SETUP`/`RECORD` handshake — every
    /// [`RtspMuxer::add_stream`] call this session will get must already
    /// have happened, since the SDP it announces describes them all — and
    /// returns one [`Sink`](crate::element::Sink) per track, each taken out by the [`MuxerTrack`]
    /// its [`RtspMuxer::add_stream`] returned.
    ///
    /// All returned `Sink`s write through the same session behind a shared
    /// lock: independently-threaded branches arrive concurrently, and
    /// neither `av_interleaved_write_frame` nor `av_write_trailer` is safe
    /// to call from two threads against one output at once. They also share
    /// one trailer, written once every track has reported `Eos` — not on
    /// whichever finishes first, which would cut the others off mid-stream.
    pub fn open(mut self) -> Result<MuxerSinks> {
        crate::ensure_ffmpeg();
        self.output
            .write_header_with(self.options.to_dictionary())
            .map_err(RtspMuxerError::from)?;
        Ok(open_tracks::<Self>(
            self.id,
            self.output,
            self.streams,
            TrackOptions {
                // Deliberately not finalizing on `Stop` — see this type's own
                // docs on why a live publish differs from a file here.
                finish_on_stop: false,
                destination: Some(self.redacted_url),
                // Each track rebases onto its own last output, not onto a
                // shared one.
                timeline: Some(|| Box::new(Timeline::default())),
            },
        ))
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

impl TrackTimeline for Timeline {
    fn stamp(&mut self, packet: &mut ffmpeg::Packet) {
        Timeline::stamp(self, packet);
    }

    fn seeked(&mut self) {
        self.pending_seek = true;
    }
}

impl Muxer for RtspMuxer {
    const ELEMENT_TYPE: ElementType = ElementType::RtspMuxer;

    fn ffmpeg_error(error: ffmpeg::Error) -> Error {
        RtspMuxerError::Ffmpeg(error).into()
    }

    fn unsupported_buffer(kind: &'static str) -> Error {
        RtspMuxerError::UnsupportedBuffer(kind).into()
    }
}

#[cfg(test)]
mod tests {
    use ffmpeg_next as ffmpeg;

    use super::{RtspMuxer, RtspMuxerError, TrackFormat};
    use crate::{elements::RtspOptions, error::Error};

    #[test]
    fn rejects_a_url_containing_a_nul_byte_before_connecting() {
        let result = RtspMuxer::create(
            "rtsp://127.0.0.1:8554/stream\0invalid",
            RtspOptions::default(),
        );

        assert!(matches!(
            result,
            Err(Error::RtspMuxerError(RtspMuxerError::InvalidUrl))
        ));
    }

    /// Allocating does not reach the network, so this must succeed with
    /// nothing listening — the handshake is `open`'s to fail at.
    #[test]
    fn creating_does_not_need_a_server() {
        let mut muxer = RtspMuxer::create("rtsp://127.0.0.1:1/stream", RtspOptions::default())
            .expect("allocating an RTSP muxer must not connect");
        let _video = muxer
            .add_stream(
                "video",
                TrackFormat::new(
                    ffmpeg::codec::Parameters::new(),
                    ffmpeg::Rational(1, 90_000),
                ),
            )
            .expect("registering a track must not connect either");
    }

    /// A server that takes the connection and never answers — or none at
    /// all where the OS lets a connect wait unanswered — held `open` for
    /// good: nothing bounded the handshake. It gives up after the
    /// configured timeout instead.
    #[test]
    fn open_gives_up_on_a_server_that_never_answers() {
        // Held for the whole test: the OS completes the TCP handshake from
        // its backlog, and nothing ever reads the request or replies.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind a port");
        let port = listener.local_addr().expect("its port").port();
        let (done, finished) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let options = RtspOptions {
                timeout: std::time::Duration::from_millis(500),
                ..Default::default()
            };
            let mut muxer = RtspMuxer::create(format!("rtsp://127.0.0.1:{port}/stream"), options)
                .expect("allocate");
            let mut parameters = ffmpeg::codec::Parameters::new();
            // SAFETY: parameters this test owns; a video stream the SDP can
            // describe, so the handshake is what `open` reaches.
            unsafe {
                let raw = parameters.as_mut_ptr();
                (*raw).codec_type = ffmpeg::ffi::AVMediaType::AVMEDIA_TYPE_VIDEO;
                (*raw).codec_id = ffmpeg::ffi::AVCodecID::AV_CODEC_ID_H264;
                (*raw).width = 64;
                (*raw).height = 64;
            }
            let _video = muxer
                .add_stream(
                    "video",
                    TrackFormat::new(parameters, ffmpeg::Rational(1, 90_000)),
                )
                .expect("register a track");
            let started = std::time::Instant::now();
            let opened = muxer.open().map(drop);
            let _ = done.send((opened, started.elapsed()));
        });
        let (opened, took) = finished
            .recv_timeout(std::time::Duration::from_secs(30))
            .expect("open is still waiting on a server that will never answer");
        assert!(
            opened.is_err(),
            "a server that never answers is not a session"
        );
        assert!(
            took < std::time::Duration::from_secs(15),
            "open took {took:?}"
        );
        drop(listener);
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
