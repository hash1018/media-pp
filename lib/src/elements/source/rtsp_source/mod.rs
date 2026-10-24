//! [`RtspSource`]: an RTSP client, and a receiver of one publisher's push —
//! this crate's own, over std's sockets, with no FFmpeg between it and the
//! network.

mod pull;
mod push;
mod receive;

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::pp_log::{PpLog, pp_error, pp_info, pp_warn};
use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    contract::{MediaKind, OutputContract, PortContract},
    element::{Attached, ElementType, Identity, Produced, Source, SourceStage, Wait},
    elements::{
        RtspOptions,
        parameter_sets::{annex_b, h264_sequence, hevc_sequence},
        rtsp::{message::Malformed, redact, sdp::Announced},
    },
    error::Result,
    pad::SrcPad,
    produce::{Received, source_stage},
};

use crate::elements::StreamInfo;

use crate::core::bell::Bell;
use receive::{BACKLOG, Item, Outbox, Reader, Session, Silence};
#[cfg(feature = "rtsp-server")]
pub(crate) use receive::{Item as PublishedItem, Outbox as PublishedOutbox};

/// Why an [`RtspSource`] could not connect or read.
///
/// Converts into the crate-wide `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum RtspSourceError {
    /// [`RtspSource::open`] was given something other than an
    /// `rtsp://[user:password@]host[:port]/path` URL. `rtsps://` is not
    /// taken.
    #[error("RtspSource::open needs an rtsp://host[:port]/path URL")]
    InvalidUrl,
    /// The server could not be reached at `address`: refused, unreachable,
    /// or no answer to the connection within [`RtspOptions::timeout`].
    #[error("cannot connect to {address}: {error}")]
    Connect {
        address: String,
        #[source]
        error: std::io::Error,
    },
    /// The server did not answer a request within
    /// [`RtspOptions::timeout`].
    #[error("the server did not answer within {0:?}")]
    NoAnswer(std::time::Duration),
    /// The server answered `method` with `status`: a path it does not
    /// have (404), a transport it will not deliver (461), and the like.
    #[error("the server answered {method} with {status}")]
    Refused { method: &'static str, status: u16 },
    /// The server asked for credentials, and the URL carried none, or none
    /// it took.
    #[error("the server refused the credentials, or asked for some the URL did not carry")]
    Unauthorized,
    /// What the other end sent is not RTSP, or not a description this
    /// reads; this names what.
    #[error("the other end sent a malformed {0}")]
    Malformed(&'static str),
    /// Nothing in the session can be received: no track in a codec this
    /// takes apart — H.264, H.265, VP8, VP9, AAC, Opus, G.711 — or a delivery it
    /// does not take; this names what was refused.
    #[error("nothing in the session can be received: {0}")]
    Unsupported(String),
    /// The connection failed while the session was set up or read.
    #[error("the connection failed: {0}")]
    Io(#[from] std::io::Error),
    /// A server whose session was playing sent no media for
    /// [`RtspOptions::timeout`], which this names: a camera unplugged, a
    /// network gone, with nothing to close the connection.
    #[error("the server sent nothing for {0:?}")]
    ServerSilent(std::time::Duration),
    /// The session has no stream of the kind asked for — see
    /// [`RtspSource::best`].
    #[error("the RTSP session has no {0:?} stream")]
    NoStream(ffmpeg::media::Type),
    /// [`RtspSource::listen`] was given something other than an
    /// `rtsp://host:port/path` URL, or one holding a NUL byte.
    #[error("RtspSource::listen needs an rtsp://host:port/path URL")]
    InvalidListenUrl,
    /// A port could not be bound: the URL's, for [`RtspSource::listen`], or
    /// a UDP pair for a track's RTP.
    #[error("cannot bind a port: {0}")]
    Bind(#[source] std::io::Error),
    /// No publisher connected and announced its streams within
    /// [`RtspListenOptions::wait`], which this names. The port is closed
    /// again by the time this is returned.
    #[error("no publisher announced a stream within {0:?}")]
    NoPublisher(Duration),
    /// A publisher [`RtspSource::listen`] received sent nothing for
    /// [`RtspListenOptions::timeout`], which this names — before its streams
    /// could be probed, or while they were being read.
    #[error("the publisher sent nothing for {0:?}")]
    PublisherSilent(Duration),
}

impl From<Malformed> for RtspSourceError {
    fn from(Malformed(what): Malformed) -> Self {
        Self::Malformed(what)
    }
}

/// How [`RtspSource::listen`] waits for a publisher, and how long one may
/// then go silent.
///
/// Not [`RtspOptions`]: there is no transport to choose, since the
/// publisher's `SETUP` picks one — RTP interleaved in the RTSP connection
/// (TCP) or RTP over UDP — and the listener takes either.
#[derive(Debug, Clone, Copy)]
pub struct RtspListenOptions {
    /// How long [`RtspSource::listen`] waits for a publisher to connect and
    /// finish its `ANNOUNCE`/`SETUP`/`RECORD` handshake. Past it, `listen`
    /// returns [`RtspSourceError::NoPublisher`] with the port closed again.
    /// A duration too long for the clock to add (`Duration::MAX`) waits
    /// without end, as asked.
    pub wait: Duration,
    /// How long a publisher may send nothing — between any two packets —
    /// before the session is given up as
    /// [`RtspSourceError::PublisherSilent`]: a pulled cable or a vanished
    /// host closes no connection, and without this a read would wait on it
    /// for ever.
    pub timeout: Duration,
}

impl Default for RtspListenOptions {
    /// Thirty seconds for a publisher to arrive, and the five seconds of
    /// silence [`RtspOptions`] allows a server.
    fn default() -> Self {
        Self {
            wait: Duration::from_secs(30),
            timeout: Duration::from_secs(5),
        }
    }
}

/// Receives a live RTSP stream — a camera, a server — and hands on its
/// packets, one output per stream.
///
/// It is the client/receive counterpart to [`crate::elements::RtspMuxer`]
/// (which publishes), and has the same shape as
/// [`crate::elements::FileDemuxer`]: open it, pick a stream with
/// [`RtspSource::best`], and attach a branch to that stream's `index`. It
/// hands on encoded packets, as they arrive; it is live, so the branch needs
/// no pacer, and it cannot be sought.
///
/// # Example
///
/// ```no_run
/// # fn main() -> media_pp::Result<()> {
/// use media_pp::elements::{FrameCounter, RtspOptions, RtspSource, SwDecoder};
/// use media_pp::ffmpeg::media;
/// use media_pp::pipeline::Pipeline;
///
/// let url = "rtsp://192.168.0.10:554/stream";
/// let (source, _streams) = RtspSource::open("camera", url, RtspOptions::default())?;
/// let video = source.best(media::Type::Video)?;
/// let decoder = SwDecoder::new("decoder", video.parameters().clone())?;
/// let (counter, _frames) = FrameCounter::new("counter");
///
/// let (pipeline, ()) = Pipeline::new("camera", source, |source, ctx| {
///     let branch = ctx.branch().queue("packets", 32).pipe(decoder).to(counter)?;
///     ctx.attach(source, video.index(), branch)?;
///     Ok(())
/// })?;
/// pipeline.run()?;
/// # Ok(())
/// # }
/// ```
///
/// # What it receives
///
/// RTSP 1.0, as cameras and servers speak it, by this crate's own client:
///
/// - **Codecs:** H.264, H.265, VP8 and VP9, AAC (`mpeg4-generic`), Opus, and G.711
///   (µ-law and A-law). A described track in any other codec — MJPEG,
///   MPEG-4 Part 2, a camera's ONVIF metadata — is left out, and the
///   streams are those it can take, numbered in the order described; a
///   session with none is [`RtspSourceError::Unsupported`].
/// - **Delivery:** RTP interleaved in the RTSP connection
///   ([`RtspTransport::Tcp`](crate::elements::RtspTransport::Tcp)), or
///   over a pair of UDP ports per track, put back in order as it arrives.
///   No multicast, no RTSP over HTTP, no `rtsps://`.
/// - **Credentials** in the URL — `rtsp://user:password@host/path`, `%`
///   escapes decoded — answer a server's Digest or Basic challenge, and
///   never go into a request's URL.
/// - **The session is kept** with `GET_PARAMETER` where the server takes
///   it, `OPTIONS` otherwise, every half of the timeout its `Session`
///   says, and torn down when the source stops.
///
/// H.264 and H.265 come as Annex B access units, a picture to a packet,
/// keyframes flagged. RTP gives each picture only the time it is shown at
/// (its `pts`); for a stream that reorders its pictures — B-frames — the
/// time each is decoded at is worked out from the stream's own picture
/// order counts and its sequence parameter set, so a muxer downstream is
/// handed times that never go back, and [`StreamInfo::reorders`] says
/// so before the pipeline runs. A stream that does not reorder is decoded
/// when it is shown.
///
/// What the description leaves out — a picture's size, where the
/// parameter sets come only in band — is read from the first keyframe
/// before [`RtspSource::open`] returns, for up to five seconds and the
/// timeout beyond them; [`StreamInfo::frame_rate`] is what an H.264
/// stream's VUI says, and `None` where nothing does.
///
/// # Errors and reconnecting
///
/// Deliberately does **not** retry or reconnect internally: a read failure
/// (dropped connection, camera reboot, a server silent for
/// [`RtspOptions::timeout`]) ends this source with the error on the bus,
/// the same way any other fatal source failure does, instead of looping
/// forever. Reconnecting means building a fresh
/// `RtspSource`/[`crate::pipeline::Pipeline`] — mirrors `Pipeline` itself
/// not being reusable once it ends: watch
/// [`crate::pipeline::Pipeline::bus`], and on error, call
/// [`RtspSource::open`] again. A server that closes the connection ends
/// the stream, each output after what came before.
///
/// The session is read on threads of its own — one for the connection,
/// and one for each track's UDP socket — and handed to the source through
/// a buffer of some seconds of packets: a pause or a stop reaches the
/// source at once, whatever the network does. A source not reading for
/// longer than that loses what comes past it, and a picture stream goes on
/// from its next keyframe.
///
/// # Receiving a push
///
/// [`RtspSource::listen`] turns the roles round: rather than asking a
/// server for a stream, it takes the port in its URL and waits for a
/// publisher — an encoder, a camera that pushes, another program's
/// [`RtspMuxer`](crate::elements::RtspMuxer), `ffmpeg -f rtsp` — to
/// `ANNOUNCE` its streams and `RECORD` into it. What it hands on after that
/// is the same as from [`RtspSource::open`]: one output per stream the
/// publisher announced and set up, [`RtspSource::best`] to pick one, no
/// pacer needed.
///
/// It is a receiver for one publisher, not an RTSP server — an
/// `RtspServer`, with the `rtsp-server` feature, is:
///
/// - **One publisher, once.** It waits for a single connection, and closes
///   the listening socket as soon as that one is accepted, so a second
///   publisher is refused while the first is received. When the publisher
///   is gone the source ends; to take the next one, call `listen` again —
///   the port is closed until then.
/// - **One port each.** Two listening on the same port are a mistake: the
///   second cannot bind it.
/// - **No authentication**, and the URL's path is not enforced: any
///   publisher that reaches the port is received.
/// - **No viewers.** It answers `OPTIONS`, `ANNOUNCE`, `SETUP`, `RECORD`,
///   `GET_PARAMETER`, `SET_PARAMETER` and `TEARDOWN`, never `DESCRIBE` or
///   `PLAY`.
/// - **The codecs it receives**, all of them: an announcement of any other
///   is refused (`415`), as [`RtspSourceError::Unsupported`].
///
/// # How a received session ends
///
/// The publisher's `TEARDOWN`, or its closing the connection, ends the
/// stream: each output ends after what was read before it. That holds
/// when the close arrives as a reset, as a publisher's that has gone
/// before the answer to its `TEARDOWN` reached it does. A publisher that
/// stops sending without closing — a pulled cable, a host gone — is
/// [`RtspSourceError::PublisherSilent`] on the bus after
/// [`RtspListenOptions::timeout`]. As with `open`, nothing reconnects: a
/// new session is a new `listen`.
pub struct RtspSource(SourceStage<Reading>);

source_stage!(RtspSource);

/// How long the parameters a description leaves out are waited for, beyond
/// the timeout: a stream that begins between keyframes is read until its
/// first.
const PROBE_SPAN: Duration = Duration::from_secs(5);

/// What an [`RtspSource`] hands on, a packet at a time as the session
/// delivers it: all of its work, which the framework makes the source.
struct Reading {
    pp_log: PpLog,
    /// What each stream's output declares, by stream index.
    contracts: Vec<OutputContract>,
    streams: Vec<StreamInfo>,
    rx: crossbeam_channel::Receiver<Item>,
    /// What the session's threads ring with each item they hand on.
    bell: Bell,
    /// What came while the streams were probed, handed on first.
    held: VecDeque<Item>,
    reader: Reader,
}

impl RtspSource {
    /// Connects to `url` (e.g. `rtsp://host:port/path`, with
    /// `user:password@` before the host where the server asks for
    /// credentials) and returns the element alongside every stream the
    /// server advertised that it can receive, so the caller can inspect
    /// them before deciding which of `src_pads()` to link — same pattern as
    /// `FileDemuxer::open`.
    ///
    /// Blocks while the session is set up — each request given
    /// [`RtspOptions::timeout`] for its answer — and while what the
    /// description leaves out is read from the media (see "What it
    /// receives").
    pub fn open(
        name: impl Into<String>,
        url: impl AsRef<str>,
        options: RtspOptions,
    ) -> std::result::Result<(Self, Vec<StreamInfo>), RtspSourceError> {
        crate::ensure_ffmpeg();
        let url = url.as_ref();
        let identity = Identity::new(ElementType::RtspSource, name);
        let pp_log = identity.pp_log().clone();
        let (session, refused) = pull::connect(url, options, &pp_log).inspect_err(|error| {
            pp_error!(pp_log: &pp_log, "opening {} failed: {error}", redact(url));
        })?;
        for what in &refused {
            pp_warn!(pp_log: &pp_log, "left out a track in {what}");
        }
        let (source, streams) = Self::reading(
            identity,
            session,
            Silence {
                after: options.timeout,
                error: RtspSourceError::ServerSilent,
            },
            PROBE_SPAN.saturating_add(options.timeout),
        )?;
        pp_info!(
            pp_log: &pp_log,
            "opened: url={}, transport={:?}, {} stream(s)",
            // A camera's address often carries its password.
            redact(url),
            options.transport,
            streams.len()
        );
        Ok((source, streams))
    }

    /// Listens at `url` — `rtsp://host:port/path`, where `host` is the
    /// address to bind (`0.0.0.0` for every interface, `127.0.0.1` for this
    /// machine alone) — for one publisher to push a stream into, and
    /// returns the element alongside every stream that publisher announced
    /// and set up, as [`RtspSource::open`] does for a server's.
    ///
    /// Blocks until the publisher has connected, announced its streams and
    /// started recording, and what its announcement leaves out has been
    /// read from its first keyframes — or until [`RtspListenOptions::wait`]
    /// runs out, which is [`RtspSourceError::NoPublisher`]. Either way
    /// nothing is left listening once it returns an error: the port can be
    /// listened on again at once. Run it on a thread of its own, or with a
    /// short `wait` in a loop, where the caller has other things to answer
    /// meanwhile.
    ///
    /// The publisher picks the transport in its `SETUP`, and either works:
    /// RTP interleaved in the RTSP connection (TCP), or RTP over UDP, to a
    /// pair of ports opened for each stream. See [`RtspSource`]'s
    /// "Receiving a push" for what this listener is not, and "How a
    /// received session ends".
    pub fn listen(
        name: impl Into<String>,
        url: impl AsRef<str>,
        options: RtspListenOptions,
    ) -> std::result::Result<(Self, Vec<StreamInfo>), RtspSourceError> {
        crate::ensure_ffmpeg();
        let url = url.as_ref();
        if !url.starts_with("rtsp://") {
            return Err(RtspSourceError::InvalidListenUrl);
        }
        let identity = Identity::new(ElementType::RtspSource, name);
        let pp_log = identity.pp_log().clone();
        pp_info!(
            pp_log: &pp_log,
            "listening: url={}, waiting up to {:?} for a publisher",
            redact(url),
            options.wait
        );
        let session = push::accept(url, options, &pp_log).inspect_err(|error| {
            pp_error!(pp_log: &pp_log, "no publisher received: {error}");
        })?;
        let (source, streams) = Self::reading(
            identity,
            session,
            Silence {
                after: options.timeout,
                error: RtspSourceError::PublisherSilent,
            },
            PROBE_SPAN.saturating_add(options.timeout),
        )?;
        pp_info!(
            pp_log: &pp_log,
            "publisher received: url={}, {} stream(s)",
            redact(url),
            streams.len()
        );
        Ok((source, streams))
    }

    /// The element reading `session`, and its streams: described from
    /// what the description says, and what it leaves out read from the
    /// media within `probe`.
    fn reading(
        identity: Identity,
        session: Session,
        silence: Silence,
        probe: Duration,
    ) -> std::result::Result<(Self, Vec<StreamInfo>), RtspSourceError> {
        let pp_log = identity.pp_log().clone();
        let mut described: Vec<Described> = session.tracks.iter().map(Described::new).collect();
        let (items, rx) = crossbeam_channel::bounded(BACKLOG);
        let bell = Bell::new();
        let outbox = Outbox {
            items: std::mem::ManuallyDrop::new(items),
            bell: bell.clone(),
        };
        let reader = Reader::start(session, outbox, silence, &pp_log)?;

        // What the description leaves out, read from the first keyframes.
        let mut held = VecDeque::new();
        let deadline = Instant::now().checked_add(probe);
        while !described.iter().all(|track| track.complete) {
            let left = deadline.map_or(Duration::MAX, |deadline| {
                deadline.saturating_duration_since(Instant::now())
            });
            let item = match rx.recv_timeout(left) {
                Ok(item) => item,
                Err(_) => {
                    pp_warn!(
                        pp_log: &pp_log,
                        "no keyframe said every picture's size within {probe:?}"
                    );
                    break;
                }
            };
            match item {
                Item::Packet(index, ref packet) => {
                    if let Some(track) = described.get_mut(index)
                        && !track.complete
                        && packet.is_key()
                    {
                        track.complete_from(packet);
                    }
                    held.push_back(item);
                }
                Item::End => {
                    held.push_back(item);
                    break;
                }
                Item::Failed(error) => return Err(error),
            }
        }

        let streams: Vec<StreamInfo> = described
            .into_iter()
            .enumerate()
            .map(|(index, track)| track.stream().with_index(index))
            .collect();
        let contracts = contracts(&streams);
        Ok((
            Self(SourceStage::new(
                identity,
                Reading {
                    pp_log,
                    contracts,
                    streams: streams.clone(),
                    rx,
                    bell,
                    held,
                    reader,
                },
            )),
            streams,
        ))
    }

    /// The source of a publisher an `RtspServer` took, whose `streams` its
    /// connection has described, numbered: what the connection hands on
    /// through the outbox returned, read as a pulled session is, and ended
    /// as the outbox goes.
    #[cfg(feature = "rtsp-server")]
    pub(crate) fn published(name: &str, streams: Vec<StreamInfo>) -> (Self, Outbox) {
        let identity = Identity::new(ElementType::RtspSource, name);
        let pp_log = identity.pp_log().clone();
        let (items, rx) = crossbeam_channel::bounded(BACKLOG);
        let bell = Bell::new();
        let outbox = Outbox {
            items: std::mem::ManuallyDrop::new(items),
            bell: bell.clone(),
        };
        let source = Self(SourceStage::new(
            identity,
            Reading {
                pp_log,
                contracts: contracts(&streams),
                streams,
                rx,
                bell,
                held: VecDeque::new(),
                reader: Reader::idle(),
            },
        ));
        (source, outbox)
    }

    /// The first stream of `kind`, as everything a branch for it is built
    /// from — or a [`RtspSourceError::NoStream`] naming the kind the
    /// session lacks. See
    /// [`FileDemuxer::best`](crate::elements::FileDemuxer::best).
    pub fn best(
        &self,
        kind: ffmpeg::media::Type,
    ) -> std::result::Result<StreamInfo, RtspSourceError> {
        self.0
            .inner()
            .streams
            .iter()
            .find(|stream| stream.kind() == kind)
            .cloned()
            .ok_or(RtspSourceError::NoStream(kind))
    }
}

/// What each of `streams`' outputs declares, from the medium the session
/// announced: every output hands on `MediaBuffer::Packet`, so only this
/// tells an audio stream apart from a video one.
fn contracts(streams: &[StreamInfo]) -> Vec<OutputContract> {
    streams
        .iter()
        .map(|s| match MediaKind::packet_for(s.kind()) {
            Some(kind) => OutputContract::Fixed(PortContract::packet(kind)),
            None => OutputContract::Unknown,
        })
        .collect()
}

/// A track's description as it is completed: what the session said of it,
/// and what its parameter sets say — in the description, or else in band
/// at its first keyframe.
struct Described {
    announced: Announced,
    parameters: ffmpeg::codec::Parameters,
    frame_rate: Option<ffmpeg::Rational>,
    complete: bool,
}

impl Described {
    fn new(track: &receive::Track) -> Self {
        let announced = track.announced.clone();
        let mut parameters = announced.parameters();
        let video = announced.medium == ffmpeg::media::Type::Video;
        let mut described = Self {
            frame_rate: None,
            complete: !video,
            announced,
            parameters: ffmpeg::codec::Parameters::new(),
        };
        if video {
            let sets = described.announced.extradata.clone();
            described.complete = described.read_sets(&mut parameters, &sets);
        }
        described.parameters = parameters;
        described
    }

    /// What a keyframe of the track says, where the description said too
    /// little: its parameter sets, its size.
    fn complete_from(&mut self, key: &ffmpeg::Packet) {
        let mut parameters = self.parameters.clone();
        if !self.announced.complete(&mut parameters, key) {
            return;
        }
        if let Some(unit) = key.data() {
            let unit = unit.to_vec();
            self.read_sets(&mut parameters, &unit);
        }
        self.parameters = parameters;
        self.complete = true;
    }

    /// Puts what the sequence parameter set in `sets` — Annex B — says into
    /// `parameters`: the size, how far pictures are reordered, the colour
    /// and the rate where it says them. Whether it said the size.
    fn read_sets(&mut self, parameters: &mut ffmpeg::codec::Parameters, sets: &[u8]) -> bool {
        // SAFETY: plain fields of parameters this owns alone.
        let raw = unsafe { &mut *parameters.as_mut_ptr() };
        for nal in annex_b(sets) {
            match self.announced.codec {
                ffmpeg::codec::Id::H264 if nal.first().map(|b| b & 0x1f) == Some(7) => {
                    let Some(sequence) = h264_sequence(nal) else {
                        continue;
                    };
                    raw.width = sequence.shape.width as i32;
                    raw.height = sequence.shape.height as i32;
                    raw.video_delay = sequence.reorder as i32;
                    if let Some((primaries, transfer, matrix, full)) = sequence.colour {
                        colour(raw, primaries, transfer, matrix, full);
                    }
                    if let Some((units, scale)) = sequence.timing {
                        self.frame_rate = Some(ffmpeg::Rational::new(
                            i32::try_from(scale).unwrap_or(i32::MAX),
                            i32::try_from(2 * u64::from(units)).unwrap_or(i32::MAX),
                        ));
                    }
                    return true;
                }
                ffmpeg::codec::Id::HEVC if nal.first().map(|b| (b >> 1) & 0x3f) == Some(33) => {
                    let Some(sequence) = hevc_sequence(nal) else {
                        continue;
                    };
                    raw.width = sequence.shape.width as i32;
                    raw.height = sequence.shape.height as i32;
                    raw.video_delay = sequence.reorder as i32;
                    return true;
                }
                _ => {}
            }
        }
        false
    }

    fn stream(self) -> StreamInfo {
        let stream = StreamInfo::new(self.parameters, self.announced.time_base());
        match self.frame_rate {
            Some(rate) => stream.with_frame_rate(rate),
            None => stream,
        }
    }
}

/// Puts an H.264 VUI's colour description into `raw`, as far as FFmpeg
/// names its values.
fn colour(
    raw: &mut ffmpeg::ffi::AVCodecParameters,
    primaries: u8,
    transfer: u8,
    matrix: u8,
    full: bool,
) {
    use ffmpeg::color::{Primaries, Range, Space, TransferCharacteristic};
    let primaries = match primaries {
        1 => Some(Primaries::BT709),
        4 => Some(Primaries::BT470M),
        5 => Some(Primaries::BT470BG),
        6 => Some(Primaries::SMPTE170M),
        7 => Some(Primaries::SMPTE240M),
        8 => Some(Primaries::Film),
        9 => Some(Primaries::BT2020),
        10 => Some(Primaries::SMPTE428),
        11 => Some(Primaries::SMPTE431),
        12 => Some(Primaries::SMPTE432),
        22 => Some(Primaries::JEDEC_P22),
        _ => None,
    };
    let transfer = match transfer {
        1 => Some(TransferCharacteristic::BT709),
        4 => Some(TransferCharacteristic::GAMMA22),
        5 => Some(TransferCharacteristic::GAMMA28),
        6 => Some(TransferCharacteristic::SMPTE170M),
        7 => Some(TransferCharacteristic::SMPTE240M),
        8 => Some(TransferCharacteristic::Linear),
        9 => Some(TransferCharacteristic::Log),
        10 => Some(TransferCharacteristic::LogSqrt),
        11 => Some(TransferCharacteristic::IEC61966_2_4),
        12 => Some(TransferCharacteristic::BT1361_ECG),
        13 => Some(TransferCharacteristic::IEC61966_2_1),
        14 => Some(TransferCharacteristic::BT2020_10),
        15 => Some(TransferCharacteristic::BT2020_12),
        16 => Some(TransferCharacteristic::SMPTE2084),
        17 => Some(TransferCharacteristic::SMPTE428),
        18 => Some(TransferCharacteristic::ARIB_STD_B67),
        _ => None,
    };
    let space = match matrix {
        0 => Some(Space::RGB),
        1 => Some(Space::BT709),
        4 => Some(Space::FCC),
        5 => Some(Space::BT470BG),
        6 => Some(Space::SMPTE170M),
        7 => Some(Space::SMPTE240M),
        8 => Some(Space::YCGCO),
        9 => Some(Space::BT2020NCL),
        10 => Some(Space::BT2020CL),
        11 => Some(Space::SMPTE2085),
        12 => Some(Space::ChromaDerivedNCL),
        13 => Some(Space::ChromaDerivedCL),
        14 => Some(Space::ICTCP),
        _ => None,
    };
    if let Some(primaries) = primaries {
        raw.color_primaries = primaries.into();
    }
    if let Some(transfer) = transfer {
        raw.color_trc = transfer.into();
    }
    if let Some(space) = space {
        raw.color_space = space.into();
    }
    raw.color_range = if full { Range::JPEG } else { Range::MPEG }.into();
}

impl Source for Reading {
    fn is_live(&self) -> bool {
        true
    }

    /// One per stream the session delivers, at the stream's own index.
    fn outputs(&self) -> Option<Vec<SrcPad>> {
        Some(
            self.contracts
                .iter()
                .enumerate()
                .map(|(index, contract)| SrcPad::with_contract(format!("src_{index}"), *contract))
                .collect(),
        )
    }

    /// The next packet the session delivers, on its stream's output — what
    /// came while the streams were probed first.
    ///
    /// Waits through `wait`, so a pause or a stop is answered at once; how
    /// long the session may deliver nothing is the reading threads' to
    /// tell, as [`RtspSourceError::ServerSilent`] or
    /// [`RtspSourceError::PublisherSilent`].
    fn produce(&mut self, wait: &mut Wait<'_>) -> Result<Produced> {
        let item = match self.held.pop_front() {
            Some(item) => item,
            None => match wait.recv(&self.rx, &self.bell) {
                Received::Got(item) => item,
                Received::LetGo => return Ok(Produced::Nothing),
                Received::Gone => return Ok(Produced::End),
            },
        };
        match item {
            Item::Packet(index, packet) => Ok(Produced::On(
                index,
                MediaBuffer::Packet(Arc::new(packet).into()),
            )),
            Item::End => {
                pp_info!(self, "the session ended");
                Ok(Produced::End)
            }
            Item::Failed(error) => {
                pp_error!(self, "the session failed: {error}");
                Err(error.into())
            }
        }
    }

    /// Tears the session down where this source asked for it, and waits for
    /// its threads.
    fn stopping(&mut self) {
        self.reader.stop();
    }

    fn attached(&mut self, cx: &Attached<'_>) {
        self.pp_log = cx.pp_log().clone();
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;

    /// `RtspOptions::timeout`'s whole reason for existing: without it
    /// a connection to a server that never answers waits for the OS's own
    /// connect timeout, far longer. `192.0.2.1` is RFC 5737 TEST-NET-1,
    /// reserved for documentation and guaranteed not to be routed, so this
    /// exercises the "no answer" path rather than a fast connection refusal.
    ///
    /// Only the upper bound is asserted: a network that replies with an ICMP
    /// unreachable makes this fail even sooner, which is equally correct. The
    /// bound sits well under the OS-level TCP connect timeout (~21s on
    /// Windows, far longer on Linux), so a regression that stops passing the
    /// option through is what actually trips it.
    #[test]
    fn open_gives_up_within_the_configured_timeout_instead_of_hanging() {
        let options = RtspOptions {
            timeout: Duration::from_millis(500),
            ..Default::default()
        };

        let started = Instant::now();
        let result = RtspSource::open("rtsp", "rtsp://192.0.2.1:554/none", options);
        let elapsed = started.elapsed();

        assert!(
            result.is_err(),
            "opening an unroutable address must not succeed"
        );
        assert!(
            elapsed < Duration::from_secs(10),
            "open took {elapsed:?} — the configured timeout is not bounding the connection"
        );
    }

    /// A caller that never touches `RtspOptions` still has to get a bounded
    /// `open`, since the default this type supplies is the only thing
    /// standing between them and the OS's own.
    #[test]
    fn the_default_options_still_bound_the_connection() {
        let options = RtspOptions::default();

        assert_eq!(options.transport, crate::elements::RtspTransport::Tcp);
        assert!(
            options.timeout > Duration::ZERO,
            "the default timeout must not be unbounded"
        );
    }

    /// Receiving a push, against this crate's own publisher: `RtspMuxer`
    /// pushing a paced file into `RtspSource::listen` on this machine, so
    /// that no server is needed.
    mod listen {
        use std::sync::{
            Mutex, OnceLock,
            atomic::{AtomicUsize, Ordering},
        };

        use super::*;
        use crate::{
            bus::BusEvent,
            elements::{AppSink, FileDemuxer, Pacer, RtspMuxer, RtspTransport, StreamInfo},
            pipeline::Pipeline,
            stream::StreamEvent,
        };

        /// Three seconds of picture and sound, made once for every test
        /// here: long enough for timestamps to advance well past a probe.
        fn fixture() -> &'static str {
            static FIXTURE: OnceLock<String> = OnceLock::new();
            FIXTURE.get_or_init(|| {
                crate::test_support::synthesize("rtsp-listen", 3.0, 48_000)
                    .path
                    .to_string_lossy()
                    .into_owned()
            })
        }

        /// A port nothing on this machine is using as this asks — the OS's
        /// choice, let go at once for the listener to take.
        fn free_port() -> u16 {
            std::net::TcpListener::bind("127.0.0.1:0")
                .and_then(|listener| listener.local_addr())
                .expect("an ephemeral port")
                .port()
        }

        fn url(port: u16) -> String {
            format!("rtsp://127.0.0.1:{port}/live")
        }

        /// `RtspSource::listen` on a thread of its own, since it blocks
        /// until a publisher arrives.
        fn listen_in_background(
            port: u16,
            options: RtspListenOptions,
        ) -> std::thread::JoinHandle<
            std::result::Result<(RtspSource, Vec<StreamInfo>), RtspSourceError>,
        > {
            let url = url(port);
            std::thread::spawn(move || RtspSource::listen("ingest", url, options))
        }

        /// The fixture's picture and sound published to `port` over
        /// `transport`, paced, running — and what it published. Retried
        /// until the listener, started on another thread just before, has
        /// bound the port: a connection it refuses is answered at once on
        /// loopback, and reaches no listener to disturb.
        fn publish(port: u16, transport: RtspTransport) -> (Arc<Pipeline>, Vec<StreamInfo>) {
            let deadline = Instant::now() + Duration::from_secs(15);
            loop {
                match try_publish(port, transport) {
                    Ok(published) => return published,
                    Err(error) => {
                        assert!(
                            Instant::now() < deadline,
                            "could not publish to the listener: {error}"
                        );
                        std::thread::sleep(Duration::from_millis(50));
                    }
                }
            }
        }

        fn try_publish(
            port: u16,
            transport: RtspTransport,
        ) -> crate::Result<(Arc<Pipeline>, Vec<StreamInfo>)> {
            let (source, _) = FileDemuxer::open("file", fixture())?;
            let video = source.best(ffmpeg::media::Type::Video)?;
            let audio = source.best(ffmpeg::media::Type::Audio)?;
            let published = vec![video.clone(), audio.clone()];
            let (pipeline, ()) = Pipeline::new("publisher", source, |source, ctx| {
                let options = RtspOptions {
                    transport,
                    timeout: Duration::from_secs(2),
                };
                let mut muxer = RtspMuxer::create(url(port), options)?;
                let tracks = [
                    muxer.add_stream("video", StreamInfo::from(&video))?,
                    muxer.add_stream("audio", StreamInfo::from(&audio))?,
                ];
                let mut sinks = muxer.open()?;
                for (stream, track) in [&video, &audio].into_iter().zip(tracks) {
                    let branch = ctx
                        .branch()
                        .queue(format!("queue-{}", stream.index()), 32)
                        .pipe(Pacer::new(format!("pacer-{}", stream.index())))
                        .to(sinks.take(track)?)?;
                    ctx.attach(source, stream.index(), branch)?;
                }
                Ok(())
            })?;
            pipeline.run()?;
            Ok((pipeline, published))
        }

        /// Each packet's `pts` and time base, as one output was handed them.
        type Stamps = Vec<(Option<i64>, ffmpeg::Rational, i64)>;

        /// What each output of a received session was handed.
        #[derive(Default)]
        struct Received {
            /// By output.
            packets: Vec<Mutex<Stamps>>,
            ends: AtomicUsize,
        }

        impl Received {
            fn count(&self, output: usize) -> usize {
                self.packets[output].lock().unwrap().len()
            }
        }

        fn receive(source: RtspSource, outputs: usize) -> (Arc<Pipeline>, Arc<Received>) {
            let received = Arc::new(Received {
                packets: (0..outputs).map(|_| Mutex::default()).collect(),
                ends: AtomicUsize::new(0),
            });
            let (pipeline, ()) = Pipeline::new("receiver", source, |source, ctx| {
                for output in 0..outputs {
                    let sink = AppSink::with_events(
                        format!("output-{output}"),
                        {
                            let received = Arc::clone(&received);
                            move |buffer| {
                                if let MediaBuffer::Packet(packet) = buffer {
                                    received.packets[output].lock().unwrap().push((
                                        packet.pts(),
                                        packet.time_base(),
                                        packet.duration(),
                                    ));
                                }
                                Ok(())
                            }
                        },
                        {
                            let received = Arc::clone(&received);
                            move |event: &StreamEvent| {
                                if let StreamEvent::Eos = event {
                                    received.ends.fetch_add(1, Ordering::SeqCst);
                                }
                                Ok(())
                            }
                        },
                    );
                    let branch = ctx.branch().to(sink)?;
                    ctx.attach(source, output, branch)?;
                }
                Ok(())
            })
            .expect("wire every received stream");
            pipeline.run().expect("run the receiver");
            (pipeline, received)
        }

        /// Waits up to `within` for `done`, failing with `what` past it.
        fn wait_for(within: Duration, what: &str, mut done: impl FnMut() -> bool) {
            let deadline = Instant::now() + within;
            while !done() {
                assert!(Instant::now() < deadline, "{what}");
                std::thread::sleep(Duration::from_millis(10));
            }
        }

        /// The whole round, over `transport`: both tracks arrive described
        /// as they were published, packets come out on both outputs with
        /// advancing timestamps, and the publisher finishing ends both
        /// outputs — with no error on the bus, whatever the publisher's
        /// close arrived as.
        fn round_trip(transport: RtspTransport) {
            let port = free_port();
            let listener = listen_in_background(
                port,
                RtspListenOptions {
                    wait: Duration::from_secs(20),
                    timeout: Duration::from_secs(5),
                },
            );
            let (publisher, published) = publish(port, transport);
            let (source, streams) = listener
                .join()
                .expect("the listening thread")
                .expect("the publisher is received");

            assert_eq!(streams.len(), 2, "both published tracks are announced");
            for sent in &published {
                let got = streams
                    .iter()
                    .find(|stream| stream.kind() == sent.kind())
                    .unwrap_or_else(|| panic!("no {:?} stream received", sent.kind()));
                assert_eq!(
                    got.parameters().id(),
                    sent.parameters().id(),
                    "{:?}",
                    sent.kind()
                );
                // SAFETY: parameters both sides own, read only.
                let (sent_raw, got_raw) =
                    unsafe { (&*sent.parameters().as_ptr(), &*got.parameters().as_ptr()) };
                match sent.kind() {
                    ffmpeg::media::Type::Video => assert_eq!(
                        (got_raw.width, got_raw.height),
                        (sent_raw.width, sent_raw.height),
                        "the picture's size"
                    ),
                    _ => assert_eq!(
                        (got_raw.sample_rate, got_raw.ch_layout.nb_channels),
                        (sent_raw.sample_rate, sent_raw.ch_layout.nb_channels),
                        "the sound's rate and channels"
                    ),
                }
            }

            let (receiver, received) = receive(source, streams.len());
            wait_for(
                Duration::from_secs(30),
                "the publisher did not finish",
                || {
                    matches!(
                        publisher.bus().try_recv(),
                        Some(BusEvent::Finished) | Some(BusEvent::Error { .. })
                    )
                },
            );
            publisher.stop();
            wait_for(
                Duration::from_secs(20),
                "the publisher's finish did not end every received output",
                || received.ends.load(Ordering::SeqCst) == streams.len(),
            );
            receiver.stop();
            while let Some(event) = receiver.bus().try_recv() {
                assert!(
                    !matches!(event, BusEvent::Error { .. }),
                    "a publisher that finished is not a failure: {event}"
                );
            }

            for stream in &streams {
                let packets = received.packets[stream.index()].lock().unwrap();
                assert!(
                    packets.len() >= 10,
                    "{:?}: {} packets received",
                    stream.kind(),
                    packets.len()
                );
                assert!(
                    packets
                        .iter()
                        .all(|&(_, time_base, _)| time_base == stream.time_base()),
                    "{:?}'s packets carry its time base",
                    stream.kind()
                );
                // A muxer cutting segments needs each one's duration, which
                // RTP does not carry for a picture.
                assert!(
                    packets.iter().all(|&(_, _, duration)| duration > 0),
                    "{:?}'s packets each last a while",
                    stream.kind()
                );
                let pts: Vec<i64> = packets.iter().filter_map(|&(pts, _, _)| pts).collect();
                let (first, last) = (pts.iter().min(), pts.iter().max());
                let span = f64::from(stream.time_base())
                    * (last.copied().unwrap_or(0) - first.copied().unwrap_or(0)) as f64;
                assert!(
                    span >= 1.0,
                    "{:?}'s timestamps advance across what was sent: {span:.2}s",
                    stream.kind()
                );
            }
        }

        #[test]
        fn a_publisher_over_tcp_is_received_and_its_finish_ends_the_stream() {
            round_trip(RtspTransport::Tcp);
        }

        #[test]
        fn a_publisher_over_udp_is_received_and_its_finish_ends_the_stream() {
            round_trip(RtspTransport::Udp);
        }

        /// Nobody publishing is a typed error within the wait, and nothing
        /// is left holding the port: the OS lets it be bound by a socket
        /// that shares it with nobody, and a second listen binds it again —
        /// to give up the same way, which a port still taken would not.
        #[test]
        fn a_listen_nobody_publishes_to_gives_up_in_time_and_frees_its_port() {
            let port = free_port();
            let wait = Duration::from_millis(300);
            let options = RtspListenOptions {
                wait,
                ..Default::default()
            };

            let started = Instant::now();
            let result = RtspSource::listen("ingest", url(port), options);
            let took = started.elapsed();
            assert!(
                matches!(result, Err(RtspSourceError::NoPublisher(waited)) if waited == wait),
                "{:?}",
                result.err()
            );
            assert!(took >= wait, "gave up after {took:?}, before the wait");
            assert!(took < Duration::from_secs(5), "gave up only after {took:?}");

            drop(std::net::TcpListener::bind(("127.0.0.1", port)).expect("the port is free again"));
            assert!(matches!(
                RtspSource::listen("ingest", url(port), options),
                Err(RtspSourceError::NoPublisher(_))
            ));
        }

        /// A publisher that stops sending without closing — here, paused —
        /// is an error on the bus once the timeout has passed, rather than
        /// a read that waits for ever.
        #[test]
        fn a_publisher_gone_silent_is_an_error_after_the_timeout() {
            let port = free_port();
            let timeout = Duration::from_secs(2);
            let listener = listen_in_background(
                port,
                RtspListenOptions {
                    wait: Duration::from_secs(20),
                    timeout,
                },
            );
            let (publisher, _) = publish(port, RtspTransport::Tcp);
            let (source, streams) = listener
                .join()
                .expect("the listening thread")
                .expect("the publisher is received");
            let (receiver, received) = receive(source, streams.len());
            wait_for(Duration::from_secs(10), "nothing was received", || {
                received.count(0) > 0
            });

            publisher.pause();
            let silent_from = Instant::now();
            let error = loop {
                match receiver.bus().recv_timeout(Duration::from_secs(20)) {
                    Ok(BusEvent::Error { error, .. }) => break error,
                    Ok(_) => continue,
                    Err(_) => panic!("a silent publisher held the source"),
                }
            };
            let took = silent_from.elapsed();
            assert!(
                matches!(
                    error,
                    crate::Error::RtspSourceError(RtspSourceError::PublisherSilent(t)) if t == timeout
                ),
                "{error}"
            );
            assert!(took < timeout + Duration::from_secs(5), "took {took:?}");
            receiver.stop();
            publisher.stop();
        }

        #[test]
        fn listen_refuses_what_is_not_an_rtsp_url() {
            for url in ["http://127.0.0.1:8554/live", "rtsp://127.0.0.1:8554/a\0b"] {
                assert!(
                    matches!(
                        RtspSource::listen("ingest", url, RtspListenOptions::default()),
                        Err(RtspSourceError::InvalidListenUrl)
                    ),
                    "{url:?}"
                );
            }
        }
    }
}
