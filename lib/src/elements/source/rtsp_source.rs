use std::{
    cell::Cell,
    ffi::{CString, c_int, c_void},
    sync::Arc,
    time::{Duration, Instant},
};

use crate::pp_log::{PpLog, pp_error, pp_info};
use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    bus::Bus,
    contract::{MediaKind, OutputContract, PortContract},
    element::{Attached, ElementType, Identity, Produced, Source, SourceStage, Wait},
    elements::{
        RtspOptions,
        parameter_sets::{OutputWatches, report_change},
        rtsp::redact,
    },
    error::Result,
    pad::SrcPad,
    produce::source_stage,
};

use crate::elements::StreamInfo;

/// Why an [`RtspSource`] could not connect or read.
///
/// Converts into the crate-wide `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum RtspSourceError {
    /// FFmpeg rejected connection setup, stream reading, or shutdown.
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),
    /// The session has no stream of the kind asked for — see
    /// [`RtspSource::best`].
    #[error("the RTSP session has no {0:?} stream")]
    NoStream(ffmpeg::media::Type),
    /// [`RtspSource::listen`] was given something other than an
    /// `rtsp://host:port/path` URL, or one holding a NUL byte.
    #[error("RtspSource::listen needs an rtsp://host:port/path URL")]
    InvalidListenUrl,
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

/// How [`RtspSource::listen`] waits for a publisher, and how long one may
/// then go silent.
///
/// Not [`RtspOptions`]: there is no transport to choose, since the
/// publisher's `SETUP` picks one — RTP interleaved in the RTSP connection
/// (TCP) or RTP over UDP — and FFmpeg's listener takes either. Both bounds
/// are kept by this crate rather than by FFmpeg's own `listen_timeout` and
/// `timeout`: those cover only the accept and, for TCP, not the reads at
/// all, so a publisher that connected and then stopped sending would hold
/// the source for good.
#[derive(Debug, Clone, Copy)]
pub struct RtspListenOptions {
    /// How long [`RtspSource::listen`] waits for a publisher to connect and
    /// finish its `ANNOUNCE`/`SETUP`/`RECORD` handshake. Past it, `listen`
    /// returns [`RtspSourceError::NoPublisher`] with the port closed again.
    /// A duration too long for the clock to add (`Duration::MAX`) waits
    /// without end, as asked.
    pub wait: Duration,
    /// How long a publisher may send nothing — between any two packets,
    /// and beyond the five seconds of its stream FFmpeg may read inside
    /// `listen` to probe it — before the session is given up as
    /// [`RtspSourceError::PublisherSilent`]:
    /// a pulled cable or a vanished host closes no connection, and without
    /// this a TCP read would wait on it for ever. It is also how long a
    /// `Stop` can take to reach a source whose publisher has gone quiet,
    /// since the source takes it between reads.
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
/// # Errors and reconnecting
///
/// Deliberately does **not** retry or reconnect internally: a read failure
/// (dropped connection, camera reboot, ...) ends this source's thread with
/// the error on the bus, the same way any other fatal source failure does,
/// instead of looping forever. Reconnecting means building a fresh
/// `RtspSource`/[`crate::pipeline::Pipeline`] — mirrors `Pipeline` itself
/// not being reusable once it ends: watch
/// [`crate::pipeline::Pipeline::bus`], and on error, call
/// [`RtspSource::open`] again.
///
/// # Receiving a push
///
/// [`RtspSource::listen`] turns the roles round: rather than asking a
/// server for a stream, it takes the port in its URL and waits for a
/// publisher — an encoder, a camera that pushes, another program's
/// [`RtspMuxer`](crate::elements::RtspMuxer), `ffmpeg -f rtsp` — to
/// `ANNOUNCE` its streams and `RECORD` into it. What it hands on after that
/// is the same as from [`RtspSource::open`]: one output per stream the
/// publisher announced, [`RtspSource::best`] to pick one, no pacer needed.
///
/// It is FFmpeg's own RTSP listener (`rtsp_flags=listen`), which is a
/// receiver for one publisher and not an RTSP server:
///
/// - **One publisher, once.** It waits for a single connection, and FFmpeg
///   closes the listening socket as soon as that one is accepted, so a
///   second publisher is refused while the first is received. When the
///   publisher is gone the source ends; to take the next one, call
///   `listen` again — the port is closed until then.
/// - **One port each.** Two listening on different ports are independent.
///   Two on the same port are a mistake: on Linux and macOS the second
///   cannot bind it, but on Windows FFmpeg's `SO_REUSEADDR` lets both bind,
///   and which one a publisher reaches is the OS's choice.
/// - **No authentication**, and the URL's path is not enforced: any
///   publisher that reaches the port is received.
/// - **No viewers.** It answers `ANNOUNCE`, `SETUP`, `RECORD`, `OPTIONS`
///   and `TEARDOWN`, never `DESCRIBE` or `PLAY`; to serve what it receives,
///   publish it on with an `RtspMuxer` to a server such as MediaMTX.
///
/// # How a received session ends
///
/// The publisher closing its connection ends the stream: each output ends
/// after what was read before it, as for a clean end of
/// [`RtspSource::open`]'s session. That holds whether it sent a `TEARDOWN`
/// first or not, and when the close arrives as a reset — FFmpeg's listener
/// does not tell these apart, and its reply to a `TEARDOWN` reaches a
/// publisher that has already closed, which then resets the connection
/// (measured on Windows over TCP, from FFmpeg's own publisher: every clean
/// finish read as `WSAECONNABORTED`). A publisher process killed outright
/// ends it the same way, since its OS closes the connection for it.
///
/// A publisher that stops sending without closing — a pulled cable, a host
/// gone — is [`RtspSourceError::PublisherSilent`] on the bus after
/// [`RtspListenOptions::timeout`], and anything else that fails a read is
/// an error as it is for `open`. As with `open`, nothing reconnects: a new
/// session is a new `listen`.
///
/// Uses `Packet::read` directly instead of `Input::packets()` — the
/// latter silently retries forever inside its own `next()` on any non-EOF
/// error (network timeout, connection reset, ...), which would make a
/// stuck connection un-`Stop`-able (the pipeline's requests never get a
/// turn) and this element's "fail fast, don't retry" contract impossible to
/// keep.
pub struct RtspSource(SourceStage<Reading>);

source_stage!(RtspSource);

/// What an [`RtspSource`] hands on, a packet at a time as the session
/// delivers it: all of its work, which the framework makes the source.
struct Reading {
    pp_log: PpLog,
    input: ffmpeg::format::context::Input,
    /// What each stream's output declares, by stream index.
    contracts: Vec<OutputContract>,
    /// Whether each stream still carries the pictures it was described
    /// with — see [`crate::bus::BusEvent::FormatChanged`].
    watches: OutputWatches,
    /// Where a stream that changed is reported, once the pipeline has
    /// wired this.
    bus: Option<Bus>,
    /// For a session [`RtspSource::listen`] received, what bounds each
    /// read. Declared after `input` on purpose: FFmpeg holds a pointer to
    /// its interrupt until the input is closed, and fields drop in the
    /// order they are declared.
    listening: Option<Listening>,
}

/// The bound on a received session's reads: the interrupt FFmpeg asks,
/// armed with `timeout` before each one.
struct Listening {
    /// Boxed so that its address, which FFmpeg holds, stays put when the
    /// source moves to its thread.
    interrupt: Box<Interrupt>,
    timeout: Duration,
}

/// What a listening input's interrupt callback answers from: whether the
/// time given for the call under way has run out.
///
/// FFmpeg asks it from inside the blocking calls — every 100 ms while it
/// waits to accept, and while a read waits on the socket — on whichever
/// thread made the call, which is the one thread that owns the input at a
/// time. So a `Cell` is enough, and nothing here can panic across the FFI.
#[derive(Default)]
struct Interrupt {
    deadline: Cell<Option<Instant>>,
    /// Set once the callback has answered "stop": what tells a call cut
    /// short apart from one that failed by itself, and — since FFmpeg
    /// cannot resume a read cut off mid-packet — that the session is over.
    fired: Cell<bool>,
}

impl Interrupt {
    /// Gives the next call `within`; `None` from a duration the clock
    /// cannot add means no limit.
    fn arm(&self, within: Duration) {
        self.deadline.set(Instant::now().checked_add(within));
        self.fired.set(false);
    }

    fn fired(&self) -> bool {
        self.fired.get()
    }
}

/// How much of a received stream FFmpeg reads at most to probe it — its
/// default analyze duration, which nothing here changes. Live media takes
/// as long to arrive as it lasts, so `listen` gives the probe this long
/// before [`RtspListenOptions::timeout`] starts to count: a stream that
/// waits seconds for its first keyframe is not a silent one.
const PROBE_SPAN: Duration = Duration::from_secs(5);

/// FFmpeg's `AVIOInterruptCB::callback` for a listening input.
extern "C" fn interrupted(opaque: *mut c_void) -> c_int {
    // SAFETY: `opaque` is the `Box<Interrupt>` this was registered with in
    // `open_listening`, which outlives the input FFmpeg calls this for (see
    // `Reading::listening`), and only shared references are made to it.
    let interrupt = unsafe { &*(opaque as *const Interrupt) };
    let expired = interrupt
        .deadline
        .get()
        .is_some_and(|deadline| Instant::now() >= deadline);
    if expired {
        interrupt.fired.set(true);
    }
    c_int::from(expired)
}

/// Binds `url`'s port as FFmpeg's RTSP listener and returns once a
/// publisher has sent `RECORD` — or once `interrupt` gives up on it.
///
/// Written out rather than taken from `ffmpeg::format`, whose
/// `input_with_interrupt` takes no options, and leaks its callback's state.
fn open_listening(
    url: &CString,
    interrupt: &Interrupt,
) -> std::result::Result<ffmpeg::format::context::Input, ffmpeg::Error> {
    let mut options = ffmpeg::Dictionary::new();
    options.set("rtsp_flags", "listen");
    // Nothing else: FFmpeg's `listen_timeout` (whole seconds) and `timeout`
    // are left to `interrupt`, which bounds the handshake and every read as
    // well. And no `rtsp_transport`, so that the publisher's choice is
    // taken — and so that the client's fallback from a silent UDP session
    // to TCP, which would send `PAUSE` to a publisher, can never run.

    // SAFETY: a plain allocation, checked before use.
    let mut context = unsafe { ffi::avformat_alloc_context() };
    if context.is_null() {
        return Err(ffmpeg::Error::Other {
            errno: ffmpeg::util::error::ENOMEM,
        });
    }
    // SAFETY: `context` is the live context just allocated. The opaque
    // pointer is `interrupt`, which the caller keeps alive and in place for
    // as long as the input it gets back — see `Reading::listening`.
    unsafe {
        (*context).interrupt_callback = ffi::AVIOInterruptCB {
            callback: Some(interrupted),
            opaque: interrupt as *const Interrupt as *mut c_void,
        };
    }
    // SAFETY: `disown` hands over the dictionary's own pointer, which
    // `avformat_open_input` takes and the free below reclaims.
    let mut options = unsafe { options.disown() };
    // SAFETY: `c"rtsp"` names FFmpeg's demuxer, whose static descriptor
    // this returns — forced rather than probed, so that the options above
    // reach the RTSP demuxer whatever the URL looks like to a prober.
    let demuxer = unsafe { ffi::av_find_input_format(c"rtsp".as_ptr()) };
    // SAFETY: `context` is the context allocated above, which FFmpeg frees
    // on failure; `url` lives for the whole call; `options` is the
    // dictionary freed below either way.
    let code =
        unsafe { ffi::avformat_open_input(&mut context, url.as_ptr(), demuxer, &mut options) };
    // SAFETY: what the call left of the dictionary, or the whole of it.
    unsafe { ffi::av_dict_free(&mut options) };
    if code < 0 {
        return Err(ffmpeg::Error::from(code));
    }
    // SAFETY: the call above opened `context`; this hands it to the wrapper
    // that closes it on drop.
    Ok(unsafe { ffmpeg::format::context::Input::wrap(context) })
}

/// `error` as the OS's own, where it is a received session's connection
/// closing under it — the reset a publisher's close can arrive as. See
/// [`RtspSource`]'s "How a received session ends" on why that is an end
/// and not a failure.
fn connection_closed(error: ffmpeg::Error) -> Option<std::io::Error> {
    match error {
        // FFmpeg's network errors are the OS's own codes, Winsock's on
        // Windows, which FFmpeg's own message does not name — so the OS is
        // what reads them.
        ffmpeg::Error::Other { errno } => {
            Some(std::io::Error::from_raw_os_error(errno)).filter(|os| {
                matches!(
                    os.kind(),
                    std::io::ErrorKind::ConnectionReset | std::io::ErrorKind::ConnectionAborted
                )
            })
        }
        _ => None,
    }
}

impl RtspSource {
    /// Connects to `url` (e.g. `rtsp://host:port/path`) and returns the
    /// element alongside every stream the server advertised, so the
    /// caller can inspect them before deciding which of `src_pads()` to
    /// link — same pattern as `FileDemuxer::open`.
    pub fn open(
        name: impl Into<String>,
        url: impl AsRef<str>,
        options: RtspOptions,
    ) -> std::result::Result<(Self, Vec<StreamInfo>), RtspSourceError> {
        crate::ensure_ffmpeg();
        let input = ffmpeg::format::input_with_dictionary(url.as_ref(), options.to_dictionary())?;
        let identity = Identity::new(ElementType::RtspSource, name);
        let pp_log = identity.pp_log().clone();
        let (source, streams) = Self::reading(identity, input, None);
        pp_info!(
            pp_log: &pp_log,
            "opened: url={}, transport={:?}, {} stream(s)",
            // A camera's address often carries its password.
            redact(url.as_ref()),
            options.transport,
            streams.len()
        );
        Ok((source, streams))
    }

    /// Listens at `url` — `rtsp://host:port/path`, where `host` is the
    /// address to bind (`0.0.0.0` for every interface, `127.0.0.1` for this
    /// machine alone) — for one publisher to push a stream into, and
    /// returns the element alongside every stream that publisher announced,
    /// as [`RtspSource::open`] does for a server's.
    ///
    /// Blocks until the publisher has connected, announced its streams and
    /// started recording, and its first packets have been probed — which
    /// is when its streams are known — or until
    /// [`RtspListenOptions::wait`] runs out, which is
    /// [`RtspSourceError::NoPublisher`]. Either way nothing is left
    /// listening once it returns an error: the port can be listened on
    /// again at once. The probe reads up to five seconds of the stream —
    /// usually far less, but a stream that begins between keyframes is
    /// probed until its first — and is given that long and
    /// [`RtspListenOptions::timeout`] beyond it (see
    /// [`RtspSourceError::PublisherSilent`]). Run it on a thread of its
    /// own, or with a short `wait` in a loop, where the caller has other
    /// things to answer meanwhile.
    ///
    /// The publisher picks the transport in its `SETUP`, and either works:
    /// RTP interleaved in the RTSP connection (TCP), or RTP over UDP, for
    /// which the listener opens a pair of ports per stream from 5000
    /// upwards. For UDP, `host` must be an address the publisher reaches
    /// this machine at, not `0.0.0.0`: FFmpeg's listener tells the
    /// publisher to send its RTP to the address it was bound to, and one
    /// told `0.0.0.0` cannot — measured on Windows, an FFmpeg publisher's
    /// writes failed (`WSAEADDRNOTAVAIL`) and the session ended with nothing
    /// received. TCP takes `0.0.0.0` as well as any. See [`RtspSource`]'s
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
        let c_url = CString::new(url).map_err(|_| RtspSourceError::InvalidListenUrl)?;
        let identity = Identity::new(ElementType::RtspSource, name);
        let pp_log = identity.pp_log().clone();
        pp_info!(
            pp_log: &pp_log,
            "listening: url={}, waiting up to {:?} for a publisher",
            redact(url),
            options.wait
        );

        // Declared before the input, so that it is dropped after it.
        let interrupt = Box::new(Interrupt::default());
        interrupt.arm(options.wait);
        let mut input = match open_listening(&c_url, &interrupt) {
            Ok(input) => input,
            Err(_) if interrupt.fired() => {
                pp_error!(pp_log: &pp_log, "no publisher within {:?}", options.wait);
                return Err(RtspSourceError::NoPublisher(options.wait));
            }
            Err(error) => {
                pp_error!(pp_log: &pp_log, "listening failed: {error}");
                return Err(error.into());
            }
        };

        // What the publisher's SDP leaves out — a picture's size, which
        // its first packets say — is probed from them, as `open`'s input
        // is. FFmpeg reads at most `PROBE_SPAN` of the stream for it, which
        // a live publisher sends in that much time: given `timeout` beyond
        // that, a probe still unfinished is a publisher not sending.
        interrupt.arm(PROBE_SPAN.saturating_add(options.timeout));
        // SAFETY: `input` owns the opened context this probes.
        let code =
            unsafe { ffi::avformat_find_stream_info(input.as_mut_ptr(), std::ptr::null_mut()) };
        if interrupt.fired() {
            pp_error!(
                pp_log: &pp_log,
                "the publisher sent nothing to probe for {:?}",
                options.timeout
            );
            return Err(RtspSourceError::PublisherSilent(options.timeout));
        }
        if code < 0 {
            let error = ffmpeg::Error::from(code);
            pp_error!(pp_log: &pp_log, "probing the publisher's streams failed: {error}");
            return Err(error.into());
        }

        let (source, streams) = Self::reading(
            identity,
            input,
            Some(Listening {
                interrupt,
                timeout: options.timeout,
            }),
        );
        pp_info!(
            pp_log: &pp_log,
            "publisher received: url={}, {} stream(s)",
            redact(url),
            streams.len()
        );
        Ok((source, streams))
    }

    /// The element over an opened `input`, and the streams it has.
    fn reading(
        identity: Identity,
        input: ffmpeg::format::context::Input,
        listening: Option<Listening>,
    ) -> (Self, Vec<StreamInfo>) {
        let streams: Vec<StreamInfo> = input.streams().map(|s| StreamInfo::of(&s)).collect();

        // Per stream, from the medium the session announced: every output
        // hands on `MediaBuffer::Packet`, so only this tells an audio
        // stream apart from a video one. A medium this crate does not model
        // declares nothing and is left to the runtime check.
        let contracts = streams
            .iter()
            .map(|s| match MediaKind::packet_for(s.kind()) {
                Some(kind) => OutputContract::Fixed(PortContract::packet(kind)),
                None => OutputContract::Unknown,
            })
            .collect();

        let pp_log = identity.pp_log().clone();
        (
            Self(SourceStage::new(
                identity,
                Reading {
                    pp_log,
                    input,
                    contracts,
                    watches: OutputWatches::new(&streams),
                    bus: None,
                    listening,
                },
            )),
            streams,
        )
    }

    /// The stream of `kind` FFmpeg judges the one to play, as everything a
    /// branch for it is built from — or a [`RtspSourceError::NoStream`]
    /// naming the kind the session lacks. See
    /// [`FileDemuxer::best`](crate::elements::FileDemuxer::best).
    pub fn best(
        &self,
        kind: ffmpeg::media::Type,
    ) -> std::result::Result<StreamInfo, RtspSourceError> {
        self.0
            .inner()
            .input
            .streams()
            .best(kind)
            .map(|stream| StreamInfo::of(&stream))
            .ok_or(RtspSourceError::NoStream(kind))
    }
}

impl Reading {
    fn stream(&self, index: usize) -> Option<ffmpeg::format::stream::Stream<'_>> {
        self.input.streams().find(|s| s.index() == index)
    }
}

impl Source for Reading {
    fn is_live(&self) -> bool {
        true
    }

    /// One per stream the server advertised, at the stream's own index.
    fn outputs(&self) -> Option<Vec<SrcPad>> {
        Some(
            self.contracts
                .iter()
                .enumerate()
                .map(|(index, contract)| SrcPad::with_contract(format!("src_{index}"), *contract))
                .collect(),
        )
    }

    /// The next packet the session delivers, on its stream's output, or
    /// nothing for a stream it has no output for.
    ///
    /// Waits inside the read, which the connection's own timeout bounds —
    /// see [`RtspOptions::timeout`], and [`RtspListenOptions::timeout`] for
    /// a received session — rather than through `wait`: a `Stop` reaches
    /// this source between reads, at most that long after it is sent.
    fn produce(&mut self, _wait: &mut Wait<'_>) -> Result<Produced> {
        if let Some(listening) = &self.listening {
            listening.interrupt.arm(listening.timeout);
        }
        let mut packet = ffmpeg::Packet::empty();
        match packet.read(&mut self.input) {
            Ok(()) => {
                let index = packet.stream();
                // A stream the session started after it was opened has no
                // output to go on — `FileDemuxer` passes such a one over
                // the same way.
                if index >= self.contracts.len() {
                    return Ok(Produced::Nothing);
                }
                // FFmpeg does not guarantee a demuxer fills
                // `AVPacket::time_base`, and the packet contract is that
                // one carries it — what a decoder downstream hands on to
                // its frames, and a `Pacer` paces by. Same as
                // `FileDemuxer`'s.
                if let Some(time_base) = self.stream(index).map(|stream| stream.time_base()) {
                    packet.set_time_base(time_base);
                }
                let buf = MediaBuffer::Packet(Arc::new(packet).into());
                if let Some(change) = self.watches.changed(index, &buf) {
                    report_change(
                        self.bus.as_ref(),
                        &self.pp_log,
                        ElementType::RtspSource,
                        self.pp_log.name(),
                        index,
                        change,
                    );
                }
                Ok(Produced::On(index, buf))
            }
            // A real on-demand RTSP stream can send a clean EOF; a
            // live camera essentially never will, but treat it the
            // same way `FileDemuxer` treats running out of packets.
            Err(ffmpeg::Error::Eof) => Ok(Produced::End),
            // Anything else (connection reset, socket timeout, ...) is
            // fatal — reported and this thread ends, rather than
            // retried. See this type's own docs on why: retrying
            // belongs to whoever's watching the bus, building a fresh
            // `RtspSource` to reconnect with.
            //
            // Except in a received session, where a read the interrupt
            // cut short is the publisher gone silent, and a reset
            // connection the publisher's close — see this type's "How a
            // received session ends".
            Err(error) => match &self.listening {
                Some(listening) if listening.interrupt.fired() => {
                    pp_error!(
                        self,
                        "the publisher sent nothing for {:?}",
                        listening.timeout
                    );
                    Err(RtspSourceError::PublisherSilent(listening.timeout).into())
                }
                Some(_) => match connection_closed(error) {
                    Some(closed) => {
                        pp_info!(self, "the publisher closed its connection: {closed}");
                        Ok(Produced::End)
                    }
                    None => {
                        pp_error!(self, "read failed: {error}");
                        Err(RtspSourceError::Ffmpeg(error).into())
                    }
                },
                None => {
                    pp_error!(self, "read failed: {error}");
                    Err(RtspSourceError::Ffmpeg(error).into())
                }
            },
        }
    }

    fn attached(&mut self, cx: &Attached<'_>) {
        self.pp_log = cx.pp_log().clone();
        self.bus = Some(cx.context().bus.clone());
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;

    /// `RtspOptions::timeout`'s whole reason for existing: without it
    /// ffmpeg applies *no* timeout at all and `open` blocks forever against
    /// a server that never answers. `192.0.2.1` is RFC 5737 TEST-NET-1,
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
            "open took {elapsed:?} — the configured timeout is not reaching ffmpeg"
        );
    }

    /// Each stream's packets go out on that stream's own output, carrying
    /// its time base, and a clean end of the session ends every output.
    /// Read from a file, which FFmpeg opens as it would a session: no
    /// server is needed to see how the source hands on what it reads.
    #[test]
    fn each_stream_goes_out_on_its_own_output_and_the_end_ends_them_all() {
        use std::sync::{Mutex, atomic::AtomicUsize, atomic::Ordering};

        use crate::{elements::AppSink, pipeline::Pipeline, stream::StreamEvent};

        let Some(path) = crate::test_support::try_test_video() else {
            return;
        };
        let (source, streams) =
            RtspSource::open("rtsp", &path, RtspOptions::default()).expect("open the fixture");
        let kinds: Vec<_> = streams.iter().map(|stream| stream.kind()).collect();
        let seen: Arc<Vec<Mutex<Vec<ffmpeg::Rational>>>> =
            Arc::new(streams.iter().map(|_| Mutex::default()).collect());
        let ends = Arc::new(AtomicUsize::new(0));
        let (pipeline, ()) = Pipeline::new("rtsp", source, |source, ctx| {
            for output in 0..kinds.len() {
                let sink = AppSink::with_events(
                    format!("stream-{output}"),
                    {
                        let seen = Arc::clone(&seen);
                        move |buffer| {
                            if let MediaBuffer::Packet(packet) = buffer {
                                seen[output].lock().unwrap().push(packet.time_base());
                            }
                            Ok(())
                        }
                    },
                    {
                        let ends = Arc::clone(&ends);
                        move |event: &StreamEvent| {
                            if let StreamEvent::Eos = event {
                                ends.fetch_add(1, Ordering::SeqCst);
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
        .expect("wire every stream");
        pipeline.run().expect("run");
        let deadline = Instant::now() + Duration::from_secs(20);
        while ends.load(Ordering::SeqCst) < kinds.len() {
            assert!(Instant::now() < deadline, "not every output ended");
            std::thread::sleep(Duration::from_millis(10));
        }
        pipeline.stop();

        for (output, stream) in streams.iter().enumerate() {
            let seen = seen[output].lock().unwrap();
            assert!(!seen.is_empty(), "stream {output} handed nothing on");
            assert!(
                seen.iter()
                    .all(|&time_base| time_base == stream.time_base()),
                "stream {output}'s packets carry its time base"
            );
        }
    }

    /// A caller that never touches `RtspOptions` still has to get a bounded
    /// `open`, since the default this type supplies is the only thing
    /// standing between them and ffmpeg's unbounded one.
    #[test]
    fn the_default_options_still_bound_the_connection() {
        let options = RtspOptions::default();

        assert_eq!(options.transport, crate::elements::RtspTransport::Tcp);
        assert!(
            options.timeout > Duration::ZERO,
            "the default timeout must not be ffmpeg's unbounded one"
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
        type Stamps = Vec<(Option<i64>, ffmpeg::Rational)>;

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
                                    received.packets[output]
                                        .lock()
                                        .unwrap()
                                        .push((packet.pts(), packet.time_base()));
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
                        .all(|&(_, time_base)| time_base == stream.time_base()),
                    "{:?}'s packets carry its time base",
                    stream.kind()
                );
                let pts: Vec<i64> = packets.iter().filter_map(|&(pts, _)| pts).collect();
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
