use std::{
    collections::HashMap,
    net::UdpSocket,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use crossbeam_channel::{Receiver, Sender, TrySendError, bounded, select, unbounded};
use ffmpeg_next as ffmpeg;
use rust_hlog::{HLog, herror, hinfo};
use str0m::{
    Event, Input, Output, Rtc, RtcError,
    change::{SdpAnswer, SdpOffer, SdpPendingOffer},
    format::Codec,
    media::{Direction, MediaKind, MediaTime, Mid},
    net::{Protocol, Receive},
};
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    control::{ControlMsg, ControlReceiver, apply_one, drain_control, wait_out_pause},
    driver::{Driver, StopReceiver},
    element::{Element, ElementType, Sink, Source, SourceElement, element_hlog},
    error::Result,
    pad::SrcPad,
};

/// How often `WebRtcPeer::run` re-checks `stop`/its command channel while
/// otherwise blocked on the UDP socket — see its own docs for why this is
/// polling rather than a true multi-way wait.
const POLL_INTERVAL: Duration = Duration::from_millis(20);

/// Bound on the command channel (see [`Command`]) and on each attached
/// track's inbound buffer (`WebRtcPeer` -> its `WebRtcTrackSource`). Once
/// this many media buffers are backed up, the newest one is dropped
/// instead of piling up in memory forever — the right call for live media,
/// where a backed-up peer means falling behind, not something worth
/// buffering indefinitely for (same reasoning as
/// [`crate::queue::OverflowPolicy::DropNewest`]). Control traffic on the
/// same channel (`AddTrack`/`SetAnswer`/`AcceptOffer`) is never dropped for
/// capacity pressure — those call sites block on plain `send` instead of
/// `try_send`. A disconnected peer still rejects the command; `set_answer`
/// intentionally treats that case as a no-op.
const CHANNEL_CAPACITY: usize = 128;

/// Identifies one outbound track before/after negotiation. str0m's own
/// [`Mid`] doesn't exist until the SDP exchange that creates it completes,
/// so this is a stable handle usable from the moment
/// [`WebRtcHandle::add_track`] returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TrackId(u64);

/// Errors specific to `WebRtcPeer`/`WebRtcHandle`/`WebRtcTrackSink`.
/// Converts into the crate-wide `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum WebRtcError {
    #[error("str0m error: {0}")]
    Str0m(#[from] RtcError),

    #[error("network error: {0}")]
    Io(#[from] std::io::Error),

    #[error(
        "WebRtcTrackSink only accepts already-encoded Packet buffers \
         (an encoder's output), got a {0}"
    )]
    UnsupportedBuffer(&'static str),

    #[error("WebRtcPeer's run() has already ended")]
    Closed,
}

/// One command sent from a [`WebRtcHandle`]/[`WebRtcTrackSink`] (any
/// thread) into [`WebRtcPeer::run`]'s own thread.
enum Command {
    AddTrack(TrackId, MediaKind, Direction, Codec),
    Push(TrackId, MediaBuffer),
    SetAnswer(SdpAnswer),
    /// A fresh offer from the *remote* peer (their own renegotiation, e.g.
    /// them adding a track) — out of scope to originate ourselves in v1
    /// (see the module docs), but still something this side has to be able
    /// to *accept*, or a two-way call could only ever renegotiate from one
    /// side. `Sender` here is a one-shot rendezvous, same idea as
    /// [`crate::control::ControlSender::send`]'s ack channel.
    AcceptOffer(
        SdpOffer,
        Sender<std::result::Result<SdpAnswer, WebRtcError>>,
    ),
}

/// Where one outbound track is in str0m's own offer/answer dance — mirrors
/// str0m's own `chat.rs` example's `TrackOutState`.
enum TrackOutState {
    ToOpen(MediaKind, Direction),
    Negotiating(Mid),
    Open(Mid),
}

impl TrackOutState {
    fn mid(&self) -> Option<Mid> {
        match self {
            TrackOutState::ToOpen(..) => None,
            TrackOutState::Negotiating(mid) | TrackOutState::Open(mid) => Some(*mid),
        }
    }
}

/// The [`Driver`] — owns the [`Rtc`] session and its [`UdpSocket`], and
/// drives str0m's sans-I/O poll loop on the dedicated thread
/// [`crate::driver::DriverRunner::run`] gives it. Not a
/// [`crate::element::SourceElement`]/[`Source`]: it has no `src_pads()`
/// dataflow graph of its own — see [`Driver`]'s own docs for why a
/// connection with dynamically-appearing, independently bidirectional
/// tracks doesn't fit that shape. Whatever it produces or consumes flows
/// through the separate [`WebRtcTrackSink`]/[`WebRtcTrackSource`] pairs it
/// mints per track instead (see below).
///
/// `rtc`/`socket` must already be connected: the initial SDP offer/answer
/// and ICE candidate setup happen via str0m directly, in the caller's own
/// code, *before* [`WebRtcPeer::new`] — same posture as
/// [`crate::elements::RtspServer`] not managing RTSP client connections
/// itself. `WebRtcPeer` only takes over from there.
///
/// Every track — whether it's one this side requested via
/// [`WebRtcHandle::add_track`] or one the remote peer added (`str0m`'s
/// `Event::MediaAdded`, which — critically — *never fires for a track this
/// side added itself*) — is attached the same way, the moment its `Mid`
/// exists: a [`WebRtcTrackSink`] (to reply on) and a [`WebRtcTrackSource`]
/// (whatever the remote side sends on it) are minted together and handed
/// out through [`WebRtcHandle::next_track`], no closure required. A single
/// `Direction::SendRecv` track therefore needs exactly one
/// [`WebRtcHandle::add_track`] call (on either side) and one
/// `next_track()` on *each* side — no separate outbound API, and no
/// special-casing for which side happened to originate it.
/// (`Direction::SendOnly`/`RecvOnly` still work the same way; the unused
/// half of the pair — a `WebRtcTrackSource` nothing ever sends on, or a
/// `WebRtcTrackSink` str0m has no send capability for — is simply inert,
/// not an error.) This is the same idea as
/// [`crate::elements::TeeHandle::attach`]'s dynamic attachment, just
/// without `Tee`'s `Mutex` (nothing but this one thread ever touches
/// `tracks_in`).
#[rust_hlog::hlog]
pub struct WebRtcPeer {
    name: Arc<str>,
    rtc: Rtc,
    socket: UdpSocket,
    /// Where inbound data for each attached track goes: just a plain
    /// `Sender`, not a `Box<dyn Sink>` — the matching `Receiver` lives
    /// inside that track's own [`WebRtcTrackSource`], driven by *its own*
    /// `Pipeline` on its own thread, so nothing here needs to know about
    /// `ControlMsg` at all. The `Mutex` alongside it is the same
    /// `WebRtcTrackSource`'s [`WebRtcTrackSource::codec`] cell — written
    /// here (from `Event::MediaData`), read there, from whatever thread the
    /// caller checks it on. It's the one piece of `tracks_in` shared across
    /// threads; the map itself still isn't (see below).
    #[allow(clippy::type_complexity)]
    tracks_in: HashMap<Mid, (Sender<MediaBuffer>, Arc<Mutex<Option<Codec>>>)>,
    tracks_out: HashMap<TrackId, TrackOutState>,
    /// The codec each `tracks_out` entry was opened with — what
    /// [`WebRtcHandle::add_track`] was told to expect, consulted by
    /// `write_track` to pick the payload type matching what's actually
    /// being pushed instead of guessing at whatever this connection
    /// happened to negotiate first for that `Mid`.
    track_codec: HashMap<TrackId, Codec>,
    /// The one SDP exchange currently in flight (str0m only allows one at a
    /// time — see `chat.rs`'s own `pending.is_some()` guard), plus which
    /// `TrackId`s it covers, so [`Command::SetAnswer`] knows which entries
    /// in `tracks_out` to flip from `Negotiating` to `Open`.
    pending: Option<(SdpPendingOffer, Vec<TrackId>)>,
    /// Shared with every [`WebRtcHandle`] clone, so `TrackId`s minted here
    /// (for tracks the *remote* peer added — see the type docs) never
    /// collide with ones `WebRtcHandle::add_track` mints.
    next_id: Arc<AtomicU64>,
    /// Cloned into every [`WebRtcTrackSink`] this element hands out via
    /// [`WebRtcPeer::attach_track`] — including for tracks *this* side
    /// requested, since `WebRtcTrackSink` is otherwise only ever
    /// constructed from inside `run`.
    command_tx: Sender<Command>,
    command_rx: Receiver<Command>,
    /// The other half of [`WebRtcHandle::next_track`] — one entry per
    /// newly-attached track, in attachment order (see [`TrackId`]'s own
    /// docs for why the caller has to match on it, not just take these in
    /// order, when more than one track can appear).
    new_track_tx: Sender<(TrackId, Mid, MediaKind, WebRtcTrackSink, WebRtcTrackSource)>,
    on_offer: Box<dyn FnMut(SdpOffer) + Send>,
    on_keyframe_request: Box<dyn FnMut(TrackId) + Send>,
}

/// Cheaply-cloneable handle for requesting new tracks, completing
/// renegotiation, and picking up newly-attached tracks — same spirit as
/// [`crate::elements::AppSourceHandle`]. Cloning shares one queue of
/// pending [`WebRtcHandle::next_track`] results, same as any other
/// multi-consumer channel — only one clone's call actually receives a
/// given track, so in practice only one place in the app should be
/// draining it.
#[derive(Clone)]
pub struct WebRtcHandle {
    next_id: Arc<AtomicU64>,
    command_tx: Sender<Command>,
    new_track_rx: Receiver<(TrackId, Mid, MediaKind, WebRtcTrackSink, WebRtcTrackSource)>,
}

impl WebRtcHandle {
    /// Requests a new track of `kind`/`direction`. Blocks only while the
    /// peer's bounded command queue is full; once the command is accepted,
    /// returns the locally assigned [`TrackId`]. This does not mean SDP
    /// negotiation has completed — receive the attached track through
    /// [`WebRtcHandle::next_track`]. Returns [`WebRtcError::Closed`] without
    /// yielding a `TrackId` if the peer loop has already stopped.
    ///
    /// `codec` is what [`WebRtcTrackSink::consume`] on the resulting track
    /// will actually be fed (an encoder's output, or a packet relayed
    /// verbatim from another track) — used to pick the matching payload
    /// type out of whatever this connection negotiates for the track,
    /// instead of guessing. If this connection never negotiates `codec` for
    /// it, pushed buffers are silently dropped, same as an unopened track.
    pub fn add_track(
        &self,
        kind: MediaKind,
        direction: Direction,
        codec: Codec,
    ) -> Result<TrackId> {
        let id = TrackId(self.next_id.fetch_add(1, Ordering::Relaxed));
        self.command_tx
            .send(Command::AddTrack(id, kind, direction, codec))
            .map_err(|_| WebRtcError::Closed)?;
        Ok(id)
    }

    /// Blocks until the next track attaches — either one requested via
    /// [`WebRtcHandle::add_track`] (on either side) once its `Mid` exists,
    /// or one the remote peer added on its own. Returns the `TrackId` (so
    /// the caller can match it against what `add_track` returned — a `Mid`
    /// alone doesn't exist yet at `add_track` time, see [`TrackId`]'s own
    /// docs) alongside the `Mid`/`MediaKind` str0m assigned it and the
    /// `WebRtcTrackSink`/`WebRtcTrackSource` pair to send/receive on it.
    /// `Err` once `WebRtcPeer` (and its `run`) is gone and every
    /// already-attached track has been drained.
    pub fn next_track(
        &self,
    ) -> Result<(TrackId, Mid, MediaKind, WebRtcTrackSink, WebRtcTrackSource)> {
        self.new_track_rx
            .recv()
            .map_err(|_| WebRtcError::Closed.into())
    }

    /// Feeds a remote answer back in, completing a renegotiation started by
    /// [`WebRtcHandle::add_track`]. A no-op if `WebRtcPeer` (and its `run`)
    /// is already gone.
    pub fn set_answer(&self, answer: SdpAnswer) {
        let _ = self.command_tx.send(Command::SetAnswer(answer));
    }

    /// Accepts a fresh offer from the *remote* peer (their own
    /// renegotiation) and returns the resulting answer for the caller to
    /// ship back over its own signaling transport. Blocks until
    /// `WebRtcPeer::run` has actually applied it.
    pub fn accept_remote_offer(&self, offer: SdpOffer) -> Result<SdpAnswer> {
        let (reply_tx, reply_rx) = crossbeam_channel::bounded(0);
        self.command_tx
            .send(Command::AcceptOffer(offer, reply_tx))
            .map_err(|_| WebRtcError::Closed)?;
        reply_rx
            .recv()
            .map_err(|_| WebRtcError::Closed)?
            .map_err(Into::into)
    }
}

/// One outbound track. A plain [`Sink`] — no bespoke push API, it links
/// into a [`crate::pipeline::ChainBuilder`] exactly like
/// [`crate::elements::RtspServer`] or any other terminal sink.
/// `consume()` only ever hands off to `WebRtcPeer::run`'s own thread via a
/// channel send; the actual str0m write happens over there.
#[rust_hlog::hlog]
pub struct WebRtcTrackSink {
    id: TrackId,
    command_tx: Sender<Command>,
}

impl Element for WebRtcTrackSink {
    fn name(&self) -> Arc<str> {
        format!("webrtc-track-{}", self.id.0).into()
    }

    fn element_type(&self) -> ElementType {
        ElementType::WebRtcPeer
    }

    fn hlog(&self) -> &HLog {
        &self.hlog
    }

    fn hlog_mut(&mut self) -> &mut HLog {
        &mut self.hlog
    }
}

impl Sink for WebRtcTrackSink {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        if !matches!(buf, MediaBuffer::Packet(_) | MediaBuffer::Eos) {
            let kind = match buf {
                MediaBuffer::Video(_) => "Video",
                MediaBuffer::Audio(_) => "Audio",
                MediaBuffer::Packet(_) | MediaBuffer::Eos => unreachable!("matched above"),
            };
            herror!(self, "unsupported buffer: {kind}");
            return Err(WebRtcError::UnsupportedBuffer(kind).into());
        }
        // `WebRtcPeer::run` gone (channel disconnected) means this track is
        // dead — surface it as `Err` rather than swallowing it, so whatever
        // pipeline this `Sink` is plugged into (its own `Queue`, its own
        // `Bus`) actually learns about it instead of silently sending into
        // a void forever. Non-fatal by the same convention as any other
        // `Sink::consume` failure (see `Queue`'s own docs) — just no longer
        // an invisible one.
        //
        // A full channel (`WebRtcPeer::run` backed up) drops the newest
        // buffer instead — same as an unopened track (see `add_track`'s
        // docs) — but isn't reported on a `Bus`: unlike `WebRtcPeer::run`,
        // which only ever borrows a `Bus` for the duration of one `run()`
        // call, `WebRtcTrackSink` is a handle the caller can keep past
        // `Driver::stop()`, so storing one here would keep that `Bus`'s
        // channel open indefinitely — including past whatever's waiting on
        // `BusReceiver::iter()` to finish once every sender is gone.
        match self.command_tx.try_send(Command::Push(self.id, buf)) {
            Ok(()) | Err(TrySendError::Full(_)) => Ok(()),
            Err(TrySendError::Disconnected(_)) => {
                herror!(self, "WebRtcPeer::run gone — track is dead");
                Err(WebRtcError::Closed.into())
            }
        }
    }

    fn control(&mut self, _msg: ControlMsg) -> Result<()> {
        // Terminal, same as AppSink/RtspServer: nothing buffered or
        // downstream to flush/forward for any ControlMsg.
        Ok(())
    }
}

/// One inbound track — the mirror image of [`WebRtcTrackSink`]. A plain
/// [`SourceElement`], same shape as [`crate::elements::AppSource`]: it
/// links into its own [`crate::pipeline::Pipeline`] via `src_pads()` like
/// any other source. The difference from `AppSource` is only *who* feeds
/// it — instead of an [`crate::elements::AppSourceHandle`] the app calls
/// itself, [`WebRtcPeer::run`] pushes into the sending half of this same
/// channel internally, from its own thread, for every `Event::MediaData`
/// on this track's `Mid`. Nothing here ever calls back into caller-supplied
/// code from `WebRtcPeer::run`'s own thread — that thread only ever touches
/// this crate's own types (see the module docs for why `WebRtcPeer` hands
/// tracks out through [`WebRtcHandle::next_track`] instead of a callback).
#[rust_hlog::hlog]
pub struct WebRtcTrackSource {
    name: Arc<str>,
    pad: SrcPad,
    data_rx: Receiver<MediaBuffer>,
    codec: Arc<Mutex<Option<Codec>>>,
}

impl WebRtcTrackSource {
    fn new(
        name: impl Into<String>,
        data_rx: Receiver<MediaBuffer>,
        codec: Arc<Mutex<Option<Codec>>>,
    ) -> Self {
        let name: Arc<str> = name.into().into();
        let hlog = element_hlog(ElementType::WebRtcPeer, &name, None);
        let pad = SrcPad::new(format!("{name}_src"));
        Self {
            name,
            hlog,
            pad,
            data_rx,
            codec,
        }
    }

    /// The codec this track is actually carrying, as seen on the most
    /// recently received packet's RTP payload type — `None` until the
    /// first one arrives. Unlike [`WebRtcHandle::add_track`]'s `codec`
    /// (which the *caller* declares up front for an outbound track), an
    /// inbound track's codec isn't knowable ahead of time: SDP negotiation
    /// can accept several codecs for one `m=` line, and only the packets
    /// actually arriving say which one the remote side picked (see
    /// `Event::MediaData`'s own `params` field). Whatever's downstream
    /// (e.g. a decoder) needs a keyframe before it can do anything useful
    /// anyway, so waiting for the first packet to learn the codec isn't an
    /// extra constraint in practice.
    pub fn codec(&self) -> Option<Codec> {
        *self.codec.lock().unwrap()
    }
}

impl Element for WebRtcTrackSource {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::WebRtcPeer
    }

    fn hlog(&self) -> &HLog {
        &self.hlog
    }

    fn hlog_mut(&mut self) -> &mut HLog {
        &mut self.hlog
    }
}

impl Source for WebRtcTrackSource {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl SourceElement for WebRtcTrackSource {
    /// Identical shape to [`crate::elements::AppSource::run`]: selects on
    /// `control` and its own data channel together, so `Stop`/`Pause`
    /// never wait behind a remote peer that's gone quiet. The data channel
    /// disconnecting — `WebRtcPeer` gone, whether from `Stop` or the
    /// connection dying on its own — ends this the same way `AppSource`
    /// ends when every `AppSourceHandle` is dropped: one final `Eos`, no
    /// error.
    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        hinfo!(self, "run: starting");
        loop {
            if drain_control(control, self, bus)? {
                hinfo!(self, "run: stopped");
                return Ok(());
            }

            select! {
                recv(control.rx) -> req => {
                    match req {
                        Ok(req) => {
                            if apply_one(self, bus, req.msg, &req.ack)? {
                                hinfo!(self, "run: stopped");
                                return Ok(());
                            }
                            if req.msg == ControlMsg::Pause
                                && wait_out_pause(control, self, bus)?
                            {
                                hinfo!(self, "run: stopped");
                                return Ok(());
                            }
                        }
                        // The Pipeline itself is gone — nothing left to drive this.
                        Err(_) => {
                            hinfo!(self, "run: control channel gone, ending");
                            return Ok(());
                        }
                    }
                }
                recv(self.data_rx) -> buf => {
                    match buf {
                        Ok(buf) if buf.is_eos() => {
                            hinfo!(self, "run: reached eos");
                            break;
                        }
                        Ok(buf) => {
                            if let Err(error) = self.pad.push(buf) {
                                bus.post(
                                    &self.hlog,
                                    BusEvent::Error {
                                        element_type: ElementType::WebRtcPeer,
                                        name: self.name.clone(),
                                        error,
                                    },
                                );
                            }
                        }
                        // `WebRtcPeer` gone — this track (or the whole peer) is done.
                        Err(_) => {
                            hinfo!(self, "run: WebRtcPeer gone, ending");
                            break;
                        }
                    }
                }
            }
        }
        // The data channel ending (above) can race a `Stop` sent at the
        // same moment — e.g. stopping the *upstream* `WebRtcPeer` (via its
        // `DriverRunner`) disconnects this exact channel, and a caller
        // stopping this `Pipeline` too, right after, can land its `Stop` in
        // `control`'s queue after `select!` already picked the data arm.
        // Ack it (a no-op otherwise) so `ControlSender::send`'s rendezvous
        // never blocks forever waiting for an ack this thread would
        // otherwise never get around to sending.
        while let Some((_msg, ack)) = control.try_recv() {
            let _ = ack.send(());
        }
        self.pad.push(MediaBuffer::Eos)
    }

    /// No timeline of its own — same reasoning as
    /// [`crate::elements::AppSource::seek`]: a WebRTC connection has
    /// nothing to reposition.
    fn seek(&mut self, target: Duration) -> Result<Duration> {
        Ok(target)
    }
}

impl WebRtcPeer {
    /// `rtc`/`socket` must already be connected — see the type-level docs.
    /// `on_offer` receives every renegotiation offer this element generates
    /// (via [`WebRtcHandle::add_track`]) for the caller to ship over its
    /// own signaling transport; `on_keyframe_request` reports which
    /// outbound track the remote peer wants a keyframe for (forward this to
    /// whatever's encoding that track). Newly-attached tracks themselves
    /// come from [`WebRtcHandle::next_track`], not a constructor argument.
    pub fn new(
        name: impl Into<String>,
        rtc: Rtc,
        socket: UdpSocket,
        on_offer: impl FnMut(SdpOffer) + Send + 'static,
        on_keyframe_request: impl FnMut(TrackId) + Send + 'static,
    ) -> (Self, WebRtcHandle) {
        let name: Arc<str> = name.into().into();
        let hlog = element_hlog(ElementType::WebRtcPeer, &name, None);
        hinfo!(
            hlog: &hlog,
            "created: local_addr={:?}",
            socket.local_addr()
        );
        let (command_tx, command_rx) = bounded(CHANNEL_CAPACITY);
        let (new_track_tx, new_track_rx) = unbounded();
        let next_id = Arc::new(AtomicU64::new(0));
        (
            Self {
                name,
                hlog,
                rtc,
                socket,
                tracks_in: HashMap::new(),
                tracks_out: HashMap::new(),
                track_codec: HashMap::new(),
                pending: None,
                next_id: next_id.clone(),
                command_tx: command_tx.clone(),
                command_rx,
                new_track_tx,
                on_offer: Box::new(on_offer),
                on_keyframe_request: Box::new(on_keyframe_request),
            },
            WebRtcHandle {
                next_id,
                command_tx,
                new_track_rx,
            },
        )
    }

    /// Mints a fresh [`WebRtcTrackSink`]/[`WebRtcTrackSource`] pair for
    /// `mid`/`kind` and hands both out via [`WebRtcHandle::next_track`] —
    /// see the type docs for why this is the one path both locally- and
    /// remotely-added tracks go through.
    fn attach_track(&mut self, id: TrackId, mid: Mid, kind: MediaKind) {
        hinfo!(self, "track attached: id={id:?}, mid={mid}, kind={kind:?}");
        let reply = WebRtcTrackSink {
            id,
            command_tx: self.command_tx.clone(),
            hlog: element_hlog(
                ElementType::WebRtcPeer,
                &format!("webrtc-track-{}", id.0),
                None,
            ),
        };
        let (tx, rx) = bounded(CHANNEL_CAPACITY);
        let codec = Arc::new(Mutex::new(None));
        self.tracks_in.insert(mid, (tx, codec.clone()));
        let source = WebRtcTrackSource::new(format!("webrtc-track-{}-in", id.0), rx, codec);
        let _ = self.new_track_tx.send((id, mid, kind, reply, source));
    }

    fn apply_command(&mut self, cmd: Command) -> Result<()> {
        match cmd {
            Command::AddTrack(id, kind, direction, codec) => {
                hinfo!(
                    self,
                    "add_track requested: id={id:?}, kind={kind:?}, direction={direction:?}, codec={codec:?}"
                );
                self.tracks_out
                    .insert(id, TrackOutState::ToOpen(kind, direction));
                self.track_codec.insert(id, codec);
            }
            Command::Push(id, buf) => self.write_track(id, buf)?,
            Command::SetAnswer(answer) => {
                let Some((pending, ids)) = self.pending.take() else {
                    return Ok(());
                };
                self.rtc
                    .sdp_api()
                    .accept_answer(pending, answer)
                    .inspect_err(|error| herror!(self, "accept_answer failed: {error}"))
                    .map_err(WebRtcError::from)?;
                hinfo!(self, "renegotiation complete: {} track(s)", ids.len());
                for id in ids {
                    if let Some(state @ TrackOutState::Negotiating(_)) = self.tracks_out.get(&id) {
                        let mid = state.mid().expect("Negotiating always carries a Mid");
                        self.tracks_out.insert(id, TrackOutState::Open(mid));
                    }
                }
            }
            Command::AcceptOffer(offer, reply) => {
                let result = self
                    .rtc
                    .sdp_api()
                    .accept_offer(offer)
                    .inspect_err(|error| herror!(self, "accept_offer failed: {error}"))
                    .map_err(WebRtcError::from);
                if result.is_ok() {
                    hinfo!(self, "accepted remote offer");
                }
                let _ = reply.send(result);
            }
        }
        Ok(())
    }

    /// Starts a new SDP exchange if any track is waiting to be opened and
    /// none is already in flight (str0m only allows one pending offer at a
    /// time).
    fn negotiate_if_needed(&mut self) {
        if self.pending.is_some() {
            return;
        }
        let to_open: Vec<TrackId> = self
            .tracks_out
            .iter()
            .filter(|(_, s)| matches!(s, TrackOutState::ToOpen(..)))
            .map(|(id, _)| *id)
            .collect();
        if to_open.is_empty() {
            return;
        }

        let mut newly_negotiating = Vec::with_capacity(to_open.len());
        let mut api = self.rtc.sdp_api();
        for &id in &to_open {
            let Some(TrackOutState::ToOpen(kind, direction)) = self.tracks_out.get(&id) else {
                continue;
            };
            let (kind, direction) = (*kind, *direction);
            let mid = api.add_media(kind, direction, None, None, None);
            self.tracks_out.insert(id, TrackOutState::Negotiating(mid));
            newly_negotiating.push((id, mid, kind));
        }

        if let Some((offer, pending)) = api.apply() {
            hinfo!(self, "renegotiation started: {} track(s)", to_open.len());
            self.pending = Some((pending, to_open));
            (self.on_offer)(offer);
        }

        // str0m never fires `Event::MediaAdded` for media *this side* just
        // added (see the type docs) — so this is the only place these
        // newly-minted `Mid`s ever reach `attach_track`, unlike the remote
        // side's own `Event::MediaAdded` handling below.
        for (id, mid, kind) in newly_negotiating {
            self.attach_track(id, mid, kind);
        }
    }

    fn write_track(&mut self, id: TrackId, buf: MediaBuffer) -> Result<()> {
        let Some(TrackOutState::Open(mid)) = self.tracks_out.get(&id) else {
            // Not open yet (or unknown/never added) — dropped, see
            // `WebRtcHandle::add_track`'s docs.
            return Ok(());
        };
        let MediaBuffer::Packet(packet) = buf else {
            return Ok(()); // Eos: nothing to write, nothing to flush
        };
        let Some(writer) = self.rtc.writer(*mid) else {
            return Ok(());
        };
        // Only a locally-`add_track`ed track has a declared codec (the
        // caller told us what it intends to push — see `add_track`'s
        // docs). A remotely-added one (`Event::MediaAdded`) has no such
        // declaration available at attach time, so this falls back to
        // whatever this connection negotiated first for the `Mid` — same
        // best-effort guess this always made, just now scoped to only the
        // case that has no better option.
        let pt = match self.track_codec.get(&id) {
            Some(&codec) => writer.payload_params().find(|p| p.spec().codec == codec),
            None => writer.payload_params().next(),
        }
        .map(|p| p.pt());
        let Some(pt) = pt else {
            return Ok(()); // no negotiated codec (matching or otherwise) yet
        };
        let data = packet.data().unwrap_or(&[]).to_vec();
        let time_base = packet.time_base();
        let pts = packet.pts().unwrap_or(0).max(0) as u64;
        let Some(frequency) = str0m::media::Frequency::new(time_base.denominator() as u32) else {
            return Ok(()); // packet has no usable time base — nothing sane to write
        };
        let rtp_time = MediaTime::new(pts, frequency);
        writer
            .write(pt, Instant::now(), rtp_time, data)
            .inspect_err(|error| herror!(self, "writer.write failed: {error}"))
            .map_err(WebRtcError::from)?;
        Ok(())
    }

    /// Drains every immediately-available str0m output (retransmits and
    /// events), returning once str0m itself has nothing left to do until
    /// the returned deadline.
    fn drive_until_timeout(&mut self, bus: &Bus) -> Result<Instant> {
        loop {
            let output = self
                .rtc
                .poll_output()
                .inspect_err(|error| herror!(self, "poll_output failed: {error}"))
                .map_err(WebRtcError::from)?;
            match output {
                Output::Timeout(deadline) => return Ok(deadline),
                Output::Transmit(t) => {
                    // A single failed send (e.g. transient ICMP unreachable)
                    // isn't fatal to the whole connection — str0m's own
                    // retransmit/timeout logic handles loss.
                    let _ = self.socket.send_to(&t.contents, t.destination);
                }
                Output::Event(event) => self.handle_event(event, bus),
            }
        }
    }

    fn handle_event(&mut self, event: Event, bus: &Bus) {
        match event {
            Event::MediaAdded(added) => {
                // Only reached for media the *remote* peer added (see the
                // type docs) — by definition already fully negotiated by
                // the time we see this, so `Open` immediately: unlike a
                // locally-requested track, there's no answer left to wait
                // for before a `WebRtcTrackSink` bound to it can actually
                // send.
                let id = TrackId(self.next_id.fetch_add(1, Ordering::Relaxed));
                self.tracks_out.insert(id, TrackOutState::Open(added.mid));
                self.attach_track(id, added.mid, added.kind);
            }
            Event::MediaData(data) => {
                if let Some((tx, codec)) = self.tracks_in.get(&data.mid) {
                    // Every packet, not just the first: cheap (one lock),
                    // and correct if the remote side ever actually changes
                    // codec mid-stream (rare, but the payload type is free
                    // to vary packet-to-packet — see `WebRtcTrackSource::
                    // codec`'s own docs for why this can't be pinned down
                    // any earlier than "whatever the last packet said").
                    *codec.lock().unwrap() = Some(data.params.spec().codec);

                    let mut packet = ffmpeg::Packet::copy(&data.data);
                    // `data.time` is str0m's own RTP timestamp (numerator)
                    // over the codec's clock rate (denominator) — reused
                    // as-is for pts/dts. No B-frame reordering happens over
                    // RTP (decode order == transmit order), so pts and dts
                    // are always the same value here.
                    packet.set_time_base(ffmpeg::Rational::new(1, data.time.denom() as i32));
                    let pts = data.time.numer() as i64;
                    packet.set_pts(Some(pts));
                    packet.set_dts(Some(pts));
                    if data.is_keyframe() {
                        let flags = packet.flags() | ffmpeg::codec::packet::Flags::KEY;
                        packet.set_flags(flags);
                    }
                    match tx.try_send(MediaBuffer::Packet(Arc::new(packet))) {
                        Ok(()) => {}
                        Err(TrySendError::Full(_)) => {
                            // This track's `WebRtcTrackSource` (or whatever
                            // it feeds) isn't keeping up — drop the newest
                            // buffer rather than let this grow unbounded
                            // (see `CHANNEL_CAPACITY`'s docs).
                            bus.post(
                                &self.hlog,
                                BusEvent::Dropped {
                                    element_type: ElementType::WebRtcPeer,
                                    name: self.name.clone(),
                                },
                            );
                        }
                        Err(TrySendError::Disconnected(_)) => {
                            // This track's `WebRtcTrackSource` is gone (its
                            // own `Pipeline` finished) — stop trying to feed
                            // it.
                            self.tracks_in.remove(&data.mid);
                        }
                    }
                }
            }
            Event::KeyframeRequest(req) => {
                if let Some((&id, _)) = self
                    .tracks_out
                    .iter()
                    .find(|(_, s)| s.mid() == Some(req.mid))
                {
                    hinfo!(self, "keyframe requested: id={id:?}, mid={}", req.mid);
                    (self.on_keyframe_request)(id);
                }
            }
            Event::Connected => {
                hinfo!(self, "ICE+DTLS connected");
            }
            Event::IceConnectionStateChange(state) => {
                hinfo!(self, "ICE connection state: {state:?}");
            }
            Event::MediaChanged(changed) => {
                hinfo!(
                    self,
                    "media changed: mid={}, direction={:?}",
                    changed.mid,
                    changed.direction
                );
            }
            // `Event` is `#[non_exhaustive]` — data channels, stats, etc.
            // are still outside this element's concern for now.
            _ => {}
        }
    }
}

impl Element for WebRtcPeer {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::WebRtcPeer
    }

    fn hlog(&self) -> &HLog {
        &self.hlog
    }

    fn hlog_mut(&mut self) -> &mut HLog {
        &mut self.hlog
    }
}

impl Driver for WebRtcPeer {
    /// Drives str0m's poll loop. Every iteration: apply any commands from
    /// `WebRtcHandle`/`WebRtcTrackSink`, start a renegotiation if a track
    /// is waiting, drain str0m's own output (writing/dispatching as it
    /// goes), check `stop`, then block on the UDP socket for at most
    /// `POLL_INTERVAL` — capped below whatever str0m itself asked for, so
    /// the command channel and `stop` are never starved for longer than
    /// that even when nothing else is happening. There's no true
    /// multi-way wait across the command channel, `stop`, *and* a raw
    /// socket the way [`crate::elements::AppSource`] manages across two
    /// `crossbeam_channel`s (a `UdpSocket` isn't `select!`-able), so this
    /// is bounded polling instead — worst case `POLL_INTERVAL` of extra
    /// latency for `Stop`/a fresh `add_track`, not unboundedly stuck.
    ///
    /// `stop`/the connection dying both clear `tracks_in` immediately, so
    /// every already-handed-out `WebRtcTrackSource` sees its data channel
    /// disconnect and ends with a final `Eos` right away, instead of
    /// waiting for this whole `WebRtcPeer` to be dropped later by whatever
    /// owns its `DriverRunner`. Neither `WebRtcPeer` nor its
    /// `WebRtcTrackSource`s have a `Pause`/`Seek` concept — see
    /// [`Driver`]'s own docs for why that's not just an oversight: freezing
    /// this loop would starve ICE keepalives/DTLS retransmits, likely
    /// dropping the connection rather than gracefully suspending it.
    fn run(&mut self, stop: &StopReceiver, bus: &Bus) -> Result<()> {
        hinfo!(self, "run: starting");
        let mut buf = vec![0u8; 2000];
        loop {
            while let Ok(cmd) = self.command_rx.try_recv() {
                self.apply_command(cmd)?;
            }
            self.negotiate_if_needed();

            let deadline = self.drive_until_timeout(bus)?;
            if !self.rtc.is_alive() || stop.is_stopped() {
                hinfo!(self, "run: stopped (rtc alive: {})", self.rtc.is_alive());
                self.tracks_in.clear();
                return Ok(());
            }

            let wait = deadline
                .saturating_duration_since(Instant::now())
                .min(POLL_INTERVAL)
                .max(Duration::from_millis(1));
            self.socket
                .set_read_timeout(Some(wait))
                .inspect_err(|error| herror!(self, "set_read_timeout failed: {error}"))
                .map_err(WebRtcError::from)?;

            match self.socket.recv_from(&mut buf) {
                Ok((n, source)) => {
                    let Ok(contents) = buf[..n].try_into() else {
                        continue; // not a WebRTC datagram we recognize — ignore
                    };
                    let destination = self
                        .socket
                        .local_addr()
                        .inspect_err(|error| herror!(self, "local_addr failed: {error}"))
                        .map_err(WebRtcError::from)?;
                    self.rtc
                        .handle_input(Input::Receive(
                            Instant::now(),
                            Receive {
                                proto: Protocol::Udp,
                                source,
                                destination,
                                contents,
                            },
                        ))
                        .inspect_err(|error| herror!(self, "handle_input(Receive) failed: {error}"))
                        .map_err(WebRtcError::from)?;
                }
                Err(e)
                    if matches!(
                        e.kind(),
                        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                    ) =>
                {
                    // Nothing arrived, but str0m still needs to be told
                    // time has passed — its own internal clock only moves
                    // forward via `Input::Timeout`, and *that* is what
                    // makes the next `poll_output()` produce whatever's
                    // next (retransmits, RTCP, the initial STUN checks,
                    // ...). Skipping this on every timeout would leave
                    // str0m stuck forever waiting for input that already
                    // isn't coming.
                    self.rtc
                        .handle_input(Input::Timeout(Instant::now()))
                        .inspect_err(|error| herror!(self, "handle_input(Timeout) failed: {error}"))
                        .map_err(WebRtcError::from)?;
                }
                Err(e) => {
                    herror!(self, "recv_from failed: {e}");
                    return Err(WebRtcError::from(e).into());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        net::UdpSocket,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
            mpsc,
        },
        thread,
        time::{Duration, Instant},
    };

    use str0m::Candidate;

    use super::*;
    use crate::{driver::DriverRunner, pipeline::Pipeline};

    fn command_only_handle(capacity: usize) -> (WebRtcHandle, Receiver<Command>) {
        let (command_tx, command_rx) = bounded(capacity);
        let (_new_track_tx, new_track_rx) = unbounded();
        (
            WebRtcHandle {
                next_id: Arc::new(AtomicU64::new(0)),
                command_tx,
                new_track_rx,
            },
            command_rx,
        )
    }

    #[test]
    fn add_track_returns_the_id_that_was_enqueued() {
        let (handle, command_rx) = command_only_handle(1);

        let returned = handle
            .add_track(MediaKind::Video, Direction::SendRecv, Codec::Vp8)
            .expect("live command receiver should accept AddTrack");
        let Command::AddTrack(enqueued, kind, direction, codec) =
            command_rx.recv().expect("AddTrack should be queued")
        else {
            panic!("expected AddTrack command");
        };

        assert_eq!(returned, enqueued);
        assert_eq!(kind, MediaKind::Video);
        assert_eq!(direction, Direction::SendRecv);
        assert_eq!(codec, Codec::Vp8);
    }

    #[test]
    fn add_track_blocks_for_backpressure_then_unblocks_when_capacity_opens() {
        let (handle, command_rx) = command_only_handle(1);
        handle
            .add_track(MediaKind::Video, Direction::SendRecv, Codec::Vp8)
            .expect("first command should fill the queue");

        let (entered_tx, entered_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let blocked_handle = handle.clone();
        let worker = thread::spawn(move || {
            entered_tx.send(()).expect("test receiver alive");
            let result =
                blocked_handle.add_track(MediaKind::Audio, Direction::SendOnly, Codec::Opus);
            done_tx.send(result).expect("test receiver alive");
        });

        entered_rx.recv().expect("worker should start add_track");
        assert!(
            done_rx.recv_timeout(Duration::from_millis(50)).is_err(),
            "second AddTrack must wait while the bounded queue is full"
        );

        let _first = command_rx.recv().expect("free one queue slot");
        let second_id = done_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("add_track should unblock once capacity opens")
            .expect("command receiver is still alive");
        let Command::AddTrack(enqueued_id, ..) =
            command_rx.recv().expect("second AddTrack should be queued")
        else {
            panic!("expected AddTrack command");
        };
        assert_eq!(second_id, enqueued_id);
        worker.join().expect("worker should finish cleanly");
    }

    #[test]
    fn add_track_returns_closed_instead_of_a_phantom_id() {
        let (handle, command_rx) = command_only_handle(1);
        drop(command_rx);

        let error = handle
            .add_track(MediaKind::Video, Direction::SendRecv, Codec::Vp8)
            .expect_err("closed peer must not yield a TrackId");
        assert!(matches!(
            error,
            crate::Error::WebRtcError(WebRtcError::Closed)
        ));
    }

    #[rust_hlog::hlog]
    struct CountingSink {
        count: Arc<AtomicUsize>,
    }

    impl Element for CountingSink {
        fn name(&self) -> Arc<str> {
            "counter".into()
        }

        fn element_type(&self) -> ElementType {
            ElementType::Other
        }

        fn hlog(&self) -> &HLog {
            &self.hlog
        }

        fn hlog_mut(&mut self) -> &mut HLog {
            &mut self.hlog
        }
    }

    impl Sink for CountingSink {
        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            if matches!(buf, MediaBuffer::Packet(_)) {
                self.count.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        }

        fn control(&mut self, _msg: ControlMsg) -> Result<()> {
            Ok(())
        }
    }

    /// Two `Rtc`s, connected via a throwaway data channel — just enough to
    /// get ICE/DTLS established with zero media, mirroring how a real
    /// caller does the initial signaling before ever touching
    /// `WebRtcPeer` (see its own docs).
    fn connected_pair() -> (Rtc, UdpSocket, Rtc, UdpSocket) {
        let socket_a = UdpSocket::bind("127.0.0.1:0").expect("bind a");
        let socket_b = UdpSocket::bind("127.0.0.1:0").expect("bind b");
        let addr_a = socket_a.local_addr().expect("addr a");
        let addr_b = socket_b.local_addr().expect("addr b");

        let mut rtc_a = Rtc::builder().build(Instant::now());
        rtc_a
            .add_local_candidate(Candidate::host(addr_a, "udp").expect("candidate a"))
            .expect("add candidate a");
        let mut rtc_b = Rtc::builder().build(Instant::now());
        rtc_b
            .add_local_candidate(Candidate::host(addr_b, "udp").expect("candidate b"))
            .expect("add candidate b");

        let mut changes = rtc_a.sdp_api();
        changes.add_channel("bootstrap".to_string());
        let (offer, pending) = changes.apply().expect("adding a channel always offers");
        let answer = rtc_b.sdp_api().accept_offer(offer).expect("b accepts");
        rtc_a
            .sdp_api()
            .accept_answer(pending, answer)
            .expect("a accepts answer");

        (rtc_a, socket_a, rtc_b, socket_b)
    }

    fn push_packets(sink: &mut WebRtcTrackSink) {
        for i in 0..5 {
            // `write_track` needs a real time base to build str0m's own
            // `MediaTime` from — a bare `Packet::copy` defaults to 0/0,
            // which is silently unusable (dropped, not an error).
            let mut packet = ffmpeg::Packet::copy(&[1, 2, 3, 4]);
            packet.set_time_base(ffmpeg::Rational::new(1, 90_000));
            packet.set_pts(Some(i * 3_000));
            sink.consume(MediaBuffer::Packet(Arc::new(packet)))
                .expect("push");
        }
    }

    /// Wires `source` into its own `Pipeline`, forwarding every packet it
    /// produces into a `CountingSink` that increments `count` — standing in
    /// for the real, independent downstream pipeline a `WebRtcTrackSource`
    /// is meant to be driven from.
    fn wire_counting(source: WebRtcTrackSource, count: Arc<AtomicUsize>) -> Arc<Pipeline> {
        let sink = CountingSink {
            count,
            hlog: element_hlog(ElementType::Other, "counter", None),
        };
        Pipeline::new("test", source, |source, ctx| {
            let branch = ctx.branch().to(Box::new(sink))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("test pipeline wiring must succeed")
    }

    /// One `Direction::SendRecv` track, opened by `WebRtcHandle::add_track`
    /// on one peer and relayed to the other by hand (standing in for a
    /// real signaling transport — see `on_offer`'s docs), carries data
    /// *both* ways: peer-a pushes into the `WebRtcTrackSink` its own
    /// `next_track()` returned for the track it just added (str0m never
    /// fires `Event::MediaAdded` for that — see the type docs), and peer-b
    /// pushes back on the exact same `Mid` via the `WebRtcTrackSink` its
    /// own `next_track()` returned for the resulting `Event::MediaAdded` —
    /// no second `add_track`/renegotiation needed for the reverse
    /// direction. Also exercises real ICE/DTLS-SRTP over loopback UDP, with
    /// each `WebRtcTrackSource` driven by its own independent `Pipeline`.
    #[test]
    fn one_sendrecv_track_carries_data_both_ways() {
        let (rtc_a, socket_a, rtc_b, socket_b) = connected_pair();

        let (offer_tx, offer_rx) = crossbeam_channel::unbounded::<SdpOffer>();

        let (peer_a, handle_a) = WebRtcPeer::new(
            "peer-a",
            rtc_a,
            socket_a,
            move |offer| {
                let _ = offer_tx.send(offer);
            },
            |_id| {},
        );
        let (peer_b, handle_b) = WebRtcPeer::new("peer-b", rtc_b, socket_b, |_offer| {}, |_id| {});

        // `WebRtcPeer` is a `Driver`, not a `Pipeline` source — no wiring
        // closure needed; inbound tracks attach dynamically via
        // `next_track()` instead, each into its own `Pipeline`.
        let driver_a = DriverRunner::new(peer_a);
        let driver_b = DriverRunner::new(peer_b);
        driver_a.run();
        driver_b.run();

        // Let ICE/DTLS actually finish over the real loopback sockets
        // before renegotiating.
        thread::sleep(Duration::from_millis(200));

        let track_id = handle_a
            .add_track(MediaKind::Video, Direction::SendRecv, Codec::Vp8)
            .expect("running peer should accept AddTrack");
        // `next_track()` returns for peer-a's own track the moment
        // `add_track`'s negotiation mints a `Mid` — before the offer even
        // leaves this process, let alone before an answer comes back.
        let (returned_id, _mid, _kind, mut sink_a, source_a) = handle_a
            .next_track()
            .expect("peer-a's own track should attach");
        assert_eq!(
            track_id, returned_id,
            "next_track should report the TrackId add_track just returned"
        );

        let offer = offer_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("peer-a should generate a renegotiation offer");
        let answer = handle_b
            .accept_remote_offer(offer)
            .expect("peer-b should accept the offer");
        handle_a.set_answer(answer);
        // peer-b's track attaches as part of accepting the offer (its
        // `Event::MediaAdded`), independent of the answer round-trip.
        let (_id, _mid, _kind, mut sink_b, source_b) = handle_b
            .next_track()
            .expect("peer-b's remote track should attach");

        let received_by_a = Arc::new(AtomicUsize::new(0));
        let received_by_b = Arc::new(AtomicUsize::new(0));
        let track_pipeline_a = wire_counting(source_a, received_by_a.clone());
        let track_pipeline_b = wire_counting(source_b, received_by_b.clone());
        track_pipeline_a.run();
        track_pipeline_b.run();

        // Let the answer actually apply before pushing media through it.
        thread::sleep(Duration::from_millis(100));

        push_packets(&mut sink_a);
        thread::sleep(Duration::from_millis(300));
        assert_eq!(
            received_by_b.load(Ordering::SeqCst),
            5,
            "peer-b should receive everything peer-a pushed"
        );

        push_packets(&mut sink_b);
        thread::sleep(Duration::from_millis(300));
        assert_eq!(
            received_by_a.load(Ordering::SeqCst),
            5,
            "peer-a should receive everything peer-b pushed back, on the same track"
        );

        // Trivial now — `DriverRunner::stop` just flips a flag, no
        // rendezvous ack to race (see `StopReceiver`'s own docs).
        driver_a.stop();
        driver_b.stop();
        // Deliberately also stopped even though each `WebRtcTrackSource`'s
        // `Pipeline` may already be ending on its own by this point (the
        // `driver_a/b.stop()` above disconnects its data channel, which
        // ends it with a natural `Eos` — see that type's own docs): this is
        // a regression guard for a real `Pipeline::stop` bug this exact
        // race used to hit, now fixed at its source (see `Pipeline`'s own
        // `control_rx` field docs).
        track_pipeline_a.stop();
        track_pipeline_b.stop();

        let events_a: Vec<_> = driver_a.bus().iter().collect();
        let events_b: Vec<_> = driver_b.bus().iter().collect();
        let track_events_a: Vec<_> = track_pipeline_a.bus().iter().collect();
        let track_events_b: Vec<_> = track_pipeline_b.bus().iter().collect();
        assert!(
            !events_a.iter().any(|e| matches!(e, BusEvent::Error { .. })),
            "unexpected error event(s) on peer-a: {events_a:?}"
        );
        assert!(
            !events_b.iter().any(|e| matches!(e, BusEvent::Error { .. })),
            "unexpected error event(s) on peer-b: {events_b:?}"
        );
        assert!(
            !track_events_a
                .iter()
                .any(|e| matches!(e, BusEvent::Error { .. })),
            "unexpected error event(s) on peer-a's inbound track: {track_events_a:?}"
        );
        assert!(
            !track_events_b
                .iter()
                .any(|e| matches!(e, BusEvent::Error { .. })),
            "unexpected error event(s) on peer-b's inbound track: {track_events_b:?}"
        );
    }

    /// Regression test for the claim in `Driver::run`'s own docs: stopping
    /// one side must clear its `tracks_in` immediately, so an
    /// already-attached `WebRtcTrackSource` sees its data channel
    /// disconnect and ends with a final `Eos` on its own — not left
    /// hanging until some other, unrelated thing (e.g. a caller's own
    /// `Pipeline::stop`) intervenes. There's no "reconnect" concept here
    /// to test instead: `WebRtcPeer::run` only ever reacts to its `Rtc`
    /// going not-alive or an explicit `Stop`, treating both the same way
    /// (see that `if` right before `tracks_in.clear()`); `Stop` is the
    /// deterministic one of the two to trigger from a test. Deliberately
    /// omits the `track_pipeline.stop()` safety net
    /// `one_sendrecv_track_carries_data_both_ways` uses — the whole point
    /// here is to prove the `Pipeline` ends on its own.
    #[test]
    fn stopping_a_peer_ends_its_inbound_track_source_with_a_clean_eos() {
        let (rtc_a, socket_a, rtc_b, socket_b) = connected_pair();

        let (offer_tx, offer_rx) = crossbeam_channel::unbounded::<SdpOffer>();
        let (peer_a, handle_a) = WebRtcPeer::new(
            "peer-a",
            rtc_a,
            socket_a,
            move |offer| {
                let _ = offer_tx.send(offer);
            },
            |_id| {},
        );
        let (peer_b, handle_b) = WebRtcPeer::new("peer-b", rtc_b, socket_b, |_offer| {}, |_id| {});

        let driver_a = DriverRunner::new(peer_a);
        let driver_b = DriverRunner::new(peer_b);
        driver_a.run();
        driver_b.run();

        thread::sleep(Duration::from_millis(200));

        handle_a
            .add_track(MediaKind::Video, Direction::SendOnly, Codec::Vp8)
            .expect("running peer should accept AddTrack");
        let offer = offer_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("peer-a should generate a renegotiation offer");
        let answer = handle_b
            .accept_remote_offer(offer)
            .expect("peer-b should accept the offer");
        handle_a.set_answer(answer);

        let (_id, _mid, _kind, _sink_b, source_b) = handle_b
            .next_track()
            .expect("peer-b's remote track should attach");

        let received = Arc::new(AtomicUsize::new(0));
        let track_pipeline_b = wire_counting(source_b, received);
        track_pipeline_b.run();

        // Let the answer actually apply before tearing the connection down.
        thread::sleep(Duration::from_millis(100));

        driver_b.stop();

        // No `track_pipeline_b.stop()` — draining the bus to completion is
        // itself the proof: it only returns once every `Bus` sender,
        // including the one `WebRtcTrackSource`'s own `Pipeline::run`
        // thread holds, has actually dropped.
        let track_events_b: Vec<_> = track_pipeline_b.bus().iter().collect();
        assert!(
            track_events_b
                .iter()
                .any(|e| matches!(e, BusEvent::Eos { .. })),
            "expected the inbound track's own Pipeline to reach Eos on its \
             own once peer-b stopped, without an explicit Pipeline::stop; \
             got {track_events_b:?}"
        );
        assert!(
            !track_events_b
                .iter()
                .any(|e| matches!(e, BusEvent::Error { .. })),
            "unexpected error event(s): {track_events_b:?}"
        );

        driver_a.stop();
    }
}
