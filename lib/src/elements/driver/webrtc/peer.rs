use std::{
    collections::HashMap,
    net::UdpSocket,
    sync::{Arc, Mutex, atomic::AtomicU64},
    time::{Duration, Instant},
};

use crate::pp_log::{PpLog, pp_error, pp_info, pp_warn};
use crossbeam_channel::{Receiver, Sender, bounded, unbounded};
use ffmpeg_next as ffmpeg;
use str0m::{
    Event, Input, Rtc,
    change::{SdpOffer, SdpPendingOffer},
    format::Codec,
    media::{Direction, MediaKind, MediaTime, Mid},
    net::{Protocol, Receive},
};

use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    driver::{Driver, StopReceiver},
    element::{Element, ElementType, element_pp_log},
    error::Result,
    time::{InvalidTimeBase, MediaTimestamp},
};

use crate::core::bell::Bell;
use crate::elements::net;

use super::{
    command::{Command, OpenTrack, TrackId, TrackOutState, WebRtcError},
    connection::{self, Next, Outputs},
    inbox::{Delivered, Inbox},
    outbox::{Outbox, Outgoing},
    stream_info::{StreamInfoProbe, WebRtcStreamInfo},
    track::{WebRtcHandle, WebRtcTrackSink},
};

/// How often `WebRtcPeer::run` re-checks `stop`/its command channel while
/// otherwise blocked on the UDP socket — see its own docs for why this is
/// polling rather than a true multi-way wait.
const POLL_INTERVAL: Duration = Duration::from_millis(20);

/// Bound on the command channel (see [`Command`]), on the outbox its track
/// sinks share, and on each attached track's inbound buffer (`WebRtcPeer`
/// -> the `WebRtcSource` it is an output of). Live media that backs up this far means
/// falling behind, not something worth buffering indefinitely for: an
/// inbound track drops its newest buffer (same reasoning as
/// [`crate::queue::OverflowPolicy::DropNewest`]), and the outbox lets go of
/// what it holds and goes on from each track's next keyframe. Commands
/// (`OpenTracks`/`SetAnswer`/`AcceptOffer`) are never dropped for capacity
/// pressure — those call sites block on plain `send`. A disconnected peer
/// still rejects the command; `set_answer` intentionally treats that case
/// as a no-op.
const CHANNEL_CAPACITY: usize = 128;
/// A WebRTC connection to one remote peer, carrying its tracks in and out
/// of pipelines.
///
/// Signalling and the first connection are the application's own, made with
/// `str0m` directly; the peer then takes over the connected [`Rtc`] and its
/// socket. It is not a pipeline element: run it with a
/// [`DriverRunner`](crate::driver::DriverRunner). What it sends goes out
/// through a [`WebRtcMuxer`](crate::elements::WebRtcMuxer) of its
/// ([`WebRtcHandle::muxer`]), and what the remote peer sends comes in as a
/// [`WebRtcSource`](crate::elements::WebRtcSource)
/// ([`WebRtcHandle::next_source`]) — the same two a WHIP or WHEP connection
/// is used through. Needs the `webrtc` feature.
///
/// # Example
///
/// Sending one H.264 track over a connection already set up through `str0m`:
///
/// ```no_run
/// # fn main() -> media_pp::Result<()> {
/// use media_pp::driver::DriverRunner;
/// use media_pp::elements::{StreamInfo, WebRtcPeer};
/// # let rtc = str0m::Rtc::builder().build(std::time::Instant::now());
/// # let socket = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
/// # let video: StreamInfo = unimplemented!();
///
/// let (peer, handle) = WebRtcPeer::new(
///     "peer",
///     rtc,    // connected through your own signalling
///     socket, // the UDP socket it is connected on
///     |offer| { /* send this renegotiation offer to the remote side */ },
///     |track| { /* ask the encoder of the track named `track` for a keyframe */ },
/// );
/// let driver = DriverRunner::new(peer);
/// driver.run()?;
///
/// let mut muxer = handle.muxer("camera");
/// let track = muxer.add_stream("video", video)?; // an H.264 encoder's stream
/// let mut sinks = muxer.open()?;                 // the offer goes to `on_offer`
/// let sink = sinks.take(track)?;                 // ends the encoder's chain
/// // The remote peer's answer: `handle.set_answer(answer)`. What it sends
/// // back, it adds as tracks of its own: `handle.next_source("remote")`.
/// # Ok(())
/// # }
/// ```
///
/// Two peers connected over loopback, with the signalling between them, are in
/// the
/// [`webrtc_loopback`](https://github.com/hash1018/media-pp/tree/main/examples/webrtc/webrtc_loopback)
/// example.
///
/// # How it works
///
/// It is a [`Driver`] — owns the [`Rtc`] session and its [`UdpSocket`],
/// and drives str0m's sans-I/O poll loop on the dedicated thread
/// [`crate::driver::DriverRunner::run`] gives it. Not a source or a sink
/// of its own: a connection's tracks appear while it runs, from either
/// side, and what flows through each goes through the muxer or the source
/// it came out of.
///
/// `rtc`/`socket` must already be connected: the initial SDP offer/answer
/// and ICE candidate setup happen via str0m directly, in the caller's own
/// code, *before* [`WebRtcPeer::new`]. `WebRtcPeer` only takes over after
/// signaling has established the connection; it does not provide a signaling
/// server itself.
///
/// Every media section goes one way. What this side sends goes through a
/// muxer: on a section the remote peer added asking to be sent to, where
/// it asked for one of the track's kind that took its codec — with no
/// renegotiation — and else on a section the muxer adds, sending only, in
/// one renegotiation for all its tracks. What the remote peer sends is
/// received, as a source's outputs — the tracks one of its offers adds
/// coming out together, as one source, once str0m has said everything that
/// offer brought. A remote peer that changes the direction of one it sends
/// on is reported as [`WebRtcError::DirectionChanged`] on the [`Bus`].
pub struct WebRtcPeer {
    pp_log: PpLog,
    name: Arc<str>,
    rtc: Rtc,
    socket: UdpSocket,
    /// Where inbound data for each attached track goes: an inbox, not a
    /// `Box<dyn RawSink>` — the source reading it is driven by *its own*
    /// `Pipeline` on its own thread, so nothing here needs to know about
    /// `ControlMsg` at all. A one-slot channel beside it confirms enough of
    /// the actual payload — received H.264 SPS/PPS included — for that
    /// source to say what the track carries.
    tracks_in: HashMap<Mid, TrackInState>,
    tracks_out: HashMap<TrackId, TrackOutState>,
    /// Each sending track's name, as its muxer's `add_stream` gave it —
    /// what a keyframe request for it is reported by.
    track_names: HashMap<TrackId, Arc<str>>,
    /// Muxers' tracks offered and not all attached yet, with where their
    /// sinks go once they are.
    opening: Vec<Opening>,
    /// Sections the remote peer added asking to be sent to, in the order
    /// it added them, that no muxer's track has been sent on yet.
    requested: Vec<(Mid, MediaKind)>,
    /// The remote peer's tracks attached since its last batch went out, and
    /// the bell they share: a batch is one source, which waits on one bell.
    arriving: Vec<Arriving>,
    arriving_bell: Option<Bell>,
    /// Where each batch goes, for [`WebRtcHandle::next_source`]; let go of
    /// as the peer stops, so a call waiting there hears it.
    sources_tx: Option<Sender<Vec<Arriving>>>,
    /// Each muxer track's codec, until its [`WebRtcTrackSink`] attaches and
    /// takes it; the muxer then declares its parameters in full through
    /// [`WebRtcTrackSink::set_source_parameters`].
    track_codec: HashMap<TrackId, Codec>,
    /// The currently negotiated codec families for each attached media
    /// section, shared with its sink. A locally-created track is
    /// attached while its offer is pending, so [`Command::SetAnswer`]
    /// refreshes this cell after applying the answer.
    negotiated_codecs: HashMap<Mid, Arc<Mutex<Vec<Codec>>>>,
    /// The direction each attached track was handed out with, so
    /// `Event::MediaChanged` can tell an actual renegotiation apart from a
    /// re-announcement of the direction already in force. Keyed by `Mid`
    /// because that is what the event carries.
    track_direction: HashMap<Mid, Direction>,
    /// Set when str0m reports the remote DTLS/SCTP connection closing. The
    /// current `drive_until_timeout` call is still allowed to drain any
    /// reciprocal protocol output before the run loop tears down tracks.
    remote_closed: bool,
    /// The one SDP exchange currently in flight (str0m only allows one at a
    /// time — see `chat.rs`'s own `pending.is_some()` guard), plus which
    /// `TrackId`s it covers, so [`Command::SetAnswer`] knows which entries
    /// in `tracks_out` to flip from `Negotiating` to `Open`.
    pending: Option<(SdpPendingOffer, Vec<TrackId>)>,
    command_rx: Receiver<Command>,
    /// What every [`WebRtcTrackSink`] this element hands out puts its
    /// packets in, for `run` to write — closed as the peer goes, so a sink
    /// kept past it says so.
    outbox: Outbox,
    on_offer: Box<dyn FnMut(SdpOffer) + Send>,
    on_keyframe_request: Box<dyn FnMut(&str) + Send>,
}

/// One track the remote peer sends, attached and not yet handed out: what
/// [`WebRtcHandle::next_source`] makes a source's output of.
pub(super) struct Arriving {
    pub(super) inbox: Inbox,
    /// What its first packets say it carries, once they have.
    pub(super) described: Receiver<WebRtcStreamInfo>,
}

/// A muxer's tracks being opened.
struct Opening {
    ids: Vec<TrackId>,
    sinks: Vec<Option<WebRtcTrackSink>>,
    reply: Sender<std::result::Result<Vec<WebRtcTrackSink>, WebRtcError>>,
}

struct TrackInState {
    inbox: Inbox,
    stream_info_tx: Sender<WebRtcStreamInfo>,
    stream_info_probe: StreamInfoProbe,
    stream_info_sent: bool,
}

impl Drop for TrackInState {
    /// The peer has let go of the track, stopping or for good: its source
    /// ends once it has handed on what it holds.
    fn drop(&mut self) {
        self.inbox.close();
    }
}

impl WebRtcPeer {
    /// `rtc`/`socket` must already be connected — see the type-level docs.
    /// `on_offer` receives every renegotiation offer this element generates
    /// (as a [`WebRtcHandle::muxer`] opens) for the caller to ship over its
    /// own signaling transport; `on_keyframe_request` names the track sent
    /// — by the name its muxer's `add_stream` gave it — the remote peer
    /// wants a keyframe for, to forward to whatever encodes it.
    pub fn new(
        name: impl Into<String>,
        rtc: Rtc,
        socket: UdpSocket,
        on_offer: impl FnMut(SdpOffer) + Send + 'static,
        on_keyframe_request: impl FnMut(&str) + Send + 'static,
    ) -> (Self, WebRtcHandle) {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::WebRtcPeer, &name, None);
        pp_info!(
            pp_log: &pp_log,
            "created: local_addr={:?}",
            socket.local_addr()
        );
        let (command_tx, command_rx) = bounded(CHANNEL_CAPACITY);
        let (sources_tx, sources_rx) = unbounded();
        let next_id = Arc::new(AtomicU64::new(0));
        (
            Self {
                name,
                pp_log,
                rtc,
                socket,
                tracks_in: HashMap::new(),
                tracks_out: HashMap::new(),
                track_names: HashMap::new(),
                opening: Vec::new(),
                requested: Vec::new(),
                arriving: Vec::new(),
                arriving_bell: None,
                sources_tx: Some(sources_tx),
                track_codec: HashMap::new(),
                negotiated_codecs: HashMap::new(),
                track_direction: HashMap::new(),
                remote_closed: false,
                pending: None,
                command_rx,
                outbox: Outbox::new(CHANNEL_CAPACITY),
                on_offer: Box::new(on_offer),
                on_keyframe_request: Box::new(on_keyframe_request),
            },
            WebRtcHandle {
                next_id,
                command_tx,
                sources_rx,
            },
        )
    }

    /// Makes the sink of a track a muxer asked for, now its `Mid` exists,
    /// and hands the muxer its sinks once every one of its tracks has one.
    fn attach_outgoing(&mut self, id: TrackId, mid: Mid, kind: MediaKind) {
        pp_info!(
            self,
            "track attached: id={id:?}, mid={mid}, kind={kind:?}, sending"
        );
        self.track_direction.insert(mid, Direction::SendOnly);
        let codec = self.track_codec.remove(&id);
        // Nothing is negotiated before the answer: until it narrows the
        // list, the codec the track is offered as is what its muxer may
        // declare, and what a packet sent meanwhile is let go of as.
        let mut codecs = self.codecs_for_mid(mid);
        if codecs.is_empty() {
            codecs.extend(codec);
        }
        let negotiated_codecs = Arc::new(Mutex::new(codecs));
        self.negotiated_codecs
            .insert(mid, negotiated_codecs.clone());
        let sink = WebRtcTrackSink::new(id, kind, codec, negotiated_codecs, self.outbox.clone());
        let Some(at) = self
            .opening
            .iter()
            .position(|opening| opening.ids.contains(&id))
        else {
            return;
        };
        let opening = &mut self.opening[at];
        if let Some(slot) = opening.ids.iter().position(|known| *known == id) {
            opening.sinks[slot] = Some(sink);
        }
        if opening.sinks.iter().all(Option::is_some) {
            let opening = self.opening.remove(at);
            let sinks = opening.sinks.into_iter().flatten().collect();
            let _ = opening.reply.send(Ok(sinks));
        }
    }

    /// Takes in a track the remote peer sends on: its packets go to an
    /// inbox of its own, which goes out with the rest of its batch.
    fn attach_incoming(&mut self, mid: Mid, kind: MediaKind, direction: Direction) {
        pp_info!(
            self,
            "track attached: mid={mid}, kind={kind:?}, direction={direction:?}"
        );
        self.track_direction.insert(mid, direction);
        let negotiated_codecs = Arc::new(Mutex::new(self.codecs_for_mid(mid)));
        self.negotiated_codecs.insert(mid, negotiated_codecs);
        let bell = self.arriving_bell.get_or_insert_with(Bell::new).clone();
        let inbox = Inbox::sharing(CHANNEL_CAPACITY, &bell);
        let (described_tx, described) = bounded(1);
        self.tracks_in.insert(
            mid,
            TrackInState {
                inbox: inbox.clone(),
                stream_info_tx: described_tx,
                stream_info_probe: StreamInfoProbe::new(),
                stream_info_sent: false,
            },
        );
        self.arriving.push(Arriving { inbox, described });
    }

    /// Hands out the remote peer's tracks attached since the last batch, as
    /// one: everything str0m has said since has been taken in.
    pub(super) fn hand_out_arrivals(&mut self) {
        if self.arriving.is_empty() {
            return;
        }
        let arriving = std::mem::take(&mut self.arriving);
        self.arriving_bell = None;
        pp_info!(
            self,
            "{} track(s) from the remote peer handed out",
            arriving.len()
        );
        if let Some(tx) = &self.sources_tx {
            let _ = tx.send(arriving);
        }
    }

    fn codecs_for_mid(&mut self, mid: Mid) -> Vec<Codec> {
        let Some(writer) = self.rtc.writer(mid) else {
            return Vec::new();
        };
        let mut codecs = Vec::new();
        for codec in writer.payload_params().map(|params| params.spec().codec) {
            if !codecs.contains(&codec) {
                codecs.push(codec);
            }
        }
        codecs
    }

    fn refresh_codecs(&mut self, mid: Mid) {
        let codecs = self.codecs_for_mid(mid);
        if let Some(shared) = self.negotiated_codecs.get(&mid) {
            *shared.lock().unwrap() = codecs;
        }
    }

    fn apply_command(&mut self, cmd: Command) -> Result<()> {
        match cmd {
            Command::OpenTracks(tracks, reply) => {
                pp_info!(self, "{} track(s) to send requested", tracks.len());
                let ids: Vec<TrackId> = tracks.iter().map(|track| track.id).collect();
                self.opening.push(Opening {
                    sinks: ids.iter().map(|_| None).collect(),
                    ids,
                    reply,
                });
                for OpenTrack {
                    id,
                    kind,
                    codec,
                    name,
                } in tracks
                {
                    self.track_codec.insert(id, codec);
                    self.track_names.insert(id, name);
                    // A section the remote peer asked to be sent to, of the
                    // track's kind, that took its codec: sent on as it is.
                    let candidates: Vec<(usize, Mid)> = self
                        .requested
                        .iter()
                        .enumerate()
                        .filter(|(_, (_, of))| *of == kind)
                        .map(|(at, (mid, _))| (at, *mid))
                        .collect();
                    let asked = candidates
                        .into_iter()
                        .find(|&(_, mid)| self.codecs_for_mid(mid).contains(&codec))
                        .map(|(at, _)| at);
                    match asked {
                        Some(at) => {
                            let (mid, _) = self.requested.remove(at);
                            pp_info!(
                                self,
                                "track {id:?} sent on mid={mid}, which the remote peer asked for"
                            );
                            self.tracks_out.insert(id, TrackOutState::Open(mid));
                            self.attach_outgoing(id, mid, kind);
                        }
                        None => {
                            self.tracks_out
                                .insert(id, TrackOutState::ToOpen(kind, Direction::SendOnly));
                        }
                    }
                }
            }
            Command::SetAnswer(answer) => {
                let Some((pending, ids)) = self.pending.take() else {
                    return Ok(());
                };
                self.rtc
                    .sdp_api()
                    .accept_answer(pending, answer)
                    .inspect_err(|error| pp_error!(self, "accept_answer failed: {error}"))
                    .map_err(WebRtcError::from)?;
                pp_info!(self, "renegotiation complete: {} track(s)", ids.len());
                for id in ids {
                    if let Some(state @ TrackOutState::Negotiating(_)) = self.tracks_out.get(&id) {
                        let mid = state.mid().expect("Negotiating always carries a Mid");
                        self.tracks_out.insert(id, TrackOutState::Open(mid));
                        self.refresh_codecs(mid);
                    }
                }
            }
            Command::AcceptOffer(offer, reply) => {
                let result = self
                    .rtc
                    .sdp_api()
                    .accept_offer(offer)
                    .inspect_err(|error| pp_error!(self, "accept_offer failed: {error}"))
                    .map_err(WebRtcError::from);
                if result.is_ok() {
                    pp_info!(self, "accepted remote offer");
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
        let mut to_open: Vec<TrackId> = self
            .tracks_out
            .iter()
            .filter(|(_, s)| matches!(s, TrackOutState::ToOpen(..)))
            .map(|(id, _)| *id)
            .collect();
        if to_open.is_empty() {
            return;
        }
        // In the order they were asked for, which numbers them: a muxer's
        // tracks are its streams in order, and the remote peer's source of
        // them has its outputs in the order the offer lists them.
        to_open.sort_by_key(|id| id.0);

        let mut newly_negotiating = Vec::with_capacity(to_open.len());
        let mut api = self.rtc.sdp_api();
        for &id in &to_open {
            let Some(TrackOutState::ToOpen(kind, direction)) = self.tracks_out.get(&id) else {
                continue;
            };
            let (kind, direction) = (*kind, *direction);
            let mid = api.add_media(kind, direction, None, None, None);
            self.tracks_out.insert(id, TrackOutState::Negotiating(mid));
            newly_negotiating.push((id, mid, kind, direction));
        }

        if let Some((offer, pending)) = api.apply() {
            pp_info!(self, "renegotiation started: {} track(s)", to_open.len());
            self.pending = Some((pending, to_open));
            (self.on_offer)(offer);
        }

        // str0m never fires `Event::MediaAdded` for media *this side* just
        // added (see the type docs) — so this is the only place these
        // newly-minted `Mid`s ever reach `attach_outgoing`, unlike the remote
        // side's own `Event::MediaAdded` handling below.
        for (id, mid, kind, _) in newly_negotiating {
            self.attach_outgoing(id, mid, kind);
        }
    }

    /// Writes everything its track sinks have put in the outbox. A
    /// malformed packet is local to its one track and buffer: reported and
    /// dropped without tearing down the connection, matching `Queue`'s
    /// consume-error contract.
    fn write_outbox(&mut self, bus: &Bus) {
        for outgoing in self.outbox.take_all() {
            if let Err(error) = self.write_track(outgoing) {
                bus.post(
                    &self.pp_log,
                    BusEvent::Error {
                        element_type: ElementType::WebRtcPeer,
                        name: self.name.clone(),
                        error,
                    },
                );
            }
        }
    }

    fn write_track(&mut self, outgoing: Outgoing) -> Result<()> {
        let id = outgoing.track;
        let Some(TrackOutState::Open(mid)) = self.tracks_out.get(&id) else {
            // Not open yet (or unknown/never added) — dropped, see
            // `WebRtcHandle::muxer`'s docs.
            return Ok(());
        };
        let mid = *mid;
        connection::write(&mut self.rtc, mid, &outgoing)
            .inspect_err(|error| pp_error!(self, "writing track {id:?} failed: {error}"))?;
        Ok(())
    }

    /// Drains every immediately-available str0m output (retransmits and
    /// events), returning once str0m itself has nothing left to do until
    /// the returned deadline — see [`Outputs`].
    fn drive_until_timeout(&mut self, bus: &Bus) -> Result<Instant> {
        let mut outputs = Outputs::new();
        loop {
            let next = outputs
                .next(&mut self.rtc)
                .inspect_err(|error| pp_error!(self, "poll_output failed: {error}"))
                .map_err(WebRtcError::from)?;
            match next {
                Next::Wait(deadline) => return Ok(deadline),
                Next::Transmit(t) => {
                    // A single failed send (e.g. transient ICMP unreachable)
                    // isn't fatal to the whole connection — str0m's own
                    // retransmit/timeout logic handles loss.
                    let _ = self.socket.send_to(&t.contents, t.destination);
                }
                Next::Event(event) => self.handle_event(event, bus),
            }
        }
    }

    /// Tells the remote peer this connection is over, instead of simply
    /// going quiet.
    ///
    /// Without this a stopped peer is indistinguishable from a crashed or
    /// unplugged one: nothing is sent, and the remote only finds out when
    /// its own ICE checks time out — or, on a path that happens to return
    /// ICMP port-unreachable, when a `recv_from` fails. Neither is a
    /// contract; both are accidents of the network in between.
    ///
    /// `Rtc::close` queues a DTLS `close_notify` that only leaves via
    /// `poll_output`, and str0m requires draining until it reports a
    /// timeout, so this drives the loop one more time rather than
    /// returning straight away. Best-effort throughout: this runs while
    /// shutting down, so a send that fails has nowhere left to be
    /// reported and nothing left to retry into.
    fn close_connection(&mut self, bus: &Bus) {
        if !self.rtc.is_alive() {
            return; // already gone — nothing to notify, nothing to drain
        }
        if let Err(error) = self.rtc.close() {
            pp_warn!(self, "close failed, ending without notifying: {error}");
            return;
        }
        if let Err(error) = self.drive_until_timeout(bus) {
            pp_warn!(self, "draining close_notify failed: {error}");
            return;
        }
        pp_info!(self, "event=close phase=completed outcome=ok");
    }

    pub(super) fn handle_event(&mut self, event: Event, bus: &Bus) {
        match event {
            Event::MediaAdded(added) => {
                // Only reached for media the *remote* peer added — str0m
                // never reports this side's own. What it sends on is taken
                // in; one it asks to be sent to has nothing here to send.
                if added.direction.is_receiving() {
                    self.attach_incoming(added.mid, added.kind, added.direction);
                } else if added.direction.is_sending() {
                    pp_info!(
                        self,
                        "the remote peer added mid={} ({:?}) to be sent to, for a muxer's track",
                        added.mid,
                        added.kind
                    );
                    self.requested.push((added.mid, added.kind));
                } else {
                    pp_info!(
                        self,
                        "the remote peer added mid={}, inactive; left alone",
                        added.mid
                    );
                }
            }
            Event::MediaData(data) => {
                if let Some(track) = self.tracks_in.get_mut(&data.mid) {
                    let codec = data.params.spec();
                    if !track.stream_info_sent
                        && let Some(info) = track.stream_info_probe.observe(codec, &data.data)
                    {
                        // Signal before queueing the payload that completed
                        // the information. Earlier payloads are already in
                        // `data_tx`; all of them stay buffered while the caller
                        // builds its downstream graph.
                        track.stream_info_sent = track.stream_info_tx.try_send(info).is_ok();
                    }

                    let packet = connection::received(&data);
                    match track
                        .inbox
                        .push(MediaBuffer::Packet(Arc::new(packet).into()))
                    {
                        Delivered::Taken => {}
                        Delivered::Full => {
                            // This track's source (or whatever it feeds)
                            // isn't keeping up — drop the newest
                            // buffer rather than let this grow unbounded
                            // (see `CHANNEL_CAPACITY`'s docs).
                            bus.post(
                                &self.pp_log,
                                BusEvent::Dropped {
                                    element_type: ElementType::WebRtcPeer,
                                    name: self.name.clone(),
                                },
                            );
                        }
                        Delivered::Gone => {
                            // This track's source is gone (its own
                            // `Pipeline` finished) — stop trying to feed it.
                            self.tracks_in.remove(&data.mid);
                        }
                    }
                }
            }
            Event::KeyframeRequest(req) => {
                if let Some(name) = self
                    .tracks_out
                    .iter()
                    .find(|(_, s)| s.mid() == Some(req.mid))
                    .and_then(|(id, _)| self.track_names.get(id).cloned())
                {
                    pp_info!(self, "keyframe requested: track={name}, mid={}", req.mid);
                    (self.on_keyframe_request)(&name);
                }
            }
            Event::Connected => {
                pp_info!(self, "ICE+DTLS connected");
            }
            Event::IceConnectionStateChange(state) => {
                pp_info!(self, "ICE connection state: {state:?}");
            }
            Event::MediaChanged(changed) => {
                pp_info!(
                    self,
                    "media changed: mid={}, direction={:?}",
                    changed.mid,
                    changed.direction
                );
                self.refresh_codecs(changed.mid);
                // A track goes the one way it attached with — a muxer's sink
                // or a source's output, already the caller's — and there is
                // no way to hand out a half that did not exist then, or to
                // take back one that no longer works.
                // So a direction the remote peer actually changed makes what
                // the caller holds wrong, and saying so is all this element
                // can do about it. Recovering means tearing the track down
                // and adding a new one.
                //
                // `MediaChanged` also fires when a renegotiation re-states
                // the direction already in force; comparing keeps that from
                // being reported as a change.
                if let Some(&from) = self.track_direction.get(&changed.mid)
                    && from != changed.direction
                {
                    self.track_direction.insert(changed.mid, changed.direction);
                    bus.post(
                        &self.pp_log,
                        BusEvent::Error {
                            element_type: ElementType::WebRtcPeer,
                            name: self.name.clone(),
                            error: WebRtcError::DirectionChanged {
                                mid: changed.mid,
                                from,
                                to: changed.direction,
                            }
                            .into(),
                        },
                    );
                }
            }
            Event::Closed => {
                pp_info!(self, "event=close phase=remote_received outcome=ok");
                self.remote_closed = true;
            }
            // `Event` is `#[non_exhaustive]` — data channels, stats, etc.
            // are still outside this element's concern for now.
            _ => {}
        }
    }
}

/// Converts a `Packet`'s `(pts, time_base)` into the `MediaTime` str0m
/// expects for [`str0m::media::Writer::write`]. `MediaTime` is
/// numer/denom *seconds* (str0m rebases it to the codec's RTP clock rate
/// internally), but an FFmpeg `time_base` is numer/denom *seconds per
/// tick* — so the elapsed time is `pts * numerator / denominator`, not
/// `pts / denominator`. Most time bases in this codebase have numerator 1
/// (e.g. `1/90_000`), which would hide a naive `pts / denominator`: an
/// NTSC-style `1001/30_000` time base would make the RTP timestamp run
/// ~1001x too fast.
pub(super) fn packet_rtp_time(
    packet: &ffmpeg::Packet,
) -> std::result::Result<MediaTime, WebRtcError> {
    let pts = packet.pts().ok_or(WebRtcError::MissingPacketPts)?;
    let timestamp = MediaTimestamp::try_new(pts, packet.time_base()).map_err(
        |InvalidTimeBase {
             numerator,
             denominator,
         }| WebRtcError::InvalidPacketTimeBase {
            numerator,
            denominator,
        },
    )?;
    to_str0m_media_time(timestamp)
}

/// Converts a validated `(pts, time_base)` into the `MediaTime` str0m
/// expects for [`str0m::media::Writer::write`]. `MediaTime` is numer/denom
/// *seconds* (str0m rebases it to the codec's RTP clock rate internally),
/// but an FFmpeg `time_base` is numer/denom *seconds per tick* — so the
/// elapsed time is `pts * numerator / denominator`, not `pts /
/// denominator`. This keeps that exact `(pts * numerator, denominator)`
/// rational rather than rescaling to some fixed target base first — a
/// backend-specific conversion, so it lives here rather than on
/// `MediaTimestamp` itself.
fn to_str0m_media_time(timestamp: MediaTimestamp) -> std::result::Result<MediaTime, WebRtcError> {
    let time_base = timestamp.time_base().get();
    let numerator = time_base.numerator();
    let denominator = time_base.denominator();
    let pts = u64::try_from(timestamp.pts())
        .map_err(|_| WebRtcError::NegativePacketPts(timestamp.pts()))?;
    let frequency = str0m::media::Frequency::new(denominator as u32).ok_or(
        WebRtcError::InvalidPacketTimeBase {
            numerator,
            denominator,
        },
    )?;
    let numer = pts
        .checked_mul(numerator as u64)
        .ok_or(WebRtcError::PacketTimestampOverflow {
            pts,
            numerator,
            denominator,
        })?;
    Ok(MediaTime::new(numer, frequency))
}

impl Drop for WebRtcPeer {
    fn drop(&mut self) {
        self.outbox.close();
    }
}

impl Element for WebRtcPeer {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::WebRtcPeer
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
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
    /// latency for `Stop`/a muxer opening, not unboundedly stuck.
    ///
    /// `stop`/the connection dying both clear `tracks_in` immediately, so
    /// every already-handed-out source sees its inboxes close and ends with
    /// a final `Eos` right away, instead of waiting for this whole
    /// `WebRtcPeer` to be dropped later by whatever owns its
    /// `DriverRunner`. Neither `WebRtcPeer` nor its sources have a
    /// `Pause`/`Seek` concept — see
    /// [`Driver`]'s own docs for why that's not just an oversight: freezing
    /// this loop would starve ICE keepalives/DTLS retransmits, likely
    /// dropping the connection rather than gracefully suspending it.
    fn run(&mut self, stop: &StopReceiver, bus: &Bus) -> Result<()> {
        pp_info!(self, "started");
        let mut buf = vec![0u8; 2000];
        loop {
            while let Ok(cmd) = self.command_rx.try_recv() {
                self.apply_command(cmd)?;
            }
            self.negotiate_if_needed();
            self.write_outbox(bus);

            let deadline = self.drive_until_timeout(bus)?;
            self.hand_out_arrivals();
            if self.remote_closed || !self.rtc.is_alive() || stop.is_stopped() {
                pp_info!(
                    self,
                    "stopped rtc_alive={} remote_closed={}",
                    self.rtc.is_alive(),
                    self.remote_closed
                );
                if !self.remote_closed {
                    self.close_connection(bus);
                }
                self.tracks_in.clear();
                // Nothing more will be handed out, nor opened.
                self.sources_tx = None;
                for opening in self.opening.drain(..) {
                    let _ = opening.reply.send(Err(WebRtcError::Closed));
                }
                return Ok(());
            }

            let wait = deadline
                .saturating_duration_since(Instant::now())
                .min(POLL_INTERVAL)
                .max(Duration::from_millis(1));
            match net::recv_within(&self.socket, &mut buf, wait) {
                Ok((n, source)) => {
                    let Ok(contents) = buf[..n].try_into() else {
                        continue; // not a WebRTC datagram we recognize — ignore
                    };
                    let destination = self
                        .socket
                        .local_addr()
                        .inspect_err(|error| pp_error!(self, "local_addr failed: {error}"))
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
                        .inspect_err(|error| {
                            pp_error!(self, "handle_input(Receive) failed: {error}")
                        })
                        .map_err(WebRtcError::from)?;
                }
                Err(e) if e.kind() == std::io::ErrorKind::TimedOut => {
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
                        .inspect_err(|error| {
                            pp_error!(self, "handle_input(Timeout) failed: {error}")
                        })
                        .map_err(WebRtcError::from)?;
                }
                Err(e) => {
                    pp_error!(self, "recv_from failed: {e}");
                    return Err(WebRtcError::from(e).into());
                }
            }
        }
    }
}
