use std::{
    collections::HashMap,
    net::UdpSocket,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use crate::pp_log::{PpLog, pp_error, pp_info, pp_warn};
use crossbeam_channel::{Receiver, Sender, TrySendError, bounded, unbounded};
use ffmpeg_next as ffmpeg;
use str0m::{
    Event, Input, Output, Rtc,
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

use super::{
    command::{Command, TrackId, TrackOutState, WebRtcError},
    stream_info::{StreamInfoProbe, WebRtcStreamInfo},
    track::{AttachedTrack, TrackEndpoints, WebRtcHandle, WebRtcTrackSink, WebRtcTrackSource},
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
/// The [`Driver`] — owns the [`Rtc`] session and its [`UdpSocket`], and
/// drives str0m's sans-I/O poll loop on the dedicated thread
/// [`crate::driver::DriverRunner::run`] gives it. Not a
/// [`crate::element::SourceElement`]/[`crate::element::Source`]: it has no `src_pads()`
/// dataflow graph of its own — see [`Driver`]'s own docs for why a
/// connection with dynamically-appearing, independently bidirectional
/// tracks doesn't fit that shape. Whatever it produces or consumes flows
/// through the separate [`WebRtcTrackSink`]/[`WebRtcTrackSource`] pairs it
/// mints per track instead (see below).
///
/// `rtc`/`socket` must already be connected: the initial SDP offer/answer
/// and ICE candidate setup happen via str0m directly, in the caller's own
/// code, *before* [`WebRtcPeer::new`]. `WebRtcPeer` only takes over after
/// signaling has established the connection; it does not provide a signaling
/// server itself.
///
/// Every track — whether it's one this side requested via
/// [`WebRtcHandle::add_track`] or one the remote peer added (`str0m`'s
/// `Event::MediaAdded`, which — critically — *never fires for a track this
/// side added itself*) — is attached the same way, the moment its `Mid`
/// exists: a [`TrackEndpoints`] is minted from the track's negotiated
/// direction and handed out through [`WebRtcHandle::next_track`], no
/// closure required. A single `Direction::SendRecv` track therefore needs
/// exactly one [`WebRtcHandle::add_track`] call (on either side) and one
/// `next_track()` on *each* side — no separate outbound API, and no
/// special-casing for which side happened to originate it.
///
/// What that direction allows is decided once, here, rather than left to
/// the caller to observe: a `SendOnly` track yields a [`WebRtcTrackSink`]
/// and no source, a `RecvOnly` one a [`WebRtcTrackSource`] and no sink.
/// The endpoints are never re-issued, so a remote peer that renegotiates
/// a different direction afterwards invalidates what the caller is
/// holding; that is reported as
/// [`WebRtcError::DirectionChanged`] on the [`Bus`] rather than quietly
/// changing behavior underneath it.
///
/// Attachment itself is the same idea as
/// [`crate::elements::TeeHandle::attach`]'s dynamic attachment, just
/// without `Tee`'s `Mutex` (nothing but this one thread ever touches
/// `tracks_in`).
pub struct WebRtcPeer {
    pp_log: PpLog,
    name: Arc<str>,
    rtc: Rtc,
    socket: UdpSocket,
    /// Where inbound data for each attached track goes: just a plain
    /// `Sender`, not a `Box<dyn Sink>` — the matching `Receiver` lives
    /// inside that track's own [`WebRtcTrackSource`], driven by *its own*
    /// `Pipeline` on its own thread, so nothing here needs to know about
    /// `ControlMsg` at all. Its codec cell is the same
    /// `WebRtcTrackSource`'s [`WebRtcTrackSource::codec`] cell — written
    /// here (from `Event::MediaData`), read there, from whatever thread the
    /// caller checks it on. A separate one-slot channel confirms enough actual
    /// payload information for `wait_stream_info` (including received H.264
    /// SPS/PPS); both are shared across threads, while the map itself still
    /// isn't (see below).
    tracks_in: HashMap<Mid, TrackInState>,
    tracks_out: HashMap<TrackId, TrackOutState>,
    /// Pending outbound codec selections for locally-requested tracks. Each entry
    /// moves into that track's [`WebRtcTrackSink`] when it attaches; a
    /// remotely-added track has no such declaration and its caller supplies
    /// one through [`WebRtcTrackSink::set_source_parameters`] — or
    /// [`WebRtcTrackSink::set_codec`], with no parameters to hand — validated
    /// against the endpoint's negotiated codec list before pushing packets.
    track_codec: HashMap<TrackId, Codec>,
    /// The currently negotiated codec families for each attached media
    /// section, shared with both endpoints. A locally-created track is
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
    new_track_tx: Sender<AttachedTrack>,
    on_offer: Box<dyn FnMut(SdpOffer) + Send>,
    on_keyframe_request: Box<dyn FnMut(TrackId) + Send>,
}

struct TrackInState {
    data_tx: Sender<MediaBuffer>,
    codec: Arc<Mutex<Option<Codec>>>,
    stream_info_tx: Sender<WebRtcStreamInfo>,
    stream_info_probe: StreamInfoProbe,
    stream_info_sent: bool,
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
        let pp_log = element_pp_log(ElementType::WebRtcPeer, &name, None);
        pp_info!(
            pp_log: &pp_log,
            "created: local_addr={:?}",
            socket.local_addr()
        );
        let (command_tx, command_rx) = bounded(CHANNEL_CAPACITY);
        let (new_track_tx, new_track_rx) = unbounded();
        let next_id = Arc::new(AtomicU64::new(0));
        (
            Self {
                name,
                pp_log,
                rtc,
                socket,
                tracks_in: HashMap::new(),
                tracks_out: HashMap::new(),
                track_codec: HashMap::new(),
                negotiated_codecs: HashMap::new(),
                track_direction: HashMap::new(),
                remote_closed: false,
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
    pub(super) fn attach_track(
        &mut self,
        id: TrackId,
        mid: Mid,
        kind: MediaKind,
        direction: Direction,
    ) {
        pp_info!(
            self,
            "track attached: id={id:?}, mid={mid}, kind={kind:?}, direction={direction:?}"
        );
        self.track_direction.insert(mid, direction);
        let negotiated_codecs = Arc::new(Mutex::new(self.codecs_for_mid(mid)));
        self.negotiated_codecs
            .insert(mid, negotiated_codecs.clone());

        // Only build the inbound half when the direction can actually
        // deliver on it: `tracks_in` is what `Event::MediaData` routes
        // through, so leaving a receive-less track out of it is also what
        // makes a stray inbound packet on it visibly a dropped one rather
        // than something feeding a source nobody was given.
        let outbound_codec = self.track_codec.remove(&id);
        let sink = direction.is_sending().then(|| {
            WebRtcTrackSink::new(
                id,
                kind,
                outbound_codec,
                negotiated_codecs.clone(),
                self.command_tx.clone(),
            )
        });
        let source = direction.is_receiving().then(|| {
            let (tx, rx) = bounded(CHANNEL_CAPACITY);
            let (stream_info_tx, stream_info_rx) = bounded(1);
            let codec = Arc::new(Mutex::new(None));
            self.tracks_in.insert(
                mid,
                TrackInState {
                    data_tx: tx,
                    codec: codec.clone(),
                    stream_info_tx,
                    stream_info_probe: StreamInfoProbe::new(),
                    stream_info_sent: false,
                },
            );
            WebRtcTrackSource::new(
                id,
                kind,
                format!("webrtc-track-{}-in", id.0),
                rx,
                codec,
                negotiated_codecs.clone(),
                stream_info_rx,
            )
        });

        let endpoints = match (sink, source) {
            (Some(sink), Some(source)) => TrackEndpoints::SendRecv(sink, source),
            (Some(sink), None) => TrackEndpoints::Send(sink),
            (None, Some(source)) => TrackEndpoints::Recv(source),
            (None, None) => TrackEndpoints::Inactive,
        };
        let _ = self.new_track_tx.send(AttachedTrack {
            id,
            mid,
            kind,
            endpoints,
        });
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

    fn apply_command(&mut self, cmd: Command, bus: &Bus) -> Result<()> {
        match cmd {
            Command::AddTrack(id, kind, direction, codec) => {
                pp_info!(
                    self,
                    "add_track requested: id={id:?}, kind={kind:?}, direction={direction:?}, codec={codec:?}"
                );
                self.tracks_out
                    .insert(id, TrackOutState::ToOpen(kind, direction));
                self.track_codec.insert(id, codec);
            }
            Command::Push(id, codec, buf) => {
                // A malformed media packet is local to this one track and
                // buffer. Report and drop it without tearing down the
                // entire live WebRTC connection, matching Queue's
                // consume-error contract.
                if let Err(error) = self.write_track(id, codec, buf) {
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
            newly_negotiating.push((id, mid, kind, direction));
        }

        if let Some((offer, pending)) = api.apply() {
            pp_info!(self, "renegotiation started: {} track(s)", to_open.len());
            self.pending = Some((pending, to_open));
            (self.on_offer)(offer);
        }

        // str0m never fires `Event::MediaAdded` for media *this side* just
        // added (see the type docs) — so this is the only place these
        // newly-minted `Mid`s ever reach `attach_track`, unlike the remote
        // side's own `Event::MediaAdded` handling below.
        for (id, mid, kind, direction) in newly_negotiating {
            self.attach_track(id, mid, kind, direction);
        }
    }

    fn write_track(&mut self, id: TrackId, codec: Option<Codec>, buf: MediaBuffer) -> Result<()> {
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
        let codec = codec.ok_or(WebRtcError::OutboundCodecNotDeclared(id))?;
        // Never guess the first negotiated codec: str0m packetizes whatever
        // bytes it receives under the selected payload type, so guessing VP8
        // for an H.264 packet creates a valid-looking but mislabeled stream.
        let mut negotiated = Vec::new();
        let mut pt = None;
        for params in writer.payload_params() {
            let candidate = params.spec().codec;
            if !negotiated.contains(&candidate) {
                negotiated.push(candidate);
            }
            if candidate == codec && pt.is_none() {
                pt = Some(params.pt());
            }
        }
        let Some(pt) = pt else {
            return Err(WebRtcError::OutboundCodecNotNegotiated {
                track_id: id,
                codec,
                negotiated,
            }
            .into());
        };
        let data = packet.data().unwrap_or(&[]).to_vec();
        let rtp_time = packet_rtp_time(&packet)?;
        writer
            .write(pt, Instant::now(), rtp_time, data)
            .inspect_err(|error| pp_error!(self, "writer.write failed: {error}"))
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
                .inspect_err(|error| pp_error!(self, "poll_output failed: {error}"))
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
                // Only reached for media the *remote* peer added (see the
                // type docs) — by definition already fully negotiated by
                // the time we see this, so `Open` immediately: unlike a
                // locally-requested track, there's no answer left to wait
                // for before a `WebRtcTrackSink` bound to it can actually
                // send.
                let id = TrackId(self.next_id.fetch_add(1, Ordering::Relaxed));
                self.tracks_out.insert(id, TrackOutState::Open(added.mid));
                self.attach_track(id, added.mid, added.kind, added.direction);
            }
            Event::MediaData(data) => {
                if let Some(track) = self.tracks_in.get_mut(&data.mid) {
                    // Every packet, not just the first: cheap (one lock),
                    // and correct if the remote side ever actually changes
                    // codec mid-stream (rare, but the payload type is free
                    // to vary packet-to-packet — see `WebRtcTrackSource::
                    // codec`'s own docs for why this can't be pinned down
                    // any earlier than "whatever the last packet said").
                    let codec = data.params.spec();
                    track.codec.lock().unwrap().replace(codec.codec);
                    if !track.stream_info_sent
                        && let Some(info) = track.stream_info_probe.observe(codec, &data.data)
                    {
                        // Signal before queueing the payload that completed
                        // the information. Earlier payloads are already in
                        // `data_tx`; all of them stay buffered while the caller
                        // builds its downstream graph.
                        track.stream_info_sent = track.stream_info_tx.try_send(info).is_ok();
                    }

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
                    match track
                        .data_tx
                        .try_send(MediaBuffer::Packet(Arc::new(packet)))
                    {
                        Ok(()) => {}
                        Err(TrySendError::Full(_)) => {
                            // This track's `WebRtcTrackSource` (or whatever
                            // it feeds) isn't keeping up — drop the newest
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
                    pp_info!(self, "keyframe requested: id={id:?}, mid={}", req.mid);
                    (self.on_keyframe_request)(id);
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
                // A track's endpoints are minted once, from the direction it
                // attached with, and `next_track` has already handed them to
                // the caller — there is no way to hand out a half that did
                // not exist then, or to take back one that no longer works.
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
        pp_info!(self, "started");
        let mut buf = vec![0u8; 2000];
        loop {
            while let Ok(cmd) = self.command_rx.try_recv() {
                self.apply_command(cmd, bus)?;
            }
            self.negotiate_if_needed();

            let deadline = self.drive_until_timeout(bus)?;
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
                return Ok(());
            }

            let wait = deadline
                .saturating_duration_since(Instant::now())
                .min(POLL_INTERVAL)
                .max(Duration::from_millis(1));
            self.socket
                .set_read_timeout(Some(wait))
                .inspect_err(|error| pp_error!(self, "set_read_timeout failed: {error}"))
                .map_err(WebRtcError::from)?;

            match self.socket.recv_from(&mut buf) {
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
