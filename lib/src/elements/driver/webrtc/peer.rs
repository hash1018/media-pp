use std::{
    collections::HashMap,
    net::UdpSocket,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use crossbeam_channel::{Receiver, Sender, TrySendError, bounded, unbounded};
use ffmpeg_next as ffmpeg;
use rust_hlog::{HLog, herror, hinfo};
use str0m::{
    Event, Input, Output, Rtc,
    change::{SdpOffer, SdpPendingOffer},
    format::Codec,
    media::{MediaKind, MediaTime, Mid},
    net::{Protocol, Receive},
};

use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    driver::{Driver, StopReceiver},
    element::{Element, ElementType, element_hlog},
    error::Result,
};

use super::{
    command::{Command, TrackId, TrackOutState, WebRtcError},
    track::{WebRtcHandle, WebRtcTrackSink, WebRtcTrackSource},
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
        let reply = WebRtcTrackSink::new(id, self.command_tx.clone());
        let (tx, rx) = bounded(CHANNEL_CAPACITY);
        let codec = Arc::new(Mutex::new(None));
        self.tracks_in.insert(mid, (tx, codec.clone()));
        let source = WebRtcTrackSource::new(format!("webrtc-track-{}-in", id.0), rx, codec);
        let _ = self.new_track_tx.send((id, mid, kind, reply, source));
    }

    fn apply_command(&mut self, cmd: Command, bus: &Bus) -> Result<()> {
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
            Command::Push(id, buf) => {
                // A malformed media packet is local to this one track and
                // buffer. Report and drop it without tearing down the
                // entire live WebRTC connection, matching Queue's
                // consume-error contract.
                if let Err(error) = self.write_track(id, buf) {
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
        let rtp_time = packet_rtp_time(&packet)?;
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
    let time_base = packet.time_base();
    let numerator = time_base.numerator();
    let denominator = time_base.denominator();
    if numerator <= 0 || denominator <= 0 {
        return Err(WebRtcError::InvalidPacketTimeBase {
            numerator,
            denominator,
        });
    }
    let pts = packet.pts().ok_or(WebRtcError::MissingPacketPts)?;
    let pts = u64::try_from(pts).map_err(|_| WebRtcError::NegativePacketPts(pts))?;
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
                self.apply_command(cmd, bus)?;
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
