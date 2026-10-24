//! One session of a WebRTC server: a player or a publisher — its `str0m`
//! connection, and the driven pipeline that feeds it or that it feeds.
//!
//! A player is a pipeline `BroadcastSource → WebRtcTrackSink`, a sink for
//! each of the broadcast's tracks the player takes, writing into the
//! session's outbox; a publisher a pipeline `WebRtcTrackSource →
//! BroadcastSink` for each of its tracks, reading what the connection puts
//! in their inboxes. The session polls them on the server's thread, as what
//! they wait on wakes them, and is left the transport: the connection, its
//! socket, and when a player is sent each packet.

use std::collections::HashMap;
use std::net::UdpSocket;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Wake, Waker};
use std::time::{Duration, Instant};

use ffmpeg_next as ffmpeg;
use str0m::change::SdpOffer;
use str0m::channel::ChannelId;
use str0m::format::Codec;
use str0m::media::{MediaData, MediaKind, Mid};
use str0m::{Candidate, Event, IceConnectionState, Input, Rtc};

use crate::buffer::MediaBuffer;
use crate::bus::BusEvent;
use crate::elements::{Broadcast, BroadcastSource, StreamInfo};
use crate::pipeline::{Driven, DrivenPipeline, DrivenPipelineHandle};
use crate::pp_log::{PpLog, pp_debug, pp_info};

use super::super::command::TrackId;
use super::super::connection::{self, Next, Outputs};
use super::super::inbox::{Delivered, Inbox};
use super::super::outbox::{Outbox, Outgoing};
use super::super::stream_info::{StreamInfoProbe, str0m_codec};
use super::super::track::{WebRtcTrackSink, WebRtcTrackSource};
use super::{SessionId, SessionRole, SessionStats, Shared, WebRtcServerError, WebRtcServerEvent};

/// How much faster than it plays a player is sent what the broadcast held
/// when it connected: twice, so it catches up in as long as that was,
/// without a burst its socket cannot hold.
const CATCH_UP: f64 = 2.0;

/// How many times, in one turn of the server's loop, a player is written a
/// packet and what it makes sent — so that one player catching up cannot
/// keep the others waiting.
///
/// One packet a time: str0m keeps what it is to send in a ring of a
/// thousand RTP packets a video track, and what is written past that before
/// it is sent is written over — a few large pictures written together were
/// lost whole.
const ROUNDS: usize = 256;

/// How long a session may take to connect after its offer is answered.
const CONNECT_WITHIN: Duration = Duration::from_secs(30);

/// How long a publisher's packets are held while its tracks say what they
/// are, before what has said so is published without the rest.
const HOLD_FOR: Duration = Duration::from_secs(10);
const HOLD_AT_MOST: usize = 4096;

/// How many packets a player's outbox keeps: what it is sent of the
/// broadcast's keyframe interval as it joins, at twice its speed, with
/// room. A player further behind than this goes on from a keyframe.
const OUTBOX: usize = 4096;

/// Whether `broadcast` has anything a WebRTC player can be sent.
pub(crate) fn playable(broadcast: &Broadcast) -> bool {
    !broadcast.is_closed()
        && broadcast.tracks().is_some_and(|tracks| {
            tracks
                .iter()
                .any(|track| str0m_codec(track.parameters().id()).is_some())
        })
}

/// How often a session's connection measures it.
const STATS_EVERY: Duration = Duration::from_secs(1);

/// One session.
pub(crate) struct Session {
    id: SessionId,
    path: String,
    rtc: Rtc,
    broadcast: Broadcast,
    /// When str0m next asks for the time.
    deadline: Instant,
    opened: Instant,
    connected: bool,
    /// The media the client offered, as `str0m` added it.
    media: Vec<(Mid, MediaKind)>,
    /// The client's data channels, by label.
    channels: HashMap<String, ChannelId>,
    role: Role,
    /// Set by what the session's pipelines wait on, which their polls are
    /// handed `waker` for.
    woken: Arc<Woken>,
    waker: Waker,
    /// When it next wants driving, as its last drive said — see
    /// [`Self::due`].
    next: Instant,
    /// Whether something was put in since it was last driven: a datagram,
    /// something to send on a data channel.
    stirred: bool,
    /// Its figures, as last measured; and each medium's requests so far,
    /// resends and keyframes, which the figures add up.
    stats: SessionStats,
    requests: HashMap<Mid, (u64, u64)>,
    pp_log: PpLog,
}

// SAFETY: everything a session holds is `Send` but the codec parameters
// inside a publisher's probes and track formats, which are its own copies,
// used only through `&mut self` on the server's one thread.
unsafe impl Send for Session {}

/// Whether a session's pipelines have been woken since they were polled.
struct Woken(AtomicBool);

impl Wake for Woken {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.store(true, Ordering::Release);
    }
}

/// One of a session's pipelines, and when it asked to be polled again.
struct Driving {
    pipeline: DrivenPipeline,
    handle: DrivenPipelineHandle,
    at: Option<Instant>,
}

impl Driving {
    /// A pipeline not polled yet: polled at the session's next poll.
    fn new(pipeline: DrivenPipeline) -> Self {
        Self {
            handle: pipeline.handle(),
            pipeline,
            at: Some(Instant::now()),
        }
    }

    /// Polls it where it was woken or its time has come: whether it has
    /// finished. When it next wants polling goes into `next`.
    fn poll(
        &mut self,
        woken: bool,
        now: Instant,
        cx: &mut Context<'_>,
        next: &mut Instant,
    ) -> bool {
        if woken || self.at.is_some_and(|at| at <= now) {
            match self.pipeline.poll(cx) {
                Driven::Finished => return true,
                Driven::Pending { at } => self.at = at,
            }
        }
        if let Some(at) = self.at {
            *next = (*next).min(at.max(now));
        }
        false
    }

    /// What its elements reported since it was last asked. Read every poll,
    /// so that nothing piles up on a bus nobody else reads; the log has
    /// each already.
    fn reported(&self) -> impl Iterator<Item = BusEvent> + '_ {
        std::iter::from_fn(|| self.handle.bus().try_recv())
    }
}

enum Role {
    Play(Box<Playing>),
    Publish(Box<Publishing>),
}

/// A player's pipeline, and the packets it has made ready to be sent.
struct Playing {
    /// Made once it connects, so it starts at the newest keyframe then —
    /// at the poll after, by when the media it offered has been told.
    pipeline: Option<Driving>,
    connected: bool,
    outbox: Outbox,
    /// Where each of the broadcast's tracks is sent, by its number — the
    /// id its sink writes with — if anywhere.
    mids: Vec<Option<Mid>>,
    /// Where pacing is measured from: when the first packet was sent, and
    /// where it is on the timeline.
    paced_from: Option<(Instant, f64)>,
}

/// A publisher's tracks on their way into its path.
#[derive(Default)]
struct Publishing {
    /// Each medium sent, in the order the publisher offered them.
    tracks: Vec<Published>,
    /// Packets held while the tracks say what they are: the track's number
    /// and the packet.
    held: Vec<(usize, ffmpeg::Packet)>,
    first_packet: Option<Instant>,
    published: bool,
}

/// One medium a publisher sends.
struct Published {
    mid: Mid,
    kind: MediaKind,
    probe: StreamInfoProbe,
    format: Option<StreamInfo>,
    /// Once published: where its packets go, and the pipeline taking them
    /// into the broadcast.
    inbox: Option<Inbox>,
    pipeline: Option<Driving>,
}

impl Session {
    /// A session of `role` on `path`, answering `offer`: the session and
    /// its SDP answer.
    pub(crate) fn open(
        id: SessionId,
        path: &str,
        role: SessionRole,
        broadcast: Broadcast,
        offer: &str,
        candidates: &[Candidate],
        pp_log: PpLog,
    ) -> Result<(Self, String), WebRtcServerError> {
        let offer = SdpOffer::from_sdp_string(offer)
            .map_err(|error| WebRtcServerError::InvalidOffer(error.to_string()))?;
        let now = Instant::now();
        let mut rtc = Rtc::builder()
            .set_ice_lite(true)
            .set_stats_interval(Some(STATS_EVERY))
            .build(now);
        for candidate in candidates {
            rtc.add_local_candidate(candidate.clone());
        }
        let answer = rtc
            .sdp_api()
            .accept_offer(offer)
            .map_err(|error| WebRtcServerError::InvalidOffer(error.to_string()))?;
        let stats = SessionStats {
            session: id,
            path: path.to_owned(),
            role,
            opened: now,
            connected: false,
            remote: None,
            rtt: None,
            loss: None,
            bytes_sent: 0,
            bytes_received: 0,
            resend_requests: 0,
            keyframe_requests: 0,
            fell_behind: 0,
            updated: None,
        };
        let role = match role {
            SessionRole::Play => Role::Play(Box::new(Playing {
                pipeline: None,
                connected: false,
                outbox: Outbox::new(OUTBOX),
                mids: Vec::new(),
                paced_from: None,
            })),
            SessionRole::Publish => Role::Publish(Box::default()),
        };
        let woken = Arc::new(Woken(AtomicBool::new(false)));
        Ok((
            Self {
                id,
                path: path.to_owned(),
                rtc,
                broadcast,
                deadline: now,
                opened: now,
                connected: false,
                media: Vec::new(),
                channels: HashMap::new(),
                role,
                waker: Waker::from(Arc::clone(&woken)),
                woken,
                next: now,
                stirred: true,
                stats,
                requests: HashMap::new(),
                pp_log,
            },
            answer.to_sdp_string(),
        ))
    }

    /// Its figures, as last measured.
    pub(crate) fn stats(&self) -> &SessionStats {
        &self.stats
    }

    /// Puts its figures where the server's handle reads them.
    fn tell_stats(&mut self, shared: &Shared) {
        if let Role::Play(playing) = &self.role {
            self.stats.fell_behind = playing.outbox.fell_behind();
        }
        let (resends, keyframes) = self
            .requests
            .values()
            .fold((0, 0), |(r, k), (resends, keyframes)| {
                (r + resends, k + keyframes)
            });
        self.stats.resend_requests = resends;
        self.stats.keyframe_requests = keyframes;
        shared
            .stats
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(self.id, self.stats.clone());
    }

    pub(crate) fn path(&self) -> &str {
        &self.path
    }

    pub(crate) fn accepts(&self, input: &Input) -> bool {
        self.rtc.accepts(input)
    }

    pub(crate) fn input(&mut self, input: Input) {
        let _ = self.rtc.handle_input(input);
        self.stirred = true;
    }

    /// Whether it is to be driven now: something was put in, what its
    /// pipelines wait on woke them, or the time it asked for has come. A
    /// session with none of these has nothing to do, and the server's loop
    /// passes it by for a look at two values — with hundreds of players,
    /// most of them most of the time.
    pub(crate) fn due(&self, now: Instant) -> bool {
        self.stirred || now >= self.next || self.woken.0.load(Ordering::Acquire)
    }

    /// When it next wants driving, woken or not.
    pub(crate) fn next(&self) -> Instant {
        self.next
    }

    /// Moves the session on to `now`: its pipelines polled, what is due
    /// written, what the connection has to send sent, what happened told.
    /// When it next wants to be driven, or why it has ended.
    pub(crate) fn drive(
        &mut self,
        now: Instant,
        socket: &UdpSocket,
        shared: &Shared,
    ) -> Result<Instant, String> {
        if now >= self.deadline {
            self.rtc
                .handle_input(Input::Timeout(now))
                .map_err(|error| error.to_string())?;
        }
        let mut next = now + Duration::from_secs(1);
        self.poll(now, &mut next)?;
        // What is due is written a packet at a time, and what it makes sent
        // before the next: see `ROUNDS`.
        let mut more = true;
        for _ in 0..ROUNDS {
            let due = self.write_due(now);
            more = due == Some(now);
            if let Some(due) = due
                && !more
            {
                next = next.min(due);
            }
            // str0m sends what was written only once time has moved on
            // past it; without this it waits for the next turn of the loop,
            // and under load piles up there without bound.
            self.rtc
                .handle_input(Input::Timeout(now))
                .map_err(|error| error.to_string())?;
            self.flush(socket, shared, &mut next)?;
            if !more {
                break;
            }
        }
        if more {
            next = now;
        }
        // What the connection's events woke: a publisher's packets, a
        // player's pipeline begun.
        self.poll(now, &mut next)?;
        if !self.rtc.is_alive() {
            return Err("the connection closed".into());
        }
        if !self.connected && now.duration_since(self.opened) > CONNECT_WITHIN {
            return Err("it never connected".into());
        }
        self.next = next;
        self.stirred = false;
        Ok(next)
    }

    /// Polls the session's pipelines that were woken or are due.
    fn poll(&mut self, now: Instant, next: &mut Instant) -> Result<(), String> {
        if let Role::Play(playing) = &self.role
            && playing.connected
            && playing.pipeline.is_none()
        {
            self.begin_playing()?;
        }
        let woken = self.woken.0.swap(false, Ordering::AcqRel);
        let mut cx = Context::from_waker(&self.waker);
        match &mut self.role {
            Role::Play(playing) => {
                let Some(driving) = &mut playing.pipeline else {
                    return Ok(());
                };
                let finished = driving.poll(woken, now, &mut cx, next);
                // A player's pipeline reports nothing it can act on; the
                // log has it.
                driving.reported().for_each(drop);
                if finished {
                    return Err(if self.broadcast.is_closed() {
                        "its broadcast closed".into()
                    } else {
                        "its broadcast's tracks changed".into()
                    });
                }
            }
            Role::Publish(publishing) => {
                for track in &mut publishing.tracks {
                    let Some(driving) = &mut track.pipeline else {
                        continue;
                    };
                    driving.poll(woken, now, &mut cx, next);
                    // The broadcast refusing the publisher — another
                    // published, or it closed — ends the session.
                    for event in driving.reported() {
                        if let BusEvent::Error {
                            error: crate::error::Error::BroadcastError(error),
                            ..
                        } = event
                        {
                            return Err(error.to_string());
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Writes a player the next packet it is due: `Some(now)` where it was
    /// written, when the next one is due where it is not yet, `None` with
    /// nothing to write.
    fn write_due(&mut self, now: Instant) -> Option<Instant> {
        let Role::Play(playing) = &mut self.role else {
            return None;
        };
        let paced_from = &mut playing.paced_from;
        let outgoing = match playing.outbox.take_due(now, |outgoing| {
            let Some(at) = seconds(outgoing) else {
                return now;
            };
            let (began, from) = *paced_from.get_or_insert((now, at));
            began + Duration::from_secs_f64(((at - from) / CATCH_UP).max(0.0))
        }) {
            Ok(outgoing) => outgoing,
            Err(due) => return due,
        };
        if let Some(Some(mid)) = playing.mids.get(outgoing.track.0 as usize) {
            // What fails is one packet's: the next goes on.
            let _ = connection::write(&mut self.rtc, *mid, &outgoing);
        }
        Some(now)
    }

    /// Sends what the connection has to send, and tells what happened —
    /// see [`Outputs`].
    fn flush(
        &mut self,
        socket: &UdpSocket,
        shared: &Shared,
        next: &mut Instant,
    ) -> Result<(), String> {
        let mut outputs = Outputs::new();
        loop {
            match outputs
                .next(&mut self.rtc)
                .map_err(|error| error.to_string())?
            {
                Next::Wait(deadline) => {
                    self.deadline = deadline;
                    *next = (*next).min(deadline);
                    return Ok(());
                }
                Next::Transmit(transmit) => {
                    let _ = socket.send_to(&transmit.contents, transmit.destination);
                }
                Next::Event(event) => self.event(event, shared)?,
            }
        }
    }

    /// Closes the connection, telling the client.
    pub(crate) fn close(&mut self, socket: &UdpSocket) {
        if !self.rtc.is_alive() {
            return;
        }
        let _ = self.rtc.close();
        let mut outputs = Outputs::new();
        while let Ok(next) = outputs.next(&mut self.rtc) {
            match next {
                Next::Transmit(transmit) => {
                    let _ = socket.send_to(&transmit.contents, transmit.destination);
                }
                Next::Wait(_) => break,
                Next::Event(_) => {}
            }
        }
    }

    /// Sends `data` on the data channel `label`.
    pub(crate) fn send(&mut self, label: &str, binary: bool, data: &[u8]) -> Result<(), ()> {
        let id = *self.channels.get(label).ok_or(())?;
        let mut channel = self.rtc.channel(id).ok_or(())?;
        self.stirred = true;
        channel.write(binary, data).map(|_| ()).map_err(|_| ())
    }

    fn event(&mut self, event: Event, shared: &Shared) -> Result<(), String> {
        let tell = |event| {
            let _ = shared.events.0.send(event);
        };
        match event {
            Event::Connected => {
                self.connected = true;
                self.stats.connected = true;
                self.tell_stats(shared);
                pp_info!(pp_log: &self.pp_log, "session {} connected", self.id);
                let role = match &mut self.role {
                    Role::Play(playing) => {
                        playing.connected = true;
                        SessionRole::Play
                    }
                    Role::Publish(_) => SessionRole::Publish,
                };
                tell(WebRtcServerEvent::Connected {
                    session: self.id,
                    path: self.path.clone(),
                    role,
                });
            }
            Event::IceConnectionStateChange(IceConnectionState::Disconnected) => {
                return Err("the client left".into());
            }
            Event::Closed => return Err("the client closed the connection".into()),
            Event::MediaAdded(added) => {
                self.media.push((added.mid, added.kind));
                if let Role::Publish(publishing) = &mut self.role {
                    publishing.tracks.push(Published {
                        mid: added.mid,
                        kind: added.kind,
                        probe: StreamInfoProbe::new(),
                        format: None,
                        inbox: None,
                        pipeline: None,
                    });
                }
            }
            Event::MediaData(data) => {
                if let Role::Publish(publishing) = &mut self.role {
                    publishing.take(&data, self.id, &self.broadcast, &self.pp_log)?;
                }
            }
            Event::ChannelOpen(id, label) => {
                self.channels.insert(label.clone(), id);
                tell(WebRtcServerEvent::ChannelOpen {
                    session: self.id,
                    channel: label,
                });
            }
            Event::ChannelData(data) => {
                let label = self
                    .channels
                    .iter()
                    .find(|(_, id)| **id == data.id)
                    .map(|(label, _)| label.clone())
                    .unwrap_or_default();
                tell(WebRtcServerEvent::Data {
                    session: self.id,
                    channel: label,
                    binary: data.binary,
                    data: data.data,
                });
            }
            Event::ChannelClose(id) => self.channels.retain(|_, open| *open != id),
            Event::PeerStats(peer) => {
                let pair = peer.selected_candidate_pair.as_ref();
                self.stats.remote = pair.map(|pair| pair.remote.addr);
                // RTCP's measure where there is one; a receive-only
                // session has none, and ICE's own keepalives measure it.
                self.stats.rtt = peer
                    .rtt
                    .or_else(|| pair.and_then(|pair| pair.current_round_trip_time));
                self.stats.loss = match self.stats.role {
                    SessionRole::Play => peer.egress_loss_fraction,
                    SessionRole::Publish => peer.ingress_loss_fraction,
                }
                .or(self.stats.loss);
                self.stats.bytes_sent = peer.peer_bytes_tx;
                self.stats.bytes_received = peer.peer_bytes_rx;
                self.stats.updated = Some(peer.timestamp);
                self.tell_stats(shared);
            }
            Event::MediaEgressStats(egress) => {
                self.requests
                    .insert(egress.mid, (egress.nacks, egress.plis + egress.firs));
            }
            Event::MediaIngressStats(ingress) => {
                self.requests
                    .insert(ingress.mid, (ingress.nacks, ingress.plis + ingress.firs));
            }
            _ => {}
        }
        Ok(())
    }

    /// Begins a connected player's pipeline: a reader of its path's
    /// broadcast from its newest keyframe, and a sink for each track the
    /// player takes — the first of each kind in a codec both WebRTC and the
    /// media it offered take.
    fn begin_playing(&mut self) -> Result<(), String> {
        let Role::Play(playing) = &mut self.role else {
            return Ok(());
        };
        // Not a whole group of pictures to catch up on at twice the stream's
        // speed: past what the connection keeps to send again, a packet lost
        // in it breaks the picture until the next keyframe anyway.
        let (source, tracks) = BroadcastSource::joining_within(
            format!("webrtc-{}-reader", self.id),
            &self.broadcast,
            crate::elements::broadcast::PLAYER_JOIN_WITHIN,
        )
        .map_err(|error| error.to_string())?;
        playing.mids = vec![None; tracks.len()];
        let mut sinks = Vec::new();
        for (index, track) in tracks.iter().enumerate() {
            let Some(codec) = str0m_codec(track.parameters().id()) else {
                continue;
            };
            let kind = match track.kind() {
                ffmpeg::media::Type::Video => MediaKind::Video,
                ffmpeg::media::Type::Audio => MediaKind::Audio,
                _ => continue,
            };
            let Some(&(mid, _)) = self.media.iter().find(|(mid, offered)| {
                *offered == kind
                    && !playing.mids.contains(&Some(*mid))
                    && codecs(&mut self.rtc, *mid).contains(&codec)
            }) else {
                continue;
            };
            let mut sink = WebRtcTrackSink::new(
                TrackId(index as u64),
                kind,
                None,
                Arc::new(Mutex::new(codecs(&mut self.rtc, mid))),
                playing.outbox.clone(),
            );
            if let Err(error) = sink.set_source_parameters(track.parameters()) {
                pp_debug!(
                    pp_log: &self.pp_log,
                    "session {} cannot be sent track {index}: {error}",
                    self.id
                );
                continue;
            }
            playing.mids[index] = Some(mid);
            sinks.push((index, sink));
        }
        if sinks.is_empty() {
            return Err("nothing it offered to take can be sent".into());
        }
        pp_debug!(
            pp_log: &self.pp_log,
            "session {} sends {} of {} tracks",
            self.id,
            sinks.len(),
            tracks.len()
        );
        let (pipeline, ()) =
            DrivenPipeline::new(format!("webrtc-{}", self.id), source, |source, ctx| {
                for (index, sink) in sinks {
                    ctx.attach(source, index, ctx.branch().to(sink)?)?;
                }
                Ok(())
            })
            .map_err(|error| error.to_string())?;
        playing.pipeline = Some(Driving::new(pipeline));
        Ok(())
    }
}

/// The codecs `mid`'s negotiation took, each once.
fn codecs(rtc: &mut Rtc, mid: Mid) -> Vec<Codec> {
    let mut codecs = Vec::new();
    if let Some(writer) = rtc.writer(mid) {
        for codec in writer.payload_params().map(|params| params.spec().codec) {
            if !codecs.contains(&codec) {
                codecs.push(codec);
            }
        }
    }
    codecs
}

/// Where a packet is on its timeline, in seconds: its decoding time where
/// it has one.
fn seconds(outgoing: &Outgoing) -> Option<f64> {
    let packet = &outgoing.packet;
    let ts = packet.dts().or(packet.pts())?;
    Some(ts as f64 * f64::from(packet.time_base()))
}

impl Publishing {
    /// One of the publisher's packets: held until its tracks are published,
    /// then put in its track's inbox.
    fn take(
        &mut self,
        data: &MediaData,
        session: SessionId,
        broadcast: &Broadcast,
        pp_log: &PpLog,
    ) -> Result<(), String> {
        let Some(index) = self.tracks.iter().position(|track| track.mid == data.mid) else {
            return Ok(());
        };
        let packet = connection::received(data);
        if self.published {
            if let Some(inbox) = &self.tracks[index].inbox
                && inbox.push(MediaBuffer::packet(packet)) == Delivered::Full
            {
                pp_debug!(pp_log: pp_log, "session {session} dropped a packet it could not keep up with");
            }
            return Ok(());
        }
        let track = &mut self.tracks[index];
        if track.format.is_none()
            && let Some(info) = track.probe.observe(data.params.spec(), &data.data)
        {
            track.format = info.stream_info().ok();
        }
        self.held.push((index, packet));
        let first = *self.first_packet.get_or_insert_with(Instant::now);
        let all = self.tracks.iter().all(|track| track.format.is_some());
        if all || self.held.len() >= HOLD_AT_MOST || first.elapsed() >= HOLD_FOR {
            self.publish(session, broadcast, pp_log)?;
        }
        Ok(())
    }

    /// Publishes the tracks that have said what they are into the
    /// broadcast, each through a pipeline of its own, and hands them what
    /// was held.
    fn publish(
        &mut self,
        session: SessionId,
        broadcast: &Broadcast,
        pp_log: &PpLog,
    ) -> Result<(), String> {
        let ready: Vec<usize> = (0..self.tracks.len())
            .filter(|&index| self.tracks[index].format.is_some())
            .collect();
        if ready.is_empty() {
            self.held.clear();
            return Err("nothing it sent could be described".into());
        }
        let formats: Vec<StreamInfo> = ready
            .iter()
            .filter_map(|&index| self.tracks[index].format.clone())
            .collect();
        let sinks = broadcast
            .publish(&formats)
            .map_err(|error| error.to_string())?;
        pp_info!(pp_log: pp_log, "published {} tracks into {}", ready.len(), broadcast.name());
        for (index, sink) in ready.into_iter().zip(sinks) {
            let track = &mut self.tracks[index];
            let inbox = Inbox::new(HOLD_AT_MOST);
            let source = WebRtcTrackSource::new(
                track.kind,
                format!("webrtc-{session}-in-{index}"),
                inbox.clone(),
            );
            let (pipeline, ()) = DrivenPipeline::new(
                format!("webrtc-{session}-{index}"),
                source,
                |source, ctx| {
                    ctx.attach(source, 0, ctx.branch().to(sink)?)?;
                    Ok(())
                },
            )
            .map_err(|error| error.to_string())?;
            track.inbox = Some(inbox);
            track.pipeline = Some(Driving::new(pipeline));
        }
        for (index, packet) in std::mem::take(&mut self.held) {
            if let Some(inbox) = &self.tracks[index].inbox {
                inbox.push(MediaBuffer::packet(packet));
            }
        }
        self.published = true;
        Ok(())
    }
}
