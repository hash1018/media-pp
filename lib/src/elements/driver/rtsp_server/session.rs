//! One RTSP connection: its requests, what it is sent while it plays, and
//! what it sends while it publishes.

use std::collections::hash_map::RandomState;
use std::hash::BuildHasher;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::net::tcp::OwnedWriteHalf;
use tokio::sync::mpsc;

use crate::elements::broadcast::{Timeline, same_shape, time_base};
use crate::elements::{Broadcast, BroadcastItem, BroadcastReader, StreamInfo};
use crate::pp_log::{PpLog, pp_debug, pp_info, pp_warn};

use super::announce::{self, Announced};
use super::message::{self, Incoming, Request, Response, Transport};
use super::record::{Recording, Stopped};
use super::rtp::{self, Packetizer};
use super::{RtspPathMode, Shared};

/// How long a session lasts without a request, as the server tells its
/// client — a session lasts as long as its connection here, so this only
/// tells a client how often to say it is still there.
const SESSION_TIMEOUT: u64 = 60;

/// The methods this server answers.
const PUBLIC: &str = "OPTIONS, DESCRIBE, ANNOUNCE, SETUP, PLAY, RECORD, PAUSE, TEARDOWN, GET_PARAMETER, SET_PARAMETER";

/// How many of a publisher's UDP packets wait for its connection's task.
const UDP_BACKLOG: usize = 1024;

/// How much faster than it plays a UDP player is sent what the broadcast
/// held when it joined: twice, so it catches up in as long as that was.
/// Sent at once, the group of pictures it joins with overflows the
/// player's socket, and UDP has no way to say slow down.
const UDP_CATCH_UP: f64 = 2.0;

/// One of a publisher's RTP packets that came over UDP: its track, and the
/// packet.
pub(super) type Datagram = (usize, Vec<u8>);

/// The server's two UDP sockets: RTP is sent and taken on one, RTCP on the
/// other.
pub(super) struct Udp {
    rtp: tokio::net::UdpSocket,
    rtcp: tokio::net::UdpSocket,
}

impl Udp {
    pub(super) fn new(
        rtp: std::net::UdpSocket,
        rtcp: std::net::UdpSocket,
    ) -> std::io::Result<Self> {
        rtp.set_nonblocking(true)?;
        rtcp.set_nonblocking(true)?;
        Ok(Self {
            rtp: tokio::net::UdpSocket::from_std(rtp)?,
            rtcp: tokio::net::UdpSocket::from_std(rtcp)?,
        })
    }

    /// Reads what comes to the server's UDP ports: a publisher's RTP goes to
    /// its connection, by where it comes from; reports from players and
    /// publishers are let go of, so they never fill the sockets' buffers.
    pub(super) async fn receive(self: Arc<Self>, shared: Arc<Shared>) {
        let mut rtp = vec![0u8; 65_536];
        let mut rtcp = vec![0u8; 2048];
        loop {
            tokio::select! {
                got = self.rtp.recv_from(&mut rtp) => {
                    let Ok((length, from)) = got else { continue };
                    let route = shared.routes.lock().unwrap_or_else(|p| p.into_inner()).get(&from).cloned();
                    if let Some((track, connection)) = route {
                        // A publisher's task that has fallen this far behind
                        // loses a packet, as UDP would.
                        let _ = connection.try_send((track, rtp[..length].to_vec()));
                    }
                }
                _ = self.rtcp.recv_from(&mut rtcp) => {}
            }
        }
    }
}

/// Where one track of a playing session is sent.
#[derive(Debug, Clone, Copy)]
enum Delivery {
    /// Interleaved on the connection: RTP on `channel`, RTCP on the next.
    Tcp { channel: u8 },
    /// To the player's ports.
    Udp { rtp: SocketAddr, rtcp: SocketAddr },
}

/// What a connection has set up.
struct Session {
    id: String,
    path: String,
    broadcast: Broadcast,
    role: Role,
}

enum Role {
    Play {
        /// The broadcast's tracks as they were described.
        tracks: Vec<StreamInfo>,
        /// Which of them were described, by their `streamid`: the ones the
        /// RTP muxer carries.
        described: Vec<usize>,
        /// Where each described track is sent, once set up.
        delivery: Vec<Option<Delivery>>,
    },
    Record {
        tracks: Vec<Announced>,
        /// The channel each track's RTP comes in on, interleaved.
        channels: Vec<Option<u8>>,
        /// Holds the path against a second publisher while this one lasts.
        _claim: Claim,
    },
}

/// A playing session's reading of its broadcast.
struct Playing {
    reader: BroadcastReader,
    timeline: Timeline,
    /// Each described track's packetizer, made with its first packet.
    packetizers: Vec<Option<Packetizer>>,
    /// Where timestamp zero is in the sender reports, the same for every
    /// track so the player lines them up.
    start_us: i64,
    /// Whether the reader has been handed the run it started in.
    started: bool,
    /// Where a UDP player's sending is paced from: when the first packet
    /// was sent, and where it is on the session's timeline.
    paced_from: Option<(Instant, f64)>,
    _count: Count,
}

/// Counts a player for as long as it plays.
struct Count(Arc<Shared>);

impl Count {
    fn new(shared: &Arc<Shared>) -> Self {
        shared.playing.fetch_add(1, Ordering::AcqRel);
        Self(Arc::clone(shared))
    }
}

impl Drop for Count {
    fn drop(&mut self) {
        self.0.playing.fetch_sub(1, Ordering::AcqRel);
    }
}

/// A path held by its publisher.
struct Claim {
    shared: Arc<Shared>,
    path: String,
}

impl Claim {
    fn take(shared: &Arc<Shared>, path: &str) -> Option<Self> {
        let fresh = shared
            .publishing
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(path.to_owned());
        fresh.then(|| Self {
            shared: Arc::clone(shared),
            path: path.to_owned(),
        })
    }
}

impl Drop for Claim {
    fn drop(&mut self) {
        self.shared
            .publishing
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(&self.path);
    }
}

/// The UDP addresses a publisher's RTP comes from, routed to its
/// connection while it lasts.
struct Routes {
    shared: Arc<Shared>,
    from: Vec<SocketAddr>,
    connection: mpsc::Sender<Datagram>,
}

impl Routes {
    fn add(&mut self, from: SocketAddr, track: usize) {
        self.shared
            .routes
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(from, (track, self.connection.clone()));
        self.from.push(from);
    }
}

impl Drop for Routes {
    fn drop(&mut self) {
        let mut routes = self.shared.routes.lock().unwrap_or_else(|p| p.into_inner());
        for from in &self.from {
            routes.remove(from);
        }
    }
}

/// Everything about a connection a request can change, but what it reads
/// from while it plays.
struct Connection {
    shared: Arc<Shared>,
    peer: SocketAddr,
    session: Option<Session>,
    recording: Option<Recording>,
    routes: Routes,
    /// When a publisher last sent anything — a packet or a request.
    heard: Instant,
}

/// What a request leaves the connection to do.
enum After {
    Continue,
    Close,
}

/// Serves one connection until it closes, the client leaves, or the server
/// stops.
pub(super) async fn serve(
    stream: TcpStream,
    peer: SocketAddr,
    shared: Arc<Shared>,
    udp: Option<Arc<Udp>>,
) {
    let pp_log = shared.pp_log.clone();
    pp_debug!(pp_log: &pp_log, "connection from {peer}");
    let (mut read, mut write) = stream.into_split();
    let (datagrams, mut datagrams_in) = mpsc::channel::<Datagram>(UDP_BACKLOG);
    let mut buffer = Vec::with_capacity(4096);
    let mut connection = Connection {
        shared: Arc::clone(&shared),
        peer,
        session: None,
        recording: None,
        routes: Routes {
            shared: Arc::clone(&shared),
            from: Vec::new(),
            connection: datagrams,
        },
        heard: Instant::now(),
    };
    let mut playing: Option<Playing> = None;
    let why = 'connection: loop {
        tokio::select! {
            got = read.read_buf(&mut buffer) => {
                match got {
                    Ok(0) => break 'connection "closed by the client",
                    Err(_) => break 'connection "the connection failed",
                    Ok(_) => {}
                }
                loop {
                    let (incoming, taken) = match message::parse(&buffer) {
                        Ok(Some(parsed)) => parsed,
                        Ok(None) => break,
                        Err(message::Malformed(what)) => {
                            pp_warn!(pp_log: &pp_log, "{peer} sent what is not RTSP ({what}); closing");
                            let _ = write.write_all(&Response::new(400).to_bytes(None)).await;
                            break 'connection "a malformed request";
                        }
                    };
                    buffer.drain(..taken);
                    connection.heard = Instant::now();
                    let request = match incoming {
                        Incoming::Request(request) => request,
                        Incoming::Interleaved { channel, data } => {
                            if let Err(why) = connection.interleaved(channel, &data, &pp_log) {
                                break 'connection why;
                            }
                            continue;
                        }
                    };
                    let (response, after) = connection.answer(&request, &mut playing);
                    pp_debug!(
                        pp_log: &pp_log,
                        "{peer} {} {} -> {}",
                        request.method,
                        request.url,
                        response.status()
                    );
                    let bytes = response.to_bytes(request.header("CSeq"));
                    if send(&mut write, &bytes, shared.write_timeout).await.is_err() {
                        break 'connection "the connection failed";
                    }
                    if let After::Close = after {
                        break 'connection "torn down";
                    }
                }
            }
            Some((track, data)) = datagrams_in.recv() => {
                connection.heard = Instant::now();
                if let Err(why) = connection.recorded(track, &data, &pp_log) {
                    break 'connection why;
                }
            }
            () = tokio::time::sleep_until((connection.heard + shared.publisher_timeout).into()),
                if connection.recording.is_some() =>
            {
                break 'connection "the publisher went silent";
            }
            item = next_item(&mut playing), if playing.is_some() => {
                // Its own scope: what the session holds is not `Sync`, and
                // must not be borrowed across the send below.
                let outgoing = {
                    let (Some(session), Some(play)) = (&connection.session, &mut playing) else {
                        continue;
                    };
                    match prepare(item, session, play, &pp_log) {
                        Ok(Some(outgoing)) => outgoing,
                        Ok(None) => continue,
                        Err(why) => break 'connection why,
                    }
                };
                if let Err(why) = deliver(outgoing, &mut write, udp.as_deref(), &shared, &pp_log).await {
                    break 'connection why;
                }
            }
        }
    };
    match &connection.session {
        Some(session) => pp_info!(
            pp_log: &pp_log,
            "{peer} left {} (session {}): {why}",
            session.path,
            session.id
        ),
        None => pp_debug!(pp_log: &pp_log, "{peer} left: {why}"),
    }
}

/// The next item a playing session's reader reads.
async fn next_item(playing: &mut Option<Playing>) -> Option<BroadcastItem> {
    std::future::poll_fn(|cx| match playing.as_mut() {
        Some(playing) => playing.reader.poll_next(cx),
        None => std::task::Poll::Pending,
    })
    .await
}

async fn send(write: &mut OwnedWriteHalf, bytes: &[u8], timeout: Duration) -> Result<(), ()> {
    match tokio::time::timeout(timeout, write.write_all(bytes)).await {
        Ok(Ok(())) => Ok(()),
        _ => Err(()),
    }
}

fn session_header(id: &str) -> String {
    format!("{id};timeout={SESSION_TIMEOUT}")
}

fn session_id() -> String {
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_nanos());
    format!("{:016X}", RandomState::new().hash_one(seed))
}

impl Connection {
    /// The response to `request`, and what the connection does after it.
    fn answer(&mut self, request: &Request, playing: &mut Option<Playing>) -> (Response, After) {
        let in_session = matches!(
            (request.session(), self.session.as_ref()),
            (Some(id), Some(session)) if id == session.id
        );
        let continuing = |response| (response, After::Continue);
        match request.method.as_str() {
            "OPTIONS" => continuing(Response::new(200).header("Public", PUBLIC)),
            "GET_PARAMETER" | "SET_PARAMETER" => continuing(Response::new(200)),
            "DESCRIBE" => continuing(self.describe(request)),
            "ANNOUNCE" => continuing(self.announce(request)),
            "SETUP" => continuing(self.setup(request, playing.is_some())),
            "PLAY" if !in_session => continuing(Response::new(454)),
            "PLAY" => continuing(self.play(playing)),
            "RECORD" if !in_session => continuing(Response::new(454)),
            "RECORD" => continuing(self.record()),
            "PAUSE" if !in_session => continuing(Response::new(454)),
            "PAUSE" => {
                *playing = None;
                let id = self
                    .session
                    .as_ref()
                    .map(|s| s.id.clone())
                    .unwrap_or_default();
                continuing(Response::new(200).header("Session", session_header(&id)))
            }
            "TEARDOWN" => {
                *playing = None;
                (Response::new(200), After::Close)
            }
            _ => continuing(Response::new(501).header("Public", PUBLIC)),
        }
    }

    fn describe(&self, request: &Request) -> Response {
        let Some(target) = message::target(&request.url) else {
            return Response::new(400);
        };
        let Some((broadcast, tracks, described)) = described(&self.shared, &target.path) else {
            return Response::new(404);
        };
        let carried: Vec<StreamInfo> = described
            .iter()
            .map(|&index| tracks[index].clone())
            .collect();
        match rtp::sdp(&target.path, &carried) {
            Ok(sdp) => Response::new(200)
                .header(
                    "Content-Base",
                    format!("{}/", request.url.trim_end_matches('/')),
                )
                .body("application/sdp", sdp),
            Err(error) => {
                pp_warn!(
                    pp_log: &self.shared.pp_log,
                    "could not describe {} (broadcast {}): {error}",
                    target.path,
                    broadcast.name()
                );
                Response::new(500)
            }
        }
    }

    fn announce(&mut self, request: &Request) -> Response {
        let Some(target) = message::target(&request.url) else {
            return Response::new(400);
        };
        if self.session.is_some() {
            return Response::new(455);
        }
        let Some(path) = self.shared.path(&target.path) else {
            return Response::new(404);
        };
        if path.mode != RtspPathMode::Record {
            return Response::new(403);
        }
        let tracks = match std::str::from_utf8(&request.body)
            .map_err(|_| announce::Refused::Malformed)
            .and_then(announce::parse)
        {
            Ok(tracks) => tracks,
            Err(announce::Refused::Malformed) => return Response::new(400),
            Err(announce::Refused::Unsupported(what)) => {
                pp_info!(
                    pp_log: &self.shared.pp_log,
                    "{} announced {what} on {}, which this server cannot take",
                    self.peer,
                    target.path
                );
                return Response::new(415);
            }
        };
        let Some(claim) = Claim::take(&self.shared, &target.path) else {
            pp_info!(
                pp_log: &self.shared.pp_log,
                "{} announced {}, which another publisher holds",
                self.peer,
                target.path
            );
            return Response::new(455);
        };
        self.session = Some(Session {
            id: session_id(),
            path: target.path,
            broadcast: path.broadcast,
            role: Role::Record {
                channels: vec![None; tracks.len()],
                tracks,
                _claim: claim,
            },
        });
        Response::new(200)
    }

    fn setup(&mut self, request: &Request, playing: bool) -> Response {
        let Some(target) = message::target(&request.url) else {
            return Response::new(400);
        };
        if playing || self.recording.is_some() {
            return Response::new(455);
        }
        match (request.session(), self.session.as_ref()) {
            (Some(id), Some(existing)) if id != existing.id => return Response::new(454),
            (Some(_), None) => return Response::new(454),
            _ => {}
        }
        let Some((transport, _)) = request.header("Transport").and_then(message::transport) else {
            return Response::new(461);
        };
        if let Some(Session {
            role: Role::Record { .. },
            ..
        }) = &self.session
        {
            return self.setup_record(request, transport);
        }
        if let Some(existing) = &self.session
            && existing.path != target.path
        {
            return Response::new(459);
        }
        if self.session.is_none() {
            let Some((broadcast, tracks, described)) = described(&self.shared, &target.path) else {
                return Response::new(404);
            };
            self.session = Some(Session {
                id: session_id(),
                path: target.path.clone(),
                broadcast,
                role: Role::Play {
                    tracks,
                    delivery: vec![None; described.len()],
                    described,
                },
            });
        }
        let udp_ports = self.shared.udp_ports;
        let peer = self.peer;
        let Some(Session {
            id,
            role:
                Role::Play {
                    described,
                    delivery,
                    ..
                },
            ..
        }) = self.session.as_mut()
        else {
            return Response::new(500);
        };
        let stream = match target.track {
            Some(stream) => stream,
            None if described.len() == 1 => 0,
            None => return Response::new(459),
        };
        if stream >= described.len() {
            return Response::new(404);
        }
        let (chosen, answer) = match transport {
            Transport::Tcp { channels } => {
                let (rtp, rtcp) = channels.unwrap_or_else(|| default_channels(stream));
                (
                    Delivery::Tcp { channel: rtp },
                    format!("RTP/AVP/TCP;unicast;interleaved={rtp}-{rtcp}"),
                )
            }
            Transport::Udp { rtp, rtcp } => {
                let Some((server_rtp, server_rtcp)) = udp_ports else {
                    return Response::new(461);
                };
                (
                    Delivery::Udp {
                        rtp: SocketAddr::new(peer.ip(), rtp),
                        rtcp: SocketAddr::new(peer.ip(), rtcp),
                    },
                    format!(
                        "RTP/AVP;unicast;client_port={rtp}-{rtcp};server_port={server_rtp}-{server_rtcp}"
                    ),
                )
            }
        };
        delivery[stream] = Some(chosen);
        Response::new(200)
            .header("Transport", answer)
            .header("Session", session_header(id))
    }

    fn setup_record(&mut self, request: &Request, transport: Transport) -> Response {
        let udp_ports = self.shared.udp_ports;
        let peer = self.peer;
        let Some(Session {
            id,
            role: Role::Record {
                tracks, channels, ..
            },
            ..
        }) = self.session.as_mut()
        else {
            return Response::new(500);
        };
        let Some(track) = announced_track(&request.url, tracks) else {
            return Response::new(404);
        };
        let answer = match transport {
            Transport::Tcp { channels: asked } => {
                let (rtp, rtcp) = asked.unwrap_or_else(|| default_channels(track));
                channels[track] = Some(rtp);
                format!("RTP/AVP/TCP;unicast;interleaved={rtp}-{rtcp};mode=record")
            }
            Transport::Udp { rtp, rtcp } => {
                let Some((server_rtp, server_rtcp)) = udp_ports else {
                    return Response::new(461);
                };
                let id = id.clone();
                self.routes.add(SocketAddr::new(peer.ip(), rtp), track);
                return Response::new(200)
                    .header(
                        "Transport",
                        format!(
                            "RTP/AVP/UDP;unicast;client_port={rtp}-{rtcp};server_port={server_rtp}-{server_rtcp};mode=record"
                        ),
                    )
                    .header("Session", session_header(&id));
            }
        };
        Response::new(200)
            .header("Transport", answer)
            .header("Session", session_header(id))
    }

    fn play(&mut self, playing: &mut Option<Playing>) -> Response {
        let Some(Session {
            id,
            path,
            broadcast,
            role:
                Role::Play {
                    described,
                    delivery,
                    ..
                },
        }) = self.session.as_ref()
        else {
            return Response::new(455);
        };
        if delivery.iter().all(Option::is_none) {
            return Response::new(455);
        }
        let response = Response::new(200)
            .header("Session", session_header(id))
            .header("Range", "npt=0.000-");
        if playing.is_none() {
            *playing = Some(Playing {
                reader: broadcast.subscribe(),
                timeline: Timeline::default(),
                packetizers: described.iter().map(|_| None).collect(),
                start_us: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map_or(0, |since| i64::try_from(since.as_micros()).unwrap_or(0)),
                started: false,
                paced_from: None,
                _count: Count::new(&self.shared),
            });
            pp_info!(
                pp_log: &self.shared.pp_log,
                "{} plays {path} (session {id}, {} of {} tracks)",
                self.peer,
                delivery.iter().flatten().count(),
                described.len()
            );
        }
        response
    }

    fn record(&mut self) -> Response {
        let Some(Session {
            id,
            path,
            broadcast,
            role: Role::Record { tracks, .. },
        }) = self.session.as_ref()
        else {
            return Response::new(455);
        };
        let response = Response::new(200).header("Session", session_header(id));
        if self.recording.is_none() {
            pp_info!(
                pp_log: &self.shared.pp_log,
                "{} publishes {path} (session {id}, {} tracks)",
                self.peer,
                tracks.len()
            );
            self.recording = Some(Recording::new(broadcast.clone(), tracks.clone()));
        }
        response
    }

    /// A packet interleaved on the connection: a publisher's RTP, or RTCP
    /// from either side, which is let go of.
    fn interleaved(
        &mut self,
        channel: u8,
        data: &[u8],
        pp_log: &PpLog,
    ) -> Result<(), &'static str> {
        let Some(Session {
            role: Role::Record { channels, .. },
            ..
        }) = &self.session
        else {
            return Ok(());
        };
        match channels.iter().position(|&rtp| rtp == Some(channel)) {
            Some(track) => self.recorded(track, data, pp_log),
            None => Ok(()),
        }
    }

    /// One of a publisher's RTP packets of `track`, into its recording.
    fn recorded(&mut self, track: usize, data: &[u8], pp_log: &PpLog) -> Result<(), &'static str> {
        let Some(recording) = self.recording.as_mut() else {
            return Ok(());
        };
        let was = recording.published();
        match recording.take(track, data) {
            Ok(()) => {
                if !was
                    && recording.published()
                    && let Some(session) = &self.session
                {
                    pp_info!(pp_log: pp_log, "{} published into {}", self.peer, session.path);
                }
                Ok(())
            }
            Err(Stopped::Broadcast(error)) => {
                pp_info!(pp_log: pp_log, "{}'s publication ended: {error}", self.peer);
                Err("the broadcast took no more")
            }
        }
    }
}

/// Interleaved channels for track `index` where a client named none.
fn default_channels(index: usize) -> (u8, u8) {
    let channel = u8::try_from(index * 2).unwrap_or(254);
    (channel, channel.saturating_add(1))
}

/// Which announced track a `SETUP` URL names, by its control URL.
fn announced_track(url: &str, tracks: &[Announced]) -> Option<usize> {
    let url = url.trim_end_matches('/');
    tracks
        .iter()
        .position(|track| {
            let control = track.control.trim_end_matches('/');
            !control.is_empty() && (url == control || url.ends_with(&format!("/{control}")))
        })
        .or_else(|| (tracks.len() == 1).then_some(0))
}

/// What a path's broadcast describes now: its tracks, and which of them the
/// RTP muxer carries. `None` where the path is not served, or has nothing
/// to describe.
fn described(shared: &Shared, path: &str) -> Option<(Broadcast, Vec<StreamInfo>, Vec<usize>)> {
    let entry = shared.path(path)?;
    if entry.broadcast.is_closed() {
        return None;
    }
    let tracks = entry.broadcast.tracks()?;
    let described: Vec<usize> = (0..tracks.len())
        .filter(|&index| rtp::carries(&tracks[index]))
        .collect();
    if described.is_empty() {
        return None;
    }
    Some((entry.broadcast, tracks, described))
}

/// What is to be sent to a player, and where.
struct Outgoing {
    delivery: Delivery,
    packets: Vec<Vec<u8>>,
    /// Not before when, for a UDP player catching up.
    due: Option<Instant>,
}

/// What `item` has a playing session send its player, if anything: an
/// error, with why, ends the connection.
fn prepare(
    item: Option<BroadcastItem>,
    session: &Session,
    playing: &mut Playing,
    pp_log: &PpLog,
) -> Result<Option<Outgoing>, &'static str> {
    let Role::Play {
        tracks,
        described,
        delivery,
    } = &session.role
    else {
        return Ok(None);
    };
    let (track, packet, arrived) = match item {
        Some(BroadcastItem::Closed) | None => return Err("the broadcast closed"),
        Some(BroadcastItem::Started(started)) => {
            if !same_shape(tracks, &started) {
                pp_info!(
                    pp_log: pp_log,
                    "{} was published again with tracks the session's description does not hold; ending session {}",
                    session.path,
                    session.id
                );
                return Err("the broadcast's tracks changed");
            }
            if playing.started {
                playing.timeline.next_run();
            }
            playing.started = true;
            return Ok(None);
        }
        Some(BroadcastItem::Skipped) => {
            pp_debug!(
                pp_log: pp_log,
                "session {} fell behind {} and goes on from its newest keyframe",
                session.id,
                session.path
            );
            return Ok(None);
        }
        Some(BroadcastItem::Ended) => return Ok(None),
        Some(BroadcastItem::Packet {
            track,
            packet,
            arrived,
        }) => (track, packet, arrived),
    };
    let Some(stream) = described.iter().position(|&index| index == track) else {
        return Ok(None);
    };
    let Some(delivery) = delivery[stream] else {
        return Ok(None);
    };
    let info = &tracks[track];
    let Some(stamped) =
        playing
            .timeline
            .restamp(&packet, time_base(&packet, Some(info.time_base)), arrived)
    else {
        return Ok(None);
    };
    let packetizer = match &mut playing.packetizers[stream] {
        Some(packetizer) => packetizer,
        slot => match Packetizer::new(info, rtp::payload_type(stream), playing.start_us) {
            Ok(packetizer) => slot.insert(packetizer),
            Err(error) => {
                pp_warn!(
                    pp_log: pp_log,
                    "session {}: could not packetize track {track} of {}: {error}",
                    session.id,
                    session.path
                );
                return Err("a track could not be packetized");
            }
        },
    };
    let due = match delivery {
        Delivery::Udp { .. } => stamped.dts().or(stamped.pts()).map(|ts| {
            let at = ts as f64 * f64::from(time_base(&stamped, Some(info.time_base)));
            let (began, from) = *playing.paced_from.get_or_insert((Instant::now(), at));
            began + Duration::from_secs_f64(((at - from) / UDP_CATCH_UP).max(0.0))
        }),
        Delivery::Tcp { .. } => None,
    };
    match packetizer.packetize(&stamped) {
        Ok(packets) => Ok(Some(Outgoing {
            delivery,
            packets,
            due,
        })),
        Err(error) => {
            pp_debug!(
                pp_log: pp_log,
                "session {}: a packet of track {track} was not sent: {error}",
                session.id
            );
            Ok(None)
        }
    }
}

/// Sends what a player is to be sent: what fails to send over TCP ends the
/// connection, with why.
async fn deliver(
    Outgoing {
        delivery,
        packets,
        due,
    }: Outgoing,
    write: &mut OwnedWriteHalf,
    udp: Option<&Udp>,
    shared: &Shared,
    pp_log: &PpLog,
) -> Result<(), &'static str> {
    match delivery {
        Delivery::Tcp { channel } => {
            let mut framed = Vec::with_capacity(packets.iter().map(|p| p.len() + 4).sum());
            for packet in &packets {
                let channel = if rtp::is_rtcp(packet) {
                    channel.saturating_add(1)
                } else {
                    channel
                };
                framed.push(b'$');
                framed.push(channel);
                framed.extend_from_slice(&(packet.len() as u16).to_be_bytes());
                framed.extend_from_slice(packet);
            }
            send(write, &framed, shared.write_timeout)
                .await
                .map_err(|()| "the player stopped taking what it was sent")
        }
        Delivery::Udp { rtp, rtcp } => {
            let Some(udp) = udp else {
                return Err("no UDP sockets");
            };
            if let Some(due) = due {
                tokio::time::sleep_until(due.into()).await;
            }
            for packet in &packets {
                let (socket, to) = if rtp::is_rtcp(packet) {
                    (&udp.rtcp, rtcp)
                } else {
                    (&udp.rtp, rtp)
                };
                // Straight to the socket while it takes it; waited on only
                // when its buffer is full.
                let sent = match socket.try_send_to(packet, to) {
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        socket.send_to(packet, to).await
                    }
                    sent => sent,
                };
                if let Err(error) = sent {
                    pp_debug!(pp_log: pp_log, "a datagram to {to} was not sent: {error}");
                }
            }
            Ok(())
        }
    }
}
