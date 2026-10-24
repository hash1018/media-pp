//! One RTSP connection: its requests, what it is sent while it plays, and
//! what it sends while it publishes.
//!
//! What a player is sent is a driven pipeline of its own,
//! `BroadcastSource → RtspTrackSink` for each track it set up, which the
//! connection's task polls whenever it has sent all the pipeline gave it:
//! a player that takes its packets slowly leaves its reader behind in the
//! broadcast, never a queue of them here.

use std::collections::hash_map::RandomState;
use std::future::Future;
use std::hash::BuildHasher;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::task::Poll;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::net::tcp::OwnedWriteHalf;
use tokio::sync::mpsc;

use crate::elements::broadcast::same_shape;
use crate::elements::driver::access::{Access, AccessRequest, Credentials};
use crate::elements::driver::publishers::Arrivals;
use crate::elements::{Broadcast, BroadcastSource, RtspSource, StreamInfo};
use crate::pipeline::{DrivenPipeline, DrivenPipelineHandle};
use crate::pp_log::{PpLog, pp_debug, pp_info, pp_warn};

use super::digest::{self, RtspDigest};
use super::record::{Recording, Stopped};
use super::rtp::{self, Packetizer};
use super::track::{Delivery, Outbox, Outgoing, RtspTrackSink};
use super::{SessionCounts, Shared};
use crate::elements::rtsp::message::{self, Incoming, Request, Response, Transport};
use crate::elements::rtsp::sdp::{self as announce, Announced};

/// How long a session lasts without a request, as the server tells its
/// client — a session lasts as long as its connection here, so this only
/// tells a client how often to say it is still there.
const SESSION_TIMEOUT: u64 = 60;

/// The methods this server answers.
const PUBLIC: &str = "OPTIONS, DESCRIBE, ANNOUNCE, SETUP, PLAY, RECORD, PAUSE, TEARDOWN, GET_PARAMETER, SET_PARAMETER";

/// How many of a publisher's UDP packets wait for its connection's task.
const UDP_BACKLOG: usize = 1024;

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

/// What a connection has set up.
struct Session {
    id: String,
    path: String,
    role: Role,
}

enum Role {
    Play {
        broadcast: Broadcast,
        /// The broadcast's tracks as they were described.
        tracks: Vec<StreamInfo>,
        /// Which of them were described, by their `streamid`: the ones the
        /// RTP muxer carries.
        described: Vec<usize>,
        /// Where each described track is sent, once set up.
        delivery: Vec<Option<Delivery>>,
    },
    Record {
        /// Where the publisher is handed out, once it records.
        arrivals: Arrivals<RtspSource>,
        tracks: Vec<Announced>,
        /// The channel each track's RTP comes in on, interleaved.
        channels: Vec<Option<u8>>,
        /// Holds the path against a second publisher while this one lasts.
        _claim: Claim,
    },
}

/// A playing session's pipeline, and what it has cut into RTP to be sent.
struct Playing {
    pipeline: DrivenPipeline,
    handle: DrivenPipelineHandle,
    outbox: Outbox,
    /// Each described track's, by its number, where it was set up: what
    /// cuts its packets into RTP as they are sent.
    packetizers: Vec<Option<Packetizer>>,
    count: Count,
}

/// What a playing session's pipeline has for its connection next.
enum Played {
    Send(Outgoing),
    /// Its stream has ended: the broadcast closed, or was published again
    /// with tracks its description does not hold.
    Ended,
}

/// A session among the server's while it plays or publishes, and what it
/// counts of itself — see [`RtspServerHandle::session_stats`](super::RtspServerHandle::session_stats).
struct Count {
    shared: Arc<Shared>,
    id: u64,
    counts: Arc<SessionCounts>,
}

impl Count {
    fn new(shared: &Arc<Shared>, path: &str, role: Access, remote: SocketAddr, udp: bool) -> Self {
        let id = shared.next_session.fetch_add(1, Ordering::Relaxed);
        let counts = Arc::new(SessionCounts {
            path: path.to_owned(),
            role,
            remote,
            udp,
            opened: Instant::now(),
            sent: Default::default(),
            received: Default::default(),
        });
        shared
            .sessions
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(id, Arc::clone(&counts));
        Self {
            shared: Arc::clone(shared),
            id,
            counts,
        }
    }
}

impl Drop for Count {
    fn drop(&mut self) {
        self.shared
            .sessions
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(&self.id);
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
    /// The publisher among the server's sessions, while it publishes.
    publishing: Option<Count>,
    routes: Routes,
    /// When a publisher last sent anything — a packet or a request.
    heard: Instant,
    /// What the server's [`Authorize`](crate::elements::Authorize) has let
    /// this connection do, on which path: asked once for each.
    allowed: Vec<(String, Access)>,
    /// The challenge a Digest response on this connection answers.
    nonce: String,
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
        publishing: None,
        routes: Routes {
            shared: Arc::clone(&shared),
            from: Vec::new(),
            connection: datagrams,
        },
        heard: Instant::now(),
        allowed: Vec::new(),
        nonce: digest::nonce(),
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
                        // An answer to nothing this server asked.
                        Incoming::Reply(_) => continue,
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
            played = next_played(&mut playing), if playing.is_some() => {
                let outgoing = match played {
                    Played::Send(outgoing) => outgoing,
                    Played::Ended => {
                        let closed = connection.session.as_ref().is_none_or(|session| {
                            matches!(&session.role, Role::Play { broadcast, .. } if broadcast.is_closed())
                        });
                        break 'connection if closed {
                            "the broadcast closed"
                        } else {
                            "the broadcast's tracks changed"
                        };
                    }
                };
                // Cut here, as it is sent, and not borrowed across the send.
                let Some(Some(packetizer)) = playing
                    .as_mut()
                    .and_then(|play| play.packetizers.get_mut(outgoing.stream))
                else {
                    continue;
                };
                let packets = match packetizer.packetize(&outgoing.packet) {
                    Ok(packets) => packets,
                    Err(error) => {
                        pp_debug!(pp_log: &pp_log, "{peer}: a packet was not sent: {error}");
                        continue;
                    }
                };
                let bytes: usize = packets.iter().map(Vec::len).sum();
                if let Err(why) = deliver(
                    outgoing.delivery,
                    packets,
                    outgoing.due,
                    &mut write,
                    udp.as_deref(),
                    &shared,
                    &pp_log,
                )
                .await
                {
                    break 'connection why;
                }
                if let Some(play) = &playing {
                    play.count.counts.sent.fetch_add(bytes as u64, Ordering::Relaxed);
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

/// What a playing session's pipeline has for its connection next: what it
/// already gave, or else what a poll of it gives — polled only once all it
/// gave has been sent.
async fn next_played(playing: &mut Option<Playing>) -> Played {
    std::future::poll_fn(|cx| {
        let Some(playing) = playing.as_mut() else {
            return Poll::Pending;
        };
        loop {
            if let Some(outgoing) = playing.outbox.take() {
                return Poll::Ready(Played::Send(outgoing));
            }
            let polled = Pin::new(&mut playing.pipeline).poll(cx);
            // What it reports is in the log already; read here so that it
            // does not pile up on a bus nobody else reads.
            while playing.handle.bus().try_recv().is_some() {}
            match polled {
                Poll::Ready(()) => return Poll::Ready(Played::Ended),
                Poll::Pending if playing.outbox.is_empty() => return Poll::Pending,
                Poll::Pending => {}
            }
        }
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
    /// What a request the server's `Authorize` refused is answered with:
    /// asking for a user and password, which a player given
    /// `rtsp://user:password@…` then sends — hashed against this
    /// connection's challenge where it can, as a player does where it is
    /// offered both.
    fn unauthorized(&self) -> Response {
        Response::new(401)
            .header(
                "WWW-Authenticate",
                format!(
                    "Digest realm=\"{}\", nonce=\"{}\"",
                    digest::REALM,
                    self.nonce
                ),
            )
            .header(
                "WWW-Authenticate",
                format!("Basic realm=\"{}\"", digest::REALM),
            )
    }

    /// What `request` presents for `path`: a Digest response to this
    /// connection's own challenge, or what any other header says. A Digest
    /// response to any other challenge — another connection's, one made up
    /// — presents nothing.
    fn credentials(&self, request: &Request, path: &str) -> Credentials {
        let header = request.header("Authorization");
        let presented = match header.and_then(|h| h.trim().get(..7)) {
            Some(scheme) if scheme.eq_ignore_ascii_case("digest ") => header
                .and_then(|h| RtspDigest::parse(h, &request.method, &self.nonce, path))
                .map_or(Credentials::None, Credentials::Digest),
            _ => Credentials::from_header(header),
        };
        // A token too long for a URL's user name — some players keep it
        // in a buffer of 128 bytes — rides in its query instead.
        match presented {
            Credentials::None => {
                message::query_token(&request.url).map_or(Credentials::None, Credentials::Bearer)
            }
            presented => presented,
        }
    }

    /// Whether the server's `Authorize`, where it has one, lets this
    /// connection do `access` on `path`, from what `request` presents.
    fn allows(&mut self, request: &Request, path: &str, access: Access) -> bool {
        let Some(authorize) = &self.shared.authorize else {
            return true;
        };
        if self
            .allowed
            .iter()
            .any(|(allowed, granted)| allowed == path && *granted == access)
        {
            return true;
        }
        let asked = AccessRequest {
            path: path.to_owned(),
            access,
            credentials: self.credentials(request, path),
            peer: self.peer,
        };
        // The application's answer may take a while; this runtime thread's
        // other connections go on meanwhile.
        let allowed = tokio::task::block_in_place(|| authorize.allows(&asked));
        if allowed {
            pp_debug!(pp_log: &self.shared.pp_log, "{} may {access:?} {path}", self.peer);
            self.allowed.push((asked.path, access));
        } else {
            pp_info!(
                pp_log: &self.shared.pp_log,
                "refused {access:?} of {path} to {} presenting {:?}",
                self.peer,
                asked.credentials
            );
        }
        allowed
    }

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

    fn describe(&mut self, request: &Request) -> Response {
        let Some(target) = message::target(&request.url) else {
            return Response::new(400);
        };
        if !self.allows(request, &target.path, Access::Play) {
            return self.unauthorized();
        }
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
                    format!(
                        "{}/",
                        message::without_query(&request.url).trim_end_matches('/')
                    ),
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
        if !self.allows(request, &target.path, Access::Publish) {
            return self.unauthorized();
        }
        let arrivals = match self.shared.publish_target(&target.path) {
            Ok(arrivals) => arrivals,
            Err(status) => return Response::new(status),
        };
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
            role: Role::Record {
                arrivals,
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
            // A player may set up without describing first.
            if !self.allows(request, &target.path, Access::Play) {
                return self.unauthorized();
            }
            let Some((broadcast, tracks, described)) = described(&self.shared, &target.path) else {
                return Response::new(404);
            };
            self.session = Some(Session {
                id: session_id(),
                path: target.path.clone(),
                role: Role::Play {
                    broadcast,
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
        let Some(track) = announce::track_named(&request.url, tracks) else {
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
            role:
                Role::Play {
                    broadcast,
                    tracks,
                    described,
                    delivery,
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
            match self.begin_playing(id, path, broadcast, tracks, described, delivery) {
                Ok(begun) => *playing = Some(begun),
                Err(status) => return Response::new(status),
            }
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

    /// A player's pipeline: a reader of the broadcast from its newest
    /// keyframe, and a sink for each track it set up. Refused with the
    /// status to answer where the broadcast's tracks are no longer the ones
    /// described, or a track cannot be packetized.
    fn begin_playing(
        &self,
        id: &str,
        path: &str,
        broadcast: &Broadcast,
        tracks: &[StreamInfo],
        described: &[usize],
        delivery: &[Option<Delivery>],
    ) -> Result<Playing, u16> {
        // A UDP player is sent what it joins with at twice the stream's
        // speed, and has no way to ask for what it loses of it: not a whole
        // group of pictures. Over TCP the connection paces it and loses
        // nothing, so a player starts at once.
        let name = format!("rtsp-{id}-reader");
        let over_udp = delivery
            .iter()
            .any(|delivery| matches!(delivery, Some(Delivery::Udp { .. })));
        let (source, now) = if over_udp {
            BroadcastSource::joining_within(
                name,
                broadcast,
                crate::elements::broadcast::PLAYER_JOIN_WITHIN,
            )
        } else {
            BroadcastSource::new(name, broadcast)
        }
        .map_err(|_| 404_u16)?;
        if !same_shape(tracks, &now) {
            pp_info!(
                pp_log: &self.shared.pp_log,
                "{} was published again with tracks session {id}'s description does not hold",
                broadcast.name()
            );
            return Err(410);
        }
        // Where timestamp zero is in the sender reports, the same for every
        // track so the player lines them up.
        let start_us = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |since| i64::try_from(since.as_micros()).unwrap_or(0));
        let outbox = Outbox::default();
        let mut packetizers: Vec<Option<Packetizer>> = described.iter().map(|_| None).collect();
        let mut sinks = Vec::new();
        for (stream, &track) in described.iter().enumerate() {
            let Some(delivery) = delivery[stream] else {
                continue;
            };
            let info = &tracks[track];
            let packetizer =
                Packetizer::new(info, rtp::payload_type(stream), start_us).map_err(|error| {
                    pp_warn!(
                        pp_log: &self.shared.pp_log,
                        "session {id}: could not packetize track {track} of {}: {error}",
                        broadcast.name()
                    );
                    500_u16
                })?;
            packetizers[stream] = Some(packetizer);
            sinks.push((
                track,
                RtspTrackSink::new(
                    format!("rtsp-{id}-track-{track}"),
                    delivery,
                    stream,
                    info.time_base(),
                    outbox.clone(),
                ),
            ));
        }
        let (pipeline, ()) = DrivenPipeline::new(format!("rtsp-{id}"), source, |source, ctx| {
            for (track, sink) in sinks {
                ctx.attach(source, track, ctx.branch().to(sink)?)?;
            }
            Ok(())
        })
        .map_err(|error| {
            pp_warn!(pp_log: &self.shared.pp_log, "session {id} could not begin: {error}");
            500_u16
        })?;
        Ok(Playing {
            handle: pipeline.handle(),
            pipeline,
            outbox,
            packetizers,
            count: Count::new(&self.shared, path, Access::Play, self.peer, over_udp),
        })
    }

    fn record(&mut self) -> Response {
        let Some(Session {
            id,
            path,
            role: Role::Record {
                arrivals, tracks, ..
            },
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
            self.recording = Some(Recording::new(arrivals.clone(), path, tracks.clone()));
            let udp = matches!(
                &self.session,
                Some(Session { role: Role::Record { channels, .. }, .. })
                    if channels.iter().all(Option::is_none)
            );
            self.publishing = Some(Count::new(
                &self.shared,
                path,
                Access::Publish,
                self.peer,
                udp,
            ));
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
        if let Some(publishing) = &self.publishing {
            publishing
                .counts
                .received
                .fetch_add(data.len() as u64, Ordering::Relaxed);
        }
        match recording.take(track, data, pp_log) {
            Ok(()) => {
                if !was
                    && recording.published()
                    && let Some(session) = &self.session
                {
                    pp_info!(pp_log: pp_log, "{} handed out as {}'s publisher", self.peer, session.path);
                }
                Ok(())
            }
            Err(Stopped::Gone) => {
                pp_info!(pp_log: pp_log, "{}'s publication ended: nothing takes it", self.peer);
                Err("nothing takes the publisher")
            }
        }
    }
}

/// Interleaved channels for track `index` where a client named none.
fn default_channels(index: usize) -> (u8, u8) {
    let channel = u8::try_from(index * 2).unwrap_or(254);
    (channel, channel.saturating_add(1))
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

/// Sends what a player is to be sent: what fails to send over TCP ends the
/// connection, with why.
async fn deliver(
    delivery: Delivery,
    packets: Vec<Vec<u8>>,
    due: Option<Instant>,
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
