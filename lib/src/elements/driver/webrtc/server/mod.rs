//! A WebRTC server serving [`Broadcast`]s: [`WebRtcServer`].

mod annex_b;
mod http;
mod session;
#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::io;
use std::net::{SocketAddr, ToSocketAddrs, UdpSocket};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender, bounded, unbounded};
use str0m::net::{Protocol, Receive};
use str0m::{Candidate, Input};
use thiserror::Error as ThisError;

use crate::bus::Bus;
use crate::driver::{Driver, StopReceiver};
use crate::element::{Element, ElementType, element_pp_log};
use crate::elements::Broadcast;
use crate::error::Result;
use crate::pp_log::{PpLog, pp_info};

pub use http::WebRtcHttpEndpoint;
use session::Session;

/// The longest the server's loop waits on its socket before it looks at
/// what its players are to be sent and what it has been asked.
const POLL: Duration = Duration::from_millis(2);

/// How long a request waits for the server's loop to answer it.
const ANSWER_WITHIN: Duration = Duration::from_secs(5);

/// How large the server's UDP socket's buffers are asked to be — every
/// session's packets go through the one socket.
const UDP_BUFFER: usize = 8 * 1024 * 1024;

/// Why a [`WebRtcServer`] or its handle refused something.
///
/// Converts into the crate-wide `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum WebRtcServerError {
    /// The address could not be bound — taken, or not this machine's.
    #[error("could not bind {addr}: {source}")]
    Bind {
        /// The address asked for.
        addr: String,
        /// Why.
        source: io::Error,
    },
    /// The socket is bound to every interface, and no address to give
    /// players was said: see [`WebRtcServerOptions::candidates`].
    #[error(
        "bound to {0}, which a player cannot reach: name the addresses it can in WebRtcServerOptions::candidates"
    )]
    NoCandidate(SocketAddr),
    /// A path is already served under that name.
    #[error("path {0:?} is already served")]
    PathTaken(String),
    /// A path a URL could not name as it is: empty, or holding a character
    /// other than letters, digits, `-`, `_`, `.`, `~` and `/` between
    /// segments.
    #[error("{0:?} is not a path a URL can name")]
    InvalidPath(String),
    /// No path is served under that name.
    #[error("no path {0:?} is served")]
    NoSuchPath(String),
    /// The path takes no publisher — it was added as
    /// [`WebRtcPathMode::Play`] — or one is publishing into it already.
    #[error("path {0:?} takes no publisher now")]
    NotPublishable(String),
    /// The path's broadcast has nothing a WebRTC player can be sent: no
    /// track yet, or none in a codec WebRTC carries (H.264, H.265, VP8,
    /// VP9, AV1, Opus, G.711).
    #[error("path {0:?} has nothing a WebRTC player can be sent")]
    NothingToPlay(String),
    /// The offer is not SDP str0m can answer.
    #[error("the offer cannot be answered: {0}")]
    InvalidOffer(String),
    /// No session of that id, or none any more.
    #[error("no session {0}")]
    NoSuchSession(SessionId),
    /// The session has no open data channel of that label.
    #[error("session {0} has no open data channel {1:?}")]
    NoSuchChannel(SessionId, String),
    /// The server is not running — not yet, or not any more — or did not
    /// answer in time.
    #[error("the WebRTC server is not running")]
    NotRunning,
    /// The HTTP endpoint's address could not be listened on.
    #[error("could not listen on {addr} for HTTP: {source}")]
    Http {
        /// The address asked for.
        addr: String,
        /// Why.
        source: io::Error,
    },
}

/// What a [`WebRtcServer`] lets clients do on a path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebRtcPathMode {
    /// Players may play it — [`WebRtcServerHandle::play`], WHEP.
    Play,
    /// A publisher may also publish into it, one at a time —
    /// [`WebRtcServerHandle::publish`], WHIP; players play it as ever.
    Publish,
}

/// One session of a [`WebRtcServer`]: a player or a publisher. Shown, and
/// read back, as sixteen hex digits — what a WHEP or WHIP session's URL
/// ends in.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SessionId(u64);

impl std::fmt::Debug for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SessionId({self})")
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:016x}", self.0)
    }
}

impl std::str::FromStr for SessionId {
    type Err = std::num::ParseIntError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        u64::from_str_radix(s, 16).map(SessionId)
    }
}

/// What a session does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionRole {
    /// It plays its path.
    Play,
    /// It publishes into its path.
    Publish,
}

/// Something that happened to one of a [`WebRtcServer`]'s sessions — read
/// with [`WebRtcServerHandle::next_event`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WebRtcServerEvent {
    /// The session connected: ICE and DTLS are done, and media flows.
    Connected {
        /// Which session.
        session: SessionId,
        /// Its path.
        path: String,
        /// What it does.
        role: SessionRole,
    },
    /// The client opened a data channel.
    ChannelOpen {
        /// Which session.
        session: SessionId,
        /// The channel's label, as the client named it.
        channel: String,
    },
    /// The client sent something on a data channel.
    Data {
        /// Which session.
        session: SessionId,
        /// The channel's label.
        channel: String,
        /// Whether it was sent as binary rather than text.
        binary: bool,
        /// What it sent.
        data: Vec<u8>,
    },
    /// The session ended.
    Ended {
        /// Which session.
        session: SessionId,
        /// Why, in words.
        why: String,
    },
}

/// How a [`WebRtcServer`] runs.
#[derive(Debug, Clone, Default)]
pub struct WebRtcServerOptions {
    /// The addresses players and publishers reach the server's socket at —
    /// its ICE candidates. Empty, the default, is the address it is bound
    /// to, which must then be one a client can reach: not `0.0.0.0`.
    pub candidates: Vec<SocketAddr>,
}

/// A WebRTC server: plays [`Broadcast`]s to WebRTC players — browsers among
/// them — each under a path of its own, and takes a WebRTC publisher's
/// stream into one on a path that allows it. Needs the `webrtc` feature.
///
/// One UDP socket serves every session. A player is sent its path's
/// broadcast as it is — nothing decoded or encoded — from the newest
/// keyframe on, in the codecs WebRTC carries: H.264, H.265, VP8, VP9 and AV1
/// pictures, Opus and G.711 sound. A track in any other codec — AAC, say,
/// from a camera — is left out, so a camera's AAC sound reaches a browser
/// only re-encoded to Opus by a pipeline of the application's own. A
/// browser plays H.264 without B-frames, which an encoder for WebRTC leaves
/// out.
///
/// It answers offers, and leaves how they reach it to the application:
/// [`WebRtcServerHandle::play`] and [`WebRtcServerHandle::publish`] take an
/// offer and give its answer, as WHEP and WHIP do over HTTP — which
/// [`WebRtcServerHandle::serve_http`] serves, for a browser to reach. What
/// clients send on data channels comes out of
/// [`WebRtcServerHandle::next_event`], and [`WebRtcServerHandle::send`]
/// sends on them.
///
/// It is not a pipeline element: run it with a
/// [`DriverRunner`](crate::driver::DriverRunner).
///
/// # Example
///
/// ```no_run
/// # fn main() -> media_pp::Result<()> {
/// use media_pp::driver::DriverRunner;
/// use media_pp::elements::{
///     Broadcast, BroadcastOptions, WebRtcPathMode, WebRtcServer, WebRtcServerOptions,
/// };
///
/// let (server, handle) = WebRtcServer::bind("server", "192.168.0.5:8189", WebRtcServerOptions::default())?;
/// let camera = Broadcast::new("camera", BroadcastOptions::default());
/// handle.add_path("cam1", &camera, WebRtcPathMode::Play)?;
/// DriverRunner::new(server).run()?;
/// // Browsers play http://192.168.0.5:8080/cam1/whep.
/// let _http = handle.serve_http("0.0.0.0:8080")?;
/// // ... a pipeline writes the camera into `camera`.
/// # Ok(())
/// # }
/// ```
///
/// # How it works
///
/// One thread — its driver's — runs every session: a `str0m` connection
/// each, fed what the socket receives, polled for what it sends. A player is
/// a reader of its path's broadcast from the moment it connects, so it
/// starts at a keyframe; what the broadcast held then is sent at twice the
/// stream's speed, so it catches up without a burst its socket cannot hold,
/// and what comes after as it is written. Its tracks share one timeline, which
/// goes on across the broadcast's runs. A publisher's tracks are published
/// into the path's broadcast once each has said enough of itself — H.264's
/// parameter sets — and its leaving ends the run.
pub struct WebRtcServer {
    name: Arc<str>,
    pp_log: PpLog,
    socket: Option<UdpSocket>,
    candidates: Vec<Candidate>,
    commands: Receiver<Command>,
    shared: Arc<Shared>,
}

/// A handle to a [`WebRtcServer`], to say what it serves and to answer
/// offers while it runs.
///
/// Cheap to clone; every clone is the same server's. It keeps the paths'
/// broadcasts it was given — so a broadcast served stays open while its
/// path is. What it asks of the server waits for the server's loop to
/// answer, up to five seconds, and fails with
/// [`WebRtcServerError::NotRunning`] when the server is not running.
#[derive(Clone)]
pub struct WebRtcServerHandle {
    shared: Arc<Shared>,
    commands: Sender<Command>,
}

/// What a server and its handles share.
pub(crate) struct Shared {
    pub(crate) pp_log: PpLog,
    paths: Mutex<HashMap<String, Path>>,
    local_addr: SocketAddr,
    next_session: AtomicU64,
    sessions: AtomicUsize,
    events: (Sender<WebRtcServerEvent>, Receiver<WebRtcServerEvent>),
}

#[derive(Clone)]
pub(crate) struct Path {
    pub(crate) broadcast: Broadcast,
    pub(crate) mode: WebRtcPathMode,
}

/// What a handle asks of the server's loop.
pub(crate) enum Command {
    Open {
        path: String,
        role: SessionRole,
        offer: String,
        answer: Sender<std::result::Result<(SessionId, String), WebRtcServerError>>,
    },
    Close {
        session: SessionId,
        done: Sender<bool>,
    },
    Send {
        session: SessionId,
        channel: String,
        binary: bool,
        data: Vec<u8>,
        done: Sender<std::result::Result<(), WebRtcServerError>>,
    },
}

impl WebRtcServer {
    /// A server on a UDP socket bound to `addr` — `"192.168.0.5:8189"`, or
    /// `"127.0.0.1:0"` for a port of the system's choice on this machine
    /// alone — and the handle that says what it serves.
    ///
    /// # Errors
    ///
    /// [`WebRtcServerError::Bind`] where `addr` cannot be bound, and
    /// [`WebRtcServerError::NoCandidate`] where it is bound to every
    /// interface and no [`WebRtcServerOptions::candidates`] are given.
    pub fn bind(
        name: impl Into<String>,
        addr: impl ToSocketAddrs,
        options: WebRtcServerOptions,
    ) -> std::result::Result<(Self, WebRtcServerHandle), WebRtcServerError> {
        let name: Arc<str> = name.into().into();
        let addrs: Vec<SocketAddr> = addr
            .to_socket_addrs()
            .map_err(|source| WebRtcServerError::Bind {
                addr: String::new(),
                source,
            })?
            .collect();
        let shown = addrs
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let bind_error = |source| WebRtcServerError::Bind {
            addr: shown.clone(),
            source,
        };
        let first = *addrs
            .first()
            .ok_or_else(|| bind_error(io::Error::new(io::ErrorKind::InvalidInput, "no address")))?;
        let socket = udp_socket(first).map_err(bind_error)?;
        let local_addr = socket.local_addr().map_err(bind_error)?;
        let advertised = if options.candidates.is_empty() {
            if local_addr.ip().is_unspecified() {
                return Err(WebRtcServerError::NoCandidate(local_addr));
            }
            vec![local_addr]
        } else {
            options
                .candidates
                .iter()
                .map(|candidate| {
                    // A candidate's port is the socket's where it says none.
                    if candidate.port() == 0 {
                        SocketAddr::new(candidate.ip(), local_addr.port())
                    } else {
                        *candidate
                    }
                })
                .collect()
        };
        let candidates = advertised
            .iter()
            .filter_map(|addr| Candidate::host(*addr, "udp").ok())
            .collect();
        let pp_log = element_pp_log(ElementType::WebRtcServer, &name, None);
        pp_info!(pp_log: &pp_log, "bound to {local_addr}, candidates {advertised:?}");
        let (commands_tx, commands) = bounded(256);
        let shared = Arc::new(Shared {
            pp_log: pp_log.clone(),
            paths: Mutex::new(HashMap::new()),
            local_addr,
            next_session: AtomicU64::new(seed()),
            sessions: AtomicUsize::new(0),
            events: unbounded(),
        });
        Ok((
            Self {
                name,
                pp_log,
                socket: Some(socket),
                candidates,
                commands,
                shared: Arc::clone(&shared),
            },
            WebRtcServerHandle {
                shared,
                commands: commands_tx,
            },
        ))
    }
}

/// Where session numbers start: a different place each run, so a session's
/// URL from one is not another's in the next.
fn seed() -> u64 {
    use std::hash::BuildHasher;
    std::collections::hash_map::RandomState::new().hash_one(Instant::now()) >> 16
}

fn udp_socket(addr: SocketAddr) -> io::Result<UdpSocket> {
    use socket2::{Domain, Protocol as SocketProtocol, Socket, Type};
    let socket = Socket::new(
        Domain::for_address(addr),
        Type::DGRAM,
        Some(SocketProtocol::UDP),
    )?;
    // Not fatal: a smaller buffer only loses more under load.
    let _ = socket.set_send_buffer_size(UDP_BUFFER);
    let _ = socket.set_recv_buffer_size(UDP_BUFFER);
    socket.bind(&addr.into())?;
    Ok(socket.into())
}

impl WebRtcServerHandle {
    /// Serves `broadcast` under `path`, as `mode` allows, from now on.
    ///
    /// # Errors
    ///
    /// [`WebRtcServerError::InvalidPath`] for a path a URL could not name
    /// as it is, and [`WebRtcServerError::PathTaken`] for one already
    /// served; what was served under it goes on unchanged.
    pub fn add_path(
        &self,
        path: &str,
        broadcast: &Broadcast,
        mode: WebRtcPathMode,
    ) -> std::result::Result<(), WebRtcServerError> {
        if !valid_path(path) {
            return Err(WebRtcServerError::InvalidPath(path.to_owned()));
        }
        let mut paths = self.shared.paths.lock().unwrap_or_else(|p| p.into_inner());
        if paths.contains_key(path) {
            return Err(WebRtcServerError::PathTaken(path.to_owned()));
        }
        paths.insert(
            path.to_owned(),
            Path {
                broadcast: broadcast.clone(),
                mode,
            },
        );
        drop(paths);
        pp_info!(
            pp_log: &self.shared.pp_log,
            "path added: {path} broadcast={} mode={mode:?}",
            broadcast.name()
        );
        Ok(())
    }

    /// Stops serving `path`: nobody new opens it. Whether it was served.
    /// Sessions already open go on.
    pub fn remove_path(&self, path: &str) -> bool {
        let removed = self
            .shared
            .paths
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(path)
            .is_some();
        if removed {
            pp_info!(pp_log: &self.shared.pp_log, "path removed: {path}");
        }
        removed
    }

    /// Answers `offer` — SDP from a player — with a session playing `path`:
    /// the session, and the SDP answer to send back.
    ///
    /// # Errors
    ///
    /// [`WebRtcServerError::NoSuchPath`],
    /// [`WebRtcServerError::NothingToPlay`] where its broadcast has nothing
    /// WebRTC carries, [`WebRtcServerError::InvalidOffer`], and
    /// [`WebRtcServerError::NotRunning`].
    pub fn play(
        &self,
        path: &str,
        offer: &str,
    ) -> std::result::Result<(SessionId, String), WebRtcServerError> {
        self.open(path, SessionRole::Play, offer)
    }

    /// Answers `offer` — SDP from a publisher — with a session publishing
    /// into `path`: the session, and the SDP answer to send back.
    ///
    /// # Errors
    ///
    /// [`WebRtcServerError::NoSuchPath`],
    /// [`WebRtcServerError::NotPublishable`] where the path takes no
    /// publisher now, [`WebRtcServerError::InvalidOffer`], and
    /// [`WebRtcServerError::NotRunning`].
    pub fn publish(
        &self,
        path: &str,
        offer: &str,
    ) -> std::result::Result<(SessionId, String), WebRtcServerError> {
        self.open(path, SessionRole::Publish, offer)
    }

    fn open(
        &self,
        path: &str,
        role: SessionRole,
        offer: &str,
    ) -> std::result::Result<(SessionId, String), WebRtcServerError> {
        let (answer, answered) = bounded(1);
        self.ask(Command::Open {
            path: path.to_owned(),
            role,
            offer: offer.to_owned(),
            answer,
        })?;
        answered
            .recv_timeout(ANSWER_WITHIN)
            .map_err(|_| WebRtcServerError::NotRunning)?
    }

    /// Ends `session`. Whether there was one.
    pub fn close(&self, session: SessionId) -> bool {
        let (done, closed) = bounded(1);
        if self.ask(Command::Close { session, done }).is_err() {
            return false;
        }
        closed.recv_timeout(ANSWER_WITHIN).unwrap_or(false)
    }

    /// Sends `data` on `session`'s data channel `channel` — as binary, or
    /// as text where `binary` is false.
    ///
    /// # Errors
    ///
    /// [`WebRtcServerError::NoSuchSession`],
    /// [`WebRtcServerError::NoSuchChannel`], and
    /// [`WebRtcServerError::NotRunning`].
    pub fn send(
        &self,
        session: SessionId,
        channel: &str,
        binary: bool,
        data: &[u8],
    ) -> std::result::Result<(), WebRtcServerError> {
        let (done, sent) = bounded(1);
        self.ask(Command::Send {
            session,
            channel: channel.to_owned(),
            binary,
            data: data.to_vec(),
            done,
        })?;
        sent.recv_timeout(ANSWER_WITHIN)
            .map_err(|_| WebRtcServerError::NotRunning)?
    }

    fn ask(&self, command: Command) -> std::result::Result<(), WebRtcServerError> {
        self.commands
            .send_timeout(command, ANSWER_WITHIN)
            .map_err(|_| WebRtcServerError::NotRunning)
    }

    /// The next thing that happened to a session, waiting up to `timeout`
    /// for one. Every clone of the handle reads the one queue, so each event
    /// goes to one reader.
    pub fn next_event(&self, timeout: Duration) -> Option<WebRtcServerEvent> {
        self.shared.events.1.recv_timeout(timeout).ok()
    }

    /// The address its socket is bound to.
    pub fn local_addr(&self) -> SocketAddr {
        self.shared.local_addr
    }

    /// How many sessions are open, players and publishers together.
    pub fn sessions(&self) -> usize {
        self.shared.sessions.load(Ordering::Acquire)
    }

    /// Serves WHEP and WHIP over HTTP on `addr`, for browsers and other
    /// clients that speak them — see [`WebRtcHttpEndpoint`].
    ///
    /// # Errors
    ///
    /// [`WebRtcServerError::Http`] where `addr` cannot be listened on.
    pub fn serve_http(
        &self,
        addr: impl ToSocketAddrs,
    ) -> std::result::Result<WebRtcHttpEndpoint, WebRtcServerError> {
        WebRtcHttpEndpoint::bind(addr, self.clone())
    }
}

impl Shared {
    pub(crate) fn path(&self, name: &str) -> Option<Path> {
        self.paths
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(name)
            .cloned()
    }
}

impl std::fmt::Debug for WebRtcServerHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebRtcServerHandle")
            .field("local_addr", &self.shared.local_addr)
            .finish()
    }
}

/// Whether `path` is one a URL names as it is.
pub(crate) fn valid_path(path: &str) -> bool {
    !path.is_empty()
        && path.split('/').all(|segment| !segment.is_empty())
        && path
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '~' | '/'))
}

impl Element for WebRtcServer {
    fn name(&self) -> Arc<str> {
        Arc::clone(&self.name)
    }

    fn element_type(&self) -> ElementType {
        ElementType::WebRtcServer
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Driver for WebRtcServer {
    /// Runs every session until asked to stop; each is closed then.
    fn run(&mut self, stop: &StopReceiver, _bus: &Bus) -> Result<()> {
        let Some(socket) = self.socket.take() else {
            return Ok(());
        };
        pp_info!(self, "started");
        let mut sessions: HashMap<SessionId, Session> = HashMap::new();
        let mut buffer = vec![0u8; 65_536];
        let local = self.shared.local_addr;
        let mut publishing: HashMap<String, SessionId> = HashMap::new();
        while !stop.is_stopped() {
            while let Ok(command) = self.commands.try_recv() {
                self.command(command, &socket, &mut sessions, &mut publishing);
            }
            let now = Instant::now();
            let mut next = now + POLL;
            let mut ended = Vec::new();
            for (id, session) in &mut sessions {
                match session.drive(now, &socket, &self.shared) {
                    Ok(deadline) => next = next.min(deadline),
                    Err(why) => ended.push((*id, why)),
                }
            }
            for (id, why) in ended {
                self.end(id, why, &mut sessions, &mut publishing);
            }
            let wait = next
                .saturating_duration_since(Instant::now())
                .clamp(Duration::from_micros(200), POLL);
            if socket.set_read_timeout(Some(wait)).is_err() {
                continue;
            }
            // What arrives now, and whatever else is already waiting.
            let mut first = true;
            loop {
                let received = socket.recv_from(&mut buffer);
                let (length, source) = match received {
                    Ok(received) => received,
                    Err(_) => break,
                };
                let arrived = Instant::now();
                if let Ok(contents) = buffer[..length].try_into() {
                    let input = Input::Receive(
                        arrived,
                        Receive {
                            proto: Protocol::Udp,
                            source,
                            destination: local,
                            contents,
                        },
                    );
                    if let Some(session) = sessions
                        .values_mut()
                        .find(|session| session.accepts(&input))
                    {
                        session.input(input);
                    }
                }
                if first {
                    first = false;
                    let _ = socket.set_nonblocking(true);
                }
            }
            let _ = socket.set_nonblocking(false);
        }
        let ids: Vec<SessionId> = sessions.keys().copied().collect();
        for id in ids {
            if let Some(session) = sessions.get_mut(&id) {
                session.close(&socket);
            }
            self.end(
                id,
                "the server stopped".into(),
                &mut sessions,
                &mut publishing,
            );
        }
        pp_info!(self, "stopped");
        Ok(())
    }
}

impl WebRtcServer {
    fn command(
        &mut self,
        command: Command,
        socket: &UdpSocket,
        sessions: &mut HashMap<SessionId, Session>,
        publishing: &mut HashMap<String, SessionId>,
    ) {
        match command {
            Command::Open {
                path,
                role,
                offer,
                answer,
            } => {
                let opened = self.open(&path, role, &offer, publishing);
                let reply = opened.map(|(id, session, sdp)| {
                    sessions.insert(id, session);
                    self.shared.sessions.fetch_add(1, Ordering::AcqRel);
                    if role == SessionRole::Publish {
                        publishing.insert(path.clone(), id);
                    }
                    pp_info!(self, "session {id}: {role:?} {path}");
                    (id, sdp)
                });
                if let Err(error) = &reply {
                    pp_info!(self, "an offer for {path} ({role:?}) was refused: {error}");
                }
                let _ = answer.send(reply);
            }
            Command::Close { session, done } => {
                let found = sessions.contains_key(&session);
                if found {
                    if let Some(open) = sessions.get_mut(&session) {
                        open.close(socket);
                    }
                    self.end(session, "closed".into(), sessions, publishing);
                }
                let _ = done.send(found);
            }
            Command::Send {
                session,
                channel,
                binary,
                data,
                done,
            } => {
                let sent = match sessions.get_mut(&session) {
                    Some(open) => open
                        .send(&channel, binary, &data)
                        .map_err(|()| WebRtcServerError::NoSuchChannel(session, channel.clone())),
                    None => Err(WebRtcServerError::NoSuchSession(session)),
                };
                let _ = done.send(sent);
            }
        }
    }

    fn open(
        &self,
        path: &str,
        role: SessionRole,
        offer: &str,
        publishing: &HashMap<String, SessionId>,
    ) -> std::result::Result<(SessionId, Session, String), WebRtcServerError> {
        let entry = self
            .shared
            .path(path)
            .ok_or_else(|| WebRtcServerError::NoSuchPath(path.to_owned()))?;
        if role == SessionRole::Publish
            && (entry.mode != WebRtcPathMode::Publish || publishing.contains_key(path))
        {
            return Err(WebRtcServerError::NotPublishable(path.to_owned()));
        }
        if role == SessionRole::Play && !session::playable(&entry.broadcast) {
            return Err(WebRtcServerError::NothingToPlay(path.to_owned()));
        }
        let id = SessionId(self.shared.next_session.fetch_add(1, Ordering::Relaxed));
        let (session, answer) = Session::open(
            id,
            path,
            role,
            entry.broadcast,
            offer,
            &self.candidates,
            self.pp_log.clone(),
        )?;
        Ok((id, session, answer))
    }

    fn end(
        &self,
        id: SessionId,
        why: String,
        sessions: &mut HashMap<SessionId, Session>,
        publishing: &mut HashMap<String, SessionId>,
    ) {
        let Some(session) = sessions.remove(&id) else {
            return;
        };
        self.shared.sessions.fetch_sub(1, Ordering::AcqRel);
        publishing.retain(|_, publisher| *publisher != id);
        pp_info!(self, "session {id} ({}) ended: {why}", session.path());
        let _ = self
            .shared
            .events
            .0
            .send(WebRtcServerEvent::Ended { session: id, why });
    }
}
