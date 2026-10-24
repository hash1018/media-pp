//! A WebRTC server serving [`Broadcast`]s and taking publishers:
//! [`WebRtcServer`].

mod http;
mod session;
#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::collections::hash_map::RandomState;
use std::hash::BuildHasher;
use std::io;
use std::net::{SocketAddr, ToSocketAddrs, UdpSocket};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender, bounded, unbounded};
use str0m::net::{Protocol, Receive};
use str0m::{Candidate, Input};
use thiserror::Error as ThisError;

use crate::bus::Bus;
use crate::driver::{Driver, StopReceiver};
use crate::element::{Element, ElementType, element_pp_log};
use crate::elements::driver::publishers::{self, Arrivals, OnDrop, Publishers};
use crate::elements::{Authorize, Broadcast, WebRtcSource};
use crate::error::Result;
use crate::pp_log::{PpLog, pp_info, pp_warn};

pub use http::WebRtcHttpEndpoint;
use session::{Session, Target};

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
    /// A path is already served under that name — or, for
    /// [`WebRtcServerHandle::take_publishers`], already takes publishers.
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
    /// The path takes no publisher — players alone may use it — or one is
    /// publishing on it already.
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

/// One session of a [`WebRtcServer`]: a player or a publisher. Shown, and
/// read back, as thirty-two hex digits — what a WHEP or WHIP session's URL
/// ends in, and all it takes to end the session over HTTP: so half of it
/// is drawn from a key this server alone knows, and one session's id says
/// nothing of another's.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SessionId(u128);

impl std::fmt::Debug for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SessionId({self})")
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:032x}", self.0)
    }
}

impl std::str::FromStr for SessionId {
    type Err = std::num::ParseIntError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        u128::from_str_radix(s, 16).map(SessionId)
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

/// How one of a [`WebRtcServer`]'s sessions is doing: what its client
/// reports and its connection measures, about once a second — read with
/// [`WebRtcServerHandle::session_stats`], for a dashboard.
///
/// The counts only rise, from the session's opening; a rate is two
/// readings apart. Where the client has not reported yet — a session
/// still connecting, a client that sends no reports — a figure is `None`.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SessionStats {
    /// Which session.
    pub session: SessionId,
    /// Its path.
    pub path: String,
    /// What it does.
    pub role: SessionRole,
    /// When it was opened.
    pub opened: Instant,
    /// Whether it has connected: ICE and DTLS done, media flowing.
    pub connected: bool,
    /// Where the client's packets come from, once it has connected.
    pub remote: Option<SocketAddr>,
    /// The round trip to the client and back, as last measured.
    pub rtt: Option<Duration>,
    /// The share of packets lost on the way, from 0 to 1, over the last
    /// second or so: of what a player was sent, as it reports, or of what
    /// a publisher sent, as the server counts.
    pub loss: Option<f32>,
    /// The bytes sent to the client, everything the connection carried.
    pub bytes_sent: u64,
    /// The bytes received from it.
    pub bytes_received: u64,
    /// How many times a lost packet was asked to be sent again: by a
    /// player of the server, or by the server of a publisher.
    pub resend_requests: u64,
    /// How many times a keyframe was asked for, the same way: a player
    /// whose picture broke, or the server of a publisher.
    pub keyframe_requests: u64,
    /// How many times a player fell so far behind that what was waiting
    /// for it was let go of and it went on from a keyframe — a picture it
    /// skipped. Always 0 for a publisher.
    pub fell_behind: u64,
    /// When these figures were last measured.
    pub updated: Option<Instant>,
}

/// How a [`WebRtcServer`] runs.
#[derive(Debug, Clone)]
pub struct WebRtcServerOptions {
    /// The addresses players and publishers reach the server's sockets at
    /// — its ICE candidates. Empty, the default, is the address it is
    /// bound to, which must then be one a client can reach: not `0.0.0.0`.
    /// A candidate without a port — port 0 — is at the socket's.
    pub candidates: Vec<SocketAddr>,
    /// How many threads it runs its sessions on, each with a UDP socket of
    /// its own: a new session goes to the thread with the fewest. Two by
    /// default. Bound to port 0, each thread's socket is at a port the
    /// system chooses; bound to a port, the threads' sockets are at that
    /// port and the ones after it — as many as there are threads, each
    /// candidate's port counting up with them — so a firewall or a router
    /// forwards that range.
    pub threads: usize,
    /// Who may play and publish which paths over
    /// [`WebRtcServerHandle::serve_http`] — WHEP and WHIP — asked for each
    /// offer before it is answered; `None`, the default, lets everybody.
    /// [`WebRtcServerHandle::play`] and [`WebRtcServerHandle::publish`]
    /// called by the application itself are not asked: what reaches them
    /// the application has let through already. See [`Authorize`].
    pub authorize: Option<Authorize>,
    /// STUN and TURN servers a client of
    /// [`WebRtcServerHandle::serve_http`] may reach this server through,
    /// told it as WHIP and WHEP tell them: a `Link` header each, on the
    /// answer to its offer and on the answer to an `OPTIONS` whose
    /// credentials [`Self::authorize`] lets through, so that a client
    /// behind a network that blocks UDP can ask for a TURN relay before it
    /// makes its offer. Its page passes them to the browser. Empty, the
    /// default, tells none.
    pub ice_servers: Vec<IceServer>,
}

/// A STUN or TURN server, as [`WebRtcServerOptions::ice_servers`] tells a
/// client it: what `RTCIceServer` holds in a browser.
#[derive(Clone, PartialEq, Eq)]
pub struct IceServer {
    /// Its URLs: `stun:host:port`, `turn:host:port?transport=udp`,
    /// `turns:host:443?transport=tcp`.
    pub urls: Vec<String>,
    /// The user name a TURN server takes, where it takes one.
    pub username: Option<String>,
    /// The password a TURN server takes with it.
    pub credential: Option<String>,
}

impl std::fmt::Debug for IceServer {
    /// Never shows the credential.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IceServer")
            .field("urls", &self.urls)
            .field("username", &self.username)
            .finish_non_exhaustive()
    }
}

impl Default for WebRtcServerOptions {
    fn default() -> Self {
        Self {
            candidates: Vec::new(),
            threads: 2,
            authorize: None,
            ice_servers: Vec::new(),
        }
    }
}

/// A WebRTC server: plays [`Broadcast`]s to WebRTC players — browsers among
/// them — each under a path of its own, and takes WebRTC publishers on the
/// paths that take them, each handed out as a [`WebRtcSource`] of what it
/// sends ([`WebRtcServerHandle::take_publishers`]). Needs the `webrtc`
/// feature.
///
/// A few UDP sockets serve every session — one for each of its threads (see
/// [`WebRtcServerOptions::threads`]). A player is sent its path's
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
/// use media_pp::elements::{Broadcast, BroadcastOptions, WebRtcServer, WebRtcServerOptions};
///
/// let (server, handle) = WebRtcServer::bind("server", "192.168.0.5:8189", WebRtcServerOptions::default())?;
/// let camera = Broadcast::new("camera", BroadcastOptions::default());
/// handle.add_path("cam1", &camera)?;
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
/// Each of its threads runs the sessions given it, each a `str0m`
/// connection, fed what the thread's socket receives and polled for what it
/// sends; a thread keeps one core busy at most (`docs/benchmarks/streaming`
/// has what a player costs it).
///
/// What goes through a session is a pipeline of its own with no thread of
/// its own — a [`DrivenPipeline`](crate::pipeline::DrivenPipeline) the
/// session's thread polls as what it waits on wakes it. A player's is a
/// [`BroadcastSource`](crate::elements::BroadcastSource) of its path's
/// broadcast and a sink for each track it takes, begun as it connects, so it starts at a keyframe; its
/// tracks share one timeline, which goes on across the broadcast's runs.
/// What the broadcast held then is sent at twice the stream's speed, so it
/// catches up without a burst its socket cannot hold, and what comes after
/// as it is written — at most a second of it: where the newest keyframe is
/// further behind, the player waits for the next one, as a packet lost in a
/// longer stretch of large pictures is past what the connection keeps to
/// send again. A publisher is handed out as a [`WebRtcSource`] of its
/// tracks once each has said enough of itself — H.264's parameter sets —
/// and its leaving ends the source. The sinks are the ones a
/// [`WebRtcPeer`](super::WebRtcPeer)'s muxer sends through, and the source
/// the one its remote tracks come out of: a peer and a server's session are
/// one connection, driven by different owners.
///
/// A publisher is taken on a path apart from what players play under the
/// same name — WHIP to `/<path>/whip`, WHEP from `/<path>/whep` — and a
/// pipeline of the application's carries what it sends wherever it wants
/// it, a broadcast its players read among them, converted or not.
pub struct WebRtcServer {
    name: Arc<str>,
    pp_log: PpLog,
    shards: Vec<Shard>,
    shared: Arc<Shared>,
}

/// One of a server's threads: its socket, and the sessions it is given.
struct Shard {
    index: usize,
    socket: Option<UdpSocket>,
    local: SocketAddr,
    candidates: Vec<Candidate>,
    commands: Receiver<Command>,
    pp_log: PpLog,
}

/// A handle to a [`WebRtcServer`], to say what it serves and to answer
/// offers while it runs.
///
/// Cheap to clone; every clone is the same server's. It keeps the paths'
/// broadcasts it was given — so a broadcast served stays open while its
/// path is. What it asks of the server waits for the server's loop to
/// answer, up to five seconds, and fails with
/// [`WebRtcServerError::NotRunning`] when the server is not running. Once
/// the server has stopped, the paths' [`Publishers`] answer `None`.
#[derive(Clone)]
pub struct WebRtcServerHandle {
    shared: Arc<Shared>,
    /// Each thread's.
    commands: Vec<Sender<Command>>,
}

/// What a server and its handles share.
pub(crate) struct Shared {
    pub(crate) pp_log: PpLog,
    paths: Mutex<HashMap<String, Broadcast>>,
    /// Where a publisher of each path that takes them is handed out, apart
    /// from what its players play, under the number of the registration,
    /// so that a `Publishers` gone takes away only its own.
    publishers: Mutex<HashMap<String, (u64, Arrivals<WebRtcSource>)>>,
    next_registration: AtomicU64,
    local_addr: SocketAddr,
    next_session: AtomicU64,
    /// Keys what a session's id is made of beyond its number, so that one
    /// cannot be guessed from another.
    session_key: RandomState,
    pub(crate) authorize: Option<Authorize>,
    pub(crate) ice_servers: Vec<IceServer>,
    sessions: AtomicUsize,
    /// How many sessions each thread has.
    per_thread: Vec<AtomicUsize>,
    /// Which session publishes into each path that has a publisher.
    publishing: Mutex<HashMap<String, SessionId>>,
    events: (Sender<WebRtcServerEvent>, Receiver<WebRtcServerEvent>),
    /// Each open session's latest figures, written by its thread as
    /// they change.
    pub(crate) stats: Mutex<HashMap<SessionId, SessionStats>>,
}

/// What a handle asks of a thread.
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

/// How many of a session number's low bits say which thread it is on.
const THREAD_BITS: u32 = 8;

impl SessionId {
    /// The thread it is on, of `threads`.
    fn thread(self, threads: usize) -> usize {
        (self.0 & ((1 << THREAD_BITS) - 1)) as usize % threads.max(1)
    }
}

impl WebRtcServer {
    /// Whether a player is sent `track`: one in a codec WebRTC carries —
    /// H.264, H.265, H.266, VP8, VP9 and AV1 pictures, Opus and G.711 sound.
    /// A broadcast's track in any other is left out of what its players are
    /// sent. Which of these a browser plays is the browser's: most play no
    /// H.265, and none pictures that come out of order (B-frames).
    pub fn carries(track: &crate::elements::StreamInfo) -> bool {
        super::stream_info::str0m_codec(track.parameters().id()).is_some()
    }

    /// A server on UDP sockets bound to `addr` — `"192.168.0.5:8189"`, or
    /// `"127.0.0.1:0"` for ports of the system's choice on this machine
    /// alone — and the handle that says what it serves.
    ///
    /// # Errors
    ///
    /// [`WebRtcServerError::Bind`] where `addr`, or a port after it that a
    /// thread needs, cannot be bound, and [`WebRtcServerError::NoCandidate`]
    /// where it is bound to every interface and no
    /// [`WebRtcServerOptions::candidates`] are given.
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
        let threads = options.threads.clamp(1, 1 << THREAD_BITS);
        let pp_log = element_pp_log(ElementType::WebRtcServer, &name, None);
        let mut shards = Vec::with_capacity(threads);
        let mut senders = Vec::with_capacity(threads);
        for index in 0..threads {
            let port = match first.port() {
                0 => 0,
                port => port.checked_add(index as u16).ok_or_else(|| {
                    bind_error(io::Error::new(io::ErrorKind::InvalidInput, "ports run out"))
                })?,
            };
            let socket = udp_socket(SocketAddr::new(first.ip(), port)).map_err(bind_error)?;
            let local = socket.local_addr().map_err(bind_error)?;
            let advertised: Vec<SocketAddr> = if options.candidates.is_empty() {
                if local.ip().is_unspecified() {
                    return Err(WebRtcServerError::NoCandidate(local));
                }
                vec![local]
            } else {
                options
                    .candidates
                    .iter()
                    .map(|candidate| match candidate.port() {
                        0 => SocketAddr::new(candidate.ip(), local.port()),
                        port => SocketAddr::new(candidate.ip(), port.saturating_add(index as u16)),
                    })
                    .collect()
            };
            pp_info!(
                pp_log: &pp_log,
                "thread {index} bound to {local}, candidates {advertised:?}"
            );
            let (sender, commands) = bounded(256);
            senders.push(sender);
            shards.push(Shard {
                index,
                socket: Some(socket),
                local,
                candidates: advertised
                    .iter()
                    .filter_map(|addr| Candidate::host(*addr, "udp").ok())
                    .collect(),
                commands,
                pp_log: pp_log.clone(),
            });
        }
        let shared = Arc::new(Shared {
            pp_log: pp_log.clone(),
            paths: Mutex::new(HashMap::new()),
            publishers: Mutex::default(),
            next_registration: AtomicU64::new(0),
            local_addr: shards[0].local,
            next_session: AtomicU64::new(seed()),
            sessions: AtomicUsize::new(0),
            per_thread: (0..threads).map(|_| AtomicUsize::new(0)).collect(),
            publishing: Mutex::new(HashMap::new()),
            events: unbounded(),
            stats: Mutex::new(HashMap::new()),
            session_key: RandomState::new(),
            authorize: options.authorize.clone(),
            ice_servers: options.ice_servers.clone(),
        });
        Ok((
            Self {
                name,
                pp_log,
                shards,
                shared: Arc::clone(&shared),
            },
            WebRtcServerHandle {
                shared,
                commands: senders,
            },
        ))
    }
}

/// Where session numbers start: a different place each run, so a session's
/// URL from one is not another's in the next.
fn seed() -> u64 {
    use std::hash::BuildHasher;
    std::collections::hash_map::RandomState::new().hash_one(Instant::now()) >> 24
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
    /// Serves `broadcast` under `path` to players, from now on.
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
    ) -> std::result::Result<(), WebRtcServerError> {
        if !valid_path(path) {
            return Err(WebRtcServerError::InvalidPath(path.to_owned()));
        }
        let mut paths = self.shared.paths.lock().unwrap_or_else(|p| p.into_inner());
        if paths.contains_key(path) {
            return Err(WebRtcServerError::PathTaken(path.to_owned()));
        }
        paths.insert(path.to_owned(), broadcast.clone());
        drop(paths);
        pp_info!(
            pp_log: &self.shared.pp_log,
            "path added: {path} broadcast={}",
            broadcast.name()
        );
        Ok(())
    }

    /// Takes publishers on `path` — [`Self::publish`], WHIP to
    /// `/<path>/whip` — from now on, each handed out as a [`WebRtcSource`]
    /// of what it sends through the [`Publishers`] returned, and none once
    /// that is dropped. Apart from what players play under the same name.
    /// One publisher at a time.
    ///
    /// # Errors
    ///
    /// [`WebRtcServerError::InvalidPath`] for a path a URL could not name
    /// as it is, and [`WebRtcServerError::PathTaken`] for one that already
    /// takes publishers.
    pub fn take_publishers(
        &self,
        path: &str,
    ) -> std::result::Result<Publishers<WebRtcSource>, WebRtcServerError> {
        if !valid_path(path) {
            return Err(WebRtcServerError::InvalidPath(path.to_owned()));
        }
        let mut taken = self
            .shared
            .publishers
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        if taken.contains_key(path) {
            return Err(WebRtcServerError::PathTaken(path.to_owned()));
        }
        let registration = self
            .shared
            .next_registration
            .fetch_add(1, Ordering::Relaxed);
        let shared = Arc::downgrade(&self.shared);
        let name = path.to_owned();
        let (arrivals, publishers) = publishers::publishers(
            path,
            OnDrop::new(move || release(&shared, &name, registration)),
        );
        taken.insert(path.to_owned(), (registration, arrivals));
        drop(taken);
        pp_info!(pp_log: &self.shared.pp_log, "publishers taken on {path}");
        Ok(publishers)
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

    /// Answers `offer` — SDP from a publisher — with a session publishing on
    /// `path`, handed out through its [`Publishers`] once its tracks have
    /// said what they carry: the session, and the SDP answer to send back.
    ///
    /// # Errors
    ///
    /// [`WebRtcServerError::NoSuchPath`] where nothing is served or taken
    /// under `path`, [`WebRtcServerError::NotPublishable`] where it takes no
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
        // The thread with the fewest sessions.
        let thread = (0..self.commands.len())
            .min_by_key(|&thread| self.shared.per_thread[thread].load(Ordering::Acquire))
            .unwrap_or(0);
        self.ask(
            thread,
            Command::Open {
                path: path.to_owned(),
                role,
                offer: offer.to_owned(),
                answer,
            },
        )?;
        answered
            .recv_timeout(ANSWER_WITHIN)
            .map_err(|_| WebRtcServerError::NotRunning)?
    }

    /// Ends `session`. Whether there was one.
    pub fn close(&self, session: SessionId) -> bool {
        let (done, closed) = bounded(1);
        if self
            .ask(
                session.thread(self.commands.len()),
                Command::Close { session, done },
            )
            .is_err()
        {
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
        self.ask(
            session.thread(self.commands.len()),
            Command::Send {
                session,
                channel: channel.to_owned(),
                binary,
                data: data.to_vec(),
                done,
            },
        )?;
        sent.recv_timeout(ANSWER_WITHIN)
            .map_err(|_| WebRtcServerError::NotRunning)?
    }

    fn ask(&self, thread: usize, command: Command) -> std::result::Result<(), WebRtcServerError> {
        self.commands[thread]
            .send_timeout(command, ANSWER_WITHIN)
            .map_err(|_| WebRtcServerError::NotRunning)
    }

    /// The next thing that happened to a session, waiting up to `timeout`
    /// for one. Every clone of the handle reads the one queue, so each event
    /// goes to one reader.
    pub fn next_event(&self, timeout: Duration) -> Option<WebRtcServerEvent> {
        self.shared.events.1.recv_timeout(timeout).ok()
    }

    /// The address its first thread's socket is bound to.
    pub fn local_addr(&self) -> SocketAddr {
        self.shared.local_addr
    }

    /// Every open session's figures, as its connection last measured
    /// them — about once a second — in no particular order. See
    /// [`SessionStats`].
    pub fn session_stats(&self) -> Vec<SessionStats> {
        self.shared
            .stats
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .values()
            .cloned()
            .collect()
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
    pub(crate) fn path(&self, name: &str) -> Option<Broadcast> {
        self.paths
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(name)
            .cloned()
    }

    /// Where a publisher of `name` is handed out, where it takes any.
    fn publishers_of(&self, name: &str) -> Option<Arrivals<WebRtcSource>> {
        self.publishers
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(name)
            .map(|(_, arrivals)| arrivals.clone())
    }
}

/// Takes away what [`WebRtcServerHandle::take_publishers`] registered for
/// `path` as `registration`: no publisher begins on it from here, and one
/// under way goes on into the source it was handed out as.
fn release(shared: &Weak<Shared>, path: &str, registration: u64) {
    let Some(shared) = shared.upgrade() else {
        return;
    };
    let mut taken = shared.publishers.lock().unwrap_or_else(|p| p.into_inner());
    if taken.get(path).is_some_and(|(id, _)| *id == registration) {
        taken.remove(path);
        drop(taken);
        pp_info!(pp_log: &shared.pp_log, "publishers let go of on {path}");
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
    /// Runs every thread's sessions until asked to stop; each is closed
    /// then. The first thread is the driver's own.
    fn run(&mut self, stop: &StopReceiver, _bus: &Bus) -> Result<()> {
        if self.shards.iter().any(|shard| shard.socket.is_none()) {
            return Ok(());
        }
        pp_info!(self, "started: {} threads", self.shards.len());
        let shared = &*self.shared;
        let name = &self.name;
        std::thread::scope(|scope| {
            let (first, rest) = self.shards.split_at_mut(1);
            for shard in rest {
                let index = shard.index;
                let spawned = std::thread::Builder::new()
                    .name(format!("{name}-webrtc-{index}"))
                    .spawn_scoped(scope, move || shard.run(stop, shared));
                if let Err(error) = spawned {
                    pp_warn!(
                        pp_log: &shared.pp_log,
                        "thread {index} could not start: {error}"
                    );
                }
            }
            first[0].run(stop, shared);
        });
        // Nothing more will be handed out: each path's `Publishers` hears it.
        self.shared
            .publishers
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clear();
        pp_info!(self, "stopped");
        Ok(())
    }
}

impl Shard {
    /// Runs this thread's sessions until asked to stop.
    fn run(&mut self, stop: &StopReceiver, shared: &Shared) {
        let Some(socket) = self.socket.take() else {
            return;
        };
        let mut sessions: HashMap<SessionId, Session> = HashMap::new();
        // Which session each address sends from — looked at first, rather
        // than asking every session whether a datagram is its.
        let mut senders: HashMap<SocketAddr, SessionId> = HashMap::new();
        let mut buffer = vec![0u8; 65_536];
        let local = self.local;
        while !stop.is_stopped() {
            while let Ok(command) = self.commands.try_recv() {
                self.command(command, &socket, &mut sessions, shared);
            }
            // A session a handle closed leaves its addresses behind; a
            // session has one or two, so many more than sessions is that.
            if senders.len() > 2 * sessions.len() + 16 {
                senders.retain(|_, id| sessions.contains_key(id));
            }
            let now = Instant::now();
            let mut next = now + POLL;
            let mut ended = Vec::new();
            for (id, session) in &mut sessions {
                if !session.due(now) {
                    next = next.min(session.next());
                    continue;
                }
                match session.drive(now, &socket, shared) {
                    Ok(deadline) => next = next.min(deadline),
                    Err(why) => ended.push((*id, why)),
                }
            }
            if !ended.is_empty() {
                senders.retain(|_, id| !ended.iter().any(|(gone, _)| gone == id));
            }
            for (id, why) in ended {
                self.end(id, why, &mut sessions, shared);
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
                let stun = buffer[..length].first().is_some_and(|first| *first < 4);
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
                    // An address a session has taken STUN from is that
                    // session's for what else comes from it — DTLS and
                    // media. Not asked of `accepts`, which takes them only
                    // once the session has been polled since the client's
                    // nomination: a ClientHello right behind the nominating
                    // STUN, read in the same batch, was refused, and every
                    // connection waited for the client to send it again a
                    // second later. STUN is asked, as a new client may have
                    // the address a session it does not belong to had.
                    let known = senders.get(&source).copied().filter(|id| {
                        sessions
                            .get(id)
                            .is_some_and(|session| !stun || session.accepts(&input))
                    });
                    let found = known.or_else(|| {
                        let (id, _) = sessions
                            .iter()
                            .find(|(_, session)| session.accepts(&input))?;
                        senders.insert(source, *id);
                        Some(*id)
                    });
                    if let Some(session) = found.and_then(|id| sessions.get_mut(&id)) {
                        session.input(input);
                    }
                }
                // A session that took STUN is driven before anything else is
                // read: str0m begins DTLS as it is polled after the client's
                // nomination, and drops what DTLS comes before ("prior to
                // DTLS start") — a ClientHello read in the same batch was
                // lost to the client's retransmission. What else waits is
                // read once the sessions have been driven, at once.
                if stun {
                    break;
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
            self.end(id, "the server stopped".into(), &mut sessions, shared);
        }
    }

    fn command(
        &mut self,
        command: Command,
        socket: &UdpSocket,
        sessions: &mut HashMap<SessionId, Session>,
        shared: &Shared,
    ) {
        match command {
            Command::Open {
                path,
                role,
                offer,
                answer,
            } => {
                let reply = self
                    .open(&path, role, &offer, shared)
                    .map(|(id, session, sdp)| {
                        shared
                            .stats
                            .lock()
                            .unwrap_or_else(|p| p.into_inner())
                            .insert(id, session.stats().clone());
                        sessions.insert(id, session);
                        shared.sessions.fetch_add(1, Ordering::AcqRel);
                        shared.per_thread[self.index].fetch_add(1, Ordering::AcqRel);
                        pp_info!(
                            pp_log: &self.pp_log,
                            "session {id}: {role:?} {path} on thread {}",
                            self.index
                        );
                        (id, sdp)
                    });
                if let Err(error) = &reply {
                    pp_info!(
                        pp_log: &self.pp_log,
                        "an offer for {path} ({role:?}) was refused: {error}"
                    );
                }
                let _ = answer.send(reply);
            }
            Command::Close { session, done } => {
                let found = sessions.contains_key(&session);
                if found {
                    if let Some(open) = sessions.get_mut(&session) {
                        open.close(socket);
                    }
                    self.end(session, "closed".into(), sessions, shared);
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
        shared: &Shared,
    ) -> std::result::Result<(SessionId, Session, String), WebRtcServerError> {
        let target = match role {
            SessionRole::Play => {
                let broadcast = shared
                    .path(path)
                    .ok_or_else(|| WebRtcServerError::NoSuchPath(path.to_owned()))?;
                if !session::playable(&broadcast) {
                    return Err(WebRtcServerError::NothingToPlay(path.to_owned()));
                }
                Target::Play(broadcast)
            }
            SessionRole::Publish => {
                Target::Publish(shared.publishers_of(path).ok_or_else(|| {
                    if shared.path(path).is_some() {
                        WebRtcServerError::NotPublishable(path.to_owned())
                    } else {
                        WebRtcServerError::NoSuchPath(path.to_owned())
                    }
                })?)
            }
        };
        let number = shared.next_session.fetch_add(1, Ordering::Relaxed);
        let id = SessionId(
            u128::from(shared.session_key.hash_one(number)) << 64
                | u128::from((number << THREAD_BITS) | self.index as u64),
        );
        // A publisher holds its path from here: looked at and taken under
        // one lock, so that two threads cannot each give it one.
        let mut publishing = shared.publishing.lock().unwrap_or_else(|p| p.into_inner());
        if role == SessionRole::Publish && publishing.contains_key(path) {
            return Err(WebRtcServerError::NotPublishable(path.to_owned()));
        }
        let (session, answer) = Session::open(
            id,
            path,
            target,
            offer,
            &self.candidates,
            self.pp_log.clone(),
        )?;
        if role == SessionRole::Publish {
            publishing.insert(path.to_owned(), id);
        }
        Ok((id, session, answer))
    }

    fn end(
        &self,
        id: SessionId,
        why: String,
        sessions: &mut HashMap<SessionId, Session>,
        shared: &Shared,
    ) {
        let Some(session) = sessions.remove(&id) else {
            return;
        };
        shared
            .stats
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(&id);
        shared.sessions.fetch_sub(1, Ordering::AcqRel);
        shared.per_thread[self.index].fetch_sub(1, Ordering::AcqRel);
        shared
            .publishing
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .retain(|_, publisher| *publisher != id);
        pp_info!(
            pp_log: &self.pp_log,
            "session {id} ({}) ended: {why}",
            session.path()
        );
        let _ = shared
            .events
            .0
            .send(WebRtcServerEvent::Ended { session: id, why });
    }
}
