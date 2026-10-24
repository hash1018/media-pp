//! An RTSP server serving [`Broadcast`]s and taking publishers:
//! [`RtspServer`].

mod digest;
mod record;
pub(crate) mod rtp;
mod session;
#[cfg(test)]
mod tests;
mod track;

pub use digest::RtspDigest;

use std::collections::{HashMap, HashSet};
use std::io;
use std::net::{SocketAddr, TcpListener, ToSocketAddrs, UdpSocket};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use thiserror::Error as ThisError;

use crate::bus::Bus;
use crate::driver::{Driver, StopReceiver};
use crate::element::{Element, ElementType, element_pp_log};
use crate::elements::driver::publishers::{self, Arrivals, OnDrop, Publishers};
use crate::elements::driver::served::Served;
use crate::elements::{Access, Authorize, Broadcast, BroadcastOptions, RtspMuxer, RtspSource};
use crate::error::Result;
use crate::pp_log::{PpLog, pp_info, pp_warn};

/// How often the server looks at whether it has been asked to stop.
const STOP_POLL: Duration = Duration::from_millis(50);

/// Why an [`RtspServer`] or its handle refused something.
///
/// Converts into the crate-wide `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum RtspServerError {
    /// The address could not be listened on — taken, or not this machine's.
    #[error("could not listen on {addr}: {source}")]
    Bind {
        /// The address asked for.
        addr: String,
        /// Why.
        source: io::Error,
    },
    /// A path is already served under that name — or, for
    /// [`RtspServerHandle::take_publishers`], already takes publishers.
    #[error("path {0:?} is already served")]
    PathTaken(String),
    /// A path a URL could not name as it is: empty, or holding a character
    /// other than letters, digits, `-`, `_`, `.`, `~` and `/` between
    /// segments.
    #[error("{0:?} is not a path a URL can name")]
    InvalidPath(String),
    /// The server's threads could not be started.
    #[error("could not start the server's runtime: {0}")]
    Runtime(io::Error),
}

/// How an [`RtspServer`] runs.
#[derive(Debug, Clone)]
pub struct RtspServerOptions {
    /// How many threads it runs its connections on. Two by default.
    pub threads: usize,
    /// Whether players may ask for RTP over UDP, on two ports the server
    /// opens beside its own; otherwise only interleaved on the RTSP
    /// connection, which every player can ask for. On by default.
    pub udp: bool,
    /// How long sending one packet to a player over TCP may take before the
    /// player is dropped as gone. Ten seconds by default.
    pub write_timeout: Duration,
    /// How long a publisher may send nothing before it is taken to have
    /// gone, and its run ended — one over UDP, or one that stopped without
    /// closing its connection. Ten seconds by default.
    pub publisher_timeout: Duration,
    /// Who may play and publish which paths, asked for each client before
    /// it may; `None`, the default, lets everybody. See [`Authorize`].
    pub authorize: Option<Authorize>,
}

impl Default for RtspServerOptions {
    fn default() -> Self {
        Self {
            threads: 2,
            udp: true,
            write_timeout: Duration::from_secs(10),
            publisher_timeout: Duration::from_secs(10),
            authorize: None,
        }
    }
}

/// An RTSP server: plays what is written through each of its paths' muxers
/// to RTSP players ([`RtspServerHandle::path`]), and takes RTSP publishers — `ANNOUNCE` and `RECORD` — on the
/// paths that take them, each handed out as an [`RtspSource`] of what it
/// sends ([`RtspServerHandle::take_publishers`]). Needs the `rtsp-server`
/// feature.
///
/// A path's stream is written once, through an [`RtspMuxer`] its
/// [`RtspServerPath`] makes, by whatever pipeline the application has — an
/// [`RtspSource`](crate::elements::RtspSource) pulling a camera, an
/// encoder — and every player reads it: nothing is decoded or encoded for
/// them, and each is sent only its own copy of the RTP packets. A player
/// starts at the newest keyframe the path holds, or waits for the next one
/// written; one that cannot keep up is moved on to a newer keyframe rather
/// than slowing the others or the input. Each muxer opened is a run of the
/// path's stream: an input that drops and comes back opens another, and
/// its players go on.
///
/// It is not a pipeline element: run it with a
/// [`DriverRunner`](crate::driver::DriverRunner), and take paths through
/// its [`RtspServerHandle`] before or while it runs.
///
/// # Example
///
/// ```no_run
/// # fn main() -> media_pp::Result<()> {
/// use media_pp::driver::DriverRunner;
/// use media_pp::elements::{RtspOptions, RtspServer, RtspServerOptions, RtspSource};
/// use media_pp::pipeline::Pipeline;
///
/// let (server, handle) = RtspServer::bind("server", "0.0.0.0:8554", RtspServerOptions::default())?;
/// let cam1 = handle.path("cam1")?;
/// let server = DriverRunner::new(server);
/// server.run()?;
///
/// // The camera, pulled once, written through the path's muxer; players
/// // open rtsp://<this machine>:8554/cam1.
/// let (source, streams) = RtspSource::open("camera", "rtsp://192.168.0.10/stream", RtspOptions::default())?;
/// let mut muxer = cam1.muxer();
/// let tracks = streams
///     .iter()
///     .map(|stream| muxer.add_stream(format!("track-{}", stream.index()), stream))
///     .collect::<media_pp::Result<Vec<_>>>()?;
/// let mut sinks = muxer.open()?;
/// let (input, ()) = Pipeline::new("camera", source, |source, ctx| {
///     for (stream, track) in streams.iter().zip(tracks) {
///         ctx.attach(source, stream.index(), ctx.branch().to(sinks.take(track)?)?)?;
///     }
///     Ok(())
/// })?;
/// input.run()?;
/// # Ok(())
/// # }
/// ```
///
/// # How it works
///
/// [`Self::bind`] listens at once, so a port already taken is an error
/// there; connections wait until it runs. It runs on threads of its own
/// ([`RtspServerOptions::threads`]), a runtime it owns inside its driver's
/// thread, and stops — every connection closed — when its runner is stopped
/// or dropped.
///
/// A player starts at the newest keyframe the path holds — within a
/// second — or at the next one written, so it is at most that far behind
/// the input. What came after that keyframe is sent at once to a player
/// over TCP, whose connection paces it, and at twice the stream's speed to
/// one over UDP, which nothing paces: sent at once, it overflows the
/// player's socket. A UDP player whose newest keyframe is more than a second
/// behind waits for the next one instead, since what it loses of a longer
/// stretch it cannot ask for again. A UDP player costs the server several times what a TCP
/// one does — a system call for every RTP packet — so many of them want
/// more [`RtspServerOptions::threads`]; `docs/benchmarks/streaming` has the
/// figures.
///
/// Each player is a pipeline of its own with no thread of its own — a
/// [`DrivenPipeline`](crate::pipeline::DrivenPipeline) of a
/// [`BroadcastSource`](crate::elements::BroadcastSource) of its path's
/// broadcast, and a sink for each track it set up — which its connection
/// polls once it has sent all the pipeline gave it, so a player that takes
/// its packets slowly is left behind in the broadcast rather than queued
/// for here. It is sent RTP as FFmpeg's `rtp` muxer cuts it — any codec
/// that muxer carries, described by the SDP it writes. A player's tracks
/// share one timeline, so they stay together, and that timeline goes on
/// across the broadcast's runs: one that begins with tracks of a different
/// shape ends the player's session, since its description no longer holds.
/// A session lasts as long as its connection.
///
/// A publisher is taken on a path apart from what players play under the
/// same name: a pipeline of the application's carries what it sends — an
/// `RtspSource` like a pulled camera's — wherever the application wants it,
/// a broadcast its players read among them, converted or not.
pub struct RtspServer {
    name: Arc<str>,
    pp_log: PpLog,
    listener: Option<TcpListener>,
    udp: Option<(UdpSocket, UdpSocket)>,
    shared: Arc<Shared>,
    threads: usize,
}

/// A handle to an [`RtspServer`], to say what it serves while it runs.
///
/// Cheap to clone; every clone is the same server's, and keeps nothing of
/// it but what it shares with its paths: once the server has stopped,
/// taking a path still succeeds and nobody connects to it, and the paths'
/// [`Publishers`] answer `None`.
#[derive(Clone)]
pub struct RtspServerHandle {
    shared: Arc<Shared>,
}

/// What a server and its handles share.
pub(crate) struct Shared {
    pub(crate) pp_log: PpLog,
    paths: Mutex<HashMap<String, Path>>,
    pub(crate) local_addr: SocketAddr,
    /// The server's RTP and RTCP ports, where it takes UDP.
    pub(crate) udp_ports: Option<(u16, u16)>,
    pub(crate) write_timeout: Duration,
    pub(crate) publisher_timeout: Duration,
    pub(crate) authorize: Option<Authorize>,
    /// Every session playing or publishing now, by a number of its own —
    /// see [`RtspServerHandle::session_stats`].
    pub(crate) sessions: Mutex<HashMap<u64, Arc<SessionCounts>>>,
    pub(crate) next_session: AtomicU64,
    /// The paths a publisher holds.
    pub(crate) publishing: Mutex<HashSet<String>>,
    /// Where a publisher of each path that takes them is handed out, apart
    /// from what its players read — see
    /// [`RtspServerHandle::take_publishers`] — under the number of the
    /// registration, so that a `Publishers` gone takes away only its own.
    publishers: Mutex<HashMap<String, (u64, Arrivals<RtspSource>)>>,
    next_registration: AtomicU64,
    /// Where a publisher's RTP over UDP comes from, and which of its tracks
    /// and connection it goes to.
    pub(crate) routes:
        Mutex<HashMap<SocketAddr, (usize, tokio::sync::mpsc::Sender<session::Datagram>)>>,
}

#[derive(Clone)]
pub(crate) struct Path {
    pub(crate) broadcast: Broadcast,
    /// The number of the registration, so that an `RtspServerPath` gone
    /// takes away only its own.
    registration: u64,
}

impl Shared {
    pub(crate) fn path(&self, name: &str) -> Option<Path> {
        self.paths
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(name)
            .cloned()
    }

    /// Where a publisher of `name` is handed out. `Err` with the status to
    /// answer where nothing takes it: 403 for a path players alone may use,
    /// 404 for none.
    pub(crate) fn publish_target(
        &self,
        name: &str,
    ) -> std::result::Result<Arrivals<RtspSource>, u16> {
        if let Some((_, arrivals)) = self
            .publishers
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(name)
        {
            return Ok(arrivals.clone());
        }
        match self.path(name) {
            Some(_) => Err(403),
            None => Err(404),
        }
    }
}

impl RtspServer {
    /// Whether a player is sent `track`: one in a codec RTP carries here —
    /// H.264, H.265, VP8, VP9, MJPEG, MPEG-4 and -2, AAC, Opus, MP3,
    /// G.711, G.722 and the rest of FFmpeg's packetizers. A broadcast's track
    /// in any other is left out of what its players are described; an
    /// application that wants it sent converts it first.
    pub fn carries(track: &crate::elements::StreamInfo) -> bool {
        rtp::carries(track)
    }

    /// A server listening on `addr` — `"0.0.0.0:8554"` for every interface
    /// on RTSP's own port, `"127.0.0.1:0"` for a port of the system's choice
    /// on this machine alone — and the handle that says what it serves.
    ///
    /// # Errors
    ///
    /// [`RtspServerError::Bind`] where `addr` cannot be listened on, or —
    /// with [`RtspServerOptions::udp`] — two UDP ports beside it cannot be
    /// opened.
    pub fn bind(
        name: impl Into<String>,
        addr: impl ToSocketAddrs,
        options: RtspServerOptions,
    ) -> std::result::Result<(Self, RtspServerHandle), RtspServerError> {
        let name: Arc<str> = name.into().into();
        let addrs: Vec<SocketAddr> = addr
            .to_socket_addrs()
            .map_err(|source| RtspServerError::Bind {
                addr: String::new(),
                source,
            })?
            .collect();
        let shown = addrs
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let bind_error = |source| RtspServerError::Bind {
            addr: shown.clone(),
            source,
        };
        let listener = TcpListener::bind(&addrs[..]).map_err(bind_error)?;
        let local_addr = listener.local_addr().map_err(bind_error)?;
        let udp = if options.udp {
            let open = || udp_socket(SocketAddr::new(local_addr.ip(), 0)).map_err(bind_error);
            Some((open()?, open()?))
        } else {
            None
        };
        let udp_ports = match &udp {
            Some((rtp, rtcp)) => Some((
                rtp.local_addr().map_err(bind_error)?.port(),
                rtcp.local_addr().map_err(bind_error)?.port(),
            )),
            None => None,
        };
        let pp_log = element_pp_log(ElementType::RtspServer, &name, None);
        pp_info!(
            pp_log: &pp_log,
            "listening on {local_addr}, udp={udp_ports:?} threads={}",
            options.threads
        );
        let shared = Arc::new(Shared {
            pp_log: pp_log.clone(),
            paths: Mutex::new(HashMap::new()),
            local_addr,
            udp_ports,
            write_timeout: options.write_timeout,
            publisher_timeout: options.publisher_timeout,
            authorize: options.authorize,
            sessions: Mutex::default(),
            next_session: AtomicU64::new(0),
            publishing: Mutex::new(HashSet::new()),
            publishers: Mutex::default(),
            next_registration: AtomicU64::new(0),
            routes: Mutex::new(HashMap::new()),
        });
        Ok((
            Self {
                name,
                pp_log,
                listener: Some(listener),
                udp,
                shared: Arc::clone(&shared),
                threads: options.threads.max(1),
            },
            RtspServerHandle { shared },
        ))
    }
}

impl RtspServerHandle {
    /// Serves `path` — `rtsp://<address>/<path>` — to players, for as long
    /// as the [`RtspServerPath`] returned lives: what is written through
    /// its muxers is what they play.
    ///
    /// # Errors
    ///
    /// [`RtspServerError::InvalidPath`] for a path a URL could not name as it
    /// is, and [`RtspServerError::PathTaken`] for one already served; what
    /// was served under it goes on unchanged.
    pub fn path(&self, path: &str) -> std::result::Result<RtspServerPath, RtspServerError> {
        if !valid_path(path) {
            return Err(RtspServerError::InvalidPath(path.to_owned()));
        }
        let mut paths = self.shared.paths.lock().unwrap_or_else(|p| p.into_inner());
        if paths.contains_key(path) {
            return Err(RtspServerError::PathTaken(path.to_owned()));
        }
        let registration = self
            .shared
            .next_registration
            .fetch_add(1, Ordering::Relaxed);
        let broadcast = Broadcast::new(path, BroadcastOptions::default());
        paths.insert(
            path.to_owned(),
            Path {
                broadcast: broadcast.clone(),
                registration,
            },
        );
        drop(paths);
        pp_info!(pp_log: &self.shared.pp_log, "path served: {path}");
        let shared = Arc::downgrade(&self.shared);
        let name = path.to_owned();
        Ok(RtspServerPath {
            path: path.into(),
            served: Served::new(
                broadcast,
                OnDrop::new(move || unserve(&shared, &name, registration)),
            ),
        })
    }

    /// Takes publishers on `path` — `ANNOUNCE` and `RECORD` to
    /// `rtsp://<address>/<path>` — from now on, each handed out as an
    /// [`RtspSource`] of what it sends through the [`Publishers`] returned,
    /// and none once that is dropped. Apart from what players play under
    /// the same name: an application that makes what they play of what is
    /// published — converts it, say — gives them one URL for both. One
    /// publisher at a time.
    ///
    /// # Errors
    ///
    /// [`RtspServerError::InvalidPath`] for a path no URL can carry, and
    /// [`RtspServerError::PathTaken`] where `path` already takes publishers.
    pub fn take_publishers(
        &self,
        path: &str,
    ) -> std::result::Result<Publishers<RtspSource>, RtspServerError> {
        if !valid_path(path) {
            return Err(RtspServerError::InvalidPath(path.to_owned()));
        }
        let mut taken = self
            .shared
            .publishers
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        if taken.contains_key(path) {
            return Err(RtspServerError::PathTaken(path.to_owned()));
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

    /// The address it listens on — with the port the system chose, where
    /// it was asked for port 0.
    pub fn local_addr(&self) -> SocketAddr {
        self.shared.local_addr
    }

    /// The URL a player opens `path` at, by the address it listens on.
    pub fn url(&self, path: &str) -> String {
        format!("rtsp://{}/{path}", self.shared.local_addr)
    }

    /// How many players are playing now, every path together.
    pub fn players(&self) -> usize {
        self.shared
            .sessions
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .values()
            .filter(|session| session.role == Access::Play)
            .count()
    }

    /// Every session playing or publishing now, in no particular order —
    /// see [`RtspSessionStats`]. A session is counted from its `PLAY` or
    /// `RECORD` until its connection ends. Copies a few figures of each
    /// under a lock; nothing is asked of the sessions themselves.
    pub fn session_stats(&self) -> Vec<RtspSessionStats> {
        self.shared
            .sessions
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .values()
            .map(|session| RtspSessionStats {
                path: session.path.clone(),
                role: session.role,
                remote: session.remote,
                udp: session.udp,
                opened: session.opened,
                bytes_sent: session.sent.load(Ordering::Relaxed),
                bytes_received: session.received.load(Ordering::Relaxed),
            })
            .collect()
    }
}

/// A path an [`RtspServer`] serves to players, from
/// [`RtspServerHandle::path`], for as long as it lives: dropped, nobody new
/// plays it, and its players' sessions end.
///
/// What its players play is what is written through its muxers — each
/// [`Self::muxer`] opened a run of its stream, which they go on across, as
/// they would across a camera's reconnections. One muxer at a time: one
/// opened while another's run goes on ends that run. Holds nothing of the
/// server.
#[derive(Debug)]
pub struct RtspServerPath {
    path: Arc<str>,
    served: Served,
}

impl RtspServerPath {
    /// The path it serves.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// A muxer of a run of the path's stream: given its tracks with
    /// `add_stream` and opened as one to an RTSP server is, a sink per
    /// track. Opening it begins a run; its sinks all let go of end it. A
    /// track in a codec RTP does not carry here is left out of what
    /// players are described — see [`RtspServer::carries`].
    pub fn muxer(&self) -> RtspMuxer {
        RtspMuxer::serving(&self.path, self.served.broadcast.clone())
    }

    /// How many players play it now: the RTSP sessions reading what its
    /// muxers write.
    pub fn players(&self) -> usize {
        self.served.broadcast.stats().readers
    }

    /// What its players read, for this crate's own tests.
    #[cfg(test)]
    pub(crate) fn broadcast(&self) -> &Broadcast {
        &self.served.broadcast
    }
}

/// Takes away the path [`RtspServerHandle::path`] registered as
/// `registration`: nobody new plays it.
fn unserve(shared: &Weak<Shared>, path: &str, registration: u64) {
    let Some(shared) = shared.upgrade() else {
        return;
    };
    let mut paths = shared.paths.lock().unwrap_or_else(|p| p.into_inner());
    if paths
        .get(path)
        .is_some_and(|served| served.registration == registration)
    {
        paths.remove(path);
        drop(paths);
        pp_info!(pp_log: &shared.pp_log, "path no longer served: {path}");
    }
}

/// Takes away what [`RtspServerHandle::take_publishers`] registered for
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

/// One session of an [`RtspServer`] — a player or a publisher — as
/// [`RtspServerHandle::session_stats`] last read it.
///
/// The byte counts only rise, from the session's `PLAY` or `RECORD`; a rate
/// is two readings apart. They are of RTP, without the framing that
/// interleaves it on the connection.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct RtspSessionStats {
    /// Its path.
    pub path: String,
    /// Whether it plays its path or publishes into it.
    pub role: Access,
    /// Where its connection comes from.
    pub remote: SocketAddr,
    /// Whether its media goes over UDP rather than its connection.
    pub udp: bool,
    /// When it began to play or publish.
    pub opened: Instant,
    /// What it has been sent: a player's media. Nothing for a publisher.
    pub bytes_sent: u64,
    /// What it has sent: a publisher's media. Nothing for a player.
    pub bytes_received: u64,
}

/// What a session counts of itself while it lasts, for
/// [`RtspServerHandle::session_stats`].
pub(crate) struct SessionCounts {
    pub(crate) path: String,
    pub(crate) role: Access,
    pub(crate) remote: SocketAddr,
    pub(crate) udp: bool,
    pub(crate) opened: Instant,
    pub(crate) sent: AtomicU64,
    pub(crate) received: AtomicU64,
}

impl std::fmt::Debug for RtspServerHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RtspServerHandle")
            .field("local_addr", &self.shared.local_addr)
            .finish()
    }
}

/// A UDP socket bound to `addr`, its buffers as large as the system lets
/// them be up to [`UDP_BUFFER`]: every UDP player's packets leave through
/// the one socket, and a keyframe for each arrives at once — a buffer of the
/// system's default size (64 KiB on Windows) holds a fraction of one.
fn udp_socket(addr: SocketAddr) -> io::Result<UdpSocket> {
    use socket2::{Domain, Protocol, Socket, Type};
    let socket = Socket::new(Domain::for_address(addr), Type::DGRAM, Some(Protocol::UDP))?;
    // Not fatal: a smaller buffer only loses more under load.
    let _ = socket.set_send_buffer_size(UDP_BUFFER);
    let _ = socket.set_recv_buffer_size(UDP_BUFFER);
    socket.bind(&addr.into())?;
    Ok(socket.into())
}

/// How large the server's UDP sockets' buffers are asked to be.
const UDP_BUFFER: usize = 8 * 1024 * 1024;

/// Whether `path` is one a URL names as it is.
fn valid_path(path: &str) -> bool {
    !path.is_empty()
        && path
            .split('/')
            .all(|segment| !segment.is_empty() && !segment.starts_with("streamid="))
        && path
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '~' | '/'))
}

impl Element for RtspServer {
    fn name(&self) -> Arc<str> {
        Arc::clone(&self.name)
    }

    fn element_type(&self) -> ElementType {
        ElementType::RtspServer
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Driver for RtspServer {
    /// Accepts connections and serves each on the server's runtime until
    /// asked to stop; every connection is closed then.
    fn run(&mut self, stop: &StopReceiver, _bus: &Bus) -> Result<()> {
        let Some(listener) = self.listener.take() else {
            return Ok(());
        };
        let udp = self.udp.take();
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(self.threads)
            .thread_name(format!("{}-rtsp", self.name))
            .enable_io()
            .enable_time()
            .build()
            .map_err(RtspServerError::Runtime)?;
        let shared = Arc::clone(&self.shared);
        let pp_log = self.pp_log.clone();
        let result: io::Result<()> = runtime.block_on(async move {
            listener.set_nonblocking(true)?;
            let listener = tokio::net::TcpListener::from_std(listener)?;
            let udp = match udp {
                Some((rtp, rtcp)) => Some(Arc::new(session::Udp::new(rtp, rtcp)?)),
                None => None,
            };
            if let Some(udp) = &udp {
                tokio::spawn(Arc::clone(udp).receive(Arc::clone(&shared)));
            }
            pp_info!(pp_log: &pp_log, "started");
            let mut tick = tokio::time::interval(STOP_POLL);
            loop {
                tokio::select! {
                    accepted = listener.accept() => match accepted {
                        Ok((stream, peer)) => {
                            let _ = stream.set_nodelay(true);
                            tokio::spawn(session::serve(stream, peer, Arc::clone(&shared), udp.clone()));
                        }
                        Err(error) => {
                            pp_warn!(pp_log: &pp_log, "accepting a connection failed: {error}");
                            tokio::time::sleep(STOP_POLL).await;
                        }
                    },
                    _ = tick.tick() => {
                        if stop.is_stopped() {
                            break;
                        }
                    }
                }
            }
            Ok(())
        });
        runtime.shutdown_timeout(Duration::from_secs(1));
        // Nothing more will be handed out: each path's `Publishers` hears it.
        self.shared
            .publishers
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clear();
        pp_info!(self, "stopped");
        result.map_err(|error| RtspServerError::Runtime(error).into())
    }
}
