//! An RTSP server serving [`Broadcast`]s: [`RtspServer`].

mod announce;
mod depacketize;
mod digest;
mod message;
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
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use thiserror::Error as ThisError;

use crate::bus::Bus;
use crate::driver::{Driver, StopReceiver};
use crate::element::{Element, ElementType, element_pp_log};
use crate::elements::{Authorize, Broadcast};
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
    /// A path is already served under that name.
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

/// What an [`RtspServer`] lets clients do on a path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtspPathMode {
    /// Players may play it; what is written into its broadcast comes from
    /// the application.
    Play,
    /// A publisher may also push into it — `ANNOUNCE` and `RECORD` — one at
    /// a time, whose stream is published into its broadcast; players play
    /// it as ever.
    Record,
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

/// An RTSP server: plays [`Broadcast`]s to RTSP players, each under a path
/// of its own, and — on a path that allows it — takes an RTSP publisher's
/// stream into one. Needs the `rtsp-server` feature.
///
/// A broadcast is written once, by whatever pipeline the application has —
/// an [`RtspSource`](crate::elements::RtspSource) pulling a camera, an
/// encoder — and every player reads it: nothing is decoded or encoded for
/// them, and each is sent only its own copy of the RTP packets. A player
/// starts at the newest keyframe the broadcast holds, so it shows a picture
/// at once; one that cannot keep up is moved on to a newer keyframe rather
/// than slowing the others or the input. An input that drops and comes back
/// publishes again into the same broadcast, and its players go on.
///
/// It is not a pipeline element: run it with a
/// [`DriverRunner`](crate::driver::DriverRunner), and add paths through its
/// [`RtspServerHandle`] before or while it runs.
///
/// # Example
///
/// ```no_run
/// # fn main() -> media_pp::Result<()> {
/// use media_pp::driver::DriverRunner;
/// use media_pp::elements::{
///     Broadcast, BroadcastOptions, RtspOptions, RtspPathMode, RtspServer, RtspServerOptions,
///     RtspSource, TrackFormat,
/// };
/// use media_pp::pipeline::Pipeline;
///
/// let (server, handle) = RtspServer::bind("server", "0.0.0.0:8554", RtspServerOptions::default())?;
/// let camera = Broadcast::new("camera", BroadcastOptions::default());
/// handle.add_path("cam1", &camera, RtspPathMode::Play)?;
/// let server = DriverRunner::new(server);
/// server.run()?;
///
/// // The camera, pulled once, written into the broadcast; players open
/// // rtsp://<this machine>:8554/cam1.
/// let (source, streams) = RtspSource::open("camera", "rtsp://192.168.0.10/stream", RtspOptions::default())?;
/// let tracks: Vec<TrackFormat> = streams.iter().map(TrackFormat::from).collect();
/// let sinks = camera.publish(&tracks)?;
/// let (input, ()) = Pipeline::new("camera", source, |source, ctx| {
///     for (stream, sink) in streams.iter().zip(sinks) {
///         let branch = ctx.branch().queue("packets", 64).to(sink)?;
///         ctx.attach(source, stream.index, branch)?;
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
/// A player starts at the newest keyframe, so it is up to a group of
/// pictures behind the input — as far as the input's keyframes are apart —
/// and stays so. What came after that keyframe is sent at once to a player
/// over TCP, whose connection paces it, and at twice the stream's speed to
/// one over UDP, which nothing paces: sent at once, it overflows the
/// player's socket. A UDP player costs the server several times what a TCP
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
/// Cheap to clone; every clone is the same server's. It keeps the paths'
/// broadcasts it was given — so a broadcast served stays open while its
/// path is — and nothing else of the server: once the server has stopped,
/// adding a path still succeeds and nobody connects to it.
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
    /// How many players are playing now.
    pub(crate) playing: AtomicUsize,
    /// The paths a publisher holds.
    pub(crate) publishing: Mutex<HashSet<String>>,
    /// Where a publisher's RTP over UDP comes from, and which of its tracks
    /// and connection it goes to.
    pub(crate) routes:
        Mutex<HashMap<SocketAddr, (usize, tokio::sync::mpsc::Sender<session::Datagram>)>>,
}

#[derive(Clone)]
pub(crate) struct Path {
    pub(crate) broadcast: Broadcast,
    pub(crate) mode: RtspPathMode,
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

impl RtspServer {
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
            playing: AtomicUsize::new(0),
            publishing: Mutex::new(HashSet::new()),
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
    /// Serves `broadcast` under `path` — at `rtsp://<address>/<path>` — as
    /// `mode` allows, from now on.
    ///
    /// # Errors
    ///
    /// [`RtspServerError::InvalidPath`] for a path a URL could not name as it
    /// is, and [`RtspServerError::PathTaken`] for one already served; what
    /// was served under it goes on unchanged.
    pub fn add_path(
        &self,
        path: &str,
        broadcast: &Broadcast,
        mode: RtspPathMode,
    ) -> std::result::Result<(), RtspServerError> {
        if !valid_path(path) {
            return Err(RtspServerError::InvalidPath(path.to_owned()));
        }
        let mut paths = self.shared.paths.lock().unwrap_or_else(|p| p.into_inner());
        if paths.contains_key(path) {
            return Err(RtspServerError::PathTaken(path.to_owned()));
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
    /// Players already playing it go on until its broadcast closes or they
    /// leave.
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
        self.shared.playing.load(Ordering::Acquire)
    }
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
        pp_info!(self, "stopped");
        result.map_err(|error| RtspServerError::Runtime(error).into())
    }
}
