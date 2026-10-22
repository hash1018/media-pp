//! An RTMP server taking publishers' streams into [`Broadcast`]s:
//! [`RtmpServer`].

mod amf;
mod chunk;
mod flv;
mod ingest;
mod session;
#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet};
use std::io;
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
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

/// How many connections it serves at once; one more is closed at once.
const MAX_CONNECTIONS: usize = 256;

/// Why an [`RtmpServer`] or its handle refused something.
///
/// Converts into the crate-wide `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum RtmpServerError {
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
    /// A path an RTMP URL could not name: one without both an application
    /// and a stream name — `live/cam1` — or holding a character other than
    /// letters, digits, `-`, `_`, `.`, `~` and `/` between segments.
    #[error("{0:?} is not a path an RTMP URL can name")]
    InvalidPath(String),
}

/// How an [`RtmpServer`] runs.
#[derive(Debug, Clone)]
pub struct RtmpServerOptions {
    /// How long a publisher may send nothing — the handshake included —
    /// before it is taken to have gone, and its run ended. Ten seconds by
    /// default.
    pub publisher_timeout: Duration,
    /// Who may publish into which paths, asked for each publisher before it
    /// may; `None`, the default, lets everybody. See [`Authorize`]: RTMP
    /// has no `Authorization` header, so the credentials are what the
    /// stream name's query says — `?user=…&pass=…` as
    /// [`Credentials::Basic`](crate::elements::Credentials::Basic), or
    /// `?token=…` as [`Credentials::Bearer`](crate::elements::Credentials::Bearer).
    pub authorize: Option<Authorize>,
}

impl Default for RtmpServerOptions {
    fn default() -> Self {
        Self {
            publisher_timeout: Duration::from_secs(10),
            authorize: None,
        }
    }
}

/// An RTMP server: takes publishers' streams — OBS, FFmpeg, an encoder or
/// a camera pushing to `rtmp://<this machine>/<app>/<stream>` — into
/// [`Broadcast`]s, each under a path of its own. Needs the `rtmp-server`
/// feature.
///
/// It takes, and plays to no one: a path's broadcast is read by whatever
/// the application has — an `RtspServer` or a `WebRtcServer` serving it
/// to players, a recording. Nothing is
/// decoded: H.264 and AAC as RTMP has always carried them, and H.265, AV1,
/// VP9 and Opus as Enhanced RTMP does, go into the broadcast as they come.
/// A publisher that drops and comes back publishes again into the same
/// broadcast, and its readers go on.
///
/// It is not a pipeline element: run it with a
/// [`DriverRunner`](crate::driver::DriverRunner), and add paths through its
/// [`RtmpServerHandle`] before or while it runs.
///
/// # Example
///
/// With the `rtsp-server` feature too:
///
/// ```no_run
/// # #[cfg(feature = "rtsp-server")]
/// # fn main() -> media_pp::Result<()> {
/// use media_pp::driver::DriverRunner;
/// use media_pp::elements::{
///     Broadcast, BroadcastOptions, RtmpServer, RtmpServerOptions, RtspPathMode, RtspServer,
///     RtspServerOptions,
/// };
///
/// // OBS publishes to rtmp://<this machine>/live with the stream key
/// // `studio`; RTSP players play rtsp://<this machine>:8554/live/studio.
/// let studio = Broadcast::new("studio", BroadcastOptions::default());
/// let (rtmp, ingest) = RtmpServer::bind("rtmp", "0.0.0.0:1935", RtmpServerOptions::default())?;
/// ingest.add_path("live/studio", &studio)?;
/// let (rtsp, players) = RtspServer::bind("rtsp", "0.0.0.0:8554", RtspServerOptions::default())?;
/// players.add_path("live/studio", &studio, RtspPathMode::Play)?;
/// let rtmp = DriverRunner::new(rtmp);
/// rtmp.run()?;
/// let rtsp = DriverRunner::new(rtsp);
/// rtsp.run()?;
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "rtsp-server"))]
/// # fn main() {}
/// ```
///
/// # How it works
///
/// [`Self::bind`] listens at once, so a port already taken is an error
/// there; connections wait until it runs. Each connection is served on a
/// thread of its own — publishers are few, and each sends at its own pace
/// — up to 256 at once. A path takes one publisher at a time. What a
/// publisher sends is written into its path's broadcast once its tracks
/// are known: those its metadata names, each with its configuration, and
/// a picture's size, from the metadata or its first keyframe. A new
/// configuration mid-stream — a new resolution — publishes the broadcast
/// again. Stopping the server closes every connection and waits for its
/// thread.
pub struct RtmpServer {
    name: Arc<str>,
    pp_log: PpLog,
    listener: Option<TcpListener>,
    shared: Arc<Shared>,
}

/// A handle to an [`RtmpServer`], to say what it takes while it runs.
///
/// Cheap to clone; every clone is the same server's. It keeps the paths'
/// broadcasts it was given — so a broadcast served stays open while its
/// path is — and nothing else of the server: once the server has stopped,
/// adding a path still succeeds and nobody publishes to it.
#[derive(Clone)]
pub struct RtmpServerHandle {
    shared: Arc<Shared>,
}

/// What a server, its connections and its handles share.
pub(crate) struct Shared {
    pub(crate) pp_log: PpLog,
    paths: Mutex<HashMap<String, Broadcast>>,
    local_addr: SocketAddr,
    pub(crate) publisher_timeout: Duration,
    pub(crate) authorize: Option<Authorize>,
    /// The paths a publisher holds.
    pub(crate) publishing: Mutex<HashSet<String>>,
    /// How many publishers are publishing now.
    pub(crate) publishers: AtomicUsize,
    /// Each open connection's socket, by number, so that stopping can
    /// close them.
    connections: Mutex<HashMap<u64, TcpStream>>,
    next_connection: AtomicU64,
}

impl Shared {
    pub(crate) fn path(&self, name: &str) -> Option<Broadcast> {
        self.paths
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(name)
            .cloned()
    }
}

impl RtmpServer {
    /// A server listening on `addr` — `"0.0.0.0:1935"` for every interface
    /// on RTMP's own port, `"127.0.0.1:0"` for a port of the system's choice
    /// on this machine alone — and the handle that says what it takes.
    ///
    /// # Errors
    ///
    /// [`RtmpServerError::Bind`] where `addr` cannot be listened on.
    pub fn bind(
        name: impl Into<String>,
        addr: impl ToSocketAddrs,
        options: RtmpServerOptions,
    ) -> std::result::Result<(Self, RtmpServerHandle), RtmpServerError> {
        let name: Arc<str> = name.into().into();
        let addrs: Vec<SocketAddr> = addr
            .to_socket_addrs()
            .map_err(|source| RtmpServerError::Bind {
                addr: String::new(),
                source,
            })?
            .collect();
        let shown = addrs
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let bind_error = |source| RtmpServerError::Bind {
            addr: shown.clone(),
            source,
        };
        let listener = TcpListener::bind(&addrs[..]).map_err(bind_error)?;
        let local_addr = listener.local_addr().map_err(bind_error)?;
        listener.set_nonblocking(true).map_err(bind_error)?;
        let pp_log = element_pp_log(ElementType::RtmpServer, &name, None);
        pp_info!(pp_log: &pp_log, "listening on {local_addr}");
        let shared = Arc::new(Shared {
            pp_log: pp_log.clone(),
            paths: Mutex::new(HashMap::new()),
            local_addr,
            publisher_timeout: options.publisher_timeout,
            authorize: options.authorize,
            publishing: Mutex::new(HashSet::new()),
            publishers: AtomicUsize::new(0),
            connections: Mutex::new(HashMap::new()),
            next_connection: AtomicU64::new(0),
        });
        Ok((
            Self {
                name,
                pp_log,
                listener: Some(listener),
                shared: Arc::clone(&shared),
            },
            RtmpServerHandle { shared },
        ))
    }
}

impl RtmpServerHandle {
    /// Takes publishers into `broadcast` under `path` — `app/stream`, what
    /// `rtmp://<address>/app/stream` names, or what a publisher given the
    /// server `rtmp://<address>/app` and the stream key `stream` publishes
    /// to — from now on.
    ///
    /// # Errors
    ///
    /// [`RtmpServerError::InvalidPath`] for a path an RTMP URL could not
    /// name, and [`RtmpServerError::PathTaken`] for one already served; what
    /// was served under it goes on unchanged.
    pub fn add_path(
        &self,
        path: &str,
        broadcast: &Broadcast,
    ) -> std::result::Result<(), RtmpServerError> {
        if !valid_path(path) {
            return Err(RtmpServerError::InvalidPath(path.to_owned()));
        }
        let mut paths = self.shared.paths.lock().unwrap_or_else(|p| p.into_inner());
        if paths.contains_key(path) {
            return Err(RtmpServerError::PathTaken(path.to_owned()));
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

    /// Stops taking publishers into `path`. Whether it was served. A
    /// publisher already publishing goes on.
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

    /// The address it listens on.
    pub fn local_addr(&self) -> SocketAddr {
        self.shared.local_addr
    }

    /// The URL a publisher publishes to, to publish into `path`.
    pub fn url(&self, path: &str) -> String {
        format!("rtmp://{}/{path}", self.shared.local_addr)
    }

    /// How many publishers are publishing now.
    pub fn publishers(&self) -> usize {
        self.shared.publishers.load(Ordering::Acquire)
    }
}

impl std::fmt::Debug for RtmpServerHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RtmpServerHandle")
            .field("local_addr", &self.shared.local_addr)
            .finish()
    }
}

/// Whether `path` is one an RTMP URL names as it is: an application and a
/// stream name at least.
fn valid_path(path: &str) -> bool {
    path.split('/').count() >= 2
        && path.split('/').all(|segment| !segment.is_empty())
        && path
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '~' | '/'))
}

impl Element for RtmpServer {
    fn name(&self) -> Arc<str> {
        Arc::clone(&self.name)
    }

    fn element_type(&self) -> ElementType {
        ElementType::RtmpServer
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Driver for RtmpServer {
    /// Accepts connections and serves each on a thread of its own until
    /// asked to stop; every connection is closed then, and its thread
    /// joined.
    fn run(&mut self, stop: &StopReceiver, _bus: &Bus) -> Result<()> {
        let Some(listener) = self.listener.take() else {
            return Ok(());
        };
        pp_info!(self, "started");
        let mut threads: Vec<JoinHandle<()>> = Vec::new();
        while !stop.is_stopped() {
            match listener.accept() {
                Ok((stream, peer)) => {
                    threads.retain(|thread| !thread.is_finished());
                    if threads.len() >= MAX_CONNECTIONS {
                        pp_warn!(
                            self,
                            "refused a connection from {peer}: {MAX_CONNECTIONS} already"
                        );
                        continue;
                    }
                    let id = self.shared.next_connection.fetch_add(1, Ordering::Relaxed);
                    let Ok(kept) = stream.try_clone() else {
                        continue;
                    };
                    self.shared
                        .connections
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .insert(id, kept);
                    let shared = Arc::clone(&self.shared);
                    let spawned = thread::Builder::new()
                        .name(format!("{}-rtmp", self.name))
                        .spawn(move || {
                            session::serve(stream, peer, &shared);
                            shared
                                .connections
                                .lock()
                                .unwrap_or_else(|p| p.into_inner())
                                .remove(&id);
                        });
                    match spawned {
                        Ok(thread) => threads.push(thread),
                        Err(error) => {
                            pp_warn!(self, "could not serve {peer}: {error}");
                            self.shared
                                .connections
                                .lock()
                                .unwrap_or_else(|p| p.into_inner())
                                .remove(&id);
                        }
                    }
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    thread::sleep(STOP_POLL);
                }
                Err(error) => {
                    pp_warn!(self, "accepting a connection failed: {error}");
                    thread::sleep(STOP_POLL);
                }
            }
        }
        for stream in self
            .shared
            .connections
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .values()
        {
            let _ = stream.shutdown(Shutdown::Both);
        }
        for thread in threads {
            let _ = thread.join();
        }
        pp_info!(self, "stopped");
        Ok(())
    }
}
