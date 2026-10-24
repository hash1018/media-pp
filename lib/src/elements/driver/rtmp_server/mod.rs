//! An RTMP server taking publishers, each as an [`RtmpSource`]:
//! [`RtmpServer`].

mod amf;
mod chunk;
mod flv;
mod ingest;
mod session;
mod source;
#[cfg(test)]
mod tests;

pub use source::RtmpSource;

use std::collections::{HashMap, HashSet};
use std::io;
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use thiserror::Error as ThisError;

use crate::bus::Bus;
use crate::driver::{Driver, StopReceiver};
use crate::element::{Element, ElementType, element_pp_log};
use crate::elements::Authorize;
use crate::elements::driver::publishers::{self, Arrivals, OnDrop, Publishers};
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
    /// A path already takes publishers.
    #[error("path {0:?} already takes publishers")]
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

/// An RTMP server: takes publishers — OBS, FFmpeg, an encoder or a camera
/// pushing to `rtmp://<this machine>/<app>/<stream>` — each handed out as an
/// [`RtmpSource`] of what it sends, through the [`Publishers`] of its path
/// ([`RtmpServerHandle::take_publishers`]). Needs the `rtmp-server` feature.
///
/// It takes, and plays to no one: what a publisher sends goes wherever the
/// application's pipeline of its source takes it — a broadcast an
/// `RtspServer` or a `WebRtcServer` serves to players, a recording. Nothing
/// is decoded: H.264 and AAC as RTMP has always carried them, and H.265,
/// AV1, VP9 and Opus as Enhanced RTMP does, come out as they come in. A
/// publisher that drops and comes back is handed out again, as a source of
/// its own.
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
/// use media_pp::elements::{RtmpServer, RtmpServerOptions, RtspServer, RtspServerOptions};
/// use media_pp::pipeline::Pipeline;
///
/// // OBS publishes to rtmp://<this machine>/live with the stream key
/// // `studio`; RTSP players play rtsp://<this machine>:8554/live/studio.
/// let (rtmp, ingest) = RtmpServer::bind("rtmp", "0.0.0.0:1935", RtmpServerOptions::default())?;
/// let publishers = ingest.take_publishers("live/studio")?;
/// let (rtsp, players) = RtspServer::bind("rtsp", "0.0.0.0:8554", RtspServerOptions::default())?;
/// let studio = players.path("live/studio")?;
/// let rtmp = DriverRunner::new(rtmp);
/// rtmp.run()?;
/// let rtsp = DriverRunner::new(rtsp);
/// rtsp.run()?;
///
/// // Each publisher, as it comes, a run of what the players play.
/// while let Some((source, streams)) = publishers.next() {
///     let mut muxer = studio.muxer();
///     let tracks = streams
///         .iter()
///         .map(|stream| muxer.add_stream(format!("track-{}", stream.index()), stream))
///         .collect::<media_pp::Result<Vec<_>>>()?;
///     let mut sinks = muxer.open()?;
///     let (pipeline, ()) = Pipeline::new("studio", source, |source, ctx| {
///         for (stream, track) in streams.iter().zip(tracks) {
///             ctx.attach(source, stream.index(), ctx.branch().to(sinks.take(track)?)?)?;
///         }
///         Ok(())
///     })?;
///     pipeline.run()?;
///     // ... until its bus says `Finished`: the publisher has gone.
/// #   break;
/// }
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
/// — up to 256 at once. A path takes one publisher at a time. A publisher
/// is handed out once its tracks are known: those its metadata names, each
/// with its configuration, and a picture's size, from the metadata or its
/// first keyframe. A new configuration mid-stream — a new resolution —
/// ends its source and hands out another. Stopping the server closes every
/// connection and waits for its thread, and the paths' [`Publishers`]
/// answer `None`.
pub struct RtmpServer {
    name: Arc<str>,
    pp_log: PpLog,
    listener: Option<TcpListener>,
    shared: Arc<Shared>,
}

/// A handle to an [`RtmpServer`], to say what it takes while it runs.
///
/// Cheap to clone; every clone is the same server's, and keeps nothing of
/// it but what it shares with the paths' [`Publishers`].
#[derive(Clone)]
pub struct RtmpServerHandle {
    shared: Arc<Shared>,
}

/// What a server, its connections and its handles share.
pub(crate) struct Shared {
    pub(crate) pp_log: PpLog,
    /// Where a publisher of each path that takes them is handed out, under
    /// the number of the registration, so that a `Publishers` gone takes
    /// away only its own.
    paths: Mutex<HashMap<String, (u64, Arrivals<RtmpSource>)>>,
    next_registration: AtomicU64,
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
    /// Where a publisher of `name` is handed out, where it takes any.
    pub(crate) fn path(&self, name: &str) -> Option<Arrivals<RtmpSource>> {
        self.paths
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(name)
            .map(|(_, arrivals)| arrivals.clone())
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
            next_registration: AtomicU64::new(0),
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
    /// Takes publishers on `path` — `app/stream`, what
    /// `rtmp://<address>/app/stream` names, or what a publisher given the
    /// server `rtmp://<address>/app` and the stream key `stream` publishes
    /// to — from now on, each handed out as an [`RtmpSource`] of what it
    /// sends through the [`Publishers`] returned, and none once that is
    /// dropped.
    ///
    /// # Errors
    ///
    /// [`RtmpServerError::InvalidPath`] for a path an RTMP URL could not
    /// name, and [`RtmpServerError::PathTaken`] for one that already takes
    /// publishers.
    pub fn take_publishers(
        &self,
        path: &str,
    ) -> std::result::Result<Publishers<RtmpSource>, RtmpServerError> {
        if !valid_path(path) {
            return Err(RtmpServerError::InvalidPath(path.to_owned()));
        }
        let mut paths = self.shared.paths.lock().unwrap_or_else(|p| p.into_inner());
        if paths.contains_key(path) {
            return Err(RtmpServerError::PathTaken(path.to_owned()));
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
        paths.insert(path.to_owned(), (registration, arrivals));
        drop(paths);
        pp_info!(pp_log: &self.shared.pp_log, "publishers taken on {path}");
        Ok(publishers)
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

/// Takes away what [`RtmpServerHandle::take_publishers`] registered for
/// `path` as `registration`: no publisher begins on it from here, and one
/// under way goes on into the source it was handed out as.
fn release(shared: &Weak<Shared>, path: &str, registration: u64) {
    let Some(shared) = shared.upgrade() else {
        return;
    };
    let mut paths = shared.paths.lock().unwrap_or_else(|p| p.into_inner());
    if paths.get(path).is_some_and(|(id, _)| *id == registration) {
        paths.remove(path);
        drop(paths);
        pp_info!(pp_log: &shared.pp_log, "publishers let go of on {path}");
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
        // Nothing more will be handed out: each path's `Publishers` hears it.
        self.shared
            .paths
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clear();
        pp_info!(self, "stopped");
        Ok(())
    }
}
