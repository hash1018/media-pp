use std::{
    collections::HashSet,
    ffi::CString,
    net::{TcpListener, TcpStream, UdpSocket},
    ops::RangeInclusive,
    path::PathBuf,
    process::{Child, Command, Stdio},
    ptr,
    sync::{
        LazyLock, Mutex,
        atomic::{AtomicU32, Ordering},
    },
    time::{Duration, Instant},
};

use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Element, ElementType, Sink},
    error::Result,
};

/// How long to wait for the spawned `mediamtx` to accept its first TCP
/// connection before giving up. Generous because multiple `RtspServer`s
/// starting at once (each its own `mediamtx` process) can make an
/// individual instance slower to come up than it would be alone.
const MEDIAMTX_STARTUP_TIMEOUT: Duration = Duration::from_secs(10);

/// Ports currently claimed by a live `RtspServer` in this process, shared
/// across every instance — the RTSP port itself, plus RTP/RTCP if
/// [`ViewerTransport::TcpAndUdp`] reserved any. A bare "bind port 0, read
/// back what the OS assigned, then bind that same number again in
/// `mediamtx`" trick is racy even against *other `RtspServer`s in this
/// same process* — this closes that gap by having every instance check
/// and reserve through the same table before trusting a port. It can't do
/// anything about a completely unrelated process grabbing the port in
/// between; that race is inherent to the technique and accepted for this
/// crate's dev/demo scope.
static RESERVED_PORTS: LazyLock<Mutex<HashSet<u16>>> = LazyLock::new(|| Mutex::new(HashSet::new()));

/// Whether a candidate port needs to be free as a TCP or a UDP port —
/// independent OS namespaces, so [`is_free`] has to test with the matching
/// socket type.
#[derive(Debug, Clone, Copy)]
enum PortKind {
    Tcp,
    Udp,
}

/// How [`RtspServer::new`] resolves which port to actually bind, and what
/// to do if a preferred one is already taken. An explicit argument rather
/// than a hidden default — mirrors [`crate::queue::OverflowPolicy`] — since
/// which of these is right depends entirely on the caller: a single
/// well-known port vs. an arbitrary number of these running side by side.
#[derive(Debug, Clone)]
pub enum PortPolicy {
    /// Use exactly the port in `url`. Fails with [`RtspServerError::PortInUse`]
    /// if it's already bound — by another `RtspServer` in this process or
    /// by anything else on the system.
    Exact,
    /// Try the exact port in `url` first; if it's taken, fall back to any
    /// free port in `range`.
    PreferExact { range: RangeInclusive<u16> },
    /// Ignore whatever port `url` has (if any) and always pick a free port
    /// in `range` — the right choice when an arbitrary number of
    /// `RtspServer`s might run at once and none of them needs a
    /// predictable port. `range` bounds where those end up (e.g. so only
    /// that range needs opening in a firewall), instead of leaving it to
    /// whatever the OS's own ephemeral port allocator would pick.
    Any { range: RangeInclusive<u16> },
}

/// Transport [`RtspServer::new`] uses for its own publish leg (this
/// process pushing packets into `mediamtx`) — independent of
/// [`ViewerTransport`], which governs what transport *viewers* connect
/// with afterwards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PublishTransport {
    /// Lossless. Recommended, and the default: at this test clip's ~25
    /// Mbit/s, `Udp` dropped enough packets on loopback to corrupt HEVC
    /// fragmentation units and leave viewers with no decodable frames at
    /// all, even though `ANNOUNCE`/`RECORD` themselves succeeded.
    #[default]
    Tcp,
    /// Lower latency than `Tcp`, but no retransmission — only safe if the
    /// content's bitrate and the network path between this process and
    /// `mediamtx` can be trusted not to drop packets.
    Udp,
}

impl PublishTransport {
    fn as_ffmpeg_option(self) -> &'static str {
        match self {
            PublishTransport::Tcp => "tcp",
            PublishTransport::Udp => "udp",
        }
    }
}

/// Transports mediamtx accepts from *viewers* connecting to watch —
/// independent of [`PublishTransport`], which only governs this process's
/// own leg into mediamtx.
#[derive(Debug, Clone, Default)]
pub enum ViewerTransport {
    /// TCP only. The default: needs no dedicated RTP/RTCP ports, so it
    /// can't collide with another concurrent `RtspServer`'s.
    #[default]
    TcpOnly,
    /// Both TCP and UDP; mediamtx negotiates whichever a connecting viewer
    /// asks for. Needs two extra UDP ports (RTP + RTCP), picked from
    /// `range` the same way [`PortPolicy::Any`] picks the RTSP port —
    /// tracked in the same table, so concurrent `RtspServer`s don't
    /// collide on these either.
    TcpAndUdp { range: RangeInclusive<u16> },
}

/// Errors specific to `RtspServer`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum RtspServerError {
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),

    #[error(
        "RtspServer only remuxes compressed Packets, got a decoded {0}; \
         link it straight after a demuxer, not a decoder"
    )]
    UnsupportedBuffer(&'static str),

    #[error("RTSP URL contains a NUL byte")]
    InvalidUrl,

    #[error("no host in RTSP URL {0:?}")]
    UrlMissingAuthority(String),

    #[error("invalid port {0:?} in RTSP URL")]
    InvalidPort(String),

    #[error("PortPolicy::Exact/PreferExact needs a real, non-zero port in the URL")]
    PortRequired,

    #[error("port {0} is already in use")]
    PortInUse(u16),

    #[error("no free port in {0:?}")]
    NoFreePortInRange(RangeInclusive<u16>),

    #[error(
        "mediamtx exited during startup ({0}) — likely lost a race for one \
         of its ports against something outside this process"
    )]
    MediamtxExited(std::process::ExitStatus),

    #[error("failed to write a temporary mediamtx config to {0}: {1}")]
    WriteConfig(PathBuf, std::io::Error),

    #[error(
        "failed to spawn `mediamtx` (needs the `rtsp-server` feature enabled, \
         which vendors it next to the built binary): {0}"
    )]
    SpawnMediamtx(std::io::Error),

    #[error("mediamtx didn't start listening on {0} within {1:?}")]
    MediamtxNotReady(String, Duration),
}

/// Terminal sink that spawns [MediaMTX](https://github.com/bluenviron/mediamtx)
/// as a child process and publishes incoming compressed `Packet`s into it
/// (via RTSP `ANNOUNCE`/`RECORD`), so this element is a self-contained RTSP
/// server from the outside — nothing else needs to be running first, and
/// anyone can watch by pointing a player (`ffplay`, VLC, ...) at `url`. This
/// only remuxes — it does not encode — so upstream must already produce a
/// codec RTSP can carry (H.264/H.265/AAC, ...); the natural source is
/// [`crate::elements::FileDemuxer`]'s packet pad, not a decoder's frame
/// output.
///
/// Internally this is still a *publish* (RTSP client pushing into a real
/// media server), not ffmpeg's own `rtsp_flags=listen` muxer — that mode
/// only ever serves one client with no reconnect story, and hung outright
/// on at least one ffmpeg build encountered during development. Delegating
/// the actual serving to MediaMTX gets multi-viewer support and reconnects
/// for free. The generated config also turns off mediamtx's other
/// protocols (RTMP/HLS/WebRTC/SRT/MoQ), since otherwise several fixed
/// ports (not just the RTSP one `PortPolicy` governs) would collide the
/// moment a second `RtspServer` started.
///
/// Requires the crate's `rtsp-server` feature, which vendors the
/// `mediamtx` binary (`third_party/mediamtx/`) and copies it next to
/// whatever binary ends up depending on this crate (see `lib/build.rs`).
/// `Command::new("mediamtx")` finds it there because Windows searches the
/// calling executable's own directory before `PATH` — no `PATH` setup
/// needed. Windows-only for now, since only a `mediamtx.exe` is vendored.
///
/// `new` spawns `mediamtx`, waits for it to start accepting connections,
/// then publishes to it — so it blocks briefly but doesn't wait on an
/// external process being started separately. The child is killed (and
/// its ports released back for another `RtspServer` to use) when this is
/// dropped.
///
/// Packets arrive as fast as upstream produces them — pace them to real
/// playback speed first (put a [`crate::elements::Pacer`] upstream) or
/// viewers will receive the whole file far faster than realtime.
pub struct RtspServer {
    name: String,
    url: String,
    output: ffmpeg::format::context::Output,
    input_time_base: ffmpeg::Rational,
    mediamtx: Child,
    ports: Vec<u16>,
    /// Last dts actually written to `output` (in *output* time_base
    /// units) — what `consume` re-anchors `pts_offset` against right
    /// after a `Seek`. Anchored on dts, not pts: the muxer rejects a
    /// non-monotonic dts outright (`write_interleaved` errors), while
    /// pts is allowed to reorder (B-frames) and has no such hard
    /// requirement. Matching *pts* continuity across the seam instead
    /// doesn't guarantee dts stays increasing — a B-frame's own pts
    /// trails its dts, so if the last packet written before the seek
    /// happened to be one, the pts-matched offset undershoots dts by
    /// however far behind it was. Same `pts_offset` still applies to
    /// both pts and dts (see that field), so each packet's own pts/dts
    /// gap is preserved regardless of which one the anchor is based on.
    last_output_dts: Option<i64>,
    /// Same role as `last_output_dts`, for packets where `dts()` is
    /// `None` (anchor falls back to this — see `consume`).
    last_output_pts: Option<i64>,
    /// Added to every packet's (rescaled) pts/dts before writing.
    /// Constant between seeks — recomputed only when `pending_seek` is
    /// consumed, so relative spacing between consecutive frames is
    /// preserved; only the absolute baseline shifts.
    pts_offset: i64,
    /// Set by `control(Seek)`, consumed by the next `consume(Packet)` —
    /// deferred because the actual jump size (in output time_base units)
    /// isn't knowable until that packet's raw pts is in hand.
    pending_seek: bool,
}

impl RtspServer {
    /// `params`/`time_base` describe the incoming packets (see
    /// [`crate::elements::FileDemuxer::stream_parameters`] /
    /// [`crate::elements::FileDemuxer::stream_time_base`]) — they must
    /// match whatever gets pushed into `consume`. `url` is what this
    /// element publishes to, e.g. `"rtsp://127.0.0.1:8554/stream"` — its
    /// port is only a starting point, though, not necessarily final: see
    /// `port_policy`. Viewers should connect to [`RtspServer::url`]
    /// afterwards, which reflects whatever port was actually used.
    ///
    /// `publish_transport` only affects this process's own leg into
    /// `mediamtx`; `viewer_transport` only affects what viewers can use
    /// afterwards — the two are independent.
    pub fn new(
        name: impl Into<String>,
        url: &str,
        port_policy: PortPolicy,
        viewer_transport: ViewerTransport,
        publish_transport: PublishTransport,
        params: ffmpeg::codec::Parameters,
        time_base: ffmpeg::Rational,
    ) -> Result<Self> {
        let parsed = parse_url(url)?;
        let mut port_guard = PortGuard::default();

        let port = reserve_port(&parsed.host, parsed.port, &port_policy, PortKind::Tcp)?;
        port_guard.add(port);
        let viewer = resolve_viewer_transport(&parsed.host, &viewer_transport, &mut port_guard)?;

        let authority = format!("{}:{port}", parsed.host);
        let url = format!("rtsp://{authority}{}", parsed.path);

        let config_path = write_temp_config(&authority, &viewer)?;
        let mediamtx = Command::new("mediamtx")
            .arg(&config_path)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .inspect_err(|_| {
                let _ = std::fs::remove_file(&config_path);
            })
            .map_err(RtspServerError::SpawnMediamtx)?;
        let mut mediamtx_guard = MediamtxGuard(Some(mediamtx));

        let ready = wait_for_listener(&authority, mediamtx_guard.child_mut());
        let _ = std::fs::remove_file(&config_path); // mediamtx already read it (or tried to)
        ready?;

        let output = Self::publish(&url, params, time_base, publish_transport)?;

        Ok(Self {
            name: name.into(),
            url,
            output,
            input_time_base: time_base,
            mediamtx: mediamtx_guard.disarm(),
            ports: port_guard.disarm(),
            last_output_dts: None,
            last_output_pts: None,
            pts_offset: 0,
            pending_seek: false,
        })
    }

    /// The RTSP URL viewers should connect to — same as the `url` passed
    /// to [`RtspServer::new`], except with its actual port substituted in.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Publishes to the (already-confirmed-listening) `mediamtx` over
    /// `transport`.
    fn publish(
        url: &str,
        params: ffmpeg::codec::Parameters,
        time_base: ffmpeg::Rational,
        transport: PublishTransport,
    ) -> Result<ffmpeg::format::context::Output> {
        let mut output = alloc_output(url)?;

        {
            let mut stream = output
                .add_stream(ffmpeg::encoder::find(ffmpeg::codec::Id::None))
                .map_err(RtspServerError::from)?;
            stream.set_parameters(params);
            // Avoid incompatible codec-tag issues remuxing into a
            // different container than the source demuxer used — same fix
            // as ffmpeg-next's own `remux` example.
            unsafe {
                (*stream.parameters().as_mut_ptr()).codec_tag = 0;
            }
            stream.set_time_base(time_base);
        }

        // No `rtsp_flags` set: without `listen`, the RTSP muxer's
        // `write_header` connects *out* to `url` and does ANNOUNCE/RECORD
        // against the `mediamtx` instance just spawned above.
        let mut options = ffmpeg::Dictionary::new();
        options.set("rtsp_transport", transport.as_ffmpeg_option());
        output
            .write_header_with(options)
            .map_err(RtspServerError::from)?;

        Ok(output)
    }
}

/// Releases every port in `self.0` from [`RESERVED_PORTS`] on drop unless
/// [`Self::disarm`] was called first — so any early return (via `?`)
/// between reserving a port and fully constructing an `RtspServer` still
/// frees whatever had been reserved so far, without every fallible step in
/// between having to remember to.
#[derive(Default)]
struct PortGuard(Vec<u16>);

impl PortGuard {
    fn add(&mut self, port: u16) {
        self.0.push(port);
    }

    fn disarm(mut self) -> Vec<u16> {
        std::mem::take(&mut self.0)
    }
}

impl Drop for PortGuard {
    fn drop(&mut self) {
        if !self.0.is_empty() {
            let mut reserved = RESERVED_PORTS.lock().unwrap();
            for port in &self.0 {
                reserved.remove(port);
            }
        }
    }
}

/// Kills `self.0` on drop unless [`Self::disarm`] was called first — same
/// rationale as [`PortGuard`], for the spawned `mediamtx` child instead of
/// a port.
struct MediamtxGuard(Option<Child>);

impl MediamtxGuard {
    fn child_mut(&mut self) -> &mut Child {
        self.0.as_mut().expect("MediamtxGuard already disarmed")
    }

    fn disarm(mut self) -> Child {
        self.0.take().expect("MediamtxGuard already disarmed")
    }
}

impl Drop for MediamtxGuard {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// `url` split into the pieces [`reserve_port`] and reassembly need. `port`
/// is `None` when `url` had none, or had an explicit `0` — both mean "no
/// preference".
struct ParsedUrl {
    host: String,
    port: Option<u16>,
    path: String,
}

fn parse_url(url: &str) -> Result<ParsedUrl> {
    let rest = url.strip_prefix("rtsp://").unwrap_or(url);
    let path_start = rest.find('/').unwrap_or(rest.len());
    let (authority, path) = rest.split_at(path_start);
    if authority.is_empty() {
        return Err(RtspServerError::UrlMissingAuthority(url.to_string()).into());
    }

    let (host, port) = match authority.rsplit_once(':') {
        Some((host, port_str)) => {
            let port: u16 = port_str
                .parse()
                .map_err(|_| RtspServerError::InvalidPort(port_str.to_string()))?;
            (host, if port == 0 { None } else { Some(port) })
        }
        None => (authority, None),
    };

    Ok(ParsedUrl {
        host: host.to_string(),
        port,
        path: path.to_string(),
    })
}

/// Picks the port `RtspServer::new` will actually use, per `policy`, and
/// reserves it in [`RESERVED_PORTS`] before returning it.
fn reserve_port(
    host: &str,
    requested: Option<u16>,
    policy: &PortPolicy,
    kind: PortKind,
) -> Result<u16> {
    let mut reserved = RESERVED_PORTS.lock().unwrap();

    if let PortPolicy::Exact = policy {
        let port = requested.ok_or(RtspServerError::PortRequired)?;
        return if is_free(&reserved, host, port, kind) {
            reserved.insert(port);
            Ok(port)
        } else {
            Err(RtspServerError::PortInUse(port).into())
        };
    }

    let (preferred, range) = match policy {
        PortPolicy::PreferExact { range } => (requested, range.clone()),
        PortPolicy::Any { range } => (None, range.clone()),
        PortPolicy::Exact => unreachable!("handled above"),
    };

    for port in preferred.into_iter().chain(range.clone()) {
        if is_free(&reserved, host, port, kind) {
            reserved.insert(port);
            return Ok(port);
        }
    }

    Err(RtspServerError::NoFreePortInRange(range).into())
}

/// Not already claimed by another `RtspServer` in this process, and
/// actually bindable right now as the given `kind` of port.
fn is_free(reserved: &HashSet<u16>, host: &str, port: u16, kind: PortKind) -> bool {
    if reserved.contains(&port) {
        return false;
    }
    match kind {
        PortKind::Tcp => TcpListener::bind((host, port)).is_ok(),
        PortKind::Udp => UdpSocket::bind((host, port)).is_ok(),
    }
}

/// The pieces of a generated mediamtx config that depend on
/// [`ViewerTransport`]: which transports to advertise, and (only for
/// [`ViewerTransport::TcpAndUdp`]) which RTP/RTCP ports to bind them to.
struct ViewerConfig {
    transports_line: &'static str,
    addresses_lines: String,
}

/// Resolves `viewer_transport` into config lines, reserving RTP/RTCP ports
/// (via `port_guard`, so they're released if a later step fails) if it
/// calls for any.
fn resolve_viewer_transport(
    host: &str,
    viewer_transport: &ViewerTransport,
    port_guard: &mut PortGuard,
) -> Result<ViewerConfig> {
    match viewer_transport {
        ViewerTransport::TcpOnly => Ok(ViewerConfig {
            transports_line: "[tcp]",
            addresses_lines: String::new(),
        }),
        ViewerTransport::TcpAndUdp { range } => {
            let policy = PortPolicy::Any {
                range: range.clone(),
            };
            let rtp = reserve_port(host, None, &policy, PortKind::Udp)?;
            port_guard.add(rtp);
            let rtcp = reserve_port(host, None, &policy, PortKind::Udp)?;
            port_guard.add(rtcp);

            Ok(ViewerConfig {
                transports_line: "[udp, tcp]",
                addresses_lines: format!("rtpAddress: :{rtp}\nrtcpAddress: :{rtcp}\n"),
            })
        }
    }
}

/// Writes a minimal mediamtx config to a fresh file in the OS temp dir and
/// returns its path: bind to `authority`, apply `viewer`'s transport
/// settings, and allow publishing to any path (mediamtx's own default —
/// no config file, or one without a `paths` section — rejects `ANNOUNCE`
/// to a path it doesn't already know about with `400 Bad Request`).
fn write_temp_config(authority: &str, viewer: &ViewerConfig) -> Result<PathBuf> {
    static COUNTER: AtomicU32 = AtomicU32::new(0);

    let path = std::env::temp_dir().join(format!(
        "media-pp-mediamtx-{}-{}.yml",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    // Everything but RTSP disabled: mediamtx also listens for RTMP/HLS/
    // WebRTC/SRT/MoQ by default, each on a fixed port — harmless with one
    // instance, but a second concurrent `RtspServer` (its own mediamtx,
    // only its RTSP — and, with `ViewerTransport::TcpAndUdp`, RTP/RTCP —
    // ports governed by a `PortPolicy`/range) would collide on those and
    // fail to start even though its other ports were free. We only ever
    // publish over RTSP, so there's nothing to lose by turning the rest
    // off. (MoQ additionally writes a self-signed TLS cert —
    // `auto.crt`/`auto.key` — into whatever directory mediamtx happens to
    // be run from the first time it starts, which is one more reason to
    // skip it.)
    let contents = format!(
        "rtspAddress: {authority}\n\
         rtspTransports: {}\n\
         {}\
         rtmp: false\n\
         hls: false\n\
         webrtc: false\n\
         srt: false\n\
         moq: false\n\
         paths:\n  all_others:\n",
        viewer.transports_line, viewer.addresses_lines,
    );
    std::fs::write(&path, contents).map_err(|e| RtspServerError::WriteConfig(path.clone(), e))?;
    Ok(path)
}

/// Polls `authority` with plain TCP connects until one succeeds (mediamtx
/// is accepting RTSP connections by then), `mediamtx` itself exits (it
/// failed to bind *something* — one of its config's several addresses,
/// not necessarily `authority`; the actual reason only ever reached this
/// process's discarded stdout/stderr), or `MEDIAMTX_STARTUP_TIMEOUT` runs
/// out.
///
/// Checking the child's own liveness (not just whether *some* listener
/// answers on `authority`) matters because of the same cross-process
/// port race [`RESERVED_PORTS`]'s docs describe: if two `RtspServer`s in
/// different processes raced onto the same RTSP port, connecting there
/// might succeed even though *this* `mediamtx` never came up at all —
/// silently handing this instance's packets to a sibling process's
/// server instead of its own. A dead child means the connect, if it ever
/// succeeds, wasn't to this `mediamtx`.
fn wait_for_listener(authority: &str, mediamtx: &mut Child) -> Result<()> {
    let deadline = Instant::now() + MEDIAMTX_STARTUP_TIMEOUT;
    loop {
        if let Ok(Some(status)) = mediamtx.try_wait() {
            return Err(RtspServerError::MediamtxExited(status).into());
        }
        if TcpStream::connect(authority).is_ok() {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(RtspServerError::MediamtxNotReady(
                authority.to_string(),
                MEDIAMTX_STARTUP_TIMEOUT,
            )
            .into());
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// Allocates an RTSP output context without opening a generic `AVIOContext`
/// for it. `ffmpeg_next::format::output_as`/`_with` always calls
/// `avio_open2`, but the RTSP muxer sets `AVFMT_NOFILE` — it manages its
/// own control/RTP sockets internally (inside `write_header`/`write_frame`)
/// and never touches `ctx->pb`. Forcing an `avio_open2` on an `rtsp://` URL
/// fails outright on ffmpeg builds that don't register a generic `rtsp`
/// `AVIOContext` protocol (this one included — `rtsp` shows up in `ffmpeg
/// -formats` but not in `ffmpeg -protocols`), even though the RTSP *muxer*
/// works fine. Real `ffmpeg`/`ffplay` skip the open for any `AVFMT_NOFILE`
/// format for the same reason (see `ffmpeg.c`'s `open_output_file`); this
/// mirrors that.
fn alloc_output(url: &str) -> Result<ffmpeg::format::context::Output> {
    let c_url = CString::new(url).map_err(|_| RtspServerError::InvalidUrl)?;
    let c_format = CString::new("rtsp").expect("static format name has no interior NUL");

    unsafe {
        let mut ps: *mut ffi::AVFormatContext = ptr::null_mut();
        let ret = ffi::avformat_alloc_output_context2(
            &mut ps,
            ptr::null_mut(),
            c_format.as_ptr(),
            c_url.as_ptr(),
        );
        if ret < 0 {
            return Err(RtspServerError::Ffmpeg(ffmpeg::Error::from(ret)).into());
        }

        Ok(ffmpeg::format::context::Output::wrap(ps))
    }
}

impl Element for RtspServer {
    fn name(&self) -> &str {
        &self.name
    }

    fn element_type(&self) -> ElementType {
        ElementType::RtspServer
    }
}

impl Sink for RtspServer {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Packet(packet) => {
                let mut packet = (*packet).clone();
                let output_time_base = self
                    .output
                    .stream(0)
                    .expect("stream 0 was added in new()")
                    .time_base();
                packet.rescale_ts(self.input_time_base, output_time_base);

                if let Some(raw_pts) = packet.pts() {
                    if self.pending_seek {
                        // Continue smoothly from whatever was last
                        // written, ignoring how far this actually jumped
                        // — a viewer only ever sees the output side.
                        // Anchored on dts (see `last_output_dts`) since
                        // that's what the muxer actually requires to be
                        // monotonic; falls back to pts only for a packet
                        // with no dts at all.
                        self.pts_offset = match (self.last_output_dts, packet.dts()) {
                            (Some(last_dts), Some(raw_dts)) => last_dts + 1 - raw_dts,
                            _ => match self.last_output_pts {
                                Some(last_pts) => last_pts + 1 - raw_pts,
                                None => 0, // nothing written yet: raw pts is fine as-is
                            },
                        };
                        self.pending_seek = false;
                    }
                    let corrected_pts = raw_pts + self.pts_offset;
                    packet.set_pts(Some(corrected_pts));
                    if let Some(raw_dts) = packet.dts() {
                        // Same offset as pts, so the original pts/dts gap
                        // (relevant for B-frame reordering) is preserved.
                        let corrected_dts = raw_dts + self.pts_offset;
                        packet.set_dts(Some(corrected_dts));
                        self.last_output_dts = Some(corrected_dts);
                    }
                    self.last_output_pts = Some(corrected_pts);
                }

                packet.set_stream(0);
                packet.set_position(-1);
                packet
                    .write_interleaved(&mut self.output)
                    .map_err(RtspServerError::from)?;
                Ok(())
            }
            MediaBuffer::Eos => Ok(self.output.write_trailer().map_err(RtspServerError::from)?),
            MediaBuffer::Video(_) => Err(RtspServerError::UnsupportedBuffer("Video").into()),
            MediaBuffer::Audio(_) => Err(RtspServerError::UnsupportedBuffer("Audio").into()),
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        // Terminal, nothing to forward. `Stop` doesn't write a trailer
        // here (unlike natural `Eos`) — `Stop` means abandon, and `Drop`
        // already kills `mediamtx` unconditionally.
        //
        // `Seek`: just flag it — see `pending_seek`'s docs for why the
        // actual pts correction waits for `consume`'s next packet
        // instead of happening here.
        if let ControlMsg::Seek(_) = msg {
            self.pending_seek = true;
        }
        Ok(())
    }
}

impl Drop for RtspServer {
    fn drop(&mut self) {
        // mediamtx has no portable graceful-shutdown API from here (no
        // SIGTERM equivalent in `std` on Windows); killing it is enough
        // since it owns no state we need flushed — the stream itself was
        // already ended by `write_trailer` above on Eos.
        let _ = self.mediamtx.kill();
        let _ = self.mediamtx.wait();
        let mut reserved = RESERVED_PORTS.lock().unwrap();
        for port in &self.ports {
            reserved.remove(port);
        }
    }
}
