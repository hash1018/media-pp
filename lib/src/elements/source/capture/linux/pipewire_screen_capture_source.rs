use std::{
    sync::{
        Arc, Mutex,
        mpsc::{self, RecvTimeoutError},
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

use enumflags2::BitFlags;
use ffmpeg_next as ffmpeg;
use pipewire as pw;
use pw::{properties::properties, spa};
use spa::pod::Pod;
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_error, pp_info, pp_warn};

use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    control::{ControlReceiver, drain_control},
    element::{Element, ElementType, Source, SourceElement, element_pp_log},
    error::Result,
    pad::SrcPad,
    pool::UnboundObjectPool,
    schedule::PeriodicSchedule,
};

/// How long [`PipeWireScreenCaptureSource::open`] waits for the PipeWire stream to
/// finish negotiating a format after the portal handshake already succeeded.
/// Only covers the machine-to-machine part — the user is done choosing by the
/// time this starts — so a generous-but-finite bound is enough to turn a
/// compositor that never answers into an error instead of a permanent hang.
const NEGOTIATION_TIMEOUT: Duration = Duration::from_secs(5);

/// Bounds `Stop` latency at very low configured [`PipeWireScreenCaptureOptions::fps`]
/// values, where "wait until the next tick" on its own could otherwise be a
/// long, unresponsive block. Same idea as `DxgiCaptureSource`'s own constant.
const POLL_GRANULARITY: Duration = Duration::from_millis(100);

/// Errors specific to `PipeWireScreenCaptureSource`. Converts into the crate-wide
/// `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum PipeWireScreenCaptureSourceError {
    #[error("xdg-desktop-portal error: {0}")]
    Portal(String),

    /// The user dismissed the screen-share dialog, or a supplied
    /// [`PipeWireScreenCaptureOptions::restore_token`] was rejected and the
    /// re-prompt was then cancelled. Distinct from [`Self::Portal`] because
    /// it is a routine outcome — the caller decides whether to re-prompt,
    /// give up, or fall back — not a malfunction to report as a failure.
    #[error("the user cancelled the screen-share dialog")]
    Cancelled,

    /// The portal reported success but handed back no stream at all. Nothing
    /// this element can capture from.
    #[error("the portal returned no screen-cast stream")]
    NoStream,

    #[error("pipewire error: {0}")]
    PipeWire(String),

    /// The stream connected but never produced a `Format` param within
    /// [`NEGOTIATION_TIMEOUT`], so this element never learned what size or
    /// pixel layout to expect.
    #[error("timed out waiting for the PipeWire stream to negotiate a format")]
    NegotiationTimeout,

    /// The compositor offered only formats this element does not accept. v1
    /// negotiates CPU-mapped `BGRx`/`BGRA` only — see
    /// [`PipeWireScreenCaptureSource`]'s own docs on why DMA-BUF is out of scope.
    #[error("compositor negotiated unsupported video format {0:?}")]
    UnsupportedFormat(u32),

    #[error("PipeWireScreenCaptureSource doesn't support seeking a live capture")]
    SeekUnsupported,
}

/// Which kinds of source the portal's picker offers the user — see
/// [`PipeWireScreenCaptureOptions::source_kind`].
///
/// This is a *filter on the dialog*, not a selection. Wayland gives no way to
/// name a particular monitor, window, or rectangle: the compositor's own
/// picker decides, and this only narrows what it lists. See
/// [`PipeWireScreenCaptureSource`]'s docs for the full consequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureSourceKind {
    /// List whole monitors only.
    Monitor,
    /// List individual application windows only.
    Window,
    /// List both, and let the user pick either.
    MonitorOrWindow,
}

/// Construction-time options for [`PipeWireScreenCaptureSource::open`].
#[derive(Debug, Clone)]
pub struct PipeWireScreenCaptureOptions {
    /// The constant rate frames are emitted at. Like
    /// `DxgiCaptureOptions::fps`, this is a fixed *output* rate, not a cap on
    /// an otherwise irregular one — see [`PipeWireScreenCaptureSource`]'s docs on
    /// why the capture side's own rate is both variable and unrelated.
    /// `30` by default, matching every other source in this crate.
    pub fps: u32,
    /// What the portal's picker offers — see [`CaptureSourceKind`].
    /// [`CaptureSourceKind::Monitor`] by default.
    pub source_kind: CaptureSourceKind,
    /// Whether the compositor draws the mouse cursor into the captured
    /// pixels. Off by default, matching `DxgiCaptureOptions`' own default.
    ///
    /// Unlike the DXGI path, this is not CPU-side compositing this element
    /// performs: it selects the portal's `Embedded` cursor mode and the
    /// compositor does the drawing, so it costs nothing here and works for
    /// every source kind.
    pub include_cursor: bool,
    /// A token from a previous session's [`PipeWireScreenCaptureSource::open`]
    /// (see its return value). Supplying it reconnects to the same source
    /// **without showing the dialog again**; leaving it `None` always
    /// prompts.
    ///
    /// This is the only way to make repeat runs deterministic — there is no
    /// monitor index or rectangle to ask for. A token the compositor no
    /// longer recognises is not an error: the portal falls back to prompting,
    /// and `open` returns a fresh token for next time.
    pub restore_token: Option<String>,
}

impl Default for PipeWireScreenCaptureOptions {
    fn default() -> Self {
        Self {
            fps: 30,
            source_kind: CaptureSourceKind::Monitor,
            include_cursor: false,
            restore_token: None,
        }
    }
}

/// The latest captured image, written by the PipeWire thread and read by
/// [`SourceElement::run`] on the pipeline's source thread.
///
/// Holds tightly packed `width * 4` BGRA rows rather than an
/// `ffmpeg::frame::Video`: the PipeWire buffer's own stride is whatever the
/// compositor chose, so a copy has to happen regardless, and a plain `Vec`
/// keeps the locked region free of any ffmpeg allocation behaviour.
struct Latest {
    pixels: Vec<u8>,
    /// `false` until the first real (non-empty) frame lands, so `run` can
    /// tell "nothing captured yet" apart from "a genuinely black screen".
    have_frame: bool,
    /// Set when the PipeWire thread hits an unrecoverable stream error, so
    /// `run` can surface it instead of silently emitting stale frames
    /// forever.
    error: Option<String>,
}

/// What the PipeWire thread reports back to `open` once, at startup.
enum Startup {
    Ready { width: u32, height: u32 },
    Failed(PipeWireScreenCaptureSourceError),
}

/// Sent into the PipeWire thread's own main loop to end it.
struct Terminate;

/// Captures the Wayland desktop through xdg-desktop-portal's ScreenCast
/// portal and a PipeWire stream, emitting full-range BGRA CPU frames.
///
/// # What this captures
///
/// **Not a monitor this element chooses.** Wayland deliberately offers no API
/// to name a monitor, window, or rectangle; the compositor's own picker does.
/// [`PipeWireScreenCaptureOptions::source_kind`] only narrows what that picker
/// lists, and [`PipeWireScreenCaptureOptions::restore_token`] is the only way to
/// reuse an earlier choice without re-prompting. `open` is therefore
/// interactive and can block for as long as the user takes to answer — and
/// can fail with [`PipeWireScreenCaptureSourceError::Cancelled`], a failure mode
/// with no `DxgiCaptureSource` equivalent.
///
/// This is the substantive difference from `DxgiCaptureSource`, which is why
/// the two are separate types rather than one struct with a backend switch:
/// that element's `CaptureArea` is not expressible here at all.
///
/// # Frame rate
///
/// [`PipeWireScreenCaptureOptions::fps`] is a fixed *output* rate. The compositor
/// produces frames on damage — the negotiated PipeWire framerate is variable
/// (`0/1`) with only a maximum — so an idle desktop may deliver a handful of
/// frames per second while a busy one approaches the monitor's refresh rate.
/// `run` decouples the two by re-emitting the latest captured image on its own
/// schedule, exactly as `DxgiCaptureSource` does, so downstream sees a steady
/// rate either way. Raising `fps` above the chosen monitor's refresh rate only
/// repeats frames; it does not capture more.
///
/// # Frame format
///
/// `Pixel::BGRA`, tagged `color_space = RGB` / `color_range = JPEG` — the same
/// full-range RGB contract every capture source in this crate emits, so the
/// same downstream conversions apply. Only CPU-mapped PipeWire buffers
/// (`MemFd`/`MemPtr`) are negotiated: DMA-BUF is deliberately out of scope
/// while this crate has no Linux GPU element that could consume a
/// GPU-resident frame without an immediate round trip back to system memory.
pub struct PipeWireScreenCaptureSource {
    name: Arc<str>,
    pp_log: PpLog,
    pad: SrcPad,
    width: u32,
    height: u32,
    /// The configured output rate, kept because it is the unit `pts` counts in
    /// and so what [`PipeWireScreenCaptureSource::time_base`] must report —
    /// `frame_interval` below is derived from it for scheduling.
    fps: u32,
    frame_interval: Duration,
    /// Monotonic frame counter used as the emitted `pts`.
    frame_index: i64,
    /// Reused across every emitted frame — see [`UnboundObjectPool`]'s own
    /// docs on why frames are pooled rather than allocated per push.
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
    latest: Arc<Mutex<Latest>>,
    /// Ends the PipeWire thread's main loop. `Option` only so `Drop` can take
    /// it; always `Some` while this element is alive.
    terminate: Option<pw::channel::Sender<Terminate>>,
    /// Joined by `Drop` — this element owns the thread it spawned.
    worker: Option<JoinHandle<()>>,
}

impl PipeWireScreenCaptureSource {
    /// Runs the portal handshake, starts the PipeWire stream, and waits for
    /// format negotiation to finish.
    ///
    /// **Blocks on user interaction** unless
    /// [`PipeWireScreenCaptureOptions::restore_token`] is supplied and still valid:
    /// the compositor shows its screen-share dialog and this call does not
    /// return until the user answers.
    ///
    /// Returns the element, the negotiated capture size, and a restore token
    /// to persist for the next run (`None` if the compositor declined to
    /// issue one). The size comes from the stream's negotiated format rather
    /// than the portal's reported monitor size, because compositor scaling can
    /// make the two differ.
    pub fn open(
        name: impl Into<String>,
        options: PipeWireScreenCaptureOptions,
    ) -> std::result::Result<(Self, u32, u32, Option<String>), PipeWireScreenCaptureSourceError>
    {
        let name = name.into();
        let pp_log = element_pp_log(ElementType::PipeWireScreenCaptureSource, &name, None);

        let cast = portal_handshake(&options)?;
        let restore_token = cast.restore_token.clone();

        let latest = Arc::new(Mutex::new(Latest {
            pixels: Vec::new(),
            have_frame: false,
            error: None,
        }));
        let (startup_tx, startup_rx) = mpsc::channel::<Startup>();
        let (terminate_tx, terminate_rx) = pw::channel::channel::<Terminate>();

        let worker = {
            let latest = latest.clone();
            let fps = options.fps.max(1);
            std::thread::Builder::new()
                .name(format!("{name}-pipewire"))
                .spawn(move || {
                    if let Err(error) =
                        run_pipewire(cast, fps, latest.clone(), &startup_tx, terminate_rx)
                    {
                        // `open` may already have returned by the time a
                        // stream fails, so report through both paths: the
                        // startup channel (ignored if nobody is listening
                        // any more) and the shared slot `run` polls.
                        let message = error.to_string();
                        let _ = startup_tx.send(Startup::Failed(error));
                        if let Ok(mut latest) = latest.lock() {
                            latest.error.get_or_insert(message);
                        }
                    }
                })
                .map_err(|e| PipeWireScreenCaptureSourceError::PipeWire(e.to_string()))?
        };

        // From here on the thread is running, so every early return has to
        // tear it down rather than leaking it.
        let startup = match startup_rx.recv_timeout(NEGOTIATION_TIMEOUT) {
            Ok(startup) => startup,
            Err(RecvTimeoutError::Timeout) => {
                let _ = terminate_tx.send(Terminate);
                let _ = worker.join();
                return Err(PipeWireScreenCaptureSourceError::NegotiationTimeout);
            }
            Err(RecvTimeoutError::Disconnected) => {
                let _ = worker.join();
                return Err(PipeWireScreenCaptureSourceError::PipeWire(
                    "the PipeWire thread exited before negotiating a format".into(),
                ));
            }
        };
        let (width, height) = match startup {
            Startup::Ready { width, height } => (width, height),
            Startup::Failed(error) => {
                let _ = terminate_tx.send(Terminate);
                let _ = worker.join();
                return Err(error);
            }
        };

        let fps = options.fps.max(1); // a `0` fps is nonsensical; treat it as 1 rather than dividing by zero
        pp_info!(
            pp_log: &pp_log,
            "opened: {}x{} via xdg-desktop-portal, source_kind={:?}, include_cursor={}, fps={}, restored={}",
            width,
            height,
            options.source_kind,
            options.include_cursor,
            fps,
            options.restore_token.is_some()
        );

        Ok((
            Self {
                pad: SrcPad::new(format!("{name}_src")),
                name: name.into(),
                pp_log,
                width,
                height,
                fps,
                frame_interval: Duration::from_secs_f64(1.0 / fps as f64),
                frame_index: 0,
                pool: UnboundObjectPool::new(
                    0,
                    move || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, width, height),
                    |_| {},
                ),
                latest,
                terminate: Some(terminate_tx),
                worker: Some(worker),
            },
            width,
            height,
            restore_token,
        ))
    }

    /// The unit each emitted frame's `pts` is expressed in.
    ///
    /// Frames are stamped with a plain frame counter, so this is `1/fps` — the
    /// configured output rate, not the compositor's irregular capture rate.
    /// Same contract as `DxgiCaptureSource::time_base`.
    pub fn time_base(&self) -> ffmpeg::Rational {
        ffmpeg::Rational::new(1, self.fps as i32)
    }

    /// Copies the latest captured image into a fresh pooled frame.
    ///
    /// Copying rather than sharing is what lets each emitted frame carry its
    /// own correctly-incrementing `pts` even when several emissions in a row
    /// show identical content — an `Arc`-shared frame can't have its `pts`
    /// rewritten in place once downstream might hold a clone. Same reasoning
    /// `DxgiCaptureSource::emit_frame_cpu` documents.
    ///
    /// Returns `None` when nothing has been captured yet, so `run` can skip
    /// the tick instead of pushing an uninitialised frame.
    fn emit_frame(&mut self) -> Option<crate::pool::UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        let mut frame = self.pool.get();
        let row_bytes = self.width as usize * 4;
        {
            let latest = self.latest.lock().ok()?;
            if !latest.have_frame || latest.pixels.len() < row_bytes * self.height as usize {
                return None;
            }
            let dst_stride = frame.stride(0);
            let dst = frame.data_mut(0);
            for row in 0..self.height as usize {
                dst[row * dst_stride..row * dst_stride + row_bytes]
                    .copy_from_slice(&latest.pixels[row * row_bytes..(row + 1) * row_bytes]);
            }
        }
        frame.set_pts(Some(self.frame_index));
        frame.set_color_space(ffmpeg::color::Space::RGB);
        frame.set_color_range(ffmpeg::color::Range::JPEG);
        self.frame_index += 1;
        Some(frame)
    }

    /// The PipeWire thread's error, if it has hit one. Taken rather than
    /// peeked so `run` reports it exactly once.
    fn take_worker_error(&self) -> Option<String> {
        self.latest.lock().ok()?.error.take()
    }
}

impl Drop for PipeWireScreenCaptureSource {
    fn drop(&mut self) {
        // Dropping the sender alone would not wake a blocked main loop, so
        // signal first, then join the thread this element owns.
        if let Some(terminate) = self.terminate.take() {
            let _ = terminate.send(Terminate);
        }
        if let Some(worker) = self.worker.take()
            && worker.join().is_err()
        {
            pp_warn!(self, "the PipeWire capture thread panicked");
        }
    }
}

impl Element for PipeWireScreenCaptureSource {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::PipeWireScreenCaptureSource
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for PipeWireScreenCaptureSource {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl SourceElement for PipeWireScreenCaptureSource {
    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        pp_info!(self, "started");
        let mut schedule = PeriodicSchedule::new(self.frame_interval, Instant::now());
        loop {
            let outcome = drain_control(control, self, bus)?;
            if outcome.stopped {
                pp_info!(self, "stopped");
                return Ok(());
            }
            if outcome.paused_for > Duration::ZERO {
                schedule.resume_after_pause(outcome.paused_for, Instant::now());
            }

            if let Some(error) = self.take_worker_error() {
                pp_error!(self, "capture failed: {error}");
                return Err(PipeWireScreenCaptureSourceError::PipeWire(error).into());
            }

            // Nothing to poll: the PipeWire thread fills `latest` on its own.
            // Sleeping in `POLL_GRANULARITY` slices keeps `Stop` responsive at
            // low `fps` without busy-waiting.
            let remaining = schedule.remaining(Instant::now());
            if !remaining.is_zero() {
                std::thread::sleep(remaining.min(POLL_GRANULARITY));
                continue;
            }

            let Some(frame) = self.emit_frame() else {
                // Still advance even though there's nothing to emit this
                // tick — otherwise the next iteration's `remaining` is zero
                // and this busy-loops instead of waiting for the next tick.
                schedule.advance_after_tick(Instant::now());
                continue;
            };
            if let Err(error) = self.pad.push(MediaBuffer::Video(Arc::new(frame))) {
                bus.post(
                    &self.pp_log,
                    BusEvent::Error {
                        element_type: ElementType::PipeWireScreenCaptureSource,
                        name: self.name.clone(),
                        error,
                    },
                );
            }
            // Advance only now that this tick's own work (copy + push, which a
            // slow downstream can stretch arbitrarily) is done — see
            // `TestVideoSource::run`'s identical correction.
            schedule.advance_after_tick(Instant::now());
        }
    }

    fn seek(&mut self, _target: Duration) -> Result<Duration> {
        Err(PipeWireScreenCaptureSourceError::SeekUnsupported.into())
    }
}

/// What the portal handshake resolved to.
struct PortalCast {
    fd: std::os::fd::OwnedFd,
    node_id: u32,
    restore_token: Option<String>,
}

/// Drives the whole ScreenCast portal handshake to completion synchronously.
///
/// `ashpd` is async-only, so this blocks on its futures with `pollster`
/// against zbus's own `async-io` backend. That combination is deliberate:
/// nothing here requires the caller to have installed a runtime, which keeps
/// `open` a plain blocking constructor like every other element's.
fn portal_handshake(
    options: &PipeWireScreenCaptureOptions,
) -> std::result::Result<PortalCast, PipeWireScreenCaptureSourceError> {
    use ashpd::desktop::{
        PersistMode,
        screencast::{CursorMode, Screencast, SelectSourcesOptions, SourceType, StartCastOptions},
    };

    fn portal_err(error: ashpd::Error) -> PipeWireScreenCaptureSourceError {
        match error {
            ashpd::Error::Response(ashpd::desktop::ResponseError::Cancelled) => {
                PipeWireScreenCaptureSourceError::Cancelled
            }
            other => PipeWireScreenCaptureSourceError::Portal(other.to_string()),
        }
    }

    pollster::block_on(async {
        let proxy = Screencast::new().await.map_err(portal_err)?;
        let session = proxy
            .create_session(Default::default())
            .await
            .map_err(portal_err)?;

        let sources = match options.source_kind {
            CaptureSourceKind::Monitor => BitFlags::from(SourceType::Monitor),
            CaptureSourceKind::Window => BitFlags::from(SourceType::Window),
            CaptureSourceKind::MonitorOrWindow => SourceType::Monitor | SourceType::Window,
        };
        let mut select = SelectSourcesOptions::default()
            .set_sources(sources)
            .set_multiple(false)
            .set_cursor_mode(if options.include_cursor {
                CursorMode::Embedded
            } else {
                CursorMode::Hidden
            })
            // Ask the compositor to remember the choice so a caller that
            // persists the returned token can skip the dialog next time.
            .set_persist_mode(PersistMode::ExplicitlyRevoked);
        if let Some(token) = &options.restore_token {
            select = select.set_restore_token(token.as_str());
        }
        proxy
            .select_sources(&session, select)
            .await
            .map_err(portal_err)?;

        let response = proxy
            .start(&session, None, StartCastOptions::default())
            .await
            .map_err(portal_err)?
            .response()
            .map_err(portal_err)?;

        let stream = response
            .streams()
            .first()
            .ok_or(PipeWireScreenCaptureSourceError::NoStream)?;
        let node_id = stream.pipe_wire_node_id();
        let restore_token = response.restore_token().map(str::to_owned);

        let fd = proxy
            .open_pipe_wire_remote(&session, Default::default())
            .await
            .map_err(portal_err)?;

        // The cast lives exactly as long as the portal session object. Closing
        // it here would tear the stream down before it produced a frame, and
        // the session has no lifetime tie to the returned fd, so it is kept
        // alive deliberately for the process's remaining lifetime.
        std::mem::forget(session);

        Ok(PortalCast {
            fd,
            node_id,
            restore_token,
        })
    })
}

/// Copies `height` rows of `row_bytes` out of a `src_stride`-strided source
/// into a tightly packed destination.
///
/// Repacking here rather than downstream keeps `emit_frame` dealing with only
/// one stride (its own destination frame's): the compositor is free to pick
/// any padded stride it likes, and on this path it is the only place a
/// mismatch between the mapped buffer and the negotiated size could turn into
/// an out-of-bounds read.
///
/// Returns `false` without writing anything when the source cannot supply that
/// many full rows, or the destination cannot hold them — a short or malformed
/// PipeWire mapping is dropped, never partially consumed.
fn repack_rows(
    dst: &mut [u8],
    src: &[u8],
    src_stride: usize,
    row_bytes: usize,
    height: usize,
) -> bool {
    if src_stride < row_bytes
        || src.len() < src_stride.saturating_mul(height.saturating_sub(1)) + row_bytes
        || dst.len() < row_bytes.saturating_mul(height)
    {
        return false;
    }
    for row in 0..height {
        dst[row * row_bytes..(row + 1) * row_bytes]
            .copy_from_slice(&src[row * src_stride..row * src_stride + row_bytes]);
    }
    true
}

/// The PipeWire thread body: owns the main loop, context, core, and stream,
/// none of which are `Send`, and so all of which must be created and dropped
/// here rather than handed across from `open`.
fn run_pipewire(
    cast: PortalCast,
    fps: u32,
    latest: Arc<Mutex<Latest>>,
    startup: &mpsc::Sender<Startup>,
    terminate: pw::channel::Receiver<Terminate>,
) -> std::result::Result<(), PipeWireScreenCaptureSourceError> {
    fn pw_err(error: impl std::fmt::Display) -> PipeWireScreenCaptureSourceError {
        PipeWireScreenCaptureSourceError::PipeWire(error.to_string())
    }

    pw::init();
    let mainloop = pw::main_loop::MainLoopRc::new(None).map_err(pw_err)?;
    let context = pw::context::ContextRc::new(&mainloop, None).map_err(pw_err)?;
    let core = context.connect_fd_rc(cast.fd, None).map_err(pw_err)?;

    // Attach to the outer loop, but let the callback own its own clone: the
    // `AttachedReceiver` borrows the `Loop` for as long as it lives, so the
    // handle it is attached to has to outlive it.
    let quit_loop = mainloop.clone();
    let _terminate = terminate.attach(mainloop.loop_(), move |_| quit_loop.quit());

    let stream = pw::stream::StreamBox::new(
        &core,
        "media-pp-screen-capture",
        properties! {
            *pw::keys::MEDIA_TYPE => "Video",
            *pw::keys::MEDIA_CATEGORY => "Capture",
            *pw::keys::MEDIA_ROLE => "Screen",
        },
    )
    .map_err(pw_err)?;

    // Negotiated size, shared between the two callbacks below. Both run on
    // this thread's own loop, so a `Cell` is enough — no lock needed.
    let size = Arc::new(Mutex::new((0u32, 0u32)));

    let _listener = stream
        .add_local_listener_with_user_data(())
        .param_changed({
            let size = size.clone();
            let latest = latest.clone();
            let startup = startup.clone();
            move |_, (), id, param| {
                let Some(param) = param else { return };
                if id != spa::param::ParamType::Format.as_raw() {
                    return;
                }
                let Ok((media_type, media_subtype)) = spa::param::format_utils::parse_format(param)
                else {
                    return;
                };
                if media_type != spa::param::format::MediaType::Video
                    || media_subtype != spa::param::format::MediaSubtype::Raw
                {
                    return;
                }
                let mut info = spa::param::video::VideoInfoRaw::new();
                if info.parse(param).is_err() {
                    return;
                }
                let format = info.format();
                if format != spa::param::video::VideoFormat::BGRx
                    && format != spa::param::video::VideoFormat::BGRA
                {
                    let _ = startup.send(Startup::Failed(
                        PipeWireScreenCaptureSourceError::UnsupportedFormat(format.as_raw()),
                    ));
                    return;
                }
                let (width, height) = (info.size().width, info.size().height);
                if width == 0 || height == 0 {
                    return;
                }
                if let Ok(mut size) = size.lock() {
                    *size = (width, height);
                }
                if let Ok(mut latest) = latest.lock() {
                    // A renegotiation to a different size (the user resized a
                    // captured window) invalidates whatever was buffered.
                    latest.pixels = vec![0; width as usize * height as usize * 4];
                    latest.have_frame = false;
                }
                let _ = startup.send(Startup::Ready { width, height });
            }
        })
        .process({
            let size = size.clone();
            move |stream, ()| {
                let Some(mut buffer) = stream.dequeue_buffer() else {
                    return;
                };
                let datas = buffer.datas_mut();
                if datas.is_empty() {
                    return;
                }
                let data = &mut datas[0];
                if data.chunk().size() == 0 {
                    return; // a tick with no new content
                }
                let stride = data.chunk().stride();
                let Ok((width, height)) = size.lock().map(|s| *s) else {
                    return;
                };
                if width == 0 || height == 0 {
                    return;
                }
                let row_bytes = width as usize * 4;
                let src_stride = if stride > 0 {
                    stride as usize
                } else {
                    row_bytes
                };
                let Some(pixels) = data.data() else { return };
                let Ok(mut latest) = latest.lock() else {
                    return;
                };
                if latest.pixels.len() < row_bytes * height as usize {
                    latest.pixels = vec![0; row_bytes * height as usize];
                }
                if repack_rows(
                    &mut latest.pixels,
                    pixels,
                    src_stride,
                    row_bytes,
                    height as usize,
                ) {
                    latest.have_frame = true;
                }
            }
        })
        .register()
        .map_err(pw_err)?;

    let obj = spa::pod::object!(
        spa::utils::SpaTypes::ObjectParamFormat,
        spa::param::ParamType::EnumFormat,
        spa::pod::property!(
            spa::param::format::FormatProperties::MediaType,
            Id,
            spa::param::format::MediaType::Video
        ),
        spa::pod::property!(
            spa::param::format::FormatProperties::MediaSubtype,
            Id,
            spa::param::format::MediaSubtype::Raw
        ),
        // CPU-mapped BGRx/BGRA only — see this element's docs on why DMA-BUF
        // is deliberately not offered.
        spa::pod::property!(
            spa::param::format::FormatProperties::VideoFormat,
            Choice,
            Enum,
            Id,
            spa::param::video::VideoFormat::BGRx,
            spa::param::video::VideoFormat::BGRx,
            spa::param::video::VideoFormat::BGRA,
        ),
        spa::pod::property!(
            spa::param::format::FormatProperties::VideoSize,
            Choice,
            Range,
            Rectangle,
            spa::utils::Rectangle {
                width: 1920,
                height: 1080
            },
            spa::utils::Rectangle {
                width: 1,
                height: 1
            },
            spa::utils::Rectangle {
                width: 8192,
                height: 8192
            }
        ),
        // The compositor answers with a variable rate (`0/1`) capped at this
        // maximum; the emitted rate is this element's own concern, not this.
        spa::pod::property!(
            spa::param::format::FormatProperties::VideoFramerate,
            Choice,
            Range,
            Fraction,
            spa::utils::Fraction { num: fps, denom: 1 },
            spa::utils::Fraction { num: 0, denom: 1 },
            spa::utils::Fraction {
                num: fps.max(60),
                denom: 1
            }
        ),
    );
    let values: Vec<u8> = spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &spa::pod::Value::Object(obj),
    )
    .map_err(pw_err)?
    .0
    .into_inner();
    let mut params = [Pod::from_bytes(&values).ok_or_else(|| {
        PipeWireScreenCaptureSourceError::PipeWire("failed to build format pod".into())
    })?];

    stream
        .connect(
            spa::utils::Direction::Input,
            Some(cast.node_id),
            pw::stream::StreamFlags::AUTOCONNECT | pw::stream::StreamFlags::MAP_BUFFERS,
            &mut params,
        )
        .map_err(pw_err)?;

    mainloop.run();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a padded source: every row is `row_bytes` of `fill` followed by
    /// `pad` bytes of `0xAA`, so a repack that reads the padding is visible in
    /// the output rather than silently plausible.
    fn padded_src(row_bytes: usize, pad: usize, height: usize) -> Vec<u8> {
        let mut src = Vec::with_capacity((row_bytes + pad) * height);
        for row in 0..height {
            src.extend(std::iter::repeat_n(row as u8, row_bytes));
            src.extend(std::iter::repeat_n(0xAA, pad));
        }
        src
    }

    #[test]
    fn repack_drops_padding_between_rows() {
        let (row_bytes, pad, height) = (8, 4, 3);
        let src = padded_src(row_bytes, pad, height);
        let mut dst = vec![0u8; row_bytes * height];

        assert!(repack_rows(
            &mut dst,
            &src,
            row_bytes + pad,
            row_bytes,
            height
        ));
        for row in 0..height {
            assert_eq!(
                &dst[row * row_bytes..(row + 1) * row_bytes],
                &vec![row as u8; row_bytes][..],
                "row {row} should hold only its own pixels, never the padding"
            );
        }
    }

    #[test]
    fn repack_accepts_a_source_whose_last_row_omits_its_padding() {
        // A compositor may map exactly `stride * (h - 1) + row_bytes`: the
        // final row's padding is outside the mapping and must not be required.
        let (row_bytes, pad, height) = (8, 4, 3);
        let mut src = padded_src(row_bytes, pad, height);
        src.truncate((row_bytes + pad) * (height - 1) + row_bytes);
        let mut dst = vec![0u8; row_bytes * height];

        assert!(repack_rows(
            &mut dst,
            &src,
            row_bytes + pad,
            row_bytes,
            height
        ));
        assert_eq!(&dst[2 * row_bytes..], &vec![2u8; row_bytes][..]);
    }

    #[test]
    fn repack_rejects_a_short_source_without_writing() {
        let (row_bytes, height) = (8, 3);
        let src = vec![0x11; row_bytes * height - 1];
        let mut dst = vec![0u8; row_bytes * height];

        assert!(!repack_rows(&mut dst, &src, row_bytes, row_bytes, height));
        assert!(
            dst.iter().all(|&b| b == 0),
            "a rejected buffer must leave the destination untouched, \
             so `have_frame` never turns on for a partial copy"
        );
    }

    #[test]
    fn repack_rejects_a_stride_narrower_than_one_row() {
        let (row_bytes, height) = (8, 2);
        let src = vec![0x22; row_bytes * height];
        let mut dst = vec![0u8; row_bytes * height];

        assert!(!repack_rows(
            &mut dst,
            &src,
            row_bytes - 1,
            row_bytes,
            height
        ));
        assert!(dst.iter().all(|&b| b == 0));
    }

    #[test]
    fn repack_rejects_a_destination_too_small_for_the_frame() {
        let (row_bytes, height) = (8, 3);
        let src = vec![0x33; row_bytes * height];
        // Mirrors a renegotiation to a larger size racing ahead of the
        // reallocation of `Latest::pixels`.
        let mut dst = vec![0u8; row_bytes * (height - 1)];

        assert!(!repack_rows(&mut dst, &src, row_bytes, row_bytes, height));
        assert!(dst.iter().all(|&b| b == 0));
    }

    #[test]
    fn default_options_match_the_documented_defaults() {
        let options = PipeWireScreenCaptureOptions::default();
        assert_eq!(options.fps, 30);
        assert_eq!(options.source_kind, CaptureSourceKind::Monitor);
        assert!(!options.include_cursor);
        assert!(options.restore_token.is_none());
    }

    #[test]
    fn seek_is_rejected_as_a_typed_error() {
        // `open` needs a portal dialog, so the rejection is asserted against
        // the error itself rather than through a constructed element.
        let error: crate::Error = PipeWireScreenCaptureSourceError::SeekUnsupported.into();
        assert!(
            matches!(
                error,
                crate::Error::PipeWireScreenCaptureSourceError(
                    PipeWireScreenCaptureSourceError::SeekUnsupported
                )
            ),
            "seek failures must reach callers as a typed variant, not a stringly error"
        );
    }
}
