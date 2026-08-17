/// # Closing a captured window ends the capture
///
/// Not via the stream, which reports nothing useful: a closed window, a
/// fullscreen-starved monitor, and a desktop where nothing happens to be moving
/// are identical from there — zero frames, zero buffers, zero callbacks. The
/// stream even stays `Streaming` throughout.
///
/// The portal knows, though, and says so: closing a captured window makes it
/// emit the session's `Closed` signal, measured at about six seconds after the
/// fact on GNOME 50. This element watches for it and ends `run` with
/// [`PipeWireScreenCaptureSourceError::SourceGone`]. A capture that merely
/// stopped receiving frames is left alone, since it may still recover.
/// Reopening is
/// the caller's decision, the same contract `DxgiCaptureSourceError::AccessLost`
/// sets on Windows.
///
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
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
use spa::sys as spa_sys;
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

/// How long the portal-session watcher waits before re-checking whether the
/// element is still alive. Only bounds shutdown latency; the signal itself
/// arrives whenever it arrives.
const SESSION_POLL_GRANULARITY: Duration = Duration::from_millis(250);

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
    /// `NEGOTIATION_TIMEOUT`, so this element never learned what size or
    /// pixel layout to expect.
    #[error("timed out waiting for the PipeWire stream to negotiate a format")]
    NegotiationTimeout,

    /// The compositor offered only formats this element does not accept. v1
    /// negotiates CPU-mapped `BGRx`/`BGRA` only — see
    /// [`PipeWireScreenCaptureSource`]'s own docs on why DMA-BUF is out of scope.
    #[error("compositor negotiated unsupported video format {0:?}")]
    UnsupportedFormat(u32),

    /// What was being captured no longer exists: the portal closed the session
    /// (a captured window was closed), or the stream itself errored or
    /// disconnected after having run.
    ///
    /// Broken out of [`Self::PipeWire`] because it is terminal — unlike a
    /// stall, this will not recover — following the "fail fast, the caller
    /// decides whether to reopen" contract `DxgiCaptureSourceError::AccessLost`
    /// sets.
    #[error("the captured source is gone: {0}")]
    SourceGone(String),

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
    /// The size `pixels` is currently in.
    ///
    /// Not necessarily the size [`PipeWireScreenCaptureSource::open`] reported:
    /// the compositor renegotiates the format whenever a captured *window* is
    /// resized, while everything downstream stays built for the original size.
    /// `emit_frame` reconciles the two rather than trusting either alone.
    /// The size currently being emitted, and the size `pool`'s frames are
    /// allocated to. Follows the compositor's renegotiation — a captured
    /// *window* changes size when the user resizes it — rather than staying at
    /// whatever `open` first saw.
    width: u32,
    height: u32,
    /// Set when a buffer arrived that this element could not read — currently
    /// only a GPU-resident (DMA-BUF) buffer, which has no CPU mapping. Reported
    /// once by `run` rather than silently dropped, because dropping every frame
    /// looks exactly like a frozen desktop and gives a caller nothing to act on.
    unmappable_buffers: u64,
    /// `false` until the first real (non-empty) frame lands, so `run` can
    /// tell "nothing captured yet" apart from "a genuinely black screen".
    have_frame: bool,
    /// Set when the PipeWire thread hits an unrecoverable stream error, so
    /// `run` can surface it instead of silently emitting stale frames
    /// forever.
    error: Option<PipeWireScreenCaptureSourceError>,
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
/// # A fullscreen client can stall a monitor capture — capture the window instead
///
/// While a client is fullscreen and the compositor can scan its buffer out
/// directly, GNOME/Mutter may stop composing that monitor and feed a
/// [`CaptureSourceKind::Monitor`] stream nothing at all — not empty buffers, no
/// buffers — for as long as that lasts, then resume the moment fullscreen ends.
/// `run` keeps emitting at the configured rate throughout, so downstream sees
/// repeats of the last captured image rather than any error.
///
/// Nothing this element negotiates changes that: GNOME's own built-in recorder
/// collapses to well under 1fps under exactly the same conditions and recovers
/// at exactly the same moment.
///
/// [`CaptureSourceKind::Window`] does not have the problem. A window stream is
/// fed from that window's own surface rather than from the monitor's composited
/// output, so it keeps delivering at full rate while the window is fullscreen —
/// measured at 25-35fps against 0fps for a monitor stream under identical
/// conditions. Prefer it whenever the caller knows it wants one application.
///
/// # A window stream is monitor-sized, whatever the window's own size
///
/// GNOME sizes a [`CaptureSourceKind::Window`] stream to the monitor rather
/// than to the window, and keeps it there. Measured against GNOME 50: a window
/// smaller than the monitor arrives top-left aligned with the rest of the frame
/// black; a maximized or fullscreen one fills it exactly; and a window dragged
/// across two monitors, wider than either, is **clipped** at the stream's width
/// rather than widening it. So a window capture can never show more than one
/// monitor's worth of pixels, and capturing a small window still encodes a
/// full-size frame that is mostly black.
///
/// # Closing a captured window cannot be detected
///
/// It looks exactly like the fullscreen stall above. Measured on GNOME 50,
/// closing a captured window leaves the PipeWire stream in `Streaming` — no
/// error, no disconnect — and simply stops the process callbacks. A monitor
/// stream starved by a fullscreen client produces the identical signature:
/// zero frames, zero empty buffers, zero callbacks. The only difference is that
/// one recovers and the other never will, which is not knowable at the time.
///
/// So this element reports a stall for both and keeps running, rather than
/// guessing. A caller that must distinguish them has to look outside the
/// stream — at whether the window it asked for still exists. Ending the capture
/// on a long stall would kill recordings that were about to resume.
///
/// [`PipeWireScreenCaptureSourceError::SourceGone`] therefore only covers a
/// stream that genuinely reports an error or disconnects, which a closed window
/// on this compositor does not.
///
/// # A captured window can be resized — put a `Scaler` downstream
///
/// The portal does not forbid a compositor from renegotiating mid-stream, and
/// when one does **this element follows**, emitting frames at the new size. The
/// size [`PipeWireScreenCaptureSource::open`] reports is therefore the *initial*
/// size, not a promise for the whole stream.
///
/// GNOME never actually does this — see the section above; resizing, maximizing,
/// and spanning monitors all left the stream at its original size. Treat this as
/// insurance for other compositors rather than as behaviour to expect here.
///
/// Chain a [`crate::elements::Scaler`] before anything built for a fixed
/// geometry. It rebuilds its scaling context whenever its input dimensions
/// change and always emits its own configured size, so an encoder and muxer
/// behind it keep working across a resize. Without one, an encoder rejects the
/// first differently-sized frame outright — a loud failure, deliberately, in
/// preference to this element silently cropping away whatever moved outside the
/// original rectangle.
///
/// Scaling here instead would be the same hidden conversion every other source
/// in this crate refuses to do.
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
    /// Cleared on `Drop` to end `session_watcher`, which otherwise blocks on a
    /// signal that may never come.
    watching: Arc<AtomicBool>,
    /// Watches the portal session for the one signal that distinguishes a
    /// vanished source from an idle screen — see `watch_session_closed`.
    session_watcher: Option<JoinHandle<()>>,
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

        let mut cast = portal_handshake(&options)?;
        let restore_token = cast.restore_token.clone();
        // Taken out before `cast` moves into the PipeWire thread: the session
        // is watched here, the stream is opened there.
        let session = cast.session.take();

        let latest = Arc::new(Mutex::new(Latest {
            pixels: Vec::new(),
            width: 0,
            height: 0,
            unmappable_buffers: 0,
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
                        let reported =
                            PipeWireScreenCaptureSourceError::PipeWire(error.to_string());
                        let _ = startup_tx.send(Startup::Failed(error));
                        if let Ok(mut latest) = latest.lock() {
                            latest.error.get_or_insert(reported);
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

        let watching = Arc::new(AtomicBool::new(true));
        let session_watcher = watch_session_closed(session, latest.clone(), watching.clone());

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
                watching,
                session_watcher,
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
        let (src_width, src_height, ready) = {
            let latest = self.latest.lock().ok()?;
            (latest.width, latest.height, latest.have_frame)
        };
        if !ready || src_width == 0 || src_height == 0 {
            return None;
        }
        // Follow the compositor rather than clamping to the opening size: a
        // resized window would otherwise be silently cropped, losing whatever
        // moved outside the original rectangle. Downstream absorbs the change
        // through `Scaler`, which rebuilds its own context per input size — see
        // this element's docs on why that split, not scaling here, is the one
        // consistent with the rest of the crate.
        if (src_width, src_height) != (self.width, self.height) {
            pp_info!(
                self,
                "capture resized: {}x{} -> {}x{}",
                self.width,
                self.height,
                src_width,
                src_height
            );
            self.width = src_width;
            self.height = src_height;
            // Pooled frames are allocated to a fixed size, so the old pool
            // cannot serve the new one. Outstanding frames stay valid: each
            // `UnboundObjectPoolRef` keeps its own pool's share alive.
            self.pool = UnboundObjectPool::new(
                0,
                move || {
                    ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, src_width, src_height)
                },
                |_| {},
            );
        }

        let row_bytes = src_width as usize * 4;
        let mut frame = self.pool.get();
        {
            let latest = self.latest.lock().ok()?;
            // Re-checked under the lock: the PipeWire thread may have
            // renegotiated again between the read above and here.
            if latest.width != src_width || latest.height != src_height {
                return None;
            }
            let dst_stride = frame.stride(0);
            let dst = frame.data_mut(0);
            if !copy_rows_into(
                dst,
                dst_stride,
                &latest.pixels,
                row_bytes,
                src_height as usize,
            ) {
                return None;
            }
        }
        frame.set_pts(Some(self.frame_index));
        frame.set_color_space(ffmpeg::color::Space::RGB);
        frame.set_color_range(ffmpeg::color::Range::JPEG);
        self.frame_index += 1;
        Some(frame)
    }

    /// How many unreadable buffers have arrived since the last check, if any.
    /// Taken rather than peeked so the warning is not repeated every tick.
    fn take_unmappable_report(&self) -> Option<u64> {
        let mut latest = self.latest.lock().ok()?;
        (latest.unmappable_buffers > 0).then(|| std::mem::take(&mut latest.unmappable_buffers))
    }

    /// The PipeWire thread's error, if it has hit one. Taken rather than
    /// peeked so `run` reports it exactly once.
    fn take_worker_error(&self) -> Option<PipeWireScreenCaptureSourceError> {
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
        self.watching.store(false, Ordering::Release);
        if let Some(watcher) = self.session_watcher.take()
            && watcher.join().is_err()
        {
            pp_warn!(self, "the portal session watcher panicked");
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

            if let Some(count) = self.take_unmappable_report() {
                pp_warn!(
                    self,
                    "dropped {count} GPU-resident (DMA-BUF) buffer(s): this element only \
                     reads CPU-mapped buffers, so the captured image is frozen at the last \
                     readable frame"
                );
            }
            if let Some(error) = self.take_worker_error() {
                pp_error!(self, "capture failed: {error}");
                return Err(error.into());
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
    /// Kept rather than dropped: the cast lives exactly as long as this, and
    /// its `Closed` signal is the one thing that tells a vanished source apart
    /// from a merely idle one (see `watch_session_closed`).
    session: Option<ashpd::desktop::Session<ashpd::desktop::screencast::Screencast>>,
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

        Ok(PortalCast {
            session: Some(session),
            fd,
            node_id,
            restore_token,
        })
    })
}

/// Whether a stream state transition means the thing being captured is gone,
/// and what to say about it.
///
/// `Unconnected` is also where every stream starts, so it only counts as a loss
/// once the stream really was running — otherwise opening one would report
/// itself as already broken.
fn disconnect_reason(
    old: &pw::stream::StreamState,
    new: &pw::stream::StreamState,
) -> Option<String> {
    use pw::stream::StreamState::{Error, Paused, Streaming, Unconnected};
    match new {
        Error(message) => Some(message.clone()),
        Unconnected if matches!(old, Paused | Streaming) => {
            Some("the stream was disconnected".to_owned())
        }
        _ => None,
    }
}

/// Copies `rows` tightly packed rows of `row_bytes` into a destination whose
/// own rows are `dst_stride` apart.
///
/// FFmpeg allocates each frame's line size with its own padding, so the
/// destination stride is generally wider than the pixels; only the real
/// `row_bytes` are written. Returns `false` without writing anything when
/// either side cannot supply or hold that many rows.
fn copy_rows_into(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    row_bytes: usize,
    rows: usize,
) -> bool {
    if row_bytes == 0
        || dst_stride < row_bytes
        || src.len() < row_bytes.saturating_mul(rows)
        || dst.len() < dst_stride.saturating_mul(rows.saturating_sub(1)) + row_bytes
    {
        return false;
    }
    for row in 0..rows {
        dst[row * dst_stride..row * dst_stride + row_bytes]
            .copy_from_slice(&src[row * row_bytes..(row + 1) * row_bytes]);
    }
    true
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

/// Watches the portal session for its `Closed` signal, which the compositor
/// emits when what was being captured no longer exists — a captured window
/// being closed, measured at ~6s after the fact on GNOME 50.
///
/// This is the *only* way to tell that case apart from a stall. The PipeWire
/// stream reports nothing: a closed window, a fullscreen-starved monitor, and a
/// desktop where simply nothing is moving all present identically — zero
/// frames, zero buffers, zero callbacks. One of them never recovers, and only
/// the portal knows which.
///
/// Returns `None` when there is no session to watch, which leaves the element
/// reporting stalls exactly as before rather than failing to open.
fn watch_session_closed(
    session: Option<ashpd::desktop::Session<ashpd::desktop::screencast::Screencast>>,
    latest: Arc<Mutex<Latest>>,
    watching: Arc<AtomicBool>,
) -> Option<JoinHandle<()>> {
    use futures_util::StreamExt;

    let session = session?;
    std::thread::Builder::new()
        .name("portal-session-watch".into())
        .spawn(move || {
            pollster::block_on(async move {
                let Ok(mut closed) = session.receive_closed().await else {
                    return;
                };
                // Polled in slices rather than awaited outright: `Drop` has to
                // be able to end this thread, and the signal may never come.
                while watching.load(Ordering::Acquire) {
                    let next = std::pin::pin!(closed.next());
                    let slice = std::pin::pin!(async_io::Timer::after(SESSION_POLL_GRANULARITY));
                    match futures_util::future::select(next, slice).await {
                        futures_util::future::Either::Left((signal, _)) => {
                            if let Ok(mut latest) = latest.lock() {
                                latest.error.get_or_insert(
                                    PipeWireScreenCaptureSourceError::SourceGone(
                                        if signal.is_some() {
                                            "the portal closed the session".to_owned()
                                        } else {
                                            "the portal session ended".to_owned()
                                        },
                                    ),
                                );
                            }
                            return;
                        }
                        futures_util::future::Either::Right(_) => {}
                    }
                }
            })
        })
        .ok()
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
        .state_changed({
            let latest = latest.clone();
            move |_, (), old, new| {
                // A closed window takes its node with it, and the callbacks
                // simply stop. Without watching the state that is
                // indistinguishable from a compositor going quiet, and the
                // capture would run forever repeating its last frame.
                if let Some(message) = disconnect_reason(&old, &new)
                    && let Ok(mut latest) = latest.lock()
                {
                    latest
                        .error
                        .get_or_insert(PipeWireScreenCaptureSourceError::SourceGone(message));
                }
            }
        })
        .param_changed({
            let size = size.clone();
            let latest = latest.clone();
            let startup = startup.clone();
            move |stream, (), id, param| {
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
                    latest.width = width;
                    latest.height = height;
                    latest.have_frame = false;
                }

                // Constrain the buffer negotiation to memory this element can
                // actually read. Without this the compositor is free to hand
                // over DMA-BUF once the captured content becomes GPU-resident
                // (a fullscreen video, say); those have no CPU mapping, every
                // frame is dropped, and the capture silently freezes at the
                // last readable image. See this element's docs on why DMA-BUF
                // support is out of scope rather than merely unimplemented.
                let data_type = (1 << spa::buffer::DataType::MemFd.as_raw())
                    | (1 << spa::buffer::DataType::MemPtr.as_raw());
                let buffers = spa::pod::object!(
                    spa::utils::SpaTypes::ObjectParamBuffers,
                    spa::param::ParamType::Buffers,
                    spa::pod::Property::new(
                        spa_sys::SPA_PARAM_BUFFERS_dataType,
                        spa::pod::Value::Choice(spa::pod::ChoiceValue::Int(spa::utils::Choice(
                            spa::utils::ChoiceFlags::empty(),
                            spa::utils::ChoiceEnum::Flags {
                                default: data_type,
                                flags: vec![data_type],
                            },
                        ),)),
                    ),
                );
                if let Ok(bytes) = spa::pod::serialize::PodSerializer::serialize(
                    std::io::Cursor::new(Vec::new()),
                    &spa::pod::Value::Object(buffers),
                ) {
                    let bytes = bytes.0.into_inner();
                    if let Some(pod) = Pod::from_bytes(&bytes) {
                        let _ = stream.update_params(&mut [pod]);
                    }
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
                let Some(pixels) = data.data() else {
                    // No CPU mapping: a DMA-BUF slipped through negotiation.
                    // Count it so `run` can say so; returning quietly here is
                    // what made a dead capture look like a static desktop.
                    if let Ok(mut latest) = latest.lock() {
                        latest.unmappable_buffers += 1;
                    }
                    return;
                };
                let Ok(mut latest) = latest.lock() else {
                    return;
                };
                if latest.pixels.len() < row_bytes * height as usize {
                    latest.pixels = vec![0; row_bytes * height as usize];
                }
                latest.width = width;
                latest.height = height;
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

    /// Builds `rows` tightly packed rows of `row_bytes`, each filled with its
    /// own row index.
    fn image(row_bytes: usize, rows: usize) -> Vec<u8> {
        (0..rows)
            .flat_map(|r| std::iter::repeat_n(r as u8, row_bytes))
            .collect()
    }

    #[test]
    fn a_stream_error_is_reported_as_a_lost_source() {
        use pw::stream::StreamState::{Error, Streaming};
        assert_eq!(
            disconnect_reason(&Streaming, &Error("node destroyed".into())),
            Some("node destroyed".to_owned())
        );
    }

    #[test]
    fn disconnecting_after_running_is_a_lost_source() {
        // What a closed window looks like: the node goes away and the stream
        // falls back to Unconnected without ever erroring.
        use pw::stream::StreamState::{Paused, Streaming, Unconnected};
        assert!(disconnect_reason(&Streaming, &Unconnected).is_some());
        assert!(disconnect_reason(&Paused, &Unconnected).is_some());
    }

    #[test]
    fn the_states_a_stream_opens_through_are_not_a_lost_source() {
        // Every stream starts Unconnected and climbs; reporting a loss here
        // would make every successful open look like an immediate failure.
        use pw::stream::StreamState::{Connecting, Paused, Streaming, Unconnected};
        assert!(disconnect_reason(&Unconnected, &Connecting).is_none());
        assert!(disconnect_reason(&Connecting, &Paused).is_none());
        assert!(disconnect_reason(&Paused, &Streaming).is_none());
        assert!(disconnect_reason(&Streaming, &Paused).is_none());
    }

    #[test]
    fn rows_are_written_at_the_destination_stride_not_packed() {
        // FFmpeg pads each line, so writing packed would put row 1 inside
        // row 0's padding and skew the whole image.
        let (row_bytes, rows, stride) = (8, 3, 12);
        let src = image(row_bytes, rows);
        let mut dst = vec![0xEEu8; stride * rows];

        assert!(copy_rows_into(&mut dst, stride, &src, row_bytes, rows));
        for row in 0..rows {
            assert_eq!(
                &dst[row * stride..row * stride + row_bytes],
                &vec![row as u8; row_bytes][..],
                "row {row} lands at its own stride offset"
            );
            assert!(
                dst[row * stride + row_bytes..(row + 1) * stride]
                    .iter()
                    .all(|&b| b == 0xEE),
                "the destination's own padding is left alone"
            );
        }
    }

    #[test]
    fn a_source_shorter_than_it_claims_is_refused_without_writing() {
        let (row_bytes, rows) = (8, 3);
        let src = vec![0x77u8; row_bytes * rows - 1];
        let mut dst = vec![0u8; row_bytes * rows];

        assert!(!copy_rows_into(&mut dst, row_bytes, &src, row_bytes, rows));
        assert!(
            dst.iter().all(|&b| b == 0),
            "a rejected copy must leave the frame untouched rather than half-written"
        );
    }

    #[test]
    fn a_destination_too_small_for_the_frame_is_refused() {
        // Mirrors a renegotiation racing ahead of the pool rebuild.
        let (row_bytes, rows) = (8, 3);
        let src = image(row_bytes, rows);
        let mut dst = vec![0u8; row_bytes * (rows - 1)];

        assert!(!copy_rows_into(&mut dst, row_bytes, &src, row_bytes, rows));
        assert!(dst.iter().all(|&b| b == 0));
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
