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
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::{self, RecvTimeoutError},
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

use enumflags2::BitFlags;
use ffmpeg_next as ffmpeg;
#[cfg(feature = "cuda")]
use ffmpeg_next::ffi;
use pipewire as pw;
use pw::{properties::properties, spa};
use spa::pod::Pod;
use spa::sys as spa_sys;
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_error, pp_info, pp_warn};

#[cfg(feature = "cuda")]
use crate::{
    elements::CudaUploadError,
    platform::cuda::{CudaDevice, CudaFrameFormat, frame::create_hw_frames_ctx},
    platform::ffmpeg::AvBufferRef,
    platform::linux::dmabuf_cuda::{
        CudaBgraSurface, DmaBufCudaError, DmaBufCudaImporter, DmaBufPlane,
    },
};

use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    contract::{MediaKind, MemoryDomain, OutputContract, PortContract},
    control::{ControlReceiver, drain_control},
    element::{Element, ElementType, Source, SourceElement, element_pp_log},
    elements::VideoFormat,
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

    /// GPU capture could not be set up, or a captured buffer could not be
    /// imported. Terminal: every following buffer would fail the same way,
    /// and this mode deliberately has no CPU fallback to drop back to — see
    /// [`PipeWireScreenCaptureSource::open_gpu`].
    #[cfg(feature = "cuda")]
    #[error("GPU capture failed: {0}")]
    GpuImport(#[from] DmaBufCudaError),

    /// The CUDA pool captured surfaces are allocated from could not be built,
    /// or would not hand one out. The same error `CudaUpload` reports for the
    /// same operations, since this allocates from an identically shaped
    /// frames context.
    #[cfg(feature = "cuda")]
    #[error("GPU capture's CUDA frame pool failed: {0}")]
    CudaFrames(#[from] crate::elements::CudaUploadError),

    /// A pooled CUDA frame arrived with no surface pointer to import into.
    #[cfg(feature = "cuda")]
    #[error("the pooled CUDA frame carries no surface")]
    NoCudaSurface,

    /// Taking this element's own reference to the caller's CUDA device
    /// context failed.
    #[cfg(feature = "cuda")]
    #[error("failed to reference the CUDA device context")]
    CudaDeviceRef,

    /// GPU capture negotiated DMA-BUF and the compositor sent something else
    /// anyway. Not a fallback point: the frames context, the import, and
    /// everything downstream are built for CUDA surfaces.
    #[cfg(feature = "cuda")]
    #[error("GPU capture expected a DMA-BUF buffer, got {0}")]
    NotDmaBuf(String),
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
    /// Set when a buffer's chunk did not describe a whole frame, so nothing
    /// was copied out of it. Reported for the same reason as
    /// `unmappable_buffers`: dropping every frame silently is
    /// indistinguishable from a frozen desktop.
    short_buffers: u64,
    /// `false` until the first real (non-empty) frame lands, so `run` can
    /// tell "nothing captured yet" apart from "a genuinely black screen".
    have_frame: bool,
    /// The latest captured CUDA surface, in GPU mode only — `pixels` stays
    /// empty there and this carries the frame instead.
    ///
    /// A whole `AVFrame` rather than a raw pointer: the surface belongs to a
    /// refcounted pool, and holding this reference is what stops that
    /// surface being handed out again while it is still the latest capture.
    #[cfg(feature = "cuda")]
    frame: Option<ffmpeg::frame::Video>,
    /// Set when the PipeWire thread hits an unrecoverable stream error, so
    /// `run` can surface it instead of silently emitting stale frames
    /// forever.
    error: Option<PipeWireScreenCaptureSourceError>,
}

/// What `run_pipewire` captures into, chosen by which constructor was used.
enum CaptureTarget {
    /// CPU-mapped buffers (`MemFd`/`MemPtr`) copied into plain BGRA frames.
    Cpu,
    /// DMA-BUF imported into CUDA BGRA surfaces — see
    /// [`PipeWireScreenCaptureSource::open_gpu`].
    #[cfg(feature = "cuda")]
    Gpu(Arc<AvBufferRef>),
}

/// GPU-mode capture state, owned by the PipeWire thread: the DMA-BUF import
/// and the CUDA pool its results land in.
#[cfg(feature = "cuda")]
struct GpuCapture {
    importer: DmaBufCudaImporter,
    /// This element's own reference to the shared device context, released
    /// when this is dropped.
    hw_device_ctx: Arc<AvBufferRef>,
    /// The BGRA CUDA pool captured surfaces come from. Rebuilt when the
    /// compositor renegotiates the size — pooled surfaces are allocated to a
    /// fixed one, and outstanding frames keep the old context alive through
    /// their own references.
    hw_frames_ctx: Option<AvBufferRef>,
    width: u32,
    height: u32,
}

#[cfg(feature = "cuda")]
impl GpuCapture {
    fn new(
        hw_device_ctx: Arc<AvBufferRef>,
    ) -> std::result::Result<Self, PipeWireScreenCaptureSourceError> {
        // Built before the stream connects: its modifier list is what the
        // format negotiation offers the compositor. On failure `?` drops
        // `hw_device_ctx`, which is what releases the reference `open_gpu`
        // took.
        let importer = DmaBufCudaImporter::new()?;
        Ok(Self {
            importer,
            hw_device_ctx,
            hw_frames_ctx: None,
            width: 0,
            height: 0,
        })
    }

    /// Imports one captured DMA-BUF into a fresh CUDA surface.
    fn capture(
        &mut self,
        plane: DmaBufPlane,
        width: u32,
        height: u32,
    ) -> std::result::Result<ffmpeg::frame::Video, PipeWireScreenCaptureSourceError> {
        self.ensure_frames_ctx(width, height)?;

        let mut frame = ffmpeg::frame::Video::empty();
        let frames_ctx = self
            .hw_frames_ctx
            .as_ref()
            .ok_or(CudaUploadError::HwFramesAlloc)?;
        let code =
            // SAFETY: `frames_ctx` is this capture's own pool, and `frame` is the empty
            // local the surface is being allocated into.
            unsafe { ffi::av_hwframe_get_buffer(frames_ctx.as_ptr(), frame.as_mut_ptr(), 0) };
        if code < 0 {
            return Err(CudaUploadError::HwFrameGet(code).into());
        }
        let surface = CudaBgraSurface::from_frame(&frame)
            .ok_or(PipeWireScreenCaptureSourceError::NoCudaSurface)?;
        self.importer.copy_into(plane, width, height, surface)?;

        // The same full-range RGB contract the CPU path stamps on its frames;
        // downstream cannot read it off a CUDA surface.
        frame.set_color_space(ffmpeg::color::Space::RGB);
        frame.set_color_range(ffmpeg::color::Range::JPEG);
        Ok(frame)
    }

    fn ensure_frames_ctx(
        &mut self,
        width: u32,
        height: u32,
    ) -> std::result::Result<(), PipeWireScreenCaptureSourceError> {
        if self.hw_frames_ctx.is_some() && (self.width, self.height) == (width, height) {
            return Ok(());
        }
        // SAFETY: `create_hw_frames_ctx`'s contract is a live device context, which
        // is what the owned `AvBufferRef` beside it is.
        let frames_ctx = unsafe {
            create_hw_frames_ctx(&self.hw_device_ctx, CudaFrameFormat::Bgra, width, height)
        }
        .map_err(CudaUploadError::from)?;
        self.hw_frames_ctx = Some(frames_ctx);
        self.width = width;
        self.height = height;
        Ok(())
    }
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
/// # A captured window can be resized — put a `SwScaler` downstream
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
/// Chain a [`crate::elements::SwScaler`] before anything built for a fixed
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
/// same downstream conversions apply. [`Self::open`] negotiates CPU-mapped
/// PipeWire buffers (`MemFd`/`MemPtr`) only, so a GPU-resident capture never
/// arrives as a buffer it would have to drop.
///
/// # GPU capture (`cuda` feature)
///
/// When built with `cuda`, the `open_gpu` constructor captures the same desktop
/// into **CUDA-resident** `Pixel::CUDA` frames instead, by negotiating DMA-BUF
/// and importing each buffer onto the GPU. That is the mode to use in front of
/// `CudaEncoder`: no compositor readback, no host copy, and
/// no `CudaUpload` in the pipeline. It is a separate
/// constructor rather than an option because it needs a
/// `CudaDevice` to allocate against, and the two modes
/// negotiate mutually exclusive buffer kinds.
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
    /// Whether captured frames are CUDA surfaces rather than CPU pixels.
    /// Fixed at construction by which of `open`/`open_gpu` was used.
    #[cfg(feature = "cuda")]
    gpu: bool,
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
    /// Returns the element, the capture's [`VideoFormat`], and a restore token
    /// to persist for the next run (`None` if the compositor declined to
    /// issue one) — the same shape `DxgiCaptureSource::open`
    /// returns, so a caller can build a matching downstream
    /// [`crate::elements::SwScaler`]/[`crate::elements::SwEncoder`]/
    /// [`crate::elements::Mp4Muxer`] from one value. The size comes from the
    /// stream's negotiated format rather than the portal's reported monitor
    /// size, because compositor scaling can make the two differ.
    pub fn open(
        name: impl Into<String>,
        options: PipeWireScreenCaptureOptions,
    ) -> std::result::Result<(Self, VideoFormat, Option<String>), PipeWireScreenCaptureSourceError>
    {
        Self::open_with(name, options, CaptureTarget::Cpu)
    }

    /// Opens the same capture, but emits **CUDA-resident** `Pixel::CUDA`
    /// frames (`CudaFrameFormat::Bgra`) instead of CPU ones, so a recording
    /// pipeline is `PipeWireScreenCaptureSource -> CudaEncoder -> Mp4Muxer`
    /// with no `CudaUpload` in between and no captured
    /// pixel ever touching system memory.
    ///
    /// `device` must be the same [`CudaDevice`] every other CUDA element in
    /// the pipeline is built from — this element takes its own FFmpeg
    /// reference, so `device` itself need not outlive the call. The EGL
    /// device the import runs on is selected by its `EGL_CUDA_DEVICE_NV`, so
    /// the imported buffer and the frame it lands in are on one GPU by
    /// construction rather than by agreement.
    ///
    /// # No fallback
    ///
    /// This negotiates DMA-BUF **only**. A compositor that cannot deliver it
    /// fails `open_gpu` rather than quietly reverting to the CPU path: the
    /// point of this mode is the absence of a round trip through system
    /// memory, and silently reintroducing one would misreport what the
    /// pipeline does. Call [`Self::open`] to choose that path deliberately.
    ///
    /// Needs the driver's own libraries at *run* time — `libEGL.so.1`,
    /// `libGLESv2.so.2`, and `libcuda.so.1`, all `dlopen`ed, so a build needs
    /// no development packages for them.
    #[cfg(feature = "cuda")]
    pub fn open_gpu(
        name: impl Into<String>,
        options: PipeWireScreenCaptureOptions,
        device: &CudaDevice,
    ) -> std::result::Result<(Self, VideoFormat, Option<String>), PipeWireScreenCaptureSourceError>
    {
        // Referenced here rather than in the PipeWire thread so a failure is
        // reported by `open_gpu` itself.
        Self::open_with(name, options, CaptureTarget::Gpu(device.retain()))
    }

    fn open_with(
        name: impl Into<String>,
        options: PipeWireScreenCaptureOptions,
        target: CaptureTarget,
    ) -> std::result::Result<(Self, VideoFormat, Option<String>), PipeWireScreenCaptureSourceError>
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
            short_buffers: 0,
            have_frame: false,
            #[cfg(feature = "cuda")]
            frame: None,
            error: None,
        }));
        let (startup_tx, startup_rx) = mpsc::channel::<Startup>();
        let (terminate_tx, terminate_rx) = pw::channel::channel::<Terminate>();

        #[cfg(feature = "cuda")]
        let gpu = matches!(target, CaptureTarget::Gpu(_));

        // Which memory the emitted frames live in, settled here so the pad
        // below can declare it. Without the `cuda` feature there is no
        // `open_gpu` to call at all, so every frame is a CPU one.
        #[cfg(feature = "cuda")]
        let memory = if gpu {
            MemoryDomain::Cuda
        } else {
            MemoryDomain::System
        };
        #[cfg(not(feature = "cuda"))]
        let memory = MemoryDomain::System;

        let worker = {
            let latest = latest.clone();
            let fps = options.fps.max(1);
            std::thread::Builder::new()
                .name(format!("{name}-pipewire"))
                .spawn(move || {
                    if let Err(error) =
                        run_pipewire(cast, fps, target, latest.clone(), &startup_tx, terminate_rx)
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
        // Same `1 / fps` convention as `PipeWireScreenCaptureSource::time_base`
        // — computed here too since `Self` is moved into the tuple below before
        // that method could be called on it.
        let time_base = ffmpeg::Rational::new(1, fps as i32);
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
                // `open` emits CPU frames and `open_gpu` CUDA-resident
                // ones, and which of the two is settled here — so a
                // downstream filter can be checked against it.
                pad: SrcPad::with_contract(
                    format!("{name}_src"),
                    OutputContract::Fixed(PortContract::frame(MediaKind::VideoFrame, memory)),
                ),
                name: name.into(),
                pp_log,
                width,
                height,
                fps,
                frame_interval: Duration::from_secs_f64(1.0 / fps as f64),
                frame_index: 0,
                // GPU mode pools only the small `AVFrame` wrapper: the
                // surface it references comes from the CUDA frames context
                // the PipeWire thread allocates from. Same split as
                // `CudaUpload`.
                #[cfg(feature = "cuda")]
                pool: if gpu {
                    UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {})
                } else {
                    UnboundObjectPool::new(
                        0,
                        move || {
                            ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, width, height)
                        },
                        |_| {},
                    )
                },
                #[cfg(not(feature = "cuda"))]
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
                #[cfg(feature = "cuda")]
                gpu,
            },
            VideoFormat {
                width,
                height,
                time_base,
            },
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
        #[cfg(feature = "cuda")]
        if self.gpu {
            return self.emit_gpu_frame();
        }
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
        // through `SwScaler`, which rebuilds its own context per input size — see
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

    /// Takes a new reference to the latest captured CUDA surface and stamps
    /// this tick's `pts` on it.
    ///
    /// No pixel copy, unlike [`Self::emit_frame`]'s CPU path, and none is
    /// needed for the same reason that one exists: `av_frame_ref` gives this
    /// tick its own `AVFrame` to stamp, while the surface underneath stays
    /// the PipeWire thread's. That thread allocates the *next* capture from
    /// the frames context's pool, which will not hand back a surface that is
    /// still referenced — so an already-pushed frame's pixels cannot change
    /// under whatever is still reading them.
    #[cfg(feature = "cuda")]
    fn emit_gpu_frame(
        &mut self,
    ) -> Option<crate::pool::UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        let mut frame = self.pool.get();
        let (src_width, src_height) = {
            let latest = self.latest.lock().ok()?;
            if !latest.have_frame {
                return None;
            }
            let source = latest.frame.as_ref()?;
            // SAFETY: `destination` is the pooled wrapper's own `AVFrame` and `source`
            // is the latest captured frame, still held by the lock above; they are
            // distinct, so the unref-then-ref pair neither aliases nor drops the
            // capture it is about to reference.
            unsafe {
                let destination = frame.as_mut_ptr();
                // Releases the previous capture this pooled wrapper still
                // referenced, which is what lets its surface return to the
                // frames pool instead of being held for this element's life.
                ffi::av_frame_unref(destination);
                if ffi::av_frame_ref(destination, source.as_ptr()) < 0 {
                    return None;
                }
            }
            (latest.width, latest.height)
        };
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
        }
        frame.set_pts(Some(self.frame_index));
        self.frame_index += 1;
        Some(frame)
    }

    /// How many unreadable buffers have arrived since the last check, if any.
    /// Taken rather than peeked so the warning is not repeated every tick.
    fn take_unmappable_report(&self) -> Option<u64> {
        let mut latest = self.latest.lock().ok()?;
        (latest.unmappable_buffers > 0).then(|| std::mem::take(&mut latest.unmappable_buffers))
    }

    /// How many buffers arrived whose chunk did not describe a whole frame.
    /// Taken rather than peeked, for the same reason as the report above.
    fn take_short_report(&self) -> Option<u64> {
        let mut latest = self.latest.lock().ok()?;
        (latest.short_buffers > 0).then(|| std::mem::take(&mut latest.short_buffers))
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
    fn is_live(&self) -> bool {
        true
    }

    fn is_seekable(&self) -> bool {
        false
    }

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
            if let Some(count) = self.take_short_report() {
                pp_warn!(
                    self,
                    "dropped {count} buffer(s) whose chunk described less than a \
                     {}x{} frame: the captured image is whatever last arrived whole",
                    self.width,
                    self.height
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
/// The pixels inside one mapped buffer.
///
/// SPA describes the valid region of a buffer with its chunk's own `offset`
/// and `size`, not with the bounds of the mapping, and a node is free to hand
/// over a mapping whose image starts partway in. Reading from the start of
/// the mapping instead would copy whatever precedes the image and call it a
/// desktop.
///
/// Both bounds are clamped to the mapping. `None` means the chunk describes
/// nothing readable; whether what it does describe is a *whole* frame is
/// `repack_rows`' own check, which refuses a short source rather than writing
/// part of one.
fn chunk_image(pixels: &[u8], offset: usize, size: usize) -> Option<&[u8]> {
    let start = offset.min(pixels.len());
    let end = offset.saturating_add(size).min(pixels.len());
    (end > start).then(|| &pixels[start..end])
}

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
    target: CaptureTarget,
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

    // GPU mode's import lives on this thread for its whole life: its EGL
    // context is current here and nowhere else.
    #[cfg(feature = "cuda")]
    let mut gpu = match target {
        CaptureTarget::Cpu => None,
        CaptureTarget::Gpu(hw_device_ctx) => Some(GpuCapture::new(hw_device_ctx)?),
    };
    #[cfg(not(feature = "cuda"))]
    let CaptureTarget::Cpu = target;

    // The modifiers the import can accept, which is exactly what the format
    // negotiation may offer: empty means CPU mode, which offers none at all.
    #[cfg(feature = "cuda")]
    let modifiers: Vec<u64> = gpu
        .as_ref()
        .map(|gpu| gpu.importer.modifiers().to_vec())
        .unwrap_or_default();
    #[cfg(not(feature = "cuda"))]
    let modifiers: Vec<u64> = Vec::new();

    // Negotiated size, shared between the two callbacks below. Both run on
    // this thread's own loop, so a `Cell` is enough — no lock needed.
    let size = Arc::new(Mutex::new((0u32, 0u32)));
    // The modifier the compositor fixated on, which every import of a buffer
    // from this stream has to be told.
    let modifier = Arc::new(Mutex::new(0u64));
    // Bumped on every negotiated format. The compositor reallocates its
    // buffers each time, so this is what tells the import that whatever it
    // cached describes memory that no longer exists.
    let negotiation = Arc::new(AtomicU64::new(0));

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
            let modifiers = modifiers.clone();
            let modifier = modifier.clone();
            let negotiation = negotiation.clone();
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

                if !modifiers.is_empty() {
                    // GPU mode. The compositor answers the unfixated offer
                    // with the modifiers it shares with this driver and asks
                    // for one to be chosen; until that round trip is done it
                    // allocates no buffers, so this returns rather than
                    // reporting the format as negotiated.
                    let (negotiated, fixation_required) = negotiated_modifier(param);
                    let Some(negotiated) = negotiated else {
                        let _ = startup.send(Startup::Failed(
                            PipeWireScreenCaptureSourceError::PipeWire(
                                "the compositor negotiated no DMA-BUF modifier".into(),
                            ),
                        ));
                        return;
                    };
                    if let Ok(mut modifier) = modifier.lock() {
                        *modifier = negotiated;
                    }
                    if fixation_required {
                        if let Some(bytes) =
                            fixated_format_pod(format, negotiated, width, height, fps)
                            && let Some(pod) = Pod::from_bytes(&bytes)
                        {
                            let _ = stream.update_params(&mut [pod]);
                        } else {
                            let _ = startup.send(Startup::Failed(
                                PipeWireScreenCaptureSourceError::PipeWire(
                                    "failed to build the fixated format pod".into(),
                                ),
                            ));
                        }
                        return;
                    }
                }

                negotiation.fetch_add(1, Ordering::Release);
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
                // actually use. In CPU mode that is the mapped kinds only:
                // without this the compositor is free to hand over DMA-BUF
                // once the captured content becomes GPU-resident (a
                // fullscreen video, say), those have no CPU mapping, every
                // frame is dropped, and the capture silently freezes at the
                // last readable image. GPU mode is the exact mirror — a
                // mapped buffer there has no importable fd, and this mode has
                // no CPU path to fall back to.
                let data_type = if modifiers.is_empty() {
                    (1 << spa::buffer::DataType::MemFd.as_raw())
                        | (1 << spa::buffer::DataType::MemPtr.as_raw())
                } else {
                    1 << spa::buffer::DataType::DmaBuf.as_raw()
                };
                if let Some(bytes) = buffers_pod(data_type)
                    && let Some(pod) = Pod::from_bytes(&bytes)
                {
                    let _ = stream.update_params(&mut [pod]);
                }

                let _ = startup.send(Startup::Ready { width, height });
            }
        })
        .process({
            let size = size.clone();
            #[cfg(feature = "cuda")]
            let modifier = modifier.clone();
            #[cfg(feature = "cuda")]
            let negotiation = negotiation.clone();
            // Moved in: the import runs here, on the loop thread its EGL
            // context is current on.
            #[cfg(feature = "cuda")]
            let mut gpu = gpu.take();
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
                let (offset, chunk_size) =
                    (data.chunk().offset() as usize, data.chunk().size() as usize);
                let Ok((width, height)) = size.lock().map(|s| *s) else {
                    return;
                };
                if width == 0 || height == 0 {
                    return;
                }

                #[cfg(feature = "cuda")]
                if let Some(gpu) = gpu.as_mut() {
                    if data.type_() != spa::buffer::DataType::DmaBuf {
                        // Negotiation asked for DMA-BUF only, so this is the
                        // compositor contradicting itself rather than a
                        // condition to fall back from.
                        if let Ok(mut latest) = latest.lock() {
                            latest.error.get_or_insert(
                                PipeWireScreenCaptureSourceError::NotDmaBuf(format!(
                                    "{:?}",
                                    data.type_()
                                )),
                            );
                        }
                        return;
                    }
                    // Buffers from an earlier negotiation are gone, whether
                    // or not the size changed with it.
                    gpu.importer
                        .sync_negotiation(negotiation.load(Ordering::Acquire));
                    let plane = DmaBufPlane {
                        fd: data.fd() as std::ffi::c_int,
                        offset: data.chunk().offset(),
                        stride,
                        modifier: modifier
                            .lock()
                            .map(|modifier| *modifier)
                            .unwrap_or_default(),
                    };
                    match gpu.capture(plane, width, height) {
                        Ok(frame) => {
                            if let Ok(mut latest) = latest.lock() {
                                latest.width = width;
                                latest.height = height;
                                latest.frame = Some(frame);
                                latest.have_frame = true;
                            }
                        }
                        Err(error) => {
                            if let Ok(mut latest) = latest.lock() {
                                latest.error.get_or_insert(error);
                            }
                        }
                    }
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
                let Some(image) = chunk_image(pixels, offset, chunk_size) else {
                    if let Ok(mut latest) = latest.lock() {
                        latest.short_buffers += 1;
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
                    image,
                    src_stride,
                    row_bytes,
                    height as usize,
                ) {
                    latest.have_frame = true;
                } else {
                    // The chunk described less than a frame; the previous
                    // image is left alone rather than half-overwritten.
                    latest.short_buffers += 1;
                }
            }
        })
        .register()
        .map_err(pw_err)?;

    let values = format_pod(fps, &modifiers).ok_or_else(|| {
        PipeWireScreenCaptureSourceError::PipeWire("failed to build format pod".into())
    })?;
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

/// The `EnumFormat` this element offers the compositor.
///
/// One pod covers both modes: `modifiers` empty asks for plain BGRx/BGRA,
/// which the compositor can only satisfy from mapped memory, and a non-empty
/// list adds the DMA-BUF modifier property that turns the same request into a
/// GPU-buffer one. The list is offered `DONT_FIXATE`, which is what makes the
/// compositor answer with the modifiers it shares with this driver instead of
/// picking blindly — see `negotiated_modifier`.
fn format_pod(fps: u32, modifiers: &[u64]) -> Option<Vec<u8>> {
    use spa::pod::{ChoiceValue, Property, PropertyFlags, Value};
    use spa::utils::{Choice, ChoiceEnum, ChoiceFlags, Fraction, Id, Rectangle};

    let mut properties = vec![
        Property::new(
            spa_sys::SPA_FORMAT_mediaType,
            Value::Id(Id(spa::param::format::MediaType::Video.as_raw())),
        ),
        Property::new(
            spa_sys::SPA_FORMAT_mediaSubtype,
            Value::Id(Id(spa::param::format::MediaSubtype::Raw.as_raw())),
        ),
        Property::new(
            spa_sys::SPA_FORMAT_VIDEO_format,
            Value::Choice(ChoiceValue::Id(Choice(
                ChoiceFlags::empty(),
                ChoiceEnum::Enum {
                    default: Id(spa::param::video::VideoFormat::BGRx.as_raw()),
                    alternatives: vec![
                        Id(spa::param::video::VideoFormat::BGRx.as_raw()),
                        Id(spa::param::video::VideoFormat::BGRA.as_raw()),
                    ],
                },
            ))),
        ),
    ];
    if !modifiers.is_empty() {
        let mut modifier = Property::new(
            spa_sys::SPA_FORMAT_VIDEO_modifier,
            Value::Choice(ChoiceValue::Long(Choice(
                ChoiceFlags::empty(),
                ChoiceEnum::Enum {
                    default: modifiers[0] as i64,
                    alternatives: modifiers.iter().map(|&m| m as i64).collect(),
                },
            ))),
        );
        // MANDATORY says a format without a modifier is not acceptable —
        // otherwise the compositor may answer with mapped memory this mode
        // cannot import.
        modifier.flags = PropertyFlags::MANDATORY | PropertyFlags::DONT_FIXATE;
        properties.push(modifier);
    }
    properties.push(Property::new(
        spa_sys::SPA_FORMAT_VIDEO_size,
        Value::Choice(ChoiceValue::Rectangle(Choice(
            ChoiceFlags::empty(),
            ChoiceEnum::Range {
                default: Rectangle {
                    width: 1920,
                    height: 1080,
                },
                min: Rectangle {
                    width: 1,
                    height: 1,
                },
                max: Rectangle {
                    width: 8192,
                    height: 8192,
                },
            },
        ))),
    ));
    // The compositor answers with a variable rate (`0/1`) capped at this
    // maximum; the emitted rate is this element's own concern, not this.
    properties.push(Property::new(
        spa_sys::SPA_FORMAT_VIDEO_framerate,
        Value::Choice(ChoiceValue::Fraction(Choice(
            ChoiceFlags::empty(),
            ChoiceEnum::Range {
                default: Fraction { num: fps, denom: 1 },
                min: Fraction { num: 0, denom: 1 },
                max: Fraction {
                    num: fps.max(60),
                    denom: 1,
                },
            },
        ))),
    ));

    serialize_object(spa::pod::Object {
        type_: spa::utils::SpaTypes::ObjectParamFormat.as_raw(),
        id: spa::param::ParamType::EnumFormat.as_raw(),
        properties,
    })
}

/// The answer to a compositor that asked for the modifier to be fixated: the
/// same format with exactly one modifier left and the `DONT_FIXATE` flag
/// gone. Until this is sent the stream allocates no buffers.
fn fixated_format_pod(
    format: spa::param::video::VideoFormat,
    modifier: u64,
    width: u32,
    height: u32,
    fps: u32,
) -> Option<Vec<u8>> {
    use spa::pod::{ChoiceValue, Property, PropertyFlags, Value};
    use spa::utils::{Choice, ChoiceEnum, ChoiceFlags, Fraction, Id, Rectangle};

    let mut modifier = Property::new(
        spa_sys::SPA_FORMAT_VIDEO_modifier,
        Value::Choice(ChoiceValue::Long(Choice(
            ChoiceFlags::empty(),
            ChoiceEnum::Enum {
                default: modifier as i64,
                alternatives: vec![modifier as i64],
            },
        ))),
    );
    modifier.flags = PropertyFlags::MANDATORY;

    serialize_object(spa::pod::Object {
        type_: spa::utils::SpaTypes::ObjectParamFormat.as_raw(),
        id: spa::param::ParamType::EnumFormat.as_raw(),
        properties: vec![
            Property::new(
                spa_sys::SPA_FORMAT_mediaType,
                Value::Id(Id(spa::param::format::MediaType::Video.as_raw())),
            ),
            Property::new(
                spa_sys::SPA_FORMAT_mediaSubtype,
                Value::Id(Id(spa::param::format::MediaSubtype::Raw.as_raw())),
            ),
            Property::new(
                spa_sys::SPA_FORMAT_VIDEO_format,
                Value::Id(Id(format.as_raw())),
            ),
            modifier,
            Property::new(
                spa_sys::SPA_FORMAT_VIDEO_size,
                Value::Rectangle(Rectangle { width, height }),
            ),
            Property::new(
                spa_sys::SPA_FORMAT_VIDEO_framerate,
                Value::Choice(ChoiceValue::Fraction(Choice(
                    ChoiceFlags::empty(),
                    ChoiceEnum::Range {
                        default: Fraction { num: fps, denom: 1 },
                        min: Fraction { num: 0, denom: 1 },
                        max: Fraction {
                            num: fps.max(60),
                            denom: 1,
                        },
                    },
                ))),
            ),
        ],
    })
}

/// The `Buffers` param restricting negotiation to the memory kinds this
/// element can use — see its one call site on why each mode names exactly
/// one set.
fn buffers_pod(data_type: i32) -> Option<Vec<u8>> {
    serialize_object(spa::pod::object!(
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
    ))
}

/// The modifier a `Format` param carries, and whether the compositor still
/// wants it fixated.
///
/// Read out of the pod directly rather than through `VideoInfoRaw::flags`:
/// the flags that carry this (`SPA_VIDEO_FLAG_MODIFIER`,
/// `..._FIXATION_REQUIRED`) only exist behind libspa's `v0_3_65`/`v0_3_75`
/// features, and this crate deliberately builds against a lower minimum —
/// see `lib/Cargo.toml`. The property itself has been there throughout.
fn negotiated_modifier(pod: &Pod) -> (Option<u64>, bool) {
    // SAFETY: `pod.as_raw_ptr()` is a live `spa_pod` owned by the callback's
    // parameter, and a POD's own `size` field excludes its header — which is
    // why the header is added back to get the whole object's extent.
    let bytes = unsafe {
        let raw = pod.as_raw_ptr();
        std::slice::from_raw_parts(
            raw as *const u8,
            (*raw).size as usize + std::mem::size_of::<spa_sys::spa_pod>(),
        )
    };
    let Ok((_, spa::pod::Value::Object(object))) =
        spa::pod::deserialize::PodDeserializer::deserialize_any_from(bytes)
    else {
        return (None, false);
    };
    let Some(property) = object
        .properties
        .into_iter()
        .find(|property| property.key == spa_sys::SPA_FORMAT_VIDEO_modifier)
    else {
        return (None, false);
    };
    let fixation_required = property
        .flags
        .contains(spa::pod::PropertyFlags::DONT_FIXATE);
    let modifier = match property.value {
        spa::pod::Value::Long(modifier) => Some(modifier as u64),
        spa::pod::Value::Choice(spa::pod::ChoiceValue::Long(spa::utils::Choice(_, choice))) => {
            match choice {
                spa::utils::ChoiceEnum::None(modifier) => Some(modifier as u64),
                spa::utils::ChoiceEnum::Enum { default, .. } => Some(default as u64),
                spa::utils::ChoiceEnum::Range { default, .. } => Some(default as u64),
                spa::utils::ChoiceEnum::Step { default, .. } => Some(default as u64),
                spa::utils::ChoiceEnum::Flags { default, .. } => Some(default as u64),
            }
        }
        _ => None,
    };
    (modifier, fixation_required)
}

fn serialize_object(object: spa::pod::Object) -> Option<Vec<u8>> {
    spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &spa::pod::Value::Object(object),
    )
    .ok()
    .map(|serialized| serialized.0.into_inner())
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

    /// Parses a serialized param pod back into the object it encodes, so a
    /// test asserts on what the compositor would actually receive rather than
    /// on the builder's own inputs.
    fn parse_object(bytes: &[u8]) -> spa::pod::Object {
        let (_, value) = spa::pod::deserialize::PodDeserializer::deserialize_any_from(bytes)
            .expect("the builder produced a decodable pod");
        match value {
            spa::pod::Value::Object(object) => object,
            other => panic!("expected an object pod, got {other:?}"),
        }
    }

    fn property(object: &spa::pod::Object, key: u32) -> Option<&spa::pod::Property> {
        object.properties.iter().find(|prop| prop.key == key)
    }

    fn modifier_alternatives(property: &spa::pod::Property) -> Vec<u64> {
        match &property.value {
            spa::pod::Value::Choice(spa::pod::ChoiceValue::Long(spa::utils::Choice(
                _,
                spa::utils::ChoiceEnum::Enum { alternatives, .. },
            ))) => alternatives.iter().map(|&m| m as u64).collect(),
            other => panic!("expected a Long enum choice, got {other:?}"),
        }
    }

    #[test]
    fn the_cpu_format_offers_no_modifier_at_all() {
        // What keeps the compositor from answering with a DMA-BUF the CPU
        // path could only drop.
        let pod = format_pod(30, &[]).expect("the CPU format pod builds");
        let object = parse_object(&pod);
        assert!(property(&object, spa_sys::SPA_FORMAT_VIDEO_modifier).is_none());
        assert!(property(&object, spa_sys::SPA_FORMAT_VIDEO_format).is_some());
    }

    #[test]
    fn the_gpu_format_offers_every_modifier_unfixated() {
        let modifiers = [0x0300_0000_0060_6010u64, 0x00ff_ffff_ffff_ffff];
        let pod = format_pod(30, &modifiers).expect("the DMA-BUF format pod builds");
        let object = parse_object(&pod);
        let offered =
            property(&object, spa_sys::SPA_FORMAT_VIDEO_modifier).expect("modifier is offered");
        // Both flags matter: MANDATORY rules out an answer with no modifier
        // (which would be mapped memory), DONT_FIXATE is what makes the
        // compositor reply with the modifiers it shares with this driver
        // instead of picking one blindly.
        assert!(offered.flags.contains(spa::pod::PropertyFlags::MANDATORY));
        assert!(offered.flags.contains(spa::pod::PropertyFlags::DONT_FIXATE));
        assert_eq!(modifier_alternatives(offered), modifiers);
    }

    #[test]
    fn the_fixated_format_names_one_modifier_and_drops_dont_fixate() {
        let pod = fixated_format_pod(
            spa::param::video::VideoFormat::BGRx,
            0x0300_0000_0060_6010,
            1920,
            1080,
            30,
        )
        .expect("the fixated format pod builds");
        let object = parse_object(&pod);
        let fixated =
            property(&object, spa_sys::SPA_FORMAT_VIDEO_modifier).expect("modifier is fixated");
        assert!(!fixated.flags.contains(spa::pod::PropertyFlags::DONT_FIXATE));
        assert_eq!(modifier_alternatives(fixated), [0x0300_0000_0060_6010]);
        // The size has to be concrete too: a range here reads as a fresh
        // offer rather than an answer, and the compositor keeps renegotiating.
        assert!(matches!(
            property(&object, spa_sys::SPA_FORMAT_VIDEO_size).map(|prop| &prop.value),
            Some(spa::pod::Value::Rectangle(spa::utils::Rectangle {
                width: 1920,
                height: 1080
            }))
        ));
    }

    #[test]
    fn a_modifier_choice_is_read_back_as_needing_fixation() {
        let pod = format_pod(30, &[0x0300_0000_0060_6010, 0x0300_0000_0060_6011])
            .expect("the DMA-BUF format pod builds");
        let (modifier, fixation_required) =
            negotiated_modifier(Pod::from_bytes(&pod).expect("a valid pod"));
        assert_eq!(modifier, Some(0x0300_0000_0060_6010));
        assert!(fixation_required);
    }

    #[test]
    fn a_fixed_modifier_is_read_back_as_settled() {
        let pod = fixated_format_pod(
            spa::param::video::VideoFormat::BGRx,
            0x0300_0000_00e0_8014,
            800,
            600,
            30,
        )
        .expect("the fixated format pod builds");
        let (modifier, fixation_required) =
            negotiated_modifier(Pod::from_bytes(&pod).expect("a valid pod"));
        assert_eq!(modifier, Some(0x0300_0000_00e0_8014));
        assert!(!fixation_required);
    }

    #[test]
    fn a_format_without_a_modifier_negotiates_none() {
        let pod = format_pod(30, &[]).expect("the CPU format pod builds");
        assert_eq!(
            negotiated_modifier(Pod::from_bytes(&pod).expect("a valid pod")),
            (None, false)
        );
    }

    #[test]
    fn the_buffers_param_names_exactly_the_requested_memory_kinds() {
        let dma_buf = 1 << spa::buffer::DataType::DmaBuf.as_raw();
        let pod = buffers_pod(dma_buf).expect("the buffers pod builds");
        let object = parse_object(&pod);
        let data_type =
            property(&object, spa_sys::SPA_PARAM_BUFFERS_dataType).expect("dataType is named");
        match &data_type.value {
            spa::pod::Value::Choice(spa::pod::ChoiceValue::Int(spa::utils::Choice(
                _,
                spa::utils::ChoiceEnum::Flags { default, flags },
            ))) => {
                assert_eq!(*default, dma_buf);
                assert_eq!(flags, &vec![dma_buf]);
                // Mapped memory is not in the set: GPU mode has no CPU path
                // to fall back to, so a mapped buffer must not be offered.
                assert_eq!(*default & (1 << spa::buffer::DataType::MemFd.as_raw()), 0);
            }
            other => panic!("expected an Int flags choice, got {other:?}"),
        }
    }

    /// The reference `open_gpu` takes is taken *before* the portal handshake,
    /// which blocks on a dialog the user can cancel — and before a thread
    /// that may fail to spawn. Every one of those paths abandons the carrier,
    /// so dropping it has to be what releases the reference. The refcount is
    /// that contract in observable form.
    #[cfg(feature = "cuda")]
    #[test]
    fn an_abandoned_device_reference_is_released() {
        let Some((device, _cuda_lock)) = crate::test_support::try_cuda_device() else {
            return;
        };
        let owner = device.retain();
        let before = Arc::strong_count(&owner);
        {
            let held = owner.clone();
            assert_eq!(
                Arc::strong_count(&held),
                before + 1,
                "the carrier holds a reference of its own while it lives"
            );
        }
        assert_eq!(
            Arc::strong_count(&owner),
            before,
            "an abandoned carrier must release the reference open_gpu took"
        );
    }

    /// The same contract once the reference has reached the `GpuCapture` that
    /// adopts it, which is the path a successful `open_gpu` takes. A machine
    /// where the import cannot open at all exercises the other half: `new`
    /// fails and releases the reference on its way out.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_gpu_capture_releases_its_device_reference() {
        let Some((device, _cuda_lock)) = crate::test_support::try_cuda_device() else {
            return;
        };
        let owner = device.retain();
        let before = Arc::strong_count(&owner);
        let carrier = owner.clone();
        match GpuCapture::new(carrier) {
            Ok(gpu) => {
                assert_eq!(
                    Arc::strong_count(&owner),
                    before + 1,
                    "the capture holds the reference while it lives"
                );
                drop(gpu);
            }
            Err(error) => {
                eprintln!("no usable DMA-BUF import here ({error}); checking release only")
            }
        }
        assert_eq!(
            Arc::strong_count(&owner),
            before,
            "the reference must be released whether the capture was built or not"
        );
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

    /// A node may hand over a mapping whose image starts partway in, and the
    /// chunk is what says where. Reading from the mapping's start copies
    /// whatever precedes the image instead.
    #[test]
    fn pixels_are_read_from_the_chunk_the_node_described() {
        let mut buffer = vec![0xFFu8; 64];
        buffer[16..48].fill(0x11);

        let image = chunk_image(&buffer, 16, 32).expect("the chunk holds an image");
        assert_eq!(image.len(), 32);
        assert!(
            image.iter().all(|&byte| byte == 0x11),
            "an ignored offset reads the bytes before the image"
        );
    }

    #[test]
    fn a_chunk_reaching_past_its_mapping_is_clamped_to_it() {
        let buffer = vec![0x22u8; 64];

        assert_eq!(chunk_image(&buffer, 48, 64).map(<[u8]>::len), Some(16));
        assert_eq!(chunk_image(&buffer, 64, 16), None);
        assert_eq!(chunk_image(&buffer, 0, 0), None);
    }

    /// What the clamping is for: a chunk that covers less than the frame must
    /// leave the previous image alone rather than write part of a new one.
    #[test]
    fn a_chunk_shorter_than_a_frame_is_refused_rather_than_half_copied() {
        let mut destination = vec![0u8; 4 * 4 * 4];
        let mapping = vec![0x33u8; 64];
        let image = chunk_image(&mapping, 0, 32).expect("the chunk holds something");

        assert!(
            !repack_rows(&mut destination, image, 16, 16, 4),
            "half a frame is not a frame"
        );
        assert!(
            destination.iter().all(|&byte| byte == 0),
            "a refused frame must not have written anything"
        );
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
