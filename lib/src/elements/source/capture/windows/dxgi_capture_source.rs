use std::{
    ffi::c_void,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use crate::pp_log::{PpLog, pp_error, pp_info};
use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;
use windows::{
    Win32::{
        Foundation::{HMODULE, POINT, RECT},
        Graphics::{
            Direct3D::D3D_DRIVER_TYPE_UNKNOWN,
            Direct3D11::{
                D3D11_BIND_FLAG, D3D11_BIND_SHADER_RESOURCE, D3D11_BOX, D3D11_CPU_ACCESS_READ,
                D3D11_CREATE_DEVICE_FLAG, D3D11_MAP_READ, D3D11_SDK_VERSION, D3D11_TEXTURE2D_DESC,
                D3D11_USAGE_DEFAULT, D3D11_USAGE_STAGING, D3D11CreateDevice, ID3D11Device,
                ID3D11DeviceContext, ID3D11Resource, ID3D11Texture2D,
            },
            Dxgi::{
                Common::{DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_SAMPLE_DESC},
                CreateDXGIFactory1, DXGI_ERROR_ACCESS_LOST, DXGI_ERROR_WAIT_TIMEOUT,
                DXGI_OUTDUPL_FRAME_INFO, DXGI_OUTDUPL_MOVE_RECT, DXGI_OUTDUPL_POINTER_SHAPE_INFO,
                DXGI_OUTDUPL_POINTER_SHAPE_TYPE_COLOR,
                DXGI_OUTDUPL_POINTER_SHAPE_TYPE_MASKED_COLOR,
                DXGI_OUTDUPL_POINTER_SHAPE_TYPE_MONOCHROME, IDXGIAdapter1, IDXGIDevice,
                IDXGIFactory1, IDXGIOutput1, IDXGIOutputDuplication, IDXGIResource,
            },
        },
    },
    core::Interface,
};

use crate::{
    buffer::{MediaBuffer, picture_is_referenced, release_picture},
    bus::{Bus, BusEvent},
    contract::{MediaKind, MemoryDomain, OutputContract, PortContract},
    control::{ControlReceiver, drain_control},
    element::{Element, ElementType, Source, SourceElement, element_pp_log},
    elements::VideoFormat,
    error::{D3d11FrameWrapError, D3d11SharedDeviceError, Result},
    pad::SrcPad,
    platform::windows::{
        d3d11::protect_shared_device,
        d3d11va::{d3d11va_texture, wrap_d3d11_texture},
    },
    pool::{UnboundObjectPool, UnboundObjectPoolRef},
    schedule::PeriodicSchedule,
};

/// How often [`DxgiCaptureSource::run`]'s poll loop re-checks
/// `drain_control`/whether it's time to emit, even mid-wait for the next
/// real desktop change — bounds `Stop` latency at very low configured
/// [`DxgiCaptureOptions::fps`] values, where "wait until the next tick" on
/// its own could otherwise be a long, unresponsive block. Same idea as
/// [`crate::queue::Queue`]'s own `STOP_POLL_INTERVAL`.
const POLL_GRANULARITY: Duration = Duration::from_millis(100);

/// How long `AcquireNextFrame` is allowed to wait for the desktop to change:
/// not at all.
///
/// That wait is not a free way to pass the time. It holds the D3D11 device's
/// own lock for its whole duration, so under [`CaptureMode::Gpu`], where the
/// device is deliberately shared with a compositor, a downloader, or an
/// encoder, every one of them stalls for exactly as long as this call blocks.
/// Waiting out a whole frame interval inside it dragged a 60 fps
/// `D3d11VideoCompositor` sharing the device down to roughly 40, and
/// configuring the capture at 30 fps — a *longer* wait per tick — took that
/// compositor to about 21. Even 2 ms of it, paid on every tick as a still
/// screen does since nothing is ever already queued, cost that compositor
/// 3 to 5 fps of its configured 60.
///
/// Nothing is lost by not waiting. Desktop Duplication queues a change until
/// it is acquired, so a poll that finds none simply finds it at the next
/// tick, and [`DxgiCaptureSource::run`] polls at the last moment before each
/// emission — which bounds that to the one tick a fixed-rate emitter
/// quantizes to anyway.
const ACQUIRE_TIMEOUT_MS: u32 = 0;

/// Errors specific to `DxgiCaptureSource`. Converts into the crate-wide
/// `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum DxgiCaptureSourceError {
    /// FFmpeg could not allocate a reference-counted wrapper for a captured texture.
    #[error(transparent)]
    FrameWrap(#[from] D3d11FrameWrapError),
    /// A DXGI, D3D11, or Win32 operation failed.

    #[error("windows error: {0}")]
    Windows(#[from] windows::core::Error),
    /// The flat adapter/output enumeration has no requested index.

    #[error("no DXGI output at index {0} (across every adapter)")]
    NoSuchOutput(u32),

    /// `DXGI_ERROR_ACCESS_LOST` specifically, broken out of the generic
    /// [`DxgiCaptureSourceError::Windows`] variant because it's the single
    /// most common *recoverable* failure mode for desktop duplication —
    /// a lock screen, a UAC prompt, a display mode change, or a
    /// fullscreen-exclusive app/overlay stealing the duplication lock all
    /// surface this way. Same "fail fast, caller rebuilds a fresh one"
    /// contract [`crate::elements::RtspSource`] already documents: this
    /// element doesn't retry internally, callers that want to survive a
    /// lock-screen cycle watch for this specific error and call
    /// [`DxgiCaptureSource::open`] again.
    #[error("DXGI_ERROR_ACCESS_LOST — desktop duplication needs to be reopened")]
    AccessLost,

    /// FFmpeg could not take a second reference to the picture last copied,
    /// which is how [`CaptureMode::Cpu`] emits a tick that captured nothing.
    #[error("failed to reference the captured picture (code {0})")]
    FrameRef(i32),

    /// Seeking was requested on a live desktop capture.
    #[error("DxgiCaptureSource doesn't support seeking a live capture")]
    SeekUnsupported,

    /// The requested region does not intersect an attached output.
    #[error("CaptureArea::Region {0:?} doesn't overlap any display output")]
    RegionOutsideDesktop(CaptureRect),

    /// See [`CaptureArea::Region`]'s own docs on why this is a hard
    /// failure rather than an automatic CPU-bridged fallback.
    #[error(
        "CaptureArea::Region spans outputs on more than one GPU adapter — \
         zero-copy compositing across adapters isn't supported"
    )]
    RegionSpansMultipleAdapters,

    /// A caller-supplied device does not belong to the captured output's adapter.
    #[error("the supplied D3D11 device belongs to a different adapter than the capture area")]
    DeviceAdapterMismatch,

    /// The capture device cannot be shared across a pipeline's threads.
    #[error(transparent)]
    SharedDevice(#[from] D3d11SharedDeviceError),

    /// Cursor composition was requested for a multi-output region.
    #[error("include_cursor isn't supported when CaptureArea::Region spans more than one output")]
    CursorUnsupportedForRegion,
}

/// How [`DxgiCaptureSource::open`] captures each frame — see
/// [`DxgiCaptureOptions::capture_mode`].
#[derive(Debug, Clone)]
pub enum CaptureMode {
    /// The original behavior: `AcquireNextFrame`'s resource is copied into
    /// a CPU-readable staging texture, `Map`ped, and copied row-by-row
    /// into a plain `Pixel::BGRA` CPU frame. No external device required —
    /// `open` creates its own, internal to this element, from the chosen
    /// output's own adapter.
    ///
    /// `include_cursor` composites the mouse cursor onto every emitted
    /// frame. Off by default: the base capture path (desktop pixels only)
    /// needs no extra work, and most consumers (recording, streaming a
    /// presentation) don't want the cursor baked in at all. Only exists on
    /// this variant — cursor compositing is CPU-side pixel blending
    /// (`composite_cursor`), which has nothing to run against under
    /// [`CaptureMode::Gpu`], where the captured image never touches the
    /// CPU at all; putting the field here instead of as a separate
    /// `DxgiCaptureOptions` flag makes that combination unrepresentable
    /// rather than a runtime error to guard against. Also unsupported
    /// (a hard `open`-time error, see
    /// [`DxgiCaptureSourceError::CursorUnsupportedForRegion`]) when
    /// [`DxgiCaptureOptions::area`] is a [`CaptureArea::Region`] spanning
    /// more than one output — see that variant's own docs on why.
    Cpu {
        /// Whether to blend the current mouse pointer into emitted CPU frames.
        include_cursor: bool,
    },
    /// Captures straight to a GPU-resident frame tagged `Pixel::D3D11`
    /// (BGRA — desktop content has no reason to go through YUV) — no
    /// `Map`, no CPU pixel copy at all, just GPU-side `CopyResource`/
    /// `CopySubresourceRegion` calls (each contributing output's
    /// duplication resource -> this element's own per-output "latest
    /// capture" texture, then those -> a fresh per-emission composite
    /// texture every tick, so an in-flight pushed frame's content can't
    /// change under whatever's still reading it — same reasoning
    /// [`crate::elements::D3d11Upload`] documents for building a fresh
    /// texture per call rather than reusing one).
    ///
    /// [`DxgiCaptureSource::open`] creates a device on the resolved adapter
    /// and returns it for downstream reuse. Alternatively,
    /// [`DxgiCaptureSource::open_with_device`] accepts an existing shared
    /// device and validates its adapter LUID before opening duplication.
    /// Either path establishes one exact `ID3D11Device` for every D3D11
    /// element sharing this capture.
    ///
    /// No cursor option — see [`CaptureMode::Cpu`]'s own docs on why.
    Gpu,
}

/// A capture region in absolute virtual-desktop pixel coordinates — the
/// same origin/units Win32 itself uses for multi-monitor layout
/// (`GetSystemMetrics(SM_XVIRTUALSCREEN)`, `MONITORINFOEX::rcMonitor`,
/// `DXGI_OUTPUT_DESC::DesktopCoordinates`), not local to any one monitor.
/// See [`CaptureArea::Region`].
#[derive(Debug, Clone, Copy)]
pub struct CaptureRect {
    /// Absolute virtual-desktop x coordinate of the left edge.
    pub x: i32,
    /// Absolute virtual-desktop y coordinate of the top edge.
    pub y: i32,
    /// Region width in pixels.
    pub width: u32,
    /// Region height in pixels.
    pub height: u32,
}

/// Which portion of the desktop [`DxgiCaptureSource::open`] duplicates —
/// see [`DxgiCaptureOptions::area`].
#[derive(Debug, Clone, Copy)]
pub enum CaptureArea {
    /// The `output_index`'th output's entire desktop — a flat index
    /// across every adapter's every output, in enumeration order
    /// (adapter 0's outputs, then adapter 1's, ...) — "monitor 0",
    /// "monitor 1", regardless of which GPU each is attached to. `0` is
    /// whatever Windows considers the first output of the first adapter,
    /// not necessarily the primary monitor.
    ///
    /// The simple case: exactly one `IDXGIOutputDuplication`, no
    /// cropping, no compositing.
    Output {
        /// Flat index across every adapter's outputs.
        output_index: u32,
    },
    /// An arbitrary rectangle in absolute virtual-desktop coordinates —
    /// for callers that only know "this screen area" (e.g. a
    /// user-dragged region-selection UI), not which monitor index owns
    /// it. May overlap more than one output: `open` resolves every
    /// output the rectangle intersects and opens one
    /// `IDXGIOutputDuplication` per contributing output, then stitches
    /// each output's contribution into one composite
    /// `rect.width x rect.height` image every capture tick —
    /// `CopySubresourceRegion` under [`CaptureMode::Gpu`], a plain
    /// per-row memory copy under [`CaptureMode::Cpu`] — placing each
    /// piece at its correct offset, no scaling or blending.
    ///
    /// **Every intersected output must share the same adapter.** Desktop
    /// Duplication resources can't be copied directly across
    /// `ID3D11Device`s from different adapters without a CPU round trip
    /// — `open` checks every intersected output's adapter *before*
    /// opening any duplication, so a rejected region never partially
    /// opens anything, and fails outright
    /// ([`DxgiCaptureSourceError::RegionSpansMultipleAdapters`]) rather
    /// than silently falling back to a CPU bridge for the mismatched
    /// output — same "hard, loud failure, never a silent auto-copy"
    /// reasoning as `D3d12Renderer`'s own device-mismatch guard.
    ///
    /// `include_cursor` (see [`CaptureMode::Cpu`]) is only valid when the
    /// region resolves to a single output —
    /// [`DxgiCaptureSourceError::CursorUnsupportedForRegion`] otherwise.
    /// The cursor can legitimately straddle a monitor boundary inside a
    /// stitched composite; handling that correctly isn't done, simplest
    /// to reject rather than silently draw it wrong.
    Region(CaptureRect),
}

/// Construction-time options for [`DxgiCaptureSource::open`].
#[derive(Debug, Clone)]
pub struct DxgiCaptureOptions {
    /// Which output(s) to capture from — see [`CaptureArea`].
    pub area: CaptureArea,
    /// The constant rate frames are emitted at — see [`DxgiCaptureSource`]'s
    /// own docs on why this is a fixed output rate (like
    /// [`crate::elements::TestVideoSource::new`]'s `framerate`), not a cap
    /// on an otherwise irregular one. `30` by default, matching
    /// `TestVideoSource`'s own default.
    pub fps: u32,
    /// CPU (the original behavior) or GPU (zero-copy) capture — see
    /// [`CaptureMode`]. `CaptureMode::Cpu { include_cursor: false }` by
    /// default, so existing callers building `DxgiCaptureOptions { ..
    /// ..Default::default() }` keep today's behavior unchanged.
    pub capture_mode: CaptureMode,
}

impl Default for DxgiCaptureOptions {
    fn default() -> Self {
        Self {
            area: CaptureArea::Output { output_index: 0 },
            fps: 30,
            capture_mode: CaptureMode::Cpu {
                include_cursor: false,
            },
        }
    }
}

/// One cached mouse cursor shape — refreshed only when
/// `DXGI_OUTDUPL_FRAME_INFO::PointerShapeBufferSize` says it changed
/// (the shape rarely changes frame-to-frame; re-fetching it on every
/// frame would be wasted work).
struct CursorShape {
    kind: u32,
    width: u32,
    height: u32,
    pitch: u32,
    data: Vec<u8>,
}

/// The cursor as one comparable value: everything
/// [`DxgiCaptureSource::copy_picture`] would draw of it.
///
/// The mouse moves independently of the desktop image, so a tick that
/// captured nothing can still owe a new picture. The shape is compared by a
/// counter bumped on every refresh rather than by its pixels — a shape is
/// replaced wholesale, never edited. Left at its default when
/// `include_cursor` is off, where none of it is drawn.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
struct CursorState {
    visible: bool,
    x: i32,
    y: i32,
    shape: u64,
}

/// One contributing output's own duplication plus this element's copy of
/// its capture, and where that output's portion belongs in the final
/// composite image. Exactly one of these exists under
/// [`CaptureArea::Output`]; [`CaptureArea::Region`] has one per output it
/// overlaps.
struct CaptureUnit {
    duplication: IDXGIOutputDuplication,
    /// This output's own full-resolution capture — CPU-readable
    /// (`D3D11_USAGE_STAGING`) under [`CaptureMode::Cpu`], GPU-only
    /// (`D3D11_USAGE_DEFAULT`, no bind flags) under [`CaptureMode::Gpu`]
    /// — only the fresh composite texture
    /// [`DxgiCaptureSource::emit_frame_gpu`] builds needs to be
    /// shader-bindable, not this one. Sized to this output's own
    /// resolution, not the final composite's — see `source_box` for the
    /// (possibly smaller) piece of it actually used.
    ///
    /// `None` where a frame goes straight into a composite instead: one
    /// output under [`CaptureMode::Gpu`], where a surface of this element's
    /// own would only be something to copy out of again.
    staging_texture: Option<ID3D11Texture2D>,
    /// The sub-rectangle of `staging_texture`, in this output's own
    /// local pixel coordinates, that actually falls inside the
    /// requested capture area. The whole texture under
    /// [`CaptureArea::Output`]; a crop under [`CaptureArea::Region`] —
    /// every pixel outside this box was requested by nobody, no reason
    /// to copy it into the composite.
    source_box: D3D11_BOX,
    /// Where `source_box`'s pixels land in the final composite image.
    dest_x: u32,
    dest_y: u32,
    /// Whether this unit has captured at least one real image yet — the
    /// element as a whole is ready to emit only once every unit's own
    /// flag is `true` (see [`DxgiCaptureSource::all_captured`]), so a
    /// freshly opened multi-output region never emits with part of the
    /// composite still blank.
    has_captured: bool,
}

/// Captures the desktop via Windows' DXGI Desktop Duplication API
/// (`IDXGIOutputDuplication`) — GStreamer's `d3d11screencapturesrc`
/// equivalent. One src pad, pushing `Pixel::BGRA` frames (no internal
/// color conversion — same division of labor as every other source in
/// this crate: chain a [`crate::elements::SwScaler`] downstream if
/// something needs YUV420P, e.g. `D3d12Renderer`'s
/// CPU-upload path or [`crate::elements::SwEncoder`]).
///
/// Emits at a **constant** rate — [`DxgiCaptureOptions::fps`] — not one
/// push per real desktop change. An earlier version of this pushed
/// variable-rate (VFR): a real wall-clock pts per actual change, nothing
/// in between. That turned out to cause real problems, both for muxing
/// (most consumers assume something closer to a steady rate) and,
/// concretely, for live rendering: `D3d12Renderer` presents on a
/// vsync-locked swap chain (`Present(1, ..)`) that only ever shows the
/// *latest* submitted frame each tick, silently dropping anything else
/// queued behind it — submission timing straight off an irregular VFR
/// source has no relationship to that vsync grid, so real changes would
/// unpredictably race into the same tick (one silently discarded) or
/// land in a gap (stale frame held an extra tick), which is visible
/// judder even though the *average* rate was exactly right.
///
/// So instead: this element always keeps the most recently captured
/// desktop image on hand, and [`SourceElement::run`]'s own loop emits it
/// — the same one again if nothing changed since the last tick — at a
/// steady `1 / fps` cadence, entirely on the one thread `run()` already
/// has (no extra threads spawned; see this crate's own "elements never
/// spawn their own threads" rule). Same shape as
/// [`crate::elements::TestVideoSource`]: [`DxgiCaptureSource::time_base`]
/// is `1 / fps` and `pts` is a plain incrementing tick counter, one per
/// *emitted* frame, not per real capture.
///
/// Confirmed (`examples/render/screen_preview_cpu`, with and without a
/// downstream [`crate::elements::Pacer`]) that this constant-rate,
/// drift-free schedule is what actually mattered — not whether a
/// separate `Pacer` stage exists. The VFR version needed one to paper
/// over its own irregular submission timing; once emission here is
/// steady and drift-free, `SwScaler`'s modest, fairly consistent per-frame
/// conversion cost isn't enough on its own to reintroduce the same vsync
/// misalignment, so a straight `DxgiCaptureSource -> SwScaler -> D3d12Renderer`
/// chain stays smooth with no `Pacer` at all. `Pacer` remains genuinely
/// useful for other reasons (multi-stream sync against a shared `Clock`,
/// or a stage with real per-frame variance like `SwEncoder`), just not
/// load-bearing here purely for vsync alignment the way it first
/// appeared to be.
///
/// Deliberately does **not** retry internally on `DXGI_ERROR_ACCESS_LOST`
/// (lock screen, UAC prompt, display mode change, ...) — same "fail fast,
/// caller rebuilds" contract as [`crate::elements::RtspSource`]; watch for
/// [`DxgiCaptureSourceError::AccessLost`] and call
/// [`DxgiCaptureSource::open`] again.
///
/// Runs until `Stop` — never reaches `Eos` on its own, same as
/// `TestVideoSource` (there's no natural end to a live desktop capture).
///
/// May capture from more than one output at once — see
/// [`CaptureArea::Region`] — in which case every field below that used
/// to describe "the" duplication instead describes one `CaptureUnit`
/// per contributing output.
pub struct DxgiCaptureSource {
    pp_log: PpLog,
    name: Arc<str>,
    /// Only used by [`CaptureMode::Gpu`]'s [`DxgiCaptureSource::emit_frame`]
    /// path, to build each tick's fresh per-emission composite texture —
    /// unused after construction in [`CaptureMode::Cpu`], but harmless to
    /// hold either way (one extra COM reference, same device already
    /// owns).
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    units: Vec<CaptureUnit>,
    /// The final composite image's dimensions — `rect.width`/`rect.height`
    /// under [`CaptureArea::Region`], the single output's own resolution
    /// under [`CaptureArea::Output`].
    width: u32,
    height: u32,
    gpu_mode: bool,
    include_cursor: bool,
    cursor_shape: Option<CursorShape>,
    /// The cursor's last known position/visibility — *not*
    /// `DXGI_OUTDUPL_FRAME_INFO::PointerPosition` read fresh every call.
    /// Per Microsoft's own Desktop Duplication sample, that field is only
    /// actually refreshed on a call where `LastMouseUpdateTime != 0` (the
    /// mouse itself changed on *this* call) — on any other call its
    /// contents aren't meaningful. Updated only when `LastMouseUpdateTime
    /// != 0` and composited fresh onto every emitted frame (independent
    /// of whether the desktop image itself changed that tick), so a
    /// moving cursor over an otherwise-static screen still shows up.
    cursor_position: POINT,
    cursor_visible: bool,
    /// The most recently captured composite desktop image, CPU-side —
    /// plain, not pool-backed (never shared/pushed directly downstream;
    /// see `run`'s own emit step, which copies out of this into a fresh
    /// pooled frame every tick). Each unit's own crop is written directly
    /// into its correct offset here as it's polled (see `poll_capture`),
    /// so this is always the up-to-date composite, not something
    /// assembled at emit time; re-copied from as-is on every tick where
    /// nothing new arrived, which is what makes this element emit at a
    /// constant rate rather than only on real changes. Only under
    /// [`CaptureMode::Cpu`] — `None` under [`CaptureMode::Gpu`], which
    /// composites straight from each unit's own `staging_texture` at
    /// emit time instead (see [`DxgiCaptureSource::emit_frame_gpu`]).
    staging: Option<ffmpeg::frame::Video>,
    /// See [`DxgiCaptureOptions::fps`] — kept alongside `frame_interval`
    /// so [`DxgiCaptureSource::time_base`] doesn't have to recover it from
    /// a `Duration`.
    fps: i32,
    /// `1 / fps`.
    frame_interval: Duration,
    /// This element's `pts` tick counter — one per *emitted* frame (see
    /// [`DxgiCaptureSource::time_base`]'s own docs), not per real capture.
    frame_index: i64,
    pad: SrcPad,
    /// Reused across every emitted frame — see [`UnboundObjectPool`]'s
    /// docs. Pre-sized to `width`/`height` up front, same reasoning as
    /// `SwScaler`'s own pool.
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
    /// [`CaptureMode::Cpu`]'s wrappers: empty frames pointed at
    /// `last_picture`, carrying only this tick's own timestamp. The same
    /// split [`CaptureMode::Gpu`] already has between `pool`'s wrappers and
    /// the composite texture inside them — with the picture in CPU memory,
    /// the wrapper is what lets several ticks in a row show it under their
    /// own `pts`. Unused under [`CaptureMode::Gpu`].
    wrapper_pool: UnboundObjectPool<ffmpeg::frame::Video>,
    /// The composite the last [`CaptureMode::Gpu`] emission drew, held as the
    /// frame that owns its texture: every emission is a reference to this, so
    /// a tick that captured nothing hands out another one, and the buffer's
    /// reference count is what says whether the texture may be drawn into
    /// again. `None` before the first emission and under [`CaptureMode::Cpu`].
    /// See [`DxgiCaptureSource::emit_frame_gpu`].
    composite: Option<ffmpeg::frame::Video>,
    /// Composites drawn earlier, kept rather than freed. A screen-sized
    /// texture is 8 MiB at 1080p, and a screen with real motion changes on
    /// every tick, so allocating one per changed frame is what this avoids —
    /// see [`DxgiCaptureSource::build_composite`], which takes one back only
    /// once nothing refers to it. Grows to however many emitted frames are in
    /// flight at once, and no further.
    spare_composites: Vec<ffmpeg::frame::Video>,
    /// [`CaptureMode::Cpu`]'s equivalent of `composite`: the pooled
    /// frame the desktop image was last copied into, kept so a tick that
    /// captured nothing can point a wrapper at it instead of copying
    /// identical pixels again. `None` before the first emission and under
    /// [`CaptureMode::Gpu`]. See [`DxgiCaptureSource::emit_frame_cpu`].
    last_picture: Option<Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>>,
    /// Pictures a wrapper already pushed downstream may still be pointing
    /// at. A wrapper shares the picture's *buffer*, not its pool slot, so a
    /// replaced picture waits here until nothing but the frame itself
    /// references its pixels — otherwise the next copy would write over what
    /// a frame still queued downstream is showing. See
    /// [`picture_is_referenced`].
    retired: Vec<Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>>,
    /// The cursor as it was drawn into `last_picture`, so a tick where only
    /// the mouse moved still copies a new picture. See [`CursorState`].
    picture_cursor: CursorState,
    /// Bumped every time `cursor_shape` is replaced — see [`CursorState`].
    cursor_shape_version: u64,
    /// Scratch for the per-frame change metadata DXGI writes — see
    /// [`DxgiCaptureSource::changed_regions`]. One buffer, grown to the
    /// largest frame seen and reused, rather than an allocation per frame on
    /// the path whose whole point is to stop copying per frame.
    metadata: Vec<u8>,
    /// Whether an acquired frame goes straight into a composite, with no
    /// surface of this element's own in between: one output under
    /// [`CaptureMode::Gpu`]. Settled at construction, since it is a property
    /// of the capture area and the mode — see [`DxgiCaptureSource::open`].
    direct: bool,
    /// Whether any unit has captured since the picture a repeat would offer
    /// again was built — the `composite` texture under
    /// [`CaptureMode::Gpu`], `last_picture`'s pixels under
    /// [`CaptureMode::Cpu`]. That is, whether producing it again would give
    /// different pixels.
    captured_since_picture: bool,
}

// SAFETY: every D3D11/DXGI handle here is a `windows-rs` COM interface
// wrapper — thread-safe to hand off (refcounting is interlocked), and
// `&mut self` on every method that touches them (mirrors `D3d12Decoder`/
// `SwScaler`'s own reasoning) already rules out concurrent access from
// multiple threads.
unsafe impl Send for DxgiCaptureSource {}

impl DxgiCaptureSource {
    /// Opens whichever output(s) [`DxgiCaptureOptions::area`] resolves to
    /// and starts duplicating them. Returns the element alongside the
    /// captured composite's actual [`VideoFormat`] — what the caller
    /// needs to build a matching downstream
    /// [`crate::elements::SwScaler`]/[`crate::elements::Pacer`], same
    /// pattern as [`crate::elements::RtspSource::open`] returning stream
    /// info — plus, under [`CaptureMode::Gpu`], the `ID3D11Device` this
    /// capture was opened on (`None` under [`CaptureMode::Cpu`], where
    /// nothing downstream needs to share it). This is always built from
    /// whichever adapter `area` actually resolves to — see
    /// [`CaptureMode::Gpu`]'s own docs on why callers should build every
    /// other D3D11 element sharing this capture from the returned device
    /// rather than a separately-created one.
    pub fn open(
        name: impl Into<String>,
        options: DxgiCaptureOptions,
    ) -> std::result::Result<(Self, VideoFormat, Option<ID3D11Device>), DxgiCaptureSourceError>
    {
        Self::open_impl(name, options, None)
    }

    /// Opens the capture on a caller-owned D3D11 device.
    ///
    /// The device must belong to the same DXGI adapter as every output touched
    /// by [`DxgiCaptureOptions::area`]; a mismatch is rejected before desktop
    /// duplication or textures are created. Under [`CaptureMode::Gpu`], this
    /// lets capture, filters, compositors, encoders, and renderers share one
    /// device without a system-memory round trip. Device injection does not
    /// remove the existing GPU copies: duplication surfaces first refresh the
    /// internal latest-image textures, then each emission gets an independent
    /// composite texture that downstream may retain after `ReleaseFrame`.
    /// A `D3D11_CREATE_DEVICE_SINGLETHREADED` device is rejected; otherwise
    /// this enables its immediate context's runtime multithread protection.
    pub fn open_with_device(
        name: impl Into<String>,
        options: DxgiCaptureOptions,
        device: &ID3D11Device,
    ) -> std::result::Result<(Self, VideoFormat), DxgiCaptureSourceError> {
        let (source, format, _returned_device) = Self::open_impl(name, options, Some(device))?;
        Ok((source, format))
    }

    fn open_impl(
        name: impl Into<String>,
        options: DxgiCaptureOptions,
        supplied_device: Option<&ID3D11Device>,
    ) -> std::result::Result<(Self, VideoFormat, Option<ID3D11Device>), DxgiCaptureSourceError>
    {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::DxgiCaptureSource, &name, None);

        // SAFETY: creates the documented DXGI factory interface without
        // borrowing caller-owned storage.
        let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1() }?;
        let gpu_mode = matches!(options.capture_mode, CaptureMode::Gpu);
        // `CaptureMode::Gpu` has no `include_cursor` field at all (see its
        // own docs) — nothing to extract there, so `false` unconditionally.
        let include_cursor = match &options.capture_mode {
            CaptureMode::Cpu { include_cursor } => *include_cursor,
            CaptureMode::Gpu => false,
        };

        let (targets, requested) = resolve_area(&factory, &options.area)?;
        if include_cursor && targets.len() > 1 {
            return Err(DxgiCaptureSourceError::CursorUnsupportedForRegion);
        }
        let width = (requested.right - requested.left) as u32;
        let height = (requested.bottom - requested.top) as u32;

        // Every target shares one adapter (`resolve_area` already checked).
        // Either create the device there or validate the caller's device
        // before opening any duplication.
        let adapter = &targets[0].0;
        let device = if let Some(device) = supplied_device {
            validate_device_adapter(device, adapter)?;
            device.clone()
        } else {
            let mut device: Option<ID3D11Device> = None;
            let mut context: Option<ID3D11DeviceContext> = None;
            // SAFETY: the selected live adapter is passed with UNKNOWN driver
            // type as required, optional feature levels use D3D defaults, and
            // device/context are live correctly typed out-parameters.
            unsafe {
                D3D11CreateDevice(
                    &adapter.cast::<windows::Win32::Graphics::Dxgi::IDXGIAdapter>()?,
                    D3D_DRIVER_TYPE_UNKNOWN,
                    HMODULE::default(),
                    D3D11_CREATE_DEVICE_FLAG(0),
                    None,
                    D3D11_SDK_VERSION,
                    Some(&mut device),
                    None,
                    Some(&mut context),
                )?;
            }
            device.expect("D3D11CreateDevice succeeded without producing a device")
        };
        let context = protect_shared_device(&device)?;
        // Cloned before `device` moves into `Self` below — the only copy
        // handed back to the caller (a COM ref-count bump, not a deep
        // copy).
        let returned_device = gpu_mode.then(|| device.clone());
        let dxgi_device: IDXGIDevice = device.cast()?;

        // One output under [`CaptureMode::Gpu`] is the case where an acquired
        // frame can go straight into a composite, with no surface of this
        // element's own in between — see [`DxgiCaptureSource::poll_capture`].
        // Several outputs cannot: a composite is only complete once every one
        // of them has contributed, so each keeps its own latest image to be
        // assembled from.
        let direct = gpu_mode && targets.len() == 1;

        let mut units = Vec::with_capacity(targets.len());
        for (_, output, desktop_rect) in &targets {
            // SAFETY: output and DXGI device are live and originate from the
            // same adapter established by `resolve_area`.
            let duplication = unsafe { output.DuplicateOutput(&dxgi_device) }?;
            // SAFETY: the duplication interface is live and returns its plain
            // immutable description by value.
            let desc = unsafe { duplication.GetDesc() };
            let unit_width = desc.ModeDesc.Width;
            let unit_height = desc.ModeDesc.Height;

            // The overlap between this output's own desktop rectangle
            // and the requested one, expressed two ways: as a source box
            // local to this output's own texture, and as a destination
            // offset into the composite. `resolve_area` already
            // guarantees a non-empty overlap for every target it
            // returns.
            let overlap_left = desktop_rect.left.max(requested.left);
            let overlap_top = desktop_rect.top.max(requested.top);
            let overlap_right = desktop_rect.right.min(requested.right);
            let overlap_bottom = desktop_rect.bottom.min(requested.bottom);
            let source_box = D3D11_BOX {
                left: (overlap_left - desktop_rect.left) as u32,
                top: (overlap_top - desktop_rect.top) as u32,
                front: 0,
                right: (overlap_right - desktop_rect.left) as u32,
                bottom: (overlap_bottom - desktop_rect.top) as u32,
                back: 1,
            };
            let dest_x = (overlap_left - requested.left) as u32;
            let dest_y = (overlap_top - requested.top) as u32;

            // Cpu: CPU-readable staging texture, `Map`ped every real
            // capture (see `poll_capture`). Gpu: GPU-only — this
            // per-output texture is only ever a `CopySubresourceRegion`
            // source, never sampled directly, so unlike the composite
            // `emit_frame_gpu` builds, no bind flags are needed here.
            let staging_desc = D3D11_TEXTURE2D_DESC {
                Width: unit_width,
                Height: unit_height,
                MipLevels: 1,
                ArraySize: 1,
                Format: DXGI_FORMAT_B8G8R8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Usage: if gpu_mode {
                    D3D11_USAGE_DEFAULT
                } else {
                    D3D11_USAGE_STAGING
                },
                BindFlags: D3D11_BIND_FLAG(0).0 as u32,
                CPUAccessFlags: if gpu_mode {
                    0
                } else {
                    D3D11_CPU_ACCESS_READ.0 as u32
                },
                MiscFlags: 0,
            };
            let staging_texture = if direct {
                // The frame goes into a composite instead, so this surface
                // would only ever be copied out of into that same composite.
                None
            } else {
                let mut staging_texture: Option<ID3D11Texture2D> = None;
                // SAFETY: `staging_desc` is fully initialized for the selected
                // mode, no initial data is supplied, and the output slot is live.
                unsafe { device.CreateTexture2D(&staging_desc, None, Some(&mut staging_texture)) }?;
                Some(
                    staging_texture.expect("CreateTexture2D succeeded without producing a texture"),
                )
            };

            units.push(CaptureUnit {
                duplication,
                staging_texture,
                source_box,
                dest_x,
                dest_y,
                has_captured: false,
            });
        }

        // Which domain this emits in is settled by `capture_mode` right
        // here, so downstream can be checked against it even though the
        // captured size and format are runtime values.
        let pad = SrcPad::with_contract(
            format!("{name}_src"),
            OutputContract::Fixed(PortContract::frame(
                MediaKind::VideoFrame,
                if gpu_mode {
                    MemoryDomain::D3d11
                } else {
                    MemoryDomain::System
                },
            )),
        );
        // Gpu: only the small CPU-side `AVFrame` wrapper is ever pooled
        // (`ffmpeg::frame::Video::empty` — same as `D3d11Upload`'s own
        // pool); the GPU texture itself is a fresh allocation every
        // `emit_frame` call (see that method's own docs on why). Cpu:
        // pre-sized real `Pixel::BGRA` CPU buffers, as before.
        let pool = if gpu_mode {
            // Wrappers, so a returned one lets go of the composite it was
            // showing rather than pinning that texture until its next use.
            UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, release_picture)
        } else {
            UnboundObjectPool::new(
                0,
                move || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, width, height),
                |_| {},
            )
        };
        let staging = (!gpu_mode)
            .then(|| ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, width, height));

        let fps = options.fps.max(1); // a `0` fps is nonsensical; treat it as 1 rather than dividing by zero
        // Same `1 / fps` convention as `DxgiCaptureSource::time_base` — computed
        // here too since `Self` is moved into the tuple below before that method
        // could be called on it.
        let time_base = ffmpeg::Rational::new(1, fps as i32);
        pp_info!(
            pp_log: &pp_log,
            "opened: {}x{} composite from {} output(s), include_cursor={}, fps={}, gpu_mode={}",
            width,
            height,
            units.len(),
            include_cursor,
            fps,
            gpu_mode
        );

        Ok((
            Self {
                name,
                pp_log,
                device,
                context,
                units,
                width,
                height,
                gpu_mode,
                include_cursor,
                cursor_shape: None,
                cursor_position: POINT::default(),
                cursor_visible: false,
                staging,
                fps: fps as i32,
                frame_interval: Duration::from_secs_f64(1.0 / fps as f64),
                frame_index: 0,
                pad,
                pool,
                wrapper_pool: UnboundObjectPool::new(
                    0,
                    ffmpeg::frame::Video::empty,
                    release_picture,
                ),
                composite: None,
                spare_composites: Vec::new(),
                last_picture: None,
                retired: Vec::new(),
                picture_cursor: CursorState::default(),
                cursor_shape_version: 0,
                metadata: Vec::new(),
                direct,
                captured_since_picture: false,
            },
            VideoFormat {
                width,
                height,
                time_base,
            },
            returned_device,
        ))
    }

    /// The unit each emitted frame's `pts` is expressed in — what you
    /// need to construct a matching [`crate::elements::Pacer`]. `1 /
    /// fps`, same convention as [`crate::elements::TestVideoSource::time_base`].
    pub fn time_base(&self) -> ffmpeg::Rational {
        ffmpeg::Rational::new(1, self.fps)
    }

    /// Whether every contributing output has captured at least one real
    /// image yet — see [`CaptureUnit::has_captured`]'s own docs.
    fn all_captured(&self) -> bool {
        self.units.iter().all(|unit| unit.has_captured)
    }

    /// Which parts of `unit_index`'s output changed in the frame just
    /// acquired, in that output's own pixel coordinates — or `None` for "copy
    /// all of it", which is what a frame with no usable metadata means.
    ///
    /// Desktop Duplication reports this per frame, and most of what a desktop
    /// does is small: a caret, a clock, a menu opening. Copying the whole
    /// surface for it is 8 MiB at 1080p where the change is a few hundred
    /// kilobytes, and that copy is the largest per-frame cost this element
    /// has on a screen that *is* changing — the one case none of the repeat
    /// handling can help with.
    ///
    /// Two lists describe a frame. The dirty rectangles are regions drawn
    /// afresh. The move rectangles are regions the compositor blitted from
    /// somewhere else in the previous frame — a scrolling window. Both are
    /// answered here by copying that region out of the *acquired* texture,
    /// which already holds the finished desktop image: a move is only an
    /// optimization DXGI offers, never a region it left unwritten, so the
    /// destination of one is exactly as copyable as a dirty rectangle. That
    /// is what keeps this from needing D3D11's forbidden overlapping
    /// same-resource copy, or a scratch texture to route one through.
    fn changed_regions(
        &mut self,
        unit_index: usize,
        info: &DXGI_OUTDUPL_FRAME_INFO,
    ) -> Option<Vec<RECT>> {
        let size = info.TotalMetadataBufferSize as usize;
        if size == 0 || !self.units[unit_index].has_captured {
            // No metadata to read, or nothing to add it to: the first frame
            // of a duplication has to fill the whole surface, whatever the
            // metadata says changed since a frame this element never saw.
            return None;
        }
        if self.metadata.len() < size {
            self.metadata.resize(size, 0);
        }
        let duplication = self.units[unit_index].duplication.clone();

        let mut moves_bytes = 0u32;
        // SAFETY: the buffer is at least `TotalMetadataBufferSize` long and
        // correctly aligned for the move-rect array DXGI writes into it, and
        // `moves_bytes` is a live out-parameter.
        let moves = unsafe {
            duplication.GetFrameMoveRects(
                self.metadata.len() as u32,
                self.metadata.as_mut_ptr().cast::<DXGI_OUTDUPL_MOVE_RECT>(),
                &mut moves_bytes,
            )
        };
        if moves.is_err() {
            return None;
        }
        let move_count = moves_bytes as usize / size_of::<DXGI_OUTDUPL_MOVE_RECT>();
        // SAFETY: DXGI wrote `moves_bytes` of move rectangles into the buffer
        // above, so that many entries are initialized and owned by this call.
        let moved: Vec<RECT> = unsafe {
            std::slice::from_raw_parts(
                self.metadata.as_ptr().cast::<DXGI_OUTDUPL_MOVE_RECT>(),
                move_count,
            )
        }
        .iter()
        .map(|rect| rect.DestinationRect)
        .collect();

        // The dirty rectangles go after the move ones, which is what the
        // single `TotalMetadataBufferSize` covers.
        let dirty_offset = moves_bytes as usize;
        let mut dirty_bytes = 0u32;
        // SAFETY: the remaining buffer is what DXGI asked for, and
        // `dirty_bytes` is a live out-parameter.
        let dirty = unsafe {
            duplication.GetFrameDirtyRects(
                (self.metadata.len() - dirty_offset) as u32,
                self.metadata.as_mut_ptr().add(dirty_offset).cast::<RECT>(),
                &mut dirty_bytes,
            )
        };
        if dirty.is_err() {
            return None;
        }
        let dirty_count = dirty_bytes as usize / size_of::<RECT>();
        // SAFETY: as above, for the dirty-rectangle half of the same buffer.
        let dirty = unsafe {
            std::slice::from_raw_parts(
                self.metadata.as_ptr().add(dirty_offset).cast::<RECT>(),
                dirty_count,
            )
        };

        // Clipped to the piece of this output the caller actually asked for:
        // nothing outside `source_box` is ever composited, so copying it
        // would be work for pixels no frame will carry.
        let source_box = self.units[unit_index].source_box;
        let regions: Vec<RECT> = moved
            .iter()
            .chain(dirty)
            .filter_map(|rect| clip_to_box(*rect, &source_box))
            .collect();
        // Nothing usable to copy — a frame that only moved the cursor is
        // already handled before this is reached, so this means metadata that
        // described nothing inside the captured area.
        if regions.is_empty() {
            return None;
        }
        // A change that covers most of the surface is cheaper to answer with
        // the whole surface. Copying a sub-rectangle out of the acquired
        // desktop texture costs measurably more per pixel than copying all of
        // it — on this hardware, a screen-sized window repainting at 30 fps
        // measured about four points of a core more through the rectangles
        // than through the whole-resource copy, for pixels it would have
        // copied anyway. Below the threshold that per-pixel premium is what
        // buys the saving; above it, it is all there is.
        let area: i64 = regions
            .iter()
            .map(|rect| i64::from(rect.right - rect.left) * i64::from(rect.bottom - rect.top))
            .sum();
        let captured = i64::from(source_box.right - source_box.left)
            * i64::from(source_box.bottom - source_box.top);
        (area * 4 < captured * 3).then_some(regions)
    }

    /// Refreshes `self.cursor_shape` from `unit_index`'s duplication
    /// interface's current pointer shape buffer. Only called when
    /// `DXGI_OUTDUPL_FRAME_INFO::PointerShapeBufferSize > 0` — i.e. the
    /// shape actually changed since the last call. Only ever called with
    /// `unit_index == 0`: `open` rejects `include_cursor` whenever more
    /// than one unit exists (see [`CaptureArea::Region`]'s own docs).
    fn refresh_cursor_shape(
        &mut self,
        unit_index: usize,
        buffer_size: usize,
    ) -> std::result::Result<(), windows::core::Error> {
        let mut buffer = vec![0u8; buffer_size];
        let mut required = 0u32;
        let mut info = DXGI_OUTDUPL_POINTER_SHAPE_INFO::default();
        // SAFETY: `unit_index` names a live duplication, `buffer` is writable
        // for the advertised shape size, and `required`/`info` are live
        // out-parameters. Success guarantees `required <= buffer.len()`.
        unsafe {
            self.units[unit_index].duplication.GetFramePointerShape(
                buffer.len() as u32,
                buffer.as_mut_ptr() as *mut c_void,
                &mut required,
                &mut info,
            )?;
        }
        buffer.truncate(required as usize);
        self.cursor_shape = Some(CursorShape {
            kind: info.Type,
            width: info.Width,
            height: info.Height,
            pitch: info.Pitch,
            data: buffer,
        });
        // A different shape from here on — see [`CursorState`].
        self.cursor_shape_version += 1;
        Ok(())
    }

    /// Tries once to capture a new image from every contributing output,
    /// within `timeout_ms` total — Desktop Duplication has no "wait on
    /// any of these" primitive, so `timeout_ms` is split evenly across
    /// `self.units` (more outputs means a longer total `poll_capture`
    /// call for the same per-unit responsiveness, still bounded overall
    /// by `POLL_GRANULARITY` same as the single-output case always was).
    /// Under [`CaptureMode::Cpu`], each unit that captured a new image
    /// writes its own crop directly into `self.staging` at its own
    /// composite offset. Always refreshes the cached cursor
    /// position/shape (see their own docs) regardless, since the mouse
    /// can move independently of the desktop image. A
    /// `DXGI_ERROR_WAIT_TIMEOUT` on any one unit (nothing changed within
    /// its share of `timeout_ms`) is not an error — that unit is simply
    /// unchanged this call.
    fn poll_capture(&mut self, timeout_ms: u32) -> std::result::Result<(), DxgiCaptureSourceError> {
        let per_unit_timeout = timeout_ms / self.units.len() as u32;
        for index in 0..self.units.len() {
            let mut info = DXGI_OUTDUPL_FRAME_INFO::default();
            let mut resource: Option<IDXGIResource> = None;
            // SAFETY: this live duplication has no outstanding acquired frame;
            // `info` and `resource` are correctly typed live out-parameters.
            let acquire = unsafe {
                self.units[index].duplication.AcquireNextFrame(
                    per_unit_timeout,
                    &mut info,
                    &mut resource,
                )
            };
            let resource = match acquire {
                Ok(()) => resource.expect("AcquireNextFrame succeeded without a resource"),
                Err(error) if error.code() == DXGI_ERROR_WAIT_TIMEOUT => continue,
                Err(error) if error.code() == DXGI_ERROR_ACCESS_LOST => {
                    return Err(DxgiCaptureSourceError::AccessLost);
                }
                Err(error) => return Err(error.into()),
            };

            if self.include_cursor {
                if info.PointerShapeBufferSize > 0 {
                    self.refresh_cursor_shape(index, info.PointerShapeBufferSize as usize)?;
                }
                // See `cursor_position`/`cursor_visible`'s own docs: only
                // trust `info.PointerPosition` on the call where the
                // mouse itself actually changed.
                if info.LastMouseUpdateTime != 0 {
                    self.cursor_position = info.PointerPosition.Position;
                    self.cursor_visible = info.PointerPosition.Visible.as_bool();
                }
            }

            // `AcquireNextFrame` succeeds not just when the desktop image
            // itself changed, but also on a *cursor-only* update (the
            // pointer moved/blinked with the picture underneath it
            // untouched) — `AccumulatedFrames == 0` is how DXGI signals
            // that case (see Microsoft's own Desktop Duplication sample).
            // The cursor position was already refreshed above
            // regardless; there's just no new *image* to copy out, so
            // release and move to the next unit.
            if info.AccumulatedFrames == 0 {
                // SAFETY: balances this unit's successful `AcquireNextFrame`;
                // no borrowed resource access remains.
                unsafe { self.units[index].duplication.ReleaseFrame() }?;
                continue;
            }

            // Every fallible step here (the two `cast`s, the copy itself)
            // must still release DXGI's own frame lock on the way out —
            // an early `?` before `ReleaseFrame()` would leave this unit
            // unable to `AcquireNextFrame` again until it's torn down
            // entirely, so the copy's own result is captured instead of
            // propagated directly.
            // What actually moved on this output, so the copy below is the
            // size of the change rather than the size of the screen — see
            // `changed_regions`.
            let mut changed = self.changed_regions(index, &info);
            // Where this frame lands. With one output on the GPU path that is
            // a composite nothing is reading, which is the whole of the work:
            // the frame is never copied again on its way out. A composite that
            // is not the one drawn into last is missing everything that
            // happened before this frame, so it takes the whole picture rather
            // than what changed since.
            let target = if self.direct {
                let (composite, current) = self.composite_to_draw_into()?;
                if !current {
                    changed = None;
                }
                Some(composite)
            } else {
                None
            };
            let copy_result: std::result::Result<(), DxgiCaptureSourceError> = (|| {
                let texture: ID3D11Texture2D = resource.cast()?;
                let source: ID3D11Resource = texture.cast()?;
                let destination: ID3D11Resource = match &target {
                    Some(composite) => composite_resource(composite)?,
                    None => self.units[index]
                        .staging_texture
                        .as_ref()
                        .expect("a unit without a composite of its own has its own surface")
                        .cast::<ID3D11Resource>()?,
                };
                let unit = &self.units[index];
                // A unit's own surface mirrors its whole output, so a region
                // lands where it came from. A composite holds only the piece
                // that was asked for, at the offset that piece occupies in
                // the capture area — the same mapping `source_box` and
                // `dest_x`/`dest_y` describe for the assembled path.
                let destination_of = |left: i32, top: i32| -> (u32, u32) {
                    match &target {
                        Some(_) => (
                            unit.dest_x + (left as u32 - unit.source_box.left),
                            unit.dest_y + (top as u32 - unit.source_box.top),
                        ),
                        None => (left as u32, top as u32),
                    }
                };
                match &changed {
                    None => match &target {
                        // The composite is the capture area and the acquired
                        // texture is the whole output; those are the same
                        // extent only when the area is the whole output, so
                        // the crop is taken explicitly.
                        // SAFETY: source and destination belong to this
                        // context, and `source_box` is the intersection
                        // `open` computed, so it and its destination offset
                        // are inside both textures.
                        Some(_) => unsafe {
                            let (x, y) = destination_of(
                                unit.source_box.left as i32,
                                unit.source_box.top as i32,
                            );
                            self.context.CopySubresourceRegion(
                                &destination,
                                0,
                                x,
                                y,
                                0,
                                &source,
                                0,
                                Some(&unit.source_box as *const D3D11_BOX),
                            );
                        },
                        // SAFETY: acquired source and this unit's own surface
                        // belong to this context and have identical dimensions
                        // and format; the immediate context is source-thread
                        // confined.
                        None => unsafe { self.context.CopyResource(&destination, &source) },
                    },
                    Some(regions) => {
                        for region in regions {
                            let box_ = D3D11_BOX {
                                left: region.left as u32,
                                top: region.top as u32,
                                front: 0,
                                right: region.right as u32,
                                bottom: region.bottom as u32,
                                back: 1,
                            };
                            let (x, y) = destination_of(region.left, region.top);
                            // SAFETY: as above, and every region was clipped
                            // to `source_box` by `changed_regions`, so the box
                            // and its destination offset are both inside the
                            // two textures.
                            unsafe {
                                self.context.CopySubresourceRegion(
                                    &destination,
                                    0,
                                    x,
                                    y,
                                    0,
                                    &source,
                                    0,
                                    Some(&box_ as *const D3D11_BOX),
                                );
                            }
                        }
                    }
                }
                Ok(())
            })();

            // Release DXGI's own frame as soon as we've copied it out,
            // rather than holding it while we map/read (Cpu mode) the
            // (independent) staging copy below — and unconditionally,
            // even if the copy above failed.
            // SAFETY: balances the successful acquisition above after the
            // desktop image has been copied and no acquired-resource use remains.
            let release_result = unsafe { self.units[index].duplication.ReleaseFrame() };
            if let Some(composite) = target {
                match &copy_result {
                    // The composite now holds this frame, so emitting is only
                    // wrapping it — there is nothing left to build.
                    Ok(()) => {
                        self.composite = Some(composite);
                        self.captured_since_picture = false;
                    }
                    // A composite half-drawn or not drawn at all: back among
                    // the spares, and the one being shown stays the one being
                    // shown.
                    Err(_) => self.spare_composites.push(composite),
                }
            }
            copy_result?;
            release_result?;

            if self.gpu_mode {
                // No `Map`/CPU copy at all — what this unit captured is
                // already where `emit_frame_gpu` reads it from, either the
                // composite above or this unit's own surface. See
                // `CaptureMode::Gpu`'s own docs.
                self.units[index].has_captured = true;
                if !self.direct {
                    // This unit's latest image changed, so the next composite
                    // is no longer the one `composite` already holds.
                    self.captured_since_picture = true;
                }
                continue;
            }
            // Cpu: the same, for the picture `last_picture` already holds —
            // set before the copy below rather than after it, so a failure
            // part way through cannot leave a half-written `staging` looking
            // like the picture that was already emitted.
            self.captured_since_picture = true;

            // Cpu mode: Map this unit's own staging texture and copy
            // just its `source_box` crop into the shared composite
            // buffer at `dest_x`/`dest_y` — no separate composite GPU
            // texture needed, the crop lands directly in CPU memory at
            // its final position.
            let mut mapped = Default::default();
            // SAFETY: CPU mode created this live staging texture with READ
            // access; `mapped` is a live out-parameter and no earlier map is
            // outstanding for it.
            unsafe {
                self.context.Map(
                    &staging_of(&self.units[index])?,
                    0,
                    D3D11_MAP_READ,
                    0,
                    Some(&mut mapped),
                )?;
            }
            {
                let unit = &self.units[index];
                let box_ = unit.source_box;
                let crop_width = (box_.right - box_.left) as usize;
                let crop_height = (box_.bottom - box_.top) as usize;
                let row_bytes = crop_width * 4;
                let staging = self
                    .staging
                    .as_mut()
                    .expect("CaptureMode::Cpu always has a staging buffer");
                let dst_stride = staging.stride(0);
                let dst = staging.data_mut(0);
                for row in 0..crop_height {
                    let src_row = box_.top as usize + row;
                    // SAFETY: the mapped pointer remains valid until `Unmap`;
                    // `source_box` is the output intersection, so row/column
                    // offsets and `row_bytes` stay within RowPitch and height.
                    let src = unsafe {
                        std::slice::from_raw_parts(
                            (mapped.pData as *const u8)
                                .add(src_row * mapped.RowPitch as usize + box_.left as usize * 4),
                            row_bytes,
                        )
                    };
                    let dst_row = unit.dest_y as usize + row;
                    let dst_col = unit.dest_x as usize * 4;
                    dst[dst_row * dst_stride + dst_col..dst_row * dst_stride + dst_col + row_bytes]
                        .copy_from_slice(src);
                }
            }
            // SAFETY: balances the successful map above after all slices made
            // from `mapped.pData` have gone out of scope.
            unsafe {
                self.context.Unmap(&staging_of(&self.units[index])?, 0);
            }
            self.units[index].has_captured = true;
        }
        Ok(())
    }

    /// Builds the next frame to push — the unit of work `run` does once
    /// per emission tick, real change or repeat — and stamps the next
    /// `pts` and this source's fixed color description. Dispatches to
    /// [`DxgiCaptureSource::emit_frame_cpu`] or
    /// [`DxgiCaptureSource::emit_frame_gpu`] depending on `self.gpu_mode`.
    ///
    /// Both modes produce BGRA, so both get the same description, and it
    /// matches what [`crate::elements::D3d11VideoCompositor`] already stamps
    /// on its own BGRA output: `Space::RGB` because these are RGB samples
    /// with no luma/chroma matrix applied, and `Range::JPEG` because desktop
    /// pixels are full-range 0-255, not studio-swing.
    ///
    /// Downstream consumers read this rather than guessing:
    /// `D3d11VideoCompositor` uses it to pick its NV12 conversion matrix,
    /// and [`crate::elements::D3d11NvencEncoder`] forwards it. Note it does
    /// **not** change what a BGRA-input NVENC recording is tagged with —
    /// NVENC converts RGB to YUV inside its own encode block with a fixed
    /// matrix and tags the bitstream to match what it actually did, which
    /// is why that path stays self-consistent either way.
    fn emit_frame(
        &mut self,
    ) -> std::result::Result<
        crate::pool::UnboundObjectPoolRef<ffmpeg::frame::Video>,
        DxgiCaptureSourceError,
    > {
        let mut frame = if self.gpu_mode {
            self.emit_frame_gpu()?
        } else {
            self.emit_frame_cpu()?
        };
        frame.set_pts(Some(self.frame_index));
        frame.set_color_space(ffmpeg::color::Space::RGB);
        frame.set_color_range(ffmpeg::color::Range::JPEG);
        self.frame_index += 1;
        Ok(frame)
    }

    /// Emits one wrapper around the picture `self.staging` was last copied
    /// into, copying a new one first when this tick has something new to
    /// show. The wrapper is what carries this tick's own `pts` (stamped by
    /// the caller, `emit_frame`): `self.staging` itself is written in place
    /// by `poll_capture` and an `Arc`-shared frame cannot have its `pts`
    /// rewritten once downstream may hold a clone, so what is shared is a
    /// picture nothing writes to again.
    ///
    /// # Why a tick with nothing new copies nothing
    ///
    /// The same reason [`DxgiCaptureSource::emit_frame_gpu`] builds nothing:
    /// this element emits at a constant rate rather than only when the
    /// desktop changes, so most ticks on a mostly-still screen would copy
    /// the full picture — 8 MiB per tick at 1080p, around 480 MB/s at 60 fps
    /// — to produce pixels identical to the ones just emitted. Such a tick
    /// points another wrapper at the picture already copied instead. What
    /// downstream sees does not change: a frame every tick, each with its
    /// own wrapper and its own advancing `pts`.
    ///
    /// The cursor is part of "something new" here, not just the desktop
    /// image: it is drawn into the picture, and it moves independently — see
    /// [`CursorState`].
    fn emit_frame_cpu(
        &mut self,
    ) -> std::result::Result<UnboundObjectPoolRef<ffmpeg::frame::Video>, DxgiCaptureSourceError>
    {
        let cursor = self.cursor_state();
        if self.last_picture.is_none()
            || self.captured_since_picture
            || cursor != self.picture_cursor
        {
            self.copy_picture(cursor);
        }

        let mut wrapper = self.wrapper_pool.get();
        let picture = self
            .last_picture
            .as_ref()
            .expect("copy_picture leaves a picture behind");
        // SAFETY: `ptr` is the pooled wrapper's own `AVFrame`, unreferenced
        // before it is given a new one, and the source is the picture this
        // element holds the pooled reference to — both live, and distinct
        // from each other.
        unsafe {
            let ptr = wrapper.as_mut_ptr();
            ffi::av_frame_unref(ptr);
            let code = ffi::av_frame_ref(ptr, picture.as_ptr());
            if code < 0 {
                return Err(DxgiCaptureSourceError::FrameRef(code));
            }
        }
        Ok(wrapper)
    }

    /// The cursor as this tick would draw it — see [`CursorState`].
    fn cursor_state(&self) -> CursorState {
        if !self.include_cursor {
            return CursorState::default();
        }
        CursorState {
            visible: self.cursor_visible && self.cursor_shape.is_some(),
            x: self.cursor_position.x,
            y: self.cursor_position.y,
            shape: self.cursor_shape_version,
        }
    }

    /// Copies `self.staging` (the latest captured composite image, however
    /// stale — already assembled from every unit's own crop by
    /// `poll_capture`) into a pooled frame, compositing the cursor onto that
    /// copy if enabled, and makes it the picture emitted from here on.
    ///
    /// Into a frame the pool considers free, never over the previous
    /// picture: that one is still what earlier wrappers are showing.
    fn copy_picture(&mut self, cursor: CursorState) {
        // The picture being replaced may still be under a wrapper pushed
        // downstream, so it waits in `retired` rather than going straight
        // back to the pool — see that field's own docs.
        if let Some(previous) = self.last_picture.take() {
            self.retired.push(previous);
        }
        self.retired
            .retain(|picture| picture_is_referenced(picture));

        let mut frame = self.pool.get();
        let staging = self
            .staging
            .as_ref()
            .expect("CaptureMode::Cpu always has a staging buffer");
        {
            let dst_stride = frame.stride(0);
            let src_stride = staging.stride(0);
            let row_bytes = self.width as usize * 4;
            let src = staging.data(0);
            let dst = frame.data_mut(0);
            for row in 0..self.height as usize {
                dst[row * dst_stride..row * dst_stride + row_bytes]
                    .copy_from_slice(&src[row * src_stride..row * src_stride + row_bytes]);
            }
            if cursor.visible
                && let Some(shape) = &self.cursor_shape
            {
                composite_cursor(
                    dst,
                    dst_stride,
                    self.width,
                    self.height,
                    cursor.x,
                    cursor.y,
                    shape,
                );
            }
        }
        self.last_picture = Some(Arc::new(frame));
        self.picture_cursor = cursor;
        self.captured_since_picture = false;
    }

    /// `CaptureMode::Gpu`'s equivalent of [`DxgiCaptureSource::emit_frame_cpu`]:
    /// builds a composite `ID3D11Texture2D` (`self.width` x `self.height`) and
    /// `CopySubresourceRegion`s every unit's own `source_box` crop into it at
    /// that unit's `dest_x`/`dest_y` — one GPU-side copy per contributing
    /// output. Wraps it as a `Pixel::D3D11` frame via [`wrap_d3d11_texture`],
    /// reusing the pooled `AVFrame` wrapper (the pool built by `open` for
    /// [`CaptureMode::Gpu`] only holds these small wrappers, not GPU memory —
    /// see that pool's own construction site).
    ///
    /// # Why a tick that captured nothing builds nothing
    ///
    /// This element emits at a constant rate rather than only when the desktop
    /// changes, so most ticks on a mostly-still screen find every unit's
    /// `staging_texture` exactly as the last one left it. Rebuilding then costs
    /// a fresh full-size texture and a full-size copy to produce an image
    /// pixel-identical to the one just emitted — at 1080p60 that is around
    /// 480 MB/s of allocation for no new pixels. Such a tick re-wraps
    /// the composite it already has instead. The rate downstream sees does not change: a
    /// frame is still pushed every tick, with its own wrapper and its own
    /// advancing `pts`.
    ///
    /// Handing two frames the same texture is sound because a composite is
    /// written only while nothing is reading it: a tick that *did* capture
    /// something draws into one no emitted frame still refers to, and
    /// allocates another when every composite it holds is in flight. So a
    /// frame still held downstream can never see its pixels change — the
    /// property [`crate::pool::UnboundObjectPool`]'s own contract protects for
    /// pooled frames, established here by never writing to a published texture
    /// at all.
    fn emit_frame_gpu(
        &mut self,
    ) -> std::result::Result<
        crate::pool::UnboundObjectPoolRef<ffmpeg::frame::Video>,
        DxgiCaptureSourceError,
    > {
        // Nothing to assemble on the direct path: `poll_capture` drew this
        // tick's capture into the composite as it arrived, and a tick that
        // captured nothing left the one before it in place.
        if !self.direct && (self.composite.is_none() || self.captured_since_picture) {
            self.build_composite()?;
        }

        let mut wrapper = self.pool.get();
        let composite = self.composite.as_ref().expect(
            "a composite exists by the time anything is emitted — `run` waits for every unit \
             to have captured, which is what puts one there",
        );
        // SAFETY: `ptr` is the pooled wrapper's own `AVFrame`, unreferenced
        // before it is given a new one, and the source is the composite this
        // element holds — both live, and distinct from each other. Every
        // emission being a reference to that one frame is also what makes its
        // buffer's reference count say whether the texture is still being
        // read, which is what `build_composite` asks.
        unsafe {
            let ptr = wrapper.as_mut_ptr();
            ffi::av_frame_unref(ptr);
            let code = ffi::av_frame_ref(ptr, composite.as_ptr());
            if code < 0 {
                return Err(DxgiCaptureSourceError::FrameRef(code));
            }
        }
        Ok(wrapper)
    }

    /// Draws every unit's latest capture into a composite nothing is reading,
    /// and makes it the one emitted from here on.
    ///
    /// Into a texture rather than a fresh one wherever possible. A tick that
    /// captured something used to allocate a screen-sized texture — 8 MiB at
    /// 1080p, and at 60 fps on a screen with real motion that is an allocation
    /// per frame, which is exactly the case none of the repeat handling above
    /// can help with. What it may *not* do is draw over pixels an emitted
    /// frame is still showing, so a composite becomes eligible again only once
    /// [`picture_is_referenced`] reads false for it: every emission is a
    /// reference to the composite's own frame, so that count is the number of
    /// frames still in flight over it.
    ///
    /// That rule is also what keeps a reused texture from being mistaken for
    /// an unchanged picture downstream. Everything that recognises a repeat by
    /// address — the video compositors, `ChangeGate` — holds the frame whose
    /// address it compares, which is exactly the reference this checks, so a
    /// texture cannot come back carrying different pixels while anything still
    /// names it.
    fn build_composite(&mut self) -> std::result::Result<(), DxgiCaptureSourceError> {
        // The composite being replaced goes back among the spares rather than
        // being freed; it is reusable as soon as nothing refers to it.
        if let Some(previous) = self.composite.take() {
            self.spare_composites.push(previous);
        }
        let reusable = self
            .spare_composites
            .iter()
            .position(|spare| !picture_is_referenced(spare));
        let composite = match reusable {
            Some(index) => self.spare_composites.swap_remove(index),
            None => self.new_composite()?,
        };

        let (texture_raw, _) =
            d3d11va_texture(&composite).expect("a composite is a D3D11 frame with its texture");
        // SAFETY: `texture_raw` is borrowed from the live composite frame that
        // owns it; cloning the wrapper acquires an independent COM reference
        // for the copies below.
        let texture = unsafe {
            ID3D11Texture2D::from_raw_borrowed(&texture_raw)
                .expect("a composite's texture pointer is never null")
                .clone()
        };
        let dst_resource: ID3D11Resource = texture.cast()?;
        for unit in &self.units {
            let src_resource: ID3D11Resource = staging_of(unit)?;
            let box_ = unit.source_box;
            // SAFETY: source/destination resources belong to this device;
            // `source_box` was derived from their intersection and its
            // destination offset is bounded by the composite dimensions.
            unsafe {
                self.context.CopySubresourceRegion(
                    &dst_resource,
                    0,
                    unit.dest_x,
                    unit.dest_y,
                    0,
                    &src_resource,
                    0,
                    Some(&box_ as *const D3D11_BOX),
                );
            }
        }

        self.composite = Some(composite);
        self.captured_since_picture = false;
        Ok(())
    }

    /// The composite this tick's capture should be drawn into, and whether it
    /// already holds the frame before this one.
    ///
    /// The one drawn into last, when nothing is reading it: then only what
    /// changed since has to be copied, which is what makes a still window on
    /// a busy screen cost the window rather than the screen. Otherwise a
    /// composite nothing is reading, which is missing everything up to now
    /// and so takes the whole picture. Taken out of `self` either way — the
    /// caller puts it back once the copy has either succeeded or failed, so a
    /// failure cannot leave a half-drawn composite as the one being shown.
    fn composite_to_draw_into(
        &mut self,
    ) -> std::result::Result<(ffmpeg::frame::Video, bool), DxgiCaptureSourceError> {
        if let Some(current) = self.composite.take() {
            if !picture_is_referenced(&current) {
                return Ok((current, true));
            }
            self.spare_composites.push(current);
        }
        let reusable = self
            .spare_composites
            .iter()
            .position(|spare| !picture_is_referenced(spare));
        let composite = match reusable {
            // Oldest first, which is what `remove` preserves and
            // `swap_remove` would not: a composite the GPU may still be
            // sampling is one the driver has to wait for before this frame's
            // copy into it can start, and the oldest free one is the one it
            // is least likely to still be reading.
            Some(index) => self.spare_composites.remove(index),
            None => self.new_composite()?,
        };
        Ok((composite, false))
    }

    /// One more composite texture, wrapped as the frame that owns it.
    fn new_composite(&self) -> std::result::Result<ffmpeg::frame::Video, DxgiCaptureSourceError> {
        let desc = D3D11_TEXTURE2D_DESC {
            Width: self.width,
            Height: self.height,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        let mut texture: Option<ID3D11Texture2D> = None;
        // SAFETY: `desc` is a fully initialized GPU texture description, no
        // initial data is supplied, and `texture` is a live out-parameter.
        unsafe {
            self.device
                .CreateTexture2D(&desc, None, Some(&mut texture))?;
        }
        let texture = texture.expect("CreateTexture2D succeeded without producing a texture");
        Ok(wrap_d3d11_texture(texture, self.width, self.height)?)
    }
}

impl Element for DxgiCaptureSource {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::DxgiCaptureSource
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for DxgiCaptureSource {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl SourceElement for DxgiCaptureSource {
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
            // The wait for the next tick happens here, in a sleep that holds
            // nothing, rather than inside `AcquireNextFrame` — see
            // `ACQUIRE_TIMEOUT_MS` on why that call is never given time to
            // wait in. Bounded by `POLL_GRANULARITY` so `Stop` stays
            // responsive at a low configured `fps`.
            let remaining = schedule.remaining(Instant::now());
            if !remaining.is_zero() {
                thread::sleep(remaining.min(POLL_GRANULARITY));
                continue;
            }
            if let Err(error) = self.poll_capture(ACQUIRE_TIMEOUT_MS) {
                pp_error!(self, "capture failed: {error}");
                return Err(error.into());
            }

            if !schedule.is_due(Instant::now()) {
                continue;
            }

            if !self.all_captured() {
                // Still advance even though there's nothing to emit this
                // tick — otherwise `next_due` sits in the past and the
                // next iteration's `poll_timeout` above is zero, busy-looping
                // instead of waiting for the next tick.
                schedule.advance_after_tick(Instant::now());
                continue; // nothing real captured yet — nothing to emit
            }
            let frame = match self.emit_frame() {
                Ok(frame) => frame,
                Err(error) => {
                    pp_error!(self, "emit_frame failed: {error}");
                    return Err(error.into());
                }
            };
            if let Err(error) = self.pad.push(MediaBuffer::Video(Arc::new(frame))) {
                bus.post(
                    &self.pp_log,
                    BusEvent::Error {
                        element_type: ElementType::DxgiCaptureSource,
                        name: self.name.clone(),
                        error,
                    },
                );
            }
            // Advance only now that this tick's own work (emit + push,
            // which a slow downstream/GPU readback can stretch
            // arbitrarily) is done — see `TestVideoSource::run`'s
            // identical correction for why the placement matters.
            schedule.advance_after_tick(Instant::now());
        }
    }

    fn seek(&mut self, _target: Duration) -> Result<Duration> {
        Err(DxgiCaptureSourceError::SeekUnsupported.into())
    }
}

/// Confirms `device` was created on the same adapter as `target`.
///
/// Compared by adapter LUID rather than by interface pointer: the two sides are
/// obtained independently, so the same physical adapter can arrive as different
/// COM objects, while the LUID stays stable across DXGI interface versions.
fn validate_device_adapter(
    device: &ID3D11Device,
    target: &IDXGIAdapter1,
) -> std::result::Result<(), DxgiCaptureSourceError> {
    let dxgi_device: IDXGIDevice = device.cast()?;
    // SAFETY: `dxgi_device` is live, and `GetAdapter` hands back an owned
    // reference this function is responsible for.
    let device_adapter = unsafe { dxgi_device.GetAdapter()? };
    // SAFETY: `device_adapter` is the live reference returned just above, and
    // `GetDesc` only fills a plain descriptor it does not retain.
    let device_luid = unsafe { device_adapter.GetDesc()? }.AdapterLuid;
    let target_adapter: windows::Win32::Graphics::Dxgi::IDXGIAdapter = target.cast()?;
    // SAFETY: `target_adapter` is live from the cast above, and `GetDesc` again
    // only fills a plain descriptor it does not retain.
    let target_luid = unsafe { target_adapter.GetDesc()? }.AdapterLuid;
    if (device_luid.LowPart, device_luid.HighPart) != (target_luid.LowPart, target_luid.HighPart) {
        return Err(DxgiCaptureSourceError::DeviceAdapterMismatch);
    }
    Ok(())
}

fn pick_output(
    factory: &IDXGIFactory1,
    output_index: u32,
) -> std::result::Result<(IDXGIAdapter1, IDXGIOutput1), DxgiCaptureSourceError> {
    let mut remaining = output_index;
    let mut adapter_index = 0u32;
    loop {
        // SAFETY: adapters are enumerated monotonically on this live factory
        // until DXGI reports exhaustion.
        let adapter = match unsafe { factory.EnumAdapters1(adapter_index) } {
            Ok(adapter) => adapter,
            Err(_) => return Err(DxgiCaptureSourceError::NoSuchOutput(output_index)),
        };
        let mut output_i = 0u32;
        loop {
            // SAFETY: outputs are enumerated monotonically on the live adapter
            // until DXGI reports exhaustion.
            let output = match unsafe { adapter.EnumOutputs(output_i) } {
                Ok(output) => output,
                Err(_) => break,
            };
            if remaining == 0 {
                let output1: IDXGIOutput1 = output.cast()?;
                return Ok((adapter, output1));
            }
            remaining -= 1;
            output_i += 1;
        }
        adapter_index += 1;
    }
}

/// One output resolved by [`resolve_area`]: its own adapter, the output
/// itself, and its absolute-desktop `DesktopCoordinates`.
type ResolvedOutput = (IDXGIAdapter1, IDXGIOutput1, RECT);

/// Resolves `area` into the concrete output(s) it captures from, plus the
/// absolute-desktop rectangle actually requested. [`CaptureArea::Output`]
/// always resolves to exactly one target (that output's own
/// `DesktopCoordinates` doubles as the requested rectangle — the whole
/// monitor). [`CaptureArea::Region`] resolves to every output whose own
/// desktop rectangle intersects the requested one —
/// A unit's own surface, as the resource a copy or a map takes.
///
/// Only reached where the unit has one: the paths that do not draw straight
/// into a composite — [`CaptureMode::Cpu`], and several outputs assembled
/// into one image.
fn staging_of(unit: &CaptureUnit) -> std::result::Result<ID3D11Resource, DxgiCaptureSourceError> {
    Ok(unit
        .staging_texture
        .as_ref()
        .expect("a unit that does not draw into a composite has its own surface")
        .cast()?)
}

/// The texture a composite frame owns, as the resource a copy takes.
fn composite_resource(
    composite: &ffmpeg::frame::Video,
) -> std::result::Result<ID3D11Resource, DxgiCaptureSourceError> {
    let (texture_raw, _) =
        d3d11va_texture(composite).expect("a composite is a D3D11 frame with its texture");
    // SAFETY: `texture_raw` is borrowed from the live composite frame that
    // owns it; cloning the wrapper acquires an independent COM reference for
    // the copy that follows.
    let texture = unsafe {
        ID3D11Texture2D::from_raw_borrowed(&texture_raw)
            .expect("a composite's texture pointer is never null")
            .clone()
    };
    Ok(texture.cast()?)
}

/// The part of `rect` inside `source_box`, or `None` when they do not meet.
///
/// A changed region DXGI reports covers the whole output, while a capture may
/// have asked for a piece of it ([`CaptureArea::Region`]); and a rectangle
/// arrives as `RECT`, which is signed, so this is also what keeps a negative
/// or inverted one out of the unsigned copy that follows.
fn clip_to_box(rect: RECT, source_box: &D3D11_BOX) -> Option<RECT> {
    let left = rect.left.max(source_box.left as i32);
    let top = rect.top.max(source_box.top as i32);
    let right = rect.right.min(source_box.right as i32);
    let bottom = rect.bottom.min(source_box.bottom as i32);
    (left < right && top < bottom).then_some(RECT {
        left,
        top,
        right,
        bottom,
    })
}

/// [`DxgiCaptureSourceError::RegionOutsideDesktop`] if none do — and fails
/// with [`DxgiCaptureSourceError::RegionSpansMultipleAdapters`] if those
/// outputs aren't all on the same adapter, checked here before
/// [`DxgiCaptureSource::open`] opens any duplication.
fn resolve_area(
    factory: &IDXGIFactory1,
    area: &CaptureArea,
) -> std::result::Result<(Vec<ResolvedOutput>, RECT), DxgiCaptureSourceError> {
    match *area {
        CaptureArea::Output { output_index } => {
            let (adapter, output) = pick_output(factory, output_index)?;
            // SAFETY: the output is a live DXGI interface; `GetDesc` returns
            // its plain descriptor without retaining caller pointers.
            let desktop_rect = unsafe {
                output
                    .cast::<windows::Win32::Graphics::Dxgi::IDXGIOutput>()?
                    .GetDesc()
            }?
            .DesktopCoordinates;
            Ok((vec![(adapter, output, desktop_rect)], desktop_rect))
        }
        CaptureArea::Region(rect) => {
            let requested = RECT {
                left: rect.x,
                top: rect.y,
                right: rect.x + rect.width as i32,
                bottom: rect.y + rect.height as i32,
            };
            let mut targets = Vec::new();
            let mut adapter_index = 0u32;
            loop {
                // SAFETY: adapters are enumerated monotonically on the live
                // factory until DXGI reports exhaustion.
                let adapter = match unsafe { factory.EnumAdapters1(adapter_index) } {
                    Ok(adapter) => adapter,
                    Err(_) => break,
                };
                let mut output_i = 0u32;
                loop {
                    // SAFETY: outputs are enumerated monotonically on this live
                    // adapter until DXGI reports exhaustion.
                    let output = match unsafe { adapter.EnumOutputs(output_i) } {
                        Ok(output) => output,
                        Err(_) => break,
                    };
                    let output1: IDXGIOutput1 = output.cast()?;
                    // SAFETY: `output1` and its base interface are live;
                    // `GetDesc` returns a plain value.
                    let desktop_rect = unsafe {
                        output1
                            .cast::<windows::Win32::Graphics::Dxgi::IDXGIOutput>()?
                            .GetDesc()
                    }?
                    .DesktopCoordinates;
                    let intersects = desktop_rect.left < requested.right
                        && desktop_rect.right > requested.left
                        && desktop_rect.top < requested.bottom
                        && desktop_rect.bottom > requested.top;
                    if intersects {
                        targets.push((adapter.clone(), output1, desktop_rect));
                    }
                    output_i += 1;
                }
                adapter_index += 1;
            }
            if targets.is_empty() {
                return Err(DxgiCaptureSourceError::RegionOutsideDesktop(rect));
            }
            // SAFETY: the first target's live adapter returns its immutable
            // descriptor by value.
            let first_luid = unsafe {
                targets[0]
                    .0
                    .cast::<windows::Win32::Graphics::Dxgi::IDXGIAdapter>()?
                    .GetDesc()
            }?
            .AdapterLuid;
            for (adapter, _, _) in &targets[1..] {
                // SAFETY: each target adapter is live and returns its immutable
                // descriptor by value for the identity comparison.
                let luid = unsafe {
                    adapter
                        .cast::<windows::Win32::Graphics::Dxgi::IDXGIAdapter>()?
                        .GetDesc()
                }?
                .AdapterLuid;
                if (luid.LowPart, luid.HighPart) != (first_luid.LowPart, first_luid.HighPart) {
                    return Err(DxgiCaptureSourceError::RegionSpansMultipleAdapters);
                }
            }
            Ok((targets, requested))
        }
    }
}

/// Blends [`CursorShape`] onto `dst` (a `Pixel::BGRA` plane, `dst_stride`
/// bytes per row, `dst_width`x`dst_height` pixels) at `(pos_x, pos_y)`,
/// clipped to `dst`'s bounds — the position can legitimately fall partly
/// outside this output's captured region on a multi-monitor setup.
/// Implements the three DXGI pointer shape kinds per MSDN's
/// `DXGI_OUTDUPL_POINTER_SHAPE_TYPE` docs. A pure function over byte
/// buffers (no D3D calls) so it's unit-testable without a live capture.
fn composite_cursor(
    dst: &mut [u8],
    dst_stride: usize,
    dst_width: u32,
    dst_height: u32,
    pos_x: i32,
    pos_y: i32,
    shape: &CursorShape,
) {
    if shape.kind == DXGI_OUTDUPL_POINTER_SHAPE_TYPE_MONOCHROME.0 as u32 {
        let mask_height = shape.height / 2;
        for row in 0..mask_height {
            for col in 0..shape.width {
                let byte_col = (col / 8) as usize;
                let bit = 7 - (col % 8);
                let and_byte = shape.data[row as usize * shape.pitch as usize + byte_col];
                let xor_byte =
                    shape.data[(mask_height + row) as usize * shape.pitch as usize + byte_col];
                let and_bit = (and_byte >> bit) & 1;
                let xor_bit = (xor_byte >> bit) & 1;
                let (x, y) = (pos_x + col as i32, pos_y + row as i32);
                if x < 0 || y < 0 || x as u32 >= dst_width || y as u32 >= dst_height {
                    continue;
                }
                let offset = y as usize * dst_stride + x as usize * 4;
                match (and_bit, xor_bit) {
                    (0, 0) => {
                        dst[offset] = 0;
                        dst[offset + 1] = 0;
                        dst[offset + 2] = 0;
                        dst[offset + 3] = 255;
                    }
                    (0, 1) => {
                        dst[offset] = 255;
                        dst[offset + 1] = 255;
                        dst[offset + 2] = 255;
                        dst[offset + 3] = 255;
                    }
                    (1, 0) => {}
                    _ => {
                        dst[offset] ^= 0xFF;
                        dst[offset + 1] ^= 0xFF;
                        dst[offset + 2] ^= 0xFF;
                    }
                }
            }
        }
    } else if shape.kind == DXGI_OUTDUPL_POINTER_SHAPE_TYPE_COLOR.0 as u32 {
        for row in 0..shape.height {
            for col in 0..shape.width {
                let idx = row as usize * shape.pitch as usize + col as usize * 4;
                let (b, g, r, a) = (
                    shape.data[idx],
                    shape.data[idx + 1],
                    shape.data[idx + 2],
                    shape.data[idx + 3],
                );
                let (x, y) = (pos_x + col as i32, pos_y + row as i32);
                if x < 0 || y < 0 || x as u32 >= dst_width || y as u32 >= dst_height {
                    continue;
                }
                let offset = y as usize * dst_stride + x as usize * 4;
                let inv = 255 - a as u32;
                dst[offset] = ((b as u32 * a as u32 + dst[offset] as u32 * inv) / 255) as u8;
                dst[offset + 1] =
                    ((g as u32 * a as u32 + dst[offset + 1] as u32 * inv) / 255) as u8;
                dst[offset + 2] =
                    ((r as u32 * a as u32 + dst[offset + 2] as u32 * inv) / 255) as u8;
                dst[offset + 3] = 255;
            }
        }
    } else if shape.kind == DXGI_OUTDUPL_POINTER_SHAPE_TYPE_MASKED_COLOR.0 as u32 {
        for row in 0..shape.height {
            for col in 0..shape.width {
                let idx = row as usize * shape.pitch as usize + col as usize * 4;
                let (b, g, r, a) = (
                    shape.data[idx],
                    shape.data[idx + 1],
                    shape.data[idx + 2],
                    shape.data[idx + 3],
                );
                let (x, y) = (pos_x + col as i32, pos_y + row as i32);
                if x < 0 || y < 0 || x as u32 >= dst_width || y as u32 >= dst_height {
                    continue;
                }
                let offset = y as usize * dst_stride + x as usize * 4;
                if a == 0xFF {
                    dst[offset] ^= b;
                    dst[offset + 1] ^= g;
                    dst[offset + 2] ^= r;
                } else {
                    dst[offset] = b;
                    dst[offset + 1] = g;
                    dst[offset + 2] = r;
                    dst[offset + 3] = 255;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{buffer::picture_id, platform::windows::d3d11va::d3d11va_texture};

    /// [`CaptureMode::Cpu`]'s half of the same contract: a tick that
    /// captured nothing points another wrapper at the picture already
    /// copied, and one that captured something copies into a frame the
    /// pool considers free rather than over pixels a wrapper is showing.
    ///
    /// Asserted on the picture identity rather than on a frame rate,
    /// because the saving is a full-size copy per tick, not a rate: the
    /// element still emits every tick either way, which is the part that
    /// must not change.
    ///
    /// Hardware test: skips when the machine has no desktop duplication.
    #[test]
    fn an_unchanged_tick_reuses_the_picture_it_already_copied() {
        let (mut source, _format, _device) = match DxgiCaptureSource::open(
            "reuse-cpu",
            DxgiCaptureOptions {
                area: CaptureArea::Output { output_index: 0 },
                fps: 60,
                capture_mode: CaptureMode::Cpu {
                    include_cursor: false,
                },
            },
        ) {
            Ok(opened) => opened,
            Err(error) => {
                eprintln!("skipping: no desktop duplication available here ({error})");
                return;
            }
        };

        // Wait for a real capture rather than assuming one is ready: until a
        // unit has copied something there is nothing to emit, which is what
        // `all_captured` gates on.
        let deadline = Instant::now() + Duration::from_secs(5);
        while !source.all_captured() && Instant::now() < deadline {
            source.poll_capture(16).expect("poll the duplication");
        }
        if !source.all_captured() {
            eprintln!("skipping: the desktop never produced a frame to capture");
            return;
        }

        let first = source.emit_frame_cpu().expect("copy the first picture");
        let picture = picture_id(&first);

        // No `poll_capture` in between and the cursor is not drawn, so
        // nothing this tick would show has changed.
        let second = source.emit_frame_cpu().expect("emit an unchanged tick");
        assert_eq!(
            picture_id(&second),
            picture,
            "an unchanged tick copied the picture again instead of re-wrapping it"
        );

        // A tick that did capture must not copy over the picture the frames
        // above are still showing.
        source.captured_since_picture = true;
        let third = source.emit_frame_cpu().expect("emit a changed tick");
        assert_ne!(
            picture_id(&third),
            picture,
            "a changed tick copied over a picture still referenced downstream"
        );
    }

    /// A tick that captured nothing re-wraps the composite instead of
    /// building an identical one, and one that captured something builds a
    /// new one rather than overwriting what may still be held downstream.
    ///
    /// Asserted on the texture identity rather than on a frame rate, because
    /// the saving is an allocation and a full-size copy per tick, not a rate:
    /// the element still emits every tick either way, which is the part that
    /// must not change.
    ///
    /// Hardware test: skips when the machine has no desktop duplication.
    #[test]
    fn an_unchanged_tick_reuses_the_composite_it_already_built() {
        let (mut source, _format, _device) = match DxgiCaptureSource::open(
            "reuse",
            DxgiCaptureOptions {
                area: CaptureArea::Output { output_index: 0 },
                fps: 60,
                capture_mode: CaptureMode::Gpu,
            },
        ) {
            Ok(opened) => opened,
            Err(error) => {
                eprintln!("skipping: no desktop duplication available here ({error})");
                return;
            }
        };

        // Wait for the first real capture rather than assuming one is ready:
        // `emit_frame_gpu` has nothing to composite until a unit has copied
        // something, which is what `all_captured` gates on.
        let deadline = Instant::now() + Duration::from_secs(5);
        while !source.all_captured() && Instant::now() < deadline {
            source.poll_capture(16).expect("poll the duplication");
        }
        if !source.all_captured() {
            eprintln!("skipping: the desktop never produced a frame to capture");
            return;
        }

        let first = source.emit_frame_gpu().expect("build the first composite");
        let first_texture = d3d11va_texture(&first).expect("a D3D11 frame").0;

        // No `poll_capture` in between, so nothing was captured and the
        // composite cannot have changed.
        let second = source.emit_frame_gpu().expect("emit an unchanged tick");
        assert_eq!(
            d3d11va_texture(&second).expect("a D3D11 frame").0,
            first_texture,
            "an unchanged tick rebuilt the composite instead of re-wrapping it"
        );

        // The next capture must not be drawn over a texture those frames are
        // still showing, and a composite it has never drawn into holds
        // nothing of the frames before this one — so it takes the whole
        // picture rather than what changed since.
        let (target, up_to_date) = source
            .composite_to_draw_into()
            .expect("a composite to draw into");
        let target_texture = d3d11va_texture(&target).expect("a D3D11 frame").0;
        assert_ne!(
            target_texture, first_texture,
            "a capture would have been drawn over a composite still referenced downstream"
        );
        assert!(
            !up_to_date,
            "a composite this element has not drawn into cannot be treated as current"
        );
        // What `poll_capture` does once its copy has succeeded.
        source.composite = Some(target);

        // With nothing reading it any more, the composite drawn last is the
        // one to draw into again — that is what lets a tick copy only the
        // region DXGI says changed instead of the whole screen.
        drop(first);
        drop(second);
        let (again, up_to_date) = source
            .composite_to_draw_into()
            .expect("a composite to draw into");
        assert_eq!(
            d3d11va_texture(&again).expect("a D3D11 frame").0,
            target_texture,
            "a capture allocated or recycled a composite while the current one was free"
        );
        assert!(
            up_to_date,
            "the composite drawn into last holds the frame before this one"
        );
    }

    /// Every changed region DXGI reports is answered inside the piece of the
    /// output actually being captured, and a rectangle that falls outside it
    /// is not copied at all.
    #[test]
    fn a_changed_region_is_clipped_to_what_was_asked_for() {
        let source_box = D3D11_BOX {
            left: 100,
            top: 50,
            front: 0,
            right: 400,
            bottom: 250,
            back: 1,
        };
        let inside = RECT {
            left: 150,
            top: 60,
            right: 200,
            bottom: 100,
        };
        assert_eq!(clip_to_box(inside, &source_box), Some(inside));

        let straddling = RECT {
            left: 0,
            top: 0,
            right: 150,
            bottom: 80,
        };
        assert_eq!(
            clip_to_box(straddling, &source_box),
            Some(RECT {
                left: 100,
                top: 50,
                right: 150,
                bottom: 80
            }),
            "the part inside is what gets copied"
        );

        let outside = RECT {
            left: 500,
            top: 300,
            right: 600,
            bottom: 400,
        };
        assert_eq!(clip_to_box(outside, &source_box), None);

        let inverted = RECT {
            left: 200,
            top: 100,
            right: 150,
            bottom: 80,
        };
        assert_eq!(
            clip_to_box(inverted, &source_box),
            None,
            "a rectangle with no area cannot become an unsigned copy extent"
        );
    }
    fn blank_frame(width: u32, height: u32) -> (Vec<u8>, usize) {
        let stride = width as usize * 4;
        (vec![0u8; stride * height as usize], stride)
    }

    #[test]
    fn monochrome_cursor_draws_black_white_and_leaves_transparent_alone() {
        let (mut dst, stride) = blank_frame(4, 4);
        // 2x2 mask: AND=0/XOR=0 (black), AND=0/XOR=1 (white),
        // AND=1/XOR=0 (unchanged), AND=1/XOR=1 (invert).
        // AND row: bits 0,0,1,1 -> 0b00110000 in the top nibble (MSB first)
        // XOR row: bits 0,1,0,1 -> 0b01010000
        let and_row = 0b0011_0000u8;
        let xor_row = 0b0101_0000u8;
        dst[stride + 2 * 4] = 200; // pre-existing pixel at (2,1) to check invert
        dst[stride + 2 * 4 + 1] = 100;
        dst[stride + 2 * 4 + 2] = 50;
        let shape = CursorShape {
            kind: DXGI_OUTDUPL_POINTER_SHAPE_TYPE_MONOCHROME.0 as u32,
            width: 4,
            height: 4, // 2 rows AND + 2 rows XOR
            pitch: 1,
            data: vec![and_row, and_row, xor_row, xor_row],
        };
        composite_cursor(&mut dst, stride, 4, 4, 0, 0, &shape);

        // (0,0): and=0,xor=0 -> black
        assert_eq!(&dst[0..4], &[0, 0, 0, 255]);
        // (1,0): and=0,xor=1 -> white
        assert_eq!(&dst[4..8], &[255, 255, 255, 255]);
        // (2,1): and=1,xor=0 -> unchanged (pre-existing pixel)
        let off = stride + 2 * 4;
        assert_eq!(&dst[off..off + 3], &[200, 100, 50]);
        // (3,1): and=1,xor=1 -> inverted from 0 -> 255
        let off = stride + 3 * 4;
        assert_eq!(&dst[off..off + 3], &[255, 255, 255]);
    }

    #[test]
    fn color_cursor_alpha_blends_over_destination() {
        let (mut dst, stride) = blank_frame(2, 1);
        dst[0..4].copy_from_slice(&[10, 20, 30, 255]); // dst pixel (0,0)
        let shape = CursorShape {
            kind: DXGI_OUTDUPL_POINTER_SHAPE_TYPE_COLOR.0 as u32,
            width: 1,
            height: 1,
            pitch: 4,
            data: vec![200, 150, 100, 255], // fully opaque src -> fully replaces
        };
        composite_cursor(&mut dst, stride, 2, 1, 0, 0, &shape);
        assert_eq!(&dst[0..4], &[200, 150, 100, 255]);
    }

    #[test]
    fn masked_color_cursor_xors_when_alpha_is_full_and_replaces_otherwise() {
        let (mut dst, stride) = blank_frame(2, 1);
        dst[0..4].copy_from_slice(&[0b1010_1010, 0, 0, 255]);
        dst[4..8].copy_from_slice(&[1, 2, 3, 255]);
        let shape = CursorShape {
            kind: DXGI_OUTDUPL_POINTER_SHAPE_TYPE_MASKED_COLOR.0 as u32,
            width: 2,
            height: 1,
            pitch: 8,
            data: vec![
                0b0101_0101,
                0,
                0,
                0xFF, // xor at (0,0)
                77,
                88,
                99,
                0x00, // replace at (1,0)
            ],
        };
        composite_cursor(&mut dst, stride, 2, 1, 0, 0, &shape);
        assert_eq!(dst[0], 0b1010_1010 ^ 0b0101_0101);
        assert_eq!(&dst[4..8], &[77, 88, 99, 255]);
    }

    /// Every emitted frame has to describe its own color, in both capture
    /// modes: `D3d11VideoCompositor` reads exactly these two fields to pick
    /// an NV12 conversion matrix, and leaving them unset makes it fall back
    /// to a guess instead of using what this source actually produces.
    ///
    /// Skips when the machine has no desktop to duplicate (a headless or
    /// session-0 runner), since that is a real environment rather than a
    /// failure.
    #[test]
    fn emitted_frames_describe_full_range_rgb_in_both_modes() {
        use crate::{buffer::MediaBuffer, elements::AppSink, pipeline::Pipeline};
        use std::sync::{Arc, Mutex};

        for capture_mode in [
            CaptureMode::Cpu {
                include_cursor: false,
            },
            CaptureMode::Gpu,
        ] {
            let options = DxgiCaptureOptions {
                fps: 30,
                capture_mode: capture_mode.clone(),
                ..DxgiCaptureOptions::default()
            };
            let Ok((source, _format, _device)) = DxgiCaptureSource::open("test-capture", options)
            else {
                eprintln!("skipping {capture_mode:?}: no duplicable desktop on this machine");
                continue;
            };

            let seen: Arc<Mutex<Option<(ffmpeg::color::Space, ffmpeg::color::Range)>>> =
                Arc::new(Mutex::new(None));
            let recorded = seen.clone();
            let sink = AppSink::new("test-capture-sink", move |buf| {
                if let MediaBuffer::Video(frame) = buf {
                    let mut slot = recorded
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    slot.get_or_insert((frame.color_space(), frame.color_range()));
                }
                Ok(())
            });

            let pipeline = Pipeline::new("capture-color", source, |source, ctx| {
                let branch = ctx.branch().to(Box::new(sink))?;
                ctx.attach(source, 0, branch)?;
                Ok(())
            })
            .expect("wiring a capture source to an AppSink should succeed");
            pipeline.run().unwrap();
            std::thread::sleep(std::time::Duration::from_millis(300));
            pipeline.stop();

            let observed = seen
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .take();
            let Some((space, range)) = observed else {
                eprintln!("skipping {capture_mode:?}: capture produced no frame in time");
                continue;
            };
            assert_eq!(
                space,
                ffmpeg::color::Space::RGB,
                "{capture_mode:?} must describe its BGRA samples as RGB, not leave them unspecified"
            );
            assert_eq!(
                range,
                ffmpeg::color::Range::JPEG,
                "{capture_mode:?} must describe desktop pixels as full range"
            );
        }
    }

    #[test]
    fn gpu_capture_accepts_its_adapter_device_from_the_caller() {
        let options = DxgiCaptureOptions {
            capture_mode: CaptureMode::Gpu,
            ..DxgiCaptureOptions::default()
        };
        let Ok((source, _format, Some(device))) =
            DxgiCaptureSource::open("device-provider", options.clone())
        else {
            eprintln!("skipping: no duplicable desktop on this machine");
            return;
        };
        drop(source);

        let (source, _format) =
            DxgiCaptureSource::open_with_device("device-consumer", options, &device)
                .expect("the device created for this output must pass adapter validation");
        assert_eq!(source.device.as_raw(), device.as_raw());
    }
}
