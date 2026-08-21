use std::{
    ffi::c_void,
    sync::Arc,
    time::{Duration, Instant},
};

use crate::pp_log::{PpLog, pp_error, pp_info};
use ffmpeg_next as ffmpeg;
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
                DXGI_OUTDUPL_FRAME_INFO, DXGI_OUTDUPL_POINTER_SHAPE_INFO,
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
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    control::{ControlReceiver, drain_control},
    element::{Element, ElementType, Source, SourceElement, element_pp_log},
    elements::VideoFormat,
    elements::filter::decoder::d3d11va_decoder::wrap_d3d11_texture,
    error::Result,
    pad::SrcPad,
    pool::UnboundObjectPool,
    schedule::PeriodicSchedule,
};

/// How often [`DxgiCaptureSource::run`]'s poll loop re-checks
/// `drain_control`/whether it's time to emit, even mid-wait for the next
/// real desktop change — bounds `Stop` latency at very low configured
/// [`DxgiCaptureOptions::fps`] values, where "wait until the next tick" on
/// its own could otherwise be a long, unresponsive block. Same idea as
/// [`crate::queue::Queue`]'s own `STOP_POLL_INTERVAL`.
const POLL_GRANULARITY: Duration = Duration::from_millis(100);

/// Errors specific to `DxgiCaptureSource`. Converts into the crate-wide
/// `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum DxgiCaptureSourceError {
    #[error("windows error: {0}")]
    Windows(#[from] windows::core::Error),

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

    #[error("DxgiCaptureSource doesn't support seeking a live capture")]
    SeekUnsupported,

    #[error("CaptureArea::Region {0:?} doesn't overlap any display output")]
    RegionOutsideDesktop(CaptureRect),

    /// See [`CaptureArea::Region`]'s own docs on why this is a hard
    /// failure rather than an automatic CPU-bridged fallback.
    #[error(
        "CaptureArea::Region spans outputs on more than one GPU adapter — \
         zero-copy compositing across adapters isn't supported"
    )]
    RegionSpansMultipleAdapters,

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
    Cpu { include_cursor: bool },
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
    /// Unlike [`CaptureMode::Cpu`], this variant carries no device of its
    /// own to inject: `open` always builds the device itself, from
    /// whichever adapter [`DxgiCaptureOptions::area`] actually selects
    /// (the *only* place that resolves "which adapter" — see
    /// `resolve_area`), and hands it back as `open`'s own return value
    /// for the caller to reuse. That's the one `ID3D11Device` every other
    /// D3D11 element sharing this capture's output should be built from
    /// (e.g. `render_common::D3d11GpuContext::new(Some(device))`) — for
    /// `open`'s own zero-copy path to mean anything, see
    /// [`crate::elements::D3d11Renderer`]'s own docs on why. Taking a
    /// caller-supplied device here instead would only reopen the exact
    /// adapter-mismatch problem this design avoids: two independently
    /// resolved "which adapter" answers that would need to be checked
    /// against each other instead of structurally being the same one.
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
    pub x: i32,
    pub y: i32,
    pub width: u32,
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
    Output { output_index: u32 },
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
    staging_texture: ID3D11Texture2D,
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
/// something needs YUV420P, e.g. [`crate::elements::D3d12Renderer`]'s
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
/// Confirmed (`examples/render/screen_capture`, with and without a
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
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::DxgiCaptureSource, &name, None);

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

        // Always built from `area`'s own adapter (every target shares
        // one — `resolve_area` already checked), `Cpu` and `Gpu` alike —
        // see `CaptureMode::Gpu`'s own docs on why this element is the
        // sole place that resolves "which adapter", rather than trusting
        // (and validating) a caller-supplied device.
        let adapter = &targets[0].0;
        let mut device: Option<ID3D11Device> = None;
        let mut context: Option<ID3D11DeviceContext> = None;
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
        let device = device.expect("D3D11CreateDevice succeeded without producing a device");
        let context = context.expect("D3D11CreateDevice succeeded without producing a context");
        // Cloned before `device` moves into `Self` below — the only copy
        // handed back to the caller (a COM ref-count bump, not a deep
        // copy).
        let returned_device = gpu_mode.then(|| device.clone());
        let dxgi_device: IDXGIDevice = device.cast()?;

        let mut units = Vec::with_capacity(targets.len());
        for (_, output, desktop_rect) in &targets {
            let duplication = unsafe { output.DuplicateOutput(&dxgi_device) }?;
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
            let mut staging_texture: Option<ID3D11Texture2D> = None;
            unsafe { device.CreateTexture2D(&staging_desc, None, Some(&mut staging_texture)) }?;
            let staging_texture =
                staging_texture.expect("CreateTexture2D succeeded without producing a texture");

            units.push(CaptureUnit {
                duplication,
                staging_texture,
                source_box,
                dest_x,
                dest_y,
                has_captured: false,
            });
        }

        let pad = SrcPad::new(format!("{name}_src"));
        // Gpu: only the small CPU-side `AVFrame` wrapper is ever pooled
        // (`ffmpeg::frame::Video::empty` — same as `D3d11Upload`'s own
        // pool); the GPU texture itself is a fresh allocation every
        // `emit_frame` call (see that method's own docs on why). Cpu:
        // pre-sized real `Pixel::BGRA` CPU buffers, as before.
        let pool = if gpu_mode {
            UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {})
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
                unsafe { self.units[index].duplication.ReleaseFrame() }?;
                continue;
            }

            // Every fallible step here (the two `cast`s, the copy itself)
            // must still release DXGI's own frame lock on the way out —
            // an early `?` before `ReleaseFrame()` would leave this unit
            // unable to `AcquireNextFrame` again until it's torn down
            // entirely, so the copy's own result is captured instead of
            // propagated directly.
            let copy_result: std::result::Result<(), DxgiCaptureSourceError> = (|| {
                let texture: ID3D11Texture2D = resource.cast()?;
                unsafe {
                    self.context.CopyResource(
                        &self.units[index].staging_texture.cast::<ID3D11Resource>()?,
                        &texture.cast::<ID3D11Resource>()?,
                    );
                }
                Ok(())
            })();

            // Release DXGI's own frame as soon as we've copied it out,
            // rather than holding it while we map/read (Cpu mode) the
            // (independent) staging copy below — and unconditionally,
            // even if the copy above failed.
            let release_result = unsafe { self.units[index].duplication.ReleaseFrame() };
            copy_result?;
            release_result?;

            if self.gpu_mode {
                // No `Map`/CPU copy at all — `staging_texture` itself
                // *is* this unit's latest capture; `emit_frame_gpu` reads
                // straight from it. See `CaptureMode::Gpu`'s own docs.
                self.units[index].has_captured = true;
                continue;
            }

            // Cpu mode: Map this unit's own staging texture and copy
            // just its `source_box` crop into the shared composite
            // buffer at `dest_x`/`dest_y` — no separate composite GPU
            // texture needed, the crop lands directly in CPU memory at
            // its final position.
            let mut mapped = Default::default();
            unsafe {
                self.context.Map(
                    &self.units[index].staging_texture.cast::<ID3D11Resource>()?,
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
            unsafe {
                self.context.Unmap(
                    &self.units[index].staging_texture.cast::<ID3D11Resource>()?,
                    0,
                );
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
            self.emit_frame_cpu()
        };
        frame.set_pts(Some(self.frame_index));
        frame.set_color_space(ffmpeg::color::Space::RGB);
        frame.set_color_range(ffmpeg::color::Range::JPEG);
        self.frame_index += 1;
        Ok(frame)
    }

    /// Copies `self.staging` (the latest captured composite image,
    /// however stale — already assembled from every unit's own crop by
    /// `poll_capture`) into a fresh pooled CPU frame, compositing the
    /// cursor onto that copy if enabled. Copying instead of sharing
    /// `self.staging` directly is what lets each emitted frame carry its
    /// own distinct, correctly-incrementing `pts` (stamped by the
    /// caller, `emit_frame`) even when several emissions in a row show
    /// the same content — an `Arc`-shared frame can't have its `pts`
    /// safely rewritten in place once downstream might already hold a
    /// clone of it.
    fn emit_frame_cpu(&mut self) -> crate::pool::UnboundObjectPoolRef<ffmpeg::frame::Video> {
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
            if self.include_cursor
                && self.cursor_visible
                && let Some(shape) = &self.cursor_shape
            {
                composite_cursor(
                    dst,
                    dst_stride,
                    self.width,
                    self.height,
                    self.cursor_position.x,
                    self.cursor_position.y,
                    shape,
                );
            }
        }
        frame
    }

    /// `CaptureMode::Gpu`'s equivalent of [`DxgiCaptureSource::emit_frame_cpu`]:
    /// builds a fresh composite `ID3D11Texture2D` (`self.width` x
    /// `self.height`) and `CopySubresourceRegion`s every unit's own
    /// `source_box` crop into it at that unit's `dest_x`/`dest_y` — one
    /// GPU-side copy per contributing output, same reasoning
    /// `D3d11Upload::upload` documents for allocating fresh every call
    /// rather than reusing one texture: each unit's own `staging_texture`
    /// gets overwritten by its next real capture, so a frame already
    /// pushed downstream needs its own, independently stable copy of
    /// what it was capturing at push time. Wraps the fresh texture as a
    /// `Pixel::D3D11` frame via [`wrap_d3d11_texture`], reusing the
    /// pooled `AVFrame` wrapper (the pool built by `open` for
    /// `CaptureMode::Gpu` only holds these small wrappers, not GPU
    /// memory — see that pool's own construction site).
    fn emit_frame_gpu(
        &mut self,
    ) -> std::result::Result<
        crate::pool::UnboundObjectPoolRef<ffmpeg::frame::Video>,
        DxgiCaptureSourceError,
    > {
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
        unsafe {
            self.device
                .CreateTexture2D(&desc, None, Some(&mut texture))?;
        }
        let texture = texture.expect("CreateTexture2D succeeded without producing a texture");
        let dst_resource: ID3D11Resource = texture.cast()?;
        for unit in &self.units {
            let src_resource: ID3D11Resource = unit.staging_texture.cast()?;
            let box_ = unit.source_box;
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

        let mut frame = self.pool.get();
        // Overwrites the pooled slot's previous contents in place —
        // `ffmpeg::frame::Video`'s own `Drop` runs on whatever was there
        // before, releasing that frame's GPU texture right here (same
        // pattern `D3d11Upload::consume` documents).
        *frame = wrap_d3d11_texture(texture, self.width, self.height);
        Ok(frame)
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

            let poll_timeout = schedule.remaining(Instant::now()).min(POLL_GRANULARITY);
            if let Err(error) = self.poll_capture(poll_timeout.as_millis() as u32) {
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

fn pick_output(
    factory: &IDXGIFactory1,
    output_index: u32,
) -> std::result::Result<(IDXGIAdapter1, IDXGIOutput1), DxgiCaptureSourceError> {
    let mut remaining = output_index;
    let mut adapter_index = 0u32;
    loop {
        let adapter = match unsafe { factory.EnumAdapters1(adapter_index) } {
            Ok(adapter) => adapter,
            Err(_) => return Err(DxgiCaptureSourceError::NoSuchOutput(output_index)),
        };
        let mut output_i = 0u32;
        loop {
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
                let adapter = match unsafe { factory.EnumAdapters1(adapter_index) } {
                    Ok(adapter) => adapter,
                    Err(_) => break,
                };
                let mut output_i = 0u32;
                loop {
                    let output = match unsafe { adapter.EnumOutputs(output_i) } {
                        Ok(output) => output,
                        Err(_) => break,
                    };
                    let output1: IDXGIOutput1 = output.cast()?;
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
            let first_luid = unsafe {
                targets[0]
                    .0
                    .cast::<windows::Win32::Graphics::Dxgi::IDXGIAdapter>()?
                    .GetDesc()
            }?
            .AdapterLuid;
            for (adapter, _, _) in &targets[1..] {
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
            pipeline.run();
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
}
