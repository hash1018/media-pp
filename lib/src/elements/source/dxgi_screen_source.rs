use std::{
    ffi::c_void,
    sync::Arc,
    time::{Duration, Instant},
};

use ffmpeg_next as ffmpeg;
use rust_hlog::{HLog, herror, hinfo};
use thiserror::Error as ThisError;
use windows::{
    Win32::{
        Foundation::{HMODULE, POINT},
        Graphics::{
            Direct3D::D3D_DRIVER_TYPE_UNKNOWN,
            Direct3D11::{
                D3D11_BIND_FLAG, D3D11_BIND_SHADER_RESOURCE, D3D11_CPU_ACCESS_READ,
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
    element::{Element, ElementType, Source, SourceElement, element_hlog},
    elements::filter::decoder::d3d11va_decoder::wrap_d3d11_texture,
    error::Result,
    pad::SrcPad,
    pool::UnboundObjectPool,
};

/// How often [`DxgiScreenSource::run`]'s poll loop re-checks
/// `drain_control`/whether it's time to emit, even mid-wait for the next
/// real desktop change — bounds `Stop` latency at very low configured
/// [`DxgiScreenOptions::fps`] values, where "wait until the next tick" on
/// its own could otherwise be a long, unresponsive block. Same idea as
/// [`crate::queue::Queue`]'s own `STOP_POLL_INTERVAL`.
const POLL_GRANULARITY: Duration = Duration::from_millis(100);

/// Errors specific to `DxgiScreenSource`. Converts into the crate-wide
/// `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum DxgiScreenSourceError {
    #[error("windows error: {0}")]
    Windows(#[from] windows::core::Error),

    #[error("no DXGI output at index {0} (across every adapter)")]
    NoSuchOutput(u32),

    /// `DXGI_ERROR_ACCESS_LOST` specifically, broken out of the generic
    /// [`DxgiScreenSourceError::Windows`] variant because it's the single
    /// most common *recoverable* failure mode for desktop duplication —
    /// a lock screen, a UAC prompt, a display mode change, or a
    /// fullscreen-exclusive app/overlay stealing the duplication lock all
    /// surface this way. Same "fail fast, caller rebuilds a fresh one"
    /// contract [`crate::elements::RtspSource`] already documents: this
    /// element doesn't retry internally, callers that want to survive a
    /// lock-screen cycle watch for this specific error and call
    /// [`DxgiScreenSource::open`] again.
    #[error("DXGI_ERROR_ACCESS_LOST — desktop duplication needs to be reopened")]
    AccessLost,

    #[error("DxgiScreenSource doesn't support seeking a live capture")]
    SeekUnsupported,

    #[error(
        "CaptureMode::Gpu's device is on a different adapter than output_index \
         selects — open it against the same adapter that output belongs to"
    )]
    DeviceAdapterMismatch,
}

/// How [`DxgiScreenSource::open`] captures each frame — see
/// [`DxgiScreenOptions::capture_mode`].
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
    /// ([`composite_cursor`]), which has nothing to run against under
    /// [`CaptureMode::Gpu`], where the captured image never touches the
    /// CPU at all; putting the field here instead of as a separate
    /// `DxgiScreenOptions` flag makes that combination unrepresentable
    /// rather than a runtime error to guard against.
    Cpu { include_cursor: bool },
    /// Captures straight to a GPU-resident frame tagged `Pixel::D3D11`
    /// (BGRA — desktop content has no reason to go through YUV) on
    /// `device` — no `Map`, no CPU pixel copy at all, just two GPU-side
    /// `CopyResource` calls (duplication resource -> this element's own
    /// "latest capture" texture, then that texture -> a fresh per-emission
    /// texture every tick, so an in-flight pushed frame's content can't
    /// change under whatever's still reading it — same reasoning
    /// [`crate::elements::D3d11Upload`] documents for building a fresh
    /// texture per call rather than reusing one). `device` must be opened
    /// against the *same adapter* [`DxgiScreenOptions::output_index`]
    /// selects — `open` verifies this and fails with
    /// [`DxgiScreenSourceError::DeviceAdapterMismatch`] otherwise — and
    /// should be the one `ID3D11Device` every other D3D11 element in the
    /// pipeline shares, for `open`'s own zero-copy path to mean anything —
    /// see [`crate::elements::D3d11Renderer`]'s own docs on why.
    ///
    /// No cursor option — see [`CaptureMode::Cpu`]'s own docs on why.
    Gpu { device: ID3D11Device },
}

/// Construction-time options for [`DxgiScreenSource::open`].
#[derive(Debug, Clone)]
pub struct DxgiScreenOptions {
    /// A flat index across every adapter's every output, in enumeration
    /// order (adapter 0's outputs, then adapter 1's, ...) — "monitor 0",
    /// "monitor 1", regardless of which GPU each is attached to. `0` is
    /// whatever Windows considers the first output of the first adapter,
    /// not necessarily the primary monitor.
    pub output_index: u32,
    /// The constant rate frames are emitted at — see [`DxgiScreenSource`]'s
    /// own docs on why this is a fixed output rate (like
    /// [`crate::elements::TestVideoSource::new`]'s `framerate`), not a cap
    /// on an otherwise irregular one. `30` by default, matching
    /// `TestVideoSource`'s own default.
    pub fps: u32,
    /// CPU (the original behavior) or GPU (zero-copy) capture — see
    /// [`CaptureMode`]. `CaptureMode::Cpu { include_cursor: false }` by
    /// default, so existing callers building `DxgiScreenOptions { ..
    /// ..Default::default() }` keep today's behavior unchanged.
    pub capture_mode: CaptureMode,
}

impl Default for DxgiScreenOptions {
    fn default() -> Self {
        Self {
            output_index: 0,
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

/// Captures the desktop via Windows' DXGI Desktop Duplication API
/// (`IDXGIOutputDuplication`) — GStreamer's `d3d11screencapturesrc`
/// equivalent. One src pad, pushing `Pixel::BGRA` frames (no internal
/// color conversion — same division of labor as every other source in
/// this crate: chain a [`crate::elements::Scaler`] downstream if
/// something needs YUV420P, e.g. [`crate::elements::D3d12Renderer`]'s
/// CPU-upload path or [`crate::elements::SwEncoder`]).
///
/// Emits at a **constant** rate — [`DxgiScreenOptions::fps`] — not one
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
/// [`crate::elements::TestVideoSource`]: [`DxgiScreenSource::time_base`]
/// is `1 / fps` and `pts` is a plain incrementing tick counter, one per
/// *emitted* frame, not per real capture.
///
/// Confirmed (`examples/render/screen_capture`, with and without a
/// downstream [`crate::elements::Pacer`]) that this constant-rate,
/// drift-free schedule is what actually mattered — not whether a
/// separate `Pacer` stage exists. The VFR version needed one to paper
/// over its own irregular submission timing; once emission here is
/// steady and drift-free, `Scaler`'s modest, fairly consistent per-frame
/// conversion cost isn't enough on its own to reintroduce the same vsync
/// misalignment, so a straight `DxgiScreenSource -> Scaler -> D3d12Renderer`
/// chain stays smooth with no `Pacer` at all. `Pacer` remains genuinely
/// useful for other reasons (multi-stream sync against a shared `Clock`,
/// or a stage with real per-frame variance like `SwEncoder`), just not
/// load-bearing here purely for vsync alignment the way it first
/// appeared to be.
///
/// Deliberately does **not** retry internally on `DXGI_ERROR_ACCESS_LOST`
/// (lock screen, UAC prompt, display mode change, ...) — same "fail fast,
/// caller rebuilds" contract as [`crate::elements::RtspSource`]; watch for
/// [`DxgiScreenSourceError::AccessLost`] and call
/// [`DxgiScreenSource::open`] again.
///
/// Runs until `Stop` — never reaches `Eos` on its own, same as
/// `TestVideoSource` (there's no natural end to a live desktop capture).
#[rust_hlog::hlog]
pub struct DxgiScreenSource {
    name: Arc<str>,
    /// Only used by [`CaptureMode::Gpu`]'s [`DxgiScreenSource::emit_frame`]
    /// path, to build each tick's fresh per-emission texture — unused
    /// after construction in [`CaptureMode::Cpu`], but harmless to hold
    /// either way (one extra COM reference, same device already owns).
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    duplication: IDXGIOutputDuplication,
    /// The latest captured desktop image, GPU-resident either way —
    /// CPU-readable (`D3D11_USAGE_STAGING`) under [`CaptureMode::Cpu`], or
    /// shader-bindable (`D3D11_USAGE_DEFAULT`/`D3D11_BIND_SHADER_RESOURCE`)
    /// under [`CaptureMode::Gpu`]. See [`DxgiScreenSource::poll_capture`]
    /// for which.
    staging_texture: ID3D11Texture2D,
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
    /// The most recently captured desktop image, CPU-side — plain, not
    /// pool-backed (never shared/pushed directly downstream; see `run`'s
    /// own emit step, which copies out of this into a fresh pooled frame
    /// every tick). Updated in place whenever `poll_capture` sees a real
    /// image change; re-copied from as-is on every tick where nothing new
    /// arrived, which is what makes this element emit at a constant rate
    /// rather than only on real changes. Only under [`CaptureMode::Cpu`] —
    /// `None` under [`CaptureMode::Gpu`], which reads straight from
    /// `staging_texture` instead (see `emit_frame`).
    staging: Option<ffmpeg::frame::Video>,
    /// Whether `staging` holds real captured pixels yet — `false` until
    /// the first successful capture, so `run` doesn't emit a blank frame
    /// before there's anything real to show.
    has_captured: bool,
    /// See [`DxgiScreenOptions::fps`] — kept alongside `frame_interval`
    /// so [`DxgiScreenSource::time_base`] doesn't have to recover it from
    /// a `Duration`.
    fps: i32,
    /// `1 / fps`.
    frame_interval: Duration,
    /// This element's `pts` tick counter — one per *emitted* frame (see
    /// [`DxgiScreenSource::time_base`]'s own docs), not per real capture.
    frame_index: i64,
    pad: SrcPad,
    /// Reused across every emitted frame — see [`UnboundObjectPool`]'s
    /// docs. Pre-sized to `width`/`height` up front (known from
    /// `DXGI_OUTDUPL_DESC` at `open` time), same reasoning as `Scaler`'s
    /// own pool.
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
}

// SAFETY: every D3D11/DXGI handle here is a `windows-rs` COM interface
// wrapper — thread-safe to hand off (refcounting is interlocked), and
// `&mut self` on every method that touches them (mirrors `D3d12vaDecoder`/
// `Scaler`'s own reasoning) already rules out concurrent access from
// multiple threads.
unsafe impl Send for DxgiScreenSource {}

impl DxgiScreenSource {
    /// Opens the `output_index`'th DXGI output (see
    /// [`DxgiScreenOptions::output_index`]) and starts duplicating it.
    /// Returns the element alongside the captured desktop's actual
    /// `(width, height)` — what the caller needs to build a matching
    /// downstream [`crate::elements::Scaler`]/[`crate::elements::Pacer`],
    /// same pattern as [`crate::elements::RtspSource::open`] returning
    /// stream info.
    pub fn open(
        name: impl Into<String>,
        options: DxgiScreenOptions,
    ) -> std::result::Result<(Self, u32, u32), DxgiScreenSourceError> {
        let name: Arc<str> = name.into().into();
        let hlog = element_hlog(ElementType::DxgiScreenSource, &name, None);

        let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1() }?;
        let (adapter, output) = pick_output(&factory, options.output_index)?;
        let gpu_mode = matches!(options.capture_mode, CaptureMode::Gpu { .. });
        // `CaptureMode::Gpu` has no `include_cursor` field at all (see its
        // own docs) — nothing to extract there, so `false` unconditionally.
        let include_cursor = match &options.capture_mode {
            CaptureMode::Cpu { include_cursor } => *include_cursor,
            CaptureMode::Gpu { .. } => false,
        };

        let (device, context) = match &options.capture_mode {
            CaptureMode::Cpu { .. } => {
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
                (
                    device.expect("D3D11CreateDevice succeeded without producing a device"),
                    context.expect("D3D11CreateDevice succeeded without producing a context"),
                )
            }
            CaptureMode::Gpu { device } => {
                // Same LUID check `D3d12Renderer`'s own device-mismatch
                // guard does — a device from the wrong adapter would
                // silently fail (or worse) once `DuplicateOutput`/
                // `CopyResource` actually run against `output`'s adapter.
                let selected_luid = unsafe { adapter.GetDesc() }?.AdapterLuid;
                let device_adapter = unsafe { device.cast::<IDXGIDevice>()?.GetAdapter() }?;
                let device_luid = unsafe { device_adapter.GetDesc() }?.AdapterLuid;
                if (device_luid.LowPart, device_luid.HighPart)
                    != (selected_luid.LowPart, selected_luid.HighPart)
                {
                    return Err(DxgiScreenSourceError::DeviceAdapterMismatch);
                }
                let context = unsafe { device.GetImmediateContext() }?;
                (device.clone(), context)
            }
        };

        let output1: IDXGIOutput1 = output.cast()?;
        let dxgi_device: IDXGIDevice = device.cast()?;
        let duplication = unsafe { output1.DuplicateOutput(&dxgi_device) }?;

        let desc = unsafe { duplication.GetDesc() };
        let width = desc.ModeDesc.Width;
        let height = desc.ModeDesc.Height;

        // Cpu: CPU-readable staging texture, `Map`ped every real capture
        // (see `poll_capture`). Gpu: shader-bindable, never CPU-mapped —
        // just the GPU-side "latest capture" `CopyResource` target that
        // `emit_frame` copies out of into each tick's own fresh texture.
        let staging_desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
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
            BindFlags: if gpu_mode {
                D3D11_BIND_SHADER_RESOURCE.0 as u32
            } else {
                D3D11_BIND_FLAG(0).0 as u32
            },
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
        hinfo!(
            hlog: &hlog,
            "opened: output_index={}, {}x{}, include_cursor={}, fps={}, gpu_mode={}",
            options.output_index,
            width,
            height,
            include_cursor,
            fps,
            gpu_mode
        );

        Ok((
            Self {
                name,
                hlog,
                device,
                context,
                duplication,
                staging_texture,
                width,
                height,
                gpu_mode,
                include_cursor,
                cursor_shape: None,
                cursor_position: POINT::default(),
                cursor_visible: false,
                staging,
                has_captured: false,
                fps: fps as i32,
                frame_interval: Duration::from_secs_f64(1.0 / fps as f64),
                frame_index: 0,
                pad,
                pool,
            },
            width,
            height,
        ))
    }

    /// The unit each emitted frame's `pts` is expressed in — what you
    /// need to construct a matching [`crate::elements::Pacer`]. `1 /
    /// fps`, same convention as [`crate::elements::TestVideoSource::time_base`].
    pub fn time_base(&self) -> ffmpeg::Rational {
        ffmpeg::Rational::new(1, self.fps)
    }

    /// Refreshes `self.cursor_shape` from the duplication interface's
    /// current pointer shape buffer. Only called when
    /// `DXGI_OUTDUPL_FRAME_INFO::PointerShapeBufferSize > 0` — i.e. the
    /// shape actually changed since the last call.
    fn refresh_cursor_shape(
        &mut self,
        buffer_size: usize,
    ) -> std::result::Result<(), windows::core::Error> {
        let mut buffer = vec![0u8; buffer_size];
        let mut required = 0u32;
        let mut info = DXGI_OUTDUPL_POINTER_SHAPE_INFO::default();
        unsafe {
            self.duplication.GetFramePointerShape(
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

    /// Tries once to capture a new desktop image, within `timeout_ms`.
    /// Updates `self.staging` (and `self.has_captured`) in place if the
    /// desktop image itself changed; always refreshes the cached cursor
    /// position/shape (see their own docs) regardless, since the mouse
    /// can move independently of the desktop image. A `DXGI_ERROR_WAIT_TIMEOUT`
    /// (nothing changed within `timeout_ms`) is not an error — just
    /// means `self.staging` is unchanged this call.
    fn poll_capture(&mut self, timeout_ms: u32) -> std::result::Result<(), DxgiScreenSourceError> {
        let mut info = DXGI_OUTDUPL_FRAME_INFO::default();
        let mut resource: Option<IDXGIResource> = None;
        let acquire = unsafe {
            self.duplication
                .AcquireNextFrame(timeout_ms, &mut info, &mut resource)
        };
        let resource = match acquire {
            Ok(()) => resource.expect("AcquireNextFrame succeeded without a resource"),
            Err(error) if error.code() == DXGI_ERROR_WAIT_TIMEOUT => return Ok(()),
            Err(error) if error.code() == DXGI_ERROR_ACCESS_LOST => {
                return Err(DxgiScreenSourceError::AccessLost);
            }
            Err(error) => return Err(error.into()),
        };

        if self.include_cursor {
            if info.PointerShapeBufferSize > 0 {
                self.refresh_cursor_shape(info.PointerShapeBufferSize as usize)?;
            }
            // See `cursor_position`/`cursor_visible`'s own docs: only
            // trust `info.PointerPosition` on the call where the mouse
            // itself actually changed.
            if info.LastMouseUpdateTime != 0 {
                self.cursor_position = info.PointerPosition.Position;
                self.cursor_visible = info.PointerPosition.Visible.as_bool();
            }
        }

        // `AcquireNextFrame` succeeds not just when the desktop image
        // itself changed, but also on a *cursor-only* update (the pointer
        // moved/blinked with the picture underneath it untouched) —
        // `AccumulatedFrames == 0` is how DXGI signals that case (see
        // Microsoft's own Desktop Duplication sample). The cursor
        // position was already refreshed above regardless; there's just
        // no new *image* to copy out, so release and stop here.
        if info.AccumulatedFrames == 0 {
            unsafe { self.duplication.ReleaseFrame() }?;
            return Ok(());
        }

        let texture: ID3D11Texture2D = resource.cast()?;
        unsafe {
            self.context.CopyResource(
                &self.staging_texture.cast::<ID3D11Resource>()?,
                &texture.cast::<ID3D11Resource>()?,
            );
        }

        // Release DXGI's own frame as soon as we've copied it out, rather
        // than holding it while we map/read (Cpu mode) the (independent)
        // staging copy below.
        unsafe { self.duplication.ReleaseFrame() }?;

        if self.gpu_mode {
            // No `Map`/CPU copy at all — `self.staging_texture` itself
            // *is* the latest capture; `emit_frame` reads straight from
            // it. See `CaptureMode::Gpu`'s own docs.
            self.has_captured = true;
            return Ok(());
        }

        let mut mapped = Default::default();
        unsafe {
            self.context.Map(
                &self.staging_texture.cast::<ID3D11Resource>()?,
                0,
                D3D11_MAP_READ,
                0,
                Some(&mut mapped),
            )?;
        }
        {
            let staging = self
                .staging
                .as_mut()
                .expect("CaptureMode::Cpu always has a staging buffer");
            let dst_stride = staging.stride(0);
            let row_bytes = self.width as usize * 4;
            let dst = staging.data_mut(0);
            for row in 0..self.height as usize {
                let src = unsafe {
                    std::slice::from_raw_parts(
                        (mapped.pData as *const u8).add(row * mapped.RowPitch as usize),
                        row_bytes,
                    )
                };
                dst[row * dst_stride..row * dst_stride + row_bytes].copy_from_slice(src);
            }
        }
        unsafe {
            self.context
                .Unmap(&self.staging_texture.cast::<ID3D11Resource>()?, 0);
        }
        self.has_captured = true;
        Ok(())
    }

    /// Builds the next frame to push — the unit of work `run` does once
    /// per emission tick, real change or repeat — and stamps the next
    /// `pts`. Dispatches to [`DxgiScreenSource::emit_frame_cpu`] or
    /// [`DxgiScreenSource::emit_frame_gpu`] depending on `self.gpu_mode`.
    fn emit_frame(
        &mut self,
    ) -> std::result::Result<
        crate::pool::UnboundObjectPoolRef<ffmpeg::frame::Video>,
        DxgiScreenSourceError,
    > {
        let mut frame = if self.gpu_mode {
            self.emit_frame_gpu()?
        } else {
            self.emit_frame_cpu()
        };
        frame.set_pts(Some(self.frame_index));
        self.frame_index += 1;
        Ok(frame)
    }

    /// Copies `self.staging` (the latest captured desktop image, however
    /// stale) into a fresh pooled CPU frame, compositing the cursor onto
    /// that copy if enabled. Copying instead of sharing `self.staging`
    /// directly is what lets each emitted frame carry its own distinct,
    /// correctly-incrementing `pts` (stamped by the caller, `emit_frame`)
    /// even when several emissions in a row show the same content — an
    /// `Arc`-shared frame can't have its `pts` safely rewritten in place
    /// once downstream might already hold a clone of it.
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

    /// `CaptureMode::Gpu`'s equivalent of [`DxgiScreenSource::emit_frame_cpu`]:
    /// builds a fresh `ID3D11Texture2D` (same shape as `staging_texture`)
    /// and `CopyResource`s `self.staging_texture`'s current contents into
    /// it — a second GPU-side copy, same reasoning `D3d11Upload::upload`
    /// documents for allocating fresh every call rather than reusing one
    /// texture: `staging_texture` gets overwritten by the next real
    /// capture, so a frame already pushed downstream needs its own,
    /// independently stable copy of what it was capturing at push time.
    /// Wraps the fresh texture as a `Pixel::D3D11` frame via
    /// [`wrap_d3d11_texture`], reusing the pooled `AVFrame` wrapper (the
    /// pool built by `open` for `CaptureMode::Gpu` only holds these small
    /// wrappers, not GPU memory — see that pool's own construction site).
    fn emit_frame_gpu(
        &mut self,
    ) -> std::result::Result<
        crate::pool::UnboundObjectPoolRef<ffmpeg::frame::Video>,
        DxgiScreenSourceError,
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
        unsafe {
            self.context.CopyResource(
                &texture.cast::<ID3D11Resource>()?,
                &self.staging_texture.cast::<ID3D11Resource>()?,
            );
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

impl Element for DxgiScreenSource {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::DxgiScreenSource
    }

    fn hlog(&self) -> &HLog {
        &self.hlog
    }

    fn hlog_mut(&mut self) -> &mut HLog {
        &mut self.hlog
    }
}

impl Source for DxgiScreenSource {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl SourceElement for DxgiScreenSource {
    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        hinfo!(self, "run: starting");
        let mut next_due = Instant::now();
        loop {
            if drain_control(control, self, bus)? {
                hinfo!(self, "run: stopped");
                return Ok(());
            }

            let remaining = next_due.saturating_duration_since(Instant::now());
            let poll_timeout = remaining.min(POLL_GRANULARITY);
            if let Err(error) = self.poll_capture(poll_timeout.as_millis() as u32) {
                herror!(self, "capture failed: {error}");
                return Err(error.into());
            }

            if Instant::now() < next_due {
                continue;
            }
            next_due += self.frame_interval;

            if !self.has_captured {
                continue; // nothing real captured yet — nothing to emit
            }
            let frame = match self.emit_frame() {
                Ok(frame) => frame,
                Err(error) => {
                    herror!(self, "emit_frame failed: {error}");
                    return Err(error.into());
                }
            };
            if let Err(error) = self.pad.push(MediaBuffer::Video(Arc::new(frame))) {
                bus.post(
                    &self.hlog,
                    BusEvent::Error {
                        element_type: ElementType::DxgiScreenSource,
                        name: self.name.clone(),
                        error,
                    },
                );
            }
        }
    }

    fn seek(&mut self, _target: Duration) -> Result<Duration> {
        Err(DxgiScreenSourceError::SeekUnsupported.into())
    }
}

fn pick_output(
    factory: &IDXGIFactory1,
    output_index: u32,
) -> std::result::Result<(IDXGIAdapter1, IDXGIOutput1), DxgiScreenSourceError> {
    let mut remaining = output_index;
    let mut adapter_index = 0u32;
    loop {
        let adapter = match unsafe { factory.EnumAdapters1(adapter_index) } {
            Ok(adapter) => adapter,
            Err(_) => return Err(DxgiScreenSourceError::NoSuchOutput(output_index)),
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
}
