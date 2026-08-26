use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use crossbeam_channel::{Receiver, RecvTimeoutError, bounded};
use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;
use windows::{
    Foundation::TypedEventHandler,
    Graphics::{
        Capture::{
            Direct3D11CaptureFrame, Direct3D11CaptureFramePool, GraphicsCaptureItem,
            GraphicsCaptureSession,
        },
        DirectX::{Direct3D11::IDirect3DDevice, DirectXPixelFormat},
        SizeInt32,
    },
    Win32::{
        Foundation::{CloseHandle, HMODULE, HWND, RPC_E_CHANGED_MODE, STILL_ACTIVE},
        Graphics::{
            Direct3D::D3D_DRIVER_TYPE_HARDWARE,
            Direct3D11::{
                D3D11_BIND_FLAG, D3D11_BIND_SHADER_RESOURCE, D3D11_BOX,
                D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_SDK_VERSION, D3D11_TEXTURE2D_DESC,
                D3D11_USAGE_DEFAULT, D3D11CreateDevice, ID3D11Device, ID3D11DeviceContext,
                ID3D11Resource, ID3D11Texture2D,
            },
            Dxgi::{
                Common::{DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_SAMPLE_DESC},
                IDXGIDevice,
            },
        },
        System::{
            Threading::{GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION},
            WinRT::{
                Direct3D11::{CreateDirect3D11DeviceFromDXGIDevice, IDirect3DDxgiInterfaceAccess},
                Graphics::Capture::IGraphicsCaptureItemInterop,
                RO_INIT_MULTITHREADED, RoInitialize, RoUninitialize,
            },
        },
        UI::WindowsAndMessaging::{GetWindowThreadProcessId, IsWindow},
    },
    core::{IInspectable, Interface, factory},
};

use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    contract::{MediaKind, MemoryDomain, OutputContract, PortContract},
    control::{ControlReceiver, drain_control},
    element::{Element, ElementType, Source, SourceElement, element_pp_log},
    error::{D3d11FrameWrapError, Result},
    pad::SrcPad,
    platform::windows::d3d11va::wrap_d3d11_texture,
    pool::UnboundObjectPool,
    pp_log::{PpLog, pp_error, pp_info},
    schedule::PeriodicSchedule,
};

const PIXEL_FORMAT: DirectXPixelFormat = DirectXPixelFormat::B8G8R8A8UIntNormalized;
const FRAME_POOL_BUFFERS: i32 = 2;
const POLL_GRANULARITY: Duration = Duration::from_millis(100);

/// Errors specific to [`WgcCaptureSource`].
#[derive(Debug, ThisError)]
pub enum WgcCaptureSourceError {
    /// FFmpeg could not allocate the owner used to wrap an output texture.
    #[error(transparent)]
    FrameWrap(#[from] D3d11FrameWrapError),
    /// A WinRT, D3D11, DXGI, or Win32 operation failed.
    #[error("windows error: {0}")]
    Windows(#[from] windows::core::Error),
    /// The supplied handle does not currently identify a window.
    #[error("the supplied HWND doesn't identify a live window")]
    InvalidWindow,
    /// A fixed output cadence cannot be constructed from zero frames per second.
    #[error("WgcCaptureOptions::fps must be greater than zero")]
    InvalidFps,
    /// Windows Graphics Capture is unavailable in this Windows session.
    #[error("Windows Graphics Capture isn't supported in this Windows session")]
    Unsupported,
    /// WGC interop requires a BGRA-capable D3D11 device.
    #[error("the supplied D3D11 device wasn't created with D3D11_CREATE_DEVICE_BGRA_SUPPORT")]
    MissingBgraSupport,
    /// WGC reported a frame with unusable visible dimensions.
    #[error("Windows Graphics Capture reported an invalid content size {width}x{height}")]
    InvalidContentSize { width: i32, height: i32 },
    /// A frame-pool surface unexpectedly came from another D3D11 device.
    #[error("Windows Graphics Capture returned a texture from another D3D11 device")]
    DeviceMismatch,
    /// WGC returned a surface format other than the BGRA format requested at construction.
    #[error("Windows Graphics Capture returned unsupported texture format {0:?}")]
    UnsupportedTextureFormat(DXGI_FORMAT),
    /// The backing texture cannot contain the frame's visible content.
    #[error(
        "Windows Graphics Capture texture {texture_width}x{texture_height} is smaller than visible content {content_width}x{content_height}"
    )]
    TextureTooSmall {
        texture_width: u32,
        texture_height: u32,
        content_width: u32,
        content_height: u32,
    },
    /// The frame callback disappeared while the capture session was still live.
    #[error("Windows Graphics Capture frame notifications stopped unexpectedly")]
    FrameNotificationsStopped,
    /// Seeking was requested on a live window capture.
    #[error("WgcCaptureSource doesn't support seeking a live capture")]
    SeekUnsupported,
}

/// Construction options for [`WgcCaptureSource::open`].
#[derive(Debug, Clone)]
pub struct WgcCaptureOptions {
    /// Constant rate at which the latest captured window image is emitted.
    pub fps: u32,
    /// Whether Windows should include the mouse cursor in captured frames.
    pub include_cursor: bool,
}

impl Default for WgcCaptureOptions {
    fn default() -> Self {
        Self {
            fps: 30,
            include_cursor: true,
        }
    }
}

/// Captures one existing Win32 window through Windows Graphics Capture.
///
/// The caller selects the target and passes its `HWND`; this element does not
/// show `GraphicsCapturePicker` or own application UI. It creates the WGC item
/// on its source thread, where it also owns the WinRT apartment, frame pool,
/// capture session, and event registrations for their complete lifetime.
/// Dropping or stopping the pipeline therefore releases every callback and
/// capture object before that thread exits — with one deliberate exception,
/// documented on this module's `WgcRuntime::drop`: when the process that owned
/// the target window has itself exited, the capture session is leaked rather
/// than released, because releasing it there deadlocks the source thread
/// permanently.
///
/// One source pad emits BGRA [`MediaBuffer::Video`] frames in
/// [`MemoryDomain::D3d11`]. [`Self::open`] returns the exact device it creates;
/// [`Self::open_with_device`] instead uses a caller-owned shared device. Every
/// downstream D3D11 element interacting with these textures must use that same
/// device and its shared immediate context.
/// Each WGC update is copied once into a new immutable texture, so later WGC
/// updates can never mutate a frame while downstream `Arc` clones still exist.
/// When the fixed output cadence repeats an unchanged image, only the small
/// FFmpeg metadata wrapper is new; those frames share the same texture while
/// carrying independent PTS values.
///
/// WGC itself is change-driven, but this source emits the most recent image at
/// a constant [`WgcCaptureOptions::fps`] cadence. PTS values are consecutive
/// ticks in [`Self::time_base`]. Window resizing is handled in place: the WGC
/// frame pool and this source's latest-image texture are recreated, and later
/// frames carry the new visible dimensions. BGRA frames are tagged RGB/full
/// range.
///
/// The target window closing is the natural end of this finite capture and
/// forwards [`MediaBuffer::Eos`]. `Stop` abandons immediately, while
/// [`Pipeline::finish`](crate::pipeline::Pipeline::finish) forwards ordered EOS
/// through the normal source-control path. The source is live and not seekable.
/// A downstream push error is posted to the source [`Bus`] and only that frame
/// is dropped; WGC/session/device failures are fatal because capture cannot
/// meaningfully continue.
pub struct WgcCaptureSource {
    pp_log: PpLog,
    name: Arc<str>,
    /// Raw HWND value. Stored as an integer because windows-rs deliberately
    /// does not mark borrowed Win32 handles `Send`; this element does not own
    /// the window, and reconstructs the by-value handle only on its source
    /// thread after revalidating it.
    hwnd: usize,
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    include_cursor: bool,
    fps: i32,
    frame_interval: Duration,
    frame_index: i64,
    pad: SrcPad,
    frame_pool: UnboundObjectPool<ffmpeg::frame::Video>,
}

impl WgcCaptureSource {
    /// Validates `hwnd`, creates the capture's D3D11 device, and returns both
    /// the source and a cloned device reference for downstream construction.
    /// WGC activation itself happens in [`SourceElement::run`] so its WinRT
    /// apartment is initialized and torn down on the same source thread.
    pub fn open(
        name: impl Into<String>,
        hwnd: HWND,
        options: WgcCaptureOptions,
    ) -> std::result::Result<(Self, ID3D11Device), WgcCaptureSourceError> {
        validate_open(hwnd, &options)?;

        let mut device = None;
        let mut context = None;
        // SAFETY: null adapter/software pointers select the default hardware
        // adapter, optional feature levels use D3D defaults, and both output
        // slots are correctly typed and live. BGRA support is required by the
        // WinRT Direct3D interop device built from this D3D11 device in `run`.
        unsafe {
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                HMODULE::default(),
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                None,
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut context),
            )?;
        }
        let device = device.expect("D3D11CreateDevice succeeded without producing a device");
        let context = context.expect("D3D11CreateDevice succeeded without producing a context");
        let returned_device = device.clone();
        let source = Self::from_device(name, hwnd, options, device, context);

        Ok((source, returned_device))
    }

    /// Uses a caller-owned D3D11 device for capture, allowing capture,
    /// filtering, compositing, encoding, and rendering to share one device.
    ///
    /// The device must have been created with
    /// `D3D11_CREATE_DEVICE_BGRA_SUPPORT`. WGC surfaces and emitted frames
    /// remain GPU-resident on this exact device; each new captured image still
    /// needs one GPU copy to separate its lifetime from WGC's reusable frame
    /// pool surface.
    pub fn open_with_device(
        name: impl Into<String>,
        hwnd: HWND,
        options: WgcCaptureOptions,
        device: &ID3D11Device,
    ) -> std::result::Result<Self, WgcCaptureSourceError> {
        validate_open(hwnd, &options)?;
        // SAFETY: this reads immutable creation metadata from a live device.
        let creation_flags = unsafe { device.GetCreationFlags() };
        if creation_flags & D3D11_CREATE_DEVICE_BGRA_SUPPORT.0 as u32 == 0 {
            return Err(WgcCaptureSourceError::MissingBgraSupport);
        }
        // SAFETY: returns the shared immediate context owned by this device.
        let context = unsafe { device.GetImmediateContext()? };
        Ok(Self::from_device(
            name,
            hwnd,
            options,
            device.clone(),
            context,
        ))
    }

    fn from_device(
        name: impl Into<String>,
        hwnd: HWND,
        options: WgcCaptureOptions,
        device: ID3D11Device,
        context: ID3D11DeviceContext,
    ) -> Self {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::WgcCaptureSource, &name, None);
        let fps = options.fps as i32;
        let pad = SrcPad::with_contract(format!("{name}_src"), output_contract());

        Self {
            pp_log,
            name,
            hwnd: hwnd.0 as usize,
            device,
            context,
            include_cursor: options.include_cursor,
            fps,
            frame_interval: Duration::from_secs_f64(1.0 / f64::from(options.fps)),
            frame_index: 0,
            pad,
            frame_pool: UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {}),
        }
    }

    /// PTS unit of frames emitted by this source.
    pub fn time_base(&self) -> ffmpeg::Rational {
        ffmpeg::Rational::new(1, self.fps)
    }

    fn create_texture(
        &self,
        width: u32,
        height: u32,
        bind_flags: D3D11_BIND_FLAG,
    ) -> std::result::Result<ID3D11Texture2D, WgcCaptureSourceError> {
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: bind_flags.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        let mut texture = None;
        // SAFETY: `desc` is fully initialized for an ordinary single-sample
        // texture, initial data is absent, and the output slot is live.
        unsafe {
            self.device
                .CreateTexture2D(&desc, None, Some(&mut texture))?;
        }
        Ok(texture.expect("CreateTexture2D succeeded without producing a texture"))
    }

    fn receive_latest(
        &self,
        runtime: &WgcRuntime,
        latest: &mut Option<ID3D11Texture2D>,
        visible_size: &mut Option<(u32, u32)>,
    ) -> std::result::Result<(), WgcCaptureSourceError> {
        let frame = CaptureFrameGuard(runtime.frame_pool.TryGetNextFrame()?);
        let size = frame.0.ContentSize()?;
        if size.Width <= 0 || size.Height <= 0 {
            // A minimized window can transiently report an empty client
            // area. Keep the last valid image and let the next change-driven
            // notification recover capture instead of terminating the source.
            return Ok(());
        }
        let width = size.Width as u32;
        let height = size.Height as u32;
        let surface = frame.0.Surface()?;
        let access: IDirect3DDxgiInterfaceAccess = surface.cast()?;
        // SAFETY: `surface` is a live WGC Direct3D surface for the lifetime of
        // `frame`; `GetInterface` returns an independently owned COM reference.
        let source: ID3D11Texture2D = unsafe { access.GetInterface()? };
        // SAFETY: the texture is live and returns an owned creator-device ref.
        let source_device = unsafe { source.GetDevice() }?;
        if source_device.as_raw() != self.device.as_raw() {
            return Err(WgcCaptureSourceError::DeviceMismatch);
        }
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        // SAFETY: `desc` is a live out-parameter for the live texture.
        unsafe { source.GetDesc(&mut desc) };
        if desc.Format != DXGI_FORMAT_B8G8R8A8_UNORM {
            return Err(WgcCaptureSourceError::UnsupportedTextureFormat(desc.Format));
        }
        if desc.Width < width || desc.Height < height {
            return Err(WgcCaptureSourceError::TextureTooSmall {
                texture_width: desc.Width,
                texture_height: desc.Height,
                content_width: width,
                content_height: height,
            });
        }

        // WGC owns and reuses `source`. Copy once into a fresh texture whose
        // contents are never changed again; cadence repeats can safely share
        // this texture until a later WGC update replaces `latest`.
        let captured = self.create_texture(width, height, D3D11_BIND_SHADER_RESOURCE)?;
        let destination: ID3D11Resource = captured.cast()?;
        let source: ID3D11Resource = source.cast()?;
        let source_box = D3D11_BOX {
            left: 0,
            top: 0,
            front: 0,
            right: width,
            bottom: height,
            back: 1,
        };
        // SAFETY: both resources belong to `self.device`; the destination was
        // allocated to the visible size and the validated source contains the
        // exact source box.
        unsafe {
            self.context.CopySubresourceRegion(
                &destination,
                0,
                0,
                0,
                0,
                &source,
                0,
                Some(&source_box),
            )
        };
        drop(frame);
        *latest = Some(captured);
        *visible_size = Some((width, height));

        if runtime.size()
            != (SizeInt32 {
                Width: width as i32,
                Height: height as i32,
            })
        {
            runtime.recreate(SizeInt32 {
                Width: width as i32,
                Height: height as i32,
            })?;
        }
        Ok(())
    }

    fn emit_frame(
        &mut self,
        latest: &ID3D11Texture2D,
        width: u32,
        height: u32,
    ) -> std::result::Result<
        crate::pool::UnboundObjectPoolRef<ffmpeg::frame::Video>,
        WgcCaptureSourceError,
    > {
        let mut frame = self.frame_pool.get();
        *frame = wrap_d3d11_texture(latest.clone(), width, height)?;
        frame.set_pts(Some(self.frame_index));
        frame.set_color_space(ffmpeg::color::Space::RGB);
        frame.set_color_range(ffmpeg::color::Range::JPEG);
        self.frame_index += 1;
        Ok(frame)
    }

    fn push_eos(&mut self) -> Result<()> {
        self.pad.push(MediaBuffer::Eos)
    }
}

impl Element for WgcCaptureSource {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::WgcCaptureSource
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for WgcCaptureSource {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl SourceElement for WgcCaptureSource {
    fn is_live(&self) -> bool {
        true
    }

    fn is_seekable(&self) -> bool {
        false
    }

    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        let _apartment = WinRtApartment::initialize()?;
        let hwnd = HWND(self.hwnd as *mut _);
        // The caller can close or destroy the selected window after `open` and
        // before the source thread starts; reject that stale handle before
        // creating any WGC object.
        if !unsafe { IsWindow(Some(hwnd)) }.as_bool() {
            return Err(WgcCaptureSourceError::InvalidWindow.into());
        }
        let runtime = WgcRuntime::start(hwnd, &self.device, self.include_cursor)
            .inspect_err(|error| pp_error!(self, "capture start failed: {error}"))?;
        pp_info!(
            self,
            "started: window={:?}, fps={}, include_cursor={}",
            hwnd,
            self.fps,
            self.include_cursor
        );

        let mut latest = None;
        let mut visible_size = None;
        let mut schedule = PeriodicSchedule::new(self.frame_interval, Instant::now());
        // `Closed` is the documented signal, but it does not reliably fire
        // for every way a target window goes away (observed: neither a
        // forceful process kill nor a plain `WM_CLOSE` ever raised it in
        // practice, leaving the loop parked with no EOS and no error). Treat
        // the handle no longer resolving as equally conclusive; this is
        // cheap enough to check on every wake-up, which happens at least
        // every `POLL_GRANULARITY` even without new frames.
        let target_gone = |runtime: &WgcRuntime| -> bool {
            // SAFETY: reads only whether `hwnd` still identifies a window;
            // it retains no caller storage.
            runtime.closed.load(Ordering::Acquire) || !unsafe { IsWindow(Some(hwnd)) }.as_bool()
        };
        loop {
            let outcome = drain_control(control, self, bus)?;
            if outcome.stopped {
                pp_info!(self, "stopped");
                return Ok(());
            }
            if outcome.paused_for > Duration::ZERO {
                schedule.resume_after_pause(outcome.paused_for, Instant::now());
            }
            if target_gone(&runtime) {
                pp_info!(self, "target window closed; forwarding EOS");
                return self.push_eos();
            }

            let timeout = schedule.remaining(Instant::now()).min(POLL_GRANULARITY);
            match runtime.frame_rx.recv_timeout(timeout) {
                Ok(()) => {
                    // `Closed` shares this bounded wake-up channel with
                    // `FrameArrived`. Do not touch the frame pool after the
                    // close notification won the race.
                    if target_gone(&runtime) {
                        pp_info!(self, "target window closed; forwarding EOS");
                        return self.push_eos();
                    }
                    self.receive_latest(&runtime, &mut latest, &mut visible_size)
                        .inspect_err(|error| pp_error!(self, "capture frame failed: {error}"))?;
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(WgcCaptureSourceError::FrameNotificationsStopped.into());
                }
            }

            if target_gone(&runtime) {
                pp_info!(self, "target window closed; forwarding EOS");
                return self.push_eos();
            }
            if !schedule.is_due(Instant::now()) {
                continue;
            }
            let Some((width, height)) = visible_size else {
                schedule.advance_after_tick(Instant::now());
                continue;
            };
            let frame = self
                .emit_frame(
                    latest
                        .as_ref()
                        .expect("visible size is set only with a latest texture"),
                    width,
                    height,
                )
                .inspect_err(|error| pp_error!(self, "emit frame failed: {error}"))?;
            if let Err(error) = self.pad.push(MediaBuffer::Video(Arc::new(frame))) {
                bus.post(
                    &self.pp_log,
                    BusEvent::Error {
                        element_type: ElementType::WgcCaptureSource,
                        name: self.name.clone(),
                        error,
                    },
                );
            }
            schedule.advance_after_tick(Instant::now());
        }
    }

    fn seek(&mut self, _target: Duration) -> Result<Duration> {
        Err(WgcCaptureSourceError::SeekUnsupported.into())
    }
}

fn validate_open(
    hwnd: HWND,
    options: &WgcCaptureOptions,
) -> std::result::Result<(), WgcCaptureSourceError> {
    if options.fps == 0 {
        return Err(WgcCaptureSourceError::InvalidFps);
    }
    // SAFETY: `IsWindow` only inspects the by-value handle and retains no
    // caller storage. A stale or null handle simply returns false.
    if !unsafe { IsWindow(Some(hwnd)) }.as_bool() {
        return Err(WgcCaptureSourceError::InvalidWindow);
    }
    Ok(())
}

fn output_contract() -> OutputContract {
    OutputContract::Fixed(PortContract::frame(
        MediaKind::VideoFrame,
        MemoryDomain::D3d11,
    ))
}

/// Whether the process that owned the capture target has exited. A pid of
/// zero, an unopenable process, or an unreadable exit code all answer `false`:
/// the caller uses this only to skip an otherwise-deadlocking teardown, so
/// anything short of proof that the owner is gone must take the normal path.
fn owner_exited(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    // SAFETY: opens the pid for a query-only handle; failure is reported as
    // `Err` and retains nothing.
    let Ok(process) = (unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) })
    else {
        // The pid can no longer be opened at all, which for a process that
        // existed moments ago means it is gone.
        return true;
    };
    let mut code = 0u32;
    // SAFETY: `process` is a live handle opened just above and `code` is a
    // live out-parameter.
    let read = unsafe { GetExitCodeProcess(process, &mut code) };
    // SAFETY: `process` came from `OpenProcess` and is not used afterwards.
    let _ = unsafe { CloseHandle(process) };
    read.is_ok() && code != STILL_ACTIVE.0 as u32
}

struct CaptureFrameGuard(Direct3D11CaptureFrame);

impl Drop for CaptureFrameGuard {
    fn drop(&mut self) {
        let _ = self.0.Close();
    }
}

struct WinRtApartment {
    uninitialize: bool,
}

impl WinRtApartment {
    fn initialize() -> std::result::Result<Self, WgcCaptureSourceError> {
        // SAFETY: initializes WinRT for this source thread. A successful call
        // is balanced by this guard's Drop on the same thread. Changed mode
        // means the caller already initialized another apartment, which is
        // still sufficient for free-threaded WGC objects and must not be
        // uninitialized by this library.
        match unsafe { RoInitialize(RO_INIT_MULTITHREADED) } {
            Ok(()) => Ok(Self { uninitialize: true }),
            Err(error) if error.code() == RPC_E_CHANGED_MODE => Ok(Self {
                uninitialize: false,
            }),
            Err(error) => Err(error.into()),
        }
    }
}

impl Drop for WinRtApartment {
    fn drop(&mut self) {
        if self.uninitialize {
            // SAFETY: balances this guard's successful `RoInitialize` on the
            // same source thread after every WGC object has been dropped.
            unsafe { RoUninitialize() };
        }
    }
}

struct WgcRuntime {
    /// The target window's owning process, read while the window was still
    /// alive. Teardown needs it because a destroyed window and a dead owner
    /// call for opposite handling — see [`Drop`].
    owner_pid: u32,
    item: GraphicsCaptureItem,
    frame_pool: Direct3D11CaptureFramePool,
    /// Held in a [`ManuallyDrop`](std::mem::ManuallyDrop) so [`Drop`] can
    /// decline to release it — see that impl for why a destroyed target
    /// makes releasing this object unsafe to do on this thread.
    session: std::mem::ManuallyDrop<GraphicsCaptureSession>,
    direct3d_device: IDirect3DDevice,
    frame_token: Option<i64>,
    closed_token: Option<i64>,
    frame_rx: Receiver<()>,
    closed: Arc<AtomicBool>,
    size: std::cell::Cell<SizeInt32>,
}

impl WgcRuntime {
    fn start(
        hwnd: HWND,
        device: &ID3D11Device,
        include_cursor: bool,
    ) -> std::result::Result<Self, WgcCaptureSourceError> {
        if !GraphicsCaptureSession::IsSupported()? {
            return Err(WgcCaptureSourceError::Unsupported);
        }
        let interop: IGraphicsCaptureItemInterop = factory::<GraphicsCaptureItem, _>()?;
        // SAFETY: `hwnd` was validated during construction and remains the
        // caller-selected capture target. The returned item owns its refs.
        let item: GraphicsCaptureItem = unsafe { interop.CreateForWindow(hwnd)? };
        let size = item.Size()?;
        if size.Width <= 0 || size.Height <= 0 {
            return Err(WgcCaptureSourceError::InvalidContentSize {
                width: size.Width,
                height: size.Height,
            });
        }

        let dxgi_device: IDXGIDevice = device.cast()?;
        // SAFETY: the DXGI interface belongs to the live BGRA-capable D3D11
        // device constructed by `open`; the returned inspectable owns a ref.
        let inspectable = unsafe { CreateDirect3D11DeviceFromDXGIDevice(&dxgi_device)? };
        let direct3d_device: IDirect3DDevice = inspectable.cast()?;
        let frame_pool = Direct3D11CaptureFramePool::CreateFreeThreaded(
            &direct3d_device,
            PIXEL_FORMAT,
            FRAME_POOL_BUFFERS,
            size,
        )?;
        let session = frame_pool.CreateCaptureSession(&item)?;
        if !include_cursor {
            session.SetIsCursorCaptureEnabled(false)?;
        }

        let (frame_tx, frame_rx) = bounded(1);
        let closed = Arc::new(AtomicBool::new(false));
        let mut owner_pid = 0;
        // SAFETY: `hwnd` is still a live window here, and `owner_pid` is a
        // live out-parameter. A failure leaves it zero, which `owner_exited`
        // treats as "cannot tell", i.e. the safe full-teardown path.
        unsafe { GetWindowThreadProcessId(hwnd, Some(&mut owner_pid)) };
        let mut runtime = Self {
            owner_pid,
            item,
            frame_pool,
            session: std::mem::ManuallyDrop::new(session),
            direct3d_device,
            frame_token: None,
            closed_token: None,
            frame_rx,
            closed,
            size: std::cell::Cell::new(size),
        };

        let closed_wake = frame_tx.clone();
        runtime.frame_token =
            Some(runtime.frame_pool.FrameArrived(&TypedEventHandler::<
                Direct3D11CaptureFramePool,
                IInspectable,
            >::new(move |_, _| {
                let _ = frame_tx.try_send(());
                Ok(())
            }))?);
        let closed = runtime.closed.clone();
        runtime.closed_token = Some(runtime.item.Closed(&TypedEventHandler::<
            GraphicsCaptureItem,
            IInspectable,
        >::new(move |_, _| {
            closed.store(true, Ordering::Release);
            let _ = closed_wake.try_send(());
            Ok(())
        }))?);
        runtime.session.StartCapture()?;
        Ok(runtime)
    }

    fn size(&self) -> SizeInt32 {
        self.size.get()
    }

    fn recreate(&self, size: SizeInt32) -> std::result::Result<(), WgcCaptureSourceError> {
        self.frame_pool.Recreate(
            &self.direct3d_device,
            PIXEL_FORMAT,
            FRAME_POOL_BUFFERS,
            size,
        )?;
        self.size.set(size);
        Ok(())
    }
}

impl Drop for WgcRuntime {
    fn drop(&mut self) {
        if let Some(token) = self.frame_token.take() {
            let _ = self.frame_pool.RemoveFrameArrived(token);
        }
        if let Some(token) = self.closed_token.take() {
            let _ = self.item.RemoveClosed(token);
        }
        let _ = self.frame_pool.Close();

        // Once the target's *process* is gone, every way of retiring the
        // capture session blocks this thread forever: `Close` and the final
        // `Release` both deadlock, and neither unblocks after two minutes.
        // Detaching the handlers first, closing the frame pool first, and
        // giving WGC its own device were each measured and changed nothing.
        // So the session is deliberately leaked on that one path — a COM
        // reference the process reclaims at exit, in exchange for a source
        // thread that actually finishes. Everything else still releases
        // normally.
        //
        // A destroyed window whose owner is still running is *not* that case:
        // there the session retires in milliseconds, and leaking it instead
        // crashes the process on the way out. Hence the owner check rather
        // than a window check alone.
        if owner_exited(self.owner_pid) {
            return;
        }
        let _ = self.session.Close();
        // SAFETY: the session is released exactly once, only on the path that
        // did not take the early return above, and never touched afterwards.
        unsafe { std::mem::ManuallyDrop::drop(&mut self.session) };
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::mpsc, thread, time::Duration};

    use windows::{
        Win32::UI::WindowsAndMessaging::{
            CreateWindowExW, DestroyWindow, WINDOW_EX_STYLE, WS_OVERLAPPEDWINDOW, WS_VISIBLE,
        },
        core::w,
    };

    use super::*;
    use crate::{
        Error, bus::BusEvent, contract::InputContract, elements::AppSink, pipeline::Pipeline,
        platform::windows::d3d11va::d3d11va_texture,
    };

    struct TestWindow(HWND);

    impl TestWindow {
        fn create() -> windows::core::Result<Self> {
            // The built-in STATIC class avoids registering process-global
            // test state. The window stays owned by this test thread and is
            // destroyed before the thread returns.
            let hwnd = unsafe {
                CreateWindowExW(
                    WINDOW_EX_STYLE::default(),
                    w!("STATIC"),
                    w!("media-pp WGC test window"),
                    WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                    0,
                    0,
                    320,
                    240,
                    None,
                    None,
                    None,
                    None,
                )?
            };
            Ok(Self(hwnd))
        }
    }

    impl Drop for TestWindow {
        fn drop(&mut self) {
            let _ = unsafe { DestroyWindow(self.0) };
        }
    }

    #[test]
    fn rejects_zero_fps_before_touching_the_window_or_device() {
        let result = WgcCaptureSource::open(
            "capture",
            HWND::default(),
            WgcCaptureOptions {
                fps: 0,
                ..WgcCaptureOptions::default()
            },
        );
        assert!(matches!(result, Err(WgcCaptureSourceError::InvalidFps)));
    }

    #[test]
    fn rejects_a_stale_window_handle_before_creating_a_device() {
        let result =
            WgcCaptureSource::open("capture", HWND::default(), WgcCaptureOptions::default());
        assert!(matches!(result, Err(WgcCaptureSourceError::InvalidWindow)));
    }

    #[test]
    fn contract_accepts_d3d11_video_and_refuses_system_memory() {
        let OutputContract::Fixed(produced) = output_contract() else {
            panic!("WGC output must be fixed at construction");
        };
        let d3d11 = InputContract::Fixed(PortContract::frame(
            MediaKind::VideoFrame,
            MemoryDomain::D3d11,
        ));
        let system = InputContract::Fixed(PortContract::frame(
            MediaKind::VideoFrame,
            MemoryDomain::System,
        ));
        let InputContract::Fixed(d3d11) = d3d11 else {
            unreachable!()
        };
        let InputContract::Fixed(system) = system else {
            unreachable!()
        };
        assert!(d3d11.accepts(&produced));
        assert!(!system.accepts(&produced));
    }

    #[test]
    #[ignore = "requires an interactive Windows Graphics Capture session"]
    fn captures_after_one_downstream_frame_failure() {
        crate::init().expect("initialize FFmpeg");
        let window = TestWindow::create().expect("create capture target window");
        let (created, device) = match WgcCaptureSource::open(
            "capture",
            window.0,
            WgcCaptureOptions {
                fps: 30,
                include_cursor: false,
            },
        ) {
            Ok(opened) => opened,
            Err(error) => {
                eprintln!("skipping: cannot create WGC source ({error})");
                return;
            }
        };
        drop(created);
        let mut source = WgcCaptureSource::open_with_device(
            "capture",
            window.0,
            WgcCaptureOptions {
                fps: 30,
                include_cursor: false,
            },
            &device,
        )
        .expect("the device created by WGC open must be reusable");

        let latest = source
            .create_texture(2, 2, D3D11_BIND_SHADER_RESOURCE)
            .expect("create immutable latest-image texture");
        let first = source.emit_frame(&latest, 2, 2).expect("wrap first tick");
        let second = source.emit_frame(&latest, 2, 2).expect("wrap repeat tick");
        assert_eq!(first.pts(), Some(0));
        assert_eq!(second.pts(), Some(1));
        assert_eq!(
            d3d11va_texture(&first).expect("first D3D11 texture").0,
            d3d11va_texture(&second).expect("second D3D11 texture").0,
            "cadence repeats must share the immutable latest texture"
        );
        drop(first);
        drop(second);
        let (accepted_tx, accepted_rx) = mpsc::sync_channel(1);
        let mut fail_first = true;
        let sink = AppSink::new("sink", move |buffer| {
            if !matches!(buffer, MediaBuffer::Video(_)) {
                return Ok(());
            }
            if fail_first {
                fail_first = false;
                return Err(Error::Other("simulated first-frame failure".into()));
            }
            let _ = accepted_tx.try_send(());
            Ok(())
        });
        let pipeline = Pipeline::new("wgc-recovery", source, |source, ctx| {
            let branch = ctx.branch().to(Box::new(sink))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("wire WGC pipeline");

        pipeline.run().expect("start WGC pipeline");
        accepted_rx
            .recv_timeout(Duration::from_secs(3))
            .expect("a later frame must pass after one downstream failure");
        pipeline.stop();
        let events: Vec<_> = pipeline.bus().iter().collect();
        let errors = events
            .iter()
            .filter(|event| matches!(event, BusEvent::Error { .. }))
            .count();
        assert_eq!(errors, 1, "only the simulated push failure is expected");
    }

    /// Destroying the target must end the source, not park it: `Closed` does
    /// not fire even for this in-process `DestroyWindow`, so without the
    /// handle check the loop waits forever. The bus draining at all is what
    /// proves the thread finished — `iter` only returns once every sender is
    /// gone.
    ///
    /// This does not cover the teardown deadlock [`WgcRuntime::drop`]
    /// describes. That one needs the target's whole *process* to die, which
    /// an in-process window cannot reproduce; destroying a window whose owner
    /// is still alive retires the session normally.
    #[test]
    #[ignore = "requires an interactive Windows Graphics Capture session"]
    fn destroying_the_target_window_ends_the_source() {
        crate::init().expect("initialize FFmpeg");
        let window = TestWindow::create().expect("create capture target window");
        let (source, _device) = match WgcCaptureSource::open(
            "capture",
            window.0,
            WgcCaptureOptions {
                fps: 30,
                include_cursor: false,
            },
        ) {
            Ok(opened) => opened,
            Err(error) => {
                eprintln!("skipping: cannot create WGC source ({error})");
                return;
            }
        };
        let pipeline = Pipeline::new("wgc-target-closed", source, |source, ctx| {
            let branch = ctx
                .branch()
                .to(Box::new(AppSink::new("sink", |_| Ok(()))))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("wire WGC pipeline");

        pipeline.run().expect("start WGC pipeline");
        // Let the session actually start before the target goes away, so this
        // exercises teardown of a running capture rather than a failed start.
        thread::sleep(Duration::from_millis(500));
        drop(window);

        let (done_tx, done_rx) = mpsc::sync_channel(1);
        let draining = Arc::clone(&pipeline);
        thread::spawn(move || {
            let saw_eos = draining
                .bus()
                .iter()
                .any(|event| matches!(event, BusEvent::Eos { .. }));
            let _ = done_tx.send(saw_eos);
        });
        let saw_eos = done_rx
            .recv_timeout(Duration::from_secs(15))
            .expect("the source thread must finish once its target window is destroyed");
        assert!(saw_eos, "a destroyed target window must forward EOS");
    }
}
