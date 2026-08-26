use std::{
    cell::RefCell,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
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
        Foundation::{CloseHandle, HANDLE, HMODULE, HWND, RPC_E_CHANGED_MODE, WAIT_OBJECT_0},
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
            Threading::{OpenProcess, PROCESS_SYNCHRONIZE, WaitForSingleObject},
            WinRT::{
                Direct3D11::{CreateDirect3D11DeviceFromDXGIDevice, IDirect3DDxgiInterfaceAccess},
                Graphics::Capture::IGraphicsCaptureItemInterop,
                RO_INIT_MULTITHREADED, RoInitialize, RoUninitialize,
            },
        },
        UI::{
            Accessibility::{HWINEVENTHOOK, SetWinEventHook, UnhookWinEvent},
            WindowsAndMessaging::{
                CHILDID_SELF, DispatchMessageW, EVENT_OBJECT_DESTROY, GetWindowThreadProcessId,
                MSG, OBJID_WINDOW, PM_REMOVE, PeekMessageW, TranslateMessage,
                WINEVENT_OUTOFCONTEXT,
            },
        },
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
    platform::windows::{
        d3d11::{MultithreadProtectionError, enable_multithread_protection},
        d3d11va::wrap_d3d11_texture,
    },
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
    /// The captured window went away while the capture was running.
    ///
    /// Reopening is the caller's decision, the same contract
    /// `DxgiCaptureSourceError::AccessLost` and
    /// `PipeWireScreenCaptureSourceError::SourceGone` set for the other
    /// capture sources.
    #[error("the captured window is gone")]
    TargetGone,
    /// A fixed output cadence cannot be constructed from zero frames per second.
    #[error("WgcCaptureOptions::fps must be greater than zero")]
    InvalidFps,
    /// Windows Graphics Capture is unavailable in this Windows session.
    #[error("Windows Graphics Capture isn't supported in this Windows session")]
    Unsupported,
    /// WGC interop requires a BGRA-capable D3D11 device.
    #[error("the supplied D3D11 device wasn't created with D3D11_CREATE_DEVICE_BGRA_SUPPORT")]
    MissingBgraSupport,
    /// A device created for use from only one thread cannot cross a Queue boundary.
    #[error(
        "the D3D11 device was created with D3D11_CREATE_DEVICE_SINGLETHREADED and cannot be shared by capture and downstream elements"
    )]
    SingleThreadedDevice,
    /// The target owner's lifetime could not be monitored safely.
    #[error("cannot monitor the capture target's owner process {pid}: {source}")]
    OwnerProcessUnavailable {
        pid: u32,
        #[source]
        source: windows::core::Error,
    },
    /// The helper thread that watches the original HWND could not start.
    #[error("cannot start the WGC window lifetime watcher: {0}")]
    WindowWatcherUnavailable(String),
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
/// show `GraphicsCapturePicker` or own application UI. Construction starts a
/// lifetime watcher before retaining the selected window's owner identity, so
/// a later HWND reuse cannot redirect capture to another window. Dropping the
/// source stops and joins that owned helper thread. The source thread owns the
/// WinRT apartment used for the frame pool, capture session, and event
/// registrations for their complete lifetime.
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
/// device and its shared immediate context. Construction rejects a
/// `D3D11_CREATE_DEVICE_SINGLETHREADED` device and enables the runtime's
/// immediate-context multithread protection before the source can run.
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
/// Runs until `Stop` — a live capture has no natural end, so this never
/// reaches [`MediaBuffer::Eos`] on its own. The captured window going away
/// ends `run` with [`WgcCaptureSourceError::TargetGone`] instead, which
/// reaches the caller as a [`BusEvent::Error`]; whether that means reopening,
/// [`Pipeline::stop`](crate::pipeline::Pipeline::stop), or
/// [`Pipeline::finish`](crate::pipeline::Pipeline::finish) is the caller's
/// decision, the same "fail fast, caller rebuilds" contract
/// `DxgiCaptureSource` and `PipeWireScreenCaptureSource` follow.
/// `Pipeline::finish` still forwards
/// ordered EOS through the normal source-control path. The source is live and
/// not seekable. A downstream push error is posted to the source [`Bus`] and
/// only that frame is dropped; WGC/session/device failures are fatal because
/// capture cannot meaningfully continue.
pub struct WgcCaptureSource {
    pp_log: PpLog,
    name: Arc<str>,
    /// Raw HWND value. Stored as an integer because windows-rs deliberately
    /// does not mark borrowed Win32 handles `Send`; this element does not own
    /// the window, and reconstructs the by-value handle only on its source
    /// thread only after the original window's lifetime and owner identity
    /// were retained.
    hwnd: usize,
    owner_pid: u32,
    owner_thread_id: u32,
    owner_process: ProcessHandle,
    target_watch: WindowLifetimeWatch,
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
    /// Validates `hwnd`, starts its lifetime watcher, creates the capture's D3D11
    /// device, and returns both the source and a cloned device reference for
    /// downstream construction. The frame pool and session are activated in
    /// [`SourceElement::run`] on their owning source thread.
    pub fn open(
        name: impl Into<String>,
        hwnd: HWND,
        options: WgcCaptureOptions,
    ) -> std::result::Result<(Self, ID3D11Device), WgcCaptureSourceError> {
        validate_options(&options)?;
        let target = CaptureTarget::open(hwnd)?;

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
        protect_context(&device, &context)?;
        let returned_device = device.clone();
        let source = Self::from_device(name, target, options, device, context);

        Ok((source, returned_device))
    }

    /// Uses a caller-owned D3D11 device for capture, allowing capture,
    /// filtering, compositing, encoding, and rendering to share one device.
    ///
    /// The device must have been created with
    /// `D3D11_CREATE_DEVICE_BGRA_SUPPORT` and without
    /// `D3D11_CREATE_DEVICE_SINGLETHREADED`. This method enables its immediate
    /// context's D3D11 runtime multithread protection. WGC surfaces and emitted
    /// frames remain GPU-resident on this exact device; each new captured image
    /// still needs one GPU copy to separate its lifetime from WGC's reusable
    /// frame pool surface.
    pub fn open_with_device(
        name: impl Into<String>,
        hwnd: HWND,
        options: WgcCaptureOptions,
        device: &ID3D11Device,
    ) -> std::result::Result<Self, WgcCaptureSourceError> {
        validate_options(&options)?;
        let target = CaptureTarget::open(hwnd)?;
        // SAFETY: this reads immutable creation metadata from a live device.
        let creation_flags = unsafe { device.GetCreationFlags() };
        if creation_flags & D3D11_CREATE_DEVICE_BGRA_SUPPORT.0 == 0 {
            return Err(WgcCaptureSourceError::MissingBgraSupport);
        }
        // SAFETY: returns the shared immediate context owned by this device.
        let context = unsafe { device.GetImmediateContext()? };
        protect_context(device, &context)?;
        Ok(Self::from_device(
            name,
            target,
            options,
            device.clone(),
            context,
        ))
    }

    fn from_device(
        name: impl Into<String>,
        target: CaptureTarget,
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
            hwnd: target.hwnd,
            owner_pid: target.owner_pid,
            owner_thread_id: target.owner_thread_id,
            owner_process: target.owner_process,
            target_watch: target.watch,
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
        let runtime = WgcRuntime::start(
            hwnd,
            self.owner_pid,
            self.owner_thread_id,
            self.owner_process.raw(),
            self.target_watch.flag(),
            &self.device,
            self.include_cursor,
        )
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
        loop {
            let outcome = drain_control(control, self, bus)?;
            if outcome.stopped {
                pp_info!(self, "stopped");
                return Ok(());
            }
            if outcome.paused_for > Duration::ZERO {
                schedule.resume_after_pause(outcome.paused_for, Instant::now());
            }
            if runtime.target_gone() {
                return Err(WgcCaptureSourceError::TargetGone.into());
            }

            let timeout = schedule.remaining(Instant::now()).min(POLL_GRANULARITY);
            match runtime.frame_rx.recv_timeout(timeout) {
                Ok(()) => {
                    // `Closed` shares this bounded wake-up channel with
                    // `FrameArrived`. Do not touch the frame pool after the
                    // close notification won the race.
                    if runtime.target_gone() {
                        return Err(WgcCaptureSourceError::TargetGone.into());
                    }
                    self.receive_latest(&runtime, &mut latest, &mut visible_size)
                        .inspect_err(|error| pp_error!(self, "capture frame failed: {error}"))?;
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(WgcCaptureSourceError::FrameNotificationsStopped.into());
                }
            }

            if runtime.target_gone() {
                return Err(WgcCaptureSourceError::TargetGone.into());
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

fn validate_options(options: &WgcCaptureOptions) -> std::result::Result<(), WgcCaptureSourceError> {
    if options.fps == 0 {
        return Err(WgcCaptureSourceError::InvalidFps);
    }
    Ok(())
}

fn output_contract() -> OutputContract {
    OutputContract::Fixed(PortContract::frame(
        MediaKind::VideoFrame,
        MemoryDomain::D3d11,
    ))
}

struct ProcessHandle(HANDLE);

// SAFETY: a process synchronization handle is a kernel object reference whose
// wait/close operations are thread-independent. Ownership remains unique while
// this wrapper moves with the source onto its worker thread.
unsafe impl Send for ProcessHandle {}

impl ProcessHandle {
    fn open(pid: u32) -> std::result::Result<Self, WgcCaptureSourceError> {
        // SAFETY: opens one synchronization-only reference to the live owner
        // pid. The returned handle is closed exactly once by `Drop`.
        let handle = unsafe { OpenProcess(PROCESS_SYNCHRONIZE, false, pid) }
            .map_err(|source| WgcCaptureSourceError::OwnerProcessUnavailable { pid, source })?;
        Ok(Self(handle))
    }

    fn raw(&self) -> HANDLE {
        self.0
    }
}

impl Drop for ProcessHandle {
    fn drop(&mut self) {
        // SAFETY: this type owns the successful `OpenProcess` result and does
        // not expose ownership of it elsewhere.
        let _ = unsafe { CloseHandle(self.0) };
    }
}

fn process_has_exited(process: HANDLE) -> bool {
    // SAFETY: every caller supplies the live synchronization handle retained
    // from construction. A zero-time wait observes state without blocking.
    (unsafe { WaitForSingleObject(process, 0) }) == WAIT_OBJECT_0
}

fn target_still_matches(hwnd: HWND, owner_pid: u32, owner_thread_id: u32) -> bool {
    let mut current_pid = 0;
    // SAFETY: reads the identity currently associated with this by-value HWND
    // and writes to the live pid out-parameter without retaining either.
    let current_thread_id = unsafe { GetWindowThreadProcessId(hwnd, Some(&mut current_pid)) };
    current_thread_id == owner_thread_id && current_pid == owner_pid
}

struct CaptureTarget {
    hwnd: usize,
    owner_pid: u32,
    owner_thread_id: u32,
    owner_process: ProcessHandle,
    watch: WindowLifetimeWatch,
}

impl CaptureTarget {
    fn open(hwnd: HWND) -> std::result::Result<Self, WgcCaptureSourceError> {
        // Install the destroy watcher before reading the current owner. This
        // closes the open-to-run reuse window even when Windows recycles the
        // same numeric HWND for another window on the same UI thread.
        let watch = WindowLifetimeWatch::start(hwnd)?;
        let mut owner_pid = 0;
        // SAFETY: this inspects the by-value handle and fills a live pid
        // out-parameter. Zero means the handle did not identify a window.
        let owner_thread_id = unsafe { GetWindowThreadProcessId(hwnd, Some(&mut owner_pid)) };
        if owner_thread_id == 0 || owner_pid == 0 {
            return Err(WgcCaptureSourceError::InvalidWindow);
        }
        let owner_process = ProcessHandle::open(owner_pid)?;

        if watch.destroyed() || !target_still_matches(hwnd, owner_pid, owner_thread_id) {
            return Err(WgcCaptureSourceError::InvalidWindow);
        }

        Ok(Self {
            hwnd: hwnd.0 as usize,
            owner_pid,
            owner_thread_id,
            owner_process,
            watch,
        })
    }
}

fn protect_context(
    device: &ID3D11Device,
    context: &ID3D11DeviceContext,
) -> std::result::Result<(), WgcCaptureSourceError> {
    enable_multithread_protection(device, context).map_err(|error| match error {
        MultithreadProtectionError::SingleThreadedDevice => {
            WgcCaptureSourceError::SingleThreadedDevice
        }
        MultithreadProtectionError::Windows(error) => error.into(),
    })
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

struct DestroyWatchState {
    hwnd: usize,
    destroyed: Arc<AtomicBool>,
}

thread_local! {
    static DESTROY_WATCH: RefCell<Option<DestroyWatchState>> = const { RefCell::new(None) };
}

unsafe extern "system" fn window_event_proc(
    _hook: HWINEVENTHOOK,
    event: u32,
    hwnd: HWND,
    id_object: i32,
    id_child: i32,
    _event_thread: u32,
    _event_time: u32,
) {
    if event != EVENT_OBJECT_DESTROY
        || id_object != OBJID_WINDOW.0
        || id_child != CHILDID_SELF as i32
    {
        return;
    }
    DESTROY_WATCH.with(|watch| {
        if let Some(watch) = watch.borrow().as_ref()
            && watch.hwnd == hwnd.0 as usize
        {
            watch.destroyed.store(true, Ordering::Release);
        }
    });
}

struct WindowDestroyHook {
    hook: HWINEVENTHOOK,
}

impl WindowDestroyHook {
    fn install(
        hwnd: HWND,
        destroyed: Arc<AtomicBool>,
    ) -> std::result::Result<Self, WgcCaptureSourceError> {
        DESTROY_WATCH.with(|watch| {
            debug_assert!(watch.borrow().is_none());
            *watch.borrow_mut() = Some(DestroyWatchState {
                hwnd: hwnd.0 as usize,
                destroyed: destroyed.clone(),
            });
        });
        // SAFETY: the callback is a static function, out-of-context delivery
        // keeps it in this process. A global hook avoids a race where the
        // target is destroyed before its pid/tid-filtered hook is installed;
        // the callback itself ignores every HWND except the selected value.
        // This watcher thread pumps its queue until the hook is removed.
        let hook = unsafe {
            SetWinEventHook(
                EVENT_OBJECT_DESTROY,
                EVENT_OBJECT_DESTROY,
                None,
                Some(window_event_proc),
                0,
                0,
                WINEVENT_OUTOFCONTEXT,
            )
        };
        if hook.is_invalid() {
            DESTROY_WATCH.with(|watch| *watch.borrow_mut() = None);
            return Err(windows::core::Error::from_thread().into());
        }
        Ok(Self { hook })
    }
}

impl Drop for WindowDestroyHook {
    fn drop(&mut self) {
        // SAFETY: this hook was installed successfully by `install`, remains
        // owned by this guard, and is removed on the same source thread.
        let _ = unsafe { UnhookWinEvent(self.hook) };
        DESTROY_WATCH.with(|watch| *watch.borrow_mut() = None);
    }
}

fn pump_window_events() {
    let mut message = MSG::default();
    // SAFETY: `message` is a live out-parameter. The watcher thread owns no UI
    // window, so draining and dispatching its queue only delivers the WinEvent
    // callback and other messages addressed to this helper thread.
    while unsafe { PeekMessageW(&mut message, None, 0, 0, PM_REMOVE) }.as_bool() {
        // SAFETY: `message` was initialized by successful `PeekMessageW` and
        // remains live for translation and synchronous dispatch.
        unsafe {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
}

struct WindowLifetimeWatch {
    destroyed: Arc<AtomicBool>,
    stop_tx: crossbeam_channel::Sender<()>,
    worker: Option<JoinHandle<()>>,
}

impl WindowLifetimeWatch {
    fn start(hwnd: HWND) -> std::result::Result<Self, WgcCaptureSourceError> {
        let hwnd = hwnd.0 as usize;
        let destroyed = Arc::new(AtomicBool::new(false));
        let worker_destroyed = destroyed.clone();
        let (stop_tx, stop_rx) = bounded(1);
        let (ready_tx, ready_rx) = bounded(1);
        let worker = thread::Builder::new()
            .name("wgc-window-watch".into())
            .spawn(move || {
                let hwnd = HWND(hwnd as *mut _);
                let hook = match WindowDestroyHook::install(hwnd, worker_destroyed) {
                    Ok(hook) => {
                        let _ = ready_tx.send(Ok(()));
                        hook
                    }
                    Err(error) => {
                        let _ = ready_tx.send(Err(error));
                        return;
                    }
                };
                loop {
                    pump_window_events();
                    match stop_rx.recv_timeout(Duration::from_millis(10)) {
                        Ok(()) | Err(RecvTimeoutError::Disconnected) => break,
                        Err(RecvTimeoutError::Timeout) => {}
                    }
                }
                drop(hook);
            })
            .map_err(|error| WgcCaptureSourceError::WindowWatcherUnavailable(error.to_string()))?;

        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                destroyed,
                stop_tx,
                worker: Some(worker),
            }),
            Ok(Err(error)) => {
                let _ = worker.join();
                Err(error)
            }
            Err(_) => {
                let _ = worker.join();
                Err(WgcCaptureSourceError::WindowWatcherUnavailable(
                    "watcher stopped during startup".into(),
                ))
            }
        }
    }

    fn destroyed(&self) -> bool {
        self.destroyed.load(Ordering::Acquire)
    }

    fn flag(&self) -> Arc<AtomicBool> {
        self.destroyed.clone()
    }
}

impl Drop for WindowLifetimeWatch {
    fn drop(&mut self) {
        let _ = self.stop_tx.try_send(());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

struct WgcRuntime {
    /// Stable handle opened while the owner was alive. Unlike reopening its
    /// pid during `Drop`, this cannot confuse access denial or pid reuse with
    /// process termination.
    owner_process: HANDLE,
    target_destroyed: Arc<AtomicBool>,
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
        owner_pid: u32,
        owner_thread_id: u32,
        owner_process: HANDLE,
        target_destroyed: Arc<AtomicBool>,
        device: &ID3D11Device,
        include_cursor: bool,
    ) -> std::result::Result<Self, WgcCaptureSourceError> {
        if !GraphicsCaptureSession::IsSupported()? {
            return Err(WgcCaptureSourceError::Unsupported);
        }
        if target_destroyed.load(Ordering::Acquire)
            || process_has_exited(owner_process)
            || !target_still_matches(hwnd, owner_pid, owner_thread_id)
        {
            return Err(WgcCaptureSourceError::TargetGone);
        }
        let interop: IGraphicsCaptureItemInterop = factory::<GraphicsCaptureItem, _>()?;
        // SAFETY: the lifetime watcher was installed before construction read
        // the original owner identity, and the checks on both sides of this
        // call reject any destruction or reuse that races item creation.
        let item: GraphicsCaptureItem = unsafe { interop.CreateForWindow(hwnd)? };
        if target_destroyed.load(Ordering::Acquire)
            || process_has_exited(owner_process)
            || !target_still_matches(hwnd, owner_pid, owner_thread_id)
        {
            return Err(WgcCaptureSourceError::TargetGone);
        }
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
        let mut runtime = Self {
            owner_process,
            target_destroyed,
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

    fn target_gone(&self) -> bool {
        self.closed.load(Ordering::Acquire)
            || self.target_destroyed.load(Ordering::Acquire)
            || process_has_exited(self.owner_process)
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
        // crashes the process on the way out. The synchronization handle was
        // opened while the original owner was alive, so access denial or pid
        // reuse at teardown cannot choose the leak path accidentally.
        if process_has_exited(self.owner_process) {
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
    use std::{
        sync::{Arc, Mutex, mpsc},
        thread,
        time::Duration,
    };

    use windows::{
        Win32::{
            Graphics::Direct3D11::ID3D11Multithread,
            UI::WindowsAndMessaging::{
                CreateWindowExW, DestroyWindow, WINDOW_EX_STYLE, WS_OVERLAPPEDWINDOW, WS_VISIBLE,
            },
        },
        core::{Interface, w},
    };

    use super::*;
    use crate::{
        Error,
        bus::BusEvent,
        contract::InputContract,
        elements::{AppSink, D3d11Scaler, D3d11ScalerFormat, FrameCounter},
        pipeline::Pipeline,
        platform::windows::d3d11va::d3d11va_texture,
    };

    struct TestWindow(HWND);

    impl TestWindow {
        fn create() -> windows::core::Result<Self> {
            // The built-in STATIC class avoids registering process-global
            // test state. The window stays owned by this test thread and is
            // destroyed before the thread returns.
            //
            // SAFETY: `STATIC` is a preregistered window class and every other
            // argument is a plain value or absent parent/menu/instance, so this
            // borrows nothing that has to outlive the call.
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
            // SAFETY: `DestroyWindow` runs on the thread that created this
            // window, as it requires, and the handle is destroyed exactly once.
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
    fn destroy_callback_remains_set_when_the_hwnd_value_can_be_reused() {
        let destroyed = Arc::new(AtomicBool::new(false));
        DESTROY_WATCH.with(|watch| {
            *watch.borrow_mut() = Some(DestroyWatchState {
                hwnd: 0x1234,
                destroyed: destroyed.clone(),
            });
        });
        // SAFETY: directly invokes the callback with plain test values; it
        // only compares them and updates the thread-local atomic flag.
        unsafe {
            window_event_proc(
                HWINEVENTHOOK::default(),
                EVENT_OBJECT_DESTROY,
                HWND(0x5678usize as *mut _),
                OBJID_WINDOW.0,
                CHILDID_SELF as i32,
                0,
                0,
            );
        }
        assert!(!destroyed.load(Ordering::Acquire));
        // SAFETY: same callback contract as above, now with the watched HWND.
        unsafe {
            window_event_proc(
                HWINEVENTHOOK::default(),
                EVENT_OBJECT_DESTROY,
                HWND(0x1234usize as *mut _),
                OBJID_WINDOW.0,
                CHILDID_SELF as i32,
                0,
                0,
            );
        }
        assert!(destroyed.load(Ordering::Acquire));
        // A later live window can reuse the numeric value, but no subsequent
        // event can clear the original object's terminal state.
        //
        // SAFETY: same callback contract as above, with a reused HWND value.
        unsafe {
            window_event_proc(
                HWINEVENTHOOK::default(),
                EVENT_OBJECT_DESTROY,
                HWND(0x5678usize as *mut _),
                OBJID_WINDOW.0,
                CHILDID_SELF as i32,
                0,
                0,
            );
        }
        assert!(destroyed.load(Ordering::Acquire));
        DESTROY_WATCH.with(|watch| *watch.borrow_mut() = None);
    }

    #[test]
    fn retained_process_handle_distinguishes_live_and_exited_owners() {
        let current = ProcessHandle::open(std::process::id()).expect("open current process");
        assert!(!process_has_exited(current.raw()));

        let mut child = std::process::Command::new("cmd")
            .args(["/C", "exit", "0"])
            .spawn()
            .expect("spawn short-lived owner process");
        let retained = ProcessHandle::open(child.id()).expect("retain child process identity");
        child.wait().expect("wait for owner process");
        assert!(process_has_exited(retained.raw()));
    }

    #[test]
    #[ignore = "requires an interactive Windows Graphics Capture session"]
    fn queue_and_d3d11_scaler_share_the_capture_context_safely() {
        crate::init().expect("initialize FFmpeg");
        let window = TestWindow::create().expect("create capture target window");
        let (source, device) = match WgcCaptureSource::open(
            "capture",
            window.0,
            WgcCaptureOptions {
                fps: 60,
                include_cursor: false,
            },
        ) {
            Ok(opened) => opened,
            Err(error) => {
                eprintln!("skipping: cannot create WGC source ({error})");
                return;
            }
        };
        // SAFETY: returns the one live immediate context owned by `device`.
        let context = unsafe { device.GetImmediateContext() }.expect("immediate context");
        let multithread: ID3D11Multithread = context.cast().expect("multithread interface");
        // SAFETY: reads one boolean property from the live context interface.
        assert!(unsafe { multithread.GetMultithreadProtected() }.as_bool());
        let scaler = D3d11Scaler::new(
            "scale",
            &device,
            Arc::new(Mutex::new(context)),
            D3d11ScalerFormat::Preserve,
            160,
            120,
        )
        .expect("create scaler");
        let (counter, frames) = FrameCounter::new("frames");
        let pipeline = Pipeline::new("wgc-d3d11-queue", source, |source, ctx| {
            let branch = ctx
                .branch()
                .queue("captured", 4)
                .pipe(scaler)
                .to(Box::new(counter))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("wire WGC D3D11 pipeline");

        pipeline.run().expect("start WGC pipeline");
        thread::sleep(Duration::from_secs(2));
        pipeline.stop();
        assert!(
            frames.load(Ordering::Relaxed) > 0,
            "scaled frames must flow"
        );
        let errors: Vec<_> = pipeline
            .bus()
            .iter()
            .filter(|event| matches!(event, BusEvent::Error { .. }))
            .collect();
        assert!(errors.is_empty(), "unexpected WGC/D3D11 errors: {errors:?}");
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
    /// gone — and it must drain carrying an error rather than an EOS, since a
    /// live capture has no natural end.
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
            let events: Vec<_> = draining.bus().iter().collect();
            let _ = done_tx.send(events);
        });
        let events = done_rx
            .recv_timeout(Duration::from_secs(15))
            .expect("the source thread must finish once its target window is destroyed");

        assert!(
            events.iter().any(|event| matches!(
                event,
                BusEvent::Error {
                    element_type: ElementType::WgcCaptureSource,
                    ..
                }
            )),
            "a destroyed target must be reported as an error, not EOS: {events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, BusEvent::Eos { .. })),
            "a live capture has no natural end, so nothing may report EOS: {events:?}"
        );
    }
}
