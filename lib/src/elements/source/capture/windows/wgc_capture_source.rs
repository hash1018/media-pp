use std::{
    sync::{
        Arc, Mutex, Once,
        atomic::{AtomicBool, AtomicU64, Ordering},
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
        Foundation::{
            CloseHandle, HANDLE, HMODULE, HWND, LPARAM, RPC_E_CHANGED_MODE, WAIT_OBJECT_0, WPARAM,
        },
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
            Com::CoIncrementMTAUsage,
            Threading::{
                GetCurrentThreadId, OpenProcess, PROCESS_SYNCHRONIZE, WaitForSingleObject,
            },
            WinRT::{
                Direct3D11::{CreateDirect3D11DeviceFromDXGIDevice, IDirect3DDxgiInterfaceAccess},
                Graphics::Capture::IGraphicsCaptureItemInterop,
                RO_INIT_MULTITHREADED, RoInitialize, RoUninitialize,
            },
        },
        UI::{
            Accessibility::{HWINEVENTHOOK, SetWinEventHook, UnhookWinEvent},
            WindowsAndMessaging::{
                CHILDID_SELF, DispatchMessageW, EVENT_OBJECT_DESTROY, GetMessageW,
                GetWindowThreadProcessId, MSG, OBJID_WINDOW, PM_NOREMOVE, PeekMessageW,
                PostThreadMessageW, TranslateMessage, WINEVENT_OUTOFCONTEXT, WM_QUIT, WM_USER,
            },
        },
    },
    core::{IInspectable, Interface, factory},
};

use crate::rate::{FrameRate, FrameRateHandle};
use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    contract::{MediaKind, MemoryDomain, OutputContract, PortContract},
    control::{ControlReceiver, drain_control},
    element::{Element, ElementType, Source, SourceElement, element_pp_log},
    error::{D3d11FrameWrapError, D3d11SharedDeviceError, Result},
    pad::SrcPad,
    platform::windows::{d3d11::protect_shared_device, d3d11va::wrap_d3d11_texture},
    pool::UnboundObjectPool,
    pp_log::{PpLog, pp_error, pp_info, pp_warn},
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
    /// The capture device cannot be shared across a pipeline's threads.
    #[error(transparent)]
    SharedDevice(#[from] D3d11SharedDeviceError),
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
    /// `None` when the owner denied a synchronization handle; see
    /// [`process_has_exited`] for what that costs.
    owner_process: Option<ProcessHandle>,
    target_watch: WindowLifetimeWatch,
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    include_cursor: bool,
    /// Shared rather than a plain field so [`WgcCaptureSource::frame_rate`]
    /// can hand out a handle that changes it while this is running.
    frame_rate: Arc<FrameRate>,
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
        drop(context);
        let context = protect_shared_device(&device)?;
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
        let context = protect_shared_device(device)?;
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
        if let Some(error) = target.owner_process_error {
            pp_warn!(
                pp_log: &pp_log,
                "owner process {} cannot be monitored ({error}); teardown will always take the \
                 full path and owner exit alone will not end this source",
                target.owner_pid
            );
        }

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
            frame_rate: FrameRate::new(ffmpeg::Rational::new(fps, 1)),
            frame_index: 0,
            pad,
            frame_pool: UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {}),
        }
    }

    /// PTS unit of frames emitted by this source — the reciprocal of the
    /// capture rate, and so a value that moves with [`Self::frame_rate`].
    pub fn time_base(&self) -> ffmpeg::Rational {
        self.frame_rate.get().invert()
    }

    /// Runtime control for the rate this captures at.
    ///
    /// Taken before this is moved into a `Pipeline`, which is the only chance
    /// to. Changing the rate re-means [`Self::time_base`] and every timestamp
    /// after the change — see [`crate::rate`].
    pub fn frame_rate(&self) -> FrameRateHandle {
        self.frame_rate.handle()
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
        let target = TargetIdentity {
            hwnd,
            owner_pid: self.owner_pid,
            owner_thread_id: self.owner_thread_id,
        };
        let runtime = WgcRuntime::start(
            target,
            self.owner_process.as_ref().map(ProcessHandle::raw),
            self.target_watch.flag(),
            &self.device,
            self.include_cursor,
        )
        .inspect_err(|error| pp_error!(self, "capture start failed: {error}"))?;
        pp_info!(
            self,
            "started: window={:?}, fps={}, include_cursor={}",
            hwnd,
            self.frame_rate.get(),
            self.include_cursor
        );

        let mut latest = None;
        let mut visible_size = None;
        let mut schedule = PeriodicSchedule::new(self.frame_rate.interval(), Instant::now());
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
    fn open(pid: u32) -> windows::core::Result<Self> {
        // SAFETY: opens one synchronization-only reference to the live owner
        // pid. The returned handle is closed exactly once by `Drop`.
        let handle = unsafe { OpenProcess(PROCESS_SYNCHRONIZE, false, pid) }?;
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

/// Whether the retained owner is known to have exited. `None` means the owner
/// could never be opened for synchronization — a protected or higher-integrity
/// process denies even `PROCESS_SYNCHRONIZE` — and answers `false`, because
/// every caller uses this only to take a shortcut that requires proof the owner
/// is gone.
fn process_has_exited(process: Option<HANDLE>) -> bool {
    let Some(process) = process else {
        return false;
    };
    // SAFETY: every caller supplies the live synchronization handle retained
    // from construction. A zero-time wait observes state without blocking.
    (unsafe { WaitForSingleObject(process, 0) }) == WAIT_OBJECT_0
}

/// The capture target, identified by more than its raw handle value. Windows
/// recycles `HWND` values, so the owner pid/tid read while the original window
/// was alive is what distinguishes it from a later window that happened to be
/// given the same number.
#[derive(Clone, Copy)]
struct TargetIdentity {
    hwnd: HWND,
    owner_pid: u32,
    owner_thread_id: u32,
}

impl TargetIdentity {
    /// Whether the handle still resolves to the exact window this source was
    /// opened on. A destroyed window reports thread zero, so this answers
    /// `false` for destruction and for reuse by another thread alike — the
    /// destroy hook covers the remaining case of the same UI thread recreating
    /// a window under the same handle value.
    fn still_matches(self) -> bool {
        let mut current_pid = 0;
        // SAFETY: reads the identity currently associated with this by-value
        // HWND and writes to the live pid out-parameter, retaining neither.
        let current_thread_id =
            unsafe { GetWindowThreadProcessId(self.hwnd, Some(&mut current_pid)) };
        current_thread_id == self.owner_thread_id && current_pid == self.owner_pid
    }
}

struct CaptureTarget {
    hwnd: usize,
    owner_pid: u32,
    owner_thread_id: u32,
    /// `None` when the owner could not be opened for synchronization; see
    /// [`process_has_exited`]. Reported once by [`WgcCaptureSource::from_device`].
    owner_process: Option<ProcessHandle>,
    owner_process_error: Option<windows::core::Error>,
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
        // A capturable window does not imply an openable owner: an elevated or
        // otherwise protected process denies even `PROCESS_SYNCHRONIZE`. WGC
        // still captures it, so losing owner-exit detection degrades teardown
        // to the always-safe full path rather than refusing the target.
        let (owner_process, owner_process_error) = match ProcessHandle::open(owner_pid) {
            Ok(handle) => (Some(handle), None),
            Err(error) => (None, Some(error)),
        };

        let identity = TargetIdentity {
            hwnd,
            owner_pid,
            owner_thread_id,
        };
        if watch.destroyed() || !identity.still_matches() {
            return Err(WgcCaptureSourceError::InvalidWindow);
        }

        Ok(Self {
            hwnd: hwnd.0 as usize,
            owner_pid,
            owner_thread_id,
            owner_process,
            owner_process_error,
            watch,
        })
    }
}

struct CaptureFrameGuard(Direct3D11CaptureFrame);

impl Drop for CaptureFrameGuard {
    fn drop(&mut self) {
        let _ = self.0.Close();
    }
}

/// Holds one never-released reference to the process's multithreaded
/// apartment.
///
/// `windows-rs` caches every WinRT activation factory in a process-global slot
/// and deliberately leaks the reference (see `FactoryCache::call`), which is
/// only sound while the component DLL behind it stays loaded. The *last*
/// `RoUninitialize` in a process unloads it — `GraphicsCapture.dll` here — and
/// leaves that cached pointer dangling; the next
/// `GraphicsCaptureSession::IsSupported` then jumps through a freed vtable and
/// takes the process down with an access violation. A host that stops one
/// window capture and starts another reaches exactly that, because each source
/// thread owns its own apartment and the one it leaves behind is routinely the
/// process's last.
///
/// One MTA reference that is never decremented removes the condition instead of
/// racing it: the apartment a source thread joins and leaves is no longer the
/// last one, so nothing gets unloaded out from under the cache. It is taken
/// before the first `RoInitialize` and intentionally kept for the life of the
/// process, because the cached factories it protects live that long too.
fn retain_process_mta() {
    static MTA: Once = Once::new();
    MTA.call_once(|| {
        // SAFETY: takes one reference to the process MTA and returns its
        // cookie by value, borrowing nothing. The cookie is dropped without
        // `CoDecrementMTAUsage` on purpose — see this function's docs.
        if let Err(error) = unsafe { CoIncrementMTAUsage() } {
            // Nothing here can proceed more safely by reacting to this: the
            // apartment initialization that follows reports its own failure,
            // and a capture that somehow works anyway is no worse off than it
            // was before this guard existed.
            debug_assert!(false, "CoIncrementMTAUsage failed: {error}");
        }
    });
}

struct WinRtApartment {
    uninitialize: bool,
}

impl WinRtApartment {
    fn initialize() -> std::result::Result<Self, WgcCaptureSourceError> {
        retain_process_mta();
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

/// One registered interest in a window's destruction, owned by the
/// [`WindowLifetimeWatch`] that created it. The id is what removal matches on,
/// so two sources watching the same window stay independent.
struct WatchEntry {
    id: u64,
    hwnd: usize,
    destroyed: Arc<AtomicBool>,
}

/// The watcher thread, alive exactly while [`WATCH_ENTRIES`] is non-empty.
/// `thread_id` is its message-queue address; the thread forces that queue into
/// existence before reporting itself ready, so posting `WM_QUIT` to it cannot
/// fail for a thread that has not exited.
struct WatcherThread {
    thread_id: u32,
    worker: JoinHandle<()>,
}

/// Registrations the hook callback matches against. Locked by the watcher
/// thread on every desktop window destruction, so nothing slow may happen
/// while it is held.
static WATCH_ENTRIES: Mutex<Vec<WatchEntry>> = Mutex::new(Vec::new());
/// The watcher's lifecycle. Always locked *before* [`WATCH_ENTRIES`], and
/// never held together with it across the join in [`stop_watcher_thread`] —
/// the thread being joined needs the entries lock to run its callback.
static WATCH_THREAD: Mutex<Option<WatcherThread>> = Mutex::new(None);
static NEXT_WATCH_ID: AtomicU64 = AtomicU64::new(1);

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
    let hwnd = hwnd.0 as usize;
    let Ok(entries) = WATCH_ENTRIES.lock() else {
        return;
    };
    for entry in entries.iter().filter(|entry| entry.hwnd == hwnd) {
        entry.destroyed.store(true, Ordering::Release);
    }
}

struct WindowDestroyHook(HWINEVENTHOOK);

impl WindowDestroyHook {
    fn install() -> windows::core::Result<Self> {
        // SAFETY: the callback is a static function and out-of-context delivery
        // keeps it in this process, on the thread installing the hook.
        //
        // The hook is deliberately unfiltered. Filtering it to the target's own
        // thread would mean reading that thread id first, which reopens exactly
        // the race this watcher exists to close: the original window can be
        // destroyed and its handle value reissued to a new window on the same
        // UI thread before the filtered hook is in place, and an owner pid/tid
        // recheck cannot tell those two windows apart. One unfiltered hook for
        // the whole process is the price; see `WATCH_ENTRIES` for why it is
        // paid once rather than once per capture source.
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
            return Err(windows::core::Error::from_thread());
        }
        Ok(Self(hook))
    }
}

impl Drop for WindowDestroyHook {
    fn drop(&mut self) {
        // SAFETY: this hook was installed successfully by `install`, remains
        // owned by this guard, and is removed on the thread that installed it,
        // as `UnhookWinEvent` requires.
        let _ = unsafe { UnhookWinEvent(self.0) };
    }
}

/// Blocks until `WM_QUIT`, delivering hook callbacks as they arrive. There is
/// nothing to poll for: an out-of-context WinEvent is delivered through this
/// queue, so the thread costs nothing while the desktop is quiet.
fn run_watcher_message_loop() {
    let mut message = MSG::default();
    loop {
        // SAFETY: `message` is a live out-parameter. This thread owns no UI
        // window, so its queue carries only the hook callbacks and the stop
        // message posted by `stop_watcher_thread`.
        let result = unsafe { GetMessageW(&mut message, None, 0, 0) };
        if result.0 <= 0 {
            return;
        }
        // SAFETY: `message` was filled by a successful `GetMessageW` and stays
        // live for translation and synchronous dispatch.
        unsafe {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
}

/// Starts the shared watcher if it is not already running. The caller holds
/// the [`WATCH_THREAD`] lock.
fn start_watcher_thread(
    thread: &mut Option<WatcherThread>,
) -> std::result::Result<(), WgcCaptureSourceError> {
    if thread.is_some() {
        return Ok(());
    }
    let (ready_tx, ready_rx) = bounded(1);
    let worker = thread::Builder::new()
        .name("wgc-window-watch".into())
        .spawn(move || {
            let hook = match WindowDestroyHook::install() {
                Ok(hook) => hook,
                Err(error) => {
                    let _ = ready_tx.send(Err(error));
                    return;
                }
            };
            let mut message = MSG::default();
            // SAFETY: `message` is a live out-parameter. This peek creates this
            // thread's message queue, which is what makes the later
            // `PostThreadMessageW` well defined; nothing is dispatched here.
            unsafe {
                let _ = PeekMessageW(&mut message, None, WM_USER, WM_USER, PM_NOREMOVE);
            }
            // SAFETY: reads this thread's own id and retains nothing.
            let _ = ready_tx.send(Ok(unsafe { GetCurrentThreadId() }));
            run_watcher_message_loop();
            drop(hook);
        })
        .map_err(|error| WgcCaptureSourceError::WindowWatcherUnavailable(error.to_string()))?;

    match ready_rx.recv() {
        Ok(Ok(thread_id)) => {
            *thread = Some(WatcherThread { thread_id, worker });
            Ok(())
        }
        Ok(Err(error)) => {
            let _ = worker.join();
            Err(error.into())
        }
        Err(_) => {
            let _ = worker.join();
            Err(WgcCaptureSourceError::WindowWatcherUnavailable(
                "watcher stopped during startup".into(),
            ))
        }
    }
}

/// Stops the shared watcher. The caller holds the [`WATCH_THREAD`] lock and
/// must not hold [`WATCH_ENTRIES`]: the thread being joined takes that lock in
/// its hook callback.
fn stop_watcher_thread(thread: &mut Option<WatcherThread>) {
    let Some(WatcherThread { thread_id, worker }) = thread.take() else {
        return;
    };
    // SAFETY: the thread created its message queue before reporting itself
    // ready and only leaves the loop on this message, so the post targets a
    // live queue and borrows nothing.
    let _ = unsafe { PostThreadMessageW(thread_id, WM_QUIT, WPARAM(0), LPARAM(0)) };
    let _ = worker.join();
}

/// One source's registered interest in its target window's destruction.
///
/// The hook and the thread behind this are process-wide and shared: a host
/// that captures several windows pays for one desktop-wide hook, not one per
/// source. Dropping the last watch stops that thread.
struct WindowLifetimeWatch {
    id: u64,
    destroyed: Arc<AtomicBool>,
}

impl WindowLifetimeWatch {
    fn start(hwnd: HWND) -> std::result::Result<Self, WgcCaptureSourceError> {
        let id = NEXT_WATCH_ID.fetch_add(1, Ordering::Relaxed);
        let destroyed = Arc::new(AtomicBool::new(false));

        let mut thread = WATCH_THREAD.lock().expect("watcher thread lock");
        // Registered before the thread starts, so no destruction can slip
        // between the hook going live and this window being watched.
        WATCH_ENTRIES
            .lock()
            .expect("watch entries lock")
            .push(WatchEntry {
                id,
                hwnd: hwnd.0 as usize,
                destroyed: destroyed.clone(),
            });
        if let Err(error) = start_watcher_thread(&mut thread) {
            remove_watch_entry(id);
            return Err(error);
        }
        Ok(Self { id, destroyed })
    }

    fn destroyed(&self) -> bool {
        self.destroyed.load(Ordering::Acquire)
    }

    fn flag(&self) -> Arc<AtomicBool> {
        self.destroyed.clone()
    }
}

/// Returns whether the registry is now empty.
fn remove_watch_entry(id: u64) -> bool {
    let mut entries = WATCH_ENTRIES.lock().expect("watch entries lock");
    entries.retain(|entry| entry.id != id);
    entries.is_empty()
}

impl Drop for WindowLifetimeWatch {
    fn drop(&mut self) {
        let mut thread = WATCH_THREAD.lock().expect("watcher thread lock");
        if remove_watch_entry(self.id) {
            stop_watcher_thread(&mut thread);
        }
    }
}

struct WgcRuntime {
    /// The window and owner identity this session was started on, rechecked
    /// on every wake-up — see [`WgcRuntime::target_gone`].
    target: TargetIdentity,
    /// Stable handle opened while the owner was alive. Unlike reopening its
    /// pid during `Drop`, this cannot confuse access denial or pid reuse with
    /// process termination. `None` when the owner refused one at all.
    owner_process: Option<HANDLE>,
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
        target: TargetIdentity,
        owner_process: Option<HANDLE>,
        target_destroyed: Arc<AtomicBool>,
        device: &ID3D11Device,
        include_cursor: bool,
    ) -> std::result::Result<Self, WgcCaptureSourceError> {
        if !GraphicsCaptureSession::IsSupported()? {
            return Err(WgcCaptureSourceError::Unsupported);
        }
        let gone = || {
            target_destroyed.load(Ordering::Acquire)
                || !target.still_matches()
                || process_has_exited(owner_process)
        };
        if gone() {
            return Err(WgcCaptureSourceError::TargetGone);
        }
        let interop: IGraphicsCaptureItemInterop = factory::<GraphicsCaptureItem, _>()?;
        // SAFETY: the lifetime watcher was installed before construction read
        // the original owner identity, and the checks on both sides of this
        // call reject any destruction or reuse that races item creation.
        let item: GraphicsCaptureItem = unsafe { interop.CreateForWindow(target.hwnd)? };
        if gone() {
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
            target,
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

    /// Cheapest-first, and deliberately three independent signals rather than
    /// one: `Closed` is documented but has been observed not to fire for a
    /// forceful kill or a plain `WM_CLOSE`; the destroy hook is delivered
    /// asynchronously and a higher-integrity owner may never reach it; and the
    /// identity read is the only one that costs nothing to be wrong about. A
    /// missed target death parks the loop emitting the last frame forever, so
    /// each signal stays as a backstop for the others.
    fn target_gone(&self) -> bool {
        self.closed.load(Ordering::Acquire)
            || self.target_destroyed.load(Ordering::Acquire)
            || !self.target.still_matches()
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
        // reuse at teardown cannot choose the leak path accidentally — and an
        // owner that never granted one takes this normal path too, which is
        // the safe direction to be wrong in.
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
                CreateWindowExW, DestroyWindow, FindWindowW, WINDOW_EX_STYLE, WS_OVERLAPPEDWINDOW,
                WS_VISIBLE,
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

    /// Drives the real hook end to end: a live registration, a real
    /// `DestroyWindow`, and the flag the source loop actually reads. The
    /// callback comparison alone would not prove the hook is installed, that
    /// the shared watcher thread pumps it, or that delivery reaches this
    /// process at all.
    #[test]
    fn the_shared_hook_reports_a_real_window_destruction() {
        let window = TestWindow::create().expect("create test window");
        let watch = WindowLifetimeWatch::start(window.0).expect("start the lifetime watch");
        assert!(!watch.destroyed());

        // An unrelated window's destruction must not disturb this watch.
        let other = TestWindow::create().expect("create second test window");
        drop(other);
        drop(window);

        let deadline = Instant::now() + Duration::from_secs(5);
        while !watch.destroyed() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
        assert!(
            watch.destroyed(),
            "the destroy hook must report the watched window"
        );
    }

    /// Two sources watching different windows share one hook and one thread,
    /// and neither registration may see the other's destruction.
    #[test]
    fn watches_are_independent_and_share_one_watcher_thread() {
        let first_window = TestWindow::create().expect("create first test window");
        let second_window = TestWindow::create().expect("create second test window");
        let first = WindowLifetimeWatch::start(first_window.0).expect("start first watch");
        let second = WindowLifetimeWatch::start(second_window.0).expect("start second watch");
        let (first_id, second_id) = (first.id, second.id);
        // A live registration always implies a running watcher.
        assert!(WATCH_THREAD.lock().expect("watcher thread lock").is_some());

        drop(first_window);
        let deadline = Instant::now() + Duration::from_secs(5);
        while !first.destroyed() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
        assert!(first.destroyed());
        assert!(
            !second.destroyed(),
            "one window's destruction must not end another watch"
        );

        drop(second_window);
        drop(second);
        drop(first);
        // Both registrations are gone. Whether the shared thread stopped is
        // not asserted here: another test in this process may legitimately
        // hold a watch of its own at the same moment.
        let entries = WATCH_ENTRIES.lock().expect("watch entries lock");
        assert!(
            !entries
                .iter()
                .any(|entry| entry.id == first_id || entry.id == second_id),
            "dropping a watch must unregister it"
        );
    }

    /// Starting a capture, stopping it, and starting another one must not take
    /// the process down.
    ///
    /// Each source thread owns its own WinRT apartment, so the one a stopped
    /// capture leaves behind is routinely the process's last — which used to
    /// unload `GraphicsCapture.dll` while `windows-rs` still held a cached,
    /// deliberately leaked activation factory pointing into it. The next
    /// `IsSupported` call then read a freed vtable and died with an access
    /// violation. No window or capture session is needed to reproduce that:
    /// the factory is what goes stale.
    ///
    /// Another live MTA reference elsewhere in the process hides the unload, so
    /// this can pass for the wrong reason in a busy parallel run. It cannot
    /// fail for the wrong reason, and it is exact when run on its own.
    #[test]
    fn winrt_factories_survive_an_apartment_cycle() {
        fn probe_in_a_fresh_apartment() {
            thread::spawn(|| {
                let apartment = WinRtApartment::initialize().expect("initialize WinRT apartment");
                // Primes, then later re-reads, the process-global factory cache.
                let _ = GraphicsCaptureSession::IsSupported();
                drop(apartment);
            })
            .join()
            .expect("apartment thread must not panic or fault");
        }

        probe_in_a_fresh_apartment();
        probe_in_a_fresh_apartment();
    }

    /// Capture runs on the source thread while everything past the first
    /// `Queue` runs on another, so a device that promised single-threaded use
    /// has to be refused at `open_with_device` — before a window is watched or
    /// a WGC object exists.
    #[test]
    fn rejects_a_single_threaded_device() {
        let Some(device) = crate::test_support::try_single_threaded_d3d11_device() else {
            return;
        };
        let window = TestWindow::create().expect("create capture target window");
        let result = WgcCaptureSource::open_with_device(
            "capture",
            window.0,
            WgcCaptureOptions::default(),
            &device,
        );
        assert!(matches!(
            result,
            Err(WgcCaptureSourceError::SharedDevice(
                D3d11SharedDeviceError::SingleThreaded
            ))
        ));
    }

    /// The destroy hook is asynchronous and can be missed, so the identity
    /// read is what has to keep answering for a window that went away while
    /// its owner kept running — otherwise the source loop parks forever on
    /// its last frame.
    #[test]
    fn window_identity_stops_matching_once_the_window_is_destroyed() {
        let window = TestWindow::create().expect("create test window");
        let mut owner_pid = 0;
        // SAFETY: reads the identity of the live window this test owns and
        // fills a live pid out-parameter.
        let owner_thread_id = unsafe { GetWindowThreadProcessId(window.0, Some(&mut owner_pid)) };
        let identity = TargetIdentity {
            hwnd: window.0,
            owner_pid,
            owner_thread_id,
        };
        assert!(identity.still_matches());

        // A different owner never matches, even for this very much alive
        // handle — that is the HWND-reuse case.
        assert!(
            !TargetIdentity {
                owner_pid: owner_pid.wrapping_add(1),
                ..identity
            }
            .still_matches()
        );

        drop(window);
        assert!(
            !identity.still_matches(),
            "a destroyed window must stop matching its recorded identity"
        );
    }

    #[test]
    fn retained_process_handle_distinguishes_live_and_exited_owners() {
        let current = ProcessHandle::open(std::process::id()).expect("open current process");
        assert!(!process_has_exited(Some(current.raw())));

        let mut child = std::process::Command::new("cmd")
            .args(["/C", "exit", "0"])
            .spawn()
            .expect("spawn short-lived owner process");
        let retained = ProcessHandle::open(child.id()).expect("retain child process identity");
        child.wait().expect("wait for owner process");
        assert!(process_has_exited(Some(retained.raw())));

        // An owner that never granted a handle is not evidence of exit, so it
        // must take the full-teardown path rather than the leak shortcut.
        assert!(!process_has_exited(None));
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
    /// Hosts one top-level window in a separate process, so a test can end
    /// that process outright rather than destroying a window this process
    /// owns. `title` is what [`Self::find`] matches on.
    struct WindowHostProcess {
        child: std::process::Child,
        title: String,
    }

    impl WindowHostProcess {
        fn start(title: &str) -> Option<Self> {
            let script = format!(
                "Add-Type -AssemblyName System.Windows.Forms; \
                 $f = New-Object Windows.Forms.Form; \
                 $f.Text = '{title}'; $f.Width = 320; $f.Height = 240; $f.Show(); \
                 while ($true) {{ [Windows.Forms.Application]::DoEvents(); Start-Sleep -Milliseconds 20 }}"
            );
            let child = std::process::Command::new("powershell")
                .args(["-NoProfile", "-NonInteractive", "-Command", &script])
                .spawn()
                .ok()?;
            Some(Self {
                child,
                title: title.to_string(),
            })
        }

        /// The host's window once it exists, or `None` if it never showed up.
        fn find(&self) -> Option<HWND> {
            let wide: Vec<u16> = self
                .title
                .encode_utf16()
                .chain(std::iter::once(0))
                .collect();
            let deadline = Instant::now() + Duration::from_secs(20);
            while Instant::now() < deadline {
                // SAFETY: both arguments are read-only; the class name is
                // absent and the title is a live NUL-terminated wide string.
                if let Ok(hwnd) = unsafe { FindWindowW(None, windows::core::PCWSTR(wide.as_ptr())) }
                    && !hwnd.is_invalid()
                {
                    return Some(hwnd);
                }
                thread::sleep(Duration::from_millis(100));
            }
            None
        }

        fn kill(&mut self) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }

    impl Drop for WindowHostProcess {
        fn drop(&mut self) {
            self.kill();
        }
    }

    /// Killing the window's *owner* has to end the source the same way
    /// destroying the window does — an error on the bus, no EOS — and the
    /// source thread has to actually finish.
    ///
    /// This is the common shape of "the user closed the app being captured",
    /// and it is not the same code path as a destroyed window whose owner
    /// lives on: teardown deliberately leaks the capture session here,
    /// because retiring it against a dead owner blocks the source thread
    /// forever.
    #[test]
    #[ignore = "requires an interactive Windows Graphics Capture session"]
    fn killing_the_target_windows_owner_ends_the_source() {
        crate::init().expect("initialize FFmpeg");
        let title = format!("media-pp wgc owner test {}", std::process::id());
        let Some(mut host) = WindowHostProcess::start(&title) else {
            eprintln!("skipping: cannot start the window host process");
            return;
        };
        let Some(hwnd) = host.find() else {
            eprintln!("skipping: the window host never showed a window");
            return;
        };

        let (source, _device) = match WgcCaptureSource::open(
            "capture",
            hwnd,
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
        let pipeline = Pipeline::new("wgc-owner-killed", source, |source, ctx| {
            let branch = ctx
                .branch()
                .to(Box::new(AppSink::new("sink", |_| Ok(()))))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("wire WGC pipeline");

        pipeline.run().expect("start WGC pipeline");
        // Let the session actually start, so this exercises a running capture
        // losing its owner rather than a failed start.
        thread::sleep(Duration::from_millis(500));
        host.kill();

        let (done_tx, done_rx) = mpsc::sync_channel(1);
        let draining = Arc::clone(&pipeline);
        thread::spawn(move || {
            let events: Vec<_> = draining.bus().iter().collect();
            let _ = done_tx.send(events);
        });
        let events = done_rx
            .recv_timeout(Duration::from_secs(15))
            .expect("the source thread must finish once its target's owner is gone");

        assert!(
            events.iter().any(|event| matches!(
                event,
                BusEvent::Error {
                    element_type: ElementType::WgcCaptureSource,
                    ..
                }
            )),
            "a dead owner must be reported as an error, not EOS: {events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, BusEvent::Eos { .. })),
            "a live capture has no natural end, so nothing may report EOS: {events:?}"
        );
    }

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
