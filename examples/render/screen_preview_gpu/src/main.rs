//! Captures a desktop or application window straight into GPU memory and
//! presents it without a system-memory pixel round trip.
//!
//! - Windows desktop: `DxgiCaptureSource(GPU) -> Queue -> D3d11Renderer`
//! - Windows window: `WgcCaptureSource -> Queue -> D3d11Renderer`
//! - Linux: `PipeWireScreenCaptureSource(GPU) -> Queue -> CudaConverter ->
//!   CudaRenderer(Vulkan)`
//!
//! `WgcCaptureSource` deliberately takes only an `HWND` and shows no picker
//! of its own (see its docs); on Windows, `wgc` with no `HWND` argument lists
//! capturable top-level windows on the console and prompts for one instead.
//!
//! ```text
//! cargo run -p screen_preview_gpu -- dxgi
//! cargo run -p screen_preview_gpu -- wgc [<HWND>] # Windows
//! cargo run -p screen_preview_gpu -- [monitor|window] [restore-token] # Linux
//! ```

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn main() {
    eprintln!(
        "{} supports Windows (DXGI) and Linux (PipeWire) only",
        env!("CARGO_PKG_NAME")
    );
}

#[cfg(target_os = "windows")]
fn main() -> impl std::process::Termination {
    windows_example::run()
}

#[cfg(target_os = "linux")]
fn main() -> impl std::process::Termination {
    linux_example::run()
}

#[cfg(target_os = "windows")]
mod windows_example {
    use std::{
        ffi::c_void,
        io::{self, Write},
        process::ExitCode,
    };

    use media_pp::{
        bus::BusEvent,
        element::{ElementType, SourceElement},
        elements::{
            CaptureMode, DxgiCaptureOptions, DxgiCaptureSource, WgcCaptureOptions, WgcCaptureSource,
        },
        pipeline::Pipeline,
    };
    use render_common::{D3d11GpuContext, Shutdown};
    use windows::{
        Win32::{
            Foundation::{HWND, LPARAM, TRUE},
            Graphics::Dwm::{DWMWA_CLOAKED, DwmGetWindowAttribute},
            UI::WindowsAndMessaging::{
                EnumWindows, GW_OWNER, GWL_EXSTYLE, GetWindow, GetWindowLongW, GetWindowTextW,
                IsWindowVisible, WS_EX_TOOLWINDOW,
            },
        },
        core::BOOL,
    };
    use winit::raw_window_handle::RawWindowHandle;

    #[derive(Clone, Copy)]
    enum CaptureSelection {
        Dxgi,
        Wgc(isize),
    }

    /// DxgiCaptureSource (GPU mode) -> Renderer: captures the desktop straight
    /// to a GPU-resident `Pixel::D3D11` BGRA texture on the *renderer's own*
    /// `ID3D11Device` (no `Map`, no CPU pixel copy at all — see
    /// `CaptureMode::Gpu`'s own docs) and presents it directly, no `SwScaler`
    /// (desktop content is already BGRA/RGB, no YUV conversion needed, and
    /// `D3d11Renderer` letterboxes any capture size into the window on its
    /// own). Compare against the Windows-only `screen_preview_cpu`, which captures
    /// to a plain CPU `Pixel::BGRA` frame instead and converts it to YUV420P
    /// for the D3D12 CPU-upload path.
    ///
    /// No cursor: `CaptureMode::Gpu` doesn't support cursor compositing yet
    /// (see that variant's own docs) — the Windows-only `screen_preview_cpu`
    /// CPU-capture path does.
    ///
    ///     cargo run -p screen_preview_gpu -- dxgi
    ///     cargo run -p screen_preview_gpu -- wgc 0x0000000000123456
    ///     cargo run -p screen_preview_gpu -- wgc # prompts for a window
    pub(super) fn run() -> ExitCode {
        let selection = match parse_selection() {
            Ok(selection) => selection,
            Err(()) => {
                eprintln!("usage: {} [dxgi | wgc [<HWND>]]", env!("CARGO_PKG_NAME"));
                return ExitCode::FAILURE;
            }
        };
        render_common::run_window(
            "media-pp screen_preview_gpu",
            1280,
            720,
            move |target, shutdown| {
                let RawWindowHandle::Win32(handle) = target.window else {
                    panic!("screen_preview_gpu example only supports Windows");
                };
                play(
                    selection,
                    handle.hwnd.get(),
                    target.width,
                    target.height,
                    &shutdown,
                )
            },
        );
        ExitCode::SUCCESS
    }

    fn parse_selection() -> Result<CaptureSelection, ()> {
        let mut args = std::env::args().skip(1);
        match args.next().as_deref() {
            None | Some("dxgi") if args.next().is_none() => Ok(CaptureSelection::Dxgi),
            Some("wgc") => match args.next() {
                None => prompt_window_selection().map(CaptureSelection::Wgc),
                Some(value) => {
                    let hwnd = parse_hwnd(&value).ok_or(())?;
                    if args.next().is_some() {
                        return Err(());
                    }
                    Ok(CaptureSelection::Wgc(hwnd))
                }
            },
            _ => Err(()),
        }
    }

    fn parse_hwnd(value: &str) -> Option<isize> {
        let value = value.trim();
        let hex = value
            .strip_prefix("0x")
            .or_else(|| value.strip_prefix("0X"));
        let parsed = match hex {
            Some(hex) => usize::from_str_radix(hex, 16).ok()?,
            None => value.parse::<usize>().ok()?,
        };
        (parsed != 0).then_some(parsed as isize)
    }

    struct WindowEntry {
        hwnd: isize,
        title: String,
    }

    /// Lists capturable top-level windows and prompts on stdin for one,
    /// standing in for the picker `WgcCaptureSource` deliberately doesn't
    /// show itself.
    fn prompt_window_selection() -> Result<isize, ()> {
        let windows = list_capturable_windows();
        if windows.is_empty() {
            eprintln!("no capturable windows found");
            return Err(());
        }
        println!("capture which window?");
        for (index, window) in windows.iter().enumerate() {
            println!("  [{}] {}", index + 1, window.title);
        }
        print!("> ");
        io::stdout().flush().map_err(|_| ())?;
        let mut line = String::new();
        io::stdin().read_line(&mut line).map_err(|_| ())?;
        let selected = line.trim().parse::<usize>().ok().filter(|n| *n >= 1);
        selected
            .and_then(|n| windows.get(n - 1))
            .map(|window| window.hwnd)
            .ok_or(())
    }

    /// Enumerates visible, unowned, non-tool top-level windows with a
    /// title — the same rough filter Alt+Tab uses — since every other
    /// top-level window is a popup, tooltip, or helper of one of these.
    fn list_capturable_windows() -> Vec<WindowEntry> {
        unsafe extern "system" fn collect(hwnd: HWND, lparam: LPARAM) -> BOOL {
            // SAFETY: `lparam` was set below from a live `&mut Vec<WindowEntry>`
            // that outlives the `EnumWindows` call this callback runs inside.
            let windows = unsafe { &mut *(lparam.0 as *mut Vec<WindowEntry>) };
            if is_capturable(hwnd) {
                let mut buffer = [0u16; 256];
                // SAFETY: `hwnd` is a live top-level window from `EnumWindows`
                // and `buffer` is a live, correctly sized out-parameter.
                let len = unsafe { GetWindowTextW(hwnd, &mut buffer) };
                if len > 0 {
                    windows.push(WindowEntry {
                        hwnd: hwnd.0 as isize,
                        title: String::from_utf16_lossy(&buffer[..len as usize]),
                    });
                }
            }
            TRUE
        }

        let mut windows = Vec::new();
        // SAFETY: `collect` only writes through `lparam` for the duration of
        // this call, and this thread does not touch `windows` until it
        // returns.
        unsafe {
            let _ = EnumWindows(Some(collect), LPARAM(&raw mut windows as isize));
        }
        windows
    }

    fn is_capturable(hwnd: HWND) -> bool {
        // SAFETY: `hwnd` comes from `EnumWindows`, which only yields live
        // top-level windows.
        if !unsafe { IsWindowVisible(hwnd) }.as_bool() {
            return false;
        }
        // An owned window (most dialogs and tool palettes) belongs to
        // another top-level window, not a separate capture target.
        // SAFETY: `hwnd` is live; a missing owner is reported as `Err`.
        if unsafe { GetWindow(hwnd, GW_OWNER) }.is_ok() {
            return false;
        }
        // SAFETY: `hwnd` is live; reading its extended style has no
        // preconditions beyond that.
        let ex_style = unsafe { GetWindowLongW(hwnd, GWL_EXSTYLE) } as u32;
        if ex_style & WS_EX_TOOLWINDOW.0 != 0 {
            return false;
        }
        let mut cloaked = 0u32;
        // SAFETY: `hwnd` is live and `cloaked` is a correctly sized
        // out-parameter for the `DWORD` this attribute writes.
        let cloak_query = unsafe {
            DwmGetWindowAttribute(
                hwnd,
                DWMWA_CLOAKED,
                &raw mut cloaked as *mut c_void,
                size_of::<u32>() as u32,
            )
        };
        // DWM keeps suspended UWP frame windows cloaked (invisible) even
        // though `IsWindowVisible` still reports them visible; a query
        // failure means the window isn't the kind DWM cloaks at all.
        cloak_query.is_err() || cloaked == 0
    }

    fn play(
        selection: CaptureSelection,
        hwnd: isize,
        window_width: u32,
        window_height: u32,
        shutdown: &Shutdown,
    ) -> media_pp::Result<()> {
        media_pp::init()?;
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        // Renderer first, then capture: both `open_with_device` paths use
        // this exact device. WGC requires its BGRA creation flag; DXGI proves
        // the selected output belongs to the same adapter.
        let gpu =
            D3d11GpuContext::new(None).map_err(|e| media_pp::Error::Other(format!("{e:?}")))?;

        match selection {
            CaptureSelection::Dxgi => {
                let (source, _format) = DxgiCaptureSource::open_with_device(
                    "screen",
                    DxgiCaptureOptions {
                        fps: 60,
                        capture_mode: CaptureMode::Gpu,
                        ..DxgiCaptureOptions::default()
                    },
                    gpu.device(),
                )?;
                present(
                    source,
                    ElementType::DxgiCaptureSource,
                    &gpu,
                    hwnd,
                    window_width,
                    window_height,
                    shutdown,
                )
            }
            CaptureSelection::Wgc(target) => {
                let source = WgcCaptureSource::open_with_device(
                    "window",
                    HWND(target as *mut c_void),
                    WgcCaptureOptions {
                        fps: 60,
                        include_cursor: true,
                    },
                    gpu.device(),
                )?;
                present(
                    source,
                    ElementType::WgcCaptureSource,
                    &gpu,
                    hwnd,
                    window_width,
                    window_height,
                    shutdown,
                )
            }
        }
    }

    fn present<S: SourceElement + 'static>(
        source: S,
        source_type: ElementType,
        gpu: &D3d11GpuContext,
        hwnd: isize,
        window_width: u32,
        window_height: u32,
        shutdown: &Shutdown,
    ) -> media_pp::Result<()> {
        let pipeline = Pipeline::new("screen-preview-gpu", source, |source, ctx| {
            let renderer = render_common::d3d11_window_renderer(
                "renderer",
                &gpu,
                hwnd,
                window_width,
                window_height,
            )
            .expect("failed to create renderer");

            let branch = ctx
                .branch()
                .queue("captured", 4) // thread boundary so rendering doesn't block capture
                .to(Box::new(renderer))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })?;

        // Published before `run`, so a close that arrives from here on finds
        // a pipeline to stop. `true` means one already did, and nothing has
        // presented yet.
        if shutdown.publish(std::slice::from_ref(&pipeline)) {
            return Ok(());
        }

        // `run()` starts capture on a background thread and returns right
        // away — failures show up as `BusEvent::Error`. Neither source ends on
        // its own: closing the preview is what stops DXGI, and WGC's captured
        // window going away arrives as an error from the source, not an EOS.
        pipeline.run()?;

        for event in pipeline.bus().iter() {
            match &event {
                BusEvent::Eos { name, .. } => println!("[{name}] eos"),
                BusEvent::Error { name, error, .. } => eprintln!("[{name}] error: {error}"),
                BusEvent::Dropped { name, .. } => {
                    eprintln!("[{name}] dropped a buffer (queue full)")
                }
                // `BusEvent` is `#[non_exhaustive]`; this example only acts
                // on the events above.
                _ => {}
            }
            // Only stop for EOS or an error from the selected source itself;
            // an occasional dropped/backpressured frame elsewhere is not a
            // reason to end the demo.
            let source_died = matches!(
                &event,
                BusEvent::Error { element_type, .. } if *element_type == source_type
            );
            if matches!(event, BusEvent::Eos { .. }) || source_died {
                pipeline.stop();
            }
        }
        Ok(())
    }
}

/// The Linux half of the same example: capture straight into GPU memory and
/// present it, with no pixel ever passing through system memory.
///
/// The graph is one element longer than the Windows branch, and the platform
/// forces exactly that one. DXGI hands over a BGRA texture that
/// `D3d11Renderer` presents as-is; PipeWire hands over a DMA-BUF that
/// `open_gpu` imports as a BGRA CUDA surface, and `CudaRenderer` presents
/// NV12 — so `CudaConverter` sits between them. That element exists for this
/// shape: without it a GPU capture can only be encoded (NVENC ingests BGRA
/// directly), never shown or composited.
///
/// The CLI differences are the ones `screen_record_software` documents: Wayland has no
/// way to name a monitor, so the compositor prompts on the first run and
/// hands back a restore token later runs can pass to skip the dialog.
///
///     cargo run -p screen_preview_gpu -- [monitor|window] [restore-token]
#[cfg(target_os = "linux")]
mod linux_example {
    use media_pp::{
        bus::BusEvent,
        element::ElementType,
        elements::{
            CaptureSourceKind, CudaConverter, CudaDevice, PipeWireScreenCaptureOptions,
            PipeWireScreenCaptureSource,
        },
        pipeline::Pipeline,
    };
    use render_common::{Shutdown, VulkanGpuContext, WindowTarget};

    pub(super) fn run() {
        render_common::run_window(
            "media-pp screen_preview_gpu",
            1280,
            720,
            |target, shutdown| play(target, &shutdown),
        );
    }

    fn play(target: WindowTarget, shutdown: &Shutdown) -> media_pp::Result<()> {
        media_pp::init()?;
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let source_kind = match std::env::args().nth(1).as_deref() {
            Some("window") => CaptureSourceKind::Window,
            _ => CaptureSourceKind::Monitor,
        };
        // Last so it can simply be left off: it is a long opaque string that
        // only a repeat run has.
        let restore_token = std::env::args().nth(2);
        if restore_token.is_none() {
            eprintln!("opening the portal — approve the screen-share dialog to continue...");
        }

        // One CUDA context for the whole stack: the capture allocates its
        // surfaces on it, the converter allocates from it, and the renderer
        // imports its Vulkan memory into it. Each element rejects a frame
        // from a different one.
        let cuda = CudaDevice::new().map_err(|error| media_pp::Error::Other(error.to_string()))?;
        let (source, format, restore_token) = PipeWireScreenCaptureSource::open_gpu(
            "screen",
            PipeWireScreenCaptureOptions {
                fps: 60,
                source_kind,
                include_cursor: true,
                restore_token,
            },
            &cuda,
        )?;
        let gpu = VulkanGpuContext::new(target.display).map_err(media_pp::Error::Other)?;

        let (width, height) = (format.width, format.height);
        let pipeline = Pipeline::new("screen-preview-gpu", source, |source, ctx| {
            // The capture's own size, not a rounded one: the converter is
            // fixed-size, so anything else would reject every frame. An odd
            // capture is refused here rather than at the first frame — see
            // `CudaConverter`, whose chroma has no half sample to write.
            let converter = CudaConverter::new("convert", &cuda, width, height)
                .map_err(|error| media_pp::Error::Other(error.to_string()))?;
            let renderer = render_common::cuda_window_renderer(
                "renderer",
                &gpu,
                &cuda,
                target.display,
                target.window,
                target.width,
                target.height,
            )
            .map_err(media_pp::Error::Other)?;

            let branch = ctx
                .branch()
                // Thread boundary so conversion and presentation cannot stall
                // capture; the compositor keeps producing at its own rate.
                .queue("captured", 4)
                .pipe(converter)
                .to(Box::new(renderer))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })?;

        // Published before `run`, so a close that arrives from here on finds
        // a pipeline to stop. `true` means one already did — while the portal
        // dialog was up, say — and nothing has presented yet.
        if shutdown.publish(std::slice::from_ref(&pipeline)) {
            return Ok(());
        }

        println!("presenting a {width}x{height} capture — close the window to stop");
        // `run()` starts capture on a background thread and returns right
        // away — any failure shows up as a `BusEvent::Error` here instead of
        // through a returned `Result`. This source never reaches `Eos` on its
        // own; closing the window, or the captured source going away, is what
        // ends this.
        pipeline.run()?;

        for event in pipeline.bus().iter() {
            match &event {
                BusEvent::Eos { name, .. } => println!("[{name}] eos"),
                BusEvent::Error { name, error, .. } => eprintln!("[{name}] error: {error}"),
                BusEvent::Dropped { name, .. } => {
                    eprintln!("[{name}] dropped a buffer (queue full)")
                }
                // `BusEvent` is `#[non_exhaustive]`; this example only acts on
                // the events above.
                _ => {}
            }
            // Same reasoning as the Windows branch: only stop for `Eos`, or an
            // `Error` that means the capture's own `run()` thread ended — one
            // dropped frame elsewhere is not a reason to end the demo.
            let source_died = matches!(
                &event,
                BusEvent::Error { element_type, .. }
                    if *element_type == ElementType::PipeWireScreenCaptureSource
            );
            if matches!(event, BusEvent::Eos { .. }) || source_died {
                pipeline.stop();
            }
        }

        match restore_token {
            Some(token) => println!(
                "re-run without a dialog:\n  ... {} {token}",
                if matches!(source_kind, CaptureSourceKind::Window) {
                    "window"
                } else {
                    "monitor"
                }
            ),
            None => println!("the compositor issued no restore token; the next run will prompt"),
        }
        Ok(())
    }
}
