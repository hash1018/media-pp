//! Captures the desktop straight into GPU memory and presents it without a
//! system-memory pixel round trip.
//!
//! - Windows: `DxgiCaptureSource(GPU) -> Queue -> D3d11Renderer`
//! - Linux: `PipeWireScreenCaptureSource(GPU) -> Queue -> CudaConverter ->
//!   CudaRenderer(Vulkan)`
//!
//! ```text
//! cargo run -p screen_preview_gpu
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
    use media_pp::{
        bus::BusEvent,
        element::ElementType,
        elements::{CaptureMode, DxgiCaptureOptions, DxgiCaptureSource},
        pipeline::Pipeline,
    };
    use render_common::{D3d11GpuContext, Shutdown};
    use winit::raw_window_handle::RawWindowHandle;

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
    ///     cargo run -p screen_preview_gpu
    pub(super) fn run() {
        render_common::run_window(
            "media-pp screen_preview_gpu",
            1280,
            720,
            |target, shutdown| {
                let RawWindowHandle::Win32(handle) = target.window else {
                    panic!("screen_preview_gpu example only supports Windows");
                };
                play(handle.hwnd.get(), target.width, target.height, &shutdown)
            },
        );
    }

    fn play(
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

        // Opened first: `CaptureMode::Gpu` builds its own device from
        // whichever adapter `output_index` actually selects and hands it
        // back — `render_common::d3d11_window_renderer` below is built from
        // that same returned device, required for the zero-copy path to be
        // valid at all (see `CaptureMode::Gpu`'s own docs on why the device
        // flows this direction, not the other way).
        let capture_options = DxgiCaptureOptions {
            fps: 60,
            capture_mode: CaptureMode::Gpu,
            ..DxgiCaptureOptions::default()
        };
        let (source, _format, device) = DxgiCaptureSource::open("screen", capture_options)?;
        let device = device.expect("CaptureMode::Gpu always returns a device");

        let gpu = D3d11GpuContext::new(Some(device))
            .map_err(|e| media_pp::Error::Other(format!("{e:?}")))?;

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
        // away — any failure shows up as a `BusEvent::Error` here instead of
        // through a returned `Result`. `DxgiCaptureSource` never reaches `Eos`
        // on its own — closing the window is what ends this.
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
            // Same reasoning as `screen_preview_cpu`'s own loop: only stop for
            // `Eos`, or an `Error` that means `DxgiCaptureSource`'s own `run()`
            // thread actually ended — an occasional dropped/backpressured
            // frame elsewhere isn't a reason to end the demo.
            let source_died = matches!(
                &event,
                BusEvent::Error { element_type, .. } if *element_type == ElementType::DxgiCaptureSource
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
