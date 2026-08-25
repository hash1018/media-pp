//! Previews desktop capture through the CPU-frame path —
//! `CaptureSource -> Queue -> SwScaler(NV12) -> Queue -> GPU upload ->
//! Renderer`. Windows uses DXGI and D3D12; Linux uses the desktop
//! portal/PipeWire and CUDA/Vulkan. The capture includes the cursor, converts
//! directly to window-sized NV12, and needs no encode/decode round trip or
//! separate `Pacer`.
//!
//!     cargo run -p screen_preview_cpu

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn main() {
    eprintln!(
        "{} example only supports Windows and Linux",
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
    use media_pp::ffmpeg;
    use media_pp::{
        bus::BusEvent,
        element::ElementType,
        elements::{CaptureMode, D3d12Upload, DxgiCaptureOptions, DxgiCaptureSource, SwScaler},
        pipeline::Pipeline,
    };
    use render_common::{D3d12GpuContext, Shutdown};
    use winit::raw_window_handle::RawWindowHandle;

    /// DxgiCaptureSource -> SwScaler -> D3d12Upload -> Renderer: captures the
    /// desktop live via DXGI Desktop Duplication (cursor included) at a
    /// constant frame rate (`DxgiCaptureOptions::fps`), converts/resizes it to
    /// the window's own size as `Pixel::NV12` in one pass, and uploads that to
    /// the GPU — no `SwEncoder`/`SwDecoder` round trip.
    ///
    /// No `Pacer` here, confirmed unneeded: `DxgiCaptureSource` previously
    /// emitted variable-rate (real wall-clock pts, push-on-change), and
    /// removing `Pacer` against that measurably caused judder. It's since
    /// been rewritten to emit at a constant rate on a drift-free absolute
    /// schedule instead — the same pattern `TestVideoSource` uses (see
    /// `test_video`) — and with that fixed, `SwScaler` sitting between source
    /// and renderer here doesn't add enough jitter on its own to bring the
    /// judder back. The constant-rate/drift-free change was the actual fix,
    /// not the presence of a `Pacer` stage.
    ///
    ///     cargo run -p screen_preview_cpu
    pub(super) fn run() {
        render_common::run_window(
            "media-pp screen_preview_cpu",
            1280,
            720,
            |target, shutdown| {
                let RawWindowHandle::Win32(handle) = target.window else {
                    panic!("screen_preview_cpu example only supports Windows");
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

        let capture_options = DxgiCaptureOptions {
            fps: 60,
            capture_mode: CaptureMode::Cpu {
                include_cursor: true,
            },
            ..DxgiCaptureOptions::default()
        };
        let (source, _format, _device) = DxgiCaptureSource::open("screen", capture_options)?;

        let gpu = D3d12GpuContext::new().map_err(|e| media_pp::Error::Other(format!("{e:?}")))?;

        let pipeline = Pipeline::new("screen-preview-cpu", source, |source, ctx| {
            // Converts the captured `Pixel::BGRA` desktop frames down to the
            // window's own size as `Pixel::NV12` in one pass — the layout
            // `D3d12Upload` writes, and the only one it accepts.
            let scaler = SwScaler::new(
                "to-nv12",
                ffmpeg::format::Pixel::NV12,
                window_width,
                window_height,
                ffmpeg::software::scaling::Flags::BILINEAR,
            );
            // `D3d12Renderer` draws from a device resource only, so this is
            // where the captured pixels cross to the GPU.
            let upload = D3d12Upload::new("upload", gpu.device(), window_width, window_height)
                .expect("failed to create the D3D12 upload");
            let renderer = render_common::d3d12_window_renderer(
                "renderer",
                &gpu,
                hwnd,
                window_width,
                window_height,
            )
            .expect("failed to create renderer");

            let branch = ctx
                .branch()
                .queue("captured", 4) // thread boundary so scaling doesn't block capture
                .pipe(scaler)
                .queue("frames", 8) // thread boundary so rendering doesn't block scaling
                .pipe(upload)
                .to(Box::new(renderer))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })?;

        // `run()` starts capture on a background thread and returns right
        // away — any failure shows up as a `BusEvent::Error` here instead of
        // through a returned `Result`. `DxgiCaptureSource` never reaches `Eos`
        // on its own — closing the window is what ends this.
        if shutdown.publish(std::slice::from_ref(&pipeline)) {
            return Ok(());
        }

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
            // Unlike the other render examples (a steady, self-paced
            // synthetic/file source that essentially never overruns the
            // renderer's 2-slot upload ring), live desktop capture can
            // legitimately burst faster than that ring drains — e.g. right at
            // startup, before the render thread has processed even its first
            // frame. `D3d12RendererError::Submit(NoFreeSlot)` on an occasional
            // frame is expected backpressure, not a reason to end the whole
            // demo — the `Queue` in front of the renderer already drops just
            // that one buffer and keeps going (see `Queue`'s own "report,
            // don't die" contract). Only stop for `Eos`, or an `Error` from
            // `DxgiCaptureSource` itself (its `run()` thread actually ended —
            // e.g. `DXGI_ERROR_ACCESS_LOST` from a lock screen — so nothing
            // more will ever arrive).
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

#[cfg(target_os = "linux")]
mod linux_example {
    use media_pp::{
        bus::BusEvent,
        element::ElementType,
        elements::{
            CaptureSourceKind, CudaDevice, CudaFrameFormat, CudaUpload,
            PipeWireScreenCaptureOptions, PipeWireScreenCaptureSource, SwScaler,
        },
        ffmpeg,
        pipeline::Pipeline,
    };
    use render_common::{Shutdown, VulkanGpuContext, WindowTarget};

    pub(super) fn run() {
        render_common::run_window(
            "media-pp screen_preview_cpu",
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
        let restore_token = std::env::args().nth(2);
        if restore_token.is_none() {
            eprintln!("opening the portal — approve the screen-share dialog to continue...");
        }
        let (source, capture_format, restore_token) = PipeWireScreenCaptureSource::open(
            "screen",
            PipeWireScreenCaptureOptions {
                fps: 60,
                source_kind,
                include_cursor: true,
                restore_token,
            },
        )?;

        let cuda = CudaDevice::new().map_err(|error| media_pp::Error::Other(error.to_string()))?;
        let gpu = VulkanGpuContext::new(target.display).map_err(media_pp::Error::Other)?;

        let pipeline = Pipeline::new("screen-preview-cpu", source, |source, ctx| {
            let scaler = SwScaler::new(
                "to-nv12",
                ffmpeg::format::Pixel::NV12,
                target.width,
                target.height,
                ffmpeg::software::scaling::Flags::BILINEAR,
            );
            let upload = CudaUpload::new(
                "upload",
                &cuda,
                CudaFrameFormat::Nv12,
                target.width,
                target.height,
            )
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
                .queue("captured", 4)
                .pipe(scaler)
                .queue("frames", 8)
                .pipe(upload)
                .to(Box::new(renderer))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })?;

        if shutdown.publish(std::slice::from_ref(&pipeline)) {
            return Ok(());
        }
        println!(
            "presenting a {}x{} capture — close the window to stop",
            capture_format.width, capture_format.height
        );
        pipeline.run()?;

        for event in pipeline.bus().iter() {
            match &event {
                BusEvent::Eos { name, .. } => println!("[{name}] eos"),
                BusEvent::Error { name, error, .. } => eprintln!("[{name}] error: {error}"),
                BusEvent::Dropped { name, .. } => {
                    eprintln!("[{name}] dropped a buffer (queue full)")
                }
                _ => {}
            }
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
                "re-run without a dialog:\n  cargo run -p screen_preview_cpu -- {} {token}",
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
