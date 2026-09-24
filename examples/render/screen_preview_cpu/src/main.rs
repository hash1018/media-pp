//! Previews desktop capture through the CPU-frame path. Windows is
//! `DxgiCaptureSource -> Queue -> SwScaler(NV12) -> Queue -> D3d12Upload ->
//! D3d12WindowRenderer`, converting directly to window-sized NV12 and
//! drawing it in the renderer's own window. Linux is
//! `PipeWireScreenCaptureSource -> Queue -> VulkanWindowRenderer`: the
//! renderer draws the capture's system-memory BGRA as it comes, uploading
//! and scaling it itself. The capture includes the cursor, and needs no
//! encode/decode round trip or separate `Pacer`.
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
        elements::{
            CaptureMode, D3d12Gpu, D3d12Upload, D3d12WindowRenderer, DxgiCaptureOptions,
            DxgiCaptureSource, SwScaler, WindowOptions,
        },
        pipeline::Pipeline,
    };

    /// DxgiCaptureSource -> SwScaler -> D3d12Upload -> D3d12WindowRenderer:
    /// captures the desktop live via DXGI Desktop Duplication (cursor
    /// included) at a constant frame rate (`DxgiCaptureOptions::frame_rate`),
    /// converts/resizes it to the window's own size as `Pixel::NV12` in one
    /// pass, and uploads that to the GPU — no `SwEncoder`/`SwDecoder` round
    /// trip.
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
    pub(super) fn run() -> media_pp::Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let capture_options = DxgiCaptureOptions {
            frame_rate: ffmpeg::Rational::new(60, 1),
            capture_mode: CaptureMode::Cpu {
                include_cursor: true,
            },
            ..DxgiCaptureOptions::default()
        };
        let (source, _format, _device) = DxgiCaptureSource::open("screen", capture_options)?;

        let gpu = D3d12Gpu::new()?;
        let window_options = WindowOptions {
            title: "media-pp screen_preview_cpu".into(),
            ..WindowOptions::default()
        };
        let (window_width, window_height) = (window_options.width, window_options.height);
        let (renderer, window) = D3d12WindowRenderer::open("renderer", &gpu, window_options)?;
        let shutdown = render_common::stop_on_close([window]);

        let (pipeline, ()) = Pipeline::new("screen-preview-cpu", source, |source, ctx| {
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
            // `D3d12WindowRenderer` draws from a device resource only, so this
            // is where the captured pixels cross to the GPU.
            let upload = D3d12Upload::new("upload", gpu.device())?;

            let branch = ctx
                .branch()
                .queue("captured", 4) // thread boundary so scaling doesn't block capture
                .pipe(scaler)
                .queue("frames", 8) // thread boundary so rendering doesn't block scaling
                .pipe(upload)
                .to(renderer)?;
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
            println!("{event}");
            // An error on an occasional frame downstream is not a reason to
            // end the whole demo — the `Queue` in front of the failing stage
            // already drops just that one buffer and keeps going (see
            // `Queue`'s own "report, don't die" contract). Only stop for
            // `Finished`, or an `Error` from `DxgiCaptureSource` itself (its
            // `run()` thread actually ended — e.g. `DXGI_ERROR_ACCESS_LOST`
            // from a lock screen — so nothing more will ever arrive).
            let source_died = matches!(
                &event,
                BusEvent::Error { element_type, .. } if *element_type == ElementType::DxgiCaptureSource
            );
            if matches!(event, BusEvent::Finished) || source_died {
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
            CaptureSourceKind, PipeWireScreenCaptureOptions, PipeWireScreenCaptureSource,
            VulkanGpu, VulkanWindowRenderer, WindowOptions,
        },
        ffmpeg,
        pipeline::Pipeline,
    };

    pub(super) fn run() -> media_pp::Result<()> {
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
                frame_rate: ffmpeg::Rational::new(60, 1),
                source_kind,
                include_cursor: true,
                restore_token,
            },
        )?;

        // The capture's BGRA in system memory, drawn as it comes: the renderer
        // uploads it itself and scales it to the window as it draws.
        let gpu = VulkanGpu::new()?;
        let (renderer, window) = VulkanWindowRenderer::open(
            "renderer",
            &gpu,
            WindowOptions {
                title: "media-pp screen_preview_cpu".into(),
                ..WindowOptions::default()
            },
        )?;
        let shutdown = render_common::stop_on_close([window]);

        let (pipeline, ()) = Pipeline::new("screen-preview-cpu", source, |source, ctx| {
            let branch = ctx.branch().queue("captured", 4).to(renderer)?;
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
            println!("{event}");
            let source_died = matches!(
                &event,
                BusEvent::Error { element_type, .. }
                    if *element_type == ElementType::PipeWireScreenCaptureSource
            );
            if matches!(event, BusEvent::Finished) || source_died {
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
