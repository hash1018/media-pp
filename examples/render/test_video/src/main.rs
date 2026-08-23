#[cfg(not(target_os = "windows"))]
fn main() {
    eprintln!("{} example only supports Windows", env!("CARGO_PKG_NAME"));
}

#[cfg(target_os = "windows")]
fn main() -> impl std::process::Termination {
    windows_example::run()
}

#[cfg(target_os = "windows")]
mod windows_example {
    use media_pp::{
        bus::BusEvent,
        elements::{D3d12Upload, SwScaler, TestVideoOptions, TestVideoSource},
        ffmpeg,
        pipeline::Pipeline,
    };
    use render_common::{D3d12GpuContext, Shutdown};
    use winit::raw_window_handle::RawWindowHandle;

    /// TestVideoSource -> Renderer: a synthetic moving-gradient stream, no
    /// file/camera/decoder involved at all, presented in a native window via
    /// `render_common`'s own `D3d12WindowRenderer` (wrapped as a
    /// `D3d12Renderer`) — proves `TestVideoSource`'s frames and
    /// `D3d12Renderer`'s CPU-upload path work end to end without needing a
    /// real video source.
    ///
    /// No `Pacer` here, deliberately, as an experiment: `TestVideoSource`
    /// self-paces with a drift-free absolute schedule (see its own docs) and
    /// nothing sits between it and the renderer here (no `SwScaler`, unlike
    /// `screen_capture`). Testing confirmed that schedule is enough on its own
    /// for a vsync-locked renderer to stay smooth without a separate pacing
    /// stage; `screen_capture` reached the same result after its source moved
    /// from variable-rate emission to the same absolute scheduling scheme.
    ///
    ///     cargo run -p test_video
    pub(super) fn run() {
        render_common::run_window("media-pp test_video", 1280, 720, |target, shutdown| {
            let RawWindowHandle::Win32(handle) = target.window else {
                panic!("test_video example only supports Windows");
            };
            play(handle.hwnd.get(), target.width, target.height, &shutdown)
        });
    }

    fn play(hwnd: isize, width: u32, height: u32, shutdown: &Shutdown) -> media_pp::Result<()> {
        media_pp::init()?;
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let options = TestVideoOptions {
            width,
            height,
            ..TestVideoOptions::default()
        };
        let source = TestVideoSource::new("test-video", options);

        let gpu = D3d12GpuContext::new().map_err(|e| media_pp::Error::Other(format!("{e:?}")))?;

        let pipeline = Pipeline::new("test-video", source, |source, ctx| {
            let renderer =
                render_common::d3d12_window_renderer("renderer", &gpu, hwnd, width, height)
                    .expect("failed to create renderer");
            let branch = ctx
                .branch()
                .queue("frames", 8) // thread boundary so rendering doesn't block generation
                // `D3d12Renderer` draws from a device resource only, so the
                // generated system-memory frames are converted to the NV12
                // layout `D3d12Upload` writes and uploaded here.
                .pipe(SwScaler::new(
                    "to-nv12",
                    ffmpeg::format::Pixel::NV12,
                    width,
                    height,
                    ffmpeg::software::scaling::Flags::BILINEAR,
                ))
                .pipe(
                    D3d12Upload::new("upload", gpu.device(), width, height)
                        .expect("failed to create the D3D12 upload"),
                )
                .to(Box::new(renderer))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })?;

        // `run()` starts playback on a background thread and returns right
        // away — any failure (e.g. an unsupported pixel format from
        // `Renderer`) shows up as a `BusEvent::Error` here instead of through
        // a returned `Result`. `TestVideoSource` never reaches `Eos` on its
        // own — closing the window is what ends this (see `Ok(())` below,
        // reached when the shared shell stops this published pipeline, or
        // when an error ends it below).
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
            if matches!(event, BusEvent::Eos { .. } | BusEvent::Error { .. }) {
                pipeline.stop();
            }
        }
        Ok(())
    }
}
