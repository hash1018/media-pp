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
    use media_pp::ffmpeg;
    use media_pp::{
        bus::BusEvent,
        elements::{D3d11Upload, SwScaler, TestVideoOptions, TestVideoSource},
        pipeline::Pipeline,
    };
    use render_common::{D3d11GpuContext, Shutdown};
    use winit::raw_window_handle::RawWindowHandle;

    /// TestVideoSource -> SwScaler -> D3d11Upload -> Renderer: a synthetic
    /// `Pixel::YUV420P` stream converted to `Pixel::NV12` on the CPU, then
    /// uploaded to a GPU `Pixel::D3D11` texture on the *renderer's own*
    /// `ID3D11Device` before being presented — proves `D3d11Upload`'s frames
    /// (built via plain `windows-rs` calls + `av_buffer_create`, not FFmpeg's
    /// own hwframe pool — see `D3d11Upload`'s own docs) are readable by
    /// `D3d11Renderer`'s zero-copy path. Compare against `d3d12_upload`, the
    /// D3D12 sibling of this same smoke test.
    ///
    ///     cargo run -p d3d11_upload
    pub(super) fn run() {
        render_common::run_window("media-pp d3d11_upload", 1280, 720, |target, shutdown| {
            let RawWindowHandle::Win32(handle) = target.window else {
                panic!("d3d11_upload example only supports Windows");
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

        let gpu =
            D3d11GpuContext::new(None).map_err(|e| media_pp::Error::Other(format!("{e:?}")))?;

        let pipeline = Pipeline::new("d3d11-upload", source, |source, ctx| {
            // `Pixel::NV12` — the only layout `D3d11Upload` accepts.
            let scaler = SwScaler::new(
                "to-nv12",
                ffmpeg::format::Pixel::NV12,
                width,
                height,
                ffmpeg::software::scaling::Flags::BILINEAR,
            );
            // Same device the renderer draws with — required for the
            // zero-copy path to be valid at all (see D3d11Upload::new).
            let upload = D3d11Upload::new("upload", gpu.device(), width, height);
            let renderer =
                render_common::d3d11_window_renderer("renderer", &gpu, hwnd, width, height)
                    .expect("failed to create renderer");

            let branch = ctx
                .branch()
                .queue("generated", 4) // thread boundary so scaling doesn't block generation
                .pipe(scaler)
                .queue("scaled", 4) // thread boundary so uploading doesn't block scaling
                .pipe(upload)
                .queue("frames", 8) // thread boundary so rendering doesn't block uploading
                .to(Box::new(renderer))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })?;

        // `run()` starts playback on a background thread and returns right
        // away — any failure (e.g. an unsupported pixel format anywhere in
        // the chain) shows up as a `BusEvent::Error` here instead of through a
        // returned `Result`. `TestVideoSource` never reaches `Eos` on its own
        // — closing the window is what ends this.
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
