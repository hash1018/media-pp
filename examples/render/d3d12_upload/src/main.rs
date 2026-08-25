//! TestVideoSource -> SwScaler -> D3d12Upload -> Renderer: a synthetic
//! `Pixel::YUV420P` stream converted to `Pixel::NV12` on the CPU, then
//! uploaded to a GPU `Pixel::D3D12` texture on the *renderer's own*
//! `ID3D12Device` before being presented — proves `D3d12Upload`'s frames
//! are structurally identical to `D3d12Decoder`'s own (same
//! `AVD3D12VAFrame` payload), so `D3d12Renderer` takes its zero-copy path
//! unmodified even though nothing here ever decoded anything. Every stage sits
//! behind its own `Queue` so each one is exercised on a separate thread;
//! `test_video` runs the same conversion and upload as a single-thread tail
//! instead, to show `TestVideoSource` pacing itself without a `Pacer`.
//!
//!     cargo run -p d3d12_upload

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
        elements::{D3d12Upload, SwScaler, TestVideoOptions, TestVideoSource},
        pipeline::Pipeline,
    };
    use render_common::{D3d12GpuContext, Shutdown};
    use winit::raw_window_handle::RawWindowHandle;

    pub(super) fn run() {
        render_common::run_window("media-pp d3d12_upload", 1280, 720, |target, shutdown| {
            let RawWindowHandle::Win32(handle) = target.window else {
                panic!("d3d12_upload example only supports Windows");
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

        let pipeline = Pipeline::new("d3d12-upload", source, |source, ctx| {
            // `Pixel::NV12` — the only layout `D3d12Upload`/`D3d12Renderer`'s
            // zero-copy path accepts.
            let scaler = SwScaler::new(
                "to-nv12",
                ffmpeg::format::Pixel::NV12,
                width,
                height,
                ffmpeg::software::scaling::Flags::BILINEAR,
            );
            // Same device the renderer draws with — required for the
            // zero-copy path to be valid at all (see D3d12Upload::new).
            let upload = D3d12Upload::new("upload", gpu.device(), width, height)
                .expect("failed to open D3D12Upload");
            let renderer =
                render_common::d3d12_window_renderer("renderer", &gpu, hwnd, width, height)
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
