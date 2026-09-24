//! TestVideoSource -> SwScaler -> D3d12Upload -> D3d12WindowRenderer: a
//! synthetic `Pixel::YUV420P` stream converted to `Pixel::NV12` on the CPU,
//! then uploaded to a GPU `Pixel::D3D12` texture on the *renderer's own*
//! `ID3D12Device` before being presented — proves `D3d12Upload`'s frames
//! are structurally identical to `D3d12Decoder`'s own (same
//! `AVD3D12VAFrame` payload), so the renderer takes its zero-copy path
//! unmodified even though nothing here ever decoded anything. Every stage sits
//! behind its own `Queue` so each one is exercised on a separate thread;
//! `test_video` runs the same conversion and upload as a single-thread tail
//! instead, to show `TestVideoSource` pacing itself without a `Pacer`.
//!
//! The window is the renderer's own: `D3d12WindowRenderer::open` opens it on
//! a thread of its own, the way a GStreamer video sink does, and reports what
//! happens to it — Space pauses and resumes, Escape or closing the window
//! stops. The one device every element shares is a `D3d12Gpu`.
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
    use std::{sync::mpsc::RecvTimeoutError, time::Duration};

    use media_pp::ffmpeg;
    use media_pp::{
        bus::BusEvent,
        elements::{
            D3d12Gpu, D3d12Upload, D3d12WindowRenderer, Key, SwScaler, TestVideoOptions,
            TestVideoSource, WindowEvent, WindowOptions,
        },
        pipeline::Pipeline,
    };

    pub(super) fn run() -> media_pp::Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let window_options = WindowOptions {
            title: "media-pp d3d12_upload".into(),
            ..WindowOptions::default()
        };
        let options = TestVideoOptions {
            width: window_options.width,
            height: window_options.height,
            ..TestVideoOptions::default()
        };
        let source = TestVideoSource::new("test-video", options);

        let gpu = D3d12Gpu::new()?;
        let (screen, window) = D3d12WindowRenderer::open("screen", &gpu, window_options)?;

        let (pipeline, ()) = Pipeline::new("d3d12-upload", source, |source, ctx| {
            // `Pixel::NV12` — the only layout `D3d12Upload` and the
            // renderer's zero-copy path accept.
            let scaler = SwScaler::to_format(
                "to-nv12",
                ffmpeg::format::Pixel::NV12,
                ffmpeg::software::scaling::Flags::BILINEAR,
            );
            // Same device the renderer draws with — required for the
            // zero-copy path to be valid at all (see D3d12Upload::new).
            let upload = D3d12Upload::new("upload", gpu.device())?;

            let branch = ctx
                .branch()
                .queue("generated", 4) // thread boundary so scaling doesn't block generation
                .pipe(scaler)
                .queue("scaled", 4) // thread boundary so uploading doesn't block scaling
                .pipe(upload)
                .queue("frames", 8) // thread boundary so rendering doesn't block uploading
                .to(screen)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })?;

        pipeline.run()?;

        // Two things to watch: the pipeline's bus — an error does not end a
        // pipeline on its own, and `TestVideoSource` never reaches `Eos` —
        // and the window, whose keys and closing are this program's to act
        // on.
        let mut paused = false;
        loop {
            match pipeline.bus().recv_timeout(Duration::from_millis(20)) {
                Ok(event) => {
                    println!("{event}");
                    if matches!(event, BusEvent::Finished | BusEvent::Error { .. }) {
                        break;
                    }
                }
                Err(RecvTimeoutError::Disconnected) => break,
                Err(RecvTimeoutError::Timeout) => {}
            }
            match window.try_recv() {
                Some(WindowEvent::Closed | WindowEvent::Key(Key::Escape)) => break,
                Some(WindowEvent::Key(Key::Space)) => {
                    paused = !paused;
                    if paused {
                        pipeline.pause();
                    } else {
                        pipeline.resume();
                    }
                }
                _ => {}
            }
        }
        pipeline.stop();
        Ok(())
    }
}
