//! TestVideoSource -> SwEncoder -> SwDecoder -> Pacer -> SwScaler -> GPU upload
//! -> Renderer: encodes a synthetic moving-gradient stream (via `libopenh264`)
//! and decodes it straight back — no file, camera, or container/mux involved at
//! all — presented in a native window at real playback speed. Proves
//! `SwEncoder`'s `Packet`s are actually valid, decodable H.264 (not just
//! "avcodec_open2 succeeded"): if the round trip corrupted anything, the
//! gradient would visibly glitch or freeze instead of scrolling smoothly.
//! This example keeps a `Pacer` after the encode/decode round trip. The
//! source itself is already paced accurately enough for direct rendering,
//! but the encoder and decoder add their own buffering and per-frame
//! variance; this particular chain has not been validated without the
//! final clock-anchored pacing stage.
//!
//!     cargo run -p transcode_render

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
    use media_pp::{
        bus::BusEvent,
        elements::{
            D3d12Upload, Pacer, SwDecoder, SwEncoder, SwEncoderOptions, SwScaler, TestVideoOptions,
            TestVideoSource, VideoCodec,
        },
        ffmpeg,
        pipeline::Pipeline,
    };
    use render_common::{D3d12GpuContext, Shutdown};
    use winit::raw_window_handle::RawWindowHandle;

    pub(super) fn run() {
        render_common::run_window(
            "media-pp transcode_render",
            1280,
            720,
            |target, shutdown| {
                let RawWindowHandle::Win32(handle) = target.window else {
                    panic!("transcode_render example only supports Windows");
                };
                play(handle.hwnd.get(), target.width, target.height, &shutdown)
            },
        );
    }

    fn play(hwnd: isize, width: u32, height: u32, shutdown: &Shutdown) -> media_pp::Result<()> {
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

        let gpu = D3d12GpuContext::new()?;

        let (pipeline, ()) = Pipeline::new("transcode-render", source, |source, ctx| {
            let encoder = SwEncoder::new(
                "encoder",
                SwEncoderOptions {
                    codec: VideoCodec::OpenH264,
                    width,
                    height,
                    frame_rate: options.frame_rate,
                    bit_rate: 2_000_000,
                    gop_size: 60, // ~2s @ 30fps (TestVideoOptions::default's own framerate)
                    max_b_frames: None,
                },
            )?;
            // No container/demuxer in this loop to get these from — SwEncoder
            // exposes its own codec parameters for exactly this case.
            let params = encoder.parameters();
            let decoder = SwDecoder::new("decoder", params)?;
            let pacer = Pacer::new("pacer");
            let renderer =
                render_common::d3d12_window_renderer("renderer", &gpu, hwnd, width, height)?;

            let branch = ctx
                .branch()
                .queue("to-encode", 8) // let generation run ahead of the (CPU-heavy) encoder
                .pipe(encoder)
                .queue("to-decode", 8) // let encode run ahead of decode
                .pipe(decoder)
                .queue("frames", 8) // pacer sleeps on its own thread; let decode run ahead into this
                .pipe(pacer)
                // `D3d12Renderer` draws from a device resource only, so the
                // system-memory frames are converted to the NV12 layout
                // `D3d12Upload` writes and uploaded here.
                .pipe(SwScaler::new(
                    "to-nv12",
                    ffmpeg::format::Pixel::NV12,
                    width,
                    height,
                    ffmpeg::software::scaling::Flags::BILINEAR,
                ))
                .pipe(D3d12Upload::new("upload", gpu.device())?)
                .to(renderer)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })?;

        // `run()` starts playback on a background thread and returns right
        // away. Opening the encoder and decoder already happened above, and a
        // failure there came back through `?`; a runtime failure — a bad pixel
        // format from `Renderer` — shows up as a `BusEvent::Error` here instead.
        // `TestVideoSource` never reaches `Eos` on its own — closing the window
        // is what ends this.
        if shutdown.publish(std::slice::from_ref(&pipeline)) {
            return Ok(());
        }

        pipeline.run()?;

        for event in pipeline.bus().iter() {
            println!("{event}");
            if matches!(event, BusEvent::Finished | BusEvent::Error { .. }) {
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
        elements::{
            CudaDevice, CudaFrameFormat, CudaUpload, Pacer, SwDecoder, SwEncoder, SwEncoderOptions,
            SwScaler, TestVideoOptions, TestVideoSource, VideoCodec,
        },
        ffmpeg,
        pipeline::Pipeline,
    };
    use render_common::{Shutdown, VulkanGpuContext, WindowTarget};

    pub(super) fn run() {
        render_common::run_window(
            "media-pp transcode_render",
            1280,
            720,
            |target, shutdown| play(target, &shutdown),
        );
    }

    fn play(target: WindowTarget, shutdown: &Shutdown) -> media_pp::Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let options = TestVideoOptions {
            width: target.width,
            height: target.height,
            ..TestVideoOptions::default()
        };
        let source = TestVideoSource::new("test-video", options);
        let time_base = source.time_base();
        let cuda = CudaDevice::new()?;
        let gpu = VulkanGpuContext::new(target.display)?;

        let (pipeline, ()) = Pipeline::new("transcode-render", source, |source, ctx| {
            let encoder = SwEncoder::new(
                "encoder",
                SwEncoderOptions {
                    codec: VideoCodec::OpenH264,
                    width: target.width,
                    height: target.height,
                    time_base,
                    frame_rate: options.frame_rate,
                    bit_rate: 2_000_000,
                    gop_size: 60,
                    max_b_frames: None,
                },
            )?;
            let decoder = SwDecoder::new("decoder", encoder.parameters())?;
            let pacer = Pacer::new("pacer");
            let scaler = SwScaler::new(
                "to-nv12",
                ffmpeg::format::Pixel::NV12,
                target.width,
                target.height,
                ffmpeg::software::scaling::Flags::BILINEAR,
            );
            let upload = CudaUpload::new("upload", &cuda, CudaFrameFormat::Nv12)?;
            let renderer = render_common::cuda_window_renderer(
                "renderer",
                &gpu,
                &cuda,
                target.display,
                target.window,
                target.width,
                target.height,
            )?;

            let branch = ctx
                .branch()
                .queue("to-encode", 8)
                .pipe(encoder)
                .queue("to-decode", 8)
                .pipe(decoder)
                .queue("frames", 8)
                .pipe(pacer)
                .pipe(scaler)
                .pipe(upload)
                .to(renderer)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })?;

        if shutdown.publish(std::slice::from_ref(&pipeline)) {
            return Ok(());
        }
        pipeline.run()?;
        drain_bus(&pipeline);
        Ok(())
    }

    fn drain_bus(pipeline: &Pipeline) {
        for event in pipeline.bus().iter() {
            println!("{event}");
            if matches!(event, BusEvent::Finished | BusEvent::Error { .. }) {
                pipeline.stop();
            }
        }
    }
}
