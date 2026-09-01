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
        let time_base = source.time_base();

        let gpu = D3d12GpuContext::new().map_err(|e| media_pp::Error::Other(format!("{e:?}")))?;

        let pipeline = Pipeline::new("transcode-render", source, |source, ctx| {
            let encoder = SwEncoder::new(
                "encoder",
                SwEncoderOptions {
                    codec: VideoCodec::OpenH264,
                    width,
                    height,
                    time_base,
                    frame_rate: options.framerate,
                    bit_rate: 2_000_000,
                    gop_size: 60, // ~2s @ 30fps (TestVideoOptions::default's own framerate)
                },
            )
            .expect("failed to open encoder");
            // No container/demuxer in this loop to get these from — SwEncoder
            // exposes its own codec parameters for exactly this case.
            let params = encoder.parameters();
            let decoder = SwDecoder::new("decoder", params).expect("failed to open decoder");
            let pacer = Pacer::new("pacer", time_base)?;
            let renderer =
                render_common::d3d12_window_renderer("renderer", &gpu, hwnd, width, height)
                    .expect("failed to create renderer");

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
                .pipe(
                    D3d12Upload::new("upload", gpu.device(), width, height)
                        .expect("failed to create the D3D12 upload"),
                )
                .to(Box::new(renderer))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })?;

        // `run()` starts playback on a background thread and returns right
        // away — any failure (encoder/decoder open already happened above and
        // would have panicked synchronously; a runtime failure like a bad
        // pixel format from `Renderer`) shows up as a `BusEvent::Error` here
        // instead of through a returned `Result`. `TestVideoSource` never
        // reaches `Eos` on its own — closing the window is what ends this.
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
        media_pp::init()?;
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
        let cuda = CudaDevice::new().map_err(|error| media_pp::Error::Other(error.to_string()))?;
        let gpu = VulkanGpuContext::new(target.display).map_err(media_pp::Error::Other)?;

        let pipeline = Pipeline::new("transcode-render", source, |source, ctx| {
            let encoder = SwEncoder::new(
                "encoder",
                SwEncoderOptions {
                    codec: VideoCodec::OpenH264,
                    width: target.width,
                    height: target.height,
                    time_base,
                    frame_rate: options.framerate,
                    bit_rate: 2_000_000,
                    gop_size: 60,
                },
            )?;
            let decoder = SwDecoder::new("decoder", encoder.parameters())?;
            let pacer = Pacer::new("pacer", time_base)?;
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
                .queue("to-encode", 8)
                .pipe(encoder)
                .queue("to-decode", 8)
                .pipe(decoder)
                .queue("frames", 8)
                .pipe(pacer)
                .pipe(scaler)
                .pipe(upload)
                .to(Box::new(renderer))?;
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
            match &event {
                BusEvent::Eos { name, .. } => println!("[{name}] eos"),
                BusEvent::Error { name, error, .. } => eprintln!("[{name}] error: {error}"),
                BusEvent::Dropped { name, .. } => {
                    eprintln!("[{name}] dropped a buffer (queue full)")
                }
                _ => {}
            }
            if matches!(event, BusEvent::Eos { .. } | BusEvent::Error { .. }) {
                pipeline.stop();
            }
        }
    }
}
