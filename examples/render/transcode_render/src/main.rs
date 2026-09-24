//! TestVideoSource -> SwEncoder -> SwDecoder -> Pacer -> SwScaler -> GPU upload
//! -> Renderer (on Linux, `VulkanWindowRenderer` straight after the `Pacer`,
//! drawing the decoded YUV420P as it comes): encodes a synthetic moving-gradient stream (via `libopenh264`)
//! and decodes it straight back — no file, camera, or container/mux involved at
//! all — presented in a native window at real playback speed. Proves
//! `SwEncoder`'s `Packet`s are actually valid, decodable H.264 (not just
//! "avcodec_open2 succeeded"): if the round trip corrupted anything, the
//! gradient would visibly glitch or freeze instead of scrolling smoothly.
//! This example keeps a `Pacer` after the encode/decode round trip. The
//! source itself is already paced accurately enough for direct rendering,
//! but the encoder and decoder add their own buffering and per-frame
//! variance; this particular chain has not been validated without the
//! final clock-anchored pacing stage. On Windows the frames are uploaded to
//! D3D12 and drawn into `D3d12WindowRenderer`'s own window.
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
            D3d12Gpu, D3d12Upload, D3d12WindowRenderer, Pacer, SwDecoder, SwEncoder,
            SwEncoderOptions, SwScaler, TestVideoOptions, TestVideoSource, VideoCodec,
            WindowOptions,
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

        let gpu = D3d12Gpu::new()?;
        let window_options = WindowOptions {
            title: "media-pp transcode_render".into(),
            ..WindowOptions::default()
        };
        let (width, height) = (window_options.width, window_options.height);
        let (renderer, window) = D3d12WindowRenderer::open("renderer", &gpu, window_options)?;
        let shutdown = render_common::stop_on_close([window]);

        let options = TestVideoOptions {
            width,
            height,
            ..TestVideoOptions::default()
        };
        let source = TestVideoSource::new("test-video", options);

        let (pipeline, ()) = Pipeline::new("transcode-render", source, |source, ctx| {
            let encoder = SwEncoder::new(
                "encoder",
                SwEncoderOptions {
                    codec: VideoCodec::OpenH264,
                    width,
                    height,
                    pixel_format: ffmpeg::format::Pixel::YUV420P,
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

            let branch = ctx
                .branch()
                .queue("to-encode", 8) // let generation run ahead of the (CPU-heavy) encoder
                .pipe(encoder)
                .queue("to-decode", 8) // let encode run ahead of decode
                .pipe(decoder)
                .queue("frames", 8) // pacer sleeps on its own thread; let decode run ahead into this
                .pipe(pacer)
                // `D3d12WindowRenderer` draws from a device resource only, so the
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
            Pacer, SwDecoder, SwEncoder, SwEncoderOptions, TestVideoOptions, TestVideoSource,
            VideoCodec, VulkanGpu, VulkanWindowRenderer, WindowOptions,
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

        // OpenH264 decodes to YUV420P, which the renderer draws as it comes
        // and uploads itself.
        let gpu = VulkanGpu::new()?;
        let window_options = WindowOptions {
            title: "media-pp transcode_render".into(),
            ..WindowOptions::default()
        };
        let (width, height) = (window_options.width, window_options.height);
        let (renderer, window) = VulkanWindowRenderer::open("renderer", &gpu, window_options)?;
        let shutdown = render_common::stop_on_close([window]);

        let options = TestVideoOptions {
            width,
            height,
            ..TestVideoOptions::default()
        };
        let source = TestVideoSource::new("test-video", options);

        let (pipeline, ()) = Pipeline::new("transcode-render", source, |source, ctx| {
            let encoder = SwEncoder::new(
                "encoder",
                SwEncoderOptions {
                    codec: VideoCodec::OpenH264,
                    width,
                    height,
                    pixel_format: ffmpeg::format::Pixel::YUV420P,
                    frame_rate: options.frame_rate,
                    bit_rate: 2_000_000,
                    gop_size: 60,
                    max_b_frames: None,
                },
            )?;
            let decoder = SwDecoder::new("decoder", encoder.parameters())?;
            let branch = ctx
                .branch()
                .queue("to-encode", 8)
                .pipe(encoder)
                .queue("to-decode", 8)
                .pipe(decoder)
                .queue("frames", 8)
                .pipe(Pacer::new("pacer"))
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
