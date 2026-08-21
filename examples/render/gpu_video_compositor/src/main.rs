//! Two live video sources composited on the GPU, displayed and recorded at
//! the same time — `2 x (source -> upload) -> compositor -> Tee -> {renderer,
//! download -> encode -> mp4}`.
//!
//! Both platforms run the identical graph, terminal sinks, and CLI; only the
//! GPU stack differs — `D3d11Upload`/`D3d11VideoCompositor`/`D3d11Download`
//! on Windows, `CudaUpload`/`CudaVideoCompositor`/`CudaDownload` on Linux.
//!
//!     cargo run -p gpu_video_compositor -- [output.mp4] [seconds]

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn main() {
    eprintln!(
        "{} supports Windows (D3D11) and Linux (CUDA) only",
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
    use std::{thread, time::Duration};

    use ffmpeg_next as ffmpeg;
    use media_pp::{
        bus::BusEvent,
        color::Color,
        elements::{
            D3d11Download, D3d11Upload, D3d11VideoCompositor, Mp4Muxer, SwEncoder,
            SwEncoderOptions, SwScaler, TeeBuilder, TestVideoOptions, TestVideoSource, VideoCodec,
            VideoCompositorOptions, VideoFit, VideoLayer, VideoRect,
        },
        pipeline::Pipeline,
    };
    use render_common::{D3d11GpuContext, Shutdown};
    use winit::raw_window_handle::RawWindowHandle;

    /// Two TestVideoSource pipelines -> D3d11Upload -> D3d11VideoCompositor
    /// (GPU shader compositing, no CPU round trip for the inputs or the
    /// composited output) -> Tee -> {D3d11Renderer for live display,
    /// D3d11Download -> SwScaler -> SwEncoder -> Mp4Muxer for simultaneous
    /// recording}. The foreground layer moves at runtime through its
    /// D3d11VideoLayerHandle, same as the CPU `video_compositor` example, but
    /// every frame this composites never touches the CPU until the recording
    /// branch's own D3d11Download.
    ///
    ///     cargo run -p gpu_video_compositor -- [output.mp4] [seconds]
    pub(super) fn run() {
        let path = std::env::args()
            .nth(1)
            .unwrap_or_else(|| "gpu_video_compositor.mp4".into());
        let seconds: u64 = std::env::args()
            .nth(2)
            .and_then(|value| value.parse().ok())
            .unwrap_or(5);

        render_common::run_window(
            "media-pp gpu_video_compositor",
            640,
            360,
            move |target, shutdown| {
                let RawWindowHandle::Win32(handle) = target.window else {
                    panic!("gpu_video_compositor example only supports Windows");
                };
                play(handle.hwnd.get(), &path, seconds, &shutdown)
            },
        );
    }

    fn play(hwnd: isize, path: &str, seconds: u64, shutdown: &Shutdown) -> media_pp::Result<()> {
        media_pp::init()?;
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let gpu =
            D3d11GpuContext::new(None).map_err(|e| media_pp::Error::Other(format!("{e:?}")))?;

        let output_width = 640;
        let output_height = 360;
        let frame_rate = ffmpeg::Rational::new(30, 1);
        let (compositor, compositor_handle) = D3d11VideoCompositor::new(
            "compositor",
            gpu.device(),
            gpu.context(),
            VideoCompositorOptions {
                width: output_width,
                height: output_height,
                frame_rate,
                background: Color::new(24, 24, 24),
            },
        )
        .map_err(|e| media_pp::Error::Other(e.to_string()))?;
        let time_base = compositor.time_base();

        let mut background_layer =
            VideoLayer::new(VideoRect::new(0, 0, output_width, output_height));
        background_layer.fit = VideoFit::Cover;
        let background_input = compositor_handle
            .add_source("background", background_layer)
            .map_err(|e| media_pp::Error::Other(e.to_string()))?
            .expect("compositor is alive");
        let background_sink = background_input.sink;

        let foreground_width = 192;
        let foreground_height = 144;
        let mut foreground_layer = VideoLayer::new(VideoRect::new(
            0,
            output_height as i32 - foreground_height as i32,
            foreground_width,
            foreground_height,
        ));
        foreground_layer.z_index = 1;
        foreground_layer.opacity = 0.85;
        foreground_layer.fit = VideoFit::Cover;
        let foreground_input = compositor_handle
            .add_source("foreground", foreground_layer)
            .map_err(|e| media_pp::Error::Other(e.to_string()))?
            .expect("compositor is alive");
        let foreground_sink = foreground_input.sink;
        let foreground_handle = foreground_input.layer;

        // TestVideoSource -> SwScaler(NV12) -> D3d11Upload: the CPU->GPU bridge
        // every compositor input needs (see D3d11VideoCompositor's own docs —
        // it only accepts already-GPU-resident Pixel::D3D11 frames).
        let background_source = TestVideoSource::new(
            "background-source",
            TestVideoOptions {
                width: output_width,
                height: output_height,
                framerate: frame_rate,
            },
        );
        let background_pipeline =
            Pipeline::new("background-input", background_source, |source, ctx| {
                let scaler = SwScaler::new(
                    "to-nv12",
                    ffmpeg::format::Pixel::NV12,
                    output_width,
                    output_height,
                    ffmpeg::software::scaling::Flags::BILINEAR,
                );
                let upload = D3d11Upload::new("upload", gpu.device(), output_width, output_height);
                let branch = ctx.branch().pipe(scaler).pipe(upload).to(background_sink)?;
                ctx.attach(source, 0, branch)?;
                Ok(())
            })?;

        // A different size and frame rate proves compositor inputs are
        // independent live pipelines, same as the CPU `video_compositor`
        // example — each sink retains only its latest frame and the
        // compositor emits on its own 30fps clock.
        let foreground_source = TestVideoSource::new(
            "foreground-source",
            TestVideoOptions {
                width: 320,
                height: 240,
                framerate: ffmpeg::Rational::new(15, 1),
            },
        );
        let foreground_pipeline =
            Pipeline::new("foreground-input", foreground_source, |source, ctx| {
                let scaler = SwScaler::new(
                    "to-nv12",
                    ffmpeg::format::Pixel::NV12,
                    320,
                    240,
                    ffmpeg::software::scaling::Flags::BILINEAR,
                );
                let upload = D3d11Upload::new("upload", gpu.device(), 320, 240);
                let branch = ctx.branch().pipe(scaler).pipe(upload).to(foreground_sink)?;
                ctx.attach(source, 0, branch)?;
                Ok(())
            })?;

        let encoder = SwEncoder::new(
            "encoder",
            SwEncoderOptions {
                codec: VideoCodec::OpenH264,
                width: output_width,
                height: output_height,
                time_base,
                frame_rate,
                bit_rate: 2_000_000,
                gop_size: 60,
            },
        )?;
        let mut muxer = Mp4Muxer::create(path)?;
        muxer.add_stream("video", encoder.parameters(), time_base)?;
        let muxer_sink = muxer.open()?.pop().expect("one video stream");

        let output_pipeline = Pipeline::new("composited-output", compositor, |source, ctx| {
            let renderer = render_common::d3d11_window_renderer(
                "renderer",
                &gpu,
                hwnd,
                output_width,
                output_height,
            )
            .expect("failed to create renderer");
            let render_branch = ctx.branch().queue("render", 4).to(Box::new(renderer))?;

            let download = D3d11Download::new(
                "download",
                gpu.device(),
                gpu.context(),
                output_width,
                output_height,
            )
            .map_err(|e| media_pp::Error::Other(e.to_string()))?;
            let to_yuv = SwScaler::new(
                "to-yuv",
                ffmpeg::format::Pixel::YUV420P,
                output_width,
                output_height,
                ffmpeg::software::scaling::Flags::BILINEAR,
            );
            let record_branch = ctx
                .branch()
                .queue("record", 4)
                .pipe(download)
                .pipe(to_yuv)
                .queue("encode-frames", 8)
                .pipe(encoder)
                .to(muxer_sink)?;

            let tee_branch = TeeBuilder::new("tee", ctx.clone())
                .branch(render_branch)
                .branch(record_branch)
                .build()?;
            ctx.attach(source, 0, tee_branch)?;
            Ok(())
        })?;

        // Published before `run`, so a close that arrives from here on finds
        // the pipelines to stop. `true` means one already did, and nothing
        // has presented yet.
        if shutdown.publish(&[
            background_pipeline.clone(),
            foreground_pipeline.clone(),
            output_pipeline.clone(),
        ]) {
            return Ok(());
        }

        output_pipeline.run();
        background_pipeline.run();
        foreground_pipeline.run();

        let steps = seconds.saturating_mul(30);
        let travel = output_width - foreground_width;
        for step in 0..steps {
            // Closing the window ends the animation early rather than leaving
            // it to run out the fixed duration with nothing to present to.
            if shutdown.requested() {
                break;
            }
            let x = if steps <= 1 {
                0
            } else {
                (u64::from(travel) * step / (steps - 1)) as i32
            };
            foreground_handle
                .set_rect(VideoRect::new(
                    x,
                    output_height as i32 - foreground_height as i32,
                    foreground_width,
                    foreground_height,
                ))
                .map_err(|e| media_pp::Error::Other(e.to_string()))?;
            thread::sleep(Duration::from_millis(33));
        }

        background_pipeline.stop();
        foreground_pipeline.stop();
        output_pipeline.stop();

        for pipeline in [&background_pipeline, &foreground_pipeline, &output_pipeline] {
            for event in pipeline.bus().iter() {
                if let BusEvent::Error { name, error, .. } = event {
                    eprintln!("[{name}] error: {error}");
                }
            }
        }

        println!("wrote {path}");
        Ok(())
    }
}

/// The Linux half of the same example. Deliberately the same graph, the same
/// layer settings, and the same CLI as `windows_example` — two upload-fed
/// inputs, one compositor, a Tee to a window renderer and to a
/// download-and-encode recording branch — so only the GPU stack differs.
#[cfg(target_os = "linux")]
mod linux_example {
    use std::{thread, time::Duration};

    use ffmpeg_next as ffmpeg;
    use media_pp::{
        bus::BusEvent,
        color::Color,
        elements::{
            CudaDevice, CudaDownload, CudaFrameFormat, CudaUpload, CudaVideoCompositor, Mp4Muxer,
            SwEncoder, SwEncoderOptions, SwScaler, TeeBuilder, TestVideoOptions, TestVideoSource,
            VideoCodec, VideoCompositorOptions, VideoFit, VideoLayer, VideoRect,
        },
        pipeline::Pipeline,
    };
    use render_common::{Shutdown, VulkanGpuContext, WindowTarget};

    pub(super) fn run() {
        let path = std::env::args()
            .nth(1)
            .unwrap_or_else(|| "gpu_video_compositor.mp4".into());
        let seconds: u64 = std::env::args()
            .nth(2)
            .and_then(|value| value.parse().ok())
            .unwrap_or(5);

        render_common::run_window(
            "media-pp gpu_video_compositor",
            640,
            360,
            move |target, shutdown| play(target, &path, seconds, &shutdown),
        );
    }

    fn play(
        target: WindowTarget,
        path: &str,
        seconds: u64,
        shutdown: &Shutdown,
    ) -> media_pp::Result<()> {
        media_pp::init()?;
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        // One CUDA context for the whole stack: every input uploads onto it,
        // the compositor draws on it, and the renderer imports its Vulkan
        // memory into it. Each element rejects a frame from a different one.
        let cuda = CudaDevice::new().map_err(|e| media_pp::Error::Other(e.to_string()))?;
        let gpu = VulkanGpuContext::new(target.display).map_err(media_pp::Error::Other)?;

        let output_width = 640;
        let output_height = 360;
        let frame_rate = ffmpeg::Rational::new(30, 1);
        let (compositor, compositor_handle) = CudaVideoCompositor::new(
            "compositor",
            &cuda,
            VideoCompositorOptions {
                width: output_width,
                height: output_height,
                frame_rate,
                background: Color::new(24, 24, 24),
            },
        )
        .map_err(|e| media_pp::Error::Other(e.to_string()))?;
        let time_base = compositor.time_base();

        let mut background_layer =
            VideoLayer::new(VideoRect::new(0, 0, output_width, output_height));
        background_layer.fit = VideoFit::Cover;
        let background_input = compositor_handle
            .add_source("background", background_layer)
            .map_err(|e| media_pp::Error::Other(e.to_string()))?;
        let background_sink = background_input.sink;

        let foreground_width = 192;
        let foreground_height = 144;
        let mut foreground_layer = VideoLayer::new(VideoRect::new(
            0,
            output_height as i32 - foreground_height as i32,
            foreground_width,
            foreground_height,
        ));
        foreground_layer.z_index = 1;
        foreground_layer.opacity = 0.85;
        foreground_layer.fit = VideoFit::Cover;
        let foreground_input = compositor_handle
            .add_source("foreground", foreground_layer)
            .map_err(|e| media_pp::Error::Other(e.to_string()))?;
        let foreground_sink = foreground_input.sink;
        let foreground_handle = foreground_input.layer;

        // TestVideoSource -> SwScaler(NV12) -> CudaUpload: the CPU->GPU bridge
        // every compositor input needs (see CudaVideoCompositor's own docs —
        // it only accepts already-GPU-resident NV12 CUDA frames).
        let background_source = TestVideoSource::new(
            "background-source",
            TestVideoOptions {
                width: output_width,
                height: output_height,
                framerate: frame_rate,
            },
        );
        let background_pipeline =
            Pipeline::new("background-input", background_source, |source, ctx| {
                let scaler = SwScaler::new(
                    "to-nv12",
                    ffmpeg::format::Pixel::NV12,
                    output_width,
                    output_height,
                    ffmpeg::software::scaling::Flags::BILINEAR,
                );
                let upload = CudaUpload::new(
                    "upload",
                    &cuda,
                    CudaFrameFormat::Nv12,
                    output_width,
                    output_height,
                )
                .map_err(|e| media_pp::Error::Other(e.to_string()))?;
                let branch = ctx.branch().pipe(scaler).pipe(upload).to(background_sink)?;
                ctx.attach(source, 0, branch)?;
                Ok(())
            })?;

        // A different size and frame rate proves compositor inputs are
        // independent live pipelines, same as the CPU `video_compositor`
        // example — each sink retains only its latest frame and the
        // compositor emits on its own 30fps clock.
        let foreground_source = TestVideoSource::new(
            "foreground-source",
            TestVideoOptions {
                width: 320,
                height: 240,
                framerate: ffmpeg::Rational::new(15, 1),
            },
        );
        let foreground_pipeline =
            Pipeline::new("foreground-input", foreground_source, |source, ctx| {
                let scaler = SwScaler::new(
                    "to-nv12",
                    ffmpeg::format::Pixel::NV12,
                    320,
                    240,
                    ffmpeg::software::scaling::Flags::BILINEAR,
                );
                let upload = CudaUpload::new("upload", &cuda, CudaFrameFormat::Nv12, 320, 240)
                    .map_err(|e| media_pp::Error::Other(e.to_string()))?;
                let branch = ctx.branch().pipe(scaler).pipe(upload).to(foreground_sink)?;
                ctx.attach(source, 0, branch)?;
                Ok(())
            })?;

        let encoder = SwEncoder::new(
            "encoder",
            SwEncoderOptions {
                codec: VideoCodec::OpenH264,
                width: output_width,
                height: output_height,
                time_base,
                frame_rate,
                bit_rate: 2_000_000,
                gop_size: 60,
            },
        )?;
        let mut muxer = Mp4Muxer::create(path)?;
        muxer.add_stream("video", encoder.parameters(), time_base)?;
        let muxer_sink = muxer.open()?.pop().expect("one video stream");

        let output_pipeline = Pipeline::new("composited-output", compositor, |source, ctx| {
            let renderer = render_common::cuda_window_renderer(
                "renderer",
                &gpu,
                &cuda,
                target.display,
                target.window,
                output_width,
                output_height,
            )
            .expect("failed to create renderer");
            let render_branch = ctx.branch().queue("render", 4).to(Box::new(renderer))?;

            let download = CudaDownload::new(
                "download",
                &cuda,
                CudaFrameFormat::Nv12,
                output_width,
                output_height,
            );
            let to_yuv = SwScaler::new(
                "to-yuv",
                ffmpeg::format::Pixel::YUV420P,
                output_width,
                output_height,
                ffmpeg::software::scaling::Flags::BILINEAR,
            );
            let record_branch = ctx
                .branch()
                .queue("record", 4)
                .pipe(download)
                .pipe(to_yuv)
                .queue("encode-frames", 8)
                .pipe(encoder)
                .to(muxer_sink)?;

            let tee_branch = TeeBuilder::new("tee", ctx.clone())
                .branch(render_branch)
                .branch(record_branch)
                .build()?;
            ctx.attach(source, 0, tee_branch)?;
            Ok(())
        })?;

        // Published before `run`, so a close that arrives from here on finds
        // the pipelines to stop. `true` means one already did, and nothing
        // has presented yet.
        if shutdown.publish(&[
            background_pipeline.clone(),
            foreground_pipeline.clone(),
            output_pipeline.clone(),
        ]) {
            return Ok(());
        }

        output_pipeline.run();
        background_pipeline.run();
        foreground_pipeline.run();

        let steps = seconds.saturating_mul(30);
        let travel = output_width - foreground_width;
        for step in 0..steps {
            // Closing the window ends the animation early rather than leaving
            // it to run out the fixed duration with nothing to present to.
            if shutdown.requested() {
                break;
            }
            let x = if steps <= 1 {
                0
            } else {
                (u64::from(travel) * step / (steps - 1)) as i32
            };
            foreground_handle
                .set_rect(VideoRect::new(
                    x,
                    output_height as i32 - foreground_height as i32,
                    foreground_width,
                    foreground_height,
                ))
                .map_err(|e| media_pp::Error::Other(e.to_string()))?;
            thread::sleep(Duration::from_millis(33));
        }

        background_pipeline.stop();
        foreground_pipeline.stop();
        output_pipeline.stop();

        for pipeline in [&background_pipeline, &foreground_pipeline, &output_pipeline] {
            for event in pipeline.bus().iter() {
                if let BusEvent::Error { name, error, .. } = event {
                    eprintln!("[{name}] error: {error}");
                }
            }
        }

        println!("wrote {path}");
        Ok(())
    }
}
