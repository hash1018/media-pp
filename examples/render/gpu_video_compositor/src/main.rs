use std::{thread, time::Duration};

use ffmpeg_next as ffmpeg;
use media_pp::{
    bus::BusEvent,
    elements::{
        D3d11Download, D3d11Upload, D3d11VideoCompositor, Mp4Muxer, Scaler, SwEncoder,
        SwEncoderOptions, TeeBuilder, TestVideoOptions, TestVideoSource, VideoCodec, VideoColor,
        VideoCompositorOptions, VideoFit, VideoLayer, VideoRect,
    },
    pipeline::Pipeline,
};
use render_common::D3d11GpuContext;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy},
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::{Window, WindowId},
};

/// Two TestVideoSource pipelines -> D3d11Upload -> D3d11VideoCompositor
/// (GPU shader compositing, no CPU round trip for the inputs or the
/// composited output) -> Tee -> {D3d11Renderer for live display,
/// D3d11Download -> Scaler -> SwEncoder -> Mp4Muxer for simultaneous
/// recording}. The foreground layer moves at runtime through its
/// D3d11VideoLayerHandle, same as the CPU `video_compositor` example, but
/// every frame this composites never touches the CPU until the recording
/// branch's own D3d11Download.
///
///     cargo run -p gpu_video_compositor -- [output.mp4] [seconds]
fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "gpu_video_compositor.mp4".into());
    let seconds: u64 = std::env::args()
        .nth(2)
        .and_then(|value| value.parse().ok())
        .unwrap_or(5);

    let event_loop = EventLoop::<PlaybackDone>::with_user_event()
        .build()
        .expect("failed to create event loop");
    let proxy = event_loop.create_proxy();
    let mut app = App {
        proxy,
        window: None,
        path,
        seconds,
        _playback: None,
    };
    event_loop.run_app(&mut app).expect("event loop failed");
}

struct PlaybackDone;

struct App {
    proxy: EventLoopProxy<PlaybackDone>,
    window: Option<Window>,
    path: String,
    seconds: u64,
    _playback: Option<thread::JoinHandle<()>>,
}

impl ApplicationHandler<PlaybackDone> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("media-pp gpu_video_compositor")
                    .with_inner_size(LogicalSize::new(640, 360))
                    .with_resizable(false),
            )
            .expect("failed to create window");

        let hwnd = match window
            .window_handle()
            .expect("failed to get window handle")
            .as_raw()
        {
            RawWindowHandle::Win32(handle) => handle.hwnd.get(),
            _ => panic!("gpu_video_compositor example only supports Windows"),
        };

        let proxy = self.proxy.clone();
        let path = self.path.clone();
        let seconds = self.seconds;
        self._playback = Some(thread::spawn(move || {
            if let Err(e) = play(hwnd, &path, seconds) {
                eprintln!("playback failed: {e}");
            }
            let _ = proxy.send_event(PlaybackDone);
        }));
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if self.window.as_ref().map(Window::id) != Some(window_id) {
            return;
        }
        if let WindowEvent::CloseRequested = event {
            event_loop.exit();
        }
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, _event: PlaybackDone) {
        event_loop.exit();
    }
}

fn play(hwnd: isize, path: &str, seconds: u64) -> media_pp::Result<()> {
    media_pp::init()?;

    let gpu = D3d11GpuContext::new(None).map_err(|e| media_pp::Error::Other(format!("{e:?}")))?;

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
            background: VideoColor::new(24, 24, 24),
        },
    )
    .map_err(|e| media_pp::Error::Other(e.to_string()))?;
    let time_base = compositor.time_base();

    let mut background_layer = VideoLayer::new(VideoRect::new(0, 0, output_width, output_height));
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

    // TestVideoSource -> Scaler(NV12) -> D3d11Upload: the CPU->GPU bridge
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
            let scaler = Scaler::new(
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
            let scaler = Scaler::new(
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
        let to_yuv = Scaler::new(
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

    output_pipeline.run();
    background_pipeline.run();
    foreground_pipeline.run();

    let steps = seconds.saturating_mul(30);
    let travel = output_width - foreground_width;
    for step in 0..steps {
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
