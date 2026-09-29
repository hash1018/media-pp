//! TestVideoSource -> SwScaler -> [VideoToolboxUpload] -> MetalWindowRenderer:
//! a moving test pattern drawn into a `winit` window by the library's own
//! Metal renderer, from system memory or — with `--videotoolbox` — from
//! VideoToolbox pixel buffers. `--file PATH` plays a file instead, decoded on
//! the CPU and paced, which is how to see colour: the test pattern is grey.
//! `--format` picks the layout the frames reach the renderer in: `nv12` (the
//! default), `yuv420p` or `bgra`; VideoToolbox frames are NV12 or BGRA only.
//!
//! The window, and the event loop on the main thread that AppKit asks for,
//! are this program's: it hands the renderer an `Arc` of the window, and the
//! renderer gives the window's view a Metal layer of its own and follows its
//! size. So nothing here runs inside `run_with_windows` — that is for a
//! program with no event loop of its own. `--seconds N` closes the window
//! after N seconds, resizing it once halfway through, for a run nobody
//! watches.
//!
//!     cargo run -p metal_window_render -- [--videotoolbox] [--format F] [--seconds N] [--file PATH]

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("{} example only supports macOS", env!("CARGO_PKG_NAME"));
}

#[cfg(target_os = "macos")]
fn main() -> media_pp::Result<()> {
    macos_example::run()
}

#[cfg(target_os = "macos")]
mod macos_example {
    use std::{
        sync::{Arc, mpsc},
        thread,
        time::Duration,
    };

    use media_pp::{
        bus::BusEvent,
        elements::{
            FileDemuxer, MetalWindowRenderer, Pacer, SwDecoder, SwScaler, TestVideoOptions,
            TestVideoSource, VideoToolboxDevice, VideoToolboxUpload,
        },
        ffmpeg::{self, media},
        pipeline::Pipeline,
    };
    use winit::{
        application::ApplicationHandler,
        dpi::LogicalSize,
        event::WindowEvent,
        event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy},
        window::{Window, WindowId},
    };

    const WIDTH: u32 = 1280;
    const HEIGHT: u32 = 720;

    enum Wake {
        /// Halfway through a timed run: make the window another shape.
        Resize,
        /// The pipeline has stopped and let go of the window.
        Done,
    }

    struct Options {
        videotoolbox: bool,
        format: ffmpeg::format::Pixel,
        seconds: Option<u64>,
        file: Option<String>,
    }

    const USAGE: &str = "usage: metal_window_render [--videotoolbox] [--format nv12|yuv420p|bgra] [--seconds N] [--file PATH]";

    pub(super) fn run() -> media_pp::Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;
        let mut options = Options {
            videotoolbox: false,
            format: ffmpeg::format::Pixel::NV12,
            seconds: None,
            file: None,
        };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--videotoolbox" => options.videotoolbox = true,
                "--seconds" => options.seconds = args.next().and_then(|s| s.parse().ok()),
                "--file" => options.file = args.next(),
                "--format" => {
                    options.format = match args.next().as_deref() {
                        Some("nv12") => ffmpeg::format::Pixel::NV12,
                        Some("yuv420p") => ffmpeg::format::Pixel::YUV420P,
                        Some("bgra") => ffmpeg::format::Pixel::BGRA,
                        _ => {
                            eprintln!("{USAGE}");
                            std::process::exit(1);
                        }
                    }
                }
                _ => {
                    eprintln!("{USAGE}");
                    std::process::exit(1);
                }
            }
        }
        if options.videotoolbox && options.format == ffmpeg::format::Pixel::YUV420P {
            eprintln!("VideoToolbox frames are NV12 or BGRA; {USAGE}");
            std::process::exit(1);
        }
        let device = if options.videotoolbox {
            Some(VideoToolboxDevice::new()?)
        } else {
            None
        };

        let event_loop = EventLoop::<Wake>::with_user_event()
            .build()
            .expect("an event loop");
        let mut app = App {
            options,
            device,
            proxy: event_loop.create_proxy(),
            window: None,
            stop: None,
            failed: None,
        };
        event_loop.run_app(&mut app).expect("the event loop ran");
        match app.failed {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    struct App {
        options: Options,
        device: Option<VideoToolboxDevice>,
        proxy: EventLoopProxy<Wake>,
        window: Option<Arc<Window>>,
        /// Asks the worker to stop the pipeline. Stopping is not done here,
        /// on the main thread, where the event loop has to go on turning.
        stop: Option<mpsc::Sender<()>>,
        failed: Option<media_pp::Error>,
    }

    impl App {
        fn start(&mut self, window: &Arc<Window>) -> media_pp::Result<()> {
            let renderer = MetalWindowRenderer::for_window("screen", window.clone())?;
            let format = self.options.format;
            let upload = self
                .device
                .as_ref()
                .map(|device| VideoToolboxUpload::new("upload", device));
            // At the source's own size: the renderer scales, and the
            // letterboxing is part of what this shows.
            let convert = |width, height| {
                SwScaler::new(
                    "convert",
                    format,
                    width,
                    height,
                    ffmpeg::software::scaling::Flags::BILINEAR,
                )
            };
            let pipeline = match &self.options.file {
                None => {
                    let source = TestVideoSource::new(
                        "test-video",
                        TestVideoOptions {
                            width: WIDTH,
                            height: HEIGHT,
                            ..TestVideoOptions::default()
                        },
                    );
                    Pipeline::new("metal-window-render", source, |source, ctx| {
                        // The pattern is YUV420P already, which the renderer
                        // draws as it comes: no conversion in between.
                        let branch = ctx.branch().queue("frames", 4);
                        let branch = if format == ffmpeg::format::Pixel::YUV420P {
                            branch
                        } else {
                            branch.pipe(convert(WIDTH, HEIGHT))
                        };
                        let branch = match upload {
                            Some(upload) => branch.pipe(upload),
                            None => branch,
                        };
                        ctx.attach(source, 0, branch.to(renderer)?)?;
                        Ok(())
                    })?
                    .0
                }
                Some(path) => {
                    let (source, _) = FileDemuxer::open("demux", path)?;
                    let video = source.best(media::Type::Video)?;
                    let params = video.parameters.clone();
                    // SAFETY: `params` is the stream's own codec parameters,
                    // live for as long as `params` is.
                    let (width, height) = unsafe {
                        let raw = params.as_ptr();
                        ((*raw).width as u32, (*raw).height as u32)
                    };
                    Pipeline::new("metal-window-render", source, |source, ctx| {
                        let branch = ctx
                            .branch()
                            .pipe(SwDecoder::new("decoder", params)?)
                            .queue("frames", 16)
                            .pipe(Pacer::new("pacer"))
                            .pipe(convert(width, height));
                        let branch = match upload {
                            Some(upload) => branch.pipe(upload),
                            None => branch,
                        };
                        ctx.attach(source, video.index, branch.to(renderer)?)?;
                        Ok(())
                    })?
                    .0
                }
            };
            pipeline.run()?;

            let (stop, stopped) = mpsc::channel();
            self.stop = Some(stop.clone());
            if let Some(seconds) = self.options.seconds {
                let proxy = self.proxy.clone();
                thread::spawn(move || {
                    thread::sleep(Duration::from_secs(seconds) / 2);
                    let _ = proxy.send_event(Wake::Resize);
                    thread::sleep(Duration::from_secs(seconds) / 2);
                    let _ = stop.send(());
                });
            }
            let proxy = self.proxy.clone();
            thread::spawn(move || {
                // Either the window asks, or the pipeline ends on its own.
                loop {
                    if stopped.recv_timeout(Duration::from_millis(20)).is_ok() {
                        break;
                    }
                    match pipeline.bus().try_recv() {
                        Some(event @ (BusEvent::Error { .. } | BusEvent::Finished)) => {
                            println!("{event}");
                            break;
                        }
                        Some(event) => println!("{event}"),
                        None => {}
                    }
                }
                pipeline.stop();
                drop(pipeline);
                let _ = proxy.send_event(Wake::Done);
            });
            Ok(())
        }
    }

    impl ApplicationHandler<Wake> for App {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            if self.window.is_some() {
                return;
            }
            let attributes = Window::default_attributes()
                .with_title("media-pp metal_window_render")
                .with_inner_size(LogicalSize::new(WIDTH / 2, HEIGHT / 2));
            let window = Arc::new(event_loop.create_window(attributes).expect("a window"));
            if let Err(error) = self.start(&window) {
                eprintln!("could not start: {error}");
                self.failed = Some(error);
                event_loop.exit();
                return;
            }
            self.window = Some(window);
        }

        fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
            match event {
                WindowEvent::Resized(size) => {
                    // Followed by the renderer on its own: it reads the
                    // layer's size before every frame.
                    println!("window is {}x{}", size.width, size.height);
                }
                WindowEvent::CloseRequested => {
                    if let Some(stop) = &self.stop {
                        let _ = stop.send(());
                    }
                }
                _ => {}
            }
        }

        fn user_event(&mut self, event_loop: &ActiveEventLoop, event: Wake) {
            match event {
                Wake::Resize => {
                    if let Some(window) = &self.window {
                        let _ = window.request_inner_size(LogicalSize::new(400, 400));
                    }
                }
                Wake::Done => event_loop.exit(),
            }
        }
    }
}
