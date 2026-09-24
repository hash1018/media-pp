//! TestVideoSource -> SwScaler -> [CudaUpload] -> VulkanWindowRenderer: a
//! moving test pattern drawn into a `winit` window by the library's own
//! Vulkan renderer, from system memory or — with `--cuda` — from CUDA.
//! `--file PATH` plays a file instead, decoded on the CPU and paced, which is
//! how to see colour: the test pattern is grey. `--format` picks the layout
//! the frames reach the renderer in: `nv12` (the default), `yuv420p` or
//! `bgra`; CUDA frames are NV12 or BGRA only.
//!
//! The window is this program's: it runs the event loop and hands the
//! renderer an `Arc` of the window. Resizing it is followed — on X11 by the
//! renderer itself, on Wayland through the `WindowSize` this program sets
//! from its own `Resized` events. `--seconds N` closes it after N seconds,
//! resizing it once halfway through, for a run nobody watches.
//!
//!     cargo run -p vulkan_window_render -- [--cuda] [--format F] [--seconds N] [--file PATH]

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("{} example only supports Linux", env!("CARGO_PKG_NAME"));
}

#[cfg(target_os = "linux")]
fn main() -> media_pp::Result<()> {
    linux_example::run()
}

#[cfg(target_os = "linux")]
mod linux_example {
    use std::{
        sync::{Arc, mpsc},
        thread,
        time::Duration,
    };

    use media_pp::{
        bus::BusEvent,
        elements::{
            CudaDevice, CudaFrameFormat, CudaUpload, FileDemuxer, Pacer, SwDecoder, SwScaler,
            TestVideoOptions, TestVideoSource, VulkanGpu, VulkanWindowRenderer, WindowSize,
        },
        ffmpeg::{self, media},
        pipeline::Pipeline,
    };
    use winit::{
        application::ApplicationHandler,
        dpi::PhysicalSize,
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
        cuda: bool,
        format: ffmpeg::format::Pixel,
        seconds: Option<u64>,
        file: Option<String>,
    }

    const USAGE: &str = "usage: vulkan_window_render [--cuda] [--format nv12|yuv420p|bgra] [--seconds N] [--file PATH]";

    pub(super) fn run() -> media_pp::Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Debug,
            7,
        )?;
        let mut options = Options {
            cuda: false,
            format: ffmpeg::format::Pixel::NV12,
            seconds: None,
            file: None,
        };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--cuda" => options.cuda = true,
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
        // What `CudaUpload` makes of the frames, where they go to CUDA.
        let upload = match options.format {
            ffmpeg::format::Pixel::NV12 => Some(CudaFrameFormat::Nv12),
            ffmpeg::format::Pixel::BGRA => Some(CudaFrameFormat::Bgra),
            _ => None,
        };
        if options.cuda && upload.is_none() {
            eprintln!("CUDA frames are NV12 or BGRA; {USAGE}");
            std::process::exit(1);
        }

        // The CUDA side first, and the Vulkan device paired with it: made
        // once, before anything decodes.
        let cuda = if options.cuda {
            Some(Arc::new(CudaDevice::new()?))
        } else {
            None
        };
        let gpu = match &cuda {
            Some(cuda) => VulkanGpu::for_cuda(cuda)?,
            None => VulkanGpu::new()?,
        };
        println!("drawing on {}", gpu.name());

        let event_loop = EventLoop::<Wake>::with_user_event()
            .build()
            .expect("an event loop");
        let mut app = App {
            options,
            gpu,
            cuda,
            proxy: event_loop.create_proxy(),
            window: None,
            size: None,
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
        gpu: VulkanGpu,
        cuda: Option<Arc<CudaDevice>>,
        proxy: EventLoopProxy<Wake>,
        window: Option<Arc<Window>>,
        size: Option<WindowSize>,
        /// Asks the worker to stop the pipeline. Stopping is not done here:
        /// it waits for the renderer, which may be waiting for this loop to
        /// release a swapchain image.
        stop: Option<mpsc::Sender<()>>,
        failed: Option<media_pp::Error>,
    }

    impl App {
        fn start(&mut self, window: &Arc<Window>) -> media_pp::Result<()> {
            let renderer = VulkanWindowRenderer::for_window("screen", &self.gpu, window.clone())?;
            self.size = Some(renderer.window_size());
            let format = self.options.format;
            let upload = match &self.cuda {
                Some(cuda) => {
                    let into = if format == ffmpeg::format::Pixel::BGRA {
                        CudaFrameFormat::Bgra
                    } else {
                        CudaFrameFormat::Nv12
                    };
                    Some(CudaUpload::new("upload", cuda, into))
                }
                None => None,
            };
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
                    Pipeline::new("vulkan-window-render", source, |source, ctx| {
                        let branch = ctx.branch().queue("frames", 4).pipe(convert(WIDTH, HEIGHT));
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
                    Pipeline::new("vulkan-window-render", source, |source, ctx| {
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
                .with_title("media-pp vulkan_window_render")
                .with_inner_size(PhysicalSize::new(WIDTH, HEIGHT));
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
                    println!("window is {}x{}", size.width, size.height);
                    if let Some(window_size) = &self.size {
                        window_size.set(size.width, size.height);
                    }
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
                    // Where the size is the client's own to set — Wayland —
                    // winit applies it at once and says so here, rather than
                    // with a `Resized`.
                    if let Some(window) = &self.window
                        && let Some(size) = window.request_inner_size(PhysicalSize::new(800, 800))
                    {
                        println!("window is now {}x{}", size.width, size.height);
                        if let Some(window_size) = &self.size {
                            window_size.set(size.width, size.height);
                        }
                    }
                }
                Wake::Done => event_loop.exit(),
            }
        }
    }
}
