//! FileDemuxer -> SwDecoder -> Queue -> SwOrtDetector -> SwScaler (960x540
//! RGB24) -> SwDetectionOverlay -> AppSink, drawing what the detector found
//! onto each picture and presenting it in a plain window. Everything is on
//! the CPU, blitted into a `winit` window through `softbuffer`; no GPU
//! renderer is involved.
//!
//! The detector hands each decoded picture on as it came, carrying the
//! `Detections` it found as metadata; the scaler after it carries them on,
//! and the overlay draws them onto the scaled picture — their boxes are
//! fractions of the picture, so they fit it at whatever size it is. Labels
//! are drawn where a system font is found.
//!
//! The model is an Ultralytics YOLO ONNX export — YOLOv8 and YOLO11, or
//! YOLOv10 and YOLO26. No `Pacer` in this pipeline, so frames show up as
//! fast as decode + inference allow, not at real playback speed.
//!
//!     cargo run -p detect -- path/to/model.onnx path/to/video.mp4

fn main() -> impl std::process::Termination {
    example::run()
}

mod example {
    use std::{num::NonZeroU32, rc::Rc, thread};

    use media_pp::ffmpeg::{format::Pixel, frame::Video, media, software::scaling::Flags};
    use media_pp::{
        Result,
        buffer::MediaBuffer,
        bus::BusEvent,
        elements::{
            AppSink, BoxStyle, COCO_CLASS_LABELS, DetectionOverlayOptions, Detections, FileDemuxer,
            LabelStyle, OrtDetectorOptions, SwDecoder, SwDetectionOverlay, SwOrtDetector, SwScaler,
            Treatment,
        },
        pipeline::Pipeline,
    };
    use softbuffer::{Context, Surface};
    use winit::{
        application::ApplicationHandler,
        dpi::LogicalSize,
        event::WindowEvent,
        event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy, OwnedDisplayHandle},
        window::{Window, WindowId},
    };

    const DST_FORMAT: Pixel = Pixel::RGB24;
    const DST_WIDTH: u32 = 960;
    const DST_HEIGHT: u32 = 540;
    const CONF_THRESHOLD: f32 = 0.5;
    const IOU_THRESHOLD: f32 = 0.7;
    /// Where the labels' font is looked for; without one, boxes alone.
    const FONTS: [&str; 5] = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ];

    pub(super) fn run() {
        let mut args = std::env::args().skip(1);
        let (Some(model_path), Some(video_path)) = (args.next(), args.next()) else {
            eprintln!("usage: detect <model.onnx> <video.mp4>");
            std::process::exit(1);
        };

        let event_loop = EventLoop::<AppEvent>::with_user_event()
            .build()
            .expect("failed to create event loop");
        let context = Context::new(event_loop.owned_display_handle())
            .expect("failed to create softbuffer context");
        let proxy = event_loop.create_proxy();
        let mut app = App {
            model_path,
            video_path,
            proxy,
            context,
            window: None,
            surface: None,
            latest_frame: None,
            // Kept alive for the app's duration so the playback thread doesn't
            // outlive the window it's sending frames to; not otherwise joined —
            // the window closes itself once playback finishes (see
            // `AppEvent::Done` below).
            _playback: None,
        };
        event_loop.run_app(&mut app).expect("event loop failed");
    }

    enum AppEvent {
        /// One rendered 960x540 XRGB8888 frame, ready to blit straight into the
        /// surface — see `render_frame`.
        Frame(Box<[u32]>),
        Done,
    }

    struct App {
        model_path: String,
        video_path: String,
        proxy: EventLoopProxy<AppEvent>,
        context: Context<OwnedDisplayHandle>,
        window: Option<Rc<Window>>,
        surface: Option<Surface<OwnedDisplayHandle, Rc<Window>>>,
        latest_frame: Option<Box<[u32]>>,
        _playback: Option<thread::JoinHandle<()>>,
    }

    impl ApplicationHandler<AppEvent> for App {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            if self.window.is_some() {
                return;
            }

            let window = Rc::new(
                event_loop
                    .create_window(
                        Window::default_attributes()
                            .with_title("media-pp detect")
                            .with_inner_size(LogicalSize::new(DST_WIDTH, DST_HEIGHT))
                            // Fixed to the scaler's size — no resize handling below.
                            .with_resizable(false),
                    )
                    .expect("failed to create window"),
            );
            let mut surface =
                Surface::new(&self.context, window.clone()).expect("failed to create surface");
            surface
                .resize(
                    NonZeroU32::new(DST_WIDTH).unwrap(),
                    NonZeroU32::new(DST_HEIGHT).unwrap(),
                )
                .expect("failed to size surface");
            self.surface = Some(surface);
            self.window = Some(window);

            let model_path = self.model_path.clone();
            let video_path = self.video_path.clone();
            let proxy = self.proxy.clone();
            self._playback = Some(thread::spawn(move || {
                if let Err(e) = play(&model_path, &video_path, proxy.clone()) {
                    eprintln!("detection failed: {e}");
                }
                let _ = proxy.send_event(AppEvent::Done);
            }));
        }

        fn window_event(
            &mut self,
            event_loop: &ActiveEventLoop,
            window_id: WindowId,
            event: WindowEvent,
        ) {
            if self.window.as_ref().map(|w| w.id()) != Some(window_id) {
                return;
            }
            match event {
                WindowEvent::CloseRequested => event_loop.exit(),
                WindowEvent::RedrawRequested => {
                    let (Some(surface), Some(pixels)) = (&mut self.surface, &self.latest_frame)
                    else {
                        return;
                    };
                    let mut buffer = surface.buffer_mut().expect("failed to get surface buffer");
                    buffer.copy_from_slice(pixels);
                    buffer.present().expect("failed to present frame");
                }
                _ => {}
            }
        }

        fn user_event(&mut self, event_loop: &ActiveEventLoop, event: AppEvent) {
            match event {
                AppEvent::Frame(pixels) => {
                    self.latest_frame = Some(pixels);
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
                AppEvent::Done => event_loop.exit(),
            }
        }
    }

    fn play(model_path: &str, video_path: &str, proxy: EventLoopProxy<AppEvent>) -> Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let (source, _) = FileDemuxer::open("demux", video_path)?;
        let video = source.best(media::Type::Video)?;
        let params = video.parameters.clone();

        let (pipeline, ()) = Pipeline::new("detect", source, |source, ctx| {
            let decoder = SwDecoder::new("decoder", params)?;
            let scaler =
                SwScaler::new("scaler", DST_FORMAT, DST_WIDTH, DST_HEIGHT, Flags::BILINEAR);
            let detector = SwOrtDetector::new(
                "detector",
                model_path,
                OrtDetectorOptions {
                    conf_threshold: CONF_THRESHOLD,
                    iou_threshold: IOU_THRESHOLD,
                    // The model's own class names, which an Ultralytics
                    // export carries.
                    labels: None,
                    ..OrtDetectorOptions::default()
                },
            )?;
            let overlay = SwDetectionOverlay::new("overlay", {
                let font = FONTS.iter().find_map(|path| std::fs::read(path).ok());
                DetectionOverlayOptions {
                    others: Treatment::boxes(BoxStyle {
                        label: font.is_some().then(LabelStyle::default),
                        ..BoxStyle::default()
                    }),
                    font,
                    ..DetectionOverlayOptions::default()
                }
            })?;
            let render_proxy = proxy.clone();
            let sink = AppSink::new("draw", move |buf| {
                let MediaBuffer::Video(frame) = &buf else {
                    return Ok(());
                };
                let Some(found) = buf
                    .metadata()
                    .and_then(|metadata| metadata.get::<Detections>())
                else {
                    return Ok(());
                };
                for detection in &found.items {
                    // A model that names no classes is taken for stock COCO
                    // weights, as most exports without names are.
                    let label = found
                        .label(detection)
                        .or_else(|| COCO_CLASS_LABELS.get(detection.class_id).copied())
                        .unwrap_or("unknown");
                    println!("{label} ({:.0}%)", detection.score * 100.0);
                }
                // A closed window (event loop already exited) just means
                // there's nothing left to draw into — not a pipeline error.
                let _ = render_proxy.send_event(AppEvent::Frame(render_frame(frame)));
                Ok(())
            });

            let branch = ctx
                .branch()
                .pipe(decoder) // same thread as the demux — cheap enough not to need a queue
                .queue("frames", 8) // detector/scaler run on their own thread
                .pipe(detector)
                .pipe(scaler)
                .pipe(overlay)
                .to(sink)?;
            ctx.attach(source, video.index, branch)?;
            Ok(())
        })?;

        pipeline.run()?;

        for event in pipeline.bus().iter() {
            println!("{event}");
            if matches!(event, BusEvent::Finished | BusEvent::Error { .. }) {
                pipeline.stop();
            }
        }

        Ok(())
    }

    /// Blits `frame`'s packed RGB24 bytes, boxes already drawn on them, into
    /// an XRGB8888 buffer — softbuffer's own pixel format — skipping
    /// `stride`'s per-row padding.
    fn render_frame(frame: &Video) -> Box<[u32]> {
        let width = frame.width() as usize;
        let height = frame.height() as usize;
        let stride = frame.stride(0);
        let data = frame.data(0);

        let mut pixels = vec![0u32; width * height].into_boxed_slice();
        for y in 0..height {
            let row = &data[y * stride..y * stride + width * 3];
            for x in 0..width {
                let rgb = &row[x * 3..x * 3 + 3];
                pixels[y * width + x] =
                    ((rgb[0] as u32) << 16) | ((rgb[1] as u32) << 8) | rgb[2] as u32;
            }
        }
        pixels
    }
}
