use std::{num::NonZeroU32, rc::Rc, thread};

use ffmpeg_next::{format::Pixel, frame::Video, media, software::scaling::Flags};
use media_pp::{
    Error, Result,
    bus::BusEvent,
    element::Source,
    elements::{COCO_CLASS_LABELS, Detection, FileDemuxer, OrtDetector, Scaler, SwDecoder},
    pipeline::{ChainBuilder, Pipeline},
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
const DST_WIDTH: u32 = 640;
const DST_HEIGHT: u32 = 640;
const CONF_THRESHOLD: f32 = 0.5;
const IOU_THRESHOLD: f32 = 0.7;
const BOX_COLOR: u32 = 0x00ff00; // XRGB8888 — softbuffer's pixel format
const BOX_THICKNESS: usize = 2;

/// Demux -> SwDecoder -> Scaler (640x640 RGB24) -> OrtDetector, drawing every
/// detection straight onto the same 640x640 frame `OrtDetector` saw and
/// presenting it in a plain window. Deliberately not DX12: `Dx12Renderer`
/// has no hook for drawing an overlay, so this blits pixels straight into a
/// `winit` window via `softbuffer` instead — no GPU, no `renderer-engine`.
///
/// No `Pacer` in this pipeline, so frames show up as fast as decode +
/// inference allow, not at real playback speed.
///
///     cargo run -p detect -- path/to/model.onnx [path/to/video.mp4]
fn main() {
    let mut args = std::env::args().skip(1);
    let Some(model_path) = args.next() else {
        eprintln!("usage: detect <model.onnx> [video, default test-video/h265.mp4]");
        std::process::exit(1);
    };
    let video_path = args.next().unwrap_or_else(|| "test-video/h265.mp4".into());

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
    /// One rendered 640x640 XRGB8888 frame, ready to blit straight into the
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
                        // Fixed to the model's own input size — no resize handling below.
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
                let (Some(surface), Some(pixels)) = (&mut self.surface, &self.latest_frame) else {
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
    media_pp::init()?;

    let (source, streams) = FileDemuxer::open("demux", video_path)?;
    let video = streams
        .iter()
        .find(|s| s.kind == media::Type::Video)
        .ok_or_else(|| Error::Other("no video stream in file".into()))?;
    let params = source
        .stream_parameters(video.index)
        .ok_or_else(|| Error::Other("stream disappeared".into()))?;

    let pipeline = Pipeline::new("detect", source, |source, bus, _clock, id| {
        let decoder = SwDecoder::new("decoder", params).expect("failed to open decoder");
        let scaler = Scaler::new("scaler", DST_FORMAT, DST_WIDTH, DST_HEIGHT, Flags::BILINEAR);
        let render_proxy = proxy.clone();
        let detector = OrtDetector::new(
            "detector",
            model_path,
            CONF_THRESHOLD,
            IOU_THRESHOLD,
            move |frame, detections| {
                for detection in detections {
                    let label = COCO_CLASS_LABELS
                        .get(detection.class_id)
                        .copied()
                        .unwrap_or("unknown");
                    println!("{label} ({:.0}%)", detection.score * 100.0);
                }
                // A closed window (event loop already exited) just means
                // there's nothing left to draw into — not a pipeline error.
                let _ = render_proxy.send_event(AppEvent::Frame(render_frame(frame, detections)));
                Ok(())
            },
        )
        .expect("failed to load model");

        let branch = ChainBuilder::new(bus.clone(), id)
            .pipe(decoder) // same thread as the demux — cheap enough not to need a queue
            .queue("frames", 8) // scaler/detector run on their own thread
            .pipe(scaler)
            .build(Box::new(detector));
        source.src_pads()[video.index].link(branch);
    });

    pipeline.run();

    for event in pipeline.bus().iter() {
        match &event {
            BusEvent::Eos { name, .. } => println!("[{name}] eos"),
            BusEvent::Error { name, error, .. } => eprintln!("[{name}] error: {error}"),
            BusEvent::Dropped { name, .. } => eprintln!("[{name}] dropped a buffer (queue full)"),
            BusEvent::Seeked { .. } => {}
        }
        if matches!(event, BusEvent::Eos { .. } | BusEvent::Error { .. }) {
            pipeline.stop();
        }
    }

    Ok(())
}

/// Blits `frame`'s packed RGB24 bytes into a 640x640 XRGB8888 buffer
/// (softbuffer's own pixel format — skipping `stride`'s per-row padding,
/// same as `OrtDetector::detect`'s own tensor conversion), then draws every
/// detection's box directly on top — no rescaling needed, since `frame` is
/// the exact same pixel space `detections`' coordinates are already in.
fn render_frame(frame: &Video, detections: &[Detection]) -> Box<[u32]> {
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

    for detection in detections {
        draw_box(&mut pixels, width, height, detection);
    }

    pixels
}

fn draw_box(pixels: &mut [u32], width: usize, height: usize, detection: &Detection) {
    let x0 = detection.x.max(0.0) as usize;
    let y0 = detection.y.max(0.0) as usize;
    let x1 = ((detection.x + detection.width).max(0.0) as usize).min(width.saturating_sub(1));
    let y1 = ((detection.y + detection.height).max(0.0) as usize).min(height.saturating_sub(1));

    for x in x0..=x1 {
        for t in 0..BOX_THICKNESS {
            set_pixel(pixels, width, height, x, y0 + t);
            set_pixel(pixels, width, height, x, y1.saturating_sub(t));
        }
    }
    for y in y0..=y1 {
        for t in 0..BOX_THICKNESS {
            set_pixel(pixels, width, height, x0 + t, y);
            set_pixel(pixels, width, height, x1.saturating_sub(t), y);
        }
    }
}

fn set_pixel(pixels: &mut [u32], width: usize, height: usize, x: usize, y: usize) {
    if x < width && y < height {
        pixels[y * width + x] = BOX_COLOR;
    }
}
