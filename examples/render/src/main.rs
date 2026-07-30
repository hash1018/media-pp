use std::{sync::Arc, thread};

use ffmpeg_next::media;
use media_pp::{
    Error,
    bus::BusEvent,
    clock::Clock,
    element::Source,
    elements::{Decoder, Dx12Renderer, FileDemuxer, Pacer},
    pipeline::{ChainBuilder, Pipeline},
};
use renderer_engine::engine::RendererEngine;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::{Window, WindowId},
};

/// Demux -> Decoder -> Queue -> Pacer -> Renderer: decodes a video file
/// and presents it in a native window at real playback speed, via
/// `renderer_engine`'s DX12 `WindowRenderer`.
///
///     cargo run -p render -- path/to/video.mp4
fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "test-video/h265.mp4".into());

    let event_loop = EventLoop::new().expect("failed to create event loop");
    let mut app = App {
        path,
        window: None,
        // Kept alive for the app's duration so the window doesn't outlive
        // the thread rendering into it; not otherwise joined — closing
        // the window just ends the process.
        _playback: None,
    };
    event_loop.run_app(&mut app).expect("event loop failed");
}

struct App {
    path: String,
    window: Option<Window>,
    _playback: Option<thread::JoinHandle<()>>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("media-pp render")
                    .with_inner_size(LogicalSize::new(1280, 720))
                    // Pacer/Renderer are wired up once, sized to the
                    // window's initial size — no resize handling here.
                    .with_resizable(false),
            )
            .expect("failed to create window");

        let hwnd = match window
            .window_handle()
            .expect("failed to get window handle")
            .as_raw()
        {
            RawWindowHandle::Win32(handle) => handle.hwnd.get(),
            _ => panic!("render example only supports Windows"),
        };
        let size = window.inner_size();

        let path = self.path.clone();
        self._playback = Some(thread::spawn(move || {
            if let Err(e) = play(&path, hwnd, size.width, size.height) {
                eprintln!("playback failed: {e}");
            }
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
}

fn play(path: &str, hwnd: isize, width: u32, height: u32) -> media_pp::Result<()> {
    media_pp::init()?;

    let (source, streams) = FileDemuxer::open("demux", path)?;
    let video = streams
        .iter()
        .find(|s| s.kind == media::Type::Video)
        .ok_or_else(|| Error::Other("no video stream in file".into()))?;
    let params = source
        .stream_parameters(video.index)
        .ok_or_else(|| Error::Other("stream disappeared".into()))?;
    let time_base = source
        .stream_time_base(video.index)
        .ok_or_else(|| Error::Other("stream disappeared".into()))?;

    let engine = RendererEngine::new().map_err(|e| Error::Other(format!("{e:?}")))?;
    let clock = Arc::new(Clock::new());

    let mut pipeline = Pipeline::new(source, |source, bus| {
        let decoder = Decoder::new("decoder", params).expect("failed to open decoder");
        let pacer = Pacer::new("pacer", time_base, clock);
        let renderer = Dx12Renderer::new("renderer", &engine, hwnd, width, height)
            .expect("failed to create renderer");
        let branch = ChainBuilder::new(bus.clone())
            .pipe(decoder) // same thread as the demux — cheap enough not to need a queue
            .queue("frames", 32) // pacer sleeps on its own thread; let decode run ahead into this
            .pipe(pacer)
            .build(Box::new(renderer));
        source.src_pads()[video.index].link(branch);
    });

    // Drain the bus even if `run()` failed — the *specific* element error
    // (e.g. an unsupported pixel format from `Renderer`) was already
    // posted there before the failure propagated up through `?`.
    let ran = pipeline.run();
    for event in pipeline.bus().iter() {
        match event {
            BusEvent::Error { element, message } => eprintln!("[{element}] error: {message}"),
            BusEvent::Eos { element } => println!("[{element}] eos"),
            BusEvent::Dropped { element } => eprintln!("[{element}] dropped a buffer (queue full)"),
        }
    }
    ran?;
    Ok(())
}
