use std::thread;

use ffmpeg_next::media;
use media_pp::{
    Error,
    bus::BusEvent,
    element::Source,
    elements::{Dx12Renderer, FileDemuxer, Pacer, SwDecoder},
    pipeline::{ChainBuilder, Pipeline},
};
use renderer_engine::engine::RendererEngine;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy},
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::{Window, WindowId},
};

/// Demux -> SwDecoder -> Queue -> Pacer -> Renderer: decodes a video file
/// and presents it in a native window at real playback speed, via
/// `renderer_engine`'s DX12 `WindowRenderer`.
///
///     cargo run -p sw_decode_render -- path/to/video.mp4
fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "test-video/h265.mp4".into());

    let event_loop = EventLoop::<PlaybackDone>::with_user_event()
        .build()
        .expect("failed to create event loop");
    let proxy = event_loop.create_proxy();
    let mut app = App {
        path,
        proxy,
        window: None,
        // Kept alive for the app's duration so the window doesn't outlive
        // the thread rendering into it; not otherwise joined — the window
        // closes itself once playback finishes (see `user_event` below).
        _playback: None,
    };
    event_loop.run_app(&mut app).expect("event loop failed");
}

/// Sent from the playback thread once `play()` returns, so the window
/// closes itself when the pipeline finishes instead of sitting there
/// until someone closes it by hand.
struct PlaybackDone;

struct App {
    path: String,
    proxy: EventLoopProxy<PlaybackDone>,
    window: Option<Window>,
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
                    .with_title("media-pp sw_decode_render")
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
            _ => panic!("sw_decode_render example only supports Windows"),
        };
        let size = window.inner_size();

        let path = self.path.clone();
        let proxy = self.proxy.clone();
        self._playback = Some(thread::spawn(move || {
            if let Err(e) = play(&path, hwnd, size.width, size.height) {
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

    let pipeline = Pipeline::new("sw-decode-render", source, |source, bus, clock, id| {
        let decoder = SwDecoder::new("decoder", params).expect("failed to open decoder");
        let pacer = Pacer::new("pacer", time_base, clock.clone());
        let renderer = Dx12Renderer::new("renderer", &engine, hwnd, width, height)
            .expect("failed to create renderer");
        let branch = ChainBuilder::new(bus.clone(), id)
            .pipe(decoder) // same thread as the demux — cheap enough not to need a queue
            .queue("frames", 32) // pacer sleeps on its own thread; let decode run ahead into this
            .pipe(pacer)
            .build(Box::new(renderer));
        source.src_pads()[video.index].link(branch);
    });

    // `run()` starts playback on a background thread and returns right
    // away — any failure (e.g. an unsupported pixel format from
    // `Renderer`) shows up as a `BusEvent::Error` here instead of
    // through a returned `Result`.
    pipeline.run();

    // Errors no longer end the pipeline on their own (see `BusEvent`'s
    // docs) — watch for one here and `stop()`, or this window would just
    // sit open (showing a frozen last frame) instead of closing after a
    // renderer failure. Single video stream, so `Eos` calling `stop()` is
    // a harmless no-op too.
    for event in pipeline.bus().iter() {
        match &event {
            BusEvent::Eos { name, .. } => println!("[{name}] eos"),
            BusEvent::Error { name, error, .. } => eprintln!("[{name}] error: {error}"),
            BusEvent::Dropped { name, .. } => eprintln!("[{name}] dropped a buffer (queue full)"),
            BusEvent::Seeked {
                name,
                requested,
                landed,
                ..
            } => println!("[{name}] seeked: requested {requested:.2?}, landed {landed:.2?}"),
        }
        if matches!(event, BusEvent::Eos { .. } | BusEvent::Error { .. }) {
            pipeline.stop();
        }
    }
    Ok(())
}
