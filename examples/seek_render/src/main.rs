use std::{
    io::{self, BufRead},
    thread,
    time::Duration,
};

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

/// Demux -> SwDecoder -> Queue -> Pacer -> Renderer, same chain as
/// `sw_decode_render`, plus a terminal prompt that reads timestamps and
/// calls `Pipeline::seek` with them while the window is open — proves
/// `seek` actually changes what's on screen, not just that it compiles.
///
///     cargo run -p seek_render -- path/to/video.mp4
///     (then in the same terminal: type `30` or `1:15` + Enter to jump
///      there, or `q` + Enter to stop early)
fn main() {
    // `renderer-engine` logs internal render-thread failures via `log`
    // (e.g. a DX12 present error) that would otherwise be silently
    // dropped — without a logger installed, `log::error!` calls are
    // no-ops. Set `RUST_LOG=debug` (or `error`/`warn`) to see them.
    env_logger::init();

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
                    .with_title("media-pp seek_render")
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
            _ => panic!("seek_render example only supports Windows"),
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

    let pipeline = Pipeline::new("seek-render", source, |source, ctx| {
        let decoder = SwDecoder::new("decoder", params).expect("failed to open decoder");
        let pacer = Pacer::new("pacer", time_base, ctx.clock.clone());
        let renderer = Dx12Renderer::new("renderer", &engine, hwnd, width, height)
            .expect("failed to create renderer");
        let branch = ChainBuilder::new(ctx.clone())
            .pipe(decoder) // same thread as the demux — cheap enough not to need a queue
            .queue("frames", 32) // pacer sleeps on its own thread; let decode run ahead into this
            .pipe(pacer)
            .build(Box::new(renderer));
        source.src_pads()[video.index].link(branch);
    });

    // `run()` starts playback on a background thread and returns right
    // away — that's what makes this terminal prompt possible on the same
    // thread that would otherwise just be blocked waiting for it.
    pipeline.run();

    // Reads seek requests for as long as the process lives, on its own
    // thread — a blocked stdin read can't also notice natural playback
    // completion, so it doesn't try to; the whole process (this thread
    // included) exits once the window closes below.
    {
        let pipeline = pipeline.clone();
        thread::spawn(move || read_seek_commands(&pipeline));
    }

    // Same output `log_events()` would print, but also calls `stop()` on
    // `Eos`/`Error` — errors no longer end the pipeline on their own (see
    // `BusEvent`'s docs), so without this an error here (e.g. the
    // renderer's GPU upload ring running out of slots) would just get
    // printed forever instead of ending playback. `Eos` calling `stop()`
    // too is a harmless no-op in this example (single video stream, one
    // `Eos` means everything's already finished) — a multi-stream
    // pipeline would need to wait for every branch's `Eos`, not stop on
    // the first one.
    for event in pipeline.bus().iter() {
        match &event {
            BusEvent::Eos { name, .. } => println!("[{name}] eos"),
            BusEvent::Error { name, error, .. } => eprintln!("[{name}] error: {error}"),
            BusEvent::Dropped { name, .. } => {
                eprintln!("[{name}] dropped a buffer (queue full)")
            }
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

fn read_seek_commands(pipeline: &Pipeline) {
    println!("type a time in seconds (e.g. `30`) or mm:ss (e.g. `1:15`) + Enter to seek there");
    println!("(or `q` + Enter to stop early)");
    for line in io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line.eq_ignore_ascii_case("q") {
            pipeline.stop();
            break;
        }
        match parse_timestamp(line) {
            Some(target) => {
                println!("seeking to {target:.2?}...");
                pipeline.seek(target);
            }
            None => eprintln!("couldn't parse {line:?} — use seconds (`30`) or mm:ss (`1:15`)"),
        }
    }
}

/// `"90"` (plain seconds) or `"1:30"` (mm:ss) -> `Duration`. Fractional
/// seconds work in both forms (`"1.5"`, `"1:01.5"`).
fn parse_timestamp(s: &str) -> Option<Duration> {
    let secs = match s.split_once(':') {
        Some((min, sec)) => min.parse::<f64>().ok()? * 60.0 + sec.parse::<f64>().ok()?,
        None => s.parse::<f64>().ok()?,
    };
    if secs.is_finite() && secs >= 0.0 {
        Some(Duration::from_secs_f64(secs))
    } else {
        None
    }
}
