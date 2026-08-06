use std::thread;

use media_pp::{
    bus::BusEvent,
    element::Source,
    elements::{TestVideoOptions, TestVideoSource},
    pipeline::{ChainBuilder, Pipeline},
};
use render_common::D3d12GpuContext;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy},
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::{Window, WindowId},
};

/// TestVideoSource -> Renderer: a synthetic moving-gradient stream, no
/// file/camera/decoder involved at all, presented in a native window via
/// `render_common`'s own `D3d12WindowRenderer` (wrapped as a
/// `D3d12Renderer`) — proves `TestVideoSource`'s frames and
/// `D3d12Renderer`'s CPU-upload path work end to end without needing a
/// real video source.
///
/// No `Pacer` here, deliberately, as an experiment: `TestVideoSource`
/// self-paces with a drift-free absolute schedule (see its own docs) and
/// nothing sits between it and the renderer here (no `Scaler`, unlike
/// `screen_capture`). Testing confirmed that schedule is enough on its own
/// for a vsync-locked renderer to stay smooth without a separate pacing
/// stage; `screen_capture` reached the same result after its source moved
/// from variable-rate emission to the same absolute scheduling scheme.
///
///     cargo run -p test_video
fn main() {
    let event_loop = EventLoop::<PlaybackDone>::with_user_event()
        .build()
        .expect("failed to create event loop");
    let proxy = event_loop.create_proxy();
    let mut app = App {
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
                    .with_title("media-pp test_video")
                    .with_inner_size(LogicalSize::new(1280, 720))
                    // Renderer is wired up once, sized to the window's
                    // initial size — no resize handling here.
                    .with_resizable(false),
            )
            .expect("failed to create window");

        let hwnd = match window
            .window_handle()
            .expect("failed to get window handle")
            .as_raw()
        {
            RawWindowHandle::Win32(handle) => handle.hwnd.get(),
            _ => panic!("test_video example only supports Windows"),
        };
        let size = window.inner_size();

        let proxy = self.proxy.clone();
        self._playback = Some(thread::spawn(move || {
            if let Err(e) = play(hwnd, size.width, size.height) {
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

fn play(hwnd: isize, width: u32, height: u32) -> media_pp::Result<()> {
    media_pp::init()?;

    let options = TestVideoOptions {
        width,
        height,
        ..TestVideoOptions::default()
    };
    let source = TestVideoSource::new("test-video", options);

    let gpu = D3d12GpuContext::new().map_err(|e| media_pp::Error::Other(format!("{e:?}")))?;

    let pipeline = Pipeline::new("test-video", source, |source, ctx| {
        let renderer = render_common::d3d12_window_renderer("renderer", &gpu, hwnd, width, height)
            .expect("failed to create renderer");
        let branch = ChainBuilder::new(ctx.clone())
            .queue("frames", 8) // thread boundary so rendering doesn't block generation
            .build(Box::new(renderer));
        source.src_pads()[0].link(branch);
    });

    // `run()` starts playback on a background thread and returns right
    // away — any failure (e.g. an unsupported pixel format from
    // `Renderer`) shows up as a `BusEvent::Error` here instead of through
    // a returned `Result`. `TestVideoSource` never reaches `Eos` on its
    // own — closing the window is what ends this (see `Ok(())` below,
    // reached only via `pipeline.stop()` from outside this function, or
    // an error).
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
