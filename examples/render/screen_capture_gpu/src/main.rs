use std::thread;

use media_pp::{
    bus::BusEvent,
    element::{ElementType, Source},
    elements::{CaptureMode, DxgiScreenOptions, DxgiScreenSource},
    pipeline::{ChainBuilder, Pipeline},
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

/// DxgiScreenSource (GPU mode) -> Renderer: captures the desktop straight
/// to a GPU-resident `Pixel::D3D11` BGRA texture on the *renderer's own*
/// `ID3D11Device` (no `Map`, no CPU pixel copy at all — see
/// `CaptureMode::Gpu`'s own docs) and presents it directly, no `Scaler`
/// (desktop content is already BGRA/RGB, no YUV conversion needed, and
/// `D3d11Renderer` letterboxes any capture size into the window on its
/// own). Compare against `screen_capture`, which captures to a plain CPU
/// `Pixel::BGRA` frame instead and converts it to YUV420P for the D3D12
/// CPU-upload path.
///
/// No cursor: `CaptureMode::Gpu` doesn't support cursor compositing yet
/// (see that variant's own docs) — `screen_capture`'s CPU path does.
///
///     cargo run -p screen_capture_gpu
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
                    .with_title("media-pp screen_capture_gpu")
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
            _ => panic!("screen_capture_gpu example only supports Windows"),
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

fn play(hwnd: isize, window_width: u32, window_height: u32) -> media_pp::Result<()> {
    media_pp::init()?;

    // Built first: `DxgiScreenSource`'s GPU mode needs a device up front
    // (and verifies it's on the same adapter `output_index` selects) —
    // the same device `render_common::d3d11_window_renderer` below draws
    // with, required for the zero-copy path to be valid at all.
    let gpu = D3d11GpuContext::new().map_err(|e| media_pp::Error::Other(format!("{e:?}")))?;

    let capture_options = DxgiScreenOptions {
        fps: 60,
        capture_mode: CaptureMode::Gpu {
            device: gpu.device().clone(),
        },
        ..DxgiScreenOptions::default()
    };
    let (source, _capture_width, _capture_height) =
        DxgiScreenSource::open("screen", capture_options)?;

    let pipeline = Pipeline::new("screen-capture-gpu", source, |source, ctx| {
        let renderer = render_common::d3d11_window_renderer(
            "renderer",
            &gpu,
            hwnd,
            window_width,
            window_height,
        )
        .expect("failed to create renderer");

        let branch = ChainBuilder::new(ctx.clone())
            .queue("captured", 4) // thread boundary so rendering doesn't block capture
            .build(Box::new(renderer));
        source.src_pads()[0].link(branch);
    });

    // `run()` starts capture on a background thread and returns right
    // away — any failure shows up as a `BusEvent::Error` here instead of
    // through a returned `Result`. `DxgiScreenSource` never reaches `Eos`
    // on its own — closing the window is what ends this.
    pipeline.run();

    for event in pipeline.bus().iter() {
        match &event {
            BusEvent::Eos { name, .. } => println!("[{name}] eos"),
            BusEvent::Error { name, error, .. } => eprintln!("[{name}] error: {error}"),
            BusEvent::Dropped { name, .. } => eprintln!("[{name}] dropped a buffer (queue full)"),
            BusEvent::Seeked { .. } => {}
        }
        // Same reasoning as `screen_capture`'s own loop: only stop for
        // `Eos`, or an `Error` that means `DxgiScreenSource`'s own `run()`
        // thread actually ended — an occasional dropped/backpressured
        // frame elsewhere isn't a reason to end the demo.
        let source_died = matches!(
            &event,
            BusEvent::Error { element_type, .. } if *element_type == ElementType::DxgiScreenSource
        );
        if matches!(event, BusEvent::Eos { .. }) || source_died {
            pipeline.stop();
        }
    }
    Ok(())
}
