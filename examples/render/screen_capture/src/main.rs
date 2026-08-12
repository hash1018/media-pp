use std::thread;

use ffmpeg_next as ffmpeg;
use media_pp::{
    bus::BusEvent,
    element::ElementType,
    elements::{CaptureMode, DxgiCaptureOptions, DxgiCaptureSource, Scaler},
    pipeline::Pipeline,
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

/// DxgiCaptureSource -> Scaler -> Renderer: captures the desktop live via
/// DXGI Desktop Duplication (cursor included) at a constant frame rate
/// (`DxgiCaptureOptions::fps`) and converts/resizes it to the window's own
/// size as `Pixel::YUV420P` before rendering — no `SwEncoder`/`SwDecoder`
/// round trip.
///
/// No `Pacer` here, confirmed unneeded: `DxgiCaptureSource` previously
/// emitted variable-rate (real wall-clock pts, push-on-change), and
/// removing `Pacer` against that measurably caused judder. It's since
/// been rewritten to emit at a constant rate on a drift-free absolute
/// schedule instead — the same pattern `TestVideoSource` uses (see
/// `test_video`) — and with that fixed, `Scaler` sitting between source
/// and renderer here doesn't add enough jitter on its own to bring the
/// judder back. The constant-rate/drift-free change was the actual fix,
/// not the presence of a `Pacer` stage.
///
///     cargo run -p screen_capture
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
                    .with_title("media-pp screen_capture")
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
            _ => panic!("screen_capture example only supports Windows"),
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

    let capture_options = DxgiCaptureOptions {
        fps: 60,
        capture_mode: CaptureMode::Cpu {
            include_cursor: true,
        },
        ..DxgiCaptureOptions::default()
    };
    let (source, _capture_width, _capture_height, _device) =
        DxgiCaptureSource::open("screen", capture_options)?;

    let gpu = D3d12GpuContext::new().map_err(|e| media_pp::Error::Other(format!("{e:?}")))?;

    let pipeline = Pipeline::new("screen-capture", source, |source, ctx| {
        // Converts the captured `Pixel::BGRA` desktop frames down to the
        // window's own size as `Pixel::YUV420P` in one pass —
        // `D3d12Renderer`'s CPU-upload path only understands
        // YUV420P/D3D12, not BGRA.
        let scaler = Scaler::new(
            "to-yuv",
            ffmpeg::format::Pixel::YUV420P,
            window_width,
            window_height,
            ffmpeg::software::scaling::Flags::BILINEAR,
        );
        let renderer = render_common::d3d12_window_renderer(
            "renderer",
            &gpu,
            hwnd,
            window_width,
            window_height,
        )
        .expect("failed to create renderer");

        let branch = ctx
            .branch()
            .queue("captured", 4) // thread boundary so scaling doesn't block capture
            .pipe(scaler)
            .queue("frames", 8) // thread boundary so rendering doesn't block scaling
            .to(Box::new(renderer))?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })?;

    // `run()` starts capture on a background thread and returns right
    // away — any failure shows up as a `BusEvent::Error` here instead of
    // through a returned `Result`. `DxgiCaptureSource` never reaches `Eos`
    // on its own — closing the window is what ends this.
    pipeline.run();

    for event in pipeline.bus().iter() {
        match &event {
            BusEvent::Eos { name, .. } => println!("[{name}] eos"),
            BusEvent::Error { name, error, .. } => eprintln!("[{name}] error: {error}"),
            BusEvent::Dropped { name, .. } => eprintln!("[{name}] dropped a buffer (queue full)"),
            BusEvent::Seeked { .. } => {}
        }
        // Unlike the other render examples (a steady, self-paced
        // synthetic/file source that essentially never overruns the
        // renderer's 2-slot upload ring), live desktop capture can
        // legitimately burst faster than that ring drains — e.g. right at
        // startup, before the render thread has processed even its first
        // frame. `D3d12RendererError::Submit(NoFreeSlot)` on an occasional
        // frame is expected backpressure, not a reason to end the whole
        // demo — the `Queue` in front of the renderer already drops just
        // that one buffer and keeps going (see `Queue`'s own "report,
        // don't die" contract). Only stop for `Eos`, or an `Error` from
        // `DxgiCaptureSource` itself (its `run()` thread actually ended —
        // e.g. `DXGI_ERROR_ACCESS_LOST` from a lock screen — so nothing
        // more will ever arrive).
        let source_died = matches!(
            &event,
            BusEvent::Error { element_type, .. } if *element_type == ElementType::DxgiCaptureSource
        );
        if matches!(event, BusEvent::Eos { .. }) || source_died {
            pipeline.stop();
        }
    }
    Ok(())
}
