use std::thread;

use ffmpeg_next as ffmpeg;
use media_pp::{
    bus::BusEvent,
    element::Source,
    elements::{D3d11Upload, Scaler, TestVideoOptions, TestVideoSource},
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

/// TestVideoSource -> Scaler -> D3d11Upload -> Renderer: a synthetic
/// `Pixel::YUV420P` stream converted to `Pixel::NV12` on the CPU, then
/// uploaded to a GPU `Pixel::D3D11` texture on the *renderer's own*
/// `ID3D11Device` before being presented — proves `D3d11Upload`'s frames
/// (built via plain `windows-rs` calls + `av_buffer_create`, not FFmpeg's
/// own hwframe pool — see `D3d11Upload`'s own docs) are readable by
/// `D3d11Renderer`'s zero-copy path. Compare against `d3d12_upload`, the
/// D3D12 sibling of this same smoke test.
///
///     cargo run -p d3d11_upload
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
                    .with_title("media-pp d3d11_upload")
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
            _ => panic!("d3d11_upload example only supports Windows"),
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

    let gpu = D3d11GpuContext::new().map_err(|e| media_pp::Error::Other(format!("{e:?}")))?;

    let pipeline = Pipeline::new("d3d11-upload", source, |source, ctx| {
        // `Pixel::NV12` — the only layout `D3d11Upload` accepts.
        let scaler = Scaler::new(
            "to-nv12",
            ffmpeg::format::Pixel::NV12,
            width,
            height,
            ffmpeg::software::scaling::Flags::BILINEAR,
        );
        // Same device the renderer draws with — required for the
        // zero-copy path to be valid at all (see D3d11Upload::new).
        let upload = D3d11Upload::new("upload", gpu.device(), width, height);
        let renderer = render_common::d3d11_window_renderer("renderer", &gpu, hwnd, width, height)
            .expect("failed to create renderer");

        let branch = ChainBuilder::new(ctx.clone())
            .queue("generated", 4) // thread boundary so scaling doesn't block generation
            .pipe(scaler)
            .queue("scaled", 4) // thread boundary so uploading doesn't block scaling
            .pipe(upload)
            .queue("frames", 8) // thread boundary so rendering doesn't block uploading
            .build(Box::new(renderer));
        source.src_pads()[0].link(branch);
    });

    // `run()` starts playback on a background thread and returns right
    // away — any failure (e.g. an unsupported pixel format anywhere in
    // the chain) shows up as a `BusEvent::Error` here instead of through a
    // returned `Result`. `TestVideoSource` never reaches `Eos` on its own
    // — closing the window is what ends this.
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
