#[cfg(not(target_os = "windows"))]
fn main() {
    eprintln!("{} example only supports Windows", env!("CARGO_PKG_NAME"));
}

#[cfg(target_os = "windows")]
fn main() -> impl std::process::Termination {
    windows_example::run()
}

#[cfg(target_os = "windows")]
mod windows_example {
    use std::thread;

    use ffmpeg_next::media;
    use media_pp::{
        Error,
        bus::BusEvent,
        elements::{D3d11Decoder, FileDemuxer, Pacer},
        pipeline::Pipeline,
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

    /// Demux -> D3d11Decoder -> Queue -> Pacer -> Renderer: decodes on the GPU
    /// via D3D11VA hardware acceleration and presents the frames in a native
    /// window at real playback speed, without ever copying the decoded pixels
    /// back to system memory — `D3d11Renderer` draws straight from the
    /// decoder's own D3D11 texture. The D3D11 sibling of `hw_decode_render`
    /// (which does the same thing via D3D12VA instead).
    ///
    /// `D3d11Decoder` never touches FFmpeg's `hw_frames_ctx`/
    /// `AVD3D11VAFramesContext` itself — only `hw_device_ctx` and `get_format`
    /// (see `D3d11Decoder`'s own docs) — so libavcodec's own internal D3D11VA
    /// hwaccel init handles frames-context allocation entirely inside
    /// already-correct C code, unlike the hand-mirrored struct path that
    /// crashed when this project tried to drive it manually (see
    /// `D3d11Upload`'s/`wrap_d3d11_texture`'s own docs on that history). This
    /// example is what actually proves that's safe on real hardware.
    ///
    ///     cargo run -p d3d11_decode_render -- path/to/video.mp4
    pub(super) fn run() {
        let Some(path) = std::env::args().nth(1) else {
            eprintln!("usage: d3d11_decode_render <video.mp4>");
            std::process::exit(1);
        };

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
                        .with_title("media-pp d3d11_decode_render")
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
                _ => panic!("d3d11_decode_render example only supports Windows"),
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
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

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

        let gpu = D3d11GpuContext::new(None).map_err(|e| Error::Other(format!("{e:?}")))?;

        let pipeline = Pipeline::new("d3d11-decode-render", source, |source, ctx| {
            // Same device the renderer draws with — required for the
            // zero-copy path to be valid at all (see D3d11Decoder::new).
            // `extra_hw_frames` must cover the `"frames"` queue's own depth
            // below (see D3d11Decoder::new's own docs on why, unlike
            // D3d12vaDecoder) — decode can legitimately run that far ahead of
            // playback while the queue buffers up.
            let decoder = D3d11Decoder::new("decoder", params, gpu.device(), 32)
                .expect("failed to open D3D11VA decoder");
            let pacer = Pacer::new("pacer", time_base, ctx.clock.clone())?;
            let renderer =
                render_common::d3d11_window_renderer("renderer", &gpu, hwnd, width, height)
                    .expect("failed to create renderer");
            let branch = ctx
                .branch()
                .pipe(decoder) // same thread as the demux — cheap enough not to need a queue
                .queue("frames", 32) // pacer sleeps on its own thread; let decode run ahead into this
                .pipe(pacer)
                .to(Box::new(renderer))?;
            ctx.attach(source, video.index, branch)?;
            Ok(())
        })?;

        // `run()` starts playback on a background thread and returns right
        // away — any failure (including the source's own) shows up as a
        // `BusEvent::Error` here instead of through a returned `Result`.
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
}
