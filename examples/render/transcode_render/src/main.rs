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

    use media_pp::{
        bus::BusEvent,
        elements::{
            Pacer, SwDecoder, SwEncoder, SwEncoderOptions, TestVideoOptions, TestVideoSource,
            VideoCodec,
        },
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

    /// TestVideoSource -> SwEncoder -> SwDecoder -> Pacer -> Renderer: encodes
    /// a synthetic moving-gradient stream (via `libopenh264`) and decodes it
    /// straight back — no file, camera, or container/mux involved at all —
    /// presented in a native window at real playback speed. Proves
    /// `SwEncoder`'s `Packet`s are actually valid, decodable H.264 (not just
    /// "avcodec_open2 succeeded"): if the round trip corrupted anything, the
    /// gradient would visibly glitch or freeze instead of scrolling smoothly.
    /// This example keeps a `Pacer` after the encode/decode round trip. The
    /// source itself is already paced accurately enough for direct rendering,
    /// but the encoder and decoder add their own buffering and per-frame
    /// variance; this particular chain has not been validated without the
    /// final clock-anchored pacing stage.
    ///
    ///     cargo run -p transcode_render
    pub(super) fn run() {
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
                        .with_title("media-pp transcode_render")
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
                _ => panic!("transcode_render example only supports Windows"),
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
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let options = TestVideoOptions {
            width,
            height,
            ..TestVideoOptions::default()
        };
        let source = TestVideoSource::new("test-video", options);
        let time_base = source.time_base();

        let gpu = D3d12GpuContext::new().map_err(|e| media_pp::Error::Other(format!("{e:?}")))?;

        let pipeline = Pipeline::new("transcode-render", source, |source, ctx| {
            let encoder = SwEncoder::new(
                "encoder",
                SwEncoderOptions {
                    codec: VideoCodec::OpenH264,
                    width,
                    height,
                    time_base,
                    frame_rate: options.framerate,
                    bit_rate: 2_000_000,
                    gop_size: 60, // ~2s @ 30fps (TestVideoOptions::default's own framerate)
                },
            )
            .expect("failed to open encoder");
            // No container/demuxer in this loop to get these from — SwEncoder
            // exposes its own codec parameters for exactly this case.
            let params = encoder.parameters();
            let decoder = SwDecoder::new("decoder", params).expect("failed to open decoder");
            let pacer = Pacer::new("pacer", time_base, ctx.clock.clone())?;
            let renderer =
                render_common::d3d12_window_renderer("renderer", &gpu, hwnd, width, height)
                    .expect("failed to create renderer");

            let branch = ctx
                .branch()
                .queue("to-encode", 8) // let generation run ahead of the (CPU-heavy) encoder
                .pipe(encoder)
                .queue("to-decode", 8) // let encode run ahead of decode
                .pipe(decoder)
                .queue("frames", 8) // pacer sleeps on its own thread; let decode run ahead into this
                .pipe(pacer)
                .to(Box::new(renderer))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })?;

        // `run()` starts playback on a background thread and returns right
        // away — any failure (encoder/decoder open already happened above and
        // would have panicked synchronously; a runtime failure like a bad
        // pixel format from `Renderer`) shows up as a `BusEvent::Error` here
        // instead of through a returned `Result`. `TestVideoSource` never
        // reaches `Eos` on its own — closing the window is what ends this.
        pipeline.run();

        for event in pipeline.bus().iter() {
            match &event {
                BusEvent::Eos { name, .. } => println!("[{name}] eos"),
                BusEvent::Error { name, error, .. } => eprintln!("[{name}] error: {error}"),
                BusEvent::Dropped { name, .. } => {
                    eprintln!("[{name}] dropped a buffer (queue full)")
                }
                // `BusEvent` is `#[non_exhaustive]`; this example only acts
                // on the events above.
                _ => {}
            }
            if matches!(event, BusEvent::Eos { .. } | BusEvent::Error { .. }) {
                pipeline.stop();
            }
        }
        Ok(())
    }
}
