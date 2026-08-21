#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn main() {
    eprintln!(
        "{} supports Windows (DXGI) and Linux (PipeWire) only",
        env!("CARGO_PKG_NAME")
    );
}

#[cfg(target_os = "windows")]
fn main() -> impl std::process::Termination {
    windows_example::run()
}

#[cfg(target_os = "linux")]
fn main() -> impl std::process::Termination {
    linux_example::run()
}

#[cfg(target_os = "windows")]
mod windows_example {
    use std::{
        sync::{Arc, Mutex},
        thread,
    };

    use media_pp::{
        bus::BusEvent,
        element::ElementType,
        elements::{CaptureMode, DxgiCaptureOptions, DxgiCaptureSource},
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

    /// DxgiCaptureSource (GPU mode) -> Renderer: captures the desktop straight
    /// to a GPU-resident `Pixel::D3D11` BGRA texture on the *renderer's own*
    /// `ID3D11Device` (no `Map`, no CPU pixel copy at all — see
    /// `CaptureMode::Gpu`'s own docs) and presents it directly, no `SwScaler`
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
    pub(super) fn run() {
        let event_loop = EventLoop::<PlaybackDone>::with_user_event()
            .build()
            .expect("failed to create event loop");
        let proxy = event_loop.create_proxy();
        let mut app = App {
            proxy,
            window: None,
            shutdown: Arc::new(Shutdown::default()),
            stopper: None,
            playback: None,
        };
        event_loop.run_app(&mut app).expect("event loop failed");
    }

    /// Sent from the playback thread once `play()` returns, so the window
    /// closes itself when the pipeline finishes instead of sitting there
    /// until someone closes it by hand.
    struct PlaybackDone;

    /// What lets the window outlive the thread presenting into it.
    ///
    /// Two things have to happen in order when the window is closed, and the
    /// event loop thread can do neither of them directly:
    ///
    /// * [`Pipeline::stop`] blocks until every element's control cascade
    ///   acknowledges, and the renderer's own `consume` can only acknowledge
    ///   once the swapchain releases an image — which needs this event loop
    ///   to keep pumping. Stopping from the event loop thread therefore
    ///   deadlocks, so [`App::window_event`] hands the stop to its own thread.
    /// * The window must not be dropped until the capture thread is done
    ///   presenting to that `HWND`, so [`App::drop`] joins first.
    ///
    /// `requested` covers the order where the window is closed before the
    /// capture thread has a pipeline to stop at all.
    #[derive(Default)]
    struct Shutdown {
        state: Mutex<ShutdownState>,
    }

    #[derive(Default)]
    struct ShutdownState {
        requested: bool,
        pipeline: Option<Arc<Pipeline>>,
    }

    impl Shutdown {
        /// Main thread: records the close and hands back the pipeline to stop,
        /// if the capture thread got far enough to publish one.
        fn request(&self) -> Option<Arc<Pipeline>> {
            let mut state = self.state.lock().expect("shutdown state poisoned");
            state.requested = true;
            state.pipeline.clone()
        }

        /// Capture thread: publishes the pipeline and reports whether a close
        /// already arrived, which is what makes the two orders equivalent.
        fn publish(&self, pipeline: Arc<Pipeline>) -> bool {
            let mut state = self.state.lock().expect("shutdown state poisoned");
            state.pipeline = Some(pipeline);
            state.requested
        }
    }

    struct App {
        proxy: EventLoopProxy<PlaybackDone>,
        window: Option<Window>,
        shutdown: Arc<Shutdown>,
        /// Runs [`Pipeline::stop`] off the event loop thread; see [`Shutdown`].
        stopper: Option<thread::JoinHandle<()>>,
        /// Joined — not merely held — in [`App::drop`], since dropping a
        /// `JoinHandle` detaches its thread rather than waiting for it.
        playback: Option<thread::JoinHandle<()>>,
    }

    impl App {
        /// Ends the capture without blocking the event loop. Idempotent: a
        /// second close request while the first stop is still running is
        /// ignored.
        fn begin_stop(&mut self) {
            if self.stopper.is_some() {
                return;
            }
            let shutdown = self.shutdown.clone();
            self.stopper = Some(thread::spawn(move || {
                if let Some(pipeline) = shutdown.request() {
                    pipeline.stop();
                }
            }));
        }
    }

    impl Drop for App {
        fn drop(&mut self) {
            self.begin_stop();
            if let Some(stopper) = self.stopper.take() {
                let _ = stopper.join();
            }
            if let Some(playback) = self.playback.take() {
                let _ = playback.join();
            }
            // Only now is it safe to let go of the window.
            self.window = None;
        }
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
            let shutdown = self.shutdown.clone();
            self.playback = Some(thread::spawn(move || {
                if let Err(e) = play(hwnd, size.width, size.height, &shutdown) {
                    eprintln!("playback failed: {e}");
                }
                let _ = proxy.send_event(PlaybackDone);
            }));
            self.window = Some(window);
        }

        fn window_event(
            &mut self,
            _event_loop: &ActiveEventLoop,
            window_id: WindowId,
            event: WindowEvent,
        ) {
            if self.window.as_ref().map(Window::id) != Some(window_id) {
                return;
            }
            if let WindowEvent::CloseRequested = event {
                // Not `exit()`: the window has to stay mapped, and this loop
                // has to keep dispatching, until capture is done with it.
                // `PlaybackDone` is what exits. See `Shutdown`.
                self.begin_stop();
            }
        }

        fn user_event(&mut self, event_loop: &ActiveEventLoop, _event: PlaybackDone) {
            event_loop.exit();
        }
    }

    fn play(
        hwnd: isize,
        window_width: u32,
        window_height: u32,
        shutdown: &Shutdown,
    ) -> media_pp::Result<()> {
        media_pp::init()?;
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        // Opened first: `CaptureMode::Gpu` builds its own device from
        // whichever adapter `output_index` actually selects and hands it
        // back — `render_common::d3d11_window_renderer` below is built from
        // that same returned device, required for the zero-copy path to be
        // valid at all (see `CaptureMode::Gpu`'s own docs on why the device
        // flows this direction, not the other way).
        let capture_options = DxgiCaptureOptions {
            fps: 60,
            capture_mode: CaptureMode::Gpu,
            ..DxgiCaptureOptions::default()
        };
        let (source, _format, device) = DxgiCaptureSource::open("screen", capture_options)?;
        let device = device.expect("CaptureMode::Gpu always returns a device");

        let gpu = D3d11GpuContext::new(Some(device))
            .map_err(|e| media_pp::Error::Other(format!("{e:?}")))?;

        let pipeline = Pipeline::new("screen-capture-gpu", source, |source, ctx| {
            let renderer = render_common::d3d11_window_renderer(
                "renderer",
                &gpu,
                hwnd,
                window_width,
                window_height,
            )
            .expect("failed to create renderer");

            let branch = ctx
                .branch()
                .queue("captured", 4) // thread boundary so rendering doesn't block capture
                .to(Box::new(renderer))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })?;

        // Published before `run`, so a close that arrives from here on finds
        // a pipeline to stop. `true` means one already did, and nothing has
        // presented yet.
        if shutdown.publish(pipeline.clone()) {
            return Ok(());
        }

        // `run()` starts capture on a background thread and returns right
        // away — any failure shows up as a `BusEvent::Error` here instead of
        // through a returned `Result`. `DxgiCaptureSource` never reaches `Eos`
        // on its own — closing the window is what ends this.
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
            // Same reasoning as `screen_capture`'s own loop: only stop for
            // `Eos`, or an `Error` that means `DxgiCaptureSource`'s own `run()`
            // thread actually ended — an occasional dropped/backpressured
            // frame elsewhere isn't a reason to end the demo.
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
}

/// The Linux half of the same example: capture straight into GPU memory and
/// present it, with no pixel ever passing through system memory.
///
/// The graph is one element longer than the Windows branch, and the platform
/// forces exactly that one. DXGI hands over a BGRA texture that
/// `D3d11Renderer` presents as-is; PipeWire hands over a DMA-BUF that
/// `open_gpu` imports as a BGRA CUDA surface, and `CudaRenderer` presents
/// NV12 — so `CudaConverter` sits between them. That element exists for this
/// shape: without it a GPU capture can only be encoded (NVENC ingests BGRA
/// directly), never shown or composited.
///
/// The CLI differences are the ones `screen_record` documents: Wayland has no
/// way to name a monitor, so the compositor prompts on the first run and
/// hands back a restore token later runs can pass to skip the dialog.
///
///     cargo run -p screen_capture_gpu -- [monitor|window] [restore-token]
#[cfg(target_os = "linux")]
mod linux_example {
    use std::{
        sync::{Arc, Mutex},
        thread,
    };

    use media_pp::{
        bus::BusEvent,
        element::ElementType,
        elements::{
            CaptureSourceKind, CudaConverter, CudaDevice, PipeWireScreenCaptureOptions,
            PipeWireScreenCaptureSource,
        },
        pipeline::Pipeline,
    };
    use render_common::VulkanGpuContext;
    use winit::{
        application::ApplicationHandler,
        dpi::LogicalSize,
        event::WindowEvent,
        event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy},
        raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle},
        window::{Window, WindowId},
    };

    pub(super) fn run() {
        let event_loop = EventLoop::<PlaybackDone>::with_user_event()
            .build()
            .expect("failed to create event loop");
        let proxy = event_loop.create_proxy();
        let mut app = App {
            proxy,
            window: None,
            shutdown: Arc::new(Shutdown::default()),
            stopper: None,
            playback: None,
        };
        event_loop.run_app(&mut app).expect("event loop failed");
    }

    /// Sent from the capture thread once `play()` returns, so the window
    /// closes itself when the pipeline finishes.
    struct PlaybackDone;

    /// What lets the window outlive the thread presenting into it.
    ///
    /// Two things have to happen in order when the window is closed, and the
    /// event loop thread can do neither of them directly:
    ///
    /// * [`Pipeline::stop`] blocks until every element's control cascade
    ///   acknowledges, and the renderer's own `consume` can only acknowledge
    ///   once the compositor releases a swapchain image — which needs this
    ///   event loop to keep dispatching. Stopping from the event loop thread
    ///   therefore deadlocks, so [`App::window_event`] hands the stop to its
    ///   own thread and keeps pumping.
    /// * The window must not be dropped until the capture thread is done with
    ///   it. On Wayland, tearing down the `wl_surface` under a
    ///   `vkQueuePresentKHR` is a use-after-free, not a lost frame, so
    ///   [`App::drop`] joins before letting go of the window.
    ///
    /// `requested` covers the order where the window is closed while
    /// `open_gpu` is still blocked on the portal dialog and there is no
    /// pipeline to stop at all.
    #[derive(Default)]
    struct Shutdown {
        state: Mutex<ShutdownState>,
    }

    #[derive(Default)]
    struct ShutdownState {
        requested: bool,
        pipeline: Option<Arc<Pipeline>>,
    }

    impl Shutdown {
        /// Main thread: records the close and hands back the pipeline to stop,
        /// if the capture thread got far enough to publish one.
        fn request(&self) -> Option<Arc<Pipeline>> {
            let mut state = self.state.lock().expect("shutdown state poisoned");
            state.requested = true;
            state.pipeline.clone()
        }

        /// Capture thread: publishes the pipeline and reports whether a close
        /// already arrived, which is what makes the two orders equivalent.
        fn publish(&self, pipeline: Arc<Pipeline>) -> bool {
            let mut state = self.state.lock().expect("shutdown state poisoned");
            state.pipeline = Some(pipeline);
            state.requested
        }
    }

    /// Where the renderer presents, handed to the capture thread.
    ///
    /// # Safety of the `Send` impl
    ///
    /// On Wayland these are raw pointers into the window the main thread
    /// owns, valid only while it lives. `App` holds the window until
    /// `run_app` returns, and [`App::finish`] — the only path to that return
    /// — stops the pipeline and joins the capture thread first, so the
    /// pointers outlive every use of them. See [`Shutdown`] for why merely
    /// holding the `JoinHandle` was not enough.
    struct WindowTarget {
        display: RawDisplayHandle,
        window: RawWindowHandle,
        width: u32,
        height: u32,
    }

    // SAFETY: see the type's own docs — the window outlives the capture thread.
    unsafe impl Send for WindowTarget {}

    struct App {
        proxy: EventLoopProxy<PlaybackDone>,
        window: Option<Window>,
        shutdown: Arc<Shutdown>,
        /// Runs [`Pipeline::stop`] off the event loop thread; see [`Shutdown`].
        stopper: Option<thread::JoinHandle<()>>,
        /// Joined — not merely held — in [`App::drop`], since dropping a
        /// `JoinHandle` detaches its thread rather than waiting for it.
        playback: Option<thread::JoinHandle<()>>,
    }

    impl App {
        /// Ends the capture without blocking the event loop. Idempotent: a
        /// second close request while the first stop is still running is
        /// ignored.
        fn begin_stop(&mut self) {
            if self.stopper.is_some() {
                return;
            }
            let shutdown = self.shutdown.clone();
            self.stopper = Some(thread::spawn(move || {
                if let Some(pipeline) = shutdown.request() {
                    pipeline.stop();
                }
            }));
        }
    }

    impl Drop for App {
        fn drop(&mut self) {
            self.begin_stop();
            if let Some(stopper) = self.stopper.take() {
                let _ = stopper.join();
            }
            if let Some(playback) = self.playback.take() {
                let _ = playback.join();
            }
            // Only now is it safe to let go of the window.
            self.window = None;
        }
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
            let size = window.inner_size();
            let target = WindowTarget {
                display: window
                    .display_handle()
                    .expect("failed to get display handle")
                    .as_raw(),
                window: window
                    .window_handle()
                    .expect("failed to get window handle")
                    .as_raw(),
                width: size.width,
                height: size.height,
            };

            let proxy = self.proxy.clone();
            let shutdown = self.shutdown.clone();
            self.playback = Some(thread::spawn(move || {
                if let Err(error) = play(target, &shutdown) {
                    eprintln!("capture failed: {error}");
                }
                let _ = proxy.send_event(PlaybackDone);
            }));
            self.window = Some(window);
        }

        fn window_event(
            &mut self,
            _event_loop: &ActiveEventLoop,
            window_id: WindowId,
            event: WindowEvent,
        ) {
            if self.window.as_ref().map(Window::id) != Some(window_id) {
                return;
            }
            if let WindowEvent::CloseRequested = event {
                // Not `exit()`: the window has to stay mapped, and this loop
                // has to keep dispatching, until capture is done with it.
                // `PlaybackDone` is what exits. See `Shutdown`.
                self.begin_stop();
            }
        }

        fn user_event(&mut self, event_loop: &ActiveEventLoop, _event: PlaybackDone) {
            event_loop.exit();
        }
    }

    fn play(target: WindowTarget, shutdown: &Shutdown) -> media_pp::Result<()> {
        media_pp::init()?;
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let source_kind = match std::env::args().nth(1).as_deref() {
            Some("window") => CaptureSourceKind::Window,
            _ => CaptureSourceKind::Monitor,
        };
        // Last so it can simply be left off: it is a long opaque string that
        // only a repeat run has.
        let restore_token = std::env::args().nth(2);
        if restore_token.is_none() {
            eprintln!("opening the portal — approve the screen-share dialog to continue...");
        }

        // One CUDA context for the whole stack: the capture allocates its
        // surfaces on it, the converter allocates from it, and the renderer
        // imports its Vulkan memory into it. Each element rejects a frame
        // from a different one.
        let cuda = CudaDevice::new().map_err(|error| media_pp::Error::Other(error.to_string()))?;
        let (source, format, restore_token) = PipeWireScreenCaptureSource::open_gpu(
            "screen",
            PipeWireScreenCaptureOptions {
                fps: 60,
                source_kind,
                include_cursor: true,
                restore_token,
            },
            &cuda,
        )?;
        let gpu = VulkanGpuContext::new(target.display).map_err(media_pp::Error::Other)?;

        let (width, height) = (format.width, format.height);
        let pipeline = Pipeline::new("screen-capture-gpu", source, |source, ctx| {
            // The capture's own size, not a rounded one: the converter is
            // fixed-size, so anything else would reject every frame. An odd
            // capture is refused here rather than at the first frame — see
            // `CudaConverter`, whose chroma has no half sample to write.
            let converter = CudaConverter::new("convert", &cuda, width, height)
                .map_err(|error| media_pp::Error::Other(error.to_string()))?;
            let renderer = render_common::cuda_window_renderer(
                "renderer",
                &gpu,
                &cuda,
                target.display,
                target.window,
                target.width,
                target.height,
            )
            .map_err(media_pp::Error::Other)?;

            let branch = ctx
                .branch()
                // Thread boundary so conversion and presentation cannot stall
                // capture; the compositor keeps producing at its own rate.
                .queue("captured", 4)
                .pipe(converter)
                .to(Box::new(renderer))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })?;

        // Published before `run`, so a close that arrives from here on finds
        // a pipeline to stop. `true` means one already did — while the portal
        // dialog was up, say — and nothing has presented yet.
        if shutdown.publish(pipeline.clone()) {
            return Ok(());
        }

        println!("presenting a {width}x{height} capture — close the window to stop");
        // `run()` starts capture on a background thread and returns right
        // away — any failure shows up as a `BusEvent::Error` here instead of
        // through a returned `Result`. This source never reaches `Eos` on its
        // own; closing the window, or the captured source going away, is what
        // ends this.
        pipeline.run();

        for event in pipeline.bus().iter() {
            match &event {
                BusEvent::Eos { name, .. } => println!("[{name}] eos"),
                BusEvent::Error { name, error, .. } => eprintln!("[{name}] error: {error}"),
                BusEvent::Dropped { name, .. } => {
                    eprintln!("[{name}] dropped a buffer (queue full)")
                }
                // `BusEvent` is `#[non_exhaustive]`; this example only acts on
                // the events above.
                _ => {}
            }
            // Same reasoning as the Windows branch: only stop for `Eos`, or an
            // `Error` that means the capture's own `run()` thread ended — one
            // dropped frame elsewhere is not a reason to end the demo.
            let source_died = matches!(
                &event,
                BusEvent::Error { element_type, .. }
                    if *element_type == ElementType::PipeWireScreenCaptureSource
            );
            if matches!(event, BusEvent::Eos { .. }) || source_died {
                pipeline.stop();
            }
        }

        match restore_token {
            Some(token) => println!(
                "re-run without a dialog:\n  ... {} {token}",
                if matches!(source_kind, CaptureSourceKind::Window) {
                    "window"
                } else {
                    "monitor"
                }
            ),
            None => println!("the compositor issued no restore token; the next run will prompt"),
        }
        Ok(())
    }
}
