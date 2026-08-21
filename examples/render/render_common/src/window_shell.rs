//! The window every GPU render example presents into, and the shutdown
//! ordering that presenting from another thread requires.
//!
//! # Why this is shared rather than copied per example
//!
//! Opening a window is the easy part. Closing one is not, and getting it
//! wrong is a use-after-free rather than a dropped frame: the pipeline
//! presents from its own thread, so the `Window` must not be dropped until
//! that thread is finished with it. Two orderings make that hard, and both
//! were live defects in examples that had each grown their own copy of this
//! scaffolding.
//!
//! * **Exiting the event loop drops the window.** On Wayland that destroys
//!   the `wl_surface` the swapchain presents to, and the NVIDIA WSI
//!   dispatches the display queue from whichever thread calls
//!   `vkQueuePresentKHR` — so a present still in flight walks a freed proxy
//!   and the process dies in `wl_proxy_destroy`. Holding the worker's
//!   `JoinHandle` does not prevent this: dropping a `JoinHandle` detaches its
//!   thread rather than waiting for it. [`App::drop`] joins, then lets go of
//!   the window.
//!
//! * **Stopping from the event loop thread deadlocks.**
//!   [`Pipeline::stop`] returns only once every element's control cascade has
//!   acknowledged, and a renderer can only acknowledge once the compositor
//!   releases a swapchain image — which needs this event loop to keep
//!   dispatching. So `CloseRequested` hands the stop to its own thread and
//!   keeps pumping; the window stays mapped, and `PlaybackDone` is what
//!   actually exits.
//!
//! [`Shutdown`] carries the third case: a close that arrives before the
//! worker has any pipeline to stop, which is ordinary rather than rare when
//! opening the source blocks on a portal dialog.

use std::{
    sync::{Arc, Mutex},
    thread,
};

use media_pp::pipeline::Pipeline;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle},
    window::{Window, WindowId},
};

/// Where the worker should present, handed to it once the window exists.
///
/// # Safety of the `Send` impl
///
/// These are raw pointers into a window the main thread owns, valid only
/// while it lives. [`App::drop`] is what makes handing them to another thread
/// sound: it joins the worker before the `Window` is dropped, on every path
/// out of the event loop. See this module's own docs.
pub struct WindowTarget {
    /// Only the Linux renderer reads this, for `VulkanGpuContext`; the
    /// Windows one presents through `window` alone. Compiling for one
    /// platform at a time makes that a dead-code warning on the other, not a
    /// reason to split the struct.
    #[allow(dead_code)]
    pub display: RawDisplayHandle,
    pub window: RawWindowHandle,
    pub width: u32,
    pub height: u32,
}

// SAFETY: see the type's own docs — the window outlives the worker thread.
unsafe impl Send for WindowTarget {}

/// The handshake between the window and the worker that owns the pipelines.
///
/// The worker [`publish`](Self::publish)es what it built; the window records
/// a close and stops whatever has been published by then. Both go through one
/// lock, so the two orders are equivalent: a close that beats `publish` is
/// reported back by `publish` itself.
#[derive(Default)]
pub struct Shutdown {
    state: Mutex<ShutdownState>,
}

#[derive(Default)]
struct ShutdownState {
    requested: bool,
    pipelines: Vec<Arc<Pipeline>>,
}

impl Shutdown {
    /// Worker: publishes the pipelines a close should stop, and reports
    /// whether one already arrived — in which case the worker should return
    /// instead of running anything.
    ///
    /// Call this after building and before [`Pipeline::run`], so no window
    /// event can find a running pipeline it cannot reach.
    pub fn publish(&self, pipelines: &[Arc<Pipeline>]) -> bool {
        let mut state = self.state.lock().expect("shutdown state poisoned");
        state.pipelines = pipelines.to_vec();
        state.requested
    }

    /// Whether the window has been closed, for a worker whose own loop would
    /// otherwise run to a fixed length with nothing left to present to.
    pub fn requested(&self) -> bool {
        self.state
            .lock()
            .expect("shutdown state poisoned")
            .requested
    }

    /// Window: records the close and hands back what to stop.
    fn request(&self) -> Vec<Arc<Pipeline>> {
        let mut state = self.state.lock().expect("shutdown state poisoned");
        state.requested = true;
        state.pipelines.clone()
    }
}

/// Opens a `width` x `height` window titled `title`, runs `play` on its own
/// thread with a target to present into, and returns once `play` has finished
/// and the window is closed — in that order.
///
/// The window is not resizable: every renderer in these examples is wired up
/// once at the size it is given.
pub fn run_window<F>(title: &str, width: u32, height: u32, play: F)
where
    F: FnOnce(WindowTarget, Arc<Shutdown>) -> media_pp::Result<()> + Send + 'static,
{
    let event_loop = EventLoop::<PlaybackDone>::with_user_event()
        .build()
        .expect("failed to create event loop");
    let proxy = event_loop.create_proxy();
    let mut app = App {
        title: title.to_string(),
        width,
        height,
        proxy,
        play: Some(play),
        window: None,
        shutdown: Arc::new(Shutdown::default()),
        stopper: None,
        playback: None,
    };
    event_loop.run_app(&mut app).expect("event loop failed");
}

/// Sent once `play` returns, so the window closes itself when the work is
/// done instead of sitting there until someone closes it by hand.
struct PlaybackDone;

struct App<F> {
    title: String,
    width: u32,
    height: u32,
    proxy: EventLoopProxy<PlaybackDone>,
    /// Taken by `resumed`, which may be called more than once.
    play: Option<F>,
    window: Option<Window>,
    shutdown: Arc<Shutdown>,
    /// Runs [`Pipeline::stop`] off the event loop thread; see the module docs.
    stopper: Option<thread::JoinHandle<()>>,
    /// Joined — not merely held — in [`App::drop`].
    playback: Option<thread::JoinHandle<()>>,
}

impl<F> App<F> {
    /// Ends the work without blocking the event loop. Idempotent: a second
    /// close request while the first stop is still running is ignored.
    fn begin_stop(&mut self) {
        if self.stopper.is_some() {
            return;
        }
        let shutdown = self.shutdown.clone();
        self.stopper = Some(thread::spawn(move || {
            for pipeline in shutdown.request() {
                pipeline.stop();
            }
        }));
    }
}

impl<F> Drop for App<F> {
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

impl<F> ApplicationHandler<PlaybackDone> for App<F>
where
    F: FnOnce(WindowTarget, Arc<Shutdown>) -> media_pp::Result<()> + Send + 'static,
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title(self.title.clone())
                    .with_inner_size(LogicalSize::new(self.width, self.height))
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

        let play = self.play.take().expect("resumed already started the work");
        let shutdown = self.shutdown.clone();
        let proxy = self.proxy.clone();
        self.playback = Some(thread::spawn(move || {
            if let Err(error) = play(target, shutdown) {
                eprintln!("playback failed: {error}");
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
            // Not `exit()`: the window has to stay mapped, and this loop has
            // to keep dispatching, until the worker is done with it.
            // `PlaybackDone` is what exits. See the module docs.
            self.begin_stop();
        }
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, _event: PlaybackDone) {
        event_loop.exit();
    }
}
