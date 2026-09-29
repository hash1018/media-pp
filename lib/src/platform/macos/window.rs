//! A window a renderer opens for itself, and the main-thread event loop it
//! needs — the macOS counterpart of `platform::windows::window` and
//! `platform::linux::x11_window`, which each run a window on a thread of
//! its own.
//!
//! AppKit allows none of that: every window, and the one event loop that
//! serves them all, belong to the process's main thread. So a window here
//! is made on the main thread — by a block sent to its dispatch queue when
//! asked from anywhere else — and is served by whatever event loop the main
//! thread runs: [`run_with_windows`]'s, for a program that has none of its
//! own, or the application's (`NSApplication`'s, `winit`'s), which drains
//! the same queue. Drawing into it needs no main thread: its
//! `CAMetalLayer` is drawn from whichever thread the pipeline runs the
//! renderer on.

use std::{
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::Duration,
};

use crossbeam_channel::Sender;
use dispatch2::{DispatchQueue, MainThreadBound};
use objc2::{
    DefinedClass, MainThreadMarker, MainThreadOnly, define_class, msg_send,
    rc::Retained,
    runtime::{NSObject, NSObjectProtocol, ProtocolObject},
};
use objc2_app_kit::{
    NSApplication, NSApplicationActivationPolicy, NSBackingStoreType, NSEvent,
    NSEventModifierFlags, NSEventType, NSResponder, NSView, NSWindow, NSWindowDelegate,
    NSWindowStyleMask,
};
use objc2_foundation::{NSNotification, NSPoint, NSRect, NSSize, NSString};
use objc2_quartz_core::CAMetalLayer;
use thiserror::Error as ThisError;

use crate::elements::{Key, MouseButton, WindowEvent, WindowGone, WindowOptions};

/// How long a window may take to be made on the main thread before the main
/// thread is taken to be running no event loop at all.
const MAIN_THREAD_TIMEOUT: Duration = Duration::from_secs(5);

/// Why a window could not be opened.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ThisError)]
pub enum WindowError {
    /// The main thread runs no event loop to make the window on: a program
    /// opens windows inside [`run_with_windows`](crate::elements::run_with_windows),
    /// or runs an event loop of its own on its main thread.
    #[error(
        "the main thread runs no event loop to open a window on; run the program inside media_pp::elements::run_with_windows"
    )]
    NoMainLoop,
}

/// Runs `work` on a thread of its own while the main thread runs AppKit's
/// event loop, and returns what it returns once it has — the macOS form of
/// a program's `main`, for one that opens windows and has no event loop of
/// its own. See [`crate::elements::run_with_windows`].
pub(crate) fn run_with_windows<R: Send + 'static>(work: impl FnOnce() -> R + Send + 'static) -> R {
    let mtm = MainThreadMarker::new().expect("run_with_windows is called from the main thread");
    let app = NSApplication::sharedApplication(mtm);
    app.setActivationPolicy(NSApplicationActivationPolicy::Regular);
    let (done_tx, done_rx) = mpsc::channel();
    let worker = std::thread::Builder::new()
        .name("main".into())
        .spawn(move || {
            let outcome = catch_unwind(AssertUnwindSafe(work));
            let _ = done_tx.send(());
            DispatchQueue::main().exec_async(stop_main_loop);
            outcome
        })
        .expect("a thread for the program's work");
    // Ended by the worker once it has returned; `run` returns only for a
    // stop, so a stop that came before it started is still seen, as the
    // block that asks for it runs inside it.
    if done_rx.try_recv().is_err() {
        app.run();
    }
    match worker.join() {
        Ok(Ok(result)) => result,
        Ok(Err(panic)) | Err(panic) => resume_unwind(panic),
    }
}

/// Stops the event loop `run_with_windows` runs. Called on the main thread.
fn stop_main_loop() {
    // SAFETY: a block on the main dispatch queue runs on the main thread.
    let mtm = unsafe { MainThreadMarker::new_unchecked() };
    let app = NSApplication::sharedApplication(mtm);
    app.stop(None);
    // `stop` takes effect once the loop has handled an event; this is one.
    let wake = NSEvent::otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2(
        NSEventType::ApplicationDefined,
        NSPoint::new(0.0, 0.0),
        NSEventModifierFlags::empty(),
        0.0,
        0,
        None,
        0,
        0,
        0,
    );
    if let Some(wake) = wake {
        app.postEvent_atStart(&wake, true);
    }
}

/// Where a window's events go: taken when its renderer is dropped, so
/// `WindowEvents::recv` returns `None` from then whatever the main thread
/// is doing.
type Events = Arc<Mutex<Option<Sender<WindowEvent>>>>;

fn send(events: &Events, event: WindowEvent) {
    if let Some(events) = events.lock().ok().and_then(|events| events.clone()) {
        let _ = events.send(event);
    }
}

/// What a window and its [`Control`]s share.
#[derive(Debug, Default)]
struct Shared {
    /// Whether it was last asked to fill the screen.
    fullscreen: AtomicBool,
    /// Set when its renderer lets go of it.
    gone: AtomicBool,
}

/// The AppKit objects of one window, which only the main thread touches.
struct Parts {
    window: Retained<NSWindow>,
    /// The window holds its delegate weakly; this keeps it.
    _delegate: Retained<WindowDelegate>,
}

/// A `CAMetalLayer`, drawn into from any thread — which Core Animation
/// allows, and which is all a renderer does with it.
pub(crate) struct MetalLayer(pub(crate) Retained<CAMetalLayer>);

// SAFETY: Core Animation's layers are thread-safe to use from any thread;
// a Metal layer's drawables are made to be taken and presented off the main
// thread.
unsafe impl Send for MetalLayer {}
// SAFETY: as above.
unsafe impl Sync for MetalLayer {}

/// A window on the main thread, closed when this drops.
pub(crate) struct OwnedWindow {
    parts: Option<Arc<MainThreadBound<Parts>>>,
    layer: Arc<MetalLayer>,
    events: Events,
    shared: Arc<Shared>,
}

impl OwnedWindow {
    /// Opens a window with a picture area of `options`' size — in points,
    /// as AppKit sizes windows, so twice as many pixels each way on a Retina
    /// display — and returns once it exists. What happens to it goes to
    /// `events`.
    pub(crate) fn open(
        options: &WindowOptions,
        events: Sender<WindowEvent>,
    ) -> Result<Self, WindowError> {
        let events: Events = Arc::new(Mutex::new(Some(events)));
        let title = options.title.clone();
        let size = NSSize::new(f64::from(options.width), f64::from(options.height));
        let made = if let Some(mtm) = MainThreadMarker::new() {
            create(mtm, &title, size, &events)
        } else {
            let (made_tx, made_rx) = mpsc::channel();
            let cancelled = Arc::new(AtomicBool::new(false));
            {
                let events = Arc::clone(&events);
                let cancelled = Arc::clone(&cancelled);
                DispatchQueue::main().exec_async(move || {
                    // Given up on: an event loop started after the wait
                    // ended must not open a window nobody draws into.
                    if cancelled.load(Ordering::Acquire) {
                        return;
                    }
                    // SAFETY: a block on the main dispatch queue runs on the
                    // main thread.
                    let mtm = unsafe { MainThreadMarker::new_unchecked() };
                    let _ = made_tx.send(create(mtm, &title, size, &events));
                });
            }
            match made_rx.recv_timeout(MAIN_THREAD_TIMEOUT) {
                Ok(made) => made,
                Err(_) => {
                    cancelled.store(true, Ordering::Release);
                    return Err(WindowError::NoMainLoop);
                }
            }
        };
        let (parts, layer) = made;
        Ok(Self {
            parts: Some(Arc::new(parts)),
            layer: Arc::new(layer),
            events,
            shared: Arc::default(),
        })
    }

    /// The layer the window's picture area is drawn through.
    pub(crate) fn layer(&self) -> Arc<MetalLayer> {
        Arc::clone(&self.layer)
    }

    /// What changes the window's title and fullscreen state from another
    /// thread.
    pub(crate) fn control(&self) -> Control {
        Control {
            parts: self.parts.as_ref().map(Arc::downgrade).unwrap_or_default(),
            shared: Arc::clone(&self.shared),
        }
    }
}

impl Drop for OwnedWindow {
    fn drop(&mut self) {
        self.shared.gone.store(true, Ordering::Release);
        if let Ok(mut events) = self.events.lock() {
            events.take();
        }
        // Closed, and let go of, on the main thread, where its parts may be
        // touched — asynchronously, since the main thread's event loop may
        // already have ended, in which case the process is and the window
        // goes with it.
        if let Some(parts) = self.parts.take() {
            DispatchQueue::main().exec_async(move || {
                // SAFETY: a block on the main dispatch queue runs on the
                // main thread.
                let mtm = unsafe { MainThreadMarker::new_unchecked() };
                parts.get(mtm).window.close();
                drop(parts);
            });
        }
    }
}

/// Makes a window, its view and the layer it is drawn through.
fn create(
    mtm: MainThreadMarker,
    title: &str,
    size: NSSize,
    events: &Events,
) -> (MainThreadBound<Parts>, MetalLayer) {
    let app = NSApplication::sharedApplication(mtm);
    // A program with no bundle starts out allowed no windows in front of
    // others, nor a Dock icon; an application with a loop of its own has
    // chosen already.
    if app.activationPolicy() == NSApplicationActivationPolicy::Prohibited {
        app.setActivationPolicy(NSApplicationActivationPolicy::Regular);
    }
    let frame = NSRect::new(NSPoint::new(0.0, 0.0), size);
    let style = NSWindowStyleMask::Titled
        | NSWindowStyleMask::Closable
        | NSWindowStyleMask::Miniaturizable
        | NSWindowStyleMask::Resizable;
    // SAFETY: a fresh window with a plain style; it is kept, not released,
    // when it is closed, since this holds it until it lets go.
    let window = unsafe {
        let window = NSWindow::initWithContentRect_styleMask_backing_defer(
            NSWindow::alloc(mtm),
            frame,
            style,
            NSBackingStoreType::Buffered,
            false,
        );
        window.setReleasedWhenClosed(false);
        window
    };
    window.setTitle(&NSString::from_str(title));

    let layer = CAMetalLayer::new();
    layer.setContentsScale(window.backingScaleFactor());
    let view = VideoView::new(mtm, Arc::clone(events), frame);
    // The layer first, then asking for one, is what makes the view host
    // this layer rather than make one of its own.
    view.setLayer(Some(&layer));
    view.setWantsLayer(true);
    window.setContentView(Some(&view));

    let delegate = WindowDelegate::new(mtm, Arc::clone(events), layer.clone());
    window.setDelegate(Some(ProtocolObject::from_ref(&*delegate)));
    window.center();
    window.makeKeyAndOrderFront(None);
    window.makeFirstResponder(Some(&view));
    app.activate();
    (
        MainThreadBound::new(
            Parts {
                window,
                _delegate: delegate,
            },
            mtm,
        ),
        MetalLayer(layer),
    )
}

/// The picture area's size in pixels: its content view's, at the window's
/// scale.
fn content_pixels(window: &NSWindow) -> (u32, u32) {
    let size = window
        .contentView()
        .map(|view| view.bounds().size)
        .unwrap_or(NSSize::new(0.0, 0.0));
    let scale = window.backingScaleFactor();
    (
        (size.width * scale).round().max(0.0) as u32,
        (size.height * scale).round().max(0.0) as u32,
    )
}

/// Changes an [`OwnedWindow`] from any thread — see
/// [`crate::elements::WindowControl`].
#[derive(Clone)]
pub(crate) struct Control {
    parts: Weak<MainThreadBound<Parts>>,
    shared: Arc<Shared>,
}

impl std::fmt::Debug for Control {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Control").finish_non_exhaustive()
    }
}

impl Control {
    /// Runs `change` on the window on the main thread, later, unless it is
    /// gone by then.
    fn on_main(&self, change: impl FnOnce(&NSWindow) + Send + 'static) -> Result<(), WindowGone> {
        if self.shared.gone.load(Ordering::Acquire) {
            return Err(WindowGone);
        }
        let parts = self.parts.clone();
        DispatchQueue::main().exec_async(move || {
            // SAFETY: a block on the main dispatch queue runs on the main
            // thread.
            let mtm = unsafe { MainThreadMarker::new_unchecked() };
            if let Some(parts) = parts.upgrade() {
                change(&parts.get(mtm).window);
            }
        });
        Ok(())
    }

    pub(crate) fn set_title(&self, title: &str) -> Result<(), WindowGone> {
        let title = title.to_owned();
        self.on_main(move |window| window.setTitle(&NSString::from_str(&title)))
    }

    pub(crate) fn set_fullscreen(&self, fullscreen: bool) -> Result<(), WindowGone> {
        if self.shared.gone.load(Ordering::Acquire) {
            return Err(WindowGone);
        }
        if self.shared.fullscreen.swap(fullscreen, Ordering::AcqRel) == fullscreen {
            return Ok(());
        }
        self.on_main(move |window| {
            let now = window.styleMask().contains(NSWindowStyleMask::FullScreen);
            if now != fullscreen {
                window.toggleFullScreen(None);
            }
        })
    }

    pub(crate) fn is_fullscreen(&self) -> bool {
        self.shared.fullscreen.load(Ordering::Acquire)
    }
}

/// What the picture area's view holds.
struct ViewIvars {
    events: Events,
}

define_class!(
    /// The window's content view: hosts the Metal layer, and reports keys
    /// and clicks.
    #[unsafe(super(NSView, NSResponder, NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "MediaPpVideoView"]
    #[ivars = ViewIvars]
    struct VideoView;

    unsafe impl NSObjectProtocol for VideoView {}

    impl VideoView {
        #[unsafe(method(acceptsFirstResponder))]
        fn accepts_first_responder(&self) -> bool {
            true
        }

        /// Measured from the top left, as the events report it.
        #[unsafe(method(isFlipped))]
        fn is_flipped(&self) -> bool {
            true
        }

        #[unsafe(method(keyDown:))]
        fn key_down(&self, event: &NSEvent) {
            let characters = event
                .charactersIgnoringModifiers()
                .map(|characters| characters.to_string());
            let key = Key::from_mac(event.keyCode(), characters.as_deref());
            send(&self.ivars().events, WindowEvent::Key(key));
        }

        #[unsafe(method(mouseDown:))]
        fn mouse_down(&self, event: &NSEvent) {
            self.click(event, MouseButton::Left);
        }

        #[unsafe(method(rightMouseDown:))]
        fn right_mouse_down(&self, event: &NSEvent) {
            self.click(event, MouseButton::Right);
        }

        #[unsafe(method(otherMouseDown:))]
        fn other_mouse_down(&self, event: &NSEvent) {
            if event.buttonNumber() == 2 {
                self.click(event, MouseButton::Middle);
            }
        }
    }
);

impl VideoView {
    fn new(mtm: MainThreadMarker, events: Events, frame: NSRect) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(ViewIvars { events });
        // SAFETY: `NSView`'s designated initializer, on the freshly
        // allocated view whose ivars are set.
        unsafe { msg_send![super(this), initWithFrame: frame] }
    }

    /// A button press at `event`'s place, in pixels from the picture area's
    /// top left — a second one in quick succession as a double click.
    fn click(&self, event: &NSEvent, button: MouseButton) {
        let point = self.convertPoint_fromView(event.locationInWindow(), None);
        let scale = self
            .window()
            .map_or(1.0, |window| window.backingScaleFactor());
        let (x, y) = (
            (point.x * scale).round() as i32,
            (point.y * scale).round() as i32,
        );
        let event = if event.clickCount() == 2 {
            WindowEvent::DoubleClick { button, x, y }
        } else {
            WindowEvent::MouseDown { button, x, y }
        };
        send(&self.ivars().events, event);
    }
}

/// What the window's delegate holds.
struct DelegateIvars {
    events: Events,
    layer: Retained<CAMetalLayer>,
}

define_class!(
    /// Hides the window for a close, and reports resizing.
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "MediaPpWindowDelegate"]
    #[ivars = DelegateIvars]
    struct WindowDelegate;

    unsafe impl NSObjectProtocol for WindowDelegate {}

    unsafe impl NSWindowDelegate for WindowDelegate {
        /// Hidden rather than closed: the renderer goes on presenting into
        /// it until it is dropped, as on the other platforms.
        #[unsafe(method(windowShouldClose:))]
        fn window_should_close(&self, window: &NSWindow) -> bool {
            window.orderOut(None);
            send(&self.ivars().events, WindowEvent::Closed);
            false
        }

        #[unsafe(method(windowDidResize:))]
        fn window_did_resize(&self, notification: &NSNotification) {
            if let Some(window) = window_of(notification) {
                let (width, height) = content_pixels(&window);
                send(&self.ivars().events, WindowEvent::Resized { width, height });
            }
        }

        /// A window moved to a screen of another scale draws at that one.
        #[unsafe(method(windowDidChangeBackingProperties:))]
        fn window_did_change_backing_properties(&self, notification: &NSNotification) {
            if let Some(window) = window_of(notification) {
                self.ivars()
                    .layer
                    .setContentsScale(window.backingScaleFactor());
            }
        }
    }
);

impl WindowDelegate {
    fn new(mtm: MainThreadMarker, events: Events, layer: Retained<CAMetalLayer>) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(DelegateIvars { events, layer });
        // SAFETY: `NSObject`'s own initializer, on the freshly allocated
        // object whose ivars are set.
        unsafe { msg_send![super(this), init] }
    }
}

/// The window a window notification is about.
fn window_of(notification: &NSNotification) -> Option<Retained<NSWindow>> {
    notification
        .object()
        .and_then(|object| object.downcast::<NSWindow>().ok())
}
