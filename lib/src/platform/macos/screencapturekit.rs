//! What `ScreenCaptureKitSource` asks of ScreenCaptureKit before it
//! captures: the displays and windows there are, how many pixels each is,
//! and whether this program may record the screen at all — and waiting for
//! the answers ScreenCaptureKit gives only through completion handlers.

use std::{sync::mpsc, time::Duration};

use objc2::rc::Retained;
use objc2_core_foundation::CGRect;
use objc2_core_graphics::{
    CGDisplayCopyDisplayMode, CGDisplayMode, CGMainDisplayID, CGPreflightScreenCaptureAccess,
    CGRequestScreenCaptureAccess,
};
use objc2_foundation::NSError;
use objc2_screen_capture_kit::{SCDisplay, SCShareableContent, SCWindow};

/// How long ScreenCaptureKit gets to answer a request — for what there is to
/// capture, or to start or stop a stream. Its answers take milliseconds; one
/// that never comes must not wedge `open` or a pipeline's stop.
const ANSWER_TIMEOUT: Duration = Duration::from_secs(5);

/// One display, as [`crate::elements::ScreenCaptureKitSource::list_displays`]
/// lists it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScreenCaptureKitDisplay {
    /// Core Graphics' display ID, which stays the same for one display while
    /// it is connected, and is what a capture names it by.
    pub id: u32,
    /// Its size in pixels — what a capture of it delivers — rather than in
    /// the points the system lays windows out in, half as many each way on
    /// a Retina display.
    pub width: u32,
    /// Its height in pixels.
    pub height: u32,
    /// Whether it is the main display, the one with the menu bar.
    pub is_main: bool,
}

/// One window, as [`crate::elements::ScreenCaptureKitSource::list_windows`]
/// lists it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScreenCaptureKitWindow {
    /// Core Graphics' window ID, which a capture names it by, for as long
    /// as the window exists.
    pub id: u32,
    /// Its title, empty where it has none.
    pub title: String,
    /// The name of the application it belongs to.
    pub application: String,
    /// The process that owns it.
    pub pid: u32,
    /// Its size in pixels, on the display it is mostly on — what a capture
    /// of it delivers.
    pub width: u32,
    /// Its height in pixels.
    pub height: u32,
}

/// Why ScreenCaptureKit would not say what there is to capture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ContentError {
    /// This program has not been allowed to record the screen.
    PermissionDenied,
    /// It said something else, or nothing in time.
    Failed(String),
}

/// Whether this program may record the screen, asking the user the first
/// time — which macOS does by showing its own prompt and sending them to
/// System Settings, and answers `false` at once. A program allowed there
/// has to be started again before it may.
///
/// The permission is the responsible application's: the program's own,
/// bundled, or the terminal's that started it.
pub(crate) fn may_record() -> bool {
    CGPreflightScreenCaptureAccess() || CGRequestScreenCaptureAccess()
}

/// Waits for the one answer a completion handler sends through the channel
/// its `ask` is handed.
fn wait_for<T>(what: &str, ask: impl FnOnce(mpsc::Sender<T>)) -> Result<T, String> {
    let (tx, rx) = mpsc::channel();
    ask(tx);
    rx.recv_timeout(ANSWER_TIMEOUT)
        .map_err(|_| format!("ScreenCaptureKit did not {what} in {ANSWER_TIMEOUT:?}"))
}

/// `error`'s description, or `None` for a null one — how a completion
/// handler says it succeeded.
///
/// # Safety
///
/// `error` is null or a live `NSError`.
pub(crate) unsafe fn describe(error: *mut NSError) -> Option<String> {
    // SAFETY: the caller's.
    unsafe { error.as_ref() }.map(|error| error.localizedDescription().to_string())
}

/// Runs `call`, handing it a completion handler that takes an optional
/// error, and waits for that handler — how a stream is started and stopped.
pub(crate) fn complete(
    what: &str,
    call: impl FnOnce(&block2::DynBlock<dyn Fn(*mut NSError)>),
) -> Result<(), String> {
    wait_for(what, |tx| {
        let handler = block2::RcBlock::new(move |error: *mut NSError| {
            // SAFETY: ScreenCaptureKit hands over null or a live error.
            let _ = tx.send(unsafe { describe(error) });
        });
        call(&handler);
    })?
    .map_or(Ok(()), Err)
}

/// Every display and window there is to capture, on this Space or
/// another — an application in full screen is on a Space of its own.
pub(crate) fn shareable_content() -> Result<Retained<SCShareableContent>, ContentError> {
    if !CGPreflightScreenCaptureAccess() {
        return Err(ContentError::PermissionDenied);
    }
    let answer = wait_for("say what there is to capture", |tx| {
        let handler = block2::RcBlock::new(
            move |content: *mut SCShareableContent, error: *mut NSError| {
                // SAFETY: ScreenCaptureKit hands over a live content object
                // or a live error, whichever it has; the content is retained
                // for as long as this keeps it.
                let answer = unsafe {
                    match Retained::retain(content) {
                        Some(content) => Ok(content),
                        None => Err(describe(error).unwrap_or_else(|| "no answer".into())),
                    }
                };
                let _ = tx.send(answer);
            },
        );
        // SAFETY: a class method taking a completion handler, which it
        // copies and calls once.
        unsafe {
            SCShareableContent::getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler(
                true, false, &handler,
            );
        }
    })
    .map_err(ContentError::Failed)?;
    answer.map_err(ContentError::Failed)
}

/// `display`'s size in pixels: its current mode's, which is what
/// ScreenCaptureKit captures at best, and its size in points where there is
/// no mode to read.
fn display_pixels(display: &SCDisplay) -> (u32, u32) {
    // SAFETY: plain queries of a live display object and its mode.
    unsafe {
        let id = display.displayID();
        let mode = CGDisplayCopyDisplayMode(id);
        let pixels = mode.as_deref().map(|mode| {
            (
                CGDisplayMode::pixel_width(Some(mode)),
                CGDisplayMode::pixel_height(Some(mode)),
            )
        });
        let (width, height) = match pixels {
            Some((width, height)) if width > 0 && height > 0 => (width, height),
            _ => (display.width() as usize, display.height() as usize),
        };
        (width as u32, height as u32)
    }
}

/// How many pixels a point is on `display`: 2 on a Retina display, 1 on
/// most others.
fn display_scale(display: &SCDisplay) -> f64 {
    let (width, _) = display_pixels(display);
    // SAFETY: a plain query of a live display object.
    let points = unsafe { display.frame() }.size.width;
    if points > 0.0 {
        f64::from(width) / points
    } else {
        1.0
    }
}

pub(crate) fn describe_display(display: &SCDisplay) -> ScreenCaptureKitDisplay {
    let (width, height) = display_pixels(display);
    // SAFETY: a plain query of a live display object.
    let id = unsafe { display.displayID() };
    ScreenCaptureKitDisplay {
        id,
        width,
        height,
        is_main: id == CGMainDisplayID(),
    }
}

/// The display the middle of `frame` is on, of `content`'s, or the main one.
fn display_of(content: &SCShareableContent, frame: CGRect) -> Option<Retained<SCDisplay>> {
    let x = frame.origin.x + frame.size.width / 2.0;
    let y = frame.origin.y + frame.size.height / 2.0;
    // SAFETY: plain queries of live display objects.
    let displays = unsafe { content.displays() };
    let on = displays.iter().find(|display| {
        // SAFETY: as above.
        let bounds = unsafe { display.frame() };
        (bounds.origin.x..bounds.origin.x + bounds.size.width).contains(&x)
            && (bounds.origin.y..bounds.origin.y + bounds.size.height).contains(&y)
    });
    on.or_else(|| {
        displays
            .iter()
            // SAFETY: as above.
            .find(|display| unsafe { display.displayID() } == CGMainDisplayID())
    })
}

/// A window's size in pixels: its frame, in points, at the scale of the
/// display it is mostly on.
pub(crate) fn window_pixels(content: &SCShareableContent, window: &SCWindow) -> (u32, u32) {
    // SAFETY: a plain query of a live window object.
    let frame = unsafe { window.frame() };
    let scale = display_of(content, frame).map_or(1.0, |display| display_scale(&display));
    (
        (frame.size.width * scale).round().max(0.0) as u32,
        (frame.size.height * scale).round().max(0.0) as u32,
    )
}

pub(crate) fn describe_window(
    content: &SCShareableContent,
    window: &SCWindow,
) -> ScreenCaptureKitWindow {
    let (width, height) = window_pixels(content, window);
    // SAFETY: plain queries of a live window object and its application.
    unsafe {
        let application = window.owningApplication();
        ScreenCaptureKitWindow {
            id: window.windowID(),
            title: window
                .title()
                .map(|title| title.to_string())
                .unwrap_or_default(),
            application: application
                .as_ref()
                .map(|application| application.applicationName().to_string())
                .unwrap_or_default(),
            pid: application
                .as_ref()
                .map_or(0, |application| application.processID() as u32),
            width,
            height,
        }
    }
}

/// The windows worth offering: an application's, at the normal level —
/// not the menu bar, the Dock or a status item — bigger than a pixel, and,
/// where it is not on screen, with a title: an application keeps windows
/// it never shows, and those have none.
pub(crate) fn capturable_windows(content: &SCShareableContent) -> Vec<Retained<SCWindow>> {
    // SAFETY: plain queries of live window objects.
    unsafe {
        content
            .windows()
            .iter()
            .filter(|window| {
                let frame = window.frame();
                window.windowLayer() == 0
                    && window.owningApplication().is_some()
                    && frame.size.width >= 1.0
                    && frame.size.height >= 1.0
                    && (window.isOnScreen()
                        || window.title().is_some_and(|title| title.length() > 0))
            })
            .collect()
    }
}

/// The display `id` names, where it is still there.
pub(crate) fn find_display(content: &SCShareableContent, id: u32) -> Option<Retained<SCDisplay>> {
    // SAFETY: plain queries of live display objects.
    unsafe { content.displays() }
        .iter()
        // SAFETY: as above.
        .find(|display| unsafe { display.displayID() } == id)
}

/// The window `id` names, where it is still there.
pub(crate) fn find_window(content: &SCShareableContent, id: u32) -> Option<Retained<SCWindow>> {
    // SAFETY: plain queries of live window objects.
    unsafe { content.windows() }
        .iter()
        // SAFETY: as above.
        .find(|window| unsafe { window.windowID() } == id)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whatever there is to capture, each display is listed once with a
    /// size in pixels at least its size in points, and exactly one is the
    /// main one; each window offered is found again by its id. Skips where
    /// this program may not record the screen, and asks nobody.
    #[test]
    fn listed_displays_and_windows_can_be_found_again() {
        let content = match shareable_content() {
            Ok(content) => content,
            Err(ContentError::PermissionDenied) => {
                eprintln!("skipping: this program has not been allowed to record the screen");
                return;
            }
            Err(ContentError::Failed(reason)) => panic!("{reason}"),
        };
        // SAFETY: plain queries of live display objects.
        let displays = unsafe { content.displays() };
        assert!(!displays.is_empty(), "a Mac has a display");
        assert_eq!(
            displays
                .iter()
                .filter(|display| describe_display(display).is_main)
                .count(),
            1
        );
        for display in displays.iter() {
            let listed = describe_display(&display);
            // SAFETY: as above.
            let points = unsafe { (display.width(), display.height()) };
            assert!(
                i64::from(listed.width) >= points.0 as i64
                    && i64::from(listed.height) >= points.1 as i64,
                "{listed:?} against {points:?} points"
            );
            assert!(find_display(&content, listed.id).is_some());
        }
        for window in capturable_windows(&content) {
            let listed = describe_window(&content, &window);
            assert!(listed.width >= 1 && listed.height >= 1, "{listed:?}");
            assert!(find_window(&content, listed.id).is_some());
        }
    }
}
