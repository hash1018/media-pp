//! A window a renderer opens for itself, on a thread of its own — the way a
//! GStreamer video sink does when nobody hands it one.
//!
//! Not `winit`: its event loop is one per process and would collide with an
//! application's own, where each of these is a plain Win32 window with its
//! own message loop, so there can be as many as there are renderers.

use std::{
    cell::Cell,
    ffi::c_void,
    sync::{
        Arc, Once,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
};

use crossbeam_channel::Sender;
use windows::{
    Win32::{
        Foundation::{HWND, LPARAM, LRESULT, RECT, WPARAM},
        Graphics::Gdi::{
            GetMonitorInfoW, MONITOR_DEFAULTTONEAREST, MONITORINFO, MonitorFromWindow, ValidateRect,
        },
        System::LibraryLoader::GetModuleHandleW,
        UI::WindowsAndMessaging::{
            AdjustWindowRectEx, CREATESTRUCTW, CS_DBLCLKS, CS_HREDRAW, CS_VREDRAW, CW_USEDEFAULT,
            CreateWindowExW, DefWindowProcW, DestroyWindow, DispatchMessageW, GWL_STYLE,
            GWLP_USERDATA, GetMessageW, GetWindowLongPtrW, GetWindowPlacement, HWND_TOP, IDC_ARROW,
            LoadCursorW, MSG, PostMessageW, PostQuitMessage, RegisterClassExW, SW_HIDE, SW_SHOW,
            SWP_FRAMECHANGED, SWP_NOMOVE, SWP_NOOWNERZORDER, SWP_NOSIZE, SWP_NOZORDER,
            SetWindowLongPtrW, SetWindowPlacement, SetWindowPos, SetWindowTextW, ShowWindow,
            TranslateMessage, WINDOW_EX_STYLE, WINDOWPLACEMENT, WM_APP, WM_CLOSE, WM_DESTROY,
            WM_KEYDOWN, WM_LBUTTONDBLCLK, WM_LBUTTONDOWN, WM_MBUTTONDBLCLK, WM_MBUTTONDOWN,
            WM_NCCREATE, WM_PAINT, WM_RBUTTONDBLCLK, WM_RBUTTONDOWN, WM_SIZE, WNDCLASSEXW,
            WS_OVERLAPPEDWINDOW,
        },
    },
    core::{HSTRING, PCWSTR, w},
};

use crate::elements::{Key, MouseButton, WindowEvent, WindowGone, WindowOptions};

/// Asks the window's own thread to destroy it — `DestroyWindow` only works
/// from the thread that created the window.
const WM_APP_DESTROY: u32 = WM_APP + 1;

/// Asks the window's own thread to fill the screen with it (`WPARAM` 1) or
/// put it back (0) — done there, where its saved placement lives.
const WM_APP_FULLSCREEN: u32 = WM_APP + 2;

const CLASS_NAME: PCWSTR = w!("media-pp.window");

/// A window on its own thread, destroyed and joined when this drops.
pub(crate) struct OwnedWindow {
    hwnd: isize,
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
}

/// What a window and the [`Control`]s of it share.
#[derive(Debug, Default)]
struct Shared {
    /// Whether it was last asked to fill the screen.
    fullscreen: AtomicBool,
    /// Set before the window is destroyed, so no control reaches a handle
    /// Windows may since have given another window.
    gone: AtomicBool,
}

impl OwnedWindow {
    /// Opens a window with a client area of `options`' size and returns once
    /// it exists. What happens to it — keys, clicks, resizing, a request to
    /// close — goes to `events`.
    pub(crate) fn open(
        options: &WindowOptions,
        events: Sender<WindowEvent>,
    ) -> windows::core::Result<Self> {
        let (created_tx, created_rx) = mpsc::channel();
        let title = HSTRING::from(options.title.as_str());
        let (width, height) = (options.width, options.height);
        let thread = thread::Builder::new()
            .name(format!("window:{}", options.title))
            .spawn(move || run(title, width, height, events, created_tx))
            .map_err(|error| {
                windows::core::Error::new(windows::Win32::Foundation::E_FAIL, error.to_string())
            })?;
        match created_rx.recv() {
            Ok(Ok(hwnd)) => Ok(Self {
                hwnd,
                shared: Arc::default(),
                thread: Some(thread),
            }),
            Ok(Err(error)) => {
                let _ = thread.join();
                Err(error)
            }
            Err(_) => {
                let _ = thread.join();
                Err(windows::core::Error::new(
                    windows::Win32::Foundation::E_FAIL,
                    "the window thread ended before creating its window",
                ))
            }
        }
    }

    pub(crate) fn hwnd(&self) -> HWND {
        HWND(self.hwnd as *mut c_void)
    }

    /// What changes the window's title and fullscreen state from another
    /// thread.
    pub(crate) fn control(&self) -> Control {
        Control {
            hwnd: self.hwnd,
            shared: Arc::clone(&self.shared),
        }
    }
}

impl Drop for OwnedWindow {
    fn drop(&mut self) {
        self.shared.gone.store(true, Ordering::Release);
        // SAFETY: posting to a window this thread may not own is what
        // `PostMessageW` is for; if the window is already gone it fails, and
        // the thread has ended on its own.
        let _ = unsafe { PostMessageW(Some(self.hwnd()), WM_APP_DESTROY, WPARAM(0), LPARAM(0)) };
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Changes an [`OwnedWindow`] from any thread — see
/// [`crate::elements::WindowControl`].
#[derive(Debug, Clone)]
pub(crate) struct Control {
    hwnd: isize,
    shared: Arc<Shared>,
}

impl Control {
    fn hwnd(&self) -> Result<HWND, WindowGone> {
        if self.shared.gone.load(Ordering::Acquire) {
            return Err(WindowGone);
        }
        Ok(HWND(self.hwnd as *mut c_void))
    }

    pub(crate) fn set_title(&self, title: &str) -> Result<(), WindowGone> {
        let hwnd = self.hwnd()?;
        // SAFETY: a window this control's window was, not yet destroyed —
        // `gone` is set before it is. From another thread this sends the
        // window thread `WM_SETTEXT`, which its message loop answers.
        unsafe { SetWindowTextW(hwnd, &HSTRING::from(title)) }.map_err(|_| WindowGone)
    }

    pub(crate) fn set_fullscreen(&self, fullscreen: bool) -> Result<(), WindowGone> {
        let hwnd = self.hwnd()?;
        if self.shared.fullscreen.swap(fullscreen, Ordering::AcqRel) == fullscreen {
            return Ok(());
        }
        // SAFETY: as for `set_title`; the window thread does the rest.
        unsafe {
            PostMessageW(
                Some(hwnd),
                WM_APP_FULLSCREEN,
                WPARAM(usize::from(fullscreen)),
                LPARAM(0),
            )
        }
        .map_err(|_| WindowGone)
    }

    pub(crate) fn is_fullscreen(&self) -> bool {
        self.shared.fullscreen.load(Ordering::Acquire)
    }
}

/// What the window procedure reaches through `GWLP_USERDATA`, owned by the
/// window thread for the window's whole life.
struct ThreadState {
    events: Sender<WindowEvent>,
    /// The style and placement to put back when it leaves the full screen;
    /// `Some` while it fills it.
    restore: Cell<Option<Restore>>,
}

#[derive(Clone, Copy)]
struct Restore {
    style: isize,
    placement: WINDOWPLACEMENT,
}

/// The window thread: create, report, pump until destroyed.
fn run(
    title: HSTRING,
    width: u32,
    height: u32,
    events: Sender<WindowEvent>,
    created: mpsc::Sender<windows::core::Result<isize>>,
) {
    // Owned by this thread for the window's whole life; the window procedure
    // borrows it through `GWLP_USERDATA`.
    let state = Box::new(ThreadState {
        events,
        restore: Cell::new(None),
    });
    let hwnd = match create(&title, width, height, &state) {
        Ok(hwnd) => hwnd,
        Err(error) => {
            let _ = created.send(Err(error));
            return;
        }
    };
    let _ = created.send(Ok(hwnd.0 as isize));
    let mut message = MSG::default();
    // SAFETY: the standard message loop for windows this thread created.
    unsafe {
        while GetMessageW(&mut message, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
    drop(state);
}

fn create(
    title: &HSTRING,
    width: u32,
    height: u32,
    state: &ThreadState,
) -> windows::core::Result<HWND> {
    static REGISTER: Once = Once::new();
    // SAFETY: plain Win32 calls with locals; the class names a `'static`
    // procedure, and `state` outlives the window (see `run`).
    unsafe {
        let instance = GetModuleHandleW(None)?;
        REGISTER.call_once(|| {
            let class = WNDCLASSEXW {
                cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
                // Double clicks reported as such, rather than as two presses.
                style: CS_HREDRAW | CS_VREDRAW | CS_DBLCLKS,
                lpfnWndProc: Some(window_procedure),
                hInstance: instance.into(),
                hCursor: LoadCursorW(None, IDC_ARROW).unwrap_or_default(),
                lpszClassName: CLASS_NAME,
                ..Default::default()
            };
            RegisterClassExW(&class);
        });
        // The size asked for is the picture's, so the frame goes around it.
        let mut frame = RECT {
            left: 0,
            top: 0,
            right: width as i32,
            bottom: height as i32,
        };
        AdjustWindowRectEx(&mut frame, WS_OVERLAPPEDWINDOW, false, WINDOW_EX_STYLE(0))?;
        let hwnd = CreateWindowExW(
            WINDOW_EX_STYLE(0),
            CLASS_NAME,
            title,
            WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            frame.right - frame.left,
            frame.bottom - frame.top,
            None,
            None,
            Some(instance.into()),
            Some(std::ptr::from_ref(state).cast()),
        )?;
        let _ = ShowWindow(hwnd, SW_SHOW);
        Ok(hwnd)
    }
}

/// Fills the monitor the window is on with it, without its frame, keeping
/// what to put back. Raymond Chen's recipe: the style and placement saved,
/// the frame styles taken off, the window moved over the monitor.
///
/// # Safety
///
/// On the window's own thread.
unsafe fn enter_fullscreen(hwnd: HWND, state: &ThreadState) {
    if state.restore.get().is_some() {
        return;
    }
    // SAFETY: the caller's promise; every out-pointer is a local.
    unsafe {
        let style = GetWindowLongPtrW(hwnd, GWL_STYLE);
        let mut placement = WINDOWPLACEMENT {
            length: std::mem::size_of::<WINDOWPLACEMENT>() as u32,
            ..Default::default()
        };
        let mut monitor = MONITORINFO {
            cbSize: std::mem::size_of::<MONITORINFO>() as u32,
            ..Default::default()
        };
        if GetWindowPlacement(hwnd, &mut placement).is_err()
            || !GetMonitorInfoW(
                MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST),
                &mut monitor,
            )
            .as_bool()
        {
            return;
        }
        SetWindowLongPtrW(hwnd, GWL_STYLE, style & !(WS_OVERLAPPEDWINDOW.0 as isize));
        let screen = monitor.rcMonitor;
        let _ = SetWindowPos(
            hwnd,
            Some(HWND_TOP),
            screen.left,
            screen.top,
            screen.right - screen.left,
            screen.bottom - screen.top,
            SWP_NOOWNERZORDER | SWP_FRAMECHANGED,
        );
        state.restore.set(Some(Restore { style, placement }));
    }
}

/// Puts the window back as it was before [`enter_fullscreen`].
///
/// # Safety
///
/// On the window's own thread.
unsafe fn leave_fullscreen(hwnd: HWND, state: &ThreadState) {
    let Some(restore) = state.restore.take() else {
        return;
    };
    // SAFETY: the caller's promise; the placement is the one saved.
    unsafe {
        SetWindowLongPtrW(hwnd, GWL_STYLE, restore.style);
        let _ = SetWindowPlacement(hwnd, &restore.placement);
        let _ = SetWindowPos(
            hwnd,
            None,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED,
        );
    }
}

/// The client coordinates a mouse message carries: signed, since a press
/// held while the pointer leaves the window reports it outside.
fn mouse_position(lparam: LPARAM) -> (i32, i32) {
    let x = (lparam.0 & 0xFFFF) as u16 as i16;
    let y = ((lparam.0 >> 16) & 0xFFFF) as u16 as i16;
    (i32::from(x), i32::from(y))
}

unsafe extern "system" fn window_procedure(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // SAFETY: `GWLP_USERDATA` holds the `ThreadState` `run` keeps alive until
    // the loop has ended, set from `CREATESTRUCTW::lpCreateParams` on the
    // first message that carries it.
    unsafe {
        if message == WM_NCCREATE {
            let create = &*(lparam.0 as *const CREATESTRUCTW);
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, create.lpCreateParams as isize);
        }
        let state = (GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *const ThreadState).as_ref();
        let send = |event| {
            if let Some(state) = state {
                let _ = state.events.send(event);
            }
        };
        let mouse = |button, double| {
            let (x, y) = mouse_position(lparam);
            send(if double {
                WindowEvent::DoubleClick { button, x, y }
            } else {
                WindowEvent::MouseDown { button, x, y }
            });
            LRESULT(0)
        };
        match message {
            WM_KEYDOWN => {
                send(WindowEvent::Key(Key::from_virtual_key(wparam.0 as u32)));
                LRESULT(0)
            }
            WM_LBUTTONDOWN => mouse(MouseButton::Left, false),
            WM_LBUTTONDBLCLK => mouse(MouseButton::Left, true),
            WM_RBUTTONDOWN => mouse(MouseButton::Right, false),
            WM_RBUTTONDBLCLK => mouse(MouseButton::Right, true),
            WM_MBUTTONDOWN => mouse(MouseButton::Middle, false),
            WM_MBUTTONDBLCLK => mouse(MouseButton::Middle, true),
            WM_SIZE => {
                let (width, height) = (lparam.0 as u32 & 0xFFFF, (lparam.0 as u32 >> 16) & 0xFFFF);
                send(WindowEvent::Resized { width, height });
                LRESULT(0)
            }
            // Hidden rather than destroyed: the renderer still presents into
            // it, and what closing means — stop, pause, quit — is the
            // application's to decide from `Closed`.
            WM_CLOSE => {
                // Hidden before it is reported, so whoever acts on `Closed`
                // finds the window already out of sight.
                let _ = ShowWindow(hwnd, SW_HIDE);
                send(WindowEvent::Closed);
                LRESULT(0)
            }
            // The swap chain paints it; there is nothing to draw here, only
            // the region to validate so Windows stops asking.
            WM_PAINT => {
                let _ = ValidateRect(Some(hwnd), None);
                LRESULT(0)
            }
            WM_APP_FULLSCREEN => {
                if let Some(state) = state {
                    if wparam.0 != 0 {
                        enter_fullscreen(hwnd, state);
                    } else {
                        leave_fullscreen(hwnd, state);
                    }
                }
                LRESULT(0)
            }
            WM_APP_DESTROY => {
                let _ = DestroyWindow(hwnd);
                LRESULT(0)
            }
            WM_DESTROY => {
                SetWindowLongPtrW(hwnd, GWLP_USERDATA, 0);
                PostQuitMessage(0);
                LRESULT(0)
            }
            _ => DefWindowProcW(hwnd, message, wparam, lparam),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use windows::Win32::UI::WindowsAndMessaging::{GetWindowRect, GetWindowTextW, IsWindowVisible};

    use super::*;

    fn test_window(title: &str) -> Option<(OwnedWindow, crossbeam_channel::Receiver<WindowEvent>)> {
        let (events_tx, events) = crossbeam_channel::unbounded();
        let options = WindowOptions {
            title: title.into(),
            width: 160,
            height: 120,
        };
        match OwnedWindow::open(&options, events_tx) {
            Ok(window) => Some((window, events)),
            Err(error) => {
                eprintln!("skipping: no window can be opened here ({error})");
                None
            }
        }
    }

    /// Keys and a request to close come out as events; closing hides the
    /// window rather than destroying it; dropping destroys it, ends its
    /// thread, and with it the event channel.
    #[test]
    fn a_window_reports_keys_and_closing_and_goes_when_dropped() {
        let Some((window, events)) = test_window("media-pp test window") else {
            return;
        };
        // SAFETY: posting to a live window.
        unsafe {
            PostMessageW(Some(window.hwnd()), WM_KEYDOWN, WPARAM(0x20), LPARAM(0)).unwrap();
            PostMessageW(Some(window.hwnd()), WM_CLOSE, WPARAM(0), LPARAM(0)).unwrap();
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut seen = Vec::new();
        while !seen.contains(&WindowEvent::Closed) && Instant::now() < deadline {
            if let Ok(event) = events.recv_timeout(Duration::from_millis(100)) {
                seen.push(event);
            }
        }
        assert!(seen.contains(&WindowEvent::Key(Key::Space)), "{seen:?}");
        assert!(seen.contains(&WindowEvent::Closed), "{seen:?}");
        // SAFETY: reads a live window's visibility.
        let visible = unsafe { IsWindowVisible(window.hwnd()) }.as_bool();
        assert!(!visible, "closed means hidden");

        drop(window);
        assert!(
            events.recv_timeout(Duration::from_secs(5)).is_err(),
            "the window's thread is gone, and its sender with it"
        );
        assert!(events.recv().is_err());
    }

    /// A press and a double click come out with their button and where in
    /// the picture area they were — negative too, outside it.
    #[test]
    fn clicks_come_out_with_their_button_and_place() {
        let Some((window, events)) = test_window("media-pp click test") else {
            return;
        };
        let at = |x: i16, y: i16| LPARAM(((y as u16 as isize) << 16) | x as u16 as isize);
        // SAFETY: posting to a live window.
        unsafe {
            PostMessageW(Some(window.hwnd()), WM_LBUTTONDOWN, WPARAM(0), at(10, 20)).unwrap();
            PostMessageW(Some(window.hwnd()), WM_RBUTTONDBLCLK, WPARAM(0), at(-3, 7)).unwrap();
        }
        let mut seen = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(5);
        while seen.len() < 2 && Instant::now() < deadline {
            if let Ok(event) = events.recv_timeout(Duration::from_millis(100))
                && matches!(
                    event,
                    WindowEvent::MouseDown { .. } | WindowEvent::DoubleClick { .. }
                )
            {
                seen.push(event);
            }
        }
        assert_eq!(
            seen,
            [
                WindowEvent::MouseDown {
                    button: MouseButton::Left,
                    x: 10,
                    y: 20
                },
                WindowEvent::DoubleClick {
                    button: MouseButton::Right,
                    x: -3,
                    y: 7
                },
            ]
        );
    }

    fn rect(hwnd: HWND) -> RECT {
        let mut rect = RECT::default();
        // SAFETY: reads a live window's rectangle into a local.
        unsafe { GetWindowRect(hwnd, &mut rect) }.unwrap();
        rect
    }

    fn wait_for(what: impl Fn() -> bool) -> bool {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !what() {
            if Instant::now() > deadline {
                return false;
            }
            thread::sleep(Duration::from_millis(10));
        }
        true
    }

    /// A control changes the title, fills the monitor with the window and
    /// puts it back where it was, and once the window is gone says so.
    #[test]
    fn a_control_changes_the_title_and_the_full_screen() {
        let Some((window, _events)) = test_window("media-pp control test") else {
            return;
        };
        let control = window.control();
        let hwnd = window.hwnd();

        control.set_title("renamed").unwrap();
        let mut text = [0u16; 32];
        // SAFETY: reads a live window's title into a local.
        let length = unsafe { GetWindowTextW(hwnd, &mut text) } as usize;
        assert_eq!(String::from_utf16_lossy(&text[..length]), "renamed");

        let before = rect(hwnd);
        let mut monitor = MONITORINFO {
            cbSize: std::mem::size_of::<MONITORINFO>() as u32,
            ..Default::default()
        };
        // SAFETY: reads the window's monitor into a local.
        unsafe {
            assert!(
                GetMonitorInfoW(
                    MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST),
                    &mut monitor
                )
                .as_bool()
            );
        }
        control.set_fullscreen(true).unwrap();
        assert!(control.is_fullscreen());
        assert!(
            wait_for(|| rect(hwnd) == monitor.rcMonitor),
            "fills the monitor: {:?} is not {:?}",
            rect(hwnd),
            monitor.rcMonitor
        );
        control.clone().set_fullscreen(false).unwrap();
        assert!(!control.is_fullscreen(), "a clone shares the state");
        assert!(
            wait_for(|| rect(hwnd) == before),
            "put back: {:?} is not {:?}",
            rect(hwnd),
            before
        );

        drop(window);
        assert_eq!(control.set_title("gone"), Err(WindowGone));
        assert_eq!(control.set_fullscreen(true), Err(WindowGone));
    }
}
