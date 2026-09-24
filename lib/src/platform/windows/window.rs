//! A window a renderer opens for itself, on a thread of its own — the way a
//! GStreamer video sink does when nobody hands it one.
//!
//! Not `winit`: its event loop is one per process and would collide with an
//! application's own, where each of these is a plain Win32 window with its
//! own message loop, so there can be as many as there are renderers.

use std::{
    ffi::c_void,
    sync::{Once, mpsc},
    thread::{self, JoinHandle},
};

use crossbeam_channel::Sender;
use windows::{
    Win32::{
        Foundation::{HWND, LPARAM, LRESULT, RECT, WPARAM},
        Graphics::Gdi::ValidateRect,
        System::LibraryLoader::GetModuleHandleW,
        UI::WindowsAndMessaging::{
            AdjustWindowRectEx, CREATESTRUCTW, CS_HREDRAW, CS_VREDRAW, CW_USEDEFAULT,
            CreateWindowExW, DefWindowProcW, DestroyWindow, DispatchMessageW, GWLP_USERDATA,
            GetMessageW, GetWindowLongPtrW, IDC_ARROW, LoadCursorW, MSG, PostMessageW,
            PostQuitMessage, RegisterClassExW, SW_HIDE, SW_SHOW, SetWindowLongPtrW, ShowWindow,
            TranslateMessage, WINDOW_EX_STYLE, WM_APP, WM_CLOSE, WM_DESTROY, WM_KEYDOWN,
            WM_NCCREATE, WM_PAINT, WM_SIZE, WNDCLASSEXW, WS_OVERLAPPEDWINDOW,
        },
    },
    core::{HSTRING, PCWSTR, w},
};

use crate::elements::{Key, WindowEvent, WindowOptions};

/// Asks the window's own thread to destroy it — `DestroyWindow` only works
/// from the thread that created the window.
const WM_APP_DESTROY: u32 = WM_APP + 1;

const CLASS_NAME: PCWSTR = w!("media-pp.window");

/// A window on its own thread, destroyed and joined when this drops.
pub(crate) struct OwnedWindow {
    hwnd: isize,
    thread: Option<JoinHandle<()>>,
}

impl OwnedWindow {
    /// Opens a window with a client area of `options`' size and returns once
    /// it exists. What happens to it — keys, resizing, a request to close —
    /// goes to `events`.
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
}

impl Drop for OwnedWindow {
    fn drop(&mut self) {
        // SAFETY: posting to a window this thread may not own is what
        // `PostMessageW` is for; if the window is already gone it fails, and
        // the thread has ended on its own.
        let _ = unsafe { PostMessageW(Some(self.hwnd()), WM_APP_DESTROY, WPARAM(0), LPARAM(0)) };
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
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
    let events = Box::new(events);
    let hwnd = match create(&title, width, height, &events) {
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
    drop(events);
}

fn create(
    title: &HSTRING,
    width: u32,
    height: u32,
    events: &Sender<WindowEvent>,
) -> windows::core::Result<HWND> {
    static REGISTER: Once = Once::new();
    // SAFETY: plain Win32 calls with locals; the class names a `'static`
    // procedure, and `events` outlives the window (see `run`).
    unsafe {
        let instance = GetModuleHandleW(None)?;
        REGISTER.call_once(|| {
            let class = WNDCLASSEXW {
                cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
                style: CS_HREDRAW | CS_VREDRAW,
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
            Some(std::ptr::from_ref(events).cast()),
        )?;
        let _ = ShowWindow(hwnd, SW_SHOW);
        Ok(hwnd)
    }
}

unsafe extern "system" fn window_procedure(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // SAFETY: `GWLP_USERDATA` holds the `Sender` `run` keeps alive until the
    // loop has ended, set from `CREATESTRUCTW::lpCreateParams` on the first
    // message that carries it.
    unsafe {
        if message == WM_NCCREATE {
            let create = &*(lparam.0 as *const CREATESTRUCTW);
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, create.lpCreateParams as isize);
        }
        let events =
            (GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *const Sender<WindowEvent>).as_ref();
        let send = |event| {
            if let Some(events) = events {
                let _ = events.send(event);
            }
        };
        match message {
            WM_KEYDOWN => {
                send(WindowEvent::Key(Key::from_virtual_key(wparam.0 as u32)));
                LRESULT(0)
            }
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

    use windows::Win32::UI::WindowsAndMessaging::IsWindowVisible;

    use super::*;

    /// Keys and a request to close come out as events; closing hides the
    /// window rather than destroying it; dropping destroys it, ends its
    /// thread, and with it the event channel.
    #[test]
    fn a_window_reports_keys_and_closing_and_goes_when_dropped() {
        let (events_tx, events) = crossbeam_channel::unbounded();
        let options = WindowOptions {
            title: "media-pp test window".into(),
            width: 160,
            height: 120,
        };
        let window = match OwnedWindow::open(&options, events_tx) {
            Ok(window) => window,
            Err(error) => {
                eprintln!("skipping: no window can be opened here ({error})");
                return;
            }
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
}
