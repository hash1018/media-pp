//! A window a renderer opens for itself, on a thread of its own — the way a
//! GStreamer video sink does when nobody hands it one. The Linux half of
//! `platform::windows::window`.
//!
//! Plain X11 through libxcb rather than `winit`: `winit`'s event loop is one
//! per process and would collide with an application's own, where each of
//! these is a connection and a thread of its own, so there can be as many as
//! there are renderers. X11 rather than Wayland because a Wayland client
//! draws its own title bar and borders — GNOME's compositor draws none — and
//! a renderer has no business drawing window decorations; on a Wayland
//! desktop this is an XWayland window, and looks like any other.

use std::{
    ffi::c_void,
    num::NonZeroU32,
    ptr::NonNull,
    sync::Arc,
    thread::{self, JoinHandle},
};

use crossbeam_channel::Sender;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle, XcbDisplayHandle, XcbWindowHandle};
use x11rb::{
    connection::Connection,
    protocol::{
        Event,
        xproto::{
            AtomEnum, ConfigureNotifyEvent, ConnectionExt as _, CreateWindowAux, EventMask,
            PropMode, WindowClass,
        },
    },
    wrapper::ConnectionExt as _,
    xcb_ffi::XCBConnection,
};

use crate::elements::{Key, WindowEvent, WindowOptions};

/// A window on its own connection and thread, destroyed and joined when
/// this drops.
pub(crate) struct OwnedWindow {
    connection: Arc<XCBConnection>,
    screen: usize,
    window: u32,
    thread: Option<JoinHandle<()>>,
}

impl OwnedWindow {
    /// Opens a window of `options`' size and returns once it exists and is
    /// mapped. What happens to it — keys, resizing, a request to close —
    /// goes to `events`.
    pub(crate) fn open(
        options: &WindowOptions,
        events: Sender<WindowEvent>,
    ) -> Result<Self, String> {
        let (connection, screen) =
            XCBConnection::connect(None).map_err(|error| format!("no X display: {error}"))?;
        let connection = Arc::new(connection);
        let (window, wm_delete_window) =
            create(&connection, screen, options).map_err(|error| error.to_string())?;
        let keymap = Keymap::read(&connection).map_err(|error| error.to_string())?;

        // The thread only reads events; nothing about the window's creation
        // is left for it, so a failure above never has a thread to join.
        let thread = {
            let connection = Arc::clone(&connection);
            let window = Watched {
                window,
                wm_delete_window,
                size: [options.width, options.height],
            };
            thread::Builder::new()
                .name(format!("window:{}", options.title))
                .spawn(move || run(&connection, window, keymap, &events))
                .map_err(|error| error.to_string())?
        };
        Ok(Self {
            connection,
            screen,
            window,
            thread: Some(thread),
        })
    }

    /// The window's X11 id, for a test to send it what a window manager
    /// would.
    #[cfg(test)]
    pub(crate) fn id(&self) -> u32 {
        self.window
    }

    /// What Vulkan makes a surface from: the XCB connection and the window.
    pub(crate) fn handles(&self) -> (RawDisplayHandle, RawWindowHandle) {
        let display = XcbDisplayHandle::new(
            NonNull::new(self.connection.get_raw_xcb_connection().cast::<c_void>()),
            self.screen as i32,
        );
        let window = XcbWindowHandle::new(NonZeroU32::new(self.window).expect("X11 ids are not 0"));
        (display.into(), window.into())
    }
}

impl Drop for OwnedWindow {
    fn drop(&mut self) {
        // The thread ends on the `DestroyNotify` this brings, or on the
        // connection failing; either way there is nothing to wait for but it.
        let _ = self.connection.destroy_window(self.window);
        let _ = self.connection.flush();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// What the window thread watches for.
struct Watched {
    window: u32,
    /// The atom a close button's request carries.
    wm_delete_window: u32,
    /// The size last reported, so a move is not reported as a resize.
    size: [u32; 2],
}

/// Creates and maps the window, and says which atom a request to close it
/// will carry.
fn create(
    connection: &XCBConnection,
    screen: usize,
    options: &WindowOptions,
) -> Result<(u32, u32), Box<dyn std::error::Error>> {
    let root = &connection.setup().roots[screen];
    let window = connection.generate_id()?;
    connection.create_window(
        x11rb::COPY_DEPTH_FROM_PARENT,
        window,
        root.root,
        0,
        0,
        options.width.try_into()?,
        options.height.try_into()?,
        0,
        WindowClass::INPUT_OUTPUT,
        root.root_visual,
        &CreateWindowAux::new()
            // Black until the first frame, and in the bars around it.
            .background_pixel(root.black_pixel)
            .event_mask(EventMask::KEY_PRESS | EventMask::STRUCTURE_NOTIFY),
    )?;
    let wm_delete_window = intern(connection, b"WM_DELETE_WINDOW")?;
    // A close button asks rather than kills: without this the window
    // manager would disconnect the whole client, the renderer's presenting
    // connection with it.
    connection.change_property32(
        PropMode::REPLACE,
        window,
        intern(connection, b"WM_PROTOCOLS")?,
        AtomEnum::ATOM,
        &[wm_delete_window],
    )?;
    // Instance and class, each NUL-terminated: what a desktop groups the
    // window under.
    connection.change_property8(
        PropMode::REPLACE,
        window,
        AtomEnum::WM_CLASS,
        AtomEnum::STRING,
        b"media-pp\0media-pp\0",
    )?;
    let title = options.title.as_bytes();
    connection.change_property8(
        PropMode::REPLACE,
        window,
        AtomEnum::WM_NAME,
        AtomEnum::STRING,
        title,
    )?;
    connection.change_property8(
        PropMode::REPLACE,
        window,
        intern(connection, b"_NET_WM_NAME")?,
        intern(connection, b"UTF8_STRING")?,
        title,
    )?;
    connection.map_window(window)?;
    connection.flush()?;
    Ok((window, wm_delete_window))
}

fn intern(connection: &XCBConnection, name: &[u8]) -> Result<u32, Box<dyn std::error::Error>> {
    Ok(connection.intern_atom(false, name)?.reply()?.atom)
}

/// The window thread: report what happens until the window is destroyed.
fn run(
    connection: &XCBConnection,
    mut watched: Watched,
    mut keymap: Keymap,
    events: &Sender<WindowEvent>,
) {
    let window = watched.window;
    while let Ok(event) = connection.wait_for_event() {
        match event {
            Event::KeyPress(press) => {
                let _ = events.send(WindowEvent::Key(keymap.key(press.detail)));
            }
            // Moves as well as resizes; only a new size is news.
            Event::ConfigureNotify(ConfigureNotifyEvent { width, height, .. }) => {
                let now = [u32::from(width), u32::from(height)];
                if now != watched.size {
                    watched.size = now;
                    let _ = events.send(WindowEvent::Resized {
                        width: now[0],
                        height: now[1],
                    });
                }
            }
            // Hidden rather than destroyed, as on Windows: the renderer still
            // presents into it, and what closing means — stop, pause, quit — is
            // the application's to decide from `Closed`. Hidden before it is
            // reported, so whoever acts on `Closed` finds it already gone.
            Event::ClientMessage(message)
                if message.window == window
                    && message.data.as_data32()[0] == watched.wm_delete_window =>
            {
                let _ = connection.unmap_window(window);
                let _ = connection.flush();
                let _ = events.send(WindowEvent::Closed);
            }
            Event::MappingNotify(_) => {
                if let Ok(fresh) = Keymap::read(connection) {
                    keymap = fresh;
                }
            }
            Event::DestroyNotify(destroyed) if destroyed.window == window => break,
            _ => {}
        }
    }
}

/// Which key each keycode is: the first keysym of each, which is the
/// unshifted one — a lowercase letter, a digit.
struct Keymap {
    min_keycode: u8,
    per_keycode: usize,
    keysyms: Vec<u32>,
}

impl Keymap {
    fn read(connection: &XCBConnection) -> Result<Self, Box<dyn std::error::Error>> {
        let setup = connection.setup();
        let (min, max) = (setup.min_keycode, setup.max_keycode);
        let reply = connection
            .get_keyboard_mapping(min, max - min + 1)?
            .reply()?;
        Ok(Self {
            min_keycode: min,
            per_keycode: usize::from(reply.keysyms_per_keycode),
            keysyms: reply.keysyms,
        })
    }

    fn key(&self, keycode: u8) -> Key {
        let keysym = keycode
            .checked_sub(self.min_keycode)
            .and_then(|index| self.keysyms.get(usize::from(index) * self.per_keycode))
            .copied()
            .unwrap_or(0);
        Key::from_keysym(keysym)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    /// A window opens with the size asked for, is mapped, and goes when
    /// dropped: its thread ends, and the event channel with it.
    #[test]
    fn a_window_opens_and_goes_when_dropped() {
        let (events_tx, events) = crossbeam_channel::unbounded();
        let options = WindowOptions {
            title: "media-pp test window".into(),
            width: 160,
            height: 120,
        };
        let window = match OwnedWindow::open(&options, events_tx) {
            Ok(window) => window,
            Err(error) => {
                eprintln!("skipping: no X11 window can be opened here ({error})");
                return;
            }
        };
        let geometry = window
            .connection
            .get_geometry(window.window)
            .expect("a request")
            .reply()
            .expect("the window exists");
        assert_eq!((geometry.width, geometry.height), (160, 120));
        assert!(matches!(
            window.handles(),
            (RawDisplayHandle::Xcb(_), RawWindowHandle::Xcb(_))
        ));

        drop(window);
        // Every sender went with the thread: nothing more can arrive.
        loop {
            match events.recv_timeout(Duration::from_secs(5)) {
                Ok(_) => continue,
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                    panic!("the window thread did not end")
                }
            }
        }
    }
}
