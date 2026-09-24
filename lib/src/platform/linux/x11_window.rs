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
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
};

use crossbeam_channel::Sender;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle, XcbDisplayHandle, XcbWindowHandle};
use x11rb::{
    connection::Connection,
    protocol::{
        Event,
        xproto::{
            AtomEnum, ButtonPressEvent, ClientMessageEvent, ConfigureNotifyEvent,
            ConnectionExt as _, CreateWindowAux, EventMask, PropMode, WindowClass,
        },
    },
    wrapper::ConnectionExt as _,
    xcb_ffi::XCBConnection,
};

use crate::elements::{Key, MouseButton, WindowEvent, WindowGone, WindowOptions};

/// A window on its own connection and thread, destroyed and joined when
/// this drops.
pub(crate) struct OwnedWindow {
    connection: Arc<XCBConnection>,
    screen: usize,
    window: u32,
    control: Control,
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
        let (window, atoms) =
            create(&connection, screen, options).map_err(|error| error.to_string())?;
        let control = Control {
            shared: Arc::new(ControlShared {
                connection: Arc::clone(&connection),
                window,
                root: connection.setup().roots[screen].root,
                atoms,
                fullscreen: AtomicBool::new(false),
                gone: AtomicBool::new(false),
            }),
        };
        let keymap = Keymap::read(&connection).map_err(|error| error.to_string())?;

        // The thread only reads events; nothing about the window's creation
        // is left for it, so a failure above never has a thread to join.
        let thread = {
            let connection = Arc::clone(&connection);
            let window = Watched {
                window,
                wm_delete_window: atoms.wm_delete_window,
                size: [options.width, options.height],
                last_press: None,
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
            control,
            thread: Some(thread),
        })
    }

    /// What changes the window's title and fullscreen state from another
    /// thread.
    pub(crate) fn control(&self) -> Control {
        self.control.clone()
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
        // Before the window goes, so no control reaches an id the server may
        // since have given another window.
        self.control.shared.gone.store(true, Ordering::Release);
        // The thread ends on the `DestroyNotify` this brings, or on the
        // connection failing; either way there is nothing to wait for but it.
        let _ = self.connection.destroy_window(self.window);
        let _ = self.connection.flush();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Changes an [`OwnedWindow`] from any thread — see
/// [`crate::elements::WindowControl`]. Requests go out on the window's own
/// connection, which is safe to share across threads.
#[derive(Debug, Clone)]
pub(crate) struct Control {
    shared: Arc<ControlShared>,
}

#[derive(Debug)]
struct ControlShared {
    connection: Arc<XCBConnection>,
    window: u32,
    root: u32,
    atoms: Atoms,
    /// Whether it was last asked to fill the screen.
    fullscreen: AtomicBool,
    /// Set before the window is destroyed.
    gone: AtomicBool,
}

impl Control {
    fn live(&self) -> Result<&ControlShared, WindowGone> {
        if self.shared.gone.load(Ordering::Acquire) {
            return Err(WindowGone);
        }
        Ok(&self.shared)
    }

    pub(crate) fn set_title(&self, title: &str) -> Result<(), WindowGone> {
        let shared = self.live()?;
        set_title(&shared.connection, shared.window, &shared.atoms, title).map_err(|_| WindowGone)
    }

    /// Asks the window manager, as EWMH has an application do: a
    /// `_NET_WM_STATE` message to the root window, adding or removing
    /// `_NET_WM_STATE_FULLSCREEN`.
    pub(crate) fn set_fullscreen(&self, fullscreen: bool) -> Result<(), WindowGone> {
        let shared = self.live()?;
        if shared.fullscreen.swap(fullscreen, Ordering::AcqRel) == fullscreen {
            return Ok(());
        }
        // Action (1 add, 0 remove), the property, no second one, and the
        // source: 1, an ordinary application.
        let message = ClientMessageEvent::new(
            32,
            shared.window,
            shared.atoms.net_wm_state,
            [
                u32::from(fullscreen),
                shared.atoms.net_wm_state_fullscreen,
                0,
                1,
                0,
            ],
        );
        let sent = shared.connection.send_event(
            false,
            shared.root,
            EventMask::SUBSTRUCTURE_REDIRECT | EventMask::SUBSTRUCTURE_NOTIFY,
            message,
        );
        sent.and_then(|_| shared.connection.flush())
            .map_err(|_| WindowGone)
    }

    pub(crate) fn is_fullscreen(&self) -> bool {
        self.shared.fullscreen.load(Ordering::Acquire)
    }
}

/// The atoms a window's requests and events name, interned once.
#[derive(Debug, Clone, Copy)]
struct Atoms {
    /// What a close button's request carries.
    wm_delete_window: u32,
    net_wm_name: u32,
    utf8_string: u32,
    net_wm_state: u32,
    net_wm_state_fullscreen: u32,
}

/// Both titles a window has: the old `WM_NAME` and EWMH's UTF-8
/// `_NET_WM_NAME`, which a modern window manager shows.
fn set_title(
    connection: &XCBConnection,
    window: u32,
    atoms: &Atoms,
    title: &str,
) -> Result<(), x11rb::errors::ConnectionError> {
    let title = title.as_bytes();
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
        atoms.net_wm_name,
        atoms.utf8_string,
        title,
    )?;
    connection.flush()
}

/// How often the monitor showing `window` refreshes: the mode of the RandR
/// CRTC its centre is on, read on a connection of this call's own — so any
/// X11 window, the renderer's or one it was given. `None` where the server
/// has no RandR or the window is on no monitor.
pub(crate) fn refresh_interval(window: u32) -> Option<std::time::Duration> {
    use x11rb::protocol::randr::{ConnectionExt as _, ModeFlag};

    let (connection, screen) = XCBConnection::connect(None).ok()?;
    let root = connection.setup().roots.get(screen)?.root;
    let geometry = connection.get_geometry(window).ok()?.reply().ok()?;
    let centre = connection
        .translate_coordinates(
            window,
            root,
            (geometry.width / 2) as i16,
            (geometry.height / 2) as i16,
        )
        .ok()?
        .reply()
        .ok()?;
    let (x, y) = (i32::from(centre.dst_x), i32::from(centre.dst_y));
    let resources = connection
        .randr_get_screen_resources_current(root)
        .ok()?
        .reply()
        .ok()?;
    for &crtc in &resources.crtcs {
        let Some(info) = connection
            .randr_get_crtc_info(crtc, resources.config_timestamp)
            .ok()
            .and_then(|cookie| cookie.reply().ok())
        else {
            continue;
        };
        let (left, top) = (i32::from(info.x), i32::from(info.y));
        let on = info.mode != 0
            && (left..left + i32::from(info.width)).contains(&x)
            && (top..top + i32::from(info.height)).contains(&y);
        if !on {
            continue;
        }
        let mode = resources.modes.iter().find(|mode| mode.id == info.mode)?;
        let mut lines = f64::from(mode.vtotal);
        if mode.mode_flags.contains(ModeFlag::DOUBLE_SCAN) {
            lines *= 2.0;
        }
        if mode.mode_flags.contains(ModeFlag::INTERLACE) {
            lines /= 2.0;
        }
        let hertz = f64::from(mode.dot_clock) / (f64::from(mode.htotal) * lines);
        return (hertz.is_finite() && hertz >= 1.0)
            .then(|| std::time::Duration::from_secs_f64(1.0 / hertz));
    }
    None
}

/// What the window thread watches for.
struct Watched {
    window: u32,
    /// The atom a close button's request carries.
    wm_delete_window: u32,
    /// The size last reported, so a move is not reported as a resize.
    size: [u32; 2],
    /// The last press not yet part of a double click: its button, time and
    /// place.
    last_press: Option<(u8, u32, i32, i32)>,
}

/// How long and how far apart two presses of one button may be and still be
/// a double click. X11 has no double click of its own; half a second and a
/// few pixels is what most toolkits settle on.
const DOUBLE_CLICK_MS: u32 = 500;
const DOUBLE_CLICK_DISTANCE: i32 = 4;

impl Watched {
    /// A press as the event it is: the second of a quick pair is a double
    /// click, and ends the pair.
    fn press(&mut self, press: &ButtonPressEvent) -> Option<WindowEvent> {
        let button = match press.detail {
            1 => MouseButton::Left,
            2 => MouseButton::Middle,
            3 => MouseButton::Right,
            // The wheel, and buttons past it.
            _ => return None,
        };
        let (x, y) = (i32::from(press.event_x), i32::from(press.event_y));
        let double = self
            .last_press
            .is_some_and(|(detail, time, last_x, last_y)| {
                detail == press.detail
                    && press.time.wrapping_sub(time) <= DOUBLE_CLICK_MS
                    && (x - last_x).abs() <= DOUBLE_CLICK_DISTANCE
                    && (y - last_y).abs() <= DOUBLE_CLICK_DISTANCE
            });
        if double {
            self.last_press = None;
            Some(WindowEvent::DoubleClick { button, x, y })
        } else {
            self.last_press = Some((press.detail, press.time, x, y));
            Some(WindowEvent::MouseDown { button, x, y })
        }
    }
}

/// Creates and maps the window, and interns the atoms its requests and
/// events name.
fn create(
    connection: &XCBConnection,
    screen: usize,
    options: &WindowOptions,
) -> Result<(u32, Atoms), Box<dyn std::error::Error>> {
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
            .event_mask(
                EventMask::KEY_PRESS | EventMask::BUTTON_PRESS | EventMask::STRUCTURE_NOTIFY,
            ),
    )?;
    let atoms = Atoms {
        wm_delete_window: intern(connection, b"WM_DELETE_WINDOW")?,
        net_wm_name: intern(connection, b"_NET_WM_NAME")?,
        utf8_string: intern(connection, b"UTF8_STRING")?,
        net_wm_state: intern(connection, b"_NET_WM_STATE")?,
        net_wm_state_fullscreen: intern(connection, b"_NET_WM_STATE_FULLSCREEN")?,
    };
    // A close button asks rather than kills: without this the window
    // manager would disconnect the whole client, the renderer's presenting
    // connection with it.
    connection.change_property32(
        PropMode::REPLACE,
        window,
        intern(connection, b"WM_PROTOCOLS")?,
        AtomEnum::ATOM,
        &[atoms.wm_delete_window],
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
    set_title(connection, window, &atoms, &options.title)?;
    connection.map_window(window)?;
    connection.flush()?;
    Ok((window, atoms))
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
            Event::ButtonPress(press) if press.event == window => {
                if let Some(event) = watched.press(&press) {
                    let _ = events.send(event);
                }
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

    fn press(detail: u8, time: u32, x: i16, y: i16) -> ButtonPressEvent {
        ButtonPressEvent {
            response_type: 4,
            detail,
            sequence: 0,
            time,
            root: 0,
            event: 1,
            child: 0,
            root_x: x,
            root_y: y,
            event_x: x,
            event_y: y,
            state: 0u16.into(),
            same_screen: true,
        }
    }

    /// A quick second press of the same button, near the first, is a
    /// double click and ends the pair; a slow one, another button's, or the
    /// wheel's is not.
    #[test]
    fn a_quick_second_press_is_a_double_click() {
        let mut watched = Watched {
            window: 1,
            wm_delete_window: 0,
            size: [0, 0],
            last_press: None,
        };
        let down = |button, x, y| Some(WindowEvent::MouseDown { button, x, y });
        let double = |button, x, y| Some(WindowEvent::DoubleClick { button, x, y });
        assert_eq!(
            watched.press(&press(1, 1000, 10, 10)),
            down(MouseButton::Left, 10, 10)
        );
        assert_eq!(
            watched.press(&press(1, 1300, 12, 11)),
            double(MouseButton::Left, 12, 11)
        );
        // The pair is spent: a third press starts over.
        assert_eq!(
            watched.press(&press(1, 1400, 12, 11)),
            down(MouseButton::Left, 12, 11)
        );
        assert_eq!(
            watched.press(&press(1, 2000, 12, 11)),
            down(MouseButton::Left, 12, 11)
        );
        assert_eq!(
            watched.press(&press(3, 2100, 12, 11)),
            down(MouseButton::Right, 12, 11)
        );
        assert_eq!(watched.press(&press(4, 2150, 12, 11)), None, "the wheel");
    }

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
