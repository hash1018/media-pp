//! What a window renderer's own window says, and how it is opened —
//! backend-independent, so the D3D11, D3D12 and Vulkan window renderers
//! share it.

use std::time::Duration;

use crossbeam_channel::Receiver;
use thiserror::Error as ThisError;

#[cfg(target_os = "linux")]
use crate::platform::linux::x11_window::Control;
#[cfg(target_os = "windows")]
use crate::platform::windows::window::Control;

/// How a window renderer's own window is opened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowOptions {
    /// The window's title.
    pub title: String,
    /// The size of the area the picture is drawn in, in pixels — the window
    /// is that plus its frame. The picture keeps its own aspect ratio inside
    /// it, with black bars as needed.
    pub width: u32,
    /// See [`Self::width`].
    pub height: u32,
}

impl Default for WindowOptions {
    fn default() -> Self {
        Self {
            title: "media-pp".into(),
            width: 1280,
            height: 720,
        }
    }
}

/// Something that happened to a window renderer's own window.
///
/// The renderer only reports these; what a key or a close means — pause,
/// seek, stop — is the application's to decide, the way a GStreamer video
/// sink forwards navigation events rather than acting on them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WindowEvent {
    /// A key went down (repeating while held).
    Key(Key),
    /// The picture area is now this size; the renderer has already followed.
    Resized {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
    },
    /// The user asked to close the window. It is hidden, not destroyed —
    /// the renderer keeps presenting into it until it is dropped — so a
    /// pipeline goes on until the application stops it.
    Closed,
    /// A mouse button went down, at `x`, `y` in the picture area — pixels
    /// from its top left, the black bars included.
    MouseDown {
        /// Which button.
        button: MouseButton,
        /// Pixels from the left of the picture area.
        x: i32,
        /// Pixels from the top of the picture area.
        y: i32,
    },
    /// A second press of the same button in quick succession — in place of
    /// the second [`Self::MouseDown`], so a double click is one
    /// `MouseDown` and one `DoubleClick`. What counts as quick is the
    /// desktop's own setting on Windows, and half a second on Linux.
    DoubleClick {
        /// Which button.
        button: MouseButton,
        /// Pixels from the left of the picture area.
        x: i32,
        /// Pixels from the top of the picture area.
        y: i32,
    },
}

/// A mouse button, as a window renderer reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MouseButton {
    /// The primary button.
    Left,
    /// The secondary button.
    Right,
    /// The wheel's button.
    Middle,
}

/// A key, as a window renderer reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Key {
    /// Space bar.
    Space,
    /// Enter / Return.
    Enter,
    /// Escape.
    Escape,
    /// Left arrow.
    Left,
    /// Right arrow.
    Right,
    /// Up arrow.
    Up,
    /// Down arrow.
    Down,
    /// Backspace.
    Backspace,
    /// A letter (lowercase), a digit, or the full stop, comma, minus or plus
    /// — the four punctuation keys every keyboard layout has a key of its
    /// own for, and what a player steps and changes its speed with. The plus
    /// is the key Windows calls that on every layout, which on a US keyboard
    /// says `=` unshifted.
    Char(char),
    /// Any other key, by the platform's own code for it — on Windows, the
    /// virtual-key code; on Linux, the X11 keysym.
    Other(u32),
}

impl Key {
    /// Reads a Windows virtual-key code.
    #[cfg(target_os = "windows")]
    pub(crate) fn from_virtual_key(code: u32) -> Self {
        match code {
            0x20 => Self::Space,
            0x0D => Self::Enter,
            0x1B => Self::Escape,
            0x25 => Self::Left,
            0x26 => Self::Up,
            0x27 => Self::Right,
            0x28 => Self::Down,
            0x08 => Self::Backspace,
            0x30..=0x39 => Self::Char(char::from(code as u8)),
            0x41..=0x5A => Self::Char(char::from(code as u8).to_ascii_lowercase()),
            // VK_OEM_PLUS, VK_OEM_COMMA, VK_OEM_MINUS, VK_OEM_PERIOD: the one
            // virtual-key codes Windows defines as the same key on every layout.
            0xBB => Self::Char('+'),
            0xBC => Self::Char(','),
            0xBD => Self::Char('-'),
            0xBE => Self::Char('.'),
            other => Self::Other(other),
        }
    }

    /// Reads an X11 keysym — the unshifted one, so a letter is lowercase.
    #[cfg(target_os = "linux")]
    pub(crate) fn from_keysym(keysym: u32) -> Self {
        match keysym {
            0x0020 => Self::Space,
            0xFF0D | 0xFF8D => Self::Enter,
            0xFF1B => Self::Escape,
            0xFF51 => Self::Left,
            0xFF52 => Self::Up,
            0xFF53 => Self::Right,
            0xFF54 => Self::Down,
            0xFF08 => Self::Backspace,
            0x0030..=0x0039 | 0x0061..=0x007A => Self::Char(char::from(keysym as u8)),
            0x0041..=0x005A => Self::Char(char::from(keysym as u8).to_ascii_lowercase()),
            // The same four punctuation keys as on Windows, and no more, so a
            // program reads the same keys on both. The plus key says `=`
            // unshifted on a US layout and `+` on most others.
            0x002C..=0x002E => Self::Char(char::from(keysym as u8)),
            0x002B | 0x003D => Self::Char('+'),
            other => Self::Other(other),
        }
    }
}

/// What a window renderer's own window reports, as it happens.
///
/// Returned beside a renderer that opened its window itself. Cheap to move;
/// it keeps nothing alive — once the renderer and its window are gone,
/// [`Self::recv`] returns `None`.
#[derive(Debug)]
pub struct WindowEvents {
    pub(crate) events: Receiver<WindowEvent>,
}

impl WindowEvents {
    /// Waits for the next event; `None` once the window is gone.
    pub fn recv(&self) -> Option<WindowEvent> {
        self.events.recv().ok()
    }

    /// The next event if one is waiting, without blocking.
    pub fn try_recv(&self) -> Option<WindowEvent> {
        self.events.try_recv().ok()
    }

    /// Waits up to `timeout` for the next event — for a loop that also has a
    /// pipeline's bus to watch. `None` both when the time is up and when the
    /// window is gone; [`Self::recv`] tells the second apart.
    pub fn recv_timeout(&self, timeout: Duration) -> Option<WindowEvent> {
        self.events.recv_timeout(timeout).ok()
    }
}

/// What an application can change about a window renderer's own window
/// while it is drawn into: its title, and whether it fills the screen.
///
/// Taken from the renderer — its `window_control` — before the renderer goes
/// into a pipeline, and only from one that opened its window itself: a
/// window the application gave it is the application's to change. Cloning is
/// cheap, and a clone controls the same window. It keeps nothing alive: once
/// the renderer and its window are gone, every call returns [`WindowGone`].
///
/// Each call asks the window's own thread or the display server and returns
/// without waiting for the window to have changed; the renderer follows the
/// new size on its own, as it follows any other resize.
#[derive(Debug, Clone)]
pub struct WindowControl {
    pub(crate) control: Control,
}

/// The window a [`WindowControl`] was for is gone: its renderer was
/// dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ThisError)]
#[error("the window is gone")]
pub struct WindowGone;

impl WindowControl {
    /// Changes the window's title.
    pub fn set_title(&self, title: &str) -> Result<(), WindowGone> {
        self.control.set_title(title)
    }

    /// Fills the screen the window is on with it, without a frame, or puts
    /// it back where and how big it was. Asking for what it already is does
    /// nothing.
    pub fn set_fullscreen(&self, fullscreen: bool) -> Result<(), WindowGone> {
        self.control.set_fullscreen(fullscreen)
    }

    /// Whether it was last asked to fill the screen — through this control
    /// or a clone of it.
    pub fn is_fullscreen(&self) -> bool {
        self.control.is_fullscreen()
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[cfg(target_os = "linux")]
    #[test]
    fn keysyms_read_as_the_keys_a_player_uses() {
        assert_eq!(Key::from_keysym(0x20), Key::Space);
        assert_eq!(Key::from_keysym(0xFF51), Key::Left);
        assert_eq!(Key::from_keysym(0xFF53), Key::Right);
        assert_eq!(Key::from_keysym(0xFF1B), Key::Escape);
        assert_eq!(Key::from_keysym(0xFF0D), Key::Enter);
        assert_eq!(Key::from_keysym(0x66), Key::Char('f'));
        assert_eq!(Key::from_keysym(0x46), Key::Char('f'));
        assert_eq!(Key::from_keysym(0x31), Key::Char('1'));
        // A player's frame step keys.
        assert_eq!(Key::from_keysym(0x2E), Key::Char('.'));
        assert_eq!(Key::from_keysym(0x2C), Key::Char(','));
        assert_eq!(Key::from_keysym(0x2D), Key::Char('-'));
        // And its speed keys.
        assert_eq!(Key::from_keysym(0x3D), Key::Char('+'));
        assert_eq!(Key::from_keysym(0x2B), Key::Char('+'));
        assert_eq!(Key::from_keysym(0xFF08), Key::Backspace);
        assert_eq!(Key::from_keysym(0x2F), Key::Other(0x2F));
        assert_eq!(Key::from_keysym(0xFFBE), Key::Other(0xFFBE));
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn virtual_keys_read_as_the_keys_a_player_uses() {
        assert_eq!(Key::from_virtual_key(0x20), Key::Space);
        assert_eq!(Key::from_virtual_key(0x25), Key::Left);
        assert_eq!(Key::from_virtual_key(0x27), Key::Right);
        assert_eq!(Key::from_virtual_key(0x1B), Key::Escape);
        assert_eq!(Key::from_virtual_key(0x46), Key::Char('f'));
        assert_eq!(Key::from_virtual_key(0x31), Key::Char('1'));
        // A player's frame step keys.
        assert_eq!(Key::from_virtual_key(0xBE), Key::Char('.'));
        assert_eq!(Key::from_virtual_key(0xBC), Key::Char(','));
        assert_eq!(Key::from_virtual_key(0xBD), Key::Char('-'));
        // And its speed keys.
        assert_eq!(Key::from_virtual_key(0xBB), Key::Char('+'));
        assert_eq!(Key::from_virtual_key(0x08), Key::Backspace);
        assert_eq!(Key::from_virtual_key(0x70), Key::Other(0x70));
    }
}
