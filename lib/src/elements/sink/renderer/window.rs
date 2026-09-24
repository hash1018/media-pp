//! What a window renderer's own window says, and how it is opened —
//! backend-independent, so the D3D11 and D3D12 window renderers share it.

use std::time::Duration;

use crossbeam_channel::Receiver;

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
    /// A letter (lowercase) or a digit.
    Char(char),
    /// Any other key, by the platform's own code for it — on Windows, the
    /// virtual-key code.
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
            0x30..=0x39 => Self::Char(char::from(code as u8)),
            0x41..=0x5A => Self::Char(char::from(code as u8).to_ascii_lowercase()),
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

#[cfg(all(test, target_os = "windows"))]
mod tests {
    use super::*;

    #[test]
    fn virtual_keys_read_as_the_keys_a_player_uses() {
        assert_eq!(Key::from_virtual_key(0x20), Key::Space);
        assert_eq!(Key::from_virtual_key(0x25), Key::Left);
        assert_eq!(Key::from_virtual_key(0x27), Key::Right);
        assert_eq!(Key::from_virtual_key(0x1B), Key::Escape);
        assert_eq!(Key::from_virtual_key(0x46), Key::Char('f'));
        assert_eq!(Key::from_virtual_key(0x31), Key::Char('1'));
        assert_eq!(Key::from_virtual_key(0x70), Key::Other(0x70));
    }
}
