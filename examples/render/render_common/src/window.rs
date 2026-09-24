//! Ending the work when a renderer's own window is closed — the same on both
//! platforms, since `D3d11WindowRenderer`, `D3d12WindowRenderer` and
//! `VulkanWindowRenderer` all report their window through the library's one
//! `WindowEvents`.

use std::{sync::Arc, thread};

use media_pp::elements::{Key, WindowEvent, WindowEvents};

use crate::Shutdown;

/// A [`Shutdown`] that closing any of `windows`, or pressing Escape in one,
/// sets off: whatever has been published by then is stopped.
///
/// Each window is watched on a thread of its own, which ends when its window
/// does — when the renderer that opened it is dropped — so nothing here has
/// to be joined. Stopping happens on that thread, not the window's own, so a
/// renderer waiting for its window never waits on the stop.
pub fn stop_on_close(windows: impl IntoIterator<Item = WindowEvents>) -> Arc<Shutdown> {
    let shutdown = Arc::new(Shutdown::default());
    for window in windows {
        let shutdown = Arc::clone(&shutdown);
        thread::spawn(move || {
            while let Some(event) = window.recv() {
                if matches!(event, WindowEvent::Closed | WindowEvent::Key(Key::Escape)) {
                    for pipeline in shutdown.request() {
                        pipeline.stop();
                    }
                }
            }
        });
    }
    shutdown
}
