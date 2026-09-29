//! The handshake between whatever closes a window and the worker that owns
//! the pipelines presenting into it — a renderer's own window, watched by
//! [`crate::stop_on_close`].

use std::sync::{Arc, Mutex, Weak};

use media_pp::pipeline::Pipeline;

/// The handshake between the window and the worker that owns the pipelines.
///
/// The worker [`publish`](Self::publish)es what it built; the window records
/// a close and stops whatever has been published by then. Both go through one
/// lock, so the two orders are equivalent: a close that beats `publish` is
/// reported back by `publish` itself.
///
/// It holds the pipelines weakly. What owns a pipeline is the worker that
/// built it, and dropping it there is what joins its threads — the ones that
/// drop its elements, a renderer among them. A strong hold here outlived
/// the worker whenever this outlived it, as it does wherever the
/// thread watching a window keeps it until the window is gone: nothing then
/// joined those threads, and the process could exit in the middle of a
/// renderer tearing itself down.
#[derive(Default)]
pub struct Shutdown {
    state: Mutex<ShutdownState>,
}

#[derive(Default)]
struct ShutdownState {
    requested: bool,
    pipelines: Vec<Weak<Pipeline>>,
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
        state.pipelines = pipelines.iter().map(Arc::downgrade).collect();
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

    /// Window: records the close and hands back what to stop — whichever
    /// of the published pipelines still exist.
    pub(crate) fn request(&self) -> Vec<Arc<Pipeline>> {
        let mut state = self.state.lock().expect("shutdown state poisoned");
        state.requested = true;
        state.pipelines.iter().filter_map(Weak::upgrade).collect()
    }
}
