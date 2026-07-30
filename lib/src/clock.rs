use std::{sync::OnceLock, time::Instant};

/// Shared wall-clock reference for pacing decoded frames to their
/// presentation time. Whichever branch (video, audio, ...) processes a
/// frame first sets the anchor; every other branch reads the same one, so
/// they agree on t=0 instead of each drifting from its own first frame.
#[derive(Default)]
pub struct Clock {
    start: OnceLock<Instant>,
}

impl Clock {
    pub fn new() -> Self {
        Self::default()
    }

    /// The instant playback started, set on first call.
    pub fn start(&self) -> Instant {
        *self.start.get_or_init(Instant::now)
    }
}
