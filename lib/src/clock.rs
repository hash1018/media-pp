use std::{sync::Mutex, time::Instant};

/// Shared wall-clock reference for pacing decoded frames to their
/// presentation time. Whichever branch (video, audio, ...) processes a
/// frame first sets the anchor; every other branch reads the same one, so
/// they agree on t=0 instead of each drifting from its own first frame.
///
/// Owned by [`crate::pipeline::Pipeline`] (one per pipeline, shared with
/// every [`crate::elements::Pacer`] via the `wire` closure) so
/// `Pipeline::pause`/`resume` can keep it in sync with the rest of the
/// pipeline — see those for why a `Pacer`, mid-playback, needs this to be
/// pause-aware and not just a fixed anchor.
pub struct Clock {
    state: Mutex<State>,
}

#[derive(Clone, Copy)]
enum State {
    /// Never started — `start()` anchors to *now* on first call.
    Unset,
    Running {
        start: Instant,
    },
    Paused {
        start: Instant,
        paused_at: Instant,
    },
}

impl Default for Clock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(State::Unset),
        }
    }

    /// The instant playback started, set on first call — shifted forward
    /// on every [`Clock::resume`] by however long the clock spent paused,
    /// so `now - start()` stays continuous across a pause/resume cycle
    /// instead of jumping by the pause's real duration. Callers that pace
    /// against this (see [`crate::elements::Pacer::wait_for`]) need to
    /// call it fresh each time, not cache the first result — the whole
    /// point is that it can move.
    pub fn start(&self) -> Instant {
        let mut state = self.state.lock().unwrap();
        match *state {
            State::Unset => {
                let now = Instant::now();
                *state = State::Running { start: now };
                now
            }
            State::Running { start } => start,
            State::Paused { start, .. } => start,
        }
    }

    /// Freezes the clock in place. No-op if unset (nothing running yet)
    /// or already paused.
    pub fn pause(&self) {
        let mut state = self.state.lock().unwrap();
        if let State::Running { start } = *state {
            *state = State::Paused {
                start,
                paused_at: Instant::now(),
            };
        }
    }

    /// Undoes [`Clock::pause`] by shifting `start` forward by however long
    /// this pause lasted. No-op if not currently paused.
    pub fn resume(&self) {
        let mut state = self.state.lock().unwrap();
        if let State::Paused { start, paused_at } = *state {
            let shift = Instant::now().saturating_duration_since(paused_at);
            *state = State::Running {
                start: start + shift,
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn pause_shifts_start_forward_by_the_pause_duration() {
        let clock = Clock::new();
        let first = clock.start();

        clock.pause();
        std::thread::sleep(Duration::from_millis(30));
        clock.resume();

        let after = clock.start();
        assert!(
            after >= first + Duration::from_millis(20),
            "expected start() to shift forward by roughly the pause duration"
        );
    }
}
