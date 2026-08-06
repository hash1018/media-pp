use std::{
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

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
    /// Incremented before a control request starts cascading through the
    /// pipeline. A `Pacer` compares this with the last generation it
    /// acknowledged in `control()` so a long presentation-time wait can
    /// return promptly and let the owning worker process that request.
    interrupt_epoch: AtomicU64,
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
            interrupt_epoch: AtomicU64::new(0),
        }
    }

    /// Signals paced waits to return without changing the clock's playback
    /// anchor. The actual pause/seek/stop state change still happens through
    /// the ordinary synchronous control cascade.
    pub(crate) fn interrupt(&self) {
        self.interrupt_epoch.fetch_add(1, Ordering::Release);
    }

    pub(crate) fn interrupt_epoch(&self) -> u64 {
        self.interrupt_epoch.load(Ordering::Acquire)
    }

    /// The instant playback started, set on first call — shifted forward
    /// on every [`Clock::resume`] by however long the clock spent paused,
    /// so `now - start()` stays continuous across a pause/resume cycle
    /// instead of jumping by the pause's real duration. Callers that pace
    /// against this (see `Pacer::wait_for`) need to
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

    /// Back to the same "never started" state as a freshly constructed
    /// `Clock` — the next [`Clock::start`] call re-anchors t=0 to
    /// *that* moment, same lazy-first-caller-wins semantics as initial
    /// startup (see the type docs). Unconditional, regardless of current
    /// state.
    ///
    /// Called on [`crate::control::ControlMsg::Seek`]
    /// (see [`crate::pipeline::Pipeline::seek`]): the old anchor measured
    /// real time elapsed *for the pre-seek position* — after a jump, a
    /// `Pacer`'s `elapsed_secs` (relative to its own now-reset
    /// `first_pts`) starts over from ~0 too, so pairing it with the
    /// stale anchor would compute a `due` far in the past and skip
    /// sleeping entirely, dumping every post-seek frame with no pacing.
    /// This is the wall-clock half of that same fix — `Pacer::first_pts`
    /// resetting is the pts half; both are needed together.
    pub fn reset(&self) {
        let mut state = self.state.lock().unwrap();
        *state = State::Unset;
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

    /// Regression test for the bug found manually testing `seek_render`:
    /// without `reset()`, a `Pacer` re-anchoring only its `first_pts` (not
    /// the shared `Clock`) after a seek computed `due` times far in the
    /// past — `start()` kept returning the *original* anchor no matter
    /// how long ago that was — so every post-seek frame skipped its sleep
    /// entirely. `reset()` must make the next `start()` anchor to a fresh
    /// "now", not the original one.
    #[test]
    fn reset_makes_the_next_start_anchor_to_a_fresh_now() {
        let clock = Clock::new();
        let original = clock.start();

        std::thread::sleep(Duration::from_millis(30));
        clock.reset();
        let after_reset = clock.start();

        assert!(
            after_reset >= original + Duration::from_millis(20),
            "expected start() after reset() to anchor to a fresh instant, \
             not keep returning the original one"
        );
    }
}
