//! The pipeline's shared, pause-aware wall-clock reference.
//!
//! One [`Clock`] per [`Pipeline`](crate::pipeline::Pipeline), shared with
//! every [`Pacer`](crate::elements::Pacer) at wiring time. Whichever branch
//! processes a frame first anchors t=0 and the others read that same anchor,
//! which is what keeps video and audio from each drifting away from their own
//! first frame.

use std::{
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
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
    /// The media timestamp this clock's zero stands for, in nanoseconds —
    /// see [`Clock::media_origin_ns`].
    media_origin_ns: Mutex<Option<i64>>,
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
    /// Creates an unstarted clock whose first [`Self::start`] call establishes
    /// the shared playback anchor.
    pub fn new() -> Self {
        Self {
            state: Mutex::new(State::Unset),
            interrupt_epoch: AtomicU64::new(0),
            media_origin_ns: Mutex::new(None),
        }
    }

    /// The media timestamp this clock's zero stands for, establishing it
    /// from `first` if nothing has yet.
    ///
    /// One origin for the whole pipeline, not one per stream, and that is
    /// the whole point. **A container's streams do not start at the same
    /// timestamp** — an MP4's video commonly starts one frame in while its
    /// audio starts at zero, tens of milliseconds apart — and that offset is
    /// part of what keeps the picture with the sound. A stream that zeroed on
    /// its own first timestamp would throw it away and be played as though it
    /// began with every other, which is audible as lip sync a frame out.
    ///
    /// So the first stream to produce a timestamp sets the origin and every
    /// other measures from it, keeping whatever the container gave them.
    /// Cleared by [`Clock::reset`], because a seek starts a new timeline and
    /// the streams land at their own places in it.
    ///
    /// Callers hold onto what this returns rather than asking per buffer:
    /// the answer cannot change until a reset, and the reset reaches them as
    /// a control message first.
    pub(crate) fn media_origin_ns(&self, first: i64) -> i64 {
        *self
            .media_origin_ns
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get_or_insert(first)
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

    /// Pause-aware time elapsed since this clock was first anchored.
    pub(crate) fn elapsed(&self) -> Duration {
        let state = self.state.lock().unwrap();
        match *state {
            State::Unset => Duration::ZERO,
            State::Running { start } => Instant::now().saturating_duration_since(start),
            State::Paused { start, paused_at } => paused_at.saturating_duration_since(start),
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
        // The media origin goes with it: a seek lands each stream at its own
        // place in a new timeline, and the first one to arrive there is what
        // the others should be measured against — see `media_origin_ns`.
        *self
            .media_origin_ns
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
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
