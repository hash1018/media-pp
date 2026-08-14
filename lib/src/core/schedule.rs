//! Wall-clock scheduling helpers shared by every periodic
//! [`crate::element::SourceElement`] (`TestVideoSource`, `DxgiCaptureSource`,
//! `VideoCompositor`, `D3d11VideoCompositor`, ...): an absolute `next_due`
//! deadline that must survive a [`crate::control::ControlMsg::Pause`]/
//! `Resume` cycle without losing its phase, yet also must not let one
//! abnormally slow tick turn into a burst of back-to-back catch-up work.
//! [`PeriodicSchedule`] is exactly the small piece of arithmetic these
//! sources used to duplicate (and, for a while, disagree on) inline in
//! their own `run()` loops — see its own docs for the two distinct causes
//! of "behind schedule" it separates.

use std::time::{Duration, Instant};

/// An absolute, drift-free deadline (`next_due += interval` each tick, not
/// "sleep `interval` since the last tick" — the latter accumulates the
/// nonzero cost of the tick's own work on top of every single interval)
/// that a periodic source calls into once per loop iteration.
///
/// Every method that can move `next_due` takes `now` as a parameter
/// instead of calling [`Instant::now()`] itself, so this type's own tests
/// can drive it without real sleeps.
#[derive(Debug, Clone, Copy)]
pub struct PeriodicSchedule {
    next_due: Instant,
    interval: Duration,
}

impl PeriodicSchedule {
    /// `next_due` starts at `now` — the first tick is due immediately.
    pub fn new(interval: Duration, now: Instant) -> Self {
        Self {
            next_due: now,
            interval,
        }
    }

    /// `true` once `next_due` has arrived.
    pub fn is_due(&self, now: Instant) -> bool {
        now >= self.next_due
    }

    /// How long until `next_due`, or `Duration::ZERO` if it has already
    /// passed — safe to feed straight into [`std::thread::sleep`] or a
    /// bounded poll timeout without a separate `is_due` guard.
    pub fn remaining(&self, now: Instant) -> Duration {
        self.next_due.saturating_duration_since(now)
    }

    /// Folds a measured [`crate::control::ControlOutcome::paused_for`]
    /// back into the schedule: shifts `next_due` forward by exactly how
    /// long `Pause` held it, so `Resume` picks up from the same phase it
    /// had before freezing — real (`Instant`) time keeps moving during a
    /// pause, but the media timeline must not, or every source pacing
    /// against wall-clock time would see `Resume` as a burst of owed work.
    /// If the shift still lands in the past (the pause cascade itself ran
    /// long, or `paused_for` under-counted it), falls back to the same
    /// "one interval from now" resync [`PeriodicSchedule::advance_after_tick`]
    /// uses, for the same reason: no caller of this type may ever emit a
    /// back-to-back burst, regardless of why it fell behind.
    pub fn resume_after_pause(&mut self, paused_for: Duration, now: Instant) {
        self.next_due += paused_for;
        self.resync_if_behind(now);
    }

    /// Advances to the next tick after handling the current one. If that
    /// still lands in the past — a single tick's own work (composition,
    /// capture, generation, push) took longer than `interval` — drops
    /// every missed tick instead of letting them all fire back-to-back
    /// with no sleep between them the next time this loop runs, and
    /// resumes cadence anchored at "one interval from now" rather than the
    /// stale deadline. This is a different cause of falling behind than
    /// [`PeriodicSchedule::resume_after_pause`]'s (real work taking too
    /// long vs. time frozen by `Pause`) but the same corrective action.
    pub fn advance_after_tick(&mut self, now: Instant) {
        self.next_due += self.interval;
        self.resync_if_behind(now);
    }

    fn resync_if_behind(&mut self, now: Instant) {
        if self.next_due < now {
            self.next_due = now + self.interval;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const INTERVAL: Duration = Duration::from_millis(100);

    #[test]
    fn resume_after_pause_preserves_phase_when_the_shift_is_enough() {
        let t0 = Instant::now();
        let mut schedule = PeriodicSchedule::new(INTERVAL, t0);
        schedule.advance_after_tick(t0); // steady state: next_due = t0 + 100ms
        // Pause lands 10ms before that deadline and holds for 500ms, so
        // the resume instant is t0 + 590ms.
        let resumed_at = t0 + Duration::from_millis(590);
        schedule.resume_after_pause(Duration::from_millis(500), resumed_at);

        // 10ms of the original interval was left when Pause started
        // (t0+90ms), so the preserved deadline is t0+590ms + 10ms.
        assert_eq!(
            schedule.remaining(resumed_at),
            Duration::from_millis(10),
            "expected the pre-pause phase to be preserved, not reset to resume + a full interval"
        );
    }

    #[test]
    fn resume_after_pause_resyncs_when_the_shift_still_lands_in_the_past() {
        let t0 = Instant::now();
        let mut schedule = PeriodicSchedule::new(INTERVAL, t0);
        // paused_for under-counts how long Pause actually held real time
        // (e.g. a slow downstream cascade outside the measured window):
        // shifting by it alone would still leave `next_due` behind `now`.
        let now = t0 + Duration::from_secs(10);
        schedule.resume_after_pause(Duration::from_millis(1), now);

        assert_eq!(
            schedule.remaining(now),
            INTERVAL,
            "expected a resync to exactly one interval from now, not a burst \
             of back-to-back catch-up ticks"
        );
    }

    #[test]
    fn advance_after_tick_holds_steady_cadence_when_ticks_keep_up() {
        let t0 = Instant::now();
        let mut schedule = PeriodicSchedule::new(INTERVAL, t0);
        schedule.advance_after_tick(t0);
        assert_eq!(schedule.remaining(t0), INTERVAL);
        schedule.advance_after_tick(t0 + INTERVAL);
        assert_eq!(schedule.remaining(t0 + INTERVAL), INTERVAL);
    }

    #[test]
    fn advance_after_tick_drops_missed_ticks_instead_of_bursting() {
        let t0 = Instant::now();
        let mut schedule = PeriodicSchedule::new(INTERVAL, t0);
        // One abnormally slow tick eats 15 intervals' worth of real time.
        let slow_tick_done = t0 + INTERVAL * 15;
        schedule.advance_after_tick(slow_tick_done);

        assert_eq!(
            schedule.remaining(slow_tick_done),
            INTERVAL,
            "expected a resync to one interval from now, not 14 missed \
             ticks all firing back-to-back"
        );
    }

    #[test]
    fn is_due_and_remaining_agree() {
        let t0 = Instant::now();
        let mut schedule = PeriodicSchedule::new(INTERVAL, t0);
        assert!(schedule.is_due(t0), "the first tick is due immediately");
        schedule.advance_after_tick(t0); // steady state: next_due = t0 + 100ms
        assert!(!schedule.is_due(t0 + INTERVAL / 2));
        assert_eq!(schedule.remaining(t0 + INTERVAL / 2), INTERVAL / 2);
        assert!(schedule.is_due(t0 + INTERVAL));
        assert_eq!(schedule.remaining(t0 + INTERVAL), Duration::ZERO);
    }
}
