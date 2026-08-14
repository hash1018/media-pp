//! Wall-clock scheduling helpers shared by every
//! [`crate::element::SourceElement`] that paces itself against real time
//! instead of an upstream `pts`. Two distinct shapes of that problem live
//! here:
//!
//! - [`PeriodicSchedule`] — an absolute `next_due` deadline
//!   (`TestVideoSource`, `DxgiCaptureSource`, `VideoCompositor`,
//!   `D3d11VideoCompositor`, ...) that must survive a
//!   [`crate::control::ControlMsg::Pause`]/`Resume` cycle without losing
//!   its phase, yet also must not let one abnormally slow tick turn into a
//!   burst of back-to-back catch-up work.
//! - [`ActiveTimeline`] — an elapsed-time budget (`WasapiCaptureSource`,
//!   `TestAudioSource`, `AudioMixer`, ...) measuring how much *active*
//!   (non-`Pause`d) wall-clock time has passed since this source started,
//!   used to decide how many samples/ticks are owed so far.
//!
//! Both used to be duplicated (and, for a while, disagreed on) inline in
//! each source's own `run()` loop — see each type's own docs for exactly
//! what they replace.

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

/// How much *active* (non-`Pause`d) wall-clock time has passed since this
/// source started — the elapsed-time counterpart to [`PeriodicSchedule`]'s
/// absolute deadline, for sources that decide how many samples/ticks are
/// owed so far rather than waiting for one fixed-size tick at a time (e.g.
/// `WasapiCaptureSource` filling a silence gap, `AudioMixer`/
/// `TestAudioSource` deciding how many samples a mix tick owes).
///
/// Tracks a single shifting `anchor` rather than a separate elapsed-time
/// accumulator: `now.saturating_duration_since(anchor)` after shifting
/// `anchor` forward by a pause's duration gives exactly the same value as
/// `now.saturating_duration_since(start) - paused_total` would with a
/// fixed `start` and a separately accumulated `paused_total` — one field
/// instead of two, and it composes for free across any number of
/// `Pause`/`Resume` cycles. Unlike [`crate::clock::Clock`] (which every
/// [`crate::elements::Pacer`] in a pipeline shares, and which observes
/// `pause()`/`resume()` as they happen in real time), each caller of this
/// type owns its own private instance and only ever learns about a pause
/// after the fact — one summed [`crate::control::ControlOutcome::paused_for`]
/// per [`crate::control::drain_control`] call — so there is no in-progress
/// "currently paused" state to represent here.
#[derive(Debug, Clone, Copy)]
pub struct ActiveTimeline {
    anchor: Instant,
}

impl ActiveTimeline {
    pub fn new(now: Instant) -> Self {
        Self { anchor: now }
    }

    /// Folds a measured [`crate::control::ControlOutcome::paused_for`]
    /// back in, so [`ActiveTimeline::elapsed`] does not count time spent
    /// frozen inside `Pause` as active — see the type's own docs for why
    /// shifting the anchor is equivalent to subtracting an accumulated
    /// total.
    pub fn account_pause(&mut self, paused_for: Duration) {
        self.anchor += paused_for;
    }

    /// Active time elapsed since `now`(construction) minus every
    /// `paused_for` folded in since. Saturates to zero rather than
    /// underflowing/panicking if `now` predates the (possibly
    /// pause-shifted) anchor.
    pub fn elapsed(&self, now: Instant) -> Duration {
        now.saturating_duration_since(self.anchor)
    }
}

#[cfg(test)]
mod active_timeline_tests {
    use super::*;

    #[test]
    fn elapsed_excludes_time_spent_paused() {
        let t0 = Instant::now();
        let mut timeline = ActiveTimeline::new(t0);

        let before_pause = t0 + Duration::from_millis(200);
        assert_eq!(timeline.elapsed(before_pause), Duration::from_millis(200));

        // A 500ms Pause/Resume cycle measured 500ms after `before_pause`.
        timeline.account_pause(Duration::from_millis(500));
        let after_pause = before_pause + Duration::from_millis(500);
        assert_eq!(
            timeline.elapsed(after_pause),
            Duration::from_millis(200),
            "the 500ms spent paused must not count as active time"
        );
    }

    #[test]
    fn multiple_pauses_accumulate() {
        let t0 = Instant::now();
        let mut timeline = ActiveTimeline::new(t0);
        timeline.account_pause(Duration::from_millis(100));
        timeline.account_pause(Duration::from_millis(250));

        assert_eq!(
            timeline.elapsed(t0 + Duration::from_secs(1)),
            Duration::from_millis(650),
            "1s of real time minus 350ms total paused"
        );
    }

    #[test]
    fn elapsed_saturates_instead_of_underflowing() {
        let t0 = Instant::now();
        let mut timeline = ActiveTimeline::new(t0);
        timeline.account_pause(Duration::from_secs(10));

        assert_eq!(timeline.elapsed(t0), Duration::ZERO);
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
