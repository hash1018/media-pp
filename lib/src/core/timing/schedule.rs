//! Wall-clock scheduling shared by the sources that pace themselves
//! against real time instead of an upstream `pts`: [`PeriodicSchedule`],
//! an absolute `next_due` deadline (`TestVideoSource`, `DxgiCaptureSource`,
//! the video compositors, ...) that must not let one abnormally slow tick
//! turn into a burst of back-to-back catch-up work.
//!
//! What a pause leaves out is not this module's: a
//! [`crate::element::Source`] keeps its schedule on
//! [`crate::element::Wait::now`], a clock that stands still while the
//! pipeline is paused.

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

    /// How long until `next_due`, or `Duration::ZERO` if it has already
    /// passed — safe to feed straight into [`std::thread::sleep`] or a
    /// bounded poll timeout without checking whether it is due first.
    pub fn remaining(&self, now: Instant) -> Duration {
        self.next_due.saturating_duration_since(now)
    }

    /// Advances to the next tick after handling the current one. If that
    /// still lands in the past — a single tick's own work (composition,
    /// capture, generation, push) took longer than `interval` — drops
    /// every missed tick instead of letting them all fire back-to-back
    /// with no sleep between them the next time this loop runs, and
    /// resumes cadence anchored at "one interval from now" rather than the
    /// stale deadline.
    ///
    /// Returns how many deadlines were skipped that way, which is the count
    /// of ticks this schedule never produced — zero whenever the tick
    /// finished in time.
    pub fn advance_after_tick(&mut self, now: Instant) -> u64 {
        self.next_due += self.interval;
        let missed = self.ticks_behind(now);
        self.resync_if_behind(now);
        missed
    }

    /// How many deadlines, the pending one included, have already passed.
    fn ticks_behind(&self, now: Instant) -> u64 {
        if self.next_due >= now {
            return 0;
        }
        let late = now.duration_since(self.next_due).as_nanos();
        let interval = self.interval.as_nanos().max(1);
        u64::try_from(late / interval + 1).unwrap_or(u64::MAX)
    }

    /// Changes the cadence, taking effect at the next tick.
    ///
    /// The deadline is re-anchored to one *new* interval from `now` rather
    /// than kept: the pending one was measured against the old cadence, and
    /// on a change from slow to fast it can already be far enough out that
    /// keeping it would stall the source for the remainder of an interval it
    /// no longer has. Re-anchoring also means a change from fast to slow
    /// cannot leave a deadline already in the past, which is the burst every
    /// other method here exists to prevent.
    ///
    /// One tick's worth of phase is the cost, once, at the moment the caller
    /// asked for a different rate.
    pub fn set_interval(&mut self, interval: Duration, now: Instant) {
        self.interval = interval;
        self.next_due = now + interval;
    }

    /// The cadence currently being kept, so a caller driving this from
    /// changeable configuration can tell whether it still matches without
    /// storing a second copy of the answer.
    pub fn interval(&self) -> Duration {
        self.interval
    }

    fn resync_if_behind(&mut self, now: Instant) {
        if self.next_due < now {
            self.next_due = now + self.interval;
        }
    }
}

/// A live source's schedule as its `produce` follows it: begun as the first
/// tick is asked for, kept at whatever rate each is asked with, and moved
/// on past a tick only once [`Self::made`] says that tick's work is done.
///
/// The placement is the point. A tick's work — making a frame and pushing
/// it, which a slow downstream can stretch arbitrarily — has to be behind
/// the `now` that [`PeriodicSchedule::advance_after_tick`] judges a resync
/// by; moved on before it, one abnormally slow tick's catch-up frame slips
/// through uncapped before the next is asked for.
#[derive(Debug, Default)]
pub(crate) struct LiveSchedule {
    schedule: Option<PeriodicSchedule>,
    /// Whether a tick's work was done since the schedule last moved.
    made: bool,
}

/// Where a live source's next tick stands — see [`LiveSchedule::next`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct Tick {
    now: Instant,
    /// When it is due, on the clock it was asked on.
    pub(crate) due: Instant,
    /// How many ticks the last one's work overran, and were dropped.
    pub(crate) missed: u64,
    /// Whether it was asked for at a rate other than the last one's, which
    /// the schedule is anchored afresh to.
    pub(crate) rate_changed: bool,
}

impl LiveSchedule {
    /// The next tick at `now`, `interval` after the last — on the clock a
    /// pause does not move, [`crate::element::Wait::now`], so playing on
    /// after one is not a burst of the ticks it would have owed.
    pub(crate) fn next(&mut self, interval: Duration, now: Instant) -> Tick {
        let schedule = self
            .schedule
            .get_or_insert_with(|| PeriodicSchedule::new(interval, now));
        let missed = if std::mem::take(&mut self.made) {
            schedule.advance_after_tick(now)
        } else {
            0
        };
        let rate_changed = schedule.interval() != interval;
        if rate_changed {
            schedule.set_interval(interval, now);
        }
        Tick {
            now,
            due: now + schedule.remaining(now),
            missed,
            rate_changed,
        }
    }

    /// The tick [`Self::next`] answered was made — or passed with nothing
    /// to emit, which moves the schedule on all the same, or the next would
    /// be due at once.
    pub(crate) fn made(&mut self) {
        self.made = true;
    }
}

impl Tick {
    /// Where a wait for this tick, `slice` at a time, stops this time: the
    /// tick, or `slice` from when it was asked for if that comes first.
    pub(crate) fn until(&self, slice: Duration) -> Instant {
        self.now
            .checked_add(slice)
            .map_or(self.due, |end| end.min(self.due))
    }

    /// Waits for the tick, `slice` at a time, so what a source looks at
    /// between waits — a rate set meanwhile, a capture thread's report — is
    /// looked at even at a low rate. `true` once it has come; `false` where
    /// a slice ended first or the pipeline had something for the thread —
    /// answer [`Produced::Nothing`](crate::element::Produced::Nothing)
    /// then, and be asked again.
    pub(crate) fn come(&self, wait: &mut crate::element::Wait<'_>, slice: Duration) -> bool {
        wait.until(self.until(slice)) && wait.now() >= self.due
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const INTERVAL: Duration = Duration::from_millis(100);

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

    /// What was dropped is answered, so a compositor can say how much of its
    /// frame rate it never drew. The first tick is due at `t0` itself, so the
    /// deadlines a tick finishing at `done` overran are `t0 + INTERVAL`
    /// through `done` — the last included, since the schedule resyncs past
    /// it rather than firing it.
    #[test]
    fn advance_after_tick_answers_how_many_ticks_it_dropped() {
        let t0 = Instant::now();
        let mut schedule = PeriodicSchedule::new(INTERVAL, t0);
        assert_eq!(schedule.advance_after_tick(t0), 0, "on time");
        assert_eq!(
            schedule.advance_after_tick(t0 + INTERVAL),
            0,
            "done exactly at its own deadline is still on time"
        );

        let mut schedule = PeriodicSchedule::new(INTERVAL, t0);
        assert_eq!(
            schedule.advance_after_tick(t0 + INTERVAL * 3 / 2),
            1,
            "half an interval over skips the one deadline it ran past"
        );

        let mut schedule = PeriodicSchedule::new(INTERVAL, t0);
        assert_eq!(schedule.advance_after_tick(t0 + INTERVAL * 15), 15);
        assert_eq!(
            schedule.advance_after_tick(t0 + INTERVAL * 16),
            0,
            "and after the resync the cadence is kept again"
        );
    }

    #[test]
    fn remaining_counts_down_to_the_deadline() {
        let t0 = Instant::now();
        let mut schedule = PeriodicSchedule::new(INTERVAL, t0);
        assert_eq!(
            schedule.remaining(t0),
            Duration::ZERO,
            "the first tick is due immediately"
        );
        schedule.advance_after_tick(t0); // steady state: next_due = t0 + 100ms
        assert_eq!(schedule.remaining(t0 + INTERVAL / 2), INTERVAL / 2);
        assert_eq!(schedule.remaining(t0 + INTERVAL), Duration::ZERO);
    }

    /// A rate change has to take effect from the moment it is asked for, not
    /// from a deadline set under the old one.
    #[test]
    fn set_interval_reanchors_instead_of_keeping_a_stale_deadline() {
        let start = Instant::now();
        let mut schedule = PeriodicSchedule::new(Duration::from_millis(100), start);
        schedule.advance_after_tick(start);
        assert_eq!(schedule.interval(), Duration::from_millis(100));

        // Slow to fast: the pending deadline is 100 ms out, which is four of
        // the new intervals. Keeping it would stall the source for three of
        // them before the new rate was ever kept.
        let now = start + Duration::from_millis(10);
        schedule.set_interval(Duration::from_millis(25), now);
        assert_eq!(schedule.interval(), Duration::from_millis(25));
        assert_eq!(schedule.remaining(now), Duration::from_millis(25));
        assert!(
            schedule
                .remaining(now + Duration::from_millis(25))
                .is_zero()
        );
    }

    /// And fast to slow must not leave a deadline already behind, which
    /// would fire immediately and then again a full interval later.
    #[test]
    fn set_interval_never_leaves_the_deadline_in_the_past() {
        let start = Instant::now();
        let mut schedule = PeriodicSchedule::new(Duration::from_millis(10), start);
        let now = start + Duration::from_millis(500);
        schedule.set_interval(Duration::from_millis(200), now);
        assert_eq!(schedule.remaining(now), Duration::from_millis(200));
    }

    /// Asked again before the tick was made — a wait let go — the schedule
    /// stays where it was; once made, it moves on by one interval.
    #[test]
    fn a_live_schedule_moves_on_only_past_a_tick_made() {
        let start = Instant::now();
        let mut live = LiveSchedule::default();
        let first = live.next(INTERVAL, start);
        assert_eq!(first.due, start, "the first tick is due at once");
        let again = live.next(INTERVAL, start + INTERVAL / 2);
        assert_eq!(again.due, start + INTERVAL / 2, "still the first, overdue");
        live.made();
        let second = live.next(INTERVAL, start + INTERVAL / 2);
        assert_eq!(second.due, start + INTERVAL);
        assert_eq!(second.missed, 0);
        assert!(!second.rate_changed);
    }

    /// A tick whose work overran says how many it cost, and one asked for
    /// at another rate is anchored afresh.
    #[test]
    fn a_live_schedule_counts_overruns_and_follows_its_rate() {
        let start = Instant::now();
        let mut live = LiveSchedule::default();
        live.next(INTERVAL, start);
        live.made();
        let late = live.next(INTERVAL, start + INTERVAL * 3 + INTERVAL / 2);
        assert_eq!(late.missed, 3);
        let now = start + INTERVAL * 4;
        let faster = live.next(INTERVAL / 2, now);
        assert!(faster.rate_changed);
        assert_eq!(faster.due, now + INTERVAL / 2);
    }

    /// A wait slice ends at the tick or `slice` on, whichever is first, and
    /// one too long to add to an instant ends at the tick.
    #[test]
    fn a_tick_is_waited_for_a_slice_at_a_time() {
        let start = Instant::now();
        let mut live = LiveSchedule::default();
        live.next(INTERVAL, start);
        live.made();
        let tick = live.next(INTERVAL, start);
        assert_eq!(tick.until(INTERVAL / 4), start + INTERVAL / 4);
        assert_eq!(tick.until(INTERVAL * 2), start + INTERVAL);
        assert_eq!(tick.until(Duration::MAX), start + INTERVAL);
    }
}
