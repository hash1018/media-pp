//! The rate a periodic source emits at, changeable while it runs.
//!
//! Every element that paces itself with a [`crate::schedule::PeriodicSchedule`]
//! took that rate at construction and had no way to be told another. This is
//! the pair that lets one be: the element holds an [`Arc<FrameRate>`] its own
//! loop reads each tick, and hands out a [`FrameRateHandle`] that another
//! thread can write.
//!
//! # What a change means
//!
//! For a source whose output `pts` is a tick counter — every compositor here,
//! and every capture — the time base is the reciprocal of this rate. So a
//! change re-means every timestamp after it, while the ones already downstream
//! were stamped under the old one. Nothing in this type can repair that: a
//! muxer holding a time base from `avformat_write_header` will not be told,
//! and an encoder's rate control was configured once.
//!
//! Each element documents its own version of that on its own setter. The rule
//! they share: safe while nothing downstream is reading timestamps, and the
//! caller's to know.

use std::{
    sync::{
        Arc, Weak,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use ffmpeg_next as ffmpeg;

/// The rate an element is emitting at, shared between its own loop and
/// whoever can change it.
///
/// One word rather than a lock: a reader is a tick loop that wants this and
/// nothing else, and a `Mutex` here would be contended by exactly one writer
/// that almost never writes. Packed rather than two atomics so a reader can
/// never catch a numerator from one setting and a denominator from another.
#[derive(Debug)]
pub struct FrameRate(AtomicU64);

impl FrameRate {
    /// Starts at `rate`, which the caller has already validated — a
    /// constructor rejecting its own options says so with its own error type,
    /// which is more use than a `None` from here.
    pub fn new(rate: ffmpeg::Rational) -> Arc<Self> {
        Arc::new(Self(AtomicU64::new(pack(rate))))
    }

    /// The rate as it stands.
    pub fn get(&self) -> ffmpeg::Rational {
        unpack(self.0.load(Ordering::Relaxed))
    }

    /// How long one frame lasts at the current rate — what a
    /// [`crate::schedule::PeriodicSchedule`] is driven with.
    pub fn interval(&self) -> Duration {
        let rate = self.get();
        Duration::from_secs_f64(rate.denominator() as f64 / rate.numerator() as f64)
    }

    /// Changes the rate, taking effect at the next tick of whatever loop
    /// reads this. `false` for a rate that is not positive, which leaves the
    /// running one alone rather than pacing on a nonsense interval.
    pub fn set(&self, rate: ffmpeg::Rational) -> bool {
        if rate.numerator() <= 0 || rate.denominator() <= 0 {
            return false;
        }
        self.0.store(pack(rate), Ordering::Relaxed);
        true
    }

    /// A handle another thread can change this through.
    ///
    /// `Weak`, so holding one does not keep the element alive: a handle
    /// outliving what it controls answers rather than resurrecting it.
    pub fn handle(self: &Arc<Self>) -> FrameRateHandle {
        FrameRateHandle(Arc::downgrade(self))
    }
}

/// Runtime control for one element's output rate.
///
/// Cheap to clone and cheap to call — a store, and no allocation or lock. It
/// keeps nothing alive: once the element is gone, both methods say so instead
/// of taking effect on something that is no longer running.
#[derive(Debug, Clone)]
pub struct FrameRateHandle(Weak<FrameRate>);

impl FrameRateHandle {
    /// Changes the rate, taking effect at the element's next tick.
    ///
    /// Returns `false` for a rate that is not positive, and for an element
    /// that has already been dropped. See this module's own docs for what a
    /// change means to anything reading the element's timestamps.
    pub fn set(&self, rate: ffmpeg::Rational) -> bool {
        self.0.upgrade().is_some_and(|shared| shared.set(rate))
    }

    /// The rate the element is actually emitting at, or `None` once it is
    /// gone.
    ///
    /// Read back rather than remembered by the caller: a rate [`Self::set`]
    /// refused leaves the old one running, and the two disagreeing is how a
    /// recording ends up configured for a rate nothing is producing.
    pub fn get(&self) -> Option<ffmpeg::Rational> {
        Some(self.0.upgrade()?.get())
    }
}

fn pack(rate: ffmpeg::Rational) -> u64 {
    ((rate.numerator() as u32 as u64) << 32) | rate.denominator() as u32 as u64
}

fn unpack(packed: u64) -> ffmpeg::Rational {
    ffmpeg::Rational::new((packed >> 32) as u32 as i32, packed as u32 as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rate_survives_being_packed_and_read_back() {
        for rate in [
            ffmpeg::Rational::new(60, 1),
            ffmpeg::Rational::new(30, 1),
            ffmpeg::Rational::new(24000, 1001),
            ffmpeg::Rational::new(1, 1),
        ] {
            let shared = FrameRate::new(rate);
            assert_eq!(shared.get(), rate, "{rate}");
            assert_eq!(shared.handle().get(), Some(rate), "{rate}");
        }
    }

    #[test]
    fn the_interval_is_one_frame_long() {
        let shared = FrameRate::new(ffmpeg::Rational::new(50, 1));
        assert_eq!(shared.interval(), Duration::from_millis(20));

        assert!(shared.handle().set(ffmpeg::Rational::new(25, 1)));
        assert_eq!(shared.interval(), Duration::from_millis(40));
    }

    /// A rate that cannot be kept is refused, and refusing leaves the running
    /// one alone rather than a source ticking on a nonsense interval.
    #[test]
    fn an_impossible_rate_is_refused_and_changes_nothing() {
        let shared = FrameRate::new(ffmpeg::Rational::new(60, 1));
        let handle = shared.handle();

        for refused in [
            ffmpeg::Rational::new(0, 1),
            ffmpeg::Rational::new(-30, 1),
            ffmpeg::Rational::new(30, 0),
        ] {
            assert!(!handle.set(refused), "{refused} was accepted");
            assert_eq!(shared.get(), ffmpeg::Rational::new(60, 1));
        }
    }

    /// The handle must not keep the element alive, and must answer once it is
    /// gone rather than pretending the change took.
    #[test]
    fn a_handle_outliving_its_element_answers_instead_of_taking_effect() {
        let shared = FrameRate::new(ffmpeg::Rational::new(60, 1));
        let handle = shared.handle();
        drop(shared);

        assert!(!handle.set(ffmpeg::Rational::new(30, 1)));
        assert_eq!(handle.get(), None);
    }
}
