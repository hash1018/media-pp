//! Settings a handle replaces while an element reads them, frame by frame.
//!
//! What the dynamics filters share. A handle replaces all of an element's
//! settings in one call — never one field at a time, which would let the
//! element see a new threshold beside an old one — and the element asks,
//! once a frame, whether anything moved. Asking reads one atomic; the lock
//! is taken only when the answer is yes, which is once per change rather
//! than once per frame.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

#[derive(Debug)]
pub(crate) struct Tuning<T> {
    options: Mutex<T>,
    revision: AtomicU64,
}

impl<T: Copy> Tuning<T> {
    pub(crate) fn new(options: T) -> Arc<Self> {
        Arc::new(Self {
            options: Mutex::new(options),
            revision: AtomicU64::new(0),
        })
    }

    /// The settings as they stand.
    pub(crate) fn get(&self) -> T {
        *self
            .options
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Replaces them, from the element's next frame.
    pub(crate) fn set(&self, options: T) {
        *self
            .options
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = options;
        self.revision.fetch_add(1, Ordering::Release);
    }

    /// The settings, if they or `sample_rate` have moved since `seen` —
    /// which this brings up to date. An element works its per-sample rates
    /// out again from what this answers, and from nothing else.
    pub(crate) fn fresh(&self, seen: &mut Option<(u64, u32)>, sample_rate: u32) -> Option<T> {
        let now = (self.revision.load(Ordering::Acquire), sample_rate);
        if *seen == Some(now) {
            return None;
        }
        *seen = Some(now);
        Some(self.get())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Asked twice with nothing changed, the second answer is nothing; a
    /// new setting or a new rate is noticed once.
    #[test]
    fn a_change_is_noticed_once_and_nothing_else_is() {
        let tuning = Tuning::new(1);
        let mut seen = None;
        assert_eq!(tuning.fresh(&mut seen, 48_000), Some(1), "the first look");
        assert_eq!(tuning.fresh(&mut seen, 48_000), None);
        tuning.set(2);
        assert_eq!(tuning.fresh(&mut seen, 48_000), Some(2));
        assert_eq!(tuning.fresh(&mut seen, 48_000), None);
        assert_eq!(tuning.fresh(&mut seen, 44_100), Some(2), "a new rate");
    }
}
