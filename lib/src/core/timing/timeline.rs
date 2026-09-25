//! Which timeline a buffer belongs to.
//!
//! A seek starts a new timeline: what was read before it is from a position
//! the pipeline has left. The synchronous `Flush` cascade discards what is
//! already queued, but it is only as good as every thread's timing — a
//! source that reads on a moment too long, a buffer a thread was holding as
//! the flush went past, and old media is handed on after the new position's.
//! So every buffer also carries the number of the timeline it was made on,
//! and what is behind the pipeline's current one is dropped where it crosses
//! a thread, whatever the cascade missed.
//!
//! # How a buffer gets its number
//!
//! Without being told. Within one thread a buffer is made while the one
//! before it is being consumed, so the number lives with the thread: a
//! source's thread takes the pipeline's current number when it starts and
//! again as it applies a `Seek`, and everything its consume chain makes on
//! that thread is on that timeline. A [`Queue`](crate::queue::Queue) is the
//! one place a buffer changes thread; it takes the number of the thread
//! handing it over and gives it to its worker's thread as the buffer goes
//! on. No element, and no [`MediaBuffer`], has to know.
//!
//! A number means something only against the timeline that gave it, so a
//! queue takes one only from a thread of its own pipeline. A thread nobody
//! numbered — an element's own worker, a test driving elements by hand, a
//! thread of another pipeline handing over to this one — hands buffers over
//! [`UNNUMBERED`], which is never behind: failing open is the old
//! behaviour, where failing closed would drop a stream outright.
//!
//! # What is here for later
//!
//! Only the number, for now: what a timeline *is* — where it starts, at
//! what rate and in which direction it runs — is what a rate or a reverse
//! seek will add to it, and each buffer can then be read against its own.

use std::{
    cell::{Cell, RefCell},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use crate::buffer::MediaBuffer;

/// The number of a buffer made on a thread that no pipeline numbered.
pub(crate) const UNNUMBERED: u64 = 0;

/// A pipeline's current timeline, shared by every thread it runs on.
#[derive(Debug)]
pub(crate) struct Timeline {
    current: AtomicU64,
}

impl Timeline {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            current: AtomicU64::new(UNNUMBERED + 1),
        })
    }

    /// The number of the timeline the pipeline is on.
    pub(crate) fn current(&self) -> u64 {
        self.current.load(Ordering::Acquire)
    }

    /// Starts a new timeline, and answers its number. Everything numbered
    /// before this is behind from now on.
    pub(crate) fn begin(&self) -> u64 {
        self.current.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Whether a buffer made on timeline `number` is from a position the
    /// pipeline has since left.
    pub(crate) fn is_behind(&self, number: u64) -> bool {
        number != UNNUMBERED && number < self.current()
    }
}

/// A buffer and the number of the timeline it was made on — what a queue
/// carries across a thread.
pub(crate) struct Numbered {
    pub(crate) number: u64,
    pub(crate) buf: MediaBuffer,
}

impl Numbered {
    /// `buf`, numbered as this thread's buffers are on `timeline` — a
    /// queue's own, `None` for one spawned by hand — or [`UNNUMBERED`] where
    /// this thread is not one of that timeline's.
    pub(crate) fn on(timeline: Option<&Arc<Timeline>>, buf: MediaBuffer) -> Self {
        let number = timeline.map_or(UNNUMBERED, |timeline| {
            let ours = TIMELINE.with(|slot| {
                slot.borrow()
                    .as_ref()
                    .is_some_and(|entered| Arc::ptr_eq(entered, timeline))
            });
            if ours { current() } else { UNNUMBERED }
        });
        Self { number, buf }
    }
}

thread_local! {
    /// The timeline this thread's pipeline is on, for [`follow`].
    static TIMELINE: RefCell<Option<Arc<Timeline>>> = const { RefCell::new(None) };
    /// The number this thread's buffers are made on.
    static NUMBER: Cell<u64> = const { Cell::new(UNNUMBERED) };
}

/// Makes this thread one of `timeline`'s, making buffers on its current
/// number — what a source's thread does as it starts, and a queue's worker.
pub(crate) fn enter(timeline: &Arc<Timeline>) {
    TIMELINE.with(|slot| *slot.borrow_mut() = Some(Arc::clone(timeline)));
    NUMBER.with(|number| number.set(timeline.current()));
}

/// Moves this thread onto its pipeline's current timeline — what a source
/// does as it applies a `Seek`. Nothing, on a thread no pipeline entered.
pub(crate) fn follow() {
    let current = TIMELINE.with(|slot| slot.borrow().as_ref().map(|timeline| timeline.current()));
    if let Some(current) = current {
        NUMBER.with(|number| number.set(current));
    }
}

/// The number buffers made on this thread are on.
pub(crate) fn current() -> u64 {
    NUMBER.with(Cell::get)
}

/// Makes the buffers this thread makes from now on part of timeline
/// `number` — what a queue's worker does before handing a buffer on, so
/// what the elements after it make of it is on the same one.
pub(crate) fn carry_on(number: u64) {
    NUMBER.with(|slot| slot.set(number));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_thread_follows_its_pipeline_onto_a_new_timeline_only_when_told() {
        let timeline = Timeline::new();
        std::thread::spawn({
            let timeline = Arc::clone(&timeline);
            move || {
                assert_eq!(current(), UNNUMBERED, "a thread nobody entered");
                enter(&timeline);
                let first = current();
                assert!(!timeline.is_behind(first));

                let second = timeline.begin();
                assert_eq!(current(), first, "until it follows, it is where it was");
                assert!(timeline.is_behind(first));
                follow();
                assert_eq!(current(), second);
                assert!(!timeline.is_behind(second));
            }
        })
        .join()
        .unwrap();
    }

    #[test]
    fn a_buffer_is_numbered_only_for_the_timeline_its_thread_is_on() {
        let ours = Timeline::new();
        let theirs = Timeline::new();
        ours.begin();
        std::thread::spawn(move || {
            let eos = || MediaBuffer::Eos;
            assert_eq!(Numbered::on(Some(&ours), eos()).number, UNNUMBERED);
            enter(&ours);
            assert_eq!(Numbered::on(Some(&ours), eos()).number, ours.current());
            assert_eq!(Numbered::on(Some(&theirs), eos()).number, UNNUMBERED);
            assert_eq!(Numbered::on(None, eos()).number, UNNUMBERED);
        })
        .join()
        .unwrap();
    }

    #[test]
    fn an_unnumbered_buffer_is_never_behind() {
        let timeline = Timeline::new();
        timeline.begin();
        timeline.begin();
        assert!(!timeline.is_behind(UNNUMBERED));
    }
}
