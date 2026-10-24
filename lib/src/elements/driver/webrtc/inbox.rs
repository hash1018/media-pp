//! What a connection has received on one track, for the source that
//! hands it on: [`Inbox`].

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex, MutexGuard},
    task::Waker,
};

use crate::buffer::MediaBuffer;
use crate::core::bell::Bell;

/// Packets one track of a connection has received, from the thread that
/// drives the connection to the source that hands them on — on a thread of
/// its own, which waits on [`Self::bell`], or driven, which leaves its
/// poll's waker here. Bounded: a source that does not keep up has the
/// newest let go of, as live media is.
#[derive(Clone)]
pub(super) struct Inbox(Arc<Shared>);

struct Shared {
    state: Mutex<State>,
    /// Rung at every packet and at the end, for a source waiting on a
    /// thread of its own — one for all the inboxes a source reads.
    bell: Bell,
}

struct State {
    queue: VecDeque<MediaBuffer>,
    capacity: usize,
    /// The connection has let go of the track: once what is held is handed
    /// on, the stream ends.
    closed: bool,
    /// The source has gone: nothing is taken any more.
    reader_gone: bool,
    waker: Option<Waker>,
}

/// What [`Inbox::push`] did.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Delivered {
    Taken,
    /// Full: let go of.
    Full,
    /// Its source is gone.
    Gone,
}

/// What a source finds in its inbox.
pub(super) enum Taken {
    Packet(MediaBuffer),
    /// Nothing yet.
    Empty,
    /// Nothing, and nothing more will come.
    Ended,
}

impl Inbox {
    #[cfg(test)]
    pub(super) fn new(capacity: usize) -> Self {
        Self::ringing(capacity, &Bell::new())
    }

    /// `count` inboxes rung by one bell, for a source that reads them all.
    pub(super) fn group(count: usize, capacity: usize) -> Vec<Self> {
        let bell = Bell::new();
        (0..count).map(|_| Self::ringing(capacity, &bell)).collect()
    }

    /// One rung by `bell`, which the other inboxes of the source that reads
    /// it share — for tracks that arrive one at a time, and are read
    /// together.
    pub(super) fn sharing(capacity: usize, bell: &Bell) -> Self {
        Self::ringing(capacity, bell)
    }

    fn ringing(capacity: usize, bell: &Bell) -> Self {
        Self(Arc::new(Shared {
            state: Mutex::new(State {
                queue: VecDeque::new(),
                capacity: capacity.max(1),
                closed: false,
                reader_gone: false,
                waker: None,
            }),
            bell: bell.clone(),
        }))
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.0
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Wakes whatever waits on this inbox — outside the lock.
    fn ring(&self, waker: Option<Waker>) {
        if let Some(waker) = waker {
            waker.wake();
        }
        self.0.bell.ring();
    }

    pub(super) fn push(&self, buf: MediaBuffer) -> Delivered {
        let waker = {
            let mut state = self.lock();
            if state.reader_gone {
                return Delivered::Gone;
            }
            if state.queue.len() >= state.capacity {
                return Delivered::Full;
            }
            state.queue.push_back(buf);
            state.waker.take()
        };
        self.ring(waker);
        Delivered::Taken
    }

    /// The connection has let go of the track: its source ends once it has
    /// handed on what is held.
    pub(super) fn close(&self) {
        let waker = {
            let mut state = self.lock();
            state.closed = true;
            state.waker.take()
        };
        self.ring(waker);
    }

    /// The next packet, or whether more may come — leaving `waker`, where
    /// there is one, to be woken when it does.
    pub(super) fn take(&self, waker: Option<&Waker>) -> Taken {
        let mut state = self.lock();
        if let Some(buf) = state.queue.pop_front() {
            return Taken::Packet(buf);
        }
        if state.closed {
            return Taken::Ended;
        }
        if let Some(waker) = waker
            && !state
                .waker
                .as_ref()
                .is_some_and(|left| left.will_wake(waker))
        {
            state.waker = Some(waker.clone());
        }
        Taken::Empty
    }

    /// What a source on a thread of its own waits on: rung at every packet
    /// and at the end.
    pub(crate) fn bell(&self) -> &Bell {
        &self.0.bell
    }

    /// The source has gone: what is held is let go of, and nothing more
    /// is taken.
    pub(super) fn reader_gone(&self) {
        let mut state = self.lock();
        state.reader_gone = true;
        state.queue.clear();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::Wake;

    use super::*;

    fn packet() -> MediaBuffer {
        MediaBuffer::packet(ffmpeg_next::Packet::copy(&[1]))
    }

    struct Count(AtomicUsize);

    impl Wake for Count {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// A packet wakes the waker left by a source that found nothing, and
    /// the end does too, once what was held has been handed on.
    #[test]
    fn a_packet_and_the_end_wake_a_waiting_source() {
        let inbox = Inbox::new(4);
        let count = Arc::new(Count(AtomicUsize::new(0)));
        let waker = Waker::from(Arc::clone(&count));
        assert!(matches!(inbox.take(Some(&waker)), Taken::Empty));
        assert_eq!(inbox.push(packet()), Delivered::Taken);
        assert_eq!(count.0.load(Ordering::SeqCst), 1);
        assert!(inbox.bell().take(), "and rings the bell");
        assert!(matches!(inbox.take(Some(&waker)), Taken::Packet(_)));
        assert!(matches!(inbox.take(Some(&waker)), Taken::Empty));
        assert_eq!(inbox.push(packet()), Delivered::Taken);
        inbox.close();
        assert!(matches!(inbox.take(None), Taken::Packet(_)));
        assert!(matches!(inbox.take(None), Taken::Ended));
    }

    #[test]
    fn full_it_lets_go_of_the_newest_and_gone_it_takes_nothing() {
        let inbox = Inbox::new(1);
        assert_eq!(inbox.push(packet()), Delivered::Taken);
        assert_eq!(inbox.push(packet()), Delivered::Full);
        inbox.reader_gone();
        assert_eq!(inbox.push(packet()), Delivered::Gone);
    }
}
