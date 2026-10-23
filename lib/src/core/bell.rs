//! [`Bell`]: what wakes a thread waiting on several things at once — a
//! source's, between what it makes — when one of them has something.
//!
//! A thread waiting on two channels at once — its pipeline's requests and
//! its own input — could block in a crossbeam `Select`, and once did.
//! Crossbeam spins and yields before it parks a thread, and again before
//! each park it times; for a thread that waits seventy times a second, as a
//! camera's source does between pictures and sound, that cost more than
//! everything else it did — a hundred cameras' sources spent one and a half
//! cores waiting. A bell parks the thread at once instead, and whatever
//! hands it something rings the bell, which unparks it.
//!
//! The thread looks at what it waits on before it parks, and the bell knows
//! the thread from the moment it [`listens`](Bell::listen): a ring between
//! the look and the park leaves the park's token, so the park returns at
//! once, and nothing is missed. A ring with nothing behind it — the thread
//! woken by another ring, or by anything else that unparks it — is looked
//! at and waited past.

use std::sync::{Arc, Mutex};
use std::thread::Thread;

/// Wakes whichever thread last listened for it. Cloned to everything that
/// may hand that thread something; cheap to ring, and a ring with nobody
/// listening does nothing.
#[derive(Clone, Default)]
pub(crate) struct Bell {
    listener: Arc<Mutex<Option<Thread>>>,
}

impl Bell {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Wakes the thread listening, if one is.
    pub(crate) fn ring(&self) {
        if let Some(thread) = &*self.listener.lock().unwrap_or_else(|p| p.into_inner()) {
            thread.unpark();
        }
    }

    /// Makes this thread the one a ring wakes. Called before the thread
    /// looks at what it waits on, so that a ring after the look is not
    /// missed.
    pub(crate) fn listen(&self) {
        let current = std::thread::current();
        let mut listener = self.listener.lock().unwrap_or_else(|p| p.into_inner());
        if listener.as_ref().map(Thread::id) != Some(current.id()) {
            *listener = Some(current);
        }
    }

    /// Whether anything but the one waiting holds this bell — for a list of
    /// bells to let go of those whose waiter is gone.
    pub(crate) fn is_shared(&self) -> bool {
        Arc::strong_count(&self.listener) > 1
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;

    /// A ring wakes the thread listening at once, and one rung before the
    /// park is not missed.
    #[test]
    fn a_ring_wakes_the_listener_even_before_it_parks() {
        let bell = Bell::new();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let waiter = {
            let bell = bell.clone();
            std::thread::spawn(move || {
                bell.listen();
                ready_tx.send(()).unwrap();
                // Rung while this sleeps: the park that follows returns.
                std::thread::sleep(Duration::from_millis(100));
                let parked = Instant::now();
                std::thread::park_timeout(Duration::from_secs(10));
                parked.elapsed()
            })
        };
        ready_rx.recv().unwrap();
        bell.ring();
        assert!(waiter.join().unwrap() < Duration::from_secs(5));
    }

    #[test]
    fn a_ring_with_nobody_listening_does_nothing() {
        let bell = Bell::new();
        bell.ring();
        let other = bell.clone();
        assert!(bell.is_shared());
        drop(other);
        assert!(!bell.is_shared());
    }
}
