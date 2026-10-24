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

use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::Thread;

/// Wakes whichever thread last listened for it. Cloned to everything that
/// may hand that thread something; cheap to ring. A ring is kept until it
/// is [`taken`](Self::take): one rung before anyone listens is found by the
/// first look, as a crossbeam channel of one kept it.
#[derive(Clone, Default)]
pub(crate) struct Bell {
    shared: Arc<Shared>,
}

#[derive(Default)]
struct Shared {
    listener: Mutex<Option<Thread>>,
    rung: AtomicBool,
}

impl Bell {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Wakes the thread listening, if one is, and keeps the ring for its
    /// next look.
    pub(crate) fn ring(&self) {
        self.shared.rung.store(true, Ordering::Release);
        if let Some(thread) = &*self
            .shared
            .listener
            .lock()
            .unwrap_or_else(|p| p.into_inner())
        {
            thread.unpark();
        }
    }

    /// Whether it has been rung since it was last taken — taking the ring.
    pub(crate) fn take(&self) -> bool {
        self.shared.rung.swap(false, Ordering::AcqRel)
    }

    /// Makes this thread the one a ring wakes. Called before the thread
    /// looks at what it waits on, so that a ring after the look is not
    /// missed.
    pub(crate) fn listen(&self) {
        let current = std::thread::current();
        let mut listener = self
            .shared
            .listener
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        if listener.as_ref().map(Thread::id) != Some(current.id()) {
            *listener = Some(current);
        }
    }

    /// Whether anything but the one waiting holds this bell — for a list of
    /// bells to let go of those whose waiter is gone.
    pub(crate) fn is_shared(&self) -> bool {
        Arc::strong_count(&self.shared) > 1
    }
}

/// A crossbeam sender that rings a bell with each message it sends, and as
/// it goes — what feeds a source that waits on the channel with
/// [`Wait::recv`](crate::element::Wait). Sends as crossbeam's does.
pub(crate) struct Ringing<T> {
    /// Let go of in [`Drop`] before the bell is rung, so that whoever the
    /// ring wakes finds it gone.
    tx: ManuallyDrop<crossbeam_channel::Sender<T>>,
    bell: Bell,
}

impl<T> Ringing<T> {
    pub(crate) fn send(&self, message: T) -> Result<(), crossbeam_channel::SendError<T>> {
        self.tx.send(message)?;
        self.bell.ring();
        Ok(())
    }

    pub(crate) fn try_send(&self, message: T) -> Result<(), crossbeam_channel::TrySendError<T>> {
        self.tx.try_send(message)?;
        self.bell.ring();
        Ok(())
    }
}

impl<T> Clone for Ringing<T> {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
            bell: self.bell.clone(),
        }
    }
}

impl<T> Drop for Ringing<T> {
    /// The last sender gone is something for the receiver to see: it goes
    /// first, then rings. Rung while it was still there, a receiver woken
    /// found the channel as it was and waited again, for a ring that never
    /// came.
    fn drop(&mut self) {
        // SAFETY: dropped once, here, and not touched after.
        unsafe { ManuallyDrop::drop(&mut self.tx) };
        self.bell.ring();
    }
}

/// A channel of `capacity` whose sender rings the bell it is returned with.
pub(crate) fn ringing<T>(capacity: usize) -> (Ringing<T>, crossbeam_channel::Receiver<T>, Bell) {
    let (tx, rx) = crossbeam_channel::bounded(capacity);
    let bell = Bell::new();
    (
        Ringing {
            tx: ManuallyDrop::new(tx),
            bell: bell.clone(),
        },
        rx,
        bell,
    )
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
    fn a_ring_with_nobody_listening_is_kept_for_the_first_look() {
        let bell = Bell::new();
        assert!(!bell.take());
        bell.ring();
        assert!(bell.take(), "kept");
        assert!(!bell.take(), "and taken");
        let other = bell.clone();
        assert!(bell.is_shared());
        drop(other);
        assert!(!bell.is_shared());
    }

    /// The ring a sender gives as it goes comes once it has gone: whoever
    /// it wakes finds the channel without it. Rung first, a receiver woken
    /// found its sender still there, waited again, and was never woken for
    /// it going — an offline render whose inputs had all ended waited for
    /// them for ever.
    #[test]
    fn a_sender_going_rings_once_it_has_gone() {
        for _ in 0..200 {
            let (tx, rx, bell) = ringing::<()>(1);
            let (ready_tx, ready_rx) = std::sync::mpsc::channel();
            let looking = std::thread::spawn(move || {
                ready_tx.send(()).unwrap();
                while !bell.take() {
                    std::hint::spin_loop();
                }
                rx.try_recv()
            });
            // Looking before it goes, so that the ring is seen at once.
            ready_rx.recv().unwrap();
            drop(tx);
            assert_eq!(
                looking.join().unwrap(),
                Err(crossbeam_channel::TryRecvError::Disconnected)
            );
        }
    }
}
