//! A server's publishers on one path, each handed out as a source:
//! [`Publishers`].

use std::mem::ManuallyDrop;
use std::sync::{Arc, Mutex};
use std::task::{Poll, Waker};

use crossbeam_channel::{Receiver, Sender, TryRecvError, unbounded};

use crate::elements::StreamInfo;

/// The publishers a server takes on one path: each, as it begins, handed
/// out as the source of what it sends, with its streams — one run of the
/// path's stream.
///
/// An `RtspServerHandle`'s, an `RtmpServerHandle`'s or a
/// `WebRtcServerHandle`'s `take_publishers` makes it, and the path takes
/// publishers for as long as it lives: dropped, the path takes none from
/// then on, and one publishing goes on into the source it was handed as.
/// The server takes one publisher at a time, and refuses another while it
/// publishes.
///
/// A publisher is handed out once its streams are known — what its
/// announcement leaves out read from its first packets, as the source of
/// a pull says it. What it sends meanwhile waits in the source, and what it
/// sends while nothing reads it waits as far as the source keeps, a few
/// seconds; past that a picture stream goes on from its next keyframe.
/// It ends as the publisher does — leaves, stops sending, or begins
/// anew with other streams, which is handed out again as a source of its
/// own — and dropping the source ends the publisher's connection.
///
/// Not `Clone`: each publisher goes to one place. Holds nothing of the
/// server.
pub struct Publishers<S> {
    path: String,
    rx: Receiver<(S, Vec<StreamInfo>)>,
    wakers: Arc<Wakers>,
    /// Takes the path's registration away as this goes.
    _taken: Box<dyn Send + Sync>,
}

impl<S> std::fmt::Debug for Publishers<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Publishers")
            .field("path", &self.path)
            .finish()
    }
}

/// What waits for the next publisher in [`Publishers::next_async`].
#[derive(Default)]
struct Wakers(Mutex<Vec<Waker>>);

impl Wakers {
    fn wake(&self) {
        let woken = std::mem::take(&mut *self.0.lock().unwrap_or_else(|p| p.into_inner()));
        woken.into_iter().for_each(Waker::wake);
    }
}

impl<S> Publishers<S> {
    /// The path it takes publishers on.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Blocks until the next publisher has begun and said what it sends,
    /// and answers its source and streams. `None` once the server has
    /// stopped.
    pub fn next(&self) -> Option<(S, Vec<StreamInfo>)> {
        self.rx.recv().ok()
    }

    /// [`Self::next`], waited for without a thread: what an async runtime's
    /// task awaits. Any runtime's — it is woken by the server's thread.
    pub async fn next_async(&self) -> Option<(S, Vec<StreamInfo>)> {
        std::future::poll_fn(|cx| {
            if let Some(ready) = self.ready() {
                return Poll::Ready(ready);
            }
            self.wakers
                .0
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(cx.waker().clone());
            // Looked at again with the waker in place: a publisher handed
            // out between the first look and now woke nothing.
            match self.ready() {
                Some(ready) => Poll::Ready(ready),
                None => Poll::Pending,
            }
        })
        .await
    }

    fn ready(&self) -> Option<Option<(S, Vec<StreamInfo>)>> {
        match self.rx.try_recv() {
            Ok(publisher) => Some(Some(publisher)),
            Err(TryRecvError::Disconnected) => Some(None),
            Err(TryRecvError::Empty) => None,
        }
    }
}

/// The server's side of a [`Publishers`]: where each publisher is handed
/// as it begins. Cheap to clone; the last clone let go of ends the
/// `Publishers`' wait with `None`.
pub(crate) struct Arrivals<S>(Arc<ArrivalsInner<S>>);

impl<S> Clone for Arrivals<S> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

struct ArrivalsInner<S> {
    /// Let go of in [`Drop`] before the waiting are woken, so that what is
    /// woken finds it gone.
    tx: ManuallyDrop<Sender<(S, Vec<StreamInfo>)>>,
    wakers: Arc<Wakers>,
}

impl<S> Drop for ArrivalsInner<S> {
    fn drop(&mut self) {
        // SAFETY: dropped once, here, and not touched after.
        unsafe { ManuallyDrop::drop(&mut self.tx) };
        self.wakers.wake();
    }
}

impl<S> Arrivals<S> {
    /// Hands out a publisher's source and streams. Whether anything takes
    /// them: `false` once the `Publishers` has gone.
    pub(crate) fn hand(&self, source: S, streams: Vec<StreamInfo>) -> bool {
        let handed = self.0.tx.send((source, streams)).is_ok();
        self.0.wakers.wake();
        handed
    }
}

/// A [`Publishers`] of `path` and the server's side of it; `taken` is kept
/// by the `Publishers`, and dropped as it goes — what takes the path's
/// registration away.
pub(crate) fn publishers<S>(
    path: &str,
    taken: impl Send + Sync + 'static,
) -> (Arrivals<S>, Publishers<S>) {
    let (tx, rx) = unbounded();
    let wakers = Arc::new(Wakers::default());
    (
        Arrivals(Arc::new(ArrivalsInner {
            tx: ManuallyDrop::new(tx),
            wakers: Arc::clone(&wakers),
        })),
        Publishers {
            path: path.to_owned(),
            rx,
            wakers,
            _taken: Box::new(taken),
        },
    )
}

/// Runs `f` as it is dropped: what a [`Publishers`] keeps to take its
/// path's registration away.
pub(crate) struct OnDrop<F: FnOnce()>(Option<F>);

impl<F: FnOnce()> OnDrop<F> {
    pub(crate) fn new(f: F) -> Self {
        Self(Some(f))
    }
}

impl<F: FnOnce()> Drop for OnDrop<F> {
    fn drop(&mut self) {
        if let Some(f) = self.0.take() {
            f();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::task::{Context, Wake};
    use std::time::Duration;

    use super::*;

    struct Flag(AtomicBool);

    impl Wake for Flag {
        fn wake(self: Arc<Self>) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    fn poll_once<S>(
        publishers: &Publishers<S>,
        flag: &Arc<Flag>,
    ) -> Poll<Option<(S, Vec<StreamInfo>)>> {
        let waker = Waker::from(Arc::clone(flag));
        let mut cx = Context::from_waker(&waker);
        let mut next = std::pin::pin!(publishers.next_async());
        next.as_mut().poll(&mut cx)
    }

    #[test]
    fn a_publisher_handed_out_wakes_the_wait_and_the_last_arrival_going_ends_it() {
        let (arrivals, publishers) = publishers::<u32>("cam", ());
        let flag = Arc::new(Flag(AtomicBool::new(false)));
        assert!(poll_once(&publishers, &flag).is_pending());

        assert!(arrivals.hand(7, Vec::new()));
        assert!(
            flag.0.load(Ordering::SeqCst),
            "handing a publisher out wakes"
        );
        assert!(matches!(
            poll_once(&publishers, &flag),
            Poll::Ready(Some((7, _)))
        ));

        flag.0.store(false, Ordering::SeqCst);
        assert!(poll_once(&publishers, &flag).is_pending());
        let clone = arrivals.clone();
        drop(arrivals);
        assert!(!flag.0.load(Ordering::SeqCst), "a clone still hands out");
        drop(clone);
        assert!(flag.0.load(Ordering::SeqCst), "the last going wakes");
        assert!(matches!(poll_once(&publishers, &flag), Poll::Ready(None)));
        assert!(publishers.next().is_none());
    }

    #[test]
    fn dropping_the_publishers_takes_the_registration_away_and_refuses_what_comes() {
        let taken = Arc::new(AtomicBool::new(false));
        let (arrivals, publishers) = publishers::<u32>("cam", {
            let taken = Arc::clone(&taken);
            OnDrop::new(move || taken.store(true, Ordering::SeqCst))
        });
        let waiting = std::thread::spawn({
            let arrivals = arrivals.clone();
            move || {
                std::thread::sleep(Duration::from_millis(50));
                arrivals.hand(1, Vec::new())
            }
        });
        assert!(matches!(publishers.next(), Some((1, _))));
        assert!(waiting.join().unwrap());

        drop(publishers);
        assert!(taken.load(Ordering::SeqCst));
        assert!(!arrivals.hand(2, Vec::new()), "nothing takes it any more");
    }
}
