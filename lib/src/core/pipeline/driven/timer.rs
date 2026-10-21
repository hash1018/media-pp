//! What wakes a driven pipeline polled as a [`Future`](std::future::Future)
//! by the time it asked for: one thread for the whole process, started the
//! first time one is asked for, which wakes each waker it is handed once
//! its time has come.
//!
//! One for the process rather than one an executor brings, so that a
//! [`DrivenPipeline`](super::DrivenPipeline) goes to any executor as it is.
//! It lives as long as the process does: it holds nothing of a pipeline but
//! a waker until it wakes it.

use std::{
    cmp::{Ordering, Reverse},
    collections::BinaryHeap,
    sync::OnceLock,
    task::Waker,
    time::Instant,
};

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, unbounded};

/// The timer thread's channel — `None` where it could not be started, and
/// then a waker is woken at once, so a pipeline is polled early rather
/// than never.
static TIMER: OnceLock<Option<Sender<Due>>> = OnceLock::new();

/// Wakes `waker` once `at` has come.
pub(super) fn wake_at(at: Instant, waker: Waker) {
    let Some(timer) = TIMER.get_or_init(start) else {
        waker.wake();
        return;
    };
    if let Err(unsent) = timer.send(Due { at, waker }) {
        unsent.0.waker.wake();
    }
}

fn start() -> Option<Sender<Due>> {
    let (tx, rx) = unbounded();
    std::thread::Builder::new()
        .name("media-pp:driven-timer".to_owned())
        .spawn(move || run(&rx))
        .ok()
        .map(|_| tx)
}

fn run(rx: &Receiver<Due>) {
    let mut due = BinaryHeap::<Reverse<Due>>::new();
    loop {
        let now = Instant::now();
        while due.peek().is_some_and(|Reverse(next)| next.at <= now) {
            if let Some(Reverse(next)) = due.pop() {
                next.waker.wake();
            }
        }
        let received = match due.peek() {
            Some(Reverse(next)) => rx.recv_deadline(next.at),
            None => rx.recv().map_err(|_| RecvTimeoutError::Disconnected),
        };
        match received {
            Ok(next) => due.push(Reverse(next)),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => return,
        }
    }
}

/// A waker, and when to wake it — ordered by when alone.
struct Due {
    at: Instant,
    waker: Waker,
}

impl PartialEq for Due {
    fn eq(&self, other: &Self) -> bool {
        self.at == other.at
    }
}

impl Eq for Due {}

impl PartialOrd for Due {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Due {
    fn cmp(&self, other: &Self) -> Ordering {
        self.at.cmp(&other.at)
    }
}
