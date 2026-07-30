use crossbeam_channel::{Receiver, Sender, unbounded};

#[derive(Debug)]
pub enum BusEvent {
    Eos {
        element: String,
    },
    Error {
        element: String,
        message: String,
    },
    /// A `Queue` with `OverflowPolicy::DropNewest` dropped a buffer
    /// because it was full.
    Dropped {
        element: String,
    },
}

/// Cross-thread event channel. Once a buffer crosses a `Queue` boundary,
/// errors can no longer be propagated up the call stack with `?` — they're
/// posted here instead so the owner of the `Pipeline` can observe them.
#[derive(Clone)]
pub struct Bus {
    tx: Sender<BusEvent>,
}

pub struct BusReceiver {
    rx: Receiver<BusEvent>,
}

impl Bus {
    pub fn new() -> (Bus, BusReceiver) {
        let (tx, rx) = unbounded();
        (Bus { tx }, BusReceiver { rx })
    }

    pub fn post(&self, event: BusEvent) {
        // Nothing to do if the receiving end is gone (pipeline dropped).
        let _ = self.tx.send(event);
    }
}

impl BusReceiver {
    pub fn recv(&self) -> Option<BusEvent> {
        self.rx.recv().ok()
    }

    pub fn try_recv(&self) -> Option<BusEvent> {
        self.rx.try_recv().ok()
    }

    pub fn iter(&self) -> impl Iterator<Item = BusEvent> + '_ {
        self.rx.iter()
    }
}
