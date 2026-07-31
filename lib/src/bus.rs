use crossbeam_channel::{Receiver, Sender, unbounded};

use crate::{element::ElementType, error::Error};

#[derive(Debug)]
pub enum BusEvent {
    Eos {
        element_type: ElementType,
        name: String,
    },
    Error {
        element_type: ElementType,
        name: String,
        error: Error,
    },
    /// A `Queue` with `OverflowPolicy::DropNewest` dropped a buffer
    /// because it was full.
    Dropped {
        element_type: ElementType,
        name: String,
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

    /// Drains every event so far, printing each in a common default
    /// format (`[name] eos`, `[name] error: ...`, `[name] dropped a
    /// buffer (queue full)`). Convenience for examples and smoke tests;
    /// anything that needs to act on specific events — e.g. deciding
    /// whether an `Error` warrants a [`crate::pipeline::Pipeline::stop`]
    /// — should match on `iter()` directly instead, where `error`'s
    /// concrete variant (see [`crate::error::Error`]) is still available,
    /// not just its `Display` text.
    pub fn log_events(&self) {
        for event in self.iter() {
            match event {
                BusEvent::Error { name, error, .. } => eprintln!("[{name}] error: {error}"),
                BusEvent::Eos { name, .. } => println!("[{name}] eos"),
                BusEvent::Dropped { name, .. } => {
                    eprintln!("[{name}] dropped a buffer (queue full)")
                }
            }
        }
    }
}
