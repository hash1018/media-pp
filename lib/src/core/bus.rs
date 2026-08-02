use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use crossbeam_channel::{Receiver, Sender, unbounded};
use rust_hlog::{HLog, herror, hinfo, hwarn};

use crate::{element::ElementType, error::Error};

#[derive(Debug)]
pub enum BusEvent {
    Eos {
        element_type: ElementType,
        name: Arc<str>,
    },
    Error {
        element_type: ElementType,
        name: Arc<str>,
        error: Error,
    },
    /// A `Queue` with `OverflowPolicy::DropNewest` dropped a buffer
    /// because it was full.
    Dropped {
        element_type: ElementType,
        name: Arc<str>,
    },
    /// Posted by [`crate::control::drain_control`] once
    /// [`crate::element::SourceElement::seek`] returns — `requested` is
    /// whatever [`crate::pipeline::Pipeline::seek`] was called with;
    /// `landed` is where the source actually ended up, which the source
    /// itself has to resolve (e.g. `FileDemuxer` can only reposition to a
    /// keyframe at or before `requested`, never exactly on top of an
    /// arbitrary timestamp — see its `seek` impl). Watch this instead of
    /// assuming `requested` took effect verbatim.
    Seeked {
        element_type: ElementType,
        name: Arc<str>,
        requested: Duration,
        landed: Duration,
    },
}

/// Cross-thread event channel. Once a buffer crosses a `Queue` boundary,
/// errors can no longer be propagated up the call stack with `?` — they're
/// posted here instead so the owner of the `Pipeline` can observe them.
#[derive(Clone)]
pub struct Bus {
    tx: Sender<BusEvent>,
    /// One entry per [`crate::pipeline::ChainBuilder`] chain built through
    /// this `Bus` (i.e. one per `.build()` call) — shared via `Arc` so
    /// every clone handed to a branch's own `ChainBuilder::new` writes into
    /// the same list. Read back by [`crate::pipeline::Pipeline::new`] right
    /// after `wire` returns, to assemble [`crate::pipeline::Pipeline::topology`].
    topology: Arc<Mutex<Vec<String>>>,
}

pub struct BusReceiver {
    rx: Receiver<BusEvent>,
}

impl Bus {
    pub fn new() -> (Bus, BusReceiver) {
        let (tx, rx) = unbounded();
        (
            Bus {
                tx,
                topology: Arc::new(Mutex::new(Vec::new())),
            },
            BusReceiver { rx },
        )
    }

    /// Records one finished chain's element list, in build order — called
    /// by [`crate::pipeline::ChainBuilder::build`]. Not meant to be called
    /// directly.
    pub(crate) fn register_branch(&self, description: String) {
        self.topology.lock().unwrap().push(description);
    }

    /// Every branch description recorded so far via
    /// [`Bus::register_branch`], in the order each `ChainBuilder::build`
    /// call finished.
    pub(crate) fn topology(&self) -> Vec<String> {
        self.topology.lock().unwrap().clone()
    }

    /// `hlog` is the posting element's own [`crate::element::Element::hlog`]
    /// — used (via `rust_hlog`'s `hlog:` macro form) instead of `event`'s
    /// own `name` so a real `sub_id`, if that element's `HLog` was built
    /// with one, actually reaches the log line.
    pub fn post(&self, hlog: &HLog, event: BusEvent) {
        // `log`'s own facade already no-ops (skipping the `format!` too)
        // when nothing has called `crate::log::init` — same reasoning as
        // the old hand-rolled check this replaced, just built into `log`
        // itself instead of written here.
        match &event {
            BusEvent::Eos { .. } => hinfo!(hlog: hlog, "eos"),
            BusEvent::Error { error, .. } => herror!(hlog: hlog, "{error}"),
            BusEvent::Dropped { .. } => {
                hwarn!(hlog: hlog, "dropped a buffer (queue full)")
            }
            BusEvent::Seeked {
                requested, landed, ..
            } => hinfo!(hlog: hlog, "seeked: requested {requested:.2?}, landed {landed:.2?}"),
        }
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
    /// buffer (queue full)`, `[name] seeked: requested ... landed ...`).
    /// Convenience for examples and smoke tests; anything that needs to
    /// act on specific events — e.g. deciding whether an `Error` warrants
    /// a [`crate::pipeline::Pipeline::stop`] — should match on `iter()`
    /// directly instead, where `error`'s concrete variant (see
    /// [`crate::error::Error`]) is still available, not just its
    /// `Display` text.
    pub fn log_events(&self) {
        for event in self.iter() {
            match event {
                BusEvent::Error { name, error, .. } => eprintln!("[{name}] error: {error}"),
                BusEvent::Eos { name, .. } => println!("[{name}] eos"),
                BusEvent::Dropped { name, .. } => {
                    eprintln!("[{name}] dropped a buffer (queue full)")
                }
                BusEvent::Seeked {
                    name,
                    requested,
                    landed,
                    ..
                } => println!("[{name}] seeked: requested {requested:.2?}, landed {landed:.2?}"),
            }
        }
    }
}
