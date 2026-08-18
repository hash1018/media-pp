use std::{sync::Arc, time::Duration};

use crate::pp_log::{PpLog, pp_error, pp_info, pp_warn};
use crossbeam_channel::{Receiver, Sender, unbounded};

use crate::{element::ElementType, error::Error, graph::ElementId};

/// Marked `#[non_exhaustive]`: this grows as elements gain conditions worth
/// reporting, and a caller acting on `Eos`/`Error` should not stop compiling
/// because some unrelated element learned to report a stall. Within this crate
/// the attribute has no effect, so [`Bus::post`] and
/// [`BusReceiver::log_events`] still fail to compile until they handle a new
/// variant — the completeness check stays where it belongs.
#[derive(Debug)]
#[non_exhaustive]
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
    /// [`crate::element::SourceElement::seek`] returns — `requested` is`1`
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
    tx: Sender<BusMessage>,
    element_id: Option<ElementId>,
}

pub struct BusReceiver {
    rx: Receiver<BusMessage>,
}

/// One bus event together with the stable graph identity of the element
/// that posted it. Drivers and standalone elements that do not belong to a
/// `PipelineGraph` use `None`.
#[derive(Debug)]
pub struct BusMessage {
    pub element_id: Option<ElementId>,
    pub event: BusEvent,
}

impl Bus {
    pub fn new() -> (Bus, BusReceiver) {
        let (tx, rx) = unbounded();
        (
            Bus {
                tx,
                element_id: None,
            },
            BusReceiver { rx },
        )
    }

    pub(crate) fn for_element(&self, element_id: ElementId) -> Bus {
        Bus {
            tx: self.tx.clone(),
            element_id: Some(element_id),
        }
    }

    /// `pp_log` is the posting element's own [`crate::element::Element::pp_log`]
    /// — used (via `crate::pp_log`'s `pp_log:` macro form) instead of `event`'s
    /// own `name` so the element's full identity, pipeline id included, reaches
    /// the log record rather than just the name carried in the event.
    pub fn post(&self, pp_log: &PpLog, event: BusEvent) {
        // Each `pp_*` macro checks `crate::log::enabled` before evaluating its
        // arguments, so posting to a bus nobody is logging costs no `format!`
        // — no hand-rolled check needed here.
        match &event {
            BusEvent::Eos { .. } => {
                pp_info!(pp_log: pp_log, "event=eos phase=reported")
            }
            BusEvent::Error { error, .. } => pp_error!(pp_log: pp_log, "{error}"),
            BusEvent::Dropped { .. } => {
                pp_warn!(pp_log: pp_log, "dropped a buffer (queue full)")
            }
            BusEvent::Seeked {
                requested, landed, ..
            } => pp_info!(pp_log: pp_log, "seeked: requested {requested:.2?}, landed {landed:.2?}"),
        }
        // Nothing to do if the receiving end is gone (pipeline dropped).
        let _ = self.tx.send(BusMessage {
            element_id: self.element_id,
            event,
        });
    }
}

impl BusReceiver {
    pub fn recv(&self) -> Option<BusEvent> {
        self.recv_message().map(|message| message.event)
    }

    pub fn try_recv(&self) -> Option<BusEvent> {
        self.try_recv_message().map(|message| message.event)
    }

    pub fn iter(&self) -> impl Iterator<Item = BusEvent> + '_ {
        self.iter_with_ids().map(|message| message.event)
    }

    pub fn recv_message(&self) -> Option<BusMessage> {
        self.rx.recv().ok()
    }

    pub fn try_recv_message(&self) -> Option<BusMessage> {
        self.rx.try_recv().ok()
    }

    pub fn iter_with_ids(&self) -> impl Iterator<Item = BusMessage> + '_ {
        self.rx.iter()
    }

    /// Blocks and prints events in a common default format (`[name] eos`,
    /// `[name] error: ...`, `[name] dropped a buffer (queue full)`,
    /// `[name] seeked: requested ... landed ...`) until every corresponding
    /// [`Bus`] sender has been dropped. This consumes both events already
    /// queued and events posted while the call is waiting; use
    /// [`BusReceiver::try_recv`] to drain only what is currently available.
    ///
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
