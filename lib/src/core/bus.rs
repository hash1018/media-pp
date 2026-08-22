//! Out-of-band reporting for what cannot be returned to the caller.
//!
//! Inside a single thread a failing stage propagates with `?`. Once a buffer
//! has crossed a [`Queue`](crate::queue::Queue) there is no longer a caller
//! to return to, so the element posts a [`BusEvent`] here instead and keeps
//! running. Whoever owns the pipeline observes those events through
//! [`BusReceiver`].
//!
//! Every event is delivered as a [`BusMessage`] carrying the stable graph
//! identity of the element that posted it, so a report can be attributed to
//! one branch even when several elements share a name.

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
    /// An element completed ordered end-of-stream processing.
    Eos {
        /// Built-in kind of the element that completed.
        element_type: ElementType,
        /// Caller-selected instance name of the element that completed.
        name: Arc<str>,
    },

    /// An element encountered a failure that could not be returned through
    /// the synchronous call stack.
    Error {
        /// Built-in kind of the element reporting the failure.
        element_type: ElementType,
        /// Caller-selected instance name of the element reporting the failure.
        name: Arc<str>,
        /// Typed crate or component error reported by the element.
        error: Error,
    },
    /// A `Queue` with `OverflowPolicy::DropNewest` dropped a buffer
    /// because it was full.
    Dropped {
        /// Built-in kind of the queue that dropped the buffer.
        element_type: ElementType,
        /// Caller-selected instance name of the queue that dropped the buffer.
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
        /// Built-in kind of the source that performed the seek.
        element_type: ElementType,
        /// Caller-selected instance name of the source that performed the seek.
        name: Arc<str>,
        /// Absolute media position requested by [`crate::pipeline::Pipeline::seek`].
        requested: Duration,
        /// Absolute media position at which the source actually resumed.
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

/// The receiving half of a [`Bus`], held by whoever owns the pipeline.
///
/// Draining it blocks until every `Bus` sender has been dropped, which is how
/// a caller waits for a pipeline to actually finish rather than polling for it.
pub struct BusReceiver {
    rx: Receiver<BusMessage>,
}

/// One bus event together with the stable graph identity of the element
/// that posted it. Drivers and standalone elements that do not belong to a
/// `PipelineGraph` use `None`.
#[derive(Debug)]
pub struct BusMessage {
    /// Stable graph identity of the posting element, or `None` for a driver or
    /// standalone element outside a pipeline graph.
    pub element_id: Option<ElementId>,

    /// Event payload posted by the element.
    pub event: BusEvent,
}

impl Bus {
    /// Creates an unbounded event channel with no graph element identity.
    ///
    /// Pipeline construction derives element-specific senders internally.
    /// Standalone elements and drivers can use the returned sender directly;
    /// their [`BusMessage::element_id`] remains `None`.
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

    /// Logs and enqueues an event without blocking on the receiver.
    ///
    /// If the receiving half has already been dropped, the event is discarded;
    /// posting never turns pipeline teardown into another error.
    ///
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
    /// Blocks until the next event arrives.
    ///
    /// Returns `None` only after every corresponding [`Bus`] sender has been
    /// dropped and all already-queued events have been received. This discards
    /// the posting element's stable graph ID; use [`Self::recv_message`] when
    /// duplicate element names must be distinguished.
    pub fn recv(&self) -> Option<BusEvent> {
        self.recv_message().map(|message| message.event)
    }

    /// Receives one currently queued event without blocking.
    ///
    /// Returns `None` both when the channel is currently empty and when every
    /// sender has disconnected. Use [`Self::try_recv_message`] to retain the
    /// posting element's stable graph ID.
    pub fn try_recv(&self) -> Option<BusEvent> {
        self.try_recv_message().map(|message| message.event)
    }

    /// Iterates over events until every corresponding [`Bus`] sender drops.
    ///
    /// The iterator blocks while the channel is still connected but empty.
    /// It discards stable graph IDs; use [`Self::iter_with_ids`] when duplicate
    /// element names must be distinguished.
    pub fn iter(&self) -> impl Iterator<Item = BusEvent> + '_ {
        self.iter_with_ids().map(|message| message.event)
    }

    /// Blocks until the next event and its stable posting-element ID arrive.
    ///
    /// Returns `None` after the channel disconnects and its queued messages
    /// have been drained.
    pub fn recv_message(&self) -> Option<BusMessage> {
        self.rx.recv().ok()
    }

    /// Receives one currently queued message without blocking.
    ///
    /// Returns `None` for both an empty connected channel and a disconnected
    /// channel.
    pub fn try_recv_message(&self) -> Option<BusMessage> {
        self.rx.try_recv().ok()
    }

    /// Iterates over messages, preserving stable graph element IDs, until all
    /// corresponding [`Bus`] senders have been dropped.
    ///
    /// The iterator blocks while the channel remains connected but empty.
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
