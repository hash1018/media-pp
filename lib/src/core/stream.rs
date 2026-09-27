//! The stream plane: what describes a stream travels in it, in order with
//! its buffers — see `docs/stream-events.md`.
//!
//! So far the one event is the [`Segment`] every stream begins with and
//! every seek begins again, which says no more yet than which timeline it
//! opens and whether it follows a flush. Its buffers are on that timeline:
//! every buffer after a segment, up to the next, carries the number
//! [`crate::timeline`] gives it, and that number is the segment's `id`.
//!
//! # How an event travels
//!
//! As control does on one thread — the element's own reaction first, then
//! on through its pads, which the framework does for it — but never ahead
//! of the data. Where control jumps a backlog, an event waits its turn: a
//! [`Queue`](crate::queue::Queue) carries it in its channel behind what came
//! before it, and a [`Tee`](crate::elements::Tee) hands it to each branch
//! with the buffers, keeping it with what a preroll holds back for one. An
//! element that answers an event with buffers of its own pushes them from
//! its reaction, so they go on ahead of the event, as what an element
//! drains goes on ahead of an `Eos`.
//!
//! Events are never dropped for room and never waited for: a queue takes
//! one past its capacity. An event from a timeline the pipeline has left is
//! dropped where a buffer from it would be.
//!
//! A branch joined to a stream already under way — one attached to a `Tee`,
//! a line a bin or a rack fills anew — is handed the stream's last segment
//! before anything else, so no element is handed a buffer outside one.
//!
//! # Not public yet
//!
//! The hook an element reacts through,
//! [`Sink::stream_event`](crate::element::Sink::stream_event), takes an
//! [`Event`], which nothing outside this crate can name, so nothing outside
//! can override it until the protocol settles. An element outside the crate
//! is handed the stream through the framework's wrappers, which pass every
//! event on around it.

use std::{fmt, sync::Arc};

use crate::{
    buffer::MediaBuffer,
    error::Result,
    pad::SrcPad,
    pp_log::{PpLog, pp_trace},
};

/// Something about the stream, carried in it.
#[derive(Clone)]
pub(crate) enum StreamEvent {
    /// A run of buffers on one timeline begins: everything after it, up to
    /// the next, belongs to it.
    Segment(Arc<Segment>),
}

/// Where a run of buffers begins — see this module's docs.
pub(crate) struct Segment {
    /// The timeline it opens: the number [`crate::timeline`] gives every
    /// buffer after it.
    pub(crate) id: u64,
    /// Whether it follows a flush, so that what an element holds from
    /// before it belongs to a timeline the pipeline has left.
    pub(crate) flushed: bool,
}

impl StreamEvent {
    /// A segment opening the timeline this thread makes buffers on.
    pub(crate) fn segment(flushed: bool) -> Self {
        Self::Segment(Arc::new(Segment {
            id: crate::timeline::current(),
            flushed,
        }))
    }
}

/// Read by the trace records at each boundary an event crosses.
impl fmt::Display for StreamEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Segment(segment) => {
                write!(f, "segment id={} flushed={}", segment.id, segment.flushed)
            }
        }
    }
}

/// A buffer or an event: what a stream is made of, in the order it goes.
#[derive(Clone)]
pub(crate) enum Item {
    Buffer(MediaBuffer),
    Event(StreamEvent),
}

/// An event as [`Sink::stream_event`](crate::element::Sink::stream_event)
/// is handed it: a type only this crate can name, which keeps that hook this
/// crate's own — see this module's docs.
#[derive(Clone, Copy)]
pub struct Event<'a>(pub(crate) &'a StreamEvent);

/// Passes `event` through every one of `pads`, a failure on one keeping it
/// from none of the others, and answers the first failure — as
/// [`crate::control::forward`] does for control.
pub(crate) fn forward(pads: &mut [SrcPad], event: &StreamEvent) -> Result<()> {
    let mut first = Ok(());
    for pad in pads {
        let outcome = pad.push_event(event);
        if first.is_ok() {
            first = outcome;
        }
    }
    first
}

/// Begins a segment on every one of `pads`, on the timeline this thread is
/// on — what a source's thread does as it starts, and again as it applies a
/// seek. Traced pad by pad, as a source's `Eos` is: this is where the
/// segment enters the graph.
pub(crate) fn begin_segment(pads: &mut [SrcPad], flushed: bool, pp_log: &PpLog) -> Result<()> {
    let event = StreamEvent::segment(flushed);
    let mut first = Ok(());
    for pad in pads.iter_mut().filter(|pad| pad.is_linked()) {
        pp_trace!(
            pp_log: pp_log,
            "event={event} phase=sending pad={}",
            pad.name()
        );
        let outcome = pad.push_event(&event);
        match &outcome {
            Ok(()) => pp_trace!(
                pp_log: pp_log,
                "event={event} phase=sent pad={} outcome=ok",
                pad.name()
            ),
            Err(error) => pp_trace!(
                pp_log: pp_log,
                "event={event} phase=sent pad={} outcome=error error={error}",
                pad.name()
            ),
        }
        if first.is_ok() {
            first = outcome;
        }
    }
    first
}
