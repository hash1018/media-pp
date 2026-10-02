//! The stream plane: what describes a stream travels in it, in order with
//! its buffers — see `docs/stream-events.md`.
//!
//! Two events: the [`Segment`], and the end of the stream. Every stream
//! begins with a segment,
//! every seek begins one again, and so do a looping file at each lap and a
//! [`PipelineBridge`](crate::elements::PipelineBridge) where its feeding
//! side flushed or another input's stream begins — those last not on a
//! timeline of their own, since no seek of this pipeline began them. It
//! says which timeline it is on, whether it follows a flush, where it
//! begins both as a caller names a place in the media and on the timeline
//! its buffers are stamped on — two places a looping file keeps a lap or
//! more apart — and, for an accurate seek, where on that timeline what is
//! shown begins, which is what a decoder drops what comes before by. And
//! every stream that ends ends with [`StreamEvent::Eos`], behind its last
//! buffer: stateful elements hand on what they still hold ahead of it, and
//! a muxer finishes its file on it. Unlike a `Stop`, it asks for the stream
//! to be completed rather than abandoned.
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
//! one past its capacity.
//!
//! A branch joined to a stream already under way — one attached to a `Tee`,
//! a line a bin or a rack fills anew — is handed the stream's last segment
//! before anything else, so no element is handed a buffer outside one.
//!
//! # A seek's flush
//!
//! The `Flush` a seek begins with, the pipeline's own, puts every pad it
//! passes and every queue's worker into flushing: from then until the
//! flushed segment the source begins after the seek comes the same way,
//! whatever arrives there is dropped, buffers and events alike. What a
//! thread was still handing on from the position the seek left — a source
//! reading on a moment too long, a buffer a queue took as the flush went
//! past — goes no further than the first place the flush has reached, and
//! nothing has to say which seek a buffer came from. A `Flush` from
//! anywhere else — an element driven by hand, a bridge passing its feeding
//! side's on — is only the elements' to react to, and flushes no pad.
//!
//! # Reacting to one
//!
//! An element reacts through
//! [`RawSink::stream_event`](crate::element::RawSink::stream_event): a filter by
//! pushing what it answers the event with — a decoder what it still holds
//! at the end — which the framework then follows with the event itself
//! through the filter's pads. A [`Filter`](crate::element::Filter) or
//! a [`Sink`](crate::element::Sink) is never handed one: the framework
//! drains, resets and ends it.

use std::{fmt, sync::Arc, time::Duration};

use crate::{
    buffer::MediaBuffer,
    error::Result,
    pad::SrcPad,
    pp_log::{PpLog, pp_trace},
};

/// Something about the stream, carried in it — see this module's docs.
#[derive(Clone)]
#[non_exhaustive]
pub enum StreamEvent {
    /// A run of buffers on one timeline begins: everything after it, up to
    /// the next, belongs to it.
    Segment(Arc<Segment>),
    /// The stream ends: nothing follows it but another stream's segment.
    Eos,
}

/// Where a run of buffers begins — see this module's docs. Made by the
/// source that begins it; an element only reads one.
#[non_exhaustive]
pub struct Segment {
    /// Which of the pipeline's timelines it is on — which seek's, and one
    /// for the stream as it starts: the pipeline's own count, which each
    /// seek moves on before it flushes. A lap of a looping file and what a
    /// bridge begins are on the timeline they come in, a seek not having
    /// begun them.
    pub id: u64,
    /// Whether it follows a flush, so that what an element holds from
    /// before it belongs to a timeline the pipeline has left.
    pub flushed: bool,
    /// Where it begins, as a caller names a place in the media — a seek's
    /// target.
    pub position: Duration,
    /// The same place on the timeline its buffers are stamped on, which a
    /// looping file carries a lap further on for every lap played.
    pub start: Duration,
    /// Where what is shown begins, on the same timeline, for an accurate
    /// seek: the source lands on a keyframe before it and what is decoded
    /// from there up to it is only there to decode what follows. `None`
    /// where everything is shown from wherever the source landed — a
    /// stream starting, a seek to a keyframe.
    pub show_from: Option<Duration>,
    /// Whether its buffers come played backwards — a picture's stretches
    /// read from the end, their `pts` going down. A rate that changes
    /// without turning round begins no segment, so this says only which
    /// way, and the playback clock says how fast.
    pub backwards: bool,
}

/// Read by the trace records at each boundary an event crosses.
impl fmt::Display for StreamEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Segment(segment) => write!(
                f,
                "segment id={} flushed={} position={:?} start={:?} show_from={:?} backwards={}",
                segment.id,
                segment.flushed,
                segment.position,
                segment.start,
                segment.show_from,
                segment.backwards
            ),
            Self::Eos => f.write_str("eos"),
        }
    }
}

/// A buffer or an event: what a stream is made of, in the order it goes.
#[derive(Clone)]
pub(crate) enum Item {
    Buffer(MediaBuffer),
    Event(StreamEvent),
}

/// A place in a source's media, as
/// [`SeekableSource::on_timeline`](crate::element::SeekableSource::on_timeline)
/// is asked about it: a type only this crate can name, which keeps that
/// hook this crate's own.
#[derive(Clone, Copy)]
pub struct Position(pub(crate) Duration);

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

/// Begins `segment` on every one of `pads` — what a source's thread does as
/// it starts, and again as it applies a seek. Traced pad by pad, as a source's `Eos` is: this is where the
/// segment enters the graph.
pub(crate) fn begin_segment(pads: &mut [SrcPad], segment: Segment, pp_log: &PpLog) -> Result<()> {
    let event = StreamEvent::Segment(Arc::new(segment));
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

/// Hands `event` to a filter driven by hand as the graph would: its own
/// reaction, then the event on through its pads, a failure in either
/// keeping the other from none of them.
#[cfg(test)]
pub(crate) fn deliver<F: crate::element::RawFilter + ?Sized>(
    filter: &mut F,
    event: &StreamEvent,
) -> Result<()> {
    let reacted = filter.stream_event(event);
    let forwarded = forward(filter.src_pads(), event);
    reacted.and(forwarded)
}
