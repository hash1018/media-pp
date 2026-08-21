//! The one output port data leaves an element through.
//!
//! There is no separate fan-out primitive in this model: an element with more
//! than one [`SrcPad`] *is* a tee. [`FileDemuxer`](crate::elements::FileDemuxer)
//! is the plain form — one pad per container stream, wired once — and
//! [`Tee`](crate::elements::Tee) is the deliberate exception whose pads can
//! change while it runs.

use std::sync::Arc;

use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{ElementType, Sink},
    error::Result,
    pp_log::{PpLog, pp_trace},
};

/// An output port an [`Element`](crate::element::Element) owns. Data only
/// ever leaves an element through one of its src pads — there is no other
/// way to reach downstream.
///
/// This is what fan-out is, in this design: an element with more than one
/// src pad *is* a tee — there's no separate "Tee" primitive in the
/// pad/element model itself. [`crate::elements::FileDemuxer`] is the
/// plain form of it: one pad per container stream, chosen once at wiring
/// time via a normal `&mut [SrcPad]`. [`crate::elements::Tee`] is the one
/// deliberate exception to that plain shape — its pads live behind a lock
/// instead, so a `TeeHandle` can add or remove one from a different
/// thread than whatever is driving it; see that module for why.
pub struct SrcPad {
    name: String,
    peer: Option<Box<dyn Sink>>,
}

impl SrcPad {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            peer: None,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn is_linked(&self) -> bool {
        self.peer.is_some()
    }

    /// The linked sink's own identity, without touching the link itself —
    /// lets a caller that just saw [`SrcPad::push`]/[`SrcPad::control`]
    /// fail (e.g. [`crate::elements::Tee`], fanning out to several pads at
    /// once) report *which* downstream element the failure actually came
    /// from, instead of only knowing its own. `None` for an unlinked pad.
    pub fn peer_identity(&self) -> Option<(ElementType, Arc<str>)> {
        self.peer
            .as_ref()
            .map(|sink| (sink.element_type(), sink.name()))
    }

    /// Runtime half of a connection. Pipeline users connect through
    /// [`crate::element::Context::attach`], which keeps the graph and this
    /// peer in sync. Kept crate-visible for element-level unit tests.
    pub(crate) fn link(&mut self, sink: Box<dyn Sink>) {
        self.peer = Some(sink);
    }

    /// Pushes a buffer to whatever this pad is linked to. Pushing into an
    /// unlinked pad silently drops the buffer (e.g. a demuxer stream
    /// nobody cared to link).
    pub fn push(&mut self, buf: MediaBuffer) -> Result<()> {
        match &mut self.peer {
            Some(sink) => sink.consume(buf),
            None => Ok(()),
        }
    }

    /// Sends a source-originated EOS with explicit pad-level trace records.
    /// Filters are traced by the pipeline's common element wrapper; this is
    /// for the source boundary where EOS first enters the dataflow graph.
    pub(crate) fn push_eos(&mut self, pp_log: &PpLog) -> Result<()> {
        if self.peer.is_none() {
            pp_trace!(
                pp_log: pp_log,
                "event=eos phase=skipped pad={} reason=unlinked",
                self.name
            );
            return Ok(());
        }

        pp_trace!(
            pp_log: pp_log,
            "event=eos phase=sending pad={}",
            self.name
        );
        let result = self.push(MediaBuffer::Eos);
        match &result {
            Ok(()) => pp_trace!(
                pp_log: pp_log,
                "event=eos phase=sent pad={} outcome=ok",
                self.name
            ),
            Err(error) => pp_trace!(
                pp_log: pp_log,
                "event=eos phase=sent pad={} outcome=error error={error}",
                self.name
            ),
        }
        result
    }

    /// Forwards a [`ControlMsg`] to whatever this pad is linked to —
    /// mirrors [`SrcPad::push`], just for control instead of data.
    /// Pushing into an unlinked pad is a no-op, same as `push`.
    pub fn control(&mut self, msg: ControlMsg) -> Result<()> {
        match &mut self.peer {
            Some(sink) => sink.control(msg),
            None => Ok(()),
        }
    }
}
