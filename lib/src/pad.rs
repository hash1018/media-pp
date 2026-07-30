use crate::{buffer::MediaBuffer, element::Sink, error::Result};

/// An output port an [`Element`](crate::element::Element) owns. Data only
/// ever leaves an element through one of its src pads — there is no other
/// way to reach downstream.
///
/// This is what fan-out is, in this design: an element with more than one
/// src pad (a demuxer with one pad per container stream, say) *is* a tee.
/// There's no separate "Tee" type needed.
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

    /// Links this pad to a downstream `Sink` — a plain element, a `Queue`
    /// (thread boundary), or anything else that implements `Sink`.
    /// Replaces any previous link.
    pub fn link(&mut self, sink: Box<dyn Sink>) {
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
}
