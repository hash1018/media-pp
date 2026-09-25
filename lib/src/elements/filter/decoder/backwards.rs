//! What a video decoder holds of a stretch played backwards.
//!
//! Played backwards, each video decoder here is a
//! [`ReversibleSink`](crate::element::ReversibleSink): told where each
//! stretch begins and ends, it holds the pictures of one and hands them on
//! last first. They all hold them in this, so which decoder a pipeline has
//! does not change what reverse looks like.

use crate::buffer::MediaBuffer;

#[derive(Default)]
pub(super) struct Stretch {
    /// Whether a stretch is under way: between `begin` and `end`.
    holding: bool,
    /// Its pictures, in the order decoded.
    held: Vec<MediaBuffer>,
}

impl Stretch {
    /// A stretch begins: what is decoded from here is held.
    pub(super) fn begin(&mut self) {
        self.holding = true;
    }

    /// Whether a decoded picture is to be held rather than handed on.
    pub(super) fn holding(&self) -> bool {
        self.holding
    }

    /// Keeps a decoded picture of the stretch under way.
    pub(super) fn hold(&mut self, buffer: MediaBuffer) {
        self.held.push(buffer);
    }

    /// The stretch is over: its pictures, last first.
    pub(super) fn end(&mut self) -> impl Iterator<Item = MediaBuffer> + use<> {
        self.holding = false;
        std::mem::take(&mut self.held).into_iter().rev()
    }

    /// Drops the stretch under way, for a `Flush`.
    pub(super) fn reset(&mut self) {
        self.holding = false;
        self.held.clear();
    }
}
