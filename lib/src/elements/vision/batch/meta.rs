//! What a [`StreamMux`](super::StreamMux) puts on every picture it hands on:
//! which input it came from, and which batch it is part of.

use std::fmt;
use std::sync::Arc;

/// One input of a [`StreamMux`](super::StreamMux), as long as it is
/// registered: a new registration under a name already in use is a new
/// stream, with state of its own wherever state is kept per stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StreamId(pub(crate) u64);

impl StreamId {
    /// The number this stream was registered under, unique for the mux.
    pub fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for StreamId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "stream {}", self.0)
    }
}

/// Which input of a [`StreamMux`](super::StreamMux) a picture came from —
/// the [`Metadata`](crate::buffer::Metadata) it puts on each. Read it with
/// `buffer.metadata()?.get::<StreamOrigin>()`.
///
/// An element that keeps state from one picture to the next — a tracker,
/// analytics, a classifier's memory — keeps it for each `id`, and starts a
/// stream over when its `generation` moves: that input's own pipeline was
/// sought, so what came before is not where it is now.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct StreamOrigin {
    /// The input.
    pub id: StreamId,
    /// The name it was registered under.
    pub name: Arc<str>,
    /// How many times its pipeline has been flushed — sought — since it was
    /// registered.
    pub generation: u64,
}

/// Which batch a picture is part of, and where in it — the
/// [`Metadata`](crate::buffer::Metadata) a [`StreamMux`](super::StreamMux)
/// puts beside [`StreamOrigin`].
///
/// The pictures of one batch are handed on one after another, in slot
/// order, with nothing between them; an element that runs a model on a
/// batch at once holds them until the last, [`BatchSlot::is_last`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct BatchSlot {
    /// The batch, counted from 0 by the mux.
    pub batch: u64,
    /// This picture's place in it, from 0.
    pub index: usize,
    /// How many pictures it holds.
    pub size: usize,
}

impl BatchSlot {
    /// Whether this is the batch's last picture.
    pub fn is_last(&self) -> bool {
        self.index + 1 == self.size
    }
}
