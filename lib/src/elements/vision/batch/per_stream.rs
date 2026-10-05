//! State kept per stream by the vision elements after a
//! [`StreamMux`](super::StreamMux), keyed by the [`StreamOrigin`] each
//! picture carries.

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use crate::buffer::MediaBuffer;

use super::{StreamId, StreamOrigin};

/// How many pictures, of every stream, a stream may go unseen before its
/// state is let go of: an input that ended, or was removed, leaves no
/// other sign here. Long enough that a stream merely stalled — a camera
/// reconnecting — keeps what it had: at eight streams of 30 pictures a
/// second, about two minutes.
const FORGET_AFTER: u64 = 30_000;

/// How often, in pictures, the streams unseen that long are looked for.
const PRUNE_EVERY: u64 = 1024;

/// Which stream a picture is of: its [`StreamOrigin`]'s id, or `None` for a
/// picture no mux handed on — the one stream of a pipeline without one.
pub(crate) fn stream_of(buf: &MediaBuffer) -> (Option<StreamId>, u64) {
    match buf
        .metadata()
        .and_then(|metadata| metadata.get::<StreamOrigin>())
    {
        Some(origin) => (Some(origin.id), origin.generation),
        None => (None, 0),
    }
}

struct Kept<T> {
    generation: u64,
    seen: u64,
    state: T,
}

/// A `T` for each stream pictures come from.
pub(crate) struct PerStream<T> {
    states: HashMap<Option<StreamId>, Kept<T>>,
    /// Pictures asked about, of every stream.
    tick: u64,
}

impl<T> Default for PerStream<T> {
    fn default() -> Self {
        Self {
            states: HashMap::new(),
            tick: 0,
        }
    }
}

impl<T> PerStream<T> {
    /// The state of the stream `buf` is of, made by `make` — handed the
    /// picture's [`StreamOrigin`], where it has one — the first time, and
    /// whether that stream's generation has moved since it was last asked
    /// about: its pipeline was sought, and what the caller keeps across a
    /// seek is the caller's to let go of.
    pub(crate) fn get(
        &mut self,
        buf: &MediaBuffer,
        make: impl FnOnce(Option<&StreamOrigin>) -> T,
    ) -> (&mut T, bool) {
        self.tick += 1;
        if self.tick.is_multiple_of(PRUNE_EVERY) {
            let now = self.tick;
            self.states
                .retain(|_, kept| now - kept.seen <= FORGET_AFTER);
        }
        let (id, generation) = stream_of(buf);
        let now = self.tick;
        match self.states.entry(id) {
            Entry::Occupied(entry) => {
                let kept = entry.into_mut();
                kept.seen = now;
                let moved = kept.generation != generation;
                kept.generation = generation;
                (&mut kept.state, moved)
            }
            Entry::Vacant(entry) => {
                let origin = buf
                    .metadata()
                    .and_then(|metadata| metadata.get::<StreamOrigin>());
                let kept = entry.insert(Kept {
                    generation,
                    seen: now,
                    state: make(origin),
                });
                (&mut kept.state, false)
            }
        }
    }

    /// Every stream's state.
    pub(crate) fn values(&self) -> impl Iterator<Item = &T> {
        self.states.values().map(|kept| &kept.state)
    }

    /// Lets go of every stream's state.
    pub(crate) fn clear(&mut self) {
        self.states.clear();
    }
}
