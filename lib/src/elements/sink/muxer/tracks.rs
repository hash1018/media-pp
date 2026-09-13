//! Which sink is which track, for every muxer.
//!
//! A muxer's tracks are declared one at a time and come back all at once,
//! since the header describing them is written between the two. What
//! connects a declaration to its sink is a [`MuxerTrack`]: each muxer's
//! `add_stream` hands one back, and [`MuxerSinks::take`] exchanges it for
//! that track's sink. A caller never has to repeat the order it added tracks
//! in — which, with a track added only sometimes, is an order that is easy
//! to get wrong, and wrong compiles: audio packets go to the video track.

use std::sync::atomic::{AtomicU64, Ordering};

use thiserror::Error as ThisError;

use crate::{element::Sink, error::Result};

/// Errors from [`MuxerSinks::take`]. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum MuxerTrackError {
    /// The track was added to a different muxer than the one these sinks
    /// were opened from.
    #[error("track {index} belongs to another muxer, not the one these sinks were opened from")]
    ForeignTrack {
        /// Where the track was in its own muxer's order.
        index: usize,
    },
}

/// One track of one muxer, as its `add_stream` registered it — the way to
/// take that track's sink out of what the muxer's `open` returns.
///
/// Neither `Clone` nor `Copy`: [`MuxerSinks::take`] consumes it, so a sink
/// taken once cannot be asked for again.
#[derive(Debug)]
#[must_use = "a track's sink can only be taken with its MuxerTrack, and an untaken track leaves the output unfinalized"]
pub struct MuxerTrack {
    muxer: MuxerId,
    index: usize,
}

/// What a muxer's `open` returns: one sink per track, each taken out by the
/// [`MuxerTrack`] its `add_stream` returned.
///
/// Take every track. A sink never taken is dropped with this, and the
/// track it stood for then never reports itself finished — the output is
/// finalized only once every track has, so it is left unfinalized.
pub struct MuxerSinks {
    muxer: MuxerId,
    sinks: Vec<Option<Box<dyn Sink>>>,
}

impl MuxerSinks {
    /// The sink for `track`.
    ///
    /// Fails for a track added to a different muxer; `track` is consumed
    /// either way. A track of this muxer always has its sink here, since
    /// the only way to ask for one is with the track itself.
    pub fn take(&mut self, track: MuxerTrack) -> Result<Box<dyn Sink>> {
        let foreign = MuxerTrackError::ForeignTrack { index: track.index };
        if track.muxer != self.muxer {
            return Err(foreign.into());
        }
        self.sinks
            .get_mut(track.index)
            .and_then(Option::take)
            .ok_or_else(|| foreign.into())
    }
}

/// Which muxer a [`MuxerTrack`] came from — unique across the process, so a
/// track of one muxer is refused by another's sinks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MuxerId(u64);

impl MuxerId {
    pub(super) fn next() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }

    /// The track at `index` in this muxer's order.
    pub(super) fn track(self, index: usize) -> MuxerTrack {
        MuxerTrack { muxer: self, index }
    }

    /// `sinks`, one per track, in the order the tracks were added.
    pub(super) fn sinks(self, sinks: Vec<Box<dyn Sink>>) -> MuxerSinks {
        MuxerSinks {
            muxer: self,
            sinks: sinks.into_iter().map(Some).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elements::PacketCounter;
    use crate::error::Error;

    fn counter(name: &str) -> Box<dyn Sink> {
        Box::new(PacketCounter::new(name).0)
    }

    /// Whatever order they are taken in, each track gets its own sink.
    #[test]
    fn a_track_takes_its_own_sink_in_any_order() {
        let muxer = MuxerId::next();
        let (video, audio) = (muxer.track(0), muxer.track(1));
        let mut sinks = muxer.sinks(vec![counter("video"), counter("audio")]);

        assert_eq!(&*sinks.take(audio).unwrap().name(), "audio");
        assert_eq!(&*sinks.take(video).unwrap().name(), "video");
    }

    /// A track from another muxer is refused, and refusing it leaves this
    /// muxer's own track where it was.
    #[test]
    fn a_track_from_another_muxer_is_refused() {
        let ours = MuxerId::next();
        let theirs = MuxerId::next();
        let mut sinks = ours.sinks(vec![counter("ours")]);

        let refused = sinks.take(theirs.track(0));
        assert!(matches!(
            refused,
            Err(Error::MuxerTrackError(MuxerTrackError::ForeignTrack {
                index: 0
            }))
        ));
        assert_eq!(&*sinks.take(ours.track(0)).unwrap().name(), "ours");
    }
}
