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

use crate::{
    element::{BoxSink, RawSink},
    error::Result,
};

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
    /// The track's H.264 or H.265 pictures changed partway — another
    /// size, profile, chroma format or bit depth — and the output keeps
    /// the parameter sets it began with in its header, as an MP4 does.
    /// Reported once; the track writes nothing more, and the output is
    /// finished with what came before as the track ends.
    #[error(
        "track {track} changed its pictures partway, from {was} to {now}; its output keeps the first in its header"
    )]
    FormatChanged {
        /// The track's name, as its `add_stream` gave it.
        track: String,
        /// What its pictures were.
        was: String,
        /// What they changed to.
        now: String,
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
    sinks: Vec<Option<Box<dyn RawSink>>>,
}

impl MuxerSinks {
    /// The sink for `track`.
    ///
    /// Fails for a track added to a different muxer; `track` is consumed
    /// either way. A track of this muxer always has its sink here, since
    /// the only way to ask for one is with the track itself.
    pub fn take(&mut self, track: MuxerTrack) -> Result<BoxSink> {
        let foreign = MuxerTrackError::ForeignTrack { index: track.index };
        if track.muxer != self.muxer {
            return Err(foreign.into());
        }
        self.sinks
            .get_mut(track.index)
            .and_then(Option::take)
            .map(BoxSink::from_raw)
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
    pub(super) fn sinks(self, sinks: Vec<Box<dyn RawSink>>) -> MuxerSinks {
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

    fn counter(name: &str) -> Box<dyn RawSink> {
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

    /// What an encoder or a demuxed stream converts into is its own
    /// parameters and time base, not some default: a muxer told a different
    /// time base than its packets are stamped in writes them at the wrong
    /// times.
    #[test]
    fn a_track_format_is_what_its_encoder_or_stream_says() {
        use crate::elements::{AudioCodec, FileDemuxer, SwAudioEncoder, SwAudioEncoderOptions};

        let encoder = SwAudioEncoder::new(
            "encoder",
            SwAudioEncoderOptions {
                codec: AudioCodec::Aac,
                sample_rate: 44_100,
                channels: 2,
                bit_rate: 128_000,
            },
        )
        .expect("AAC is built into FFmpeg");
        let format = crate::elements::StreamInfo::from(&encoder);
        assert_eq!(format.time_base(), encoder.time_base());
        assert_eq!(format.parameters().id(), encoder.parameters().id());

        let Some(path) = crate::test_support::try_test_video() else {
            return;
        };
        let (_source, streams) = FileDemuxer::open("demux", &path).expect("open the fixture");
        for stream in &streams {
            let format = crate::elements::StreamInfo::from(stream);
            assert_eq!(format.time_base(), stream.time_base());
            assert_eq!(format.parameters().id(), stream.parameters().id());
        }
    }
}
