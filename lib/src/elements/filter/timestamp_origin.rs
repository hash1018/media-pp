use std::sync::Arc;

use ffmpeg_next as ffmpeg;

use crate::pp_log::{PpLog, pp_info};

use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKindSet, OutputContract, PortContract},
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    error::Result,
    pad::SrcPad,
};

/// Shifts a packet stream so that its timeline starts at zero, taking the
/// first packet it sees as the origin.
///
/// # What it is for
///
/// A branch attached to a [`Tee`](crate::elements::Tee) that is already
/// running inherits whatever timeline the source is up to. A
/// `D3d11VideoCompositor` stamps its output with a tick counter that starts
/// when *the compositor* did, so a recording branch attached ten minutes in
/// receives its first frame stamped ten minutes — and a muxer, which has no
/// reason to disbelieve it, writes a file whose first sample is ten minutes
/// from the start. Players show the leading gap as empty, and a thumbnailer
/// asking for an early frame finds nothing there.
///
/// Nothing upstream can fix that for the branch: the compositor's timeline is
/// correct, and it is shared with every other branch. What is wrong is only
/// the assumption that this branch's timeline is the same one. Putting this
/// between the encoder and the muxer makes the file start where the recording
/// did.
///
/// # Where it goes
///
/// After the encoder, on packets, rather than in front of it on frames.
/// Frames travel as pooled references whose slots the producer reuses, so
/// re-stamping one means holding its pool slot for as long as the copy is
/// downstream — a packet carries no such tie. Placed here it also shifts
/// `dts` alongside `pts`, which is the pair a muxer actually reads.
///
/// # What the origin is
///
/// The earliest timestamp on the first packet, which with B-frames is its
/// `dts` rather than its `pts`. Taking the earlier of the two is what keeps
/// the shifted stream from starting before zero: an encoder that reorders
/// emits its first packet for decoding before it is displayed, and a negative
/// `dts` is a thing muxers work around rather than something to hand them.
/// The reorder delay survives as a `pts` a frame or two above zero, which is
/// what it is.
///
/// # One stream at a time
///
/// Each of these establishes its own origin from its own first packet. Two of
/// them, on the two tracks of one file, would each start at their own first
/// packet and so disagree by up to one packet's worth of time — audio packets
/// being much longer than video ones, that is audible. A file whose tracks
/// must stay in sync needs one shared origin, which this deliberately does
/// not have: there is one track today, and a shared one is a different type
/// with a handle rather than a flag added here.
///
/// # Cost
///
/// One packet copy each, to leave the `Arc` this was handed untouched — a
/// sibling branch off the same `Tee` must not see this branch's timeline.
/// The same copy [`crate::elements::Mp4Muxer`] already makes to rescale.
pub struct TimestampOrigin {
    pp_log: PpLog,
    name: Arc<str>,
    /// The first timestamp seen, subtracted from every packet after it.
    /// `None` until the first packet that carries one.
    origin: Option<i64>,
    pad: SrcPad,
}

impl TimestampOrigin {
    pub fn new(name: impl Into<String>) -> Self {
        let name: Arc<str> = name.into().into();
        let pad = SrcPad::with_contract(format!("{name}_src"), OutputContract::Passthrough);
        let pp_log = element_pp_log(ElementType::TimestampOrigin, &name, None);
        Self {
            pp_log,
            name,
            origin: None,
            pad,
        }
    }

    /// The origin to subtract, establishing it from this packet if it is the
    /// first one carrying a timestamp at all.
    ///
    /// `None` for a packet with neither `pts` nor `dts`: there is nothing to
    /// shift, and nothing to shift it by, so such a packet must not become
    /// the origin either — the next one that does carry a timestamp is what
    /// this branch actually starts at.
    fn origin_for(&mut self, packet: &ffmpeg::Packet) -> Option<i64> {
        if let Some(origin) = self.origin {
            return Some(origin);
        }
        let earliest = match (packet.dts(), packet.pts()) {
            (Some(dts), Some(pts)) => dts.min(pts),
            (Some(only), None) | (None, Some(only)) => only,
            (None, None) => return None,
        };
        self.origin = Some(earliest);
        pp_info!(self, "timeline starts at {earliest}");
        Some(earliest)
    }
}

impl Element for TimestampOrigin {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::TimestampOrigin
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for TimestampOrigin {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for TimestampOrigin {
    /// Encoded media of either kind. It reads timestamps and nothing else, so
    /// which medium the packets carry does not matter — but frames are not
    /// what it handles, and a chain that wired them here would be wrong
    /// before it ever ran.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::Packets(MediaKindSet::PACKETS))
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        let MediaBuffer::Packet(packet) = &buf else {
            // `Eos` above all: a muxer finalizes on it, and it carries no
            // timeline to move.
            return self.pad.push(buf);
        };
        let Some(origin) = self.origin_for(packet) else {
            return self.pad.push(buf);
        };
        // Copied rather than moved: the `Arc` may be shared with a sibling
        // branch off the same `Tee`, which must not see this branch's
        // timeline.
        let mut rebased = packet.as_ref().clone();
        if let Some(pts) = packet.pts() {
            rebased.set_pts(Some(pts - origin));
        }
        if let Some(dts) = packet.dts() {
            rebased.set_dts(Some(dts - origin));
        }
        self.pad.push(MediaBuffer::Packet(Arc::new(rebased)))
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        // `Stop` abandons this run, so a pipeline started again begins a new
        // timeline at zero like the first one did.
        //
        // `Flush` deliberately does not: it announces a new position in the
        // *same* timeline, and forgetting the origin would restart the output
        // at zero mid-stream — timestamps going backwards, which is not
        // something a muxer accepts or a caller asked for.
        if msg == ControlMsg::Stop {
            self.origin = None;
        }
        self.pad.control(msg)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    struct CapturingSink {
        pp_log: PpLog,
        received: Arc<Mutex<Vec<MediaBuffer>>>,
    }

    impl Element for CapturingSink {
        fn name(&self) -> Arc<str> {
            "capture".into()
        }
        fn element_type(&self) -> ElementType {
            ElementType::Other
        }
        fn pp_log(&self) -> &PpLog {
            &self.pp_log
        }
        fn pp_log_mut(&mut self) -> &mut PpLog {
            &mut self.pp_log
        }
    }

    impl Sink for CapturingSink {
        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            self.received.lock().unwrap().push(buf);
            Ok(())
        }
        fn control(&mut self, _msg: ControlMsg) -> Result<()> {
            Ok(())
        }
    }

    fn capture(element: &mut TimestampOrigin) -> Arc<Mutex<Vec<MediaBuffer>>> {
        let received = Arc::new(Mutex::new(Vec::new()));
        element.src_pads()[0].link(Box::new(CapturingSink {
            received: received.clone(),
            pp_log: element_pp_log(ElementType::Other, "capture", None),
        }));
        received
    }

    fn packet(pts: Option<i64>, dts: Option<i64>) -> MediaBuffer {
        let mut packet = ffmpeg::Packet::new(4);
        packet.set_pts(pts);
        packet.set_dts(dts);
        MediaBuffer::Packet(Arc::new(packet))
    }

    /// `(pts, dts)` of everything that reached the sink.
    fn stamps(received: &Arc<Mutex<Vec<MediaBuffer>>>) -> Vec<(Option<i64>, Option<i64>)> {
        received
            .lock()
            .unwrap()
            .iter()
            .filter_map(|buf| match buf {
                MediaBuffer::Packet(packet) => Some((packet.pts(), packet.dts())),
                _ => None,
            })
            .collect()
    }

    /// The whole point: a branch attached to a running `Tee` receives the
    /// source's timeline, and a file made of it must still start at zero.
    #[test]
    fn a_stream_that_starts_late_is_moved_to_zero() {
        let mut origin = TimestampOrigin::new("origin");
        let received = capture(&mut origin);

        for pts in [247_803, 248_059, 248_315] {
            origin.consume(packet(Some(pts), Some(pts))).unwrap();
        }

        assert_eq!(
            stamps(&received),
            vec![
                (Some(0), Some(0)),
                (Some(256), Some(256)),
                (Some(512), Some(512))
            ],
            "the first packet is the origin and the spacing after it is unchanged"
        );
    }

    /// A reordering encoder emits its first packet for decoding before it is
    /// displayed. Taking the `pts` as the origin would put that packet's
    /// `dts` below zero, which is exactly what muxers have workarounds for.
    #[test]
    fn a_reordered_stream_does_not_start_before_zero() {
        let mut origin = TimestampOrigin::new("origin");
        let received = capture(&mut origin);

        // What NVENC produces with B-frames: dts trails pts by the reorder
        // delay, and the first packet's dts is the earliest thing in the
        // stream.
        origin
            .consume(packet(Some(247_803), Some(247_035)))
            .unwrap();
        origin
            .consume(packet(Some(248_827), Some(247_291)))
            .unwrap();

        assert_eq!(
            stamps(&received),
            vec![(Some(768), Some(0)), (Some(1792), Some(256))],
            "the earliest timestamp is the origin, so nothing lands below zero"
        );
    }

    /// The buffer this was handed may be shared with a sibling branch off the
    /// same `Tee` — a packet counter, a second muxer — and that branch's
    /// timeline is not this one's.
    #[test]
    fn the_packet_it_was_given_is_left_alone() {
        let mut origin = TimestampOrigin::new("origin");
        let _received = capture(&mut origin);

        let buf = packet(Some(247_803), Some(247_803));
        let MediaBuffer::Packet(shared) = &buf else {
            panic!("built a packet");
        };
        let shared = Arc::clone(shared);
        origin.consume(buf).unwrap();

        assert_eq!(
            (shared.pts(), shared.dts()),
            (Some(247_803), Some(247_803)),
            "rewrote the timeline of a buffer a sibling branch also holds"
        );
    }

    /// A packet with no timestamps at all is not this branch's beginning:
    /// there is nothing in it to start a timeline from, and the next one that
    /// does carry a timestamp is where the recording really starts.
    #[test]
    fn an_unstamped_packet_does_not_become_the_origin() {
        let mut origin = TimestampOrigin::new("origin");
        let received = capture(&mut origin);

        origin.consume(packet(None, None)).unwrap();
        origin.consume(packet(Some(500), Some(500))).unwrap();

        assert_eq!(stamps(&received), vec![(None, None), (Some(0), Some(0))]);
    }

    /// `Eos` is what makes a muxer write its trailer. Swallowing or delaying
    /// it would leave a file that never finishes.
    #[test]
    fn end_of_stream_is_forwarded() {
        let mut origin = TimestampOrigin::new("origin");
        let received = capture(&mut origin);

        origin.consume(packet(Some(500), Some(500))).unwrap();
        origin.consume(MediaBuffer::Eos).unwrap();

        assert!(
            received
                .lock()
                .unwrap()
                .last()
                .is_some_and(MediaBuffer::is_eos),
            "a muxer downstream would never finalize its file"
        );
    }

    /// `Stop` abandons the run, so a pipeline started again begins at zero
    /// rather than continuing to subtract the first run's origin.
    #[test]
    fn stop_lets_the_next_run_start_at_zero_again() {
        let mut origin = TimestampOrigin::new("origin");
        let received = capture(&mut origin);

        origin.consume(packet(Some(1_000), Some(1_000))).unwrap();
        origin.control(ControlMsg::Stop).unwrap();
        origin.consume(packet(Some(9_000), Some(9_000))).unwrap();

        assert_eq!(
            stamps(&received),
            vec![(Some(0), Some(0)), (Some(0), Some(0))]
        );
    }

    /// A seek moves within the same timeline. Forgetting the origin there
    /// would send the output backwards to zero mid-stream, which is not
    /// something a muxer accepts.
    #[test]
    fn flush_keeps_the_origin_it_established() {
        let mut origin = TimestampOrigin::new("origin");
        let received = capture(&mut origin);

        origin.consume(packet(Some(1_000), Some(1_000))).unwrap();
        origin.control(ControlMsg::Flush).unwrap();
        origin.consume(packet(Some(3_000), Some(3_000))).unwrap();

        assert_eq!(
            stamps(&received),
            vec![(Some(0), Some(0)), (Some(2_000), Some(2_000))]
        );
    }
}
