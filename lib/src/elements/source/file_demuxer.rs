use std::{collections::VecDeque, path::Path, sync::Arc, time::Duration};

use crate::pp_log::{PpLog, pp_error, pp_info};
use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    contract::{MediaKind, OutputContract, PortContract},
    control::{ControlReceiver, drain_control},
    element::{Element, ElementType, Source, SourceElement, element_pp_log},
    pad::SrcPad,
};

/// Errors specific to `FileDemuxer`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum FileDemuxError {
    /// FFmpeg rejected opening, reading, or seeking the input container.
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),
}

/// Metadata about one stream in an opened container, reported up front so
/// callers can decide what to build downstream before the pipeline runs.
#[derive(Debug, Clone, Copy)]
pub struct StreamInfo {
    /// Zero-based stream index used by the matching source pad.
    pub index: usize,
    /// Media kind reported by the container, such as audio or video.
    pub kind: ffmpeg::media::Type,
}

/// Demuxes a file, exposing one src pad per container stream (indexed the
/// same way as `StreamInfo::index`). Linking a pad "selects" that stream;
/// leaving it unlinked just drops its packets. Real demuxer I/O is
/// blocking, so this is meant to be run as the pipeline's source thread.
///
/// Fan-out (e.g. routing video and audio to separate branches) needs no
/// separate "Tee" element here — it's just a matter of linking more than
/// one of these pads.
pub struct FileDemuxer {
    pp_log: PpLog,
    name: Arc<str>,
    input: ffmpeg::format::context::Input,
    pads: Vec<SrcPad>,
    /// Packets read but not yet delivered, in file order.
    ///
    /// `seek` puts one here: peeking a packet right after `Input::seek` is how
    /// it learns where playback actually landed (see `seek`'s docs), and that
    /// packet is real data that still has to be delivered.
    ///
    /// `run` also parks a packet here when its own pad cannot accept one yet.
    /// A container interleaves every stream into one read cursor, so refusing
    /// to read at all while a single pad is blocked stalls the streams that
    /// *are* ready — during preroll that starves whichever branch has not yet
    /// taken its sample, and the seek times out waiting for it. Holding the
    /// blocked pad's packets keeps the cursor moving; per-pad order is what
    /// matters and each pad's packets stay in the order they were read.
    pending: VecDeque<(usize, ffmpeg::Rational, ffmpeg::Packet)>,
    /// Total payload parked in `pending`.
    pending_bytes: usize,
}

impl FileDemuxer {
    /// Opens the file and returns it alongside every stream it contains,
    /// so the caller can inspect them (count, media type, ...) before
    /// deciding which of `src_pads()` to link.
    pub fn open(
        name: impl Into<String>,
        path: impl AsRef<Path>,
    ) -> Result<(Self, Vec<StreamInfo>), FileDemuxError> {
        let input = ffmpeg::format::input(&path)?;

        let streams: Vec<StreamInfo> = input
            .streams()
            .map(|s| StreamInfo {
                index: s.index(),
                kind: s.parameters().medium(),
            })
            .collect();

        let pads = streams
            .iter()
            .map(|s| {
                // Per stream, from the medium the container announced:
                // both pads emit `MediaBuffer::Packet`, so only this tells
                // an audio stream apart from a video one. A medium this
                // crate does not model (subtitles, data) declares nothing
                // and is left to the runtime check.
                match MediaKind::packet_for(s.kind) {
                    Some(kind) => SrcPad::with_contract(
                        format!("src_{}", s.index),
                        OutputContract::Fixed(PortContract::packet(kind)),
                    ),
                    None => SrcPad::new(format!("src_{}", s.index)),
                }
            })
            .collect();

        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::FileDemuxer, &name, None);
        pp_info!(
            pp_log: &pp_log,
            "opened: path={}, {} stream(s)",
            path.as_ref().display(),
            streams.len()
        );
        Ok((
            Self {
                name,
                pp_log,
                input,
                pads,
                pending: VecDeque::new(),
                pending_bytes: 0,
            },
            streams,
        ))
    }

    /// Codec parameters for one of this file's streams — what you need to
    /// construct a matching [`crate::elements::SwDecoder`] for it.
    pub fn stream_parameters(&self, index: usize) -> Option<ffmpeg::codec::Parameters> {
        self.stream(index).map(|s| s.parameters())
    }

    /// The unit decoded frame timestamps for this stream are expressed in —
    /// what you need to construct a matching [`crate::elements::Pacer`] for
    /// it.
    pub fn stream_time_base(&self, index: usize) -> Option<ffmpeg::Rational> {
        self.stream(index).map(|s| s.time_base())
    }

    fn stream(&self, index: usize) -> Option<ffmpeg::format::stream::Stream<'_>> {
        self.input.streams().find(|s| s.index() == index)
    }

    /// Pushes `item` if its pad can take one now, otherwise parks it.
    ///
    /// A downstream failure drops just that one packet — the same
    /// "report, don't die" contract `Queue`'s worker gives a failing `Sink` —
    /// rather than ending this whole source thread over it. `Pipeline::stop`
    /// is how a caller who decides an error is fatal actually ends things.
    fn deliver_or_park(
        &mut self,
        item: (usize, ffmpeg::Rational, ffmpeg::Packet),
        bus: &Bus,
    ) -> crate::error::Result<()> {
        let (index, time_base, mut packet) = item;
        // `AVCodecParameters` does not carry the container stream's timestamp
        // unit, and FFmpeg does not guarantee that demuxers populate
        // `AVPacket::time_base`. Stamping it here — the one place every packet
        // leaves this source through, parked or not — means a packet held for
        // a blocked pad arrives downstream describing itself the same way an
        // immediately delivered one does.
        packet.set_time_base(time_base);
        let Some(pad) = self.pads.get_mut(index) else {
            // No pad for this stream: nobody selected it, so it is dropped
            // rather than parked. Parking it would grow without bound.
            return Ok(());
        };
        if !pad.ready_consume() {
            self.pending_bytes = self.pending_bytes.saturating_add(packet.size());
            self.pending.push_back((index, time_base, packet));
            return Ok(());
        }
        if let Err(error) = pad.push(MediaBuffer::Packet(Arc::new(packet))) {
            bus.post(
                &self.pp_log,
                BusEvent::Error {
                    element_type: ElementType::FileDemuxer,
                    name: self.name.clone(),
                    error,
                },
            );
        }
        Ok(())
    }

    /// Delivers every parked packet whose pad has since become ready, oldest
    /// first, leaving the rest in place.
    fn drain_pending(&mut self, bus: &Bus) -> crate::error::Result<()> {
        let mut deferred = VecDeque::with_capacity(self.pending.len());
        while let Some((index, time_base, packet)) = self.pending.pop_front() {
            let ready = self
                .pads
                .get_mut(index)
                .is_some_and(SrcPad::ready_consume)
                // A pad this packet has no home for is dropped, not carried.
                || self.pads.get(index).is_none();
            if !ready {
                deferred.push_back((index, time_base, packet));
                continue;
            }
            self.pending_bytes = self.pending_bytes.saturating_sub(packet.size());
            self.deliver_or_park((index, time_base, packet), bus)?;
        }
        // `deliver_or_park` may have parked something again; keep the older
        // entries in front of it so per-pad order survives.
        while let Some(item) = self.pending.pop_back() {
            deferred.push_back(item);
        }
        self.pending = deferred;
        Ok(())
    }

    /// Whether reading another packet would only deepen the parked backlog.
    ///
    /// Not a tuning knob: a branch that is briefly behind parks a handful of
    /// packets and clears them within milliseconds. These ceilings only bound
    /// a pad that has stopped accepting altogether, so the read cursor cannot
    /// pull an arbitrary amount of the file into memory waiting for it. Both
    /// are needed — a few large keyframes reach the byte limit at a packet
    /// count that would never trip on its own.
    fn pending_blocked(&mut self) -> bool {
        const MAX_PENDING_PACKETS: usize = 4_096;
        const MAX_PENDING_BYTES: usize = 64 * 1024 * 1024;

        self.pending.len() >= MAX_PENDING_PACKETS
            || self.pending_bytes >= MAX_PENDING_BYTES
            || !self.pads.iter_mut().any(SrcPad::ready_consume)
    }
}

impl Element for FileDemuxer {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::FileDemuxer
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for FileDemuxer {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        &mut self.pads
    }
}

impl SourceElement for FileDemuxer {
    fn is_live(&self) -> bool {
        false
    }

    fn is_seekable(&self) -> bool {
        true
    }

    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> crate::error::Result<()> {
        pp_info!(self, "started");
        // Deliberately re-creates `self.input.packets()` fresh every
        // iteration (cheap — it's just a short-lived wrapper, not a
        // stateful cursor of its own) instead of holding one `for` loop's
        // iterator across the whole function, the way this used to read.
        // That iterator borrows `input` for as long as it's alive; `Seek`
        // needs `drain_control` to be able to call `self.seek()` — a
        // *second* mutable borrow of `input` — in between reads, which a
        // single loop-spanning iterator would rule out.
        loop {
            if drain_control(control, self, bus)?.stopped {
                // Stop: abandon in place, no final Eos.
                pp_info!(self, "stopped");
                return Ok(());
            }
            // Deliver whatever is parked and now accepted, oldest first.
            // Skipping a still-blocked pad's entry to reach a later one is
            // safe: only each pad's own order has to hold, and this preserves
            // it because entries for one pad are never reordered against each
            // other.
            self.drain_pending(bus)?;
            // Read on unless everything is blocked, or the parked backlog has
            // grown past what one branch briefly falling behind can explain.
            // Sleeping is the only option then: the container cannot hand out
            // a different stream's packet without reading this one.
            if self.pending_blocked() {
                std::thread::sleep(Duration::from_millis(1));
                continue;
            }
            let next = self
                .input
                .packets()
                .next()
                .map(|(s, p)| (s.index(), s.time_base(), p));
            let Some((index, time_base, packet)) = next else {
                if self.pending.is_empty() {
                    break;
                }
                // Nothing left to read, but a blocked pad still owes delivery.
                std::thread::sleep(Duration::from_millis(1));
                continue;
            };
            // `deliver_or_park` stamps the stream time base; every packet
            // leaves this source through it, parked or not.
            self.deliver_or_park((index, time_base, packet), bus)?;
        }
        for pad in self.pads.iter_mut() {
            pad.push_eos(&self.pp_log)?;
        }
        pp_info!(self, "event=eos phase=source_completed outcome=ok");
        Ok(())
    }

    fn seek(&mut self, target: Duration) -> crate::error::Result<Duration> {
        // `Input::seek` takes microseconds (`AV_TIME_BASE` units) when
        // seeking the whole container (stream index -1, which is what it
        // uses internally) rather than one specific stream — an unbounded
        // range (`..`) just means "as close to `ts` as ffmpeg can manage",
        // no extra min/max constraint. In practice that means *backward*
        // to the nearest keyframe at or before `target`: never forward,
        // and never onto a non-keyframe, since either would leave nothing
        // downstream can decode/remux from. A sparse-keyframe file can
        // make that keyframe well before `target` — e.g. a single
        // 10-second file with keyframes only at 0s and 8.3s means every
        // `target` under 8.3s lands back at 0s.
        let ts = target.as_micros().min(i64::MAX as u128) as i64;
        self.input.seek(ts, ..).inspect_err(|error| {
            pp_error!(self, "seek to {target:?} failed: {error}");
        })?;

        // `avformat_seek_file` only reports success/failure, not where it
        // landed — the one way to find out is to read the next packet and
        // look at its own timestamp. That packet is real data (not a
        // probe to throw away), so it's stashed in `pending` for `run`'s
        // next iteration instead of being dropped here.
        match self.input.packets().next() {
            Some((stream, packet)) => {
                let time_base = stream.time_base();
                let landed = packet
                    .pts()
                    .or_else(|| packet.dts())
                    .map(|ts| ts_to_duration(ts, time_base))
                    .unwrap_or(Duration::ZERO);
                self.pending.push_back((stream.index(), time_base, packet));
                Ok(landed)
            }
            // Nothing left to read right after seeking (`target` at/past
            // EOF) — there's no packet to learn a real position from, so
            // just report the request back as-is.
            None => Ok(target),
        }
    }
}

fn ts_to_duration(ts: i64, time_base: ffmpeg::Rational) -> Duration {
    let secs = ts as f64 * f64::from(time_base.numerator()) / f64::from(time_base.denominator());
    Duration::from_secs_f64(secs.max(0.0))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use super::*;
    use crate::control;
    use crate::test_support::try_test_video;

    struct CountingSink {
        pp_log: PpLog,
        count: Arc<AtomicUsize>,
        saw_eos: Arc<AtomicBool>,
        expected_time_base: ffmpeg::Rational,
        time_base_matches: Arc<AtomicBool>,
    }

    impl Element for CountingSink {
        fn name(&self) -> Arc<str> {
            "counting-sink".into()
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

    impl crate::element::Sink for CountingSink {
        fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
            match buf {
                MediaBuffer::Eos => self.saw_eos.store(true, Ordering::SeqCst),
                MediaBuffer::Packet(packet) => {
                    if packet.time_base() != self.expected_time_base {
                        self.time_base_matches.store(false, Ordering::SeqCst);
                    }
                    self.count.fetch_add(1, Ordering::SeqCst);
                }
                _ => {}
            }
            Ok(())
        }

        fn control(&mut self, _msg: crate::control::ControlMsg) -> crate::error::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn open_reports_stream_parameters_for_a_valid_index_and_none_out_of_range() {
        let Some(path) = try_test_video() else { return };
        let (demuxer, streams) = FileDemuxer::open("demux", &path).expect("open test video");
        let video = streams
            .iter()
            .find(|s| s.kind == ffmpeg::media::Type::Video)
            .expect("test video has a video stream");

        assert!(demuxer.stream_parameters(video.index).is_some());
        assert!(demuxer.stream_time_base(video.index).is_some());

        let out_of_range = streams.len() + 1;
        assert!(
            demuxer.stream_parameters(out_of_range).is_none(),
            "an out-of-range stream index must report nothing, not panic"
        );
        assert!(demuxer.stream_time_base(out_of_range).is_none());
    }

    /// Drives `FileDemuxer::run` directly (no `Pipeline`) to prove the
    /// basic contract on its own: every packet on a linked pad's stream
    /// arrives, and running off the end of the file delivers a final
    /// `Eos` rather than just stopping silently.
    #[test]
    fn run_delivers_every_packet_on_a_linked_pad_then_eos() {
        let Some(path) = try_test_video() else { return };
        let (mut demuxer, streams) = FileDemuxer::open("demux", &path).expect("open test video");
        let video = streams
            .iter()
            .find(|s| s.kind == ffmpeg::media::Type::Video)
            .expect("test video has a video stream");

        let count = Arc::new(AtomicUsize::new(0));
        let saw_eos = Arc::new(AtomicBool::new(false));
        let time_base_matches = Arc::new(AtomicBool::new(true));
        let expected_time_base = demuxer
            .stream_time_base(video.index)
            .expect("video stream has a time base");
        demuxer.src_pads()[video.index].link(Box::new(CountingSink {
            count: count.clone(),
            saw_eos: saw_eos.clone(),
            expected_time_base,
            time_base_matches: time_base_matches.clone(),
            pp_log: element_pp_log(ElementType::Other, "counting-sink", None),
        }));

        let (bus, bus_rx) = Bus::new();
        let (_tx, rx) = control::channel();
        demuxer
            .run(&rx, &bus)
            .expect("run must reach eos cleanly, not error");

        assert!(
            count.load(Ordering::SeqCst) > 0,
            "expected at least one packet delivered to the linked pad"
        );
        assert!(
            saw_eos.load(Ordering::SeqCst),
            "expected an Eos once the file is exhausted"
        );
        assert!(
            time_base_matches.load(Ordering::SeqCst),
            "every delivered packet must carry its stream time base"
        );
        drop(bus);
        assert!(
            bus_rx.iter().all(|e| !matches!(e, BusEvent::Error { .. })),
            "run must not report any errors demuxing a well-formed file"
        );
    }

    #[test]
    fn seek_read_ahead_packet_carries_its_stream_time_base_when_delivered() {
        let Some(path) = try_test_video() else { return };
        let (mut demuxer, _) = FileDemuxer::open("demux", &path).expect("open test video");

        demuxer
            .seek(Duration::from_secs(1))
            .expect("seek within the test video");
        let (pending_index, expected_time_base, _) = demuxer
            .pending
            .front()
            .expect("seek must retain the first packet at or after the target");
        let pending_index = *pending_index;
        let expected_time_base = *expected_time_base;

        let count = Arc::new(AtomicUsize::new(0));
        let saw_eos = Arc::new(AtomicBool::new(false));
        let time_base_matches = Arc::new(AtomicBool::new(true));
        demuxer.src_pads()[pending_index].link(Box::new(CountingSink {
            count: count.clone(),
            saw_eos,
            expected_time_base,
            time_base_matches: time_base_matches.clone(),
            pp_log: element_pp_log(ElementType::Other, "counting-sink", None),
        }));

        let (bus, _bus_rx) = Bus::new();
        let (_tx, rx) = control::channel();
        demuxer
            .run(&rx, &bus)
            .expect("run after seek must reach eos cleanly");

        assert!(
            count.load(Ordering::SeqCst) > 0,
            "the packet retained by seek must be delivered"
        );
        assert!(
            time_base_matches.load(Ordering::SeqCst),
            "the packet retained by seek must carry its stream time base"
        );
    }
}
