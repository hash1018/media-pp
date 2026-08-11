use std::{path::Path, sync::Arc, time::Duration};

use ffmpeg_next as ffmpeg;
use rust_hlog::{HLog, herror, hinfo};
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    control::{ControlReceiver, drain_control},
    element::{Element, ElementType, Source, SourceElement, element_hlog},
    pad::SrcPad,
};

/// Errors specific to `FileDemuxer`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum FileDemuxError {
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),
}

/// Metadata about one stream in an opened container, reported up front so
/// callers can decide what to build downstream before the pipeline runs.
#[derive(Debug, Clone, Copy)]
pub struct StreamInfo {
    pub index: usize,
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
#[rust_hlog::hlog]
pub struct FileDemuxer {
    name: Arc<str>,
    input: ffmpeg::format::context::Input,
    pads: Vec<SrcPad>,
    /// One packet read ahead of `run`'s own loop, set only by `seek` —
    /// peeking a packet right after `Input::seek` is how it learns where
    /// playback actually landed (see `seek`'s docs), and that packet still
    /// needs to be delivered, not discarded, so it's stashed here for
    /// `run`'s next iteration to pick up instead of reading a fresh one.
    pending: Option<(usize, ffmpeg::Packet)>,
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
            .map(|s| SrcPad::new(format!("src_{}", s.index)))
            .collect();

        let name: Arc<str> = name.into().into();
        let hlog = element_hlog(ElementType::FileDemuxer, &name, None);
        hinfo!(
            hlog: &hlog,
            "opened: path={}, {} stream(s)",
            path.as_ref().display(),
            streams.len()
        );
        Ok((
            Self {
                name,
                hlog,
                input,
                pads,
                pending: None,
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
}

impl Element for FileDemuxer {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::FileDemuxer
    }

    fn hlog(&self) -> &HLog {
        &self.hlog
    }

    fn hlog_mut(&mut self) -> &mut HLog {
        &mut self.hlog
    }
}

impl Source for FileDemuxer {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        &mut self.pads
    }
}

impl SourceElement for FileDemuxer {
    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> crate::error::Result<()> {
        hinfo!(self, "run: starting");
        // Deliberately re-creates `self.input.packets()` fresh every
        // iteration (cheap — it's just a short-lived wrapper, not a
        // stateful cursor of its own) instead of holding one `for` loop's
        // iterator across the whole function, the way this used to read.
        // That iterator borrows `input` for as long as it's alive; `Seek`
        // needs `drain_control` to be able to call `self.seek()` — a
        // *second* mutable borrow of `input` — in between reads, which a
        // single loop-spanning iterator would rule out.
        loop {
            if drain_control(control, self, bus)? {
                // Stop: abandon in place, no final Eos.
                hinfo!(self, "run: stopped");
                return Ok(());
            }
            // `seek` (called from within `drain_control`, above) already
            // consumed one packet to find out where it landed — deliver
            // that before reading a fresh one, or it'd be silently lost.
            let next = match self.pending.take() {
                Some(next) => Some(next),
                None => self.input.packets().next().map(|(s, p)| (s.index(), p)),
            };
            let Some((index, packet)) = next else {
                break;
            };
            if let Some(pad) = self.pads.get_mut(index) {
                // A downstream failure drops just this one packet — same
                // "report, don't die" contract `Queue`'s worker gives a
                // failing `Sink` — rather than ending this whole source
                // thread over it. `Pipeline::stop` is how a caller who
                // decides an error is fatal actually ends things.
                if let Err(error) = pad.push(MediaBuffer::Packet(Arc::new(packet))) {
                    bus.post(
                        &self.hlog,
                        BusEvent::Error {
                            element_type: ElementType::FileDemuxer,
                            name: self.name.clone(),
                            error,
                        },
                    );
                }
            }
        }
        for pad in self.pads.iter_mut() {
            pad.push(MediaBuffer::Eos)?;
        }
        hinfo!(self, "run: reached eos");
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
            herror!(self, "seek to {target:?} failed: {error}");
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
                self.pending = Some((stream.index(), packet));
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

    /// Real video file, one directory up from this crate — same fixture
    /// `pipeline::tests::test_video` drives its own `FileDemuxer`
    /// integration tests through; this module drives `FileDemuxer`
    /// directly instead, without a `Pipeline` around it.
    fn test_video() -> &'static str {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../test-video/h265.mp4")
    }

    #[rust_hlog::hlog]
    struct CountingSink {
        count: Arc<AtomicUsize>,
        saw_eos: Arc<AtomicBool>,
    }

    impl Element for CountingSink {
        fn name(&self) -> Arc<str> {
            "counting-sink".into()
        }

        fn element_type(&self) -> ElementType {
            ElementType::Other
        }

        fn hlog(&self) -> &HLog {
            &self.hlog
        }

        fn hlog_mut(&mut self) -> &mut HLog {
            &mut self.hlog
        }
    }

    impl crate::element::Sink for CountingSink {
        fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
            match buf {
                MediaBuffer::Eos => self.saw_eos.store(true, Ordering::SeqCst),
                _ => {
                    self.count.fetch_add(1, Ordering::SeqCst);
                }
            }
            Ok(())
        }

        fn control(&mut self, _msg: crate::control::ControlMsg) -> crate::error::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn open_reports_stream_parameters_for_a_valid_index_and_none_out_of_range() {
        let (demuxer, streams) = FileDemuxer::open("demux", test_video()).expect("open test video");
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
        let (mut demuxer, streams) =
            FileDemuxer::open("demux", test_video()).expect("open test video");
        let video = streams
            .iter()
            .find(|s| s.kind == ffmpeg::media::Type::Video)
            .expect("test video has a video stream");

        let count = Arc::new(AtomicUsize::new(0));
        let saw_eos = Arc::new(AtomicBool::new(false));
        demuxer.src_pads()[video.index].link(Box::new(CountingSink {
            count: count.clone(),
            saw_eos: saw_eos.clone(),
            hlog: element_hlog(ElementType::Other, "counting-sink", None),
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
        drop(bus);
        assert!(
            bus_rx.iter().all(|e| !matches!(e, BusEvent::Error { .. })),
            "run must not report any errors demuxing a well-formed file"
        );
    }
}
