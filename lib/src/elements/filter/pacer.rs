use std::{sync::Arc, thread, time::Instant};

use ffmpeg_next as ffmpeg;

use crate::{
    buffer::MediaBuffer,
    clock::Clock,
    element::{Element, Sink, Source},
    pad::SrcPad,
};

/// Delays each buffer until its presentation time, so downstream sees
/// frames (or, upstream of a decoder, compressed packets) at real playback
/// speed instead of as fast as demux/decode can produce them. A `Filter`:
/// receives via `Sink`, sleeps in `consume`, then pushes the same buffer on
/// through its own (single) src pad.
///
/// Must run on its own thread (put a [`crate::queue::Queue`] upstream of
/// it) — sleeping here would otherwise stall whatever's feeding it
/// (demux/decode), which needs to keep running ahead into that queue's
/// buffer.
///
/// `clock` is shared across every `Pacer` in the pipeline (one per stream
/// — video, audio, ...) so they all agree on the same t=0 instead of each
/// anchoring to its own first frame.
pub struct Pacer {
    name: String,
    time_base: ffmpeg::Rational,
    clock: Arc<Clock>,
    /// (wall-clock anchor, first frame's pts) — set on this pacer's first
    /// timestamped frame.
    origin: Option<(Instant, i64)>,
    pad: SrcPad,
}

impl Pacer {
    pub fn new(name: impl Into<String>, time_base: ffmpeg::Rational, clock: Arc<Clock>) -> Self {
        let name = name.into();
        let pad = SrcPad::new(format!("{name}_src"));
        Self {
            name,
            time_base,
            clock,
            origin: None,
            pad,
        }
    }

    /// Blocks until `pts` is due, based on this pacer's origin (set here,
    /// on the first call) and the shared `clock`'s anchor. Frames without a
    /// pts (`None`) pass straight through — nothing to pace against.
    fn wait_for(&mut self, pts: Option<i64>) {
        let Some(pts) = pts else { return };
        let &mut (anchor, first_pts) = self.origin.get_or_insert((self.clock.start(), pts));

        let elapsed_ticks = pts - first_pts;
        let elapsed_secs = elapsed_ticks as f64 * f64::from(self.time_base);
        if elapsed_secs <= 0.0 {
            return;
        }

        let due = anchor + std::time::Duration::from_secs_f64(elapsed_secs);
        let now = Instant::now();
        if due > now {
            thread::sleep(due - now);
        }
    }
}

impl Element for Pacer {
    fn name(&self) -> &str {
        &self.name
    }
}

impl Source for Pacer {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for Pacer {
    fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
        match &buf {
            MediaBuffer::Packet(packet) => self.wait_for(packet.pts()),
            MediaBuffer::Video(frame) => self.wait_for(frame.pts()),
            MediaBuffer::Audio(frame) => self.wait_for(frame.pts()),
            MediaBuffer::Eos => {}
        }
        self.pad.push(buf)
    }
}
