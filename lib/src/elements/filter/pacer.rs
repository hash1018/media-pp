use std::{sync::Arc, thread, time::Instant};

use ffmpeg_next as ffmpeg;

use crate::{
    buffer::MediaBuffer,
    clock::Clock,
    control::ControlMsg,
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
    /// This pacer's first timestamped frame's pts — set on first call.
    /// Deliberately *not* paired with a cached wall-clock anchor: the
    /// anchor has to come fresh from `clock.start()` on every call
    /// instead, since [`Clock::pause`]/[`Clock::resume`] can shift it —
    /// caching it once here would mean a paused-then-resumed pipeline
    /// blasts through however many frames piled up during the pause
    /// (their `due` times would all already be in the past relative to a
    /// stale anchor).
    first_pts: Option<i64>,
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
            first_pts: None,
            pad,
        }
    }

    /// Blocks until `pts` is due, based on this pacer's `first_pts` (set
    /// here, on the first call) and the shared `clock`'s *current*
    /// anchor. Frames without a pts (`None`) pass straight through —
    /// nothing to pace against.
    fn wait_for(&mut self, pts: Option<i64>) {
        let Some(pts) = pts else { return };
        let first_pts = *self.first_pts.get_or_insert(pts);

        let elapsed_ticks = pts - first_pts;
        let elapsed_secs = elapsed_ticks as f64 * f64::from(self.time_base);
        if elapsed_secs <= 0.0 {
            return;
        }

        let due = self.clock.start() + std::time::Duration::from_secs_f64(elapsed_secs);
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

    fn control(&mut self, msg: ControlMsg) -> crate::error::Result<()> {
        // Pause/Stop: nothing local to react to — freezing this pacer is
        // entirely a function of upstream not feeding it (see `Queue`'s
        // worker loop) and `Clock` being paused.
        //
        // Seek: the next frame this pacer sees will have a pts from the
        // new position, unrelated to `first_pts` (from before the jump)
        // — reset it so `wait_for` re-anchors instead of computing a
        // huge/negative `elapsed_ticks` against a stale reference.
        //
        // `Clock::reset()` has to happen *here*, not eagerly in
        // `Pipeline::seek` before the cascade even starts — this call is
        // reached only once any data item this worker thread was already
        // in the middle of consuming (started before the `Seek` control
        // message got priority — see `Queue`'s `discard_stale_data`,
        // which can only drop what's still *sitting in the channel*, not
        // something already handed to `consume()`) has fully finished.
        // If the reset happened earlier instead, that leftover pre-seek
        // frame's own `wait_for` call could still be in flight and call
        // `clock.start()` *after* the reset but *before* the real
        // post-seek frame does, re-poisoning the fresh anchor with a
        // stale timestamp — every post-seek frame would then compute a
        // `due` time already in the past and skip pacing entirely. Pause
        // /resume can safely touch `Clock` from `Pipeline` directly
        // instead, because `Pause` itself guarantees every worker is
        // already quiesced first — nothing else could be racing to call
        // `clock.start()` at that point. `Seek` has no such guarantee, so
        // it needs the same in-cascade timing `first_pts` already gets.
        if let ControlMsg::Seek(_) = msg {
            self.first_pts = None;
            self.clock.reset();
        }
        self.pad.control(msg)
    }
}
