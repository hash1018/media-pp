use std::{
    collections::VecDeque,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use ffmpeg_next as ffmpeg;
use rust_hlog::{HLog, hinfo};

use crate::{
    buffer::MediaBuffer,
    clock::Clock,
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_hlog},
    pad::SrcPad,
};

/// Maximum time a paced wait sleeps without checking whether a control
/// request needs the owning worker back.
const INTERRUPT_POLL_INTERVAL: Duration = Duration::from_millis(10);

/// Delays each buffer until its presentation time, so downstream sees
/// frames (or, upstream of a decoder, compressed packets) at real playback
/// speed instead of as fast as demux/decode can produce them. A `Filter`:
/// receives via `Sink`, waits in short interruptible sleeps inside
/// `consume`, then pushes the same buffer through its own (single) src pad.
/// A pending pause/seek/stop interrupts that wait so the owning worker can
/// process control: pause retains the in-flight buffer for resume, while
/// seek and stop discard it.
///
/// Normally place a [`crate::queue::Queue`] upstream so the paced waits do
/// not stall the demux/decoder feeding it and those stages can run ahead
/// into the queue. The type does not enforce that placement; without the
/// queue, pacing simply blocks the upstream caller on the same thread.
///
/// `clock` is shared across every `Pacer` in the pipeline (one per stream
/// — video, audio, ...) so they all agree on the same t=0 instead of each
/// anchoring to its own first frame.
#[rust_hlog::hlog]
pub struct Pacer {
    name: Arc<str>,
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
    /// The latest pipeline interrupt this pacer has acknowledged through
    /// `control()`. A newer clock epoch means pause/seek/stop is waiting for
    /// the current `consume()` call to return. `Queue`'s own worker only
    /// checks its control channel *between* buffers (see its type docs) —
    /// it can't preempt a `consume()` call already in flight, and this
    /// pacer's own wait is exactly that kind of long-running call.
    interrupt_epoch: u64,
    /// Buffers whose paced wait was interrupted before the owning worker
    /// could process pause/seek/stop. Pause retains them for resume; seek
    /// and stop discard them in `control()`.
    pending: VecDeque<MediaBuffer>,
    pad: SrcPad,
}

impl Pacer {
    pub fn new(name: impl Into<String>, time_base: ffmpeg::Rational, clock: Arc<Clock>) -> Self {
        let name: Arc<str> = name.into().into();
        let hlog = element_hlog(ElementType::Pacer, &name, None);
        hinfo!(hlog: &hlog, "created: time_base={time_base}");
        let pad = SrcPad::new(format!("{name}_src"));
        let interrupt_epoch = clock.interrupt_epoch();
        Self {
            name,
            hlog,
            time_base,
            clock,
            first_pts: None,
            interrupt_epoch,
            pending: VecDeque::new(),
            pad,
        }
    }

    /// Blocks until `pts` is due, based on this pacer's `first_pts` (set
    /// here, on the first call) and the shared `clock`'s *current*
    /// anchor. Returns `false` if pause/seek/stop interrupts the wait; the
    /// caller retains that in-flight buffer and returns so the owning worker
    /// can process the pending control request. Frames without a pts (`None`)
    /// pass straight through.
    fn wait_for(&mut self, pts: Option<i64>) -> bool {
        if self.clock.interrupt_epoch() != self.interrupt_epoch {
            return false;
        }
        let Some(pts) = pts else { return true };
        let first_pts = *self.first_pts.get_or_insert(pts);

        let elapsed_ticks = pts - first_pts;
        let elapsed_secs = elapsed_ticks as f64 * f64::from(self.time_base);
        if elapsed_secs <= 0.0 {
            return true;
        }

        let due = self.clock.start() + std::time::Duration::from_secs_f64(elapsed_secs);
        loop {
            if self.clock.interrupt_epoch() != self.interrupt_epoch {
                return false;
            }
            let now = Instant::now();
            if due <= now {
                return true;
            }
            thread::sleep((due - now).min(INTERRUPT_POLL_INTERVAL));
        }
    }
}

impl Element for Pacer {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::Pacer
    }

    fn hlog(&self) -> &HLog {
        &self.hlog
    }

    fn hlog_mut(&mut self) -> &mut HLog {
        &mut self.hlog
    }
}

impl Source for Pacer {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for Pacer {
    fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
        self.pending.push_back(buf);
        while let Some(buf) = self.pending.pop_front() {
            let ready = match &buf {
                MediaBuffer::Packet(packet) => self.wait_for(packet.pts()),
                MediaBuffer::Video(frame) => self.wait_for(frame.pts()),
                MediaBuffer::Audio(frame) => self.wait_for(frame.pts()),
                MediaBuffer::Eos => true,
            };
            if !ready {
                self.pending.push_front(buf);
                return Ok(());
            }
            self.pad.push(buf)?;
        }
        Ok(())
    }

    fn control(&mut self, msg: ControlMsg) -> crate::error::Result<()> {
        // Acknowledge the interrupt that made any in-flight wait return.
        // Seek additionally resets both halves of pacing so the next frame
        // establishes a fresh pts and wall-clock anchor.
        self.interrupt_epoch = self.clock.interrupt_epoch();
        match msg {
            ControlMsg::Seek(_) => {
                self.pending.clear();
                self.first_pts = None;
                self.clock.reset();
            }
            ControlMsg::Stop => self.pending.clear(),
            ControlMsg::Pause | ControlMsg::Resume => {}
        }
        self.pad.control(msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{sync::mpsc, time::Duration};

    fn packet(pts: i64) -> MediaBuffer {
        let mut packet = ffmpeg::Packet::empty();
        packet.set_pts(Some(pts));
        MediaBuffer::Packet(Arc::new(packet))
    }

    #[test]
    fn long_wait_returns_promptly_when_control_interrupts_it() {
        let clock = Arc::new(Clock::new());
        let mut pacer = Pacer::new("pacer", ffmpeg::Rational::new(1, 1), clock.clone());
        assert!(
            pacer.wait_for(Some(0)),
            "first pts should establish the anchor"
        );

        let (started_tx, started_rx) = mpsc::channel();
        let worker = thread::spawn(move || {
            started_tx.send(()).expect("test receiver alive");
            pacer.wait_for(Some(60))
        });

        started_rx.recv().expect("paced wait should start");
        thread::sleep(Duration::from_millis(20));
        clock.interrupt();

        assert!(
            !worker.join().expect("paced wait should return"),
            "an interrupted paced wait must return before its due time"
        );
    }

    #[test]
    fn pause_retains_interrupted_buffer_but_seek_and_stop_discard_it() {
        let clock = Arc::new(Clock::new());
        let mut pacer = Pacer::new("pacer", ffmpeg::Rational::new(1, 1), clock.clone());

        clock.interrupt();
        pacer.consume(packet(0)).expect("interrupted consume");
        assert_eq!(pacer.pending.len(), 1);

        pacer.control(ControlMsg::Pause).expect("pause");
        assert_eq!(pacer.pending.len(), 1, "pause must retain the buffer");

        pacer
            .control(ControlMsg::Seek(Duration::ZERO))
            .expect("seek");
        assert!(pacer.pending.is_empty(), "seek must discard stale data");

        clock.interrupt();
        pacer.consume(packet(1)).expect("interrupted consume");
        assert_eq!(pacer.pending.len(), 1);
        pacer.control(ControlMsg::Stop).expect("stop");
        assert!(pacer.pending.is_empty(), "stop must abandon pending data");
    }
}
