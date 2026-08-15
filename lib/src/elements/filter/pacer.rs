use std::{
    collections::VecDeque,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use crate::pp_log::{PpLog, pp_info};
use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    clock::Clock,
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    pad::SrcPad,
    time::{InvalidTimeBase, MediaTimestamp, TimeBase},
};

/// Errors specific to [`Pacer`].
#[derive(Debug, ThisError)]
pub enum PacerError {
    /// `time_base` came from
    /// [`crate::element::SourceElement::stream_time_base`]/an encoder's own
    /// time base — i.e. from a demuxed file or an otherwise externally
    /// supplied stream, not a value this crate controls. A malformed or
    /// unusual stream can legitimately have an invalid one.
    #[error(
        "invalid time base {numerator}/{denominator}: both numerator and denominator must be positive"
    )]
    InvalidTimeBase { numerator: i32, denominator: i32 },

    /// `pts` is external input too (see [`PacerError::InvalidTimeBase`]) —
    /// an adversarial or corrupt jump this far from this pacer's own
    /// `first_pts` overflows the subtraction used to compute how long to
    /// wait, leaving nothing sane to pace against.
    #[error("pts {pts} is too far from this pacer's first pts {first_pts} to pace against")]
    TimestampDeltaOverflow { pts: i64, first_pts: i64 },
}

/// [`TimeBase::new_unchecked`] is fine here — `1/1_000_000_000` is a
/// hardcoded constant known valid, not external input.
fn nanoseconds() -> TimeBase {
    TimeBase::new_unchecked(ffmpeg::Rational::new(1, 1_000_000_000))
}

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
pub struct Pacer {
    pp_log: PpLog,
    name: Arc<str>,
    time_base: TimeBase,
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
    pub fn new(
        name: impl Into<String>,
        time_base: ffmpeg::Rational,
        clock: Arc<Clock>,
    ) -> Result<Self, PacerError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::Pacer, &name, None);
        pp_info!(pp_log: &pp_log, "created: time_base={time_base}");
        let pad = SrcPad::new(format!("{name}_src"));
        let interrupt_epoch = clock.interrupt_epoch();
        let time_base = TimeBase::try_new(time_base).map_err(
            |InvalidTimeBase {
                 numerator,
                 denominator,
             }| PacerError::InvalidTimeBase {
                numerator,
                denominator,
            },
        )?;
        Ok(Self {
            name,
            pp_log,
            time_base,
            clock,
            first_pts: None,
            interrupt_epoch,
            pending: VecDeque::new(),
            pad,
        })
    }

    /// Blocks until `pts` is due, based on this pacer's `first_pts` (set
    /// here, on the first call) and the shared `clock`'s *current*
    /// anchor. Returns `Ok(false)` if pause/seek/stop interrupts the wait;
    /// the caller retains that in-flight buffer and returns so the owning
    /// worker can process the pending control request. Frames without a
    /// pts (`None`) pass straight through. `Err` only for a `pts` too
    /// pathological to pace against at all (see
    /// [`PacerError::TimestampDeltaOverflow`]) — the caller drops that one
    /// buffer rather than treating it as interrupted.
    fn wait_for(&mut self, pts: Option<i64>) -> Result<bool, PacerError> {
        if self.clock.interrupt_epoch() != self.interrupt_epoch {
            return Ok(false);
        }
        let Some(pts) = pts else { return Ok(true) };
        let first_pts = *self.first_pts.get_or_insert(pts);

        let elapsed_ticks = pts
            .checked_sub(first_pts)
            .ok_or(PacerError::TimestampDeltaOverflow { pts, first_pts })?;
        if elapsed_ticks <= 0 {
            return Ok(true);
        }
        // Integer rescale straight to nanoseconds rather than
        // `elapsed_ticks as f64 * f64::from(time_base)` — the latter loses
        // precision (and the numerator, if computed by naive division)
        // over a long-running stream; see `MediaTimestamp`'s own docs.
        let elapsed_ns = MediaTimestamp::new_unchecked(elapsed_ticks, self.time_base)
            .rescale(nanoseconds())
            .max(0) as u64;

        let due = self.clock.start() + Duration::from_nanos(elapsed_ns);
        loop {
            if self.clock.interrupt_epoch() != self.interrupt_epoch {
                return Ok(false);
            }
            let now = Instant::now();
            if due <= now {
                return Ok(true);
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

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
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
                MediaBuffer::Packet(packet) => self.wait_for(packet.pts())?,
                MediaBuffer::Video(frame) => self.wait_for(frame.pts())?,
                MediaBuffer::Audio(frame) => self.wait_for(frame.pts())?,
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
        let mut pacer = Pacer::new("pacer", ffmpeg::Rational::new(1, 1), clock.clone()).unwrap();
        assert!(
            pacer.wait_for(Some(0)).unwrap(),
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
            !worker
                .join()
                .expect("paced wait should return")
                .expect("interrupted wait is Ok(false), not an error"),
            "an interrupted paced wait must return before its due time"
        );
    }

    #[test]
    fn pause_retains_interrupted_buffer_but_seek_and_stop_discard_it() {
        let clock = Arc::new(Clock::new());
        let mut pacer = Pacer::new("pacer", ffmpeg::Rational::new(1, 1), clock.clone()).unwrap();

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

    #[test]
    fn new_rejects_an_invalid_time_base() {
        let clock = Arc::new(Clock::new());
        for rational in [
            ffmpeg::Rational::new(0, 1),
            ffmpeg::Rational::new(1, 0),
            ffmpeg::Rational::new(-1, 1),
            ffmpeg::Rational::new(1, -1),
        ] {
            assert!(
                matches!(
                    Pacer::new("pacer", rational, clock.clone()),
                    Err(PacerError::InvalidTimeBase { .. })
                ),
                "expected {rational} to be rejected"
            );
        }
    }

    /// Regression test: a `pts` this far from `first_pts` used to overflow
    /// `pts - first_pts` silently (a plain `-`) or let the buffer through
    /// unpaced (an earlier `checked_sub` that swallowed the error). Now
    /// it's a typed `PacerError` `consume` propagates via `?`, and — since
    /// `Queue`/a pushing source both treat a `Sink::consume` failure as
    /// "drop this one buffer, report on the bus, keep going" — a Pacer
    /// that hits this on one buffer must still pace the next one normally.
    #[test]
    fn a_pathological_pts_jump_is_a_typed_error_not_silent_passthrough() {
        let clock = Arc::new(Clock::new());
        let mut pacer = Pacer::new("pacer", ffmpeg::Rational::new(1, 1), clock).unwrap();

        assert!(pacer.consume(packet(-1)).is_ok(), "establishes first_pts");

        let error = pacer
            .consume(packet(i64::MAX))
            .expect_err("pts far enough from first_pts to overflow the subtraction");
        assert!(matches!(
            error,
            crate::Error::PacerError(PacerError::TimestampDeltaOverflow {
                pts: i64::MAX,
                first_pts: -1,
            })
        ));
        assert!(
            pacer.pending.is_empty(),
            "the overflowing buffer must not get stuck in `pending`"
        );

        // The pacer itself must still be usable afterward: a `Some` result
        // (not a further error) for an ordinary pts relative to the same
        // `first_pts`.
        assert!(pacer.wait_for(Some(0)).is_ok());
    }
}
