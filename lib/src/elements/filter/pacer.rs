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
    contract::{InputContract, OutputContract},
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    pad::SrcPad,
    time::{InvalidTimeBase, MediaTimestamp, TimeBase},
};

/// Errors specific to [`Pacer`].
#[derive(Debug, ThisError)]
pub enum PacerError {
    /// `time_base` came from
    /// [`crate::elements::FileDemuxer::stream_time_base`]/an encoder's own
    /// time base — i.e. from a demuxed file or an otherwise externally
    /// supplied stream, not a value this crate controls. A malformed or
    /// unusual stream can legitimately have an invalid one.
    #[error(
        "invalid time base {numerator}/{denominator}: both numerator and denominator must be positive"
    )]
    InvalidTimeBase {
        /// Invalid rational numerator.
        numerator: i32,
        /// Invalid rational denominator.
        denominator: i32,
    },

    /// `pts` is external input too (see [`PacerError::InvalidTimeBase`]) —
    /// an adversarial or corrupt value that cannot be turned into a wait at
    /// all. Either it does not fit in nanoseconds at this stream's time
    /// base, or it is so far from the pipeline's media origin that the
    /// subtraction overflows.
    ///
    /// Reported rather than swallowed because the alternative is the buffer
    /// going through *unpaced*, which is a whole stream arriving at once.
    #[error("pts {pts} cannot be paced against this pipeline's timeline")]
    UnpaceableTimestamp {
        /// The timestamp, in its own stream's units.
        pts: i64,
    },

    /// A buffer arrived before the pipeline gave this pacer its clock, which
    /// cannot happen through ordinary wiring: `attach_context` runs when the
    /// branch is built and a branch cannot carry buffers before it exists.
    ///
    /// Reported rather than passed through, because pacing nothing is a
    /// whole stream arriving at once and there is no quieter way for that to
    /// go wrong.
    #[error("this pacer was never wired into a pipeline, so it has no clock to pace against")]
    NotAttached,
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
/// anchoring to its own first frame. That agreement is what keeps the
/// picture with the sound: the offset a container gives its streams is part
/// of their sync, and a pacer zeroing on its own stream would throw it away.
/// The clock's own `media_origin_ns` is where that agreement is kept.
pub struct Pacer {
    pp_log: PpLog,
    name: Arc<str>,
    time_base: TimeBase,
    /// The pipeline's, given by [`Element::attach_context`] — see there for
    /// why this is not something the caller supplies.
    clock: Option<Arc<Clock>>,
    /// The media timestamp this pipeline's t=0 stands for, in nanoseconds —
    /// asked of the shared [`Clock`] on the first timestamped buffer and
    /// held until a seek clears it.
    ///
    /// The clock's and not this pacer's, which is the whole point: a
    /// container's streams do not start together — see
    /// [`Clock::media_origin_ns`] — and a pacer that zeroed on its own
    /// first timestamp would play them as though they did.
    ///
    /// Deliberately *not* paired with a cached wall-clock anchor: the
    /// anchor has to come fresh from `clock.start()` on every call
    /// instead, since [`Clock::pause`]/[`Clock::resume`] can shift it —
    /// caching it once here would mean a paused-then-resumed pipeline
    /// blasts through however many frames piled up during the pause
    /// (their `due` times would all already be in the past relative to a
    /// stale anchor).
    origin_ns: Option<i64>,
    /// The latest pipeline interrupt this pacer has acknowledged through
    /// `control()`. A newer clock epoch means pause/seek/stop is waiting for
    /// the current `consume()` call to return. `Queue`'s own worker only
    /// checks its control channel *between* buffers (see its type docs) —
    /// it can't preempt a `consume()` call already in flight, and this
    /// pacer's own wait is exactly that kind of long-running call.
    interrupt_epoch: u64,
    /// Preroll advances data without consulting the paused pipeline clock.
    prerolling: bool,
    /// Buffers whose paced wait was interrupted before the owning worker
    /// could process pause/seek/stop. Pause retains them for resume; seek
    /// and stop discard them in `control()`.
    pending: VecDeque<MediaBuffer>,
    pad: SrcPad,
}

impl Pacer {
    /// Creates a pacer using `time_base` to convert input PTS values to wall
    /// time.
    ///
    /// The clock it paces against is the pipeline's and arrives when this is
    /// wired into one — see [`Element::attach_context`].
    pub fn new(name: impl Into<String>, time_base: ffmpeg::Rational) -> Result<Self, PacerError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::Pacer, &name, None);
        pp_info!(pp_log: &pp_log, "created: time_base={time_base}");
        let pad = SrcPad::with_contract(format!("{name}_src"), OutputContract::Passthrough);
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
            clock: None,
            origin_ns: None,
            interrupt_epoch: 0,
            prerolling: false,
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
    /// [`PacerError::UnpaceableTimestamp`]) — the caller drops that one
    /// buffer rather than treating it as interrupted.
    fn wait_for(&mut self, pts: Option<i64>) -> Result<bool, PacerError> {
        let clock = self.clock.clone().ok_or(PacerError::NotAttached)?;
        if clock.interrupt_epoch() != self.interrupt_epoch {
            return Ok(false);
        }
        if self.prerolling {
            return Ok(true);
        }
        let Some(pts) = pts else { return Ok(true) };
        // Rescaled before the subtraction, not after, because the origin is
        // shared with streams in other units — a container's audio and video
        // rarely count in the same ticks. Integer rescale rather than
        // `pts as f64 * f64::from(time_base)`: the latter loses precision
        // (and the numerator, if computed by naive division) over a
        // long-running stream; see `MediaTimestamp`'s own docs.
        let pts_ns = MediaTimestamp::new_unchecked(pts, self.time_base).rescale(nanoseconds());
        // `av_rescale_q_rnd` answers a value it cannot represent with
        // `INT64_MIN`, which is also FFmpeg's "no timestamp". Checked before
        // the origin is established, so a pathological first buffer cannot
        // become the timeline every other stream is measured against.
        if pts_ns == i64::MIN {
            return Err(PacerError::UnpaceableTimestamp { pts });
        }
        let origin_ns = *self
            .origin_ns
            .get_or_insert_with(|| clock.media_origin_ns(pts_ns));
        // Read before the early return below, so the clock anchors on the
        // first buffer *any* stream releases rather than on the first one
        // that happens to have something to wait for. An offset measured
        // from it is then measured from when playback actually began.
        let anchor = clock.start();

        let elapsed_ns = pts_ns
            .checked_sub(origin_ns)
            .ok_or(PacerError::UnpaceableTimestamp { pts })?;
        // At or before the origin: the stream that set it, on its own first
        // buffer, and any stream that starts earlier still. Neither has
        // anything to wait for.
        if elapsed_ns <= 0 {
            return Ok(true);
        }

        let due = anchor + Duration::from_nanos(elapsed_ns as u64);
        loop {
            if clock.interrupt_epoch() != self.interrupt_epoch {
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

    fn attach_context(&mut self, context: &Arc<crate::element::Context>) {
        self.interrupt_epoch = context.clock.interrupt_epoch();
        self.clock = Some(Arc::clone(&context.clock));
    }
}

impl Source for Pacer {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for Pacer {
    /// Pacing is a delay, not a transform: every kind is held until its
    /// own PTS comes due and then forwarded unchanged.
    fn input_contract(&self) -> InputContract {
        InputContract::Any
    }

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
        // Flush discards an interrupted old-timeline buffer; Seek then resets
        // the timestamp and clock anchors for the new timeline.
        // A pacer that was never wired has no clock to acknowledge, and
        // nothing is going to send it control either — see
        // `PacerError::NotAttached`.
        if let Some(clock) = &self.clock {
            self.interrupt_epoch = clock.interrupt_epoch();
        }
        match msg {
            ControlMsg::Flush => self.pending.clear(),
            ControlMsg::Seek(_) => {
                self.origin_ns = None;
                if let Some(clock) = &self.clock {
                    clock.reset();
                }
            }
            ControlMsg::Stop => self.pending.clear(),
            ControlMsg::Preroll(_) => {
                self.prerolling = true;
            }
            ControlMsg::Pause | ControlMsg::Resume => {
                self.prerolling = false;
            }
            ControlMsg::CheckSeek(_) => {}
        }
        self.pad.control(msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::PrerollContext;
    use std::{sync::mpsc, time::Duration};

    /// A pacer wired the way a pipeline wires one, around a clock the test
    /// keeps so it can interrupt and pause it.
    fn paced(name: &str, time_base: ffmpeg::Rational, clock: &Arc<Clock>) -> Pacer {
        let mut pacer = Pacer::new(name, time_base).expect("valid time base");
        pacer.attach_context(&Arc::new(crate::element::Context::for_test_with_clock(
            crate::bus::Bus::new().0,
            "test",
            crate::graph::PipelineGraph::new(),
            crate::graph::ElementId::for_test(1),
            Arc::clone(clock),
        )));
        pacer
    }

    fn packet(pts: i64) -> MediaBuffer {
        let mut packet = ffmpeg::Packet::empty();
        packet.set_pts(Some(pts));
        MediaBuffer::Packet(Arc::new(packet))
    }

    #[test]
    fn long_wait_returns_promptly_when_control_interrupts_it() {
        let clock = Arc::new(Clock::new());
        let mut pacer = paced("pacer", ffmpeg::Rational::new(1, 1), &clock);
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
    fn pause_retains_interrupted_buffer_but_flush_and_stop_discard_it() {
        let clock = Arc::new(Clock::new());
        let mut pacer = paced("pacer", ffmpeg::Rational::new(1, 1), &clock);

        clock.interrupt();
        pacer.consume(packet(0)).expect("interrupted consume");
        assert_eq!(pacer.pending.len(), 1);

        pacer.control(ControlMsg::Pause).expect("pause");
        assert_eq!(pacer.pending.len(), 1, "pause must retain the buffer");

        pacer.control(ControlMsg::Flush).expect("flush");
        assert!(pacer.pending.is_empty(), "flush must discard stale data");

        pacer
            .control(ControlMsg::Seek(Duration::ZERO))
            .expect("seek");

        clock.interrupt();
        pacer.consume(packet(1)).expect("interrupted consume");
        assert_eq!(pacer.pending.len(), 1);
        pacer.control(ControlMsg::Stop).expect("stop");
        assert!(pacer.pending.is_empty(), "stop must abandon pending data");
    }

    #[test]
    fn new_rejects_an_invalid_time_base() {
        for rational in [
            ffmpeg::Rational::new(0, 1),
            ffmpeg::Rational::new(1, 0),
            ffmpeg::Rational::new(-1, 1),
            ffmpeg::Rational::new(1, -1),
        ] {
            assert!(
                matches!(
                    Pacer::new("pacer", rational),
                    Err(PacerError::InvalidTimeBase { .. })
                ),
                "expected {rational} to be rejected"
            );
        }
    }

    /// Preroll has to outrun the paused clock — that is the whole reason a
    /// `Pacer` reacts to it. Suppressing pre-target media is *not* its job:
    /// that needs the time base a decoder has on every decoded branch, and a
    /// `Pacer` is only on some of them.
    /// Preroll has to outrun the paused clock — that is the whole reason a
    /// `Pacer` reacts to it, and a paused pipeline could otherwise never
    /// deliver a preview sample. Suppressing pre-target media is *not* its
    /// job: that needs the time base a decoder has on every decoded branch,
    /// and a `Pacer` is only on some of them.
    #[test]
    fn preroll_forwards_without_waiting_out_the_presentation_time() {
        let clock = Arc::new(Clock::new());
        let mut pacer = paced("pacer", ffmpeg::Rational::new(1, 1), &clock);
        let context = Arc::new(PrerollContext::for_seek([], Duration::from_secs(2)));
        pacer
            .control(ControlMsg::Preroll(context))
            .expect("preroll");

        let started = Instant::now();
        pacer.consume(packet(0)).expect("first preroll packet");
        pacer.consume(packet(60)).expect("distant preroll packet");

        assert!(
            started.elapsed() < Duration::from_millis(100),
            "a minute of presentation time must not be waited out during preroll"
        );
    }

    /// A pacer that was never wired into a pipeline has no clock, and the
    /// one thing it must not do is let the buffer through: unpaced is a
    /// whole stream arriving at once, and silently.
    ///
    /// Unreachable through ordinary wiring — `attach_context` runs when a
    /// branch is built, and a branch cannot carry buffers before it exists —
    /// which is exactly why it is worth a typed error rather than a
    /// debug_assert nobody runs.
    #[test]
    fn an_unwired_pacer_refuses_rather_than_passing_a_buffer_through() {
        let mut pacer = Pacer::new("pacer", ffmpeg::Rational::new(1, 1)).unwrap();
        assert!(matches!(
            pacer.consume(packet(0)),
            Err(crate::error::Error::PacerError(PacerError::NotAttached))
        ));
    }

    /// Regression test for the sync this exists to keep. A container's
    /// streams do not start together — `sample.mp4`'s audio starts at zero
    /// and its video one frame in, 33 ms later — and that offset is part of
    /// what puts the picture with the sound.
    ///
    /// Each pacer used to zero on its own stream's first timestamp, which
    /// released both first buffers at once and played the sound 33 ms early
    /// for the rest of the file. The origin is the *pipeline's* now, so the
    /// stream that starts later waits for its turn.
    #[test]
    fn streams_that_start_apart_stay_apart() {
        let clock = Arc::new(Clock::new());
        // Milliseconds, so the numbers below read as what they are.
        let unit = ffmpeg::Rational::new(1, 1000);
        let mut audio = paced("audio", unit, &clock);
        let mut video = paced("video", unit, &clock);

        let started = Instant::now();
        assert!(
            audio.wait_for(Some(0)).unwrap(),
            "the first stream sets the origin and has nothing to wait for"
        );
        assert!(
            started.elapsed() < Duration::from_millis(20),
            "it must not wait for itself"
        );

        assert!(video.wait_for(Some(80)).unwrap(), "paced, not refused");
        assert!(
            started.elapsed() >= Duration::from_millis(70),
            "a stream starting 80ms into the file must be held back by it, \
             not released alongside the one that starts at zero"
        );
    }

    /// Regression test: a `pts` this far from the origin used to overflow
    /// the subtraction silently (a plain `-`) or let the buffer through
    /// unpaced (an earlier `checked_sub` that swallowed the error). Now
    /// it's a typed `PacerError` `consume` propagates via `?`, and — since
    /// `Queue`/a pushing source both treat a `Sink::consume` failure as
    /// "drop this one buffer, report on the bus, keep going" — a Pacer
    /// that hits this on one buffer must still pace the next one normally.
    #[test]
    fn a_pathological_pts_jump_is_a_typed_error_not_silent_passthrough() {
        let clock = Arc::new(Clock::new());
        let mut pacer = paced("pacer", ffmpeg::Rational::new(1, 1), &clock);

        assert!(pacer.consume(packet(-1)).is_ok(), "establishes the origin");

        let error = pacer
            .consume(packet(i64::MAX))
            .expect_err("pts far enough from the origin to overflow the subtraction");
        assert!(matches!(
            error,
            crate::Error::PacerError(PacerError::UnpaceableTimestamp { pts: i64::MAX })
        ));
        assert!(
            pacer.pending.is_empty(),
            "the overflowing buffer must not get stuck in `pending`"
        );

        // The pacer itself must still be usable afterward: a `Some` result
        // (not a further error) for an ordinary pts relative to the same
        // origin.
        assert!(pacer.wait_for(Some(0)).is_ok());
    }
}
