use std::{collections::VecDeque, sync::Arc, thread, time::Duration};

use crate::pp_log::{PpLog, pp_info, pp_warn};
use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, OutputContract},
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    pad::SrcPad,
    playback_clock::PlaybackClock,
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
    /// all: it does not fit in nanoseconds at this stream's time base.
    ///
    /// Reported rather than swallowed because the alternative is the buffer
    /// going through *unpaced*, which is a whole stream arriving at once.
    #[error("pts {pts} cannot be paced against this pipeline's timeline")]
    UnpaceableTimestamp {
        /// The timestamp, in its own stream's units.
        pts: i64,
    },

    /// A buffer arrived before the pipeline gave this pacer its playback
    /// clock, which
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
/// Every `Pacer` in a pipeline (one per stream — video, audio, ...) measures
/// against the same [`crate::playback_clock::PlaybackClock`], so they agree
/// on one t=0 instead of each anchoring to its own first frame. That
/// agreement is what keeps the picture with the sound: the offset a
/// container gives its streams is part of their sync, and a pacer zeroing on
/// its own stream would throw it away.
///
/// Which clock that is can change while it runs. A pipeline starts on the
/// pause-aware wall clock and hands the position to an audio renderer once
/// its endpoint is running, and a pacer follows: what it waits on is where
/// playback has actually reached, not where a wall-clock deadline computed
/// at the start says it should be.
pub struct Pacer {
    pp_log: PpLog,
    name: Arc<str>,
    time_base: TimeBase,
    /// The pipeline's, given by [`Element::attach_context`] — see there for
    /// why this is not something the caller supplies.
    ///
    /// The playback clock rather than the wall one, and it holds the origin
    /// too: a container's streams do not start at the same timestamp, and a
    /// pacer that zeroed on its own first timestamp would play them as
    /// though they did. Nothing is cached here — not the origin, not an
    /// anchor — because both can move underneath: a pause shifts the wall
    /// timeline, a seek clears the origin, and an audio renderer taking the
    /// clock replaces the rate the position advances at.
    playback_clock: Option<Arc<PlaybackClock>>,
    /// The latest pipeline interrupt this pacer has acknowledged through
    /// `control()`. A newer clock epoch means pause/seek/stop is waiting for
    /// the current `consume()` call to return. `Queue`'s own worker only
    /// checks its control channel *between* buffers (see its type docs) —
    /// it can't preempt a `consume()` call already in flight, and this
    /// pacer's own wait is exactly that kind of long-running call.
    interrupt_epoch: u64,
    /// The longest a single buffer may hold this pacer before its timestamp
    /// is read as a new timeline rather than a distant one, or `None` for a
    /// timeline that cannot restart — see
    /// [`Pacer::with_discontinuity_limit`].
    discontinuity_limit: Option<Duration>,
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
            playback_clock: None,
            discontinuity_limit: None,
            interrupt_epoch: 0,
            prerolling: false,
            pending: VecDeque::new(),
            pad,
        })
    }

    /// The same, for a stream whose timeline can restart under it.
    ///
    /// A file's timestamps only ever move forward from where they began, so
    /// a buffer due far ahead is a real gap in the stream and waiting it out
    /// is the correct thing to do. A live sender is not like that: a camera
    /// that reboots, or an RTP timestamp base that wraps, hands over a
    /// timestamp with no relation to the one before it, and a pacer that
    /// believes it sleeps for as long as the jump says — a still picture,
    /// no error, and nothing to reconnect from, since as far as the pipeline
    /// is concerned it is working.
    ///
    /// Past `limit` such a jump is read as a new timeline: the origin
    /// re-anchors so the buffer that carried it is due now, and a warning
    /// says so. Both branches of one source see the same jump and re-anchor
    /// within a buffer of each other, so the picture keeps its sound.
    ///
    /// Pick a `limit` longer than the longest gap the stream can really
    /// have — for most cameras a second or two of nothing is already a
    /// problem, not a pause. Shorter than the spacing between its own
    /// frames and every ordinary wait reads as a jump, which is this pacer
    /// no longer pacing at all.
    pub fn with_discontinuity_limit(
        name: impl Into<String>,
        time_base: ffmpeg::Rational,
        limit: Duration,
    ) -> Result<Self, PacerError> {
        let mut pacer = Self::new(name, time_base)?;
        pp_info!(pp_log: &pacer.pp_log, "timeline jumps beyond {limit:?} re-anchor it");
        pacer.discontinuity_limit = Some(limit);
        Ok(pacer)
    }

    /// Blocks until `pts` is due against the pipeline's playback clock.
    ///
    /// Returns `Ok(false)` if pause/seek/stop interrupts the wait; the caller
    /// retains that in-flight buffer and returns so the owning worker can
    /// process the pending control request. Frames without a pts (`None`)
    /// pass straight through. `Err` only for a `pts` too pathological to
    /// pace against at all (see [`PacerError::UnpaceableTimestamp`]) — the
    /// caller drops that one buffer rather than treating it as interrupted.
    ///
    /// How long is left is asked again on every pass rather than computed
    /// once into a deadline. Under a wall-clock master the two are the same
    /// arithmetic; under an audio one they are not, because the position
    /// advances at the device's rate and a deadline named up front would be
    /// wrong by however far that rate differs.
    fn wait_for(&mut self, pts: Option<i64>) -> Result<bool, PacerError> {
        let playback = self.playback_clock.clone().ok_or(PacerError::NotAttached)?;
        if playback.interrupt_epoch() != self.interrupt_epoch {
            return Ok(false);
        }
        if self.prerolling {
            return Ok(true);
        }
        let Some(pts) = pts else { return Ok(true) };
        // Rescaled before anything is compared, because the origin this is
        // measured against is shared with streams in other units — a
        // container's audio and video rarely count in the same ticks.
        // Integer rescale rather than `pts as f64 * f64::from(time_base)`:
        // the latter loses precision (and the numerator, if computed by
        // naive division) over a long-running stream; see `MediaTimestamp`'s
        // own docs.
        let pts_ns = MediaTimestamp::new_unchecked(pts, self.time_base).rescale(nanoseconds());
        // `av_rescale_q_rnd` answers a value it cannot represent with
        // `INT64_MIN`, which is also FFmpeg's "no timestamp". Checked before
        // the clock is asked, so a pathological first buffer cannot become
        // the timeline every other stream is measured against.
        if pts_ns == i64::MIN {
            return Err(PacerError::UnpaceableTimestamp { pts });
        }
        loop {
            if playback.interrupt_epoch() != self.interrupt_epoch {
                return Ok(false);
            }
            let remaining = playback.remaining(pts_ns);
            if remaining.is_zero() {
                return Ok(true);
            }
            if let Some(limit) = self.discontinuity_limit
                && remaining > limit
            {
                // A jump this far forward is a sender that restarted its
                // timeline, not a stream with a gap that long in it — see
                // `with_discontinuity_limit`.
                pp_warn!(
                    self,
                    "timeline jumped {remaining:?} ahead; re-anchoring on it"
                );
                playback.re_anchor(pts_ns);
                return Ok(true);
            }
            thread::sleep(remaining.min(INTERRUPT_POLL_INTERVAL));
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
        self.interrupt_epoch = context.playback_clock.interrupt_epoch();
        self.playback_clock = Some(Arc::clone(&context.playback_clock));
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
        // Flush discards an interrupted old-timeline buffer; Seek then drops
        // the origin so the new timeline anchors on whichever stream reaches
        // its landing place first.
        // A pacer that was never wired has no clock to acknowledge, and
        // nothing is going to send it control either — see
        // `PacerError::NotAttached`.
        if let Some(playback) = &self.playback_clock {
            self.interrupt_epoch = playback.interrupt_epoch();
        }
        match msg {
            ControlMsg::Flush => self.pending.clear(),
            ControlMsg::Seek(_) => {
                // The wall clock is left alone, which it could not be while
                // the origin was paired with `Clock::start()`: post-seek
                // timestamps restart near zero, and against a stale anchor
                // every one of them was already overdue. The playback clock
                // holds its origin as an elapsed offset instead, so it
                // re-anchors itself on the next buffer and the pipeline's
                // monotonic time — which a seek does not stop — keeps
                // running for everything else that reads it.
                if let Some(playback) = &self.playback_clock {
                    playback.reset_for_seek();
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
    use crate::clock::Clock;
    use crate::control::PrerollContext;
    use std::{
        sync::mpsc,
        time::{Duration, Instant},
    };

    /// One pipeline's context, around a clock the test keeps so it can
    /// interrupt and pause it.
    ///
    /// Shared between every pacer a test wires, because that is what a
    /// pipeline does and what the origin depends on: two contexts are two
    /// playback clocks, and two streams measured against separate origins
    /// have lost the offset between them before the test starts.
    fn context(clock: &Arc<Clock>) -> Arc<crate::element::Context> {
        Arc::new(crate::element::Context::for_test_with_clock(
            crate::bus::Bus::new().0,
            "test",
            crate::graph::PipelineGraph::new(),
            crate::graph::ElementId::for_test(1),
            Arc::clone(clock),
        ))
    }

    /// A pacer wired the way a pipeline wires one.
    fn paced(
        name: &str,
        time_base: ffmpeg::Rational,
        context: &Arc<crate::element::Context>,
    ) -> Pacer {
        let mut pacer = Pacer::new(name, time_base).expect("valid time base");
        pacer.attach_context(context);
        pacer
    }

    /// The same, with a limit on how long one buffer may hold it.
    fn paced_live(
        name: &str,
        time_base: ffmpeg::Rational,
        context: &Arc<crate::element::Context>,
        limit: Duration,
    ) -> Pacer {
        let mut pacer =
            Pacer::with_discontinuity_limit(name, time_base, limit).expect("valid time base");
        pacer.attach_context(context);
        pacer
    }

    /// A camera that reboots hands over a timestamp with no relation to the
    /// one before it. Waited out, that is a still picture for as long as the
    /// jump says — and nothing reports it, because as far as the pipeline is
    /// concerned the pacer is doing its job.
    #[test]
    fn a_live_timeline_that_jumps_is_re_anchored_rather_than_waited_out() {
        let clock = Arc::new(Clock::new());
        let context = context(&clock);
        // Milliseconds, and a limit longer than the spacing between the
        // frames below — a limit shorter than that would read the ordinary
        // wait for the next frame as a jump, which is what it is for the
        // caller to pick against its own stream.
        let mut pacer = paced_live(
            "pacer",
            ffmpeg::Rational::new(1, 1000),
            &context,
            Duration::from_millis(300),
        );
        assert!(pacer.wait_for(Some(0)).unwrap(), "the first pts anchors");

        let started = Instant::now();
        assert!(
            pacer.wait_for(Some(3_600_000)).unwrap(),
            "a jumped timestamp is released, not refused"
        );
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "an hour ahead must not be an hour of waiting: took {:?}",
            started.elapsed()
        );

        // And the new timeline is the one it paces against from here: 200ms
        // after the jump is 200ms of waiting, not another hour.
        let after = Instant::now();
        assert!(pacer.wait_for(Some(3_600_200)).unwrap(), "the next frame");
        let waited = after.elapsed();
        assert!(
            waited >= Duration::from_millis(150) && waited < Duration::from_secs(2),
            "200ms past the re-anchored origin: waited {waited:?}"
        );
    }

    /// The limit is not the default, and must not be: a file's timeline does
    /// not restart, so a gap in it is real and waiting it out is correct.
    #[test]
    fn a_pacer_without_a_limit_still_waits_out_a_distant_timestamp() {
        let clock = Arc::new(Clock::new());
        let context = context(&clock);
        let mut pacer = paced("pacer", ffmpeg::Rational::new(1, 1000), &context);
        assert!(pacer.wait_for(Some(0)).unwrap(), "the first pts anchors");

        let started = Instant::now();
        assert!(pacer.wait_for(Some(300)).unwrap(), "300ms into the stream");
        assert!(
            started.elapsed() >= Duration::from_millis(250),
            "the gap is the stream's own and has to be waited out: took {:?}",
            started.elapsed()
        );
    }

    fn packet(pts: i64) -> MediaBuffer {
        let mut packet = ffmpeg::Packet::empty();
        packet.set_pts(Some(pts));
        MediaBuffer::Packet(Arc::new(packet))
    }

    #[test]
    fn long_wait_returns_promptly_when_control_interrupts_it() {
        let clock = Arc::new(Clock::new());
        let context = context(&clock);
        let mut pacer = paced("pacer", ffmpeg::Rational::new(1, 1), &context);
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
        let context = context(&clock);
        let mut pacer = paced("pacer", ffmpeg::Rational::new(1, 1), &context);

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
        let context = context(&clock);
        let mut pacer = paced("pacer", ffmpeg::Rational::new(1, 1), &context);
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
        let context = context(&clock);
        // Milliseconds, so the numbers below read as what they are.
        let unit = ffmpeg::Rational::new(1, 1000);
        let mut audio = paced("audio", unit, &context);
        let mut video = paced("video", unit, &context);

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

    /// Where the position comes from once an audio renderer owns it.
    ///
    /// A pacer measures against the pipeline's playback clock rather than
    /// against the first timestamp it happened to see, so a stream joining a
    /// pipeline whose audio is already a second in is a second late, not at
    /// its own zero. Held apart from the wall clock on purpose: with an
    /// origin of its own this pacer would have released the two buffers
    /// below 200ms apart, and against the audio position both are already
    /// past.
    #[test]
    fn a_pacer_measures_against_the_audio_master_not_its_own_first_buffer() {
        let clock = Arc::new(Clock::new());
        let context = context(&clock);
        let mut pacer = paced("pacer", ffmpeg::Rational::new(1, 1000), &context);

        let audio = context
            .playback_clock
            .register_audio_master()
            .expect("nothing else holds the clock");
        // A second of audio played, five submitted, and still running.
        audio
            .publish(1_000_000_000, 5_000_000_000, true)
            .expect("the registration is live");

        let started = Instant::now();
        assert!(pacer.wait_for(Some(100)).unwrap(), "100ms is already past");
        assert!(pacer.wait_for(Some(300)).unwrap(), "so is 300ms");
        assert!(
            started.elapsed() < Duration::from_millis(100),
            "both are behind a position already at 1s and neither has \
             anything to wait for: took {:?}",
            started.elapsed()
        );
    }

    /// An audio master that has taken the clock but not started must not
    /// hold a pacer.
    ///
    /// This is the deadlock a renderer's deferred registration exists to
    /// avoid, seen from the other side: a branch attached to a running `Tee`
    /// sits behind a demuxer that cannot reach its first audio packet until
    /// the video queue drains, and a pacer waiting on a position no one has
    /// published yet is what would stop that queue draining.
    #[test]
    fn priming_does_not_hold_a_pacer() {
        let clock = Arc::new(Clock::new());
        let context = context(&clock);
        let mut pacer = paced("pacer", ffmpeg::Rational::new(1, 1000), &context);

        let _audio = context
            .playback_clock
            .register_audio_master()
            .expect("nothing else holds the clock");
        assert_eq!(
            context.playback_clock.master(),
            crate::playback_clock::PlaybackMaster::AudioPriming
        );

        let started = Instant::now();
        assert!(
            pacer.wait_for(Some(10_000)).unwrap(),
            "ten seconds ahead, and released anyway"
        );
        assert!(
            started.elapsed() < Duration::from_millis(100),
            "a priming master says nothing about where it is, and waiting on \
             that is the stall this is here to rule out: took {:?}",
            started.elapsed()
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
        let context = context(&clock);
        let mut pacer = paced("pacer", ffmpeg::Rational::new(1, 1), &context);

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

    /// Packets whose `pts` goes backwards are still paced to their own
    /// timeline.
    ///
    /// A `Pacer` in front of a decoder — where `webrtc_record` and
    /// `rtsp_serve` both put one — waits on `pts`, and a stream carrying
    /// B-frames hands it packets in decode order: `pts` jumps forward, then
    /// back behind a frame already released, over and over. Each of those is
    /// simply already due, so what comes out is bursty within a frame or two
    /// and correct across the stream. What must not happen is either end of
    /// getting that wrong — a wait on a timestamp read as far in the future,
    /// or an origin that moves and lets the whole stream through at once.
    #[test]
    fn a_reordered_packet_stream_is_paced_to_its_own_length() {
        use crate::elements::FileDemuxer;
        use crate::pipeline::Pipeline;
        use std::sync::atomic::{AtomicUsize, Ordering};

        const SECONDS: f64 = 2.0;

        let fixture = crate::test_support::synthesize_reordered("paced-reorder", SECONDS);
        let path = fixture.path.to_string_lossy().into_owned();
        let (demuxer, streams) = FileDemuxer::open("demuxer", &path).expect("open the fixture");
        let video = streams
            .iter()
            .find(|stream| stream.kind == ffmpeg::media::Type::Video)
            .expect("the fixture has video")
            .index;
        let time_base = demuxer.stream_time_base(video).expect("video time base");

        let seen = Arc::new(AtomicUsize::new(0));
        let counter = crate::elements::AppSink::new("paced", {
            let seen = Arc::clone(&seen);
            move |_| {
                seen.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
        });

        let started = Instant::now();
        let pipeline = Pipeline::new("paced-reorder", demuxer, move |source, context| {
            let branch = context
                .branch()
                .pipe(Pacer::new("pacer", time_base)?)
                .to(Box::new(counter))?;
            context.attach(source, video, branch)?;
            Ok(())
        })
        .expect("wire the paced stream");
        pipeline.run().expect("run it");
        for event in pipeline.bus().iter() {
            if matches!(event, crate::bus::BusEvent::Eos { .. }) {
                break;
            }
        }
        let elapsed = started.elapsed();
        pipeline.stop();

        assert!(
            seen.load(Ordering::Relaxed) > 0,
            "no packet reached the far side of the pacer"
        );
        // Generous on both sides: what this is watching for is a stall or a
        // whole stream let through at once, not a few milliseconds either
        // way.
        let content = Duration::from_secs_f64(SECONDS);
        assert!(
            elapsed >= content.mul_f64(0.5),
            "a reordered stream was let through in {elapsed:?}, well under the \
             {content:?} it describes"
        );
        assert!(
            elapsed <= content.mul_f64(2.5),
            "a reordered stream took {elapsed:?} to pace {content:?} of packets"
        );
    }
}
