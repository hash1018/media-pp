use std::{
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use crate::pp_log::{PpLog, pp_info};
use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    control::{ControlReceiver, drain_control},
    element::{Element, ElementType, Source, SourceElement, element_pp_log},
    pad::SrcPad,
    pool::UnboundObjectPool,
    schedule::PeriodicSchedule,
};

/// Errors specific to `TestVideoSource`. Converts into the crate-wide
/// `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum TestVideoSourceError {
    #[error("TestVideoSource doesn't support seeking a generated stream")]
    SeekUnsupported,
}

/// Construction-time options for [`TestVideoSource::new`].
#[derive(Debug, Clone, Copy)]
pub struct TestVideoOptions {
    pub width: u32,
    pub height: u32,
    /// How fast `pts` advances per generated frame, and — since
    /// [`TestVideoSource`] self-paces to this same rate on a drift-free
    /// absolute schedule (see its own docs) — how fast frames actually
    /// get generated/pushed in real time, precisely enough that a
    /// downstream [`crate::elements::Pacer`] against
    /// [`TestVideoSource::time_base`] turns out not to be needed purely
    /// for smooth `D3d12Renderer` output (confirmed in
    /// `examples/render/test_video`).
    pub framerate: ffmpeg::Rational,
}

impl Default for TestVideoOptions {
    fn default() -> Self {
        Self {
            width: 640,
            height: 480,
            framerate: ffmpeg::Rational::new(30, 1),
        }
    }
}

/// Generates a synthetic, moving-diagonal-gradient video stream —
/// GStreamer's `videotestsrc` equivalent. No real decode/demux involved:
/// `run()` fabricates one `Pixel::YUV420P` frame per tick, stamps it with
/// an increasing `pts` (one tick per frame, in [`TestVideoSource::time_base`]'s
/// units), and pushes it straight downstream — useful for exercising
/// `SwScaler`/`Pacer`/`D3d12Renderer`/etc. without a real file or camera.
/// `D3d12Renderer` in particular already handles `Pixel::YUV420P` on its
/// CPU-upload path, so this can feed a renderer directly, no decoder
/// needed.
///
/// Self-paces to `options.framerate` on a drift-free absolute schedule
/// (`next_due += frame_interval` each tick in `run`, not "sleep
/// `frame_interval` since the last push" — the latter accumulates drift,
/// since generation itself always takes some nonzero time) — unlike
/// `FileDemuxer`/`RtspSource` (which push as fast as they can and leave
/// real-time pacing entirely to a downstream `Pacer`), this element's
/// "real time" isn't defined by anything external; it's whatever
/// `framerate` says it should be, so there's no reason not to generate at
/// exactly that rate itself.
///
/// Confirmed (`examples/render/test_video`, with and without a
/// downstream `Pacer`) that this is actually enough on its own for
/// smooth `D3d12Renderer` output, vsync-locked presentation included — an
/// earlier version of this doc claimed self-pacing alone was *not*
/// enough and a `Pacer` was still required, reasoning that only the
/// *average* rate was being kept correct, not *when* each frame lines up
/// against the vsync grid. That reasoning wasn't wrong about the
/// mechanism, but the fix turned out to already be in place here: a
/// relative "since last push" schedule genuinely can drift out of phase
/// over time, but this element was rewritten to the absolute schedule
/// described above specifically to close that gap, and testing without a
/// `Pacer` afterward showed no judder. See
/// `crate::elements::DxgiCaptureSource`'s own docs for the same
/// conclusion reached the same way, including a case (`SwScaler` sitting
/// between source and renderer) this element doesn't have.
///
/// Runs until `Stop` — never reaches `Eos` on its own (no frame-count
/// limit is exposed, deliberately, mirroring a live camera source more
/// than a file).
pub struct TestVideoSource {
    pp_log: PpLog,
    name: Arc<str>,
    options: TestVideoOptions,
    pad: SrcPad,
    frame_index: i64,
    /// `1 / options.framerate`, precomputed once — how long to wait
    /// between generated frames. `Duration::ZERO` (never sleeps, same as
    /// this element's old unpaced behavior) if `framerate`'s numerator is
    /// `0`, which would otherwise make this an infinite/undefined
    /// duration.
    frame_interval: Duration,
    /// Reused across every generated frame — see [`UnboundObjectPool`]'s
    /// docs. `init` builds a fresh `Pixel::YUV420P` frame at this
    /// element's fixed size; the next `generate_frame` call overwrites
    /// every pixel anyway, so `release` has nothing to reset.
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
}

impl TestVideoSource {
    pub fn new(name: impl Into<String>, options: TestVideoOptions) -> Self {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::TestVideoSource, &name, None);
        let pad = SrcPad::new(format!("{name}_src"));
        pp_info!(
            pp_log: &pp_log,
            "created: {}x{}, framerate={}",
            options.width,
            options.height,
            options.framerate
        );
        let (width, height) = (options.width, options.height);
        let pool = UnboundObjectPool::new(
            0,
            move || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::YUV420P, width, height),
            |_| {},
        );
        // See `frame_interval`'s own docs on the `numerator() > 0` guard.
        let frame_interval = if options.framerate.numerator() > 0 {
            Duration::from_secs_f64(
                options.framerate.denominator() as f64 / options.framerate.numerator() as f64,
            )
        } else {
            Duration::ZERO
        };
        Self {
            name,
            pp_log,
            options,
            pad,
            frame_index: 0,
            frame_interval,
            pool,
        }
    }

    /// The unit each generated frame's `pts` is expressed in — what you
    /// need to construct a matching [`crate::elements::Pacer`].
    pub fn time_base(&self) -> ffmpeg::Rational {
        ffmpeg::Rational::new(
            self.options.framerate.denominator(),
            self.options.framerate.numerator(),
        )
    }

    /// Fabricates the next frame: a diagonal gradient on the Y plane that
    /// shifts by one step per frame (so it visibly moves once played
    /// back), flat neutral chroma (grayscale — color isn't the point,
    /// motion/format correctness is).
    fn generate_frame(&mut self) -> crate::pool::UnboundObjectPoolRef<ffmpeg::frame::Video> {
        let mut frame = self.pool.get();

        let offset = self.frame_index;
        let width = self.options.width as usize;
        let y_stride = frame.stride(0);
        let y_height = frame.plane_height(0) as usize;
        {
            let y_plane = frame.data_mut(0);
            for row in 0..y_height {
                for col in 0..width {
                    y_plane[row * y_stride + col] =
                        ((col as i64 + row as i64 + offset) % 256) as u8;
                }
            }
        }
        for plane in [1usize, 2usize] {
            frame.data_mut(plane).fill(128);
        }

        frame.set_pts(Some(self.frame_index));
        self.frame_index += 1;
        frame
    }
}

impl Element for TestVideoSource {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::TestVideoSource
    }

    fn pp_log(&self) -> &crate::pp_log::PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut crate::pp_log::PpLog {
        &mut self.pp_log
    }
}

impl Source for TestVideoSource {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl SourceElement for TestVideoSource {
    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> crate::error::Result<()> {
        pp_info!(self, "started");
        let mut schedule = PeriodicSchedule::new(self.frame_interval, Instant::now());
        loop {
            let outcome = drain_control(control, self, bus)?;
            if outcome.stopped {
                pp_info!(self, "stopped");
                return Ok(());
            }
            if outcome.paused_for > Duration::ZERO {
                schedule.resume_after_pause(outcome.paused_for, Instant::now());
            }
            thread::sleep(schedule.remaining(Instant::now()));

            let frame = self.generate_frame();
            // A downstream failure drops just this one frame — same
            // "report, don't die" contract `Queue`'s worker gives a
            // failing `Sink` — rather than ending this whole source
            // thread over it.
            if let Err(error) = self.pad.push(MediaBuffer::Video(Arc::new(frame))) {
                bus.post(
                    &self.pp_log,
                    BusEvent::Error {
                        element_type: ElementType::TestVideoSource,
                        name: self.name.clone(),
                        error,
                    },
                );
            }
            // Advance only now that this tick's own work (generate + push,
            // which a slow downstream can stretch arbitrarily) is done —
            // `advance_after_tick`'s resync check needs `now` to reflect
            // that, or one abnormally slow tick's own catch-up frame slips
            // through uncapped before the next iteration ever notices.
            schedule.advance_after_tick(Instant::now());
        }
    }

    fn seek(&mut self, _target: std::time::Duration) -> crate::error::Result<std::time::Duration> {
        Err(TestVideoSourceError::SeekUnsupported.into())
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Mutex, thread, time::Duration};

    use crate::pp_log::PpLog;

    use super::*;
    use crate::{control::ControlMsg, element::Sink, pipeline::Pipeline};

    type VideoObservation = (ffmpeg::format::Pixel, u32, u32, Option<i64>);
    type RecordedFrames = Arc<Mutex<Vec<VideoObservation>>>;

    /// Captures every frame's `(format, width, height, pts)` it sees, in
    /// order — enough to check both pixel format/size and that `pts`
    /// actually advances frame over frame.
    struct RecordingSink {
        pp_log: PpLog,
        seen: RecordedFrames,
    }

    impl Element for RecordingSink {
        fn name(&self) -> Arc<str> {
            "recorder".into()
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

    impl Sink for RecordingSink {
        fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
            if let MediaBuffer::Video(frame) = buf {
                self.seen.lock().unwrap().push((
                    frame.format(),
                    frame.width(),
                    frame.height(),
                    frame.pts(),
                ));
            }
            Ok(())
        }
        fn control(&mut self, _msg: ControlMsg) -> crate::error::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn generates_correctly_sized_yuv420p_frames_with_increasing_pts() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = RecordingSink {
            seen: seen.clone(),
            pp_log: element_pp_log(ElementType::Other, "recorder", None),
        };
        let source = TestVideoSource::new(
            "test-video",
            TestVideoOptions {
                width: 16,
                height: 16,
                framerate: ffmpeg::Rational::new(30, 1),
            },
        );

        let pipeline = Pipeline::new("test", source, |source, ctx| {
            let branch = ctx.branch().to(Box::new(sink))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("test pipeline wiring must succeed");

        pipeline.run();
        // Long enough to observe several ticks at the 30fps `framerate`
        // above (self-paced since `TestVideoSource` now generates at that
        // rate itself — see its own docs), not just one or two.
        thread::sleep(Duration::from_millis(200));
        pipeline.stop();

        // Blocks until every `Bus` handle has dropped — i.e. the source
        // thread has actually exited, not just acked `Stop`.
        pipeline.bus().log_events();

        let frames = seen.lock().unwrap();
        assert!(!frames.is_empty(), "expected at least one generated frame");
        for window in frames.windows(2) {
            let (format, width, height, pts) = window[0];
            assert_eq!(format, ffmpeg::format::Pixel::YUV420P);
            assert_eq!((width, height), (16, 16));
            assert!(
                window[1].3 > pts,
                "expected pts to strictly increase frame over frame, got {:?} then {:?}",
                pts,
                window[1].3
            );
        }
    }

    #[test]
    fn seek_is_explicitly_unsupported() {
        let mut source = TestVideoSource::new("test-video", TestVideoOptions::default());
        assert!(source.seek(Duration::from_secs(1)).is_err());
    }

    /// Records the wall-clock `Instant` each frame arrives at, rather than
    /// its content — what
    /// [`resuming_after_a_pause_does_not_dump_a_burst_of_catch_up_frames`]
    /// needs to tell a steady post-resume framerate apart from a burst.
    struct TimestampSink {
        pp_log: PpLog,
        seen: Arc<Mutex<Vec<Instant>>>,
    }

    impl Element for TimestampSink {
        fn name(&self) -> Arc<str> {
            "timestamp-recorder".into()
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

    impl Sink for TimestampSink {
        fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
            if matches!(buf, MediaBuffer::Video(_)) {
                self.seen.lock().unwrap().push(Instant::now());
            }
            Ok(())
        }
        fn control(&mut self, _msg: ControlMsg) -> crate::error::Result<()> {
            Ok(())
        }
    }

    /// Regression test for the pause/resume scheduling bug: `next_due` is
    /// an absolute `Instant` deadline, and real time keeps moving while
    /// [`Pipeline::pause`] blocks this source's own loop inside
    /// `drain_control`. Without shifting `next_due` forward by however
    /// long the pause actually lasted (`ControlOutcome::paused_for`),
    /// `Resume` would find a deadline that's been sitting in the past the
    /// whole time it was frozen and dump every "missed" frame back to
    /// back instead of picking the steady framerate back up.
    #[test]
    fn resuming_after_a_pause_does_not_dump_a_burst_of_catch_up_frames() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = TimestampSink {
            seen: seen.clone(),
            pp_log: element_pp_log(ElementType::Other, "timestamp-recorder", None),
        };
        let source = TestVideoSource::new(
            "test-video",
            TestVideoOptions {
                width: 16,
                height: 16,
                framerate: ffmpeg::Rational::new(50, 1), // 20ms/frame
            },
        );

        let pipeline = Pipeline::new("pause-resume-test", source, |source, ctx| {
            let branch = ctx.branch().to(Box::new(sink))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("test pipeline wiring must succeed");

        pipeline.run();
        thread::sleep(Duration::from_millis(60));
        pipeline.pause();
        thread::sleep(Duration::from_millis(400));

        let resumed_at = Instant::now();
        pipeline.resume();
        thread::sleep(Duration::from_millis(120));
        pipeline.stop();
        pipeline.bus().log_events();

        let after_resume = seen
            .lock()
            .unwrap()
            .iter()
            .filter(|&&t| t >= resumed_at)
            .count();
        // At 50fps, ~120ms of real time after resume owes ~6 frames.
        // Well under this bound if paced steadily; a 400ms pause treated
        // as owed catch-up work would dump ~20 frames virtually at once,
        // comfortably clearing it.
        assert!(
            after_resume <= 12,
            "expected a steady framerate after resume, not a burst of catch-up frames: \
             {after_resume} frames arrived within 120ms of resuming"
        );
    }

    struct SlowFirstFrameSink {
        pp_log: PpLog,
        tx: crossbeam_channel::Sender<Instant>,
        slow_duration: Duration,
        delayed: bool,
    }

    impl Element for SlowFirstFrameSink {
        fn name(&self) -> Arc<str> {
            "slow-sink".into()
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

    impl Sink for SlowFirstFrameSink {
        fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
            if matches!(buf, MediaBuffer::Video(_)) {
                if !self.delayed {
                    self.delayed = true;
                    thread::sleep(self.slow_duration);
                }
                // Timestamped after any delay, not before — this marks
                // when the downstream actually became free again, the
                // reference point the next frame's arrival gets measured
                // against.
                let _ = self.tx.send(Instant::now());
            }
            Ok(())
        }
        fn control(&mut self, _msg: ControlMsg) -> crate::error::Result<()> {
            Ok(())
        }
    }

    /// Regression test for the missing processing-delay clamp:
    /// `next_due += self.frame_interval` alone (no follow-up "did that
    /// still land in the past?" check) let one abnormally slow downstream
    /// `consume()` call leave `next_due` many intervals behind `now`, and
    /// every one of those intervals would fire back-to-back with no sleep
    /// between them as soon as the loop got a chance to run again — a
    /// burst of catch-up frames. `SwVideoCompositor::run` already guarded
    /// its own composition step this way; `TestVideoSource::run` now
    /// applies the same clamp right after advancing `next_due` — and only
    /// *after* generate+push (this test's other regression: advancing
    /// before push meant the clamp couldn't see the slow tick's own delay
    /// until the following iteration, letting exactly one immediate
    /// catch-up frame slip through right after the slow one finished).
    #[test]
    fn a_slow_sink_does_not_cause_a_burst_of_catch_up_frames() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let sink = SlowFirstFrameSink {
            tx,
            slow_duration: Duration::from_millis(300),
            delayed: false,
            pp_log: element_pp_log(ElementType::Other, "slow-sink", None),
        };
        let source = TestVideoSource::new(
            "test-video",
            TestVideoOptions {
                width: 16,
                height: 16,
                framerate: ffmpeg::Rational::new(20, 1), // 50ms/frame
            },
        );

        let pipeline = Pipeline::new("slow-sink-test", source, |source, ctx| {
            let branch = ctx.branch().to(Box::new(sink))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("test pipeline wiring must succeed");

        pipeline.run();
        let slow_done = rx
            .recv_timeout(Duration::from_secs(1))
            .expect("expected the first (slow) frame to finish");
        let after_slow = rx
            .recv_timeout(Duration::from_millis(500))
            .expect("expected the frame right after the slow one");
        let steady = rx
            .recv_timeout(Duration::from_millis(500))
            .expect("expected a third frame at steady cadence");
        pipeline.stop();
        pipeline.bus().log_events();

        let immediate_gap = after_slow.saturating_duration_since(slow_done);
        assert!(
            immediate_gap >= Duration::from_millis(25),
            "expected the frame right after the slow one to wait a steady \
             ~50ms interval, not follow immediately just because the slow \
             sink had finally caught up: got {immediate_gap:?}"
        );

        let gap = steady.saturating_duration_since(after_slow);
        assert!(
            gap >= Duration::from_millis(25),
            "expected steady ~50ms cadence once the slow sink caught up, not a \
             burst of catch-up frames immediately following it: got {gap:?}"
        );
    }
}
