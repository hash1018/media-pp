use std::{sync::Arc, time::Duration};

use crate::pp_log::{PpLog, pp_info};
use ffmpeg_next as ffmpeg;

use crate::{
    buffer::MediaBuffer,
    contract::{MediaKind, MemoryDomain, OutputContract, PortContract},
    element::{Attached, ElementType, Identity, Produced, Source, SourceStage, Wait},
    pool::UnboundObjectPool,
    produce::source_stage,
    schedule::LiveSchedule,
};

/// The size and rate of the pictures a [`TestVideoSource`] makes.
#[derive(Debug, Clone, Copy)]
pub struct TestVideoOptions {
    /// Width of each generated frame in pixels.
    pub width: u32,
    /// Height of each generated frame in pixels.
    pub height: u32,
    /// How many pictures a second it makes: both how fast `pts` advances
    /// and how fast pictures actually come, since it keeps its own time —
    /// no [`crate::elements::Pacer`] is needed after it.
    pub frame_rate: ffmpeg::Rational,
}

impl Default for TestVideoOptions {
    fn default() -> Self {
        Self {
            width: 640,
            height: 480,
            frame_rate: ffmpeg::Rational::new(30, 1),
        }
    }
}

/// Makes a moving test picture at a set size and rate, for trying a pipeline
/// without a file or a camera.
///
/// GStreamer's `videotestsrc` equivalent. It hands on decoded YUV 4:2:0
/// pictures in host memory, already at their own rate, so it can feed an
/// encoder, a scaler or a renderer directly — no decoder and no
/// [`Pacer`](crate::elements::Pacer). It is live and runs until stopped.
///
/// # Example
///
/// ```no_run
/// # fn main() -> media_pp::Result<()> {
/// use media_pp::elements::{FrameCounter, TestVideoOptions, TestVideoSource};
/// use media_pp::ffmpeg::Rational;
/// use media_pp::pipeline::Pipeline;
///
/// let options = TestVideoOptions { width: 1280, height: 720, frame_rate: Rational(60, 1) };
/// let source = TestVideoSource::new("pattern", options);
/// let (counter, _frames) = FrameCounter::new("counter");
/// let (pipeline, ()) = Pipeline::new("test", source, |source, ctx| {
///     let branch = ctx.branch().to(counter)?;
///     ctx.attach(source, 0, branch)?;
///     Ok(())
/// })?;
/// pipeline.run()?;
/// # Ok(())
/// # }
/// ```
///
/// # How it works
///
/// No decode or demux is involved: each time the pipeline asks, it makes
/// one `Pixel::YUV420P` picture, stamped with a `pts` one higher than the
/// last in [`TestVideoSource::time_base`]'s units, and hands it on — enough
/// to exercise a scaler, an upload or a renderer without a file or a camera.
///
/// It keeps its own time at `options.frame_rate`, on an absolute schedule:
/// each picture is due one frame interval after the one before was *due*,
/// not after it was made, so the time making one takes never adds up into
/// drift. A file source hands its pictures on as fast as it reads them and
/// leaves the pace to a `Pacer`; this one has no outside clock to follow, so
/// it keeps the rate itself, and a renderer after it shows it smoothly with
/// no `Pacer` between. It is live, like a camera.
///
/// It runs until stopped — it never ends its stream on its own, as a camera
/// does not.
pub struct TestVideoSource(SourceStage<Generating>);

source_stage!(TestVideoSource);

/// What a [`TestVideoSource`] makes, one frame when asked: all of its
/// work, which the framework makes the source.
struct Generating {
    pp_log: PpLog,
    options: TestVideoOptions,
    frame_index: i64,
    /// `1 / options.frame_rate`, precomputed once — how long to wait
    /// between generated frames. `Duration::ZERO` (never sleeps, same as
    /// this element's old unpaced behavior) if `frame_rate`'s numerator is
    /// `0`, which would otherwise make this an infinite/undefined
    /// duration.
    frame_interval: Duration,
    /// Reused across every generated frame — see [`UnboundObjectPool`]'s
    /// docs. `init` builds a fresh `Pixel::YUV420P` frame at this
    /// element's fixed size; the next `generate_frame` call overwrites
    /// every pixel anyway, so `release` has nothing to reset.
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
    /// When each frame is due, on the clock a pause does not move — made
    /// as the first is asked for.
    schedule: LiveSchedule,
}

impl TestVideoSource {
    /// Creates an unbounded synthetic video source with the requested output
    /// definition.
    pub fn new(name: impl Into<String>, options: TestVideoOptions) -> Self {
        let identity = Identity::new(ElementType::TestVideoSource, name);
        let pp_log = identity.pp_log().clone();
        pp_info!(
            pp_log: &pp_log,
            "created: {}x{}, frame_rate={}",
            options.width,
            options.height,
            options.frame_rate
        );
        let (width, height) = (options.width, options.height);
        let pool = UnboundObjectPool::new(
            0,
            move || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::YUV420P, width, height),
            |_| {},
        );
        // See `frame_interval`'s own docs on the `numerator() > 0` guard.
        let frame_interval = if options.frame_rate.numerator() > 0 {
            Duration::from_secs_f64(
                options.frame_rate.denominator() as f64 / options.frame_rate.numerator() as f64,
            )
        } else {
            Duration::ZERO
        };
        Self(SourceStage::new(
            identity,
            Generating {
                pp_log,
                options,
                frame_index: 0,
                frame_interval,
                pool,
                schedule: LiveSchedule::default(),
            },
        ))
    }

    /// The unit each picture's `pts` counts in: one frame interval. Each
    /// picture carries it as well, so nothing after the source needs to be
    /// told it.
    pub fn time_base(&self) -> ffmpeg::Rational {
        self.0.inner().time_base()
    }

    /// The next frame at once, not when it is due — for a test fixture
    /// written as fast as it encodes, stamped as a paced run stamps it.
    #[cfg(test)]
    pub(crate) fn next_frame(&mut self) -> MediaBuffer {
        MediaBuffer::Video(Arc::new(self.0.inner_mut().generate_frame()).into())
    }
}

impl Generating {
    fn time_base(&self) -> ffmpeg::Rational {
        ffmpeg::Rational::new(
            self.options.frame_rate.denominator(),
            self.options.frame_rate.numerator(),
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
        crate::buffer::set_time_base(&mut frame, self.time_base());
        self.frame_index += 1;
        frame
    }
}

impl Source for Generating {
    fn is_live(&self) -> bool {
        true
    }

    fn output_contract(&self) -> OutputContract {
        OutputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::System)
                .with_layouts(crate::contract::PixelLayoutSet::YUV420P),
        )
    }

    /// The next frame, once it is due on the schedule — kept on the clock a
    /// pause does not move, so playing on after one is not a burst of the
    /// frames it would have owed.
    fn produce(&mut self, wait: &mut Wait<'_>) -> crate::error::Result<Produced> {
        let tick = self.schedule.next(self.frame_interval, wait.now());
        if !tick.come(wait, Duration::MAX) {
            return Ok(Produced::Nothing);
        }
        let frame = self.generate_frame();
        self.schedule.made();
        Ok(Produced::Buffer(MediaBuffer::Video(Arc::new(frame).into())))
    }

    fn attached(&mut self, cx: &Attached<'_>) {
        self.pp_log = cx.pp_log().clone();
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::Mutex,
        thread,
        time::{Duration, Instant},
    };

    use crate::pp_log::PpLog;

    use super::*;
    use crate::element::{Element, element_pp_log};
    use crate::{
        element::{RawSink, RawSource},
        pipeline::Pipeline,
    };

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

    impl RawSink for RecordingSink {
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
                frame_rate: ffmpeg::Rational::new(30, 1),
            },
        );

        let (pipeline, ()) = Pipeline::new("test", source, |source, ctx| {
            let branch = ctx.branch().to(sink)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("test pipeline wiring must succeed");

        pipeline.run().unwrap();
        // Long enough to observe several ticks at the 30fps `frame_rate`
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
    fn a_generated_stream_cannot_be_sought() {
        let mut source = TestVideoSource::new("test-video", TestVideoOptions::default());
        assert!(source.as_seekable().is_none());
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

    impl RawSink for TimestampSink {
        fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
            if matches!(buf, MediaBuffer::Video(_)) {
                self.seen.lock().unwrap().push(Instant::now());
            }
            Ok(())
        }
    }

    /// Regression test for the pause/resume scheduling bug: the
    /// `LiveSchedule`'s deadline is absolute, and real time keeps moving
    /// while [`Pipeline::pause`] holds this source. The framework keeps a
    /// pause out of the clock `Wait::now` reads; counted in, `Resume`
    /// would find a deadline that had been sitting in the past the whole
    /// time it was frozen and dump every "missed" frame back to back
    /// instead of picking the steady frame rate back up.
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
                frame_rate: ffmpeg::Rational::new(50, 1), // 20ms/frame
            },
        );

        let (pipeline, ()) = Pipeline::new("pause-resume-test", source, |source, ctx| {
            let branch = ctx.branch().to(sink)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("test pipeline wiring must succeed");

        pipeline.run().unwrap();
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

    impl RawSink for SlowFirstFrameSink {
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
    }

    /// Regression test for the missing processing-delay clamp: adding the
    /// frame interval to the last deadline alone (no follow-up "did that
    /// still land in the past?" check) let one abnormally slow downstream
    /// push leave the deadline many intervals behind now, and every one of
    /// those intervals would fire back-to-back with no wait between them —
    /// a burst of catch-up frames. `LiveSchedule`, which the compositors
    /// and the screen captures share, drops the missed ticks instead, and only *after*
    /// generate+push (this test's other regression: advancing before push
    /// meant the clamp couldn't see the slow tick's own delay until the
    /// following one, letting exactly one immediate catch-up frame slip
    /// through right after the slow one finished).
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
                frame_rate: ffmpeg::Rational::new(20, 1), // 50ms/frame
            },
        );

        let (pipeline, ()) = Pipeline::new("slow-sink-test", source, |source, ctx| {
            let branch = ctx.branch().to(sink)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("test pipeline wiring must succeed");

        pipeline.run().unwrap();
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
