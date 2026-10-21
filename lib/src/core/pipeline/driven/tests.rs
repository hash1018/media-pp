use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    task::{Context as TaskContext, Wake, Waker},
    thread::{self, ThreadId},
    time::{Duration, Instant},
};

use ffmpeg_next::{self as ffmpeg, Rescale, format::Pixel};

use super::*;
use crate::{
    buffer::{MediaBuffer, set_time_base},
    bus::BusEvent,
    element::{Element, ElementType, Produced, RawSink, Source, Wait, element_pp_log},
    elements::{AppSource, FrameCounter, Pacer, TestVideoOptions, TestVideoSource},
    error::Error,
    graph::GraphError,
    pp_log::PpLog,
    stream::StreamEvent,
};

/// What a [`Recorder`] heard, in order.
#[derive(Debug, Clone, PartialEq)]
enum Heard {
    Segment,
    /// A buffer, at its time, and the thread that handed it on.
    Data(Option<Duration>, ThreadId),
    Paused,
    Resumed,
    Stopped(ThreadId),
    Eos,
}

#[derive(Clone, Default)]
struct Record(Arc<Mutex<Vec<Heard>>>);

impl Record {
    fn push(&self, heard: Heard) {
        self.0.lock().unwrap().push(heard);
    }

    fn heard(&self) -> Vec<Heard> {
        self.0.lock().unwrap().clone()
    }

    fn data(&self) -> usize {
        self.heard()
            .iter()
            .filter(|heard| matches!(heard, Heard::Data(..)))
            .count()
    }

    fn has(&self, wanted: impl Fn(&Heard) -> bool) -> bool {
        self.heard().iter().any(wanted)
    }
}

/// A terminal that writes down everything it is handed and told.
struct Recorder {
    pp_log: PpLog,
    record: Record,
    panics: bool,
}

fn recorder(record: &Record) -> Recorder {
    Recorder {
        pp_log: element_pp_log(ElementType::Other, "recorder", None),
        record: record.clone(),
        panics: false,
    }
}

impl Element for Recorder {
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

impl RawSink for Recorder {
    fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
        assert!(!self.panics, "a sink that panics");
        let at = buf.pts().zip(buf.time_base()).map(|(pts, unit)| {
            Duration::from_nanos(pts.rescale(unit, ffmpeg::Rational(1, 1_000_000_000)) as u64)
        });
        self.record.push(Heard::Data(at, thread::current().id()));
        Ok(())
    }

    fn stream_event(&mut self, event: &StreamEvent) -> crate::error::Result<()> {
        self.record.push(match event {
            StreamEvent::Segment(_) => Heard::Segment,
            StreamEvent::Eos => Heard::Eos,
        });
        Ok(())
    }

    fn pausing(&mut self) -> crate::error::Result<()> {
        self.record.push(Heard::Paused);
        Ok(())
    }

    fn resuming(&mut self) -> crate::error::Result<()> {
        self.record.push(Heard::Resumed);
        Ok(())
    }

    fn stopping(&mut self) -> crate::error::Result<()> {
        self.record.push(Heard::Stopped(thread::current().id()));
        Ok(())
    }
}

fn camera() -> TestVideoSource {
    TestVideoSource::new(
        "camera",
        TestVideoOptions {
            width: 32,
            height: 24,
            ..TestVideoOptions::default()
        },
    )
}

/// A driven pipeline of `source` into a [`Recorder`] of `record`.
fn recorded<M>(
    source: impl IntoSource<M>,
    record: &Record,
) -> (DrivenPipeline, DrivenPipelineHandle) {
    let sink = recorder(record);
    let (pipeline, ()) = DrivenPipeline::new("driven", source, |source, ctx| {
        ctx.attach(source, 0, ctx.branch().to(sink)?)?;
        Ok(())
    })
    .unwrap();
    let handle = pipeline.handle();
    (pipeline, handle)
}

fn wait_for(what: &str, done: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !done() {
        assert!(Instant::now() < deadline, "{what}");
        thread::sleep(Duration::from_millis(2));
    }
}

/// Wakes the thread that polls.
struct Unpark(thread::Thread);

impl Wake for Unpark {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

/// Polls `pipeline` on this thread for `long`, as its owner would: again
/// when woken or when it asked to be. Says where the last poll left it.
fn poll_for(pipeline: &mut DrivenPipeline, long: Duration) -> Driven {
    let waker = Waker::from(Arc::new(Unpark(thread::current())));
    let mut cx = TaskContext::from_waker(&waker);
    let end = Instant::now() + long;
    loop {
        let driven = pipeline.poll(&mut cx);
        let now = Instant::now();
        let Driven::Pending { at } = driven else {
            return driven;
        };
        if now >= end {
            return driven;
        }
        let until = at.map_or(end, |at| at.min(end));
        thread::park_timeout(until.saturating_duration_since(now));
    }
}

/// A picture at `index` of a 25-a-second stream.
fn picture(index: i64) -> MediaBuffer {
    let mut frame = ffmpeg::frame::Video::new(Pixel::GRAY8, 16, 16);
    frame.set_pts(Some(index));
    set_time_base(&mut frame, ffmpeg::Rational(1, 25));
    MediaBuffer::video(frame)
}

/// Makes pictures as fast as it is asked, `left` of them, then ends.
struct Burst {
    made: i64,
    left: Option<i64>,
}

impl Source for Burst {
    fn is_live(&self) -> bool {
        false
    }

    fn drivable(&self) -> bool {
        true
    }

    fn produce(&mut self, _wait: &mut Wait<'_>) -> crate::error::Result<Produced> {
        if self.left.is_some_and(|left| self.made == left) {
            return Ok(Produced::End);
        }
        self.made += 1;
        Ok(Produced::Buffer(picture(self.made)))
    }
}

/// Polled, it hands on what its source makes — and has no thread of its
/// own: between polls, nothing moves.
#[test]
fn polled_it_hands_on_what_its_source_makes_and_nothing_between_polls() {
    let (counter, pictures) = FrameCounter::new("pictures");
    let (mut pipeline, ()) = DrivenPipeline::new("driven", camera(), |source, ctx| {
        ctx.attach(source, 0, ctx.branch().to(counter)?)?;
        Ok(())
    })
    .unwrap();
    assert!(matches!(
        poll_for(&mut pipeline, Duration::from_millis(400)),
        Driven::Pending { at: Some(_) }
    ));
    let made = pictures.get();
    assert!(made >= 4, "30 a second for 0.4 s, and {made} came");
    thread::sleep(Duration::from_millis(200));
    assert_eq!(pictures.get(), made, "nothing moves between polls");
}

/// What has a thread of its own is refused as it is wired, by name: a
/// queue, in a branch or a fan-out's, an element behind a queue of its
/// own, and a source that cannot be driven. A branch with none of them
/// still wires.
#[test]
fn its_wiring_refuses_what_runs_on_a_thread_of_its_own() {
    fn refused(error: Error) -> GraphError {
        match error {
            Error::GraphError(error) => error,
            other => panic!("refused with {other}"),
        }
    }
    let threaded = |name: &str| GraphError::ThreadInDrivenPipeline { name: name.into() };

    let (counter, _) = FrameCounter::new("pictures");
    let queued = DrivenPipeline::new("driven", camera(), |source, ctx| {
        ctx.attach(source, 0, ctx.branch().queue("frames", 4).to(counter)?)?;
        Ok(())
    });
    assert_eq!(refused(queued.err().unwrap()), threaded("frames"));

    let (counter, _) = FrameCounter::new("pictures");
    let paced = DrivenPipeline::new("driven", camera(), |source, ctx| {
        ctx.attach(
            source,
            0,
            ctx.branch().pipe(Pacer::new("pace")).to(counter)?,
        )?;
        Ok(())
    });
    assert_eq!(refused(paced.err().unwrap()), threaded("pace"));

    let (first, _) = FrameCounter::new("first");
    let (second, _) = FrameCounter::new("second");
    let fanned = DrivenPipeline::new("driven", camera(), |source, ctx| {
        let first = ctx.branch().to(first)?;
        let second = ctx.branch().queue("second-queue", 4).to(second)?;
        let tee = ctx.tee("tee").branch(first).branch(second).build()?;
        ctx.attach(source, 0, tee)?;
        Ok(())
    });
    assert_eq!(refused(fanned.err().unwrap()), threaded("second-queue"));

    let (app, _feed) = AppSource::new("app", 4);
    let (counter, _) = FrameCounter::new("pictures");
    let fed = DrivenPipeline::new("driven", app, |source, ctx| {
        ctx.attach(source, 0, ctx.branch().to(counter)?)?;
        Ok(())
    });
    assert_eq!(
        refused(fed.err().unwrap()),
        GraphError::SourceNotDrivable { name: "app".into() }
    );

    let (first, _) = FrameCounter::new("first");
    let (second, _) = FrameCounter::new("second");
    let wired = DrivenPipeline::new("driven", camera(), |source, ctx| {
        let first = ctx.branch().to(first)?;
        let second = ctx.branch().to(second)?;
        let tee = ctx.tee("tee").branch(first).branch(second).build()?;
        ctx.attach(source, 0, tee)?;
        Ok(())
    });
    assert!(wired.is_ok(), "a fan-out with no queue wires");
}

/// One poll hands on a share of what there is, and no more, so that a
/// source with more to give does not keep the thread's other pipelines
/// waiting — and says it wants the thread back at once.
#[test]
fn one_poll_hands_on_a_share_and_no_more() {
    let record = Record::default();
    let (mut pipeline, _handle) = recorded(
        Burst {
            made: 0,
            left: None,
        }
        .named("burst"),
        &record,
    );
    let waker = Waker::noop();
    let mut cx = TaskContext::from_waker(waker);
    let before = Instant::now();
    let Driven::Pending { at: Some(at) } = pipeline.poll(&mut cx) else {
        panic!("a source with more wants to be polled again");
    };
    assert!(at <= Instant::now() && at >= before, "at once");
    assert_eq!(record.data(), 32);
    assert_eq!(record.heard()[0], Heard::Segment, "its stream begins first");
}

/// A source at the end of what it has ends its stream, and the pipeline
/// finishes: every terminal has its end, and the bus says so.
#[test]
fn a_source_that_ends_finishes_the_pipeline() {
    let record = Record::default();
    let (mut pipeline, handle) = recorded(
        Burst {
            made: 0,
            left: Some(5),
        }
        .named("burst"),
        &record,
    );
    assert_eq!(
        poll_for(&mut pipeline, Duration::from_secs(5)),
        Driven::Finished
    );
    assert!(handle.is_finished());
    let heard = record.heard();
    assert_eq!(heard.len(), 7, "{heard:?}");
    assert_eq!(heard[0], Heard::Segment);
    assert_eq!(heard[6], Heard::Eos);
    drop(pipeline);
    let events: Vec<_> = std::iter::from_fn(|| handle.bus().try_recv()).collect();
    assert!(
        events
            .iter()
            .any(|event| matches!(event, BusEvent::Finished)),
        "{events:?}"
    );
}

/// Paused, nothing reaches a terminal until it is resumed — and the pause
/// costs the source's schedule nothing: it plays on where it was.
#[test]
fn pause_holds_it_and_resume_plays_on_where_it_was() {
    let record = Record::default();
    let (pipeline, handle) = recorded(camera(), &record);
    let runner = DrivenRunner::new(1).unwrap();
    runner.spawn(pipeline);
    wait_for("pictures", || record.data() >= 3);

    handle.pause();
    wait_for("the pause reaches the terminal", || {
        record.has(|heard| *heard == Heard::Paused)
    });
    thread::sleep(Duration::from_millis(300));
    let heard = record.heard();
    let paused_at = heard.iter().position(|h| *h == Heard::Paused).unwrap();
    assert_eq!(heard.len(), paused_at + 1, "nothing after the pause");
    let last = heard
        .iter()
        .rev()
        .find_map(|heard| match heard {
            Heard::Data(at, _) => *at,
            _ => None,
        })
        .unwrap();

    handle.resume();
    wait_for("pictures again", || {
        record.heard()[paused_at..]
            .iter()
            .any(|heard| matches!(heard, Heard::Data(..)))
    });
    let heard = record.heard();
    assert_eq!(heard[paused_at + 1], Heard::Resumed, "{heard:?}");
    let Heard::Data(Some(next), _) = heard[paused_at + 2] else {
        panic!("{heard:?}");
    };
    assert!(
        next - last < Duration::from_millis(150),
        "played on {:?} after the last picture before a 300 ms pause",
        next - last
    );
    handle.stop();
}

/// A pipeline paused before its first poll makes nothing until resumed.
#[test]
fn paused_before_its_first_poll_it_makes_nothing() {
    let record = Record::default();
    let (mut pipeline, handle) = recorded(camera(), &record);
    handle.pause();
    poll_for(&mut pipeline, Duration::from_millis(200));
    assert_eq!(record.heard(), [Heard::Segment, Heard::Paused]);
    handle.resume();
    poll_for(&mut pipeline, Duration::from_millis(300));
    assert!(record.data() > 0, "{:?}", record.heard());
}

/// Stopped, it finishes at its next poll: every element hears the stop,
/// nothing more is handed on, the runner lets go of it, and its bus ends.
#[test]
fn stop_ends_it_and_every_element_hears_it() {
    let record = Record::default();
    let (pipeline, handle) = recorded(camera(), &record);
    let runner = DrivenRunner::new(2).unwrap();
    runner.spawn(pipeline);
    wait_for("pictures", || record.data() >= 2);
    handle.stop();
    wait_for("finished", || handle.is_finished());
    wait_for("let go of", || runner.pipelines() == 0);
    let heard = record.heard();
    assert!(matches!(heard.last(), Some(Heard::Stopped(_))), "{heard:?}");
    let events: Vec<_> = handle.bus().iter().collect();
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, BusEvent::Error { .. })),
        "{events:?}"
    );
}

/// Finished, it ends its stream behind what it handed on, and the bus says
/// it has played everything.
#[test]
fn finish_ends_its_stream_in_order() {
    let record = Record::default();
    let (pipeline, handle) = recorded(camera(), &record);
    let runner = DrivenRunner::new(1).unwrap();
    runner.spawn(pipeline);
    wait_for("pictures", || record.data() >= 2);
    handle.pause();
    handle.finish();
    wait_for("finished", || handle.is_finished());
    assert_eq!(record.heard().last(), Some(&Heard::Eos));
    let events: Vec<_> = handle.bus().iter().collect();
    assert!(
        events
            .iter()
            .any(|event| matches!(event, BusEvent::Finished)),
        "{events:?}"
    );
}

/// Dropped, it stops on the thread that drops it — nothing to join.
#[test]
fn dropped_it_stops_on_the_dropping_thread() {
    let record = Record::default();
    let (mut pipeline, handle) = recorded(camera(), &record);
    poll_for(&mut pipeline, Duration::from_millis(100));
    drop(pipeline);
    assert_eq!(
        record.heard().last(),
        Some(&Heard::Stopped(thread::current().id()))
    );
    assert!(handle.is_finished());
    handle.stop();
}

/// A panic in a poll is reported on its bus as the pipeline's failure and
/// ends it, and the other pipelines on the thread go on.
#[test]
fn a_panic_is_reported_and_the_others_go_on() {
    let runner = DrivenRunner::new(1).unwrap();
    let failing = Record::default();
    let mut sink = recorder(&failing);
    sink.panics = true;
    let (pipeline, failed) = DrivenPipeline::new("failing", camera(), |source, ctx| {
        ctx.attach(source, 0, ctx.branch().to(sink)?)?;
        Ok(())
    })
    .map(|(pipeline, ())| {
        let handle = pipeline.handle();
        (pipeline, handle)
    })
    .unwrap();
    let record = Record::default();
    let (other, going) = recorded(camera(), &record);
    runner.spawn(pipeline);
    runner.spawn(other);
    wait_for("the failing one ends", || failed.is_finished());
    let events: Vec<_> = failed.bus().iter().collect();
    assert!(
        events.iter().any(|event| matches!(
            event,
            BusEvent::Error {
                error: Error::Panicked { .. },
                ..
            }
        )),
        "{events:?}"
    );
    let before = record.data();
    wait_for("the other goes on", || record.data() > before + 2);
    assert!(!going.is_finished());
}

/// What a source hands the poll's waker to wakes the pipeline, which is
/// then polled again with nothing else to wake it.
#[test]
fn what_a_source_hands_the_waker_wakes_it() {
    #[derive(Clone, Default)]
    struct Feed {
        pictures: Arc<Mutex<VecDeque<i64>>>,
        waker: Arc<Mutex<Option<Waker>>>,
    }
    struct Fed(Feed);
    impl Source for Fed {
        fn is_live(&self) -> bool {
            true
        }
        fn drivable(&self) -> bool {
            true
        }
        fn produce(&mut self, wait: &mut Wait<'_>) -> crate::error::Result<Produced> {
            if let Some(index) = self.0.pictures.lock().unwrap().pop_front() {
                return Ok(Produced::Buffer(picture(index)));
            }
            *self.0.waker.lock().unwrap() = wait.waker().cloned();
            Ok(Produced::Nothing)
        }
    }

    let feed = Feed::default();
    let record = Record::default();
    let (pipeline, handle) = recorded(Fed(feed.clone()).named("fed"), &record);
    let runner = DrivenRunner::new(1).unwrap();
    runner.spawn(pipeline);
    wait_for("waiting on its feed", || {
        feed.waker.lock().unwrap().is_some()
    });
    feed.pictures.lock().unwrap().extend([1, 2, 3]);
    if let Some(waker) = feed.waker.lock().unwrap().take() {
        waker.wake();
    }
    wait_for("what was fed", || record.data() == 3);
    handle.stop();
}

/// As a [`Future`], the time it asks for wakes it: a source that waits on
/// the clock is polled on time by an executor that knows nothing of it.
#[test]
fn as_a_future_the_time_it_asks_for_wakes_it() {
    let record = Record::default();
    let (mut pipeline, handle) = recorded(camera(), &record);
    let waker = Waker::from(Arc::new(Unpark(thread::current())));
    let mut cx = TaskContext::from_waker(&waker);
    let end = Instant::now() + Duration::from_millis(500);
    while Instant::now() < end {
        assert!(Pin::new(&mut pipeline).poll(&mut cx).is_pending());
        // Woken by the timer, never by this deadline.
        thread::park_timeout(Duration::from_secs(1));
    }
    let made = record.data();
    assert!(
        (5..=20).contains(&made),
        "30 a second for 0.5 s, and {made} came"
    );
    handle.stop();
    assert!(Pin::new(&mut pipeline).poll(&mut cx).is_ready());
}

/// The runner spreads its pipelines over its threads, and polls each on
/// the one it was given.
#[test]
fn the_runner_spreads_pipelines_over_its_threads() {
    let runner = DrivenRunner::new(2).unwrap();
    let mut records = Vec::new();
    let mut handles = Vec::new();
    for _ in 0..8 {
        let record = Record::default();
        let (pipeline, handle) = recorded(camera(), &record);
        runner.spawn(pipeline);
        records.push(record);
        handles.push(handle);
    }
    assert_eq!(runner.pipelines(), 8);
    for record in &records {
        wait_for("pictures", || record.data() >= 2);
    }
    let threads: std::collections::HashSet<_> = records
        .iter()
        .map(|record| {
            let threads: std::collections::HashSet<_> = record
                .heard()
                .iter()
                .filter_map(|heard| match heard {
                    Heard::Data(_, thread) => Some(*thread),
                    _ => None,
                })
                .collect();
            assert_eq!(threads.len(), 1, "one pipeline, one thread");
            threads.into_iter().next().unwrap()
        })
        .collect();
    assert_eq!(threads.len(), 2);
    drop(runner);
    for (record, handle) in records.iter().zip(&handles) {
        assert!(handle.is_finished());
        assert!(matches!(record.heard().last(), Some(Heard::Stopped(_))));
    }
}

/// A test tone, driven, makes sound — its tick is a time it keeps, so the
/// poll that asks again finds it come (it once worked out a new one from
/// each ask, and a driven poll never reached it).
#[test]
fn a_test_tone_sounds_driven() {
    let record = Record::default();
    let (mut pipeline, _handle) = recorded(
        crate::elements::TestAudioSource::new("tone", crate::elements::TestAudioOptions::default()),
        &record,
    );
    poll_for(&mut pipeline, Duration::from_millis(300));
    assert!(record.data() >= 3, "{:?}", record.heard());
}
