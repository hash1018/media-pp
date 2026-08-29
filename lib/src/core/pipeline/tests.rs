use std::{
    sync::{
        Mutex,
        atomic::{AtomicBool, AtomicUsize},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use super::*;
use ffmpeg_next::{self as ffmpeg, Rescale};

use crate::contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract};
use crate::elements::{
    FileDemuxer, Pacer, SwDecoder, TeeBuilder, TestAudioOptions, TestAudioSource, TestVideoOptions,
    TestVideoSource,
};
use crate::graph::GraphError;
use crate::test_support::try_test_video;
use crate::{
    control::{ControlReceiver, drain_control},
    element::{Source, SourceElement},
    pad::SrcPad,
};

#[test]
fn partial_thread_spawn_failure_stops_and_joins_started_sources() {
    let pipeline = PipelineBuilder::new("spawn-failure")
        .add_source(
            TestVideoSource::new("first", TestVideoOptions::default()),
            |_source, _ctx| Ok(()),
        )
        .unwrap()
        .add_source(
            TestVideoSource::new("second", TestVideoOptions::default()),
            |_source, _ctx| Ok(()),
        )
        .unwrap()
        .build();

    let mut spawn_count = 0;
    let error = pipeline
        .run_with_spawner(|thread_name, task| {
            spawn_count += 1;
            if spawn_count == 2 {
                Err(std::io::Error::other("injected spawn failure"))
            } else {
                thread::Builder::new().name(thread_name).spawn(task)
            }
        })
        .expect_err("the injected second spawn failure must be returned");

    assert!(matches!(error, crate::Error::ThreadSpawnError(_)));
    assert_eq!(pipeline.running.load(Ordering::Acquire), 0);
    assert!(pipeline.workers.lock().unwrap().is_empty());
}

/// End-to-end: `run()` (async — starts the background thread and
/// returns right away), then `pause()`/`stop()` (skipping `resume()`)
/// from the test's own thread — exercises the whole cascade (source's
/// `drain_control` loop -> `Queue`'s worker) at once, not just `Queue`
/// in isolation (see `queue::tests`). Mainly guards against the
/// deadlock this design is built to avoid: draining the bus
/// afterward must return promptly, not hang forever waiting on a
/// control message — or a `Bus` handle — that never arrives/drops.
#[test]
fn pause_then_stop_returns_promptly() {
    let Some(path) = try_test_video() else { return };
    let (source, streams) = FileDemuxer::open("demux", &path).expect("open test video");
    let video = streams
        .iter()
        .find(|s| s.kind == ffmpeg_next::media::Type::Video)
        .expect("test video has a video stream");
    let index = video.index;

    let pipeline = Pipeline::new("test", source, |source, ctx| {
        let branch = ctx.branch().queue("q", 4).to(Box::new(NoOpSink {
            name: "noop".into(),
            pp_log: element_pp_log(ElementType::Other, "noop", None),
        }))?;
        ctx.attach(source, index, branch)?;
        Ok(())
    })
    .expect("test pipeline wiring must succeed");

    pipeline.run().unwrap();

    // Give the background thread a moment to actually start looping
    // so `pause()`/`stop()` land while `running` is true, not before.
    thread::sleep(Duration::from_millis(50));
    pipeline.pause();
    pipeline.stop();

    // Blocks until every `Bus` handle in the pipeline has been
    // dropped — i.e. until the background thread has actually
    // finished, not just acked `stop()`.
    let events: Vec<_> = pipeline.bus().iter().collect();
    assert!(
        !events.iter().any(|e| matches!(e, BusEvent::Error { .. })),
        "unexpected error event(s): {events:?}"
    );
}

#[test]
fn seek_check_rejects_a_live_source_before_flushing() {
    let source = TestVideoSource::new("live", TestVideoOptions::default());
    let pipeline = Pipeline::new("seek-check", source, |source, ctx| {
        let branch = ctx.branch().to(Box::new(NoOpSink {
            name: "noop".into(),
            pp_log: element_pp_log(ElementType::Other, "noop", None),
        }))?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("pipeline wiring");

    pipeline.run().expect("run");
    let error = pipeline
        .seek(Duration::from_secs(1), SeekMode::Accurate)
        .expect_err("live source must reject seek");
    pipeline.stop();

    let crate::Error::SeekError(error) = error else {
        panic!("expected SeekError, got {error:?}");
    };
    assert_eq!(error.rejections().len(), 1);
    assert_eq!(
        error.rejections()[0].reason,
        crate::control::SeekRejectReason::LiveSource
    );
}

struct SeekLoopSource {
    pp_log: PpLog,
    pad: SrcPad,
    seeks: Arc<AtomicUsize>,
}

impl Element for SeekLoopSource {
    fn name(&self) -> Arc<str> {
        "seek-loop".into()
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

impl Source for SeekLoopSource {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl SourceElement for SeekLoopSource {
    fn is_live(&self) -> bool {
        false
    }

    fn is_seekable(&self) -> bool {
        true
    }

    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        loop {
            if drain_control(control, self, bus)?.stopped {
                return Ok(());
            }
            if self.pad.ready_consume() {
                self.pad
                    .push(MediaBuffer::Packet(Arc::new(ffmpeg_next::Packet::empty())))?;
            }
            thread::sleep(Duration::from_millis(1));
        }
    }

    fn seek(&mut self, target: Duration) -> Result<Duration> {
        self.seeks.fetch_add(1, Ordering::SeqCst);
        Ok(target)
    }
}

struct ControlRecordingSink {
    pp_log: PpLog,
    count: Arc<AtomicUsize>,
    controls: Arc<Mutex<Vec<&'static str>>>,
    preroll_targets: Arc<Mutex<Vec<Option<Duration>>>>,
}

impl Element for ControlRecordingSink {
    fn name(&self) -> Arc<str> {
        "control-recorder".into()
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

impl Sink for ControlRecordingSink {
    fn consume(&mut self, _buf: MediaBuffer) -> Result<()> {
        self.count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        let label = match msg {
            ControlMsg::Pause => "pause",
            ControlMsg::Resume => "resume",
            ControlMsg::Stop => "stop",
            ControlMsg::Flush => "flush",
            ControlMsg::CheckSeek(_) => "check-seek",
            ControlMsg::Preroll(context) => {
                self.preroll_targets.lock().unwrap().push(context.target());
                "preroll"
            }
            ControlMsg::Seek(_) => "seek",
        };
        self.controls.lock().unwrap().push(label);
        Ok(())
    }
}

#[test]
fn paused_seek_prerolls_one_timeline_and_restores_pause() {
    let seeks = Arc::new(AtomicUsize::new(0));
    let count = Arc::new(AtomicUsize::new(0));
    let controls = Arc::new(Mutex::new(Vec::new()));
    let preroll_targets = Arc::new(Mutex::new(Vec::new()));
    let source = SeekLoopSource {
        pp_log: element_pp_log(ElementType::Other, "seek-loop", None),
        pad: SrcPad::new("src"),
        seeks: Arc::clone(&seeks),
    };
    let pipeline = Pipeline::new("paused-seek-preroll", source, |source, ctx| {
        let branch = ctx.branch().to(Box::new(ControlRecordingSink {
            pp_log: element_pp_log(ElementType::Other, "control-recorder", None),
            count: Arc::clone(&count),
            controls: Arc::clone(&controls),
            preroll_targets: Arc::clone(&preroll_targets),
        }))?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("pipeline wiring");

    pipeline.run().expect("run");
    while count.load(Ordering::SeqCst) == 0 {
        thread::yield_now();
    }
    pipeline.pause();
    controls.lock().unwrap().clear();

    pipeline
        .seek(Duration::from_secs(2), SeekMode::Accurate)
        .expect("paused seek");
    assert_eq!(seeks.load(Ordering::SeqCst), 1);
    let after_preroll = count.load(Ordering::SeqCst);
    thread::sleep(Duration::from_millis(20));
    assert_eq!(count.load(Ordering::SeqCst), after_preroll);
    assert_eq!(
        controls.lock().unwrap().as_slice(),
        ["check-seek", "flush", "seek", "preroll", "pause"]
    );
    assert_eq!(
        preroll_targets.lock().unwrap().as_slice(),
        [Some(Duration::from_secs(2))]
    );

    controls.lock().unwrap().clear();
    preroll_targets.lock().unwrap().clear();
    pipeline
        .seek(Duration::from_secs(3), SeekMode::Keyframe)
        .expect("paused keyframe seek");
    let after_keyframe_preroll = count.load(Ordering::SeqCst);
    assert_eq!(after_keyframe_preroll, after_preroll + 1);
    thread::sleep(Duration::from_millis(20));
    assert_eq!(count.load(Ordering::SeqCst), after_keyframe_preroll);
    assert_eq!(
        controls.lock().unwrap().as_slice(),
        ["check-seek", "flush", "seek", "preroll", "pause"]
    );
    assert_eq!(preroll_targets.lock().unwrap().as_slice(), [None]);

    pipeline.resume();
    while count.load(Ordering::SeqCst) == after_keyframe_preroll {
        thread::yield_now();
    }
    pipeline.stop();
}

#[test]
fn playing_seek_uses_an_internal_pause_then_resumes() {
    let seeks = Arc::new(AtomicUsize::new(0));
    let count = Arc::new(AtomicUsize::new(0));
    let controls = Arc::new(Mutex::new(Vec::new()));
    let preroll_targets = Arc::new(Mutex::new(Vec::new()));
    let source = SeekLoopSource {
        pp_log: element_pp_log(ElementType::Other, "seek-loop", None),
        pad: SrcPad::new("src"),
        seeks: Arc::clone(&seeks),
    };
    let pipeline = Pipeline::new("playing-seek-preroll", source, |source, ctx| {
        let branch = ctx.branch().to(Box::new(ControlRecordingSink {
            pp_log: element_pp_log(ElementType::Other, "control-recorder", None),
            count: Arc::clone(&count),
            controls: Arc::clone(&controls),
            preroll_targets,
        }))?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("pipeline wiring");

    pipeline.run().expect("run");
    while count.load(Ordering::SeqCst) == 0 {
        thread::yield_now();
    }
    controls.lock().unwrap().clear();

    pipeline
        .seek(Duration::from_secs(2), SeekMode::Accurate)
        .expect("playing seek");
    let after_preroll = count.load(Ordering::SeqCst);
    while count.load(Ordering::SeqCst) == after_preroll {
        thread::yield_now();
    }
    assert_eq!(seeks.load(Ordering::SeqCst), 1);
    assert_eq!(
        controls.lock().unwrap()[..6],
        ["check-seek", "pause", "flush", "seek", "preroll", "resume"]
    );
    pipeline.stop();
}

/// Fails on its first `run`, the way a live capture whose source disappears
/// does.
struct FailingSource {
    pp_log: PpLog,
    pad: SrcPad,
}

impl Element for FailingSource {
    fn name(&self) -> Arc<str> {
        "failing".into()
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

impl Source for FailingSource {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl SourceElement for FailingSource {
    fn is_live(&self) -> bool {
        false
    }

    fn is_seekable(&self) -> bool {
        false
    }

    fn run(&mut self, _control: &ControlReceiver, _bus: &Bus) -> Result<()> {
        Err(crate::Error::Other("the source went away".into()))
    }

    fn seek(&mut self, target: Duration) -> Result<Duration> {
        Ok(target)
    }
}

/// Records whether it was ever told to stop — what a muxer needs before it can
/// finalize a track.
struct StopRecordingSink {
    pp_log: PpLog,
    stopped: Arc<AtomicBool>,
}

impl Element for StopRecordingSink {
    fn name(&self) -> Arc<str> {
        "stop-recorder".into()
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

impl Sink for StopRecordingSink {
    fn consume(&mut self, _buf: MediaBuffer) -> Result<()> {
        Ok(())
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        if msg == ControlMsg::Stop {
            self.stopped.store(true, Ordering::Release);
        }
        Ok(())
    }
}

struct BurstSource {
    pp_log: PpLog,
    pad: SrcPad,
    ready: Arc<AtomicBool>,
    buffers: usize,
}

impl Element for BurstSource {
    fn name(&self) -> Arc<str> {
        "burst".into()
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

impl Source for BurstSource {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl SourceElement for BurstSource {
    fn is_live(&self) -> bool {
        false
    }

    fn is_seekable(&self) -> bool {
        false
    }

    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        for _ in 0..self.buffers {
            self.pad
                .push(MediaBuffer::Packet(Arc::new(ffmpeg_next::Packet::empty())))?;
        }
        self.ready.store(true, Ordering::Release);
        loop {
            if drain_control(control, self, bus)?.stopped {
                return Ok(());
            }
            thread::yield_now();
        }
    }

    fn seek(&mut self, target: Duration) -> Result<Duration> {
        Ok(target)
    }
}

struct SlowEosSink {
    pp_log: PpLog,
    count: Arc<AtomicUsize>,
    saw_eos: Arc<AtomicBool>,
}

impl Element for SlowEosSink {
    fn name(&self) -> Arc<str> {
        "slow-eos".into()
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

impl Sink for SlowEosSink {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        if buf.is_eos() {
            self.saw_eos.store(true, Ordering::Release);
        } else {
            thread::sleep(Duration::from_millis(5));
            self.count.fetch_add(1, Ordering::AcqRel);
        }
        Ok(())
    }

    fn control(&mut self, _msg: ControlMsg) -> Result<()> {
        Ok(())
    }
}

/// `finish()` is deliberately not a second spelling of `stop()`: even from a
/// paused state it resumes the queue, places EOS behind the source's backlog,
/// and waits until every queued buffer and EOS have reached the terminal sink.
#[test]
fn finish_drains_queued_data_and_eos_even_while_paused() {
    const BUFFERS: usize = 24;
    let ready = Arc::new(AtomicBool::new(false));
    let count = Arc::new(AtomicUsize::new(0));
    let saw_eos = Arc::new(AtomicBool::new(false));
    let source = BurstSource {
        pp_log: element_pp_log(ElementType::Other, "burst", None),
        pad: SrcPad::new("burst_src"),
        ready: ready.clone(),
        buffers: BUFFERS,
    };
    let pipeline = Pipeline::new("finish-test", source, |source, ctx| {
        let branch = ctx
            .branch()
            .queue("backlog", BUFFERS)
            .to(Box::new(SlowEosSink {
                pp_log: element_pp_log(ElementType::Other, "slow-eos", None),
                count: count.clone(),
                saw_eos: saw_eos.clone(),
            }))?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("test pipeline wiring must succeed");

    pipeline.run().unwrap();
    while !ready.load(Ordering::Acquire) {
        thread::yield_now();
    }
    pipeline.pause();
    pipeline.finish();

    assert_eq!(count.load(Ordering::Acquire), BUFFERS);
    assert!(
        saw_eos.load(Ordering::Acquire),
        "terminal sink never received EOS"
    );
    let errors: Vec<_> = pipeline
        .bus()
        .iter()
        .filter(|event| matches!(event, BusEvent::Error { .. }))
        .collect();
    assert!(errors.is_empty(), "unexpected finish errors: {errors:?}");
}

/// [`PipelineBuilder`] with two independent, indefinitely-running
/// sources (standing in for a real video capture + audio capture pair
/// feeding one [`crate::elements::Mp4Muxer`]) sharing one `Pipeline`:
/// both should show up in `topology()` under their *own* root, not
/// both defaulted to whichever source was added first (the exact bug
/// `Tee`'s own registration had before `default_upstream` existed —
/// see [`Context::default_upstream`]'s docs), and a single `stop()`
/// call must reach both — if it only reached one, the other source's
/// thread would still be alive holding its own `Bus` sender clone
/// open, and `pipeline.bus().iter().collect()` below would hang
/// forever instead of returning.
#[test]
fn multi_source_pipeline_stops_every_source_from_one_stop_call() {
    let video = TestVideoSource::new("video", TestVideoOptions::default());
    let audio = TestAudioSource::new("audio", TestAudioOptions::default());

    let video_count = Arc::new(AtomicUsize::new(0));
    let audio_count = Arc::new(AtomicUsize::new(0));

    let pipeline = PipelineBuilder::new("multi-source-test")
        .add_source(video, {
            let count = video_count.clone();
            move |source, ctx| {
                let branch = ctx.branch().to(Box::new(CountingSink {
                    name: "video-sink".into(),
                    count,
                    pp_log: element_pp_log(ElementType::Other, "video-sink", None),
                }))?;
                ctx.attach(source, 0, branch)?;
                Ok(())
            }
        })
        .expect("video wiring must succeed")
        .add_source(audio, {
            let count = audio_count.clone();
            move |source, ctx| {
                let branch = ctx.branch().to(Box::new(CountingSink {
                    name: "audio-sink".into(),
                    count,
                    pp_log: element_pp_log(ElementType::Other, "audio-sink", None),
                }))?;
                ctx.attach(source, 0, branch)?;
                Ok(())
            }
        })
        .expect("audio wiring must succeed")
        .build();

    let topology = pipeline.topology();
    let mut branches: Vec<&str> = topology.split('\n').collect();
    branches.sort_unstable();
    assert_eq!(
        branches,
        vec![
            "TestAudioSource(audio) - Other(audio-sink)",
            "TestVideoSource(video) - Other(video-sink)",
        ]
    );

    pipeline.run().unwrap();
    thread::sleep(Duration::from_millis(100));
    pipeline.stop();

    // Would hang here if `stop()` only reached one of the two sources
    // — see this test's own docs.
    let events: Vec<_> = pipeline.bus().iter().collect();
    assert!(
        !events.iter().any(|e| matches!(e, BusEvent::Error { .. })),
        "unexpected error event(s): {events:?}"
    );
    assert!(
        video_count.load(Ordering::SeqCst) > 0,
        "video branch never received anything"
    );
    assert!(
        audio_count.load(Ordering::SeqCst) > 0,
        "audio branch never received anything"
    );
}

/// The shared Pacer clock freezes before Pause begins its synchronous
/// downstream cascade. A busy sink must not turn time spent waiting for
/// its Pause acknowledgement into playable media time.
#[test]
fn pipeline_clock_includes_a_slow_pause_cascade_in_its_frozen_time() {
    let pause_delay = Duration::from_millis(80);
    let source = TestVideoSource::new("video", TestVideoOptions::default());
    let pipeline = Pipeline::new("slow-pause-clock-test", source, |source, ctx| {
        let branch = ctx.branch().to(Box::new(SlowPauseSink {
            pause_delay,
            pp_log: element_pp_log(ElementType::Other, "slow-pause", None),
        }))?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("test pipeline wiring must succeed");
    let original_start = pipeline.clock().start();

    pipeline.run().unwrap();
    thread::sleep(Duration::from_millis(50));
    pipeline.pause();
    pipeline.resume();

    let shifted_start = pipeline.clock().start();
    pipeline.stop();
    pipeline.bus().log_events();

    assert!(
        shifted_start.saturating_duration_since(original_start) >= Duration::from_millis(60),
        "the {:?} Pause cascade was omitted from the shared Clock's frozen interval",
        pause_delay
    );
}

/// A live source must not retain its owning Pipeline. Dropping the last
/// external Arc implicitly stops and joins the source, then releases the
/// Pipeline itself instead of leaving both alive forever.
#[test]
fn dropping_a_running_pipeline_stops_and_releases_it() {
    let source = TestVideoSource::new("video", TestVideoOptions::default());
    let pipeline = Pipeline::new("drop-running-test", source, |_source, _ctx| Ok(()))
        .expect("test pipeline wiring must succeed");
    let weak = Arc::downgrade(&pipeline);

    pipeline.run().unwrap();
    thread::sleep(Duration::from_millis(50));

    let (dropped_tx, dropped_rx) = mpsc::sync_channel(0);
    thread::spawn(move || {
        drop(pipeline);
        let _ = dropped_tx.send(());
    });

    dropped_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("dropping a running Pipeline must stop and join its source promptly");
    assert!(
        weak.upgrade().is_none(),
        "a source worker must not retain the Pipeline after external handles are dropped"
    );
}

/// The same leak this crate already guards against for a *single*
/// source (`Tee`'s own `retained_handle_does_not_keep_tee_context_or_bus_alive`
/// test, which builds a bespoke `Context` by hand) but through the
/// real, integrated [`PipelineBuilder`] path with a *second*,
/// unrelated source also present: a `Tee` wired under one of two
/// sources, its `TeeHandle` retained well past the point the whole
/// `Pipeline` finishes. Draining `pipeline.bus()` to completion is
/// itself the proof — it doesn't return until every `Bus` sender,
/// including whatever clone the `Tee`'s own retained `Context` held,
/// has actually dropped; `tee_handle` only ever held a `Weak`
/// reference; so it couldn't have kept anything alive regardless. The
/// `branch()`/`sink_count()` checks afterward confirm the
/// underlying shared state is really gone, not just that the bus
/// happened to close for some unrelated reason.
#[test]
fn tee_handle_retained_across_a_multi_source_pipeline_does_not_leak() {
    let video = TestVideoSource::new("video", TestVideoOptions::default());
    let audio = TestAudioSource::new("audio", TestAudioOptions::default());

    let mut tee_handle_slot = None;
    let pipeline = PipelineBuilder::new("multi-source-tee-test")
        .add_source(video, |source, ctx| {
            let branch = ctx.branch().to(Box::new(NoOpSink {
                name: "video-sink".into(),
                pp_log: element_pp_log(ElementType::Other, "video-sink", None),
            }))?;
            let (tee_branch, handle) = TeeBuilder::new("tee", ctx.clone())
                .branch(branch)
                .build_dynamic()?;
            ctx.attach(source, 0, tee_branch)?;
            tee_handle_slot = Some(handle);
            Ok(())
        })
        .expect("video wiring must succeed")
        .add_source(audio, |source, ctx| {
            let branch = ctx.branch().to(Box::new(NoOpSink {
                name: "audio-sink".into(),
                pp_log: element_pp_log(ElementType::Other, "audio-sink", None),
            }))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("audio wiring must succeed")
        .build();
    let tee_handle = tee_handle_slot.expect("wire ran");

    pipeline.run().unwrap();
    thread::sleep(Duration::from_millis(100));
    pipeline.stop();

    let events: Vec<_> = pipeline.bus().iter().collect();
    assert!(
        !events.iter().any(|e| matches!(e, BusEvent::Error { .. })),
        "unexpected error event(s): {events:?}"
    );

    drop(pipeline);
    assert!(
        tee_handle.branch().is_none(),
        "Tee's shared state should be gone once its owning Pipeline is fully torn down"
    );
    assert_eq!(tee_handle.sink_count(), 0);
}

/// `seek()` mid-playback should reposition the source (no error from
/// `Input::seek`), reset/flush everything downstream without
/// deadlocking, and let packets keep flowing afterward.
#[test]
fn seek_repositions_and_playback_continues() {
    let Some(path) = try_test_video() else { return };
    let (source, streams) = FileDemuxer::open("demux", &path).expect("open test video");
    let video = streams
        .iter()
        .find(|s| s.kind == ffmpeg_next::media::Type::Video)
        .expect("test video has a video stream");
    let index = video.index;
    let time_base = source.stream_time_base(index).expect("stream disappeared");

    let count = Arc::new(AtomicUsize::new(0));
    let sink = CountingSink {
        name: "counting-sink".into(),
        count: count.clone(),
        pp_log: element_pp_log(ElementType::Other, "counting-sink", None),
    };

    // A `Pacer` here isn't incidental: without it, this whole 10s/
    // 300-packet file races through in well under the 50ms sleep
    // below (no decode, no throttling), so `seek()` would land on an
    // already-finished pipeline and silently no-op — exactly the kind
    // of thing a weak `count > 0` assertion wouldn't have caught (see
    // `seek_reports_where_it_actually_landed_when_target_is_not_a_keyframe`
    // for how this was found).
    let pipeline = Pipeline::new("test", source, |source, ctx| {
        let pacer = Pacer::new("pacer", time_base, ctx.clock.clone())?;
        let branch = ctx.branch().queue("q", 4).pipe(pacer).to(Box::new(sink))?;
        ctx.attach(source, index, branch)?;
        Ok(())
    })
    .expect("test pipeline wiring must succeed");

    pipeline.run().unwrap();
    thread::sleep(Duration::from_millis(50));
    pipeline
        .seek(Duration::from_secs(1), SeekMode::Accurate)
        .expect("seek");
    // Let packets flow again post-seek before tearing down.
    thread::sleep(Duration::from_millis(100));
    pipeline.stop();

    let events: Vec<_> = pipeline.bus().iter().collect();
    assert!(
        !events.iter().any(|e| matches!(e, BusEvent::Error { .. })),
        "unexpected error event(s): {events:?}"
    );
    assert!(
        count.load(Ordering::SeqCst) > 0,
        "expected at least one packet to arrive after the seek"
    );
    assert!(
        events.iter().any(|e| matches!(
            e,
            BusEvent::Seeked { requested, .. } if *requested == Duration::from_secs(1)
        )),
        "expected a Seeked event reporting the request; got {events:?}"
    );
}

/// Regression test for the bug found manually testing `rtsp_serve_seek`:
/// a container seek can only land on a keyframe at or before `target`
/// (see `FileDemuxer::seek`'s docs), so a `target` inside a GOP lands
/// back at that GOP's keyframe — potentially nowhere near what was
/// requested. Without `BusEvent::Seeked` reporting that gap, this looked
/// indistinguishable from `seek` silently doing nothing.
///
/// The assertions below hold for any fixture: how far back the seek
/// actually lands depends on the file's keyframe spacing, but it must
/// never land *past* the request, and the gap must be reported.
#[test]
fn seek_reports_where_it_actually_landed_when_target_is_not_a_keyframe() {
    let Some(path) = try_test_video() else { return };
    let (source, streams) = FileDemuxer::open("demux", &path).expect("open test video");
    let video = streams
        .iter()
        .find(|s| s.kind == ffmpeg_next::media::Type::Video)
        .expect("test video has a video stream");
    let index = video.index;
    let time_base = source.stream_time_base(index).expect("stream disappeared");

    // Paced for the same reason as `seek_repositions_and_playback_continues`
    // — otherwise the file finishes before `seek()` is even called.
    let pipeline = Pipeline::new("test", source, |source, ctx| {
        let pacer = Pacer::new("pacer", time_base, ctx.clock.clone())?;
        let branch = ctx
            .branch()
            .queue("q", 4)
            .pipe(pacer)
            .to(Box::new(NoOpSink {
                name: "noop".into(),
                pp_log: element_pp_log(ElementType::Other, "noop", None),
            }))?;
        ctx.attach(source, index, branch)?;
        Ok(())
    })
    .expect("test pipeline wiring must succeed");

    pipeline.run().unwrap();
    thread::sleep(Duration::from_millis(50));
    pipeline
        .seek(Duration::from_secs(3), SeekMode::Accurate)
        .expect("seek");
    thread::sleep(Duration::from_millis(100));
    pipeline.stop();

    let events: Vec<_> = pipeline.bus().iter().collect();
    let seeked = events
        .iter()
        .find_map(|e| match e {
            BusEvent::Seeked {
                requested, landed, ..
            } => Some((*requested, *landed)),
            _ => None,
        })
        .expect("expected a Seeked event");
    assert_eq!(seeked.0, Duration::from_secs(3));
    assert!(
        seeked.1 <= seeked.0,
        "a container seek must land at or before the request, got {:?} for {:?}",
        seeked.1,
        seeked.0
    );
}

struct NoOpSink {
    name: Arc<str>,
    pp_log: PpLog,
}
impl Element for NoOpSink {
    fn name(&self) -> Arc<str> {
        self.name.clone()
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
impl Sink for NoOpSink {
    fn consume(&mut self, _buf: MediaBuffer) -> Result<()> {
        Ok(())
    }
    fn control(&mut self, _msg: ControlMsg) -> Result<()> {
        Ok(())
    }
}

struct SlowPauseSink {
    pp_log: PpLog,
    pause_delay: Duration,
}

impl Element for SlowPauseSink {
    fn name(&self) -> Arc<str> {
        "slow-pause".into()
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

impl Sink for SlowPauseSink {
    fn consume(&mut self, _buf: MediaBuffer) -> Result<()> {
        Ok(())
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        if msg == ControlMsg::Pause {
            thread::sleep(self.pause_delay);
        }
        Ok(())
    }
}

struct CountingSink {
    pp_log: PpLog,
    name: Arc<str>,
    count: Arc<AtomicUsize>,
}
impl Element for CountingSink {
    fn name(&self) -> Arc<str> {
        self.name.clone()
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
impl Sink for CountingSink {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        // Anything but `Eos` counts — covers `FileDemuxer`'s `Packet`s
        // (what every other test using this sink actually sends) and
        // `TestVideoSource`/`TestAudioSource`'s `Video`/`Audio` frames
        // (what `multi_source_pipeline_stops_every_source_from_one_stop_call`
        // sends) alike.
        if !buf.is_eos() {
            self.count.fetch_add(1, Ordering::SeqCst);
        }
        Ok(())
    }
    fn control(&mut self, _msg: ControlMsg) -> Result<()> {
        Ok(())
    }
}

/// `ChainBuilder::build`'s terminal — and, transitively via `.pipe()`/
/// `.queue()`, everything upstream of it — should come out tagged with
/// the pipeline id it was built with, not left as `None`.
#[test]
fn chain_builder_stamps_pipeline_id_into_terminal_pp_log() {
    let (bus, _bus_rx) = Bus::new();
    let sink = NoOpSink {
        name: "noop".into(),
        pp_log: element_pp_log(ElementType::Other, "noop", None),
    };
    let graph = PipelineGraph::new();
    let source_id = graph.add_source(ElementType::Other, "source".into());
    let context = Arc::new(Context::for_test(bus, "my-pipeline", graph, source_id));
    let built = context.branch().to(Box::new(sink)).unwrap();
    assert_eq!(built.root.pp_log().pipeline_id(), Some("my-pipeline"));
    assert_eq!(built.root.pp_log().element(), "Other");
    assert_eq!(built.root.pp_log().name(), "noop");
}

/// `Pipeline::new`'s `id` should come back unchanged from
/// [`Pipeline::id`], and the `wire` closure's own `ctx.pipeline_id`
/// (used to build a matching `ChainBuilder`) should be that same
/// value — not, say, whatever `source.name()` happens to be.
#[test]
fn pipeline_id_is_whatever_new_was_given() {
    let Some(path) = try_test_video() else { return };
    let (source, streams) = FileDemuxer::open("demux", &path).expect("open test video");
    let video = streams
        .iter()
        .find(|s| s.kind == ffmpeg_next::media::Type::Video)
        .expect("test video has a video stream");
    let index = video.index;

    let pipeline = Pipeline::new("my-pipeline", source, |source, ctx| {
        let branch = ctx.branch().to(Box::new(NoOpSink {
            name: "noop".into(),
            pp_log: element_pp_log(ElementType::Other, "noop", None),
        }))?;
        ctx.attach(source, index, branch)?;
        Ok(())
    })
    .expect("test pipeline wiring must succeed");

    assert_eq!(pipeline.id(), "my-pipeline");
}

/// [`Pipeline::topology`] should render the source plus every element
/// added via `.queue()`/`.pipe()` and the terminal, in order, joined by
/// `" - "` — and one line per branch when more than one src pad is
/// linked.
#[test]
fn topology_lists_source_through_terminal_per_branch() {
    let Some(path) = try_test_video() else { return };
    let (source, streams) = FileDemuxer::open("demux", &path).expect("open test video");
    let video = streams
        .iter()
        .find(|s| s.kind == ffmpeg_next::media::Type::Video)
        .expect("test video has a video stream");
    let index = video.index;
    let time_base = source.stream_time_base(index).expect("stream disappeared");

    let pipeline = Pipeline::new("test", source, |source, ctx| {
        let pacer = Pacer::new("pacer", time_base, ctx.clock.clone())?;
        let branch = ctx
            .branch()
            .queue("q", 4)
            .pipe(pacer)
            .to(Box::new(NoOpSink {
                name: "noop".into(),
                pp_log: element_pp_log(ElementType::Other, "noop", None),
            }))?;
        ctx.attach(source, index, branch)?;
        Ok(())
    })
    .expect("test pipeline wiring must succeed");

    assert_eq!(
        pipeline.topology(),
        "FileDemuxer(demux) - Queue(q) - Pacer(pacer) - Other(noop)"
    );
    assert_eq!(
        pipeline.graph().topology_diagram(),
        concat!(
            "FileDemuxer(demux)#1\n",
            "└── [src_0] → Queue(q)#2\n",
            "              └── [q_src] → Pacer(pacer)#3\n",
            "                            └── [pacer_src] → Other(noop)#4",
        )
    );

    // `pipeline` is dropped here without ever being `run()`, taking
    // its `.queue()`-spawned worker thread down with it — regression
    // coverage for the `Queue::drop` fix (see
    // `queue::tests::dropping_without_stop_or_eos_does_not_hang`):
    // this used to hang the test process forever.
}

/// Initial branches handed to [`TeeBuilder`] should render as starting
/// under `Tee(...)`, not the pipeline's source. The whole initial
/// fan-out is committed as one subgraph.
#[test]
fn topology_attributes_tee_branches_to_the_tee_not_the_source() {
    let Some(path) = try_test_video() else { return };
    let (source, streams) = FileDemuxer::open("demux", &path).expect("open test video");
    let video = streams
        .iter()
        .find(|s| s.kind == ffmpeg_next::media::Type::Video)
        .expect("test video has a video stream");
    let index = video.index;

    let pipeline = Pipeline::new("test", source, |source, ctx| {
        let branch_a = ctx.branch().to(Box::new(NoOpSink {
            name: "sink-a".into(),
            pp_log: element_pp_log(ElementType::Other, "sink-a", None),
        }))?;
        let branch_b = ctx.branch().to(Box::new(NoOpSink {
            name: "sink-b".into(),
            pp_log: element_pp_log(ElementType::Other, "sink-b", None),
        }))?;

        let tee_branch = TeeBuilder::new("tee", ctx.clone())
            .branch(branch_a)
            .branch(branch_b)
            .build()?;
        ctx.attach(source, index, tee_branch)?;
        Ok(())
    })
    .expect("test pipeline wiring must succeed");

    let topology = pipeline.topology();
    let graph = pipeline.graph();
    assert_eq!(
        graph.revision, 2,
        "source registration plus one subgraph commit"
    );
    assert_eq!(graph.nodes.len(), 4);
    assert_eq!(graph.edges.len(), 3);
    let initial_branch_id = graph.edges[0].branch_id;
    assert!(
        graph
            .edges
            .iter()
            .all(|edge| edge.branch_id == initial_branch_id),
        "the Tee and both initial branches must commit as one subgraph"
    );
    let mut branches: Vec<&str> = topology.split('\n').collect();
    branches.sort_unstable();
    assert_eq!(
        branches,
        vec![
            "FileDemuxer(demux) - Tee(tee) - Other(sink-a)",
            "FileDemuxer(demux) - Tee(tee) - Other(sink-b)",
        ]
    );
    assert_eq!(
        graph.topology_diagram(),
        concat!(
            "FileDemuxer(demux)#1\n",
            "└── [src_0] → Tee(tee)#4\n",
            "              ├── [tee_src0] → Other(sink-a)#2\n",
            "              └── [tee_src1] → Other(sink-b)#3",
        )
    );
}

/// A fan-out that a chain reaches through [`ChainBuilder::to_branch`] should
/// render under the stage that feeds it, not under the pipeline's source.
///
/// The two-step alternative — attach the `Tee` to a mid-chain element's pad,
/// then attach that element — links the buffers identically but records the
/// edge as the source's, because the element it was really handed is not in
/// the graph yet when [`Context::attach`] runs. That put the fan-out on the
/// wrong element in every diagram and in every bus attribution derived from
/// the graph.
#[test]
fn topology_attributes_a_fan_out_to_the_stage_that_feeds_it() {
    let Some(path) = try_test_video() else { return };
    let (source, streams) = FileDemuxer::open("demux", &path).expect("open test video");
    let video = streams
        .iter()
        .find(|s| s.kind == ffmpeg_next::media::Type::Video)
        .expect("test video has a video stream");
    let index = video.index;
    let time_base = source.stream_time_base(index).expect("stream disappeared");

    let pipeline = Pipeline::new("test", source, |source, ctx| {
        let branch_a = ctx.branch().to(Box::new(NoOpSink {
            name: "sink-a".into(),
            pp_log: element_pp_log(ElementType::Other, "sink-a", None),
        }))?;
        let branch_b = ctx.branch().to(Box::new(NoOpSink {
            name: "sink-b".into(),
            pp_log: element_pp_log(ElementType::Other, "sink-b", None),
        }))?;
        let tee_branch = TeeBuilder::new("tee", ctx.clone())
            .branch(branch_a)
            .branch(branch_b)
            .build()?;

        let pacer = Pacer::new("pacer", time_base, ctx.clock.clone())?;
        let branch = ctx.branch().pipe(pacer).to_branch(tee_branch)?;
        ctx.attach(source, index, branch)?;
        Ok(())
    })
    .expect("test pipeline wiring must succeed");

    let graph = pipeline.graph();
    assert_eq!(
        graph.revision, 2,
        "the stage in front and the whole fan-out commit as one subgraph"
    );
    let branch_id = graph.edges[0].branch_id;
    assert!(
        graph.edges.iter().all(|edge| edge.branch_id == branch_id),
        "one attach must commit every edge, the joining one included"
    );
    assert_eq!(
        graph.topology_diagram(),
        concat!(
            "FileDemuxer(demux)#1\n",
            "└── [src_0] → Pacer(pacer)#5\n",
            "              └── [pacer_src] → Tee(tee)#4\n",
            "                                ├── [tee_src0] → Other(sink-a)#2\n",
            "                                └── [tee_src1] → Other(sink-b)#3",
        )
    );

    let topology = pipeline.topology();
    let mut branches: Vec<&str> = topology.split('\n').collect();
    branches.sort_unstable();
    assert_eq!(
        branches,
        vec![
            "FileDemuxer(demux) - Pacer(pacer) - Tee(tee) - Other(sink-a)",
            "FileDemuxer(demux) - Pacer(pacer) - Tee(tee) - Other(sink-b)",
        ]
    );
}

/// Once a branch is pulled off a [`Tee`] via [`TeeHandle::detach`],
/// it should stop showing up in [`Pipeline::topology`] entirely — not
/// keep rendering as still attached under `Tee(...)`, which is what a
/// stale graph node would otherwise do.
#[test]
fn topology_forgets_a_branch_once_it_is_removed_from_the_tee() {
    let Some(path) = try_test_video() else { return };
    let (source, streams) = FileDemuxer::open("demux", &path).expect("open test video");
    let video = streams
        .iter()
        .find(|s| s.kind == ffmpeg_next::media::Type::Video)
        .expect("test video has a video stream");
    let index = video.index;

    let mut tee_handle_slot = None;
    let pipeline = Pipeline::new("test", source, |source, ctx| {
        let (tee_branch, tee_handle) = TeeBuilder::new("tee", ctx.clone()).build_dynamic()?;
        ctx.attach(source, index, tee_branch)?;
        tee_handle_slot = Some(tee_handle);
        Ok(())
    })
    .expect("test pipeline wiring must succeed");

    let tee_handle = tee_handle_slot.expect("wire ran");
    let branch_a = tee_handle
        .branch()
        .expect("tee is alive")
        .to(Box::new(NoOpSink {
            name: "sink-a".into(),
            pp_log: element_pp_log(ElementType::Other, "sink-a", None),
        }))
        .unwrap();
    let branch_b = tee_handle
        .branch()
        .expect("tee is alive")
        .to(Box::new(NoOpSink {
            name: "sink-b".into(),
            pp_log: element_pp_log(ElementType::Other, "sink-b", None),
        }))
        .unwrap();
    let branch_a_id = tee_handle.attach(branch_a).unwrap();
    tee_handle.attach(branch_b).unwrap();
    tee_handle.detach(branch_a_id).unwrap();

    assert_eq!(
        pipeline.topology(),
        "FileDemuxer(demux) - Tee(tee) - Other(sink-b)"
    );
}

/// A failure past a `.queue(...)` inside a branch is reported under
/// that deeper element's own name (a `Queue`/whatever it wraps can
/// only ever speak for itself), never the `Queue`'s own name that's
/// what's actually attached to the `Tee`. The branch root is the
/// *outermost* wrapper; `detach_branch_containing` resolves the stable
/// element ID back to the owning branch regardless of depth.
#[test]
fn remove_branch_containing_resolves_through_a_queue_to_the_tee_attached_root() {
    let Some(path) = try_test_video() else { return };
    let (source, streams) = FileDemuxer::open("demux", &path).expect("open test video");
    let video = streams
        .iter()
        .find(|s| s.kind == ffmpeg_next::media::Type::Video)
        .expect("test video has a video stream");
    let index = video.index;

    let mut tee_handle_slot = None;
    let pipeline = Pipeline::new("test", source, |source, ctx| {
        let (tee_branch, tee_handle) = TeeBuilder::new("tee", ctx.clone()).build_dynamic()?;
        ctx.attach(source, index, tee_branch)?;
        tee_handle_slot = Some(tee_handle);
        Ok(())
    })
    .expect("test pipeline wiring must succeed");

    let tee_handle = tee_handle_slot.expect("wire ran");
    let branch_a = tee_handle
        .branch()
        .expect("tee is alive")
        .queue("q-a", 4)
        .to(Box::new(NoOpSink {
            name: "sink-a".into(),
            pp_log: element_pp_log(ElementType::Other, "sink-a", None),
        }))
        .unwrap();
    let branch_b = tee_handle
        .branch()
        .expect("tee is alive")
        .to(Box::new(NoOpSink {
            name: "sink-b".into(),
            pp_log: element_pp_log(ElementType::Other, "sink-b", None),
        }))
        .unwrap();
    tee_handle.attach(branch_a).unwrap();
    tee_handle.attach(branch_b).unwrap();
    // The queue, not "sink-a", is the branch root. Resolving the
    // deeply nested terminal ID still finds the correct branch.
    let sink_a_id = pipeline
        .graph()
        .nodes
        .iter()
        .find(|node| &*node.name == "sink-a")
        .expect("sink-a is attached")
        .id;
    tee_handle.detach_branch_containing(sink_a_id).unwrap();

    assert_eq!(
        pipeline.topology(),
        "FileDemuxer(demux) - Tee(tee) - Other(sink-b)"
    );
}

/// Scale check beyond the 2-branch tests above: dozens of branches on
/// one `Tee`, all present in `topology()`, then half removed — proves
/// graph attachment and recursive branch removal do not depend on
/// branch count or removal order in a way the small tests miss.
#[test]
fn topology_stays_correct_with_dozens_of_branches_added_and_then_removed() {
    let Some(path) = try_test_video() else { return };
    let (source, streams) = FileDemuxer::open("demux", &path).expect("open test video");
    let video = streams
        .iter()
        .find(|s| s.kind == ffmpeg_next::media::Type::Video)
        .expect("test video has a video stream");
    let index = video.index;

    const N: usize = 30;
    let mut tee_handle_slot = None;
    let pipeline = Pipeline::new("test", source, |source, ctx| {
        let (tee_branch, tee_handle) = TeeBuilder::new("tee", ctx.clone()).build_dynamic()?;
        ctx.attach(source, index, tee_branch)?;
        tee_handle_slot = Some(tee_handle);
        Ok(())
    })
    .expect("test pipeline wiring must succeed");

    let tee_handle = tee_handle_slot.expect("wire ran");
    let mut branch_ids = Vec::new();
    for i in 0..N {
        let name: Arc<str> = format!("sink-{i}").into();
        let branch = tee_handle
            .branch()
            .expect("tee is alive")
            .to(Box::new(NoOpSink {
                name: name.clone(),
                pp_log: element_pp_log(ElementType::Other, &name, None),
            }))
            .unwrap();
        branch_ids.push(tee_handle.attach(branch).unwrap());
    }

    let mut branches: Vec<String> = pipeline.topology().lines().map(String::from).collect();
    branches.sort();
    let mut expected: Vec<String> = (0..N)
        .map(|i| format!("FileDemuxer(demux) - Tee(tee) - Other(sink-{i})"))
        .collect();
    expected.sort();
    assert_eq!(branches, expected, "all {N} branches should show up once");

    for branch_id in branch_ids.into_iter().take(N / 2) {
        tee_handle.detach(branch_id).unwrap();
    }

    let mut remaining: Vec<String> = pipeline.topology().lines().map(String::from).collect();
    remaining.sort();
    let mut expected_remaining: Vec<String> = (N / 2..N)
        .map(|i| format!("FileDemuxer(demux) - Tee(tee) - Other(sink-{i})"))
        .collect();
    expected_remaining.sort();
    assert_eq!(
        remaining, expected_remaining,
        "only the un-removed half should remain, none of the removed ones lingering"
    );
}

#[test]
fn detached_branch_never_appears_in_topology() {
    let Some(path) = try_test_video() else { return };
    let (source, _) = FileDemuxer::open("demux", &path).expect("open test video");

    let pipeline = Pipeline::new("test", source, |_source, ctx| {
        let detached = ctx.branch().to(Box::new(NoOpSink {
            name: "never-attached".into(),
            pp_log: element_pp_log(ElementType::Other, "never-attached", None),
        }))?;
        assert_eq!(ctx.graph.snapshot().nodes.len(), 1);
        drop(detached);
        Ok(())
    })
    .expect("test pipeline wiring must succeed");

    assert_eq!(pipeline.topology(), "FileDemuxer(demux)");
}

#[test]
fn duplicate_names_are_independent_when_detaching_by_branch_id() {
    let Some(path) = try_test_video() else { return };
    let (source, streams) = FileDemuxer::open("demux", &path).expect("open test video");
    let index = streams
        .iter()
        .find(|stream| stream.kind == ffmpeg_next::media::Type::Video)
        .expect("test video has a video stream")
        .index;
    let mut handle_slot = None;
    let pipeline = Pipeline::new("test", source, |source, ctx| {
        let (tee_branch, handle) = TeeBuilder::new("tee", ctx.clone()).build_dynamic()?;
        ctx.attach(source, index, tee_branch)?;
        handle_slot = Some(handle);
        Ok(())
    })
    .expect("test pipeline wiring must succeed");
    let handle = handle_slot.expect("wire ran");

    let make_branch = || {
        handle
            .branch()
            .expect("tee is alive")
            .to(Box::new(NoOpSink {
                name: "same-name".into(),
                pp_log: element_pp_log(ElementType::Other, "same-name", None),
            }))
            .unwrap()
    };
    let first = handle.attach(make_branch()).unwrap();
    let second = handle.attach(make_branch()).unwrap();
    assert_ne!(first, second);
    assert_eq!(pipeline.topology().lines().count(), 2);

    handle.detach(first).unwrap();
    assert_eq!(pipeline.topology().lines().count(), 1);
    assert!(pipeline.topology().contains("Other(same-name)"));
}

#[test]
fn dynamic_attach_and_detach_each_publish_one_graph_revision() {
    let Some(path) = try_test_video() else { return };
    let (source, streams) = FileDemuxer::open("demux", &path).expect("open test video");
    let index = streams
        .iter()
        .find(|stream| stream.kind == ffmpeg_next::media::Type::Video)
        .expect("test video has a video stream")
        .index;
    let mut handle_slot = None;
    let pipeline = Pipeline::new("test", source, |source, ctx| {
        let (tee_branch, handle) = TeeBuilder::new("tee", ctx.clone()).build_dynamic()?;
        ctx.attach(source, index, tee_branch)?;
        handle_slot = Some(handle);
        Ok(())
    })
    .expect("test pipeline wiring must succeed");
    let handle = handle_slot.expect("wire ran");
    let before = pipeline.graph().revision;
    let detached = handle
        .branch()
        .expect("tee is alive")
        .to(Box::new(NoOpSink {
            name: "dynamic".into(),
            pp_log: element_pp_log(ElementType::Other, "dynamic", None),
        }))
        .unwrap();

    assert_eq!(pipeline.graph().revision, before);
    let branch_id = handle.attach(detached).unwrap();
    assert_eq!(pipeline.graph().revision, before + 1);
    let attached_edge = pipeline
        .graph()
        .edges
        .into_iter()
        .find(|edge| edge.branch_id == branch_id)
        .expect("dynamic branch edge is present");
    assert_eq!(&*attached_edge.from.port, "tee_src0");

    handle.detach(branch_id).unwrap();
    assert_eq!(pipeline.graph().revision, before + 2);

    let replacement = handle
        .branch()
        .expect("tee is alive")
        .to(Box::new(NoOpSink {
            name: "replacement".into(),
            pp_log: element_pp_log(ElementType::Other, "replacement", None),
        }))
        .unwrap();
    let replacement_id = handle.attach(replacement).unwrap();
    assert_eq!(pipeline.graph().revision, before + 3);
    let replacement_edge = pipeline
        .graph()
        .edges
        .into_iter()
        .find(|edge| edge.branch_id == replacement_id)
        .expect("replacement branch edge is present");
    assert_eq!(
        &*replacement_edge.from.port, "tee_src1",
        "removed Tee pad names must never be reused"
    );
}

#[test]
fn dynamic_attach_is_rejected_during_a_timeline_operation() {
    let Some(path) = try_test_video() else { return };
    let (source, streams) = FileDemuxer::open("demux", &path).expect("open test video");
    let index = streams
        .iter()
        .find(|stream| stream.kind == ffmpeg::media::Type::Video)
        .expect("test video has a video stream")
        .index;
    let mut handle = None;
    let pipeline = Pipeline::new("attach-during-seek", source, |source, ctx| {
        let (tee, tee_handle) = TeeBuilder::new("tee", ctx.clone()).build_dynamic()?;
        ctx.attach(source, index, tee)?;
        handle = Some(tee_handle);
        Ok(())
    })
    .expect("pipeline wiring");
    let handle = handle.expect("tee handle");
    let branch = handle
        .branch()
        .expect("tee alive")
        .to(Box::new(NoOpSink {
            name: "late".into(),
            pp_log: element_pp_log(ElementType::Other, "late", None),
        }))
        .expect("late branch");

    let operation = pipeline
        .operation
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let error = handle
        .attach(branch)
        .expect_err("attach must not cross a timeline operation");
    assert!(matches!(
        error,
        crate::Error::GraphError(GraphError::TimelineOperationInProgress)
    ));
    drop(operation);
}

#[test]
fn tee_handle_changes_branches_after_the_pipeline_starts() {
    let source = TestVideoSource::new("video", TestVideoOptions::default());
    let initial_count = Arc::new(AtomicUsize::new(0));
    let dynamic_count = Arc::new(AtomicUsize::new(0));
    let mut handle_slot = None;
    let pipeline = Pipeline::new("runtime-tee-test", source, |source, ctx| {
        let initial_branch = ctx.branch().to(Box::new(CountingSink {
            name: "initial".into(),
            count: initial_count.clone(),
            pp_log: element_pp_log(ElementType::Other, "initial", None),
        }))?;
        let (tee_branch, handle) = TeeBuilder::new("tee", ctx.clone())
            .branch(initial_branch)
            .build_dynamic()?;
        ctx.attach(source, 0, tee_branch)?;
        handle_slot = Some(handle);
        Ok(())
    })
    .expect("test pipeline wiring must succeed");
    let handle = handle_slot.expect("wire ran");

    pipeline.run().unwrap();
    thread::sleep(Duration::from_millis(75));
    let dynamic_branch = handle
        .branch()
        .expect("tee is alive")
        .to(Box::new(CountingSink {
            name: "dynamic".into(),
            count: dynamic_count.clone(),
            pp_log: element_pp_log(ElementType::Other, "dynamic", None),
        }))
        .unwrap();
    let branch_id = handle.attach(dynamic_branch).unwrap();
    thread::sleep(Duration::from_millis(100));
    handle.detach(branch_id).unwrap();

    let count_after_detach = dynamic_count.load(Ordering::SeqCst);
    assert!(count_after_detach > 0, "runtime branch received no frames");
    thread::sleep(Duration::from_millis(75));
    assert_eq!(
        dynamic_count.load(Ordering::SeqCst),
        count_after_detach,
        "detached branch kept receiving frames"
    );
    assert!(initial_count.load(Ordering::SeqCst) > count_after_detach);

    pipeline.stop();
    let errors: Vec<_> = pipeline
        .bus()
        .iter()
        .filter(|event| matches!(event, BusEvent::Error { .. }))
        .collect();
    assert!(errors.is_empty(), "unexpected runtime errors: {errors:?}");
}

#[test]
fn bus_messages_carry_the_posting_elements_stable_graph_id() {
    let Some(path) = try_test_video() else { return };
    let (source, streams) = FileDemuxer::open("demux", &path).expect("open test video");
    let index = streams
        .iter()
        .find(|stream| stream.kind == ffmpeg_next::media::Type::Video)
        .expect("test video has a video stream")
        .index;
    let pipeline = Pipeline::new("test", source, |source, ctx| {
        let branch = ctx.branch().to(Box::new(NoOpSink {
            name: "stable-id-sink".into(),
            pp_log: element_pp_log(ElementType::Other, "stable-id-sink", None),
        }))?;
        ctx.attach(source, index, branch)?;
        Ok(())
    })
    .expect("test pipeline wiring must succeed");
    let sink_id = pipeline
        .graph()
        .nodes
        .iter()
        .find(|node| &*node.name == "stable-id-sink")
        .expect("sink is attached")
        .id;

    pipeline.run().unwrap();
    let messages: Vec<_> = pipeline.bus().iter_with_ids().collect();
    assert!(messages.iter().any(|message| {
        message.element_id == Some(sink_id)
            && matches!(
                &message.event,
                BusEvent::Eos { name, .. } if &**name == "stable-id-sink"
            )
    }));
}

#[test]
fn a_source_that_fails_still_stops_its_own_branch() {
    // Without this the branch is only dropped, so a stateful sink — a muxer
    // waiting to write its trailer — never learns the stream is over and
    // leaves an unplayable file behind.
    let stopped = Arc::new(AtomicBool::new(false));
    let sink = StopRecordingSink {
        pp_log: PpLog::new("Other", "stop-recorder", None),
        stopped: stopped.clone(),
    };
    let source = FailingSource {
        pp_log: PpLog::new("Other", "failing", None),
        pad: SrcPad::new("failing_src"),
    };
    let pipeline = Pipeline::new("failing-source", source, |source, ctx| {
        let branch = ctx.branch().to(Box::new(sink))?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("wiring succeeds");

    pipeline.run().unwrap();
    // Drained to exhaustion, not searched lazily: the error is posted
    // before the branch is stopped, so a `find` that returns on the first
    // `Error` can observe `stopped` while the source thread is still on its
    // way there. Iterating until every `Bus` sender has dropped is what
    // makes the thread's work complete before the assertion below.
    let events: Vec<_> = pipeline.bus().iter().collect();
    assert!(
        events
            .iter()
            .any(|event| matches!(event, BusEvent::Error { .. })),
        "the failure reaches the bus"
    );
    assert!(
        stopped.load(Ordering::Acquire),
        "the branch behind a failed source must still be stopped, \
         or anything holding state downstream can never finalize it"
    );
}

// ---------------------------------------------------------------------
// Link contracts (see `crate::contract`).
//
// These cover the check itself rather than any one element: that a
// mismatch is refused before anything runs, that a `Queue` in the middle
// does not hide one, that an undeclared contract still links, and that
// the pad-to-branch boundary an attach crosses is checked too.
// ---------------------------------------------------------------------

/// A terminal sink declaring whatever a given test needs to check against.
struct DeclaringSink {
    name: Arc<str>,
    pp_log: PpLog,
    contract: InputContract,
}

impl DeclaringSink {
    fn boxed(name: &'static str, contract: InputContract) -> Box<Self> {
        Box::new(Self {
            name: name.into(),
            pp_log: element_pp_log(ElementType::Other, name, None),
            contract,
        })
    }
}

impl Element for DeclaringSink {
    fn name(&self) -> Arc<str> {
        self.name.clone()
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

impl Sink for DeclaringSink {
    fn input_contract(&self) -> InputContract {
        self.contract
    }

    fn consume(&mut self, _buf: MediaBuffer) -> Result<()> {
        Ok(())
    }

    fn control(&mut self, _msg: ControlMsg) -> Result<()> {
        Ok(())
    }
}

fn contract_context() -> Arc<Context> {
    let (bus, _rx) = Bus::new();
    let graph = PipelineGraph::new();
    let source_id = graph.add_source(ElementType::Other, "source".into());
    Arc::new(Context::for_test(bus, "contracts", graph, source_id))
}

/// A decoder built without any fixture — only `codec_type`/`codec_id`
/// decide which one `SwDecoder::new` opens. Same approach as that
/// element's own tests, so these run everywhere rather than only where
/// `MEDIA_PP_TEST_VIDEO` is set.
fn decoder(name: &str, medium: ffmpeg::media::Type, codec: ffmpeg::codec::Id) -> SwDecoder {
    let mut params = ffmpeg::codec::Parameters::new();
    // SAFETY: `as_mut_ptr` on parameters this test just created and still
    // owns exclusively; both are plain fields of `AVCodecParameters`.
    unsafe {
        (*params.as_mut_ptr()).codec_type = medium.into();
        (*params.as_mut_ptr()).codec_id = codec.into();
    }
    SwDecoder::new(name, params).expect("the built-in decoder must open")
}

fn video_decoder(name: &str) -> SwDecoder {
    decoder(name, ffmpeg::media::Type::Video, ffmpeg::codec::Id::H264)
}

fn audio_decoder(name: &str) -> SwDecoder {
    decoder(name, ffmpeg::media::Type::Audio, ffmpeg::codec::Id::AAC)
}

fn video_frames() -> InputContract {
    InputContract::Fixed(PortContract::frame(
        MediaKind::VideoFrame,
        MemoryDomain::System,
    ))
}

#[test]
fn a_decoder_reaches_a_matching_sink_through_a_queue() {
    contract_context()
        .branch()
        .pipe(video_decoder("decoder"))
        .queue("q", 4)
        .to(DeclaringSink::boxed("renderer", video_frames()))
        .expect("decoded video into a video sink is a valid link");
}

/// The one that makes the whole feature worth having: a `Queue` is a
/// thread boundary, not a transform, so a mismatch two stages apart must
/// still be caught — and must still name the decoder that actually
/// produces the frames rather than the queue that would have relayed them.
#[test]
fn a_queue_in_the_middle_does_not_hide_a_mismatch() {
    let Err(error) = contract_context()
        .branch()
        .pipe(video_decoder("decoder"))
        .queue("q", 4)
        .to(DeclaringSink::boxed(
            "muxer",
            InputContract::Fixed(PortContract::packet(MediaKind::VideoPacket)),
        ))
    else {
        panic!("decoded frames cannot be fed to a packet-only sink");
    };

    let crate::Error::GraphError(GraphError::IncompatibleLink {
        producer, consumer, ..
    }) = error
    else {
        panic!("expected an IncompatibleLink, got {error}");
    };
    assert_eq!(&*producer, "decoder");
    assert_eq!(&*consumer, "muxer");
}

/// Two `SwDecoder`s of the same `ElementType` legitimately produce
/// different kinds, decided by the stream parameters each was built with.
/// This is what makes a contract a property of the instance rather than
/// something a table keyed by `ElementType` could ever answer.
#[test]
fn an_audio_decoder_is_rejected_where_a_video_decoder_would_link() {
    contract_context()
        .branch()
        .pipe(video_decoder("video"))
        .to(DeclaringSink::boxed("encoder", video_frames()))
        .expect("the video decoder is the case this sink accepts");

    let Err(error) = contract_context()
        .branch()
        .pipe(audio_decoder("audio"))
        .to(DeclaringSink::boxed("encoder", video_frames()))
    else {
        panic!("an audio decoder cannot feed a video-only sink");
    };

    assert!(
        matches!(
            error,
            crate::Error::GraphError(GraphError::IncompatibleLink { .. })
        ),
        "got {error}"
    );
}

/// Over-rejection is the real risk of a check like this: a wrong
/// declaration refuses a pipeline that works. An element that declares
/// nothing must keep linking to anything, exactly as before contracts
/// existed.
#[test]
fn an_undeclared_contract_still_links_to_anything() {
    contract_context()
        .branch()
        .pipe(video_decoder("decoder"))
        .to(Box::new(NoOpSink {
            name: "undeclared".into(),
            pp_log: element_pp_log(ElementType::Other, "undeclared", None),
        }))
        .expect("an Unknown contract is missing information, not a refusal");
}

/// The boundary `ChainBuilder::to` cannot see: a branch is built on its
/// own and only meets the pad feeding it at attach time. A refusal there
/// has to leave both the pad and the graph exactly as they were.
#[test]
fn attaching_a_packet_pad_to_a_video_branch_changes_nothing() {
    let context = contract_context();
    let branch = context
        .branch()
        .queue("q", 4)
        .to(DeclaringSink::boxed("encoder", video_frames()))
        .expect("the branch itself is consistent");
    let before = context.graph.snapshot();

    let mut pad = SrcPad::with_contract(
        "demuxer_src",
        OutputContract::Fixed(PortContract::packet(MediaKind::VideoPacket)),
    );
    let error = context
        .attach_pad(&mut pad, branch)
        .expect_err("a packet pad cannot feed a branch that decodes nothing");

    let crate::Error::GraphError(GraphError::IncompatibleLink {
        producer, consumer, ..
    }) = error
    else {
        panic!("expected an IncompatibleLink, got {error}");
    };
    assert_eq!(&*producer, "demuxer_src");
    assert_eq!(
        &*consumer, "encoder",
        "the leading Queue defers the question rather than answering it"
    );

    assert!(!pad.is_linked(), "a refused attach must not link the pad");
    let after = context.graph.snapshot();
    assert_eq!(before.revision, after.revision);
    assert_eq!(before.nodes.len(), after.nodes.len());
}

/// The declaration on a real source pad rather than on a test double.
/// Needs a container to open, so it only runs where one is configured.
#[test]
fn a_demuxer_declares_packets_on_every_stream_pad() {
    let Some(path) = try_test_video() else {
        eprintln!("skipped: MEDIA_PP_TEST_VIDEO is not set to a readable file");
        return;
    };
    let (mut demuxer, streams) =
        FileDemuxer::open("demuxer", &path).expect("the fixture must open");

    // Per stream, from the medium the container announced — a video pad
    // and an audio pad are both `MediaBuffer::Packet` but not the same
    // contract, which is the whole point of splitting the kind.
    for (pad, stream) in demuxer.src_pads().iter().zip(&streams) {
        let expected = match MediaKind::packet_for(stream.kind) {
            Some(kind) => OutputContract::Fixed(PortContract::packet(kind)),
            None => OutputContract::Unknown,
        };
        assert_eq!(pad.contract(), expected);
    }
}

/// What the memory domain was added for. Both frames here are
/// `MediaBuffer::Video`, so nothing about the buffer type distinguishes
/// them — only the domain says one lives in system memory and the other
/// in a texture, and only that catches the missing `D3d11Upload`.
#[cfg(all(target_os = "windows", feature = "d3d11"))]
#[test]
fn a_system_memory_frame_cannot_feed_a_d3d11_filter() {
    use crate::elements::{D3d11ScalerFormat, D3d11Upload};

    let Some((device, d3d_context)) = crate::test_support::try_d3d11_device() else {
        eprintln!("skipped: no D3D11 hardware device available");
        return;
    };
    let gpu_frames = InputContract::Fixed(PortContract::frame(
        MediaKind::VideoFrame,
        MemoryDomain::D3d11,
    ));
    // A device existing is not the same as it supporting the video
    // processor this scaler opens: CI machines routinely have one without
    // the other, and that is a missing capability to skip on, not a
    // failure of what this test is checking.
    let scaler = |name: &str| {
        crate::elements::D3d11Scaler::new(
            name,
            &device,
            d3d_context.clone(),
            D3d11ScalerFormat::Preserve,
            64,
            64,
        )
    };
    if let Err(error) = scaler("probe") {
        eprintln!("skipped: this D3D11 device has no usable video processor ({error})");
        return;
    }
    let scaler = |name: &str| scaler(name).expect("the probe above already opened one");

    let Err(error) = contract_context()
        .branch()
        .pipe(video_decoder("decoder"))
        .pipe(scaler("scaler"))
        .to(DeclaringSink::boxed("renderer", gpu_frames))
    else {
        panic!("a software decoder's frames never reach a GPU scaler");
    };

    let crate::Error::GraphError(GraphError::IncompatibleLink {
        producer, consumer, ..
    }) = error
    else {
        panic!("expected an IncompatibleLink, got {error}");
    };
    assert_eq!(&*producer, "decoder");
    assert_eq!(&*consumer, "scaler");

    // The same chain with the upload that was missing.
    contract_context()
        .branch()
        .pipe(video_decoder("decoder"))
        .pipe(D3d11Upload::new("upload", &device, 64, 64))
        .pipe(scaler("scaler"))
        .to(DeclaringSink::boxed("renderer", gpu_frames))
        .expect("a D3d11Upload is exactly what makes this chain valid");
}

/// The reverse direction, which is just as easy to get wrong: a device
/// texture handed to a CPU filter.
#[cfg(all(target_os = "windows", feature = "d3d11"))]
#[test]
fn a_d3d11_frame_cannot_feed_a_cpu_filter() {
    use crate::elements::{D3d11Upload, SwScaler};

    let Some((device, _d3d_context)) = crate::test_support::try_d3d11_device() else {
        eprintln!("skipped: no D3D11 hardware device available");
        return;
    };

    let Err(error) = contract_context()
        .branch()
        .pipe(video_decoder("decoder"))
        .pipe(D3d11Upload::new("upload", &device, 64, 64))
        .pipe(SwScaler::new(
            "scaler",
            ffmpeg::format::Pixel::YUV420P,
            64,
            64,
            ffmpeg::software::scaling::Flags::BILINEAR,
        ))
        .to(DeclaringSink::boxed("sink", video_frames()))
    else {
        panic!("a device texture has no CPU-readable planes for swscale");
    };

    assert!(
        matches!(
            error,
            crate::Error::GraphError(GraphError::IncompatibleLink { .. })
        ),
        "got {error}"
    );
}

/// A port that deals in more than one kind is why the contract holds a
/// set rather than a single `MediaKind`: `FrameCounter` tallies decoded
/// buffers of either medium, and both must link.
#[test]
fn a_frame_counter_takes_either_decoded_medium() {
    use crate::elements::FrameCounter;

    for decoder in [video_decoder("video"), audio_decoder("audio")] {
        let (counter, _count) = FrameCounter::new("counter");
        contract_context()
            .branch()
            .pipe(decoder)
            .to(Box::new(counter))
            .expect("a decoded-buffer counter accepts both video and audio");
    }
}

/// The same counter pair in the other direction: `PacketCounter` deals in
/// encoded data only, so the decoder that feeds its sibling cannot feed it.
#[test]
fn a_decoded_frame_cannot_feed_a_packet_only_sink() {
    use crate::elements::PacketCounter;

    let (counter, _count) = PacketCounter::new("counter");
    let Err(error) = contract_context()
        .branch()
        .pipe(video_decoder("decoder"))
        .to(Box::new(counter))
    else {
        panic!("decoded frames are not packets");
    };

    let crate::Error::GraphError(GraphError::IncompatibleLink {
        producer, consumer, ..
    }) = error
    else {
        panic!("expected an IncompatibleLink, got {error}");
    };
    assert_eq!(&*producer, "decoder");
    assert_eq!(&*consumer, "counter");
}

/// Video and audio are both decoded frames, so nothing but the media kind
/// separates a video filter from an audio one.
#[test]
fn an_audio_filter_refuses_video_frames() {
    use crate::elements::AudioVolume;

    let (volume, _handle) = AudioVolume::new("volume");
    let Err(error) = contract_context()
        .branch()
        .pipe(video_decoder("decoder"))
        .pipe(volume)
        .to(DeclaringSink::boxed("sink", video_frames()))
    else {
        panic!("a gain filter has no samples to scale in a video frame");
    };

    assert!(
        matches!(
            error,
            crate::Error::GraphError(GraphError::IncompatibleLink { .. })
        ),
        "got {error}"
    );

    // The audio decoder is the case that filter is for.
    let (volume, _handle) = AudioVolume::new("volume");
    contract_context()
        .branch()
        .pipe(audio_decoder("decoder"))
        .pipe(volume)
        .to(DeclaringSink::boxed(
            "sink",
            InputContract::Fixed(PortContract::frame(
                MediaKind::AudioFrame,
                MemoryDomain::System,
            )),
        ))
        .expect("decoded audio through a gain filter is the intended chain");
}

/// Two GPU frames of different backends. Neither the buffer variant nor a
/// single "is on a GPU" flag separates a D3D11 texture from a CUDA
/// allocation — only naming the backend does, which is why the domain is
/// an enum rather than a boolean.
#[cfg(all(target_os = "windows", feature = "d3d11", feature = "cuda"))]
#[test]
fn a_d3d11_texture_cannot_feed_a_cuda_filter() {
    use crate::elements::{CudaScaler, CudaScalerInterp, D3d11Upload};

    let Some((device, _d3d_context)) = crate::test_support::try_d3d11_device() else {
        eprintln!("skipped: no D3D11 hardware device available");
        return;
    };
    let Some((cuda, _cuda_guard)) = crate::test_support::try_cuda_device() else {
        return;
    };

    let Err(error) = contract_context()
        .branch()
        .pipe(video_decoder("decoder"))
        .pipe(D3d11Upload::new("upload", &device, 64, 64))
        .pipe(CudaScaler::new(
            "scaler",
            &cuda,
            64,
            64,
            CudaScalerInterp::Bilinear,
        ))
        .to(DeclaringSink::boxed(
            "sink",
            InputContract::Fixed(PortContract::frame(
                MediaKind::VideoFrame,
                MemoryDomain::Cuda,
            )),
        ))
    else {
        panic!("a D3D11 texture is not reachable from a CUDA kernel");
    };

    let crate::Error::GraphError(GraphError::IncompatibleLink {
        producer, consumer, ..
    }) = error
    else {
        panic!("expected an IncompatibleLink, got {error}");
    };
    assert_eq!(&*producer, "upload");
    assert_eq!(&*consumer, "scaler");
}

/// `D3d12Renderer` takes device resources only, matching `D3d11Renderer`
/// and `CudaRenderer`. A CPU-decoded stream reaches it through
/// `D3d12Upload` — the one place a system frame crosses to the GPU —
/// rather than through a second upload path inside the sink.
#[cfg(all(target_os = "windows", feature = "d3d12"))]
#[test]
fn a_d3d12_renderer_takes_device_resources_only() {
    use std::any::Any;

    use windows::Win32::Graphics::Direct3D12::{ID3D12Device, ID3D12Fence, ID3D12Resource};

    use crate::elements::{D3d12FrameRenderer, D3d12Renderer, D3d12Upload, SubmitError};

    /// Never submitted to: the link check runs when the branch is built,
    /// so no buffer ever reaches these.
    struct StubRenderer(ID3D12Device);

    impl D3d12FrameRenderer for StubRenderer {
        fn device(&self) -> ID3D12Device {
            self.0.clone()
        }

        unsafe fn submit_nv12_texture(
            &self,
            _texture: ID3D12Resource,
            _fence: ID3D12Fence,
            _fence_value: u64,
            _width: u32,
            _height: u32,
            _keep_alive: Box<dyn Any + Send>,
        ) -> std::result::Result<(), SubmitError> {
            unreachable!("the link check never pushes a buffer")
        }

        fn resize(&self, _width: u32, _height: u32) -> std::result::Result<(), SubmitError> {
            unreachable!("the link check never resizes")
        }
    }

    let Some(device) = crate::test_support::try_d3d12_device() else {
        eprintln!("skipped: no D3D12 hardware device available");
        return;
    };
    // The second half of this test needs a working D3D12VA hw frames
    // context, which a device alone does not guarantee. Probe for it up
    // front so a machine without one skips rather than failing halfway.
    let upload = match D3d12Upload::new("upload", &device, 64, 64) {
        Ok(upload) => upload,
        Err(error) => {
            eprintln!("skipped: this D3D12 device cannot open a frames context ({error})");
            return;
        }
    };

    let Err(error) = contract_context()
        .branch()
        .pipe(video_decoder("decoder"))
        .to(Box::new(D3d12Renderer::new(
            "renderer",
            Box::new(StubRenderer(device.clone())),
        )))
    else {
        panic!("a software decoder's frames have no path to the swap chain");
    };

    let crate::Error::GraphError(GraphError::IncompatibleLink {
        producer, consumer, ..
    }) = error
    else {
        panic!("expected an IncompatibleLink, got {error}");
    };
    assert_eq!(&*producer, "decoder");
    assert_eq!(&*consumer, "renderer");

    // The upload that was missing is what makes the same chain valid.
    contract_context()
        .branch()
        .pipe(video_decoder("decoder"))
        .pipe(upload)
        .to(Box::new(D3d12Renderer::new(
            "renderer",
            Box::new(StubRenderer(device)),
        )))
        .expect("a D3D12 resource is exactly what this renderer accepts");
}

/// A passthrough element that declared nothing used to end the check at
/// itself, because `Unknown` output means nothing is known to be flowing
/// onward. `VideoSynchronizer` sits mid-branch in every A/V playback
/// pipeline, so leaving it undeclared blinded the rest of the chain.
#[test]
fn a_video_synchronizer_carries_the_contract_past_itself() {
    use crate::elements::{PacketCounter, VideoSynchronizer};

    let context = contract_context();
    let sync = VideoSynchronizer::new(
        "sync",
        ffmpeg::Rational::new(1, 90_000),
        context.playback_clock.clone(),
    )
    .expect("a valid time base opens the synchronizer");

    let (counter, _count) = PacketCounter::new("counter");
    let Err(error) = context
        .branch()
        .pipe(video_decoder("decoder"))
        .pipe(sync)
        .to(Box::new(counter))
    else {
        panic!("scheduling frames does not turn them into packets");
    };

    let crate::Error::GraphError(GraphError::IncompatibleLink {
        producer, consumer, ..
    }) = error
    else {
        panic!("expected an IncompatibleLink, got {error}");
    };
    assert_eq!(
        &*producer, "decoder",
        "a passthrough stage forwards the contract and the producer's name with it"
    );
    assert_eq!(&*consumer, "counter");
}

/// The head of a chain. A source pad that declares nothing leaves the
/// attach boundary unchecked for that whole pipeline, so the synthetic
/// and capture sources have to declare theirs too.
#[test]
fn a_video_source_cannot_be_attached_to_an_audio_branch() {
    use crate::elements::{TestVideoOptions, TestVideoSource};

    let context = contract_context();
    let branch = context
        .branch()
        .queue("q", 4)
        .to(DeclaringSink::boxed(
            "speakers",
            InputContract::Fixed(PortContract::frame(
                MediaKind::AudioFrame,
                MemoryDomain::System,
            )),
        ))
        .expect("the branch itself is consistent");

    let mut source = TestVideoSource::new("video", TestVideoOptions::default());
    let Err(error) = context.attach(&mut source, 0, branch) else {
        panic!("a video source has no samples for an audio renderer");
    };

    assert!(
        matches!(
            error,
            crate::Error::GraphError(GraphError::IncompatibleLink { .. })
        ),
        "got {error}"
    );
    assert!(
        !source.src_pads()[0].is_linked(),
        "a refused attach must not link the pad"
    );
}

/// The mistake the medium split exists for. A container's audio and video
/// pads both emit `MediaBuffer::Packet`, so before the kind carried the
/// medium this wired up cleanly and failed somewhere inside libavcodec on
/// the first packet instead.
#[test]
fn a_containers_audio_stream_cannot_feed_a_video_decoder() {
    let Some(path) = try_test_video() else {
        eprintln!("skipped: MEDIA_PP_TEST_VIDEO is not set to a readable file");
        return;
    };
    let (mut demuxer, streams) =
        FileDemuxer::open("demuxer", &path).expect("the fixture must open");
    eprintln!(
        "fixture streams: {:?}",
        streams
            .iter()
            .map(|s| (s.index, s.kind))
            .collect::<Vec<_>>()
    );
    let Some(audio) = streams
        .iter()
        .find(|s| s.kind == ffmpeg::media::Type::Audio)
    else {
        eprintln!("skipped: the fixture has no audio stream to mis-wire");
        return;
    };
    let audio_index = audio.index;
    let video = streams
        .iter()
        .find(|s| s.kind == ffmpeg::media::Type::Video)
        .expect("the fixture must have a video stream");
    let video_params = demuxer
        .stream_parameters(video.index)
        .expect("the video stream must expose parameters");

    let context = contract_context();
    let branch = context
        .branch()
        .pipe(SwDecoder::new("video-decoder", video_params).expect("the decoder must open"))
        .to(DeclaringSink::boxed("renderer", video_frames()))
        .expect("the branch itself is consistent");

    let Err(error) = context.attach(&mut demuxer, audio_index, branch) else {
        panic!("an audio stream has nothing a video decoder can decode");
    };

    let crate::Error::GraphError(GraphError::IncompatibleLink {
        produced, accepted, ..
    }) = error
    else {
        panic!("expected an IncompatibleLink, got {error}");
    };
    assert_eq!(produced.to_string(), "AudioPacket");
    assert_eq!(accepted.to_string(), "VideoPacket");

    // The video stream of the same container is what that decoder is for.
    let branch = context
        .branch()
        .pipe(
            SwDecoder::new(
                "video-decoder",
                demuxer
                    .stream_parameters(video.index)
                    .expect("the video stream must expose parameters"),
            )
            .expect("the decoder must open"),
        )
        .to(DeclaringSink::boxed("renderer", video_frames()))
        .expect("the branch itself is consistent");
    context
        .attach(&mut demuxer, video.index, branch)
        .expect("the video pad and a video decoder are exactly the intended link");
}

/// Guards the exact text README quotes, so the two cannot drift.
#[test]
fn an_incompatible_link_reads_the_way_the_readme_shows_it() {
    let (counter, _count) = crate::elements::PacketCounter::new("rec");
    let Err(error) = contract_context()
        .branch()
        .pipe(video_decoder("decoder"))
        .to(Box::new(counter))
    else {
        panic!("decoded frames are not packets");
    };
    assert_eq!(
        error.to_string(),
        "decoder produces VideoFrame (System), which rec cannot accept \
         (it takes VideoPacket|AudioPacket)"
    );
}

/// A `D3d11FrameRenderer` that is never submitted to: the link check runs
/// when a branch is built or attached, so no buffer ever reaches it.
#[cfg(all(target_os = "windows", feature = "d3d11"))]
struct StubD3d11Renderer(windows::Win32::Graphics::Direct3D11::ID3D11Device);

#[cfg(all(target_os = "windows", feature = "d3d11"))]
impl crate::elements::D3d11FrameRenderer for StubD3d11Renderer {
    fn device(&self) -> windows::Win32::Graphics::Direct3D11::ID3D11Device {
        self.0.clone()
    }

    unsafe fn submit_bgra_texture(
        &self,
        _texture: windows::Win32::Graphics::Direct3D11::ID3D11Texture2D,
        _array_index: u32,
        _width: u32,
        _height: u32,
    ) -> std::result::Result<(), crate::elements::SubmitError> {
        unreachable!("the link check never pushes a buffer")
    }

    unsafe fn submit_nv12_texture(
        &self,
        _texture: windows::Win32::Graphics::Direct3D11::ID3D11Texture2D,
        _array_index: u32,
        _width: u32,
        _height: u32,
    ) -> std::result::Result<(), crate::elements::SubmitError> {
        unreachable!("the link check never pushes a buffer")
    }

    fn resize(
        &self,
        _width: u32,
        _height: u32,
    ) -> std::result::Result<(), crate::elements::SubmitError> {
        unreachable!("the link check never resizes")
    }
}

/// A branch whose leading stages are all passthrough has no requirement of
/// its own — the one that matters belongs to an element further down, and
/// summarizing the branch as "what its first stage accepts" threw that
/// away. `VideoSynchronizer` takes a frame from any backend, so before the
/// branch was re-walked at attach time a system-memory source linked to a
/// D3D11 renderer cleanly and failed per frame at runtime.
#[cfg(all(target_os = "windows", feature = "d3d11"))]
#[test]
fn a_passthrough_at_the_head_of_a_branch_still_carries_the_downstream_requirement() {
    use crate::elements::{TestVideoOptions, TestVideoSource, VideoSynchronizer};

    let Some((device, d3d_context)) = crate::test_support::try_d3d11_device() else {
        eprintln!("skipped: no D3D11 hardware device available");
        return;
    };
    let context = contract_context();
    let renderer = crate::elements::D3d11Renderer::new(
        "renderer",
        Box::new(StubD3d11Renderer(device.clone())),
    );
    let _ = &d3d_context;

    let branch = context
        .branch()
        .pipe(
            VideoSynchronizer::new(
                "sync",
                ffmpeg::Rational::new(1, 90_000),
                context.playback_clock.clone(),
            )
            .expect("a valid time base opens the synchronizer"),
        )
        .to(Box::new(renderer))
        .expect("nothing is flowing yet, so the branch alone is consistent");
    let before = context.graph.snapshot();

    let mut source = TestVideoSource::new("video", TestVideoOptions::default());
    let Err(error) = context.attach(&mut source, 0, branch) else {
        panic!("system-memory frames never reach a D3D11 swap chain");
    };

    let crate::Error::GraphError(GraphError::IncompatibleLink {
        producer, consumer, ..
    }) = error
    else {
        panic!("expected an IncompatibleLink, got {error}");
    };
    assert_eq!(
        &*producer, "video_src",
        "the synchronizer forwards the pad's own contract rather than replacing it"
    );
    assert_eq!(
        &*consumer, "renderer",
        "the requirement belongs to the element past the passthrough stage"
    );

    assert!(!source.src_pads()[0].is_linked());
    let after = context.graph.snapshot();
    assert_eq!(before.revision, after.revision);
    assert_eq!(before.nodes.len(), after.nodes.len());
}

/// The same shape with the upload that was missing, to prove the walk is
/// not simply refusing every branch that starts with a passthrough stage.
#[cfg(all(target_os = "windows", feature = "d3d11"))]
#[test]
fn a_passthrough_at_the_head_of_a_branch_accepts_a_matching_source() {
    use crate::elements::{D3d11Upload, TestVideoOptions, TestVideoSource, VideoSynchronizer};

    let Some((device, _d3d_context)) = crate::test_support::try_d3d11_device() else {
        eprintln!("skipped: no D3D11 hardware device available");
        return;
    };
    let context = contract_context();
    let renderer = crate::elements::D3d11Renderer::new(
        "renderer",
        Box::new(StubD3d11Renderer(device.clone())),
    );

    let branch = context
        .branch()
        .pipe(
            VideoSynchronizer::new(
                "sync",
                ffmpeg::Rational::new(1, 90_000),
                context.playback_clock.clone(),
            )
            .expect("a valid time base opens the synchronizer"),
        )
        .pipe(D3d11Upload::new("upload", &device, 64, 64))
        .to(Box::new(renderer))
        .expect("the branch itself is consistent");

    let mut source = TestVideoSource::new("video", TestVideoOptions::default());
    context
        .attach(&mut source, 0, branch)
        .expect("the upload is exactly what makes this source reach the renderer");
}

// ---------------------------------------------------------------------
// Tee branches, initial and dynamic.
//
// A Tee's pads are Passthrough, so nothing about the pad alone says what
// a branch attached to it will receive. Both of these used to link
// unchecked: the initial branches because their plans were merged into
// the Tee's without their contracts, and the dynamic ones because
// TeeHandle::attach committed straight to the graph.
// ---------------------------------------------------------------------

fn audio_sink(name: &'static str) -> Box<DeclaringSink> {
    DeclaringSink::boxed(
        name,
        InputContract::Fixed(PortContract::frame(
            MediaKind::AudioFrame,
            MemoryDomain::System,
        )),
    )
}

/// An initial branch is merged into the `Tee`'s own plan, so the attach
/// that commits the `Tee` has to check it against what the `Tee` receives.
#[test]
fn an_initial_tee_branch_is_checked_against_what_the_tee_receives() {
    use crate::elements::{TestVideoOptions, TestVideoSource};

    let context = contract_context();
    let audio_branch = context
        .branch()
        .to(audio_sink("speakers"))
        .expect("the branch itself is consistent");
    let tee = TeeBuilder::new("tee", context.clone())
        .branch(audio_branch)
        .build()
        .expect("building the fan-out does not check it against a source");
    let before = context.graph.snapshot();

    let mut source = TestVideoSource::new("video", TestVideoOptions::default());
    let Err(error) = context.attach(&mut source, 0, tee) else {
        panic!("a video source has no samples for an audio sink behind a Tee");
    };

    let crate::Error::GraphError(GraphError::IncompatibleLink {
        producer, consumer, ..
    }) = error
    else {
        panic!("expected an IncompatibleLink, got {error}");
    };
    assert_eq!(
        &*producer, "video_src",
        "the Tee forwards what it was given rather than producing its own"
    );
    assert_eq!(&*consumer, "speakers");

    assert!(!source.src_pads()[0].is_linked());
    let after = context.graph.snapshot();
    assert_eq!(before.revision, after.revision);
    assert_eq!(before.nodes.len(), after.nodes.len());
}

/// Valid siblings all attach — the fan-out edge hands each branch the same
/// flow, so a Tee with several good branches is not refused.
#[test]
fn every_valid_initial_tee_branch_attaches() {
    use crate::elements::{TestVideoOptions, TestVideoSource};

    let context = contract_context();
    let one = context
        .branch()
        .to(DeclaringSink::boxed("first", video_frames()))
        .expect("consistent");
    let two = context
        .branch()
        .queue("q", 4)
        .to(DeclaringSink::boxed("second", video_frames()))
        .expect("consistent");
    let tee = TeeBuilder::new("tee", context.clone())
        .branch(one)
        .branch(two)
        .build()
        .expect("consistent");

    let mut source = TestVideoSource::new("video", TestVideoOptions::default());
    context
        .attach(&mut source, 0, tee)
        .expect("both branches take the frames this source produces");
}

/// A branch added while the pipeline runs is checked against the flow its
/// siblings already carry, which the graph recorded when the `Tee` was
/// committed. A failure leaves the fan-out exactly as it was.
#[test]
fn a_dynamic_tee_branch_is_checked_and_a_refusal_changes_nothing() {
    use crate::elements::{TestVideoOptions, TestVideoSource};

    let context = contract_context();
    let initial = context
        .branch()
        .to(DeclaringSink::boxed("renderer", video_frames()))
        .expect("consistent");
    let (tee, handle) = TeeBuilder::new("tee", context.clone())
        .branch(initial)
        .build_dynamic()
        .expect("consistent");
    let mut source = TestVideoSource::new("video", TestVideoOptions::default());
    context
        .attach(&mut source, 0, tee)
        .expect("the initial fan-out matches the source");

    let attached = context.graph.snapshot();
    let audio_branch = handle
        .branch()
        .expect("the Tee is alive")
        .to(audio_sink("speakers"))
        .expect("the branch itself is consistent");

    let Err(error) = handle.attach(audio_branch) else {
        panic!("this Tee carries video frames, not samples");
    };
    assert!(
        matches!(
            error,
            crate::Error::GraphError(GraphError::IncompatibleLink { .. })
        ),
        "got {error}"
    );

    let after = context.graph.snapshot();
    assert_eq!(
        attached.revision, after.revision,
        "a refused dynamic attach must not bump the graph revision"
    );
    assert_eq!(attached.nodes.len(), after.nodes.len());
    assert_eq!(attached.edges.len(), after.edges.len());

    // The Tee still works, and a matching branch still attaches.
    let good = handle
        .branch()
        .expect("the Tee is alive")
        .to(DeclaringSink::boxed("second", video_frames()))
        .expect("consistent");
    handle
        .attach(good)
        .expect("a video branch is what this Tee can feed");
}

/// Resolved contracts are live-graph state just like nodes and edges. A
/// detached dynamic branch must release those entries too; element IDs are
/// never reused, so leaving one behind would grow the graph for every churn
/// cycle even though snapshots and `sink_count` said the branch was gone.
#[test]
fn detaching_a_dynamic_tee_branch_releases_its_resolved_contracts() {
    use crate::elements::{TestVideoOptions, TestVideoSource};

    let context = contract_context();
    let fixed = context
        .branch()
        .to(DeclaringSink::boxed("fixed", video_frames()))
        .expect("consistent");
    let (tee, handle) = TeeBuilder::new("tee", context.clone())
        .branch(fixed)
        .build_dynamic()
        .expect("consistent");
    let mut source = TestVideoSource::new("video", TestVideoOptions::default());
    context
        .attach(&mut source, 0, tee)
        .expect("the fixed branch matches the source");
    let baseline = context.graph.resolved_output_count();

    let dynamic = handle
        .branch()
        .expect("the Tee is alive")
        .queue("dynamic-q", 4)
        .to(DeclaringSink::boxed("dynamic", video_frames()))
        .expect("consistent");
    let branch_id = handle.attach(dynamic).expect("attach the dynamic branch");
    assert!(
        context.graph.resolved_output_count() > baseline,
        "the attached branch must contribute resolved contract entries"
    );

    handle.detach(branch_id).expect("detach the dynamic branch");
    assert_eq!(
        context.graph.resolved_output_count(),
        baseline,
        "detaching must release every resolved contract owned by the branch"
    );
}

/// Records where each branch's decoded media actually landed, so a preroll can
/// be checked against the position it was asked for rather than just against
/// "something arrived".
struct PrerollProbe {
    label: &'static str,
    time_base: ffmpeg::Rational,
    samples: Arc<Mutex<Vec<i64>>>,
    pp_log: PpLog,
}

impl Element for PrerollProbe {
    fn name(&self) -> Arc<str> {
        self.label.into()
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

impl Sink for PrerollProbe {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        let pts = match &buf {
            MediaBuffer::Video(frame) => frame.pts(),
            MediaBuffer::Audio(frame) => frame.pts(),
            _ => None,
        };
        if let Some(pts) = pts {
            let ns = pts.rescale(self.time_base, ffmpeg::Rational(1, 1_000_000_000));
            self.samples.lock().unwrap().push(ns);
        }
        Ok(())
    }

    fn control(&mut self, _msg: ControlMsg) -> Result<()> {
        Ok(())
    }
}

/// A paused seek has to leave *every* decoded branch holding one sample at the
/// requested position — not just the branch that happens to carry a pacing
/// element, and not a burst of them.
///
/// Two separate defects showed up here, and the assertions below fail on
/// either. Measured against the real fixture at a 3s target:
///
/// - With the suppression gate living in `Pacer`/`VideoSynchronizer`, the
///   audio branch had neither, so it delivered 86 samples spanning 0.000s to
///   2.007s while video correctly delivered one at 3.003s — the two streams
///   ended a second apart, which on resume is a second of frozen picture.
/// - With terminals staying open until the *whole* preroll completed, the
///   branch that reached the target first kept consuming while the other
///   caught up: video ran 31 samples from 3.003s to 4.004s.
#[test]
fn a_paused_seek_leaves_every_branch_holding_one_sample_at_the_target() {
    let Some(path) = try_test_video() else { return };
    let (source, streams) = FileDemuxer::open("demux", &path).expect("open test video");
    let video = streams
        .iter()
        .find(|stream| stream.kind == ffmpeg::media::Type::Video)
        .expect("test video has a video stream");
    let Some(audio) = streams
        .iter()
        .find(|stream| stream.kind == ffmpeg::media::Type::Audio)
    else {
        eprintln!("skipping: fixture has no audio stream");
        return;
    };
    let video_params = source.stream_parameters(video.index).expect("video params");
    let audio_params = source.stream_parameters(audio.index).expect("audio params");
    let video_tb = source.stream_time_base(video.index).expect("video tb");
    let audio_tb = source.stream_time_base(audio.index).expect("audio tb");

    let target = Duration::from_secs(3);
    let video_samples = Arc::new(Mutex::new(Vec::new()));
    let audio_samples = Arc::new(Mutex::new(Vec::new()));

    let pipeline = Pipeline::new("paused-av-seek", source, |source, ctx| {
        // Only the video branch is paced, which is the ordinary shape: an
        // audio renderer schedules itself against its own device clock.
        let video_branch = ctx
            .branch()
            .pipe(SwDecoder::new("video-decoder", video_params)?)
            .pipe(Pacer::new("video-pacer", video_tb, ctx.clock.clone())?)
            .queue("video-frames", 8)
            .to(Box::new(PrerollProbe {
                label: "video",
                time_base: video_tb,
                samples: Arc::clone(&video_samples),
                pp_log: element_pp_log(ElementType::Other, "video", None),
            }))?;
        ctx.attach(source, video.index, video_branch)?;
        let audio_branch = ctx
            .branch()
            .queue("audio-packets", 8)
            .pipe(SwDecoder::new("audio-decoder", audio_params)?)
            .to(Box::new(PrerollProbe {
                label: "audio",
                time_base: audio_tb,
                samples: Arc::clone(&audio_samples),
                pp_log: element_pp_log(ElementType::Other, "audio", None),
            }))?;
        ctx.attach(source, audio.index, audio_branch)?;
        Ok(())
    })
    .expect("test pipeline wiring must succeed");

    pipeline.run().unwrap();
    thread::sleep(Duration::from_millis(100));
    pipeline.pause();
    video_samples.lock().unwrap().clear();
    audio_samples.lock().unwrap().clear();

    pipeline
        .seek(target, SeekMode::Accurate)
        .expect("a decoded A/V graph accepts a seek");
    let taken = |samples: &Arc<Mutex<Vec<i64>>>| samples.lock().unwrap().clone();
    let video = taken(&video_samples);
    let audio = taken(&audio_samples);
    pipeline.stop();

    let seconds = |ns: &[i64]| {
        ns.iter()
            .map(|ns| format!("{:.3}s", *ns as f64 / 1e9))
            .collect::<Vec<_>>()
    };
    for (label, samples) in [("video", &video), ("audio", &audio)] {
        assert_eq!(
            samples.len(),
            1,
            "{label} must hold exactly one preview sample, got {:?}",
            seconds(samples)
        );
    }
}

/// A packet terminal can finish on the landed keyframe while a sibling
/// decoder still needs many more packets to reach the accurate target. The
/// completed branch must close independently without closing the whole Tee.
#[test]
fn a_completed_tee_branch_does_not_starve_a_sibling_preroll() {
    let Some(path) = try_test_video() else { return };
    let (source, streams) = FileDemuxer::open("demux", &path).expect("open test video");
    let video = streams
        .iter()
        .find(|stream| stream.kind == ffmpeg::media::Type::Video)
        .expect("test video has a video stream");
    let params = source.stream_parameters(video.index).expect("video params");
    let time_base = source.stream_time_base(video.index).expect("video tb");
    let packets = Arc::new(AtomicUsize::new(0));
    let frames = Arc::new(Mutex::new(Vec::new()));

    let pipeline = Pipeline::new("tee-preroll", source, |source, ctx| {
        let packet_branch = ctx.branch().to(Box::new(CountingSink {
            name: "packet-terminal".into(),
            count: Arc::clone(&packets),
            pp_log: element_pp_log(ElementType::Other, "packet-terminal", None),
        }))?;
        let decoded_branch = ctx
            .branch()
            .pipe(SwDecoder::new("decoder", params)?)
            .to(Box::new(PrerollProbe {
                label: "video-terminal",
                time_base,
                samples: Arc::clone(&frames),
                pp_log: element_pp_log(ElementType::Other, "video-terminal", None),
            }))?;
        let tee = TeeBuilder::new("tee", ctx.clone())
            .branch(packet_branch)
            .branch(decoded_branch)
            .build()?;
        ctx.attach(source, video.index, tee)?;
        Ok(())
    })
    .expect("pipeline wiring");

    pipeline.run().expect("run");
    thread::sleep(Duration::from_millis(100));
    pipeline.pause();
    packets.store(0, Ordering::SeqCst);
    frames.lock().unwrap().clear();

    pipeline
        .seek(Duration::from_secs(3), SeekMode::Accurate)
        .expect("both Tee branches preroll");
    assert_eq!(packets.load(Ordering::SeqCst), 1);
    assert_eq!(frames.lock().unwrap().len(), 1);
    pipeline.stop();
}

/// Container duration can extend beyond the last video PTS (for example when
/// audio is longer). Seeking there still has a well-defined paused-preview
/// result: the last decoded video frame, not a silent EOS success that leaves
/// the old picture on screen.
#[test]
fn accurate_seek_at_known_eof_selects_the_last_presentable_frame() {
    let Some(path) = try_test_video() else { return };
    let input = ffmpeg::format::input(&path).expect("open fixture duration");
    let duration = input.duration();
    if duration <= 0 {
        eprintln!("skipping: fixture has no known container duration");
        return;
    }
    drop(input);
    let target = Duration::from_micros(duration as u64);

    let (source, streams) = FileDemuxer::open("demux", &path).expect("open test video");
    let video = streams
        .iter()
        .find(|stream| stream.kind == ffmpeg::media::Type::Video)
        .expect("test video has a video stream");
    let params = source.stream_parameters(video.index).expect("video params");
    let time_base = source.stream_time_base(video.index).expect("video tb");
    let samples = Arc::new(Mutex::new(Vec::new()));

    let pipeline = Pipeline::new("eof-preview", source, |source, ctx| {
        let branch = ctx
            .branch()
            .pipe(SwDecoder::new("decoder", params)?)
            .pipe(Pacer::new("pacer", time_base, Arc::clone(&ctx.clock))?)
            .to(Box::new(PrerollProbe {
                label: "video-terminal",
                time_base,
                samples: Arc::clone(&samples),
                pp_log: element_pp_log(ElementType::Other, "video-terminal", None),
            }))?;
        ctx.attach(source, video.index, branch)?;
        Ok(())
    })
    .expect("pipeline wiring");

    pipeline.run().expect("run");
    thread::sleep(Duration::from_millis(50));
    pipeline.pause();
    samples.lock().unwrap().clear();
    pipeline
        .seek(target, SeekMode::Accurate)
        .expect("known EOF seek prerolls");
    assert_eq!(
        samples.lock().unwrap().len(),
        1,
        "EOF seek must replace the paused picture with the last frame"
    );
    pipeline.stop();
}

/// Seekable, and deliberately silent once it has repositioned — the shape of
/// a source whose preroll can never complete, so the pipeline's wait runs to
/// its full timeout.
struct MuteAfterSeekSource {
    pp_log: PpLog,
    pad: SrcPad,
    sought: Arc<AtomicBool>,
}

impl Element for MuteAfterSeekSource {
    fn name(&self) -> Arc<str> {
        "mute-after-seek".into()
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

impl Source for MuteAfterSeekSource {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl SourceElement for MuteAfterSeekSource {
    fn is_live(&self) -> bool {
        false
    }

    fn is_seekable(&self) -> bool {
        true
    }

    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        loop {
            if drain_control(control, self, bus)?.stopped {
                return Ok(());
            }
            if !self.sought.load(Ordering::Acquire) && self.pad.ready_consume() {
                self.pad
                    .push(MediaBuffer::Packet(Arc::new(ffmpeg::Packet::empty())))?;
            }
            thread::yield_now();
        }
    }

    fn seek(&mut self, target: Duration) -> Result<Duration> {
        self.sought.store(true, Ordering::Release);
        Ok(target)
    }
}

/// `stop` promises to abandon immediately. A seek holds the operation lock
/// for the whole of its preroll wait, so without a way to end that wait from
/// outside the lock, stopping during a seek that cannot preroll waited out the
/// full timeout — and the cancellation terminals already forward on `Stop`
/// could not arrive either, since sending it needs the very same lock.
#[test]
fn stopping_during_a_seek_does_not_wait_out_the_preroll_timeout() {
    let sought = Arc::new(AtomicBool::new(false));
    let seen = Arc::new(AtomicUsize::new(0));
    let source = MuteAfterSeekSource {
        pp_log: element_pp_log(ElementType::Other, "mute-after-seek", None),
        pad: SrcPad::new("src"),
        sought: Arc::clone(&sought),
    };
    let pipeline = Pipeline::new("stop-during-seek", source, |source, ctx| {
        let branch = ctx.branch().to(Box::new(CountingSink {
            name: "sink".into(),
            count: Arc::clone(&seen),
            pp_log: element_pp_log(ElementType::Other, "sink", None),
        }))?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("pipeline wiring");

    pipeline.run().expect("run");
    while seen.load(Ordering::SeqCst) == 0 {
        thread::yield_now();
    }

    let seeking = Arc::clone(&pipeline);
    let seek = thread::spawn(move || seeking.seek(Duration::from_secs(2), SeekMode::Accurate));
    // The seek has to be inside its preroll wait for this to prove anything.
    while !sought.load(Ordering::Acquire) {
        thread::yield_now();
    }

    let started = Instant::now();
    pipeline.stop();
    let elapsed = started.elapsed();

    let outcome = seek.join().expect("seek thread");
    assert!(
        outcome.is_err(),
        "a cancelled preroll must report failure, not success"
    );
    assert!(
        elapsed < Duration::from_secs(2),
        "stop waited {elapsed:?} for a preroll it was abandoning"
    );
}

/// Never accepts a buffer, so a `Queue` in front of it parks and the terminal
/// behind it never reports a preroll sample — a branch that cannot preroll.
struct NeverReadySink {
    pp_log: PpLog,
}

impl Element for NeverReadySink {
    fn name(&self) -> Arc<str> {
        "never-ready".into()
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

impl Sink for NeverReadySink {
    fn ready_consume(&mut self) -> bool {
        false
    }

    fn consume(&mut self, _buf: MediaBuffer) -> Result<()> {
        Ok(())
    }

    fn control(&mut self, _msg: ControlMsg) -> Result<()> {
        Ok(())
    }
}

/// Preroll expects the terminals the graph had when the seek started. A branch
/// detached while that seek is still waiting takes its terminal out of the
/// graph but not out of the expected set, so the seek waited for a sample
/// nobody was left to produce.
#[test]
fn detaching_a_branch_mid_seek_does_not_strand_its_preroll() {
    let seen = Arc::new(AtomicUsize::new(0));
    let source = SeekLoopSource {
        pp_log: element_pp_log(ElementType::Other, "seek-loop", None),
        pad: SrcPad::new("src"),
        seeks: Arc::new(AtomicUsize::new(0)),
    };

    let handle = Arc::new(Mutex::new(None));
    let stash = Arc::clone(&handle);
    let pipeline = Pipeline::new("detach-mid-seek", source, move |source, ctx| {
        let live = ctx.branch().to(Box::new(CountingSink {
            name: "live".into(),
            count: Arc::clone(&seen),
            pp_log: element_pp_log(ElementType::Other, "live", None),
        }))?;
        let (tee, tee_handle) = TeeBuilder::new("tee", ctx.clone())
            .branch(live)
            .build_dynamic()?;
        ctx.attach(source, 0, tee)?;
        *stash.lock().unwrap() = Some(tee_handle);
        Ok(())
    })
    .expect("pipeline wiring");
    let tee = handle.lock().unwrap().take().expect("tee handle");

    // A branch that can never take a preroll sample: the queue in front of it
    // parks because its terminal is never ready.
    let stuck = tee
        .branch()
        .expect("dynamic tee")
        .queue("stuck-queue", 4)
        .to(Box::new(NeverReadySink {
            pp_log: element_pp_log(ElementType::Other, "never-ready", None),
        }))
        .expect("stuck branch");
    let stuck_id = tee.attach(stuck).expect("attach stuck branch");

    pipeline.run().expect("run");
    pipeline.pause();

    let seeking = Arc::clone(&pipeline);
    let seek = thread::spawn(move || seeking.seek(Duration::from_secs(1), SeekMode::Accurate));
    // Give the seek time to reach its preroll wait before the topology moves.
    thread::sleep(Duration::from_millis(200));
    tee.detach(stuck_id).expect("detach mid-seek");

    let started = Instant::now();
    let outcome = seek.join().expect("seek thread");
    let elapsed = started.elapsed();
    pipeline.stop();

    assert!(
        outcome.is_ok(),
        "the seek should complete once the branch it was waiting on is gone, got {outcome:?}"
    );
    assert!(
        elapsed < Duration::from_secs(3),
        "the seek waited {elapsed:?} on a terminal that had left the graph"
    );
}
