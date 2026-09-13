//! Starting, pausing, finishing, stopping and dropping a pipeline, and
//! what each leaves running.

use super::*;

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

/// `is_running` answers the one question a caller polling for the end has:
/// whether anything is still producing. Before `run` there is nothing;
/// after `stop` there is nothing again, however the source got there.
#[test]
fn is_running_spans_exactly_the_time_a_source_is_on_its_thread() {
    let pipeline = Pipeline::new(
        "running",
        TestVideoSource::new("gen", TestVideoOptions::default()),
        |source, ctx| {
            let branch = ctx.branch().to(Box::new(NoOpSink {
                name: "noop".into(),
                pp_log: element_pp_log(ElementType::Other, "noop", None),
            }))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        },
    )
    .expect("test pipeline wiring must succeed");

    assert!(!pipeline.is_running(), "nothing runs before run()");

    pipeline.run().unwrap();
    thread::sleep(Duration::from_millis(50));
    assert!(pipeline.is_running(), "the source thread is still going");

    pipeline.stop();
    // Blocks until every `Bus` handle is dropped, which is the source
    // thread actually being over rather than merely asked to stop.
    let _: Vec<_> = pipeline.bus().iter().collect();
    assert!(!pipeline.is_running(), "the source thread has finished");
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
/// feeding one [`crate::elements::FileMuxer`]) sharing one `Pipeline`:
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
