use std::{
    sync::{atomic::AtomicUsize, mpsc},
    thread,
    time::Duration,
};

use super::*;
use crate::elements::{
    FileDemuxer, Pacer, TeeBuilder, TestAudioOptions, TestAudioSource, TestVideoOptions,
    TestVideoSource,
};

/// Real video file, one directory up from this crate — same one every
/// example defaults to.
fn test_video() -> &'static str {
    concat!(env!("CARGO_MANIFEST_DIR"), "/../test-video/h265.mp4")
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
    let (source, streams) = FileDemuxer::open("demux", test_video()).expect("open test video");
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

    pipeline.run();

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

    pipeline.run();
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

    pipeline.run();
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

    pipeline.run();
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

    pipeline.run();
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
    let (source, streams) = FileDemuxer::open("demux", test_video()).expect("open test video");
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

    pipeline.run();
    thread::sleep(Duration::from_millis(50));
    pipeline.seek(Duration::from_secs(1));
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
/// (see `FileDemuxer::seek`'s docs) — `test-video/h265.mp4` has
/// keyframes only at 0s and ~8.33s, so any `target` between them has
/// to land back at 0s, nowhere near what was requested. Without
/// `BusEvent::Seeked` reporting that gap, this looked indistinguishable
/// from `seek` silently doing nothing.
#[test]
fn seek_reports_where_it_actually_landed_when_target_is_not_a_keyframe() {
    let (source, streams) = FileDemuxer::open("demux", test_video()).expect("open test video");
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

    pipeline.run();
    thread::sleep(Duration::from_millis(50));
    // 3s falls inside the file's first (only) GOP before the 8.33s
    // keyframe — landing anywhere near 3s would mean this stream
    // unexpectedly grew more keyframes than it's known to have.
    pipeline.seek(Duration::from_secs(3));
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
        seeked.1 < Duration::from_secs(1),
        "expected landed to fall back to the 0s keyframe, got {:?}",
        seeked.1
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
    let (source, streams) = FileDemuxer::open("demux", test_video()).expect("open test video");
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
    let (source, streams) = FileDemuxer::open("demux", test_video()).expect("open test video");
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
    let (source, streams) = FileDemuxer::open("demux", test_video()).expect("open test video");
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

/// Once a branch is pulled off a [`Tee`] via [`TeeHandle::detach`],
/// it should stop showing up in [`Pipeline::topology`] entirely — not
/// keep rendering as still attached under `Tee(...)`, which is what a
/// stale graph node would otherwise do.
#[test]
fn topology_forgets_a_branch_once_it_is_removed_from_the_tee() {
    let (source, streams) = FileDemuxer::open("demux", test_video()).expect("open test video");
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
    let (source, streams) = FileDemuxer::open("demux", test_video()).expect("open test video");
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
    let (source, streams) = FileDemuxer::open("demux", test_video()).expect("open test video");
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
    let (source, _) = FileDemuxer::open("demux", test_video()).expect("open test video");

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
    let (source, streams) = FileDemuxer::open("demux", test_video()).expect("open test video");
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
    let (source, streams) = FileDemuxer::open("demux", test_video()).expect("open test video");
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

    pipeline.run();
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
    let (source, streams) = FileDemuxer::open("demux", test_video()).expect("open test video");
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

    pipeline.run();
    let messages: Vec<_> = pipeline.bus().iter_with_ids().collect();
    assert!(messages.iter().any(|message| {
        message.element_id == Some(sink_id)
            && matches!(
                &message.event,
                BusEvent::Eos { name, .. } if &**name == "stable-id-sink"
            )
    }));
}
