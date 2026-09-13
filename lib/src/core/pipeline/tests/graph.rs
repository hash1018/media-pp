//! The graph a pipeline builds: its topology, its branches coming and
//! going, and whose name a bus message or a failure carries.

use super::*;

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
        let pacer = Pacer::new("pacer", time_base)?;
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

        let pacer = Pacer::new("pacer", time_base)?;
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

/// Passes buffers straight through, so a chain can have a stage between the
/// `Queue` and the sink that fails.
struct Passthrough {
    pp_log: PpLog,
    pad: SrcPad,
}

impl Element for Passthrough {
    fn name(&self) -> Arc<str> {
        "the-middle".into()
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

impl Source for Passthrough {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for Passthrough {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        self.pad.push(buf)
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        self.pad.control(msg)
    }
}

/// Refuses every buffer, the way a muxer whose connection has gone does.
struct AlwaysFailingSink {
    pp_log: PpLog,
}

impl Element for AlwaysFailingSink {
    fn name(&self) -> Arc<str> {
        "the-end".into()
    }

    fn element_type(&self) -> ElementType {
        ElementType::FileMuxer
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Sink for AlwaysFailingSink {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        if buf.is_eos() {
            return Ok(());
        }
        Err(crate::Error::Other("the connection went away".into()))
    }

    fn control(&mut self, _msg: ControlMsg) -> Result<()> {
        Ok(())
    }
}

/// A failure at the end of a chain is reported as the element that raised
/// it, not as whatever happened to be nearest the `Queue`.
///
/// This is what the whole tracing arrangement is for. The error travels up
/// through every stage between the sink and the queue, one `?` at a time,
/// and without an identity attached the only thing left to report it under
/// is the stage the queue happens to hand buffers to — `the-middle` here,
/// which did nothing wrong.
#[test]
fn a_failure_deep_in_a_chain_is_reported_under_the_element_that_raised_it() {
    let pipeline = Pipeline::new(
        "origin",
        TestVideoSource::new("source", TestVideoOptions::default()),
        |source, context| {
            let branch = context
                .branch()
                .queue("the-queue", 8)
                .pipe(Passthrough {
                    pp_log: PpLog::new("Other", "the-middle", None),
                    pad: SrcPad::new("the-middle_src"),
                })
                .to(Box::new(AlwaysFailingSink {
                    pp_log: PpLog::new("FileMuxer", "the-end", None),
                }))?;
            context.attach(source, 0, branch)?;
            Ok(())
        },
    )
    .expect("build");
    pipeline.run().expect("run");

    // The first error to arrive is enough: every buffer fails the same way.
    let reported = std::iter::from_fn(|| pipeline.bus().recv_message())
        .find_map(|message| match message.event {
            BusEvent::Error {
                element_type, name, ..
            } => Some((element_type, name)),
            _ => None,
        })
        .expect("the failure reached the bus");
    pipeline.stop();

    assert_eq!(
        reported.0,
        ElementType::FileMuxer,
        "reported as {:?}({}) rather than as the sink that failed",
        reported.0,
        reported.1
    );
    assert_eq!(&*reported.1, "the-end");
}
