//! What each element is doing: `Pipeline::stats`.

use super::*;

// ---- What each element is doing: `Pipeline::stats` --------------------

/// A filter that hands on what it is given and does nothing else, so the
/// counts on either side of it can be compared.
struct PassThrough {
    pp_log: PpLog,
    pad: SrcPad,
}

impl PassThrough {
    fn new() -> Self {
        Self {
            pp_log: element_pp_log(ElementType::Other, "pass", None),
            pad: SrcPad::new("pass_src"),
        }
    }
}

impl Element for PassThrough {
    fn name(&self) -> Arc<str> {
        "pass".into()
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

impl Source for PassThrough {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for PassThrough {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        self.pad.push(buf)
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        self.pad.control(msg)
    }
}

/// Waits, boundedly, for `done` — every buffer here moves on threads of
/// the pipeline's own.
fn wait_for(done: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !done() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
    assert!(done(), "timed out waiting");
}

fn element<'a>(
    stats: &'a crate::stats::PipelineStats,
    name: &str,
) -> &'a crate::stats::ElementStats {
    stats
        .elements
        .iter()
        .find(|element| &*element.name == name)
        .unwrap_or_else(|| panic!("{name} is not reported: {stats:?}"))
}

/// Every stage a chain builds is counted, and so is what leaves the source,
/// across a queue — the thread boundary where the count is easiest to lose.
#[test]
fn every_stage_counts_every_buffer_it_was_handed() {
    const BUFFERS: usize = 40;
    let count = Arc::new(AtomicUsize::new(0));
    let pipeline = Pipeline::new(
        "stats",
        BurstSource {
            pp_log: element_pp_log(ElementType::Other, "burst", None),
            pad: SrcPad::new("burst_src"),
            ready: Arc::new(AtomicBool::new(false)),
            buffers: BUFFERS,
        },
        |source, ctx| {
            let branch = ctx
                .branch()
                .pipe(PassThrough::new())
                .queue("hop", 8)
                .to(Box::new(CountingSink {
                    pp_log: element_pp_log(ElementType::Other, "end", None),
                    name: "end".into(),
                    count: count.clone(),
                }))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        },
    )
    .unwrap();
    pipeline.run().unwrap();
    wait_for(|| count.load(Ordering::SeqCst) == BUFFERS);

    let stats = pipeline.stats();
    let expected = BUFFERS as u64;
    assert_eq!(
        element(&stats, "burst").pads[0].buffers,
        expected,
        "left the source"
    );
    assert_eq!(element(&stats, "pass").buffers_in, expected);
    assert_eq!(element(&stats, "pass").pads[0].buffers, expected);
    assert_eq!(element(&stats, "hop").buffers_in, expected);
    assert_eq!(element(&stats, "end").buffers_in, expected);
    let hop = element(&stats, "hop")
        .queue
        .expect("a queue says how full it is");
    assert_eq!((hop.capacity, hop.dropped), (8, 0));
    for reported in &stats.elements {
        assert_eq!(reported.errors, 0, "{reported:?}");
        assert_eq!(reported.state, crate::stats::ElementState::Attached);
        assert!(
            reported.idle_for.is_some(),
            "{} did something",
            reported.name
        );
    }
    pipeline.stop();
}

/// A source that has stopped delivering says so, and says for how long —
/// the question a black Preview first needs answered.
#[test]
fn a_source_that_stopped_delivering_is_seen_to_have() {
    let count = Arc::new(AtomicUsize::new(0));
    let pipeline = Pipeline::new(
        "stats-idle",
        BurstSource {
            pp_log: element_pp_log(ElementType::Other, "burst", None),
            pad: SrcPad::new("burst_src"),
            ready: Arc::new(AtomicBool::new(false)),
            buffers: 3,
        },
        |source, ctx| {
            let branch = ctx.branch().to(Box::new(CountingSink {
                pp_log: element_pp_log(ElementType::Other, "end", None),
                name: "end".into(),
                count: count.clone(),
            }))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        },
    )
    .unwrap();
    pipeline.run().unwrap();
    wait_for(|| count.load(Ordering::SeqCst) == 3);

    let before = element(&pipeline.stats(), "burst")
        .idle_for
        .expect("it delivered");
    thread::sleep(Duration::from_millis(50));
    let after = element(&pipeline.stats(), "burst")
        .idle_for
        .expect("it delivered");
    assert!(
        after >= before + Duration::from_millis(40),
        "{before:?} then {after:?}"
    );

    pipeline.pause();
    assert!(
        pipeline.stats().paused,
        "and whether that is because it was paused"
    );
    pipeline.stop();
}

/// A queue told to drop rather than wait says how much it dropped, and
/// counts what arrived at it either way.
#[test]
fn a_queue_that_drops_says_how_much() {
    const BUFFERS: usize = 60;
    let count = Arc::new(AtomicUsize::new(0));
    let pipeline = Pipeline::new(
        "stats-drop",
        BurstSource {
            pp_log: element_pp_log(ElementType::Other, "burst", None),
            pad: SrcPad::new("burst_src"),
            ready: Arc::new(AtomicBool::new(false)),
            buffers: BUFFERS,
        },
        |source, ctx| {
            let branch = ctx
                .branch()
                .queue_with_policy("narrow", 2, crate::queue::OverflowPolicy::DropNewest)
                .to(Box::new(SlowEosSink {
                    pp_log: element_pp_log(ElementType::Other, "slow", None),
                    count: count.clone(),
                    saw_eos: Arc::new(AtomicBool::new(false)),
                }))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        },
    )
    .unwrap();
    pipeline.run().unwrap();
    // Everything is accounted for once what was dropped and what the sink
    // took add up — a queue gone empty can still have its last buffer in
    // the sink's hands.
    wait_for(|| {
        let stats = pipeline.stats();
        element(&stats, "narrow").queue.expect("a queue").dropped
            + element(&stats, "slow-eos").buffers_in
            == BUFFERS as u64
    });

    let stats = pipeline.stats();
    assert_eq!(
        element(&stats, "narrow").buffers_in,
        BUFFERS as u64,
        "every arrival is counted"
    );
    let narrow = element(&stats, "narrow").queue.expect("a queue");
    assert!(
        narrow.dropped > 0,
        "a burst into two slots must drop: {narrow:?}"
    );
    assert_eq!(
        narrow.dropped + element(&stats, "slow-eos").buffers_in,
        BUFFERS as u64,
        "what it dropped and what it delivered are everything it was given"
    );
    pipeline.stop();
}

/// A sink that takes its time over `Eos` — what a muxer writing its trailer
/// looks like from outside.
struct LingeringEosSink {
    pp_log: PpLog,
    linger: Duration,
}

impl Element for LingeringEosSink {
    fn name(&self) -> Arc<str> {
        "lingering".into()
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

impl Sink for LingeringEosSink {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        if buf.is_eos() {
            thread::sleep(self.linger);
        }
        Ok(())
    }

    fn control(&mut self, _msg: ControlMsg) -> Result<()> {
        Ok(())
    }
}

/// A Tee's runtime branches: reported from the attach that publishes them,
/// gone after a detach, and — finished rather than detached — reported as
/// finishing for exactly as long as they are still draining.
#[test]
fn a_runtime_branch_is_reported_while_it_exists_and_no_longer() {
    use crate::stats::ElementState;

    let initial = Arc::new(AtomicUsize::new(0));
    let mut handle_slot = None;
    let pipeline = Pipeline::new(
        "stats-tee",
        TestVideoSource::new("video", TestVideoOptions::default()),
        |source, ctx| {
            let first = ctx.branch().to(Box::new(CountingSink {
                pp_log: element_pp_log(ElementType::Other, "initial", None),
                name: "initial".into(),
                count: initial.clone(),
            }))?;
            let (tee, handle) = TeeBuilder::new("tee", ctx.clone())
                .branch(first)
                .build_dynamic()?;
            ctx.attach(source, 0, tee)?;
            handle_slot = Some(handle);
            Ok(())
        },
    )
    .unwrap();
    let handle = handle_slot.expect("wire ran");
    pipeline.run().unwrap();
    wait_for(|| initial.load(Ordering::SeqCst) > 0);

    // Detached: there, then gone.
    let counted = Arc::new(AtomicUsize::new(0));
    let branch = handle
        .branch()
        .expect("tee alive")
        .to(Box::new(CountingSink {
            pp_log: element_pp_log(ElementType::Other, "dynamic", None),
            name: "dynamic".into(),
            count: counted.clone(),
        }))
        .unwrap();
    let dynamic_id = branch.root_id();
    let branch_id = handle.attach(branch).unwrap();
    let attached = pipeline.stats();
    let dynamic = attached
        .elements
        .iter()
        .find(|element| element.id == dynamic_id)
        .expect("reported from the attach that published it");
    assert_eq!(dynamic.branch, Some(branch_id));
    assert_eq!(dynamic.state, ElementState::Attached);
    wait_for(|| counted.load(Ordering::SeqCst) > 0);
    handle.detach(branch_id).unwrap();
    assert!(
        !pipeline
            .stats()
            .elements
            .iter()
            .any(|element| element.id == dynamic_id),
        "a detached branch is dropped at once, and so is its entry"
    );

    // Finished: draining behind a queue, reported as finishing until it
    // has drained and been dropped.
    let branch = handle
        .branch()
        .expect("tee alive")
        .queue("drain", 4)
        .to(Box::new(LingeringEosSink {
            pp_log: element_pp_log(ElementType::Other, "lingering", None),
            linger: Duration::from_millis(300),
        }))
        .unwrap();
    let finishing_id = branch.root_id();
    let branch_id = handle.attach(branch).unwrap();
    let revision = pipeline.stats().revision;
    handle.finish_branch(branch_id).unwrap();
    let draining = pipeline.stats();
    assert!(
        draining.revision > revision,
        "the finish took it out of the graph"
    );
    assert!(
        draining
            .elements
            .iter()
            .any(|element| element.id == finishing_id && element.state == ElementState::Finishing),
        "still at work, so still reported: {draining:?}"
    );
    wait_for(|| {
        !pipeline
            .stats()
            .elements
            .iter()
            .any(|element| element.id == finishing_id)
    });

    // The same shape attached again is a new element, counted from zero.
    let again = handle
        .branch()
        .expect("tee alive")
        .to(Box::new(CountingSink {
            pp_log: element_pp_log(ElementType::Other, "dynamic", None),
            name: "dynamic".into(),
            count: Arc::new(AtomicUsize::new(0)),
        }))
        .unwrap();
    let again_id = again.root_id();
    handle.attach(again).unwrap();
    assert_ne!(again_id, dynamic_id, "an identity is never given twice");
    let fresh = pipeline.stats();
    let fresh = fresh
        .elements
        .iter()
        .find(|element| element.id == again_id)
        .expect("reported");
    assert!(
        fresh.buffers_in <= 1,
        "counted from its own attach: {fresh:?}"
    );
    pipeline.stop();
}

/// An application that attaches and detaches branches all day and never
/// reads a snapshot must not accumulate an entry for every element it ever
/// attached: the registry forgets dropped ones on every change, not only
/// when read.
#[test]
fn attaching_and_detaching_without_reading_keeps_the_registry_bounded() {
    let mut handle_slot = None;
    let pipeline = Pipeline::new(
        "stats-churn",
        TestVideoSource::new("video", TestVideoOptions::default()),
        |source, ctx| {
            let (tee, handle) = TeeBuilder::new("tee", ctx.clone()).build_dynamic()?;
            ctx.attach(source, 0, tee)?;
            handle_slot = Some(handle);
            Ok(())
        },
    )
    .unwrap();
    let handle = handle_slot.expect("wire ran");
    pipeline.run().unwrap();

    let baseline = pipeline.graph.registered_count();
    for _ in 0..50 {
        let branch = handle
            .branch()
            .expect("tee alive")
            .pipe(PassThrough::new())
            .to(Box::new(NoOpSink {
                name: "churn".into(),
                pp_log: element_pp_log(ElementType::Other, "churn", None),
            }))
            .unwrap();
        let id = handle.attach(branch).unwrap();
        handle.detach(id).unwrap();
    }
    // One detach's own branch may still be held while the lock is released,
    // so what is left is at most the last one's two elements.
    assert!(
        pipeline.graph.registered_count() <= baseline + 2,
        "{} entries after 50 attach-detach cycles, from {baseline}",
        pipeline.graph.registered_count()
    );
    pipeline.stop();
}
