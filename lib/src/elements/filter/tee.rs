use std::sync::{
    Arc, Mutex, MutexGuard, Weak,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

use rust_hlog::{HLog, hinfo};

use crate::{
    buffer::MediaBuffer,
    bus::BusEvent,
    control::ControlMsg,
    element::{Context, Element, ElementType, Sink, element_hlog},
    error::Result,
    graph::{BranchId, ElementId, GraphError, PlannedEdge, PortRef},
    pad::SrcPad,
    pipeline::{ChainBuilder, DetachedBranch},
};

/// Fans a single input out to multiple sinks. [`TeeBuilder`] owns the
/// initial fan-out; later branches are added and removed through a
/// [`TeeHandle`], which can be cloned and used from any thread, independent
/// of whatever thread is driving `Tee::consume`
/// (the pipeline's source/queue-worker thread). That's the whole reason
/// `Tee` doesn't implement [`crate::element::Source`] like other
/// multi-pad elements (e.g. [`crate::elements::FileDemuxer`]): its pads
/// live in individually locked branch slots instead of being a plain
/// `&mut [SrcPad]`. `consume` only holds the branch-list lock long enough
/// to take a cheap `Arc` snapshot, so a slow downstream does not block
/// unrelated attach/detach operations. Detach prevents any push that has
/// not started yet; one already executing downstream call may finish.
///
/// Cheap to fan out: `MediaBuffer` wraps its payload in an `Arc`, so
/// cloning a buffer for each output is a refcount bump, not a copy of the
/// encoded/decoded data.
#[rust_hlog::hlog]
pub struct Tee {
    id: ElementId,
    name: Arc<str>,
    shared: Arc<TeeShared>,
}

struct TeeShared {
    branches: Mutex<Vec<Arc<TeeBranch>>>,
    next_pad_id: AtomicU64,
    context: Arc<Context>,
}

struct TeeBranch {
    id: Option<BranchId>,
    root_id: ElementId,
    active: AtomicBool,
    pad: Mutex<SrcPad>,
}

/// Recovers the protected value after a panic instead of turning one
/// poisoned Tee lock into a permanent source of follow-up panics. The
/// original panic still unwinds normally; this only lets a caller that
/// catches it keep using or detach the remaining branch state.
fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Build-time configuration for a [`Tee`]. Initial branches are merged
/// with the Tee into one detached subgraph and committed by a single
/// [`Context::attach`] call. Use [`TeeBuilder::build`] for a fixed fan-out,
/// or [`TeeBuilder::build_dynamic`] when runtime changes need a
/// [`TeeHandle`].
pub struct TeeBuilder {
    tee: Tee,
    handle: TeeHandle,
    initial_branches: Vec<DetachedBranch>,
}

/// A cheaply-cloneable handle for adding or removing a [`Tee`]'s sinks
/// while the pipeline is running. It deliberately keeps only a [`Weak`]
/// reference to the `Tee`'s shared state: retaining a handle after the
/// pipeline finishes must not keep downstream sinks or the pipeline's
/// [`crate::bus::Bus`] sender alive. Once the `Tee` is gone,
/// [`TeeHandle::branch`] returns `None`.
#[derive(Clone)]
pub struct TeeHandle {
    id: ElementId,
    name: Arc<str>,
    shared: Weak<TeeShared>,
}

impl Tee {
    fn new(name: impl Into<String>, context: Arc<Context>) -> (Self, TeeHandle) {
        let name: Arc<str> = name.into().into();
        let hlog = element_hlog(ElementType::Tee, &name, None);
        hinfo!(hlog: &hlog, "created");
        let id = context.graph.reserve_element_id();
        let shared = Arc::new(TeeShared {
            branches: Mutex::new(Vec::new()),
            next_pad_id: AtomicU64::new(0),
            context,
        });
        (
            Self {
                id,
                name: name.clone(),
                hlog,
                shared: shared.clone(),
            },
            TeeHandle {
                id,
                name,
                shared: Arc::downgrade(&shared),
            },
        )
    }

    /// Posts a branch's `push` failure to the bus under *that branch's*
    /// own identity (via [`SrcPad::peer_identity`]) — unlike `Queue`,
    /// which only ever has one downstream and so can only attribute a
    /// failure to itself, `Tee` fans out to several and does know which
    /// one just failed. Reporting it that way (rather than folding every
    /// branch's failures into one generic `Tee` event) is what lets a
    /// caller watching the bus tell branches apart and identify the
    /// corresponding runtime branch. The peer identity is captured just
    /// before the downstream call, so the event stays attributable even
    /// if that branch is detached while the call is running.
    fn report_branch_error(
        &self,
        root_id: ElementId,
        peer: Option<(ElementType, Arc<str>)>,
        error: crate::error::Error,
    ) {
        let (element_type, name) = peer.unwrap_or((ElementType::Tee, self.name.clone()));
        self.shared.context.bus.for_element(root_id).post(
            &self.hlog,
            BusEvent::Error {
                element_type,
                name,
                error,
            },
        );
    }
}

impl TeeShared {
    fn next_pad(&self, tee_name: &str) -> SrcPad {
        let id = self.next_pad_id.fetch_add(1, Ordering::Relaxed);
        SrcPad::new(format!("{tee_name}_src{id}"))
    }
}

impl TeeBuilder {
    /// Starts an initially empty Tee in the supplied pipeline context.
    pub fn new(name: impl Into<String>, context: Arc<Context>) -> Self {
        let (tee, handle) = Tee::new(name, context);
        Self {
            tee,
            handle,
            initial_branches: Vec::new(),
        }
    }

    /// Adds one fixed initial branch to this fan-out subgraph.
    pub fn branch(mut self, branch: DetachedBranch) -> Self {
        self.initial_branches.push(branch);
        self
    }

    /// Returns the complete fixed fan-out without exposing runtime control.
    pub fn build(self) -> Result<DetachedBranch> {
        self.finish().map(|(branch, _handle)| branch)
    }

    /// Returns the initial subgraph together with its runtime control handle.
    /// Attach the branch through [`Context::attach`] before using the handle.
    pub fn build_dynamic(self) -> Result<(DetachedBranch, TeeHandle)> {
        self.finish()
    }

    fn finish(self) -> Result<(DetachedBranch, TeeHandle)> {
        let Self {
            tee,
            handle,
            initial_branches,
        } = self;
        let tee_id = tee.id;
        let tee_name = tee.name.clone();
        let shared = tee.shared.clone();
        let context = shared.context.clone();
        let mut tee_branch = context.branch().to(Box::new(tee))?;
        let mut runtime_branches = lock_unpoisoned(&shared.branches);

        for branch in initial_branches {
            let mut pad = shared.next_pad(&tee_name);
            let from_port: Arc<str> = pad.name().into();
            let DetachedBranch { root, plan } = branch;
            let root_id = plan.root;

            tee_branch.plan.edges.push(PlannedEdge {
                from: PortRef {
                    element: tee_id,
                    port: from_port,
                },
                to: PortRef {
                    element: root_id,
                    port: "sink".into(),
                },
            });
            tee_branch.plan.nodes.extend(plan.nodes);
            tee_branch.plan.edges.extend(plan.edges);

            pad.link(root);
            runtime_branches.push(Arc::new(TeeBranch {
                id: None,
                root_id,
                active: AtomicBool::new(true),
                pad: Mutex::new(pad),
            }));
        }
        drop(runtime_branches);

        Ok((tee_branch, handle))
    }
}

impl TeeHandle {
    /// A [`crate::pipeline::ChainBuilder`] pre-wired with this `Tee`'s own
    /// [`Context`] — lets a caller build a whole new branch (`.pipe(...)`
    /// chains, ending in `.to(...)`) at any point after the pipeline
    /// started running, then hand the result to [`TeeHandle::attach`],
    /// without needing to retain the pipeline context separately
    /// around separately. Returns `None` once the `Tee` has been dropped.
    pub fn branch(&self) -> Option<ChainBuilder> {
        let shared = self.shared.upgrade()?;
        Some(shared.context.branch())
    }

    /// Attaches a runtime branch, returning the stable ID used to remove it.
    /// Fixed initial branches belong in [`TeeBuilder`].
    pub fn attach(&self, branch: DetachedBranch) -> Result<BranchId> {
        let shared = self
            .shared
            .upgrade()
            .ok_or(GraphError::ParentNotAttached(self.id))?;
        let mut branches = lock_unpoisoned(&shared.branches);
        let mut pad = shared.next_pad(&self.name);
        let from_port: Arc<str> = pad.name().into();
        let DetachedBranch { root, plan } = branch;
        let root_id = plan.root;
        let branch_id =
            shared
                .context
                .graph
                .attach_with(self.id, from_port, plan, |branch_id| {
                    pad.link(root);
                    branches.push(Arc::new(TeeBranch {
                        id: Some(branch_id),
                        root_id,
                        active: AtomicBool::new(true),
                        pad: Mutex::new(pad),
                    }));
                    Ok(())
                })?;
        hinfo!(
            hlog: &element_hlog(ElementType::Tee, &self.name, None),
            "sink added: {} total",
            branches.len()
        );
        Ok(branch_id)
    }

    /// Detaches exactly the branch returned by [`TeeHandle::attach`]. The
    /// runtime peer and every graph node owned by it disappear in the same
    /// transaction. Names are deliberately not used as graph keys.
    pub fn detach(&self, branch_id: BranchId) -> Result<()> {
        let shared = self
            .shared
            .upgrade()
            .ok_or(GraphError::BranchNotAttached(branch_id))?;
        let mut branches = lock_unpoisoned(&shared.branches);
        let index = branches
            .iter()
            .position(|branch| branch.id == Some(branch_id))
            .ok_or(GraphError::BranchNotAttached(branch_id))?;
        hinfo!(
            hlog: &element_hlog(ElementType::Tee, &self.name, None),
            "sink removed: branch={branch_id}, {} remaining",
            branches.len() - 1
        );
        let mut removed = None;
        shared.context.graph.detach_with(branch_id, || {
            let branch = branches.remove(index);
            branch.active.store(false, Ordering::Release);
            removed = Some(branch);
            Ok(())
        })?;
        drop(branches);
        // The last Arc owns the downstream sink. Dropping it outside both
        // the branch-list and graph locks allows arbitrary Sink::drop code
        // to inspect the graph or call back into Tee without deadlocking.
        drop(removed);
        Ok(())
    }

    /// Resolves the owning branch from any element ID inside it and
    /// detaches that branch. Useful when an error is attributed to a stage
    /// behind a queue rather than to the branch root.
    pub fn detach_branch_containing(&self, element: ElementId) -> Result<()> {
        let shared = self
            .shared
            .upgrade()
            .ok_or(GraphError::ParentNotAttached(self.id))?;
        let branch_id = shared
            .context
            .graph
            .branch_containing(element)
            .ok_or(GraphError::ParentNotAttached(element))?;
        self.detach(branch_id)
    }

    pub fn sink_count(&self) -> usize {
        self.shared
            .upgrade()
            .map(|shared| lock_unpoisoned(&shared.branches).len())
            .unwrap_or(0)
    }
}

impl Element for Tee {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::Tee
    }

    fn graph_id(&self) -> Option<ElementId> {
        Some(self.id)
    }

    fn hlog(&self) -> &HLog {
        &self.hlog
    }

    fn hlog_mut(&mut self) -> &mut HLog {
        &mut self.hlog
    }
}

impl Sink for Tee {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        let branches = lock_unpoisoned(&self.shared.branches).clone();
        // One branch failing must not stop the buffer from reaching its
        // siblings — same "errors never kill anything, just get reported"
        // rule `Queue`'s worker loop follows. That buffer is dropped for
        // the failing branch only; the branch itself stays wired and gets
        // retried on the next one. Whoever's watching the bus decides
        // whether to call `TeeHandle::detach` for it.
        for branch in branches {
            if !branch.active.load(Ordering::Acquire) {
                continue;
            }
            let mut pad = lock_unpoisoned(&branch.pad);
            // Detach may have won the race while this thread waited for a
            // previous push on the same branch to finish.
            if !branch.active.load(Ordering::Acquire) {
                continue;
            }
            let peer = pad.peer_identity();
            let outcome = pad.push(buf.clone());
            drop(pad);
            if let Err(error) = outcome {
                self.report_branch_error(branch.root_id, peer, error);
            }
        }
        Ok(())
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        // Control failures follow the same isolation rule as data failures:
        // report the failed branch, but still deliver the message to every
        // sibling. This is especially important for Stop and Pause.
        let branches = lock_unpoisoned(&self.shared.branches).clone();
        for branch in branches {
            if !branch.active.load(Ordering::Acquire) {
                continue;
            }
            let mut pad = lock_unpoisoned(&branch.pad);
            if !branch.active.load(Ordering::Acquire) {
                continue;
            }
            let peer = pad.peer_identity();
            let outcome = pad.control(msg);
            drop(pad);
            if let Err(error) = outcome {
                self.report_branch_error(branch.root_id, peer, error);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        panic::{AssertUnwindSafe, catch_unwind},
        sync::{
            Barrier,
            atomic::{AtomicBool, AtomicUsize, Ordering},
            mpsc,
        },
        thread,
        time::Duration,
    };

    use super::*;
    use crate::{bus::Bus, clock::Clock, graph::PipelineGraph};

    fn packet() -> MediaBuffer {
        MediaBuffer::Packet(Arc::new(ffmpeg_next::Packet::empty()))
    }

    #[rust_hlog::hlog]
    struct CountingSink {
        name: &'static str,
        count: Arc<AtomicUsize>,
    }

    impl Element for CountingSink {
        fn name(&self) -> Arc<str> {
            self.name.into()
        }

        fn element_type(&self) -> ElementType {
            ElementType::Other
        }

        fn hlog(&self) -> &HLog {
            &self.hlog
        }

        fn hlog_mut(&mut self) -> &mut HLog {
            &mut self.hlog
        }
    }

    impl Sink for CountingSink {
        fn consume(&mut self, _buf: MediaBuffer) -> Result<()> {
            self.count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn control(&mut self, _msg: ControlMsg) -> Result<()> {
            Ok(())
        }
    }

    #[rust_hlog::hlog]
    struct AlwaysFailSink {}

    impl Element for AlwaysFailSink {
        fn name(&self) -> Arc<str> {
            "always-fail".into()
        }

        fn element_type(&self) -> ElementType {
            ElementType::Other
        }

        fn hlog(&self) -> &HLog {
            &self.hlog
        }

        fn hlog_mut(&mut self) -> &mut HLog {
            &mut self.hlog
        }
    }

    impl Sink for AlwaysFailSink {
        fn consume(&mut self, _buf: MediaBuffer) -> Result<()> {
            Err(crate::error::Error::Other(
                "simulated branch failure".into(),
            ))
        }

        fn control(&mut self, _msg: ControlMsg) -> Result<()> {
            Ok(())
        }
    }

    #[rust_hlog::hlog]
    struct ControlObservingSink {
        name: &'static str,
        count: Arc<AtomicUsize>,
        fail: bool,
    }

    impl Element for ControlObservingSink {
        fn name(&self) -> Arc<str> {
            self.name.into()
        }

        fn element_type(&self) -> ElementType {
            ElementType::Other
        }

        fn hlog(&self) -> &HLog {
            &self.hlog
        }

        fn hlog_mut(&mut self) -> &mut HLog {
            &mut self.hlog
        }
    }

    impl Sink for ControlObservingSink {
        fn consume(&mut self, _buf: MediaBuffer) -> Result<()> {
            Ok(())
        }

        fn control(&mut self, _msg: ControlMsg) -> Result<()> {
            self.count.fetch_add(1, Ordering::SeqCst);
            if self.fail {
                Err(crate::error::Error::Other(
                    "simulated control failure".into(),
                ))
            } else {
                Ok(())
            }
        }
    }

    #[rust_hlog::hlog]
    struct PanicOnceSink {
        panicked: bool,
        successful: Arc<AtomicUsize>,
    }

    impl Element for PanicOnceSink {
        fn name(&self) -> Arc<str> {
            "panic-once".into()
        }

        fn element_type(&self) -> ElementType {
            ElementType::Other
        }

        fn hlog(&self) -> &HLog {
            &self.hlog
        }

        fn hlog_mut(&mut self) -> &mut HLog {
            &mut self.hlog
        }
    }

    impl Sink for PanicOnceSink {
        fn consume(&mut self, _buf: MediaBuffer) -> Result<()> {
            if !self.panicked {
                self.panicked = true;
                panic!("simulated downstream panic");
            }
            self.successful.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn control(&mut self, _msg: ControlMsg) -> Result<()> {
            Ok(())
        }
    }

    #[rust_hlog::hlog]
    struct BlockingSink {
        entered: Option<mpsc::Sender<()>>,
        release: mpsc::Receiver<()>,
    }

    impl Element for BlockingSink {
        fn name(&self) -> Arc<str> {
            "blocking".into()
        }

        fn element_type(&self) -> ElementType {
            ElementType::Other
        }

        fn hlog(&self) -> &HLog {
            &self.hlog
        }

        fn hlog_mut(&mut self) -> &mut HLog {
            &mut self.hlog
        }
    }

    impl Sink for BlockingSink {
        fn consume(&mut self, _buf: MediaBuffer) -> Result<()> {
            if let Some(entered) = self.entered.take() {
                let _ = entered.send(());
            }
            let _ = self.release.recv();
            Ok(())
        }

        fn control(&mut self, _msg: ControlMsg) -> Result<()> {
            Ok(())
        }
    }

    #[rust_hlog::hlog]
    struct GraphInspectingDropSink {
        graph: PipelineGraph,
        dropped: Option<mpsc::Sender<()>>,
    }

    impl Element for GraphInspectingDropSink {
        fn name(&self) -> Arc<str> {
            "graph-inspecting-drop".into()
        }

        fn element_type(&self) -> ElementType {
            ElementType::Other
        }

        fn hlog(&self) -> &HLog {
            &self.hlog
        }

        fn hlog_mut(&mut self) -> &mut HLog {
            &mut self.hlog
        }
    }

    impl Sink for GraphInspectingDropSink {
        fn consume(&mut self, _buf: MediaBuffer) -> Result<()> {
            Ok(())
        }

        fn control(&mut self, _msg: ControlMsg) -> Result<()> {
            Ok(())
        }
    }

    impl Drop for GraphInspectingDropSink {
        fn drop(&mut self) {
            let _ = self.graph.snapshot();
            if let Some(dropped) = self.dropped.take() {
                let _ = dropped.send(());
            }
        }
    }

    /// A failing branch must not stop the same buffer from reaching its
    /// siblings, nor stop `Tee::consume` itself from returning `Ok` — same
    /// "errors get reported, nothing dies" rule `Queue`'s worker loop
    /// follows (see `queue::tests::a_failing_consume_drops_that_buffer_but_keeps_the_worker_alive`).
    /// Wires the failing branch in the *middle* (`before`/`after` on
    /// either side of it) so the test also proves a mid-`rest` failure
    /// doesn't short-circuit the fan-out to what comes after it,
    /// including `last`.
    #[test]
    fn a_failing_branch_does_not_block_its_siblings() {
        let (bus, bus_rx) = Bus::new();
        let graph = PipelineGraph::new();
        let source_id = graph.add_source(ElementType::Other, "source".into());
        let context = Arc::new(Context {
            bus,
            pipeline_id: "test".into(),
            graph,
            clock: Arc::new(Clock::new()),
            source_id,
        });
        let before_count = Arc::new(AtomicUsize::new(0));
        let after_count = Arc::new(AtomicUsize::new(0));
        let before = context
            .branch()
            .to(Box::new(CountingSink {
                name: "before",
                count: before_count.clone(),
                hlog: element_hlog(ElementType::Other, "before", None),
            }))
            .unwrap();
        let failing = context
            .branch()
            .to(Box::new(AlwaysFailSink {
                hlog: element_hlog(ElementType::Other, "always-fail", None),
            }))
            .unwrap();
        let after = context
            .branch()
            .to(Box::new(CountingSink {
                name: "after",
                count: after_count.clone(),
                hlog: element_hlog(ElementType::Other, "after", None),
            }))
            .unwrap();
        let tee_branch = TeeBuilder::new("tee", context.clone())
            .branch(before)
            .branch(failing)
            .branch(after)
            .build()
            .unwrap();
        let mut upstream = SrcPad::new("source_src");
        context.attach_pad(&mut upstream, tee_branch).unwrap();

        for _ in 0..3 {
            upstream
                .push(packet())
                .expect("a branch failing must not surface as an error from Tee::consume");
        }

        assert_eq!(before_count.load(Ordering::SeqCst), 3);
        assert_eq!(after_count.load(Ordering::SeqCst), 3);

        drop(upstream);
        drop(context);
        let errors: Vec<_> = bus_rx
            .iter()
            .filter(|e| matches!(e, BusEvent::Error { .. }))
            .collect();
        assert_eq!(
            errors.len(),
            3,
            "expected one Error event per failed push, not a fatal short-circuit"
        );
        assert!(
            errors.iter().all(|e| matches!(
                e,
                BusEvent::Error { name, .. } if &**name == "always-fail"
            )),
            "each Error event should be attributed to the branch that actually \
             failed, not to Tee itself — that's what lets a caller call \
             TeeHandle::detach(branch_id) straight off the bus; got {errors:?}"
        );
    }

    #[test]
    fn a_failing_control_branch_does_not_block_its_siblings() {
        let (bus, bus_rx) = Bus::new();
        let graph = PipelineGraph::new();
        let source_id = graph.add_source(ElementType::Other, "source".into());
        let context = Arc::new(Context {
            bus,
            pipeline_id: "test".into(),
            graph,
            clock: Arc::new(Clock::new()),
            source_id,
        });
        let failing_count = Arc::new(AtomicUsize::new(0));
        let healthy_count = Arc::new(AtomicUsize::new(0));
        let failing = context
            .branch()
            .to(Box::new(ControlObservingSink {
                name: "control-fail",
                count: failing_count.clone(),
                fail: true,
                hlog: element_hlog(ElementType::Other, "control-fail", None),
            }))
            .unwrap();
        let healthy = context
            .branch()
            .to(Box::new(ControlObservingSink {
                name: "control-ok",
                count: healthy_count.clone(),
                fail: false,
                hlog: element_hlog(ElementType::Other, "control-ok", None),
            }))
            .unwrap();
        let tee_branch = TeeBuilder::new("tee", context.clone())
            .branch(failing)
            .branch(healthy)
            .build()
            .unwrap();
        let mut upstream = SrcPad::new("source_src");
        context.attach_pad(&mut upstream, tee_branch).unwrap();

        upstream
            .control(ControlMsg::Pause)
            .expect("a branch control failure should be reported, not short-circuit Tee");

        assert_eq!(failing_count.load(Ordering::SeqCst), 1);
        assert_eq!(healthy_count.load(Ordering::SeqCst), 1);
        let message = bus_rx
            .try_recv_message()
            .expect("the failing control branch should post an Error event");
        assert!(matches!(
            message.event,
            BusEvent::Error { name, .. } if &*name == "control-fail"
        ));
        assert!(bus_rx.try_recv_message().is_none());
    }

    #[test]
    fn a_poisoned_branch_pad_can_be_used_after_the_panic_is_caught() {
        let (bus, _bus_rx) = Bus::new();
        let graph = PipelineGraph::new();
        let source_id = graph.add_source(ElementType::Other, "source".into());
        let context = Arc::new(Context {
            bus,
            pipeline_id: "test".into(),
            graph,
            clock: Arc::new(Clock::new()),
            source_id,
        });
        let successful = Arc::new(AtomicUsize::new(0));
        let panic_once = context
            .branch()
            .to(Box::new(PanicOnceSink {
                panicked: false,
                successful: successful.clone(),
                hlog: element_hlog(ElementType::Other, "panic-once", None),
            }))
            .unwrap();
        let tee_branch = TeeBuilder::new("tee", context.clone())
            .branch(panic_once)
            .build()
            .unwrap();
        let mut upstream = SrcPad::new("source_src");
        context.attach_pad(&mut upstream, tee_branch).unwrap();

        let first = catch_unwind(AssertUnwindSafe(|| upstream.push(packet())));
        assert!(
            first.is_err(),
            "the original downstream panic must propagate"
        );
        upstream
            .push(packet())
            .expect("the poisoned branch pad should be recovered on the next push");
        assert_eq!(successful.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_poisoned_branch_list_does_not_break_attach_or_detach() {
        let (bus, _bus_rx) = Bus::new();
        let graph = PipelineGraph::new();
        let source_id = graph.add_source(ElementType::Other, "source".into());
        let context = Arc::new(Context {
            bus,
            pipeline_id: "test".into(),
            graph,
            clock: Arc::new(Clock::new()),
            source_id,
        });
        let (tee_branch, handle) = TeeBuilder::new("tee", context.clone())
            .build_dynamic()
            .unwrap();
        let mut upstream = SrcPad::new("source_src");
        context.attach_pad(&mut upstream, tee_branch).unwrap();
        let shared = handle.shared.upgrade().unwrap();

        let poisoned = catch_unwind(AssertUnwindSafe(|| {
            let _branches = shared.branches.lock().unwrap();
            panic!("poison the branch-list lock");
        }));
        assert!(poisoned.is_err());
        assert_eq!(handle.sink_count(), 0);

        let branch = handle
            .branch()
            .unwrap()
            .to(Box::new(CountingSink {
                name: "after-poison",
                count: Arc::new(AtomicUsize::new(0)),
                hlog: element_hlog(ElementType::Other, "after-poison", None),
            }))
            .unwrap();
        let branch_id = handle.attach(branch).unwrap();
        assert_eq!(handle.sink_count(), 1);
        handle.detach(branch_id).unwrap();
        assert_eq!(handle.sink_count(), 0);
    }

    #[test]
    fn blocked_downstream_does_not_block_unrelated_attach_or_detach() {
        let (bus, _bus_rx) = Bus::new();
        let graph = PipelineGraph::new();
        let source_id = graph.add_source(ElementType::Other, "source".into());
        let context = Arc::new(Context {
            bus,
            pipeline_id: "test".into(),
            graph,
            clock: Arc::new(Clock::new()),
            source_id,
        });
        let (tee_branch, handle) = TeeBuilder::new("tee", context.clone())
            .build_dynamic()
            .unwrap();
        let mut upstream = SrcPad::new("source_src");
        context.attach_pad(&mut upstream, tee_branch).unwrap();

        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let blocking = handle
            .branch()
            .unwrap()
            .to(Box::new(BlockingSink {
                entered: Some(entered_tx),
                release: release_rx,
                hlog: element_hlog(ElementType::Other, "blocking", None),
            }))
            .unwrap();
        let blocking_id = handle.attach(blocking).unwrap();

        let push_thread = thread::spawn(move || upstream.push(packet()));
        entered_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("blocking branch was never entered");

        let new_branch = handle
            .branch()
            .unwrap()
            .to(Box::new(CountingSink {
                name: "new",
                count: Arc::new(AtomicUsize::new(0)),
                hlog: element_hlog(ElementType::Other, "new", None),
            }))
            .unwrap();
        let (attach_tx, attach_rx) = mpsc::channel();
        let attach_handle = handle.clone();
        let attach_thread = thread::spawn(move || {
            let _ = attach_tx.send(attach_handle.attach(new_branch));
        });

        let (detach_tx, detach_rx) = mpsc::channel();
        let detach_handle = handle.clone();
        let detach_thread = thread::spawn(move || {
            let _ = detach_tx.send(detach_handle.detach(blocking_id));
        });

        let attach_before_release = attach_rx.recv_timeout(Duration::from_millis(250)).ok();
        let detach_before_release = detach_rx.recv_timeout(Duration::from_millis(250)).ok();
        let attach_completed_while_blocked = attach_before_release.is_some();
        let detach_completed_while_blocked = detach_before_release.is_some();
        let _ = release_tx.send(());

        push_thread.join().unwrap().unwrap();
        attach_thread.join().unwrap();
        detach_thread.join().unwrap();
        let attach_result = attach_before_release
            .unwrap_or_else(|| attach_rx.recv_timeout(Duration::from_secs(1)).unwrap());
        let detach_result = detach_before_release
            .unwrap_or_else(|| detach_rx.recv_timeout(Duration::from_secs(1)).unwrap());
        attach_result.unwrap();
        detach_result.unwrap();

        assert!(
            attach_completed_while_blocked,
            "an unrelated attach waited for the blocked downstream"
        );
        assert!(
            detach_completed_while_blocked,
            "detach waited for an already-running downstream call"
        );
    }

    #[test]
    fn concurrent_push_attach_and_detach_stays_consistent_under_stress() {
        const MIN_PUSHES: usize = 10_000;
        const MUTATIONS: usize = 500;

        let (bus, _bus_rx) = Bus::new();
        let graph = PipelineGraph::new();
        let source_id = graph.add_source(ElementType::Other, "source".into());
        let context = Arc::new(Context {
            bus,
            pipeline_id: "test".into(),
            graph,
            clock: Arc::new(Clock::new()),
            source_id,
        });
        let initial_count = Arc::new(AtomicUsize::new(0));
        let initial = context
            .branch()
            .to(Box::new(CountingSink {
                name: "initial",
                count: initial_count.clone(),
                hlog: element_hlog(ElementType::Other, "initial", None),
            }))
            .unwrap();
        let (tee_branch, handle) = TeeBuilder::new("tee", context.clone())
            .branch(initial)
            .build_dynamic()
            .unwrap();
        let mut upstream = SrcPad::new("source_src");
        context.attach_pad(&mut upstream, tee_branch).unwrap();

        let start = Arc::new(Barrier::new(2));
        let mutating = Arc::new(AtomicBool::new(true));
        let (push_done_tx, push_done_rx) = mpsc::channel();
        let push_start = start.clone();
        let push_mutating = mutating.clone();
        let push_thread = thread::spawn(move || {
            push_start.wait();
            let packet = packet();
            let mut pushed = 0;
            let mut outcome = Ok(());
            while push_mutating.load(Ordering::Acquire) || pushed < MIN_PUSHES {
                if let Err(error) = upstream.push(packet.clone()) {
                    outcome = Err(error.to_string());
                    break;
                }
                pushed += 1;
                if pushed % 32 == 0 {
                    thread::yield_now();
                }
            }
            let _ = push_done_tx.send((upstream, outcome, pushed));
        });

        let (mutation_done_tx, mutation_done_rx) = mpsc::channel();
        let mutation_start = start;
        let mutation_handle = handle.clone();
        let mutation_thread = thread::spawn(move || {
            mutation_start.wait();
            let outcome = (|| -> std::result::Result<(), String> {
                for _ in 0..MUTATIONS {
                    let branch = mutation_handle
                        .branch()
                        .ok_or_else(|| "Tee disappeared during stress test".to_owned())?
                        .to(Box::new(CountingSink {
                            name: "dynamic",
                            count: Arc::new(AtomicUsize::new(0)),
                            hlog: element_hlog(ElementType::Other, "dynamic", None),
                        }))
                        .map_err(|error| error.to_string())?;
                    let branch_id = mutation_handle
                        .attach(branch)
                        .map_err(|error| error.to_string())?;
                    thread::yield_now();
                    mutation_handle
                        .detach(branch_id)
                        .map_err(|error| error.to_string())?;
                }
                Ok(())
            })();
            mutating.store(false, Ordering::Release);
            let _ = mutation_done_tx.send(outcome);
        });

        mutation_done_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("attach/detach stress thread timed out")
            .expect("attach/detach stress thread failed");
        let (upstream, push_outcome, pushed) = push_done_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("push stress thread timed out");
        push_outcome.expect("push stress thread failed");
        push_thread.join().unwrap();
        mutation_thread.join().unwrap();

        assert!(pushed >= MIN_PUSHES);
        assert_eq!(initial_count.load(Ordering::SeqCst), pushed);
        assert_eq!(handle.sink_count(), 1);
        let graph = context.graph.snapshot();
        assert_eq!(graph.nodes.len(), 3);
        assert_eq!(graph.edges.len(), 2);
        assert_eq!(graph.revision, 2 + (MUTATIONS as u64 * 2));
        drop(upstream);
    }

    #[test]
    fn detached_sink_is_dropped_outside_the_graph_lock() {
        let (bus, _bus_rx) = Bus::new();
        let graph = PipelineGraph::new();
        let source_id = graph.add_source(ElementType::Other, "source".into());
        let context = Arc::new(Context {
            bus,
            pipeline_id: "test".into(),
            graph: graph.clone(),
            clock: Arc::new(Clock::new()),
            source_id,
        });
        let (tee_branch, handle) = TeeBuilder::new("tee", context.clone())
            .build_dynamic()
            .unwrap();
        let mut upstream = SrcPad::new("source_src");
        context.attach_pad(&mut upstream, tee_branch).unwrap();

        let (dropped_tx, dropped_rx) = mpsc::channel();
        let branch = handle
            .branch()
            .unwrap()
            .to(Box::new(GraphInspectingDropSink {
                graph,
                dropped: Some(dropped_tx),
                hlog: element_hlog(ElementType::Other, "graph-inspecting-drop", None),
            }))
            .unwrap();
        let branch_id = handle.attach(branch).unwrap();
        let (done_tx, done_rx) = mpsc::channel();
        let detach_thread = thread::spawn(move || {
            let _ = done_tx.send(handle.detach(branch_id));
        });

        dropped_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("sink Drop deadlocked while inspecting the graph");
        done_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("detach did not finish")
            .unwrap();
        detach_thread.join().unwrap();
        drop(upstream);
    }

    #[test]
    fn retained_handle_does_not_keep_tee_context_or_bus_alive() {
        let (bus, bus_rx) = Bus::new();
        let graph = PipelineGraph::new();
        let source_id = graph.add_source(ElementType::Other, "source".into());
        let context = Arc::new(Context {
            bus,
            pipeline_id: "test".into(),
            graph,
            clock: Arc::new(Clock::new()),
            source_id,
        });
        let (tee_branch, handle) = TeeBuilder::new("tee", context.clone())
            .build_dynamic()
            .unwrap();

        drop(context);
        drop(tee_branch);

        assert!(handle.branch().is_none());
        assert_eq!(handle.sink_count(), 0);
        assert!(
            bus_rx.iter().next().is_none(),
            "a retained TeeHandle must not keep the Bus sender alive"
        );
    }
}
