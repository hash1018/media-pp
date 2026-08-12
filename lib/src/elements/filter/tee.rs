use std::sync::{Arc, Mutex, Weak};

use rust_hlog::{HLog, hinfo};

use crate::{
    buffer::MediaBuffer,
    bus::BusEvent,
    control::ControlMsg,
    element::{Context, Element, ElementType, Sink, element_hlog},
    error::Result,
    graph::{BranchId, ElementId, GraphError},
    pad::SrcPad,
    pipeline::{ChainBuilder, DetachedBranch},
};

/// Fans a single input out to a *dynamic* set of sinks — sinks are added
/// and removed through a [`TeeHandle`], which can be cloned and used from
/// any thread, independent of whatever thread is driving `Tee::consume`
/// (the pipeline's source/queue-worker thread). That's the whole reason
/// `Tee` doesn't implement [`crate::element::Source`] like other
/// multi-pad elements (e.g. [`crate::elements::FileDemuxer`]): its pads
/// live behind a lock instead of being a plain `&mut [SrcPad]`, so a
/// handle on another thread can request an add or removal while the
/// pipeline thread is in `consume`. `consume` holds that lock while it
/// visits the current sinks, so the mutation completes after the in-flight
/// buffer has finished fan-out; the new set applies to the next buffer.
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
    branches: Mutex<Vec<TeeBranch>>,
    context: Arc<Context>,
}

struct TeeBranch {
    id: BranchId,
    root_id: ElementId,
    pad: SrcPad,
}

/// A cheaply-cloneable handle for adding or removing a [`Tee`]'s sinks
/// while the pipeline is running. It deliberately keeps only a [`Weak`]
/// reference to the `Tee`'s shared state: retaining a handle after the
/// pipeline finishes must not keep downstream sinks or the pipeline's
/// [`crate::bus::Bus`] sender alive. Once the `Tee` is gone,
/// [`TeeHandle::branch`] returns `None` once the `Tee` is gone.
#[derive(Clone)]
pub struct TeeHandle {
    id: ElementId,
    name: Arc<str>,
    shared: Weak<TeeShared>,
}

impl Tee {
    /// Starts with no sinks — add some via the returned [`TeeHandle`]
    /// before (or any time after) wiring `Tee` itself into the pipeline.
    /// `context` should be the same one [`crate::pipeline::Pipeline::new`]'s
    /// (or, for a multi-source pipeline,
    /// [`crate::pipeline::PipelineBuilder::add_source`]'s) `wire` closure
    /// was handed. The `Tee` keeps that context alive while it belongs to
    /// the pipeline; `TeeHandle` accesses it weakly so the handle itself
    /// cannot extend the pipeline's lifetime.
    pub fn new(name: impl Into<String>, context: Arc<Context>) -> (Self, TeeHandle) {
        let name: Arc<str> = name.into().into();
        let hlog = element_hlog(ElementType::Tee, &name, None);
        hinfo!(hlog: &hlog, "created");
        let id = context.graph.reserve_element_id();
        let shared = Arc::new(TeeShared {
            branches: Mutex::new(Vec::new()),
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
    /// caller watching the bus tell branches apart and call
    /// corresponding runtime branch. Falls back to `Tee`'s
    /// own identity in the same call that just unlinked its peer, though
    /// in practice that can't happen: `pad`'s peer is still there
    /// whenever this is called (`SrcPad::push` never unlinks on `Err`).
    fn report_branch_error(&self, branch: &TeeBranch, error: crate::error::Error) {
        let (element_type, name) = branch
            .pad
            .peer_identity()
            .unwrap_or((ElementType::Tee, self.name.clone()));
        self.shared.context.bus.for_element(branch.root_id).post(
            &self.hlog,
            BusEvent::Error {
                element_type,
                name,
                error,
            },
        );
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

    /// Attaches a detached branch, returning the stable ID used to remove it.
    pub fn attach(&self, branch: DetachedBranch) -> Result<BranchId> {
        let shared = self
            .shared
            .upgrade()
            .ok_or(GraphError::ParentNotAttached(self.id))?;
        let mut branches = shared.branches.lock().unwrap();
        let mut pad = SrcPad::new(format!("{}_src{}", self.name, branches.len()));
        let from_port: Arc<str> = pad.name().into();
        let DetachedBranch { root, plan } = branch;
        let root_id = plan.root;
        let branch_id =
            shared
                .context
                .graph
                .attach_with(self.id, from_port, plan, |branch_id| {
                    pad.link(root);
                    branches.push(TeeBranch {
                        id: branch_id,
                        root_id,
                        pad,
                    });
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
        let mut branches = shared.branches.lock().unwrap();
        let index = branches
            .iter()
            .position(|branch| branch.id == branch_id)
            .ok_or(GraphError::BranchNotAttached(branch_id))?;
        hinfo!(
            hlog: &element_hlog(ElementType::Tee, &self.name, None),
            "sink removed: branch={branch_id}, {} remaining",
            branches.len() - 1
        );
        shared.context.graph.detach_with(branch_id, || {
            let _ = branches.remove(index).pad.unlink();
            Ok(())
        })?;
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
            .map(|shared| shared.branches.lock().unwrap().len())
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
        let mut branches = self.shared.branches.lock().unwrap();
        let Some((last, rest)) = branches.split_last_mut() else {
            return Ok(());
        };
        // One branch failing must not stop the buffer from reaching its
        // siblings — same "errors never kill anything, just get reported"
        // rule `Queue`'s worker loop follows. That buffer is dropped for
        // the failing branch only; the branch itself stays wired and gets
        // retried on the next one. Whoever's watching the bus decides
        // whether to call `TeeHandle::detach` for it.
        for branch in rest {
            if let Err(error) = branch.pad.push(buf.clone()) {
                self.report_branch_error(branch, error);
            }
        }
        if let Err(error) = last.pad.push(buf) {
            self.report_branch_error(last, error);
        }
        Ok(())
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        // Unlike `consume`, every branch gets the same `ControlMsg`
        // value directly (it's `Copy`, no need for the last-one-moves
        // split `consume` does for `MediaBuffer`).
        let mut branches = self.shared.branches.lock().unwrap();
        for branch in branches.iter_mut() {
            branch.pad.control(msg)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

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
        let (tee, handle) = Tee::new("tee", context.clone());
        let tee_branch = context.branch().to(Box::new(tee)).unwrap();
        let mut upstream = SrcPad::new("source_src");
        context.attach_pad(&mut upstream, tee_branch).unwrap();

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
        handle.attach(before).unwrap();
        handle.attach(failing).unwrap();
        handle.attach(after).unwrap();

        for _ in 0..3 {
            upstream
                .push(packet())
                .expect("a branch failing must not surface as an error from Tee::consume");
        }

        assert_eq!(before_count.load(Ordering::SeqCst), 3);
        assert_eq!(after_count.load(Ordering::SeqCst), 3);

        drop(upstream);
        drop(handle);
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
        let (tee, handle) = Tee::new("tee", context.clone());

        drop(context);
        drop(tee);

        assert!(handle.branch().is_none());
        assert_eq!(handle.sink_count(), 0);
        assert!(
            bus_rx.iter().next().is_none(),
            "a retained TeeHandle must not keep the Bus sender alive"
        );
    }
}
