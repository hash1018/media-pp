use std::{collections::HashMap, sync::Arc};

use crate::pp_log::{PpLog, pp_trace};

use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    contract::{InputContract, OutputContract},
    control::ControlMsg,
    element::{Context, Element, ElementType, Filter, Sink, Source, element_pp_log},
    error::Result,
    graph::{
        BranchId, BranchPlan, ElementId, GraphError, Incoming, NodeInfo, PlannedEdge,
        PortContracts, PortRef, ResolvedFlow,
    },
    pad::SrcPad,
    queue::{OverflowPolicy, Queue},
};

/// Builds one chain segment (a run of elements that all execute on the same
/// thread). Call [`ChainBuilder::queue`] to close the current segment behind
/// a `Queue` and start a new one on its own worker thread.
///
/// Because each element needs a handle to *its* downstream to be
/// constructed, the chain is assembled back-to-front: elements are
/// collected in call order, then folded right-to-left starting from the
/// terminal `Sink` at [`ChainBuilder::to`] time.
pub struct ChainBuilder {
    context: Arc<Context>,
    elements: Vec<Box<dyn StageBuilder>>,
    /// Nodes kept locally until this builder becomes a `DetachedBranch`
    /// and an attach operation commits the complete plan.
    planned: Vec<PlannedNode>,
    error: Option<GraphError>,
}

struct PlannedNode {
    info: NodeInfo,
    output_port: Arc<str>,
    input: InputContract,
    output: OutputContract,
}

/// A fully constructed runtime chain whose graph nodes are still detached.
/// Dropping it has no topology effect; only an attach operation commits it.
pub struct DetachedBranch {
    pub(crate) root: Box<dyn Sink>,
    pub(crate) plan: BranchPlan,
}

impl DetachedBranch {
    /// Returns the stable graph identity of the first element in this branch.
    ///
    /// The ID is reserved during construction but does not appear in the live
    /// graph until a successful [`Context::attach`].
    pub fn root_id(&self) -> ElementId {
        self.plan.root
    }
}

trait StageBuilder: Send {
    fn wrap(
        self: Box<Self>,
        downstream: Box<dyn Sink>,
        bus: &Bus,
        pipeline_id: &str,
    ) -> Result<Box<dyn Sink>>;
}

struct DirectStage<T>(T);

/// Adds uniform EOS/control boundary tracing to every direct filter without
/// requiring each built-in or downstream custom element to duplicate it.
struct FlowTracer<T> {
    inner: T,
}

impl<T: Element> Element for FlowTracer<T> {
    fn name(&self) -> Arc<str> {
        self.inner.name()
    }

    fn element_type(&self) -> ElementType {
        self.inner.element_type()
    }

    fn graph_id(&self) -> Option<ElementId> {
        self.inner.graph_id()
    }

    fn pp_log(&self) -> &PpLog {
        self.inner.pp_log()
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        self.inner.pp_log_mut()
    }
}

impl<T: Source> Source for FlowTracer<T> {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        self.inner.src_pads()
    }
}

impl<T: Filter> Sink for FlowTracer<T> {
    fn ready_consume(&mut self) -> bool {
        self.inner.ready_consume() && self.inner.src_pads().iter_mut().all(SrcPad::ready_consume)
    }

    fn input_contract(&self) -> InputContract {
        self.inner.input_contract()
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        let is_eos = buf.is_eos();
        if is_eos {
            pp_trace!(pp_log: self.inner.pp_log(), "event=eos phase=received");
        }
        let result = self.inner.consume(buf);
        if is_eos {
            match &result {
                Ok(()) => pp_trace!(
                    pp_log: self.inner.pp_log(),
                    "event=eos phase=completed outcome=ok"
                ),
                Err(error) => pp_trace!(
                    pp_log: self.inner.pp_log(),
                    "event=eos phase=completed outcome=error error={error}"
                ),
            }
        }
        result
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        pp_trace!(
            pp_log: self.inner.pp_log(),
            "event=control control={msg:?} phase=received"
        );
        let result = self.inner.control(msg.clone());
        match &result {
            Ok(()) => pp_trace!(
                pp_log: self.inner.pp_log(),
                "event=control control={msg:?} phase=completed outcome=ok"
            ),
            Err(error) => pp_trace!(
                pp_log: self.inner.pp_log(),
                "event=control control={msg:?} phase=completed outcome=error error={error}"
            ),
        }
        result
    }
}

impl<T> StageBuilder for DirectStage<T>
where
    T: Filter + 'static,
{
    fn wrap(
        self: Box<Self>,
        downstream: Box<dyn Sink>,
        _bus: &Bus,
        pipeline_id: &str,
    ) -> Result<Box<dyn Sink>> {
        let mut element = self.0;
        *element.pp_log_mut() =
            element_pp_log(element.element_type(), &element.name(), Some(pipeline_id));
        element.src_pads()[0].link(downstream);
        Ok(Box::new(FlowTracer { inner: element }))
    }
}

struct QueueStage {
    id: ElementId,
    name: String,
    capacity: usize,
    policy: OverflowPolicy,
}

/// Traces EOS/control at a terminal `Sink` and posts a `BusEvent::Eos` (under
/// the sink's own `Element::name()`) once EOS completes — mirrors what
/// `Queue` does for its own downstream, but without introducing a thread
/// boundary. This is what lets a fully direct chain (no `queue()` calls at
/// all) still report EOS on the bus. During preroll it marks a terminal ready
/// only after the wrapped sink's synchronous `consume` has returned `Ok`:
/// accepted/submitted, not necessarily scanned out, played, or remotely
/// received.
struct TerminalTracer {
    bus: Bus,
    id: ElementId,
    inner: Box<dyn Sink>,
    paused: bool,
    preroll: Option<Arc<crate::control::PrerollContext>>,
}

impl Element for TerminalTracer {
    fn name(&self) -> Arc<str> {
        self.inner.name()
    }

    fn element_type(&self) -> ElementType {
        self.inner.element_type()
    }

    fn graph_id(&self) -> Option<ElementId> {
        self.inner.graph_id()
    }

    fn pp_log(&self) -> &PpLog {
        self.inner.pp_log()
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        self.inner.pp_log_mut()
    }
}

impl Sink for TerminalTracer {
    fn ready_consume(&mut self) -> bool {
        if self.paused {
            return false;
        }
        // This terminal's own readiness, not the whole preroll's. Closing only
        // once every branch is done would let whichever reached the target
        // first keep consuming for as long as the slowest one takes, leaving
        // the streams at different positions when preroll finally completes.
        if self
            .preroll
            .as_ref()
            .is_some_and(|context| context.is_ready(self.id))
        {
            return false;
        }
        self.inner.ready_consume()
    }

    fn input_contract(&self) -> InputContract {
        self.inner.input_contract()
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        let is_eos = buf.is_eos();
        if is_eos {
            pp_trace!(pp_log: self.inner.pp_log(), "event=eos phase=received");
        }
        let result = self.inner.consume(buf);
        // `Sink::consume` defines successful terminal return as acceptance
        // into that sink's output path. This is the preroll completion point;
        // deliberately do not claim physical presentation has completed.
        if result.is_ok()
            && let Some(context) = &self.preroll
        {
            if is_eos {
                context.mark_eos(self.id);
            } else {
                context.mark_ready(self.id);
            }
        }
        if is_eos {
            match &result {
                Ok(()) => {
                    pp_trace!(
                        pp_log: self.inner.pp_log(),
                        "event=eos phase=completed outcome=ok"
                    );
                    self.bus.post(
                        self.inner.pp_log(),
                        BusEvent::Eos {
                            element_type: self.inner.element_type(),
                            name: self.inner.name(),
                        },
                    );
                }
                Err(error) => pp_trace!(
                    pp_log: self.inner.pp_log(),
                    "event=eos phase=completed outcome=error error={error}"
                ),
            }
        }
        result
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        pp_trace!(
            pp_log: self.inner.pp_log(),
            "event=control control={msg:?} phase=received"
        );
        let result = self.inner.control(msg.clone());
        if result.is_ok() {
            match &msg {
                ControlMsg::Pause => {
                    self.paused = true;
                    self.preroll = None;
                }
                ControlMsg::Resume => {
                    self.paused = false;
                    self.preroll = None;
                }
                ControlMsg::Preroll(context) => {
                    self.paused = false;
                    self.preroll = Some(Arc::clone(context));
                }
                ControlMsg::Stop => {
                    if let Some(context) = self.preroll.take() {
                        context.cancel();
                    }
                    self.paused = true;
                }
                ControlMsg::Flush | ControlMsg::CheckSeek(_) | ControlMsg::Seek(_) => {}
            }
        }
        match &result {
            Ok(()) => pp_trace!(
                pp_log: self.inner.pp_log(),
                "event=control control={msg:?} phase=completed outcome=ok"
            ),
            Err(error) => pp_trace!(
                pp_log: self.inner.pp_log(),
                "event=control control={msg:?} phase=completed outcome=error error={error}"
            ),
        }
        result
    }
}

impl StageBuilder for QueueStage {
    fn wrap(
        self: Box<Self>,
        downstream: Box<dyn Sink>,
        bus: &Bus,
        pipeline_id: &str,
    ) -> Result<Box<dyn Sink>> {
        Ok(Box::new(Queue::spawn_with_policy(
            self.name,
            self.capacity,
            downstream,
            bus.for_element(self.id),
            self.policy,
            Some(pipeline_id),
        )?))
    }
}

impl ChainBuilder {
    /// Starts a detached branch plan. Prefer [`Context::branch`] at call
    /// sites; it makes the owning pipeline explicit without cloning the
    /// context manually.
    pub fn new(context: Arc<Context>) -> Self {
        Self {
            context,
            elements: Vec::new(),
            planned: Vec::new(),
            error: None,
        }
    }

    /// Adds a single-output `Filter` (decoder, encoder, filter, ...) that
    /// receives via `Sink` and produces through its own (single) src pad.
    /// It runs on the same thread as whatever is upstream of it — direct
    /// function call, no queue.
    pub fn pipe<T: Filter + 'static>(mut self, mut element: T) -> Self {
        let name = element.name();
        let pad_count = element.src_pads().len();
        if pad_count != 1 && self.error.is_none() {
            self.error = Some(GraphError::NotSingleOutput {
                name: name.clone(),
                count: pad_count,
            });
        }
        let (output_port, output) = element
            .src_pads()
            .first()
            .map(|pad| (Arc::<str>::from(pad.name()), pad.contract()))
            .unwrap_or_else(|| ("src".into(), OutputContract::Unknown));
        self.planned.push(PlannedNode {
            info: NodeInfo {
                id: self.context.graph.reserve_element_id(),
                element_type: element.element_type(),
                name,
            },
            output_port,
            input: element.input_contract(),
            output,
        });
        self.elements.push(Box::new(DirectStage(element)));
        self
    }

    /// Introduces a thread boundary (blocking when full — see
    /// [`OverflowPolicy::Block`]): everything added after this runs on its
    /// own worker thread instead of the thread that feeds this queue.
    pub fn queue(self, name: impl Into<String>, capacity: usize) -> Self {
        self.queue_with_policy(name, capacity, OverflowPolicy::default())
    }

    /// Same as [`ChainBuilder::queue`], but lets you choose what happens
    /// when the queue is full (e.g. [`OverflowPolicy::DropNewest`] for a
    /// live source that shouldn't stall upstream).
    pub fn queue_with_policy(
        mut self,
        name: impl Into<String>,
        capacity: usize,
        policy: OverflowPolicy,
    ) -> Self {
        let name: Arc<str> = name.into().into();
        let id = self.context.graph.reserve_element_id();
        self.planned.push(PlannedNode {
            info: NodeInfo {
                id,
                element_type: ElementType::Queue,
                name: name.clone(),
            },
            output_port: format!("{name}_src").into(),
            // A Queue is a thread boundary, not a transform: it hands
            // downstream exactly what it was given. Declared here rather
            // than on the element because a Queue owns its downstream
            // sink directly and so has no pad to carry it.
            input: InputContract::Any,
            output: OutputContract::Passthrough,
        });
        self.elements.push(Box::new(QueueStage {
            id,
            name: name.to_string(),
            capacity,
            policy,
        }));
        self
    }

    /// Terminates the chain with a `Sink` (muxer, file sink, ...) and
    /// assembles everything into a single `Box<dyn Sink>` ready to be
    /// linked into a source's src pad. The terminal's own `Element::name()`
    /// is what shows up on the bus when it reports EOS.
    pub fn to(self, mut terminal: Box<dyn Sink>) -> Result<DetachedBranch> {
        if let Some(error) = self.error {
            return Err(error.into());
        }
        *terminal.pp_log_mut() = element_pp_log(
            terminal.element_type(),
            &terminal.name(),
            Some(&self.context.pipeline_id),
        );
        let terminal_info = NodeInfo {
            id: terminal
                .graph_id()
                .unwrap_or_else(|| self.context.graph.reserve_element_id()),
            element_type: terminal.element_type(),
            name: terminal.name(),
        };
        let terminal_id = terminal_info.id;

        // The terminal has no src pad, so nothing flows onward from it.
        let mut contracts: HashMap<_, _> = self
            .planned
            .iter()
            .map(|node| {
                (
                    node.info.id,
                    PortContracts {
                        input: node.input,
                        output: node.output,
                    },
                )
            })
            .collect();
        contracts.insert(
            terminal_id,
            PortContracts {
                input: terminal.input_contract(),
                output: OutputContract::Unknown,
            },
        );

        let mut nodes: Vec<_> = self.planned.iter().map(|node| node.info.clone()).collect();
        nodes.push(terminal_info);
        let edges = nodes
            .windows(2)
            .enumerate()
            .map(|(index, pair)| PlannedEdge {
                from: PortRef {
                    element: pair[0].id,
                    port: self.planned[index].output_port.clone(),
                },
                to: PortRef {
                    element: pair[1].id,
                    port: "sink".into(),
                },
            })
            .collect();
        let root_id = nodes.first().expect("terminal always supplies one node").id;
        let terminal: Box<dyn Sink> = Box::new(TerminalTracer {
            bus: self.context.bus.for_element(terminal_id),
            id: terminal_id,
            inner: terminal,
            paused: false,
            preroll: None,
        });
        let plan = BranchPlan {
            nodes,
            edges,
            contracts,
            root: root_id,
        };
        // Nothing is flowing into a detached branch yet, so this only
        // catches links downstream of a stage that produces something of
        // its own — a decoder feeding a muxer, say. The same walk runs
        // again at attach time with the real upstream contract.
        plan.resolve(None)?;

        let root = self
            .elements
            .into_iter()
            .rev()
            .try_fold(terminal, |downstream, stage| {
                stage.wrap(downstream, &self.context.bus, &self.context.pipeline_id)
            })?;
        Ok(DetachedBranch { root, plan })
    }

    /// Ends the chain at an already-assembled [`DetachedBranch`] — a
    /// [`crate::elements::TeeBuilder`]'s fan-out, in practice — instead of a
    /// plain `Sink`.
    ///
    /// A `Tee` cannot be a [`Self::pipe`] stage: its outputs live behind a
    /// lock rather than in `src_pads`, so it is not a `Source` and nothing
    /// can be chained onto it. It is still where a chain *ends*, though, and
    /// without this the only way to put stages in front of one is to attach
    /// the fan-out to a mid-chain element's own pad and then attach that
    /// element separately.
    ///
    /// That wires the buffers correctly but misrecords the topology.
    /// [`Context::attach`] always names the pipeline's source as the parent
    /// node, because the element whose pad it was handed is not in the graph
    /// yet — there is no other id it could use. The fan-out then renders as
    /// the source's own, leaving an element it never passed through.
    ///
    /// Joining the two plans here keeps the edge on the stage that really
    /// feeds the branch, so one attach commits the whole subgraph and the
    /// diagram shows the fan-out where it occurs.
    pub fn to_branch(self, downstream: DetachedBranch) -> Result<DetachedBranch> {
        if let Some(error) = self.error {
            return Err(error.into());
        }
        let DetachedBranch {
            root: sink,
            mut plan,
        } = downstream;

        // This chain's stages, in front of the plan that arrived. What the
        // branch ends in is untouched: it was assembled by its own `to`.
        let mut nodes: Vec<_> = self.planned.iter().map(|node| node.info.clone()).collect();
        let mut edges = Vec::with_capacity(self.planned.len() + plan.edges.len());
        for (index, node) in self.planned.iter().enumerate() {
            // The last stage feeds the branch's root; every other one feeds
            // the stage after it.
            let consumer = self
                .planned
                .get(index + 1)
                .map_or(plan.root, |next| next.info.id);
            edges.push(PlannedEdge {
                from: PortRef {
                    element: node.info.id,
                    port: node.output_port.clone(),
                },
                to: PortRef {
                    element: consumer,
                    port: "sink".into(),
                },
            });
            plan.contracts.insert(
                node.info.id,
                PortContracts {
                    input: node.input,
                    output: node.output,
                },
            );
        }
        // A chain with no stages of its own is the branch itself, root
        // included — there is nothing in front of it to become the new one.
        let root_id = nodes.first().map_or(plan.root, |node| node.id);
        edges.append(&mut plan.edges);
        nodes.append(&mut plan.nodes);
        let plan = BranchPlan {
            nodes,
            edges,
            contracts: plan.contracts,
            root: root_id,
        };
        // The same walk `to` runs, over the joined plan: a stage added in
        // front can be what makes a link below the branch's root impossible.
        plan.resolve(None)?;

        let root = self
            .elements
            .into_iter()
            .rev()
            .try_fold(sink, |downstream, stage| {
                stage.wrap(downstream, &self.context.bus, &self.context.pipeline_id)
            })?;
        Ok(DetachedBranch { root, plan })
    }

    /// Alias of [`Self::to`] retained for callers that prefer builder-style
    /// terminology when supplying the terminal sink.
    pub fn build(self, terminal: Box<dyn Sink>) -> Result<DetachedBranch> {
        self.to(terminal)
    }
}

impl Context {
    /// Starts a detached branch plan scoped to this source and pipeline.
    ///
    /// Building the branch allocates its runtime elements but does not publish
    /// them in the graph until [`Self::attach`] succeeds.
    pub fn branch(self: &Arc<Self>) -> ChainBuilder {
        ChainBuilder::new(self.clone())
    }

    /// Atomically attaches a completed branch to one source pad.
    ///
    /// Returns the stable branch identity used by later dynamic graph
    /// operations. An invalid index or already-linked pad returns a
    /// [`GraphError`] without changing either the runtime connection or graph.
    /// If a lifecycle or seek operation is in progress, this returns
    /// [`GraphError::TimelineOperationInProgress`] immediately; build a fresh
    /// detached branch and retry after that operation finishes.
    pub fn attach<S: Source>(
        &self,
        source: &mut S,
        pad_index: usize,
        branch: DetachedBranch,
    ) -> Result<BranchId> {
        let pads = source.src_pads();
        let pad_count = pads.len();
        let pad = pads.get_mut(pad_index).ok_or(GraphError::PadOutOfRange {
            index: pad_index,
            pad_count,
        })?;
        self.attach_pad(pad, branch)
    }

    pub(crate) fn attach_pad(&self, pad: &mut SrcPad, branch: DetachedBranch) -> Result<BranchId> {
        let _operation = match self.operation.try_lock() {
            Ok(guard) => guard,
            Err(std::sync::TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => {
                return Err(GraphError::TimelineOperationInProgress.into());
            }
        };
        if pad.is_linked() {
            return Err(GraphError::PadAlreadyLinked(pad.name().to_owned()).into());
        }
        let from_port: Arc<str> = pad.name().into();
        // Where a branch built in isolation finally meets the pad feeding
        // it. Re-walking the whole branch rather than comparing one
        // summarized input contract is what makes a leading passthrough
        // stage work: it carries this pad's contract through to whichever
        // element downstream actually constrains it. The check runs inside
        // `attach_with`'s own transaction, so a rejected branch leaves the
        // pad unlinked and the graph untouched.
        let incoming = Incoming::Known(incoming_from(pad));
        let DetachedBranch { root, plan } = branch;
        Ok(self
            .graph
            .attach_with(self.source_id, from_port, incoming, plan, |_| {
                pad.link(root);
                Ok(())
            })?)
    }
}

/// What a pad is known to be putting onto the wire, for
/// [`BranchPlan::resolve`].
///
/// A [`OutputContract::Passthrough`] pad — a [`crate::elements::Tee`]'s —
/// carries whatever reached the element that owns it, which this cannot
/// see from the pad alone; those are resolved from the live graph instead
/// (see [`crate::elements::TeeHandle::attach`]).
pub(crate) fn incoming_from(pad: &SrcPad) -> Option<ResolvedFlow> {
    match pad.contract() {
        OutputContract::Fixed(contract) => Some(ResolvedFlow {
            producer: pad.name().into(),
            contract,
        }),
        OutputContract::Passthrough | OutputContract::Unknown => None,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use super::*;
    use crate::{control::PrerollContext, element::element_pp_log};

    struct CountingTerminal {
        count: Arc<AtomicUsize>,
        pp_log: PpLog,
    }

    impl Element for CountingTerminal {
        fn name(&self) -> Arc<str> {
            "terminal".into()
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

    impl Sink for CountingTerminal {
        fn consume(&mut self, _buf: MediaBuffer) -> Result<()> {
            self.count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn control(&mut self, _msg: ControlMsg) -> Result<()> {
            Ok(())
        }
    }

    fn terminal(id: ElementId, count: Arc<AtomicUsize>) -> TerminalTracer {
        let (bus, _rx) = Bus::new();
        TerminalTracer {
            bus: bus.for_element(id),
            id,
            inner: Box::new(CountingTerminal {
                count,
                pp_log: element_pp_log(ElementType::Other, "terminal", None),
            }),
            paused: false,
            preroll: None,
        }
    }

    /// Each terminal closes on its *own* sample. Holding it open until every
    /// branch is done would let whichever reached the target first keep
    /// consuming for as long as the slowest takes, so the two streams would
    /// sit at different positions once preroll completed.
    #[test]
    fn terminal_readiness_closes_on_its_own_sample_not_the_whole_preroll() {
        let first = ElementId::for_test(1);
        let second = ElementId::for_test(2);
        let context = Arc::new(PrerollContext::new([first, second]));
        let mut first_terminal = terminal(first, Arc::new(AtomicUsize::new(0)));
        let mut second_terminal = terminal(second, Arc::new(AtomicUsize::new(0)));

        first_terminal
            .control(ControlMsg::Pause)
            .expect("pause first terminal");
        assert!(!first_terminal.ready_consume());
        first_terminal
            .control(ControlMsg::Preroll(Arc::clone(&context)))
            .expect("preroll first terminal");
        second_terminal
            .control(ControlMsg::Preroll(Arc::clone(&context)))
            .expect("preroll second terminal");

        first_terminal
            .consume(MediaBuffer::Packet(Arc::new(ffmpeg_next::Packet::empty())))
            .expect("consume first terminal sample");
        assert!(
            !first_terminal.ready_consume(),
            "the first terminal stops as soon as it has its own sample"
        );
        assert!(
            second_terminal.ready_consume(),
            "the second is still owed one"
        );
        assert!(!context.is_complete());

        second_terminal
            .consume(MediaBuffer::Packet(Arc::new(ffmpeg_next::Packet::empty())))
            .expect("consume second terminal sample");
        assert!(context.is_complete());
        assert!(!first_terminal.ready_consume());
        assert!(!second_terminal.ready_consume());

        first_terminal
            .control(ControlMsg::Resume)
            .expect("resume first terminal");
        assert!(first_terminal.ready_consume());
    }
}
