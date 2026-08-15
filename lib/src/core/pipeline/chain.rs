use std::sync::Arc;

use crate::pp_log::PpLog;

use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    control::ControlMsg,
    element::{Context, Element, ElementType, Filter, Sink, Source, element_pp_log},
    error::Result,
    graph::{BranchId, BranchPlan, ElementId, GraphError, NodeInfo, PlannedEdge, PortRef},
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
}

/// A fully constructed runtime chain whose graph nodes are still detached.
/// Dropping it has no topology effect; only an attach operation commits it.
pub struct DetachedBranch {
    pub(crate) root: Box<dyn Sink>,
    pub(crate) plan: BranchPlan,
}

impl DetachedBranch {
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
    ) -> Box<dyn Sink>;
}

struct DirectStage<T>(T);

impl<T> StageBuilder for DirectStage<T>
where
    T: Filter + 'static,
{
    fn wrap(
        self: Box<Self>,
        downstream: Box<dyn Sink>,
        _bus: &Bus,
        pipeline_id: &str,
    ) -> Box<dyn Sink> {
        let mut element = self.0;
        *element.pp_log_mut() =
            element_pp_log(element.element_type(), &element.name(), Some(pipeline_id));
        element.src_pads()[0].link(downstream);
        Box::new(element)
    }
}

struct QueueStage {
    id: ElementId,
    name: String,
    capacity: usize,
    policy: OverflowPolicy,
}

/// Wraps a terminal `Sink`, posting a `BusEvent::Eos` (under the sink's own
/// `Element::name()`) once it sees an EOS buffer pass through — mirrors
/// what `Queue` does for its own downstream, but without introducing a
/// thread boundary. This is what lets a fully direct chain (no `queue()`
/// calls at all) still report EOS on the bus.
struct EosReporter {
    bus: Bus,
    inner: Box<dyn Sink>,
}

impl Element for EosReporter {
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

impl Sink for EosReporter {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        let is_eos = buf.is_eos();
        self.inner.consume(buf)?;
        if is_eos {
            self.bus.post(
                self.inner.pp_log(),
                BusEvent::Eos {
                    element_type: self.inner.element_type(),
                    name: self.inner.name(),
                },
            );
        }
        Ok(())
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        self.inner.control(msg)
    }
}

impl StageBuilder for QueueStage {
    fn wrap(
        self: Box<Self>,
        downstream: Box<dyn Sink>,
        bus: &Bus,
        pipeline_id: &str,
    ) -> Box<dyn Sink> {
        Box::new(Queue::spawn_with_policy(
            self.name,
            self.capacity,
            downstream,
            bus.for_element(self.id),
            self.policy,
            Some(pipeline_id),
        ))
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
        let output_port = element
            .src_pads()
            .first()
            .map(|pad| Arc::<str>::from(pad.name()))
            .unwrap_or_else(|| "src".into());
        self.planned.push(PlannedNode {
            info: NodeInfo {
                id: self.context.graph.reserve_element_id(),
                element_type: element.element_type(),
                name,
            },
            output_port,
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
        let terminal: Box<dyn Sink> = Box::new(EosReporter {
            bus: self.context.bus.for_element(terminal_id),
            inner: terminal,
        });
        let root = self
            .elements
            .into_iter()
            .rev()
            .fold(terminal, |downstream, stage| {
                stage.wrap(downstream, &self.context.bus, &self.context.pipeline_id)
            });
        Ok(DetachedBranch {
            root,
            plan: BranchPlan {
                nodes,
                edges,
                root: root_id,
            },
        })
    }

    pub fn build(self, terminal: Box<dyn Sink>) -> Result<DetachedBranch> {
        self.to(terminal)
    }
}

impl Context {
    pub fn branch(self: &Arc<Self>) -> ChainBuilder {
        ChainBuilder::new(self.clone())
    }

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
        if pad.is_linked() {
            return Err(GraphError::PadAlreadyLinked(pad.name().to_owned()).into());
        }
        let from_port: Arc<str> = pad.name().into();
        let DetachedBranch { root, plan } = branch;
        Ok(self
            .graph
            .attach_with(self.source_id, from_port, plan, |_| {
                pad.link(root);
                Ok(())
            })?)
    }
}
