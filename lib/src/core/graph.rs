//! The pipeline's topology, recorded separately from the elements themselves.
//!
//! Elements own their pads and their downstream peers; nothing in that chain
//! can answer "what does this pipeline look like right now". [`PipelineGraph`]
//! keeps that record: stable [`ElementId`]/[`EdgeId`]/[`BranchId`] values that
//! survive same-name churn, and [`GraphSnapshot`] as a consistent copy to read
//! outside the lock.
//!
//! Names are labels only. IDs are what attachment, detachment, and lookup use,
//! and what lets a log record name one specific element when several share a
//! name.

use std::{
    collections::{HashMap, HashSet},
    fmt::{self, Write as _},
    sync::{Arc, Mutex},
};

use thiserror::Error as ThisError;

use crate::{
    contract::{InputContract, OutputContract, PortContract},
    element::ElementType,
    log::{Level, enabled},
    pp_log::{PpLog, pp_info},
};

/// Stable identity of one element inside a pipeline graph. Names are only
/// labels; IDs are what graph mutation and lookup use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ElementId(u64);

impl fmt::Display for ElementId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Stable identity of one connection between two element ports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EdgeId(u64);

/// Identity of one attached branch. A branch owns every node and edge that
/// arrived in the same attachment transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BranchId(u64);

impl fmt::Display for BranchId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// One element as it appears in a [`GraphSnapshot`] — its stable identity, its
/// type, and the name its caller chose.
pub struct NodeInfo {
    /// Stable identity assigned by the owning pipeline graph.
    pub id: ElementId,
    /// Built-in kind reported by the element.
    pub element_type: ElementType,
    /// Caller-selected instance name; names need not be unique within a graph.
    pub name: Arc<str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// One end of an edge: the element, and the name of the pad on it.
pub struct PortRef {
    /// Stable identity of the element that owns this port.
    pub element: ElementId,
    /// Element-defined port name used in topology output.
    pub port: Arc<str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// One link from a source pad to a downstream element, together with the
/// branch whose attachment created it.
pub struct EdgeInfo {
    /// Stable identity of this connection.
    pub id: EdgeId,
    /// Attachment transaction that created this edge.
    pub branch_id: BranchId,
    /// Upstream source pad.
    pub from: PortRef,
    /// Downstream sink port.
    pub to: PortRef,
}

#[derive(Debug, Clone)]
/// A consistent copy of the whole topology, taken under the graph's lock and
/// then read without it.
///
/// `revision` increments on every mutation, so two snapshots can be told apart
/// even when they describe the same shape. This is also what
/// [`crate::pipeline::Pipeline::topology`] renders and what a `run`/`attach`/
/// `detach` log record embeds.
pub struct GraphSnapshot {
    /// Monotonically increasing graph version. Each successful mutation
    /// increments it exactly once.
    pub revision: u64,
    /// Elements attached when this snapshot was taken.
    pub nodes: Vec<NodeInfo>,
    /// Connections attached when this snapshot was taken.
    pub edges: Vec<EdgeInfo>,
}

impl GraphSnapshot {
    /// Finds one node by stable identity within this snapshot.
    pub fn node(&self, id: ElementId) -> Option<&NodeInfo> {
        self.nodes.iter().find(|node| node.id == id)
    }

    /// Renders every root-to-leaf path in insertion order. Keeping edges
    /// separate from nodes means this also remains meaningful for fan-in
    /// graphs, where a node can have more than one upstream.
    pub fn topology(&self) -> String {
        self.paths()
            .into_iter()
            .map(|path| {
                path.into_iter()
                    .filter_map(|id| self.node(id))
                    .map(|node| format!("{:?}({})", node.element_type, node.name))
                    .collect::<Vec<_>>()
                    .join(" - ")
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Logging-only flow diagram. Each child connector starts under its
    /// upstream element, so a fan-out is visible at the element where it
    /// actually occurs instead of repeating the common path for every leaf.
    pub(crate) fn topology_diagram(&self) -> String {
        let roots: Vec<_> = self
            .nodes
            .iter()
            .filter(|node| !self.edges.iter().any(|edge| edge.to.element == node.id))
            .collect();
        let mut output = String::new();

        for (index, root) in roots.iter().enumerate() {
            let is_last = index + 1 == roots.len();
            let child_indent = if roots.len() == 1 {
                let _ = write!(output, "{:?}({})#{}", root.element_type, root.name, root.id);
                String::new()
            } else {
                let connector = if is_last { "└── " } else { "├── " };
                let _ = write!(
                    output,
                    "{connector}{:?}({})#{}",
                    root.element_type, root.name, root.id
                );
                if is_last {
                    "    ".to_owned()
                } else {
                    "│   ".to_owned()
                }
            };
            self.render_diagram_children(
                root.id,
                &child_indent,
                &mut HashSet::from([root.id]),
                &mut output,
            );
            if !is_last {
                output.push('\n');
            }
        }

        output
    }

    fn render_diagram_children(
        &self,
        parent: ElementId,
        indent: &str,
        visiting: &mut HashSet<ElementId>,
        output: &mut String,
    ) {
        let children: Vec<_> = self
            .edges
            .iter()
            .filter(|edge| edge.from.element == parent)
            .collect();

        for (index, edge) in children.iter().enumerate() {
            let Some(child) = self.node(edge.to.element) else {
                continue;
            };
            let is_last = index + 1 == children.len();
            let connector = if is_last { "└── " } else { "├── " };
            let link = format!("[{}] → ", edge.from.port);
            let _ = write!(
                output,
                "\n{indent}{connector}{link}{:?}({})#{}",
                child.element_type, child.name, child.id
            );

            if visiting.insert(child.id) {
                let continuation = if is_last { "    " } else { "│   " };
                let child_indent =
                    format!("{indent}{continuation}{}", " ".repeat(link.chars().count()));
                self.render_diagram_children(child.id, &child_indent, visiting, output);
                visiting.remove(&child.id);
            }
        }
    }

    fn paths(&self) -> Vec<Vec<ElementId>> {
        let leaves: Vec<_> = self
            .nodes
            .iter()
            .filter(|node| !self.edges.iter().any(|edge| edge.from.element == node.id))
            .collect();
        let mut rendered = Vec::new();
        for leaf in leaves {
            self.paths_to(leaf.id, &mut HashSet::new(), &mut Vec::new(), &mut rendered);
        }
        rendered
    }

    fn paths_to(
        &self,
        current: ElementId,
        visiting: &mut HashSet<ElementId>,
        suffix: &mut Vec<ElementId>,
        paths: &mut Vec<Vec<ElementId>>,
    ) {
        if !visiting.insert(current) {
            return;
        }
        suffix.push(current);
        let upstream: Vec<_> = self
            .edges
            .iter()
            .filter(|edge| edge.to.element == current)
            .map(|edge| edge.from.element)
            .collect();
        if upstream.is_empty() {
            let mut path = suffix.clone();
            path.reverse();
            paths.push(path);
        } else {
            for parent in upstream {
                self.paths_to(parent, visiting, suffix, paths);
            }
        }
        suffix.pop();
        visiting.remove(&current);
    }
}

/// Emits `event` and the topology it produced as **one** record: the event
/// word on the header line, the diagram in the body.
///
/// They cannot be two records. The private logger queues each `write_all`
/// separately, so only the lines inside a single record are guaranteed to stay
/// together — any live thread (a `Queue` worker this very call just started,
/// say) can write between two of them. Emitting the diagram separately would
/// mean it is merely *usually* adjacent to the event that caused it.
pub(crate) fn log_topology(pp_log: &PpLog, event: &str, snapshot: &GraphSnapshot) {
    if !enabled(Level::Info) {
        return;
    }
    pp_info!(pp_log: pp_log, "{event}\n{}", snapshot.topology_diagram());
}

#[derive(Debug, ThisError, PartialEq, Eq)]
/// A rejected topology change.
///
/// Every variant here is a refusal, not a partial result: attachment validates
/// before it mutates, so a graph that returns one of these is left exactly as
/// it was.
pub enum GraphError {
    /// The requested source pad index does not exist.
    #[error("source pad index {index} is out of range (source has {pad_count} pads)")]
    PadOutOfRange {
        /// Requested zero-based source pad index.
        index: usize,
        /// Number of source pads available on the element.
        pad_count: usize,
    },

    /// Attachment targeted a source pad that already owns a downstream sink.
    #[error("source pad '{0}' is already linked")]
    PadAlreadyLinked(String),

    /// A dynamic attachment named an element that is no longer in this graph.
    #[error("element {0} is not attached to this pipeline")]
    ParentNotAttached(ElementId),

    /// An attachment plan attempted to publish an already-attached node.
    #[error("element {0} is already attached to this pipeline")]
    NodeAlreadyAttached(ElementId),

    /// A detach operation named a branch that is no longer attached.
    #[error("branch {0} is not attached")]
    BranchNotAttached(BranchId),

    /// An attachment plan contained no terminal or processing element.
    #[error("a branch must contain at least one element")]
    EmptyBranch,

    /// Two elements were wired together even though what one produces can
    /// never reach the other — see [`crate::contract`]. Reported when the
    /// branch is built or attached, before anything runs, because no
    /// buffer could have made this link work.
    #[error("{producer} produces {produced}, which {consumer} cannot accept (it takes {accepted})")]
    IncompatibleLink {
        /// Name of the element or pad on the producing side.
        producer: Arc<str>,
        /// What that side emits.
        produced: PortContract,
        /// Caller-selected name of the element that rejected the link.
        consumer: Arc<str>,
        /// What the consuming side accepts.
        accepted: PortContract,
    },

    /// [`crate::pipeline::ChainBuilder::pipe`] received a filter whose output
    /// shape cannot be represented as one linear chain stage.
    #[error("ChainBuilder::pipe requires exactly one output pad, but {name} has {count}")]
    NotSingleOutput {
        /// Caller-selected name of the rejected filter.
        name: Arc<str>,
        /// Number of source pads exposed by the rejected filter.
        count: usize,
    },
}

#[derive(Debug, Clone)]
pub(crate) struct PlannedEdge {
    pub from: PortRef,
    pub to: PortRef,
}

/// What is actually flowing along one edge, once every stage upstream of
/// it has been resolved — together with the element that produced it, so a
/// rejection names the real producer rather than whichever passthrough
/// stage last relayed it.
#[derive(Debug, Clone)]
pub(crate) struct ResolvedFlow {
    pub producer: Arc<str>,
    pub contract: PortContract,
}

/// One planned element's two contracts, kept per node so a branch can be
/// re-validated against whatever it eventually gets attached to.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PortContracts {
    pub input: InputContract,
    pub output: OutputContract,
}

#[derive(Debug)]
pub(crate) struct BranchPlan {
    pub nodes: Vec<NodeInfo>,
    pub edges: Vec<PlannedEdge>,
    pub root: ElementId,
    /// Every node's contracts, keyed by element. A branch is a tree — one
    /// chain, plus a fan-out edge per initial `Tee` branch — so validating
    /// it means walking `edges` from `root` and carrying the flow along
    /// each one, not folding a linear list.
    pub contracts: HashMap<ElementId, PortContracts>,
}

impl BranchPlan {
    /// Rejects any link in this branch whose two sides cannot meet, given
    /// what is flowing into its root.
    ///
    /// `incoming` is `None` while the branch is still detached — nothing is
    /// known to be flowing yet, so only links downstream of a stage that
    /// produces something of its own get checked. The same walk runs again
    /// at attach time with the real upstream contract, which is what
    /// catches a branch whose leading stages are all passthrough: those
    /// carry the flow through untouched, so the requirement that matters
    /// belongs to an element further down.
    pub(crate) fn validate(&self, incoming: Option<ResolvedFlow>) -> Result<(), GraphError> {
        let name_of = |id: ElementId| {
            self.nodes
                .iter()
                .find(|node| node.id == id)
                .map(|node| node.name.clone())
                .unwrap_or_else(|| "<unknown>".into())
        };

        // A plan is a tree, so every node is reached exactly once; the
        // visited set only keeps a malformed plan from looping forever.
        let mut visited = HashSet::new();
        let mut pending = vec![(self.root, incoming)];
        while let Some((id, flow)) = pending.pop() {
            if !visited.insert(id) {
                continue;
            }
            let Some(contracts) = self.contracts.get(&id) else {
                continue;
            };

            if let (Some(flow), InputContract::Fixed(accepted)) = (&flow, contracts.input)
                && !accepted.accepts(&flow.contract)
            {
                return Err(GraphError::IncompatibleLink {
                    producer: flow.producer.clone(),
                    produced: flow.contract,
                    consumer: name_of(id),
                    accepted,
                });
            }

            let outgoing = match contracts.output {
                OutputContract::Fixed(contract) => Some(ResolvedFlow {
                    producer: name_of(id),
                    contract,
                }),
                OutputContract::Passthrough => flow,
                OutputContract::Unknown => None,
            };

            // Fan-out: every branch of a `Tee` receives the same buffers,
            // so each outgoing edge carries the same resolved flow.
            for edge in self.edges.iter().filter(|edge| edge.from.element == id) {
                pending.push((edge.to.element, outgoing.clone()));
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
struct BranchRecord {
    parent: ElementId,
    owned_nodes: HashSet<ElementId>,
}

#[derive(Default)]
struct GraphState {
    next_element_id: u64,
    next_edge_id: u64,
    next_branch_id: u64,
    revision: u64,
    nodes: Vec<NodeInfo>,
    edges: Vec<EdgeInfo>,
    branches: HashMap<BranchId, BranchRecord>,
}

/// Live, transactionally-updated graph behind [`crate::pipeline::Pipeline`].
/// A snapshot never observes half of an attach/detach operation.
#[derive(Clone, Default)]
pub struct PipelineGraph(Arc<Mutex<GraphState>>);

impl PipelineGraph {
    /// Creates an empty graph with revision zero and fresh ID counters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Copies nodes, edges, and revision under the graph lock, then releases
    /// the lock before returning.
    pub fn snapshot(&self) -> GraphSnapshot {
        let state = self.0.lock().unwrap();
        GraphSnapshot {
            revision: state.revision,
            nodes: state.nodes.clone(),
            edges: state.edges.clone(),
        }
    }

    /// Returns the attached branch that owns `element`, if the element was
    /// introduced by a branch attachment transaction.
    ///
    /// Source nodes are created directly and therefore return `None`.
    pub fn branch_containing(&self, element: ElementId) -> Option<BranchId> {
        let state = self.0.lock().unwrap();
        state
            .branches
            .iter()
            .find_map(|(id, branch)| branch.owned_nodes.contains(&element).then_some(*id))
    }

    pub(crate) fn reserve_element_id(&self) -> ElementId {
        let mut state = self.0.lock().unwrap();
        state.next_element_id += 1;
        ElementId(state.next_element_id)
    }

    pub(crate) fn add_source(&self, element_type: ElementType, name: Arc<str>) -> ElementId {
        let id = self.reserve_element_id();
        let mut state = self.0.lock().unwrap();
        state.nodes.push(NodeInfo {
            id,
            element_type,
            name,
        });
        state.revision += 1;
        id
    }

    /// Validates a complete branch, performs its runtime mutation while the
    /// graph is locked, then commits every node and edge as one revision.
    pub(crate) fn attach_with(
        &self,
        parent: ElementId,
        from_port: Arc<str>,
        plan: BranchPlan,
        attach_runtime: impl FnOnce(BranchId) -> Result<(), GraphError>,
    ) -> Result<BranchId, GraphError> {
        let mut state = self.0.lock().unwrap();
        if !state.nodes.iter().any(|node| node.id == parent) {
            return Err(GraphError::ParentNotAttached(parent));
        }
        if plan.nodes.is_empty() {
            return Err(GraphError::EmptyBranch);
        }
        for node in &plan.nodes {
            if state.nodes.iter().any(|current| current.id == node.id) {
                return Err(GraphError::NodeAlreadyAttached(node.id));
            }
        }

        state.next_branch_id += 1;
        let branch_id = BranchId(state.next_branch_id);
        attach_runtime(branch_id)?;

        let mut edges = Vec::with_capacity(plan.edges.len() + 1);
        state.next_edge_id += 1;
        edges.push(EdgeInfo {
            id: EdgeId(state.next_edge_id),
            branch_id,
            from: PortRef {
                element: parent,
                port: from_port,
            },
            to: PortRef {
                element: plan.root,
                port: "sink".into(),
            },
        });
        for edge in plan.edges {
            state.next_edge_id += 1;
            edges.push(EdgeInfo {
                id: EdgeId(state.next_edge_id),
                branch_id,
                from: edge.from,
                to: edge.to,
            });
        }

        let owned_nodes = plan.nodes.iter().map(|node| node.id).collect();
        state.nodes.extend(plan.nodes);
        state.edges.extend(edges);
        state.branches.insert(
            branch_id,
            BranchRecord {
                parent,
                owned_nodes,
            },
        );
        state.revision += 1;
        Ok(branch_id)
    }

    /// Performs runtime detach first, then removes this branch and any
    /// branches attached below nodes it owned as one graph revision.
    pub(crate) fn detach_with(
        &self,
        branch_id: BranchId,
        detach_runtime: impl FnOnce() -> Result<(), GraphError>,
    ) -> Result<(), GraphError> {
        let mut state = self.0.lock().unwrap();
        if !state.branches.contains_key(&branch_id) {
            return Err(GraphError::BranchNotAttached(branch_id));
        }
        detach_runtime()?;

        let mut removed_branches = HashSet::from([branch_id]);
        let mut removed_nodes = HashSet::new();
        loop {
            for id in removed_branches.clone() {
                if let Some(branch) = state.branches.get(&id) {
                    removed_nodes.extend(branch.owned_nodes.iter().copied());
                }
            }
            let before = removed_branches.len();
            for (id, branch) in &state.branches {
                if removed_nodes.contains(&branch.parent) {
                    removed_branches.insert(*id);
                }
            }
            if removed_branches.len() == before {
                break;
            }
        }

        state
            .branches
            .retain(|id, _| !removed_branches.contains(id));
        state.nodes.retain(|node| !removed_nodes.contains(&node.id));
        state
            .edges
            .retain(|edge| !removed_branches.contains(&edge.branch_id));
        state.revision += 1;
        Ok(())
    }
}
