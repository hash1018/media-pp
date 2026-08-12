use std::{
    collections::{HashMap, HashSet},
    fmt,
    sync::{Arc, Mutex},
};

use thiserror::Error as ThisError;

use crate::element::ElementType;

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
pub struct NodeInfo {
    pub id: ElementId,
    pub element_type: ElementType,
    pub name: Arc<str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortRef {
    pub element: ElementId,
    pub port: Arc<str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeInfo {
    pub id: EdgeId,
    pub branch_id: BranchId,
    pub from: PortRef,
    pub to: PortRef,
}

#[derive(Debug, Clone)]
pub struct GraphSnapshot {
    pub revision: u64,
    pub nodes: Vec<NodeInfo>,
    pub edges: Vec<EdgeInfo>,
}

impl GraphSnapshot {
    pub fn node(&self, id: ElementId) -> Option<&NodeInfo> {
        self.nodes.iter().find(|node| node.id == id)
    }

    /// Renders every root-to-leaf path in insertion order. Keeping edges
    /// separate from nodes means this also remains meaningful for fan-in
    /// graphs, where a node can have more than one upstream.
    pub fn topology(&self) -> String {
        let leaves: Vec<_> = self
            .nodes
            .iter()
            .filter(|node| !self.edges.iter().any(|edge| edge.from.element == node.id))
            .collect();
        let mut rendered = Vec::new();
        for leaf in leaves {
            let mut paths = Vec::new();
            self.paths_to(leaf.id, &mut HashSet::new(), &mut Vec::new(), &mut paths);
            for path in paths {
                rendered.push(
                    path.into_iter()
                        .filter_map(|id| self.node(id))
                        .map(|node| format!("{:?}({})", node.element_type, node.name))
                        .collect::<Vec<_>>()
                        .join(" - "),
                );
            }
        }
        rendered.join("\n")
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

#[derive(Debug, ThisError, PartialEq, Eq)]
pub enum GraphError {
    #[error("source pad index {index} is out of range (source has {pad_count} pads)")]
    PadOutOfRange { index: usize, pad_count: usize },

    #[error("source pad '{0}' is already linked")]
    PadAlreadyLinked(String),

    #[error("element {0} is not attached to this pipeline")]
    ParentNotAttached(ElementId),

    #[error("element {0} is already attached to this pipeline")]
    NodeAlreadyAttached(ElementId),

    #[error("branch {0} is not attached")]
    BranchNotAttached(BranchId),

    #[error("a branch must contain at least one element")]
    EmptyBranch,

    #[error("ChainBuilder::pipe requires exactly one output pad, but {name} has {count}")]
    NotSingleOutput { name: Arc<str>, count: usize },
}

#[derive(Debug, Clone)]
pub(crate) struct PlannedEdge {
    pub from: PortRef,
    pub to: PortRef,
}

#[derive(Debug)]
pub(crate) struct BranchPlan {
    pub nodes: Vec<NodeInfo>,
    pub edges: Vec<PlannedEdge>,
    pub root: ElementId,
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
    pub fn new() -> Self {
        Self::default()
    }

    pub fn snapshot(&self) -> GraphSnapshot {
        let state = self.0.lock().unwrap();
        GraphSnapshot {
            revision: state.revision,
            nodes: state.nodes.clone(),
            edges: state.edges.clone(),
        }
    }

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
