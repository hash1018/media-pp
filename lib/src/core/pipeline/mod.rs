//! Building and running a graph.
//!
//! [`PipelineBuilder`] assembles one; [`Pipeline`] owns it at runtime — the
//! source threads, the control cascade, the shared
//! [`Clock`](crate::clock::Clock), the [`Bus`](crate::bus::Bus), and the
//! [`PipelineGraph`](crate::graph::PipelineGraph). [`ChainBuilder`] and
//! [`DetachedBranch`] are how a branch is described before it is attached, so
//! that wiring can fail without leaving a half-built graph behind.
//!
//! Ending a pipeline is two different requests, not one:
//! [`Pipeline::finish`] emits ordered EOS from the source and lets it drain
//! every stage that holds delayed data, while [`Pipeline::stop`] abandons
//! that work.

mod builder;
mod chain;
mod runtime;

pub use builder::PipelineBuilder;
pub use chain::{ChainBuilder, DetachedBranch};
pub use runtime::{Pipeline, SeekMode};

#[cfg(test)]
use crate::pp_log::PpLog;
#[cfg(test)]
use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    control::ControlMsg,
    element::{Context, Element, ElementType, Sink, element_pp_log},
    error::Result,
    graph::PipelineGraph,
};
#[cfg(test)]
use std::sync::{Arc, atomic::Ordering};

#[cfg(test)]
mod tests;
