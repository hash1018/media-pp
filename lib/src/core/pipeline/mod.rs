mod builder;
mod chain;
mod runtime;

pub use builder::PipelineBuilder;
pub use chain::{ChainBuilder, DetachedBranch};
pub use runtime::Pipeline;

#[cfg(test)]
use crate::clog::CLog;
#[cfg(test)]
use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    control::ControlMsg,
    element::{Context, Element, ElementType, Sink, element_clog},
    error::Result,
    graph::PipelineGraph,
};
#[cfg(test)]
use std::sync::{Arc, atomic::Ordering};

#[cfg(test)]
mod tests;
