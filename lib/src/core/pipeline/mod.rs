mod builder;
mod chain;
mod runtime;

pub use builder::PipelineBuilder;
pub use chain::{ChainBuilder, DetachedBranch};
pub use runtime::Pipeline;

#[cfg(test)]
use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    control::ControlMsg,
    element::{Context, Element, ElementType, Sink, element_hlog},
    error::Result,
    graph::PipelineGraph,
};
#[cfg(test)]
use rust_hlog::HLog;
#[cfg(test)]
use std::sync::{Arc, atomic::Ordering};

#[cfg(test)]
mod tests;
