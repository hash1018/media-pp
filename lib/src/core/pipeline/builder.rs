use std::sync::{Arc, Mutex, atomic::AtomicUsize};

use crate::{
    bus::{Bus, BusReceiver},
    clock::Clock,
    control::{self, ControlReceiver, ControlSender},
    element::{Context, SourceElement, element_clog},
    error::Result,
    graph::{ElementId, PipelineGraph},
    playback_clock::PlaybackClock,
};

use super::Pipeline;

/// Accumulates one or more sources into a single [`Pipeline`] — the
/// multi-source generalization of what [`Pipeline::new`] does for exactly
/// one. Each [`PipelineBuilder::add_source`] call gets its own background
/// thread once [`PipelineBuilder::build`]'s [`Pipeline::run`] starts, but
/// they all share one [`Bus`] (so [`Pipeline::bus`] sees every source's
/// events on one channel), one [`Clock`] (so every [`crate::elements::Pacer`]
/// anywhere in the pipeline — regardless of which source's chain it's
/// under — agrees on the same t=0/pause timeline), and one
/// [`PipelineGraph`] (so [`Pipeline::topology`] renders every source's
/// own branches together).
///
/// [`Pipeline::new`] is exactly `PipelineBuilder::new(id).add_source(source,
/// wire).build()` — the ergonomic single-source special case, kept as its
/// own entry point so existing single-source callers don't need to change.
/// Reach for `PipelineBuilder` directly once there's more than one live
/// source to combine into one file/output — e.g. a video capture and an
/// audio capture both feeding the same [`crate::elements::Mp4Muxer`]: two
/// independent sources under today's [`crate::element::SourceElement`]
/// model, but one [`Pipeline`] so `run()`/`pause()`/`resume()`/`stop()`
/// only need to be called once, not once per source.
pub(super) type SourceEntry = (ElementId, Box<dyn SourceElement>);

pub struct PipelineBuilder {
    id: Arc<str>,
    bus: Bus,
    bus_rx: BusReceiver,
    clock: Arc<Clock>,
    playback_clock: Arc<PlaybackClock>,
    graph: PipelineGraph,
    sources: Vec<SourceEntry>,
    control_pairs: Vec<(ControlSender, ControlReceiver)>,
}

impl PipelineBuilder {
    pub fn new(id: impl Into<String>) -> Self {
        let id: Arc<str> = id.into().into();
        let (bus, bus_rx) = Bus::new();
        let clock = Arc::new(Clock::new());
        Self {
            id,
            bus,
            bus_rx,
            playback_clock: Arc::new(PlaybackClock::new(clock.clone())),
            clock,
            graph: PipelineGraph::new(),
            sources: Vec::new(),
            control_pairs: Vec::new(),
        }
    }

    /// Registers one more source. `wire` receives a source-scoped
    /// [`Context`]; build detached branches with [`Context::branch`] and
    /// commit them with [`Context::attach`]. A wiring error aborts the
    /// builder without publishing a partially built pipeline.
    pub fn add_source<S: SourceElement + 'static>(
        mut self,
        mut source: S,
        wire: impl FnOnce(&mut S, &Arc<Context>) -> Result<()>,
    ) -> Result<Self> {
        *source.clog_mut() = element_clog(source.element_type(), &source.name(), Some(&self.id));
        let source_id = self.graph.add_source(source.element_type(), source.name());
        let context = Arc::new(Context {
            bus: self.bus.clone(),
            pipeline_id: self.id.clone(),
            graph: self.graph.clone(),
            clock: self.clock.clone(),
            playback_clock: self.playback_clock.clone(),
            source_id,
        });
        wire(&mut source, &context)?;
        self.sources.push((source_id, Box::new(source)));
        self.control_pairs.push(control::channel());
        Ok(self)
    }

    /// Finishes construction. At least one [`PipelineBuilder::add_source`]
    /// call must have happened — an empty [`Pipeline`] has nothing for
    /// [`Pipeline::run`] to ever drive, and [`Pipeline::bus`] would block
    /// forever waiting for a source thread that will never start (nothing
    /// left holding a [`Bus`] sender to eventually drop).
    pub fn build(self) -> Arc<Pipeline> {
        assert!(
            !self.sources.is_empty(),
            "PipelineBuilder::build called with no sources added"
        );
        let (control_txs, control_rxs): (Vec<_>, Vec<_>) = self.control_pairs.into_iter().unzip();
        Arc::new(Pipeline {
            id: self.id,
            sources: Mutex::new(Some(self.sources)),
            bus: Mutex::new(Some(self.bus)),
            control_txs,
            control_rxs: Mutex::new(Some(control_rxs)),
            clock: self.clock,
            playback_clock: self.playback_clock,
            bus_rx: self.bus_rx,
            running: Arc::new(AtomicUsize::new(0)),
            workers: Mutex::new(Vec::new()),
            graph: self.graph,
        })
    }
}
