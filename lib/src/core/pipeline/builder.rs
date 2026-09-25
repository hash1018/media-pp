use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize},
};

use crate::{
    bus::{Bus, BusReceiver},
    clock::Clock,
    control::{self, ControlReceiver, ControlSender},
    element::{Context, SourceElement, element_pp_log, pipeline_pp_log},
    error::Result,
    graph::{ElementId, PipelineGraph},
    pad::SrcPad,
    playback_clock::PlaybackClock,
    stats::ElementCounters,
};

use super::{Pipeline, completion::Completion, seek::PrerollSlot};

pub(super) type SourceEntry = (ElementId, Box<dyn SourceElement>);

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
/// wire)` and then `build()` — the single-source special case.
/// Reach for `PipelineBuilder` directly once there's more than one live
/// source to combine into one file/output — e.g. a video capture and an
/// audio capture both feeding the same [`crate::elements::FileMuxer`]: two
/// independent sources under today's [`crate::element::SourceElement`]
/// model, but one [`Pipeline`] so `run()`/`pause()`/`resume()`/`stop()`
/// only need to be called once, not once per source.
pub struct PipelineBuilder {
    id: Arc<str>,
    bus: Bus,
    bus_rx: BusReceiver,
    clock: Arc<Clock>,
    playback_clock: Arc<PlaybackClock>,
    state: Arc<crate::playback_state::PlaybackState>,
    graph: PipelineGraph,
    sources: Vec<SourceEntry>,
    control_pairs: Vec<(ControlSender, ControlReceiver)>,
    operation: Arc<Mutex<()>>,
    /// Each source's counters. A source moves onto its own thread when the
    /// pipeline runs and nothing wraps it the way a chain wraps a stage,
    /// so the pipeline is what keeps them — see [`crate::stats`].
    source_counters: Vec<Arc<ElementCounters>>,
    completion: Arc<Completion>,
}

impl PipelineBuilder {
    /// Starts an empty multi-source pipeline builder with the caller-selected
    /// pipeline identity and fresh bus, graph, and clocks.
    ///
    /// Add at least one source before calling [`Self::build`].
    pub fn new(id: impl Into<String>) -> Self {
        crate::ensure_ffmpeg();
        let id: Arc<str> = id.into().into();
        let (bus, bus_rx) = Bus::new();
        let clock = Arc::new(Clock::new());
        let graph = PipelineGraph::new();
        Self {
            completion: Completion::new(graph.clone(), pipeline_pp_log(&id)),
            id,
            bus,
            bus_rx,
            playback_clock: Arc::new(PlaybackClock::new(clock.clone())),
            clock,
            state: crate::playback_state::PlaybackState::new(),
            graph,
            sources: Vec::new(),
            control_pairs: Vec::new(),
            operation: Arc::new(Mutex::new(())),
            source_counters: Vec::new(),
        }
    }

    /// Registers one more source. `wire` receives a source-scoped
    /// [`Context`]; build detached branches with [`Context::branch`] and
    /// commit them with [`Context::attach`]. A wiring error aborts the
    /// builder without publishing a partially built pipeline — the builder
    /// is consumed, so there is no half-registered source left to go on
    /// with.
    ///
    /// Hands the builder back beside whatever `wire` returned, as
    /// [`Pipeline::new`] does: `()` where there is nothing, a handle the
    /// wiring made where there is.
    ///
    /// ```ignore
    /// let (builder, ()) = PipelineBuilder::new("record").add_source(video, |source, ctx| {
    ///     /* ... */
    ///     Ok(())
    /// })?;
    /// let (builder, routing) = builder.add_source(audio, |source, ctx| {
    ///     /* ... */
    ///     Ok(routing)
    /// })?;
    /// let pipeline = builder.build();
    /// ```
    pub fn add_source<S: SourceElement + 'static, T>(
        mut self,
        mut source: S,
        wire: impl FnOnce(&mut S, &Arc<Context>) -> Result<T>,
    ) -> Result<(Self, T)> {
        *source.pp_log_mut() =
            element_pp_log(source.element_type(), &source.name(), Some(&self.id));
        let source_id = self.graph.add_source(source.element_type(), source.name());
        // A source follows a seek by repositioning itself; whether it can is
        // settled now, and a seek it cannot follow is refused before
        // anything is asked of it — see `Pipeline::check_seek`.
        if source.is_live() {
            self.graph
                .refuse_seek(source_id, crate::control::SeekRejectReason::LiveSource);
        } else if !source.is_seekable() {
            self.graph.refuse_seek(
                source_id,
                crate::control::SeekRejectReason::SourceNotSeekable,
            );
        }
        if source.as_reversible().is_none() {
            self.graph.refuse_reverse(
                source_id,
                crate::control::SeekRejectReason::SourceNotReversible,
            );
        }
        // Made before the context, which carries them to a source that
        // records its own ticks.
        let counters = ElementCounters::new();
        let context = Arc::new(Context {
            bus: self.bus.clone(),
            pipeline_id: self.id.clone(),
            graph: self.graph.clone(),
            clock: self.clock.clone(),
            playback_clock: self.playback_clock.clone(),
            state: Arc::clone(&self.state),
            operation: Arc::clone(&self.operation),
            source_id,
            source_counters: Arc::clone(&counters),
            completion: Arc::clone(&self.completion),
        });
        source.attach_context(&context);
        let wired = wire(&mut source, &context)?;
        // A source is counted by what leaves its pads; it takes nothing in.
        counters.add_pads(source.src_pads().iter().map(SrcPad::counters));
        self.graph.register_counters(source_id, &counters);
        self.source_counters.push(counters);
        self.sources.push((source_id, Box::new(source)));
        self.control_pairs.push(control::channel_in(&self.state));
        Ok((self, wired))
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
        let pp_log = pipeline_pp_log(&self.id);
        Arc::new(Pipeline {
            id: self.id,
            pp_log,
            sources: Mutex::new(Some(self.sources)),
            bus: Mutex::new(Some(self.bus)),
            control_txs,
            control_rxs: Mutex::new(Some(control_rxs)),
            clock: self.clock,
            playback_clock: self.playback_clock,
            state: self.state,
            bus_rx: self.bus_rx,
            running: Arc::new(AtomicUsize::new(0)),
            paused: AtomicBool::new(false),
            stepped: AtomicBool::new(false),
            completion: Arc::clone(&self.completion),
            operation: self.operation,
            preroll_slot: PrerollSlot::default().into(),
            workers: Mutex::new(Vec::new()),
            graph: self.graph,
            _source_counters: self.source_counters,
        })
    }
}
