use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
    time::Duration,
};

use rust_hlog::{HLog, hinfo};

use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent, BusReceiver},
    clock::Clock,
    control::{self, ControlMsg, ControlReceiver, ControlSender},
    element::{
        Context, Element, ElementInfo, ElementRegistry, ElementType, Filter, Sink, SourceElement,
        element_hlog,
    },
    error::Result,
    queue::{OverflowPolicy, Queue},
};

/// Builds one chain segment (a run of elements that all execute on the same
/// thread). Call [`ChainBuilder::queue`] to close the current segment behind
/// a `Queue` and start a new one on its own worker thread.
///
/// Because each element needs a handle to *its* downstream to be
/// constructed, the chain is assembled back-to-front: elements are
/// collected in call order, then folded right-to-left starting from the
/// terminal `Sink` at [`ChainBuilder::build`] time.
pub struct ChainBuilder {
    context: Arc<Context>,
    elements: Vec<Box<dyn StageBuilder>>,
    /// Name of the most-recently-added element in this chain, captured
    /// right when `.pipe()`/`.queue()`/`.build()` are called — so the next
    /// one can register itself as *its* `upstream`. `None` until the
    /// first is added, meaning that one's own `upstream` is registered
    /// unresolved (see [`ElementInfo::upstream`]'s own docs on how that
    /// gets resolved later).
    last: Option<Arc<str>>,
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
        assert_eq!(
            element.src_pads().len(),
            1,
            "ChainBuilder::pipe() is for single-output elements; link a multi-pad \
             element's src_pads() by hand instead"
        );
        *element.hlog_mut() =
            element_hlog(element.element_type(), &element.name(), Some(pipeline_id));
        element.src_pads()[0].link(downstream);
        Box::new(element)
    }
}

struct QueueStage {
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

    fn hlog(&self) -> &HLog {
        self.inner.hlog()
    }

    fn hlog_mut(&mut self) -> &mut HLog {
        self.inner.hlog_mut()
    }
}

impl Sink for EosReporter {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        let is_eos = buf.is_eos();
        self.inner.consume(buf)?;
        if is_eos {
            self.bus.post(
                self.inner.hlog(),
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
            bus.clone(),
            self.policy,
            Some(pipeline_id),
        ))
    }
}

impl ChainBuilder {
    /// A fresh chain defaults to hanging directly off whichever source
    /// `context` was built for (`context.default_upstream`) — not `None`
    /// — so [`Pipeline::topology`] renders it correctly even when there's
    /// more than one source in the same [`Pipeline`] (see
    /// [`PipelineBuilder`]). [`crate::elements::TeeHandle::add_sink`]
    /// still overrides this after the fact via
    /// [`ElementRegistry::set_upstream`] for a chain that turns out to
    /// start under a `Tee` instead.
    pub fn new(context: Arc<Context>) -> Self {
        let last = Some(context.default_upstream.clone());
        Self {
            context,
            elements: Vec::new(),
            last,
        }
    }

    /// Adds a single-output `Filter` (decoder, encoder, filter, ...) that
    /// receives via `Sink` and produces through its own (single) src pad.
    /// It runs on the same thread as whatever is upstream of it — direct
    /// function call, no queue.
    pub fn pipe<T: Filter + 'static>(mut self, element: T) -> Self {
        let name = element.name();
        self.context
            .registry
            .register(element.element_type(), name.clone(), self.last.take());
        self.last = Some(name);
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
        self.context
            .registry
            .register(ElementType::Queue, name.clone(), self.last.take());
        self.last = Some(name.clone());
        self.elements.push(Box::new(QueueStage {
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
    pub fn build(self, mut terminal: Box<dyn Sink>) -> Box<dyn Sink> {
        *terminal.hlog_mut() = element_hlog(
            terminal.element_type(),
            &terminal.name(),
            Some(&self.context.pipeline_id),
        );
        self.context
            .registry
            .register(terminal.element_type(), terminal.name(), self.last);
        let terminal: Box<dyn Sink> = Box::new(EosReporter {
            bus: self.context.bus.clone(),
            inner: terminal,
        });
        self.elements
            .into_iter()
            .rev()
            .fold(terminal, |downstream, stage| {
                stage.wrap(downstream, &self.context.bus, &self.context.pipeline_id)
            })
    }
}

/// Accumulates one or more sources into a single [`Pipeline`] — the
/// multi-source generalization of what [`Pipeline::new`] does for exactly
/// one. Each [`PipelineBuilder::add_source`] call gets its own background
/// thread once [`PipelineBuilder::build`]'s [`Pipeline::run`] starts, but
/// they all share one [`Bus`] (so [`Pipeline::bus`] sees every source's
/// events on one channel), one [`Clock`] (so every [`crate::elements::Pacer`]
/// anywhere in the pipeline — regardless of which source's chain it's
/// under — agrees on the same t=0/pause timeline), and one
/// [`ElementRegistry`] (so [`Pipeline::topology`] renders every source's
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
pub struct PipelineBuilder {
    id: Arc<str>,
    bus: Bus,
    bus_rx: BusReceiver,
    clock: Arc<Clock>,
    registry: ElementRegistry,
    sources: Vec<Box<dyn SourceElement>>,
    control_pairs: Vec<(ControlSender, ControlReceiver)>,
}

impl PipelineBuilder {
    pub fn new(id: impl Into<String>) -> Self {
        let id: Arc<str> = id.into().into();
        let (bus, bus_rx) = Bus::new();
        Self {
            id,
            bus,
            bus_rx,
            clock: Arc::new(Clock::new()),
            registry: ElementRegistry::new(),
            sources: Vec::new(),
            control_pairs: Vec::new(),
        }
    }

    /// Registers one more source. `wire` is called once, right away, with
    /// the freshly stamped source and a fresh [`Context`] whose
    /// `default_upstream` is *this* source's own name — so a
    /// [`ChainBuilder`] built from it defaults its first element to
    /// hanging off this source specifically, not whichever source was
    /// added first (see [`ChainBuilder::new`]'s own docs). Otherwise the
    /// same contract as [`Pipeline::new`]'s own `wire` parameter: build
    /// one or more `ChainBuilder` chains and link them via
    /// `source.src_pads()[i].link(...)`.
    pub fn add_source<S: SourceElement + 'static>(
        mut self,
        mut source: S,
        wire: impl FnOnce(&mut S, &Arc<Context>),
    ) -> Self {
        *source.hlog_mut() = element_hlog(source.element_type(), &source.name(), Some(&self.id));
        self.registry
            .register(source.element_type(), source.name(), None);
        let context = Arc::new(Context {
            bus: self.bus.clone(),
            pipeline_id: self.id.clone(),
            registry: self.registry.clone(),
            clock: self.clock.clone(),
            default_upstream: source.name(),
        });
        wire(&mut source, &context);
        self.sources.push(Box::new(source));
        self.control_pairs.push(control::channel());
        self
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
            bus_rx: self.bus_rx,
            running: AtomicUsize::new(0),
            registry: self.registry,
        })
    }
}

/// Top-level pipeline: one or more sources (see [`PipelineBuilder`], with
/// everything reachable from each source's own src pads already linked)
/// plus the bus every source reports events on and the [`Clock`] every
/// [`crate::elements::Pacer`] in it shares.
///
/// `run()` is asynchronous: it starts every source on its own background
/// thread and returns immediately, rather than blocking the caller for the
/// whole play-through. Always held as `Arc<Pipeline>` (that's what
/// [`Pipeline::new`]/[`PipelineBuilder::build`] return) — the background
/// threads need their own owning handle to outlive the `run()` call that
/// spawned them, and that's also what lets [`Pipeline::pause`]/
/// [`Pipeline::resume`]/[`Pipeline::stop`] be called from another thread
/// while it's running.
///
/// There's no separate "is it done yet" query or callback: watch
/// [`Pipeline::bus`] instead. [`BusReceiver::iter`]/
/// [`BusReceiver::log_events`] block until every [`Bus`] sender has been
/// dropped. Under the normal ownership path that happens once every
/// source's background thread (and everything reachable from it) has
/// fully finished, so draining the bus doubles as "wait for completion" —
/// with more than one source, that means waiting for *all* of them, not
/// just the first to reach `Eos`. A caller that clones the [`Context`]
/// supplied to a source's own `wire` closure also retains its `Bus`
/// sender; in that case bus draining intentionally remains blocked until
/// that extra context is dropped. A source-level failure (returned from
/// [`crate::element::SourceElement::run`] itself, as opposed to one
/// reported from inside a `Queue`) shows up there too, as a
/// [`BusEvent::Error`] under that source's own name, since there's no
/// synchronous return path left to carry it.
///
/// A `Pipeline` isn't reusable once `run()` has been called (whether it
/// finished via every source's natural `Eos` or [`Pipeline::stop`]) — a
/// second `run()` call is a no-op; build a fresh `Pipeline` for another
/// play-through.
pub struct Pipeline {
    /// This pipeline's own id — passed to [`Pipeline::new`]/
    /// [`PipelineBuilder::new`], stamped onto every source's own `hlog`
    /// there and onto every element that passes through a [`ChainBuilder`]
    /// built with it (see [`Pipeline::id`]).
    id: Arc<str>,
    sources: Mutex<Option<Vec<Box<dyn SourceElement>>>>,
    /// Taken (leaving `None` behind) the moment `run()` starts, and cloned
    /// once per source into that source's own background thread — so once
    /// a pipeline is running, `Pipeline` itself no longer holds a `Bus`
    /// sender directly. If it did, [`BusReceiver::iter`] could never
    /// observe every sender dropped (one would always still be sitting
    /// right here), and would block forever instead of unblocking once
    /// every source actually finishes.
    bus: Mutex<Option<Bus>>,
    /// One [`ControlSender`] per source, in the same order
    /// [`PipelineBuilder::add_source`] was called — [`Pipeline::stop`]/
    /// `pause`/`resume`/`seek` send to every one of these in turn (each
    /// `send` is its own synchronous rendezvous with that source's own
    /// control cascade — see [`crate::control::ControlSender::send`] — so
    /// this serializes across sources rather than fanning out in
    /// parallel; fine for the handful of sources this is meant for).
    control_txs: Vec<ControlSender>,
    /// Taken (leaving `None` behind) the moment `run()` starts, and moved
    /// one per thread — same reasoning as `bus` above. If `Pipeline` kept
    /// its own clone of each alive for its whole lifetime instead, that
    /// control channel's receiver side would never fully disconnect even
    /// after its thread has long since exited, so a
    /// [`Pipeline::stop`]/`pause`/`resume` racing that thread's own
    /// natural end (e.g. called right as it finishes on its own) could
    /// enqueue a `Request` nobody will ever read *or drop* — leaving
    /// [`crate::control::ControlSender::send`]'s rendezvous ack blocked
    /// forever instead of unblocked by the disconnect, the way it is the
    /// moment the *last* `ControlReceiver` clone actually goes away.
    control_rxs: Mutex<Option<Vec<ControlReceiver>>>,
    clock: Arc<Clock>,
    bus_rx: BusReceiver,
    /// How many source threads are still running — `0` before `run()` and
    /// again once every source's thread has finished. `AtomicUsize` rather
    /// than a per-source flag: every call site (`pause`/`resume`/`stop`/
    /// `seek`) only ever needs "is anything still running at all", never
    /// which specific source.
    running: AtomicUsize,
    /// Live registry backing [`Pipeline::elements`]/[`Pipeline::topology`] —
    /// kept for this `Pipeline`'s whole life (unlike `bus`/`control_rxs`
    /// above) so a branch registered long after construction (e.g. added to
    /// a running [`crate::elements::Tee`] via the [`Context`] owned by that
    /// `Tee`'s shared state) still shows up when either is called again.
    registry: ElementRegistry,
}

impl Pipeline {
    /// `id` names this pipeline — stamped as the source's own `hlog`
    /// sub_id right away, and folded into the [`Context`] handed to `wire`
    /// (see [`ChainBuilder`]'s own docs).
    ///
    /// `wire` is called once with the freshly created source and a
    /// [`Context`] bundling this pipeline's `Bus`, `id`, [`ElementRegistry`]
    /// (already seeded with the source itself), and `Clock` (share it with
    /// every [`crate::elements::Pacer`] via `Clock::clone` — one clock per
    /// pipeline, so every paced branch agrees on the same t=0 and the same
    /// pause/resume timeline) — everything a [`ChainBuilder`]/
    /// [`crate::elements::Tee`] needs, in one `Arc` clone instead of four
    /// separate arguments. `wire` builds one or more `ChainBuilder` chains
    /// (one per src pad that should actually be used) and links them via
    /// `source.src_pads()[i].link(...)`. Pads left unlinked just drop
    /// whatever gets pushed into them.
    ///
    /// The single-source special case of [`PipelineBuilder`] — see its own
    /// docs for combining more than one live source (e.g. a video capture
    /// and an audio capture) into one `Pipeline`.
    pub fn new<S: SourceElement + 'static>(
        id: impl Into<String>,
        source: S,
        wire: impl FnOnce(&mut S, &Arc<Context>),
    ) -> Arc<Self> {
        PipelineBuilder::new(id).add_source(source, wire).build()
    }

    /// This pipeline's own id, as passed to [`Pipeline::new`].
    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn bus(&self) -> &BusReceiver {
        &self.bus_rx
    }

    /// Every element wired into this pipeline so far (source included),
    /// each knowing what feeds it — see [`ElementInfo::upstream`]. Raw data
    /// behind [`Pipeline::topology`]; reach for that instead unless you
    /// actually need to walk the graph yourself.
    ///
    /// Freshly derived from the live [`ElementRegistry`] on every call — so
    /// a branch added to a [`crate::elements::Tee`] well after this
    /// `Pipeline` started running shows up here too, not just whatever was
    /// wired during `wire`. Only reflects wiring done through
    /// `ChainBuilder`/`Tee` — a `SrcPad::link` called by hand instead won't
    /// show up here.
    ///
    /// Every non-source entry's `upstream` is already resolved by the time
    /// it's registered — [`ChainBuilder::new`] seeds it from whichever
    /// source's [`Context`] built that chain, and
    /// [`crate::elements::TeeHandle::add_sink`] overrides it for a branch
    /// that turns out to start under a `Tee` instead — so this no longer
    /// needs a post-hoc "default to the source" pass the way it did before
    /// [`PipelineBuilder`] made more than one source (and so more than one
    /// possible default) possible.
    pub fn elements(&self) -> Vec<ElementInfo> {
        self.registry.snapshot()
    }

    /// Human-readable rundown of [`Pipeline::elements`]: one line per
    /// branch — each element nothing else in the graph feeds into (a
    /// terminal sink, or an empty [`crate::elements::Tee`] with no sinks
    /// attached yet) — formatted `Type(name) - Type(name) - ...` by
    /// walking that element's `upstream` chain back to the source.
    /// Multiple branches (fan-out across more than one src pad, or a
    /// `Tee`) are joined by newlines.
    pub fn topology(&self) -> String {
        let elements = self.elements();
        let by_name = |name: &str| elements.iter().find(|e| &*e.name == name);
        let is_leaf = |name: &str| !elements.iter().any(|e| e.upstream.as_deref() == Some(name));
        let render = |leaf: &ElementInfo| {
            let mut chain = vec![format!("{:?}({})", leaf.element_type, leaf.name)];
            let mut current = leaf;
            // Bounded by `elements.len()` rather than looping until
            // `upstream` runs out, purely defensive — a real cycle would
            // mean a bug elsewhere (`ChainBuilder`/`Tee` registration),
            // not something to hang over here.
            for _ in 0..elements.len() {
                let Some(upstream_name) = current.upstream.as_deref() else {
                    break;
                };
                let Some(upstream) = by_name(upstream_name) else {
                    break; // dangling reference — shouldn't normally happen
                };
                chain.push(format!("{:?}({})", upstream.element_type, upstream.name));
                current = upstream;
            }
            chain.reverse();
            chain.join(" - ")
        };
        elements
            .iter()
            .filter(|e| is_leaf(&e.name))
            .map(render)
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// The clock every `Pacer` in this pipeline paces against — see
    /// [`Pipeline::pause`] for why callers don't usually need to touch
    /// this directly.
    pub fn clock(&self) -> &Arc<Clock> {
        &self.clock
    }

    /// Starts driving the source on a background thread and returns
    /// immediately — see the type-level docs for how to learn when it's
    /// actually done. A no-op if this `Pipeline` is already running or
    /// has already finished a previous run — this type has no "reset"
    /// path; build a fresh `Pipeline` for another play-through.
    pub fn run(self: &Arc<Self>) {
        let Some(sources) = self.sources.lock().unwrap().take() else {
            return;
        };
        // Always `Some` in lockstep with `sources` above — all three taken
        // exactly once, on whichever `run()` call actually wins the
        // `sources` guard.
        let Some(bus) = self.bus.lock().unwrap().take() else {
            return;
        };
        let Some(control_rxs) = self.control_rxs.lock().unwrap().take() else {
            return;
        };

        self.running.store(sources.len(), Ordering::Release);
        for (mut source, control_rx) in sources.into_iter().zip(control_rxs) {
            let bus = bus.clone();
            let this = Arc::clone(self);
            thread::Builder::new()
                .name("pipeline:source".into())
                .spawn(move || {
                    hinfo!(main_id: &this.id, "pipeline: run starting ({})", source.name());
                    let source_name = source.name();
                    let source_type = source.element_type();
                    // `source.run()` itself already reports non-fatal,
                    // per-buffer failures to `bus` as it goes (see
                    // `SourceElement::run`'s docs) — a returned `Err` here
                    // means something genuinely ended this source, e.g.
                    // a `Seek` that failed outright.
                    let outcome = if let Err(error) = source.run(&control_rx, &bus) {
                        bus.post(
                            source.hlog(),
                            BusEvent::Error {
                                element_type: source_type,
                                name: source_name.clone(),
                                error,
                            },
                        );
                        "error"
                    } else {
                        "ok"
                    };
                    hinfo!(
                        main_id: &this.id,
                        "pipeline: run finished ({outcome}, {source_name})"
                    );
                    this.running.fetch_sub(1, Ordering::AcqRel);
                })
                .expect("failed to spawn pipeline source thread");
        }
    }

    /// Blocks until every element downstream of every source has paused —
    /// see [`crate::control::drain_control`] (source side) and
    /// [`crate::queue::Queue`]'s worker loop (each thread boundary). Also
    /// pauses this pipeline's `Clock`, so a `Pacer` doesn't see a jump in
    /// elapsed time once resumed. No-op if `run()` isn't currently in
    /// progress on another thread.
    pub fn pause(&self) {
        if self.running.load(Ordering::Acquire) == 0 {
            return;
        }
        self.clock.interrupt();
        for control_tx in &self.control_txs {
            control_tx.send(ControlMsg::Pause);
        }
        self.clock.pause();
    }

    /// Undoes [`Pipeline::pause`]. Resumes the `Clock` first, so it's
    /// already shifted forward by the time `Pacer`s start receiving
    /// frames again.
    pub fn resume(&self) {
        if self.running.load(Ordering::Acquire) == 0 {
            return;
        }
        self.clock.resume();
        for control_tx in &self.control_txs {
            control_tx.send(ControlMsg::Resume);
        }
    }

    /// Performs an early, full stop — abandons buffered work rather than
    /// draining to a natural `Eos`. This call is synchronous: it sends
    /// [`ControlMsg::Stop`] to every source in turn and waits for each
    /// one's own cascade to finish before moving to the next — sequential,
    /// not parallel, across sources (fine for the handful of sources this
    /// is meant for). It therefore cannot preempt an arbitrary
    /// source read or `Sink::consume` call already blocked inside user or
    /// external-library code; the call returns only after that work gives
    /// the control cascade a turn. After it returns, watch [`Pipeline::bus`]
    /// for every source's background thread to finish. Not reusable
    /// afterward — build a new `Pipeline` for the next play-through.
    pub fn stop(&self) {
        if self.running.load(Ordering::Acquire) == 0 {
            return;
        }
        self.clock.interrupt();
        for control_tx in &self.control_txs {
            control_tx.send(ControlMsg::Stop);
        }
    }

    /// Jumps to an absolute position from the start of the media. Blocks
    /// until every source has repositioned (see
    /// [`crate::element::SourceElement::seek`]) and every element
    /// downstream of each has reacted (a `Queue` drops its stale backlog, a
    /// decoder flushes, a `Pacer` re-anchors both its pts reference and
    /// this pipeline's `Clock`) — same synchronous cascade as `pause`/
    /// `resume`/`stop`. One-shot, unlike `pause`: nothing further to undo
    /// afterward, playback just continues from the new position. No-op
    /// if `run()` isn't currently in progress on another thread.
    ///
    /// Signals the clock's interrupt epoch before starting the synchronous
    /// cascade so a `Pacer` in a long wait can return its worker promptly.
    /// The clock's playback anchor is still reset later, inside
    /// [`crate::elements::Pacer::control`], after that in-flight frame is
    /// out of the way.
    ///
    /// A source that doesn't support seeking (e.g. a live capture) reports
    /// that via its own [`crate::element::SourceElement::seek`] returning
    /// an error — surfaced on [`Pipeline::bus`] as a
    /// [`BusEvent::Error`] under that source's name, same as any other
    /// per-source failure, rather than failing this call outright or
    /// skipping that source silently.
    pub fn seek(&self, target: Duration) {
        if self.running.load(Ordering::Acquire) == 0 {
            return;
        }
        self.clock.interrupt();
        for control_tx in &self.control_txs {
            control_tx.send(ControlMsg::Seek(target));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::atomic::AtomicUsize, thread, time::Duration};

    use super::*;
    use crate::{
        element::Source,
        elements::{
            FileDemuxer, Pacer, Tee, TestAudioOptions, TestAudioSource, TestVideoOptions,
            TestVideoSource,
        },
    };

    /// Real video file, one directory up from this crate — same one every
    /// example defaults to.
    fn test_video() -> &'static str {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../test-video/h265.mp4")
    }

    /// End-to-end: `run()` (async — starts the background thread and
    /// returns right away), then `pause()`/`stop()` (skipping `resume()`)
    /// from the test's own thread — exercises the whole cascade (source's
    /// `drain_control` loop -> `Queue`'s worker) at once, not just `Queue`
    /// in isolation (see `queue::tests`). Mainly guards against the
    /// deadlock this design is built to avoid: draining the bus
    /// afterward must return promptly, not hang forever waiting on a
    /// control message — or a `Bus` handle — that never arrives/drops.
    #[test]
    fn pause_then_stop_returns_promptly() {
        let (source, streams) = FileDemuxer::open("demux", test_video()).expect("open test video");
        let video = streams
            .iter()
            .find(|s| s.kind == ffmpeg_next::media::Type::Video)
            .expect("test video has a video stream");
        let index = video.index;

        let pipeline = Pipeline::new("test", source, |source, ctx| {
            let branch = ChainBuilder::new(ctx.clone())
                .queue("q", 4)
                .build(Box::new(NoOpSink {
                    name: "noop".into(),
                    hlog: element_hlog(ElementType::Other, "noop", None),
                }));
            source.src_pads()[index].link(branch);
        });

        pipeline.run();

        // Give the background thread a moment to actually start looping
        // so `pause()`/`stop()` land while `running` is true, not before.
        thread::sleep(Duration::from_millis(50));
        pipeline.pause();
        pipeline.stop();

        // Blocks until every `Bus` handle in the pipeline has been
        // dropped — i.e. until the background thread has actually
        // finished, not just acked `stop()`.
        let events: Vec<_> = pipeline.bus().iter().collect();
        assert!(
            !events.iter().any(|e| matches!(e, BusEvent::Error { .. })),
            "unexpected error event(s): {events:?}"
        );
    }

    /// [`PipelineBuilder`] with two independent, indefinitely-running
    /// sources (standing in for a real video capture + audio capture pair
    /// feeding one [`crate::elements::Mp4Muxer`]) sharing one `Pipeline`:
    /// both should show up in `topology()` under their *own* root, not
    /// both defaulted to whichever source was added first (the exact bug
    /// `Tee`'s own registration had before `default_upstream` existed —
    /// see [`Context::default_upstream`]'s docs), and a single `stop()`
    /// call must reach both — if it only reached one, the other source's
    /// thread would still be alive holding its own `Bus` sender clone
    /// open, and `pipeline.bus().iter().collect()` below would hang
    /// forever instead of returning.
    #[test]
    fn multi_source_pipeline_stops_every_source_from_one_stop_call() {
        let video = TestVideoSource::new("video", TestVideoOptions::default());
        let audio = TestAudioSource::new("audio", TestAudioOptions::default());

        let video_count = Arc::new(AtomicUsize::new(0));
        let audio_count = Arc::new(AtomicUsize::new(0));

        let pipeline = PipelineBuilder::new("multi-source-test")
            .add_source(video, {
                let count = video_count.clone();
                move |source, ctx| {
                    let branch = ChainBuilder::new(ctx.clone()).build(Box::new(CountingSink {
                        name: "video-sink".into(),
                        count,
                        hlog: element_hlog(ElementType::Other, "video-sink", None),
                    }));
                    source.src_pads()[0].link(branch);
                }
            })
            .add_source(audio, {
                let count = audio_count.clone();
                move |source, ctx| {
                    let branch = ChainBuilder::new(ctx.clone()).build(Box::new(CountingSink {
                        name: "audio-sink".into(),
                        count,
                        hlog: element_hlog(ElementType::Other, "audio-sink", None),
                    }));
                    source.src_pads()[0].link(branch);
                }
            })
            .build();

        let topology = pipeline.topology();
        let mut branches: Vec<&str> = topology.split('\n').collect();
        branches.sort_unstable();
        assert_eq!(
            branches,
            vec![
                "TestAudioSource(audio) - Other(audio-sink)",
                "TestVideoSource(video) - Other(video-sink)",
            ]
        );

        pipeline.run();
        thread::sleep(Duration::from_millis(100));
        pipeline.stop();

        // Would hang here if `stop()` only reached one of the two sources
        // — see this test's own docs.
        let events: Vec<_> = pipeline.bus().iter().collect();
        assert!(
            !events.iter().any(|e| matches!(e, BusEvent::Error { .. })),
            "unexpected error event(s): {events:?}"
        );
        assert!(
            video_count.load(Ordering::SeqCst) > 0,
            "video branch never received anything"
        );
        assert!(
            audio_count.load(Ordering::SeqCst) > 0,
            "audio branch never received anything"
        );
    }

    /// `seek()` mid-playback should reposition the source (no error from
    /// `Input::seek`), reset/flush everything downstream without
    /// deadlocking, and let packets keep flowing afterward.
    #[test]
    fn seek_repositions_and_playback_continues() {
        let (source, streams) = FileDemuxer::open("demux", test_video()).expect("open test video");
        let video = streams
            .iter()
            .find(|s| s.kind == ffmpeg_next::media::Type::Video)
            .expect("test video has a video stream");
        let index = video.index;
        let time_base = source.stream_time_base(index).expect("stream disappeared");

        let count = Arc::new(AtomicUsize::new(0));
        let sink = CountingSink {
            name: "counting-sink".into(),
            count: count.clone(),
            hlog: element_hlog(ElementType::Other, "counting-sink", None),
        };

        // A `Pacer` here isn't incidental: without it, this whole 10s/
        // 300-packet file races through in well under the 50ms sleep
        // below (no decode, no throttling), so `seek()` would land on an
        // already-finished pipeline and silently no-op — exactly the kind
        // of thing a weak `count > 0` assertion wouldn't have caught (see
        // `seek_reports_where_it_actually_landed_when_target_is_not_a_keyframe`
        // for how this was found).
        let pipeline = Pipeline::new("test", source, |source, ctx| {
            let pacer = Pacer::new("pacer", time_base, ctx.clock.clone());
            let branch = ChainBuilder::new(ctx.clone())
                .queue("q", 4)
                .pipe(pacer)
                .build(Box::new(sink));
            source.src_pads()[index].link(branch);
        });

        pipeline.run();
        thread::sleep(Duration::from_millis(50));
        pipeline.seek(Duration::from_secs(1));
        // Let packets flow again post-seek before tearing down.
        thread::sleep(Duration::from_millis(100));
        pipeline.stop();

        let events: Vec<_> = pipeline.bus().iter().collect();
        assert!(
            !events.iter().any(|e| matches!(e, BusEvent::Error { .. })),
            "unexpected error event(s): {events:?}"
        );
        assert!(
            count.load(Ordering::SeqCst) > 0,
            "expected at least one packet to arrive after the seek"
        );
        assert!(
            events.iter().any(|e| matches!(
                e,
                BusEvent::Seeked { requested, .. } if *requested == Duration::from_secs(1)
            )),
            "expected a Seeked event reporting the request; got {events:?}"
        );
    }

    /// Regression test for the bug found manually testing `rtsp_serve_seek`:
    /// a container seek can only land on a keyframe at or before `target`
    /// (see `FileDemuxer::seek`'s docs) — `test-video/h265.mp4` has
    /// keyframes only at 0s and ~8.33s, so any `target` between them has
    /// to land back at 0s, nowhere near what was requested. Without
    /// `BusEvent::Seeked` reporting that gap, this looked indistinguishable
    /// from `seek` silently doing nothing.
    #[test]
    fn seek_reports_where_it_actually_landed_when_target_is_not_a_keyframe() {
        let (source, streams) = FileDemuxer::open("demux", test_video()).expect("open test video");
        let video = streams
            .iter()
            .find(|s| s.kind == ffmpeg_next::media::Type::Video)
            .expect("test video has a video stream");
        let index = video.index;
        let time_base = source.stream_time_base(index).expect("stream disappeared");

        // Paced for the same reason as `seek_repositions_and_playback_continues`
        // — otherwise the file finishes before `seek()` is even called.
        let pipeline = Pipeline::new("test", source, |source, ctx| {
            let pacer = Pacer::new("pacer", time_base, ctx.clock.clone());
            let branch = ChainBuilder::new(ctx.clone())
                .queue("q", 4)
                .pipe(pacer)
                .build(Box::new(NoOpSink {
                    name: "noop".into(),
                    hlog: element_hlog(ElementType::Other, "noop", None),
                }));
            source.src_pads()[index].link(branch);
        });

        pipeline.run();
        thread::sleep(Duration::from_millis(50));
        // 3s falls inside the file's first (only) GOP before the 8.33s
        // keyframe — landing anywhere near 3s would mean this stream
        // unexpectedly grew more keyframes than it's known to have.
        pipeline.seek(Duration::from_secs(3));
        thread::sleep(Duration::from_millis(100));
        pipeline.stop();

        let events: Vec<_> = pipeline.bus().iter().collect();
        let seeked = events
            .iter()
            .find_map(|e| match e {
                BusEvent::Seeked {
                    requested, landed, ..
                } => Some((*requested, *landed)),
                _ => None,
            })
            .expect("expected a Seeked event");
        assert_eq!(seeked.0, Duration::from_secs(3));
        assert!(
            seeked.1 < Duration::from_secs(1),
            "expected landed to fall back to the 0s keyframe, got {:?}",
            seeked.1
        );
    }

    struct NoOpSink {
        name: Arc<str>,
        hlog: HLog,
    }
    impl Element for NoOpSink {
        fn name(&self) -> Arc<str> {
            self.name.clone()
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
    impl Sink for NoOpSink {
        fn consume(&mut self, _buf: MediaBuffer) -> Result<()> {
            Ok(())
        }
        fn control(&mut self, _msg: ControlMsg) -> Result<()> {
            Ok(())
        }
    }

    #[rust_hlog::hlog]
    struct CountingSink {
        name: Arc<str>,
        count: Arc<AtomicUsize>,
    }
    impl Element for CountingSink {
        fn name(&self) -> Arc<str> {
            self.name.clone()
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
        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            // Anything but `Eos` counts — covers `FileDemuxer`'s `Packet`s
            // (what every other test using this sink actually sends) and
            // `TestVideoSource`/`TestAudioSource`'s `Video`/`Audio` frames
            // (what `multi_source_pipeline_stops_every_source_from_one_stop_call`
            // sends) alike.
            if !buf.is_eos() {
                self.count.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        }
        fn control(&mut self, _msg: ControlMsg) -> Result<()> {
            Ok(())
        }
    }

    /// `ChainBuilder::build`'s terminal — and, transitively via `.pipe()`/
    /// `.queue()`, everything upstream of it — should come out tagged with
    /// the pipeline id it was built with, not left at `sub_id: None`.
    #[test]
    fn chain_builder_stamps_pipeline_id_into_terminal_hlog() {
        let (bus, _bus_rx) = Bus::new();
        let sink = NoOpSink {
            name: "noop".into(),
            hlog: element_hlog(ElementType::Other, "noop", None),
        };
        let context = Arc::new(Context {
            bus,
            pipeline_id: "my-pipeline".into(),
            registry: ElementRegistry::new(),
            clock: Arc::new(Clock::new()),
            default_upstream: "source".into(),
        });
        let built = ChainBuilder::new(context).build(Box::new(sink));
        assert_eq!(built.hlog().log_id(), "Other(noop):Pipeline(my-pipeline)");
    }

    /// `Pipeline::new`'s `id` should come back unchanged from
    /// [`Pipeline::id`], and the `wire` closure's own `ctx.pipeline_id`
    /// (used to build a matching `ChainBuilder`) should be that same
    /// value — not, say, whatever `source.name()` happens to be.
    #[test]
    fn pipeline_id_is_whatever_new_was_given() {
        let (source, streams) = FileDemuxer::open("demux", test_video()).expect("open test video");
        let video = streams
            .iter()
            .find(|s| s.kind == ffmpeg_next::media::Type::Video)
            .expect("test video has a video stream");
        let index = video.index;

        let pipeline = Pipeline::new("my-pipeline", source, |source, ctx| {
            let branch = ChainBuilder::new(ctx.clone()).build(Box::new(NoOpSink {
                name: "noop".into(),
                hlog: element_hlog(ElementType::Other, "noop", None),
            }));
            source.src_pads()[index].link(branch);
        });

        assert_eq!(pipeline.id(), "my-pipeline");
    }

    /// [`Pipeline::topology`] should render the source plus every element
    /// added via `.queue()`/`.pipe()` and the terminal, in order, joined by
    /// `" - "` — and one line per branch when more than one src pad is
    /// linked.
    #[test]
    fn topology_lists_source_through_terminal_per_branch() {
        let (source, streams) = FileDemuxer::open("demux", test_video()).expect("open test video");
        let video = streams
            .iter()
            .find(|s| s.kind == ffmpeg_next::media::Type::Video)
            .expect("test video has a video stream");
        let index = video.index;
        let time_base = source.stream_time_base(index).expect("stream disappeared");

        let pipeline = Pipeline::new("test", source, |source, ctx| {
            let pacer = Pacer::new("pacer", time_base, ctx.clock.clone());
            let branch = ChainBuilder::new(ctx.clone())
                .queue("q", 4)
                .pipe(pacer)
                .build(Box::new(NoOpSink {
                    name: "noop".into(),
                    hlog: element_hlog(ElementType::Other, "noop", None),
                }));
            source.src_pads()[index].link(branch);
        });

        assert_eq!(
            pipeline.topology(),
            "FileDemuxer(demux) - Queue(q) - Pacer(pacer) - Other(noop)"
        );

        // `pipeline` is dropped here without ever being `run()`, taking
        // its `.queue()`-spawned worker thread down with it — regression
        // coverage for the `Queue::drop` fix (see
        // `queue::tests::dropping_without_stop_or_eos_does_not_hang`):
        // this used to hang the test process forever.
    }

    /// A branch handed to [`TeeHandle::add_sink`] should render as starting
    /// under `Tee(...)`, not the pipeline's source — the whole reason
    /// [`Tee::new`]/[`TeeHandle::add_sink`] participate in the same
    /// [`ElementRegistry`] every `ChainBuilder` does.
    #[test]
    fn topology_attributes_tee_branches_to_the_tee_not_the_source() {
        let (source, streams) = FileDemuxer::open("demux", test_video()).expect("open test video");
        let video = streams
            .iter()
            .find(|s| s.kind == ffmpeg_next::media::Type::Video)
            .expect("test video has a video stream");
        let index = video.index;

        let pipeline = Pipeline::new("test", source, |source, ctx| {
            let branch_a = ChainBuilder::new(ctx.clone()).build(Box::new(NoOpSink {
                name: "sink-a".into(),
                hlog: element_hlog(ElementType::Other, "sink-a", None),
            }));
            let branch_b = ChainBuilder::new(ctx.clone()).build(Box::new(NoOpSink {
                name: "sink-b".into(),
                hlog: element_hlog(ElementType::Other, "sink-b", None),
            }));

            let (tee, tee_handle) = Tee::new("tee", ctx.clone());
            tee_handle.add_sink(branch_a);
            tee_handle.add_sink(branch_b);
            source.src_pads()[index].link(Box::new(tee));
        });

        let topology = pipeline.topology();
        let mut branches: Vec<&str> = topology.split('\n').collect();
        branches.sort_unstable();
        assert_eq!(
            branches,
            vec![
                "FileDemuxer(demux) - Tee(tee) - Other(sink-a)",
                "FileDemuxer(demux) - Tee(tee) - Other(sink-b)",
            ]
        );
    }

    /// Once a branch is pulled off a [`Tee`] via [`TeeHandle::remove_sink`],
    /// it should stop showing up in [`Pipeline::topology`] entirely — not
    /// keep rendering as still attached under `Tee(...)`, which is what a
    /// stale `ElementRegistry` entry would otherwise do (nothing else
    /// clears it on removal, since the registry doesn't know about
    /// `SrcPad::unlink` on its own).
    #[test]
    fn topology_forgets_a_branch_once_it_is_removed_from_the_tee() {
        let (source, streams) = FileDemuxer::open("demux", test_video()).expect("open test video");
        let video = streams
            .iter()
            .find(|s| s.kind == ffmpeg_next::media::Type::Video)
            .expect("test video has a video stream");
        let index = video.index;

        let mut tee_handle_slot = None;
        let pipeline = Pipeline::new("test", source, |source, ctx| {
            let branch_a = ChainBuilder::new(ctx.clone()).build(Box::new(NoOpSink {
                name: "sink-a".into(),
                hlog: element_hlog(ElementType::Other, "sink-a", None),
            }));
            let branch_b = ChainBuilder::new(ctx.clone()).build(Box::new(NoOpSink {
                name: "sink-b".into(),
                hlog: element_hlog(ElementType::Other, "sink-b", None),
            }));

            let (tee, tee_handle) = Tee::new("tee", ctx.clone());
            tee_handle.add_sink(branch_a);
            tee_handle.add_sink(branch_b);
            source.src_pads()[index].link(Box::new(tee));
            tee_handle_slot = Some(tee_handle);
        });

        let tee_handle = tee_handle_slot.expect("wire ran");
        tee_handle.remove_sink(0); // sink-a, added first

        assert_eq!(
            pipeline.topology(),
            "FileDemuxer(demux) - Tee(tee) - Other(sink-b)"
        );
    }

    /// Scale check beyond the 2-branch tests above: dozens of branches on
    /// one `Tee`, all present in `topology()`, then half removed — proves
    /// neither `ChainBuilder`'s eager registration nor `remove_subtree`'s
    /// fixpoint walk depend on branch count or removal order in some way
    /// the small tests wouldn't catch.
    #[test]
    fn topology_stays_correct_with_dozens_of_branches_added_and_then_removed() {
        let (source, streams) = FileDemuxer::open("demux", test_video()).expect("open test video");
        let video = streams
            .iter()
            .find(|s| s.kind == ffmpeg_next::media::Type::Video)
            .expect("test video has a video stream");
        let index = video.index;

        const N: usize = 30;
        let mut tee_handle_slot = None;
        let pipeline = Pipeline::new("test", source, |source, ctx| {
            let (tee, tee_handle) = Tee::new("tee", ctx.clone());
            for i in 0..N {
                let name: Arc<str> = format!("sink-{i}").into();
                let branch = ChainBuilder::new(ctx.clone()).build(Box::new(NoOpSink {
                    name: name.clone(),
                    hlog: element_hlog(ElementType::Other, &name, None),
                }));
                tee_handle.add_sink(branch);
            }
            source.src_pads()[index].link(Box::new(tee));
            tee_handle_slot = Some(tee_handle);
        });

        let tee_handle = tee_handle_slot.expect("wire ran");

        let mut branches: Vec<String> = pipeline.topology().lines().map(String::from).collect();
        branches.sort();
        let mut expected: Vec<String> = (0..N)
            .map(|i| format!("FileDemuxer(demux) - Tee(tee) - Other(sink-{i})"))
            .collect();
        expected.sort();
        assert_eq!(branches, expected, "all {N} branches should show up once");

        // Repeatedly removing index 0 always pulls off the current first
        // pad — since sinks were added in order 0..N, this removes
        // sink-0, then sink-1, ..., sink-(N/2 - 1).
        for _ in 0..N / 2 {
            tee_handle.remove_sink(0);
        }

        let mut remaining: Vec<String> = pipeline.topology().lines().map(String::from).collect();
        remaining.sort();
        let mut expected_remaining: Vec<String> = (N / 2..N)
            .map(|i| format!("FileDemuxer(demux) - Tee(tee) - Other(sink-{i})"))
            .collect();
        expected_remaining.sort();
        assert_eq!(
            remaining, expected_remaining,
            "only the un-removed half should remain, none of the removed ones lingering"
        );
    }
}
