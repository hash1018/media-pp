use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_info, pp_trace, pp_warn};

use crate::{
    bus::{Bus, BusEvent, BusReceiver},
    clock::Clock,
    control::{ControlMsg, ControlReceiver, ControlSender, PrerollContext, PrerollError},
    element::{Context, SourceElement},
    error::{Result, ThreadSpawnError},
    graph::{GraphSnapshot, NodeInfo, PipelineGraph, log_topology},
    playback_clock::PlaybackClock,
    playback_state::{Phase, PlaybackState},
    stats::PipelineStats,
};

use super::{PipelineBuilder, builder::SourceEntry};

/// A [`Pipeline`] asked for something its lifecycle no longer, or not yet,
/// allows. Converts into the crate-wide `Error` via `?` (see
/// [`crate::error::Error`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, ThisError)]
#[non_exhaustive]
pub enum PipelineError {
    /// [`Pipeline::run`] was called on a pipeline that had already been run.
    /// A pipeline runs once: build a new one for another play-through.
    #[error("this pipeline has already been run; build a new one to run again")]
    AlreadyStarted,

    /// [`Pipeline::seek`] was called before [`Pipeline::run`], or after every
    /// source had stopped â by [`Pipeline::stop`], [`Pipeline::finish`], or
    /// an error it could not continue past. A source parked at the end of its
    /// stream is still running, and can be sought back into.
    #[error("this pipeline is not running, so there is nothing to seek")]
    NotRunning,
}

/// How [`Pipeline::seek`] chooses the sample shown at the requested position.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SeekMode {
    /// Land at the preceding keyframe and preview the first decodable sample.
    Keyframe,
    /// Decode forward from the preceding keyframe and preview the sample that
    /// covers the requested timestamp.
    Accurate,
}

/// Top-level pipeline: one or more sources (see [`PipelineBuilder`], with
/// everything reachable from each source's own src pads already linked)
/// plus the bus every source reports events on and the [`Clock`] every
/// [`crate::elements::Pacer`] in it shares.
///
/// `run()` is asynchronous: it starts every source on its own background
/// thread and returns immediately, rather than blocking the caller for the
/// whole play-through. It returns a
/// [`ThreadSpawnError`](crate::error::ThreadSpawnError) if a source worker
/// cannot be created; any workers already created for that call are stopped
/// and joined before the error is returned. The one-shot pipeline is not
/// reusable after that failure. Returned as `Arc<Pipeline>` (that's what
/// [`Pipeline::new`]/[`PipelineBuilder::build`] return) â the background
/// threads deliberately do not retain an owning handle, so dropping the
/// last external `Arc` can stop them. The `Arc` also lets [`Pipeline::pause`]/
/// [`Pipeline::resume`]/[`Pipeline::stop`] be called from another thread
/// while it's running.
///
/// There's no separate "is it done yet" query or callback: watch
/// [`Pipeline::bus`] instead. [`BusReceiver::iter`]/
/// [`BusReceiver::log_events`] block until every [`Bus`] sender has been
/// dropped. Under the normal ownership path that happens once every
/// source's background thread (and everything reachable from it) has
/// fully finished â with more than one source, *all* of them, not just the
/// first to reach `Eos`. A source that can be sought back, `FileDemuxer`
/// among them, does not finish at the end of its media but stays there
/// until stopped; [`BusEvent::Finished`](crate::bus::BusEvent::Finished) is
/// what says the pipeline has played everything, and where a caller stops
/// it. A caller that clones the [`Context`]
/// supplied to a source's own `wire` closure also retains its `Bus`
/// sender; in that case bus draining intentionally remains blocked until
/// that extra context is dropped. A source-level failure (returned from
/// [`crate::element::SourceElement::run`] itself, as opposed to one
/// reported from inside a `Queue`) shows up there too, as a
/// [`BusEvent::Error`] under that source's own name, since there's no
/// synchronous return path left to carry it.
///
/// A `Pipeline` isn't reusable once `run()` has been called (whether it
/// finished via every source's natural `Eos`, [`Pipeline::finish`], or
/// [`Pipeline::stop`]) â a
/// second `run()` call is a no-op; build a fresh `Pipeline` for another
/// play-through.
pub struct Pipeline {
    /// This pipeline's own id â passed to [`Pipeline::new`]/
    /// [`PipelineBuilder::new`], stamped onto every source's own `pp_log`
    /// there and onto every element that passes through a [`super::ChainBuilder`]
    /// built with it (see [`Pipeline::id`]).
    pub(super) id: Arc<str>,
    /// Logging identity for pipeline-level topology records.
    pub(super) pp_log: PpLog,
    pub(super) sources: Mutex<Option<Vec<SourceEntry>>>,
    /// Taken (leaving `None` behind) the moment `run()` starts, and cloned
    /// once per source into that source's own background thread â so once
    /// a pipeline is running, `Pipeline` itself no longer holds a `Bus`
    /// sender directly. If it did, [`BusReceiver::iter`] could never
    /// observe every sender dropped (one would always still be sitting
    /// right here), and would block forever instead of unblocking once
    /// every source actually finishes.
    pub(super) bus: Mutex<Option<Bus>>,
    /// One [`ControlSender`] per source, in the same order
    /// [`PipelineBuilder::add_source`] was called â [`Pipeline::finish`]/
    /// `stop`/`pause`/`resume`/`seek` queue a request on every one of these
    /// before waking anything, and then wait for all of them together â see
    /// `Pipeline::broadcast` for why both halves of that matter.
    pub(super) control_txs: Vec<ControlSender>,
    /// Taken (leaving `None` behind) the moment `run()` starts, and moved
    /// one per thread â same reasoning as `bus` above. If `Pipeline` kept
    /// its own clone of each alive for its whole lifetime instead, that
    /// control channel's receiver side would never fully disconnect even
    /// after its thread has long since exited, so a
    /// [`Pipeline::stop`]/`pause`/`resume` racing that thread's own
    /// natural end (e.g. called right as it finishes on its own) could
    /// enqueue a `Request` nobody will ever read *or drop* â leaving
    /// [`crate::control::ControlSender::send`]'s rendezvous ack blocked
    /// forever instead of unblocked by the disconnect, the way it is the
    /// moment the *last* `ControlReceiver` clone actually goes away.
    pub(super) control_rxs: Mutex<Option<Vec<ControlReceiver>>>,
    pub(super) clock: Arc<Clock>,
    pub(super) playback_clock: Arc<PlaybackClock>,
    /// Where playback stands, for every element to read â see
    /// [`crate::playback_state`]. This is the only thing that writes it,
    /// and always before it sends the control message that goes with the
    /// change.
    pub(super) state: Arc<PlaybackState>,
    pub(super) bus_rx: BusReceiver,
    /// How many source threads are still running â `0` before `run()` and
    /// again once every source's thread has finished. `AtomicUsize` rather
    /// than a per-source flag: every call site (`pause`/`resume`/`stop`/
    /// `seek`) only ever needs "is anything still running at all", never
    /// which specific source.
    pub(super) running: Arc<AtomicUsize>,
    /// Tracks whether `Pipeline::pause` has completed without a matching
    /// resume. This cannot be inferred from `Clock`: pausing before the first
    /// media timestamp leaves an unset clock unchanged while downstream
    /// queues are nevertheless paused.
    pub(super) paused: AtomicBool,
    /// Serializes public lifecycle/timeline operations. `paused` remains the
    /// caller-requested state while seek temporarily pauses the runtime.
    pub(super) operation: Arc<Mutex<()>>,
    /// The preroll a [`Pipeline::seek`] is currently waiting on, and whether
    /// the pipeline has since been abandoned.
    ///
    /// Held here rather than only inside `seek` so a caller that wants the
    /// pipeline to end can reach it *without* the operation lock. `Stop` would
    /// otherwise have to queue behind that wait, which is the one thing it
    /// promises not to do â and the cancellation the terminals already forward
    /// on `Stop` cannot arrive either, because sending it needs the same lock.
    pub(super) preroll_slot: Mutex<PrerollSlot>,
    /// Handles for every source thread started by [`Pipeline::run`]. They
    /// are retained so dropping the pipeline can synchronously stop and
    /// join live sources instead of leaving detached work behind.
    pub(super) workers: Mutex<Vec<JoinHandle<()>>>,
    /// Live node/edge graph backing snapshots and topology rendering.
    pub(super) graph: PipelineGraph,
    /// Each source's counters, kept alive here â see
    /// [`PipelineBuilder`]'s own field.
    pub(super) _source_counters: Vec<Arc<crate::stats::ElementCounters>>,
}

impl Pipeline {
    /// `id` names this pipeline â stamped into the source's own `pp_log` as
    /// its `pipeline_id` right away, and folded into the [`Context`] handed
    /// to `wire` (see [`super::ChainBuilder`]'s own docs).
    ///
    /// `wire` is called once with the freshly created source and a
    /// [`Context`] bundling this pipeline's `Bus`, `id`, [`PipelineGraph`]
    /// (already seeded with the source itself), and `Clock` (share it with
    /// every [`crate::elements::Pacer`] via `Clock::clone` â one clock per
    /// pipeline, so every paced branch agrees on the same t=0 and the same
    /// pause/resume timeline) â everything a [`super::ChainBuilder`]/
    /// [`crate::elements::Tee`] needs, in one `Arc` clone instead of four
    /// separate arguments. `wire` creates detached chains and attaches
    /// them through [`Context::attach`]. Pads left unattached drop data.
    ///
    /// Whatever `wire` returns comes back beside the pipeline: something
    /// only the wiring can make â a [`crate::elements::TeeHandle`] from
    /// [`crate::elements::TeeBuilder::build_dynamic`], a routing to keep â
    /// is returned from it, several at once as a tuple, rather than
    /// smuggled out through a variable the closure fills in. `()` where
    /// there is nothing to hand back:
    ///
    /// ```ignore
    /// let (pipeline, ()) = Pipeline::new("play", source, |source, ctx| { /* ... */ Ok(()) })?;
    /// let (pipeline, tee) = Pipeline::new("fan", source, |source, ctx| {
    ///     let (branch, tee) = ctx.tee("tee").build_dynamic()?;
    ///     ctx.attach(source, 0, branch)?;
    ///     Ok(tee)
    /// })?;
    /// ```
    ///
    /// The single-source special case of [`PipelineBuilder`] â see its own
    /// docs for combining more than one live source (e.g. a video capture
    /// and an audio capture) into one `Pipeline`.
    pub fn new<S: SourceElement + 'static, T>(
        id: impl Into<String>,
        source: S,
        wire: impl FnOnce(&mut S, &Arc<Context>) -> Result<T>,
    ) -> Result<(Arc<Self>, T)> {
        let (builder, wired) = PipelineBuilder::new(id).add_source(source, wire)?;
        Ok((builder.build(), wired))
    }

    /// This pipeline's own id, as passed to [`Pipeline::new`].
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the receiver for asynchronous element and thread-boundary
    /// events produced by this pipeline.
    ///
    /// Calling [`BusReceiver::iter`](crate::bus::BusReceiver::iter) blocks
    /// until every sender has dropped, which normally coincides with all source
    /// and queue workers finishing â for a seekable source such as a file,
    /// once the pipeline is stopped; see
    /// [`BusEvent::Finished`](crate::bus::BusEvent::Finished). A custom element that retains a cloned
    /// [`Context`] can intentionally keep the receiver connected longer.
    pub fn bus(&self) -> &BusReceiver {
        &self.bus_rx
    }

    /// Returns a consistent node/edge snapshot of the live graph. Detached
    /// branches do not appear; a successful attach or detach increments its
    /// revision exactly once.
    pub fn graph(&self) -> GraphSnapshot {
        self.graph.snapshot()
    }

    /// Returns the nodes in a consistent snapshot of the currently attached
    /// graph.
    ///
    /// The returned values are owned copies and do not hold graph locks.
    /// Detached branch plans are absent until attachment succeeds, and removed
    /// branches disappear from later calls.
    pub fn elements(&self) -> Vec<NodeInfo> {
        self.graph().nodes
    }

    /// Human-readable rundown of [`Pipeline::elements`]: one line per
    /// branch â each element nothing else in the graph feeds into (a
    /// terminal sink, or an empty [`crate::elements::Tee`] with no sinks
    /// attached yet) â formatted `Type(name) - Type(name) - ...` by
    /// walking that element's `upstream` chain back to the source.
    /// Multiple branches (fan-out across more than one src pad, or a
    /// `Tee`) are joined by newlines.
    pub fn topology(&self) -> String {
        self.graph().topology()
    }

    /// What every element is doing, read now â see [`crate::stats`].
    ///
    /// Running totals rather than rates: take two readings and the rate is
    /// their difference over the time between them, matched by
    /// [`ElementStats::id`](crate::stats::ElementStats::id). Cheap enough to
    /// call a few times a second â the graph's lock is held only to copy
    /// the list of what is registered, and nothing on the path a buffer
    /// travels is locked by it.
    ///
    /// A branch finished with
    /// [`TeeHandle::finish_branch`](crate::elements::TeeHandle::finish_branch)
    /// goes on appearing, as
    /// [`ElementState::Finishing`](crate::stats::ElementState::Finishing),
    /// until it has drained and been dropped.
    pub fn stats(&self) -> PipelineStats {
        let (revision, registered, attached) = self.graph.registered();
        PipelineStats {
            revision,
            paused: self.paused.load(Ordering::Acquire),
            elements: crate::stats::read(registered, |id| attached.contains(&id)),
        }
    }

    /// The clock every `Pacer` in this pipeline paces against â see
    /// [`Pipeline::pause`] for why callers don't usually need to touch
    /// this directly.
    pub fn clock(&self) -> &Arc<Clock> {
        &self.clock
    }

    /// Media-position clock shared by audio output and video scheduling.
    pub fn playback_clock(&self) -> &Arc<PlaybackClock> {
        &self.playback_clock
    }

    /// Where playback is: the media time the pipeline's playback master has
    /// reached â the audio renderer's played samples when there is one, else
    /// the wall clock a [`crate::elements::Pacer`] or
    /// [`crate::elements::VideoSynchronizer`] keeps. What a progress bar
    /// shows, read from the clock rather than from whichever frame last went
    /// past, so it moves with the sound and holds still while paused.
    ///
    /// `None` while nothing paces this pipeline â a transcode runs as fast as
    /// it can and has no "now" â and from a [`Pipeline::seek`] until the
    /// first sample of the new position anchors the clock again. A position
    /// before the start of the media reads as zero.
    pub fn position(&self) -> Option<Duration> {
        self.playback_clock
            .position_ns()
            .map(|ns| Duration::from_nanos(ns.max(0) as u64))
    }

    /// Whether any source of this pipeline is still on a thread of its own.
    ///
    /// `false` before [`Pipeline::run`], and `true` from then until every
    /// source has finished â by its own `Eos`, by [`Pipeline::stop`] or
    /// [`Pipeline::finish`], or by returning an error it could not continue
    /// past. Those endings look different on the bus and identical here,
    /// which is what a caller wanting only "is this still producing?" is
    /// asking: a live capture whose target went away has to be noticed by
    /// whoever might reopen it, and that caller has no reason to care which
    /// way it ended.
    ///
    /// A source that can be sought back does not finish by its own `Eos`: a
    /// `FileDemuxer` stays at the end of its file, and this stays `true`,
    /// until the pipeline is stopped. For one of those,
    /// [`BusEvent::Finished`](crate::bus::BusEvent::Finished) is what says it
    /// has played everything.
    ///
    /// Draining the bus stays the way to learn *why* â see this type's own
    /// docs â and remains the only way to wait for the end rather than poll
    /// for it.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire) > 0
    }

    /// Starts driving the source on a background thread and returns
    /// immediately â see the type-level docs for how to learn when it's
    /// actually done. A pipeline runs once: calling this again, whether it
    /// is still running or has ended, fails with
    /// [`PipelineError::AlreadyStarted`] and changes nothing â this type has
    /// no "reset" path; build a fresh `Pipeline` for another play-through.
    ///
    /// A pipeline [`Pipeline::pause`]d before this starts paused: every
    /// source stops before producing anything, and this returns once they
    /// all have. A [`Pipeline::seek`] then puts exactly one sample through
    /// every terminal â the one at the requested position â and returns
    /// once it has arrived, which is how to take a single frame from part
    /// way into a file without decoding everything before it.
    ///
    /// If a source worker cannot be created, any source workers already
    /// started by this call are stopped and joined before the error returns.
    /// The one-shot pipeline is not reusable after that failure.
    pub fn run(&self) -> Result<()> {
        self.run_with_spawner(|thread_name, task| {
            thread::Builder::new().name(thread_name).spawn(task)
        })
    }

    pub(super) fn run_with_spawner(
        &self,
        mut spawn: impl FnMut(
            String,
            Box<dyn FnOnce() + Send + 'static>,
        ) -> std::io::Result<JoinHandle<()>>,
    ) -> Result<()> {
        // Held throughout, so a `pause` racing this either lands before the
        // sources are taken â and this starts paused â or after, as the
        // ordinary cascade.
        let _operation = lock(&self.operation);
        let Some(sources) = lock(&self.sources).take() else {
            return Err(PipelineError::AlreadyStarted.into());
        };
        // Always `Some` in lockstep with `sources` above â all three taken
        // exactly once, on whichever `run()` call actually wins the
        // `sources` guard.
        let Some(bus) = lock(&self.bus).take() else {
            return Err(PipelineError::AlreadyStarted.into());
        };
        let Some(control_rxs) = lock(&self.control_rxs).take() else {
            return Err(PipelineError::AlreadyStarted.into());
        };

        if crate::log::enabled(crate::log::Level::Info) {
            log_topology(&self.pp_log, "run", &self.graph());
        }
        // Paused before it ran: `Pause` goes into every source's control
        // channel ahead of anything else, before its thread exists, so each
        // one's first look at its control stops it before it produces. Its
        // acknowledgements are collected once the threads are running.
        let start_paused = self.paused.load(Ordering::Acquire);
        let mut pause_acks = Vec::new();
        if start_paused {
            pp_trace!(
                pp_log: &self.pp_log,
                "event=control control=Pause phase=requested reason=start_paused"
            );
            self.state.pause();
            self.clock.pause();
            pause_acks = self
                .control_txs
                .iter()
                .filter_map(|control_tx| control_tx.enqueue(ControlMsg::Pause))
                .collect();
        }
        let source_count = sources.len();
        self.running.store(source_count, Ordering::Release);
        for (index, ((source_id, source), control_rx)) in
            sources.into_iter().zip(control_rxs).enumerate()
        {
            let bus = bus.for_element(source_id);
            let running = Arc::clone(&self.running);
            let state = Arc::clone(&self.state);
            let thread_name = "pipeline:source".to_owned();
            let spawn_result = spawn(
                thread_name.clone(),
                Box::new(move || {
                    // Keep these as locals in this order. During unwinding the
                    // guard is dropped first, then the receiver, then the
                    // source. That makes a Pipeline indirectly retained by a
                    // custom source safe to drop from this worker thread.
                    let mut source = source;
                    let control_rx = control_rx;
                    let _running = RunningSourceGuard::new(running);
                    // What this source makes is on the pipeline's timeline,
                    // and moves to each new one as it applies the `Seek`.
                    crate::timeline::enter(&state);

                    let source_name = source.name();
                    let source_type = source.element_type();
                    // `source.run()` itself already reports non-fatal,
                    // per-buffer failures to `bus` as it goes (see
                    // `SourceElement::run`'s docs) â a returned `Err` here
                    // means something genuinely ended this source, e.g.
                    // a `Seek` that failed outright.
                    let outcome = if let Err(error) = source.run(&control_rx, &bus) {
                        bus.post(
                            source.pp_log(),
                            BusEvent::Error {
                                element_type: source_type,
                                name: source_name.clone(),
                                error,
                            },
                        );
                        // Nothing has told this source's branch that it is
                        // over: `run` returned instead of being stopped, and
                        // dropping it merely tears the elements down. A muxer
                        // waiting on this track would then never write its
                        // trailer, leaving an unplayable file â so cascade the
                        // same `Stop` a deliberate shutdown would have sent.
                        // `Stop` rather than `Eos` because the source failed:
                        // there is no complete stream to drain, only state to
                        // finalize.
                        // The log identity is cloned first: `src_pads` borrows
                        // the source mutably for the whole loop.
                        let source_log = source.pp_log().clone();
                        for pad in source.src_pads() {
                            if let Err(error) = pad.control(&ControlMsg::Stop) {
                                pp_warn!(
                                    pp_log: &source_log,
                                    "failed to stop the branch after a source error: {error}"
                                );
                            }
                        }
                        "error"
                    } else {
                        "ok"
                    };
                    pp_info!(pp_log: source.pp_log(), "finished outcome={outcome}");
                }),
            );
            match spawn_result {
                Ok(handle) => lock(&self.workers).push(handle),
                Err(source) => {
                    // This pipeline is one-shot, so sources already started
                    // cannot be reconstructed for a retry. Stop and join them,
                    // then account for the failed and not-yet-started sources
                    // so callers never observe a half-running pipeline after
                    // this error returns.
                    self.running
                        .fetch_sub(source_count - index, Ordering::AcqRel);
                    // A started source waiting to hand back its `Pause`
                    // acknowledgement is let go first, so it can take the
                    // `Stop` below.
                    drop(std::mem::take(&mut pause_acks));
                    self.state.interrupt();
                    for control_tx in self.control_txs.iter().take(index) {
                        control_tx.send(ControlMsg::Stop);
                    }
                    self.join_workers();
                    return Err(ThreadSpawnError::new(thread_name, source).into());
                }
            }
        }
        if start_paused {
            for ack in pause_acks {
                let _ = ack.recv();
            }
            pp_trace!(
                pp_log: &self.pp_log,
                "event=control control=Pause phase=completed outcome=ok reason=start_paused"
            );
        }
        Ok(())
    }

    /// Blocks until every element downstream of every source has paused â
    /// see [`crate::control::drain_control`] (source side) and
    /// [`crate::queue::Queue`]'s worker loop (each thread boundary). Also
    /// pauses this pipeline's `Clock` before that synchronous cascade
    /// starts, so time spent waiting for a busy downstream element to
    /// acknowledge `Pause` is frozen too and a `Pacer` doesn't see a jump
    /// once resumed.
    ///
    /// Before [`Pipeline::run`], this makes the run start paused â see its
    /// docs. Once every source has stopped, there is nothing to pause and
    /// this does nothing.
    pub fn pause(&self) {
        let _operation = self
            .operation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.running.load(Ordering::Acquire) == 0 {
            if self.not_yet_run() {
                self.paused.store(true, Ordering::Release);
            }
            return;
        }
        self.paused.store(true, Ordering::Release);
        self.pause_runtime();
    }

    /// Whether [`Pipeline::run`] has yet to take the sources.
    fn not_yet_run(&self) -> bool {
        lock(&self.sources).is_some()
    }

    /// Puts a request on every source's control channel, wakes whatever is
    /// waiting on the clock if `wake` says to, and then waits until every
    /// source has handled its request.
    ///
    /// In that order, and for every source at once. A `Pacer` or
    /// `VideoSynchronizer` interrupted mid-wait keeps its buffer and returns,
    /// so its source can take the request that interrupted it â which has to
    /// be there already. Sent one source at a time, as this used to be, a
    /// source waiting its turn behind another's cascade was interrupted with
    /// nothing to take: it read on, each buffer handed straight back by the
    /// still-interrupted pacer, and reached the end of its file in
    /// milliseconds. Waiting for them together also keeps one slow cascade
    /// from holding up the rest.
    ///
    /// Every request interrupts, whatever it is. Only some used to â pause,
    /// stop, finish, and the question that opens a seek â on the reasoning
    /// that the rest are sent while everything is already paused, so nothing
    /// can be waiting on the clock or blocked handing data on. That held
    /// until a source was not paused when it should have been (208af56):
    /// then a `Preroll` found it blocked handing a full, paused queue a
    /// packet, nothing let it go, and the seek waited on it for good. An
    /// interrupt is what lets such a thread go â a `Queue` takes the packet
    /// past its capacity, a `Pacer` lets go of its wait â so every request raises
    /// one, and none depends on the graph already being in the state the
    /// request assumes. It costs a request sent while all is paused nothing:
    /// there is nothing waiting for it to wake.
    fn broadcast(
        &self,
        enqueue: impl Fn(&ControlSender) -> Option<crossbeam_channel::Receiver<()>>,
    ) {
        let acks: Vec<_> = self.control_txs.iter().filter_map(enqueue).collect();
        self.state.interrupt();
        for ack in acks {
            let _ = ack.recv();
        }
        // Every source has taken the request and cascaded it, so whatever an
        // interrupt â this one or one raised just before â was holding back
        // can go on.
        self.state.settle();
    }

    fn pause_runtime(&self) {
        let msg = ControlMsg::Pause;
        pp_trace!(
            pp_log: &self.pp_log,
            "event=control control={msg:?} phase=requested"
        );
        self.state.pause();
        self.clock.pause();
        self.broadcast(|control_tx| control_tx.enqueue(msg.clone()));
        pp_trace!(
            pp_log: &self.pp_log,
            "event=control control={msg:?} phase=completed outcome=ok"
        );
    }

    /// Undoes [`Pipeline::pause`]. Resumes the `Clock` first, so it's
    /// already shifted forward by the time `Pacer`s start receiving
    /// frames again. Before [`Pipeline::run`], this undoes a `pause` made then, so
    /// the run starts playing.
    pub fn resume(&self) {
        let _operation = self
            .operation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.running.load(Ordering::Acquire) == 0 {
            if self.not_yet_run() {
                self.paused.store(false, Ordering::Release);
            }
            return;
        }
        self.paused.store(false, Ordering::Release);
        self.resume_runtime();
    }

    fn resume_runtime(&self) {
        let msg = ControlMsg::Resume;
        pp_trace!(
            pp_log: &self.pp_log,
            "event=control control={msg:?} phase=requested"
        );
        self.clock.resume();
        self.state.enter(Phase::Playing);
        self.broadcast(|control_tx| control_tx.enqueue(msg.clone()));
        pp_trace!(
            pp_log: &self.pp_log,
            "event=control control={msg:?} phase=completed outcome=ok"
        );
    }

    /// Performs an early, full stop â abandons buffered work rather than
    /// draining to a natural `Eos`. This call is synchronous: it sends
    /// [`ControlMsg::Stop`] to every source at once and waits until each
    /// one's own cascade has finished. It therefore cannot preempt an arbitrary
    /// source read or `Sink::consume` call already blocked inside user or
    /// external-library code; the call returns only after that work gives
    /// the control cascade a turn. After it returns, watch [`Pipeline::bus`]
    /// for every source's background thread to finish. Not reusable
    /// afterward â build a new `Pipeline` for the next play-through.
    pub fn stop(&self) {
        // Before the operation lock, not after: a seek holds that lock for as
        // long as its preroll wait, and an abandoning caller must not be made
        // to sit through it. Cancelling first turns that wait into an
        // immediate return, so this only waits for the seek's own control
        // cascade to unwind.
        self.abandon_preroll();
        let _operation = self
            .operation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.running.load(Ordering::Acquire) == 0 {
            return;
        }
        self.stop_runtime();
    }

    /// Announces the preroll a seek is about to wait on, cancelling it
    /// immediately if the pipeline has already been abandoned.
    ///
    /// That second case is not hypothetical: `stop` runs before the operation
    /// lock, so it can arrive in the window between the seek repositioning its
    /// sources and reaching this call. One mutex covers both sides â either
    /// `stop` finds the preroll here, or this finds `stop`'s flag.
    fn publish_preroll(&self, preroll: &Arc<PrerollContext>) {
        let mut slot = self
            .preroll_slot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if slot.abandoned {
            preroll.cancel();
            return;
        }
        slot.active = Some(Arc::clone(preroll));
    }

    fn retire_preroll(&self) {
        self.preroll_slot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .active = None;
    }

    /// Waits for `preroll`, rechecking the topology whenever it has not
    /// finished yet.
    ///
    /// The expected terminals are fixed when the seek starts; the graph is
    /// not. Detaching a `Tee` branch mid-seek removes its terminal without
    /// removing the obligation to hear from it, and nothing is left to report
    /// a sample for it. Rather than lock topology changes out for the whole
    /// wait, this simply stops expecting whoever has since left â which also
    /// covers any other way a terminal can disappear, not just that one.
    ///
    /// The graph snapshot only happens on a poll that found work still
    /// pending, so a preroll that completes promptly never takes one.
    fn await_preroll(
        &self,
        preroll: &PrerollContext,
        timeout: Duration,
    ) -> std::result::Result<(), PrerollError> {
        const TOPOLOGY_POLL_INTERVAL: Duration = Duration::from_millis(50);

        let deadline = Instant::now() + timeout;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            let slice = remaining.min(TOPOLOGY_POLL_INTERVAL);
            match preroll.wait(slice) {
                Err(PrerollError::TimedOut { pending }) => {
                    let live = self.graph().terminal_ids();
                    for terminal in pending
                        .iter()
                        .map(|node| node.id)
                        .filter(|terminal| !live.contains(terminal))
                    {
                        pp_trace!(
                            pp_log: &self.pp_log,
                            "event=control control=Preroll phase=pending \
                             outcome=departed terminal={terminal:?}"
                        );
                        preroll.mark_departed(terminal);
                    }
                    if remaining <= TOPOLOGY_POLL_INTERVAL {
                        // Deadline reached; report what is still owed, minus
                        // anything the prune above just resolved.
                        return preroll.wait(Duration::ZERO);
                    }
                }
                outcome => return outcome,
            }
        }
    }

    /// Ends an in-flight seek's preroll wait and refuses any that starts
    /// afterwards. Safe with none in flight, and deliberately takes no other
    /// lock: the whole point is to run *before* the operation lock a seek is
    /// holding.
    fn abandon_preroll(&self) {
        let preroll = {
            let mut slot = self
                .preroll_slot
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            slot.abandoned = true;
            slot.active.take()
        };
        if let Some(preroll) = preroll {
            preroll.cancel();
        }
    }

    fn stop_runtime(&self) {
        let msg = ControlMsg::Stop;
        self.paused.store(false, Ordering::Release);
        self.state.enter(Phase::Stopped);
        pp_trace!(
            pp_log: &self.pp_log,
            "event=control control={msg:?} phase=requested"
        );
        self.broadcast(|control_tx| control_tx.enqueue(msg.clone()));
        pp_trace!(
            pp_log: &self.pp_log,
            "event=control control={msg:?} phase=completed outcome=ok"
        );
    }

    /// Gracefully completes every source and waits for the whole graph to
    /// drain. Each source stops producing and places `MediaBuffer::Eos` behind
    /// its already-produced data; queues preserve that order, stateful codecs
    /// flush delayed output, and muxers finalize only after their EOS arrives.
    ///
    /// Unlike [`Pipeline::stop`], this does not abandon queued work. If the
    /// pipeline is paused, it resumes the control cascade first so a full
    /// paused queue cannot prevent its ordered EOS from being enqueued. The
    /// call returns only after every source thread (and the Queue workers each
    /// source owns) has finished. The pipeline is not reusable afterward.
    pub fn finish(&self) {
        // Same reasoning as `Pipeline::stop`: an in-flight seek's preroll wait
        // is about to be discarded by this completion, so there is nothing to
        // gain by sitting through it first.
        self.abandon_preroll();
        let _operation = self
            .operation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.running.load(Ordering::Acquire) == 0 {
            self.join_workers();
            return;
        }

        pp_trace!(
            pp_log: &self.pp_log,
            "event=finish phase=requested"
        );
        if self.paused.load(Ordering::Acquire) {
            self.paused.store(false, Ordering::Release);
            self.resume_runtime();
        }
        self.broadcast(ControlSender::enqueue_finish);
        self.join_workers();
        pp_trace!(
            pp_log: &self.pp_log,
            "event=finish phase=completed outcome=ok"
        );
    }

    fn join_workers(&self) {
        let current_thread = thread::current().id();
        let mut workers = self
            .workers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for worker in workers.drain(..) {
            if worker.thread().id() != current_thread {
                let _ = worker.join();
            }
        }
    }

    /// Jumps to an absolute position from the start of the media. The whole
    /// operation is serialized against lifecycle controls and internally runs
    /// `Pause -> Flush -> Seek -> Preroll -> Pause`, and `Resume` after that if it
    /// was playing — see its four stages below. Every source repositions (see
    /// [`crate::element::SourceElement::seek`]) and every downstream element
    /// reacts before preroll begins. Once every terminal in the starting
    /// topology snapshot has
    /// accepted a first sample (or EOS), a paused pipeline remains paused and
    /// a playing pipeline resumes. Fails with [`PipelineError::NotRunning`]
    /// before [`Pipeline::run`] or once every source has stopped; a source
    /// parked at the end of its stream still counts as running.
    ///
    /// Raises an interrupt before starting the synchronous cascade so a
    /// `Pacer` in a long wait can return its worker promptly.
    /// The clock's playback anchor is still reset later, inside
    /// [`Sink::control`](crate::element::Sink::control) on `Pacer`, after
    /// that in-flight frame is
    /// out of the way.
    ///
    /// Before changing anything, [`Self::check_seek`] asks the graph whether
    /// every source and branch can follow: a live or non-seekable source, or
    /// a recording muxer, returns [`crate::control::SeekError`] without
    /// flushing the current timeline.
    ///
    /// `mode` chooses whether decoding stops at the preceding keyframe or
    /// advances to the sample covering `target`.
    ///
    /// Completion means every terminal accepted its first new-timeline sample
    /// according to [`Sink::consume`](crate::element::Sink::consume). For a
    /// video renderer that includes installing or submitting the preview
    /// frame, but not waiting for physical display scanout.
    /// Whether a [`Self::seek`] would be refused, and by what â without
    /// seeking, and without asking anything running.
    ///
    /// Answered from the graph as it stands: a source that is live or cannot
    /// reposition, and a sink that cannot follow a jump in the timeline (see
    /// [`Sink::accepts_seek`](crate::element::Sink::accepts_seek)), say so as
    /// they are wired. So this works before [`Self::run`] and after the
    /// sources have stopped, costs a lock rather than a round trip through
    /// every thread, and changes as branches come and go â a recording
    /// attached to a `Tee` refuses from the moment it is attached until it is
    /// detached. What a player needs to decide whether to offer a seek bar.
    pub fn check_seek(&self) -> std::result::Result<(), crate::control::SeekError> {
        crate::control::SeekError::from_rejections(self.graph.seek_rejections())
    }

    pub fn seek(&self, target: Duration, mode: SeekMode) -> Result<()> {
        const PREROLL_TIMEOUT: Duration = Duration::from_secs(5);

        let _operation = self
            .operation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.running.load(Ordering::Acquire) == 0 {
            return Err(PipelineError::NotRunning.into());
        }
        self.check_seek()?;
        let playing = !self.paused.load(Ordering::Acquire);

        // A seek is these four stages, each leaving the graph in a state
        // the next relies on â see each for which.
        if playing {
            self.pause_runtime();
        }
        self.reposition(target);
        let preroll = self.preroll_for(|terminals| match mode {
            SeekMode::Keyframe => PrerollContext::new(terminals),
            SeekMode::Accurate => PrerollContext::for_seek(terminals, target),
        });
        let prerolled = self.preroll(&preroll, PREROLL_TIMEOUT);
        self.end_preroll(playing);
        prerolled?;
        pp_trace!(
            pp_log: &self.pp_log,
            "event=control control={:?} phase=completed outcome=ok",
            ControlMsg::Seek(target)
        );
        Ok(())
    }

    /// A seek's second stage, the graph paused: moves every source to
    /// `target` on a new timeline.
    ///
    /// After it, every source is at `target` or wherever it landed near it
    /// (see [`crate::bus::BusEvent::Seeked`]), nothing any element held from
    /// the old position is left â the `Flush` â and whatever of the old
    /// position reaches a queue later is dropped there, being on a timeline
    /// that is no longer current â see [`crate::timeline`]. Nothing has
    /// moved: every source is still paused.
    fn reposition(&self, target: Duration) {
        let msg = ControlMsg::Seek(target);
        pp_trace!(
            pp_log: &self.pp_log,
            "event=control control={msg:?} phase=requested"
        );
        self.state.interrupt();
        self.playback_clock.reset_for_seek();
        // Everything read from here on belongs to the new position, and a
        // queue drops whatever reaches it from the old one â what the
        // `Flush` below discards, and what it misses.
        self.state.begin_timeline();
        self.broadcast(|control_tx| control_tx.enqueue(ControlMsg::Flush));
        self.broadcast(|control_tx| control_tx.enqueue(msg.clone()));
    }

    /// A preroll expecting every terminal the graph has now, as `make`
    /// sets it up, and naming them for a timeout to say which it waited on.
    fn preroll_for(
        &self,
        make: impl FnOnce(Vec<crate::graph::ElementId>) -> PrerollContext,
    ) -> Arc<PrerollContext> {
        let graph = self.graph();
        let terminals = graph.terminal_ids();
        let labels: Vec<_> = terminals
            .iter()
            .filter_map(|&id| graph.node(id).cloned())
            .collect();
        Arc::new(make(terminals).labelled(labels))
    }

    /// A seek's third stage: lets data through the paused graph until every
    /// terminal has taken its sample for `preroll` â or `timeout`, or a
    /// `stop`, ends the wait.
    ///
    /// Paused before and, as far as what flows is concerned, after: each
    /// terminal takes one sample and holds, and a branch with its sample is
    /// held while its siblings catch up â see `Tee`. What ends the preroll
    /// is [`Self::end_preroll`], which must follow whatever this answers.
    fn preroll(
        &self,
        preroll: &Arc<PrerollContext>,
        timeout: Duration,
    ) -> std::result::Result<(), PrerollError> {
        // Published before the wait, so `stop` can end it rather than queue
        // behind it; cleared whatever the wait answers, so no later `stop`
        // cancels a preroll that has already finished.
        self.publish_preroll(preroll);
        self.state.enter(Phase::Prerolling(Arc::clone(preroll)));
        self.broadcast(|control_tx| control_tx.enqueue(ControlMsg::Preroll(Arc::clone(preroll))));
        let prerolled = self.await_preroll(preroll, timeout);
        self.retire_preroll();
        prerolled
    }

    /// A seek's last stage: out of the preroll by way of a pause, then on
    /// playing if `play_on`.
    ///
    /// Always the pause, even to play on. A preroll lets data through, and
    /// playing read off the state before its `Resume` had arrived let a
    /// queue hand a terminal data it had not yet been told to take. Paused,
    /// every source and queue waits for the `Resume` itself and passes it on
    /// before anything else â see `crate::playback_state`.
    fn end_preroll(&self, play_on: bool) {
        self.pause_runtime();
        if play_on {
            self.resume_runtime();
        }
    }
}

/// Decrements the live-source count even if a source panics while running.
struct RunningSourceGuard {
    running: Arc<AtomicUsize>,
}

impl RunningSourceGuard {
    fn new(running: Arc<AtomicUsize>) -> Self {
        Self { running }
    }
}

impl Drop for RunningSourceGuard {
    fn drop(&mut self) {
        self.running.fetch_sub(1, Ordering::AcqRel);
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        // Send Stop while every sender is still alive. Merely dropping the
        // senders would not wake a source polling an empty control channel.
        self.stop();

        self.join_workers();
    }
}

/// Seek's preroll wait, reachable without the operation lock.
///
/// `abandoned` is sticky because the calls that set it â `stop` and `finish` â
/// both end the pipeline for good. Once set, a seek that has not yet published
/// its preroll cancels it on arrival instead of waiting out a timeout nobody
/// is going to collect.
#[derive(Default)]
pub(super) struct PrerollSlot {
    active: Option<Arc<PrerollContext>>,
    abandoned: bool,
}

/// Locks `mutex`, taking over a poisoned one: what these guard â the run
/// state taken once, and the list of workers to join â is never left
/// half-written by a panic elsewhere.
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
