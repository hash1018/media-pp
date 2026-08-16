use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use crate::pp_log::{PpLog, pp_info, pp_trace};

use crate::{
    bus::{Bus, BusEvent, BusReceiver},
    clock::Clock,
    control::{ControlMsg, ControlReceiver, ControlSender},
    element::{Context, SourceElement},
    error::Result,
    graph::{GraphSnapshot, NodeInfo, PipelineGraph, log_topology},
    playback_clock::PlaybackClock,
};

use super::{PipelineBuilder, builder::SourceEntry};

/// Top-level pipeline: one or more sources (see [`PipelineBuilder`], with
/// everything reachable from each source's own src pads already linked)
/// plus the bus every source reports events on and the [`Clock`] every
/// [`crate::elements::Pacer`] in it shares.
///
/// `run()` is asynchronous: it starts every source on its own background
/// thread and returns immediately, rather than blocking the caller for the
/// whole play-through. Returned as `Arc<Pipeline>` (that's what
/// [`Pipeline::new`]/[`PipelineBuilder::build`] return) — the background
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
/// finished via every source's natural `Eos`, [`Pipeline::finish`], or
/// [`Pipeline::stop`]) — a
/// second `run()` call is a no-op; build a fresh `Pipeline` for another
/// play-through.
pub struct Pipeline {
    /// This pipeline's own id — passed to [`Pipeline::new`]/
    /// [`PipelineBuilder::new`], stamped onto every source's own `pp_log`
    /// there and onto every element that passes through a [`super::ChainBuilder`]
    /// built with it (see [`Pipeline::id`]).
    pub(super) id: Arc<str>,
    /// Logging identity for pipeline-level topology records.
    pub(super) pp_log: PpLog,
    pub(super) sources: Mutex<Option<Vec<SourceEntry>>>,
    /// Taken (leaving `None` behind) the moment `run()` starts, and cloned
    /// once per source into that source's own background thread — so once
    /// a pipeline is running, `Pipeline` itself no longer holds a `Bus`
    /// sender directly. If it did, [`BusReceiver::iter`] could never
    /// observe every sender dropped (one would always still be sitting
    /// right here), and would block forever instead of unblocking once
    /// every source actually finishes.
    pub(super) bus: Mutex<Option<Bus>>,
    /// One [`ControlSender`] per source, in the same order
    /// [`PipelineBuilder::add_source`] was called — [`Pipeline::finish`]/
    /// `stop`/`pause`/`resume`/`seek` send to every one of these in turn (each
    /// `send` is its own synchronous rendezvous with that source's own
    /// control cascade — see [`crate::control::ControlSender::send`] — so
    /// this serializes across sources rather than fanning out in
    /// parallel; fine for the handful of sources this is meant for).
    pub(super) control_txs: Vec<ControlSender>,
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
    pub(super) control_rxs: Mutex<Option<Vec<ControlReceiver>>>,
    pub(super) clock: Arc<Clock>,
    pub(super) playback_clock: Arc<PlaybackClock>,
    pub(super) bus_rx: BusReceiver,
    /// How many source threads are still running — `0` before `run()` and
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
    /// Handles for every source thread started by [`Pipeline::run`]. They
    /// are retained so dropping the pipeline can synchronously stop and
    /// join live sources instead of leaving detached work behind.
    pub(super) workers: Mutex<Vec<JoinHandle<()>>>,
    /// Live node/edge graph backing snapshots and topology rendering.
    pub(super) graph: PipelineGraph,
}

impl Pipeline {
    /// `id` names this pipeline — stamped into the source's own `pp_log` as
    /// its `pipeline_id` right away, and folded into the [`Context`] handed
    /// to `wire` (see [`super::ChainBuilder`]'s own docs).
    ///
    /// `wire` is called once with the freshly created source and a
    /// [`Context`] bundling this pipeline's `Bus`, `id`, [`PipelineGraph`]
    /// (already seeded with the source itself), and `Clock` (share it with
    /// every [`crate::elements::Pacer`] via `Clock::clone` — one clock per
    /// pipeline, so every paced branch agrees on the same t=0 and the same
    /// pause/resume timeline) — everything a [`super::ChainBuilder`]/
    /// [`crate::elements::Tee`] needs, in one `Arc` clone instead of four
    /// separate arguments. `wire` creates detached chains and attaches
    /// them through [`Context::attach`]. Pads left unattached drop data.
    ///
    /// The single-source special case of [`PipelineBuilder`] — see its own
    /// docs for combining more than one live source (e.g. a video capture
    /// and an audio capture) into one `Pipeline`.
    pub fn new<S: SourceElement + 'static>(
        id: impl Into<String>,
        source: S,
        wire: impl FnOnce(&mut S, &Arc<Context>) -> Result<()>,
    ) -> Result<Arc<Self>> {
        Ok(PipelineBuilder::new(id).add_source(source, wire)?.build())
    }

    /// This pipeline's own id, as passed to [`Pipeline::new`].
    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn bus(&self) -> &BusReceiver {
        &self.bus_rx
    }

    /// Returns a consistent node/edge snapshot of the live graph. Detached
    /// branches do not appear; a successful attach or detach increments its
    /// revision exactly once.
    pub fn graph(&self) -> GraphSnapshot {
        self.graph.snapshot()
    }

    pub fn elements(&self) -> Vec<NodeInfo> {
        self.graph().nodes
    }

    /// Human-readable rundown of [`Pipeline::elements`]: one line per
    /// branch — each element nothing else in the graph feeds into (a
    /// terminal sink, or an empty [`crate::elements::Tee`] with no sinks
    /// attached yet) — formatted `Type(name) - Type(name) - ...` by
    /// walking that element's `upstream` chain back to the source.
    /// Multiple branches (fan-out across more than one src pad, or a
    /// `Tee`) are joined by newlines.
    pub fn topology(&self) -> String {
        self.graph().topology()
    }

    /// The clock every `Pacer` in this pipeline paces against — see
    /// [`Pipeline::pause`] for why callers don't usually need to touch
    /// this directly.
    pub fn clock(&self) -> &Arc<Clock> {
        &self.clock
    }

    /// Media-position clock shared by audio output and video scheduling.
    pub fn playback_clock(&self) -> &Arc<PlaybackClock> {
        &self.playback_clock
    }

    /// Starts driving the source on a background thread and returns
    /// immediately — see the type-level docs for how to learn when it's
    /// actually done. A no-op if this `Pipeline` is already running or
    /// has already finished a previous run — this type has no "reset"
    /// path; build a fresh `Pipeline` for another play-through.
    pub fn run(&self) {
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

        if crate::log::enabled(crate::log::Level::Info) {
            log_topology(&self.pp_log, "run", &self.graph());
        }
        self.running.store(sources.len(), Ordering::Release);
        for ((source_id, source), control_rx) in sources.into_iter().zip(control_rxs) {
            let bus = bus.for_element(source_id);
            let running = Arc::clone(&self.running);
            let handle = thread::Builder::new()
                .name("pipeline:source".into())
                .spawn(move || {
                    // Keep these as locals in this order. During unwinding the
                    // guard is dropped first, then the receiver, then the
                    // source. That makes a Pipeline indirectly retained by a
                    // custom source safe to drop from this worker thread.
                    let mut source = source;
                    let control_rx = control_rx;
                    let _running = RunningSourceGuard::new(running);

                    let source_name = source.name();
                    let source_type = source.element_type();
                    // `source.run()` itself already reports non-fatal,
                    // per-buffer failures to `bus` as it goes (see
                    // `SourceElement::run`'s docs) — a returned `Err` here
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
                        "error"
                    } else {
                        "ok"
                    };
                    pp_info!(pp_log: source.pp_log(), "finished outcome={outcome}");
                })
                .expect("failed to spawn pipeline source thread");
            self.workers.lock().unwrap().push(handle);
        }
    }

    /// Blocks until every element downstream of every source has paused —
    /// see [`crate::control::drain_control`] (source side) and
    /// [`crate::queue::Queue`]'s worker loop (each thread boundary). Also
    /// pauses this pipeline's `Clock` before that synchronous cascade
    /// starts, so time spent waiting for a busy downstream element to
    /// acknowledge `Pause` is frozen too and a `Pacer` doesn't see a jump
    /// once resumed. No-op if `run()` isn't currently in progress on
    /// another thread.
    pub fn pause(&self) {
        if self.running.load(Ordering::Acquire) == 0 {
            return;
        }
        let msg = ControlMsg::Pause;
        pp_trace!(
            pp_log: &self.pp_log,
            "event=control control={msg:?} phase=requested"
        );
        self.clock.interrupt();
        self.clock.pause();
        self.paused.store(true, Ordering::Release);
        for control_tx in &self.control_txs {
            control_tx.send(msg);
        }
        pp_trace!(
            pp_log: &self.pp_log,
            "event=control control={msg:?} phase=completed outcome=ok"
        );
    }

    /// Undoes [`Pipeline::pause`]. Resumes the `Clock` first, so it's
    /// already shifted forward by the time `Pacer`s start receiving
    /// frames again.
    pub fn resume(&self) {
        if self.running.load(Ordering::Acquire) == 0 {
            return;
        }
        self.paused.store(false, Ordering::Release);
        let msg = ControlMsg::Resume;
        pp_trace!(
            pp_log: &self.pp_log,
            "event=control control={msg:?} phase=requested"
        );
        self.clock.resume();
        for control_tx in &self.control_txs {
            control_tx.send(msg);
        }
        pp_trace!(
            pp_log: &self.pp_log,
            "event=control control={msg:?} phase=completed outcome=ok"
        );
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
        let msg = ControlMsg::Stop;
        self.paused.store(false, Ordering::Release);
        pp_trace!(
            pp_log: &self.pp_log,
            "event=control control={msg:?} phase=requested"
        );
        self.clock.interrupt();
        for control_tx in &self.control_txs {
            control_tx.send(msg);
        }
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
        if self.running.load(Ordering::Acquire) == 0 {
            self.join_workers();
            return;
        }

        pp_trace!(
            pp_log: &self.pp_log,
            "event=finish phase=requested"
        );
        self.clock.interrupt();
        if self.paused.load(Ordering::Acquire) {
            self.resume();
        }
        for control_tx in &self.control_txs {
            control_tx.finish();
        }
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
    /// [`Sink::control`](crate::element::Sink::control) on `Pacer`, after
    /// that in-flight frame is
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
        let msg = ControlMsg::Seek(target);
        pp_trace!(
            pp_log: &self.pp_log,
            "event=control control={msg:?} phase=requested"
        );
        self.clock.interrupt();
        self.playback_clock.reset_for_seek();
        for control_tx in &self.control_txs {
            control_tx.send(msg);
        }
        pp_trace!(
            pp_log: &self.pp_log,
            "event=control control={msg:?} phase=completed outcome=ok"
        );
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
