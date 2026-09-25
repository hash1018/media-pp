use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread::JoinHandle,
    time::Duration,
};

use thiserror::Error as ThisError;

use crate::pp_log::PpLog;

use crate::{
    bus::{Bus, BusReceiver},
    clock::Clock,
    control::{ControlReceiver, ControlSender},
    element::{Context, SourceElement},
    error::Result,
    graph::{GraphSnapshot, NodeInfo, PipelineGraph},
    playback_clock::PlaybackClock,
    playback_state::PlaybackState,
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
    /// source had stopped — by [`Pipeline::stop`], [`Pipeline::finish`], or
    /// an error it could not continue past. A source parked at the end of its
    /// stream is still running, and can be sought back into.
    #[error("this pipeline is not running, so there is nothing to seek")]
    NotRunning,

    /// [`Pipeline::step`] was asked to move the picture of a pipeline that
    /// has not shown one: no terminal of it has taken a video frame yet, so
    /// there is neither a picture to step nor a place to step it from.
    #[error("nothing in this pipeline has shown a picture yet, so there is none to step")]
    NoPicture,

    /// [`Pipeline::set_rate`] was asked for a rate outside
    /// [`Pipeline::MIN_RATE`]..=[`Pipeline::MAX_RATE`] that is not
    /// [`Pipeline::REVERSE_RATE`], or one that is not a number.
    #[error("a playback rate has to be between 0.25 and 4, or -1 to play backwards")]
    UnsupportedRate,
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
/// fully finished — with more than one source, *all* of them, not just the
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
/// [`BusEvent::Error`](crate::bus::BusEvent::Error) under that source's own name, since there's no
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
    /// `stop`/`pause`/`resume`/`seek` queue a request on every one of these
    /// before waking anything, and then wait for all of them together — see
    /// `Pipeline::broadcast` for why both halves of that matter.
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
    /// Where playback stands, for every element to read — see
    /// [`crate::playback_state`]. This is the only thing that writes it,
    /// and always before it sends the control message that goes with the
    /// change.
    pub(super) state: Arc<PlaybackState>,
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
    /// Whether the picture has been stepped since playback last lined
    /// everything up to it — see [`Pipeline::step`]. The next resume seeks to
    /// the picture first.
    pub(super) stepped: AtomicBool,
    /// Which terminals have taken the end of their stream — what a step has
    /// no more pictures to ask for.
    pub(super) completion: Arc<super::completion::Completion>,
    /// Serializes public lifecycle/timeline operations. `paused` remains the
    /// caller-requested state while seek temporarily pauses the runtime.
    pub(super) operation: Arc<Mutex<()>>,
    /// The preroll a [`Pipeline::seek`] is currently waiting on, and whether
    /// the pipeline has since been abandoned.
    ///
    /// Held here rather than only inside `seek` so a caller that wants the
    /// pipeline to end can reach it *without* the operation lock. `Stop` would
    /// otherwise have to queue behind that wait, which is the one thing it
    /// promises not to do — and the cancellation the terminals already forward
    /// on `Stop` cannot arrive either, because sending it needs the same lock.
    pub(super) preroll_slot: Mutex<super::seek::PrerollSlot>,
    /// Handles for every source thread started by [`Pipeline::run`]. They
    /// are retained so dropping the pipeline can synchronously stop and
    /// join live sources instead of leaving detached work behind.
    pub(super) workers: Mutex<Vec<JoinHandle<()>>>,
    /// Live node/edge graph backing snapshots and topology rendering.
    pub(super) graph: PipelineGraph,
    /// Each source's counters, kept alive here — see
    /// [`PipelineBuilder`]'s own field.
    pub(super) _source_counters: Vec<Arc<crate::stats::ElementCounters>>,
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
    /// Whatever `wire` returns comes back beside the pipeline: something
    /// only the wiring can make — a [`crate::elements::TeeHandle`] from
    /// [`crate::elements::TeeBuilder::build_dynamic`], a routing to keep —
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
    /// The single-source special case of [`PipelineBuilder`] — see its own
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
    /// and queue workers finishing — for a seekable source such as a file,
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
    /// branch — each element nothing else in the graph feeds into (a
    /// terminal sink, or an empty [`crate::elements::Tee`] with no sinks
    /// attached yet) — formatted `Type(name) - Type(name) - ...` by
    /// walking that element's `upstream` chain back to the source.
    /// Multiple branches (fan-out across more than one src pad, or a
    /// `Tee`) are joined by newlines.
    pub fn topology(&self) -> String {
        self.graph().topology()
    }

    /// What every element is doing, read now — see [`crate::stats`].
    ///
    /// Running totals rather than rates: take two readings and the rate is
    /// their difference over the time between them, matched by
    /// [`ElementStats::id`](crate::stats::ElementStats::id). Cheap enough to
    /// call a few times a second — the graph's lock is held only to copy
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

    /// Where playback is: the media time the pipeline's playback master has
    /// reached — the audio renderer's played samples when there is one, else
    /// the wall clock a [`crate::elements::Pacer`] or
    /// [`crate::elements::VideoSynchronizer`] keeps. What a progress bar
    /// shows, read from the clock rather than from whichever frame last went
    /// past, so it moves with the sound and holds still while paused.
    ///
    /// `None` while nothing paces this pipeline — a transcode runs as fast as
    /// it can and has no "now" — and from a [`Pipeline::seek`] until the
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
    /// source has finished — by its own `Eos`, by [`Pipeline::stop`] or
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
    /// Draining the bus stays the way to learn *why* — see this type's own
    /// docs — and remains the only way to wait for the end rather than poll
    /// for it.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire) > 0
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

/// Locks `mutex`, taking over a poisoned one: what these guard — the run
/// state taken once, and the list of workers to join — is never left
/// half-written by a panic elsewhere.
pub(super) fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
