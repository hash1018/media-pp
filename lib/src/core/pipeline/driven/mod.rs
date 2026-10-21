//! A pipeline with no thread of its own — see [`DrivenPipeline`].

mod runner;
mod timer;

pub use runner::DrivenRunner;

use std::{
    future::Future,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context as TaskContext, Poll, Waker},
    time::{Duration, Instant},
};

use crate::{
    bus::{Bus, BusEvent, BusReceiver},
    clock::Clock,
    control::{ControlMsg, ControlReceiver, ControlSender},
    element::{Context, Drive, IntoSource, RawSource, Stepped, pipeline_pp_log},
    error::{Error, Result},
    graph::{PipelineGraph, log_topology},
    playback_state::{Phase, PlaybackState},
    pp_log::{PpLog, pp_info, pp_trace, pp_warn},
    stats::{ElementCounters, PipelineStats},
};

use super::{PipelineBuilder, runtime::lock};

/// How long a poll may take before it is logged: an element that waits —
/// on a device, a socket — keeps every other pipeline on the thread waiting
/// with it.
const SLOW_POLL: Duration = Duration::from_millis(20);

/// How often at most a pipeline whose polls stay slow says so.
const SLOW_POLL_EVERY: Duration = Duration::from_secs(1);

/// A pipeline with no thread of its own: whoever owns it polls it, and many
/// of them share a few threads.
///
/// Wired as a [`Pipeline`](super::Pipeline) is — the same `wire` closure,
/// the same chain builder and elements — and then, rather than run on
/// threads of its own, polled: each [`Self::poll`] takes what was asked of
/// it, hands on what its source has now, and says when it wants to be
/// polled again. A server with a pipeline for each of its players polls
/// them all on a few threads; [`DrivenRunner`] does that for an application
/// with no executor of its own, and as a [`Future`] one goes to any
/// executor that has one.
///
/// # Example
///
/// ```no_run
/// # fn main() -> media_pp::Result<()> {
/// use media_pp::elements::{FrameCounter, TestVideoOptions, TestVideoSource};
/// use media_pp::pipeline::{DrivenPipeline, DrivenRunner};
///
/// let runner = DrivenRunner::new(2)?;
/// let mut handles = Vec::new();
/// for player in 0..100 {
///     let source = TestVideoSource::new("camera", TestVideoOptions::default());
///     let (counter, _pictures) = FrameCounter::new("pictures");
///     let (pipeline, ()) = DrivenPipeline::new(format!("player-{player}"), source, |source, ctx| {
///         ctx.attach(source, 0, ctx.branch().to(counter)?)?;
///         Ok(())
///     })?;
///     handles.push(pipeline.handle());
///     runner.spawn(pipeline);
/// }
/// // ... and when they are done:
/// for handle in &handles {
///     handle.stop();
/// }
/// # Ok(())
/// # }
/// ```
///
/// # What may be in one
///
/// Nothing in it may wait, since it waits on the thread every other
/// pipeline there is polled on too. Its wiring refuses what has a thread of
/// its own, with a [`GraphError`](crate::graph::GraphError):
///
/// - no [`queue`](super::ChainBuilder::queue), in a branch or in a `Tee`'s;
/// - no element that waits on the clock behind a queue of its own, as a
///   [`Pacer`](crate::elements::Pacer) and a
///   [`VideoSynchronizer`](crate::elements::VideoSynchronizer) do;
/// - only a source that says it can be driven —
///   [`Source::drivable`](crate::element::Source::drivable).
///
/// A filter or sink that blocks — on a device, on a socket, on a full
/// channel — cannot be told by its type, and blocks the thread: a poll that
/// takes longer than 20 ms is logged as a warning under the pipeline's
/// identity, so that it can be found. Decoding and encoding do not wait and
/// are allowed, but run on the polling thread, and every pipeline polled
/// there waits for them.
///
/// # Control
///
/// [`Self::handle`] gives what pauses, resumes, stops and finishes it from
/// anywhere; each call says what it asks, wakes the pipeline, and returns
/// at once — the next poll takes it, through the same elements' hooks a
/// running pipeline's thread would. A paused pipeline is polled only to
/// take what it is asked next. It cannot be sought.
///
/// # Ending
///
/// [`Self::poll`] answers [`Driven::Finished`] once its stream has ended,
/// it was stopped or finished, or it failed: what it failed with is on the
/// bus, as for a running pipeline, and a panic in it is caught there and
/// reported the same way, so the thread goes on with the others. Dropping
/// it stops it on the dropping thread, with nothing to join.
pub struct DrivenPipeline {
    shared: Arc<Shared>,
    /// The source and what it posts on, until it ends — and then nothing,
    /// so that every element goes, and with them the bus's senders.
    running: Option<Running>,
    control_rx: ControlReceiver,
    /// Whether it has been polled — begun its stream, logged its topology.
    polled: bool,
    /// The deadline handed to the timer last, and the waker it will wake.
    armed: Option<(Instant, Waker)>,
    slow: SlowPolls,
    /// The source's counters, which nothing wraps it to keep — see
    /// [`crate::stats`].
    _source_counters: Arc<ElementCounters>,
}

struct Running {
    source: Box<dyn RawSource>,
    bus: Bus,
}

/// What a [`DrivenPipeline`] and its handles share.
struct Shared {
    id: Arc<str>,
    pp_log: PpLog,
    control_tx: ControlSender,
    state: Arc<PlaybackState>,
    clock: Arc<Clock>,
    graph: PipelineGraph,
    bus_rx: BusReceiver,
    /// The waker of the last poll, which a request wakes.
    waker: Mutex<Option<Waker>>,
    paused: AtomicBool,
    ended: AtomicBool,
    /// Serializes the handles' requests, so the state each writes and the
    /// message it sends go in the same order.
    operation: Mutex<()>,
}

impl Shared {
    fn wake(&self) {
        if let Some(waker) = lock(&self.waker).as_ref() {
            waker.wake_by_ref();
        }
    }
}

/// Where a [`DrivenPipeline::poll`] left it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Driven {
    /// Poll it again when the waker of this poll is woken, or by `at`
    /// where there is one, whether woken or not. `at` may already be
    /// past: it has more to do, and gave the others on the thread a turn
    /// first.
    Pending {
        /// When it wants to be polled again, woken or not.
        at: Option<Instant>,
    },
    /// It has ended — its stream, or by a stop, a finish or a failure,
    /// which the bus tells. Polled again, it answers this again.
    Finished,
}

impl DrivenPipeline {
    /// Wires a driven pipeline of `source`, as
    /// [`Pipeline::new`](super::Pipeline::new) wires one that runs, and
    /// hands it back beside whatever `wire` returned.
    ///
    /// Fails with what the wiring refuses — see the type's docs on what
    /// may be in one — and with what `wire` fails with.
    pub fn new<M, I: IntoSource<M>, T>(
        id: impl Into<String>,
        source: I,
        wire: impl FnOnce(&mut I::Raw, &Arc<Context>) -> Result<T>,
    ) -> Result<(Self, T)> {
        let (builder, wired) = PipelineBuilder::driven(id).add_source(source, wire)?;
        let parts = builder.into_driven();
        let (source_id, source) = parts.source;
        let (control_tx, control_rx) = parts.control;
        let pp_log = pipeline_pp_log(&parts.id);
        let pipeline = Self {
            shared: Arc::new(Shared {
                id: parts.id,
                pp_log,
                control_tx,
                state: parts.state,
                clock: parts.clock,
                graph: parts.graph,
                bus_rx: parts.bus_rx,
                waker: Mutex::new(None),
                paused: AtomicBool::new(false),
                ended: AtomicBool::new(false),
                operation: Mutex::new(()),
            }),
            running: Some(Running {
                source,
                bus: parts.bus.for_element(source_id),
            }),
            control_rx,
            polled: false,
            armed: None,
            slow: SlowPolls::default(),
            _source_counters: parts.source_counters,
        };
        Ok((pipeline, wired))
    }

    /// What controls it, from any thread — see [`DrivenPipelineHandle`].
    pub fn handle(&self) -> DrivenPipelineHandle {
        DrivenPipelineHandle {
            shared: Arc::clone(&self.shared),
        }
    }

    /// Takes what it was asked since the last poll, and hands on what its
    /// source has now — as much as one poll's share, a few dozen buffers —
    /// and says when to poll it again. Never blocks, on its own account:
    /// see the type's docs on what may be in one.
    ///
    /// `cx`'s waker is what a request — and whatever its source waits on
    /// that can wake — wakes. A deadline in [`Driven::Pending`] wakes
    /// nothing: whoever polls it keeps it, or polls it as a [`Future`],
    /// which keeps it for them.
    pub fn poll(&mut self, cx: &mut TaskContext<'_>) -> Driven {
        let Some(running) = self.running.as_mut() else {
            return Driven::Finished;
        };
        {
            let mut waker = lock(&self.shared.waker);
            if !waker
                .as_ref()
                .is_some_and(|waker| waker.will_wake(cx.waker()))
            {
                *waker = Some(cx.waker().clone());
            }
        }
        let began = Instant::now();
        if !self.polled {
            self.polled = true;
            begin(&self.shared, running);
        }
        let stepped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            running.source.drive(&mut Drive {
                control: &self.control_rx,
                bus: &running.bus,
                waker: cx.waker(),
            })
        }))
        .unwrap_or_else(|payload| Err(Error::panicked(payload.as_ref())));
        self.slow.note(&self.shared.pp_log, began.elapsed());
        match stepped {
            Ok(Stepped::Pending(at)) => return Driven::Pending { at },
            Ok(Stepped::Ended) => {
                pp_info!(pp_log: running.source.pp_log(), "finished outcome=ok");
            }
            Err(error) => {
                let source = &mut running.source;
                running.bus.post(
                    source.pp_log(),
                    BusEvent::Error {
                        element_type: source.element_type(),
                        name: source.name(),
                        error,
                    },
                );
                // As a failed source's thread does: what it fed is told it
                // is over, so a muxer behind it finalizes what it has.
                let source_log = source.pp_log().clone();
                for pad in source.src_pads() {
                    if let Err(error) = pad.control(&ControlMsg::Stop) {
                        pp_warn!(
                            pp_log: &source_log,
                            "failed to stop the branch after a source error: {error}"
                        );
                    }
                }
                pp_info!(pp_log: &source_log, "finished outcome=error");
            }
        }
        self.end();
        Driven::Finished
    }

    /// Lets go of everything it is made of, once it has ended.
    fn end(&mut self) {
        self.shared.ended.store(true, Ordering::Release);
        self.running = None;
        lock(&self.shared.waker).take();
        self.armed = None;
    }

    /// Has the timer wake `waker` at `at`, unless one it already has wakes
    /// it by then.
    fn arm(&mut self, at: Instant, waker: &Waker) {
        let now = Instant::now();
        if at <= now {
            waker.wake_by_ref();
            return;
        }
        if let Some((armed, armed_waker)) = &self.armed
            && now < *armed
            && *armed <= at
            && armed_waker.will_wake(waker)
        {
            return;
        }
        timer::wake_at(at, waker.clone());
        self.armed = Some((at, waker.clone()));
    }
}

/// What the first poll does before it asks the source for anything: the
/// topology logged, as a running pipeline's start logs it, and the
/// segment every stream begins with.
fn begin(shared: &Shared, running: &mut Running) {
    if crate::log::enabled(crate::log::Level::Info) {
        log_topology(&shared.pp_log, "run", &shared.graph.snapshot());
    }
    let source = &mut running.source;
    let source_log = source.pp_log().clone();
    let segment = crate::stream::Segment {
        id: shared.state.timeline(),
        flushed: false,
        position: Duration::ZERO,
        start: Duration::ZERO,
        show_from: None,
        backwards: false,
    };
    if let Err(error) = crate::stream::begin_segment(source.src_pads(), segment, &source_log) {
        running
            .bus
            .post_downstream_error(&source_log, source.element_type(), source.name(), error);
    }
}

impl Future for DrivenPipeline {
    type Output = ();

    /// [`DrivenPipeline::poll`], with its deadline kept: a timer thread the
    /// process has one of wakes the task by then.
    fn poll(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<()> {
        let this = self.get_mut();
        match DrivenPipeline::poll(this, cx) {
            Driven::Finished => Poll::Ready(()),
            Driven::Pending { at } => {
                if let Some(at) = at {
                    this.arm(at, cx.waker());
                }
                Poll::Pending
            }
        }
    }
}

impl Drop for DrivenPipeline {
    /// Stops it here, as [`DrivenPipelineHandle::stop`] and a poll would:
    /// every element hears the stop, and nothing is left to join.
    fn drop(&mut self) {
        if self.running.is_none() {
            return;
        }
        if self.polled {
            self.shared.state.enter(Phase::Stopped);
            let _ = self.shared.control_tx.enqueue(ControlMsg::Stop);
            let mut cx = TaskContext::from_waker(Waker::noop());
            let _ = DrivenPipeline::poll(self, &mut cx);
        }
        self.end();
    }
}

/// Controls a [`DrivenPipeline`] from any thread.
///
/// Cheap to clone. It keeps the pipeline's bus, graph and control
/// channel, never its elements: once the pipeline has ended or been
/// dropped, each request does nothing. No call waits — each says what it
/// asks, wakes the pipeline, and returns; the next poll takes it.
#[derive(Clone)]
pub struct DrivenPipelineHandle {
    shared: Arc<Shared>,
}

impl DrivenPipelineHandle {
    /// The pipeline's caller-selected identity.
    pub fn id(&self) -> &str {
        &self.shared.id
    }

    /// What its elements report — as [`Pipeline::bus`](super::Pipeline::bus)
    /// does for a running pipeline. Draining it blocks until the pipeline
    /// has ended or been dropped. What is not read is kept until every
    /// handle and the pipeline are gone.
    pub fn bus(&self) -> &BusReceiver {
        &self.shared.bus_rx
    }

    /// Whether it has ended — see [`Driven::Finished`].
    pub fn is_finished(&self) -> bool {
        self.shared.ended.load(Ordering::Acquire)
    }

    /// What every element is doing, read now — see
    /// [`Pipeline::stats`](super::Pipeline::stats).
    pub fn stats(&self) -> PipelineStats {
        let (revision, registered, attached) = self.shared.graph.registered();
        PipelineStats {
            revision,
            paused: self.shared.paused.load(Ordering::Acquire),
            elements: crate::stats::read(registered, |id| attached.contains(&id)),
        }
    }

    /// Its topology as text — see
    /// [`Pipeline::topology`](super::Pipeline::topology).
    pub fn topology(&self) -> String {
        self.shared.graph.snapshot().topology()
    }

    /// Pauses it: the next poll stops its source and passes the pause on to
    /// every element, and it makes nothing until resumed. Its source's
    /// clock stands still meanwhile, so it plays on where it was.
    pub fn pause(&self) {
        let shared = &self.shared;
        let _operation = lock(&shared.operation);
        if shared.ended.load(Ordering::Acquire) || shared.paused.swap(true, Ordering::AcqRel) {
            return;
        }
        pp_trace!(
            pp_log: &shared.pp_log,
            "event=control control=Pause phase=requested"
        );
        shared.state.pause();
        shared.clock.pause();
        let _ = shared.control_tx.enqueue(ControlMsg::Pause);
        shared.wake();
    }

    /// Undoes [`Self::pause`].
    pub fn resume(&self) {
        let shared = &self.shared;
        let _operation = lock(&shared.operation);
        if shared.ended.load(Ordering::Acquire) || !shared.paused.swap(false, Ordering::AcqRel) {
            return;
        }
        self.resume_locked();
        shared.wake();
    }

    fn resume_locked(&self) {
        let shared = &self.shared;
        pp_trace!(
            pp_log: &shared.pp_log,
            "event=control control=Resume phase=requested"
        );
        shared.clock.resume();
        shared.state.enter(Phase::Playing);
        let _ = shared.control_tx.enqueue(ControlMsg::Resume);
    }

    /// Stops it, abandoning what it holds — see
    /// [`Pipeline::stop`](super::Pipeline::stop). Its next poll answers
    /// [`Driven::Finished`].
    pub fn stop(&self) {
        let shared = &self.shared;
        let _operation = lock(&shared.operation);
        if shared.ended.load(Ordering::Acquire) {
            return;
        }
        pp_trace!(
            pp_log: &shared.pp_log,
            "event=control control=Stop phase=requested"
        );
        shared.paused.store(false, Ordering::Release);
        shared.state.enter(Phase::Stopped);
        let _ = shared.control_tx.enqueue(ControlMsg::Stop);
        shared.wake();
    }

    /// Ends its stream in order, behind what it has already handed on, so
    /// that what holds data back — an encoder, a muxer — finishes it: see
    /// [`Pipeline::finish`](super::Pipeline::finish). Paused, it is resumed
    /// first. The poll that takes it answers [`Driven::Finished`].
    pub fn finish(&self) {
        let shared = &self.shared;
        let _operation = lock(&shared.operation);
        if shared.ended.load(Ordering::Acquire) {
            return;
        }
        pp_trace!(
            pp_log: &shared.pp_log,
            "event=finish phase=requested"
        );
        shared.state.finish();
        if shared.paused.swap(false, Ordering::AcqRel) {
            self.resume_locked();
        }
        let _ = shared.control_tx.enqueue_finish();
        shared.wake();
    }
}

/// Polls that took longer than [`SLOW_POLL`], said at most every
/// [`SLOW_POLL_EVERY`].
#[derive(Default)]
struct SlowPolls {
    /// Since it was last said, how many, and the longest.
    unsaid: u32,
    longest: Duration,
    said: Option<Instant>,
}

impl SlowPolls {
    fn note(&mut self, pp_log: &PpLog, took: Duration) {
        if took < SLOW_POLL {
            return;
        }
        self.unsaid += 1;
        self.longest = self.longest.max(took);
        let now = Instant::now();
        if self
            .said
            .is_some_and(|said| now.duration_since(said) < SLOW_POLL_EVERY)
        {
            return;
        }
        pp_warn!(
            pp_log: pp_log,
            "event=poll phase=slow polls={} longest={:.1?}: something in it waits, and holds up every pipeline polled on this thread",
            self.unsaid,
            self.longest
        );
        self.unsaid = 0;
        self.longest = Duration::ZERO;
        self.said = Some(now);
    }
}

#[cfg(test)]
mod tests;
