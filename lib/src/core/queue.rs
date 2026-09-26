//! The explicit thread boundary, and with it the error boundary.
//!
//! A pipeline is synchronous until a [`Queue`] is placed in it. A `Queue` owns
//! a worker thread and a bounded channel, so upstream and downstream of it run
//! concurrently and a full channel becomes backpressure.
//!
//! Crossing it changes how failure is handled. A direct
//! [`Sink::consume`](crate::element::Sink::consume) call can return `Err` to
//! its caller; a `Queue`'s worker has no caller to return to, so a downstream
//! data error is posted to the [`Bus`](crate::bus::Bus), that buffer is
//! dropped, and the worker continues. [`OverflowPolicy`] decides what a full
//! channel does, and its own documentation explains why an unbounded wait is
//! the default and when it is the wrong one.

use std::{
    cell::Cell,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use crate::pp_log::{PpLog, pp_debug, pp_info, pp_trace};
use crossbeam_channel::{Receiver, Sender, select, unbounded};
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    contract::InputContract,
    control::{self, ControlMsg, ControlReceiver, ControlSender, RequestKind},
    element::{Context, Element, ElementType, Sink, element_pp_log},
    error::{Result, ThreadSpawnError},
    playback_state::{Bell, PlaybackState},
    stats::ElementCounters,
    timeline::{Numbered, UNNUMBERED},
};

/// Errors specific to `Queue`. Converts into the crate-wide `Error` via
/// `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum QueueError {
    /// The operating system could not create the queue's worker thread.
    ///
    /// Construction returns without retaining the supplied downstream sink.
    #[error(transparent)]
    ThreadSpawn(#[from] ThreadSpawnError),

    /// The worker has exited and can no longer receive media buffers.
    ///
    /// A queue does not restart its worker; stop or rebuild the owning pipeline.
    #[error("downstream channel closed")]
    ChannelClosed,

    /// [`OverflowPolicy::Block`] only — the channel stayed full for the
    /// whole `after`, meaning whatever's downstream of this `Queue`
    /// didn't just fall behind (ordinary, self-resolving backpressure),
    /// it's genuinely stuck. Unlike [`OverflowPolicy::DropNewest`]'s
    /// silent, expected-under-load `BusEvent::Dropped`, this is
    /// surfaced as a real error precisely because it isn't expected —
    /// see [`OverflowPolicy::Block`]'s own docs.
    #[error("downstream didn't accept a buffer within {after:?} — send timed out")]
    SendTimedOut {
        /// Maximum time spent waiting for free capacity before the current
        /// buffer was returned to the caller as undelivered.
        after: Duration,
    },
}

/// How often a worker whose downstream is not ready looks again — for a
/// downstream that becomes ready without saying so: a renderer whose device
/// has played some of what it holds, another queue whose worker has taken
/// a buffer. What does say so — the pipeline's state moving on, a terminal
/// taking its preroll sample, a request, the queue being dropped — rings
/// the worker at once; see [`crate::playback_state::Bell`]. Nothing else in
/// a queue waits on a timer.
const READINESS_POLL_INTERVAL: Duration = Duration::from_millis(20);

/// The longest a dropped queue's worker holds back for an interrupt before
/// it drains all the same — for a pipeline that is not going to settle it.
/// One that is settles it a moment after the request that ended its source,
/// and rings the worker as it does; see [`Inbox::held_back`].
const SETTLE_WAIT_LIMIT: Duration = Duration::from_secs(1);

/// What a `Queue` does when its channel is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverflowPolicy {
    /// Block the pushing thread until there's room, up to `Duration` —
    /// the right choice for offline/file processing, where correctness
    /// matters more than staying caught up. Use [`Duration::MAX`] (what
    /// [`OverflowPolicy::default`] does) for what's practically an
    /// unbounded wait — [`Sender::send_timeout`] with that duration
    /// isn't ever going to time out in a real program.
    ///
    /// A *finite* `Duration` is the escape hatch against the one thing
    /// an actually-unbounded wait can't recover from: whatever's
    /// downstream not just falling behind (ordinary backpressure, which
    /// resolves on its own as the worker keeps draining) but genuinely
    /// stuck — a `Sink::consume` call somewhere in the chain that never
    /// returns. An unbounded wait here would then also wedge whoever's
    /// pushing into this `Queue`, and transitively every `Queue`
    /// upstream of *that*, since each one's worker can't get back to its
    /// own `control_rx` until its current `downstream.consume()` call
    /// returns (see [`Queue::control`]'s own docs on why control is only
    /// ever checked *between* buffers, not able to preempt one already
    /// in flight). Timing out bounds that: it's what lets a `Stop` sent
    /// to an upstream `Queue` eventually reach it instead of waiting
    /// forever. Doesn't help if the stall is inside a raw (non-`Queue`)
    /// `Sink`'s own `consume()` call directly — nothing here retries or
    /// times out *that* call itself, only the channel send. On timeout,
    /// returns [`QueueError::SendTimedOut`] rather than losing the
    /// buffer silently — unlike [`OverflowPolicy::DropNewest`], this
    /// isn't an expected, routine condition.
    ///
    /// This timeout applies to ordinary data buffers only. `Queue` sends
    /// `MediaBuffer::Eos` with an unbounded `send` under every policy so a
    /// natural end-of-stream marker is never discarded; if downstream has
    /// stopped consuming entirely, an EOS push can therefore still block.
    Block(Duration),
    /// Drop the incoming buffer instead of blocking, and post
    /// [`BusEvent::Dropped`]. Never stalls the upstream thread — the
    /// right choice for live sources, where falling behind is worse than
    /// losing a frame.
    DropNewest,
}

impl Default for OverflowPolicy {
    fn default() -> Self {
        OverflowPolicy::Block(Duration::MAX)
    }
}

/// An explicit thread boundary.
///
/// Pushing into a `Queue` hands the buffer off through a bounded channel
/// and returns immediately — it never blocks the caller on whatever is
/// downstream (unless the channel is full and `policy` is `Block`). A
/// dedicated worker thread owns everything downstream of the queue and
/// drives it via direct `Sink::consume` calls, until it hits another
/// `Queue`.
///
/// [`ControlMsg`] crosses this same thread boundary through a separate
/// channel from data. The worker checks that channel before entering its
/// combined wait on every iteration, so a control message already pending
/// at that point jumps ahead of the data backlog. A control message that
/// arrives in the narrow window after that check can race one ready data
/// buffer in `select!`, but is checked again before another buffer is
/// pulled. Every worker acks a control message *before* acting on
/// it any further (e.g. before blocking on `Pause`), so the channel stays
/// responsive to the next one — `Resume`/`Stop` always reaches a paused
/// worker immediately, it's never stuck behind the pause itself. See the
/// worker loop below.
///
/// A queue built by a pipeline also holds back for the pipeline's
/// interrupts. `Pipeline::pause`, `stop`, `finish` and a seek interrupt a
/// paced wait before their request has reached every queue, and an
/// interrupted `Pacer` takes what it is handed without waiting; a worker
/// that went on feeding it in that time moved its whole backlog into the
/// pacer. So from the interrupt until the pipeline has every source's
/// acknowledgement the worker takes nothing but control, and the thread
/// handing buffers over is not left blocked on a channel that will not
/// drain: a buffer that finds it full goes in past its capacity, behind
/// everything already there. The channel itself has no bound for that
/// reason; the queue keeps to `capacity` otherwise.
///
/// Cheap elements (e.g. a muxer sitting right after an encoder) should
/// simply *not* have a `Queue` between them and their upstream — they run
/// as a direct call on the upstream element's thread instead of paying for
/// a dedicated thread they don't need.
///
/// A failing `downstream.consume()` doesn't end the worker thread either —
/// that buffer is dropped, `BusEvent::Error` is posted, and the loop moves
/// on to the next one. This crate never decides an error is fatal on your
/// behalf; watch [`crate::pipeline::Pipeline::bus`] and call
/// [`crate::pipeline::Pipeline::stop`] yourself if a particular error
/// means the whole pipeline should end.
///
/// Nor does `MediaBuffer::Eos`. The worker forwards it behind everything
/// queued before it, posts [`BusEvent::Eos`] once downstream has accepted
/// it, and goes back to waiting: the end of a stream is not the end of the
/// pipeline. A source that parks at its end — [`crate::elements::FileDemuxer`]
/// played to the last packet — can still be sought, and that seek's
/// `Flush`, `Seek`, and `Preroll`, and then the new stream, reach
/// downstream through this same worker. Only `ControlMsg::Stop` or dropping
/// the `Queue` ends it, and both still join it: dropping one idle after
/// `Eos` returns as soon as its worker has woken.
pub struct Queue {
    pp_log: PpLog,
    name: Arc<str>,
    tx: Sender<Numbered>,
    policy: OverflowPolicy,
    bus: Bus,
    handle: Option<JoinHandle<()>>,
    control: ControlSender,
    /// Set by [`Queue::drop`], read by the worker's own wait loops
    /// ([`worker_loop`], [`apply_control`]'s pause loop) — the one signal
    /// that reaches the worker no matter which of those it's currently
    /// blocked in, without competing with (and possibly cutting off)
    /// whatever real data/control traffic is already legitimately queued.
    /// See [`Queue::drop`] for why neither channel alone can play this
    /// role safely.
    stop: Arc<AtomicBool>,
    /// Where this queue is counted, when a pipeline built it — see
    /// [`crate::stats`]. `None` for one spawned by hand.
    counters: Option<Arc<ElementCounters>>,
    /// How many buffers may wait ahead of the worker — the channel's own
    /// bound is none; see the type docs. Zero hands each over as the worker
    /// takes it.
    capacity: usize,
    /// Rings the worker — for the pipeline's state moving on, and for this
    /// queue being dropped.
    worker_bell: Bell,
    /// Rings the thread handing buffers over — for room made, an interrupt,
    /// and the worker ending.
    room: Bell,
    /// How many `Eos` this queue has taken and not yet handed on or
    /// discarded — see [`Inbox::owes_an_end`].
    ends_owed: Arc<AtomicUsize>,
    /// The pipeline's playback state, whose timeline a buffer is handed over
    /// on — see [`crate::timeline`]. `None` for one spawned by hand.
    state: Option<Arc<PlaybackState>>,
}

impl Queue {
    /// Spawns with [`OverflowPolicy::default`]. Use
    /// [`Queue::spawn_with_policy`] to drop instead of blocking when full.
    ///
    /// `capacity` is the number of ordinary media buffers that may wait ahead
    /// of the worker; zero creates a rendezvous channel with no backlog.
    /// Returns [`QueueError::ThreadSpawn`] if its worker cannot be created.
    pub fn spawn(
        name: impl Into<String>,
        capacity: usize,
        downstream: Box<dyn Sink>,
        bus: Bus,
        pipeline_id: Option<&str>,
    ) -> Result<Queue> {
        Self::spawn_with_policy(
            name,
            capacity,
            downstream,
            bus,
            OverflowPolicy::default(),
            pipeline_id,
        )
    }

    /// Spawns the worker thread that owns `downstream` and starts pulling
    /// from the channel immediately. `pipeline_id` (typically the owning
    /// [`crate::pipeline::Pipeline`]'s own id — see
    /// [`crate::pipeline::ChainBuilder`], which is what actually passes
    /// one when this `Queue` came from a `.queue()`/`.queue_with_policy()`
    /// call) becomes this `Queue`'s `pp_log` `pipeline_id`; `None` if it
    /// wasn't built through a `Pipeline` at all (e.g. the tests below).
    /// `capacity` may be zero for a rendezvous channel; otherwise it is the
    /// maximum number of ordinary media buffers waiting ahead of the worker.
    /// Returns [`QueueError::ThreadSpawn`] without retaining `downstream` if
    /// the worker cannot be created.
    pub fn spawn_with_policy(
        name: impl Into<String>,
        capacity: usize,
        downstream: Box<dyn Sink>,
        bus: Bus,
        policy: OverflowPolicy,
        pipeline_id: Option<&str>,
    ) -> Result<Queue> {
        Self::spawn_with_policy_using(
            name,
            capacity,
            downstream,
            bus,
            policy,
            pipeline_id,
            None,
            None,
            |thread_name, task| thread::Builder::new().name(thread_name).spawn(task),
        )
    }

    /// [`Self::spawn_with_policy`] as a pipeline's chain builds it: counted,
    /// so the queue is reported with the rest of the graph, and on the
    /// pipeline's playback state, so it holds back for its interrupts.
    pub(crate) fn spawn_in_pipeline(
        name: impl Into<String>,
        capacity: usize,
        downstream: Box<dyn Sink>,
        bus: Bus,
        policy: OverflowPolicy,
        counters: Arc<ElementCounters>,
        context: &Context,
    ) -> Result<Queue> {
        Self::spawn_with_policy_using(
            name,
            capacity,
            downstream,
            bus,
            policy,
            Some(&context.pipeline_id),
            Some(counters),
            Some(Arc::clone(&context.state)),
            |thread_name, task| thread::Builder::new().name(thread_name).spawn(task),
        )
    }

    // One argument over clippy's count, and every one of them is what a
    // queue is made of; a struct would be this signature with a name on it.
    #[allow(clippy::too_many_arguments)]
    fn spawn_with_policy_using(
        name: impl Into<String>,
        capacity: usize,
        downstream: Box<dyn Sink>,
        bus: Bus,
        policy: OverflowPolicy,
        pipeline_id: Option<&str>,
        counters: Option<Arc<ElementCounters>>,
        state: Option<Arc<PlaybackState>>,
        spawn: impl FnOnce(
            String,
            Box<dyn FnOnce() + Send + 'static>,
        ) -> std::io::Result<JoinHandle<()>>,
    ) -> Result<Queue> {
        // Stored as `Arc<str>` (not `String`) so the `worker_name.clone()`
        // below, and every subsequent `BusEvent` this posts, are a
        // refcount bump instead of a fresh allocation — `Dropped` in
        // particular can fire once per buffer under sustained overflow.
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::Queue, &name, pipeline_id);
        let (tx, rx) = unbounded::<Numbered>();
        // The pipeline's state where there is a pipeline: it says when a
        // paused worker may go on. One spawned by hand keeps its own, moved
        // on by what `control` sends it.
        let (control_tx, control_rx) = match &state {
            Some(state) => control::channel_in(state),
            None => control::channel(),
        };
        let worker_name = name.clone();
        let worker_bus = bus.clone();
        let worker_pp_log = pp_log.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = stop.clone();
        if let Some(queue) = counters.as_deref().and_then(ElementCounters::queue) {
            queue.watch(tx.clone());
        }
        let worker_counters = counters.clone();
        let ends_owed = Arc::new(AtomicUsize::new(0));
        // Both rung by every move of the state this queue's control channel
        // reads — the pipeline's, or its own.
        let worker_bell = Bell::new();
        let room = Bell::new();
        control_rx.state().listen(&worker_bell);
        control_rx.state().listen(&room);
        let worker_inbox = Inbox {
            data: rx,
            state: state.clone(),
            ends_owed: ends_owed.clone(),
            bell: worker_bell.clone(),
            room: room.clone(),
            stopped_during_interrupt: Cell::new(None),
        };

        // `Builder::name` panics on interior NULs. Queue names are caller
        // input and remain unchanged for element/log identity; only the OS
        // thread's diagnostic label needs this sanitization.
        let thread_name = format!("queue:{worker_name}").replace('\0', "�");
        let handle = spawn(
            thread_name.clone(),
            Box::new(move || {
                // What this worker hands on is on the timeline of what it
                // carries — see `crate::timeline::carry_on`.
                if let Some(state) = &worker_inbox.state {
                    crate::timeline::enter(state);
                }
                worker_loop(
                    worker_inbox,
                    control_rx,
                    downstream,
                    worker_bus,
                    worker_name,
                    worker_pp_log,
                    worker_stop,
                    worker_counters,
                )
            }),
        )
        .map_err(|source| QueueError::ThreadSpawn(ThreadSpawnError::new(thread_name, source)))?;
        pp_info!(pp_log: &pp_log, "spawned: capacity={capacity}, policy={policy:?}");

        Ok(Queue {
            name,
            pp_log,
            tx,
            policy,
            bus,
            handle: Some(handle),
            control: control_tx,
            stop,
            counters,
            capacity,
            worker_bell,
            room,
            ends_owed,
            state,
        })
    }
}

impl Element for Queue {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::Queue
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Sink for Queue {
    fn ready_consume(&mut self) -> bool {
        match self.policy {
            // Dropping the incoming buffer is this policy's defined way to
            // make progress. Reporting "not ready" here would make an
            // upstream Queue stop before `consume` can perform that drop,
            // silently turning DropNewest into blocking backpressure.
            OverflowPolicy::DropNewest => true,
            OverflowPolicy::Block(_) => self.has_room(),
        }
    }

    /// A Queue neither inspects nor transforms what it carries, so it
    /// accepts every kind and — see `ChainBuilder::queue_with_policy` —
    /// passes the upstream contract straight through to whatever it
    /// feeds. Without that, a check would go dark at the first thread
    /// boundary in the pipeline.
    fn input_contract(&self) -> InputContract {
        InputContract::Any
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        // EOS must never be dropped, regardless of policy: unlike an
        // explicit Stop or Queue::drop's private stop flag, this is the
        // natural-completion signal, and it reaches downstream only after
        // everything queued before it has. The policy timeout intentionally
        // does not apply to this send.
        if buf.is_eos() {
            pp_trace!(pp_log: &self.pp_log, "event=eos phase=received");
            // Counted before it can be taken, so the worker never counts out
            // one that was not yet counted in.
            self.ends_owed.fetch_add(1, Ordering::AcqRel);
            let result = self
                .hand_over(Numbered::on(self.state.as_ref(), buf), Duration::MAX)
                .map_err(|_| QueueError::ChannelClosed.into());
            if result.is_err() {
                self.ends_owed.fetch_sub(1, Ordering::AcqRel);
            }
            match &result {
                Ok(()) => pp_trace!(
                    pp_log: &self.pp_log,
                    "event=eos phase=queued outcome=ok"
                ),
                Err(error) => pp_trace!(
                    pp_log: &self.pp_log,
                    "event=eos phase=queued outcome=error error={error}"
                ),
            }
            return result;
        }

        let counters = self.counters.as_deref();
        if let Some(counters) = counters {
            counters.arrived();
        }
        // Numbered as the thread handing it over is, and carried so: see
        // `crate::timeline`.
        let buf = Numbered::on(self.state.as_ref(), buf);
        let result = match self.policy {
            OverflowPolicy::Block(timeout) => {
                // Timed only when it had to wait: a send into room is not
                // backpressure, and timing every one would cost a clock
                // reading for nothing.
                let waited = self.tx.is_full().then(Instant::now);
                let sent = self.hand_over(buf, timeout);
                if let (Some(started), Some(queue)) =
                    (waited, counters.and_then(ElementCounters::queue))
                {
                    queue.blocked(started.elapsed());
                }
                match sent {
                    Ok(()) => Ok(()),
                    Err(HandOverError::TimedOut) => {
                        Err(QueueError::SendTimedOut { after: timeout }.into())
                    }
                    Err(HandOverError::Closed) => Err(QueueError::ChannelClosed.into()),
                }
            }
            OverflowPolicy::DropNewest => {
                if self.has_room() {
                    self.tx
                        .send(buf)
                        .map_err(|_| QueueError::ChannelClosed.into())
                } else {
                    if let Some(queue) = counters.and_then(ElementCounters::queue) {
                        queue.dropped();
                    }
                    self.bus.post(
                        &self.pp_log,
                        BusEvent::Dropped {
                            element_type: ElementType::Queue,
                            name: self.name.clone(),
                        },
                    );
                    Ok(())
                }
            }
        };
        if result.is_err()
            && let Some(counters) = counters
        {
            counters.failed();
        }
        result
    }

    fn control(&mut self, msg: &ControlMsg) -> Result<()> {
        // Blocks until the worker — and everything downstream of it — has
        // finished handling this. Never stuck behind a data backlog: the
        // worker checks this channel before every data buffer it pulls
        // (see `worker_loop`), and while paused it's blocked *only* on
        // this channel, so a `consume()` blocked sending data upstream of
        // a paused queue just sits in ordinary backpressure — nothing
        // feeds this queue while it's paused, since `Pause` blocks
        // whatever's upstream the same way, all the way back to the
        // source (see [`crate::control::drain_control`]).
        pp_trace!(
            pp_log: &self.pp_log,
            "event=control control={msg:?} phase=received"
        );
        self.control.send(msg.clone());
        pp_trace!(
            pp_log: &self.pp_log,
            "event=control control={msg:?} phase=completed outcome=ok"
        );
        Ok(())
    }
}

impl Drop for Queue {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            // Wakes the worker wherever it waits — idle, paused, or on a
            // downstream that is not ready — which is what ends one nothing
            // else would: a `.queue()`-having `Pipeline` dropped without
            // ever being `run()`, or a bare `Queue` paused and then dropped
            // without `Resume`/`Stop`. Not by closing a channel, which
            // would race whatever real data and control are still queued:
            // a woken worker drains what is waiting before it ends, as
            // `block_never_drops` and friends rely on.
            self.stop.store(true, Ordering::Relaxed);
            self.worker_bell.ring();
            pp_info!(pp_log: &self.pp_log, "dropped: joining worker");
            let _ = handle.join();
        }
    }
}

/// Why [`Queue::hand_over`] could not hand a buffer over.
enum HandOverError {
    TimedOut,
    Closed,
}

impl Queue {
    /// Whether a buffer handed over now would wait within `capacity`.
    fn has_room(&self) -> bool {
        self.tx.len() < self.capacity.max(1)
    }

    /// Whether the worker has ended, so nothing will take what is handed
    /// over.
    fn worker_gone(&self) -> bool {
        self.handle.as_ref().is_none_or(JoinHandle::is_finished)
    }

    /// Puts `buf` in the channel once there is room, waiting up to `timeout`
    /// for some — unless an interrupt comes while it waits, when `buf` goes
    /// in past the queue's capacity and this returns at once.
    ///
    /// Why it may not simply wait: while an interrupt is out the worker takes
    /// nothing (see `worker_loop`), and what ends the interrupt is the request
    /// behind it, which the thread calling this is the one to pass on. Before
    /// the worker held back, it did make room here — by feeding every
    /// buffer in the channel to an interrupted `Pacer`, which took each
    /// without waiting and held them all, so a pause left a whole queue's worth
    /// of frames out of their pool.
    ///
    /// Waits on [`Self::room`], which the worker rings as it takes a buffer
    /// and the pipeline's state rings as an interrupt is raised.
    fn hand_over(
        &self,
        buf: Numbered,
        timeout: Duration,
    ) -> std::result::Result<(), HandOverError> {
        let deadline = Instant::now().checked_add(timeout);
        let interrupted = || {
            self.state
                .as_deref()
                .is_some_and(PlaybackState::interrupt_pending)
        };
        let wait_for_room = |until: Option<Instant>| -> bool {
            let wait = until.map_or(Duration::MAX, |until| {
                until.saturating_duration_since(Instant::now())
            });
            !wait.is_zero() && {
                let _ = self.room.rings().recv_timeout(wait);
                true
            }
        };
        loop {
            if self.worker_gone() {
                return Err(HandOverError::Closed);
            }
            if self.has_room() || interrupted() {
                break;
            }
            if !wait_for_room(deadline) {
                return Err(HandOverError::TimedOut);
            }
        }
        self.tx.send(buf).map_err(|_| HandOverError::Closed)?;
        // No backlog at all: handed over is taken, or it would not be.
        if self.capacity == 0 {
            while !self.tx.is_empty() && !interrupted() && !self.worker_gone() {
                wait_for_room(None);
            }
        }
        Ok(())
    }
}

/// Where a worker takes its buffers from — and not while an interrupt is
/// out — and the bells it waits on and rings.
struct Inbox {
    data: Receiver<Numbered>,
    /// The pipeline's playback state, against whose timeline what is
    /// carried is judged before it goes on and whose interrupts the worker
    /// holds back for — `None` for a queue spawned by hand, which does
    /// neither.
    state: Option<Arc<PlaybackState>>,
    /// Shared with the [`Queue`], which counts an `Eos` in as it takes one;
    /// the worker counts it out as it hands it on or a `Flush` discards it.
    ends_owed: Arc<AtomicUsize>,
    /// What wakes the worker — see [`Queue::worker_bell`].
    bell: Bell,
    /// Rung as the worker makes room — see [`Queue::room`].
    room: Bell,
    /// When the worker first found its queue dropped while an interrupt was
    /// out — see [`Self::held_back`].
    stopped_during_interrupt: Cell<Option<Instant>>,
}

impl Inbox {
    /// Whether the worker should take nothing for now: an interrupt is out,
    /// so whatever it fed downstream would only be held by an interrupted
    /// `Pacer` rather than played.
    ///
    /// Dropped as well, all the same. A source that `finish` has ended
    /// drops its queue right after it answers the request, a moment before
    /// the pipeline settles the interrupt that went with it; a worker that
    /// drained in that moment fed an interrupted `Pacer`, which kept what it
    /// was handed — the end of the stream with it — and was dropped with it
    /// before anything asked again. The settle rings the worker, so this
    /// costs no more than that moment. Not for longer than
    /// [`SETTLE_WAIT_LIMIT`]: past it, nothing is coming to settle the
    /// interrupt, and the worker drains what it holds as it always has.
    fn held_back(&self, stop: &AtomicBool) -> bool {
        if !self
            .state
            .as_deref()
            .is_some_and(PlaybackState::interrupt_pending)
        {
            return false;
        }
        if !stop.load(Ordering::Relaxed) {
            return true;
        }
        let since = self.stopped_during_interrupt.get().unwrap_or_else(|| {
            let now = Instant::now();
            self.stopped_during_interrupt.set(Some(now));
            now
        });
        since.elapsed() < SETTLE_WAIT_LIMIT
    }

    /// Whether an `Eos` handed over is still here, on its way.
    ///
    /// What decides whether the drop of this queue is an ending or an
    /// abandonment. A source that has finished put an `Eos` behind the last
    /// of its stream, and everything up to it is owed downstream — so the
    /// worker stays until it has gone, however busy downstream is for now.
    /// A branch detached, or a pipeline dropped mid-stream, sent none: what
    /// is queued then is abandoned, and a downstream that will never take
    /// it — a wedged one, which is why a branch gets detached — must not
    /// keep the drop waiting.
    ///
    /// Nor once the pipeline is stopped, `Eos` or not. A stop abandons the
    /// stream, and nothing downstream takes anything again: the terminals
    /// hold for good. A source that ended just before a stop dropped its
    /// queue with an `Eos` in it, a moment before the `Stop` message could
    /// reach the queue behind it — and the worker waited, for a terminal
    /// that would never be ready, while the stop joined the source and the
    /// source joined it.
    fn owes_an_end(&self) -> bool {
        self.ends_owed.load(Ordering::Acquire) > 0
            && !self.state.as_deref().is_some_and(PlaybackState::is_stopped)
    }

    /// Counts out `ends` `Eos` that will not be handed on from here.
    fn discarded(&self, ends: usize) {
        self.ends_owed.fetch_sub(ends, Ordering::AcqRel);
    }
}

/// Rings the thread handing buffers over as the worker ends, so one waiting
/// for room finds there will never be any.
struct RingOnEnd(Bell);

impl Drop for RingOnEnd {
    fn drop(&mut self) {
        self.0.ring();
    }
}

/// Owns `downstream` on its own thread: pulls from `data_rx` and calls
/// `downstream.consume()`, same as before. Every iteration first checks
/// `control_rx` non-blockingly, so a control request already pending there
/// is handled before the next data buffer, however deep the backlog. A
/// request arriving immediately afterward can race one ready data item in
/// the combined `select!`; the next iteration checks control first again.
/// `Pause` blocks this
/// whole function (and therefore `downstream`) right here, without
/// touching `data_rx` at all, until `Resume`/`Stop`.
// Everything a worker is handed when it is spawned, one argument apiece;
// `spawn_with_policy_using` carries the same list for the same reason.
#[allow(clippy::too_many_arguments)]
fn worker_loop(
    inbox: Inbox,
    control_rx: ControlReceiver,
    mut downstream: Box<dyn Sink>,
    bus: Bus,
    name: Arc<str>,
    // Cloned from `Queue`'s own field before this thread was spawned —
    // same value, not rebuilt here, so a `pipeline_id` passed to
    // `spawn_with_policy` actually reaches this thread's own log lines
    // too.
    pp_log: PpLog,
    stop: Arc<AtomicBool>,
    counters: Option<Arc<ElementCounters>>,
) {
    pp_info!(pp_log: &pp_log, "worker: starting");
    let error_reporter = QueueErrorReporter {
        bus: &bus,
        downstream: (downstream.element_type(), downstream.name()),
        pp_log: &pp_log,
    };
    let worker = Worker {
        bus: &bus,
        inbox: &inbox,
        name: &name,
        pp_log: &pp_log,
        counters: counters.as_deref(),
        errors: &error_reporter,
        dropped_from: Cell::new(UNNUMBERED),
    };
    // However this ends, a thread waiting to hand a buffer over hears of it.
    let _ends = RingOnEnd(inbox.room.clone());
    let apply = |msg, ack: &Sender<()>, downstream: &mut Box<dyn Sink>| {
        apply_control(
            &inbox,
            downstream,
            msg,
            ack,
            &control_rx,
            &error_reporter,
            &stop,
        )
    };
    loop {
        if let Some((request, ack)) = control_rx.try_recv() {
            let RequestKind::Control(msg) = request else {
                let _ = ack.send(());
                continue;
            };
            if apply(msg, &ack, &mut downstream) {
                pp_info!(pp_log: &pp_log, "worker: stopped");
                return;
            }
            continue;
        }

        // Dropped: what is waiting still goes, and once nothing is, this ends.
        let stopping = stop.load(Ordering::Relaxed);
        if stopping && inbox.data.is_empty() {
            pp_info!(pp_log: &pp_log, "worker: stop flag set, ending");
            return;
        }

        // Neither while an interrupt is out — see `Inbox::held_back` — nor
        // while downstream is not ready does this take a buffer. It waits
        // for what will change either: a request, or its bell — rung as
        // the pipeline's state moves on, and as the queue is dropped. Only
        // a downstream that becomes ready without saying so is looked at
        // again on a timer; see `READINESS_POLL_INTERVAL`.
        let held_back = inbox.held_back(&stop);
        if held_back || !downstream.ready_consume() {
            // Held back, the settle rings it — unless the queue has been
            // dropped, when it also has to notice nothing is coming to.
            let poll = if held_back && !stop.load(Ordering::Relaxed) {
                Duration::MAX
            } else {
                READINESS_POLL_INTERVAL
            };
            select! {
                recv(control_rx.rx) -> request => match request {
                    Ok(request) => {
                        let RequestKind::Control(msg) = request.kind else {
                            let _ = request.ack.send(());
                            continue;
                        };
                        if apply(msg, &request.ack, &mut downstream) {
                            pp_info!(pp_log: &pp_log, "worker: stopped");
                            return;
                        }
                    }
                    Err(_) => return,
                },
                recv(inbox.bell.rings()) -> _ => {}
                default(poll) => {}
            }
            // Dropped with downstream not taking anything: what is queued
            // is abandoned — unless it holds an `Eos`, which is owed
            // downstream however long it takes: see `Inbox::owes_an_end`.
            // Ending here regardless dropped the end of the stream, which
            // is what a `finish` of a playing file lost: the source ends,
            // its queue is dropped, and the worker found its downstream
            // busy.
            if !held_back && stop.load(Ordering::Relaxed) && !inbox.owes_an_end() {
                pp_info!(pp_log: &pp_log, "worker: stop flag set, ending");
                return;
            }
            continue;
        }

        select! {
            recv(control_rx.rx) -> request => match request {
                Ok(request) => {
                    let RequestKind::Control(msg) = request.kind else {
                        let _ = request.ack.send(());
                        continue;
                    };
                    if apply(msg, &request.ack, &mut downstream) {
                        pp_info!(pp_log: &pp_log, "worker: stopped");
                        return;
                    }
                }
                Err(_) => {
                    pp_info!(pp_log: &pp_log, "worker: control channel gone, ending");
                    return; // sender (this Queue) dropped
                }
            },
            recv(inbox.data) -> buf => match buf {
                Ok(buf) => {
                    // Room for the thread handing buffers over, before this
                    // one goes on, which may take a while.
                    inbox.room.ring();
                    forward(&mut downstream, buf, &worker);
                }
                Err(_) => {
                    pp_info!(pp_log: &pp_log, "worker: producer (this Queue) gone, ending");
                    return;
                }
            },
            // The state moved on, or the queue was dropped: look again.
            recv(inbox.bell.rings()) -> _ => {}
        }
    }
}

/// What handing a buffer on needs from its worker, besides the element it
/// goes to.
struct Worker<'a> {
    bus: &'a Bus,
    inbox: &'a Inbox,
    name: &'a Arc<str>,
    pp_log: &'a PpLog,
    counters: Option<&'a ElementCounters>,
    errors: &'a QueueErrorReporter<'a>,
    /// The timeline this worker last dropped a buffer from, so a seek's
    /// leftovers are logged once rather than once a buffer.
    dropped_from: Cell<u64>,
}

/// Hands one buffer to `downstream`, counting it and reporting a failure on
/// the bus — a failure drops that buffer and nothing else.
fn forward(downstream: &mut Box<dyn Sink>, carried: Numbered, worker: &Worker<'_>) {
    let Numbered { number, buf } = carried;
    if worker
        .inbox
        .state
        .as_deref()
        .is_some_and(|state| state.is_behind(number))
    {
        // From a position the pipeline has since left — what a `Flush`
        // should have discarded and a thread's timing let past it.
        if worker.dropped_from.replace(number) != number {
            pp_debug!(
                pp_log: worker.pp_log,
                "dropping what arrives from timeline {number}, which a seek has left"
            );
        }
        if buf.is_eos() {
            worker.inbox.discarded(1);
        }
        return;
    }
    // What the elements after this make of it is on the same timeline.
    crate::timeline::carry_on(number);
    let is_eos = buf.is_eos();
    let call = worker
        .counters
        .map(|counters| counters.begin_forwarding(is_eos));
    let result = downstream.consume(buf);
    if let Some(call) = call {
        call.end(result.is_ok());
    }
    if is_eos {
        // Handed on, taken or refused: either way it is no longer here.
        worker.inbox.discarded(1);
    }
    match result {
        Ok(()) => {
            if is_eos {
                pp_trace!(
                    pp_log: worker.pp_log,
                    "event=eos phase=completed outcome=ok"
                );
                worker.bus.post(
                    worker.pp_log,
                    BusEvent::Eos {
                        element_type: ElementType::Queue,
                        name: worker.name.clone(),
                    },
                );
                // Not a return: `Eos` ends a stream, not this queue. A
                // source that parks at its end can still be sought, and the
                // `Flush`/`Seek`/`Preroll` that follow need a worker here to
                // carry them and the new stream — see the type docs.
            }
        }
        Err(error) => {
            if is_eos {
                pp_trace!(
                    pp_log: worker.pp_log,
                    "event=eos phase=completed outcome=error error={error}"
                );
            }
            // Report and move on to the next buffer — this one's dropped,
            // but nothing else dies over it. Whoever's watching the bus
            // decides whether the error is fatal enough to call
            // `Pipeline::stop`.
            worker.errors.post(error);
        }
    }
}

/// Applies one control message to `downstream`, acking it, then — only
/// for `Pause` — blocking this thread on `control_rx` alone (never
/// touching `data_rx`) until `Resume`/`Preroll`/`Stop`. Returns `true` once `Stop`
/// has been handled, meaning the caller (`worker_loop`) should exit.
fn apply_control(
    inbox: &Inbox,
    downstream: &mut Box<dyn Sink>,
    msg: ControlMsg,
    ack: &Sender<()>,
    control_rx: &ControlReceiver,
    error_reporter: &QueueErrorReporter<'_>,
    stop: &AtomicBool,
) -> bool {
    pp_trace!(
        pp_log: error_reporter.pp_log,
        "event=control control={msg:?} phase=forwarding"
    );
    inbox.discarded(discard_stale_data(&inbox.data, &msg));
    inbox.room.ring();
    forward_control(downstream, msg.clone(), error_reporter);
    let is_stop = msg == ControlMsg::Stop;
    let _ = ack.send(());
    if is_stop {
        return true;
    }
    if msg != ControlMsg::Pause {
        return false;
    }
    loop {
        // On control, and on the bell as well: `Queue::drop` rings it, and a
        // worker paused for good — a bare `Queue`, not reached through a
        // `Pipeline`, paused and dropped with no `Resume` or `Stop` ever
        // coming — has nothing else to wake it. Nothing feeds this queue
        // while paused (see the type-level docs), so there is no traffic
        // this could cut off.
        let request = select! {
            recv(control_rx.rx) -> request => request,
            recv(inbox.bell.rings()) -> _ => {
                if stop.load(Ordering::Relaxed) {
                    pp_info!(pp_log: error_reporter.pp_log, "worker: stop flag set while paused, ending");
                    return true;
                }
                continue;
            }
        };
        let (msg, ack) = match request {
            Ok(req) => {
                let RequestKind::Control(msg) = req.kind else {
                    let _ = req.ack.send(());
                    continue;
                };
                (msg, req.ack)
            }
            Err(_) => {
                pp_info!(pp_log: error_reporter.pp_log, "worker: control channel gone while paused, ending");
                return true; // sender gone — treat like Stop
            }
        };
        pp_trace!(
            pp_log: error_reporter.pp_log,
            "event=control control={msg:?} phase=forwarding"
        );
        inbox.discarded(discard_stale_data(&inbox.data, &msg));
        inbox.room.ring();
        forward_control(downstream, msg.clone(), error_reporter);
        let is_stop = msg == ControlMsg::Stop;
        let _ = ack.send(());
        if is_stop {
            return true;
        }
        if !control_rx.state().holds() {
            // Playback has moved on — to playing, or to a preroll — before
            // this message came to say so; see `crate::playback_state`.
            return false;
        }
        // Another Pause while already paused: already forwarded above, keep waiting.
    }
}

/// Reports a downstream failure on the bus, as the *downstream element's*.
///
/// # Why not as the Queue's own
///
/// A `Queue` does not fail here — what failed is whatever it handed the
/// buffer to, and the queue is only the place the failure could no longer be
/// returned to a caller. Posting it under `ElementType::Queue` said "a queue
/// broke", which is never true, and it erased the one thing an observer
/// needs: *which* element stopped working. A muxer losing its connection and
/// an encoder refusing a frame arrived indistinguishable, both labelled with
/// the name of the queue in front of them.
///
/// # Where the identity comes from
///
/// From the error, when it has one. Every stage in a built chain is wrapped
/// in a tracer that stamps a failure with its own identity on the way past,
/// and the first stamp wins — so an error arriving here already names the
/// element that raised it, however deep in the chain that was. See
/// [`Error::traced_at`](crate::error::Error::traced_at).
///
/// From the element this queue feeds, when it has not. That happens where
/// there is no chain to be traced through: a `Queue` built directly around a
/// sink, which is what a test does and what an element assembling its own
/// plumbing may do. Naming what the buffer was handed to is then both true
/// and the best available.
struct QueueErrorReporter<'a> {
    bus: &'a Bus,
    /// Taken once, before the worker loop: the element a queue feeds is
    /// fixed for the life of the thread, and reading it at failure time
    /// would need the borrow that the failing call is already holding.
    downstream: (ElementType, Arc<str>),
    pp_log: &'a PpLog,
}

impl QueueErrorReporter<'_> {
    fn post(&self, error: crate::error::Error) {
        let (element_type, name) = match error.origin() {
            Some(origin) => (origin.element_type, origin.name.clone()),
            None => (self.downstream.0, self.downstream.1.clone()),
        };
        self.bus.post(
            self.pp_log,
            BusEvent::Error {
                element_type,
                name,
                error,
            },
        );
    }
}

/// Forwards control without turning one downstream failure into a stuck
/// synchronous caller or a dead Queue worker. The request is still acked by
/// [`apply_control`], while the failure is exposed through the same Bus path
/// used for `consume` failures.
fn forward_control(
    downstream: &mut Box<dyn Sink>,
    msg: ControlMsg,
    error_reporter: &QueueErrorReporter<'_>,
) {
    if let Err(error) = downstream.control(&msg) {
        error_reporter.post(error);
    }
}

/// Drops everything already buffered in `data_rx` without processing it —
/// only for `Flush`. That data belongs to the old timeline, so delivering it
/// after the following seek would show stale frames instead of starting at
/// the new position.
/// `Pause`/`Resume`/`Stop` leave `data_rx` alone — see the type-level
/// docs on why that's safe (nothing feeds a paused/stopped queue in the
/// first place).
/// Empties the channel on `Flush`, and says how many `Eos` went with it.
fn discard_stale_data(data_rx: &Receiver<Numbered>, msg: &ControlMsg) -> usize {
    let mut ends = 0;
    if *msg == ControlMsg::Flush {
        while let Ok(held) = data_rx.try_recv() {
            ends += usize::from(held.buf.is_eos());
        }
    }
    ends
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            Arc, Mutex,
            atomic::{AtomicUsize, Ordering},
        },
        thread,
        time::Duration,
    };

    use super::*;
    use crate::{bus::Bus, control::PrerollContext};

    #[test]
    fn flush_discards_backlog_but_seek_does_not() {
        let (tx, rx) = crossbeam_channel::bounded(2);
        tx.send(Numbered::on(None, packet())).unwrap();

        discard_stale_data(&rx, &ControlMsg::Seek(Duration::from_secs(1)));
        assert!(rx.try_recv().is_ok(), "Seek must not own Queue flushing");

        tx.send(Numbered::on(None, packet())).unwrap();
        discard_stale_data(&rx, &ControlMsg::Flush);
        assert!(rx.try_recv().is_err(), "Flush must discard queued data");
    }

    /// A downstream that's slower than the producer, so a small queue
    /// behind it actually fills up during the test.
    struct SlowCounter {
        pp_log: PpLog,
        count: Arc<AtomicUsize>,
    }

    impl Element for SlowCounter {
        fn name(&self) -> Arc<str> {
            "slow-counter".into()
        }

        fn element_type(&self) -> ElementType {
            ElementType::Other
        }

        fn pp_log(&self) -> &PpLog {
            &self.pp_log
        }

        fn pp_log_mut(&mut self) -> &mut PpLog {
            &mut self.pp_log
        }
    }

    impl Sink for SlowCounter {
        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            if let MediaBuffer::Packet(_) = buf {
                thread::sleep(Duration::from_millis(20));
                self.count.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        }
    }

    fn packet() -> MediaBuffer {
        MediaBuffer::Packet(Arc::new(ffmpeg_next::Packet::empty()))
    }

    struct DropAwareSink {
        dropped: Arc<AtomicBool>,
        pp_log: PpLog,
    }

    impl Drop for DropAwareSink {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::Release);
        }
    }

    impl Element for DropAwareSink {
        fn name(&self) -> Arc<str> {
            "drop-aware".into()
        }

        fn element_type(&self) -> ElementType {
            ElementType::Other
        }

        fn pp_log(&self) -> &PpLog {
            &self.pp_log
        }

        fn pp_log_mut(&mut self) -> &mut PpLog {
            &mut self.pp_log
        }
    }

    impl Sink for DropAwareSink {
        fn consume(&mut self, _buf: MediaBuffer) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn thread_spawn_failure_is_returned_and_releases_downstream() {
        let dropped = Arc::new(AtomicBool::new(false));
        let (bus, _bus_rx) = Bus::new();

        let result = Queue::spawn_with_policy_using(
            "queue",
            1,
            Box::new(DropAwareSink {
                dropped: dropped.clone(),
                pp_log: element_pp_log(ElementType::Other, "drop-aware", None),
            }),
            bus,
            OverflowPolicy::default(),
            None,
            None,
            None,
            |_thread_name, _task| Err(std::io::Error::other("injected spawn failure")),
        );

        assert!(matches!(
            result,
            Err(crate::Error::QueueError(QueueError::ThreadSpawn(_)))
        ));
        assert!(dropped.load(Ordering::Acquire));
    }

    #[test]
    fn interior_nul_in_queue_name_does_not_panic_while_naming_the_worker() {
        let dropped = Arc::new(AtomicBool::new(false));
        let (bus, _bus_rx) = Bus::new();
        let queue = Queue::spawn(
            "nul\0queue",
            1,
            Box::new(DropAwareSink {
                dropped: dropped.clone(),
                pp_log: element_pp_log(ElementType::Other, "drop-aware", None),
            }),
            bus,
            None,
        )
        .unwrap();

        drop(queue);
        assert!(dropped.load(Ordering::Acquire));
    }

    #[test]
    fn block_never_drops() {
        let count = Arc::new(AtomicUsize::new(0));
        let sink = SlowCounter {
            count: count.clone(),
            pp_log: element_pp_log(ElementType::Other, "slow-counter", None),
        };
        let (bus, bus_rx) = Bus::new();

        let mut queue = Queue::spawn_with_policy(
            "test",
            1,
            Box::new(sink),
            bus,
            OverflowPolicy::default(),
            None,
        )
        .unwrap();
        for _ in 0..10 {
            queue.consume(packet()).unwrap();
        }
        queue.consume(MediaBuffer::Eos).unwrap();
        drop(queue); // blocks until the worker drains everything and joins

        assert_eq!(count.load(Ordering::SeqCst), 10);
        assert!(!bus_rx.iter().any(|e| matches!(e, BusEvent::Dropped { .. })));
    }

    /// Refuses to take anything until it is let go, then counts what it
    /// takes and whether an `Eos` came — a downstream that is only full for
    /// now, as a `Queue` in front of a `Pacer` is while pictures wait their
    /// turn.
    struct Gated {
        pp_log: PpLog,
        open: Arc<AtomicBool>,
        count: Arc<AtomicUsize>,
        ended: Arc<AtomicBool>,
    }

    impl Element for Gated {
        fn name(&self) -> Arc<str> {
            "gated".into()
        }

        fn element_type(&self) -> ElementType {
            ElementType::Other
        }

        fn pp_log(&self) -> &PpLog {
            &self.pp_log
        }

        fn pp_log_mut(&mut self) -> &mut PpLog {
            &mut self.pp_log
        }
    }

    impl Sink for Gated {
        fn ready_consume(&mut self) -> bool {
            self.open.load(Ordering::SeqCst)
        }

        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            match buf {
                MediaBuffer::Eos => self.ended.store(true, Ordering::SeqCst),
                _ => {
                    self.count.fetch_add(1, Ordering::SeqCst);
                }
            }
            Ok(())
        }
    }

    /// A queue dropped while its downstream is only busy still hands over
    /// everything it was given, the `Eos` last — dropping is what ends a
    /// finished source's queue, and what that source put in it is the end
    /// of its stream.
    ///
    /// The worker used to take the drop as the end whenever it found its
    /// downstream not ready at that moment, and returned with the rest still
    /// queued: a `finish` of a playing file lost its `Eos` whenever the
    /// queue in front of a `Pacer` was full. Found by the conformance
    /// sequences.
    #[test]
    fn dropping_hands_over_everything_once_a_busy_downstream_takes_it() {
        let open = Arc::new(AtomicBool::new(false));
        let count = Arc::new(AtomicUsize::new(0));
        let ended = Arc::new(AtomicBool::new(false));
        let sink = Gated {
            pp_log: element_pp_log(ElementType::Other, "gated", None),
            open: open.clone(),
            count: count.clone(),
            ended: ended.clone(),
        };
        let (bus, _bus_rx) = Bus::new();
        let mut queue = Queue::spawn_with_policy(
            "test",
            8,
            Box::new(sink),
            bus,
            OverflowPolicy::default(),
            None,
        )
        .unwrap();
        for _ in 0..5 {
            queue.consume(packet()).unwrap();
        }
        queue.consume(MediaBuffer::Eos).unwrap();
        // Busy for longer than the worker's idle wait, so it has seen the
        // drop before anything can go.
        let opener = {
            let open = open.clone();
            thread::spawn(move || {
                thread::sleep(READINESS_POLL_INTERVAL * 4);
                open.store(true, Ordering::SeqCst);
            })
        };
        drop(queue);
        opener.join().unwrap();

        assert_eq!(count.load(Ordering::SeqCst), 5, "what was queued arrived");
        assert!(ended.load(Ordering::SeqCst), "and the Eos behind it");
    }

    /// The other half: a queue dropped with no `Eos` in it is abandoned, and
    /// a downstream that will never take what is queued does not keep the
    /// drop waiting. This is how a wedged branch is detached from a `Tee`.
    #[test]
    fn dropping_without_an_end_abandons_what_a_wedged_downstream_will_not_take() {
        let count = Arc::new(AtomicUsize::new(0));
        let sink = Gated {
            pp_log: element_pp_log(ElementType::Other, "gated", None),
            open: Arc::new(AtomicBool::new(false)),
            count: count.clone(),
            ended: Arc::new(AtomicBool::new(false)),
        };
        let (bus, _bus_rx) = Bus::new();
        let mut queue = Queue::spawn_with_policy(
            "test",
            8,
            Box::new(sink),
            bus,
            OverflowPolicy::default(),
            None,
        )
        .unwrap();
        for _ in 0..5 {
            queue.consume(packet()).unwrap();
        }
        let (dropped, done) = std::sync::mpsc::channel();
        thread::spawn(move || {
            drop(queue);
            let _ = dropped.send(());
        });
        assert!(
            done.recv_timeout(Duration::from_secs(5)).is_ok(),
            "the drop waited on a downstream that will never take anything"
        );
        assert_eq!(count.load(Ordering::SeqCst), 0);
    }

    /// An `Eos` is owed only while something could still take it. Once the
    /// pipeline is stopped its terminals hold for good, so a queue dropped
    /// with an `Eos` in it ends rather than waiting on them.
    ///
    /// What obs-rs's CI hung on for hours: a capture whose source ended as
    /// it was stopped. The source dropped its queue with the `Eos` in it,
    /// the `Stop` could no longer reach the queue behind a source that had
    /// gone, and the worker waited on a terminal that the stop made hold —
    /// while the stop joined the source, and the source joined the worker.
    #[test]
    fn a_stopped_pipeline_is_owed_no_end() {
        let ended = Arc::new(AtomicBool::new(false));
        let sink = Gated {
            pp_log: element_pp_log(ElementType::Other, "gated", None),
            // Closed, as a terminal is once the pipeline is stopped.
            open: Arc::new(AtomicBool::new(false)),
            count: Arc::new(AtomicUsize::new(0)),
            ended: ended.clone(),
        };
        let state = PlaybackState::new();
        let (bus, _bus_rx) = Bus::new();
        let mut queue = Queue::spawn_with_policy_using(
            "test",
            8,
            Box::new(sink),
            bus,
            OverflowPolicy::default(),
            None,
            None,
            Some(Arc::clone(&state)),
            |thread_name, task| thread::Builder::new().name(thread_name).spawn(task),
        )
        .unwrap();
        queue.consume(packet()).unwrap();
        queue.consume(MediaBuffer::Eos).unwrap();
        state.observe(&ControlMsg::Stop);

        let (dropped, done) = std::sync::mpsc::channel();
        thread::spawn(move || {
            drop(queue);
            let _ = dropped.send(());
        });
        assert!(
            done.recv_timeout(Duration::from_secs(5)).is_ok(),
            "the drop waited to hand an Eos to a stopped pipeline"
        );
        assert!(!ended.load(Ordering::SeqCst), "and nothing took it");
    }

    #[test]
    fn block_with_a_finite_timeout_errors_instead_of_blocking_forever() {
        let count = Arc::new(AtomicUsize::new(0));
        let sink = SlowCounter {
            count: count.clone(),
            pp_log: element_pp_log(ElementType::Other, "slow-counter", None),
        };
        let (bus, _bus_rx) = Bus::new();

        // Capacity 1, downstream takes 20ms/item, timeout is 5ms — pushed
        // in a tight loop, some of these sends must outlast their own
        // timeout instead of blocking until the worker catches up.
        let mut queue = Queue::spawn_with_policy(
            "test",
            1,
            Box::new(sink),
            bus,
            OverflowPolicy::Block(Duration::from_millis(5)),
            None,
        )
        .unwrap();
        let mut timed_out = 0;
        for _ in 0..10 {
            match queue.consume(packet()) {
                Ok(()) => {}
                Err(_) => timed_out += 1,
            }
        }
        // Eos isn't subject to the timeout (see `Sink::consume`'s own
        // special-casing) — always goes through even after some sends
        // above timed out.
        queue.consume(MediaBuffer::Eos).unwrap();
        drop(queue); // blocks until the worker drains everything and joins

        assert!(
            timed_out > 0,
            "expected at least one send to time out against a downstream that can't keep up"
        );
    }

    #[test]
    fn drop_newest_drops_when_full_and_reports_on_bus() {
        let count = Arc::new(AtomicUsize::new(0));
        let sink = SlowCounter {
            count: count.clone(),
            pp_log: element_pp_log(ElementType::Other, "slow-counter", None),
        };
        let (bus, bus_rx) = Bus::new();

        let mut queue = Queue::spawn_with_policy(
            "test",
            1,
            Box::new(sink),
            bus,
            OverflowPolicy::DropNewest,
            None,
        )
        .unwrap();
        // Pushed much faster than the 20ms/item downstream can drain a
        // capacity-1 channel, so some of these must get dropped.
        for _ in 0..10 {
            assert!(
                queue.ready_consume(),
                "DropNewest makes progress by dropping even when full"
            );
            queue.consume(packet()).unwrap();
        }
        queue.consume(MediaBuffer::Eos).unwrap(); // never dropped, even under this policy
        drop(queue);

        let processed = count.load(Ordering::SeqCst);
        let dropped = bus_rx
            .iter()
            .filter(|e| matches!(e, BusEvent::Dropped { .. }))
            .count();

        assert!(
            processed < 10,
            "expected some packets to be dropped, but all {processed} were processed"
        );
        assert!(dropped > 0, "expected at least one BusEvent::Dropped");
        assert_eq!(processed + dropped, 10);
    }

    #[test]
    fn pause_stops_delivery_and_resume_lets_it_continue() {
        let count = Arc::new(AtomicUsize::new(0));
        let sink = SlowCounter {
            count: count.clone(),
            pp_log: element_pp_log(ElementType::Other, "slow-counter", None),
        };
        let (bus, _bus_rx) = Bus::new();

        let mut queue = Queue::spawn_with_policy(
            "test",
            8,
            Box::new(sink),
            bus,
            OverflowPolicy::default(),
            None,
        )
        .unwrap();
        queue.control(&ControlMsg::Pause).unwrap(); // blocks until the worker is actually paused

        for _ in 0..3 {
            queue.consume(packet()).unwrap();
        }
        // Worker is paused and not touching data_rx — nothing should have
        // been processed yet, however long we wait.
        thread::sleep(Duration::from_millis(100));
        assert_eq!(count.load(Ordering::SeqCst), 0);

        queue.control(&ControlMsg::Resume).unwrap();
        queue.consume(MediaBuffer::Eos).unwrap();
        drop(queue);

        assert_eq!(count.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn preroll_releases_a_paused_worker_into_data_processing() {
        let count = Arc::new(AtomicUsize::new(0));
        let sink = SlowCounter {
            count: count.clone(),
            pp_log: element_pp_log(ElementType::Other, "slow-counter", None),
        };
        let (bus, _bus_rx) = Bus::new();
        let mut queue = Queue::spawn_with_policy(
            "test",
            8,
            Box::new(sink),
            bus,
            OverflowPolicy::default(),
            None,
        )
        .unwrap();
        queue.control(&ControlMsg::Pause).unwrap();
        queue.consume(packet()).unwrap();

        queue
            .control(&ControlMsg::Preroll(Arc::new(PrerollContext::new([]))))
            .unwrap();
        for _ in 0..50 {
            if count.load(Ordering::SeqCst) == 1 {
                break;
            }
            thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(count.load(Ordering::SeqCst), 1);
        queue.control(&ControlMsg::Stop).unwrap();
    }

    /// Regression test: before `Queue::drop` set its own `stop` flag,
    /// dropping a `Queue` that was never fed a `Stop` control message or
    /// an `Eos` buffer left its worker thread parked on `recv()` with
    /// nothing left to wake it — `drop()`'s own `handle.join()` then hung
    /// forever. This mirrors what happens to a `.queue()`-containing
    /// `Pipeline` that's dropped without ever being `run()`, so if this
    /// test hangs, that fix regressed.
    #[test]
    fn dropping_without_stop_or_eos_does_not_hang() {
        let count = Arc::new(AtomicUsize::new(0));
        let sink = SlowCounter {
            count: count.clone(),
            pp_log: element_pp_log(ElementType::Other, "slow-counter", None),
        };
        let (bus, _bus_rx) = Bus::new();

        let queue = Queue::spawn_with_policy(
            "test",
            8,
            Box::new(sink),
            bus,
            OverflowPolicy::default(),
            None,
        )
        .unwrap();
        drop(queue);
    }

    /// Regression test for the other half of the same bug: a worker
    /// that's specifically inside `apply_control`'s pause loop (blocked on
    /// `control_rx` alone, not `data_rx`) when dropped without ever
    /// getting `Resume`/`Stop` — only reachable by pausing a bare `Queue`
    /// directly (a `Pipeline`-owned one can't be dropped in this state,
    /// see `Queue::drop`'s docs), but the `stop` flag has to wake this
    /// wait loop too, not just `worker_loop`'s.
    #[test]
    fn dropping_while_paused_does_not_hang() {
        let count = Arc::new(AtomicUsize::new(0));
        let sink = SlowCounter {
            count: count.clone(),
            pp_log: element_pp_log(ElementType::Other, "slow-counter", None),
        };
        let (bus, _bus_rx) = Bus::new();

        let mut queue = Queue::spawn_with_policy(
            "test",
            8,
            Box::new(sink),
            bus,
            OverflowPolicy::default(),
            None,
        )
        .unwrap();
        queue.control(&ControlMsg::Pause).unwrap(); // blocks until the worker is actually paused
        drop(queue);
    }

    #[test]
    fn stop_is_synchronous_and_terminates_the_worker() {
        let count = Arc::new(AtomicUsize::new(0));
        let sink = SlowCounter {
            count: count.clone(),
            pp_log: element_pp_log(ElementType::Other, "slow-counter", None),
        };
        let (bus, _bus_rx) = Bus::new();

        let mut queue = Queue::spawn_with_policy(
            "test",
            8,
            Box::new(sink),
            bus,
            OverflowPolicy::default(),
            None,
        )
        .unwrap();
        queue.consume(packet()).unwrap();
        queue.control(&ControlMsg::Stop).unwrap(); // blocks until the worker has exited
        drop(queue); // join should return immediately — the worker already returned
    }

    /// A downstream that fails on the very first `Packet` it sees, then
    /// behaves like `SlowCounter` for every one after.
    struct FailFirstThenCount {
        pp_log: PpLog,
        count: Arc<AtomicUsize>,
        failed_once: bool,
    }

    impl Element for FailFirstThenCount {
        fn name(&self) -> Arc<str> {
            "fail-first".into()
        }

        fn element_type(&self) -> ElementType {
            ElementType::Other
        }

        fn pp_log(&self) -> &PpLog {
            &self.pp_log
        }

        fn pp_log_mut(&mut self) -> &mut PpLog {
            &mut self.pp_log
        }
    }

    impl Sink for FailFirstThenCount {
        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            let MediaBuffer::Packet(_) = buf else {
                return Ok(());
            };
            if !self.failed_once {
                self.failed_once = true;
                return Err(crate::error::Error::Other("simulated failure".into()));
            }
            self.count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    struct FailControl {
        pp_log: PpLog,
    }

    impl Element for FailControl {
        fn name(&self) -> Arc<str> {
            "fail-control".into()
        }

        fn element_type(&self) -> ElementType {
            ElementType::Other
        }

        fn pp_log(&self) -> &PpLog {
            &self.pp_log
        }

        fn pp_log_mut(&mut self) -> &mut PpLog {
            &mut self.pp_log
        }
    }

    impl Sink for FailControl {
        fn consume(&mut self, _buf: MediaBuffer) -> Result<()> {
            Ok(())
        }

        fn control(&mut self, msg: &ControlMsg) -> Result<()> {
            Err(crate::error::Error::Other(format!(
                "simulated {msg:?} failure"
            )))
        }
    }

    /// Regression test for the design change prompted by the `NoFreeSlot`
    /// investigation: a `Sink::consume` failure used to end the worker
    /// thread outright (and, transitively, everything upstream once its
    /// data channel closed). Now it's just one dropped buffer — the
    /// worker keeps running, later buffers still get through, and exactly
    /// one `BusEvent::Error` shows up for the one that failed.
    #[test]
    fn a_failing_consume_drops_that_buffer_but_keeps_the_worker_alive() {
        let count = Arc::new(AtomicUsize::new(0));
        let sink = FailFirstThenCount {
            count: count.clone(),
            failed_once: false,
            pp_log: element_pp_log(ElementType::Other, "fail-first", None),
        };
        let (bus, bus_rx) = Bus::new();

        let mut queue = Queue::spawn_with_policy(
            "test",
            8,
            Box::new(sink),
            bus,
            OverflowPolicy::default(),
            None,
        )
        .unwrap();
        for _ in 0..3 {
            queue.consume(packet()).unwrap();
        }
        queue.consume(MediaBuffer::Eos).unwrap();
        drop(queue); // blocks until the worker drains everything and joins

        // First packet failed (and was dropped); the other two still went
        // through — the worker didn't die over the first one.
        assert_eq!(count.load(Ordering::SeqCst), 2);
        let errors = bus_rx
            .iter()
            .filter(|e| matches!(e, BusEvent::Error { .. }))
            .count();
        assert_eq!(
            errors, 1,
            "expected exactly one Error event, for the one buffer that failed"
        );
    }

    /// A downstream failure is reported as the downstream element's, not as
    /// the `Queue`'s.
    ///
    /// A queue does not fail here — it is only where the failure could no
    /// longer be returned to a caller. Reporting it as `ElementType::Queue`
    /// under the queue's own name, which is what this did until now, said
    /// that a queue had broken and threw away the one thing an observer
    /// needs: which element stopped working. Two outputs whose queues were
    /// named alike became indistinguishable at the exact moment one of them
    /// failed.
    #[test]
    fn a_downstream_failure_is_reported_under_the_downstream_element() {
        let sink = FailFirstThenCount {
            count: Arc::new(AtomicUsize::new(0)),
            failed_once: false,
            pp_log: element_pp_log(ElementType::Other, "fail-first", None),
        };
        let (bus, bus_rx) = Bus::new();

        let mut queue = Queue::spawn_with_policy(
            "the-queue",
            8,
            Box::new(sink),
            bus,
            OverflowPolicy::default(),
            None,
        )
        .unwrap();
        queue.consume(packet()).unwrap();
        queue.consume(MediaBuffer::Eos).unwrap();
        drop(queue);

        let reported: Vec<_> = bus_rx
            .iter()
            .filter_map(|event| match event {
                BusEvent::Error {
                    element_type, name, ..
                } => Some((element_type, name)),
                _ => None,
            })
            .collect();
        assert_eq!(reported.len(), 1, "expected one Error, got {reported:?}");
        let (element_type, name) = &reported[0];
        assert_eq!(
            *element_type,
            ElementType::Other,
            "reported as the queue rather than as what it feeds"
        );
        assert_eq!(&**name, "fail-first", "reported the queue's own name");
    }

    /// Control failures are asynchronous worker failures just like
    /// `consume` failures: they must be visible on the Bus, but must not
    /// prevent Pause/Resume/Stop acknowledgements or strand the worker.
    #[test]
    fn failing_control_is_reported_without_blocking_the_control_cascade() {
        let sink = FailControl {
            pp_log: element_pp_log(ElementType::Other, "fail-control", None),
        };
        let (bus, bus_rx) = Bus::new();
        let mut queue = Queue::spawn_with_policy(
            "test",
            1,
            Box::new(sink),
            bus,
            OverflowPolicy::default(),
            None,
        )
        .unwrap();

        queue.control(&ControlMsg::Pause).unwrap();
        queue.control(&ControlMsg::Resume).unwrap();
        queue.control(&ControlMsg::Stop).unwrap();
        drop(queue);

        let errors: Vec<_> = bus_rx
            .iter()
            .filter(|event| matches!(event, BusEvent::Error { .. }))
            .collect();
        assert_eq!(
            errors.len(),
            3,
            "Pause, Resume, and Stop failures must each be reported once"
        );
    }
    /// A downstream that refuses `Eos` and accepts everything else.
    ///
    /// What a decoder or an encoder does when its own drain fails — those
    /// propagate such a failure now rather than swallowing it, so `Eos`
    /// reaching one through a `Queue` can arrive as an error.
    struct RefuseEos {
        pp_log: PpLog,
        count: Arc<AtomicUsize>,
    }

    impl Element for RefuseEos {
        fn name(&self) -> Arc<str> {
            "refuse-eos".into()
        }

        fn element_type(&self) -> ElementType {
            ElementType::Other
        }

        fn pp_log(&self) -> &PpLog {
            &self.pp_log
        }

        fn pp_log_mut(&mut self) -> &mut PpLog {
            &mut self.pp_log
        }
    }

    impl Sink for RefuseEos {
        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            if buf.is_eos() {
                return Err(crate::error::Error::Other("simulated drain failure".into()));
            }
            self.count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    /// An `Eos` the downstream refuses is reported like any other failure —
    /// and the stream is then never declared finished.
    ///
    /// This pins a consequence rather than a mechanism, because the
    /// consequence is what a caller meets. `Eos` is definitionally the last
    /// buffer of its stream, so a worker that treats its failure the way it
    /// treats a mid-stream one has nothing left to receive until a seek
    /// starts another: it posts the error, goes back to waiting — as it
    /// does after an accepted `Eos` too — and never posts [`BusEvent::Eos`].
    ///
    /// A caller watching the bus for completion therefore waits until
    /// something calls `Pipeline::stop`, which is this crate's stated
    /// division of labour — the bus watcher decides what is fatal. It is
    /// worth knowing rather than discovering: draining the bus is not on
    /// its own a guarantee that a pipeline has finished, wherever a decode
    /// or encode drain can fail.
    #[test]
    fn a_refused_eos_is_reported_and_the_stream_is_never_declared_finished() {
        let count = Arc::new(AtomicUsize::new(0));
        let sink = RefuseEos {
            count: count.clone(),
            pp_log: element_pp_log(ElementType::Other, "refuse-eos", None),
        };
        let (bus, bus_rx) = Bus::new();

        let mut queue = Queue::spawn_with_policy(
            "test",
            8,
            Box::new(sink),
            bus,
            OverflowPolicy::default(),
            None,
        )
        .unwrap();
        for _ in 0..3 {
            queue.consume(packet()).unwrap();
        }
        queue.consume(MediaBuffer::Eos).unwrap();
        // Dropping is what ends the worker here: it sets the stop flag and
        // joins. Without it the thread would still be waiting.
        drop(queue);

        assert_eq!(
            count.load(Ordering::SeqCst),
            3,
            "everything before the Eos went through"
        );

        let events: Vec<_> = bus_rx.iter().collect();
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, BusEvent::Error { .. }))
                .count(),
            1,
            "the refused Eos is reported once, like any other failure"
        );
        assert!(
            !events.iter().any(|e| matches!(e, BusEvent::Eos { .. })),
            "and the stream is never declared finished, because it was not"
        );
    }

    /// Records what reached it, in order: each buffer's kind and each
    /// control message's name.
    struct Recorder {
        pp_log: PpLog,
        seen: Arc<std::sync::Mutex<Vec<&'static str>>>,
    }

    impl Element for Recorder {
        fn name(&self) -> Arc<str> {
            "recorder".into()
        }

        fn element_type(&self) -> ElementType {
            ElementType::Other
        }

        fn pp_log(&self) -> &PpLog {
            &self.pp_log
        }

        fn pp_log_mut(&mut self) -> &mut PpLog {
            &mut self.pp_log
        }
    }

    impl Sink for Recorder {
        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            let label = if buf.is_eos() { "eos" } else { "packet" };
            self.seen.lock().unwrap().push(label);
            Ok(())
        }

        fn control(&mut self, msg: &ControlMsg) -> Result<()> {
            let label = match msg {
                ControlMsg::Flush => "flush",
                ControlMsg::Seek(_) => "seek",
                ControlMsg::Stop => "stop",
                _ => "other-control",
            };
            self.seen.lock().unwrap().push(label);
            Ok(())
        }
    }

    /// Waits for the queue's next `BusEvent::Eos`, failing rather than
    /// hanging if it never comes.
    fn expect_eos(bus_rx: &crate::bus::BusReceiver) {
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while std::time::Instant::now() < deadline {
            match bus_rx.try_recv() {
                Some(BusEvent::Eos { .. }) => return,
                Some(_) => {}
                None => thread::sleep(Duration::from_millis(1)),
            }
        }
        panic!("the queue never declared the stream finished");
    }

    /// `Eos` ends a stream, not the queue: what a seek sends after the end
    /// of a file — `Flush`, `Seek`, and then a new stream with its own
    /// `Eos` — still reaches downstream, and `Stop` still reaches it after
    /// that.
    ///
    /// The worker used to return as soon as it had forwarded `Eos`. A
    /// pipeline whose source parks at its end waiting for a seek could then
    /// never be sought once it got there if a branch had a `Queue` in it:
    /// the new stream stopped at the queue, the terminal behind it never
    /// saw a sample, and the seek's preroll timed out.
    #[test]
    fn a_new_stream_after_eos_and_a_flush_still_reaches_downstream() {
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = Recorder {
            pp_log: element_pp_log(ElementType::Other, "recorder", None),
            seen: Arc::clone(&seen),
        };
        let (bus, bus_rx) = Bus::new();
        let mut queue = Queue::spawn("test", 8, Box::new(sink), bus, None).unwrap();

        queue.consume(packet()).unwrap();
        queue.consume(MediaBuffer::Eos).unwrap();
        expect_eos(&bus_rx);

        queue.control(&ControlMsg::Flush).unwrap();
        queue.control(&ControlMsg::Seek(Duration::ZERO)).unwrap();
        queue.consume(packet()).unwrap();
        queue.consume(MediaBuffer::Eos).unwrap();
        expect_eos(&bus_rx);

        queue.control(&ControlMsg::Stop).unwrap();
        assert_eq!(
            *seen.lock().unwrap(),
            ["packet", "eos", "flush", "seek", "packet", "eos", "stop"]
        );
        drop(queue);
    }

    /// A worker left waiting after `Eos` is still ended promptly by dropping
    /// its `Queue` — the path a pipeline's teardown takes once a source has
    /// played to its end and returned.
    #[test]
    fn dropping_after_eos_joins_the_worker_promptly() {
        let dropped = Arc::new(AtomicBool::new(false));
        let sink = DropAwareSink {
            dropped: Arc::clone(&dropped),
            pp_log: element_pp_log(ElementType::Other, "drop-aware", None),
        };
        let (bus, bus_rx) = Bus::new();
        let mut queue = Queue::spawn("test", 8, Box::new(sink), bus, None).unwrap();
        queue.consume(packet()).unwrap();
        queue.consume(MediaBuffer::Eos).unwrap();
        expect_eos(&bus_rx);

        let started = std::time::Instant::now();
        drop(queue);
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "joining an idle worker took {:?}",
            started.elapsed()
        );
        assert!(
            dropped.load(Ordering::SeqCst),
            "the join released everything downstream"
        );
    }

    /// Records the pts of every packet it is handed.
    struct PtsRecorder {
        pp_log: PpLog,
        seen: Arc<Mutex<Vec<i64>>>,
    }

    impl Element for PtsRecorder {
        fn name(&self) -> Arc<str> {
            "pts-recorder".into()
        }

        fn element_type(&self) -> ElementType {
            ElementType::Other
        }

        fn pp_log(&self) -> &PpLog {
            &self.pp_log
        }

        fn pp_log_mut(&mut self) -> &mut PpLog {
            &mut self.pp_log
        }
    }

    impl Sink for PtsRecorder {
        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            if let MediaBuffer::Packet(packet) = buf {
                self.seen.lock().unwrap().push(packet.pts().unwrap_or(-1));
            }
            Ok(())
        }
    }

    fn packet_at(pts: i64) -> MediaBuffer {
        let mut packet = ffmpeg_next::Packet::empty();
        packet.set_pts(Some(pts));
        MediaBuffer::Packet(Arc::new(packet))
    }

    fn recording_queue(
        capacity: usize,
        state: &Arc<PlaybackState>,
    ) -> (Queue, Arc<Mutex<Vec<i64>>>) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (bus, _bus_rx) = Bus::new();
        let queue = Queue::spawn_with_policy_using(
            "queue",
            capacity,
            Box::new(PtsRecorder {
                pp_log: element_pp_log(ElementType::Other, "pts-recorder", None),
                seen: Arc::clone(&seen),
            }),
            bus,
            OverflowPolicy::default(),
            None,
            None,
            Some(Arc::clone(state)),
            |thread_name, task| thread::Builder::new().name(thread_name).spawn(task),
        )
        .expect("spawn");
        (queue, seen)
    }

    /// While an interrupt is out the worker takes nothing, and a full queue
    /// does not block the thread handing buffers over — they go in past its
    /// capacity instead, so that thread can go and take the request behind
    /// the interrupt. Once it is settled, everything arrives in the order it
    /// was handed over.
    #[test]
    fn an_interrupt_holds_the_worker_back_and_lets_the_producer_go() {
        let state = PlaybackState::new();
        let (mut queue, seen) = recording_queue(1, &state);
        state.interrupt();

        let started = std::time::Instant::now();
        for pts in 0..4 {
            queue.consume(packet_at(pts)).expect("handed over");
        }
        queue
            .consume(MediaBuffer::Eos)
            .expect("even Eos goes in past capacity");
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "a full channel must not block while an interrupt is out"
        );
        assert!(
            !queue.ready_consume(),
            "what went in past capacity makes it full"
        );
        thread::sleep(Duration::from_millis(50));
        assert!(seen.lock().unwrap().is_empty(), "the worker held back");

        state.settle();
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while seen.lock().unwrap().len() < 4 && std::time::Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(*seen.lock().unwrap(), [0, 1, 2, 3]);

        // Sent after the ones past capacity have gone: nothing overtakes them.
        queue.consume(packet_at(4)).expect("send");
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while seen.lock().unwrap().len() < 5 && std::time::Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(*seen.lock().unwrap(), [0, 1, 2, 3, 4]);
    }

    /// What went in past capacity belongs to the old timeline as much as the
    /// rest of the backlog: a `Flush` drops it all.
    #[test]
    fn flush_discards_what_went_in_past_capacity() {
        let state = PlaybackState::new();
        let (mut queue, seen) = recording_queue(1, &state);
        state.interrupt();
        for pts in 0..3 {
            queue.consume(packet_at(pts)).expect("handed over");
        }
        queue.control(&ControlMsg::Flush).expect("flush");
        state.settle();
        queue.consume(packet_at(10)).expect("send");
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while seen.lock().unwrap().is_empty() && std::time::Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        thread::sleep(Duration::from_millis(50));
        assert_eq!(*seen.lock().unwrap(), [10]);
    }
    fn gated(open: &Arc<AtomicBool>, count: &Arc<AtomicUsize>) -> Box<dyn Sink> {
        Box::new(Gated {
            pp_log: element_pp_log(ElementType::Other, "gated", None),
            open: Arc::clone(open),
            count: Arc::clone(count),
            ended: Arc::new(AtomicBool::new(false)),
        })
    }

    /// A thread already waiting for room lets go the moment an interrupt is
    /// raised: the state rings it, rather than it noticing on its next look
    /// round. Nothing else ever wakes it here — downstream takes nothing —
    /// so without the ring this waits for good.
    #[test]
    fn an_interrupt_lets_go_of_a_thread_already_waiting_for_room() {
        let state = PlaybackState::new();
        let (bus, _bus_rx) = Bus::new();
        let open = Arc::new(AtomicBool::new(false));
        let count = Arc::new(AtomicUsize::new(0));
        let mut queue = Queue::spawn_with_policy_using(
            "queue",
            1,
            gated(&open, &count),
            bus,
            OverflowPolicy::default(),
            None,
            None,
            Some(Arc::clone(&state)),
            |thread_name, task| thread::Builder::new().name(thread_name).spawn(task),
        )
        .expect("spawn");
        queue.consume(packet()).expect("room for one");

        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let producer = thread::spawn(move || {
            let handed_over = queue.consume(packet()).is_ok();
            done_tx.send(handed_over).unwrap();
            queue
        });
        assert!(
            done_rx.recv_timeout(Duration::from_millis(100)).is_err(),
            "full, so it waits"
        );
        state.interrupt();
        assert_eq!(
            done_rx.recv_timeout(Duration::from_secs(5)),
            Ok(true),
            "the interrupt lets it go, the buffer handed over past capacity"
        );
        let queue = producer.join().unwrap();
        state.settle();
        drop(queue);
    }

    /// With no capacity at all, handing a buffer over waits until the worker
    /// has taken it: nothing is left waiting ahead of the worker.
    #[test]
    fn a_queue_without_capacity_hands_over_only_what_is_taken() {
        let (bus, _bus_rx) = Bus::new();
        let open = Arc::new(AtomicBool::new(false));
        let count = Arc::new(AtomicUsize::new(0));
        let queue = Queue::spawn("queue", 0, gated(&open, &count), bus, None).expect("spawn");

        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let producer = thread::spawn(move || {
            let mut queue = queue;
            queue.consume(packet()).expect("handed over");
            done_tx.send(()).unwrap();
            queue
        });
        assert!(
            done_rx.recv_timeout(Duration::from_millis(100)).is_err(),
            "not taken yet, so not handed over"
        );
        open.store(true, Ordering::SeqCst);
        done_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("taken, and so handed over");
        let queue = producer.join().unwrap();
        drop(queue);
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }
}
