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
    collections::VecDeque,
    sync::{
        Arc, Mutex, MutexGuard,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use crate::pp_log::{PpLog, pp_debug, pp_info, pp_trace};
use crossbeam_channel::{
    Receiver, RecvTimeoutError, SendTimeoutError, Sender, TrySendError, bounded, select,
};
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    clock::Clock,
    contract::InputContract,
    control::{self, ControlMsg, ControlReceiver, ControlSender, Phase, RequestKind},
    element::{Context, Element, ElementType, Sink, element_pp_log},
    error::{Result, ThreadSpawnError},
    stats::ElementCounters,
    timeline::{Numbered, Timeline, UNNUMBERED},
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

/// How often the worker's blocking wait wakes up on its own (nothing
/// ready on either channel) to check [`Queue`]'s `stop` flag — see
/// [`worker_loop`] and [`apply_control`]'s pause loop. Only ever adds
/// latency to the already-abnormal "torn down without ever being told to
/// stop" path (see [`Queue::drop`]); real data/control traffic is always
/// picked up immediately; this pause is only ever *waited out*, not
/// polled on a timer.
const STOP_POLL_INTERVAL: Duration = Duration::from_millis(20);

/// How often a wait that an interrupt may cut short looks again: the upstream
/// side blocked on a full channel, and the worker holding back while an
/// interrupt is out — see [`Queue`]'s docs on pausing. Short, because
/// a pause waits on it; not zero, because the upstream side spends it
/// blocked in ordinary backpressure too.
const INTERRUPT_POLL_INTERVAL: Duration = Duration::from_millis(5);

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
/// A queue built by a pipeline also holds back for the pipeline's clock
/// interrupts. `Pipeline::pause`, `stop`, `finish` and a seek interrupt a
/// paced wait before their request has reached every queue, and an
/// interrupted `Pacer` takes what it is handed without waiting; a worker
/// that went on feeding it in that time moved its whole backlog into the
/// pacer. So from the interrupt until the pipeline has every source's
/// acknowledgement the worker takes nothing but control, and the thread
/// handing buffers over is not left blocked on a channel that will not
/// drain: a buffer that finds it full is held over and taken after the
/// channel's own, in order.
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
/// `Eos` returns within one short internal poll interval.
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
    /// The pipeline's clock, whose interrupts this queue holds back for;
    /// `None` for one spawned by hand, which never does.
    clock: Option<Arc<Clock>>,
    /// Buffers handed over while the channel was full and an interrupt
    /// was out — see [`Overflow`].
    overflow: Overflow,
    /// How many `Eos` this queue has taken and not yet handed on or
    /// discarded — see [`Inbox::owes_an_end`].
    ends_owed: Arc<AtomicUsize>,
    /// The pipeline's timeline, whose number a buffer is handed over with —
    /// see [`crate::timeline`]. `None` for one spawned by hand.
    timeline: Option<Arc<Timeline>>,
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
            None,
            |thread_name, task| thread::Builder::new().name(thread_name).spawn(task),
        )
    }

    /// [`Self::spawn_with_policy`] as a pipeline's chain builds it: counted,
    /// so the queue is reported with the rest of the graph, and on the
    /// pipeline's clock, so it holds back for its interrupts.
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
            Some(Arc::clone(&context.clock)),
            Some(Arc::clone(&context.timeline)),
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
        clock: Option<Arc<Clock>>,
        timeline: Option<Arc<Timeline>>,
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
        let (tx, rx) = bounded::<Numbered>(capacity);
        let (control_tx, control_rx) = control::channel();
        let worker_name = name.clone();
        let worker_bus = bus.clone();
        let worker_pp_log = pp_log.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = stop.clone();
        if let Some(queue) = counters.as_deref().and_then(ElementCounters::queue) {
            queue.watch(tx.clone());
        }
        let worker_counters = counters.clone();
        let overflow = Overflow::default();
        let ends_owed = Arc::new(AtomicUsize::new(0));
        let worker_inbox = Inbox {
            data: rx,
            overflow: overflow.clone(),
            clock: clock.clone(),
            timeline: timeline.clone(),
            ends_owed: ends_owed.clone(),
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
                if let Some(timeline) = &worker_inbox.timeline {
                    crate::timeline::enter(timeline);
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
            clock,
            overflow,
            ends_owed,
            timeline,
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
            OverflowPolicy::Block(_) => !self.tx.is_full() && self.overflow.lock().is_empty(),
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
                .hand_over(Numbered::on(self.timeline.as_ref(), buf), Duration::MAX)
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
        let buf = Numbered::on(self.timeline.as_ref(), buf);
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
            // Anything held over counts as a full channel: sent past it, this
            // buffer would overtake it.
            OverflowPolicy::DropNewest => {
                let sent = if self.overflow.lock().is_empty() {
                    self.tx.try_send(buf)
                } else {
                    Err(TrySendError::Full(buf))
                };
                match sent {
                    Ok(()) => Ok(()),
                    Err(TrySendError::Full(_)) => {
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
                    Err(TrySendError::Disconnected(_)) => Err(QueueError::ChannelClosed.into()),
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
        // What was held over is part of the backlog these discard, and this
        // thread is the only one that adds to it.
        if matches!(msg, ControlMsg::Flush | ControlMsg::Stop) {
            let mut overflow = self.overflow.lock();
            let ends = overflow.iter().filter(|held| held.buf.is_eos()).count();
            overflow.clear();
            self.ends_owed.fetch_sub(ends, Ordering::AcqRel);
        }
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
            // Wakes the worker if nothing else already would — it checks
            // this on every idle wait-timeout, in both `worker_loop` and
            // `apply_control`'s pause loop, so it's the one signal that
            // reaches a genuinely-idle worker no matter which of those two
            // places it's currently blocked in (e.g. a `.queue()`-having
            // `Pipeline` dropped without ever being `run()`, or a bare
            // `Queue` paused and then dropped without `Resume`/`Stop` —
            // `handle.join()` below would otherwise hang on either).
            // Doesn't race real pending data/control the way closing a
            // channel to force this would: it's only ever consulted once
            // `select!`/`recv_timeout` has already waited out a full
            // `STOP_POLL_INTERVAL` with *nothing* ready on either channel,
            // so any already-queued `Stop`/`Eos`/data is always drained
            // first, same as `block_never_drops` and friends rely on.
            self.stop.store(true, Ordering::Relaxed);
            pp_info!(pp_log: &self.pp_log, "dropped: joining worker");
            let _ = handle.join();
        }
    }
}

/// What the upstream side could not put in the channel because it was full
/// when an interrupt came — handed over all the same, so that thread can
/// get back to the request behind the interrupt, and taken by the worker
/// once the channel has emptied, after everything sent before it.
///
/// Only the upstream side adds to it, and only while it is empty sends to the
/// channel instead; the worker takes from it only once the channel is empty.
/// Together that keeps every buffer in the order it was handed over. It
/// holds what one source makes between two looks at its control — a buffer
/// or two — and is emptied by `Flush` and `Stop` like the channel.
#[derive(Clone, Default)]
struct Overflow(Arc<Mutex<VecDeque<Numbered>>>);

impl Overflow {
    fn lock(&self) -> MutexGuard<'_, VecDeque<Numbered>> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// Why [`Queue::hand_over`] could not hand a buffer over.
enum HandOverError {
    TimedOut,
    Closed,
}

impl Queue {
    /// Puts `buf` in the channel, waiting up to `timeout` for room — unless
    /// an interrupt comes while it waits, when `buf` goes to the
    /// [`Overflow`] instead and this returns at once.
    ///
    /// Why it may not simply wait: while an interrupt is out the worker takes
    /// nothing (see `worker_loop`), and what ends the interrupt is the request
    /// behind it, which the thread calling this is the one to pass on. Before
    /// the worker held back, it did make room here — by feeding every
    /// buffer in the channel to an interrupted `Pacer`, which took each
    /// without waiting and held them all, so a pause left a whole queue's worth
    /// of frames out of their pool.
    fn hand_over(
        &self,
        mut buf: Numbered,
        timeout: Duration,
    ) -> std::result::Result<(), HandOverError> {
        let Some(clock) = self.clock.clone() else {
            return self
                .tx
                .send_timeout(buf, timeout)
                .map_err(|error| match error {
                    SendTimeoutError::Timeout(_) => HandOverError::TimedOut,
                    SendTimeoutError::Disconnected(_) => HandOverError::Closed,
                });
        };
        let deadline = Instant::now().checked_add(timeout);
        loop {
            let mut overflow = self.overflow.lock();
            if overflow.is_empty() {
                match self.tx.try_send(buf) {
                    Ok(()) => return Ok(()),
                    Err(TrySendError::Disconnected(_)) => return Err(HandOverError::Closed),
                    Err(TrySendError::Full(back)) => buf = back,
                }
            }
            if clock.interrupt_pending() {
                overflow.push_back(buf);
                return Ok(());
            }
            let slice = match deadline {
                Some(deadline) => deadline.saturating_duration_since(Instant::now()),
                None => INTERRUPT_POLL_INTERVAL,
            }
            .min(INTERRUPT_POLL_INTERVAL);
            if slice.is_zero() {
                return Err(HandOverError::TimedOut);
            }
            if overflow.is_empty() {
                drop(overflow);
                match self.tx.send_timeout(buf, slice) {
                    Ok(()) => return Ok(()),
                    Err(SendTimeoutError::Timeout(back)) => buf = back,
                    Err(SendTimeoutError::Disconnected(_)) => return Err(HandOverError::Closed),
                }
            } else {
                // What was held over goes first; the worker is taking it.
                drop(overflow);
                if self.handle.as_ref().is_none_or(JoinHandle::is_finished) {
                    return Err(HandOverError::Closed);
                }
                thread::sleep(slice);
            }
        }
    }
}

/// Where a worker takes its buffers from: the channel, then what was held
/// over — see [`Overflow`] — and neither while an interrupt is out.
struct Inbox {
    data: Receiver<Numbered>,
    overflow: Overflow,
    clock: Option<Arc<Clock>>,
    /// The pipeline's timeline, against which what is carried is judged
    /// before it goes on — `None` for a queue spawned by hand, which
    /// drops nothing for being old.
    timeline: Option<Arc<Timeline>>,
    /// Shared with the [`Queue`], which counts an `Eos` in as it takes one;
    /// the worker counts it out as it hands it on or a `Flush` discards it.
    ends_owed: Arc<AtomicUsize>,
}

impl Inbox {
    /// Whether the worker should take nothing for now: an interrupt is out,
    /// so whatever it fed downstream would only be held by an interrupted
    /// `Pacer` rather than played. Not once `stop` is set — the queue has
    /// been dropped, nothing will answer the interrupt here, and the worker
    /// drains what it holds as it always has.
    fn held_back(&self, stop: &AtomicBool) -> bool {
        self.clock.as_deref().is_some_and(Clock::interrupt_pending) && !stop.load(Ordering::Relaxed)
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
    fn owes_an_end(&self) -> bool {
        self.ends_owed.load(Ordering::Acquire) > 0
    }

    /// Counts out `ends` `Eos` that will not be handed on from here.
    fn discarded(&self, ends: usize) {
        self.ends_owed.fetch_sub(ends, Ordering::AcqRel);
    }

    /// The next buffer held over, once the channel has nothing older.
    fn take_held_over(&self) -> Option<Numbered> {
        let mut overflow = self.overflow.lock();
        if self.data.is_empty() {
            overflow.pop_front()
        } else {
            None
        }
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
    loop {
        if let Some((request, ack)) = control_rx.try_recv() {
            let RequestKind::Control(msg) = request else {
                let _ = ack.send(());
                continue;
            };
            if apply_control(
                &inbox,
                &mut downstream,
                msg,
                &ack,
                &control_rx,
                &error_reporter,
                &stop,
            ) {
                pp_info!(pp_log: &pp_log, "worker: stopped");
                return;
            }
            continue;
        }

        // Neither while an interrupt is out — see `Inbox::held_back` — nor
        // while downstream is not ready does this take a buffer; it waits on
        // control alone, which is what will change either.
        let held_back = inbox.held_back(&stop);
        if held_back || !downstream.ready_consume() {
            let wait = if held_back {
                INTERRUPT_POLL_INTERVAL
            } else {
                STOP_POLL_INTERVAL
            };
            match control_rx.rx.recv_timeout(wait) {
                Ok(request) => {
                    let RequestKind::Control(msg) = request.kind else {
                        let _ = request.ack.send(());
                        continue;
                    };
                    if apply_control(
                        &inbox,
                        &mut downstream,
                        msg,
                        &request.ack,
                        &control_rx,
                        &error_reporter,
                        &stop,
                    ) {
                        pp_info!(pp_log: &pp_log, "worker: stopped");
                        return;
                    }
                }
                // Held back, it looks again rather than ending: `stop` set
                // since then means draining what is here, as ever. So does
                // a downstream that is only full for now — a `Pacer` behind
                // it letting pictures go at their time — while an `Eos` is
                // still queued: see `Inbox::owes_an_end`. Ending here
                // regardless dropped the end of the stream, which is what a
                // `finish` of a playing file lost: the source ends, its
                // queue is dropped, and the worker found its downstream busy.
                Err(RecvTimeoutError::Timeout) => {
                    if !held_back && stop.load(Ordering::Relaxed) && !inbox.owes_an_end() {
                        pp_info!(pp_log: &pp_log, "worker: stop flag set, ending");
                        return;
                    }
                }
                Err(RecvTimeoutError::Disconnected) => return,
            }
            continue;
        }

        if let Some(buf) = inbox.take_held_over() {
            forward(&mut downstream, buf, &worker);
            continue;
        }

        select! {
            recv(control_rx.rx) -> req => {
                match req {
                    Ok(req) => {
                        let RequestKind::Control(msg) = req.kind else {
                            let _ = req.ack.send(());
                            continue;
                        };
                        if apply_control(
                            &inbox,
                            &mut downstream,
                            msg,
                            &req.ack,
                            &control_rx,
                            &error_reporter,
                            &stop,
                        ) {
                            pp_info!(pp_log: &pp_log, "worker: stopped");
                            return;
                        }
                    }
                    Err(_) => {
                        pp_info!(pp_log: &pp_log, "worker: control channel gone, ending");
                        return; // sender (this Queue) dropped
                    }
                }
            }
            recv(inbox.data) -> buf => {
                match buf {
                    Ok(buf) => {
                        forward(&mut downstream, buf, &worker);
                    }
                    Err(_) => {
                        pp_info!(pp_log: &pp_log, "worker: producer (this Queue) gone, ending");
                        return;
                    }
                }
            }
            // Only reached once neither branch above had anything ready
            // for a whole `STOP_POLL_INTERVAL` — real traffic on either
            // channel always wins first. See `Queue::drop`.
            default(STOP_POLL_INTERVAL) => {
                if stop.load(Ordering::Relaxed) {
                    pp_info!(pp_log: &pp_log, "worker: stop flag set, ending");
                    return;
                }
            }
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
        .timeline
        .as_deref()
        .is_some_and(|timeline| timeline.is_behind(number))
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
        // `recv_timeout` (not `recv`) so `Queue::drop` setting `stop` can
        // still wake a worker that's paused forever with no `Resume`/
        // `Stop` ever coming (e.g. a bare `Queue`, not reached through a
        // `Pipeline` — see `Queue::drop`'s docs on why this state is
        // otherwise unreachable there). Nothing else feeds this queue
        // while paused (see the type-level docs), so there's no
        // legitimate traffic this could ever cut off.
        let (msg, ack) = match control_rx.rx.recv_timeout(STOP_POLL_INTERVAL) {
            Ok(req) => {
                let RequestKind::Control(msg) = req.kind else {
                    let _ = req.ack.send(());
                    continue;
                };
                (msg, req.ack)
            }
            Err(RecvTimeoutError::Timeout) => {
                if stop.load(Ordering::Relaxed) {
                    pp_info!(pp_log: error_reporter.pp_log, "worker: stop flag set while paused, ending");
                    return true;
                }
                continue;
            }
            Err(RecvTimeoutError::Disconnected) => {
                pp_info!(pp_log: error_reporter.pp_log, "worker: control channel gone while paused, ending");
                return true; // sender gone — treat like Stop
            }
        };
        pp_trace!(
            pp_log: error_reporter.pp_log,
            "event=control control={msg:?} phase=forwarding"
        );
        inbox.discarded(discard_stale_data(&inbox.data, &msg));
        forward_control(downstream, msg.clone(), error_reporter);
        let is_stop = msg == ControlMsg::Stop;
        let _ = ack.send(());
        if is_stop {
            return true;
        }
        if !Phase::Paused.after(&msg).holds() {
            // Data flows again — a `Resume`, a `Preroll`: see `Phase`.
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
            Arc,
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
                thread::sleep(STOP_POLL_INTERVAL * 4);
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

    fn recording_queue(capacity: usize, clock: &Arc<Clock>) -> (Queue, Arc<Mutex<Vec<i64>>>) {
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
            Some(Arc::clone(clock)),
            None,
            |thread_name, task| thread::Builder::new().name(thread_name).spawn(task),
        )
        .expect("spawn");
        (queue, seen)
    }

    /// While an interrupt is out the worker takes nothing, and a full
    /// channel does not block the thread handing buffers over — they are
    /// held over instead, so that thread can go and take the request behind
    /// the interrupt. Once it is settled, everything arrives in the order it
    /// was handed over, what was held over after what was in the channel.
    #[test]
    fn an_interrupt_holds_the_worker_back_and_lets_the_producer_go() {
        let clock = Arc::new(Clock::new());
        let (mut queue, seen) = recording_queue(1, &clock);
        clock.interrupt();

        let started = std::time::Instant::now();
        for pts in 0..4 {
            queue.consume(packet_at(pts)).expect("handed over");
        }
        queue
            .consume(MediaBuffer::Eos)
            .expect("even Eos is held over");
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "a full channel must not block while an interrupt is out"
        );
        assert!(!queue.ready_consume(), "held-over buffers make it full");
        thread::sleep(Duration::from_millis(50));
        assert!(seen.lock().unwrap().is_empty(), "the worker held back");

        clock.settle();
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while seen.lock().unwrap().len() < 4 && std::time::Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(*seen.lock().unwrap(), [0, 1, 2, 3]);

        // Sent after the held-over ones have gone: nothing overtakes them.
        queue.consume(packet_at(4)).expect("send");
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while seen.lock().unwrap().len() < 5 && std::time::Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(*seen.lock().unwrap(), [0, 1, 2, 3, 4]);
    }

    /// What was held over belongs to the old timeline as much as the
    /// channel's backlog does: a `Flush` drops both.
    #[test]
    fn flush_discards_what_was_held_over() {
        let clock = Arc::new(Clock::new());
        let (mut queue, seen) = recording_queue(1, &clock);
        clock.interrupt();
        for pts in 0..3 {
            queue.consume(packet_at(pts)).expect("handed over");
        }
        queue.control(&ControlMsg::Flush).expect("flush");
        clock.settle();
        queue.consume(packet_at(10)).expect("send");
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while seen.lock().unwrap().is_empty() && std::time::Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        thread::sleep(Duration::from_millis(50));
        assert_eq!(*seen.lock().unwrap(), [10]);
    }
}
