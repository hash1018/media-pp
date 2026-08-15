use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use crate::clog::{CLog, cinfo};
use crossbeam_channel::{
    Receiver, RecvTimeoutError, SendTimeoutError, Sender, TrySendError, bounded, select,
};
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    control::{self, ControlMsg, ControlReceiver, ControlSender},
    element::{Element, ElementType, Sink, element_clog},
    error::Result,
};

/// Errors specific to `Queue`. Converts into the crate-wide `Error` via
/// `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum QueueError {
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
    SendTimedOut { after: Duration },
}

/// How often the worker's blocking wait wakes up on its own (nothing
/// ready on either channel) to check [`Queue`]'s `stop` flag — see
/// [`worker_loop`] and [`apply_control`]'s pause loop. Only ever adds
/// latency to the already-abnormal "torn down without ever being told to
/// stop" path (see [`Queue::drop`]); real data/control traffic is always
/// picked up immediately; this pause is only ever *waited out*, not
/// polled on a timer.
const STOP_POLL_INTERVAL: Duration = Duration::from_millis(20);

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
pub struct Queue {
    clog: CLog,
    name: Arc<str>,
    tx: Sender<MediaBuffer>,
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
}

impl Queue {
    /// Spawns with [`OverflowPolicy::default`]. Use
    /// [`Queue::spawn_with_policy`] to drop instead of blocking when full.
    pub fn spawn(
        name: impl Into<String>,
        capacity: usize,
        downstream: Box<dyn Sink>,
        bus: Bus,
        pipeline_id: Option<&str>,
    ) -> Queue {
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
    /// call) becomes this `Queue`'s `clog` sub_id; `None` if it wasn't
    /// built through a `Pipeline` at all (e.g. the tests below).
    pub fn spawn_with_policy(
        name: impl Into<String>,
        capacity: usize,
        downstream: Box<dyn Sink>,
        bus: Bus,
        policy: OverflowPolicy,
        pipeline_id: Option<&str>,
    ) -> Queue {
        // Stored as `Arc<str>` (not `String`) so the `worker_name.clone()`
        // below, and every subsequent `BusEvent` this posts, are a
        // refcount bump instead of a fresh allocation — `Dropped` in
        // particular can fire once per buffer under sustained overflow.
        let name: Arc<str> = name.into().into();
        let clog = element_clog(ElementType::Queue, &name, pipeline_id);
        cinfo!(clog: &clog, "spawned: capacity={capacity}, policy={policy:?}");
        let (tx, rx) = bounded::<MediaBuffer>(capacity);
        let (control_tx, control_rx) = control::channel();
        let worker_name = name.clone();
        let worker_bus = bus.clone();
        let worker_clog = clog.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = stop.clone();

        let handle = thread::Builder::new()
            .name(format!("queue:{worker_name}"))
            .spawn(move || {
                worker_loop(
                    rx,
                    control_rx,
                    downstream,
                    worker_bus,
                    worker_name,
                    worker_clog,
                    worker_stop,
                )
            })
            .expect("failed to spawn queue worker thread");

        Queue {
            name,
            clog,
            tx,
            policy,
            bus,
            handle: Some(handle),
            control: control_tx,
            stop,
        }
    }
}

impl Element for Queue {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::Queue
    }

    fn clog(&self) -> &CLog {
        &self.clog
    }

    fn clog_mut(&mut self) -> &mut CLog {
        &mut self.clog
    }
}

impl Sink for Queue {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        // EOS must never be dropped, regardless of policy: unlike an
        // explicit Stop or Queue::drop's private stop flag, this is the
        // natural-completion signal that tells the worker to finish only
        // after everything queued before it has reached downstream. The
        // policy timeout intentionally does not apply to this send.
        if buf.is_eos() {
            return self
                .tx
                .send(buf)
                .map_err(|_| QueueError::ChannelClosed.into());
        }

        match self.policy {
            OverflowPolicy::Block(timeout) => match self.tx.send_timeout(buf, timeout) {
                Ok(()) => Ok(()),
                Err(SendTimeoutError::Timeout(_)) => {
                    Err(QueueError::SendTimedOut { after: timeout }.into())
                }
                Err(SendTimeoutError::Disconnected(_)) => Err(QueueError::ChannelClosed.into()),
            },
            OverflowPolicy::DropNewest => match self.tx.try_send(buf) {
                Ok(()) => Ok(()),
                Err(TrySendError::Full(_)) => {
                    self.bus.post(
                        &self.clog,
                        BusEvent::Dropped {
                            element_type: ElementType::Queue,
                            name: self.name.clone(),
                        },
                    );
                    Ok(())
                }
                Err(TrySendError::Disconnected(_)) => Err(QueueError::ChannelClosed.into()),
            },
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        // Blocks until the worker — and everything downstream of it — has
        // finished handling this. Never stuck behind a data backlog: the
        // worker checks this channel before every data buffer it pulls
        // (see `worker_loop`), and while paused it's blocked *only* on
        // this channel, so a `consume()` blocked sending data upstream of
        // a paused queue just sits in ordinary backpressure — nothing
        // feeds this queue while it's paused, since `Pause` blocks
        // whatever's upstream the same way, all the way back to the
        // source (see [`crate::control::drain_control`]).
        self.control.send(msg);
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
            cinfo!(clog: &self.clog, "dropped: joining worker");
            let _ = handle.join();
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
fn worker_loop(
    data_rx: Receiver<MediaBuffer>,
    control_rx: ControlReceiver,
    mut downstream: Box<dyn Sink>,
    bus: Bus,
    name: Arc<str>,
    // Cloned from `Queue`'s own field before this thread was spawned —
    // same value, not rebuilt here, so a `pipeline_id` passed to
    // `spawn_with_policy` actually reaches this thread's own log lines
    // too.
    clog: CLog,
    stop: Arc<AtomicBool>,
) {
    cinfo!(clog: &clog, "worker: starting");
    let error_reporter = QueueErrorReporter {
        bus: &bus,
        name: &name,
        clog: &clog,
    };
    loop {
        if let Some((msg, ack)) = control_rx.try_recv() {
            if apply_control(
                &data_rx,
                &mut downstream,
                msg,
                &ack,
                &control_rx,
                &error_reporter,
                &stop,
            ) {
                cinfo!(clog: &clog, "worker: stopped");
                return;
            }
            continue;
        }

        select! {
            recv(control_rx.rx) -> req => {
                match req {
                    Ok(req) => {
                        if apply_control(
                            &data_rx,
                            &mut downstream,
                            req.msg,
                            &req.ack,
                            &control_rx,
                            &error_reporter,
                            &stop,
                        ) {
                            cinfo!(clog: &clog, "worker: stopped");
                            return;
                        }
                    }
                    Err(_) => {
                        cinfo!(clog: &clog, "worker: control channel gone, ending");
                        return; // sender (this Queue) dropped
                    }
                }
            }
            recv(data_rx) -> buf => {
                match buf {
                    Ok(buf) => {
                        let is_eos = buf.is_eos();
                        match downstream.consume(buf) {
                            Ok(()) => {
                                if is_eos {
                                    bus.post(
                                        &clog,
                                        BusEvent::Eos {
                                            element_type: ElementType::Queue,
                                            name: name.clone(),
                                        },
                                    );
                                    return;
                                }
                            }
                            Err(error) => {
                                // Report and move on to the next buffer —
                                // this one's dropped, but nothing else
                                // dies over it. Whoever's watching the bus
                                // decides whether the error is fatal
                                // enough to call `Pipeline::stop`.
                                error_reporter.post(error);
                            }
                        }
                    }
                    Err(_) => {
                        cinfo!(clog: &clog, "worker: producer (this Queue) gone, ending");
                        return;
                    }
                }
            }
            // Only reached once neither branch above had anything ready
            // for a whole `STOP_POLL_INTERVAL` — real traffic on either
            // channel always wins first. See `Queue::drop`.
            default(STOP_POLL_INTERVAL) => {
                if stop.load(Ordering::Relaxed) {
                    cinfo!(clog: &clog, "worker: stop flag set, ending");
                    return;
                }
            }
        }
    }
}

/// Applies one control message to `downstream`, acking it, then — only
/// for `Pause` — blocking this thread on `control_rx` alone (never
/// touching `data_rx`) until `Resume`/`Stop`. Returns `true` once `Stop`
/// has been handled, meaning the caller (`worker_loop`) should exit.
fn apply_control(
    data_rx: &Receiver<MediaBuffer>,
    downstream: &mut Box<dyn Sink>,
    msg: ControlMsg,
    ack: &Sender<()>,
    control_rx: &ControlReceiver,
    error_reporter: &QueueErrorReporter<'_>,
    stop: &AtomicBool,
) -> bool {
    cinfo!(clog: error_reporter.clog, "control: {msg:?}");
    discard_stale_data(data_rx, msg);
    forward_control(downstream, msg, error_reporter);
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
            Ok(req) => (req.msg, req.ack),
            Err(RecvTimeoutError::Timeout) => {
                if stop.load(Ordering::Relaxed) {
                    cinfo!(clog: error_reporter.clog, "worker: stop flag set while paused, ending");
                    return true;
                }
                continue;
            }
            Err(RecvTimeoutError::Disconnected) => {
                cinfo!(clog: error_reporter.clog, "worker: control channel gone while paused, ending");
                return true; // sender gone — treat like Stop
            }
        };
        cinfo!(clog: error_reporter.clog, "control: {msg:?}");
        discard_stale_data(data_rx, msg);
        forward_control(downstream, msg, error_reporter);
        let is_stop = msg == ControlMsg::Stop;
        let _ = ack.send(());
        if is_stop {
            return true;
        }
        if msg == ControlMsg::Resume {
            return false;
        }
        // Another Pause while already paused: already forwarded above, keep waiting.
    }
}

struct QueueErrorReporter<'a> {
    bus: &'a Bus,
    name: &'a Arc<str>,
    clog: &'a CLog,
}

impl QueueErrorReporter<'_> {
    fn post(&self, error: crate::error::Error) {
        self.bus.post(
            self.clog,
            BusEvent::Error {
                element_type: ElementType::Queue,
                name: self.name.clone(),
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
    if let Err(error) = downstream.control(msg) {
        error_reporter.post(error);
    }
}

/// Drops everything already buffered in `data_rx` without processing it —
/// only for `Seek`. That data predates the seek point (this Queue's
/// worker hasn't gotten to it yet, but it was read/produced before the
/// jump), so delivering it downstream afterward would show stale
/// frames instead of skipping straight to the new position.
/// `Pause`/`Resume`/`Stop` leave `data_rx` alone — see the type-level
/// docs on why that's safe (nothing feeds a paused/stopped queue in the
/// first place).
fn discard_stale_data(data_rx: &Receiver<MediaBuffer>, msg: ControlMsg) {
    if matches!(msg, ControlMsg::Seek(_)) {
        while data_rx.try_recv().is_ok() {}
    }
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
    use crate::bus::Bus;

    /// A downstream that's slower than the producer, so a small queue
    /// behind it actually fills up during the test.
    struct SlowCounter {
        clog: CLog,
        count: Arc<AtomicUsize>,
    }

    impl Element for SlowCounter {
        fn name(&self) -> Arc<str> {
            "slow-counter".into()
        }

        fn element_type(&self) -> ElementType {
            ElementType::Other
        }

        fn clog(&self) -> &CLog {
            &self.clog
        }

        fn clog_mut(&mut self) -> &mut CLog {
            &mut self.clog
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

        fn control(&mut self, _msg: ControlMsg) -> Result<()> {
            Ok(())
        }
    }

    fn packet() -> MediaBuffer {
        MediaBuffer::Packet(Arc::new(ffmpeg_next::Packet::empty()))
    }

    #[test]
    fn block_never_drops() {
        let count = Arc::new(AtomicUsize::new(0));
        let sink = SlowCounter {
            count: count.clone(),
            clog: element_clog(ElementType::Other, "slow-counter", None),
        };
        let (bus, bus_rx) = Bus::new();

        let mut queue = Queue::spawn_with_policy(
            "test",
            1,
            Box::new(sink),
            bus,
            OverflowPolicy::default(),
            None,
        );
        for _ in 0..10 {
            queue.consume(packet()).unwrap();
        }
        queue.consume(MediaBuffer::Eos).unwrap();
        drop(queue); // blocks until the worker drains everything and joins

        assert_eq!(count.load(Ordering::SeqCst), 10);
        assert!(!bus_rx.iter().any(|e| matches!(e, BusEvent::Dropped { .. })));
    }

    #[test]
    fn block_with_a_finite_timeout_errors_instead_of_blocking_forever() {
        let count = Arc::new(AtomicUsize::new(0));
        let sink = SlowCounter {
            count: count.clone(),
            clog: element_clog(ElementType::Other, "slow-counter", None),
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
        );
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
            clog: element_clog(ElementType::Other, "slow-counter", None),
        };
        let (bus, bus_rx) = Bus::new();

        let mut queue = Queue::spawn_with_policy(
            "test",
            1,
            Box::new(sink),
            bus,
            OverflowPolicy::DropNewest,
            None,
        );
        // Pushed much faster than the 20ms/item downstream can drain a
        // capacity-1 channel, so some of these must get dropped.
        for _ in 0..10 {
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
            clog: element_clog(ElementType::Other, "slow-counter", None),
        };
        let (bus, _bus_rx) = Bus::new();

        let mut queue = Queue::spawn_with_policy(
            "test",
            8,
            Box::new(sink),
            bus,
            OverflowPolicy::default(),
            None,
        );
        queue.control(ControlMsg::Pause).unwrap(); // blocks until the worker is actually paused

        for _ in 0..3 {
            queue.consume(packet()).unwrap();
        }
        // Worker is paused and not touching data_rx — nothing should have
        // been processed yet, however long we wait.
        thread::sleep(Duration::from_millis(100));
        assert_eq!(count.load(Ordering::SeqCst), 0);

        queue.control(ControlMsg::Resume).unwrap();
        queue.consume(MediaBuffer::Eos).unwrap();
        drop(queue);

        assert_eq!(count.load(Ordering::SeqCst), 3);
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
            clog: element_clog(ElementType::Other, "slow-counter", None),
        };
        let (bus, _bus_rx) = Bus::new();

        let queue = Queue::spawn_with_policy(
            "test",
            8,
            Box::new(sink),
            bus,
            OverflowPolicy::default(),
            None,
        );
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
            clog: element_clog(ElementType::Other, "slow-counter", None),
        };
        let (bus, _bus_rx) = Bus::new();

        let mut queue = Queue::spawn_with_policy(
            "test",
            8,
            Box::new(sink),
            bus,
            OverflowPolicy::default(),
            None,
        );
        queue.control(ControlMsg::Pause).unwrap(); // blocks until the worker is actually paused
        drop(queue);
    }

    #[test]
    fn stop_is_synchronous_and_terminates_the_worker() {
        let count = Arc::new(AtomicUsize::new(0));
        let sink = SlowCounter {
            count: count.clone(),
            clog: element_clog(ElementType::Other, "slow-counter", None),
        };
        let (bus, _bus_rx) = Bus::new();

        let mut queue = Queue::spawn_with_policy(
            "test",
            8,
            Box::new(sink),
            bus,
            OverflowPolicy::default(),
            None,
        );
        queue.consume(packet()).unwrap();
        queue.control(ControlMsg::Stop).unwrap(); // blocks until the worker has exited
        drop(queue); // join should return immediately — the worker already returned
    }

    /// A downstream that fails on the very first `Packet` it sees, then
    /// behaves like `SlowCounter` for every one after.
    struct FailFirstThenCount {
        clog: CLog,
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

        fn clog(&self) -> &CLog {
            &self.clog
        }

        fn clog_mut(&mut self) -> &mut CLog {
            &mut self.clog
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

        fn control(&mut self, _msg: ControlMsg) -> Result<()> {
            Ok(())
        }
    }

    struct FailControl {
        clog: CLog,
    }

    impl Element for FailControl {
        fn name(&self) -> Arc<str> {
            "fail-control".into()
        }

        fn element_type(&self) -> ElementType {
            ElementType::Other
        }

        fn clog(&self) -> &CLog {
            &self.clog
        }

        fn clog_mut(&mut self) -> &mut CLog {
            &mut self.clog
        }
    }

    impl Sink for FailControl {
        fn consume(&mut self, _buf: MediaBuffer) -> Result<()> {
            Ok(())
        }

        fn control(&mut self, msg: ControlMsg) -> Result<()> {
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
            clog: element_clog(ElementType::Other, "fail-first", None),
        };
        let (bus, bus_rx) = Bus::new();

        let mut queue = Queue::spawn_with_policy(
            "test",
            8,
            Box::new(sink),
            bus,
            OverflowPolicy::default(),
            None,
        );
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

    /// Control failures are asynchronous worker failures just like
    /// `consume` failures: they must be visible on the Bus, but must not
    /// prevent Pause/Resume/Stop acknowledgements or strand the worker.
    #[test]
    fn failing_control_is_reported_without_blocking_the_control_cascade() {
        let sink = FailControl {
            clog: element_clog(ElementType::Other, "fail-control", None),
        };
        let (bus, bus_rx) = Bus::new();
        let mut queue = Queue::spawn_with_policy(
            "test",
            1,
            Box::new(sink),
            bus,
            OverflowPolicy::default(),
            None,
        );

        queue.control(ControlMsg::Pause).unwrap();
        queue.control(ControlMsg::Resume).unwrap();
        queue.control(ControlMsg::Stop).unwrap();
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
}
