use std::{
    sync::Arc,
    thread::{self, JoinHandle},
};

use crossbeam_channel::{Receiver, Sender, TrySendError, bounded, select};
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    control::{self, ControlMsg, ControlReceiver, ControlSender},
    element::{Element, ElementType, Sink},
    error::Result,
};

/// Errors specific to `Queue`. Converts into the crate-wide `Error` via
/// `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum QueueError {
    #[error("downstream channel closed")]
    ChannelClosed,
}

/// What a `Queue` does when its channel is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OverflowPolicy {
    /// Block the pushing thread until there's room. Never loses data —
    /// the right choice for offline/file processing, where correctness
    /// matters more than staying caught up.
    #[default]
    Block,
    /// Drop the incoming buffer instead of blocking, and post
    /// [`BusEvent::Dropped`]. Never stalls the upstream thread — the
    /// right choice for live sources, where falling behind is worse than
    /// losing a frame.
    DropNewest,
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
/// channel from data, and the worker always checks it first — so
/// `Pause`/`Stop` never wait behind whatever's already backed up in the
/// data channel. Every worker acks a control message *before* acting on
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
    name: Arc<str>,
    tx: Sender<MediaBuffer>,
    policy: OverflowPolicy,
    bus: Bus,
    handle: Option<JoinHandle<()>>,
    control: ControlSender,
}

impl Queue {
    /// Spawns with [`OverflowPolicy::Block`]. Use
    /// [`Queue::spawn_with_policy`] to drop instead of blocking when full.
    pub fn spawn(
        name: impl Into<String>,
        capacity: usize,
        downstream: Box<dyn Sink>,
        bus: Bus,
    ) -> Queue {
        Self::spawn_with_policy(name, capacity, downstream, bus, OverflowPolicy::Block)
    }

    /// Spawns the worker thread that owns `downstream` and starts pulling
    /// from the channel immediately.
    pub fn spawn_with_policy(
        name: impl Into<String>,
        capacity: usize,
        downstream: Box<dyn Sink>,
        bus: Bus,
        policy: OverflowPolicy,
    ) -> Queue {
        // Stored as `Arc<str>` (not `String`) so the `worker_name.clone()`
        // below, and every subsequent `BusEvent` this posts, are a
        // refcount bump instead of a fresh allocation — `Dropped` in
        // particular can fire once per buffer under sustained overflow.
        let name: Arc<str> = name.into().into();
        let (tx, rx) = bounded::<MediaBuffer>(capacity);
        let (control_tx, control_rx) = control::channel();
        let worker_name = name.clone();
        let worker_bus = bus.clone();

        let handle = thread::Builder::new()
            .name(format!("queue:{worker_name}"))
            .spawn(move || worker_loop(rx, control_rx, downstream, worker_bus, worker_name))
            .expect("failed to spawn queue worker thread");

        Queue {
            name,
            tx,
            policy,
            bus,
            handle: Some(handle),
            control: control_tx,
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
}

impl Sink for Queue {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        // EOS must never be dropped, regardless of policy: it's the only
        // shutdown signal the worker thread gets, and losing it would
        // leave that thread (and this Queue's `Drop`) blocked forever.
        if buf.is_eos() {
            return self
                .tx
                .send(buf)
                .map_err(|_| QueueError::ChannelClosed.into());
        }

        match self.policy {
            OverflowPolicy::Block => self
                .tx
                .send(buf)
                .map_err(|_| QueueError::ChannelClosed.into()),
            OverflowPolicy::DropNewest => match self.tx.try_send(buf) {
                Ok(()) => Ok(()),
                Err(TrySendError::Full(_)) => {
                    self.bus.post(BusEvent::Dropped {
                        element_type: ElementType::Queue,
                        name: self.name.clone(),
                    });
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
        // Dropping `tx` (once this is the last handle) closes the data
        // channel, which lets the worker's loop end on its own if it
        // hasn't already (e.g. via `Stop`).
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Owns `downstream` on its own thread: pulls from `data_rx` and calls
/// `downstream.consume()`, same as before, except every iteration checks
/// `control_rx` *first* — so a pending `Pause`/`Stop` is handled before
/// the next data buffer, however deep the backlog. `Pause` blocks this
/// whole function (and therefore `downstream`) right here, without
/// touching `data_rx` at all, until `Resume`/`Stop`.
fn worker_loop(
    data_rx: Receiver<MediaBuffer>,
    control_rx: ControlReceiver,
    mut downstream: Box<dyn Sink>,
    bus: Bus,
    name: Arc<str>,
) {
    loop {
        if let Some((msg, ack)) = control_rx.try_recv() {
            if apply_control(&data_rx, &mut downstream, msg, &ack, &control_rx) {
                return;
            }
            continue;
        }

        select! {
            recv(control_rx.rx) -> req => {
                match req {
                    Ok(req) => {
                        if apply_control(&data_rx, &mut downstream, req.msg, &req.ack, &control_rx) {
                            return;
                        }
                    }
                    Err(_) => return, // sender (this Queue) dropped
                }
            }
            recv(data_rx) -> buf => {
                match buf {
                    Ok(buf) => {
                        let is_eos = buf.is_eos();
                        match downstream.consume(buf) {
                            Ok(()) => {
                                if is_eos {
                                    bus.post(BusEvent::Eos {
                                        element_type: ElementType::Queue,
                                        name: name.clone(),
                                    });
                                    return;
                                }
                            }
                            Err(error) => {
                                // Report and move on to the next buffer —
                                // this one's dropped, but nothing else
                                // dies over it. Whoever's watching the bus
                                // decides whether the error is fatal
                                // enough to call `Pipeline::stop`.
                                bus.post(BusEvent::Error {
                                    element_type: ElementType::Queue,
                                    name: name.clone(),
                                    error,
                                });
                            }
                        }
                    }
                    Err(_) => return, // producer side (this Queue) dropped
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
) -> bool {
    discard_stale_data(data_rx, msg);
    let _ = downstream.control(msg);
    let is_stop = msg == ControlMsg::Stop;
    let _ = ack.send(());
    if is_stop {
        return true;
    }
    if msg != ControlMsg::Pause {
        return false;
    }
    loop {
        let Some((msg, ack)) = control_rx.recv() else {
            return true; // sender gone — treat like Stop
        };
        discard_stale_data(data_rx, msg);
        let _ = downstream.control(msg);
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
        count: Arc<AtomicUsize>,
    }

    impl Element for SlowCounter {
        fn name(&self) -> Arc<str> {
            "slow-counter".into()
        }

        fn element_type(&self) -> ElementType {
            ElementType::Other
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
        };
        let (bus, bus_rx) = Bus::new();

        let mut queue =
            Queue::spawn_with_policy("test", 1, Box::new(sink), bus, OverflowPolicy::Block);
        for _ in 0..10 {
            queue.consume(packet()).unwrap();
        }
        queue.consume(MediaBuffer::Eos).unwrap();
        drop(queue); // blocks until the worker drains everything and joins

        assert_eq!(count.load(Ordering::SeqCst), 10);
        assert!(!bus_rx.iter().any(|e| matches!(e, BusEvent::Dropped { .. })));
    }

    #[test]
    fn drop_newest_drops_when_full_and_reports_on_bus() {
        let count = Arc::new(AtomicUsize::new(0));
        let sink = SlowCounter {
            count: count.clone(),
        };
        let (bus, bus_rx) = Bus::new();

        let mut queue =
            Queue::spawn_with_policy("test", 1, Box::new(sink), bus, OverflowPolicy::DropNewest);
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
        };
        let (bus, _bus_rx) = Bus::new();

        let mut queue =
            Queue::spawn_with_policy("test", 8, Box::new(sink), bus, OverflowPolicy::Block);
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

    #[test]
    fn stop_is_synchronous_and_terminates_the_worker() {
        let count = Arc::new(AtomicUsize::new(0));
        let sink = SlowCounter {
            count: count.clone(),
        };
        let (bus, _bus_rx) = Bus::new();

        let mut queue =
            Queue::spawn_with_policy("test", 8, Box::new(sink), bus, OverflowPolicy::Block);
        queue.consume(packet()).unwrap();
        queue.control(ControlMsg::Stop).unwrap(); // blocks until the worker has exited
        drop(queue); // join should return immediately — the worker already returned
    }

    /// A downstream that fails on the very first `Packet` it sees, then
    /// behaves like `SlowCounter` for every one after.
    struct FailFirstThenCount {
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
        };
        let (bus, bus_rx) = Bus::new();

        let mut queue =
            Queue::spawn_with_policy("test", 8, Box::new(sink), bus, OverflowPolicy::Block);
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
}
