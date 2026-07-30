use std::thread::{self, JoinHandle};

use crossbeam_channel::{Sender, TrySendError, bounded};
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    element::{Element, Sink},
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
/// Cheap elements (e.g. a muxer sitting right after an encoder) should
/// simply *not* have a `Queue` between them and their upstream — they run
/// as a direct call on the upstream element's thread instead of paying for
/// a dedicated thread they don't need.
pub struct Queue {
    name: String,
    tx: Sender<MediaBuffer>,
    policy: OverflowPolicy,
    bus: Bus,
    handle: Option<JoinHandle<()>>,
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
        mut downstream: Box<dyn Sink>,
        bus: Bus,
        policy: OverflowPolicy,
    ) -> Queue {
        let name = name.into();
        let (tx, rx) = bounded::<MediaBuffer>(capacity);
        let worker_name = name.clone();
        let worker_bus = bus.clone();

        let handle = thread::Builder::new()
            .name(format!("queue:{worker_name}"))
            .spawn(move || {
                for buf in rx.iter() {
                    let is_eos = buf.is_eos();
                    if let Err(e) = downstream.consume(buf) {
                        worker_bus.post(BusEvent::Error {
                            element: worker_name.clone(),
                            message: e.to_string(),
                        });
                        return;
                    }
                    if is_eos {
                        worker_bus.post(BusEvent::Eos {
                            element: worker_name.clone(),
                        });
                        return;
                    }
                }
            })
            .expect("failed to spawn queue worker thread");

        Queue {
            name,
            tx,
            policy,
            bus,
            handle: Some(handle),
        }
    }
}

impl Element for Queue {
    fn name(&self) -> &str {
        &self.name
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
                        element: self.name.clone(),
                    });
                    Ok(())
                }
                Err(TrySendError::Disconnected(_)) => Err(QueueError::ChannelClosed.into()),
            },
        }
    }
}

impl Drop for Queue {
    fn drop(&mut self) {
        // Dropping `tx` (once this is the last handle) closes the channel,
        // which lets the worker's `rx.iter()` loop end on its own.
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
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
        fn name(&self) -> &str {
            "slow-counter"
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
}
