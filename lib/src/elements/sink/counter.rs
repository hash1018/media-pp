use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

/// How many buffers a [`FrameCounter`](super::FrameCounter) or
/// [`PacketCounter`](super::PacketCounter) has counted, readable from any
/// thread while its sink runs — on a `Queue` worker as much as inline.
///
/// Read-only: only the sink it came with counts. Cloning is cheap and shares
/// the one count; a handle keeps nothing alive but that count, so once the
/// sink is dropped it goes on reading the final number.
#[derive(Debug, Clone)]
pub struct CounterHandle(Arc<AtomicUsize>);

impl CounterHandle {
    pub(super) fn new() -> Self {
        Self(Arc::new(AtomicUsize::new(0)))
    }

    pub(super) fn increment(&self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }

    /// The count so far. Never blocks.
    pub fn get(&self) -> usize {
        self.0.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ffmpeg_next as ffmpeg;

    use crate::{buffer::MediaBuffer, element::Sink, elements::PacketCounter};

    /// Every clone reads the one count its sink keeps, and goes on reading
    /// the final number once that sink is gone; what the sink does not
    /// count, such as `Eos`, leaves it alone.
    #[test]
    fn a_handle_reads_what_its_sink_counted_after_the_sink_is_gone() {
        let (mut counter, count) = PacketCounter::new("counter");
        let clone = count.clone();
        for _ in 0..3 {
            counter
                .consume(MediaBuffer::Packet(Arc::new(ffmpeg::Packet::empty())))
                .expect("a packet counts");
        }
        counter.consume(MediaBuffer::Eos).expect("Eos passes");
        drop(counter);

        assert_eq!(count.get(), 3);
        assert_eq!(clone.get(), 3);
    }
}
