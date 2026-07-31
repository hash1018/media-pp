use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Element, ElementType, Sink},
    error::Result,
};

/// Terminal sink that just counts packets. Backed by an `Arc<AtomicUsize>`
/// so the count can be read from outside the pipeline even when this sink
/// ends up running on a `Queue` worker thread.
pub struct PacketCounter {
    name: Arc<str>,
    count: Arc<AtomicUsize>,
}

impl PacketCounter {
    pub fn new(name: impl Into<String>) -> (Self, Arc<AtomicUsize>) {
        let count = Arc::new(AtomicUsize::new(0));
        (
            Self {
                name: name.into().into(),
                count: count.clone(),
            },
            count,
        )
    }
}

impl Element for PacketCounter {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::PacketCounter
    }
}

impl Sink for PacketCounter {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        if let MediaBuffer::Packet(_) = buf {
            self.count.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    fn control(&mut self, _msg: ControlMsg) -> Result<()> {
        // Terminal, nothing to flush or forward.
        Ok(())
    }
}
