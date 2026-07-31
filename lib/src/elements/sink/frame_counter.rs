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

/// Terminal sink that counts decoded frames (video or audio). Backed by
/// an `Arc<AtomicUsize>` so the count can be read from outside the
/// pipeline even when this sink ends up running on a `Queue` worker
/// thread.
pub struct FrameCounter {
    name: String,
    count: Arc<AtomicUsize>,
}

impl FrameCounter {
    pub fn new(name: impl Into<String>) -> (Self, Arc<AtomicUsize>) {
        let count = Arc::new(AtomicUsize::new(0));
        (
            Self {
                name: name.into(),
                count: count.clone(),
            },
            count,
        )
    }
}

impl Element for FrameCounter {
    fn name(&self) -> &str {
        &self.name
    }

    fn element_type(&self) -> ElementType {
        ElementType::FrameCounter
    }
}

impl Sink for FrameCounter {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        if let MediaBuffer::Video(_) | MediaBuffer::Audio(_) = buf {
            self.count.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    fn control(&mut self, _msg: ControlMsg) -> Result<()> {
        // Terminal, nothing to flush or forward.
        Ok(())
    }
}
