use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use rust_hlog::HLog;

use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Element, ElementType, Sink, element_hlog},
    error::Result,
};

/// Terminal sink that just counts packets. Backed by an `Arc<AtomicUsize>`
/// so the count can be read from outside the pipeline even when this sink
/// ends up running on a `Queue` worker thread.
#[rust_hlog::hlog]
pub struct PacketCounter {
    name: Arc<str>,
    count: Arc<AtomicUsize>,
}

impl PacketCounter {
    pub fn new(name: impl Into<String>) -> (Self, Arc<AtomicUsize>) {
        let count = Arc::new(AtomicUsize::new(0));
        let name: Arc<str> = name.into().into();
        let hlog = element_hlog(ElementType::PacketCounter, &name, None);
        (
            Self {
                name,
                hlog,
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

    fn hlog(&self) -> &HLog {
        &self.hlog
    }

    fn hlog_mut(&mut self) -> &mut HLog {
        &mut self.hlog
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
