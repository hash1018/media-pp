use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use crate::clog::{CLog, cinfo};

use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Element, ElementType, Sink, element_clog},
    error::Result,
};

/// Terminal sink that counts decoded frames (video or audio). Backed by
/// an `Arc<AtomicUsize>` so the count can be read from outside the
/// pipeline even when this sink ends up running on a `Queue` worker
/// thread.
pub struct FrameCounter {
    clog: CLog,
    name: Arc<str>,
    count: Arc<AtomicUsize>,
}

impl FrameCounter {
    pub fn new(name: impl Into<String>) -> (Self, Arc<AtomicUsize>) {
        let count = Arc::new(AtomicUsize::new(0));
        let name: Arc<str> = name.into().into();
        let clog = element_clog(ElementType::FrameCounter, &name, None);
        cinfo!(clog: &clog, "created");
        (
            Self {
                name,
                clog,
                count: count.clone(),
            },
            count,
        )
    }
}

impl Element for FrameCounter {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::FrameCounter
    }

    fn clog(&self) -> &CLog {
        &self.clog
    }

    fn clog_mut(&mut self) -> &mut CLog {
        &mut self.clog
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
