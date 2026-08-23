use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use crate::pp_log::{PpLog, pp_info};

use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKindSet, PortContract},
    control::ControlMsg,
    element::{Element, ElementType, Sink, element_pp_log},
    error::Result,
};

/// Terminal sink that just counts packets. Backed by an `Arc<AtomicUsize>`
/// so the count can be read from outside the pipeline even when this sink
/// ends up running on a `Queue` worker thread.
pub struct PacketCounter {
    pp_log: PpLog,
    name: Arc<str>,
    count: Arc<AtomicUsize>,
}

impl PacketCounter {
    /// Creates a sink and a shared counter that increments for each packet.
    pub fn new(name: impl Into<String>) -> (Self, Arc<AtomicUsize>) {
        let count = Arc::new(AtomicUsize::new(0));
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::PacketCounter, &name, None);
        pp_info!(pp_log: &pp_log, "created");
        (
            Self {
                name,
                pp_log,
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

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Sink for PacketCounter {
    /// Counts encoded packets specifically — FrameCounter is the decoded-side counterpart.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::Packets(MediaKindSet::PACKETS))
    }

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
