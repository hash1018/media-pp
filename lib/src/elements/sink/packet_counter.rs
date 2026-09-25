use std::sync::Arc;

use crate::pp_log::{PpLog, pp_info};

use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKindSet, PortContract},
    element::{Element, ElementType, Sink, element_pp_log},
    elements::CounterHandle,
    error::Result,
};

/// Terminal sink that counts packets. The [`CounterHandle`] it comes
/// with reads the count from outside the pipeline, even while this sink runs
/// on a `Queue` worker thread.
pub struct PacketCounter {
    pp_log: PpLog,
    name: Arc<str>,
    count: CounterHandle,
}

impl PacketCounter {
    /// Creates the sink and the handle that reads how many packets it has
    /// counted.
    pub fn new(name: impl Into<String>) -> (Self, CounterHandle) {
        let count = CounterHandle::new();
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
            self.count.increment();
        }
        Ok(())
    }
}
