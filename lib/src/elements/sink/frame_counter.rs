use std::sync::Arc;

use crate::pp_log::{PpLog, pp_info};

use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKindSet, MemoryDomainSet, PixelLayoutSet, PortContract},
    control::ControlMsg,
    element::{Element, ElementType, Sink, element_pp_log},
    elements::CounterHandle,
    error::Result,
};

/// Terminal sink that counts decoded frames, video or audio. The
/// [`CounterHandle`] it comes with reads the count from outside the
/// pipeline, even while this sink runs on a `Queue` worker thread.
pub struct FrameCounter {
    pp_log: PpLog,
    name: Arc<str>,
    count: CounterHandle,
}

impl FrameCounter {
    /// Creates the sink and the handle that reads how many frames it has
    /// counted.
    pub fn new(name: impl Into<String>) -> (Self, CounterHandle) {
        let count = CounterHandle::new();
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::FrameCounter, &name, None);
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

impl Element for FrameCounter {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::FrameCounter
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Sink for FrameCounter {
    /// Counts decoded buffers of either medium — it only tallies them,
    /// so it neither reads the samples nor cares which memory they live
    /// in. PacketCounter is the encoded-side counterpart.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::Frames(
            MediaKindSet::FRAMES,
            MemoryDomainSet::ALL,
            PixelLayoutSet::ALL,
        ))
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        if let MediaBuffer::Video(_) | MediaBuffer::Audio(_) = buf {
            self.count.increment();
        }
        Ok(())
    }

    fn control(&mut self, _msg: ControlMsg) -> Result<()> {
        // Terminal, nothing to flush or forward.
        Ok(())
    }
}
