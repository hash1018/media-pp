use crate::pp_log::{PpLog, pp_info};

use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKindSet, PortContract},
    element::{Attached, ElementType, Identity, Sink},
    elements::CounterHandle,
    error::Result,
    render::{SinkStage, sink_stage},
};

/// Terminal sink that counts packets. The [`CounterHandle`] it comes
/// with reads the count from outside the pipeline, even while this sink runs
/// on a `Queue` worker thread.
pub struct PacketCounter(SinkStage<Counting>);

sink_stage!(PacketCounter);

/// What a [`PacketCounter`] does with each buffer: counts it, where it is one of
/// the packets it counts.
struct Counting {
    pp_log: PpLog,
    count: CounterHandle,
}

impl PacketCounter {
    /// Creates the sink and the handle that reads how many packets it has
    /// counted.
    pub fn new(name: impl Into<String>) -> (Self, CounterHandle) {
        let count = CounterHandle::new();
        let identity = Identity::new(ElementType::PacketCounter, name);
        let pp_log = identity.pp_log().clone();
        pp_info!(pp_log: &pp_log, "created");
        (
            Self(SinkStage::new(
                identity,
                Counting {
                    pp_log,
                    count: count.clone(),
                },
            )),
            count,
        )
    }
}

impl Sink for Counting {
    /// Counts encoded packets specifically — FrameCounter is the decoded-side counterpart.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::Packets(MediaKindSet::PACKETS))
    }

    fn render(&mut self, buf: MediaBuffer) -> Result<()> {
        if let MediaBuffer::Packet(_) = buf {
            self.count.increment();
        }
        Ok(())
    }

    fn attached(&mut self, cx: &Attached<'_>) {
        self.pp_log = cx.pp_log().clone();
    }
}
