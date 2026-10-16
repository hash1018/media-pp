use crate::pp_log::{PpLog, pp_info};

use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKindSet, MemoryDomainSet, PixelLayoutSet, PortContract},
    element::{Attached, ElementType, Identity, Sink},
    elements::CounterHandle,
    error::Result,
    render::{SinkStage, sink_stage},
};

/// Ends a branch by counting the decoded pictures or sound reaching it.
///
/// It takes decoded frames, video or audio, in any memory, and does nothing
/// else with them: it is the trivial end that makes a graph runnable while
/// something upstream is being tested. For packets use
/// [`PacketCounter`](super::PacketCounter). The [`CounterHandle`] it comes with
/// reads the count from outside the pipeline, even while this sink runs on a
/// `Queue` worker thread.
///
/// # Example
///
/// ```no_run
/// # fn main() -> media_pp::Result<()> {
/// use media_pp::elements::{FrameCounter, TestVideoOptions, TestVideoSource};
/// use media_pp::pipeline::Pipeline;
///
/// let source = TestVideoSource::new("source", TestVideoOptions::default());
/// let (counter, frames) = FrameCounter::new("counter");
///
/// let (pipeline, ()) = Pipeline::new("count", source, |source, ctx| {
///     let branch = ctx.branch().to(counter)?;
///     ctx.attach(source, 0, branch)?;
///     Ok(())
/// })?;
/// pipeline.run()?;
/// println!("pictures so far: {}", frames.get());
/// # Ok(())
/// # }
/// ```
pub struct FrameCounter(SinkStage<Counting>);

sink_stage!(FrameCounter);

/// What a [`FrameCounter`] does with each buffer: counts it, where it is one of
/// the decoded frames it counts.
struct Counting {
    pp_log: PpLog,
    count: CounterHandle,
}

impl FrameCounter {
    /// Creates the sink and the handle that reads how many frames it has
    /// counted.
    pub fn new(name: impl Into<String>) -> (Self, CounterHandle) {
        let count = CounterHandle::new();
        let identity = Identity::new(ElementType::FrameCounter, name);
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

    fn render(&mut self, buf: MediaBuffer) -> Result<()> {
        if let MediaBuffer::Video(_) | MediaBuffer::Audio(_) = buf {
            self.count.increment();
        }
        Ok(())
    }

    fn attached(&mut self, cx: &Attached<'_>) {
        self.pp_log = cx.pp_log().clone();
    }
}
