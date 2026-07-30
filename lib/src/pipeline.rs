use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent, BusReceiver},
    element::{Element, Filter, Sink, SourceElement},
    error::Result,
    queue::{OverflowPolicy, Queue},
};

/// Builds one chain segment (a run of elements that all execute on the same
/// thread). Call [`ChainBuilder::queue`] to close the current segment behind
/// a `Queue` and start a new one on its own worker thread.
///
/// Because each element needs a handle to *its* downstream to be
/// constructed, the chain is assembled back-to-front: elements are
/// collected in call order, then folded right-to-left starting from the
/// terminal `Sink` at [`ChainBuilder::build`] time.
pub struct ChainBuilder {
    bus: Bus,
    elements: Vec<Box<dyn StageBuilder>>,
}

trait StageBuilder: Send {
    fn wrap(self: Box<Self>, downstream: Box<dyn Sink>, bus: &Bus) -> Box<dyn Sink>;
}

struct DirectStage<T>(T);

impl<T> StageBuilder for DirectStage<T>
where
    T: Filter + 'static,
{
    fn wrap(self: Box<Self>, downstream: Box<dyn Sink>, _bus: &Bus) -> Box<dyn Sink> {
        let mut element = self.0;
        assert_eq!(
            element.src_pads().len(),
            1,
            "ChainBuilder::pipe() is for single-output elements; link a multi-pad \
             element's src_pads() by hand instead"
        );
        element.src_pads()[0].link(downstream);
        Box::new(element)
    }
}

struct QueueStage {
    name: String,
    capacity: usize,
    policy: OverflowPolicy,
}

/// Wraps a terminal `Sink`, posting a `BusEvent::Eos` (under the sink's own
/// `Element::name()`) once it sees an EOS buffer pass through — mirrors
/// what `Queue` does for its own downstream, but without introducing a
/// thread boundary. This is what lets a fully direct chain (no `queue()`
/// calls at all) still report EOS on the bus.
struct EosReporter {
    bus: Bus,
    inner: Box<dyn Sink>,
}

impl Element for EosReporter {
    fn name(&self) -> &str {
        self.inner.name()
    }
}

impl Sink for EosReporter {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        let is_eos = buf.is_eos();
        self.inner.consume(buf)?;
        if is_eos {
            self.bus.post(BusEvent::Eos {
                element: self.inner.name().to_string(),
            });
        }
        Ok(())
    }
}

impl StageBuilder for QueueStage {
    fn wrap(self: Box<Self>, downstream: Box<dyn Sink>, bus: &Bus) -> Box<dyn Sink> {
        Box::new(Queue::spawn_with_policy(
            self.name,
            self.capacity,
            downstream,
            bus.clone(),
            self.policy,
        ))
    }
}

impl ChainBuilder {
    pub fn new(bus: Bus) -> Self {
        Self {
            bus,
            elements: Vec::new(),
        }
    }

    /// Adds a single-output `Filter` (decoder, encoder, filter, ...) that
    /// receives via `Sink` and produces through its own (single) src pad.
    /// It runs on the same thread as whatever is upstream of it — direct
    /// function call, no queue.
    pub fn pipe<T: Filter + 'static>(mut self, element: T) -> Self {
        self.elements.push(Box::new(DirectStage(element)));
        self
    }

    /// Introduces a thread boundary (blocking when full — see
    /// [`OverflowPolicy::Block`]): everything added after this runs on its
    /// own worker thread instead of the thread that feeds this queue.
    pub fn queue(self, name: impl Into<String>, capacity: usize) -> Self {
        self.queue_with_policy(name, capacity, OverflowPolicy::Block)
    }

    /// Same as [`ChainBuilder::queue`], but lets you choose what happens
    /// when the queue is full (e.g. [`OverflowPolicy::DropNewest`] for a
    /// live source that shouldn't stall upstream).
    pub fn queue_with_policy(
        mut self,
        name: impl Into<String>,
        capacity: usize,
        policy: OverflowPolicy,
    ) -> Self {
        self.elements.push(Box::new(QueueStage {
            name: name.into(),
            capacity,
            policy,
        }));
        self
    }

    /// Terminates the chain with a `Sink` (muxer, file sink, ...) and
    /// assembles everything into a single `Box<dyn Sink>` ready to be
    /// linked into a source's src pad. The terminal's own `Element::name()`
    /// is what shows up on the bus when it reports EOS.
    pub fn build(self, terminal: Box<dyn Sink>) -> Box<dyn Sink> {
        let terminal: Box<dyn Sink> = Box::new(EosReporter {
            bus: self.bus.clone(),
            inner: terminal,
        });
        self.elements
            .into_iter()
            .rev()
            .fold(terminal, |downstream, stage| {
                stage.wrap(downstream, &self.bus)
            })
    }
}

/// Top-level pipeline: a source (with everything reachable from its src
/// pads already linked) plus the bus it reports events on.
///
/// `run()` drives the source on the calling thread. It returns once the
/// source has pushed EOS *and* every queue worker thread downstream has
/// drained and joined — so by the time it returns, the whole pipeline has
/// finished, all the way to the last sink on every branch.
pub struct Pipeline {
    source: Option<Box<dyn SourceElement>>,
    bus_rx: BusReceiver,
}

impl Pipeline {
    /// `wire` is called once with the freshly created source and a `Bus`,
    /// so it can build one or more `ChainBuilder` chains (one per src pad
    /// that should actually be used) and link them via
    /// `source.src_pads()[i].link(...)`. Pads left unlinked just drop
    /// whatever gets pushed into them.
    pub fn new<S: SourceElement + 'static>(mut source: S, wire: impl FnOnce(&mut S, &Bus)) -> Self {
        let (bus, bus_rx) = Bus::new();
        wire(&mut source, &bus);
        Pipeline {
            source: Some(Box::new(source)),
            bus_rx,
        }
    }

    pub fn bus(&self) -> &BusReceiver {
        &self.bus_rx
    }

    /// Drives the source on the calling thread. By the time this returns,
    /// the source has pushed EOS *and* every `Queue` worker thread
    /// downstream has drained and joined — the whole pipeline is done.
    pub fn run(&mut self) -> Result<()> {
        let mut source = self.source.take().expect("pipeline already run");
        source.run()
        // `source` drops here (end of scope), joining any `Queue` worker
        // threads it transitively owns (through its linked src pads)
        // before `run` returns.
    }
}
