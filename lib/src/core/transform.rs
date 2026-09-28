//! Writing a filter by its media work alone — see [`Transform`].
//!
//! A filter written directly, as [`Sink`] and [`Source`], owns everything
//! the pipeline asks of one: its pad, handing its end on after what it still
//! holds, dropping what a seek left behind, saying whether it can take a
//! buffer. Each did that for itself, and each a little differently —
//! docs/stream-events.md lists what that cost. A [`Transform`] does the
//! media work and nothing else; the framework does the rest, once, the same
//! way for every one.

use std::sync::Arc;

use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, OutputContract},
    control::ControlMsg,
    element::{Context, Element, ElementType, Filter, Sink, Source},
    error::Result,
    graph::ElementId,
    pad::SrcPad,
    pp_log::PpLog,
    stash::OutputStash,
    stream::StreamEvent,
};

/// A filter written as its media work alone: each buffer in, what it makes
/// of it out.
///
/// The framework does the rest — the one output pad, the buffers handed on
/// in order, `drain` before the end of the stream goes on, `reset` when a
/// seek leaves the timeline behind, whether a buffer can be taken now. The
/// element never sees a control message, a pause, a preroll or a seek.
///
/// Put in a pipeline as any filter is, with
/// [`ChainBuilder::pipe`](crate::pipeline::ChainBuilder::pipe), which takes
/// a `Transform` as it takes a [`Filter`]; where a list of filters is asked
/// for — [`Rack`](crate::elements::Rack) — [`IntoFilter::into_filter`] makes
/// one. Implement this or [`Sink`] and [`Source`], not both: a filter that
/// routes the stream itself — splits it, holds it back, waits on a clock —
/// is written the direct way.
pub trait Transform: Element {
    /// Makes what `buf` answers to, into `out` — nothing, one buffer, or
    /// several. Never handed the end of the stream: that is `drain`'s.
    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()>;

    /// What it still holds, into `out`, before the end of the stream goes
    /// on — a codec's delayed output. Nothing by default.
    fn drain(&mut self, _out: &mut Output) -> Result<()> {
        Ok(())
    }

    /// Lets go of what belongs to the timeline a seek has left, or to a
    /// stream that has been stopped. Nothing by default.
    fn reset(&mut self) {}

    /// What it takes — see [`Sink::input_contract`]. Nothing said by default.
    fn input_contract(&self) -> InputContract {
        InputContract::Unknown
    }

    /// What it hands on — see [`SrcPad::with_contract`]. Nothing said by
    /// default.
    fn output_contract(&self) -> OutputContract {
        OutputContract::Unknown
    }

    /// Whether it can follow a seek — see [`Sink::accepts_seek`]. Yes by
    /// default.
    fn accepts_seek(&self) -> bool {
        true
    }
}

/// Where a [`Transform`] puts what it makes, in the order it is to go on.
#[derive(Default)]
pub struct Output {
    made: Vec<MediaBuffer>,
}

impl Output {
    /// Hands `buf` on, after whatever was put here before it.
    pub fn push(&mut self, buf: MediaBuffer) {
        self.made.push(buf);
    }
}

/// What a [`Transform`] is in a pipeline: the filter the framework makes of
/// it, with the pad and the rules it does not keep itself.
pub(crate) struct TransformStage<T> {
    inner: T,
    pad: SrcPad,
    /// What it made that its pad could not take while a preroll held the
    /// graph — see [`OutputStash`].
    stash: OutputStash,
}

impl<T: Transform> TransformStage<T> {
    pub(crate) fn new(inner: T) -> Self {
        let pad = SrcPad::with_contract(format!("{}_src", inner.name()), inner.output_contract());
        Self {
            inner,
            pad,
            stash: OutputStash::default(),
        }
    }

    /// The transform itself — for the element of this crate that is a
    /// newtype over its stage, to reach what it is made of.
    pub(crate) fn inner(&self) -> &T {
        &self.inner
    }

    /// The same, to change it — for a test that drives the work directly.
    #[cfg(test)]
    pub(crate) fn inner_mut(&mut self) -> &mut T {
        &mut self.inner
    }

    /// Hands on what `out` holds, through the stash, every buffer even
    /// where one is refused, and answers the first refusal.
    fn hand_on(&mut self, out: Output) -> Result<()> {
        let mut first = Ok(());
        for buf in out.made {
            let pushed = self.stash.push(&mut self.pad, buf);
            if first.is_ok() {
                first = pushed;
            }
        }
        first
    }
}

impl<T: Transform> Element for TransformStage<T> {
    fn name(&self) -> Arc<str> {
        self.inner.name()
    }

    fn element_type(&self) -> ElementType {
        self.inner.element_type()
    }

    fn graph_id(&self) -> Option<ElementId> {
        self.inner.graph_id()
    }

    fn pp_log(&self) -> &PpLog {
        self.inner.pp_log()
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        self.inner.pp_log_mut()
    }

    fn attach_context(&mut self, context: &Arc<Context>) {
        self.stash.attach(&context.state);
        self.inner.attach_context(context);
    }
}

impl<T: Transform> Source for TransformStage<T> {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl<T: Transform> Sink for TransformStage<T> {
    fn ready_consume(&mut self) -> bool {
        self.stash.ready(&mut self.pad) && self.pad.ready_consume()
    }

    fn input_contract(&self) -> InputContract {
        self.inner.input_contract()
    }

    fn accepts_seek(&self) -> bool {
        self.inner.accepts_seek()
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        let mut out = Output::default();
        self.inner.transform(buf, &mut out)?;
        self.hand_on(out)
    }

    fn control(&mut self, msg: &ControlMsg) -> Result<()> {
        if matches!(msg, ControlMsg::Flush | ControlMsg::Stop) {
            self.inner.reset();
            self.stash.clear();
        }
        Ok(())
    }

    /// What is kept goes ahead of the event the graph passes on after this:
    /// it came before it. A seek's segment finds nothing kept, its `Flush`
    /// having let it go; only one that begins a lap or a bridge's new input
    /// can, and then a preroll's terminal is handed what it would otherwise
    /// have waited for.
    ///
    /// At the end, what the transform still holds goes on too, after what
    /// was kept — and the graph passes the end on after it whether or not
    /// draining failed: a stream that is over is over.
    fn stream_event(&mut self, event: &StreamEvent) -> Result<()> {
        let kept = self.stash.release_all(&mut self.pad);
        if !matches!(event, StreamEvent::Eos) {
            return kept;
        }
        let mut out = Output::default();
        let drained = self.inner.drain(&mut out);
        let mut handed = Ok(());
        for buf in out.made {
            let pushed = self.pad.push(buf);
            if handed.is_ok() {
                handed = pushed;
            }
        }
        kept.and(drained).and(handed)
    }
}

/// What can be put where a [`Filter`] is asked for: a filter itself, or a
/// [`Transform`], which the framework makes one of.
///
/// `M` says which of the two a type is, and the compiler works it out:
/// nothing names it. A type that is both is refused as ambiguous —
/// implement one or the other.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is neither a `Filter` nor a `Transform`",
    note = "implement `Transform` for an element that makes buffers of buffers, or `Sink` and `Source` for one that routes the stream itself"
)]
pub trait IntoFilter<M>: sealed::Sealed<M> {
    /// The filter this is, boxed — for a list of filters, as a
    /// [`Rack`](crate::elements::Rack) takes.
    fn into_filter(self) -> Box<dyn Filter>;
}

/// Says a type goes in as the [`Filter`] it is — see [`IntoFilter`].
pub enum AsFilter {}

/// Says a type goes in as the [`Transform`] it is — see [`IntoFilter`].
pub enum AsTransform {}

impl<F: Filter + 'static> IntoFilter<AsFilter> for F {
    fn into_filter(self) -> Box<dyn Filter> {
        Box::new(self)
    }
}

impl<T: Transform + 'static> IntoFilter<AsTransform> for T {
    fn into_filter(self) -> Box<dyn Filter> {
        Box::new(TransformStage::new(self))
    }
}

mod sealed {
    use super::{AsFilter, AsTransform, Filter, Transform};

    /// Keeps [`super::IntoFilter`] to the two ways in it has.
    pub trait Sealed<M> {}

    impl<F: Filter + 'static> Sealed<AsFilter> for F {}
    impl<T: Transform + 'static> Sealed<AsTransform> for T {}
}

/// Makes `$name`, a newtype over `TransformStage<_>`, the filter its stage
/// is — every method the stage's. For an element of this crate that keeps
/// the public name, constructors and `Filter` it always had while its work
/// moves into a [`Transform`].
macro_rules! transform_filter {
    ($name:ident) => {
        impl $crate::element::Element for $name {
            fn name(&self) -> ::std::sync::Arc<str> {
                self.0.name()
            }
            fn element_type(&self) -> $crate::element::ElementType {
                self.0.element_type()
            }
            fn graph_id(&self) -> Option<$crate::graph::ElementId> {
                self.0.graph_id()
            }
            fn pp_log(&self) -> &$crate::pp_log::PpLog {
                self.0.pp_log()
            }
            fn pp_log_mut(&mut self) -> &mut $crate::pp_log::PpLog {
                self.0.pp_log_mut()
            }
            fn attach_context(&mut self, context: &::std::sync::Arc<$crate::element::Context>) {
                self.0.attach_context(context);
            }
        }

        impl $crate::element::Source for $name {
            fn src_pads(&mut self) -> &mut [$crate::pad::SrcPad] {
                self.0.src_pads()
            }
        }

        impl $crate::element::Sink for $name {
            fn ready_consume(&mut self) -> bool {
                self.0.ready_consume()
            }
            fn consume(&mut self, buf: $crate::buffer::MediaBuffer) -> $crate::error::Result<()> {
                self.0.consume(buf)
            }
            fn input_contract(&self) -> $crate::contract::InputContract {
                self.0.input_contract()
            }
            fn accepts_seek(&self) -> bool {
                self.0.accepts_seek()
            }
            fn control(&mut self, msg: &$crate::control::ControlMsg) -> $crate::error::Result<()> {
                self.0.control(msg)
            }
            fn stream_event(
                &mut self,
                event: &$crate::stream::StreamEvent,
            ) -> $crate::error::Result<()> {
                self.0.stream_event(event)
            }
        }
    };
}

pub(crate) use transform_filter;

#[cfg(test)]
mod tests {
    use std::sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    };

    use ffmpeg_next as ffmpeg;

    use super::*;
    use crate::{
        contract::{MediaKind, PortContract},
        element::element_pp_log,
    };

    fn packet(tag: u8) -> MediaBuffer {
        MediaBuffer::Packet(Arc::new(ffmpeg::Packet::copy(&[tag])))
    }

    /// The tag of each packet in `buffers`.
    fn tags(buffers: &[MediaBuffer]) -> Vec<Option<u8>> {
        buffers
            .iter()
            .map(|buf| match buf {
                MediaBuffer::Packet(packet) => Some(packet.data().expect("a payload")[0]),
                _ => panic!("only packets are made here"),
            })
            .collect()
    }

    /// Writes down the tag of each packet it takes, and `None` for the end.
    struct Heard(PpLog, Arc<Mutex<Vec<Option<u8>>>>);

    impl Element for Heard {
        fn name(&self) -> Arc<str> {
            "heard".into()
        }

        fn element_type(&self) -> ElementType {
            ElementType::Other
        }

        fn pp_log(&self) -> &PpLog {
            &self.0
        }

        fn pp_log_mut(&mut self) -> &mut PpLog {
            &mut self.0
        }
    }

    impl Sink for Heard {
        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            self.1.lock().unwrap().extend(tags(&[buf]));
            Ok(())
        }

        fn stream_event(&mut self, event: &StreamEvent) -> Result<()> {
            if let StreamEvent::Eos = event {
                self.1.lock().unwrap().push(None);
            }
            Ok(())
        }
    }

    /// Links a [`Heard`] to `stage`'s pad, and returns what it will have
    /// heard.
    fn hear<T: Transform>(stage: &mut TransformStage<T>) -> Arc<Mutex<Vec<Option<u8>>>> {
        let heard = Arc::new(Mutex::new(Vec::new()));
        stage.src_pads()[0].link(Box::new(Heard(
            element_pp_log(ElementType::Other, "heard", None),
            Arc::clone(&heard),
        )));
        heard
    }

    /// Hands each packet on one late — on the next one's arrival, or when
    /// drained — and fails to drain when asked to.
    struct Delay {
        pp_log: PpLog,
        held: Option<MediaBuffer>,
        resets: Arc<AtomicUsize>,
        drain_fails: bool,
    }

    impl Delay {
        fn new() -> (Self, Arc<AtomicUsize>) {
            let resets = Arc::new(AtomicUsize::new(0));
            (
                Self {
                    pp_log: element_pp_log(ElementType::Other, "delay", None),
                    held: None,
                    resets: Arc::clone(&resets),
                    drain_fails: false,
                },
                resets,
            )
        }
    }

    impl Element for Delay {
        fn name(&self) -> Arc<str> {
            "delay".into()
        }

        fn element_type(&self) -> ElementType {
            ElementType::Other
        }

        fn pp_log(&self) -> &PpLog {
            &self.pp_log
        }

        fn pp_log_mut(&mut self) -> &mut PpLog {
            &mut self.pp_log
        }
    }

    impl Transform for Delay {
        fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
            if let Some(held) = self.held.replace(buf) {
                out.push(held);
            }
            Ok(())
        }

        fn drain(&mut self, out: &mut Output) -> Result<()> {
            if let Some(held) = self.held.take() {
                out.push(held);
            }
            if self.drain_fails {
                return Err(ffmpeg::Error::InvalidData.into());
            }
            Ok(())
        }

        fn reset(&mut self) {
            self.held = None;
            self.resets.fetch_add(1, Ordering::SeqCst);
        }

        fn input_contract(&self) -> InputContract {
            InputContract::Fixed(PortContract::packet(MediaKind::VideoPacket))
        }

        fn output_contract(&self) -> OutputContract {
            OutputContract::Fixed(PortContract::packet(MediaKind::VideoPacket))
        }

        fn accepts_seek(&self) -> bool {
            false
        }
    }

    /// What it holds goes on before the end does, and the end once.
    #[test]
    fn what_a_transform_holds_goes_on_before_the_end() {
        let (delay, _) = Delay::new();
        let mut stage = TransformStage::new(delay);
        let heard = hear(&mut stage);

        stage.consume(packet(1)).expect("held");
        assert!(heard.lock().unwrap().is_empty(), "one late");
        stage.consume(packet(2)).expect("the first goes on");
        crate::stream::deliver(&mut stage, &StreamEvent::Eos).expect("drained and ended");

        assert_eq!(
            *heard.lock().unwrap(),
            [Some(1), Some(2), None],
            "the second on the drain, and the end after it"
        );
    }

    /// A seek's `Flush` — and a `Stop` — lets go of what it holds, which
    /// then never reaches what follows; nothing else resets it.
    #[test]
    fn a_flush_or_a_stop_lets_go_of_what_it_holds() {
        let (delay, resets) = Delay::new();
        let mut stage = TransformStage::new(delay);
        let heard = hear(&mut stage);

        stage.consume(packet(1)).expect("held");
        stage.control(&ControlMsg::Pause).expect("nothing to do");
        assert_eq!(resets.load(Ordering::SeqCst), 0, "a pause is not a reset");
        stage.control(&ControlMsg::Flush).expect("reset");
        assert_eq!(resets.load(Ordering::SeqCst), 1);
        crate::stream::deliver(&mut stage, &StreamEvent::Eos).expect("ended");
        assert_eq!(
            *heard.lock().unwrap(),
            [None],
            "what the flush let go of came back"
        );

        stage.control(&ControlMsg::Stop).expect("reset");
        assert_eq!(resets.load(Ordering::SeqCst), 2);
    }

    /// A drain that fails is answered with its error, and the stream still
    /// ends: what was drained before it and the end both go on.
    #[test]
    fn a_failed_drain_still_ends_the_stream() {
        let (mut delay, _) = Delay::new();
        delay.drain_fails = true;
        let mut stage = TransformStage::new(delay);
        let heard = hear(&mut stage);

        stage.consume(packet(1)).expect("held");
        assert!(
            crate::stream::deliver(&mut stage, &StreamEvent::Eos).is_err(),
            "the drain's error"
        );
        assert_eq!(*heard.lock().unwrap(), [Some(1), None]);
    }

    /// Makes two packets of each one: its own, and one tagged 100 more.
    struct Twice(PpLog);

    impl Element for Twice {
        fn name(&self) -> Arc<str> {
            "twice".into()
        }

        fn element_type(&self) -> ElementType {
            ElementType::Other
        }

        fn pp_log(&self) -> &PpLog {
            &self.0
        }

        fn pp_log_mut(&mut self) -> &mut PpLog {
            &mut self.0
        }
    }

    impl Transform for Twice {
        fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
            let [tag] = tags(std::slice::from_ref(&buf))[..] else {
                unreachable!()
            };
            out.push(buf);
            out.push(packet(tag.expect("a packet") + 100));
            Ok(())
        }
    }

    /// Refuses the packets tagged `refused`, and writes down the rest.
    struct Refusing {
        pp_log: PpLog,
        refused: u8,
        received: Arc<Mutex<Vec<MediaBuffer>>>,
    }

    impl Element for Refusing {
        fn name(&self) -> Arc<str> {
            "refusing".into()
        }

        fn element_type(&self) -> ElementType {
            ElementType::Other
        }

        fn pp_log(&self) -> &PpLog {
            &self.pp_log
        }

        fn pp_log_mut(&mut self) -> &mut PpLog {
            &mut self.pp_log
        }
    }

    impl Sink for Refusing {
        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            if tags(std::slice::from_ref(&buf)) == [Some(self.refused)] {
                return Err(ffmpeg::Error::InvalidData.into());
            }
            self.received.lock().unwrap().push(buf);
            Ok(())
        }
    }

    /// One buffer refused downstream does not keep back the ones made
    /// after it from the same input; the refusal is still answered.
    #[test]
    fn every_buffer_made_goes_on_even_after_one_is_refused() {
        let mut stage =
            TransformStage::new(Twice(element_pp_log(ElementType::Other, "twice", None)));
        let received = Arc::new(Mutex::new(Vec::new()));
        stage.src_pads()[0].link(Box::new(Refusing {
            pp_log: element_pp_log(ElementType::Other, "refusing", None),
            refused: 7,
            received: Arc::clone(&received),
        }));

        assert!(stage.consume(packet(7)).is_err(), "the refusal is answered");
        assert_eq!(tags(&received.lock().unwrap()), [Some(107)]);
    }

    /// Put where a filter is asked for, a transform is one filter with one
    /// pad, wired by what the transform says of itself.
    #[test]
    fn a_transform_is_wired_by_what_it_says() {
        let (delay, _) = Delay::new();
        let mut filter = delay.into_filter();
        assert_eq!(&*filter.name(), "delay");
        assert_eq!(filter.src_pads().len(), 1, "one pad");
        assert_eq!(filter.src_pads()[0].name(), "delay_src");
        assert_eq!(
            filter.src_pads()[0].contract(),
            OutputContract::Fixed(PortContract::packet(MediaKind::VideoPacket))
        );
        assert_eq!(
            filter.input_contract(),
            InputContract::Fixed(PortContract::packet(MediaKind::VideoPacket))
        );
        assert!(!filter.accepts_seek());

        let unsaid = Twice(element_pp_log(ElementType::Other, "twice", None)).into_filter();
        assert_eq!(unsaid.input_contract(), InputContract::Unknown);
        assert!(unsaid.accepts_seek(), "a seek is followed unless said");
    }

    /// Takes one buffer and is then not ready until reopened — a terminal
    /// that has its preroll sample.
    struct TakesOne {
        pp_log: PpLog,
        open: Arc<std::sync::atomic::AtomicBool>,
        received: Arc<Mutex<Vec<MediaBuffer>>>,
    }

    impl Element for TakesOne {
        fn name(&self) -> Arc<str> {
            "takes-one".into()
        }

        fn element_type(&self) -> ElementType {
            ElementType::Other
        }

        fn pp_log(&self) -> &PpLog {
            &self.pp_log
        }

        fn pp_log_mut(&mut self) -> &mut PpLog {
            &mut self.pp_log
        }
    }

    impl Sink for TakesOne {
        fn ready_consume(&mut self) -> bool {
            self.open.load(Ordering::SeqCst)
        }

        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            self.open.store(false, Ordering::SeqCst);
            self.received.lock().unwrap().push(buf);
            Ok(())
        }
    }

    /// In a preroll a terminal that has its sample takes nothing more: what
    /// a transform makes after it — the second of a pair here — is kept
    /// rather than pushed into it, and the stage answers not ready. Once
    /// playback goes on what was kept goes first, then what follows.
    #[test]
    fn in_a_preroll_what_the_terminal_will_not_take_is_kept() {
        let context = Arc::new(Context::for_test_with_clock(
            crate::bus::Bus::new().0,
            "test",
            crate::graph::PipelineGraph::new(),
            crate::graph::ElementId::for_test(1),
            Arc::new(crate::clock::Clock::new()),
        ));
        let mut stage =
            TransformStage::new(Twice(element_pp_log(ElementType::Other, "twice", None)));
        stage.attach_context(&context);
        let open = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let received = Arc::new(Mutex::new(Vec::new()));
        stage.src_pads()[0].link(Box::new(TakesOne {
            pp_log: element_pp_log(ElementType::Other, "takes-one", None),
            open: Arc::clone(&open),
            received: Arc::clone(&received),
        }));
        context.state.observe(&ControlMsg::Preroll(Arc::new(
            crate::control::PrerollContext::new([]),
        )));

        stage.consume(packet(1)).expect("the sample, and one kept");
        assert_eq!(tags(&received.lock().unwrap()), [Some(1)]);
        assert!(!stage.ready_consume(), "held while it keeps one");

        context.state.observe(&ControlMsg::Pause);
        context.state.observe(&ControlMsg::Resume);
        open.store(true, Ordering::SeqCst);
        assert!(stage.ready_consume(), "nothing holds it back now");
        stage.consume(packet(2)).expect("what was kept, then this");
        assert_eq!(
            tags(&received.lock().unwrap()),
            [Some(1), Some(101), Some(2), Some(102)]
        );
    }
}
