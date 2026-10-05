//! Writing a filter by its media work alone — see [`Filter`].
//!
//! A filter written directly, as [`RawSink`] and [`SrcPads`], owns everything
//! the pipeline asks of one: its pad, handing its end on after what it still
//! holds, dropping what a seek left behind, saying whether it can take a
//! buffer. Each did that for itself, and each a little differently —
//! docs/stream-events.md lists what that cost. A [`Filter`] does the
//! media work and nothing else; the framework does the rest, once, the same
//! way for every one.

use std::sync::Arc;

use crate::{
    buffer::{MediaBuffer, Metadata},
    contract::{InputContract, OutputContract},
    control::ControlMsg,
    element::{Context, Element, ElementType, Flow, RawFilter, RawSink, SrcPads},
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
/// a `Filter` as it takes a [`RawFilter`]; where a list of filters is asked
/// for — [`Rack`](crate::elements::Rack) — [`IntoFilter::into_filter`] makes
/// one. Implement this or [`RawSink`] and [`SrcPads`], not both: a filter that
/// routes the stream itself — splits it, holds it back, waits on a clock —
/// is written the direct way.
///
/// # Metadata
///
/// What a buffer carries — its [`Metadata`](crate::buffer::Metadata) — goes
/// on with what the filter makes of it: a buffer pushed while `transform`
/// handles it that is of the same sort, at the same timestamp, and carries
/// nothing of its own is given the input's. A buffer with no timestamp
/// cannot be told apart from one held from earlier, so nothing is carried
/// from or onto one. That is a picture scaled,
/// converted, keyed, uploaded or downloaded.
///
/// Nothing is carried onto another sort — an encoder's packet from a
/// picture — nor onto a buffer at another timestamp, which is what a filter
/// handing on later what it held from earlier pushes: the input in hand is
/// not what that output was made from, and metadata on the wrong picture
/// is worse than none. A filter that stamps its outputs anew and knows
/// which input each came from carries it itself, as
/// [`FrameRateLimiter`](crate::elements::FrameRateLimiter) does; one that
/// sets its own on a buffer it pushes keeps that.
pub trait Filter: Element {
    /// Makes what `buf` answers to, into `out` — nothing, one buffer, or
    /// several. Never handed the end of the stream: that is `drain`'s.
    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()>;

    /// What it still holds, into `out`, before the end of the stream goes
    /// on — a codec's delayed output. Nothing by default.
    fn drain(&mut self, _out: &mut Output) -> Result<()> {
        Ok(())
    }

    /// Lets go of what belongs to the timeline a seek has left. A stop lets
    /// go of it too — see [`Self::stopping`]. Nothing by default.
    fn reset(&mut self) {}

    /// Lets go of what the run as a whole kept, beside what
    /// [`Self::reset`] does: a stop abandons the run, and a pipeline run
    /// again begins its output anew, where a seek only moves within it — a
    /// timeline taken from the first buffer, a count of what went out. What
    /// `reset` does by default.
    fn stopping(&mut self) {
        self.reset();
    }

    /// What it takes — see [`RawSink::input_contract`]. Nothing said by default.
    fn input_contract(&self) -> InputContract {
        InputContract::Unknown
    }

    /// What it hands on — see [`SrcPad::with_contract`]. Nothing said by
    /// default.
    fn output_contract(&self) -> OutputContract {
        OutputContract::Unknown
    }

    /// Whether it can follow a seek — see [`RawSink::accepts_seek`]. Yes by
    /// default.
    fn accepts_seek(&self) -> bool {
        true
    }
}

/// Where a [`Filter`] puts what it makes, in the order it is to go on.
#[derive(Default)]
pub struct Output {
    made: Vec<MediaBuffer>,
}

impl Output {
    /// Hands `buf` on, after whatever was put here before it.
    pub fn push(&mut self, buf: MediaBuffer) {
        self.made.push(buf);
    }

    /// Gives what `input` carries to every buffer made from it — of its
    /// sort, at its timestamp — that carries none of its own: see
    /// [`Filter`]'s section on metadata.
    fn carry(&mut self, input: &Carried) {
        for buf in &mut self.made {
            if std::mem::discriminant(buf) == input.sort
                && input.pts.is_some()
                && pts_of(buf) == input.pts
                && buf.metadata().is_none()
            {
                buf.set_metadata(Some(Arc::clone(&input.metadata)));
            }
        }
    }
}

/// What a buffer handed to a filter carried, and what an output must share
/// with it to be given that.
struct Carried {
    sort: std::mem::Discriminant<MediaBuffer>,
    pts: Option<i64>,
    metadata: Arc<Metadata>,
}

impl Carried {
    /// `None` where `buf` carries nothing.
    fn of(buf: &MediaBuffer) -> Option<Self> {
        Some(Self {
            sort: std::mem::discriminant(buf),
            pts: pts_of(buf),
            metadata: Arc::clone(buf.metadata_arc()?),
        })
    }
}

/// A buffer's presentation timestamp.
fn pts_of(buf: &MediaBuffer) -> Option<i64> {
    match buf {
        MediaBuffer::Packet(packet) => packet.pts(),
        MediaBuffer::Video(frame) => frame.pts(),
        MediaBuffer::Audio(frame) => frame.pts(),
    }
}

/// What a [`Filter`] is in a pipeline: the filter the framework makes of
/// it, with the pad and the rules it does not keep itself.
pub(crate) struct FilterStage<T> {
    inner: T,
    pad: SrcPad,
    /// What it made that its pad could not take while a preroll held the
    /// graph — see [`OutputStash`].
    stash: OutputStash,
}

impl<T: Filter> FilterStage<T> {
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

impl<T: Filter> Element for FilterStage<T> {
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

impl<T: Filter> SrcPads for FilterStage<T> {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl<T: Filter> RawSink for FilterStage<T> {
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
        let carried = Carried::of(&buf);
        self.inner.transform(buf, &mut out)?;
        if let Some(carried) = &carried {
            out.carry(carried);
        }
        self.hand_on(out)
    }

    fn flow(&mut self, Flow(msg): Flow<'_>) -> Result<()> {
        match msg {
            ControlMsg::Flush => self.inner.reset(),
            ControlMsg::Stop => self.inner.stopping(),
            ControlMsg::Pause
            | ControlMsg::Resume
            | ControlMsg::Preroll(_)
            | ControlMsg::Seek(_) => {
                return Ok(());
            }
        }
        self.stash.clear();
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

/// What can be put where a filter is asked for: a [`Filter`], which the
/// framework makes one of, a [`RawFilter`] itself, or a [`BoxFilter`]
/// holding either.
///
/// `M` says which a type is, and the compiler works it out: nothing names
/// it. A type that is both a `Filter` and a `RawFilter` is refused as
/// ambiguous — implement one or the other.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is neither a `RawFilter` nor a `Filter`",
    note = "implement `Filter` for an element that makes buffers of buffers, or `RawSink` and `SrcPads` for one that routes the stream itself"
)]
pub trait IntoFilter<M>: sealed::Sealed<M> {
    /// The filter this is, whichever kind it is.
    fn into_filter(self) -> BoxFilter;
}

/// Says a type goes in as the [`RawFilter`] it is — see [`IntoFilter`].
pub enum AsRawFilter {}

/// Says a type goes in as the [`Filter`] it is — see [`IntoFilter`].
pub enum AsFilter {}

/// Says a [`BoxFilter`] goes in as the filter it holds — see
/// [`IntoFilter`].
pub enum AsBoxFilter {}

impl<F: RawFilter + 'static> IntoFilter<AsRawFilter> for F {
    fn into_filter(self) -> BoxFilter {
        BoxFilter(Box::new(self))
    }
}

impl<T: Filter + 'static> IntoFilter<AsFilter> for T {
    fn into_filter(self) -> BoxFilter {
        BoxFilter(Box::new(FilterStage::new(self)))
    }
}

impl IntoFilter<AsBoxFilter> for BoxFilter {
    fn into_filter(self) -> BoxFilter {
        self
    }
}

mod sealed {
    use super::{AsBoxFilter, AsFilter, AsRawFilter, BoxFilter, Filter, RawFilter};

    /// Keeps [`super::IntoFilter`] to the three ways in it has.
    pub trait Sealed<M> {}

    impl<F: RawFilter + 'static> Sealed<AsRawFilter> for F {}
    impl<T: Filter + 'static> Sealed<AsFilter> for T {}
    impl Sealed<AsBoxFilter> for BoxFilter {}
}

/// A filter of any kind, which kind forgotten: what to hold where the
/// filter is picked as the program runs — an upload onto whichever device
/// there is — and what a [`Rack`](crate::elements::Rack) is filled with.
/// [`ChainBuilder::pipe`](crate::pipeline::ChainBuilder::pipe) takes one as
/// it takes the filter it holds.
///
/// Made with [`BoxFilter::new`] from a [`Filter`] or a [`RawFilter`] alike.
/// It derefs to the filter inside, for what is asked of one directly.
pub struct BoxFilter(Box<dyn RawFilter>);

impl BoxFilter {
    /// `filter`, whichever kind it is.
    pub fn new<M>(filter: impl IntoFilter<M>) -> Self {
        filter.into_filter()
    }

    /// The filter inside, as the framework drives it.
    pub(crate) fn into_raw(self) -> Box<dyn RawFilter> {
        self.0
    }
}

impl std::ops::Deref for BoxFilter {
    type Target = dyn RawFilter;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl std::ops::DerefMut for BoxFilter {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.0
    }
}

impl std::fmt::Debug for BoxFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("BoxFilter").field(&self.0.name()).finish()
    }
}

/// Makes `$name`, a newtype over `FilterStage<_>`, the filter its stage
/// is — every method the stage's. For an element of this crate that keeps
/// the public name, constructors and `RawFilter` it always had while its work
/// moves into a [`Filter`].
macro_rules! filter_stage {
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

        impl $crate::element::SrcPads for $name {
            fn src_pads(&mut self) -> &mut [$crate::pad::SrcPad] {
                self.0.src_pads()
            }
        }

        impl $crate::element::RawSink for $name {
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
            fn flow(&mut self, flow: $crate::element::Flow<'_>) -> $crate::error::Result<()> {
                self.0.flow(flow)
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

pub(crate) use filter_stage;

#[cfg(test)]
mod tests {
    use std::sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    };

    use ffmpeg_next as ffmpeg;

    use super::*;
    use crate::element::RawSinkExt;
    use crate::{
        contract::{MediaKind, PortContract},
        element::element_pp_log,
    };

    fn packet(tag: u8) -> MediaBuffer {
        MediaBuffer::Packet(Arc::new(ffmpeg::Packet::copy(&[tag])).into())
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

    impl RawSink for Heard {
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
    fn hear<T: Filter>(stage: &mut FilterStage<T>) -> Arc<Mutex<Vec<Option<u8>>>> {
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

    impl Filter for Delay {
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
        let mut stage = FilterStage::new(delay);
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
        let mut stage = FilterStage::new(delay);
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

    /// Counts what it is told: resets and stops.
    struct Timeline {
        pp_log: PpLog,
        told: Arc<Mutex<Vec<&'static str>>>,
    }

    impl Element for Timeline {
        fn name(&self) -> Arc<str> {
            "timeline".into()
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

    impl Filter for Timeline {
        fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
            out.push(buf);
            Ok(())
        }

        fn reset(&mut self) {
            self.told.lock().unwrap().push("reset");
        }

        fn stopping(&mut self) {
            self.told.lock().unwrap().push("stopping");
        }
    }

    /// A seek's `Flush` goes to `reset` and a `Stop` to `stopping` alone —
    /// for a transform whose output timeline a stop begins again and a seek
    /// does not, as `TimestampOrigin`'s and `FrameRateLimiter`'s do.
    #[test]
    fn a_flush_resets_and_a_stop_stops() {
        let told = Arc::new(Mutex::new(Vec::new()));
        let mut stage = FilterStage::new(Timeline {
            pp_log: element_pp_log(ElementType::Other, "timeline", None),
            told: Arc::clone(&told),
        });
        stage.control(&ControlMsg::Pause).expect("nothing to do");
        stage.control(&ControlMsg::Flush).expect("reset");
        stage.control(&ControlMsg::Stop).expect("stopped");
        assert_eq!(*told.lock().unwrap(), ["reset", "stopping"]);
    }

    /// A drain that fails is answered with its error, and the stream still
    /// ends: what was drained before it and the end both go on.
    #[test]
    fn a_failed_drain_still_ends_the_stream() {
        let (mut delay, _) = Delay::new();
        delay.drain_fails = true;
        let mut stage = FilterStage::new(delay);
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

    impl Filter for Twice {
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

    impl RawSink for Refusing {
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
        let mut stage = FilterStage::new(Twice(element_pp_log(ElementType::Other, "twice", None)));
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

    impl RawSink for TakesOne {
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
        let mut stage = FilterStage::new(Twice(element_pp_log(ElementType::Other, "twice", None)));
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

    /// Keeps every buffer it takes, to be looked at whole.
    struct Kept(PpLog, Arc<Mutex<Vec<MediaBuffer>>>);

    impl Element for Kept {
        fn name(&self) -> Arc<str> {
            "kept".into()
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

    impl RawSink for Kept {
        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            self.1.lock().unwrap().push(buf);
            Ok(())
        }
    }

    fn keep<T: Filter>(stage: &mut FilterStage<T>) -> Arc<Mutex<Vec<MediaBuffer>>> {
        let kept = Arc::new(Mutex::new(Vec::new()));
        stage.src_pads()[0].link(Box::new(Kept(
            element_pp_log(ElementType::Other, "kept", None),
            Arc::clone(&kept),
        )));
        kept
    }

    /// A packet tagged `tag`, at `pts`.
    fn packet_at(tag: u8, pts: i64) -> MediaBuffer {
        let mut packet = ffmpeg::Packet::copy(&[tag]);
        packet.set_pts(Some(pts));
        MediaBuffer::packet(packet)
    }

    /// What a test puts on a buffer.
    #[derive(Debug, PartialEq)]
    struct Seen(&'static str);

    fn seen(buf: &MediaBuffer) -> Option<&'static str> {
        buf.metadata()?.get::<Seen>().map(|seen| seen.0)
    }

    fn marked(buf: MediaBuffer, mark: &'static str) -> MediaBuffer {
        buf.with_metadata(Metadata::new().with(Seen(mark)))
    }

    /// Makes a packet of its own for each it takes: the same bytes, at the
    /// same timestamp, or `shift` later — or a picture, where `to_picture`.
    struct Remake {
        pp_log: PpLog,
        shift: i64,
        to_picture: bool,
        own: Option<&'static str>,
    }

    impl Remake {
        fn new() -> Self {
            Self {
                pp_log: element_pp_log(ElementType::Other, "remake", None),
                shift: 0,
                to_picture: false,
                own: None,
            }
        }
    }

    impl Element for Remake {
        fn name(&self) -> Arc<str> {
            "remake".into()
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

    impl Filter for Remake {
        fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
            let MediaBuffer::Packet(packet) = &buf else {
                unreachable!("only packets are handed in here");
            };
            let pts = packet.pts().map(|pts| pts + self.shift);
            let mut made = if self.to_picture {
                let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::GRAY8, 2, 2);
                frame.set_pts(pts);
                MediaBuffer::video(frame)
            } else {
                let mut remade = ffmpeg::Packet::copy(packet.data().expect("a payload"));
                remade.set_pts(pts);
                MediaBuffer::packet(remade)
            };
            if let Some(own) = self.own {
                made = marked(made, own);
            }
            out.push(made);
            Ok(())
        }
    }

    #[test]
    fn what_a_buffer_carries_goes_on_to_what_is_made_from_it() {
        let mut stage = FilterStage::new(Remake::new());
        let kept = keep(&mut stage);
        stage.consume(marked(packet_at(1, 10), "found")).unwrap();
        stage.consume(packet_at(2, 11)).unwrap();
        let kept = kept.lock().unwrap();
        assert_eq!(seen(&kept[0]), Some("found"));
        assert_eq!(seen(&kept[1]), None, "a buffer that carried nothing");
    }

    #[test]
    fn nothing_is_carried_onto_another_timestamp_or_another_sort() {
        let mut later = Remake::new();
        later.shift = 1;
        let mut stage = FilterStage::new(later);
        let kept = keep(&mut stage);
        stage.consume(marked(packet_at(1, 10), "found")).unwrap();
        assert_eq!(seen(&kept.lock().unwrap()[0]), None, "another timestamp");

        let mut picture = Remake::new();
        picture.to_picture = true;
        let mut stage = FilterStage::new(picture);
        let kept = keep(&mut stage);
        stage.consume(marked(packet_at(1, 10), "found")).unwrap();
        assert_eq!(seen(&kept.lock().unwrap()[0]), None, "another sort");
    }

    /// The case the timestamp is there for: a filter handing on a buffer it
    /// held from earlier must not be given what the one in hand carries.
    #[test]
    fn a_buffer_held_from_earlier_is_not_given_what_the_next_carries() {
        let (delay, _) = Delay::new();
        let mut stage = FilterStage::new(delay);
        let kept = keep(&mut stage);
        stage.consume(packet_at(1, 10)).unwrap();
        stage.consume(marked(packet_at(2, 11), "found")).unwrap();
        stage.consume(packet_at(3, 12)).unwrap();
        let kept = kept.lock().unwrap();
        assert_eq!(seen(&kept[0]), None, "the first carried nothing");
        assert_eq!(seen(&kept[1]), Some("found"), "the second its own");
    }

    #[test]
    fn what_a_filter_puts_on_a_buffer_itself_is_kept() {
        let mut own = Remake::new();
        own.own = Some("own");
        let mut stage = FilterStage::new(own);
        let kept = keep(&mut stage);
        stage.consume(marked(packet_at(1, 10), "found")).unwrap();
        assert_eq!(seen(&kept.lock().unwrap()[0]), Some("own"));
    }
}
