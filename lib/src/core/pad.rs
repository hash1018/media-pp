//! The one output port data leaves an element through.
//!
//! There is no separate fan-out primitive in this model: an element with more
//! than one [`SrcPad`] *is* a tee. [`FileDemuxer`](crate::elements::FileDemuxer)
//! is the plain form — one pad per container stream, wired once — and
//! [`Tee`](crate::elements::Tee) is the deliberate exception whose pads can
//! change while it runs.

use std::sync::Arc;

use crate::{
    buffer::MediaBuffer,
    contract::OutputContract,
    control::ControlMsg,
    element::{ElementType, Sink},
    error::Result,
    pp_log::{PpLog, pp_trace},
    stats::PadCounters,
    stream::{Item, StreamEvent},
};

/// An output port an [`Element`](crate::element::Element) owns. Data only
/// ever leaves an element through one of its src pads — there is no other
/// way to reach downstream.
///
/// This is what fan-out is, in this design: an element with more than one
/// src pad *is* a tee — there's no separate "Tee" primitive in the
/// pad/element model itself. [`crate::elements::FileDemuxer`] is the
/// plain form of it: one pad per container stream, chosen once at wiring
/// time via a normal `&mut [SrcPad]`. [`crate::elements::Tee`] is the one
/// deliberate exception to that plain shape — its pads live behind a lock
/// instead, so a `TeeHandle` can add or remove one from a different
/// thread than whatever is driving it; see that module for why.
pub struct SrcPad {
    name: String,
    contract: OutputContract,
    peer: Option<Box<dyn Sink>>,
    /// What has left through this pad — see [`crate::stats`]. Every pad has
    /// one, so an element's outputs are counted without it doing anything.
    counters: Arc<PadCounters>,
    /// Whether a seek's flush has passed and its segment not yet: what is
    /// pushed meanwhile is from the position the seek left, and is dropped
    /// — see [`crate::stream`].
    flushing: bool,
    /// Whether an `Eos` has gone and nothing since: a stream ends once.
    ended: bool,
}

impl SrcPad {
    /// Creates an unlinked output pad with the caller-selected diagnostic
    /// name, declaring nothing about what it emits.
    ///
    /// Use [`SrcPad::with_contract`] on a pad whose payload is already
    /// settled by the time its element is constructed; this plain
    /// constructor leaves the link check to defer to the runtime one, the
    /// same as before contracts existed.
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            counters: PadCounters::new(&name),
            name,
            contract: OutputContract::Unknown,
            peer: None,
            flushing: false,
            ended: false,
        }
    }

    /// Creates an unlinked output pad that declares what it emits — see
    /// [`crate::contract`].
    pub fn with_contract(name: impl Into<String>, contract: OutputContract) -> Self {
        Self {
            contract,
            ..Self::new(name)
        }
    }

    /// Returns the pad name used by topology and flow diagnostics.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns what this pad declares it emits.
    pub fn contract(&self) -> OutputContract {
        self.contract
    }

    /// Returns whether this pad currently owns a downstream sink connection.
    pub fn is_linked(&self) -> bool {
        self.peer.is_some()
    }

    /// The linked sink's own identity, without touching the link itself —
    /// lets a caller that just saw [`SrcPad::push`] or a control message
    /// fail (e.g. [`crate::elements::Tee`], fanning out to several pads at
    /// once) report *which* downstream element the failure actually came
    /// from, instead of only knowing its own. `None` for an unlinked pad.
    pub fn peer_identity(&self) -> Option<(ElementType, Arc<str>)> {
        self.peer
            .as_ref()
            .map(|sink| (sink.element_type(), sink.name()))
    }

    /// Runtime half of a connection. Pipeline users connect through
    /// [`crate::element::Context::attach`], which keeps the graph and this
    /// peer in sync. Kept crate-visible for element-level unit tests.
    pub(crate) fn link(&mut self, sink: Box<dyn Sink>) {
        self.peer = Some(sink);
    }

    /// This pad's counters, for the element that owns it to be reported
    /// with — see [`crate::stats`].
    pub(crate) fn counters(&self) -> Arc<PadCounters> {
        Arc::clone(&self.counters)
    }

    /// Pushes a buffer to whatever this pad is linked to. Pushing into an
    /// unlinked pad silently drops the buffer (e.g. a demuxer stream
    /// nobody cared to link), and so does pushing between a seek's `Flush`
    /// and the start of the stream the seek begins: what is pushed then is
    /// from the position the seek left.
    pub fn push(&mut self, buf: MediaBuffer) -> Result<()> {
        if self.flushing {
            return Ok(());
        }
        let bytes = match &buf {
            MediaBuffer::Packet(packet) => packet.size() as u64,
            _ => 0,
        };
        let result = match &mut self.peer {
            Some(sink) => sink.consume(buf),
            None => Ok(()),
        };
        self.ended = false;
        self.counters.pushed(result.is_ok(), bytes);
        result
    }

    /// Returns whether the linked peer can currently accept its next buffer.
    /// An unlinked pad is ready because pushing to it is a no-op.
    pub fn ready_consume(&mut self) -> bool {
        self.peer
            .as_mut()
            .map(|sink| sink.ready_consume())
            .unwrap_or(true)
    }

    /// Sends a source-originated EOS with explicit pad-level trace records.
    /// Filters are traced by the pipeline's common element wrapper; this is
    /// for the source boundary where EOS first enters the dataflow graph.
    pub(crate) fn push_eos(&mut self, pp_log: &PpLog) -> Result<()> {
        if self.peer.is_none() {
            pp_trace!(
                pp_log: pp_log,
                "event=eos phase=skipped pad={} reason=unlinked",
                self.name
            );
            return Ok(());
        }

        pp_trace!(
            pp_log: pp_log,
            "event=eos phase=sending pad={}",
            self.name
        );
        let result = self.push_event(&StreamEvent::Eos);
        match &result {
            Ok(()) => pp_trace!(
                pp_log: pp_log,
                "event=eos phase=sent pad={} outcome=ok",
                self.name
            ),
            Err(error) => pp_trace!(
                pp_log: pp_log,
                "event=eos phase=sent pad={} outcome=error error={error}",
                self.name
            ),
        }
        result
    }

    /// Hands `event` to whatever this pad is linked to, where it goes in
    /// order with what was pushed before it — see [`crate::stream`]. Not
    /// counted: it is not data. An unlinked pad drops it, as `push` does.
    ///
    /// A stream ends once: an `Eos` handed on after one that went, with
    /// nothing in between, is dropped. Whatever sent it twice — a source
    /// asked to finish while it waited, paused, at the end it had already
    /// handed on — would have a muxer finish its file twice.
    pub(crate) fn push_event(&mut self, event: &StreamEvent) -> Result<()> {
        match event {
            StreamEvent::Segment(segment) if segment.flushed => {
                self.flushing = false;
                self.ended = false;
            }
            // From the position the seek left, as a buffer would be.
            _ if self.flushing => return Ok(()),
            // A new stream, which can end in its turn.
            StreamEvent::Segment(_) => self.ended = false,
            StreamEvent::Eos if self.ended => return Ok(()),
            StreamEvent::Eos => {}
        }
        let result = match &mut self.peer {
            Some(sink) => sink.stream_event(event),
            None => Ok(()),
        };
        // Ended only once it has gone: one refused can be sent again.
        if matches!(event, StreamEvent::Eos) {
            self.ended = result.is_ok();
        }
        result
    }

    /// Pushes a buffer or hands on an event, whichever `item` is — for an
    /// element that keeps both, in order, to hand on later.
    pub(crate) fn push_item(&mut self, item: Item) -> Result<()> {
        match item {
            Item::Buffer(buf) => self.push(buf),
            Item::Event(event) => self.push_event(&event),
        }
    }

    /// Forwards a [`ControlMsg`] to whatever this pad is linked to —
    /// mirrors [`SrcPad::push`], just for control instead of data.
    /// Pushing into an unlinked pad is a no-op, same as `push`.
    ///
    /// Not public: an element never passes control on itself — see
    /// [`crate::element::Sink::control`] — and one that did would hand
    /// everything after it each message twice.
    pub(crate) fn control(&mut self, msg: &ControlMsg) -> Result<()> {
        // A seek's own, and nobody else's — see `crate::stream`.
        if matches!(msg, ControlMsg::Flush) && crate::control::reached_directly() {
            self.flushing = true;
        }
        match &mut self.peer {
            Some(sink) => sink.control(msg),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::time::Duration;

    use super::*;
    use crate::element::{Element, element_pp_log};
    use crate::stream::Segment;

    /// Writes down what reaches it, a word apiece.
    struct Heard {
        pp_log: PpLog,
        heard: Arc<Mutex<Vec<&'static str>>>,
    }

    impl Element for Heard {
        fn name(&self) -> Arc<str> {
            "heard".into()
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

    impl Sink for Heard {
        fn consume(&mut self, _buf: MediaBuffer) -> Result<()> {
            self.heard.lock().unwrap().push("data");
            Ok(())
        }
        fn stream_event(&mut self, event: &StreamEvent) -> Result<()> {
            let word = match event {
                StreamEvent::Segment(_) => "segment",
                StreamEvent::Eos => "eos",
            };
            self.heard.lock().unwrap().push(word);
            Ok(())
        }
    }

    fn segment() -> StreamEvent {
        StreamEvent::Segment(Arc::new(Segment {
            id: 1,
            flushed: false,
            position: Duration::ZERO,
            start: Duration::ZERO,
            show_from: None,
            backwards: false,
        }))
    }

    fn packet() -> MediaBuffer {
        MediaBuffer::Packet(Arc::new(ffmpeg_next::Packet::empty()))
    }

    /// A stream ends once: an `Eos` after one that went, with nothing in
    /// between, goes no further — and once a buffer or a new segment has
    /// followed, the next one does.
    #[test]
    fn a_stream_ends_once() {
        let heard = Arc::new(Mutex::new(Vec::new()));
        let mut pad = SrcPad::new("src");
        pad.link(Box::new(Heard {
            pp_log: element_pp_log(ElementType::Other, "heard", None),
            heard: Arc::clone(&heard),
        }));
        pad.push_event(&StreamEvent::Eos).unwrap();
        pad.push_event(&StreamEvent::Eos).unwrap();
        pad.push(packet()).unwrap();
        pad.push_event(&StreamEvent::Eos).unwrap();
        pad.push_event(&segment()).unwrap();
        pad.push_event(&StreamEvent::Eos).unwrap();
        pad.push_event(&StreamEvent::Eos).unwrap();
        assert_eq!(
            *heard.lock().unwrap(),
            ["eos", "data", "eos", "segment", "eos"]
        );
    }
}
