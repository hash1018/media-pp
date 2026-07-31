use std::time::Duration;

use crate::{
    buffer::MediaBuffer,
    bus::Bus,
    control::{ControlMsg, ControlReceiver},
    error::Result,
    pad::SrcPad,
};

/// Which kind of element posted a [`crate::bus::BusEvent`] — cheap to
/// compare/match, unlike the accompanying `name: String` (an
/// instance-level identifier chosen by whoever constructed it, needed
/// alongside this to tell apart e.g. two `Queue`s in the same pipeline;
/// see [`Element::element_type`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementType {
    FileDemuxer,
    SwDecoder,
    D3d12vaDecoder,
    Pacer,
    Tee,
    Queue,
    FrameCounter,
    PacketCounter,
    Dx12Renderer,
    RtspServer,
    /// Anything outside this crate's own elements — a test double, or a
    /// custom `Sink`/`SourceElement` implemented downstream of this
    /// crate. Keeps this enum from needing to grow every time someone
    /// adds their own element.
    Other,
}

/// A node in the pipeline graph with a name. Plain identity only — says
/// nothing about whether the node has an input, an output, both, or
/// neither.
pub trait Element: Send {
    fn name(&self) -> &str;

    /// See [`ElementType`].
    fn element_type(&self) -> ElementType;
}

/// Anything that can receive a buffer pushed from upstream — the input
/// side of an element, or a plain terminal sink. Every `Sink` is named
/// (via `Element`) so bus events (e.g. EOS) can identify which one they
/// came from.
///
/// This is the only "connection" primitive in the pipeline. By default,
/// consuming a buffer is a plain function call on the caller's thread —
/// zero overhead. Thread boundaries are introduced explicitly by wrapping
/// a `Sink` in a [`crate::queue::Queue`], not by elements spawning their
/// own threads.
pub trait Sink: Element {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()>;

    /// Reacts to a [`ControlMsg`] (pause/resume/stop) and, for anything
    /// with a downstream of its own, forwards it on — same shape as
    /// `consume`, just a separate channel from `MediaBuffer` so it can
    /// reach every element (not just ones that already know how to
    /// interpret a data buffer) and, at a [`crate::queue::Queue`], jump
    /// ahead of whatever data is backed up instead of waiting behind it.
    /// No default: every `Sink` has to consciously decide what this means
    /// for it, rather than silently dropping it.
    fn control(&mut self, msg: ControlMsg) -> Result<()>;
}

/// An element with one or more output ports. It sends data downstream by
/// pushing into its own `src_pads()` (e.g. `self.src_pads()[0].push(buf)`)
/// — it's never handed a `downstream` argument from the outside. See
/// [`SrcPad`].
///
/// `Source` and `Sink` are the two halves of the duality: `Sink` is "has
/// an input", `Source` is "has an output". An element that both receives
/// and produces (a decoder, say) implements both side by side — `Sink` to
/// receive, `Source` to push whatever it produces into its own pad(s)
/// from inside `consume`. There's no separate "processing element" trait
/// or wrapper needed for that.
pub trait Source: Element {
    fn src_pads(&mut self) -> &mut [SrcPad];
}

/// A pure source: has output but no input. Drives its own thread (or the
/// caller's, if run directly) and pushes buffers into its own src pad(s)
/// until EOS or an error. Typically wraps a blocking I/O read (demuxer,
/// file/network source).
pub trait SourceElement: Source {
    /// Drives this source until `Eos` (normal completion) or `Stop` (see
    /// [`ControlMsg::Stop`]) — call [`crate::control::drain_control`]
    /// once per loop iteration to make `control` responsive between
    /// blocking reads.
    ///
    /// `bus` is this source's own way to report a failure pushing into
    /// one of its pads *without* treating it as fatal — post a
    /// [`crate::bus::BusEvent::Error`] and keep going (drop that one
    /// buffer), the same way a [`crate::queue::Queue`] handles a failing
    /// downstream `Sink` — rather than returning `Err` and ending this
    /// source's thread over one bad buffer. A returned `Err` is still
    /// how genuinely fatal failures (this source can't continue at all)
    /// reach [`crate::pipeline::Pipeline::run`], which posts it to `bus`
    /// itself.
    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()>;

    /// Repositions this source to `target`, an absolute position from the
    /// start of the media (e.g. `av_seek_frame` for
    /// [`crate::elements::FileDemuxer`]). Called by
    /// [`crate::control::drain_control`] as part of handling
    /// [`ControlMsg::Seek`], *before* that message is forwarded to the
    /// source's own pads — so whatever's read next comes from the new
    /// position by the time downstream elements are told to flush for it.
    ///
    /// Returns where this actually landed, which is allowed to differ
    /// from `target` — a container seek can only ever reposition to a
    /// keyframe at or before it (landing mid-GOP would leave downstream
    /// decoders/muxers with no reference frame to start from), so
    /// `target` is a request, not a guarantee. `drain_control` reports
    /// the gap between the two via [`crate::bus::BusEvent::Seeked`];
    /// callers that need to know where playback actually resumed should
    /// watch that instead of assuming `target` took effect verbatim.
    fn seek(&mut self, target: Duration) -> Result<Duration>;
}

/// An element with both an input and an output — decoder, encoder,
/// filter, thumbnail extractor, ... Just a name for "has a `Sink` to
/// receive and a `Source` to push what it produces into"; nothing new to
/// implement beyond those two.
pub trait Filter: Source + Sink {}

impl<T: Source + Sink> Filter for T {}
