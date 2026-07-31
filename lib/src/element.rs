use crate::{
    buffer::MediaBuffer,
    control::{ControlMsg, ControlReceiver},
    error::Result,
    pad::SrcPad,
};

/// A node in the pipeline graph with a name. Plain identity only — says
/// nothing about whether the node has an input, an output, both, or
/// neither.
pub trait Element: Send {
    fn name(&self) -> &str;
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
    fn run(&mut self, control: &ControlReceiver) -> Result<()>;
}

/// An element with both an input and an output — decoder, encoder,
/// filter, thumbnail extractor, ... Just a name for "has a `Sink` to
/// receive and a `Source` to push what it produces into"; nothing new to
/// implement beyond those two.
pub trait Filter: Source + Sink {}

impl<T: Source + Sink> Filter for T {}
