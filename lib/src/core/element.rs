use std::{sync::Arc, time::Duration};

use crate::pp_log::PpLog;

use crate::{
    buffer::MediaBuffer,
    bus::Bus,
    clock::Clock,
    control::{ControlMsg, ControlReceiver},
    error::Result,
    graph::{ElementId, PipelineGraph},
    pad::SrcPad,
    playback_clock::PlaybackClock,
};

/// Which kind of element posted a [`crate::bus::BusEvent`] — cheap to
/// compare/match, unlike the accompanying `name: Arc<str>` (an
/// instance-level identifier chosen by whoever constructed it, needed
/// alongside this to tell apart e.g. two `Queue`s in the same pipeline;
/// see [`Element::element_type`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementType {
    FileDemuxer,
    AppSource,
    RtspSource,
    TestVideoSource,
    TestAudioSource,
    DxgiCaptureSource,
    PipeWireAudioCaptureSource,
    PipeWireScreenCaptureSource,
    WasapiCaptureSource,
    AudioMixer,
    SwVideoCompositor,
    CudaVideoCompositor,
    D3d11VideoCompositor,
    WebRtcPeer,
    SwDecoder,
    CudaDecoder,
    CudaUpload,
    CudaDownload,
    CudaConverter,
    D3d12vaDecoder,
    D3d12Upload,
    D3d11Decoder,
    D3d11Upload,
    D3d11Download,
    SwEncoder,
    CudaEncoder,
    D3d11NvencEncoder,
    SwAudioEncoder,
    AudioResampler,
    AudioVolume,
    Pacer,
    VideoSynchronizer,
    SwScaler,
    CudaScaler,
    D3d11Scaler,
    Tee,
    Queue,
    FrameCounter,
    PacketCounter,
    CudaRenderer,
    D3d12Renderer,
    D3d11Renderer,
    PipeWireAudioRenderer,
    WasapiRenderer,
    RtspSink,
    AppSink,
    OrtDetector,
    HlsMuxer,
    Mp4Muxer,
    SegmentedMp4Muxer,
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
    /// Returns a cheap clone (refcount bump, not a deep copy) of this
    /// element's name — [`crate::bus::BusEvent`] stores names as
    /// `Arc<str>` for exactly this reason: a hot path like
    /// [`crate::queue::Queue`] posting `BusEvent::Dropped` once per
    /// overflowed buffer shouldn't pay for a fresh heap allocation every
    /// time it wants to report which element it is.
    fn name(&self) -> Arc<str>;

    /// See [`ElementType`].
    fn element_type(&self) -> ElementType;

    /// A pre-reserved graph identity for elements that expose dynamic
    /// attachment handles. Most elements receive an ID from
    /// `ChainBuilder` and keep the default `None` implementation.
    fn graph_id(&self) -> Option<ElementId> {
        None
    }

    /// This element's identity for [`crate::bus::Bus::post`] — same
    /// `id`/`name` as [`Element::name`], just already wrapped as the
    /// [`crate::pp_log::PpLog`] its `pp_info!`/`pp_warn!`/`pp_error!` macros need. A
    /// stored private field, not built fresh per call, for the same reason
    /// `name()` returns a cheap `Arc<str>` clone instead of a fresh `String`
    /// — see its own docs.
    fn pp_log(&self) -> &PpLog;

    /// Mutable access to the same field [`Element::pp_log`] reads — used by
    /// [`crate::pipeline::ChainBuilder`] to stamp the owning
    /// [`crate::pipeline::Pipeline`]'s id onto every element that
    /// passes through it, via [`element_pp_log`]. Not meant to be called
    /// from anywhere else.
    fn pp_log_mut(&mut self) -> &mut PpLog;
}

/// Builds the [`PpLog`] every element constructs for its own [`Element::pp_log`]
/// field, and that [`crate::pipeline::ChainBuilder`]/[`crate::pipeline::Pipeline`]
/// rebuild once they know which pipeline an element belongs to. Keeps the
/// element type, instance name, and pipeline id as separate fields, so a log
/// reader does not need to parse a combined display string. The pipeline id is
/// `None` for an element that isn't wired into a `Pipeline` at all (e.g. most
/// of this crate's own tests). Public so a custom `Element`
/// implemented outside this crate (see [`ElementType::Other`]) can build
/// its own `pp_log` field the same way.
pub fn element_pp_log(element_type: ElementType, name: &str, pipeline_id: Option<&str>) -> PpLog {
    PpLog::new(&format!("{element_type:?}"), name, pipeline_id)
}

/// Builds the [`PpLog`] used for records a [`crate::pipeline::Pipeline`]
/// emits about itself rather than about any one element — `run` and the
/// `topology` diagram. A pipeline is not a graph node and so has no
/// [`ElementType`]; its instance name is its own id. Kept here next to
/// [`element_pp_log`] so the literal element name appears exactly once.
pub(crate) fn pipeline_pp_log(pipeline_id: &str) -> PpLog {
    PpLog::new("Pipeline", pipeline_id, Some(pipeline_id))
}

/// Everything a [`crate::pipeline::ChainBuilder`]/[`crate::elements::Tee`]
/// needs to wire itself into a [`crate::pipeline::Pipeline`] — bundled into
/// one `Arc` instead of threading `bus`/`pipeline_id`/`graph`/the wall and
/// playback clocks through separately. Built once per source by
/// [`crate::pipeline::PipelineBuilder::add_source`] (what
/// [`crate::pipeline::Pipeline::new`] itself calls, for its own
/// single-source case) and handed to that source's own `wire` closure; a
/// [`crate::elements::Tee`] keeps its own clone while it is alive, and its
/// [`crate::elements::TeeHandle`] accesses that clone weakly so retaining
/// the handle cannot keep the pipeline's `Bus` open after the `Tee` itself
/// is gone.
pub struct Context {
    pub bus: Bus,
    pub pipeline_id: Arc<str>,
    pub graph: PipelineGraph,
    pub clock: Arc<Clock>,
    /// Shared media-position clock used to hand video scheduling from the
    /// wall clock to an audio output master without changing pipelines.
    pub playback_clock: Arc<PlaybackClock>,
    /// Graph identity of the source whose wiring closure owns this context.
    pub source_id: ElementId,
}

#[cfg(test)]
impl Context {
    pub(crate) fn for_test(
        bus: Bus,
        pipeline_id: impl Into<Arc<str>>,
        graph: PipelineGraph,
        source_id: ElementId,
    ) -> Self {
        let clock = Arc::new(Clock::new());
        Self {
            bus,
            pipeline_id: pipeline_id.into(),
            graph,
            playback_clock: Arc::new(PlaybackClock::new(clock.clone())),
            clock,
            source_id,
        }
    }
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

/// A pure source: has output but no input. Its `run` method drives the
/// production loop and pushes buffers into its own src pad(s) until EOS or
/// an error. [`crate::pipeline::Pipeline::run`] normally invokes that loop
/// on the pipeline's background source thread; a caller may also invoke a
/// concrete implementation directly. Sources typically wrap blocking I/O
/// reads (demuxer, file/network source).
pub trait SourceElement: Source {
    /// Drives this source until `Eos` (normal completion),
    /// [`crate::pipeline::Pipeline::finish`], or `Stop` (see
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
