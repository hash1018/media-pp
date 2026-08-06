use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use rust_hlog::HLog;

use crate::{
    buffer::MediaBuffer,
    bus::Bus,
    clock::Clock,
    control::{ControlMsg, ControlReceiver},
    error::Result,
    pad::SrcPad,
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
    DxgiScreenSource,
    WebRtcPeer,
    SwDecoder,
    D3d12vaDecoder,
    D3d12Upload,
    D3d11Decoder,
    D3d11Upload,
    SwEncoder,
    Pacer,
    Scaler,
    Tee,
    Queue,
    FrameCounter,
    PacketCounter,
    D3d12Renderer,
    D3d11Renderer,
    RtspServer,
    AppSink,
    OrtDetector,
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

    /// This element's identity for [`crate::bus::Bus::post`] — same
    /// `id`/`name` as [`Element::name`], just already wrapped as the
    /// [`rust_hlog::HLog`] its `hinfo!`/`hwarn!`/`herror!` macros need. A
    /// stored field (via `#[rust_hlog::hlog]`), not built fresh per call,
    /// for the same reason `name()` returns a cheap `Arc<str>` clone
    /// instead of a fresh `String` — see its own docs.
    fn hlog(&self) -> &HLog;

    /// Mutable access to the same field [`Element::hlog`] reads — used by
    /// [`crate::pipeline::ChainBuilder`] to stamp a `sub_id` (the owning
    /// [`crate::pipeline::Pipeline`]'s own id) onto every element that
    /// passes through it, via [`element_hlog`]. Not meant to be called
    /// from anywhere else.
    fn hlog_mut(&mut self) -> &mut HLog;
}

/// Builds the [`HLog`] every element constructs for its own [`Element::hlog`]
/// field, and that [`crate::pipeline::ChainBuilder`]/[`crate::pipeline::Pipeline`]
/// rebuild once they know which pipeline an element belongs to. Renders as
/// `ElementType(name)` for the id — so a log line names *what* failed, not
/// just the instance name a caller could've picked for two different kinds
/// of element — and `Pipeline(pipeline_id)` for the sub_id, when there is
/// one (`None` for an element that isn't wired into a `Pipeline` at all,
/// e.g. most of this crate's own tests). Public so a custom `Element`
/// implemented outside this crate (see [`ElementType::Other`]) can build
/// its own `hlog` field the same way.
pub fn element_hlog(element_type: ElementType, name: &str, pipeline_id: Option<&str>) -> HLog {
    let id = format!("{element_type:?}({name})");
    match pipeline_id {
        Some(pipeline_id) => HLog::new(&id, Some(&format!("Pipeline({pipeline_id})"))),
        None => HLog::new(&id, None),
    }
}

/// One element known to have been statically wired into a
/// [`crate::pipeline::Pipeline`] — see [`crate::pipeline::Pipeline::elements`]/
/// [`crate::pipeline::Pipeline::topology`].
#[derive(Debug, Clone)]
pub struct ElementInfo {
    pub element_type: ElementType,
    pub name: Arc<str>,
    /// The name of whatever feeds this element, if [`ElementRegistry`]
    /// could determine it. `None` means either this *is* the pipeline's
    /// own source (nothing feeds it), or this registry simply couldn't
    /// trace it — e.g. the first element of a [`crate::pipeline::ChainBuilder`]
    /// chain, until whatever links the finished chain resolves it (the
    /// pipeline's source by default, or a [`crate::elements::Tee`] — see
    /// [`crate::elements::TeeHandle::add_sink`]). Rendering code (see
    /// [`crate::pipeline::Pipeline::topology`]) treats an unresolved
    /// `None` other than the source's own entry as "attached directly to
    /// the source", the same default `ChainBuilder`/`Pipeline::new` apply.
    pub upstream: Option<Arc<str>>,
}

/// Shared, append-mostly registry of every [`ElementInfo`] wired into a
/// [`crate::pipeline::Pipeline`] — created by
/// [`crate::pipeline::Pipeline::new`] and handed out (via [`Context`]) to
/// whatever [`crate::pipeline::ChainBuilder`]/[`crate::elements::Tee`]
/// records itself as it's constructed, statically or dynamically.
/// `Pipeline` keeps its own clone alive for the whole run, so
/// [`crate::pipeline::Pipeline::elements`]/[`crate::pipeline::Pipeline::topology`]
/// can re-derive a fresh view — including anything registered well after
/// construction (e.g. a branch added to a running [`crate::elements::Tee`])
/// — on every call, rather than a snapshot frozen at construction time.
///
/// Deliberately its own type rather than piggybacked on
/// [`crate::bus::Bus`] (an earlier version of this did exactly that): a
/// `Bus` is for runtime playback events, not build-time structure — the
/// two don't belong sharing one channel just because both happen to
/// already be threaded through the same [`Context`].
#[derive(Clone, Default)]
pub struct ElementRegistry(Arc<Mutex<Vec<ElementInfo>>>);

impl ElementRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records one element, in whatever order it's discovered. `upstream`
    /// is typically "the previous element in the same chain" — `None` for
    /// a chain's first element, left for later resolution (see
    /// [`ElementInfo::upstream`]'s own docs).
    pub(crate) fn register(
        &self,
        element_type: ElementType,
        name: Arc<str>,
        upstream: Option<Arc<str>>,
    ) {
        self.0.lock().unwrap().push(ElementInfo {
            element_type,
            name,
            upstream,
        });
    }

    /// Points the entry named `name`'s `upstream` at `upstream` — used
    /// once a chain's true root becomes known only *after* it was already
    /// registered, e.g. [`crate::elements::TeeHandle::add_sink`] learning
    /// the branch it just received actually starts under that `Tee`, not
    /// wherever it'll eventually turn out to be linked. A no-op if
    /// nothing named `name` is registered (nothing to fix up).
    pub(crate) fn set_upstream(&self, name: &str, upstream: Arc<str>) {
        if let Some(info) = self.0.lock().unwrap().iter_mut().find(|e| &*e.name == name) {
            info.upstream = Some(upstream);
        }
    }

    /// Removes `name`'s own entry, then repeatedly removes anything whose
    /// `upstream` (transitively) pointed at something just removed — i.e.
    /// the whole branch rooted at `name`, not just its first element. Used
    /// by [`crate::elements::TeeHandle::remove_sink`]: once a branch is
    /// detached, it's no longer part of this pipeline's topology at all,
    /// not still shown hanging off the `Tee` it was just pulled from.
    ///
    /// Deliberately a hard delete, not a reversible "detach" a later
    /// `add_sink` could restore — mirrors [`crate::pipeline::Pipeline`]
    /// itself not supporting reuse (see its own docs): resurrecting a
    /// removed branch's old bookkeeping was designed once (a `detached`
    /// flag + a cascading "reattach" call) and deliberately dropped as
    /// unneeded complexity for a case with no real caller — if you want a
    /// branch back, build a fresh one. A no-op if nothing named `name` is
    /// registered.
    pub(crate) fn remove_subtree(&self, name: &str) {
        let mut elements = self.0.lock().unwrap();
        let Some(pos) = elements.iter().position(|e| &*e.name == name) else {
            return;
        };
        let mut removed = vec![elements.remove(pos).name];
        loop {
            let before = elements.len();
            elements.retain(|e| {
                let is_orphaned = e
                    .upstream
                    .as_deref()
                    .is_some_and(|u| removed.iter().any(|r| &**r == u));
                if is_orphaned {
                    removed.push(e.name.clone());
                }
                !is_orphaned
            });
            if elements.len() == before {
                break;
            }
        }
    }

    /// Everything registered so far, in registration order.
    pub(crate) fn snapshot(&self) -> Vec<ElementInfo> {
        self.0.lock().unwrap().clone()
    }
}

/// Everything a [`crate::pipeline::ChainBuilder`]/[`crate::elements::Tee`]
/// needs to wire itself into a [`crate::pipeline::Pipeline`] — bundled into
/// one `Arc` instead of threading `bus`/`pipeline_id`/`registry`/`clock`
/// through separately. Built once by [`crate::pipeline::Pipeline::new`] and
/// handed to its `wire` closure; a [`crate::elements::Tee`] keeps its own
/// clone while it is alive, and its [`crate::elements::TeeHandle`] accesses
/// that clone weakly so retaining the handle cannot keep the pipeline's
/// `Bus` open after the `Tee` itself is gone.
pub struct Context {
    pub bus: Bus,
    pub pipeline_id: Arc<str>,
    pub registry: ElementRegistry,
    pub clock: Arc<Clock>,
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
