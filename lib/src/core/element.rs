//! The traits every element implements, and the identity it carries.
//!
//! [`Sink`] consumes buffers, [`Source`] owns the [`SrcPad`](crate::pad::SrcPad)s
//! they leave through, and [`Filter`] is simply both. [`SourceElement`] adds
//! the one thing a graph needs exactly once: a `run` loop that drives the
//! whole pipeline.
//!
//! [`Element`] itself is the identity half — an element's type and its
//! caller-chosen name, which every log record and every
//! [`BusEvent`](crate::bus::BusEvent) is attributed to.

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use crate::pp_log::PpLog;

use crate::{
    buffer::MediaBuffer,
    bus::Bus,
    clock::Clock,
    contract::InputContract,
    control::{ControlMsg, ControlReceiver},
    error::Result,
    graph::{ElementId, PipelineGraph},
    pad::SrcPad,
    playback_clock::PlaybackClock,
    stats::{ElementCounters, TickCounters},
};

/// Which kind of element posted a [`crate::bus::BusEvent`] — cheap to
/// compare/match, unlike the accompanying `name: Arc<str>` (an
/// instance-level identifier chosen by whoever constructed it, needed
/// alongside this to tell apart e.g. two `Queue`s in the same pipeline;
/// see [`Element::element_type`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementType {
    /// File or container demultiplexer source.
    FileDemuxer,
    /// Application-fed buffer source.
    AppSource,
    /// RTSP network stream source.
    RtspSource,
    /// Synthetic video source.
    TestVideoSource,
    /// Synthetic audio source.
    TestAudioSource,
    /// Windows desktop-duplication capture source.
    DxgiCaptureSource,
    /// Windows Graphics Capture window source.
    WgcCaptureSource,
    /// PipeWire audio capture source.
    PipeWireAudioCaptureSource,
    /// PipeWire screen capture source.
    PipeWireScreenCaptureSource,
    /// Windows WASAPI audio capture source.
    WasapiCaptureSource,
    /// Windows Media Foundation camera capture source.
    MfCaptureSource,
    /// Source for textures another D3D11 device shares in.
    D3d11SharedTextureSource,
    /// Linux V4L2 camera capture source.
    V4l2CaptureSource,
    /// Multi-input audio mixer source.
    AudioMixer,
    /// One input registered with an [`ElementType::AudioMixer`], which is a
    /// `Sink` in whichever pipeline feeds it rather than part of the mixer's
    /// own.
    ///
    /// Its own variant because it reads as one: an input that called itself
    /// `AudioMixer` put a mixer at the end of every branch feeding one, and a
    /// topology diagram with four of them in it is a diagram nobody can count
    /// the mixers from.
    AudioMixerInput,
    /// CPU video compositor source.
    SwVideoCompositor,
    /// CUDA video compositor source.
    CudaVideoCompositor,
    /// D3D11 video compositor source.
    D3d11VideoCompositor,
    /// WebRTC connection driver.
    WebRtcPeer,
    /// Packet timestamp rebase filter.
    FrameRateLimiter,
    PauseGate,
    TimestampOrigin,
    /// Video change/rate gate filter.
    ChangeGate,
    /// FFmpeg software decoder filter.
    SwDecoder,
    /// CUDA hardware decoder filter.
    CudaDecoder,
    /// System-memory to CUDA upload filter.
    CudaUpload,
    /// CUDA to system-memory download filter.
    CudaDownload,
    /// CUDA pixel-format converter filter.
    CudaConverter,
    /// D3D12 hardware decoder filter.
    D3d12Decoder,
    /// System-memory to D3D12 upload filter.
    D3d12Upload,
    /// D3D12 to system-memory download filter.
    D3d12Download,
    /// D3D12 video scaler filter.
    D3d12Scaler,
    /// D3D11 hardware decoder filter.
    D3d11Decoder,
    /// System-memory to D3D11 upload filter.
    D3d11Upload,
    /// D3D11 to system-memory download filter.
    D3d11Download,
    /// Software video encoder filter.
    SwEncoder,
    /// CUDA video encoder filter.
    CudaEncoder,
    /// D3D11-backed NVENC filter.
    D3d11VideoEncoder,
    /// Software audio encoder filter.
    SwAudioEncoder,
    /// Audio format and rate converter filter.
    AudioResampler,
    /// Runtime-adjustable audio gain filter.
    AudioVolume,
    /// Noise gate.
    AudioGate,
    /// Dynamic range compressor.
    AudioCompressor,
    /// Peak limiter.
    AudioLimiter,
    /// RNNoise noise suppression filter.
    NoiseSuppressor,
    /// Timestamp-to-wall-clock pacing filter.
    Pacer,
    /// Playback-master-aware video scheduling filter.
    VideoSynchronizer,
    /// CPU video scaler filter.
    SwScaler,
    /// CPU chroma-key filter.
    SwChromaKey,
    /// CUDA video scaler filter.
    CudaScaler,
    /// CUDA chroma-key filter.
    CudaChromaKey,
    /// D3D11 video scaler filter.
    D3d11Scaler,
    /// D3D11 chroma-key filter.
    D3d11ChromaKey,
    /// CPU colour-correction / luma-key filter.
    SwVideoEffect,
    /// CUDA colour-correction / luma-key filter.
    CudaVideoEffect,
    /// D3D11 colour-correction / luma-key filter.
    D3d11VideoEffect,
    /// D3D11 HDR-to-SDR filter.
    D3d11ToneMap,
    /// Dynamic one-to-many branch filter.
    Tee,
    /// A stretch of chain whose contents are replaced while it runs.
    Rack,
    /// One video stream decoded onto a device, by whichever path takes it.
    VideoDecodeBin,
    /// One video stream encoded into H.264, by whichever encoder takes it.
    VideoEncodeBin,
    /// Sound drawn as its waveform.
    AudioWaveform,
    /// Bounded asynchronous queue filter.
    Queue,
    /// Diagnostic decoded-frame counter sink.
    FrameCounter,
    /// Diagnostic compressed-packet counter sink.
    PacketCounter,
    /// CUDA video renderer sink.
    CudaRenderer,
    /// D3D12 video renderer sink.
    D3d12Renderer,
    /// D3D12 video renderer sink with its own window, or drawing into one
    /// it was given.
    D3d12WindowRenderer,
    /// D3D11 video renderer sink.
    D3d11Renderer,
    /// D3D11 video renderer sink with its own window, or drawing into one
    /// it was given.
    D3d11WindowRenderer,
    /// Vulkan video renderer sink, drawing into a window it was given.
    VulkanWindowRenderer,
    /// PipeWire audio renderer sink.
    PipeWireAudioRenderer,
    /// Windows WASAPI audio renderer sink.
    WasapiRenderer,
    /// RTSP publishing muxer sink, one or more tracks to an external
    /// server.
    RtspMuxer,
    /// Application callback or channel sink.
    AppSink,
    /// ONNX Runtime object-detection sink.
    OrtDetector,
    /// Speech-to-text sink, through whisper.cpp.
    WhisperTranscriber,
    /// HTTP Live Streaming muxer sink.
    HlsMuxer,
    /// Container file muxer sink — the container is whichever one
    /// FFmpeg infers from the output path.
    FileMuxer,
    /// Rotating segmented container file muxer sink.
    SegmentedFileMuxer,
    /// Rolling window of encoded packets, written to a file on request.
    ReplayBuffer,
    /// RTMP publishing muxer sink, one FLV stream to an external server.
    RtmpMuxer,
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

    /// Hands this element the pipeline it is being wired into, at the moment
    /// and for the reason [`Element::pp_log_mut`] hands it the pipeline's
    /// identity: the clock, the playback clock and the bus are the
    /// pipeline's to give, not the caller's to choose.
    ///
    /// # Why this is not a constructor argument
    ///
    /// It used to be, and an element could then be handed a clock from
    /// somewhere else entirely. That fails quietly. A [`Pacer`] on a foreign
    /// clock never sees `Pipeline::pause` shift the anchor and goes on
    /// pacing through a paused pipeline; an audio renderer registered on a
    /// foreign [`PlaybackClock`] *succeeds* — nothing is holding that one —
    /// and the video scheduled against the pipeline's own clock simply never
    /// hears about it. No error either way, just timing that does not work.
    ///
    /// Taking it here instead makes the mistake unrepresentable: there is
    /// nowhere else to get one from.
    ///
    /// # What an implementation should do
    ///
    /// Take what it needs and let the rest go. Holding the context keeps the
    /// graph and the bus alive for as long as the element, and invites
    /// reaching into the pipeline at moments it is not expecting one.
    /// [`crate::elements::Tee`] is the exception and has a reason: it builds
    /// branches later.
    ///
    /// Called once, before any buffer arrives — from the branch wiring for a
    /// filter or a terminal, and from
    /// [`crate::pipeline::PipelineBuilder::add_source`] for a source. An
    /// element that needs something from it and never receives one has been
    /// built outside a pipeline, which is a wiring mistake rather than a
    /// runtime condition; say so with a typed error rather than carrying on
    /// unpaced.
    ///
    /// Claiming something exclusive here — the playback clock's audio master
    /// is the only such thing today — is settled once, when the element is
    /// wired. A second claimant loses and is not offered the role again if
    /// the first one later goes away.
    ///
    /// [`Pacer`]: crate::elements::Pacer
    /// [`PlaybackClock`]: crate::playback_clock::PlaybackClock
    fn attach_context(&mut self, _context: &Arc<Context>) {}
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
    // Every element comes through here as it is made, which makes this the
    // one place FFmpeg is readied before any of them reaches it.
    crate::ensure_ffmpeg();
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
    /// Sender used by this source and its attached branches for asynchronous
    /// errors, EOS, drops, and seek completion.
    pub bus: Bus,

    /// Caller-selected identity of the pipeline currently being wired.
    pub pipeline_id: Arc<str>,

    /// Shared live topology graph updated by successful attach and detach
    /// operations.
    pub graph: PipelineGraph,

    /// Wall-time clock shared by paced elements in this pipeline.
    pub clock: Arc<Clock>,
    /// Shared media-position clock used to hand video scheduling from the
    /// wall clock to an audio output master without changing pipelines.
    pub playback_clock: Arc<PlaybackClock>,
    /// Where the pipeline's playback stands — whether data is to flow,
    /// which preroll is running, which timeline is current. Only the
    /// pipeline writes it; see [`crate::playback_state`].
    pub(crate) state: Arc<crate::playback_state::PlaybackState>,
    /// Serializes topology attachment with pipeline timeline operations.
    ///
    /// A branch may be detached while preroll is waiting (the waiter removes
    /// the departed terminal), but publishing a new branch in the middle of a
    /// seek would leave it outside the seek's terminal snapshot and control
    /// cascade.
    pub(crate) operation: Arc<Mutex<()>>,
    /// Graph identity of the source whose wiring closure owns this context.
    pub source_id: ElementId,
    /// That source's counters — see [`crate::stats`]. Here because a
    /// source runs on its own thread with nothing wrapping it, so this is
    /// the one place it can be handed them.
    pub(crate) source_counters: Arc<ElementCounters>,
    /// Which of the pipeline's terminals have ended, shared by every
    /// source's context — see [`crate::bus::BusEvent::Finished`].
    pub(crate) completion: Arc<crate::pipeline::completion::Completion>,
}

impl Context {
    /// Where the source this context belongs to records its ticks, for a
    /// source that produces on a schedule of its own. Asking is what makes
    /// its ticks part of what [`Pipeline::stats`] reports.
    ///
    /// For the source's own [`Element::attach_context`] and nothing else:
    /// a filter wired with the same context would be recording into its
    /// source's entry.
    ///
    /// [`Pipeline::stats`]: crate::pipeline::Pipeline::stats
    pub(crate) fn source_ticks(&self) -> Arc<TickCounters> {
        self.source_counters.ticks()
    }
}

#[cfg(test)]
impl Context {
    pub(crate) fn for_test(
        bus: Bus,
        pipeline_id: impl Into<Arc<str>>,
        graph: PipelineGraph,
        source_id: ElementId,
    ) -> Self {
        Self::for_test_with_clock(bus, pipeline_id, graph, source_id, Arc::new(Clock::new()))
    }

    /// The same, around a clock the test already holds — what a test that
    /// interrupts or pauses a paced element needs, since the element now
    /// takes its clock from a context rather than from the caller.
    pub(crate) fn for_test_with_clock(
        bus: Bus,
        pipeline_id: impl Into<Arc<str>>,
        graph: PipelineGraph,
        source_id: ElementId,
        clock: Arc<Clock>,
    ) -> Self {
        let pipeline_id: Arc<str> = pipeline_id.into();
        Self {
            bus,
            completion: crate::pipeline::completion::Completion::new(
                graph.clone(),
                pipeline_pp_log(&pipeline_id),
            ),
            pipeline_id,
            graph,
            playback_clock: Arc::new(PlaybackClock::new(clock.clone())),
            clock,
            state: crate::playback_state::PlaybackState::new(),
            operation: Arc::new(Mutex::new(())),
            source_id,
            source_counters: ElementCounters::new(),
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
    /// Returns whether calling [`Self::consume`] can make progress now.
    ///
    /// Thread boundaries check this before removing the next queued buffer,
    /// so a paused or completed-preroll terminal applies backpressure without
    /// dropping that buffer. Filters should delegate to their downstream pad;
    /// sinks that are always ready may keep the default.
    fn ready_consume(&mut self) -> bool {
        true
    }

    /// Processes one buffer synchronously on the caller's thread.
    ///
    /// An error is returned directly to upstream until the call crosses a
    /// [`crate::queue::Queue`] boundary. A queue instead reports the error on
    /// its [`crate::bus::Bus`], drops that buffer, and keeps its worker alive.
    /// Implementations must forward [`MediaBuffer::Eos`] after flushing any
    /// delayed state they own.
    ///
    /// For a terminal sink, returning `Ok(())` means the buffer has been
    /// accepted into that sink's output path. Pipeline preroll uses precisely
    /// this boundary: a video renderer must not return success until it has
    /// installed the frame as its current presentation content or submitted
    /// it to its presentation queue. This does not promise physical display
    /// scanout, audible playback, or remote receipt.
    fn consume(&mut self, buf: MediaBuffer) -> Result<()>;

    /// What this sink can be fed, checked when it is wired rather than
    /// when the first buffer arrives — see [`crate::contract`].
    ///
    /// The default declares nothing, so an element that does not override
    /// it links to anything and is validated exactly as before, when a
    /// buffer reaches `consume`. Declaring a contract never replaces that
    /// runtime check; it only moves the subset of failures that are
    /// knowable at wiring time to where they are unambiguously a wiring
    /// mistake rather than a bad buffer.
    fn input_contract(&self) -> InputContract {
        InputContract::Unknown
    }

    /// Whether this element can follow its pipeline across a seek — asked
    /// once, as it is wired, like [`Self::input_contract`], and kept with
    /// the graph so the pipeline knows before it asks anything running;
    /// see [`crate::pipeline::Pipeline::check_seek`].
    ///
    /// Nearly everything can: what it holds from the old position goes on
    /// the `Flush`, and the new position's arrives as ordinary data. What
    /// cannot is an element whose output is a record of the stream as it
    /// ran — a file being written, a replay window — into which a jump in
    /// the timeline would be written. The default is yes.
    fn accepts_seek(&self) -> bool {
        true
    }

    /// Reacts to a [`ControlMsg`] — drops what belongs to the timeline a
    /// `Flush` ends, arms for a `Preroll`, lets go of a device on `Stop` —
    /// on the thread delivering it, in order with the data around it.
    ///
    /// Only the reaction. Passing the message on is not this element's to
    /// do: a filter in a graph is behind a wrapper that hands it on through
    /// the filter's [`Source::src_pads`] once this returns, whatever it
    /// returns ([`crate::control::deliver`] does the same for a filter
    /// driven by hand); a terminal has nothing after it; and an element that
    /// routes control its own way — a [`crate::queue::Queue`], across a
    /// thread, a [`crate::elements::Tee`], to branches that come and go —
    /// does that from here. So an element with nothing of its own to reset
    /// leaves this alone, and a message added later reaches every element
    /// whether or not that element knows of it.
    ///
    /// An error goes back to whoever sent the message. The message has
    /// still gone on downstream: one element failing to pause must not
    /// leave the rest of the graph running.
    fn control(&mut self, _msg: &ControlMsg) -> Result<()> {
        Ok(())
    }
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
    /// Returns every output pad owned by this element.
    ///
    /// The slice order is the element's public pad-index contract. Implementors
    /// use the mutable access to push buffers and propagate control; callers
    /// normally connect pads through [`Context::attach`] instead of linking
    /// them directly.
    fn src_pads(&mut self) -> &mut [SrcPad];
}

/// A pure source: has output but no input. Its `run` method drives the
/// production loop and pushes buffers into its own src pad(s) until EOS or
/// an error. [`crate::pipeline::Pipeline::run`] normally invokes that loop
/// on the pipeline's background source thread; a caller may also invoke a
/// concrete implementation directly. Sources typically wrap blocking I/O
/// reads (demuxer, file/network source).
pub trait SourceElement: Source {
    /// Whether this source produces data from a live, externally advancing
    /// input rather than from a finite or application-controlled timeline.
    ///
    /// A live source cannot normally produce a first buffer while a pipeline
    /// is paused, so pipeline state handling may use this distinction to
    /// report that preroll is unavailable. Every implementation must classify
    /// itself explicitly so a new live source cannot silently opt into file-
    /// style preroll behavior.
    fn is_live(&self) -> bool;

    /// Whether this source can reposition its own input timeline through
    /// [`Self::seek`].
    ///
    /// This only describes the source's capability. A seekable source does
    /// not imply that every downstream branch can accept a pipeline seek;
    /// that must be validated across the complete graph before mutation.
    fn is_seekable(&self) -> bool;

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

    /// Reacts to one control message before it is forwarded to this source's
    /// own pads — the same ordering [`Self::seek`] gets, and for the same
    /// reason: whatever this source holds must already reflect the message by
    /// the time downstream elements see it.
    ///
    /// This is the source-side counterpart of [`Sink::control`], and exists
    /// for a source that holds state of its own. [`crate::elements::FileDemuxer`]
    /// uses it for both messages that touch its read-ahead: `Flush` discards
    /// packets belonging to the timeline being left, and `Preroll` is what
    /// makes it hold a blocked pad's packets instead of waiting on that pad.
    ///
    /// The default is a no-op. A source that hands every packet straight to a
    /// pad has nothing of its own to keep in step.
    fn on_control(&mut self, _msg: &ControlMsg) {}

    /// Called as a `Pause` arrives, before it is passed downstream: what the
    /// source does to stop producing — a capture device stopped. The time
    /// it takes counts as paused. An error ends the source.
    ///
    /// The default does nothing: a source that only reads when asked has
    /// nothing running to stop.
    fn pausing(&mut self) -> Result<()> {
        Ok(())
    }

    /// Called as a pause ends — on `Resume`, or on the `Preroll` of a seek
    /// made while paused — after the request has been passed downstream and
    /// before it is acknowledged, so no caller sees the source half resumed:
    /// a capture device restarted, and whatever it queued while stopped
    /// discarded, since that is from before the pause. An error ends the
    /// source.
    ///
    /// The default does nothing.
    fn resuming(&mut self) -> Result<()> {
        Ok(())
    }

    /// Repositions this source to `target`, an absolute position from the
    /// start of the media (e.g. `av_seek_frame` for
    /// [`crate::elements::FileDemuxer`]). Called by
    /// [`crate::control::drain_control`] as part of handling
    /// [`ControlMsg::Seek`], *before* that message is forwarded to the
    /// source's own pads — so whatever's read next comes from the new
    /// position by the time downstream elements receive the new timeline
    /// announcement. Buffered and stateful old-timeline data is discarded by
    /// the preceding [`ControlMsg::Flush`].
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

/// A boxed element is the element it holds — a boxed sink a sink, a boxed
/// filter a filter — so anything that takes one takes it boxed or not.
///
/// Two things need that. [`ChainBuilder::to`](crate::pipeline::ChainBuilder::to)
/// takes any sink, and one a muxer handed over already boxed is still one.
/// And [`Rack`](crate::elements::Rack) holds its contents boxed — that is
/// what makes them exchangeable — and everything a chain gives an element
/// it builds, from the pipeline id in its log to the tracer that stamps a
/// failure with the name of whatever raised it, is written against a type
/// that implements these three traits. Without this the elements inside a
/// rack would be the one stretch of a running graph no such wrapper could
/// reach.
///
/// The delegation is exactly that. Nothing here decides anything; every
/// method is the one on the element inside.
impl<E: Element + ?Sized> Element for Box<E> {
    fn name(&self) -> Arc<str> {
        (**self).name()
    }

    fn element_type(&self) -> ElementType {
        (**self).element_type()
    }

    fn graph_id(&self) -> Option<ElementId> {
        (**self).graph_id()
    }

    fn pp_log(&self) -> &PpLog {
        (**self).pp_log()
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        (**self).pp_log_mut()
    }

    fn attach_context(&mut self, context: &Arc<Context>) {
        (**self).attach_context(context);
    }
}

impl<S: Source + ?Sized> Source for Box<S> {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        (**self).src_pads()
    }
}

impl<S: Sink + ?Sized> Sink for Box<S> {
    fn ready_consume(&mut self) -> bool {
        (**self).ready_consume()
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        (**self).consume(buf)
    }

    fn input_contract(&self) -> InputContract {
        (**self).input_contract()
    }

    fn accepts_seek(&self) -> bool {
        (**self).accepts_seek()
    }

    fn control(&mut self, msg: &ControlMsg) -> Result<()> {
        (**self).control(msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{MediaKind, PortContract};

    /// A sink whose every answer differs from the trait's default, so a
    /// boxed one that fell back to a default would be caught saying it.
    struct Opinionated {
        pp_log: PpLog,
        consumed: usize,
    }

    impl Element for Opinionated {
        fn name(&self) -> Arc<str> {
            "opinionated".into()
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

    impl Sink for Opinionated {
        fn ready_consume(&mut self) -> bool {
            false
        }

        fn consume(&mut self, _buf: MediaBuffer) -> Result<()> {
            self.consumed += 1;
            Ok(())
        }

        fn input_contract(&self) -> InputContract {
            InputContract::Fixed(PortContract::packet(MediaKind::AudioPacket))
        }

        fn control(&mut self, _msg: &ControlMsg) -> Result<()> {
            Err(crate::error::Error::Other("opinionated".into()))
        }
    }

    fn opinionated() -> Opinionated {
        Opinionated {
            pp_log: element_pp_log(ElementType::Other, "opinionated", None),
            consumed: 0,
        }
    }

    /// A boxed sink answers what the sink inside answers — including where
    /// the trait has a default the box could have fallen back to — whether
    /// it is boxed as itself or as `dyn Sink`, which is what lets
    /// `ChainBuilder::to` take either.
    #[test]
    fn a_boxed_sink_is_the_sink_inside_it() {
        fn check(mut sink: impl Sink) {
            assert_eq!(&*sink.name(), "opinionated");
            assert!(!sink.ready_consume(), "not the default of true");
            assert_eq!(
                sink.input_contract(),
                InputContract::Fixed(PortContract::packet(MediaKind::AudioPacket)),
                "not the default of Unknown"
            );
            sink.consume(MediaBuffer::Eos).unwrap();
            assert!(
                sink.control(&ControlMsg::Pause).is_err(),
                "not the default of doing nothing"
            );
        }
        check(Box::new(opinionated()));
        check(Box::new(opinionated()) as Box<dyn Sink>);
        check(Box::new(Box::new(opinionated()) as Box<dyn Sink>));

        let mut boxed: Box<dyn Sink> = Box::new(opinionated());
        boxed.consume(MediaBuffer::Eos).unwrap();
        let mut twice = Box::new(boxed);
        twice.consume(MediaBuffer::Eos).unwrap();
    }
}
