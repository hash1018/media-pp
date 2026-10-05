use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use arc_swap::ArcSwapOption;
use ffmpeg_next::{self as ffmpeg, ffi};
use objc2_metal::{MTLDevice, MTLPixelFormat, MTLTextureUsage};
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_error, pp_info};
use crate::rate::FrameRate;

use super::super::sw_video_compositor::VideoCompositorOptions;
use super::super::text_layer::TextMask;
use super::super::timed_inputs::{
    OfflineCompositor, Picture, TimedFeed, TimedInput, Untimed, produce_offline,
};
use super::super::video_layer::{
    self, LayerGeometry, MAX_DIMENSION, VideoInputId, VideoLayer, VideoLayerError, VideoRect,
    layer_geometry,
};
use crate::elements::source::render_mode::{MediaTime, RenderMode};
use crate::playback_state::Bell;
use crate::{
    buffer::{MediaBuffer, picture_id, release_picture},
    color::{Color, ColorDescription},
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    control::ControlMsg,
    element::{
        Context, Element, ElementType, Flow, Produced, RawSink, Source, SourceStage, Wait,
        element_pp_log,
    },
    elements::{VideoToolboxDevice, VideoToolboxFrameFormat},
    error::Result,
    platform::{
        ffmpeg::AvBufferRef,
        macos::{
            metal::{Kernel, MetalError, MetalGpu, Pass, Texture, write_texture},
            pixel_buffer::PixelBuffer,
            videotoolbox::{NotVideoToolbox, create_frames_ctx, sw_format_of},
        },
    },
    pool::{UnboundObjectPool, UnboundObjectPoolRef},
    produce::source_stage,
    schedule::PeriodicSchedule,
    stats::TickCounters,
    stream::StreamEvent,
};

mod text_handle;
mod video_handle;

pub use text_handle::MetalTextLayerHandle;
use text_handle::TextLayerState;
pub use video_handle::MetalVideoLayerHandle;

const OUTPUT_POOL_SIZE: usize = 4;
/// How long a live compositor waits at a time before looking at its rate
/// again: a rate set while it waits is kept from then, rather than a tick of
/// the old one later.
const RATE_POLL_INTERVAL: Duration = Duration::from_millis(5);

/// The compositor's kernels, compiled from this one source.
const SHADER: &str = include_str!("../../../../../shaders/metal/composite.metal");

/// Errors specific to [`MetalVideoCompositor`].
#[derive(Debug, ThisError)]
pub enum MetalVideoCompositorError {
    /// A Metal call this compositor made failed.
    #[error(transparent)]
    Metal(#[from] MetalError),

    /// FFmpeg could not make the pool output frames come from.
    #[error("{0}")]
    Pool(String),

    /// FFmpeg could not take an output frame from the pool.
    #[error("failed to take an output frame from the VideoToolbox pool (code {0})")]
    FrameGet(i32),

    /// FFmpeg could not take a second reference to the frame last composed,
    /// which is how an unchanged picture is offered again.
    #[error("failed to reference the previous composite (code {0})")]
    FrameRef(i32),

    /// A background that is not opaque, on an NV12 canvas, which has no
    /// alpha to keep it in.
    #[error(
        "background_alpha {0} is not supported: an NV12 MetalVideoCompositor has no alpha; compose in VideoToolboxFrameFormat::Bgra"
    )]
    TranslucentBackground(u8),

    /// Output dimensions are odd, too small, or above the safety limit.
    #[error(
        "invalid output dimensions {width}x{height}; each dimension must be even and 2..={MAX_DIMENSION}"
    )]
    InvalidOutputDimensions {
        /// Invalid output width in pixels.
        width: u32,
        /// Invalid output height in pixels.
        height: u32,
    },

    /// The output frame-rate numerator or denominator is non-positive.
    #[error("invalid frame rate {0}; numerator and denominator must both be positive")]
    InvalidFrameRate(ffmpeg::Rational),

    /// A layer destination rectangle is zero-sized or exceeds the safety limit.
    #[error(
        "invalid layer dimensions {width}x{height}; each dimension must be 1..={MAX_DIMENSION}"
    )]
    InvalidLayerDimensions {
        /// Invalid layer width in output pixels.
        width: u32,
        /// Invalid layer height in output pixels.
        height: u32,
    },

    /// A layer opacity is non-finite or outside `0.0..=1.0`.
    #[error("layer opacity must be finite and between 0.0 and 1.0, got {0}")]
    InvalidOpacity(f32),

    /// A layer's source region is empty. Hiding a layer is
    /// [`VideoLayer::visible`]; asking it to draw nothing is a mistake.
    #[error("layer source region has invalid dimensions {width}x{height}")]
    InvalidSourceRegion {
        /// Region width as given.
        width: u32,
        /// Region height as given.
        height: u32,
    },

    /// An input frame reports a zero width or height, or one larger than
    /// the pixel buffer it is in.
    #[error("input frame has invalid dimensions {width}x{height}")]
    InvalidInputDimensions {
        /// Invalid input width in pixels.
        width: u32,
        /// Invalid input height in pixels.
        height: u32,
    },

    /// Aspect-ratio fitting would create an intermediate image above the safety limit.
    #[error("scaled layer would exceed {MAX_DIMENSION}px: {width}x{height}")]
    ScaledLayerTooLarge {
        /// Computed scaled width in pixels.
        width: u32,
        /// Computed scaled height in pixels.
        height: u32,
    },

    /// A runtime layer handle refers to an input that has been removed or replaced.
    #[error("the compositor input has been removed")]
    SourceRemoved,

    /// An input frame is not a VideoToolbox frame.
    #[error("MetalVideoCompositor draws VideoToolbox frames, got a {0:?} frame; upload it first")]
    NotVideoToolbox(ffmpeg::format::Pixel),

    /// An input frame says it is a VideoToolbox frame but carries no frames
    /// context to say what it holds.
    #[error("MetalVideoCompositor was handed a VideoToolbox frame with no frames context")]
    NoFramesContext,

    /// An input frame holds a layout this compositor does not draw.
    #[error("MetalVideoCompositor draws NV12 and BGRA frames, got {0:?}")]
    UnsupportedLayout(ffmpeg::format::Pixel),

    /// An input sink received a buffer other than decoded video.
    #[error(
        "MetalVideoCompositorInputSink only accepts decoded Video frames, got a {0}; link it after a decoder or upload"
    )]
    UnsupportedBuffer(&'static str),

    /// The supplied bytes are not a supported TrueType or OpenType font.
    #[error("invalid font data: {0}")]
    InvalidFont(String),

    /// The glyph pixel height is non-positive or non-finite.
    #[error("font size must be finite and greater than zero, got {0}")]
    InvalidFontSize(f32),

    /// Rasterizing the requested text would exceed supported dimensions.
    #[error("rasterized text is too large: {width}x{height}")]
    TextTooLarge {
        /// Computed raster width in pixels.
        width: u64,
        /// Computed raster height in pixels.
        height: u64,
    },

    /// Host memory for the rasterized pixel buffer could not be reserved.
    #[error("could not allocate {bytes} bytes for rasterized text")]
    AllocationFailed {
        /// Number of bytes requested for the text bitmap.
        bytes: usize,
    },

    /// The compositor this handle belongs to has stopped, so there is nothing
    /// left to add to.
    #[error("the compositor has stopped")]
    Stopped,

    /// An offline compositor's rate cannot change while it runs — see
    /// [`crate::elements::SwVideoCompositorError::FixedFrameRate`].
    #[error("an offline compositor's frame rate is fixed at construction")]
    FixedFrameRate,

    /// An offline compositor was given a frame it cannot place in time —
    /// one with no `pts`, or no time base to read it in.
    #[error("an offline compositor's input frame has no {0}")]
    UntimedFrame(&'static str),
}

pub(crate) struct VideoInput {
    id: VideoInputId,
    /// The hot producer/consumer path is an atomic latest-value slot: input
    /// pipelines replace the pointer without taking the layer lock, and the
    /// compositor acquires a stable Arc snapshot independently.
    latest_frame: ArcSwapOption<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    layer: Mutex<VideoLayer>,
    /// An offline compositor's input fed through a sink — see the CPU
    /// compositor's field of the same name.
    timed: Option<TimedFeed>,
}

pub(crate) struct CompositorShared {
    inputs: Mutex<HashMap<Arc<str>, Arc<VideoInput>>>,
    /// Text layers are kept apart from video inputs because they are not
    /// inputs at all: nothing pushes frames into one.
    text_layers: Mutex<Vec<(Arc<str>, Arc<TextLayerState>)>>,
    next_input_id: AtomicU64,
    /// The output rate, which the tick loop reads each pass and
    /// [`MetalVideoCompositorHandle::set_frame_rate`] writes.
    frame_rate: Arc<FrameRate>,
    mode: RenderMode,
    /// Rung as anything an offline compositor waits for changes — a frame
    /// or an end arriving at an input, an input added or removed.
    arrived: Bell,
    /// The output frame an offline compositor is making next.
    next_output: AtomicI64,
    /// Whether an offline compositor has ever had an input fed through a sink.
    fed: AtomicBool,
}

impl CompositorShared {
    /// Output frame `index` as a time, counted in output frames.
    fn output_time(&self, index: i64) -> MediaTime {
        MediaTime::new(index, self.frame_rate.get().invert())
            .expect("the frame rate was validated as positive")
    }
}

/// A cheaply cloneable handle for adding and removing compositor inputs —
/// the Metal sibling of [`crate::elements::SwVideoCompositorHandle`].
///
/// Holds the compositor weakly: once it is dropped, every method reports
/// [`MetalVideoCompositorError::Stopped`] or nothing.
#[derive(Clone)]
pub struct MetalVideoCompositorHandle {
    shared: Weak<CompositorShared>,
}

/// The sink and layer handle [`MetalVideoCompositorHandle::add_source`]
/// returns — see [`crate::elements::CompositorInput`].
pub type MetalVideoCompositorInput = crate::elements::CompositorInput<MetalVideoLayerHandle>;

impl MetalVideoCompositorHandle {
    /// Registers an input and returns its terminal Sink plus independent
    /// runtime layer control. Reusing `name` replaces the old registration;
    /// the replaced sink and layer handle can no longer affect this
    /// compositor.
    ///
    /// Each input keeps only its latest frame, which the compositor draws at
    /// every tick of its own rate. A live source keeps its own time; a file
    /// does not, and without a [`crate::elements::Pacer`] in front of this
    /// sink it is read as fast as it decodes, and all but the last frame of
    /// each tick are passed over.
    ///
    /// Offline the input is read by its timestamps instead, and held back a
    /// frame ahead of the output — see [`crate::elements::RenderMode::Offline`]
    /// and [`crate::elements::SwVideoCompositorHandle::add_source`].
    pub fn add_source(
        &self,
        name: impl Into<String>,
        layer: VideoLayer,
    ) -> std::result::Result<MetalVideoCompositorInput, MetalVideoCompositorError> {
        let layer = self.register(name, layer, true)?;
        Ok(MetalVideoCompositorInput {
            sink: crate::element::BoxSink::new(MetalVideoCompositorInputSink {
                pp_log: element_pp_log(ElementType::MetalVideoCompositor, &layer.name, None),
                name: layer.name.clone(),
                id: layer.id,
                shared: self.shared.clone(),
                input: layer.input.clone(),
            }),
            layer,
        })
    }

    /// Registers an input and returns *only* its layer handle — no `RawSink` —
    /// for a caller that sets this input's picture itself through
    /// [`MetalVideoLayerHandle::set_frame`] rather than wiring a pipeline
    /// into it. Replaces any registration of the same name, as
    /// [`Self::add_source`] does.
    ///
    /// Offline too the picture is whatever was last set: it has no
    /// timestamps to be placed by, and is never waited for.
    pub fn add_layer(
        &self,
        name: impl Into<String>,
        layer: VideoLayer,
    ) -> std::result::Result<MetalVideoLayerHandle, MetalVideoCompositorError> {
        self.register(name, layer, false)
    }

    /// Registers an input, fed through a sink where `fed` says so — which,
    /// offline, is an input whose frames are placed by their timestamps.
    fn register(
        &self,
        name: impl Into<String>,
        layer: VideoLayer,
        fed: bool,
    ) -> std::result::Result<MetalVideoLayerHandle, MetalVideoCompositorError> {
        validate_layer(layer)?;
        let Some(shared) = self.shared.upgrade() else {
            return Err(MetalVideoCompositorError::Stopped);
        };
        let name: Arc<str> = name.into().into();
        let timed = (fed && !shared.mode.is_live()).then(|| {
            shared.fed.store(true, Ordering::Release);
            let next = shared.next_output.load(Ordering::Acquire);
            TimedFeed::new(shared.arrived.clone(), shared.output_time(next))
        });
        let id = VideoInputId(shared.next_input_id.fetch_add(1, Ordering::Relaxed));
        let input = Arc::new(VideoInput {
            id,
            latest_frame: ArcSwapOption::empty(),
            layer: Mutex::new(layer),
            timed,
        });
        shared
            .inputs
            .lock()
            .unwrap()
            .insert(name.clone(), input.clone());
        shared.arrived.ring();
        Ok(MetalVideoLayerHandle {
            id,
            name,
            input: Arc::downgrade(&input),
            shared: self.shared.clone(),
        })
    }

    /// Removes the registration under `name`, if any.
    pub fn remove_source(&self, name: &str) {
        if let Some(shared) = self.shared.upgrade() {
            shared.inputs.lock().unwrap().remove(name);
            shared.arrived.ring();
        }
    }

    /// Changes the rate this compositor emits at, from the next tick — the
    /// same contract as `VulkanVideoCompositorHandle::set_frame_rate`:
    /// [`MetalVideoCompositorError::InvalidFrameRate`] for a rate that is
    /// not positive, [`MetalVideoCompositorError::Stopped`] once the
    /// compositor is gone, [`MetalVideoCompositorError::FixedFrameRate`]
    /// offline, the running rate left alone in each case.
    /// [`MetalVideoCompositor::time_base`] is the reciprocal of this and the
    /// output `pts` a tick counter in those units, so a change re-means every
    /// timestamp after it.
    pub fn set_frame_rate(
        &self,
        frame_rate: ffmpeg::Rational,
    ) -> std::result::Result<(), MetalVideoCompositorError> {
        let shared = self
            .shared
            .upgrade()
            .ok_or(MetalVideoCompositorError::Stopped)?;
        if !shared.mode.is_live() {
            return Err(MetalVideoCompositorError::FixedFrameRate);
        }
        shared
            .frame_rate
            .set(frame_rate)
            .map_err(|_| MetalVideoCompositorError::InvalidFrameRate(frame_rate))
    }

    /// The rate this compositor is emitting at, or `None` once it is gone.
    pub fn frame_rate(&self) -> Option<ffmpeg::Rational> {
        let shared = self.shared.upgrade()?;
        Some(shared.frame_rate.get())
    }

    /// How many inputs are registered, or zero once the compositor is gone.
    pub fn source_count(&self) -> usize {
        self.shared
            .upgrade()
            .map(|shared| shared.inputs.lock().unwrap().len())
            .unwrap_or(0)
    }
}

/// The terminal Sink for one compositor input. Keeps only the latest frame:
/// the compositor emits on its own clock, so an input that runs faster than
/// the output rate simply has its older frames dropped.
pub struct MetalVideoCompositorInputSink {
    pp_log: PpLog,
    name: Arc<str>,
    id: VideoInputId,
    shared: Weak<CompositorShared>,
    input: Weak<VideoInput>,
}

impl MetalVideoCompositorInputSink {
    /// Drops this registration, but only if it is still the current one —
    /// a replaced sink must not remove its replacement.
    fn detach(&self) {
        let Some(shared) = self.shared.upgrade() else {
            return;
        };
        let mut inputs = shared.inputs.lock().unwrap();
        if inputs
            .get(&self.name)
            .is_some_and(|current| current.id == self.id)
        {
            inputs.remove(&self.name);
            drop(inputs);
            shared.arrived.ring();
        }
    }
}

impl Element for MetalVideoCompositorInputSink {
    /// Remembers the pipeline this input is fed by, which an offline
    /// compositor rings as it makes room.
    fn attach_context(&mut self, context: &Arc<Context>) {
        if let Some(input) = self.input.upgrade()
            && let Some(timed) = &input.timed
        {
            timed.fed_by(&context.state);
        }
    }

    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::MetalVideoCompositor
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl RawSink for MetalVideoCompositorInputSink {
    /// Every layer is drawn on the GPU, from a VideoToolbox frame of its own.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::VideoToolbox)
                .with_layouts(crate::contract::PixelLayoutSet::NV12_OR_BGRA),
        )
    }

    /// Always, live: an input keeps only its latest frame. Offline, not
    /// once it holds a frame past the output time being made.
    fn ready_consume(&mut self) -> bool {
        self.input
            .upgrade()
            .and_then(|input| input.timed.as_ref().map(TimedFeed::wants_more))
            .unwrap_or(true)
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Video(frame) => {
                if self.shared.strong_count() == 0 {
                    return Err(MetalVideoCompositorError::Stopped.into());
                }
                // Validated here rather than when drawn, so a misconfigured
                // input names *itself* in the error.
                validate_input_frame(&frame).inspect_err(|error| pp_error!(self, "{error}"))?;
                let Some(input) = self.input.upgrade() else {
                    return Ok(());
                };
                let Some(timed) = &input.timed else {
                    input.latest_frame.store(Some(frame));
                    return Ok(());
                };
                timed.push(frame).map_err(|untimed| {
                    MetalVideoCompositorError::UntimedFrame(match untimed {
                        Untimed::NoTimestamp => "timestamp",
                        Untimed::NoTimeBase => "time base",
                    })
                    .into()
                })
            }
            other => {
                pp_error!(self, "unsupported buffer: {}", other.kind());
                Err(MetalVideoCompositorError::UnsupportedBuffer(other.kind()).into())
            }
        }
    }

    /// Its stream's end. Offline it is part of the picture: what it holds is
    /// still shown to its last frame's end, after which it is no longer
    /// waited for. Live it shows nothing from here — its layer goes with its
    /// stream — but the input stays, so a file sought back after its end is
    /// shown again; a `Stop` is what removes it.
    fn stream_event(&mut self, event: &StreamEvent) -> Result<()> {
        if let (StreamEvent::Eos, Some(input)) = (event, self.input.upgrade()) {
            match &input.timed {
                Some(timed) => timed.end(),
                None => input.latest_frame.store(None),
            }
        }
        Ok(())
    }

    fn flow(&mut self, Flow(msg): Flow<'_>) -> Result<()> {
        // Terminal for its own branch: nothing downstream to forward to. A
        // `Stop` means this upstream pipeline is done, so the registration
        // goes with it — same as `SwVideoCompositorInputSink`.
        if matches!(msg, ControlMsg::Stop) {
            self.detach();
        }
        // What an input held before a seek's flush is not what comes after:
        // live, nothing is shown until the new position's first picture.
        if matches!(msg, ControlMsg::Flush)
            && let Some(input) = self.input.upgrade()
        {
            input.latest_frame.store(None);
            if let Some(timed) = &input.timed {
                timed.clear();
            }
        }
        Ok(())
    }
}

#[derive(Clone)]
struct InputSnapshot {
    id: VideoInputId,
    layer: VideoLayer,
    frame: Option<Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>>,
}

impl InputSnapshot {
    /// Same input, in the same place, showing the same pixels — compared by
    /// the pixel buffer they live in.
    fn same_as(&self, other: &Self) -> bool {
        self.id == other.id
            && self.layer == other.layer
            && match (&self.frame, &other.frame) {
                (Some(drawn), Some(now)) => picture_id(drawn) == picture_id(now),
                (None, None) => true,
                _ => false,
            }
    }
}

/// One text layer as the compositor found it, read once so that drawing it
/// and deciding whether it changed both see the same values.
#[derive(Clone)]
struct TextSnapshot {
    mask: Option<Arc<TextMask>>,
    x: i32,
    y: i32,
    /// `f32` bits, compared rather than interpreted.
    opacity: u32,
    visible: bool,
    color: Color,
    z_index: i32,
}

impl PartialEq for TextSnapshot {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x
            && self.y == other.y
            && self.opacity == other.opacity
            && self.visible == other.visible
            && self.z_index == other.z_index
            && self.color == other.color
            && match (&self.mask, &other.mask) {
                // Identity again: `set_text` replaces a mask wholesale.
                (Some(drawn), Some(now)) => Arc::ptr_eq(drawn, now),
                (None, None) => true,
                _ => false,
            }
    }
}

/// What the last composite was made from, and what it produced: a tick that
/// finds everything as the last one left it hands out that frame again — a
/// new timestamp on a new reference to the same pixel buffer, never a copy.
struct Composed {
    inputs: Vec<InputSnapshot>,
    texts: Vec<TextSnapshot>,
    /// A reference to the frame last emitted, which also keeps its pixel
    /// buffer out of the pool so nothing draws over what may be handed out
    /// again.
    frame: ffmpeg::frame::Video,
}

impl Composed {
    fn matches(&self, inputs: &[InputSnapshot], texts: &[TextSnapshot]) -> bool {
        self.inputs.len() == inputs.len()
            && self.texts.len() == texts.len()
            && std::iter::zip(&self.inputs, inputs).all(|(drawn, now)| drawn.same_as(now))
            && self.texts == texts
    }
}

/// What one dispatch draws — the shader's `Step`, at buffer 0.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Step {
    region: [i32; 4],
    image: [f32; 4],
    uv: [f32; 4],
    r: [f32; 4],
    g: [f32; 4],
    b: [f32; 4],
    blend: [f32; 4],
}

impl Step {
    fn bytes(&self) -> [u8; 112] {
        let mut bytes = [0u8; 112];
        let floats = [self.image, self.uv, self.r, self.g, self.b, self.blend];
        let words = self
            .region
            .iter()
            .map(|value| value.to_ne_bytes())
            .chain(floats.iter().flatten().map(|value| value.to_ne_bytes()));
        for (chunk, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            *chunk = word;
        }
        bytes
    }

    /// The threads to cover the region, one a pixel.
    fn threads(&self) -> (u32, u32) {
        (self.region[2] as u32, self.region[3] as u32)
    }
}

/// The layout of a picture a video layer draws.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LayerLayout {
    Nv12,
    Bgra,
}

/// One layer to draw, resolved from its snapshot.
enum Draw {
    Video {
        buffer: PixelBuffer,
        layout: LayerLayout,
        step: Step,
    },
    Text {
        mask: Arc<TextMask>,
        step: Step,
    },
}

/// The compositor's kernels.
struct Kernels {
    fill: Kernel,
    nv12: Kernel,
    bgra: Kernel,
    text: Kernel,
    /// The canvas into the output frame, in its format.
    finish: Kernel,
}

impl Kernels {
    fn new(
        gpu: &MetalGpu,
        format: VideoToolboxFrameFormat,
    ) -> std::result::Result<Self, MetalError> {
        let finish = match format {
            VideoToolboxFrameFormat::Bgra => "to_bgra",
            VideoToolboxFrameFormat::Nv12 => "to_nv12",
        };
        let mut kernels = gpu
            .kernels(
                SHADER,
                &["fill", "layer_nv12", "layer_bgra", "text", finish],
            )?
            .into_iter();
        let mut next = || kernels.next().expect("one kernel for each name");
        Ok(Self {
            fill: next(),
            nv12: next(),
            bgra: next(),
            text: next(),
            finish: next(),
        })
    }
}

/// Composites the latest frames from any number of independent VideoToolbox
/// input pipelines into one fixed-rate `Pixel::VIDEOTOOLBOX` stream on the
/// GPU, without any frame being copied — the Metal sibling of
/// [`crate::elements::SwVideoCompositor`], `D3d11VideoCompositor`,
/// `CudaVideoCompositor` and `VulkanVideoCompositor`, driving the same
/// [`VideoLayer`]/[`VideoRect`]/[`VideoFit`](video_layer::VideoFit) API.
///
/// Like the others this is a source, not a one-input filter:
/// upstream pipelines terminate at the sinks returned by
/// [`MetalVideoCompositorHandle::add_source`], while this element's own
/// pipeline drives output on its own clock, or offline by the inputs'
/// timestamps — see [`RenderMode`].
///
/// # How it draws
///
/// Each frame is one command buffer of compute dispatches on Metal: the
/// canvas is filled with the background, each layer in stacking order is
/// sampled from its picture — scaled bilinearly, cropped, converted from its
/// own colour description — and blended onto it, text layers from their
/// coverage masks, and the canvas is written into the output frame, as BGRA
/// or as NV12. The pictures are read, and the output written, through
/// textures made over the `IOSurface`s their pixel buffers are in, so no
/// pixel is copied in or out; the command buffer is waited for before the
/// frame is handed on.
///
/// Inputs are NV12 or BGRA VideoToolbox frames — what
/// [`crate::elements::VideoToolboxDecoder`], [`crate::elements::VideoToolboxUpload`],
/// `AvFoundationCaptureSource` and `ScreenCaptureKitSource` make — from any
/// [`VideoToolboxDevice`], since a pixel buffer belongs to none. A BGRA layer
/// keeps its alpha, which is blended as the other compositors blend it,
/// [`VideoLayer::premultiplied_alpha`] included. The output is what
/// `VideoToolboxEncoder` takes, in either format.
pub struct MetalVideoCompositor(SourceStage<Compositing>);

source_stage!(MetalVideoCompositor);

/// What a [`MetalVideoCompositor`] does when asked: the next frame of the
/// composition. All of its work, which the framework makes the source.
struct Compositing {
    pp_log: PpLog,
    name: Arc<str>,
    shared: Arc<CompositorShared>,
    options: VideoCompositorOptions,
    frame_index: i64,
    gpu: MetalGpu,
    /// The pool output frames come from.
    hw_frames_ctx: AvBufferRef,
    format: VideoToolboxFrameFormat,
    /// Where every layer is blended, in R, G, B, A.
    canvas: Texture,
    kernels: Kernels,
    /// Text masks on the GPU, by the address of the mask they were made
    /// from, which `set_text` replaces rather than changes.
    masks: HashMap<usize, Texture>,
    /// The last composite and what it was made from — see [`Composed`].
    composed: Option<Composed>,
    /// Reuses only the small CPU-side `AVFrame` wrapper; the pixel buffer
    /// itself comes from `hw_frames_ctx`'s own pool.
    output_pool: UnboundObjectPool<ffmpeg::frame::Video>,
    /// Where each tick is recorded, once the pipeline has handed it over.
    ticks: Option<Arc<TickCounters>>,
    /// When each live tick is due, on the clock a pause does not move —
    /// made as the first is asked for.
    schedule: Option<PeriodicSchedule>,
    /// Whether a frame went on since the schedule last moved: it moves on
    /// once that frame has been handed on, as the next is asked for.
    ticked: bool,
}

// SAFETY: the FFmpeg buffers have no thread affinity of their own, and the
// Metal objects are this element's alone, thread-safe as Metal's are, and
// touched only through `&mut self` on its single source thread.
unsafe impl Send for Compositing {}

impl MetalVideoCompositor {
    /// A compositor whose frames are made on `device`, composing in BGRA —
    /// see [`Self::with_format`].
    ///
    /// Output dimensions must be even, so the same composition can be made
    /// in NV12.
    pub fn new(
        name: impl Into<String>,
        device: &VideoToolboxDevice,
        options: VideoCompositorOptions,
    ) -> std::result::Result<(Self, MetalVideoCompositorHandle), MetalVideoCompositorError> {
        Self::with_format(name, device, options, VideoToolboxFrameFormat::Bgra)
    }

    /// The same, composing in `format`: BGRA for a picture that may be
    /// transparent where no layer drew, and the only format that takes a
    /// [`background_alpha`](VideoCompositorOptions::background_alpha) other
    /// than 255; NV12, BT.709 at limited range, for half the bytes.
    pub fn with_format(
        name: impl Into<String>,
        device: &VideoToolboxDevice,
        options: VideoCompositorOptions,
        format: VideoToolboxFrameFormat,
    ) -> std::result::Result<(Self, MetalVideoCompositorHandle), MetalVideoCompositorError> {
        crate::ensure_ffmpeg();
        validate_output_options(options, format)?;
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::MetalVideoCompositor, &name, None);
        // SAFETY: `create_frames_ctx`'s contract is a live device context,
        // which the device's own reference is for the length of the call.
        let hw_frames_ctx = unsafe {
            create_frames_ctx(
                &device.retain(),
                format.pixel(),
                options.width,
                options.height,
            )
        }
        .map_err(|error| MetalVideoCompositorError::Pool(error.to_string()))?;
        let gpu = MetalGpu::new()?;
        let canvas = gpu.texture(
            MTLPixelFormat::RGBA8Unorm,
            options.width,
            options.height,
            MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite,
            false,
        )?;
        let kernels = Kernels::new(&gpu, format)?;

        let shared = Arc::new(CompositorShared {
            inputs: Mutex::new(HashMap::new()),
            text_layers: Mutex::new(Vec::new()),
            next_input_id: AtomicU64::new(1),
            frame_rate: FrameRate::new(options.frame_rate),
            mode: options.mode,
            arrived: Bell::new(),
            next_output: AtomicI64::new(0),
            fed: AtomicBool::new(false),
        });
        pp_info!(
            pp_log: &pp_log,
            "created: {}x{}, frame_rate={}, format=VideoToolbox/{:?} on {}",
            options.width,
            options.height,
            options.frame_rate,
            format,
            gpu.device.name()
        );
        Ok((
            Self(SourceStage::new(Compositing {
                name,
                pp_log,
                shared: shared.clone(),
                options,
                frame_index: 0,
                gpu,
                hw_frames_ctx,
                format,
                canvas,
                kernels,
                masks: HashMap::new(),
                composed: None,
                output_pool: UnboundObjectPool::new(
                    OUTPUT_POOL_SIZE,
                    ffmpeg::frame::Video::empty,
                    // A wrapper that held its reference while pooled would
                    // keep a pixel buffer out of the pool for nothing.
                    release_picture,
                ),
                ticks: None,
                schedule: None,
                ticked: false,
            })),
            MetalVideoCompositorHandle {
                shared: Arc::downgrade(&shared),
            },
        ))
    }

    /// Every output frame is a VideoToolbox frame.
    pub fn format(&self) -> ffmpeg::format::Pixel {
        ffmpeg::format::Pixel::VIDEOTOOLBOX
    }

    /// What the output frames hold.
    pub fn frame_format(&self) -> VideoToolboxFrameFormat {
        self.0.inner().format
    }

    /// Returns the fixed output width in pixels.
    pub fn width(&self) -> u32 {
        self.0.inner().options.width
    }

    /// Returns the fixed output height in pixels.
    pub fn height(&self) -> u32 {
        self.0.inner().options.height
    }

    /// The output frame rate, which is what construction was given unless
    /// [`MetalVideoCompositorHandle::set_frame_rate`] has changed it since.
    pub fn frame_rate(&self) -> ffmpeg::Rational {
        self.0.inner().frame_rate()
    }

    /// The reciprocal of [`Self::frame_rate`] — output PTS advance by one
    /// tick in this base per composed frame, so this moves with the rate.
    pub fn time_base(&self) -> ffmpeg::Rational {
        self.0.inner().time_base()
    }
}

#[cfg(test)]
impl MetalVideoCompositor {
    /// What it is made of, for the tests that compose by hand.
    fn compositing(&mut self) -> &mut Compositing {
        self.0.inner_mut()
    }
}

impl Compositing {
    fn frame_rate(&self) -> ffmpeg::Rational {
        self.shared.frame_rate.get()
    }

    fn time_base(&self) -> ffmpeg::Rational {
        self.frame_rate().invert()
    }

    fn snapshots(&self) -> Vec<InputSnapshot> {
        let inputs: Vec<_> = self
            .shared
            .inputs
            .lock()
            .unwrap()
            .values()
            .cloned()
            .collect();
        inputs
            .into_iter()
            .map(|input| InputSnapshot {
                id: input.id,
                layer: *input.layer.lock().unwrap(),
                frame: input.latest_frame.load_full(),
            })
            .collect()
    }

    fn text_snapshots(&self) -> Vec<TextSnapshot> {
        self.shared
            .text_layers
            .lock()
            .unwrap()
            .iter()
            .map(|(_, state)| TextSnapshot {
                mask: state.mask.load_full(),
                x: state.x.load(Ordering::Relaxed),
                y: state.y.load(Ordering::Relaxed),
                opacity: state.opacity.load(Ordering::Relaxed),
                visible: state.visible.load(Ordering::Relaxed),
                color: state.color,
                z_index: state.z_index.load(Ordering::Relaxed),
            })
            .collect()
    }

    /// Hands out the frame already composed, under this tick's timestamp.
    fn repeat_frame(
        &mut self,
    ) -> std::result::Result<UnboundObjectPoolRef<ffmpeg::frame::Video>, MetalVideoCompositorError>
    {
        let mut output = self.output_pool.get();
        let composed = self
            .composed
            .as_ref()
            .expect("only reached with a previous composite");
        // SAFETY: `ptr` is the pooled wrapper's own `AVFrame`, unreferenced
        // before it is given a new one, and the source is the reference this
        // element has held since it composed that frame — both live.
        unsafe {
            let ptr = output.as_mut_ptr();
            ffi::av_frame_unref(ptr);
            let code = ffi::av_frame_ref(ptr, composed.frame.as_ptr());
            if code < 0 {
                return Err(MetalVideoCompositorError::FrameRef(code));
            }
        }
        output.set_pts(Some(self.frame_index));
        crate::buffer::set_time_base(&mut output, self.time_base());
        self.frame_index += 1;
        Ok(output)
    }

    /// Takes one frame from the output pool, saying what its colour is.
    fn output_frame(
        &mut self,
    ) -> std::result::Result<UnboundObjectPoolRef<ffmpeg::frame::Video>, MetalVideoCompositorError>
    {
        let mut output = self.output_pool.get();
        // SAFETY: `ptr` is the pooled wrapper's own `AVFrame`; the unref
        // hands its previous pixel buffer back to the pool before a new one
        // is taken from it, which this element holds for its life.
        unsafe {
            let ptr = output.as_mut_ptr();
            ffi::av_frame_unref(ptr);
            let code = ffi::av_hwframe_get_buffer(self.hw_frames_ctx.as_ptr(), ptr, 0);
            if code < 0 {
                return Err(MetalVideoCompositorError::FrameGet(code));
            }
        }
        // What the canvas is, so a download, a scaler or an encoder reads it
        // right: BT.709, as RGB at full range or NV12 at limited.
        match self.format {
            VideoToolboxFrameFormat::Nv12 => ColorDescription::BT709_LIMITED.describe(&mut output),
            VideoToolboxFrameFormat::Bgra => ColorDescription {
                space: ffmpeg::color::Space::RGB,
                range: ffmpeg::color::Range::JPEG,
                ..ColorDescription::BT709_LIMITED
            }
            .describe(&mut output),
        }
        Ok(output)
    }

    /// Every snapshot resolved into what to draw, in stacking order: by
    /// `z_index`, a video layer before a text layer of the same one.
    fn draws(
        &self,
        snapshots: &[InputSnapshot],
        texts: &[TextSnapshot],
    ) -> std::result::Result<Vec<Draw>, MetalVideoCompositorError> {
        let (canvas_width, canvas_height) = (self.options.width, self.options.height);
        let mut ordered: Vec<(i32, bool, usize, Draw)> = Vec::new();
        for (index, snapshot) in snapshots.iter().enumerate() {
            let layer = snapshot.layer;
            if !layer.visible || layer.opacity == 0.0 {
                continue;
            }
            let Some(frame) = &snapshot.frame else {
                continue;
            };
            let (layout, buffer) = validate_input_frame(frame)?;
            let Some(source) =
                video_layer::source_region(layer.source, frame.width(), frame.height())
            else {
                // A crop the frame is too small for: nothing of this layer is
                // in the picture, which is not an error.
                continue;
            };
            // The region's own size, not the frame's: a crop decides what the
            // fit is fitting.
            let geometry = layer_geometry(source.width, source.height, layer.rect, layer.fit)
                .map_err(layer_error)?;
            let Some(region) = clipped_region(&geometry, canvas_width, canvas_height) else {
                continue;
            };
            let rows = if layout == LayerLayout::Nv12 {
                ColorDescription::of(frame).yuv_to_rgb_rows(frame.height())
            } else {
                [[0.0; 4]; 3]
            };
            let premultiplied = layout == LayerLayout::Bgra && layer.premultiplied_alpha;
            // Against the pixel buffer's own size, which a decoder's may
            // exceed the picture's by its padding.
            let (image_width, image_height) = buffer.size();
            let step = Step {
                region,
                image: [
                    geometry.image_x as f32,
                    geometry.image_y as f32,
                    geometry.image_width as f32,
                    geometry.image_height as f32,
                ],
                uv: [
                    source.width as f32 / image_width as f32,
                    source.height as f32 / image_height as f32,
                    source.x as f32 / image_width as f32,
                    source.y as f32 / image_height as f32,
                ],
                r: rows[0],
                g: rows[1],
                b: rows[2],
                blend: [layer.opacity, f32::from(u8::from(premultiplied)), 0.0, 0.0],
            };
            ordered.push((
                layer.z_index,
                false,
                index,
                Draw::Video {
                    buffer,
                    layout,
                    step,
                },
            ));
        }
        for (index, text) in texts.iter().enumerate() {
            let opacity = f32::from_bits(text.opacity);
            if !text.visible || opacity <= 0.0 {
                continue;
            }
            let Some(mask) = &text.mask else {
                continue;
            };
            let left = i64::from(text.x).max(0);
            let top = i64::from(text.y).max(0);
            let right = (i64::from(text.x) + i64::from(mask.width)).min(i64::from(canvas_width));
            let bottom = (i64::from(text.y) + i64::from(mask.height)).min(i64::from(canvas_height));
            if left >= right || top >= bottom {
                continue;
            }
            let colour = |value: u8| f32::from(value) / 255.0;
            let step = Step {
                region: [
                    left as i32,
                    top as i32,
                    (right - left) as i32,
                    (bottom - top) as i32,
                ],
                image: [text.x as f32, text.y as f32, 0.0, 0.0],
                r: [
                    colour(text.color.red),
                    colour(text.color.green),
                    colour(text.color.blue),
                    1.0,
                ],
                blend: [opacity, 0.0, 0.0, 0.0],
                ..Step::default()
            };
            ordered.push((
                text.z_index,
                true,
                index,
                Draw::Text {
                    mask: Arc::clone(mask),
                    step,
                },
            ));
        }
        ordered.sort_by_key(|&(z_index, is_text, index, _)| (z_index, is_text, index));
        Ok(ordered.into_iter().map(|(_, _, _, draw)| draw).collect())
    }

    fn compose_frame(
        &mut self,
    ) -> std::result::Result<UnboundObjectPoolRef<ffmpeg::frame::Video>, MetalVideoCompositorError>
    {
        let mut snapshots = self.snapshots();
        snapshots.sort_by(|left, right| {
            left.layer
                .z_index
                .cmp(&right.layer.z_index)
                .then_with(|| left.id.cmp(&right.id))
        });
        let texts = self.text_snapshots();

        // Nothing moved and no input produced a frame, so this tick's
        // picture is the one already composed — see [`Composed`].
        if self
            .composed
            .as_ref()
            .is_some_and(|composed| composed.matches(&snapshots, &texts))
        {
            return self.repeat_frame();
        }

        let draws = self.draws(&snapshots, &texts)?;
        // A mask no layer shows any more was last drawn by a command buffer
        // that has been waited for.
        let shown: HashSet<usize> = texts
            .iter()
            .filter_map(|text| text.mask.as_ref().map(|mask| Arc::as_ptr(mask) as usize))
            .collect();
        self.masks.retain(|key, _| shown.contains(key));

        let mut output = self.output_frame()?;
        self.draw_into(&draws, &output)?;

        // Held so the next tick can tell whether it has anything to draw,
        // and so the pixel buffer it may hand out again stays out of the pool.
        let mut kept = ffmpeg::frame::Video::empty();
        // SAFETY: both are live `AVFrame`s, so this adds a reference to the
        // frame just composed rather than copying it.
        let code = unsafe { ffi::av_frame_ref(kept.as_mut_ptr(), output.as_ptr()) };
        if code < 0 {
            return Err(MetalVideoCompositorError::FrameRef(code));
        }
        self.composed = Some(Composed {
            inputs: snapshots,
            texts,
            frame: kept,
        });

        output.set_pts(Some(self.frame_index));
        crate::buffer::set_time_base(&mut output, self.time_base());
        self.frame_index += 1;
        Ok(output)
    }

    /// Draws `draws` into `output` — the background, every layer, and the
    /// canvas into the frame — and waits for the GPU to have done it.
    fn draw_into(
        &mut self,
        draws: &[Draw],
        output: &ffmpeg::frame::Video,
    ) -> std::result::Result<(), MetalVideoCompositorError> {
        // The textures every dispatch reads, made first, so a failure leaves
        // nothing half encoded; each lives until the pass has finished.
        let read = MTLTextureUsage::ShaderRead;
        let mut layers: Vec<Vec<Texture>> = Vec::with_capacity(draws.len());
        for draw in draws {
            layers.push(match draw {
                Draw::Video {
                    buffer,
                    layout: LayerLayout::Nv12,
                    ..
                } => vec![
                    self.gpu.plane(buffer, 0, MTLPixelFormat::R8Unorm, read)?,
                    self.gpu.plane(buffer, 1, MTLPixelFormat::RG8Unorm, read)?,
                ],
                Draw::Video {
                    buffer,
                    layout: LayerLayout::Bgra,
                    ..
                } => vec![
                    self.gpu
                        .plane(buffer, 0, MTLPixelFormat::BGRA8Unorm, read)?,
                ],
                Draw::Text { mask, .. } => {
                    let key = Arc::as_ptr(mask) as usize;
                    if !self.masks.contains_key(&key) {
                        let texture = self.gpu.texture(
                            MTLPixelFormat::R8Unorm,
                            mask.width,
                            mask.height,
                            read,
                            true,
                        )?;
                        write_texture(
                            &texture,
                            &mask.coverage,
                            mask.width as usize,
                            mask.width,
                            mask.height,
                        );
                        self.masks.insert(key, texture);
                    }
                    vec![self.masks[&key].clone()]
                }
            });
        }
        let output_buffer =
            PixelBuffer::of_frame(output).expect("a frame of this element's own pool");
        let write = MTLTextureUsage::ShaderWrite;
        let targets = match self.format {
            VideoToolboxFrameFormat::Bgra => {
                vec![
                    self.gpu
                        .plane(&output_buffer, 0, MTLPixelFormat::BGRA8Unorm, write)?,
                ]
            }
            VideoToolboxFrameFormat::Nv12 => vec![
                self.gpu
                    .plane(&output_buffer, 0, MTLPixelFormat::R8Unorm, write)?,
                self.gpu
                    .plane(&output_buffer, 1, MTLPixelFormat::RG8Unorm, write)?,
            ],
        };

        let mut pass: Pass = self.gpu.pass()?;
        let (width, height) = (self.options.width, self.options.height);
        let background = Step {
            region: [0, 0, width as i32, height as i32],
            r: [
                f32::from(self.options.background.red) / 255.0,
                f32::from(self.options.background.green) / 255.0,
                f32::from(self.options.background.blue) / 255.0,
                f32::from(self.options.background_alpha) / 255.0,
            ],
            ..Step::default()
        };
        pass.dispatch(
            &self.kernels.fill,
            &[&self.canvas],
            Some(&background.bytes()),
            background.threads(),
        );
        for (draw, textures) in draws.iter().zip(&layers) {
            let (kernel, step) = match draw {
                Draw::Video {
                    layout: LayerLayout::Nv12,
                    step,
                    ..
                } => (&self.kernels.nv12, step),
                Draw::Video {
                    layout: LayerLayout::Bgra,
                    step,
                    ..
                } => (&self.kernels.bgra, step),
                Draw::Text { step, .. } => (&self.kernels.text, step),
            };
            let bound: Vec<&Texture> = std::iter::once(&self.canvas).chain(textures).collect();
            pass.dispatch(kernel, &bound, Some(&step.bytes()), step.threads());
        }
        let bound: Vec<&Texture> = std::iter::once(&self.canvas).chain(&targets).collect();
        let threads = match self.format {
            VideoToolboxFrameFormat::Bgra => (width, height),
            // One thread a 2x2 block.
            VideoToolboxFrameFormat::Nv12 => (width / 2, height / 2),
        };
        pass.dispatch(&self.kernels.finish, &bound, None, threads);
        pass.finish()?;
        drop(layers);
        drop(targets);
        Ok(())
    }

    /// This tick's frame, with what composing it took recorded.
    fn make_frame(&mut self) -> std::result::Result<MediaBuffer, MetalVideoCompositorError> {
        let composing = Instant::now();
        let output = self.compose_frame()?;
        if let Some(ticks) = &self.ticks {
            ticks.made(composing.elapsed());
        }
        Ok(MediaBuffer::Video(Arc::new(output)))
    }
}

impl Element for Compositing {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::MetalVideoCompositor
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }

    fn attach_context(&mut self, context: &Arc<Context>) {
        self.ticks = Some(context.source_ticks());
    }
}

impl Source for Compositing {
    fn is_live(&self) -> bool {
        self.options.mode.is_live()
    }

    fn output_contract(&self) -> OutputContract {
        OutputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::VideoToolbox)
                .with_layouts(self.format.layouts()),
        )
    }

    fn produce(&mut self, wait: &mut Wait<'_>) -> Result<Produced> {
        match self.options.mode {
            RenderMode::Live => self.produce_live(wait),
            RenderMode::Offline { end } => produce_offline(self, wait, end),
        }
    }
}

impl Compositing {
    /// This tick's frame, once it is due at its own rate — see
    /// [`RenderMode::Live`] and the software compositor's own.
    fn produce_live(&mut self, wait: &mut Wait<'_>) -> Result<Produced> {
        let now = wait.now();
        // Followed here rather than at construction, so a rate set while
        // this is running is kept from the next tick on.
        let interval = self.shared.frame_rate.interval();
        let schedule = self
            .schedule
            .get_or_insert_with(|| PeriodicSchedule::new(interval, now));
        // Moved on only once the last frame has been handed on: a push a
        // slow downstream holds is part of its tick, and the deadlines it
        // overran are ticks missed.
        let missed = if std::mem::take(&mut self.ticked) {
            schedule.advance_after_tick(now)
        } else {
            0
        };
        let rate_changed = schedule.interval() != interval;
        if rate_changed {
            schedule.set_interval(interval, now);
        }
        let due = now + schedule.remaining(now);
        if let Some(ticks) = &self.ticks {
            ticks.missed(missed);
        }
        if rate_changed {
            pp_info!(self, "frame rate is now {}", self.frame_rate());
        }
        if !wait.until(due.min(now + RATE_POLL_INTERVAL)) || wait.now() < due {
            return Ok(Produced::Nothing);
        }
        let frame = self.make_frame()?;
        self.ticked = true;
        Ok(Produced::Buffer(frame))
    }
}

impl TimedInput for VideoInput {
    fn timed(&self) -> Option<&TimedFeed> {
        self.timed.as_ref()
    }

    fn show(&self, picture: Option<Picture>) {
        self.latest_frame.store(picture);
    }
}

impl OfflineCompositor for Compositing {
    type Input = VideoInput;

    fn inputs(&self) -> Vec<Arc<VideoInput>> {
        self.shared
            .inputs
            .lock()
            .unwrap()
            .values()
            .cloned()
            .collect()
    }

    fn frame_index(&self) -> i64 {
        self.frame_index
    }

    fn output_time(&self, index: i64) -> MediaTime {
        self.shared.output_time(index)
    }

    fn fed(&self) -> bool {
        self.shared.fed.load(Ordering::Acquire)
    }

    fn arrived(&self) -> Bell {
        self.shared.arrived.clone()
    }

    fn making(&self, index: i64) {
        self.shared.next_output.store(index, Ordering::Release);
    }

    fn draw(&mut self) -> Result<MediaBuffer> {
        Ok(self.make_frame()?)
    }
}

impl Drop for Compositing {
    fn drop(&mut self) {
        pp_info!(self, "dropped: releasing the frames context");
    }
}

/// The part of the canvas a layer's scaled picture covers: the picture
/// clipped to its rectangle and to the canvas, as `[x, y, width, height]`,
/// or `None` where nothing of it is left.
fn clipped_region(
    geometry: &LayerGeometry,
    canvas_width: u32,
    canvas_height: u32,
) -> Option<[i32; 4]> {
    let clip_left = i64::from(geometry.clip.x).max(0);
    let clip_top = i64::from(geometry.clip.y).max(0);
    let clip_right =
        (i64::from(geometry.clip.x) + i64::from(geometry.clip.width)).min(i64::from(canvas_width));
    let clip_bottom = (i64::from(geometry.clip.y) + i64::from(geometry.clip.height))
        .min(i64::from(canvas_height));
    let left = geometry.image_x.max(clip_left);
    let top = geometry.image_y.max(clip_top);
    let right = (geometry.image_x + i64::from(geometry.image_width)).min(clip_right);
    let bottom = (geometry.image_y + i64::from(geometry.image_height)).min(clip_bottom);
    (left < right && top < bottom).then(|| {
        [
            left as i32,
            top as i32,
            (right - left) as i32,
            (bottom - top) as i32,
        ]
    })
}

/// What `frame` holds and the pixel buffer it is in, where it is a picture
/// this compositor draws: an NV12 or BGRA VideoToolbox frame no larger than
/// its buffer.
fn validate_input_frame(
    frame: &ffmpeg::frame::Video,
) -> std::result::Result<(LayerLayout, PixelBuffer), MetalVideoCompositorError> {
    let layout = match sw_format_of(frame) {
        Ok(ffmpeg::format::Pixel::NV12) => LayerLayout::Nv12,
        Ok(ffmpeg::format::Pixel::BGRA) => LayerLayout::Bgra,
        Ok(other) => return Err(MetalVideoCompositorError::UnsupportedLayout(other)),
        Err(NotVideoToolbox::Format(format)) => {
            return Err(MetalVideoCompositorError::NotVideoToolbox(format));
        }
        Err(NotVideoToolbox::NoFramesContext) => {
            return Err(MetalVideoCompositorError::NoFramesContext);
        }
    };
    let buffer = PixelBuffer::of_frame(frame)
        .ok_or(MetalVideoCompositorError::NotVideoToolbox(frame.format()))?;
    let (buffer_width, buffer_height) = buffer.size();
    if frame.width() == 0
        || frame.height() == 0
        || frame.width() > buffer_width
        || frame.height() > buffer_height
    {
        return Err(MetalVideoCompositorError::InvalidInputDimensions {
            width: frame.width(),
            height: frame.height(),
        });
    }
    Ok((layout, buffer))
}

fn validate_output_options(
    options: VideoCompositorOptions,
    format: VideoToolboxFrameFormat,
) -> std::result::Result<(), MetalVideoCompositorError> {
    if options.width < 2
        || options.height < 2
        || !options.width.is_multiple_of(2)
        || !options.height.is_multiple_of(2)
        || options.width > MAX_DIMENSION
        || options.height > MAX_DIMENSION
    {
        return Err(MetalVideoCompositorError::InvalidOutputDimensions {
            width: options.width,
            height: options.height,
        });
    }
    if options.frame_rate.numerator() <= 0 || options.frame_rate.denominator() <= 0 {
        return Err(MetalVideoCompositorError::InvalidFrameRate(
            options.frame_rate,
        ));
    }
    // NV12 has nowhere to keep it — refused rather than quietly made opaque.
    if options.background_alpha != 255 && format == VideoToolboxFrameFormat::Nv12 {
        return Err(MetalVideoCompositorError::TranslucentBackground(
            options.background_alpha,
        ));
    }
    Ok(())
}

/// Thin adapters over the shared, backend-agnostic checks in
/// [`super::super::video_layer`].
fn validate_layer(layer: VideoLayer) -> std::result::Result<(), MetalVideoCompositorError> {
    video_layer::validate_layer(layer).map_err(layer_error)?;
    validate_opacity(layer.opacity)
}

fn validate_rect(rect: VideoRect) -> std::result::Result<(), MetalVideoCompositorError> {
    video_layer::validate_rect(rect).map_err(layer_error)
}

fn validate_opacity(opacity: f32) -> std::result::Result<(), MetalVideoCompositorError> {
    video_layer::validate_opacity(opacity).map_err(layer_error)
}

fn layer_error(error: VideoLayerError) -> MetalVideoCompositorError {
    match error {
        VideoLayerError::InvalidDimensions { width, height } => {
            MetalVideoCompositorError::InvalidLayerDimensions { width, height }
        }
        VideoLayerError::InvalidOpacity(opacity) => {
            MetalVideoCompositorError::InvalidOpacity(opacity)
        }
        VideoLayerError::InvalidInputDimensions { width, height } => {
            MetalVideoCompositorError::InvalidInputDimensions { width, height }
        }
        VideoLayerError::ScaledLayerTooLarge { width, height } => {
            MetalVideoCompositorError::ScaledLayerTooLarge { width, height }
        }
        VideoLayerError::InvalidSourceRegion { width, height } => {
            MetalVideoCompositorError::InvalidSourceRegion { width, height }
        }
    }
}

#[cfg(test)]
mod tests;

super::super::control::compositor_control!(
    MetalVideoCompositorHandle,
    MetalVideoLayerHandle,
    MetalTextLayerHandle
);
