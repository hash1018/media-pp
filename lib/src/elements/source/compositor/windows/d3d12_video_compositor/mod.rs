use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use arc_swap::ArcSwapOption;
use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;
use windows::{
    Win32::Graphics::{
        Direct3D12::*,
        Dxgi::Common::{DXGI_FORMAT_NV12, DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8G8_UNORM},
    },
    core::{Interface, s},
};

use crate::pp_log::{PpLog, pp_error, pp_info};
use crate::rate::FrameRate;

use super::super::sw_video_compositor::VideoCompositorOptions;
use super::super::timed_inputs::{
    OfflineCompositor, Picture, TimedFeed, TimedInput, Untimed, produce_offline,
};
use super::super::video_layer::{
    self, MAX_DIMENSION, VideoInputId, VideoLayer, VideoLayerError, VideoRect, layer_geometry,
};
use crate::elements::source::render_mode::{MediaTime, RenderMode};
use crate::playback_state::Bell;
use crate::{
    buffer::{MediaBuffer, picture_id, release_picture},
    color::ColorDescription,
    contract::{
        InputContract, MediaKind, MemoryDomain, OutputContract, PixelLayoutSet, PortContract,
    },
    control::ControlMsg,
    element::{
        Context, Element, ElementType, Flow, Produced, RawSink, Source, SourceStage, Wait,
        element_pp_log,
    },
    error::Result,
    platform::ffmpeg::AvBufferRef,
    platform::windows::{
        d3d12_compute::{
            Commands, Kernel, copy_plane, plane_view, plane_write_view, transition, uav_barrier,
            writable_texture,
        },
        d3d12_gpu::D3d12Gpu,
        d3d12va::{
            create_hw_device_ctx, create_hw_frames_ctx, d3d12va_texture, set_d3d12va_fence_value,
        },
    },
    pool::{UnboundObjectPool, UnboundObjectPoolRef},
    produce::source_stage,
    schedule::LiveSchedule,
    stats::TickCounters,
    stream::StreamEvent,
};

mod text_handle;
mod video_handle;

pub use text_handle::D3d12TextLayerHandle;
use text_handle::{D3d12Mask, TextLayerState};
pub use video_handle::D3d12VideoLayerHandle;

const COMPOSE: &[u8] = include_bytes!("../../../../../shaders/d3d12/compose.hlsl");

/// How long a live compositor waits at a time before looking at its rate
/// again — see the CUDA compositor's constant of the same name.
const RATE_POLL_INTERVAL: Duration = Duration::from_millis(5);

/// What text layers' masks are uploaded through, made at the first and
/// shared by every text layer's handle.
type Uploads = Arc<Mutex<Option<Commands>>>;

/// The canvas's two planes' views, ahead of each layer's two.
const CANVAS_VIEWS: usize = 2;

/// Errors specific to [`D3d12VideoCompositor`].
#[derive(Debug, ThisError)]
pub enum D3d12VideoCompositorError {
    /// A D3D12 call failed, or a shader would not compile.
    #[error(transparent)]
    D3d12(#[from] windows::core::Error),

    /// The GPU cannot load what it mixes into through a typed UAV, which
    /// drawing a translucent layer or text takes.
    #[error("the GPU cannot load R8 and R8G8 through a typed UAV")]
    NoTypedUavLoads,

    /// FFmpeg could not make a D3D12VA device context of the device.
    #[error("FFmpeg could not create a D3D12VA device context (code {0})")]
    HwDeviceInit(i32),

    /// FFmpeg could not make the pool output frames come from.
    #[error("FFmpeg could not create the D3D12VA frames context (code {0})")]
    HwFramesAlloc(i32),

    /// FFmpeg could not take a frame from the output pool.
    #[error("failed to take an output frame from the D3D12 pool (code {0})")]
    HwFrameGet(i32),

    /// FFmpeg could not take a second reference to the frame last composed,
    /// which is how an unchanged picture is handed on again.
    #[error("failed to reference the previous composite (code {0})")]
    FrameRef(i32),

    /// A background that is not opaque: the canvas is NV12, which has no
    /// alpha.
    #[error(
        "background_alpha {0} is not supported: D3d12VideoCompositor composes in NV12, which has no alpha"
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

    /// A layer destination rectangle is zero-sized or exceeds the safety
    /// limit.
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

    /// A layer's source region is empty.
    #[error("layer source region has invalid dimensions {width}x{height}")]
    InvalidSourceRegion {
        /// Region width as given.
        width: u32,
        /// Region height as given.
        height: u32,
    },

    /// An input frame reports a zero width or height.
    #[error("input frame has invalid dimensions {width}x{height}")]
    InvalidInputDimensions {
        /// Invalid input width in pixels.
        width: u32,
        /// Invalid input height in pixels.
        height: u32,
    },

    /// Aspect-ratio fitting would make an image above the safety limit.
    #[error("scaled layer would exceed {MAX_DIMENSION}px: {width}x{height}")]
    ScaledLayerTooLarge {
        /// Computed scaled width in pixels.
        width: u32,
        /// Computed scaled height in pixels.
        height: u32,
    },

    /// A runtime layer handle refers to an input that has been removed or
    /// replaced.
    #[error("the compositor input has been removed")]
    SourceRemoved,

    /// An input picture that is not an NV12 D3D12 picture.
    #[error("D3d12VideoCompositor draws NV12 D3D12 pictures, not {0}")]
    UnsupportedPicture(String),

    /// An input picture made on another device than the compositor's.
    #[error("the picture was made on another D3D12 device than the compositor's")]
    DeviceMismatch,

    /// An input picture larger than the texture it says it is in.
    #[error("a {picture:?} picture in a {texture:?} texture")]
    PictureOutsideTexture {
        /// The picture's width and height.
        picture: (u32, u32),
        /// Its texture's.
        texture: (u32, u32),
    },

    /// An input sink received a buffer other than decoded video.
    #[error(
        "D3d12VideoCompositorInputSink only accepts decoded Video frames, got a {0}; link it after a decoder or upload"
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

    /// The compositor this handle belongs to has stopped.
    #[error("the compositor has stopped")]
    Stopped,

    /// An offline compositor's rate cannot change while it runs — see
    /// [`crate::elements::SwVideoCompositorError::FixedFrameRate`].
    #[error("an offline compositor's frame rate is fixed at construction")]
    FixedFrameRate,

    /// An offline compositor was given a frame it cannot place in time.
    #[error("an offline compositor's input frame has no {0}")]
    UntimedFrame(&'static str),
}

pub(crate) struct VideoInput {
    id: VideoInputId,
    /// The latest picture, replaced without taking the layer's lock — see
    /// the CUDA compositor's field of the same name.
    latest_frame: ArcSwapOption<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    layer: Mutex<VideoLayer>,
    /// An offline compositor's input fed through a sink.
    timed: Option<TimedFeed>,
}

pub(crate) struct CompositorShared {
    inputs: Mutex<HashMap<Arc<str>, Arc<VideoInput>>>,
    /// Text layers, drawn among the video layers by their `z_index`.
    text_layers: Mutex<Vec<(Arc<str>, Arc<TextLayerState>)>>,
    next_input_id: AtomicU64,
    frame_rate: Arc<FrameRate>,
    /// The device every input picture has to be from, and that a text
    /// layer's mask is uploaded to.
    gpu: D3d12Gpu,
    /// What a text layer's mask is uploaded through, made at the first.
    uploads: Uploads,
    mode: RenderMode,
    /// Rung as anything an offline compositor waits for changes.
    arrived: Bell,
    /// The output frame an offline compositor is making next.
    next_output: AtomicI64,
    /// Whether an offline compositor has ever had an input fed through a
    /// sink.
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
/// the D3D12 sibling of [`crate::elements::SwVideoCompositorHandle`].
#[derive(Clone)]
pub struct D3d12VideoCompositorHandle {
    shared: Weak<CompositorShared>,
}

/// The sink and layer handle [`D3d12VideoCompositorHandle::add_source`]
/// returns — see [`crate::elements::CompositorInput`].
pub type D3d12VideoCompositorInput = crate::elements::CompositorInput<D3d12VideoLayerHandle>;

impl D3d12VideoCompositorHandle {
    /// Registers an input and returns its terminal Sink plus independent
    /// runtime layer control. Reusing `name` replaces the old registration;
    /// the replaced sink and layer handle can no longer affect this
    /// compositor.
    ///
    /// Live, each input keeps only its latest frame, drawn at every tick of
    /// the compositor's own rate; offline it is read by its timestamps —
    /// see [`crate::elements::SwVideoCompositorHandle::add_source`].
    pub fn add_source(
        &self,
        name: impl Into<String>,
        layer: VideoLayer,
    ) -> std::result::Result<D3d12VideoCompositorInput, D3d12VideoCompositorError> {
        let layer = self.register(name, layer, true)?;
        Ok(D3d12VideoCompositorInput {
            sink: crate::element::BoxSink::new(D3d12VideoCompositorInputSink {
                pp_log: element_pp_log(ElementType::D3d12VideoCompositor, &layer.name, None),
                name: layer.name.clone(),
                id: layer.id,
                shared: self.shared.clone(),
                input: layer.input.clone(),
                told_stopped: false,
            }),
            layer,
        })
    }

    /// Registers an input whose picture is set through its handle —
    /// [`D3d12VideoLayerHandle::set_frame`] — rather than a pipeline.
    pub fn add_layer(
        &self,
        name: impl Into<String>,
        layer: VideoLayer,
    ) -> std::result::Result<D3d12VideoLayerHandle, D3d12VideoCompositorError> {
        self.register(name, layer, false)
    }

    /// Registers an input, fed through a sink where `fed` says so.
    fn register(
        &self,
        name: impl Into<String>,
        layer: VideoLayer,
        fed: bool,
    ) -> std::result::Result<D3d12VideoLayerHandle, D3d12VideoCompositorError> {
        validate_layer(layer)?;
        let Some(shared) = self.shared.upgrade() else {
            return Err(D3d12VideoCompositorError::Stopped);
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
        let replaced = shared
            .inputs
            .lock()
            .unwrap()
            .insert(name.clone(), input.clone());
        // Dropped outside the lock: its last picture goes back to its pool.
        drop(replaced);
        shared.arrived.ring();
        Ok(D3d12VideoLayerHandle {
            id,
            name,
            input: Arc::downgrade(&input),
            shared: self.shared.clone(),
        })
    }

    /// Removes the registration under `name`, if any.
    pub fn remove_source(&self, name: &str) {
        if let Some(shared) = self.shared.upgrade() {
            let removed = shared.inputs.lock().unwrap().remove(name);
            drop(removed);
            shared.arrived.ring();
        }
    }

    /// Changes the rate this compositor emits at, from the next tick — the
    /// same contract, and caveat, as
    /// `CudaVideoCompositorHandle::set_frame_rate`'s.
    pub fn set_frame_rate(
        &self,
        frame_rate: ffmpeg::Rational,
    ) -> std::result::Result<(), D3d12VideoCompositorError> {
        let shared = self
            .shared
            .upgrade()
            .ok_or(D3d12VideoCompositorError::Stopped)?;
        if !shared.mode.is_live() {
            return Err(D3d12VideoCompositorError::FixedFrameRate);
        }
        shared
            .frame_rate
            .set(frame_rate)
            .map_err(|_| D3d12VideoCompositorError::InvalidFrameRate(frame_rate))
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
/// the compositor emits on its own clock.
pub struct D3d12VideoCompositorInputSink {
    pp_log: PpLog,
    name: Arc<str>,
    id: VideoInputId,
    shared: Weak<CompositorShared>,
    input: Weak<VideoInput>,
    /// Whether this sink has said once that its compositor stopped: a
    /// pipeline still feeding one hears it once, not for every buffer.
    told_stopped: bool,
}

impl D3d12VideoCompositorInputSink {
    /// What a buffer reaching a stopped compositor gets: the error the
    /// first time, so a pipeline still feeding it hears why, and nothing
    /// after, rather than the same error once a buffer.
    fn stopped(&mut self) -> Result<()> {
        if std::mem::replace(&mut self.told_stopped, true) {
            return Ok(());
        }
        Err(D3d12VideoCompositorError::Stopped.into())
    }

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
            let removed = inputs.remove(&self.name);
            drop(inputs);
            drop(removed);
            shared.arrived.ring();
        }
    }
}

impl Element for D3d12VideoCompositorInputSink {
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
        ElementType::D3d12VideoCompositor
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl RawSink for D3d12VideoCompositorInputSink {
    /// Every layer is drawn on the device, from NV12 pictures.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::D3d12)
                .with_layouts(PixelLayoutSet::NV12),
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
                let Some(shared) = self.shared.upgrade() else {
                    return self.stopped();
                };
                // Checked here, so a misconfigured input names itself.
                validate_input_frame(&frame, shared.gpu.device())
                    .inspect_err(|error| pp_error!(self, "{error}"))?;
                let Some(input) = self.input.upgrade() else {
                    return Ok(());
                };
                let Some(timed) = &input.timed else {
                    input.latest_frame.store(Some(frame.into_payload()));
                    return Ok(());
                };
                timed.push(frame.into_payload()).map_err(|untimed| {
                    D3d12VideoCompositorError::UntimedFrame(match untimed {
                        Untimed::NoTimestamp => "timestamp",
                        Untimed::NoTimeBase => "time base",
                    })
                    .into()
                })
            }
            other => {
                pp_error!(self, "unsupported buffer: {}", other.kind());
                Err(D3d12VideoCompositorError::UnsupportedBuffer(other.kind()).into())
            }
        }
    }

    /// Its stream's end — see
    /// `CudaVideoCompositorInputSink`'s.
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
        if matches!(msg, ControlMsg::Stop) {
            self.detach();
        }
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
    /// What it is registered as, to name it by where it cannot be drawn.
    name: Arc<str>,
    layer: VideoLayer,
    frame: Option<Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>>,
}

impl InputSnapshot {
    /// Same input, in the same place, showing the same pixels — compared by
    /// the picture's identity, as the CUDA compositor's are.
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

/// One text layer as the compositor found it.
#[derive(Clone)]
struct TextSnapshot {
    mask: Option<Arc<D3d12Mask>>,
    x: i32,
    y: i32,
    /// `f32` bits, compared rather than interpreted.
    opacity: u32,
    visible: bool,
    color: crate::color::Color,
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
                (Some(drawn), Some(now)) => Arc::ptr_eq(drawn, now),
                (None, None) => true,
                _ => false,
            }
    }
}

/// What the last composite was made from, and what it produced: a tick that
/// finds everything as it was hands that frame on again — see the CUDA
/// compositor's `Composed`.
struct Composed {
    inputs: Vec<InputSnapshot>,
    texts: Vec<TextSnapshot>,
    /// A reference to the frame last handed on, which keeps its texture out
    /// of the output pool.
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

/// Composites the latest pictures of any number of D3D12 input pipelines
/// into one fixed-rate NV12 D3D12 stream, on the GPU — the D3D12 sibling of
/// [`crate::elements::SwVideoCompositor`] and
/// `CudaVideoCompositor`, driving the same
/// [`VideoLayer`]/[`VideoRect`]/[`VideoFit`](video_layer::VideoFit) API, on
/// any vendor's GPU.
///
/// A source, not a filter: upstream pipelines end at the sinks
/// [`D3d12VideoCompositorHandle::add_source`] returns, and this element's
/// own pipeline emits on its own clock — live, at its rate; offline, an
/// output frame each time every input has what it shows.
///
/// # How it draws
///
/// Each layer is drawn by compute shaders of this crate's own
/// (`shaders/d3d12/compose.hlsl`), a pass a plane: every canvas pixel the
/// layer covers samples its picture bilinearly where it falls — which
/// scales, crops and places the layer in one step — converts it from the
/// picture's colour to the canvas's, and mixes it in at the layer's
/// opacity. Text is mixed in its colour through its coverage mask. The
/// canvas is two planes of the element's own, copied whole into a frame of
/// its own pool once drawn, so what comes out is an ordinary D3D12VA
/// frame, waited for through its fence as any is.
///
/// # NV12, BT.709
///
/// Inputs are NV12 D3D12 pictures of the compositor's own device, each read
/// in its own colour — BT.601, BT.709, full or limited range, as it says —
/// and the canvas is NV12 in BT.709 limited range, said so on every frame.
/// A P010 picture is refused: bring it to SDR first with
/// [`crate::elements::D3d12ToneMap`].
///
/// # Even coordinates
///
/// NV12's chroma is subsampled 2x2, so what a layer covers is aligned to
/// even pixels: drawn up to one pixel inside where it was asked for.
pub struct D3d12VideoCompositor(SourceStage<Compositing>);

source_stage!(D3d12VideoCompositor);

/// What a [`D3d12VideoCompositor`] does when asked: the next frame.
struct Compositing {
    pp_log: PpLog,
    name: Arc<str>,
    shared: Arc<CompositorShared>,
    options: VideoCompositorOptions,
    frame_index: i64,
    device: ID3D12Device,
    /// `compose.hlsl`'s `fill`, and under its root signature the rest.
    fill: Kernel,
    layer_luma: ID3D12PipelineState,
    layer_chroma: ID3D12PipelineState,
    text_luma: ID3D12PipelineState,
    text_chroma: ID3D12PipelineState,
    commands: Commands,
    /// The canvas's planes and each layer's views, shader-visible; made
    /// again, larger, for a frame with more layers.
    heap: Option<(ID3D12DescriptorHeap, usize)>,
    heap_step: usize,
    /// The canvas: luma, and chroma at half the size each way.
    canvas: (ID3D12Resource, ID3D12Resource),
    /// This element's own reference to the device its frames are made on.
    _hw_device_ctx: AvBufferRef,
    /// The pool output frames come from.
    hw_frames_ctx: AvBufferRef,
    /// The last composite and what it was made from.
    composed: Option<Composed>,
    /// Reuses only the CPU-side `AVFrame` wrapper; each texture comes from
    /// `hw_frames_ctx`.
    output_pool: UnboundObjectPool<ffmpeg::frame::Video>,
    /// The pictures the last frame was drawn from, held until the GPU has
    /// read them: let go of, a picture's texture would go back to its
    /// producer's pool, to be written again under the reading.
    reading: Vec<Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>>,
    ticks: Option<Arc<TickCounters>>,
    schedule: LiveSchedule,
    /// Where a layer it cannot draw is said, once the pipeline has handed
    /// it over.
    bus: Option<crate::bus::Bus>,
}

// SAFETY: the FFmpeg buffers have no thread affinity, and the D3D12 objects
// are free-threaded and touched only through `&mut self`, on this element's
// one source thread.
unsafe impl Send for Compositing {}

impl D3d12VideoCompositor {
    /// A compositor drawing on `gpu` — the same [`D3d12Gpu`] every input
    /// pipeline's D3D12 elements were made with; a picture of another
    /// device is refused by the input sink.
    ///
    /// Output dimensions must be even, and the background opaque.
    pub fn new(
        name: impl Into<String>,
        gpu: &D3d12Gpu,
        options: VideoCompositorOptions,
    ) -> std::result::Result<(Self, D3d12VideoCompositorHandle), D3d12VideoCompositorError> {
        crate::ensure_ffmpeg();
        validate_output_options(options)?;
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::D3d12VideoCompositor, &name, None);
        let device = gpu.device().clone();
        let mut supported = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        // SAFETY: a live device, and a correctly sized out-parameter of the
        // feature asked about.
        unsafe {
            device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS,
                (&raw mut supported).cast(),
                size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS>() as u32,
            )?;
        }
        if !supported.TypedUAVLoadAdditionalFormats.as_bool() {
            return Err(D3d12VideoCompositorError::NoTypedUavLoads);
        }
        let (width, height) = (options.width, options.height);
        // SAFETY: every call is on a live device.
        let (fill, layer_luma, layer_chroma, text_luma, text_chroma, heap_step, canvas) = unsafe {
            let fill = Kernel::new(&device, COMPOSE, s!("compose.hlsl"), s!("fill"))?;
            let sibling = |entry| fill.sibling(&device, COMPOSE, s!("compose.hlsl"), entry);
            let (layer_luma, layer_chroma, text_luma, text_chroma) = (
                sibling(s!("layer_luma"))?,
                sibling(s!("layer_chroma"))?,
                sibling(s!("text_luma"))?,
                sibling(s!("text_chroma"))?,
            );
            let heap_step = device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
                as usize;
            let canvas = (
                writable_texture(&device, DXGI_FORMAT_R8_UNORM, (width, height))?,
                writable_texture(&device, DXGI_FORMAT_R8G8_UNORM, (width / 2, height / 2))?,
            );
            (
                fill,
                layer_luma,
                layer_chroma,
                text_luma,
                text_chroma,
                heap_step,
                canvas,
            )
        };
        let commands = Commands::new(&device, gpu.queue())?;
        // SAFETY: a live device, whose context owns its own reference.
        let hw_device_ctx = unsafe { create_hw_device_ctx(&device) }
            .map_err(D3d12VideoCompositorError::HwDeviceInit)?;
        // SAFETY: a live device context, which the owned reference is.
        let hw_frames_ctx = unsafe { create_hw_frames_ctx(&hw_device_ctx, width, height, 0) }
            .map_err(D3d12VideoCompositorError::HwFramesAlloc)?;
        let shared = Arc::new(CompositorShared {
            inputs: Mutex::new(HashMap::new()),
            text_layers: Mutex::new(Vec::new()),
            next_input_id: AtomicU64::new(1),
            frame_rate: FrameRate::new(options.frame_rate),
            gpu: gpu.clone(),
            uploads: Arc::default(),
            mode: options.mode,
            arrived: Bell::new(),
            next_output: AtomicI64::new(0),
            fed: AtomicBool::new(false),
        });
        pp_info!(
            pp_log: &pp_log,
            "created: {width}x{height}, frame_rate={}, format=D3D12/NV12",
            options.frame_rate
        );
        Ok((
            Self(SourceStage::new(Compositing {
                pp_log,
                name,
                shared: shared.clone(),
                options,
                frame_index: 0,
                device,
                fill,
                layer_luma,
                layer_chroma,
                text_luma,
                text_chroma,
                commands,
                heap: None,
                heap_step,
                canvas,
                _hw_device_ctx: hw_device_ctx,
                hw_frames_ctx,
                composed: None,
                output_pool: UnboundObjectPool::new(
                    4,
                    ffmpeg::frame::Video::empty,
                    release_picture,
                ),
                reading: Vec::new(),
                ticks: None,
                schedule: LiveSchedule::default(),
                bus: None,
            })),
            D3d12VideoCompositorHandle {
                shared: Arc::downgrade(&shared),
            },
        ))
    }

    /// Every output frame is a D3D12 NV12 picture.
    pub fn format(&self) -> ffmpeg::format::Pixel {
        ffmpeg::format::Pixel::D3D12
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
    /// [`D3d12VideoCompositorHandle::set_frame_rate`] has changed it since.
    pub fn frame_rate(&self) -> ffmpeg::Rational {
        self.0.inner().frame_rate()
    }

    /// The reciprocal of [`Self::frame_rate`]: output PTS advance by one
    /// tick in this base per composed frame.
    pub fn time_base(&self) -> ffmpeg::Rational {
        self.0.inner().time_base()
    }
}

#[cfg(test)]
impl D3d12VideoCompositor {
    /// What it is made of, for the tests that compose by hand.
    fn compositing(&mut self) -> &mut Compositing {
        self.0.inner_mut()
    }
}

/// One video layer of a frame, as its draw needs it.
struct LayerDraw {
    texture: ID3D12Resource,
    constants: [u32; 28],
    region: Region,
    z_index: i32,
}

/// One text layer of a frame, as its draw needs it.
struct TextDraw {
    mask: Arc<D3d12Mask>,
    constants: [u32; 28],
    region: Region,
    z_index: i32,
}

/// What is drawn of a layer, on the canvas, in luma pixels: every one even.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Region {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

/// The constants a draw is handed, in the order `compose.hlsl` takes them.
fn constants(region: Region, floats: [[f32; 4]; 6]) -> [u32; 28] {
    let mut words = [0u32; 28];
    words[..4].copy_from_slice(&[region.x, region.y, region.width, region.height]);
    for (at, value) in floats.iter().flatten().enumerate() {
        words[4 + at] = value.to_bits();
    }
    words
}

/// `color` as the canvas holds it: BT.709 limited-range `Y, Cb, Cr`, each
/// 0 to 1.
fn canvas_colour(color: crate::color::Color) -> [f32; 3] {
    let [r, g, b] = [color.red, color.green, color.blue].map(|value| f32::from(value) / 255.0);
    let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    [
        (16.0 + 219.0 * luma) / 255.0,
        (128.0 + 224.0 * (b - luma) / 1.8556) / 255.0,
        (128.0 + 224.0 * (r - luma) / 1.5748) / 255.0,
    ]
}

/// Three rows that turn a picture's normalized `(Y, Cb, Cr, 1)` — of its
/// colour, by `rows` into R'G'B' — into the canvas's BT.709 limited-range
/// `Y, Cb, Cr`, each 0 to 1.
fn to_canvas(rows: [[f32; 4]; 3]) -> [[f32; 4]; 3] {
    let (kr, kb) = (0.2126f32, 0.0722f32);
    let kg = 1.0 - kr - kb;
    let (y_scale, c_scale) = (219.0 / 255.0, 224.0 / 255.0);
    let (y_offset, c_offset) = (16.0 / 255.0, 128.0 / 255.0);
    // Each output as weights of R', G', B' and an offset.
    let luma = [kr, kg, kb];
    let outputs = [
        ([kr * y_scale, kg * y_scale, kb * y_scale], y_offset),
        (
            [
                -luma[0] / 1.8556 * c_scale,
                -luma[1] / 1.8556 * c_scale,
                (1.0 - kb) / 1.8556 * c_scale,
            ],
            c_offset,
        ),
        (
            [
                (1.0 - kr) / 1.5748 * c_scale,
                -luma[1] / 1.5748 * c_scale,
                -luma[2] / 1.5748 * c_scale,
            ],
            c_offset,
        ),
    ];
    outputs.map(|(weights, offset)| {
        let mut row = [0.0f32, 0.0, 0.0, offset];
        for (weight, rgb) in weights.iter().zip(&rows) {
            for (out, value) in row.iter_mut().zip(rgb) {
                *out += weight * value;
            }
        }
        row
    })
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
            .iter()
            .map(|(name, input)| (name.clone(), input.clone()))
            .collect();
        inputs
            .into_iter()
            .map(|(name, input)| InputSnapshot {
                id: input.id,
                name,
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

    /// Hands out the frame already composed, under this tick's timestamp: a
    /// reference to the same texture, nothing drawn.
    fn repeat_frame(
        &mut self,
    ) -> std::result::Result<UnboundObjectPoolRef<ffmpeg::frame::Video>, D3d12VideoCompositorError>
    {
        let mut output = self.output_pool.get();
        let composed = self
            .composed
            .as_ref()
            .expect("only reached with a previous composite");
        // SAFETY: the pooled wrapper's own `AVFrame`, unreferenced before it
        // is given a new one, and the reference this element has held since
        // it composed that frame.
        unsafe {
            let ptr = output.as_mut_ptr();
            ffi::av_frame_unref(ptr);
            let code = ffi::av_frame_ref(ptr, composed.frame.as_ptr());
            if code < 0 {
                return Err(D3d12VideoCompositorError::FrameRef(code));
            }
        }
        output.set_pts(Some(self.frame_index));
        crate::buffer::set_time_base(&mut output, self.time_base());
        self.frame_index += 1;
        Ok(output)
    }

    /// One frame from the output pool.
    fn output_frame(
        &mut self,
    ) -> std::result::Result<UnboundObjectPoolRef<ffmpeg::frame::Video>, D3d12VideoCompositorError>
    {
        let mut output = self.output_pool.get();
        // SAFETY: the pooled wrapper's own `AVFrame`, its previous texture
        // handed back first; the frames context is this element's own.
        unsafe {
            let ptr = output.as_mut_ptr();
            ffi::av_frame_unref(ptr);
            let code = ffi::av_hwframe_get_buffer(self.hw_frames_ctx.as_ptr(), ptr, 0);
            if code < 0 {
                return Err(D3d12VideoCompositorError::HwFrameGet(code));
            }
        }
        // What the canvas is, said so for whatever reads it next.
        ColorDescription::BT709_LIMITED.describe(&mut output);
        Ok(output)
    }

    /// What `snapshot`'s layer draws, or `None` where it draws nothing.
    fn layer_draw(
        &self,
        snapshot: &InputSnapshot,
    ) -> std::result::Result<Option<LayerDraw>, D3d12VideoCompositorError> {
        if !snapshot.layer.visible || snapshot.layer.opacity == 0.0 {
            return Ok(None);
        }
        let Some(frame) = &snapshot.frame else {
            return Ok(None);
        };
        let input = validate_input_frame(frame, &self.device)?;
        let Some(source) =
            video_layer::source_region(snapshot.layer.source, frame.width(), frame.height())
        else {
            return Ok(None);
        };
        let geometry = layer_geometry(
            source.width,
            source.height,
            snapshot.layer.rect,
            snapshot.layer.fit,
        )
        .map_err(layer_error)?;
        let (canvas_width, canvas_height) = (self.options.width, self.options.height);
        let Some((x, width)) = axis(
            geometry.image_x,
            geometry.image_width,
            i64::from(geometry.clip.x),
            geometry.clip.width,
            canvas_width,
        ) else {
            return Ok(None);
        };
        let Some((y, height)) = axis(
            geometry.image_y,
            geometry.image_height,
            i64::from(geometry.clip.y),
            geometry.clip.height,
            canvas_height,
        ) else {
            return Ok(None);
        };
        let region = Region {
            x,
            y,
            width,
            height,
        };
        let rows = to_canvas(ColorDescription::of(frame).yuv_to_rgb_rows(frame.height()));
        let floats = [
            [
                geometry.image_x as f32,
                geometry.image_y as f32,
                source.width as f32 / geometry.image_width as f32,
                source.height as f32 / geometry.image_height as f32,
            ],
            [
                source.x as f32,
                source.y as f32,
                1.0 / input.texture_size.0 as f32,
                1.0 / input.texture_size.1 as f32,
            ],
            rows[0],
            rows[1],
            rows[2],
            [
                source.width as f32,
                source.height as f32,
                0.0,
                snapshot.layer.opacity,
            ],
        ];
        Ok(Some(LayerDraw {
            texture: input.texture,
            constants: constants(region, floats),
            region,
            z_index: snapshot.layer.z_index,
        }))
    }

    /// What `text`'s layer draws, or `None` where it draws nothing.
    fn text_draw(&self, text: &TextSnapshot) -> Option<TextDraw> {
        if !text.visible {
            return None;
        }
        let mask = text.mask.as_ref()?;
        let opacity = f32::from_bits(text.opacity);
        if opacity <= 0.0 {
            return None;
        }
        let (x, mask_x, width) = text_axis(text.x, mask.width, self.options.width)?;
        let (y, mask_y, height) = text_axis(text.y, mask.height, self.options.height)?;
        let region = Region {
            x,
            y,
            width,
            height,
        };
        let colour = canvas_colour(text.color);
        let floats = [
            [mask_x as f32, mask_y as f32, 0.0, 0.0],
            [0.0; 4],
            [0.0; 4],
            [0.0; 4],
            [0.0; 4],
            [colour[0], colour[1], colour[2], opacity],
        ];
        Some(TextDraw {
            mask: mask.clone(),
            constants: constants(region, floats),
            region,
            z_index: text.z_index,
        })
    }

    fn compose_frame(
        &mut self,
    ) -> std::result::Result<UnboundObjectPoolRef<ffmpeg::frame::Video>, D3d12VideoCompositorError>
    {
        let mut snapshots = self.snapshots();
        snapshots.sort_by(|left, right| {
            left.layer
                .z_index
                .cmp(&right.layer.z_index)
                .then_with(|| left.id.cmp(&right.id))
        });
        let texts = self.text_snapshots();
        if self
            .composed
            .as_ref()
            .is_some_and(|composed| composed.matches(&snapshots, &texts))
        {
            return self.repeat_frame();
        }

        // What the last frame queued still reads the views and its
        // pictures: it runs out before any is touched or let go of.
        self.commands.wait_idle()?;
        self.reading.clear();

        let mut layers = Vec::new();
        let mut reading = Vec::new();
        for snapshot in &snapshots {
            match self.layer_draw(snapshot) {
                Ok(Some(draw)) => {
                    reading.push(snapshot.frame.clone().expect("drawn from a frame"));
                    layers.push(draw);
                }
                Ok(None) => {}
                // One layer that cannot be drawn leaves the rest drawn, and
                // is said under its own name.
                Err(error) => self.report_layer(&snapshot.name, error),
            }
        }
        let text_draws: Vec<TextDraw> = texts
            .iter()
            .filter_map(|text| self.text_draw(text))
            .collect();

        let mut output = self.output_frame()?;
        let (to, to_fence, to_value) = d3d12va_texture(&output)
            .filter(|(texture, fence, _)| !texture.is_null() && !fence.is_null())
            .ok_or_else(|| {
                D3d12VideoCompositorError::UnsupportedPicture("a frame of its own pool".into())
            })?;
        // SAFETY: the pooled frame owns these non-null objects; cloning the
        // borrowed wrappers takes references of its own.
        let (to, to_fence) = unsafe {
            (
                ID3D12Resource::from_raw_borrowed(&to)
                    .expect("not null")
                    .clone(),
                ID3D12Fence::from_raw_borrowed(&to_fence)
                    .expect("not null")
                    .clone(),
            )
        };

        let views = CANVAS_VIEWS + 2 * (layers.len() + text_draws.len());
        if self.heap.as_ref().is_none_or(|(_, held)| *held < views) {
            // SAFETY: a live device and a description built here.
            let heap: ID3D12DescriptorHeap = unsafe {
                self.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        NumDescriptors: views as u32,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })?
            };
            self.heap = Some((heap, views));
        }
        let heap = self.heap.as_ref().expect("made above").0.clone();
        let step = self.heap_step;
        // SAFETY: a live heap's handles, each `step` apart, inside it.
        let (cpu_start, gpu_start) = unsafe {
            (
                heap.GetCPUDescriptorHandleForHeapStart(),
                heap.GetGPUDescriptorHandleForHeapStart(),
            )
        };
        let cpu = |at: usize| D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: cpu_start.ptr + at * step,
        };
        let gpu = |at: usize| D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: gpu_start.ptr + (at * step) as u64,
        };
        let (luma, chroma) = (self.canvas.0.clone(), self.canvas.1.clone());
        // SAFETY: the GPU is idle on this element's work, waited for above,
        // so the heap's views may be written; each names a live resource.
        unsafe {
            self.device.CreateUnorderedAccessView(
                &luma,
                None,
                Some(&plane_write_view(DXGI_FORMAT_R8_UNORM, 0)),
                cpu(0),
            );
            self.device.CreateUnorderedAccessView(
                &chroma,
                None,
                Some(&plane_write_view(DXGI_FORMAT_R8G8_UNORM, 0)),
                cpu(1),
            );
            for (index, layer) in layers.iter().enumerate() {
                let at = CANVAS_VIEWS + 2 * index;
                self.device.CreateShaderResourceView(
                    &layer.texture,
                    Some(&plane_view(DXGI_FORMAT_R8_UNORM, 0)),
                    cpu(at),
                );
                self.device.CreateShaderResourceView(
                    &layer.texture,
                    Some(&plane_view(DXGI_FORMAT_R8G8_UNORM, 1)),
                    cpu(at + 1),
                );
            }
            for (index, text) in text_draws.iter().enumerate() {
                let at = CANVAS_VIEWS + 2 * (layers.len() + index);
                self.device.CreateShaderResourceView(
                    &text.mask.texture,
                    Some(&plane_view(DXGI_FORMAT_R8_UNORM, 0)),
                    cpu(at),
                );
                // Never read by the text kernels; a view of nothing.
                self.device.CreateShaderResourceView(
                    None,
                    Some(&plane_view(DXGI_FORMAT_R8G8_UNORM, 0)),
                    cpu(at + 1),
                );
            }
        }

        // Video and text in one stacking order: by `z_index`, a video layer
        // before a text layer of the same one, each kind as it is in.
        let mut order: Vec<(i32, bool, usize)> = layers
            .iter()
            .enumerate()
            .map(|(index, layer)| (layer.z_index, false, index))
            .chain(
                text_draws
                    .iter()
                    .enumerate()
                    .map(|(index, text)| (text.z_index, true, index)),
            )
            .collect();
        order.sort();

        // Each picture read once in the list, however many layers show it.
        let mut pictures: Vec<ID3D12Resource> = Vec::new();
        for layer in &layers {
            if !pictures
                .iter()
                .any(|picture| picture.as_raw() == layer.texture.as_raw())
            {
                pictures.push(layer.texture.clone());
            }
        }
        let [background_y, background_cb, background_cr] = canvas_colour(self.options.background);
        let background = constants(
            Region {
                x: 0,
                y: 0,
                width: self.options.width,
                height: self.options.height,
            },
            [
                [0.0; 4],
                [0.0; 4],
                [0.0; 4],
                [0.0; 4],
                [0.0; 4],
                [background_y, background_cb, background_cr, 1.0],
            ],
        );

        let list = self.commands.begin(Some(&self.fill.pipeline))?;
        // SAFETY: the list is open and the GPU idle on this element's work;
        // every view and barrier names a live resource this element, the
        // pictures it holds in `reading` or the output frame hold until the
        // queue has run what reads it.
        unsafe {
            list.SetDescriptorHeaps(&[Some(heap.clone())]);
            list.SetComputeRootSignature(&self.fill.root_signature);
            list.SetComputeRootDescriptorTable(1, gpu(0));
            for picture in &pictures {
                transition(
                    &list,
                    picture,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                );
            }
            list.SetComputeRoot32BitConstants(0, 28, background.as_ptr().cast(), 0);
            list.Dispatch(
                self.options.width.div_ceil(8),
                self.options.height.div_ceil(8),
                1,
            );
            uav_barrier(&list, &luma);
            uav_barrier(&list, &chroma);
            for (_, is_text, index) in order {
                let (at, constants, region, planes) = if is_text {
                    let text = &text_draws[index];
                    (
                        CANVAS_VIEWS + 2 * (layers.len() + index),
                        &text.constants,
                        text.region,
                        [&self.text_luma, &self.text_chroma],
                    )
                } else {
                    let layer = &layers[index];
                    (
                        CANVAS_VIEWS + 2 * index,
                        &layer.constants,
                        layer.region,
                        [&self.layer_luma, &self.layer_chroma],
                    )
                };
                list.SetComputeRootDescriptorTable(2, gpu(at));
                list.SetComputeRoot32BitConstants(0, 28, constants.as_ptr().cast(), 0);
                list.SetPipelineState(planes[0]);
                list.Dispatch(region.width.div_ceil(8), region.height.div_ceil(8), 1);
                list.SetPipelineState(planes[1]);
                list.Dispatch(
                    (region.width / 2).div_ceil(8),
                    (region.height / 2).div_ceil(8),
                    1,
                );
                uav_barrier(&list, &luma);
                uav_barrier(&list, &chroma);
            }
            for picture in &pictures {
                transition(
                    &list,
                    picture,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                );
            }
            // The canvas into the frame handed on.
            for plane in [&luma, &chroma] {
                transition(
                    &list,
                    plane,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                );
            }
            transition(
                &list,
                &to,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );
            copy_plane(&list, (&to, 0), (&luma, 0));
            copy_plane(&list, (&to, 1), (&chroma, 0));
            transition(
                &list,
                &to,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COMMON,
            );
            for plane in [&luma, &chroma] {
                transition(
                    &list,
                    plane,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                );
            }
        }
        // After whatever wrote each picture, and whatever last read the
        // frame drawn into; the frame's own fence says when it is drawn.
        let fences: Vec<(ID3D12Fence, u64)> = reading
            .iter()
            .filter_map(|frame| {
                let (_, fence, value) = d3d12va_texture(frame)?;
                // SAFETY: the held frame owns this non-null fence.
                let fence = unsafe { ID3D12Fence::from_raw_borrowed(&fence)?.clone() };
                Some((fence, value))
            })
            .collect();
        let mut waits: Vec<(&ID3D12Fence, u64)> = fences
            .iter()
            .map(|(fence, value)| (fence, *value))
            .collect();
        waits.push((&to_fence, to_value));
        let drawn = to_value + 1;
        self.commands.submit(&waits, &[(&to_fence, drawn)])?;
        set_d3d12va_fence_value(&mut output, drawn);
        self.reading = reading;

        let mut kept = ffmpeg::frame::Video::empty();
        // SAFETY: both are live `AVFrame`s, so this adds a reference to the
        // frame just composed rather than copying it.
        let code = unsafe { ffi::av_frame_ref(kept.as_mut_ptr(), output.as_ptr()) };
        if code < 0 {
            return Err(D3d12VideoCompositorError::FrameRef(code));
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

    /// Says that input `name`'s layer could not be drawn, and why.
    fn report_layer(&self, name: &Arc<str>, error: D3d12VideoCompositorError) {
        match &self.bus {
            Some(bus) => bus.post(
                &self.pp_log,
                crate::bus::BusEvent::Error {
                    element_type: ElementType::D3d12VideoCompositor,
                    name: name.clone(),
                    error: error.into(),
                },
            ),
            None => pp_error!(self, "{name} is not drawn: {error}"),
        }
    }

    /// This tick's frame, with what composing it took recorded.
    fn make_frame(&mut self) -> std::result::Result<MediaBuffer, D3d12VideoCompositorError> {
        let composing = Instant::now();
        let output = self.compose_frame()?;
        if let Some(ticks) = &self.ticks {
            ticks.made(composing.elapsed());
        }
        Ok(MediaBuffer::Video(Arc::new(output).into()))
    }

    /// This tick's frame, once it is due at its own rate.
    fn produce_live(&mut self, wait: &mut Wait<'_>) -> Result<Produced> {
        let tick = self
            .schedule
            .next(self.shared.frame_rate.interval(), wait.now());
        if let Some(ticks) = &self.ticks {
            ticks.missed(tick.missed);
        }
        if tick.rate_changed {
            pp_info!(self, "frame rate is now {}", self.frame_rate());
        }
        if !tick.come(wait, RATE_POLL_INTERVAL) {
            return Ok(Produced::Nothing);
        }
        let frame = self.make_frame()?;
        self.schedule.made();
        Ok(Produced::Buffer(frame))
    }
}

impl Element for Compositing {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::D3d12VideoCompositor
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }

    fn attach_context(&mut self, context: &Arc<Context>) {
        self.ticks = Some(context.source_ticks());
        self.bus = Some(context.bus.clone());
    }
}

impl Source for Compositing {
    fn is_live(&self) -> bool {
        self.options.mode.is_live()
    }

    fn output_contract(&self) -> OutputContract {
        OutputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::D3d12)
                .with_layouts(PixelLayoutSet::NV12),
        )
    }

    fn produce(&mut self, wait: &mut Wait<'_>) -> Result<Produced> {
        match self.options.mode {
            RenderMode::Live => self.produce_live(wait),
            RenderMode::Offline { end } => produce_offline(self, wait, end),
        }
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
        // What is queued still reads the canvas, the views and the pictures
        // held: it runs out before they go.
        let _ = self.commands.wait_idle();
        pp_info!(self, "dropped");
    }
}

/// One axis of what a layer draws: where its scaled image, its rectangle
/// and the canvas all overlap, rounded inward to even pixels. The start and
/// the extent, or `None` where nothing of it is seen.
fn axis(
    image_origin: i64,
    image_extent: u32,
    clip_origin: i64,
    clip_extent: u32,
    canvas_extent: u32,
) -> Option<(u32, u32)> {
    let start = image_origin.max(clip_origin).max(0);
    let end = (image_origin + i64::from(image_extent))
        .min(clip_origin + i64::from(clip_extent))
        .min(i64::from(canvas_extent));
    let start = start + start.rem_euclid(2);
    let end = end - end.rem_euclid(2);
    (end > start).then(|| (start as u32, (end - start) as u32))
}

/// One axis of a text mask's clip, the origin aligned down to even so that
/// where in the mask a drawn pixel is stays even too: the start on the
/// canvas, the start in the mask and the extent.
fn text_axis(origin: i32, extent: u32, canvas_extent: u32) -> Option<(u32, u32, u32)> {
    let origin = i64::from(origin) - i64::from(origin).rem_euclid(2);
    let start = origin.max(0);
    let end = (origin + i64::from(extent)).min(i64::from(canvas_extent));
    let end = end - end.rem_euclid(2);
    (end > start).then(|| (start as u32, (start - origin) as u32, (end - start) as u32))
}

/// What an input picture is, checked to be one this draws.
struct InputPicture {
    texture: ID3D12Resource,
    /// Its texture's width and height, which its picture may fall short of.
    texture_size: (u32, u32),
}

/// `frame`, checked to be an NV12 D3D12 picture of `device`, inside its
/// texture.
fn validate_input_frame(
    frame: &ffmpeg::frame::Video,
    device: &ID3D12Device,
) -> std::result::Result<InputPicture, D3d12VideoCompositorError> {
    let (texture, fence, _) = d3d12va_texture(frame)
        .filter(|(texture, fence, _)| !texture.is_null() && !fence.is_null())
        .ok_or_else(|| {
            D3d12VideoCompositorError::UnsupportedPicture(format!("{:?}", frame.format()))
        })?;
    let _ = fence;
    // SAFETY: the live frame owns this non-null texture; cloning the
    // borrowed wrapper takes a reference of its own.
    let texture = unsafe {
        ID3D12Resource::from_raw_borrowed(&texture)
            .expect("not null")
            .clone()
    };
    let mut made_on: Option<ID3D12Device> = None;
    // SAFETY: a live resource and a correctly typed out-parameter.
    unsafe { texture.GetDevice(&mut made_on) }?;
    if made_on.is_none_or(|made_on| made_on.as_raw() != device.as_raw()) {
        return Err(D3d12VideoCompositorError::DeviceMismatch);
    }
    // SAFETY: a live resource's plain description, returned by value.
    let desc = unsafe { texture.GetDesc() };
    if desc.Format != DXGI_FORMAT_NV12 {
        return Err(D3d12VideoCompositorError::UnsupportedPicture(format!(
            "a texture of DXGI format {}",
            desc.Format.0
        )));
    }
    let texture_size = (desc.Width as u32, desc.Height);
    let (width, height) = (frame.width(), frame.height());
    if width == 0 || height == 0 {
        return Err(D3d12VideoCompositorError::InvalidInputDimensions { width, height });
    }
    if width > texture_size.0 || height > texture_size.1 {
        return Err(D3d12VideoCompositorError::PictureOutsideTexture {
            picture: (width, height),
            texture: texture_size,
        });
    }
    Ok(InputPicture {
        texture,
        texture_size,
    })
}

fn validate_output_options(
    options: VideoCompositorOptions,
) -> std::result::Result<(), D3d12VideoCompositorError> {
    if options.width < 2
        || options.height < 2
        || !options.width.is_multiple_of(2)
        || !options.height.is_multiple_of(2)
        || options.width > MAX_DIMENSION
        || options.height > MAX_DIMENSION
    {
        return Err(D3d12VideoCompositorError::InvalidOutputDimensions {
            width: options.width,
            height: options.height,
        });
    }
    if options.frame_rate.numerator() <= 0 || options.frame_rate.denominator() <= 0 {
        return Err(D3d12VideoCompositorError::InvalidFrameRate(
            options.frame_rate,
        ));
    }
    if options.background_alpha != 255 {
        return Err(D3d12VideoCompositorError::TranslucentBackground(
            options.background_alpha,
        ));
    }
    Ok(())
}

fn validate_layer(layer: VideoLayer) -> std::result::Result<(), D3d12VideoCompositorError> {
    video_layer::validate_layer(layer).map_err(layer_error)?;
    validate_opacity(layer.opacity)
}

fn validate_rect(rect: VideoRect) -> std::result::Result<(), D3d12VideoCompositorError> {
    video_layer::validate_rect(rect).map_err(layer_error)
}

fn validate_opacity(opacity: f32) -> std::result::Result<(), D3d12VideoCompositorError> {
    video_layer::validate_opacity(opacity).map_err(layer_error)
}

fn layer_error(error: VideoLayerError) -> D3d12VideoCompositorError {
    match error {
        VideoLayerError::InvalidDimensions { width, height } => {
            D3d12VideoCompositorError::InvalidLayerDimensions { width, height }
        }
        VideoLayerError::InvalidOpacity(opacity) => {
            D3d12VideoCompositorError::InvalidOpacity(opacity)
        }
        VideoLayerError::InvalidInputDimensions { width, height } => {
            D3d12VideoCompositorError::InvalidInputDimensions { width, height }
        }
        VideoLayerError::ScaledLayerTooLarge { width, height } => {
            D3d12VideoCompositorError::ScaledLayerTooLarge { width, height }
        }
        VideoLayerError::InvalidSourceRegion { width, height } => {
            D3d12VideoCompositorError::InvalidSourceRegion { width, height }
        }
    }
}

#[cfg(test)]
mod tests;

super::super::control::compositor_control!(
    D3d12VideoCompositorHandle,
    D3d12VideoLayerHandle,
    D3d12TextLayerHandle
);
