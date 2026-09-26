use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use arc_swap::ArcSwapOption;
use ash::vk;
use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_debug, pp_error, pp_info};
use crate::rate::FrameRate;

use super::super::sw_video_compositor::VideoCompositorOptions;
use super::super::text_layer::TextMask;
use super::super::timed_inputs::{
    OfflineCompositor, Picture, TimedFeed, TimedInput, Untimed, run_offline,
};
use super::super::video_layer::{
    self, LayerGeometry, MAX_DIMENSION, VideoInputId, VideoLayer, VideoLayerError, VideoRect,
    layer_geometry,
};
use crate::elements::source::render_mode::{MediaTime, RenderMode};
use crate::playback_state::Bell;
use crate::{
    buffer::{MediaBuffer, picture_id, release_picture},
    bus::{Bus, BusEvent},
    color::{Color, ColorDescription},
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    control::{ControlMsg, ControlReceiver, drain_control},
    element::{Context, Element, ElementType, Sink, Source, SourceElement, element_pp_log},
    elements::VulkanDevice,
    error::Result,
    pad::SrcPad,
    platform::{
        ffmpeg::AvBufferRef,
        vulkan::{
            device::DeviceShared,
            frame_access::{Claim, abandon, claim, images_of},
            frames::{NotOurs, create_frames_ctx, sw_format_of},
            gpu::{HostBuffer, Image, Kernel, Recording, Sampler, View, VulkanError},
        },
    },
    pool::{UnboundObjectPool, UnboundObjectPoolRef},
    schedule::PeriodicSchedule,
    stats::TickCounters,
};

mod text_handle;
mod video_handle;

use text_handle::TextLayerState;
pub use text_handle::VulkanTextLayerHandle;
pub use video_handle::VulkanVideoLayerHandle;

const OUTPUT_POOL_SIZE: usize = 4;
const CONTROL_POLL_INTERVAL: Duration = Duration::from_millis(5);

/// The compositor's kernels, compiled from this one source.
const SHADER: &str = include_str!("../../../../../shaders/vulkan/composite.wgsl");

/// What a [`VulkanVideoCompositor`] composes in, and so what each frame it
/// emits holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VulkanFrameFormat {
    /// 8-bit BGRA, with alpha: a picture that can be transparent where no
    /// layer drew, to be laid over something else.
    #[default]
    Bgra,
    /// NV12, BT.709 at limited range: what an encoder takes, and half the
    /// bytes. Opaque.
    Nv12,
}

impl VulkanFrameFormat {
    fn pixel(self) -> ffmpeg::format::Pixel {
        match self {
            Self::Bgra => ffmpeg::format::Pixel::BGRA,
            Self::Nv12 => ffmpeg::format::Pixel::NV12,
        }
    }

    pub(crate) fn layouts(self) -> crate::contract::PixelLayoutSet {
        match self {
            Self::Bgra => crate::contract::PixelLayoutSet::BGRA,
            Self::Nv12 => crate::contract::PixelLayoutSet::NV12,
        }
    }
}

/// Errors specific to [`VulkanVideoCompositor`].
#[derive(Debug, ThisError)]
pub enum VulkanVideoCompositorError {
    /// A Vulkan call this compositor made failed.
    #[error(transparent)]
    Vulkan(#[from] VulkanError),

    /// FFmpeg could not make the pool output frames come from.
    #[error("{0}")]
    Pool(String),

    /// FFmpeg could not take an output frame from the pool.
    #[error("failed to take an output frame from the Vulkan pool (code {0})")]
    FrameGet(i32),

    /// FFmpeg could not take a second reference to the frame last composed,
    /// which is how an unchanged picture is offered again.
    #[error("failed to reference the previous composite (code {0})")]
    FrameRef(i32),

    /// A background that is not opaque, on an NV12 canvas, which has no
    /// alpha to keep it in.
    #[error(
        "background_alpha {0} is not supported: an NV12 VulkanVideoCompositor has no alpha; compose in VulkanFrameFormat::Bgra"
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

    /// An input frame reports a zero width or height.
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

    /// An input frame is not a Vulkan frame.
    #[error("VulkanVideoCompositor draws Vulkan frames, got a {0:?} frame; upload it first")]
    NotVulkan(ffmpeg::format::Pixel),

    /// An input frame was made on another `VkDevice` than the compositor's.
    #[error("VulkanVideoCompositor was handed a frame from another Vulkan device")]
    ForeignDevice,

    /// An input frame holds a layout this compositor does not draw.
    #[error("VulkanVideoCompositor draws NV12 and BGRA frames, got {0:?}")]
    UnsupportedLayout(ffmpeg::format::Pixel),

    /// An input sink received a buffer other than decoded video.
    #[error(
        "VulkanVideoCompositorInputSink only accepts decoded Video frames, got a {0}; link it after a decoder or upload"
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
    /// [`VulkanVideoCompositorHandle::set_frame_rate`] writes.
    frame_rate: Arc<FrameRate>,
    /// The compositor's own device, so every input sink can refuse a frame
    /// from another before it reaches a recording. Only ever compared.
    device_ctx: *const ffi::AVHWDeviceContext,
    mode: RenderMode,
    /// Rung by an offline compositor's input sinks as frames or their ends
    /// arrive.
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

// SAFETY: `device_ctx` is only ever compared, never dereferenced, and the
// compositor holds its own reference to the context for its whole life, so
// the pointer cannot go stale while any input sink is alive.
unsafe impl Send for CompositorShared {}

// SAFETY: as above for the raw pointer, and every field beside it carries its
// own synchronization — the maps are behind mutexes and the counters are
// atomic. This is what lets an input sink on one thread share this with the
// compositor's own.
unsafe impl Sync for CompositorShared {}

/// A cheaply cloneable handle for adding and removing compositor inputs —
/// the Vulkan sibling of [`crate::elements::SwVideoCompositorHandle`].
///
/// Holds the compositor weakly: once it is dropped, every method reports
/// [`VulkanVideoCompositorError::Stopped`] or nothing.
#[derive(Clone)]
pub struct VulkanVideoCompositorHandle {
    shared: Weak<CompositorShared>,
}

/// The sink and layer handle [`VulkanVideoCompositorHandle::add_source`]
/// returns — see [`crate::elements::CompositorInput`].
pub type VulkanVideoCompositorInput = crate::elements::CompositorInput<VulkanVideoLayerHandle>;

impl VulkanVideoCompositorHandle {
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
    ) -> std::result::Result<VulkanVideoCompositorInput, VulkanVideoCompositorError> {
        let layer = self.register(name, layer, true)?;
        Ok(VulkanVideoCompositorInput {
            sink: Box::new(VulkanVideoCompositorInputSink {
                pp_log: element_pp_log(ElementType::VulkanVideoCompositor, &layer.name, None),
                name: layer.name.clone(),
                id: layer.id,
                shared: self.shared.clone(),
                input: layer.input.clone(),
            }),
            layer,
        })
    }

    /// Registers an input and returns *only* its layer handle — no `Sink` —
    /// for a caller that sets this input's picture itself through
    /// [`VulkanVideoLayerHandle::set_frame`] rather than wiring a pipeline
    /// into it. Replaces any registration of the same name, as
    /// [`Self::add_source`] does.
    ///
    /// Offline too the picture is whatever was last set: it has no
    /// timestamps to be placed by, and is never waited for.
    pub fn add_layer(
        &self,
        name: impl Into<String>,
        layer: VideoLayer,
    ) -> std::result::Result<VulkanVideoLayerHandle, VulkanVideoCompositorError> {
        self.register(name, layer, false)
    }

    /// Registers an input, fed through a sink where `fed` says so — which,
    /// offline, is an input whose frames are placed by their timestamps.
    fn register(
        &self,
        name: impl Into<String>,
        layer: VideoLayer,
        fed: bool,
    ) -> std::result::Result<VulkanVideoLayerHandle, VulkanVideoCompositorError> {
        validate_layer(layer)?;
        let Some(shared) = self.shared.upgrade() else {
            return Err(VulkanVideoCompositorError::Stopped);
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
        Ok(VulkanVideoLayerHandle {
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
        }
    }

    /// Changes the rate this compositor emits at, from the next tick — the
    /// same contract as `CudaVideoCompositorHandle::set_frame_rate`:
    /// [`VulkanVideoCompositorError::InvalidFrameRate`] for a rate that is
    /// not positive, [`VulkanVideoCompositorError::Stopped`] once the
    /// compositor is gone, [`VulkanVideoCompositorError::FixedFrameRate`]
    /// offline, the running rate left alone in each case.
    /// [`VulkanVideoCompositor::time_base`] is the reciprocal of this and the
    /// output `pts` a tick counter in those units, so a change re-means every
    /// timestamp after it.
    pub fn set_frame_rate(
        &self,
        frame_rate: ffmpeg::Rational,
    ) -> std::result::Result<(), VulkanVideoCompositorError> {
        let shared = self
            .shared
            .upgrade()
            .ok_or(VulkanVideoCompositorError::Stopped)?;
        if !shared.mode.is_live() {
            return Err(VulkanVideoCompositorError::FixedFrameRate);
        }
        shared
            .frame_rate
            .set(frame_rate)
            .map_err(|_| VulkanVideoCompositorError::InvalidFrameRate(frame_rate))
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
pub struct VulkanVideoCompositorInputSink {
    pp_log: PpLog,
    name: Arc<str>,
    id: VideoInputId,
    shared: Weak<CompositorShared>,
    input: Weak<VideoInput>,
}

impl VulkanVideoCompositorInputSink {
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
        }
    }
}

impl Element for VulkanVideoCompositorInputSink {
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
        ElementType::VulkanVideoCompositor
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Sink for VulkanVideoCompositorInputSink {
    /// Every layer is drawn on the device, from a Vulkan frame of its own.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::Vulkan)
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
                let Some(shared) = self.shared.upgrade() else {
                    return Err(VulkanVideoCompositorError::Stopped.into());
                };
                // Validated here rather than when drawn, so a misconfigured
                // input names *itself* in the error.
                validate_input_frame(&frame, shared.device_ctx)
                    .inspect_err(|error| pp_error!(self, "{error}"))?;
                let Some(input) = self.input.upgrade() else {
                    return Ok(());
                };
                let Some(timed) = &input.timed else {
                    input.latest_frame.store(Some(frame));
                    return Ok(());
                };
                timed.push(frame).map_err(|untimed| {
                    VulkanVideoCompositorError::UntimedFrame(match untimed {
                        Untimed::NoTimestamp => "timestamp",
                        Untimed::NoTimeBase => "time base",
                    })
                    .into()
                })
            }
            MediaBuffer::Eos => {
                // Offline an input's end is part of the picture: what it holds
                // is still shown to its last frame's end.
                if let Some(input) = self.input.upgrade()
                    && let Some(timed) = &input.timed
                {
                    timed.end();
                } else {
                    pp_debug!(self, "input reached eos; leaving its last frame in place");
                }
                Ok(())
            }
            other => {
                pp_error!(self, "unsupported buffer: {}", other.kind());
                Err(VulkanVideoCompositorError::UnsupportedBuffer(other.kind()).into())
            }
        }
    }

    fn control(&mut self, msg: &ControlMsg) -> Result<()> {
        // Terminal for its own branch: nothing downstream to forward to. A
        // `Stop` means this upstream pipeline is done, so the registration
        // goes with it — same as `SwVideoCompositorInputSink`.
        if matches!(msg, ControlMsg::Stop) {
            self.detach();
        }
        // Offline, what an input held before a flush is not what comes after.
        if matches!(msg, ControlMsg::Flush)
            && let Some(input) = self.input.upgrade()
            && let Some(timed) = &input.timed
        {
            timed.clear();
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
    /// the image they live in, as the CUDA compositor's are.
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
/// new timestamp on a new reference to the same image, never a copy.
struct Composed {
    inputs: Vec<InputSnapshot>,
    texts: Vec<TextSnapshot>,
    /// A reference to the frame last emitted, which also keeps its image out
    /// of the output pool so nothing draws over what may be handed out again.
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

/// What one dispatch draws — the shader's `Step`, as its push constants.
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

    /// Workgroups to cover the region, eight pixels square each.
    fn groups(&self) -> (u32, u32) {
        (
            (self.region[2] as u32).div_ceil(8),
            (self.region[3] as u32).div_ceil(8),
        )
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
        frame: Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
        layout: LayerLayout,
        step: Step,
    },
    Text {
        mask: Arc<TextMask>,
        step: Step,
    },
}

/// A text mask on the GPU, kept while its layer shows it.
struct GpuMask {
    image: Image,
    view: View,
    /// Whether its pixels are on the GPU yet, which the first recording
    /// that draws it sees to.
    uploaded: bool,
}

/// The compositor's kernels.
struct Kernels {
    fill: Kernel,
    nv12: Kernel,
    bgra: Kernel,
    text: Kernel,
    /// Only for an NV12 canvas.
    to_nv12: Option<Kernel>,
}

impl Kernels {
    fn new(
        shared: &Arc<DeviceShared>,
        format: VulkanFrameFormat,
    ) -> std::result::Result<Self, VulkanError> {
        use vk::DescriptorType as D;
        let immediates = std::mem::size_of::<Step>() as u32;
        let canvas = (0, D::STORAGE_IMAGE);
        Ok(Self {
            fill: Kernel::new(shared, SHADER, c"fill", &[canvas], immediates)?,
            nv12: Kernel::new(
                shared,
                SHADER,
                c"layer_nv12",
                &[
                    canvas,
                    (1, D::SAMPLER),
                    (2, D::SAMPLED_IMAGE),
                    (3, D::SAMPLED_IMAGE),
                ],
                immediates,
            )?,
            bgra: Kernel::new(
                shared,
                SHADER,
                c"layer_bgra",
                &[canvas, (1, D::SAMPLER), (2, D::SAMPLED_IMAGE)],
                immediates,
            )?,
            text: Kernel::new(
                shared,
                SHADER,
                c"text",
                &[canvas, (2, D::SAMPLED_IMAGE)],
                immediates,
            )?,
            to_nv12: match format {
                VulkanFrameFormat::Nv12 => Some(Kernel::new(
                    shared,
                    SHADER,
                    c"to_nv12",
                    &[canvas, (4, D::STORAGE_IMAGE), (5, D::STORAGE_IMAGE)],
                    immediates,
                )?),
                VulkanFrameFormat::Bgra => None,
            },
        })
    }
}

/// Composites the latest frames from any number of independent Vulkan input
/// pipelines into one fixed-rate `Pixel::VULKAN` stream, without any frame
/// leaving the GPU — the Vulkan sibling of
/// [`crate::elements::SwVideoCompositor`], `D3d11VideoCompositor` and
/// `CudaVideoCompositor`, driving the same
/// [`VideoLayer`]/[`VideoRect`]/[`VideoFit`](video_layer::VideoFit) API, and
/// running wherever Vulkan does: on any GPU, on Windows and Linux alike.
///
/// Like the others this is a [`SourceElement`], not a one-input filter:
/// upstream pipelines terminate at the sinks returned by
/// [`VulkanVideoCompositorHandle::add_source`], while this element's own
/// pipeline drives output on its own clock, or offline by the inputs'
/// timestamps — see [`RenderMode`].
///
/// # How it draws
///
/// Each frame is one recording on a compute queue of the device FFmpeg made:
/// the canvas is filled with the background, each layer in stacking order is
/// sampled from its picture — scaled bilinearly, cropped, converted from its
/// own colour description — and blended onto it, text layers from their
/// coverage masks, and the canvas is copied into the output frame, or turned
/// into NV12 on the way. Its frames are used as FFmpeg's own Vulkan elements
/// use them, under FFmpeg's lock and its semaphores, so a decoder's picture is
/// drawn once the decoder has written it and an encoder reads the output once
/// it is drawn; the recording is waited for before the frame is handed on.
///
/// Inputs are NV12 or BGRA Vulkan frames of the compositor's own device —
/// what [`crate::elements::VulkanDecoder`] and [`crate::elements::VulkanUpload`]
/// make. A BGRA layer keeps its alpha, which is blended as the D3D11
/// compositor blends it, [`VideoLayer::premultiplied_alpha`] included.
pub struct VulkanVideoCompositor {
    pp_log: PpLog,
    name: Arc<str>,
    shared: Arc<CompositorShared>,
    options: VideoCompositorOptions,
    frame_index: i64,
    /// The device, whose reference to FFmpeg's context is also what keeps
    /// `shared.device_ctx` a valid identity.
    device: Arc<DeviceShared>,
    /// The pool output frames come from.
    hw_frames_ctx: AvBufferRef,
    format: VulkanFrameFormat,
    /// Where every layer is blended: B, G, R and A in an RGBA image, the
    /// byte order of a BGRA frame — see the shader.
    canvas: Image,
    canvas_view: View,
    kernels: Kernels,
    sampler: Sampler,
    recording: Recording,
    /// Text masks on the GPU, by the address of the mask they were made
    /// from, which `set_text` replaces rather than changes.
    masks: HashMap<usize, GpuMask>,
    /// The last composite and what it was made from — see [`Composed`].
    composed: Option<Composed>,
    /// Reuses only the small CPU-side `AVFrame` wrapper; the image itself
    /// comes from `hw_frames_ctx`'s own pool.
    output_pool: UnboundObjectPool<ffmpeg::frame::Video>,
    pad: SrcPad,
    /// Where each tick is recorded, once the pipeline has handed it over.
    ticks: Option<Arc<TickCounters>>,
}

// SAFETY: the FFmpeg buffers have no thread affinity of their own, the
// Vulkan objects are this element's alone and touched only through `&mut
// self` on its single source thread, and queue access goes through FFmpeg's
// lock.
unsafe impl Send for VulkanVideoCompositor {}

impl VulkanVideoCompositor {
    /// A compositor on `device`, composing in BGRA — see
    /// [`Self::with_format`]. `device` must be the same [`VulkanDevice`]
    /// every input's Vulkan elements were built from; a frame from another is
    /// refused by the input sink.
    ///
    /// Output dimensions must be even, so the same composition can be made
    /// in NV12.
    pub fn new(
        name: impl Into<String>,
        device: &VulkanDevice,
        options: VideoCompositorOptions,
    ) -> std::result::Result<(Self, VulkanVideoCompositorHandle), VulkanVideoCompositorError> {
        Self::with_format(name, device, options, VulkanFrameFormat::Bgra)
    }

    /// The same, composing in `format`: BGRA for a picture that may be
    /// transparent where no layer drew, and the only format that takes a
    /// [`background_alpha`](VideoCompositorOptions::background_alpha) other
    /// than 255; NV12 for one an encoder takes.
    pub fn with_format(
        name: impl Into<String>,
        device: &VulkanDevice,
        options: VideoCompositorOptions,
        format: VulkanFrameFormat,
    ) -> std::result::Result<(Self, VulkanVideoCompositorHandle), VulkanVideoCompositorError> {
        crate::ensure_ffmpeg();
        validate_output_options(options, format)?;
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::VulkanVideoCompositor, &name, None);
        let gpu = Arc::clone(device.shared());
        let hw_device_ctx = device.retain();
        let usage = match format {
            // Copied into from the canvas.
            VulkanFrameFormat::Bgra => {
                vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::SAMPLED
            }
            // Written plane by plane by `to_nv12`, and read by an encoder:
            // FFmpeg's own choice, which is every use the format allows —
            // storage and encoding among them — on images whose planes can
            // each be viewed on their own.
            VulkanFrameFormat::Nv12 => vk::ImageUsageFlags::empty(),
        };
        // SAFETY: `create_frames_ctx`'s contract is a live device context,
        // which the reference just taken is.
        let hw_frames_ctx = unsafe {
            create_frames_ctx(
                &hw_device_ctx,
                format.pixel(),
                options.width,
                options.height,
                usage,
            )
        }
        .map_err(|error| VulkanVideoCompositorError::Pool(error.to_string()))?;
        let canvas = Image::new(
            &gpu,
            vk::Format::R8G8B8A8_UNORM,
            options.width,
            options.height,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
        )?;
        let canvas_view = View::new(
            &gpu,
            canvas.image,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageAspectFlags::COLOR,
        )?;
        let kernels = Kernels::new(&gpu, format)?;
        let sampler = Sampler::linear(&gpu)?;
        let recording = Recording::new(&gpu)?;

        let shared = Arc::new(CompositorShared {
            inputs: Mutex::new(HashMap::new()),
            text_layers: Mutex::new(Vec::new()),
            next_input_id: AtomicU64::new(1),
            frame_rate: FrameRate::new(options.frame_rate),
            device_ctx: device.device_ctx(),
            mode: options.mode,
            arrived: Bell::new(),
            next_output: AtomicI64::new(0),
            fed: AtomicBool::new(false),
        });
        pp_info!(
            pp_log: &pp_log,
            "created: {}x{}, frame_rate={}, format=Vulkan/{:?} on {}",
            options.width,
            options.height,
            options.frame_rate,
            format,
            device.name()
        );
        Ok((
            Self {
                name: name.clone(),
                pp_log,
                shared: shared.clone(),
                options,
                frame_index: 0,
                device: gpu,
                hw_frames_ctx,
                format,
                canvas,
                canvas_view,
                kernels,
                sampler,
                recording,
                masks: HashMap::new(),
                composed: None,
                output_pool: UnboundObjectPool::new(
                    OUTPUT_POOL_SIZE,
                    ffmpeg::frame::Video::empty,
                    // A wrapper that held its reference while pooled would
                    // keep an image out of the frames pool for nothing.
                    release_picture,
                ),
                pad: SrcPad::with_contract(
                    format!("{name}_src"),
                    OutputContract::Fixed(
                        PortContract::frame(MediaKind::VideoFrame, MemoryDomain::Vulkan)
                            .with_layouts(format.layouts()),
                    ),
                ),
                ticks: None,
            },
            VulkanVideoCompositorHandle {
                shared: Arc::downgrade(&shared),
            },
        ))
    }

    /// Every output frame is a Vulkan frame.
    pub fn format(&self) -> ffmpeg::format::Pixel {
        ffmpeg::format::Pixel::VULKAN
    }

    /// What the output frames hold.
    pub fn frame_format(&self) -> VulkanFrameFormat {
        self.format
    }

    /// Returns the fixed output width in pixels.
    pub fn width(&self) -> u32 {
        self.options.width
    }

    /// Returns the fixed output height in pixels.
    pub fn height(&self) -> u32 {
        self.options.height
    }

    /// The output frame rate, which is what construction was given unless
    /// [`VulkanVideoCompositorHandle::set_frame_rate`] has changed it since.
    pub fn frame_rate(&self) -> ffmpeg::Rational {
        self.shared.frame_rate.get()
    }

    /// The reciprocal of [`Self::frame_rate`] — output PTS advance by one
    /// tick in this base per composed frame, so this moves with the rate.
    pub fn time_base(&self) -> ffmpeg::Rational {
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
    ) -> std::result::Result<UnboundObjectPoolRef<ffmpeg::frame::Video>, VulkanVideoCompositorError>
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
                return Err(VulkanVideoCompositorError::FrameRef(code));
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
    ) -> std::result::Result<UnboundObjectPoolRef<ffmpeg::frame::Video>, VulkanVideoCompositorError>
    {
        let mut output = self.output_pool.get();
        // SAFETY: `ptr` is the pooled wrapper's own `AVFrame`; the unref
        // hands its previous image back to the frames pool before a new one
        // is taken from it, which this element holds for its life.
        unsafe {
            let ptr = output.as_mut_ptr();
            ffi::av_frame_unref(ptr);
            let code = ffi::av_hwframe_get_buffer(self.hw_frames_ctx.as_ptr(), ptr, 0);
            if code < 0 {
                return Err(VulkanVideoCompositorError::FrameGet(code));
            }
        }
        // What the canvas is, so a download, a scaler or an encoder reads it
        // right: BT.709, as RGB at full range or NV12 at limited.
        match self.format {
            VulkanFrameFormat::Nv12 => ColorDescription::BT709_LIMITED.describe(&mut output),
            VulkanFrameFormat::Bgra => ColorDescription {
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
    ) -> std::result::Result<Vec<Draw>, VulkanVideoCompositorError> {
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
            let layout = validate_input_frame(frame, self.shared.device_ctx)?;
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
            // SAFETY: validated above as a live Vulkan frame of this device.
            let images = unsafe { images_of(frame) };
            let rows = if layout == LayerLayout::Nv12 {
                ColorDescription::of(frame).yuv_to_rgb_rows(frame.height())
            } else {
                [[0.0; 4]; 3]
            };
            let premultiplied = layout == LayerLayout::Bgra && layer.premultiplied_alpha;
            let step = Step {
                region,
                image: [
                    geometry.image_x as f32,
                    geometry.image_y as f32,
                    geometry.image_width as f32,
                    geometry.image_height as f32,
                ],
                // Against the image's own size, which a decoder's exceeds the
                // picture's by its padding.
                uv: [
                    source.width as f32 / images.width as f32,
                    source.height as f32 / images.height as f32,
                    source.x as f32 / images.width as f32,
                    source.y as f32 / images.height as f32,
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
                    frame: Arc::clone(frame),
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
    ) -> std::result::Result<UnboundObjectPoolRef<ffmpeg::frame::Video>, VulkanVideoCompositorError>
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
        // A mask no layer shows any more was last drawn by a recording that
        // has been waited for.
        let shown: HashSet<usize> = texts
            .iter()
            .filter_map(|text| text.mask.as_ref().map(|mask| Arc::as_ptr(mask) as usize))
            .collect();
        self.masks.retain(|key, _| shown.contains(key));

        let mut output = self.output_frame()?;
        self.record_and_submit(&draws, &output)?;

        // Held so the next tick can tell whether it has anything to draw,
        // and so the image it may hand out again stays out of the pool.
        let mut kept = ffmpeg::frame::Video::empty();
        // SAFETY: both are live `AVFrame`s, so this adds a reference to the
        // image just composed rather than copying it.
        let code = unsafe { ffi::av_frame_ref(kept.as_mut_ptr(), output.as_ptr()) };
        if code < 0 {
            return Err(VulkanVideoCompositorError::FrameRef(code));
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

    /// Records `draws` into `output`, submits the recording and waits for
    /// it.
    fn record_and_submit(
        &mut self,
        draws: &[Draw],
        output: &ffmpeg::frame::Video,
    ) -> std::result::Result<(), VulkanVideoCompositorError> {
        let gpu = Arc::clone(&self.device);
        // Made before any frame is claimed, so a failure here leaves every
        // frame as it was.
        let mut staging = Vec::new();
        for draw in draws {
            if let Draw::Text { mask, .. } = draw {
                let key = Arc::as_ptr(mask) as usize;
                let gpu_mask = match self.masks.entry(key) {
                    std::collections::hash_map::Entry::Occupied(slot) => slot.into_mut(),
                    std::collections::hash_map::Entry::Vacant(slot) => {
                        let image = Image::new(
                            &gpu,
                            vk::Format::R8_UNORM,
                            mask.width,
                            mask.height,
                            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                        )?;
                        let view = View::new(
                            &gpu,
                            image.image,
                            vk::Format::R8_UNORM,
                            vk::ImageAspectFlags::COLOR,
                        )?;
                        slot.insert(GpuMask {
                            image,
                            view,
                            uploaded: false,
                        })
                    }
                };
                if !gpu_mask.uploaded {
                    let mut buffer = HostBuffer::new(&gpu, mask.coverage.len())?;
                    buffer.bytes().copy_from_slice(&mask.coverage);
                    staging.push((key, buffer));
                }
            }
        }
        let mut views = Vec::new();
        let mut video_views = Vec::with_capacity(draws.len());
        for draw in draws {
            let Draw::Video { frame, layout, .. } = draw else {
                video_views.push(Vec::new());
                continue;
            };
            // SAFETY: validated when the draw was made, as a live Vulkan
            // frame of this device.
            let images = unsafe { images_of(frame) };
            let planes = plane_views(&gpu, &images.images, *layout)?;
            video_views.push(planes.iter().map(|view| view.view).collect());
            views.extend(planes);
        }
        let output_views = match self.format {
            VulkanFrameFormat::Nv12 => {
                // SAFETY: a frame of this element's own pool.
                let images = unsafe { images_of(output) };
                let planes = plane_views(&gpu, &images.images, LayerLayout::Nv12)?;
                let handles = [planes[0].view, planes[1].view];
                views.extend(planes);
                Some(handles)
            }
            VulkanFrameFormat::Bgra => None,
        };

        // From here each frame says it is what this recording makes it: the
        // recording has to be submitted, or every claim abandoned.
        let mut claims = Claim::default();
        let mut claimed = HashSet::new();
        for draw in draws {
            if let Draw::Video { frame, .. } = draw
                // SAFETY: a live frame; only its `AVVkFrame`'s address is taken.
                && claimed.insert(unsafe { (*frame.as_ptr()).data[0] as usize })
            {
                // SAFETY: validated as a live Vulkan frame of this device, and
                // claimed once however many layers show it.
                claims.extend(unsafe {
                    claim(
                        frame,
                        vk::PipelineStageFlags2::COMPUTE_SHADER,
                        vk::AccessFlags2::SHADER_SAMPLED_READ,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    )
                });
            }
        }
        // SAFETY: a frame of this element's own pool, on its device.
        claims.extend(unsafe {
            match self.format {
                VulkanFrameFormat::Bgra => claim(
                    output,
                    vk::PipelineStageFlags2::COPY,
                    vk::AccessFlags2::TRANSFER_WRITE,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                ),
                VulkanFrameFormat::Nv12 => claim(
                    output,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    vk::ImageLayout::GENERAL,
                ),
            }
        });

        // SAFETY: a frame of this element's own pool.
        let output_image = unsafe { images_of(output) }.images[0].0;
        let recorded = self.record(
            draws,
            &claims,
            &staging,
            &video_views,
            output_views,
            output_image,
        );
        let submitted = recorded.and_then(|()| {
            self.recording
                .submit(&claims.waits, &claims.signals)
                .map_err(VulkanVideoCompositorError::from)
        });
        if let Err(error) = submitted {
            abandon(&gpu.device, &claims);
            return Err(error);
        }
        // Everything the recording used lives until it has finished.
        let waited = self.recording.wait();
        drop(views);
        drop(staging);
        waited?;
        for mask in self.masks.values_mut() {
            mask.uploaded = true;
        }
        Ok(())
    }

    /// Records the whole composite: the barriers, the masks' uploads, the
    /// background, every layer, and the canvas into the output frame.
    fn record(
        &mut self,
        draws: &[Draw],
        claims: &Claim,
        staging: &[(usize, HostBuffer)],
        video_views: &[Vec<vk::ImageView>],
        output_views: Option<[vk::ImageView; 2]>,
        output_image: vk::Image,
    ) -> std::result::Result<(), VulkanVideoCompositorError> {
        self.recording.begin()?;
        let gpu = Arc::clone(&self.device);
        let device = &gpu.device;
        let commands = self.recording.commands;
        let whole = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);

        // Every frame to where this uses it, the canvas to where it is
        // drawn (its last contents discarded), and each new mask to where it
        // is copied into.
        let mut barriers = claims.barriers.clone();
        barriers.push(
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(
                    vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
                )
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(self.canvas.image)
                .subresource_range(whole),
        );
        for (key, _) in staging {
            barriers.push(
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(self.masks[key].image.image)
                    .subresource_range(whole),
            );
        }
        // SAFETY: the command buffer is recording; every image named is live
        // and outlives the recording, and every barrier's layouts are the
        // ones the images are in.
        unsafe {
            device.cmd_pipeline_barrier2(
                commands,
                &vk::DependencyInfo::default().image_memory_barriers(&barriers),
            );
        }

        if !staging.is_empty() {
            let mut uploaded = Vec::with_capacity(staging.len());
            for (key, buffer) in staging {
                let mask = &self.masks[key];
                // SAFETY: the buffer holds the mask's `width * height`
                // tightly packed bytes and the image is in the layout a copy
                // writes.
                unsafe {
                    device.cmd_copy_buffer_to_image(
                        commands,
                        buffer.buffer,
                        mask.image.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[vk::BufferImageCopy::default()
                            .image_subresource(
                                vk::ImageSubresourceLayers::default()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .layer_count(1),
                            )
                            .image_extent(vk::Extent3D {
                                width: mask.image.width,
                                height: mask.image.height,
                                depth: 1,
                            })],
                    );
                }
                uploaded.push(
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::COPY)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(mask.image.image)
                        .subresource_range(whole),
                );
            }
            // SAFETY: as for the barriers above.
            unsafe {
                device.cmd_pipeline_barrier2(
                    commands,
                    &vk::DependencyInfo::default().image_memory_barriers(&uploaded),
                );
            }
        }

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
        let canvas = self.canvas_view.view;
        dispatch(
            device,
            &mut self.recording,
            &self.kernels.fill,
            &background,
            Bindings {
                canvas,
                sampler: None,
                sampled: &[],
            },
        )?;
        for (draw, planes) in draws.iter().zip(video_views) {
            canvas_barrier(device, commands);
            match draw {
                Draw::Video { layout, step, .. } => {
                    let kernel = match layout {
                        LayerLayout::Nv12 => &self.kernels.nv12,
                        LayerLayout::Bgra => &self.kernels.bgra,
                    };
                    let sampled: Vec<_> = planes
                        .iter()
                        .enumerate()
                        .map(|(index, &view)| (index as u32 + 2, view))
                        .collect();
                    dispatch(
                        device,
                        &mut self.recording,
                        kernel,
                        step,
                        Bindings {
                            canvas,
                            sampler: Some(self.sampler.sampler),
                            sampled: &sampled,
                        },
                    )?;
                }
                Draw::Text { mask, step } => {
                    let view = self.masks[&(Arc::as_ptr(mask) as usize)].view.view;
                    dispatch(
                        device,
                        &mut self.recording,
                        &self.kernels.text,
                        step,
                        Bindings {
                            canvas,
                            sampler: None,
                            sampled: &[(2, view)],
                        },
                    )?;
                }
            }
        }

        match (self.format, output_views) {
            (VulkanFrameFormat::Nv12, Some([luma, chroma])) => {
                canvas_barrier(device, commands);
                let kernel = self
                    .kernels
                    .to_nv12
                    .as_ref()
                    .expect("made for an NV12 canvas");
                let step = Step::default();
                let set = self.recording.set(kernel.set_layout)?;
                let canvas = [vk::DescriptorImageInfo::default()
                    .image_view(self.canvas_view.view)
                    .image_layout(vk::ImageLayout::GENERAL)];
                let luma = [vk::DescriptorImageInfo::default()
                    .image_view(luma)
                    .image_layout(vk::ImageLayout::GENERAL)];
                let chroma = [vk::DescriptorImageInfo::default()
                    .image_view(chroma)
                    .image_layout(vk::ImageLayout::GENERAL)];
                let writes = [
                    storage_write(set, 0, &canvas),
                    storage_write(set, 4, &luma),
                    storage_write(set, 5, &chroma),
                ];
                // SAFETY: the set is this recording's, of the kernel's
                // layout, and every view is live until it has finished.
                unsafe {
                    device.update_descriptor_sets(&writes, &[]);
                    device.cmd_bind_pipeline(
                        commands,
                        vk::PipelineBindPoint::COMPUTE,
                        kernel.pipeline,
                    );
                    device.cmd_bind_descriptor_sets(
                        commands,
                        vk::PipelineBindPoint::COMPUTE,
                        kernel.pipeline_layout,
                        0,
                        &[set],
                        &[],
                    );
                    device.cmd_push_constants(
                        commands,
                        kernel.pipeline_layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        &step.bytes(),
                    );
                    device.cmd_dispatch(
                        commands,
                        (width / 2).div_ceil(8),
                        (height / 2).div_ceil(8),
                        1,
                    );
                }
            }
            _ => {
                // SAFETY: the copy reads the canvas once every dispatch has
                // written it, into the output image the claim put in the
                // layout a copy writes; both are `width` by `height`, and
                // their formats are the same size — the canvas holds the
                // frame's own byte order.
                unsafe {
                    let barrier = [vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                        .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                        .old_layout(vk::ImageLayout::GENERAL)
                        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(self.canvas.image)
                        .subresource_range(whole)];
                    device.cmd_pipeline_barrier2(
                        commands,
                        &vk::DependencyInfo::default().image_memory_barriers(&barrier),
                    );
                    let target = output_image;
                    let layers = vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1);
                    device.cmd_copy_image(
                        commands,
                        self.canvas.image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        target,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[vk::ImageCopy::default()
                            .src_subresource(layers)
                            .dst_subresource(layers)
                            .extent(vk::Extent3D {
                                width,
                                height,
                                depth: 1,
                            })],
                    );
                }
            }
        }
        Ok(())
    }

    fn push_frame(&mut self, bus: &Bus) -> std::result::Result<(), VulkanVideoCompositorError> {
        let composing = Instant::now();
        let output = self.compose_frame()?;
        if let Some(ticks) = &self.ticks {
            ticks.made(composing.elapsed());
        }
        if let Err(error) = self.pad.push(MediaBuffer::Video(Arc::new(output))) {
            bus.post(
                &self.pp_log,
                BusEvent::Error {
                    element_type: ElementType::VulkanVideoCompositor,
                    name: self.name.clone(),
                    error,
                },
            );
        }
        Ok(())
    }
}

/// What one dispatch reads besides its push constants: the canvas, the
/// sampler where the kernel samples a picture, and images at their bindings.
struct Bindings<'a> {
    canvas: vk::ImageView,
    sampler: Option<vk::Sampler>,
    sampled: &'a [(u32, vk::ImageView)],
}

/// Records one dispatch of `kernel` over `step`'s region.
fn dispatch(
    device: &ash::Device,
    recording: &mut Recording,
    kernel: &Kernel,
    step: &Step,
    bindings: Bindings<'_>,
) -> std::result::Result<(), VulkanVideoCompositorError> {
    let commands = recording.commands;
    let set = recording.set(kernel.set_layout)?;
    let canvas = [vk::DescriptorImageInfo::default()
        .image_view(bindings.canvas)
        .image_layout(vk::ImageLayout::GENERAL)];
    let sampler = bindings
        .sampler
        .map(|sampler| [vk::DescriptorImageInfo::default().sampler(sampler)]);
    let sampled = bindings.sampled;
    let images: Vec<[vk::DescriptorImageInfo; 1]> = sampled
        .iter()
        .map(|&(_, view)| {
            [vk::DescriptorImageInfo::default()
                .image_view(view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]
        })
        .collect();
    let mut writes = vec![storage_write(set, 0, &canvas)];
    if let Some(sampler) = &sampler {
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .image_info(sampler),
        );
    }
    for ((binding, _), info) in sampled.iter().zip(&images) {
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(*binding)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(info),
        );
    }
    let (groups_x, groups_y) = step.groups();
    // SAFETY: the set is this recording's, of the kernel's layout, and every
    // view and the sampler live until it has finished.
    unsafe {
        device.update_descriptor_sets(&writes, &[]);
        device.cmd_bind_pipeline(commands, vk::PipelineBindPoint::COMPUTE, kernel.pipeline);
        device.cmd_bind_descriptor_sets(
            commands,
            vk::PipelineBindPoint::COMPUTE,
            kernel.pipeline_layout,
            0,
            &[set],
            &[],
        );
        device.cmd_push_constants(
            commands,
            kernel.pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            &step.bytes(),
        );
        device.cmd_dispatch(commands, groups_x, groups_y, 1);
    }
    Ok(())
}

/// A storage-image descriptor at `binding` of `set`.
fn storage_write<'a>(
    set: vk::DescriptorSet,
    binding: u32,
    info: &'a [vk::DescriptorImageInfo; 1],
) -> vk::WriteDescriptorSet<'a> {
    vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .image_info(info)
}

/// Orders one dispatch's writes to the canvas before the next one's reads.
fn canvas_barrier(device: &ash::Device, commands: vk::CommandBuffer) {
    let barrier = [vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .dst_access_mask(
            vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
        )];
    // SAFETY: the command buffer is recording, and a memory barrier names no
    // resource.
    unsafe {
        device.cmd_pipeline_barrier2(
            commands,
            &vk::DependencyInfo::default().memory_barriers(&barrier),
        );
    }
}

/// A view of each plane of a picture in `layout`: NV12's luma and its
/// chroma, from the planes of one multi-planar image or from an image each;
/// BGRA's one.
fn plane_views(
    gpu: &Arc<DeviceShared>,
    images: &[(vk::Image, vk::Format)],
    layout: LayerLayout,
) -> std::result::Result<Vec<View>, VulkanVideoCompositorError> {
    let views = match (layout, images) {
        (LayerLayout::Bgra, [(image, _), ..]) => vec![View::new(
            gpu,
            *image,
            vk::Format::B8G8R8A8_UNORM,
            vk::ImageAspectFlags::COLOR,
        )?],
        (LayerLayout::Nv12, [(image, _)]) => vec![
            View::new(
                gpu,
                *image,
                vk::Format::R8_UNORM,
                vk::ImageAspectFlags::PLANE_0,
            )?,
            View::new(
                gpu,
                *image,
                vk::Format::R8G8_UNORM,
                vk::ImageAspectFlags::PLANE_1,
            )?,
        ],
        (LayerLayout::Nv12, [(luma, _), (chroma, _), ..]) => vec![
            View::new(
                gpu,
                *luma,
                vk::Format::R8_UNORM,
                vk::ImageAspectFlags::COLOR,
            )?,
            View::new(
                gpu,
                *chroma,
                vk::Format::R8G8_UNORM,
                vk::ImageAspectFlags::COLOR,
            )?,
        ],
        _ => {
            return Err(VulkanVideoCompositorError::UnsupportedLayout(
                ffmpeg::format::Pixel::VULKAN,
            ));
        }
    };
    Ok(views)
}

impl Element for VulkanVideoCompositor {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::VulkanVideoCompositor
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

impl Source for VulkanVideoCompositor {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl SourceElement for VulkanVideoCompositor {
    fn is_live(&self) -> bool {
        self.options.mode.is_live()
    }

    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        pp_info!(self, "started");
        match self.options.mode {
            RenderMode::Live => self.run_live(control, bus),
            RenderMode::Offline { end } => run_offline(self, control, bus, end),
        }
    }
}

impl VulkanVideoCompositor {
    /// Emits at its own rate by the wall clock — see [`RenderMode::Live`].
    fn run_live(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        let mut schedule = PeriodicSchedule::new(self.shared.frame_rate.interval(), Instant::now());
        loop {
            let outcome = drain_control(control, self, bus)?;
            if outcome.stopped {
                pp_info!(self, "stopped");
                return Ok(());
            }
            if outcome.paused_for > Duration::ZERO {
                schedule.resume_after_pause(outcome.paused_for, Instant::now());
            }

            // Followed here rather than at construction, so a rate set while
            // this is running is kept from the next tick on.
            let interval = self.shared.frame_rate.interval();
            let now = Instant::now();
            if schedule.interval() != interval {
                pp_info!(self, "frame rate is now {}", self.frame_rate());
                schedule.set_interval(interval, now);
            }
            if !schedule.is_due(now) {
                thread::sleep(schedule.remaining(now).min(CONTROL_POLL_INTERVAL));
                continue;
            }

            self.push_frame(bus)?;
            let missed = schedule.advance_after_tick(Instant::now());
            if let Some(ticks) = &self.ticks {
                ticks.missed(missed);
            }
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

impl OfflineCompositor for VulkanVideoCompositor {
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

    fn draw(&mut self, bus: &Bus) -> Result<()> {
        Ok(self.push_frame(bus)?)
    }

    fn end_render(&mut self) -> Result<()> {
        self.pad.push_eos(&self.pp_log)
    }
}

impl Drop for VulkanVideoCompositor {
    fn drop(&mut self) {
        pp_info!(self, "dropped: releasing hw contexts");
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

fn validate_input_frame(
    frame: &ffmpeg::frame::Video,
    device_ctx: *const ffi::AVHWDeviceContext,
) -> std::result::Result<LayerLayout, VulkanVideoCompositorError> {
    let layout = match sw_format_of(frame, device_ctx) {
        Ok(ffmpeg::format::Pixel::NV12) => LayerLayout::Nv12,
        Ok(ffmpeg::format::Pixel::BGRA) => LayerLayout::Bgra,
        Ok(other) => return Err(VulkanVideoCompositorError::UnsupportedLayout(other)),
        Err(NotOurs::NotVulkan(format)) => {
            return Err(VulkanVideoCompositorError::NotVulkan(format));
        }
        Err(NotOurs::ForeignDevice) => return Err(VulkanVideoCompositorError::ForeignDevice),
    };
    if frame.width() == 0 || frame.height() == 0 {
        return Err(VulkanVideoCompositorError::InvalidInputDimensions {
            width: frame.width(),
            height: frame.height(),
        });
    }
    Ok(layout)
}

fn validate_output_options(
    options: VideoCompositorOptions,
    format: VulkanFrameFormat,
) -> std::result::Result<(), VulkanVideoCompositorError> {
    if options.width < 2
        || options.height < 2
        || !options.width.is_multiple_of(2)
        || !options.height.is_multiple_of(2)
        || options.width > MAX_DIMENSION
        || options.height > MAX_DIMENSION
    {
        return Err(VulkanVideoCompositorError::InvalidOutputDimensions {
            width: options.width,
            height: options.height,
        });
    }
    if options.frame_rate.numerator() <= 0 || options.frame_rate.denominator() <= 0 {
        return Err(VulkanVideoCompositorError::InvalidFrameRate(
            options.frame_rate,
        ));
    }
    // NV12 has nowhere to keep it — refused rather than quietly made opaque.
    if options.background_alpha != 255 && format == VulkanFrameFormat::Nv12 {
        return Err(VulkanVideoCompositorError::TranslucentBackground(
            options.background_alpha,
        ));
    }
    Ok(())
}

/// Thin adapters over the shared, backend-agnostic checks in
/// [`super::super::video_layer`].
fn validate_layer(layer: VideoLayer) -> std::result::Result<(), VulkanVideoCompositorError> {
    video_layer::validate_layer(layer).map_err(layer_error)?;
    validate_opacity(layer.opacity)
}

fn validate_rect(rect: VideoRect) -> std::result::Result<(), VulkanVideoCompositorError> {
    video_layer::validate_rect(rect).map_err(layer_error)
}

fn validate_opacity(opacity: f32) -> std::result::Result<(), VulkanVideoCompositorError> {
    video_layer::validate_opacity(opacity).map_err(layer_error)
}

fn layer_error(error: VideoLayerError) -> VulkanVideoCompositorError {
    match error {
        VideoLayerError::InvalidDimensions { width, height } => {
            VulkanVideoCompositorError::InvalidLayerDimensions { width, height }
        }
        VideoLayerError::InvalidOpacity(opacity) => {
            VulkanVideoCompositorError::InvalidOpacity(opacity)
        }
        VideoLayerError::InvalidInputDimensions { width, height } => {
            VulkanVideoCompositorError::InvalidInputDimensions { width, height }
        }
        VideoLayerError::ScaledLayerTooLarge { width, height } => {
            VulkanVideoCompositorError::ScaledLayerTooLarge { width, height }
        }
        VideoLayerError::InvalidSourceRegion { width, height } => {
            VulkanVideoCompositorError::InvalidSourceRegion { width, height }
        }
    }
}

#[cfg(test)]
mod tests;

super::super::control::compositor_control!(
    VulkanVideoCompositorHandle,
    VulkanVideoLayerHandle,
    VulkanTextLayerHandle
);
