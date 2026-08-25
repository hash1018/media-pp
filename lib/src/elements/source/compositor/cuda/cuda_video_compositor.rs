use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicU64, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use arc_swap::ArcSwapOption;
use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_debug, pp_error, pp_info};

use super::super::sw_video_compositor::VideoCompositorOptions;
use super::super::text_layer::{TextLayer, TextRasterError, rasterize_coverage};
use super::super::video_layer::{
    self, LayerGeometry, MAX_DIMENSION, VideoFit, VideoInputId, VideoLayer, VideoLayerError,
    VideoRect, layer_geometry,
};
use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    color::Color,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    control::{ControlMsg, ControlReceiver, drain_control},
    element::{Element, ElementType, Sink, Source, SourceElement, element_pp_log},
    elements::{CudaScalerInterp, filter::scaler::cuda::scale_graph::CudaScaleGraph},
    error::Result,
    pad::SrcPad,
    platform::cuda::{
        CudaDevice, CudaFrameFormat,
        driver::{CudaDriver, CudaDriverError, CudaMask, Nv12Region, Nv12Surface},
        frame::create_hw_frames_ctx,
    },
    platform::ffmpeg::AvBufferRef,
    pool::{UnboundObjectPool, UnboundObjectPoolRef},
    schedule::PeriodicSchedule,
};

const OUTPUT_POOL_SIZE: usize = 4;
const CONTROL_POLL_INTERVAL: Duration = Duration::from_millis(5);

/// Errors specific to [`CudaVideoCompositor`].
#[derive(Debug, ThisError)]
pub enum CudaVideoCompositorError {
    /// FFmpeg rejected frame or hardware-context allocation.
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),

    /// A CUDA driver call failed.
    #[error(transparent)]
    Driver(#[from] CudaDriverError),

    /// Scaling an input layer into its destination rectangle failed.
    #[error("failed to scale a layer: {0}")]
    Scale(#[from] crate::elements::CudaScalerError),

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

    /// The input frame is not backed by CUDA hardware surfaces.
    #[error("CudaVideoCompositorInputSink only composites CUDA frames, got {0:?}")]
    UnsupportedFormat(ffmpeg::format::Pixel),

    /// The CUDA surface is not in the NV12 format required by the compositor kernel.
    #[error("CudaVideoCompositor only composites NV12 surfaces, got {0:?}")]
    UnsupportedSurfaceFormat(ffmpeg::format::Pixel),

    /// A CUDA frame does not retain the hardware frames context that owns it.
    #[error("CUDA frame has no hardware frames context")]
    MissingFramesContext,

    /// An input frame belongs to a different CUDA context.
    #[error("CUDA frame belongs to a different CUDA context than this compositor")]
    ForeignContext,

    /// A CUDA frame contains no usable device-memory plane.
    #[error("CUDA frame carries no device pointers")]
    MissingPlane,

    /// An input sink received a buffer other than decoded video.
    #[error(
        "CudaVideoCompositorInputSink only accepts decoded Video frames, got a {0}; link it after a decoder or upload"
    )]
    UnsupportedBuffer(&'static str),

    /// FFmpeg could not allocate the compositor's CUDA output frame pool.
    #[error("failed to allocate the CUDA frames context")]
    HwFramesAlloc,

    /// FFmpeg could not acquire a frame from the CUDA output pool.
    #[error("failed to take an output frame from the CUDA pool (code {0})")]
    HwFrameGet(i32),

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

    /// Seeking was requested on a live compositor with no stored timeline.
    #[error("CudaVideoCompositor doesn't support seeking a live composition")]
    SeekUnsupported,
}

struct VideoInput {
    id: VideoInputId,
    /// The hot producer/consumer path is an atomic latest-value slot:
    /// input pipelines replace the pointer without taking the layer lock,
    /// and the compositor acquires a stable Arc snapshot independently.
    latest_frame: ArcSwapOption<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    layer: Mutex<VideoLayer>,
}

struct CompositorShared {
    inputs: Mutex<HashMap<Arc<str>, Arc<VideoInput>>>,
    /// Text layers are kept apart from video inputs because they are not
    /// inputs at all: nothing pushes frames into one. They are drawn after
    /// every video layer, in registration order.
    text_layers: Mutex<Vec<(Arc<str>, Arc<TextLayerState>)>>,
    next_input_id: AtomicU64,
    /// Shared so a [`CudaTextLayerHandle`] can upload a freshly rasterized
    /// mask from whichever thread called `set_text`.
    driver: Arc<CudaDriver>,
    /// Captured from the compositor's own [`CudaDevice`] so every input sink
    /// can reject a frame from another CUDA context before it ever reaches a
    /// device pointer. Only ever compared.
    device_ctx: *const ffi::AVHWDeviceContext,
}

// SAFETY: `device_ctx` is only ever compared, never dereferenced, and the
// compositor holds its own reference to the context for its whole life, so
// the pointer cannot go stale while any input sink is alive.
unsafe impl Send for CompositorShared {}

// SAFETY: as above for the raw pointer, and every field beside it carries its
// own synchronization — the maps are behind mutexes, the counters are atomic,
// and the driver is itself `Sync`. This is what lets an input sink on one
// thread share this with the compositor's own.
unsafe impl Sync for CompositorShared {}

/// A cheaply cloneable handle for adding and removing compositor inputs —
/// the CUDA sibling of [`crate::elements::SwVideoCompositorHandle`].
#[derive(Clone)]
pub struct CudaVideoCompositorHandle {
    shared: Weak<CompositorShared>,
}

/// The two endpoints created for one compositor input registration.
/// Move `sink` into the upstream pipeline and retain `layer` in application
/// code for runtime placement changes.
pub struct CudaVideoCompositorInput {
    /// Terminal sink to attach to the input pipeline branch.
    pub sink: Box<dyn Sink>,
    /// Runtime control for this input's placement and visibility.
    pub layer: CudaVideoLayerHandle,
}

impl CudaVideoCompositorHandle {
    /// Registers an input and returns its terminal Sink plus independent
    /// runtime layer control. Reusing `name` replaces the old registration;
    /// the replaced sink and layer handle can no longer affect this
    /// compositor, exactly as
    /// [`crate::elements::SwVideoCompositorHandle::add_source`] documents.
    pub fn add_source(
        &self,
        name: impl Into<String>,
        layer: VideoLayer,
    ) -> std::result::Result<CudaVideoCompositorInput, CudaVideoCompositorError> {
        validate_layer(layer)?;
        let Some(shared) = self.shared.upgrade() else {
            return Err(CudaVideoCompositorError::SourceRemoved);
        };
        let name: Arc<str> = name.into().into();
        let id = VideoInputId(shared.next_input_id.fetch_add(1, Ordering::Relaxed));
        let input = Arc::new(VideoInput {
            id,
            latest_frame: ArcSwapOption::empty(),
            layer: Mutex::new(layer),
        });
        shared
            .inputs
            .lock()
            .unwrap()
            .insert(name.clone(), input.clone());

        Ok(CudaVideoCompositorInput {
            sink: Box::new(CudaVideoCompositorInputSink {
                pp_log: element_pp_log(ElementType::CudaVideoCompositor, &name, None),
                name: name.clone(),
                id,
                shared: Arc::downgrade(&shared),
                input: Arc::downgrade(&input),
            }),
            layer: CudaVideoLayerHandle {
                id,
                name,
                input: Arc::downgrade(&input),
            },
        })
    }

    /// Removes the registration under `name`, if any.
    pub fn remove_source(&self, name: &str) {
        if let Some(shared) = self.shared.upgrade() {
            shared.inputs.lock().unwrap().remove(name);
        }
    }

    /// Returns the number of inputs currently registered, or zero after shutdown.
    pub fn source_count(&self) -> usize {
        self.shared
            .upgrade()
            .map(|shared| shared.inputs.lock().unwrap().len())
            .unwrap_or(0)
    }
}

/// Runtime placement control for one registered input — the CUDA sibling of
/// [`crate::elements::SwVideoLayerHandle`], with the same API.
#[derive(Clone)]
pub struct CudaVideoLayerHandle {
    id: VideoInputId,
    name: Arc<str>,
    input: Weak<VideoInput>,
}

impl CudaVideoLayerHandle {
    /// Returns the stable identity of this particular input registration.
    pub fn id(&self) -> VideoInputId {
        self.id
    }

    /// Returns the registration name, which may be reused by a newer input.
    pub fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    /// Returns the current settings, or `None` after the input is removed.
    pub fn layer(&self) -> Option<VideoLayer> {
        self.input
            .upgrade()
            .map(|input| *input.layer.lock().unwrap())
    }

    /// Atomically replaces every layer setting.
    pub fn set_layer(
        &self,
        layer: VideoLayer,
    ) -> std::result::Result<(), CudaVideoCompositorError> {
        validate_layer(layer)?;
        self.update(|current| *current = layer)
    }

    /// Replaces the destination rectangle while retaining other settings.
    pub fn set_rect(&self, rect: VideoRect) -> std::result::Result<(), CudaVideoCompositorError> {
        validate_rect(rect)?;
        self.update(|layer| layer.rect = rect)
    }

    /// Replaces opacity after validating the `0.0..=1.0` range.
    pub fn set_opacity(&self, opacity: f32) -> std::result::Result<(), CudaVideoCompositorError> {
        validate_opacity(opacity)?;
        self.update(|layer| layer.opacity = opacity)
    }

    /// Changes the stacking order; larger values are drawn later.
    pub fn set_z_index(&self, z_index: i32) -> std::result::Result<(), CudaVideoCompositorError> {
        self.update(|layer| layer.z_index = z_index)
    }

    /// Shows or hides the input without removing its registration.
    pub fn set_visible(&self, visible: bool) -> std::result::Result<(), CudaVideoCompositorError> {
        self.update(|layer| layer.visible = visible)
    }

    /// Changes how the input aspect ratio maps into its rectangle.
    pub fn set_fit(&self, fit: VideoFit) -> std::result::Result<(), CudaVideoCompositorError> {
        self.update(|layer| layer.fit = fit)
    }

    fn update(
        &self,
        change: impl FnOnce(&mut VideoLayer),
    ) -> std::result::Result<(), CudaVideoCompositorError> {
        let Some(input) = self.input.upgrade() else {
            return Err(CudaVideoCompositorError::SourceRemoved);
        };
        change(&mut input.layer.lock().unwrap());
        Ok(())
    }
}

/// The terminal Sink for one compositor input. Keeps only the latest frame:
/// the compositor emits on its own clock, so an input that runs faster than
/// the output rate simply has its older frames dropped.
pub struct CudaVideoCompositorInputSink {
    pp_log: PpLog,
    name: Arc<str>,
    id: VideoInputId,
    shared: Weak<CompositorShared>,
    input: Weak<VideoInput>,
}

impl CudaVideoCompositorInputSink {
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

impl Element for CudaVideoCompositorInputSink {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::CudaVideoCompositor
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Sink for CudaVideoCompositorInputSink {
    /// Every layer is composited on the device, so each input arrives there just as the composed output does.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::frame(
            MediaKind::VideoFrame,
            MemoryDomain::Cuda,
        ))
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Video(frame) => {
                let Some(shared) = self.shared.upgrade() else {
                    return Err(CudaVideoCompositorError::SourceRemoved.into());
                };
                // Validated here rather than at compose time so a
                // misconfigured input names *itself* in the error, and so a
                // frame from another CUDA context never reaches a
                // `cuMemcpy2D` at all.
                validate_input_frame(&frame, shared.device_ctx)
                    .inspect_err(|error| pp_error!(self, "{error}"))?;
                if let Some(input) = self.input.upgrade() {
                    input.latest_frame.store(Some(frame));
                }
                Ok(())
            }
            MediaBuffer::Eos => {
                pp_debug!(self, "input reached eos; leaving its last frame in place");
                Ok(())
            }
            other => {
                pp_error!(self, "unsupported buffer: {}", other.kind());
                Err(CudaVideoCompositorError::UnsupportedBuffer(other.kind()).into())
            }
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        // Terminal for its own branch: nothing downstream to forward to. A
        // `Stop` means this upstream pipeline is done, so the registration
        // goes with it — same as `SwVideoCompositorInputSink`.
        if matches!(msg, ControlMsg::Stop) {
            self.detach();
        }
        Ok(())
    }
}

struct InputSnapshot {
    id: VideoInputId,
    layer: VideoLayer,
    frame: Option<Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>>,
}

/// Composites the latest frames from any number of independent CUDA input
/// pipelines into one fixed-rate NV12 `Pixel::CUDA` stream, without any frame
/// ever leaving the GPU — the CUDA sibling of
/// [`crate::elements::SwVideoCompositor`] and
/// `D3d11VideoCompositor`, driving the same
/// [`VideoLayer`]/[`VideoRect`]/[`VideoFit`] API.
///
/// Like the other two, this is a [`SourceElement`], not a one-input filter:
/// upstream pipelines terminate at the sinks returned by
/// [`CudaVideoCompositorHandle::add_source`], while this element's own
/// pipeline drives output on its independent clock. Input frame PTS values
/// therefore do not become output PTS; output advances by one tick in
/// [`CudaVideoCompositor::time_base`] for every composed frame.
///
/// # How it draws
///
/// Each layer is resized by `scale_cuda` and then placed with a 2D
/// device-to-device copy. That combination is what makes every
/// [`VideoFit`] work: `Cover` needs cropping, which no CUDA filter in
/// libavfilter offers, but a copy simply takes a sub-rectangle. Moving,
/// hiding, or reordering a layer costs nothing at all — only a change in a
/// layer's *scaled size* rebuilds anything.
///
/// # NV12 only
///
/// Inputs and output are NV12 CUDA surfaces. A capture produces BGRA and
/// nothing on the CUDA path converts RGB to YUV (see
/// [`crate::elements::CudaScaler`]), so a capture has to be converted on the
/// CPU before it can be composited here. Decoded video needs nothing:
/// [`crate::elements::CudaDecoder`] already produces NV12.
///
/// # Blending
///
/// A translucent layer is mixed by a small CUDA kernel this crate ships as
/// PTX text and the driver JIT-compiles at startup — see `platform::cuda`'s
/// `BLEND_PTX`. Nothing about that needs a CUDA toolkit, so `opacity` works
/// here exactly as it does on the other two backends. An opaque layer skips
/// the kernel entirely and is placed with a plain copy, which is why the
/// common case costs no arithmetic at all.
///
/// # Even coordinates
///
/// NV12 chroma is subsampled 2x2, so a layer's placement and size are
/// aligned to even pixels. A rectangle at an odd coordinate is drawn up to
/// one pixel away from where it was asked for; nothing else changes.
pub struct CudaVideoCompositor {
    pp_log: PpLog,
    name: Arc<str>,
    shared: Arc<CompositorShared>,
    options: VideoCompositorOptions,
    frame_interval: Duration,
    frame_index: i64,
    driver: Arc<CudaDriver>,
    /// This element's own reference to the shared context, released in
    /// `Drop` — it is what keeps `shared.device_ctx` a valid identity.
    _hw_device_ctx: Arc<AvBufferRef>,
    /// The pool output surfaces are allocated from.
    hw_frames_ctx: AvBufferRef,
    scalers: HashMap<VideoInputId, CudaScaleGraph>,
    /// Reuses only the small CPU-side `AVFrame` wrapper; the surface itself
    /// comes from `hw_frames_ctx`'s own pool.
    output_pool: UnboundObjectPool<ffmpeg::frame::Video>,
    pad: SrcPad,
}

// SAFETY: both buffers are heap-allocated FFmpeg buffers with no thread
// affinity of their own, `CudaDriver` pushes its context onto whichever
// thread calls it, and every field is otherwise touched only through
// `&mut self` on this element's single source thread.
unsafe impl Send for CudaVideoCompositor {}

impl CudaVideoCompositor {
    /// `device` must be the same [`CudaDevice`] every input pipeline's CUDA
    /// elements were built from; a frame from another context is rejected by
    /// the input sink rather than read from meaningless pointers.
    ///
    /// Output dimensions must be even — see this type's own docs on chroma
    /// alignment.
    pub fn new(
        name: impl Into<String>,
        device: &CudaDevice,
        options: VideoCompositorOptions,
    ) -> std::result::Result<(Self, CudaVideoCompositorHandle), CudaVideoCompositorError> {
        validate_output_options(options)?;
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::CudaVideoCompositor, &name, None);

        let driver = Arc::new(CudaDriver::retain_primary()?);
        let hw_device_ctx = device.retain();
        // SAFETY: `create_hw_frames_ctx`'s contract is a live device context, which
        // is what the owned `AvBufferRef` beside it is.
        let hw_frames_ctx = match unsafe {
            create_hw_frames_ctx(
                &hw_device_ctx,
                CudaFrameFormat::Nv12,
                options.width,
                options.height,
            )
        } {
            Ok(ctx) => ctx,
            Err(error) => {
                pp_error!(pp_log: &pp_log, "failed to build the output pool: {error}");
                return Err(CudaVideoCompositorError::HwFramesAlloc);
            }
        };
        // SAFETY: `hw_device_ctx` owns a live `AVBufferRef` for a CUDA device
        // context, whose `data` is that `AVHWDeviceContext` by FFmpeg's own
        // definition. Only the pointer's identity is kept, to compare against an
        // incoming frame's; the reference held alongside it is what keeps that
        // identity from being reused by a different context.
        let device_ctx = unsafe { (*hw_device_ctx.as_ptr()).data as *const ffi::AVHWDeviceContext };

        let shared = Arc::new(CompositorShared {
            inputs: Mutex::new(HashMap::new()),
            text_layers: Mutex::new(Vec::new()),
            next_input_id: AtomicU64::new(1),
            driver: driver.clone(),
            device_ctx,
        });
        let frame_interval = Duration::from_secs_f64(
            options.frame_rate.denominator() as f64 / options.frame_rate.numerator() as f64,
        );
        pp_info!(
            pp_log: &pp_log,
            "created: {}x{}, frame_rate={}, format=CUDA/NV12",
            options.width,
            options.height,
            options.frame_rate
        );
        Ok((
            Self {
                name: name.clone(),
                pp_log,
                shared: shared.clone(),
                options,
                frame_interval,
                frame_index: 0,
                driver,
                _hw_device_ctx: hw_device_ctx,
                hw_frames_ctx,
                scalers: HashMap::new(),
                output_pool: UnboundObjectPool::new(
                    OUTPUT_POOL_SIZE,
                    ffmpeg::frame::Video::empty,
                    |_| {},
                ),
                pad: SrcPad::with_contract(
                    format!("{name}_src"),
                    OutputContract::Fixed(PortContract::frame(
                        MediaKind::VideoFrame,
                        MemoryDomain::Cuda,
                    )),
                ),
            },
            CudaVideoCompositorHandle {
                shared: Arc::downgrade(&shared),
            },
        ))
    }

    /// Every output frame is a CUDA-resident NV12 surface.
    pub fn format(&self) -> ffmpeg::format::Pixel {
        ffmpeg::format::Pixel::CUDA
    }

    /// Returns the fixed output width in pixels.
    pub fn width(&self) -> u32 {
        self.options.width
    }

    /// Returns the fixed output height in pixels.
    pub fn height(&self) -> u32 {
        self.options.height
    }

    /// Returns the configured output frame rate.
    pub fn frame_rate(&self) -> ffmpeg::Rational {
        self.options.frame_rate
    }

    /// The reciprocal of [`Self::frame_rate`] — output PTS advance by one
    /// tick in this base per composed frame.
    pub fn time_base(&self) -> ffmpeg::Rational {
        self.options.frame_rate.invert()
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

    /// Takes one surface from the output pool. The wrapper is pooled; the
    /// surface behind it comes from `hw_frames_ctx`, same split as
    /// [`crate::elements::CudaUpload`].
    fn output_frame(
        &mut self,
    ) -> std::result::Result<UnboundObjectPoolRef<ffmpeg::frame::Video>, CudaVideoCompositorError>
    {
        let mut output = self.output_pool.get();
        // SAFETY: `ptr` is the pooled wrapper's own `AVFrame`, and the unref before
        // the allocation is what hands its previous surface back — see the comment
        // beside it. The frames context is this element's own, held for its life.
        unsafe {
            let ptr = output.as_mut_ptr();
            // Releasing the previous surface here is what returns it to the
            // frames pool rather than holding it for this element's life.
            ffi::av_frame_unref(ptr);
            let code = ffi::av_hwframe_get_buffer(self.hw_frames_ctx.as_ptr(), ptr, 0);
            if code < 0 {
                return Err(CudaVideoCompositorError::HwFrameGet(code));
            }
        }
        Ok(output)
    }

    fn compose_frame(
        &mut self,
    ) -> std::result::Result<UnboundObjectPoolRef<ffmpeg::frame::Video>, CudaVideoCompositorError>
    {
        let mut snapshots = self.snapshots();
        let active: HashSet<_> = snapshots.iter().map(|snapshot| snapshot.id).collect();
        self.scalers.retain(|id, _| active.contains(id));
        snapshots.sort_by(|left, right| {
            left.layer
                .z_index
                .cmp(&right.layer.z_index)
                .then_with(|| left.id.cmp(&right.id))
        });

        let mut output = self.output_frame()?;
        let canvas =
            Nv12Surface::from_frame(&output).ok_or(CudaVideoCompositorError::MissingPlane)?;
        self.driver.fill_nv12(
            canvas,
            self.options.width,
            self.options.height,
            self.options.background,
        )?;

        let mut blended = false;
        for snapshot in snapshots {
            if !snapshot.layer.visible || snapshot.layer.opacity == 0.0 {
                continue;
            }
            let Some(frame) = snapshot.frame else {
                continue;
            };
            let frames_ctx = validate_input_frame(&frame, self.shared.device_ctx)?;
            let geometry = layer_geometry(
                frame.width(),
                frame.height(),
                snapshot.layer.rect,
                snapshot.layer.fit,
            )
            .map_err(layer_error)?;
            let Some(placement) = Placement::new(geometry, self.options.width, self.options.height)
            else {
                // Entirely outside the canvas or clipped away to nothing.
                continue;
            };

            let scaled = self
                .scalers
                .entry(snapshot.id)
                .or_insert_with(|| CudaScaleGraph::new(CudaScalerInterp::Bilinear))
                .scale(
                    &frame,
                    frames_ctx,
                    placement.image_width,
                    placement.image_height,
                )?;
            let Some(scaled) = scaled.into_iter().next_back() else {
                // `scale_cuda` emits one frame per frame; nothing to draw if
                // it held this one back for any reason.
                continue;
            };
            let layer_surface =
                Nv12Surface::from_frame(&scaled).ok_or(CudaVideoCompositorError::MissingPlane)?;
            if snapshot.layer.opacity >= 1.0 {
                // A copy moves whole rows at the memory system's own rate,
                // where the kernel reads, mixes, and writes every byte. Worth
                // keeping apart, since opaque is the common case.
                self.driver
                    .blit_nv12(layer_surface, canvas, placement.region)?;
            } else {
                let alpha = (snapshot.layer.opacity * 255.0).round().clamp(0.0, 255.0) as u8;
                self.driver
                    .blend_nv12(layer_surface, canvas, placement.region, alpha)?;
                blended = true;
            }
        }

        let text_layers: Vec<_> = self
            .shared
            .text_layers
            .lock()
            .unwrap()
            .iter()
            .map(|(_, state)| state.clone())
            .collect();
        for text in text_layers {
            if !text.visible.load(Ordering::Relaxed) {
                continue;
            }
            let Some(mask) = text.mask.load_full() else {
                continue;
            };
            let opacity = f32::from_bits(text.opacity.load(Ordering::Relaxed));
            if opacity <= 0.0 {
                continue;
            }
            let Some(placement) = TextPlacement::new(
                text.x.load(Ordering::Relaxed),
                text.y.load(Ordering::Relaxed),
                mask.width,
                mask.height,
                self.options.width,
                self.options.height,
            ) else {
                continue;
            };
            self.driver.blend_mask_nv12(
                canvas,
                placement.destination_x,
                placement.destination_y,
                &mask,
                placement.mask_x,
                placement.mask_y,
                placement.width,
                placement.height,
                text.color,
                (opacity * 255.0).round().clamp(0.0, 255.0) as u8,
            )?;
            blended = true;
        }

        if blended {
            // Blends are kernel launches, so they are asynchronous; a copy is
            // not. One wait per frame is what makes the finished surface safe
            // to hand downstream.
            self.driver.synchronize()?;
        }
        output.set_pts(Some(self.frame_index));
        self.frame_index += 1;
        Ok(output)
    }

    fn push_frame(&mut self, bus: &Bus) -> std::result::Result<(), CudaVideoCompositorError> {
        let output = self.compose_frame()?;
        if let Err(error) = self.pad.push(MediaBuffer::Video(Arc::new(output))) {
            bus.post(
                &self.pp_log,
                BusEvent::Error {
                    element_type: ElementType::CudaVideoCompositor,
                    name: self.name.clone(),
                    error,
                },
            );
        }
        Ok(())
    }
}

impl Element for CudaVideoCompositor {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::CudaVideoCompositor
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for CudaVideoCompositor {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl SourceElement for CudaVideoCompositor {
    fn is_live(&self) -> bool {
        true
    }

    fn is_seekable(&self) -> bool {
        false
    }

    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        pp_info!(self, "started");
        let mut schedule = PeriodicSchedule::new(self.frame_interval, Instant::now());
        loop {
            let outcome = drain_control(control, self, bus)?;
            if outcome.stopped {
                pp_info!(self, "stopped");
                return Ok(());
            }
            if outcome.paused_for > Duration::ZERO {
                schedule.resume_after_pause(outcome.paused_for, Instant::now());
            }

            let now = Instant::now();
            if !schedule.is_due(now) {
                thread::sleep(schedule.remaining(now).min(CONTROL_POLL_INTERVAL));
                continue;
            }

            self.push_frame(bus)?;
            schedule.advance_after_tick(Instant::now());
        }
    }

    fn seek(&mut self, _target: Duration) -> Result<Duration> {
        Err(CudaVideoCompositorError::SeekUnsupported.into())
    }
}

impl Drop for CudaVideoCompositor {
    fn drop(&mut self) {
        pp_info!(self, "dropped: releasing hw contexts");
    }
}

/// Where one layer's scaled image actually lands, in even coordinates.
///
/// The shared [`layer_geometry`] already decides the scaled size and its
/// placement; this turns that into the rectangle a `cuMemcpy2D` can move,
/// which means clipping it to both the layer's own rectangle and the canvas,
/// and aligning everything to the 2x2 chroma grid.
struct Placement {
    image_width: u32,
    image_height: u32,
    region: Nv12Region,
}

impl Placement {
    fn new(geometry: LayerGeometry, canvas_width: u32, canvas_height: u32) -> Option<Self> {
        // The scaled surface itself has to be NV12-shaped.
        let image_width = align_down(geometry.image_width).max(2);
        let image_height = align_down(geometry.image_height).max(2);
        // Aligning the image's own origin keeps `source = destination -
        // image_origin` even as well, which is what lets both offsets be
        // valid chroma coordinates at the same time.
        let image_x = align_down_signed(geometry.image_x);
        let image_y = align_down_signed(geometry.image_y);

        let (destination_x, source_x, width) = axis(
            image_x,
            image_width,
            i64::from(geometry.clip.x),
            geometry.clip.width,
            canvas_width,
        )?;
        let (destination_y, source_y, height) = axis(
            image_y,
            image_height,
            i64::from(geometry.clip.y),
            geometry.clip.height,
            canvas_height,
        )?;

        Some(Self {
            image_width,
            image_height,
            region: Nv12Region {
                source_x,
                source_y,
                destination_x,
                destination_y,
                width,
                height,
            },
        })
    }
}

/// One axis of the clip: the visible span is where the scaled image, the
/// layer's rectangle, and the canvas all overlap. Returns the destination
/// offset, the matching offset inside the scaled image, and the extent —
/// all even, or `None` when nothing of the layer is visible.
fn axis(
    image_origin: i64,
    image_extent: u32,
    clip_origin: i64,
    clip_extent: u32,
    canvas_extent: u32,
) -> Option<(u32, u32, u32)> {
    let start = image_origin.max(clip_origin).max(0);
    let end = (image_origin + i64::from(image_extent))
        .min(clip_origin + i64::from(clip_extent))
        .min(i64::from(canvas_extent));
    // Round the visible span inward so both ends stay on the chroma grid.
    let start = align_up_nonnegative(start);
    let end = align_down_signed(end);
    if end <= start {
        return None;
    }
    Some((
        start as u32,
        (start - image_origin) as u32,
        (end - start) as u32,
    ))
}

fn align_down(value: u32) -> u32 {
    value & !1
}

fn align_down_signed(value: i64) -> i64 {
    value - value.rem_euclid(2)
}

fn align_up_nonnegative(value: i64) -> i64 {
    value + value.rem_euclid(2)
}

/// Reads the frames context out of a CUDA frame after checking it is one
/// this compositor can actually draw: the right pixel format, a real frames
/// context, this compositor's own CUDA context, and an NV12 layout.
fn validate_input_frame(
    frame: &ffmpeg::frame::Video,
    device_ctx: *const ffi::AVHWDeviceContext,
) -> std::result::Result<*mut ffi::AVBufferRef, CudaVideoCompositorError> {
    if frame.format() != ffmpeg::format::Pixel::CUDA {
        return Err(CudaVideoCompositorError::UnsupportedFormat(frame.format()));
    }
    if frame.width() == 0 || frame.height() == 0 {
        return Err(CudaVideoCompositorError::InvalidInputDimensions {
            width: frame.width(),
            height: frame.height(),
        });
    }
    // SAFETY: `frame` is a live `frame::Video` already confirmed to be
    // `Pixel::CUDA`, so `as_ptr` yields an initialized `AVFrame` and a hardware
    // frame's `hw_frames_ctx` is either null — rejected here — or an
    // `AVBufferRef` whose `data` is an `AVHWFramesContext`. Only pointer
    // identity is compared, never dereferenced past that.
    unsafe {
        let frames_ref = (*frame.as_ptr()).hw_frames_ctx;
        if frames_ref.is_null() {
            return Err(CudaVideoCompositorError::MissingFramesContext);
        }
        let frames_ctx = (*frames_ref).data as *const ffi::AVHWFramesContext;
        if !std::ptr::eq((*frames_ctx).device_ctx, device_ctx) {
            return Err(CudaVideoCompositorError::ForeignContext);
        }
        let sw_format = ffmpeg::format::Pixel::from((*frames_ctx).sw_format);
        if sw_format != ffmpeg::format::Pixel::NV12 {
            return Err(CudaVideoCompositorError::UnsupportedSurfaceFormat(
                sw_format,
            ));
        }
        Ok(frames_ref)
    }
}

fn validate_output_options(
    options: VideoCompositorOptions,
) -> std::result::Result<(), CudaVideoCompositorError> {
    if options.width < 2
        || options.height < 2
        || !options.width.is_multiple_of(2)
        || !options.height.is_multiple_of(2)
        || options.width > MAX_DIMENSION
        || options.height > MAX_DIMENSION
    {
        return Err(CudaVideoCompositorError::InvalidOutputDimensions {
            width: options.width,
            height: options.height,
        });
    }
    if options.frame_rate.numerator() <= 0 || options.frame_rate.denominator() <= 0 {
        return Err(CudaVideoCompositorError::InvalidFrameRate(
            options.frame_rate,
        ));
    }
    Ok(())
}

/// Thin adapters over the shared, backend-agnostic checks in
/// [`super::super::video_layer`] — translate its [`VideoLayerError`] into
/// this backend's own variants, plus the one restriction only this backend
/// has.
fn validate_layer(layer: VideoLayer) -> std::result::Result<(), CudaVideoCompositorError> {
    video_layer::validate_layer(layer).map_err(layer_error)?;
    validate_opacity(layer.opacity)
}

fn validate_rect(rect: VideoRect) -> std::result::Result<(), CudaVideoCompositorError> {
    video_layer::validate_rect(rect).map_err(layer_error)
}

fn validate_opacity(opacity: f32) -> std::result::Result<(), CudaVideoCompositorError> {
    video_layer::validate_opacity(opacity).map_err(layer_error)
}

fn layer_error(error: VideoLayerError) -> CudaVideoCompositorError {
    match error {
        VideoLayerError::InvalidDimensions { width, height } => {
            CudaVideoCompositorError::InvalidLayerDimensions { width, height }
        }
        VideoLayerError::InvalidOpacity(opacity) => {
            CudaVideoCompositorError::InvalidOpacity(opacity)
        }
        VideoLayerError::InvalidInputDimensions { width, height } => {
            CudaVideoCompositorError::InvalidInputDimensions { width, height }
        }
        VideoLayerError::ScaledLayerTooLarge { width, height } => {
            CudaVideoCompositorError::ScaledLayerTooLarge { width, height }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex as StdMutex;

    use super::*;
    use crate::{
        color::Color,
        elements::{CudaDownload, CudaUpload},
        test_support::try_cuda_device,
    };

    struct CapturingSink {
        pp_log: PpLog,
        received: Arc<StdMutex<Vec<MediaBuffer>>>,
    }

    impl Element for CapturingSink {
        fn name(&self) -> Arc<str> {
            "capture".into()
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

    impl Sink for CapturingSink {
        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            self.received.lock().unwrap().push(buf);
            Ok(())
        }
        fn control(&mut self, _msg: ControlMsg) -> Result<()> {
            Ok(())
        }
    }

    fn capture(element: &mut dyn Source) -> Arc<StdMutex<Vec<MediaBuffer>>> {
        let received = Arc::new(StdMutex::new(Vec::new()));
        element.src_pads()[0].link(Box::new(CapturingSink {
            received: received.clone(),
            pp_log: element_pp_log(ElementType::Other, "capture", None),
        }));
        received
    }

    fn options(width: u32, height: u32) -> VideoCompositorOptions {
        VideoCompositorOptions {
            width,
            height,
            frame_rate: ffmpeg::Rational::new(30, 1),
            background: Color::BLACK,
        }
    }

    /// A CUDA-resident NV12 frame of one flat luma value, which is what makes
    /// a composed output checkable pixel by pixel.
    fn cuda_frame(device: &CudaDevice, width: u32, height: u32, luma: u8) -> Option<MediaBuffer> {
        cuda_frame_with_pts(device, width, height, luma, 0)
    }

    fn cuda_frame_with_pts(
        device: &CudaDevice,
        width: u32,
        height: u32,
        luma: u8,
        pts: i64,
    ) -> Option<MediaBuffer> {
        let Ok(mut upload) =
            CudaUpload::new("upload", device, CudaFrameFormat::Nv12, width, height)
        else {
            eprintln!("skipping: this machine has no usable CUDA frames context");
            return None;
        };
        let uploaded = capture(&mut upload);
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, width, height);
        frame.set_pts(Some(pts));
        let y_stride = frame.stride(0);
        frame.data_mut(0)[..y_stride * height as usize].fill(luma);
        let uv_stride = frame.stride(1);
        frame.data_mut(1)[..uv_stride * (height / 2) as usize].fill(128);
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut slot = pool.get();
        *slot = frame;
        upload
            .consume(MediaBuffer::Video(Arc::new(slot)))
            .expect("upload");
        Some(uploaded.lock().unwrap().remove(0))
    }

    /// Brings one composed CUDA frame back so its pixels can be asserted on.
    fn download(
        device: &CudaDevice,
        frame: UnboundObjectPoolRef<ffmpeg::frame::Video>,
        width: u32,
        height: u32,
    ) -> Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        let mut download =
            CudaDownload::new("download", device, CudaFrameFormat::Nv12, width, height);
        let received = capture(&mut download);
        download
            .consume(MediaBuffer::Video(Arc::new(frame)))
            .expect("download");
        let buf = received.lock().unwrap().remove(0);
        match buf {
            MediaBuffer::Video(frame) => frame,
            other => panic!("expected a Video buffer, got {}", other.kind()),
        }
    }

    fn luma_at(frame: &ffmpeg::frame::Video, x: usize, y: usize) -> u8 {
        frame.data(0)[y * frame.stride(0) + x]
    }

    /// The composition contract: layers land where their rectangles say, the
    /// higher `z_index` wins where they overlap, and everything else is the
    /// background.
    #[test]
    fn composes_layers_in_z_order_at_their_rectangles() {
        let Some((device, _cuda_lock)) = try_cuda_device() else {
            return;
        };
        let (width, height) = (128u32, 128u32);
        let Ok((mut compositor, handle)) =
            CudaVideoCompositor::new("compositor", &device, options(width, height))
        else {
            eprintln!("skipping: this machine cannot open a CUDA compositor");
            return;
        };

        let mut back = handle
            .add_source(
                "back",
                VideoLayer {
                    z_index: 0,
                    fit: VideoFit::Stretch,
                    ..VideoLayer::new(VideoRect::new(0, 0, 64, 64))
                },
            )
            .expect("add back");
        let mut front = handle
            .add_source(
                "front",
                VideoLayer {
                    z_index: 1,
                    fit: VideoFit::Stretch,
                    ..VideoLayer::new(VideoRect::new(32, 32, 64, 64))
                },
            )
            .expect("add front");

        let Some(back_frame) = cuda_frame(&device, 32, 32, 100) else {
            return;
        };
        let Some(front_frame) = cuda_frame(&device, 32, 32, 200) else {
            return;
        };
        back.sink.consume(back_frame).expect("back frame");
        front.sink.consume(front_frame).expect("front frame");

        let composed = compositor.compose_frame().expect("compose");
        assert_eq!(composed.format(), ffmpeg::format::Pixel::CUDA);
        assert_eq!(composed.pts(), Some(0));
        let out = download(&device, composed, width, height);

        assert_eq!(luma_at(&out, 10, 10), 100, "the back layer is missing");
        assert_eq!(luma_at(&out, 80, 80), 200, "the front layer is missing");
        assert_eq!(
            luma_at(&out, 40, 40),
            200,
            "the higher z_index must win where the layers overlap"
        );
        assert_eq!(
            luma_at(&out, 120, 10),
            16,
            "everything outside a layer must be the background"
        );
    }

    /// Runtime control: moving and hiding a layer changes the next frame, and
    /// costs no rebuild of anything — the whole reason this backend places
    /// with a copy rather than a filter.
    #[test]
    fn layer_handle_moves_and_hides_a_live_source() {
        let Some((device, _cuda_lock)) = try_cuda_device() else {
            return;
        };
        let (width, height) = (128u32, 128u32);
        let Ok((mut compositor, handle)) =
            CudaVideoCompositor::new("compositor", &device, options(width, height))
        else {
            eprintln!("skipping: this machine cannot open a CUDA compositor");
            return;
        };
        let mut input = handle
            .add_source(
                "layer",
                VideoLayer {
                    fit: VideoFit::Stretch,
                    ..VideoLayer::new(VideoRect::new(0, 0, 32, 32))
                },
            )
            .expect("add");
        let Some(frame) = cuda_frame(&device, 32, 32, 180) else {
            return;
        };
        input.sink.consume(frame).expect("frame");

        let first = compositor.compose_frame().expect("compose");
        let out = download(&device, first, width, height);
        assert_eq!(luma_at(&out, 10, 10), 180);
        assert_eq!(luma_at(&out, 74, 74), 16);

        input
            .layer
            .set_rect(VideoRect::new(64, 64, 32, 32))
            .expect("move");
        let moved = compositor.compose_frame().expect("compose");
        let out = download(&device, moved, width, height);
        assert_eq!(luma_at(&out, 10, 10), 16, "the layer did not leave");
        assert_eq!(luma_at(&out, 74, 74), 180, "the layer did not arrive");

        input.layer.set_visible(false).expect("hide");
        let hidden = compositor.compose_frame().expect("compose");
        let out = download(&device, hidden, width, height);
        assert_eq!(luma_at(&out, 74, 74), 16, "a hidden layer was still drawn");
    }

    /// `Cover` is the fit no CUDA filter can express — it needs a crop, which
    /// is exactly what a 2D copy does. It must fill its rectangle completely,
    /// where `Contain` leaves background visible in the same rectangle.
    #[test]
    fn cover_fills_its_rectangle_where_contain_letterboxes() {
        let Some((device, _cuda_lock)) = try_cuda_device() else {
            return;
        };
        let (width, height) = (128u32, 128u32);

        // A wide source in a square rectangle: `Contain` letterboxes it
        // vertically, `Cover` crops the sides away instead.
        let composed_with = |fit: VideoFit| {
            let (mut compositor, handle) =
                CudaVideoCompositor::new("compositor", &device, options(width, height))
                    .expect("compositor");
            let mut input = handle
                .add_source(
                    "layer",
                    VideoLayer {
                        fit,
                        ..VideoLayer::new(VideoRect::new(0, 0, 64, 64))
                    },
                )
                .expect("add");
            let frame = cuda_frame(&device, 64, 16, 200).expect("frame");
            input.sink.consume(frame).expect("frame");
            let composed = compositor.compose_frame().expect("compose");
            download(&device, composed, width, height)
        };

        let contain = composed_with(VideoFit::Contain);
        assert_eq!(
            luma_at(&contain, 32, 32),
            200,
            "Contain must draw the image in the middle of its rectangle"
        );
        assert_eq!(
            luma_at(&contain, 32, 4),
            16,
            "Contain must leave background above the image"
        );

        let cover = composed_with(VideoFit::Cover);
        assert_eq!(
            luma_at(&cover, 32, 32),
            200,
            "Cover must draw the image in the middle of its rectangle"
        );
        assert_eq!(
            luma_at(&cover, 32, 4),
            200,
            "Cover must fill its rectangle, cropping the overflow"
        );
        assert_eq!(
            luma_at(&cover, 32, 80),
            16,
            "Cover must not draw outside its rectangle"
        );
    }

    /// A translucent layer is mixed with what is under it, by the kernel the
    /// driver JIT-compiles from this crate's own PTX. The expected value is
    /// the same expression evaluated here, so a wrong blend cannot pass.
    #[test]
    fn a_translucent_layer_is_blended_with_the_background() {
        let Some((device, _cuda_lock)) = try_cuda_device() else {
            return;
        };
        let (width, height) = (128u32, 128u32);
        let Ok((mut compositor, handle)) =
            CudaVideoCompositor::new("compositor", &device, options(width, height))
        else {
            eprintln!("skipping: this machine cannot open a CUDA compositor");
            return;
        };
        let mut input = handle
            .add_source(
                "layer",
                VideoLayer {
                    opacity: 0.5,
                    fit: VideoFit::Stretch,
                    ..VideoLayer::new(VideoRect::new(0, 0, 64, 64))
                },
            )
            .expect("a translucent layer registers");
        let Some(frame) = cuda_frame(&device, 32, 32, 200) else {
            return;
        };
        input.sink.consume(frame).expect("frame");

        let composed = compositor.compose_frame().expect("compose");
        let out = download(&device, composed, width, height);

        // Background is `Color::BLACK`, which is luma 16 in limited range.
        let alpha = (0.5f32 * 255.0).round() as u32;
        let expected = ((200 * alpha + 16 * (255 - alpha) + 127) / 255) as u8;
        assert_eq!(
            luma_at(&out, 10, 10),
            expected,
            "the layer was not blended with the background"
        );
        assert_eq!(
            luma_at(&out, 100, 100),
            16,
            "outside the layer must stay background"
        );

        // The endpoints still behave as before: fully opaque replaces,
        // fully transparent draws nothing.
        input.layer.set_opacity(1.0).expect("opaque");
        let composed = compositor.compose_frame().expect("compose");
        let out = download(&device, composed, width, height);
        assert_eq!(luma_at(&out, 10, 10), 200);

        input.layer.set_opacity(0.0).expect("transparent");
        let composed = compositor.compose_frame().expect("compose");
        let out = download(&device, composed, width, height);
        assert_eq!(luma_at(&out, 10, 10), 16);
    }

    /// A font every machine running these tests has. Skips rather than
    /// fails when it is missing, the same way a hardware test skips without
    /// a device.
    fn try_font() -> Option<Vec<u8>> {
        for path in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        ] {
            if let Ok(data) = std::fs::read(path) {
                return Some(data);
            }
        }
        eprintln!("skipping: no DejaVuSans on this machine to rasterize with");
        None
    }

    /// The text layer's contract: nothing is drawn until `set_text`, then the
    /// glyphs land on the canvas, and clearing the text removes them again.
    #[test]
    fn a_text_layer_draws_after_set_text_and_clears_when_emptied() {
        let Some((device, _cuda_lock)) = try_cuda_device() else {
            return;
        };
        let Some(font) = try_font() else {
            return;
        };
        let (width, height) = (256u32, 128u32);
        let Ok((mut compositor, handle)) =
            CudaVideoCompositor::new("compositor", &device, options(width, height))
        else {
            eprintln!("skipping: this machine cannot open a CUDA compositor");
            return;
        };

        let mut layer = TextLayer::new(font);
        layer.font_size = 48.0;
        layer.color = Color::WHITE;
        layer.x = 8;
        layer.y = 8;
        let text = handle
            .add_text_layer("clock", layer)
            .expect("add a text layer");

        // Nothing rasterized yet: the canvas is pure background.
        let composed = compositor.compose_frame().expect("compose");
        let out = download(&device, composed, width, height);
        let background = luma_at(&out, 10, 10);
        assert_eq!(background, 16, "an empty text layer drew something");

        text.set_text("HELLO").expect("set_text");
        let composed = compositor.compose_frame().expect("compose");
        let out = download(&device, composed, width, height);
        let lit = (0..64u32)
            .flat_map(|y| (0..200u32).map(move |x| (x, y)))
            .filter(|(x, y)| luma_at(&out, *x as usize, *y as usize) > background + 40)
            .count();
        assert!(
            lit > 100,
            "the text did not reach the canvas ({lit} bright pixels)"
        );

        // Text with no drawable glyphs clears the layer rather than erroring.
        text.set_text("   ").expect("blank set_text");
        let composed = compositor.compose_frame().expect("compose");
        let out = download(&device, composed, width, height);
        for y in 0..64usize {
            for x in 0..200usize {
                assert_eq!(
                    luma_at(&out, x, y),
                    background,
                    "clearing the text left something at ({x}, {y})"
                );
            }
        }
    }

    /// Position, visibility, and opacity all act on the next composed frame,
    /// and opacity is a real blend here rather than an on/off switch.
    #[test]
    fn text_position_visibility_and_opacity_take_effect() {
        let Some((device, _cuda_lock)) = try_cuda_device() else {
            return;
        };
        let Some(font) = try_font() else {
            return;
        };
        let (width, height) = (256u32, 128u32);
        let Ok((mut compositor, handle)) =
            CudaVideoCompositor::new("compositor", &device, options(width, height))
        else {
            eprintln!("skipping: this machine cannot open a CUDA compositor");
            return;
        };
        let mut layer = TextLayer::new(font);
        layer.font_size = 48.0;
        layer.color = Color::WHITE;
        let text = handle
            .add_text_layer("clock", layer)
            .expect("add a text layer");
        text.set_text("IIII").expect("set_text");

        let brightest = |frame: &ffmpeg::frame::Video, x0: usize, x1: usize| {
            (0..64usize)
                .flat_map(|y| (x0..x1).map(move |x| (x, y)))
                .map(|(x, y)| luma_at(frame, x, y))
                .max()
                .unwrap_or(0)
        };

        let composed = compositor.compose_frame().expect("compose");
        let out = download(&device, composed, width, height);
        assert!(brightest(&out, 0, 100) > 100, "text is not at the origin");
        assert_eq!(
            brightest(&out, 150, 250),
            16,
            "text is already on the right"
        );

        text.set_position(150, 0);
        let composed = compositor.compose_frame().expect("compose");
        let out = download(&device, composed, width, height);
        assert_eq!(brightest(&out, 0, 100), 16, "the text did not leave");
        assert!(brightest(&out, 150, 250) > 100, "the text did not arrive");

        // Half opacity over a luma-16 background must land near the midpoint
        // rather than at either end.
        text.set_opacity(0.5).expect("half opacity");
        let composed = compositor.compose_frame().expect("compose");
        let out = download(&device, composed, width, height);
        let half = brightest(&out, 150, 250);
        assert!(
            (100..=170).contains(&half),
            "half-opacity text should be mid-grey, got {half}"
        );

        text.set_visible(false);
        let composed = compositor.compose_frame().expect("compose");
        let out = download(&device, composed, width, height);
        assert_eq!(brightest(&out, 150, 250), 16, "a hidden text layer drew");
    }

    /// Input validation happens where the input is named, so a CPU frame or
    /// one from another CUDA context never reaches a device pointer.
    #[test]
    fn a_cpu_frame_and_a_foreign_context_frame_are_typed_errors() {
        let Some((device, _cuda_lock)) = try_cuda_device() else {
            return;
        };
        let Ok((_compositor, handle)) =
            CudaVideoCompositor::new("compositor", &device, options(64, 64))
        else {
            eprintln!("skipping: this machine cannot open a CUDA compositor");
            return;
        };
        let mut input = handle
            .add_source("layer", VideoLayer::new(VideoRect::new(0, 0, 32, 32)))
            .expect("add");

        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut slot = pool.get();
        *slot = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, 32, 32);
        let error = input
            .sink
            .consume(MediaBuffer::Video(Arc::new(slot)))
            .expect_err("a CPU frame must not be composited");
        assert!(
            error.to_string().contains("only composites CUDA frames"),
            "expected UnsupportedFormat, got {error}"
        );

        // Directly, not `try_cuda_device` again: the lock it returns is
        // already held for this test and does not nest.
        let other_device = CudaDevice::new().expect("a second CUDA device");
        let Some(foreign) = cuda_frame(&other_device, 32, 32, 100) else {
            return;
        };
        let error = input
            .sink
            .consume(foreign)
            .expect_err("a frame from a foreign CUDA context must not be composited");
        assert!(
            error.to_string().contains("different CUDA context"),
            "expected ForeignContext, got {error}"
        );
    }

    /// Output timing: this element emits on its own clock, so PTS are
    /// contiguous ticks of its own time base regardless of what the inputs
    /// carried.
    #[test]
    fn output_pts_are_contiguous_ticks_of_its_own_time_base() {
        let Some((device, _cuda_lock)) = try_cuda_device() else {
            return;
        };
        let Ok((mut compositor, handle)) =
            CudaVideoCompositor::new("compositor", &device, options(64, 64))
        else {
            eprintln!("skipping: this machine cannot open a CUDA compositor");
            return;
        };
        assert_eq!(compositor.time_base(), ffmpeg::Rational::new(1, 30));
        let mut input = handle
            .add_source("layer", VideoLayer::new(VideoRect::new(0, 0, 32, 32)))
            .expect("add");
        let Some(mut frame) = cuda_frame(&device, 32, 32, 100) else {
            return;
        };
        if let MediaBuffer::Video(video) = &mut frame {
            // A wildly different input timeline, which must not leak through.
            // `get_mut` rather than a cast through `Arc::as_ptr`: the frame has
            // not been published yet, so this is the one moment it is uniquely
            // owned and can be written at all.
            Arc::get_mut(video)
                .expect("an unpublished frame is uniquely owned")
                .set_pts(Some(9_999));
        }
        input.sink.consume(frame).expect("frame");

        for expected in 0..3 {
            let composed = compositor.compose_frame().expect("compose");
            assert_eq!(composed.pts(), Some(expected));
        }
    }
}

/// One registered text layer's live state, shared between its handle and the
/// compositor's own thread.
struct TextLayerState {
    font: ab_glyph::FontArc,
    font_size: f32,
    color: Color,
    /// Replaced wholesale by `set_text`, which is what makes a text change
    /// atomic from the compositor's point of view: it either draws the whole
    /// previous mask or the whole new one.
    mask: ArcSwapOption<CudaMask>,
    x: AtomicI32,
    y: AtomicI32,
    /// `f32` bits — the compositor only ever reads it, and a torn read is
    /// impossible for a 32-bit atomic.
    opacity: AtomicU32,
    visible: AtomicBool,
}

/// Runtime control for one text layer — the CUDA sibling of
/// `D3d11TextLayerHandle`.
///
/// Cloning is cheap and every clone controls the same layer. Unlike a
/// [`CudaVideoLayerHandle`], this one does real work on `set_text`: it
/// rasterizes the string on the CPU and uploads the resulting coverage mask
/// to the GPU, so it is not something to call per frame if the text has not
/// actually changed.
#[derive(Clone)]
pub struct CudaTextLayerHandle {
    state: Arc<TextLayerState>,
    driver: Arc<CudaDriver>,
}

impl CudaTextLayerHandle {
    /// Rasterizes `text` and uploads it, replacing whatever was drawn
    /// before. Text with no drawable glyphs (empty, whitespace, control
    /// characters) clears the layer.
    pub fn set_text(&self, text: &str) -> std::result::Result<(), CudaVideoCompositorError> {
        let rasterized = rasterize_coverage(&self.state.font, self.state.font_size, text)
            .map_err(text_raster_error)?;
        let Some(mask) = rasterized else {
            self.state.mask.store(None);
            return Ok(());
        };
        // Uploaded before it is published, so the compositor never sees a
        // half-written mask — the same reason `add_source` validates before
        // it replaces a registration.
        let uploaded = self
            .driver
            .upload_mask(&mask.coverage, mask.width, mask.height)?;
        self.state.mask.store(Some(Arc::new(uploaded)));
        Ok(())
    }

    /// Moves the layer's top-left corner. Coordinates are aligned to even
    /// pixels when drawn, for the chroma reason
    /// [`CudaVideoCompositor`] documents.
    pub fn set_position(&self, x: i32, y: i32) {
        self.state.x.store(x, Ordering::Relaxed);
        self.state.y.store(y, Ordering::Relaxed);
    }

    /// Unlike a video layer's, this opacity is free: the text is already
    /// drawn through the blend kernel, which takes the layer's own alpha as
    /// one more factor.
    pub fn set_opacity(&self, opacity: f32) -> std::result::Result<(), CudaVideoCompositorError> {
        validate_opacity(opacity)?;
        self.state
            .opacity
            .store(opacity.to_bits(), Ordering::Relaxed);
        Ok(())
    }

    /// Shows or hides the text without discarding its uploaded mask.
    pub fn set_visible(&self, visible: bool) {
        self.state.visible.store(visible, Ordering::Relaxed);
    }
}

impl CudaVideoCompositorHandle {
    /// Registers a text layer and returns its control handle. Reusing `name`
    /// replaces the previous registration.
    ///
    /// The font is parsed here so bad font data fails at registration rather
    /// than at the first `set_text`. Nothing is drawn until `set_text` is
    /// called — a text layer with no text is not an error, just empty.
    pub fn add_text_layer(
        &self,
        name: impl Into<String>,
        text_layer: TextLayer,
    ) -> std::result::Result<CudaTextLayerHandle, CudaVideoCompositorError> {
        if !text_layer.font_size.is_finite() || text_layer.font_size <= 0.0 {
            return Err(CudaVideoCompositorError::InvalidFontSize(
                text_layer.font_size,
            ));
        }
        let Some(shared) = self.shared.upgrade() else {
            return Err(CudaVideoCompositorError::SourceRemoved);
        };
        let font = ab_glyph::FontArc::try_from_vec(text_layer.font_data)
            .map_err(|error| CudaVideoCompositorError::InvalidFont(error.to_string()))?;
        let state = Arc::new(TextLayerState {
            font,
            font_size: text_layer.font_size,
            color: text_layer.color,
            mask: ArcSwapOption::empty(),
            x: AtomicI32::new(text_layer.x),
            y: AtomicI32::new(text_layer.y),
            opacity: AtomicU32::new(1.0f32.to_bits()),
            visible: AtomicBool::new(true),
        });
        // Reusing a name replaces that registration, the same contract
        // `add_source` has; the replaced handle then controls a layer nothing
        // draws any more.
        let name: Arc<str> = name.into().into();
        let mut layers = shared.text_layers.lock().unwrap();
        match layers.iter_mut().find(|(existing, _)| *existing == name) {
            Some(slot) => slot.1 = state.clone(),
            None => layers.push((name, state.clone())),
        }
        drop(layers);
        Ok(CudaTextLayerHandle {
            state,
            driver: shared.driver.clone(),
        })
    }
}

fn text_raster_error(error: TextRasterError) -> CudaVideoCompositorError {
    match error {
        TextRasterError::TooLarge { width, height } => {
            CudaVideoCompositorError::TextTooLarge { width, height }
        }
        TextRasterError::AllocationFailed { bytes } => {
            CudaVideoCompositorError::AllocationFailed { bytes }
        }
    }
}

/// Where a text mask lands on the canvas, clipped and aligned to the 2x2
/// chroma grid — the text counterpart of [`Placement`], simpler because a
/// mask is never scaled: it is drawn at the size it was rasterized.
struct TextPlacement {
    destination_x: u32,
    destination_y: u32,
    mask_x: u32,
    mask_y: u32,
    width: u32,
    height: u32,
}

impl TextPlacement {
    fn new(
        x: i32,
        y: i32,
        mask_width: u32,
        mask_height: u32,
        canvas_width: u32,
        canvas_height: u32,
    ) -> Option<Self> {
        let (destination_x, mask_x, width) = text_axis(x, mask_width, canvas_width)?;
        let (destination_y, mask_y, height) = text_axis(y, mask_height, canvas_height)?;
        Some(Self {
            destination_x,
            destination_y,
            mask_x,
            mask_y,
            width,
            height,
        })
    }
}

/// One axis of a text mask's clip. Aligning the origin down keeps
/// `mask = destination - origin` even as well, so both are valid chroma
/// coordinates — the same trick [`axis`] uses for video layers.
fn text_axis(origin: i32, extent: u32, canvas_extent: u32) -> Option<(u32, u32, u32)> {
    let origin = align_down_signed(i64::from(origin));
    let start = align_up_nonnegative(origin.max(0));
    let end = align_down_signed((origin + i64::from(extent)).min(i64::from(canvas_extent)));
    if end <= start {
        return None;
    }
    Some((start as u32, (start - origin) as u32, (end - start) as u32))
}
