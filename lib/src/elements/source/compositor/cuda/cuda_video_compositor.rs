use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicU64, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use arc_swap::ArcSwapOption;
use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_debug, pp_error, pp_info};

use super::super::sw_video_compositor::VideoCompositorOptions;
use super::super::video_layer::{
    self, LayerGeometry, MAX_DIMENSION, VideoFit, VideoInputId, VideoLayer, VideoLayerError,
    VideoRect, layer_geometry,
};
use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    control::{ControlMsg, ControlReceiver, drain_control},
    element::{Element, ElementType, Sink, Source, SourceElement, element_pp_log},
    elements::{
        CudaScalerInterp,
        filter::scaler::cuda::scale_graph::CudaScaleGraph,
        filter::upload::cuda_upload::{create_hw_frames_ctx, free_buffer},
    },
    error::Result,
    pad::SrcPad,
    platform::cuda::{
        CudaDevice, CudaFrameFormat,
        driver::{CudaDriver, CudaDriverError, Nv12Region, Nv12Surface},
    },
    pool::{UnboundObjectPool, UnboundObjectPoolRef},
    schedule::PeriodicSchedule,
};

const OUTPUT_POOL_SIZE: usize = 4;
const CONTROL_POLL_INTERVAL: Duration = Duration::from_millis(5);

/// Errors specific to [`CudaVideoCompositor`].
#[derive(Debug, ThisError)]
pub enum CudaVideoCompositorError {
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),

    #[error(transparent)]
    Driver(#[from] CudaDriverError),

    #[error("failed to scale a layer: {0}")]
    Scale(#[from] crate::elements::CudaScalerError),

    #[error(
        "invalid output dimensions {width}x{height}; each dimension must be even and 2..={MAX_DIMENSION}"
    )]
    InvalidOutputDimensions { width: u32, height: u32 },

    #[error("invalid frame rate {0}; numerator and denominator must both be positive")]
    InvalidFrameRate(ffmpeg::Rational),

    #[error(
        "invalid layer dimensions {width}x{height}; each dimension must be 1..={MAX_DIMENSION}"
    )]
    InvalidLayerDimensions { width: u32, height: u32 },

    #[error("layer opacity must be finite and between 0.0 and 1.0, got {0}")]
    InvalidOpacity(f32),

    /// Partial transparency needs a blend, and a blend needs a kernel this
    /// crate has no way to compile — see this element's own docs.
    #[error(
        "CudaVideoCompositor cannot blend: opacity must be 0.0 or 1.0, got {0}. \
         Composite on the CPU with SwVideoCompositor if a layer has to be translucent"
    )]
    OpacityUnsupported(f32),

    #[error("input frame has invalid dimensions {width}x{height}")]
    InvalidInputDimensions { width: u32, height: u32 },

    #[error("scaled layer would exceed {MAX_DIMENSION}px: {width}x{height}")]
    ScaledLayerTooLarge { width: u32, height: u32 },

    #[error("the compositor input has been removed")]
    SourceRemoved,

    #[error("CudaVideoCompositorInputSink only composites CUDA frames, got {0:?}")]
    UnsupportedFormat(ffmpeg::format::Pixel),

    #[error("CudaVideoCompositor only composites NV12 surfaces, got {0:?}")]
    UnsupportedSurfaceFormat(ffmpeg::format::Pixel),

    #[error("CUDA frame has no hardware frames context")]
    MissingFramesContext,

    #[error("CUDA frame belongs to a different CUDA context than this compositor")]
    ForeignContext,

    #[error("CUDA frame carries no device pointers")]
    MissingPlane,

    #[error(
        "CudaVideoCompositorInputSink only accepts decoded Video frames, got a {0}; link it after a decoder or upload"
    )]
    UnsupportedBuffer(&'static str),

    #[error("failed to allocate the CUDA frames context")]
    HwFramesAlloc,

    #[error("failed to take an output frame from the CUDA pool (code {0})")]
    HwFrameGet(i32),

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
    next_input_id: AtomicU64,
    /// Captured from the compositor's own [`CudaDevice`] so every input sink
    /// can reject a frame from another CUDA context before it ever reaches a
    /// device pointer. Only ever compared.
    device_ctx: *const ffi::AVHWDeviceContext,
}

// SAFETY: `device_ctx` is only ever compared, never dereferenced, and the
// compositor holds its own reference to the context for its whole life, so
// the pointer cannot go stale while any input sink is alive.
unsafe impl Send for CompositorShared {}
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
    pub sink: Box<dyn Sink>,
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

    pub fn source_count(&self) -> usize {
        self.shared
            .upgrade()
            .map(|shared| shared.inputs.lock().unwrap().len())
            .unwrap_or(0)
    }
}

/// Runtime placement control for one registered input — the CUDA sibling of
/// [`crate::elements::SwVideoLayerHandle`], with the one difference this
/// backend cannot hide: `set_opacity` accepts only 0.0 and 1.0.
#[derive(Clone)]
pub struct CudaVideoLayerHandle {
    id: VideoInputId,
    name: Arc<str>,
    input: Weak<VideoInput>,
}

impl CudaVideoLayerHandle {
    pub fn id(&self) -> VideoInputId {
        self.id
    }

    pub fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    pub fn layer(&self) -> Option<VideoLayer> {
        self.input
            .upgrade()
            .map(|input| *input.layer.lock().unwrap())
    }

    pub fn set_layer(
        &self,
        layer: VideoLayer,
    ) -> std::result::Result<(), CudaVideoCompositorError> {
        validate_layer(layer)?;
        self.update(|current| *current = layer)
    }

    pub fn set_rect(&self, rect: VideoRect) -> std::result::Result<(), CudaVideoCompositorError> {
        validate_rect(rect)?;
        self.update(|layer| layer.rect = rect)
    }

    /// Only 0.0 and 1.0 are accepted; anything between them returns
    /// [`CudaVideoCompositorError::OpacityUnsupported`] rather than silently
    /// rounding to an opaque layer.
    pub fn set_opacity(&self, opacity: f32) -> std::result::Result<(), CudaVideoCompositorError> {
        validate_opacity(opacity)?;
        self.update(|layer| layer.opacity = opacity)
    }

    pub fn set_z_index(&self, z_index: i32) -> std::result::Result<(), CudaVideoCompositorError> {
        self.update(|layer| layer.z_index = z_index)
    }

    pub fn set_visible(&self, visible: bool) -> std::result::Result<(), CudaVideoCompositorError> {
        self.update(|layer| layer.visible = visible)
    }

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
/// [`crate::elements::D3d11VideoCompositor`], driving the same
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
/// # No blending
///
/// A layer is opaque or hidden; `opacity` between 0.0 and 1.0 is rejected by
/// [`CudaVideoLayerHandle::set_opacity`]. Blending means reading, weighting,
/// and writing every pixel, which needs a CUDA kernel — and compiling one
/// would require the CUDA toolkit at build time, which this crate
/// deliberately does not depend on (the driver alone is enough for
/// everything else it does). Use [`crate::elements::SwVideoCompositor`] when
/// a layer has to be translucent.
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
    driver: CudaDriver,
    /// This element's own reference to the shared context, released in
    /// `Drop` — it is what keeps `shared.device_ctx` a valid identity.
    hw_device_ctx: *mut ffi::AVBufferRef,
    /// The pool output surfaces are allocated from.
    hw_frames_ctx: *mut ffi::AVBufferRef,
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

        let driver = CudaDriver::retain_primary()?;
        let hw_device_ctx = unsafe { ffi::av_buffer_ref(device.as_ptr()) };
        let hw_frames_ctx = match unsafe {
            create_hw_frames_ctx(
                hw_device_ctx,
                CudaFrameFormat::Nv12,
                options.width,
                options.height,
            )
        } {
            Ok(ctx) => ctx,
            Err(error) => {
                unsafe { free_buffer(hw_device_ctx) };
                pp_error!(pp_log: &pp_log, "failed to build the output pool: {error}");
                return Err(CudaVideoCompositorError::HwFramesAlloc);
            }
        };
        let device_ctx = unsafe { (*hw_device_ctx).data as *const ffi::AVHWDeviceContext };

        let shared = Arc::new(CompositorShared {
            inputs: Mutex::new(HashMap::new()),
            next_input_id: AtomicU64::new(1),
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
                hw_device_ctx,
                hw_frames_ctx,
                scalers: HashMap::new(),
                output_pool: UnboundObjectPool::new(
                    OUTPUT_POOL_SIZE,
                    ffmpeg::frame::Video::empty,
                    |_| {},
                ),
                pad: SrcPad::new(format!("{name}_src")),
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

    pub fn width(&self) -> u32 {
        self.options.width
    }

    pub fn height(&self) -> u32 {
        self.options.height
    }

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
        unsafe {
            let ptr = output.as_mut_ptr();
            // Releasing the previous surface here is what returns it to the
            // frames pool rather than holding it for this element's life.
            ffi::av_frame_unref(ptr);
            let code = ffi::av_hwframe_get_buffer(self.hw_frames_ctx, ptr, 0);
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
            self.driver
                .blit_nv12(layer_surface, canvas, placement.region)?;
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
        unsafe {
            free_buffer(self.hw_frames_ctx);
            free_buffer(self.hw_device_ctx);
        }
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
    video_layer::validate_opacity(opacity).map_err(layer_error)?;
    if opacity != 0.0 && opacity != 1.0 {
        return Err(CudaVideoCompositorError::OpacityUnsupported(opacity));
    }
    Ok(())
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

    /// The one capability this backend cannot offer. It has to be a typed
    /// error rather than a silently opaque layer.
    #[test]
    fn translucent_layers_are_refused_with_a_typed_error() {
        let Some((device, _cuda_lock)) = try_cuda_device() else {
            return;
        };
        let Ok((_compositor, handle)) =
            CudaVideoCompositor::new("compositor", &device, options(64, 64))
        else {
            eprintln!("skipping: this machine cannot open a CUDA compositor");
            return;
        };
        let error = handle
            .add_source(
                "layer",
                VideoLayer {
                    opacity: 0.5,
                    ..VideoLayer::new(VideoRect::new(0, 0, 32, 32))
                },
            )
            .err()
            .expect("a translucent layer must not register");
        assert!(
            error.to_string().contains("cannot blend"),
            "expected OpacityUnsupported, got {error}"
        );

        let input = handle
            .add_source("layer", VideoLayer::new(VideoRect::new(0, 0, 32, 32)))
            .expect("an opaque layer registers");
        let error = input
            .layer
            .set_opacity(0.25)
            .expect_err("a translucent update must be refused");
        assert!(
            error.to_string().contains("cannot blend"),
            "expected OpacityUnsupported, got {error}"
        );
        // The two values a copy *can* express stay available.
        input.layer.set_opacity(0.0).expect("fully transparent");
        input.layer.set_opacity(1.0).expect("fully opaque");
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
            unsafe {
                (*(Arc::as_ptr(video) as *mut UnboundObjectPoolRef<ffmpeg::frame::Video>))
                    .set_pts(Some(9_999))
            };
        }
        input.sink.consume(frame).expect("frame");

        for expected in 0..3 {
            let composed = compositor.compose_frame().expect("compose");
            assert_eq!(composed.pts(), Some(expected));
        }
    }
}
