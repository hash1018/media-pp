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
use ffmpeg_next as ffmpeg;
use rust_hlog::{HLog, hinfo};
use thiserror::Error as ThisError;

use super::video_layer::{
    self, LayerGeometry, MAX_DIMENSION, VideoFit, VideoInputId, VideoLayer, VideoLayerError,
    VideoRect,
};
use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    color::Color,
    control::{ControlMsg, ControlReceiver, drain_control},
    element::{Element, ElementType, Sink, Source, SourceElement, element_hlog},
    error::Result,
    pad::SrcPad,
    pool::{UnboundObjectPool, UnboundObjectPoolRef},
};

const OUTPUT_POOL_SIZE: usize = 4;
const CONTROL_POLL_INTERVAL: Duration = Duration::from_millis(5);

/// The compositor's fixed output definition. Every emitted frame is an
/// opaque [`ffmpeg::format::Pixel::BGRA`] frame at `width` x `height`.
#[derive(Debug, Clone, Copy)]
pub struct VideoCompositorOptions {
    pub width: u32,
    pub height: u32,
    pub frame_rate: ffmpeg::Rational,
    pub background: Color,
}

impl Default for VideoCompositorOptions {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            frame_rate: ffmpeg::Rational::new(30, 1),
            background: Color::BLACK,
        }
    }
}

/// Errors specific to [`VideoCompositor`].
#[derive(Debug, ThisError)]
pub enum VideoCompositorError {
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),

    #[error(
        "invalid output dimensions {width}x{height}; each dimension must be 1..={MAX_DIMENSION}"
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

    #[error("input frame has invalid dimensions {width}x{height}")]
    InvalidInputDimensions { width: u32, height: u32 },

    #[error("scaled layer would exceed {MAX_DIMENSION}px: {width}x{height}")]
    ScaledLayerTooLarge { width: u32, height: u32 },

    #[error("the compositor input has been removed")]
    SourceRemoved,

    #[error(
        "VideoCompositorInputSink only accepts decoded Video frames, got a {0}; link it after a decoder or video source"
    )]
    UnsupportedBuffer(&'static str),

    #[error("VideoCompositor doesn't support seeking a live composition")]
    SeekUnsupported,
}

struct VideoInput {
    id: VideoInputId,
    /// The hot producer/consumer path is an atomic latest-value slot:
    /// input pipelines replace the pointer without taking the layer lock,
    /// and the compositor acquires a stable Arc snapshot independently.
    latest_frame: ArcSwapOption<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    /// Layer changes are infrequent and update several related fields as
    /// one coherent value, so a small dedicated lock remains appropriate.
    layer: Mutex<VideoLayer>,
}

struct CompositorShared {
    inputs: Mutex<HashMap<Arc<str>, Arc<VideoInput>>>,
    next_input_id: AtomicU64,
}

/// A cheaply cloneable handle for adding and removing compositor inputs.
/// It mirrors [`crate::elements::MixerHandle`], but each registration also
/// returns a [`VideoLayerHandle`] for changing that input's placement.
#[derive(Clone)]
pub struct VideoCompositorHandle {
    shared: Weak<CompositorShared>,
}

/// The two endpoints created for one compositor input registration.
/// Move `sink` into the upstream pipeline and retain `layer` in application
/// code for runtime placement changes.
pub struct VideoCompositorInput {
    pub sink: Box<dyn Sink>,
    pub layer: VideoLayerHandle,
}

impl VideoCompositorHandle {
    /// Registers an input and returns its terminal Sink plus independent
    /// runtime layer control. Reusing `name` replaces the old registration;
    /// old sinks and layer handles become harmlessly stale.
    pub fn add_source(
        &self,
        name: impl Into<String>,
        layer: VideoLayer,
    ) -> std::result::Result<Option<VideoCompositorInput>, VideoCompositorError> {
        validate_layer(layer)?;
        let Some(shared) = self.shared.upgrade() else {
            return Ok(None);
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

        Ok(Some(VideoCompositorInput {
            sink: Box::new(VideoCompositorInputSink {
                name: name.clone(),
                hlog: element_hlog(ElementType::VideoCompositor, &name, None),
                shared: self.shared.clone(),
                input: Arc::downgrade(&input),
            }),
            layer: VideoLayerHandle {
                id,
                name,
                input: Arc::downgrade(&input),
            },
        }))
    }

    /// Removes `name` immediately. Existing input sinks and layer handles
    /// become disconnected and cannot affect a later same-name source.
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

/// Thread-safe runtime placement control for one compositor input.
/// Retaining it does not keep the input or compositor alive.
#[derive(Clone)]
pub struct VideoLayerHandle {
    id: VideoInputId,
    name: Arc<str>,
    input: Weak<VideoInput>,
}

impl VideoLayerHandle {
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

    pub fn set_layer(&self, layer: VideoLayer) -> std::result::Result<(), VideoCompositorError> {
        validate_layer(layer)?;
        self.update(|current| *current = layer)
    }

    pub fn set_rect(&self, rect: VideoRect) -> std::result::Result<(), VideoCompositorError> {
        validate_rect(rect)?;
        self.update(|layer| layer.rect = rect)
    }

    pub fn set_opacity(&self, opacity: f32) -> std::result::Result<(), VideoCompositorError> {
        validate_opacity(opacity)?;
        self.update(|layer| layer.opacity = opacity)
    }

    pub fn set_z_index(&self, z_index: i32) -> std::result::Result<(), VideoCompositorError> {
        self.update(|layer| layer.z_index = z_index)
    }

    pub fn set_visible(&self, visible: bool) -> std::result::Result<(), VideoCompositorError> {
        self.update(|layer| layer.visible = visible)
    }

    pub fn set_fit(&self, fit: VideoFit) -> std::result::Result<(), VideoCompositorError> {
        self.update(|layer| layer.fit = fit)
    }

    fn update(
        &self,
        update: impl FnOnce(&mut VideoLayer),
    ) -> std::result::Result<(), VideoCompositorError> {
        let input = self
            .input
            .upgrade()
            .ok_or(VideoCompositorError::SourceRemoved)?;
        update(&mut input.layer.lock().unwrap());
        Ok(())
    }
}

/// One terminal video input returned by
/// [`VideoCompositorHandle::add_source`]. It stores only the latest frame,
/// so a fast producer cannot build an unbounded queue behind a slower
/// compositor output rate.
#[rust_hlog::hlog]
pub struct VideoCompositorInputSink {
    name: Arc<str>,
    shared: Weak<CompositorShared>,
    input: Weak<VideoInput>,
}

impl VideoCompositorInputSink {
    fn detach(&self) {
        let (Some(shared), Some(input)) = (self.shared.upgrade(), self.input.upgrade()) else {
            return;
        };
        let mut inputs = shared.inputs.lock().unwrap();
        let is_current = inputs
            .get(&self.name)
            .is_some_and(|current| Arc::ptr_eq(current, &input));
        if is_current {
            inputs.remove(&self.name);
        }
    }
}

impl Element for VideoCompositorInputSink {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::VideoCompositor
    }

    fn hlog(&self) -> &HLog {
        &self.hlog
    }

    fn hlog_mut(&mut self) -> &mut HLog {
        &mut self.hlog
    }
}

impl Sink for VideoCompositorInputSink {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        let Some(input) = self.input.upgrade() else {
            return Ok(());
        };
        match buf {
            MediaBuffer::Video(frame) => {
                input.latest_frame.store(Some(frame));
                Ok(())
            }
            MediaBuffer::Eos => {
                self.detach();
                Ok(())
            }
            MediaBuffer::Packet(_) => Err(VideoCompositorError::UnsupportedBuffer("Packet").into()),
            MediaBuffer::Audio(_) => Err(VideoCompositorError::UnsupportedBuffer("Audio").into()),
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        match msg {
            ControlMsg::Stop => self.detach(),
            ControlMsg::Seek(_) => {
                if let Some(input) = self.input.upgrade() {
                    input.latest_frame.store(None);
                }
            }
            ControlMsg::Pause | ControlMsg::Resume => {}
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ScaleDefinition {
    source_format: ffmpeg::format::Pixel,
    source_width: u32,
    source_height: u32,
    target_width: u32,
    target_height: u32,
}

struct InputScaler {
    definition: Option<ScaleDefinition>,
    context: Option<ffmpeg::software::scaling::Context>,
    output: ffmpeg::frame::Video,
}

impl InputScaler {
    fn new() -> Self {
        Self {
            definition: None,
            context: None,
            output: ffmpeg::frame::Video::empty(),
        }
    }

    fn scale(
        &mut self,
        source: &ffmpeg::frame::Video,
        width: u32,
        height: u32,
    ) -> std::result::Result<&ffmpeg::frame::Video, ffmpeg::Error> {
        let definition = ScaleDefinition {
            source_format: source.format(),
            source_width: source.width(),
            source_height: source.height(),
            target_width: width,
            target_height: height,
        };
        if self.definition != Some(definition) {
            self.context = Some(ffmpeg::software::scaling::Context::get(
                definition.source_format,
                definition.source_width,
                definition.source_height,
                ffmpeg::format::Pixel::BGRA,
                width,
                height,
                ffmpeg::software::scaling::Flags::BILINEAR,
            )?);
            self.output = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, width, height);
            self.definition = Some(definition);
        }
        self.context
            .as_mut()
            .expect("created with a changed definition or retained from a matching one")
            .run(source, &mut self.output)?;
        Ok(&self.output)
    }
}

/// Composites the latest frames from any number of independent input
/// pipelines into one fixed-rate opaque BGRA video stream.
///
/// Like [`crate::elements::AudioMixer`], this is a [`SourceElement`], not
/// a conventional one-input filter: upstream pipelines terminate at the
/// sinks returned by [`VideoCompositorHandle::add_source`], while this
/// element's own pipeline drives output on its independent clock. Input
/// frame PTS values therefore do not become output PTS; output advances by
/// one tick in [`VideoCompositor::time_base`] for every composed frame.
#[rust_hlog::hlog]
pub struct VideoCompositor {
    name: Arc<str>,
    shared: Arc<CompositorShared>,
    options: VideoCompositorOptions,
    frame_interval: Duration,
    frame_index: i64,
    scalers: HashMap<VideoInputId, InputScaler>,
    output_pool: UnboundObjectPool<ffmpeg::frame::Video>,
    pad: SrcPad,
}

// SAFETY: `SwsContext` has no thread affinity and every scaling context is
// exclusively accessed through `&mut self` on the compositor's one source
// thread. ffmpeg-next simply omits Send for this wrapper, as with Scaler.
unsafe impl Send for VideoCompositor {}

impl VideoCompositor {
    pub fn new(
        name: impl Into<String>,
        options: VideoCompositorOptions,
    ) -> std::result::Result<(Self, VideoCompositorHandle), VideoCompositorError> {
        validate_output_options(options)?;
        let name: Arc<str> = name.into().into();
        let hlog = element_hlog(ElementType::VideoCompositor, &name, None);
        let shared = Arc::new(CompositorShared {
            inputs: Mutex::new(HashMap::new()),
            next_input_id: AtomicU64::new(1),
        });
        let frame_interval = Duration::from_secs_f64(
            options.frame_rate.denominator() as f64 / options.frame_rate.numerator() as f64,
        );
        let (width, height) = (options.width, options.height);
        let output_pool = UnboundObjectPool::new(
            OUTPUT_POOL_SIZE,
            move || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, width, height),
            |_| {},
        );
        hinfo!(
            hlog: &hlog,
            "created: {}x{}, frame_rate={}, format=BGRA",
            width,
            height,
            options.frame_rate
        );
        Ok((
            Self {
                name: name.clone(),
                hlog,
                shared: shared.clone(),
                options,
                frame_interval,
                frame_index: 0,
                scalers: HashMap::new(),
                output_pool,
                pad: SrcPad::new(format!("{name}_src")),
            },
            VideoCompositorHandle {
                shared: Arc::downgrade(&shared),
            },
        ))
    }

    pub fn format(&self) -> ffmpeg::format::Pixel {
        ffmpeg::format::Pixel::BGRA
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

    pub fn time_base(&self) -> ffmpeg::Rational {
        ffmpeg::Rational::new(
            self.options.frame_rate.denominator(),
            self.options.frame_rate.numerator(),
        )
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

    fn compose_frame(
        &mut self,
    ) -> std::result::Result<UnboundObjectPoolRef<ffmpeg::frame::Video>, VideoCompositorError> {
        let mut snapshots = self.snapshots();
        let active: HashSet<_> = snapshots.iter().map(|snapshot| snapshot.id).collect();
        self.scalers.retain(|id, _| active.contains(id));
        snapshots.sort_by(|left, right| {
            left.layer
                .z_index
                .cmp(&right.layer.z_index)
                .then_with(|| left.id.cmp(&right.id))
        });

        let mut output = self.output_pool.get();
        fill_background(&mut output, self.options.background);
        for snapshot in snapshots {
            if !snapshot.layer.visible || snapshot.layer.opacity == 0.0 {
                continue;
            }
            let Some(frame) = snapshot.frame else {
                continue;
            };
            let geometry = layer_geometry(
                frame.width(),
                frame.height(),
                snapshot.layer.rect,
                snapshot.layer.fit,
            )?;
            let scaled = self
                .scalers
                .entry(snapshot.id)
                .or_insert_with(InputScaler::new)
                .scale(&frame, geometry.image_width, geometry.image_height)
                .map_err(VideoCompositorError::from)?;
            blend_bgra(&mut output, scaled, geometry, snapshot.layer.opacity);
        }
        output.set_pts(Some(self.frame_index));
        self.frame_index += 1;
        Ok(output)
    }

    fn push_frame(&mut self, bus: &Bus) -> std::result::Result<(), VideoCompositorError> {
        let output = self.compose_frame()?;
        if let Err(error) = self.pad.push(MediaBuffer::Video(Arc::new(output))) {
            bus.post(
                &self.hlog,
                BusEvent::Error {
                    element_type: ElementType::VideoCompositor,
                    name: self.name.clone(),
                    error,
                },
            );
        }
        Ok(())
    }
}

impl Element for VideoCompositor {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::VideoCompositor
    }

    fn hlog(&self) -> &HLog {
        &self.hlog
    }

    fn hlog_mut(&mut self) -> &mut HLog {
        &mut self.hlog
    }
}

impl Source for VideoCompositor {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl SourceElement for VideoCompositor {
    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        hinfo!(self, "run: starting");
        let mut next_due = Instant::now();
        loop {
            let outcome = drain_control(control, self, bus)?;
            if outcome.stopped {
                hinfo!(self, "run: stopped");
                return Ok(());
            }
            if outcome.paused_for > Duration::ZERO {
                // Shift the deadline forward by exactly how long Pause
                // held it, so Resume picks up from the same phase it had
                // before freezing instead of resyncing to a fresh cadence
                // anchored at whenever this loop happens to run again —
                // same correction as `TestVideoSource`/`DxgiCaptureSource`.
                next_due += outcome.paused_for;
                let now = Instant::now();
                if next_due < now {
                    next_due = now + self.frame_interval;
                }
            }

            let now = Instant::now();
            if now < next_due {
                thread::sleep((next_due - now).min(CONTROL_POLL_INTERVAL));
                continue;
            }

            self.push_frame(bus)?;
            next_due += self.frame_interval;
            let now = Instant::now();
            if next_due < now {
                // A slow composition must not cause a burst of catch-up
                // frames. Drop missed ticks and resume cadence from the
                // next real output deadline.
                next_due = now + self.frame_interval;
            }
        }
    }

    fn seek(&mut self, _target: Duration) -> Result<Duration> {
        Err(VideoCompositorError::SeekUnsupported.into())
    }
}

fn validate_output_options(
    options: VideoCompositorOptions,
) -> std::result::Result<(), VideoCompositorError> {
    if options.width == 0
        || options.height == 0
        || options.width > MAX_DIMENSION
        || options.height > MAX_DIMENSION
    {
        return Err(VideoCompositorError::InvalidOutputDimensions {
            width: options.width,
            height: options.height,
        });
    }
    if options.frame_rate.numerator() <= 0 || options.frame_rate.denominator() <= 0 {
        return Err(VideoCompositorError::InvalidFrameRate(options.frame_rate));
    }
    Ok(())
}

/// Thin adapters over the shared, backend-agnostic logic in
/// [`super::video_layer`] — translate its [`VideoLayerError`] into this
/// backend's own [`VideoCompositorError`] variants so every existing call
/// site/error consumer here keeps seeing the same error shape it always
/// has.
fn map_layer_error(error: VideoLayerError) -> VideoCompositorError {
    match error {
        VideoLayerError::InvalidDimensions { width, height } => {
            VideoCompositorError::InvalidLayerDimensions { width, height }
        }
        VideoLayerError::InvalidOpacity(opacity) => VideoCompositorError::InvalidOpacity(opacity),
        VideoLayerError::InvalidInputDimensions { width, height } => {
            VideoCompositorError::InvalidInputDimensions { width, height }
        }
        VideoLayerError::ScaledLayerTooLarge { width, height } => {
            VideoCompositorError::ScaledLayerTooLarge { width, height }
        }
    }
}

fn validate_layer(layer: VideoLayer) -> std::result::Result<(), VideoCompositorError> {
    video_layer::validate_layer(layer).map_err(map_layer_error)
}

fn validate_rect(rect: VideoRect) -> std::result::Result<(), VideoCompositorError> {
    video_layer::validate_rect(rect).map_err(map_layer_error)
}

fn validate_opacity(opacity: f32) -> std::result::Result<(), VideoCompositorError> {
    video_layer::validate_opacity(opacity).map_err(map_layer_error)
}

fn layer_geometry(
    source_width: u32,
    source_height: u32,
    rect: VideoRect,
    fit: VideoFit,
) -> std::result::Result<LayerGeometry, VideoCompositorError> {
    video_layer::layer_geometry(source_width, source_height, rect, fit).map_err(map_layer_error)
}

fn fill_background(frame: &mut ffmpeg::frame::Video, color: Color) {
    let width = frame.width() as usize;
    let height = frame.height() as usize;
    let stride = frame.stride(0);
    let data = frame.data_mut(0);
    for row in 0..height {
        for pixel in data[row * stride..row * stride + width * 4].chunks_exact_mut(4) {
            pixel.copy_from_slice(&[color.blue, color.green, color.red, 255]);
        }
    }
}

fn blend_bgra(
    destination: &mut ffmpeg::frame::Video,
    source: &ffmpeg::frame::Video,
    geometry: LayerGeometry,
    opacity: f32,
) {
    let output_width = i64::from(destination.width());
    let output_height = i64::from(destination.height());
    let clip_left = i64::from(geometry.clip.x).max(0);
    let clip_top = i64::from(geometry.clip.y).max(0);
    let clip_right =
        (i64::from(geometry.clip.x) + i64::from(geometry.clip.width)).min(output_width);
    let clip_bottom =
        (i64::from(geometry.clip.y) + i64::from(geometry.clip.height)).min(output_height);
    let left = geometry.image_x.max(clip_left);
    let top = geometry.image_y.max(clip_top);
    let right = (geometry.image_x + i64::from(geometry.image_width)).min(clip_right);
    let bottom = (geometry.image_y + i64::from(geometry.image_height)).min(clip_bottom);
    if left >= right || top >= bottom {
        return;
    }

    let source_stride = source.stride(0);
    let destination_stride = destination.stride(0);
    let source_data = source.data(0);
    let destination_data = destination.data_mut(0);
    for output_y in top..bottom {
        let source_y = (output_y - geometry.image_y) as usize;
        let destination_y = output_y as usize;
        for output_x in left..right {
            let source_x = (output_x - geometry.image_x) as usize;
            let destination_x = output_x as usize;
            let source_offset = source_y * source_stride + source_x * 4;
            let destination_offset = destination_y * destination_stride + destination_x * 4;
            let source_pixel = &source_data[source_offset..source_offset + 4];
            let destination_pixel =
                &mut destination_data[destination_offset..destination_offset + 4];
            let alpha = (f32::from(source_pixel[3]) / 255.0) * opacity;
            let inverse = 1.0 - alpha;
            for channel in 0..3 {
                destination_pixel[channel] = (f32::from(source_pixel[channel]) * alpha
                    + f32::from(destination_pixel[channel]) * inverse)
                    .round()
                    .clamp(0.0, 255.0) as u8;
            }
            destination_pixel[3] = 255;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            Mutex as StdMutex,
            atomic::{AtomicBool, Ordering as AtomicOrdering},
        },
        thread,
    };

    use super::*;

    #[rust_hlog::hlog]
    struct CapturingSink {
        received: Arc<StdMutex<Vec<MediaBuffer>>>,
    }

    impl Element for CapturingSink {
        fn name(&self) -> Arc<str> {
            "capture".into()
        }

        fn element_type(&self) -> ElementType {
            ElementType::Other
        }

        fn hlog(&self) -> &HLog {
            &self.hlog
        }

        fn hlog_mut(&mut self) -> &mut HLog {
            &mut self.hlog
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

    fn options(width: u32, height: u32) -> VideoCompositorOptions {
        VideoCompositorOptions {
            width,
            height,
            frame_rate: ffmpeg::Rational::new(30, 1),
            background: Color::BLACK,
        }
    }

    fn solid_frame(
        width: u32,
        height: u32,
        color: Color,
    ) -> Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        let pool = UnboundObjectPool::new(
            0,
            move || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, width, height),
            |_| {},
        );
        let mut frame = pool.get();
        fill_background(&mut frame, color);
        Arc::new(frame)
    }

    fn pixel(frame: &ffmpeg::frame::Video, x: usize, y: usize) -> [u8; 4] {
        let offset = y * frame.stride(0) + x * 4;
        frame.data(0)[offset..offset + 4].try_into().unwrap()
    }

    fn input(
        handle: &VideoCompositorHandle,
        name: &str,
        layer: VideoLayer,
    ) -> (Box<dyn Sink>, VideoLayerHandle) {
        let input = handle.add_source(name, layer).unwrap().unwrap();
        (input.sink, input.layer)
    }

    #[test]
    fn composes_inputs_in_z_order_and_preserves_output_contract() {
        let (mut compositor, handle) = VideoCompositor::new("compositor", options(4, 4)).unwrap();
        let mut background = VideoLayer::new(VideoRect::new(0, 0, 4, 4));
        background.fit = VideoFit::Stretch;
        let (mut red_sink, _) = input(&handle, "red", background);
        let mut overlay = VideoLayer::new(VideoRect::new(1, 1, 2, 2));
        overlay.z_index = 1;
        overlay.fit = VideoFit::Stretch;
        let (mut blue_sink, _) = input(&handle, "blue", overlay);
        red_sink
            .consume(MediaBuffer::Video(solid_frame(4, 4, Color::new(255, 0, 0))))
            .unwrap();
        blue_sink
            .consume(MediaBuffer::Video(solid_frame(2, 2, Color::new(0, 0, 255))))
            .unwrap();

        let frame = compositor.compose_frame().unwrap();
        assert_eq!(frame.format(), ffmpeg::format::Pixel::BGRA);
        assert_eq!((frame.width(), frame.height()), (4, 4));
        assert_eq!(frame.pts(), Some(0));
        assert_eq!(pixel(&frame, 0, 0), [0, 0, 255, 255]);
        assert_eq!(pixel(&frame, 1, 1), [255, 0, 0, 255]);
    }

    #[test]
    fn layer_handle_moves_blends_and_hides_a_live_source() {
        let (mut compositor, handle) = VideoCompositor::new("compositor", options(3, 1)).unwrap();
        let layer = VideoLayer::new(VideoRect::new(0, 0, 1, 1));
        let (mut sink, layer_handle) = input(&handle, "white", layer);
        sink.consume(MediaBuffer::Video(solid_frame(1, 1, Color::WHITE)))
            .unwrap();

        layer_handle.set_rect(VideoRect::new(1, 0, 1, 1)).unwrap();
        layer_handle.set_opacity(0.5).unwrap();
        let blended = compositor.compose_frame().unwrap();
        assert_eq!(pixel(&blended, 0, 0), [0, 0, 0, 255]);
        assert_eq!(pixel(&blended, 1, 0), [128, 128, 128, 255]);

        layer_handle.set_visible(false).unwrap();
        let hidden = compositor.compose_frame().unwrap();
        assert_eq!(pixel(&hidden, 1, 0), [0, 0, 0, 255]);
        assert_eq!(hidden.pts(), Some(1));
    }

    #[test]
    fn input_keeps_only_the_latest_frame() {
        let (mut compositor, handle) = VideoCompositor::new("compositor", options(1, 1)).unwrap();
        let (mut sink, _) = input(
            &handle,
            "latest",
            VideoLayer::new(VideoRect::new(0, 0, 1, 1)),
        );
        sink.consume(MediaBuffer::Video(solid_frame(1, 1, Color::new(255, 0, 0))))
            .unwrap();
        sink.consume(MediaBuffer::Video(solid_frame(1, 1, Color::new(0, 255, 0))))
            .unwrap();

        let frame = compositor.compose_frame().unwrap();
        assert_eq!(pixel(&frame, 0, 0), [0, 255, 0, 255]);
    }

    #[test]
    fn frame_replacement_and_composition_run_concurrently() {
        let (mut compositor, handle) = VideoCompositor::new("compositor", options(1, 1)).unwrap();
        let (mut sink, _) = input(&handle, "live", VideoLayer::new(VideoRect::new(0, 0, 1, 1)));
        let red = solid_frame(1, 1, Color::new(255, 0, 0));
        let green = solid_frame(1, 1, Color::new(0, 255, 0));
        let done = Arc::new(AtomicBool::new(false));
        let producer_done = done.clone();
        let producer = thread::spawn(move || {
            for index in 0..2_000 {
                let frame = if index % 2 == 0 {
                    red.clone()
                } else {
                    green.clone()
                };
                sink.consume(MediaBuffer::Video(frame)).unwrap();
            }
            // Make the final observable value deterministic after the
            // concurrent replacement phase ends.
            sink.consume(MediaBuffer::Video(green)).unwrap();
            producer_done.store(true, AtomicOrdering::Release);
        });

        while !done.load(AtomicOrdering::Acquire) {
            let frame = compositor.compose_frame().unwrap();
            assert!(matches!(
                pixel(&frame, 0, 0),
                [0, 0, 0, 255] | [0, 0, 255, 255] | [0, 255, 0, 255]
            ));
        }
        producer.join().unwrap();
        let final_frame = compositor.compose_frame().unwrap();
        assert_eq!(pixel(&final_frame, 0, 0), [0, 255, 0, 255]);
    }

    #[test]
    fn replacing_a_name_invalidates_old_sink_and_layer_handle() {
        let (mut compositor, handle) = VideoCompositor::new("compositor", options(1, 1)).unwrap();
        let layer = VideoLayer::new(VideoRect::new(0, 0, 1, 1));
        let (mut old_sink, old_layer) = input(&handle, "camera", layer);
        let (mut new_sink, _) = input(&handle, "camera", layer);
        assert!(matches!(
            old_layer.set_visible(false),
            Err(VideoCompositorError::SourceRemoved)
        ));
        old_sink
            .consume(MediaBuffer::Video(solid_frame(1, 1, Color::new(255, 0, 0))))
            .unwrap();
        new_sink
            .consume(MediaBuffer::Video(solid_frame(1, 1, Color::new(0, 0, 255))))
            .unwrap();

        let frame = compositor.compose_frame().unwrap();
        assert_eq!(pixel(&frame, 0, 0), [255, 0, 0, 255]);
        assert_eq!(handle.source_count(), 1);
    }

    #[test]
    fn stop_removes_only_the_current_registration() {
        let (_compositor, handle) = VideoCompositor::new("compositor", options(1, 1)).unwrap();
        let layer = VideoLayer::new(VideoRect::new(0, 0, 1, 1));
        let (mut old_sink, _) = input(&handle, "camera", layer);
        let (_new_sink, _) = input(&handle, "camera", layer);
        old_sink.control(ControlMsg::Stop).unwrap();
        assert_eq!(handle.source_count(), 1);

        handle.remove_source("camera");
        assert_eq!(handle.source_count(), 0);
    }

    #[test]
    fn contain_and_cover_preserve_aspect_ratio() {
        let rect = VideoRect::new(10, 20, 100, 100);
        let contain = layer_geometry(160, 90, rect, VideoFit::Contain).unwrap();
        assert_eq!((contain.image_width, contain.image_height), (100, 56));
        assert_eq!((contain.image_x, contain.image_y), (10, 42));

        let cover = layer_geometry(160, 90, rect, VideoFit::Cover).unwrap();
        assert_eq!((cover.image_width, cover.image_height), (178, 100));
        assert_eq!((cover.image_x, cover.image_y), (-29, 20));
    }

    #[test]
    fn rejects_invalid_layers_and_non_video_buffers() {
        let (_compositor, handle) = VideoCompositor::new("compositor", options(1, 1)).unwrap();
        let invalid = VideoLayer {
            opacity: 1.5,
            ..VideoLayer::new(VideoRect::new(0, 0, 1, 1))
        };
        assert!(matches!(
            handle.add_source("invalid", invalid),
            Err(VideoCompositorError::InvalidOpacity(1.5))
        ));

        let (mut sink, _) = input(
            &handle,
            "valid",
            VideoLayer::new(VideoRect::new(0, 0, 1, 1)),
        );
        let error = sink
            .consume(MediaBuffer::Packet(Arc::new(ffmpeg::Packet::empty())))
            .unwrap_err();
        assert!(matches!(
            error,
            crate::Error::VideoCompositorError(VideoCompositorError::UnsupportedBuffer("Packet"))
        ));
    }

    #[test]
    fn pushes_fixed_format_frames_with_contiguous_pts() {
        let (mut compositor, _) = VideoCompositor::new("compositor", options(2, 2)).unwrap();
        let received = Arc::new(StdMutex::new(Vec::new()));
        compositor.src_pads()[0].link(Box::new(CapturingSink {
            received: received.clone(),
            hlog: element_hlog(ElementType::Other, "capture", None),
        }));
        let (bus, _) = Bus::new();
        compositor.push_frame(&bus).unwrap();
        compositor.push_frame(&bus).unwrap();

        let received = received.lock().unwrap();
        let pts: Vec<_> = received
            .iter()
            .filter_map(|buffer| match buffer {
                MediaBuffer::Video(frame) => Some(frame.pts()),
                _ => None,
            })
            .collect();
        assert_eq!(pts, vec![Some(0), Some(1)]);
    }

    #[rust_hlog::hlog]
    struct TimestampSink {
        tx: crossbeam_channel::Sender<Instant>,
    }

    impl Element for TimestampSink {
        fn name(&self) -> Arc<str> {
            "timestamp-recorder".into()
        }
        fn element_type(&self) -> ElementType {
            ElementType::Other
        }
        fn hlog(&self) -> &HLog {
            &self.hlog
        }
        fn hlog_mut(&mut self) -> &mut HLog {
            &mut self.hlog
        }
    }

    impl Sink for TimestampSink {
        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            if matches!(buf, MediaBuffer::Video(_)) {
                let _ = self.tx.send(Instant::now());
            }
            Ok(())
        }
        fn control(&mut self, _msg: ControlMsg) -> Result<()> {
            Ok(())
        }
    }

    /// Regression test: `VideoCompositor::run` never folded
    /// `ControlOutcome::paused_for` back into `next_due` — a `Pause` let
    /// real time blow straight past the stale deadline, so the loop
    /// iteration right after `Resume` always found `next_due` already in
    /// the past and pushed immediately, resetting the output cadence's
    /// phase to the resume instant instead of preserving wherever it was
    /// before the freeze. A slow 10fps (100ms/frame) rate keeps the
    /// expected gap (near-zero vs. near-one-interval) well clear of
    /// scheduling jitter. Pausing is triggered synchronously right after a
    /// frame is observed, so `next_due` is a known ~100ms away the instant
    /// `Pause` is drained (`run`'s own control check only happens at the
    /// top of the loop, after that deadline has already been advanced).
    #[test]
    fn resuming_after_a_pause_preserves_output_phase() {
        use crate::pipeline::Pipeline;

        let (tx, rx) = crossbeam_channel::unbounded();
        let sink = TimestampSink {
            tx,
            hlog: element_hlog(ElementType::Other, "timestamp-recorder", None),
        };
        let (compositor, _handle) = VideoCompositor::new(
            "compositor",
            VideoCompositorOptions {
                frame_rate: ffmpeg::Rational::new(10, 1),
                ..options(2, 2)
            },
        )
        .unwrap();

        let pipeline = Pipeline::new("phase-test", compositor, |source, ctx| {
            let branch = ctx.branch().to(Box::new(sink))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("test pipeline wiring must succeed");

        pipeline.run();
        // Warm up, then pause the instant a frame is observed — `next_due`
        // is then a known one interval away.
        for _ in 0..2 {
            rx.recv_timeout(Duration::from_millis(500))
                .expect("expected steady frames before pausing");
        }
        pipeline.pause();
        thread::sleep(Duration::from_millis(500));

        let resumed_at = Instant::now();
        pipeline.resume();
        let first_after_resume = rx
            .recv_timeout(Duration::from_millis(500))
            .expect("expected a frame after resume");
        pipeline.stop();
        pipeline.bus().log_events();

        let gap = first_after_resume.saturating_duration_since(resumed_at);
        assert!(
            gap >= Duration::from_millis(50),
            "expected the post-pause frame to land close to a full 100ms \
             interval after resume (phase preserved from before the \
             pause), not almost immediately (phase reset to the resume \
             instant): got {gap:?}"
        );
    }
}
