//! GPU (D3D11) video compositor — [`D3d11VideoCompositor`] itself plus
//! everything specific to driving it: input registration
//! ([`D3d11VideoCompositorHandle`]), per-input placement control
//! ([`video_handle::D3d11VideoLayerHandle`]), and the dynamic-text
//! sibling ([`text_handle::D3d11TextLayerHandle`]) each split into their
//! own file since neither is small enough to justify inlining here.

use std::{
    collections::HashMap,
    ffi::c_void,
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicU64, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use crate::pp_log::{PpLog, pp_info};
use arc_swap::ArcSwapOption;
use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;
use windows::{
    Win32::Foundation::RECT,
    Win32::Graphics::{
        Direct3D::{D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, D3D_SRV_DIMENSION_TEXTURE2DARRAY},
        Direct3D11::*,
        Dxgi::Common::{
            DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_NV12, DXGI_FORMAT_R8_UNORM,
            DXGI_FORMAT_R8G8_UNORM, DXGI_SAMPLE_DESC,
        },
    },
    core::{Interface, s},
};

mod text_handle;
mod video_handle;

pub use text_handle::{D3d11TextLayerError, D3d11TextLayerHandle};
pub use video_handle::D3d11VideoLayerHandle;

use super::super::{
    text_layer::TextLayer,
    video_layer::{self, LayerGeometry, MAX_DIMENSION, VideoLayer, VideoLayerError},
};
use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    control::{ControlMsg, ControlReceiver, drain_control},
    element::{Element, ElementType, Sink, Source, SourceElement, element_pp_log},
    elements::VideoCompositorOptions,
    error::{D3d11FrameWrapError, Result},
    pad::SrcPad,
    platform::windows::{
        d3d11::compile_shader,
        d3d11va::{d3d11va_texture, wrap_d3d11_texture},
    },
    pool::{UnboundObjectPool, UnboundObjectPoolRef},
    schedule::PeriodicSchedule,
};

const OUTPUT_POOL_SIZE: usize = 4;
const CONTROL_POLL_INTERVAL: Duration = Duration::from_millis(5);

const BGRA_SHADER_SOURCE: &[u8] =
    include_bytes!("../../../../../shaders/d3d11/composite_bgra.hlsl");
const NV12_SHADER_SOURCE: &[u8] =
    include_bytes!("../../../../../shaders/d3d11/composite_nv12.hlsl");

/// Errors specific to [`D3d11VideoCompositor`].
#[derive(Debug, ThisError)]
pub enum D3d11VideoCompositorError {
    /// FFmpeg could not allocate a reference-counted wrapper for an output texture.
    #[error(transparent)]
    FrameWrap(#[from] D3d11FrameWrapError),

    /// A Direct3D device, resource, or command operation failed.
    #[error("windows error: {0}")]
    Windows(#[from] windows::core::Error),

    /// The output canvas dimensions are zero or exceed the safety limit.
    #[error(
        "invalid output dimensions {width}x{height}; each dimension must be 1..={MAX_DIMENSION}"
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

    /// The input frame is not backed by a D3D11 texture.
    #[error(
        "D3d11VideoCompositorInputSink only accepts Pixel::D3D11 frames, got {0:?}; \
         upload/decode/capture to GPU first"
    )]
    UnsupportedFormat(ffmpeg::format::Pixel),

    /// A frame tagged as D3D11 does not contain a valid texture reference.
    #[error(
        "frame claimed the D3D11 pixel format but carries no texture — must come from \
         D3d11Upload/D3d11Decoder/DxgiCaptureSource's GPU mode"
    )]
    InvalidD3d11Frame,

    /// The D3D11 texture uses a DXGI format unsupported by the compositor shaders.
    #[error(
        "D3d11VideoCompositor only draws DXGI_FORMAT_B8G8R8A8_UNORM or DXGI_FORMAT_NV12 input \
         textures, got {0:?}"
    )]
    UnsupportedTextureFormat(DXGI_FORMAT),

    /// The frame selects a texture-array slice outside the resource bounds.
    #[error("D3D11 texture array index {index} is outside ArraySize {array_size}")]
    InvalidArrayIndex {
        /// Invalid texture-array index from the frame.
        index: isize,
        /// Number of slices in the backing texture array.
        array_size: u32,
    },

    /// The visible frame region is larger than its backing texture.
    #[error(
        "frame dimensions {frame_width}x{frame_height} exceed the backing D3D11 texture's \
         {texture_width}x{texture_height} dimensions"
    )]
    FrameExceedsTexture {
        /// Declared frame width in pixels.
        frame_width: u32,
        /// Declared frame height in pixels.
        frame_height: u32,
        /// Backing texture width in pixels.
        texture_width: u32,
        /// Backing texture height in pixels.
        texture_height: u32,
    },

    /// The input texture was created by a different D3D11 device.
    #[error(
        "a Pixel::D3D11 frame's texture lives on a different ID3D11Device than this \
         D3d11VideoCompositor was created with — every D3D11 element in one pipeline must share \
         exactly one device for zero-copy to be valid"
    )]
    DeviceMismatch,

    /// An input sink received a buffer other than decoded video.
    #[error(
        "D3d11VideoCompositorInputSink only accepts decoded Video frames, got a {0}; link it \
         after a decoder or video source"
    )]
    UnsupportedBuffer(&'static str),

    /// Seeking was requested on a live compositor with no stored timeline.
    #[error("D3d11VideoCompositor doesn't support seeking a live composition")]
    SeekUnsupported,
}

fn map_layer_error(error: VideoLayerError) -> D3d11VideoCompositorError {
    match error {
        VideoLayerError::InvalidDimensions { width, height } => {
            D3d11VideoCompositorError::InvalidLayerDimensions { width, height }
        }
        VideoLayerError::InvalidOpacity(opacity) => {
            D3d11VideoCompositorError::InvalidOpacity(opacity)
        }
        VideoLayerError::InvalidInputDimensions { width, height } => {
            D3d11VideoCompositorError::InvalidInputDimensions { width, height }
        }
        VideoLayerError::ScaledLayerTooLarge { width, height } => {
            D3d11VideoCompositorError::ScaledLayerTooLarge { width, height }
        }
    }
}

struct GpuVideoInput {
    id: video_layer::VideoInputId,
    /// Same shape as the CPU `SwVideoCompositor`'s own `VideoInput` field —
    /// `ffmpeg::frame::Video` is the same Rust type whether its pixel data
    /// is CPU-resident or (as with every `Pixel::D3D11` frame in this
    /// crate) a GPU texture pointer smuggled through `data[0]`. Replacing
    /// the pointer is a lock-free atomic swap; the compositor takes a
    /// stable `Arc` snapshot independently.
    latest_frame: ArcSwapOption<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    layer: Mutex<VideoLayer>,
}

struct D3d11CompositorShared {
    inputs: Mutex<HashMap<Arc<str>, Arc<GpuVideoInput>>>,
    next_input_id: AtomicU64,
    /// Lets [`D3d11VideoCompositorHandle::add_text_layer`] build GPU
    /// resources guaranteed to match this compositor's own device, without
    /// requiring a separately-threaded-through device parameter that could
    /// drift from the real one.
    device: ID3D11Device,
}

/// A cheaply cloneable handle for adding and removing
/// [`D3d11VideoCompositor`] inputs — the GPU sibling of
/// [`crate::elements::SwVideoCompositorHandle`], same shape and behavior.
#[derive(Clone)]
pub struct D3d11VideoCompositorHandle {
    shared: Weak<D3d11CompositorShared>,
}

/// The two endpoints created for one compositor input registration.
pub struct D3d11VideoCompositorInput {
    /// Terminal sink to attach to the input pipeline branch.
    pub sink: Box<dyn Sink>,
    /// Runtime control for this input's placement and visibility.
    pub layer: D3d11VideoLayerHandle,
}

impl D3d11VideoCompositorHandle {
    /// Registers an input under `name` and returns its layer handle —
    /// shared by [`Self::add_source`] (which additionally wraps a `Sink`
    /// around the same registration) and [`Self::add_layer`] (which
    /// returns just this handle, for handle-driven inputs like
    /// [`crate::elements::TextLayer`] that never receive `Pipeline`
    /// frames and would otherwise build a `Sink` only to discard it).
    fn register_input(
        &self,
        name: impl Into<String>,
        layer: VideoLayer,
    ) -> std::result::Result<Option<D3d11VideoLayerHandle>, D3d11VideoCompositorError> {
        video_layer::validate_layer(layer).map_err(map_layer_error)?;
        let Some(shared) = self.shared.upgrade() else {
            return Ok(None);
        };
        let name: Arc<str> = name.into().into();
        let id = video_layer::VideoInputId(shared.next_input_id.fetch_add(1, Ordering::Relaxed));
        let input = Arc::new(GpuVideoInput {
            id,
            latest_frame: ArcSwapOption::empty(),
            layer: Mutex::new(layer),
        });
        shared
            .inputs
            .lock()
            .unwrap()
            .insert(name.clone(), input.clone());

        Ok(Some(D3d11VideoLayerHandle {
            id,
            name,
            input: Arc::downgrade(&input),
        }))
    }

    /// Registers an input and returns its terminal Sink plus independent
    /// runtime layer control — see
    /// [`crate::elements::SwVideoCompositorHandle::add_source`]'s own docs
    /// (identical contract: reusing `name` replaces the old registration,
    /// old sinks/layer handles become harmlessly stale).
    pub fn add_source(
        &self,
        name: impl Into<String>,
        layer: VideoLayer,
    ) -> std::result::Result<Option<D3d11VideoCompositorInput>, D3d11VideoCompositorError> {
        let Some(layer_handle) = self.register_input(name, layer)? else {
            return Ok(None);
        };
        Ok(Some(D3d11VideoCompositorInput {
            sink: Box::new(D3d11VideoCompositorInputSink {
                name: layer_handle.name.clone(),
                pp_log: element_pp_log(ElementType::D3d11VideoCompositor, &layer_handle.name, None),
                shared: self.shared.clone(),
                input: layer_handle.input.clone(),
            }),
            layer: layer_handle,
        }))
    }

    /// Registers an input and returns *only* its layer handle — no `Sink`
    /// at all — for a caller that drives this input's frames directly via
    /// [`D3d11VideoLayerHandle::set_frame`] instead of wiring a `Pipeline`
    /// branch into it (e.g. [`crate::elements::TextLayer`]). Same
    /// replace-on-reuse contract as [`Self::add_source`].
    pub fn add_layer(
        &self,
        name: impl Into<String>,
        layer: VideoLayer,
    ) -> std::result::Result<Option<D3d11VideoLayerHandle>, D3d11VideoCompositorError> {
        self.register_input(name, layer)
    }

    /// Removes `name`; existing sinks and handles become stale immediately.
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

    /// Registers a new text layer and returns a [`D3d11TextLayerHandle`]
    /// ready for [`D3d11TextLayerHandle::set_text`] — the text-specific
    /// sibling of [`Self::add_layer`] (which stays generic, unaware of
    /// text), taking a [`TextLayer`] the same way `add_source` takes a
    /// [`VideoLayer`]. Always uses this compositor's own device
    /// internally, so unlike a hand-assembled `add_layer` +
    /// separately-supplied device there's no way to accidentally construct
    /// a `D3d11TextLayerHandle` against the wrong device — the one class
    /// of bug a caller-supplied device would allow. Returns `None` if the
    /// compositor has already been dropped, matching [`Self::add_layer`]'s
    /// own contract.
    pub fn add_text_layer(
        &self,
        name: impl Into<String>,
        text_layer: TextLayer,
    ) -> std::result::Result<Option<D3d11TextLayerHandle>, D3d11TextLayerError> {
        let Some(device) = self.shared.upgrade().map(|shared| shared.device.clone()) else {
            return Ok(None);
        };
        // Validate everything that can fail before replacing an existing
        // registration with the same name.
        let font = D3d11TextLayerHandle::parse_font(text_layer.font_data, text_layer.font_size)?;
        // Placeholder rect: no text has been rasterized yet, so its exact
        // size is unknown. `D3d11TextLayerHandle::set_text` immediately overwrites
        // this with the real bitmap size the first time it's called.
        let placeholder = VideoLayer::new(video_layer::VideoRect::new(
            text_layer.x,
            text_layer.y,
            1,
            1,
        ));
        let Some(layer) = self.add_layer(name, placeholder)? else {
            return Ok(None);
        };
        Ok(Some(D3d11TextLayerHandle::new(
            layer,
            &device,
            font,
            text_layer.font_size,
            text_layer.color,
        )))
    }
}

/// One terminal video input returned by
/// [`D3d11VideoCompositorHandle::add_source`] — the GPU sibling of
/// [`crate::elements::SwVideoCompositorInputSink`], same behavior (stores
/// only the latest frame; a fast producer can't build an unbounded queue
/// behind a slower compositor output rate).
pub struct D3d11VideoCompositorInputSink {
    pp_log: PpLog,
    name: Arc<str>,
    shared: Weak<D3d11CompositorShared>,
    input: Weak<GpuVideoInput>,
}

impl D3d11VideoCompositorInputSink {
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

impl Element for D3d11VideoCompositorInputSink {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::D3d11VideoCompositor
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Sink for D3d11VideoCompositorInputSink {
    /// Every layer is composited on the GPU, so each input takes a
    /// device texture just as the composed output produces one.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::frame(
            MediaKind::VideoFrame,
            MemoryDomain::D3d11,
        ))
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        let Some(input) = self.input.upgrade() else {
            return Ok(());
        };
        match buf {
            MediaBuffer::Video(frame) => {
                if frame.format() != ffmpeg::format::Pixel::D3D11 {
                    return Err(D3d11VideoCompositorError::UnsupportedFormat(frame.format()).into());
                }
                input.latest_frame.store(Some(frame));
                Ok(())
            }
            MediaBuffer::Eos => {
                self.detach();
                Ok(())
            }
            MediaBuffer::Packet(_) => {
                Err(D3d11VideoCompositorError::UnsupportedBuffer("Packet").into())
            }
            MediaBuffer::Audio(_) => {
                Err(D3d11VideoCompositorError::UnsupportedBuffer("Audio").into())
            }
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
    id: video_layer::VideoInputId,
    layer: VideoLayer,
    frame: Option<Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>>,
}

struct OutputTarget {
    texture: ID3D11Texture2D,
    render_target_view: ID3D11RenderTargetView,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct LayerConstants {
    red: [f32; 4],
    green: [f32; 4],
    blue: [f32; 4],
    opacity: f32,
    _padding: [f32; 3],
    uv_scale: [f32; 2],
    _uv_padding: [f32; 2],
}

impl LayerConstants {
    fn bgra(opacity: f32, uv_scale: [f32; 2]) -> Self {
        Self {
            red: [0.0; 4],
            green: [0.0; 4],
            blue: [0.0; 4],
            opacity,
            _padding: [0.0; 3],
            uv_scale,
            _uv_padding: [0.0; 2],
        }
    }

    fn nv12(frame: &ffmpeg::frame::Video, opacity: f32, uv_scale: [f32; 2]) -> Self {
        let [red, green, blue] =
            yuv_to_rgb_rows(frame.color_space(), frame.color_range(), frame.height());
        Self {
            red,
            green,
            blue,
            opacity,
            _padding: [0.0; 3],
            uv_scale,
            _uv_padding: [0.0; 2],
        }
    }
}

fn visible_uv_scale(
    frame_width: u32,
    frame_height: u32,
    texture_width: u32,
    texture_height: u32,
) -> std::result::Result<[f32; 2], D3d11VideoCompositorError> {
    if frame_width > texture_width || frame_height > texture_height {
        return Err(D3d11VideoCompositorError::FrameExceedsTexture {
            frame_width,
            frame_height,
            texture_width,
            texture_height,
        });
    }
    Ok([
        frame_width as f32 / texture_width as f32,
        frame_height as f32 / texture_height as f32,
    ])
}

/// Composites the latest frames from any number of independent GPU-backed
/// input pipelines into one fixed-rate opaque `Pixel::D3D11` (BGRA)
/// stream, entirely on the GPU via a D3D11 pixel shader — the D3D11
/// sibling of [`crate::elements::SwVideoCompositor`] (which does the same
/// job on the CPU via `libswscale`). Same [`VideoLayer`]/[`video_layer::VideoRect`]/
/// [`video_layer::VideoFit`] API — a caller's layer-control code doesn't
/// change shape when switching between the two.
///
/// Every input must already be a `Pixel::D3D11` frame (from
/// [`crate::elements::D3d11Upload`], [`crate::elements::D3d11Decoder`], or
/// `DxgiCaptureSource`'s GPU mode) on the exact same
/// `ID3D11Device`/shared context this compositor was built with — see
/// [`D3d11VideoCompositor::new`]'s own docs on why `context` specifically
/// must be shared, not just the device.
///
/// Like [`crate::elements::SwVideoCompositor`], this is a [`SourceElement`],
/// not a conventional one-input filter: upstream pipelines terminate at
/// the sinks returned by [`D3d11VideoCompositorHandle::add_source`], while
/// this element's own pipeline drives output on its independent clock.
pub struct D3d11VideoCompositor {
    pp_log: PpLog,
    name: Arc<str>,
    shared: Arc<D3d11CompositorShared>,
    options: VideoCompositorOptions,
    frame_interval: Duration,
    frame_index: i64,
    device: ID3D11Device,
    context: Arc<Mutex<ID3D11DeviceContext>>,
    vertex_shader: ID3D11VertexShader,
    bgra_pixel_shader: ID3D11PixelShader,
    nv12_pixel_shader: ID3D11PixelShader,
    sampler: ID3D11SamplerState,
    blend_state: ID3D11BlendState,
    rasterizer_state: ID3D11RasterizerState,
    layer_buffer: ID3D11Buffer,
    /// Checked-out frames remain unavailable until every downstream `Arc`
    /// is dropped. The unbounded pool grows when all prefilled frames are
    /// still queued or being consumed instead of aliasing a live texture.
    output_pool: UnboundObjectPool<ffmpeg::frame::Video>,
    /// Immutable RTVs cached by the corresponding texture's COM pointer.
    output_views: HashMap<usize, ID3D11RenderTargetView>,
    pad: SrcPad,
}

// SAFETY: every field is either a `windows-rs` COM interface wrapper
// (free-threaded for the calls used here, the context-touching ones behind
// `context`'s own `Mutex`) or plain data. `&mut self` on every
// method that touches non-`Arc`/`Mutex` state rules out concurrent access
// to those parts from multiple threads.
unsafe impl Send for D3d11VideoCompositor {}

impl D3d11VideoCompositor {
    /// `device` must be the same `ID3D11Device` every producer feeding this
    /// compositor's inputs uses. `context` must be the exact same shared
    /// `Arc<Mutex<ID3D11DeviceContext>>` every other context-touching D3D11
    /// consumer in this pipeline uses (e.g.
    /// `render_common::D3d11GpuContext::context()`) — reading a texture via
    /// `CopySubresourceRegion`/`Map` ([`crate::elements::D3d11Download`]) or drawing
    /// with it (a window renderer) are both context-level operations, and
    /// only funneling every one of them through one shared, mutex-guarded
    /// context is what lets this whole stack skip explicit GPU fences once
    /// a consumer has submitted its read. Output texture reuse is governed
    /// separately by the output frame's downstream `Arc` lifetime, because
    /// a frame waiting inside a `Queue` has not submitted that read yet.
    pub fn new(
        name: impl Into<String>,
        device: &ID3D11Device,
        context: Arc<Mutex<ID3D11DeviceContext>>,
        options: VideoCompositorOptions,
    ) -> std::result::Result<(Self, D3d11VideoCompositorHandle), D3d11VideoCompositorError> {
        validate_output_options(options)?;
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::D3d11VideoCompositor, &name, None);
        let shared = Arc::new(D3d11CompositorShared {
            inputs: Mutex::new(HashMap::new()),
            next_input_id: AtomicU64::new(1),
            device: device.clone(),
        });
        let frame_interval = Duration::from_secs_f64(
            options.frame_rate.denominator() as f64 / options.frame_rate.numerator() as f64,
        );

        // SAFETY: `device` is live; the helper reads only static shader bytes
        // and fully initialized descriptors and returns owned COM interfaces.
        let (
            vertex_shader,
            bgra_pixel_shader,
            nv12_pixel_shader,
            sampler,
            blend_state,
            rasterizer_state,
            layer_buffer,
        ) = unsafe { build_pipeline_state(device)? };

        pp_info!(
            pp_log: &pp_log,
            "created: {}x{}, frame_rate={}, format=D3D11(BGRA)",
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
                device: device.clone(),
                context,
                vertex_shader,
                bgra_pixel_shader,
                nv12_pixel_shader,
                sampler,
                blend_state,
                rasterizer_state,
                layer_buffer,
                output_pool: UnboundObjectPool::new(
                    OUTPUT_POOL_SIZE,
                    ffmpeg::frame::Video::empty,
                    |_| {},
                ),
                output_views: HashMap::new(),
                pad: SrcPad::with_contract(
                    format!("{name}_src"),
                    OutputContract::Fixed(PortContract::frame(
                        MediaKind::VideoFrame,
                        MemoryDomain::D3d11,
                    )),
                ),
            },
            D3d11VideoCompositorHandle {
                shared: Arc::downgrade(&shared),
            },
        ))
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

    /// Returns the reciprocal of [`Self::frame_rate`], used as output PTS units.
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

    /// Composites one output frame. A single bad input layer (mismatched
    /// device, unsupported texture format, oversized frame, ...) is
    /// skipped and reported on `bus` as a [`BusEvent::Error`] rather than
    /// failing the whole call — same "elements don't die on data errors"
    /// contract as the rest of this crate (see
    /// [`crate::element::SourceElement::run`]'s own docs: a `Result::Err`
    /// returned from here would propagate all the way out of `run()` and
    /// end this compositor's output permanently, which one misbehaving
    /// input has no business doing). Only a genuine infrastructure failure
    /// (e.g. `create_output_target` failing) still returns `Err`.
    fn compose_frame(
        &mut self,
        bus: &Bus,
    ) -> std::result::Result<UnboundObjectPoolRef<ffmpeg::frame::Video>, D3d11VideoCompositorError>
    {
        let mut snapshots = self.snapshots();
        snapshots.sort_by(|left, right| {
            left.layer
                .z_index
                .cmp(&right.layer.z_index)
                .then_with(|| left.id.cmp(&right.id))
        });

        let (canvas_width, canvas_height) = (self.options.width, self.options.height);
        let mut output_frame = self.output_pool.get();
        if d3d11va_texture(&output_frame).is_none() {
            // SAFETY: the compositor's device is live and the nonzero canvas
            // dimensions were validated at construction; the helper returns
            // owned texture and view interfaces.
            let target =
                unsafe { create_output_target(&self.device, canvas_width, canvas_height)? };
            let key = target.texture.as_raw() as usize;
            self.output_views
                .insert(key, target.render_target_view.clone());
            *output_frame = wrap_d3d11_texture(target.texture, canvas_width, canvas_height)?;
        }
        let (output_raw, _) = d3d11va_texture(&output_frame)
            .expect("output pool frames are initialized immediately after checkout");
        // SAFETY: `output_raw` is borrowed from the still-live pooled frame;
        // cloning the wrapper acquires an independent COM reference.
        let output_texture = unsafe {
            ID3D11Texture2D::from_raw_borrowed(&output_raw)
                .expect("pooled output texture pointer must not be null")
                .clone()
        };
        let output_view = self
            .output_views
            .get(&(output_texture.as_raw() as usize))
            .expect("every initialized output texture has a cached RTV")
            .clone();

        let context = self
            .context
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // SAFETY: target view, pipeline state, and every snapshot resource are
        // live on this context's device. The mutex serializes the immediate
        // context; each layer validates its own texture before drawing, and
        // all bindings are cleared before resources can be released.
        unsafe {
            let background = &self.options.background;
            context.ClearRenderTargetView(
                &output_view,
                &[
                    f32::from(background.red) / 255.0,
                    f32::from(background.green) / 255.0,
                    f32::from(background.blue) / 255.0,
                    1.0,
                ],
            );
            context.OMSetRenderTargets(Some(&[Some(output_view)]), None);
            context.OMSetBlendState(&self.blend_state, None, 0xffff_ffff);
            context.RSSetState(&self.rasterizer_state);
            context.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            context.VSSetShader(&self.vertex_shader, None);
            context.PSSetSamplers(0, Some(&[Some(self.sampler.clone())]));
            context.PSSetConstantBuffers(0, Some(&[Some(self.layer_buffer.clone())]));

            for snapshot in &snapshots {
                if !snapshot.layer.visible || snapshot.layer.opacity == 0.0 {
                    continue;
                }
                let Some(frame) = &snapshot.frame else {
                    continue;
                };
                if let Err(error) = self.draw_layer(
                    &context,
                    frame,
                    &snapshot.layer,
                    canvas_width,
                    canvas_height,
                ) {
                    // Skip just this layer — a bad frame on one input (wrong
                    // device, unsupported format, ...) is a data error, not
                    // grounds to fail the whole composited output.
                    bus.post(
                        &self.pp_log,
                        BusEvent::Error {
                            element_type: ElementType::D3d11VideoCompositor,
                            name: self.name.clone(),
                            error: error.into(),
                        },
                    );
                }
            }

            // Always release resource bindings, including when a layer
            // above was skipped after earlier layers had already drawn.
            context.PSSetShaderResources(0, Some(&[None, None]));
            context.OMSetRenderTargets(None, None);
        };
        drop(context);

        output_frame.set_pts(Some(self.frame_index));
        output_frame.set_color_space(ffmpeg::color::Space::RGB);
        output_frame.set_color_range(ffmpeg::color::Range::JPEG);
        self.frame_index += 1;
        Ok(output_frame)
    }

    unsafe fn draw_layer(
        &self,
        context: &ID3D11DeviceContext,
        frame: &ffmpeg::frame::Video,
        layer: &VideoLayer,
        canvas_width: u32,
        canvas_height: u32,
    ) -> std::result::Result<(), D3d11VideoCompositorError> {
        let (texture_raw, index) =
            d3d11va_texture(frame).ok_or(D3d11VideoCompositorError::InvalidD3d11Frame)?;
        // SAFETY: `texture_raw` is a borrowed raw `ID3D11Texture2D*` — see
        // `d3d11va_texture`'s own docs; `.clone()` (`AddRef`) gives an
        // independently ref-counted handle valid for this draw call.
        let texture = unsafe {
            ID3D11Texture2D::from_raw_borrowed(&texture_raw)
                .expect("D3d11 frame's texture pointer must not be null")
                .clone()
        };

        // SAFETY: `texture` is a live cloned COM interface; `GetDevice`
        // returns an owned reference to its creating device.
        let texture_device =
            unsafe { texture.GetDevice() }.map_err(D3d11VideoCompositorError::from)?;
        if texture_device.as_raw() != self.device.as_raw() {
            return Err(D3d11VideoCompositorError::DeviceMismatch);
        }

        let mut desc = D3D11_TEXTURE2D_DESC::default();
        // SAFETY: `desc` is a live out-parameter for the live texture.
        unsafe { texture.GetDesc(&mut desc) };
        if index < 0 || index as u64 >= u64::from(desc.ArraySize) {
            return Err(D3d11VideoCompositorError::InvalidArrayIndex {
                index,
                array_size: desc.ArraySize,
            });
        }
        let array_index = index as u32;
        let uv_scale = visible_uv_scale(frame.width(), frame.height(), desc.Width, desc.Height)?;

        let geometry =
            video_layer::layer_geometry(frame.width(), frame.height(), layer.rect, layer.fit)
                .map_err(map_layer_error)?;
        let Some((viewport, scissor)) = clipped_viewport(&geometry, canvas_width, canvas_height)
        else {
            return Ok(());
        };

        let constants = match desc.Format {
            DXGI_FORMAT_B8G8R8A8_UNORM => LayerConstants::bgra(layer.opacity, uv_scale),
            DXGI_FORMAT_NV12 => LayerConstants::nv12(frame, layer.opacity, uv_scale),
            other => return Err(D3d11VideoCompositorError::UnsupportedTextureFormat(other)),
        };
        // SAFETY: `constants` is readable for the declared constant-buffer
        // size, and the live context owns the destination buffer.
        unsafe {
            context.UpdateSubresource(
                &self.layer_buffer,
                0,
                None,
                (&raw const constants).cast::<c_void>(),
                0,
                0,
            );
            context.RSSetViewports(Some(&[viewport]));
            context.RSSetScissorRects(Some(&[scissor]));
        }

        match desc.Format {
            DXGI_FORMAT_B8G8R8A8_UNORM => {
                let srv_desc = plane_srv_desc(DXGI_FORMAT_B8G8R8A8_UNORM, array_index);
                let mut srv = None;
                // SAFETY: validation established the BGRA format and bounded
                // array slice; `srv` is a live out-parameter, and the resulting
                // view and shader remain live through the synchronous draw.
                unsafe {
                    self.device
                        .CreateShaderResourceView(&texture, Some(&srv_desc), Some(&mut srv))
                        .map_err(D3d11VideoCompositorError::from)?;
                    context.PSSetShader(&self.bgra_pixel_shader, None);
                    context.PSSetShaderResources(0, Some(&[srv]));
                    context.Draw(3, 0);
                }
            }
            DXGI_FORMAT_NV12 => {
                let luma_desc = plane_srv_desc(DXGI_FORMAT_R8_UNORM, array_index);
                let chroma_desc = plane_srv_desc(DXGI_FORMAT_R8G8_UNORM, array_index);
                let mut luma_srv = None;
                let mut chroma_srv = None;
                // SAFETY: validation established an NV12 texture and bounded
                // slice; both plane descriptors and out-parameters match that
                // texture, and the views remain live through the draw.
                unsafe {
                    self.device
                        .CreateShaderResourceView(&texture, Some(&luma_desc), Some(&mut luma_srv))
                        .map_err(D3d11VideoCompositorError::from)?;
                    self.device
                        .CreateShaderResourceView(
                            &texture,
                            Some(&chroma_desc),
                            Some(&mut chroma_srv),
                        )
                        .map_err(D3d11VideoCompositorError::from)?;
                    context.PSSetShader(&self.nv12_pixel_shader, None);
                    context.PSSetShaderResources(0, Some(&[luma_srv, chroma_srv]));
                    context.Draw(3, 0);
                }
            }
            _ => unreachable!("texture format was validated before updating constants"),
        }
        Ok(())
    }

    fn push_frame(&mut self, bus: &Bus) -> std::result::Result<(), D3d11VideoCompositorError> {
        let output = self.compose_frame(bus)?;
        if let Err(error) = self.pad.push(MediaBuffer::Video(Arc::new(output))) {
            bus.post(
                &self.pp_log,
                BusEvent::Error {
                    element_type: ElementType::D3d11VideoCompositor,
                    name: self.name.clone(),
                    error,
                },
            );
        }
        Ok(())
    }
}

impl Element for D3d11VideoCompositor {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::D3d11VideoCompositor
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for D3d11VideoCompositor {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl SourceElement for D3d11VideoCompositor {
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
        Err(D3d11VideoCompositorError::SeekUnsupported.into())
    }
}

fn validate_output_options(
    options: VideoCompositorOptions,
) -> std::result::Result<(), D3d11VideoCompositorError> {
    if options.width == 0
        || options.height == 0
        || options.width > MAX_DIMENSION
        || options.height > MAX_DIMENSION
    {
        return Err(D3d11VideoCompositorError::InvalidOutputDimensions {
            width: options.width,
            height: options.height,
        });
    }
    if options.frame_rate.numerator() <= 0 || options.frame_rate.denominator() <= 0 {
        return Err(D3d11VideoCompositorError::InvalidFrameRate(
            options.frame_rate,
        ));
    }
    Ok(())
}

/// Builds three affine rows that turn normalized `(Y, Cb, Cr, 1)` samples
/// into RGB. Unspecified color metadata follows the common SD/HD fallback:
/// BT.601 through 576 lines and BT.709 above it; unspecified range is
/// treated as MPEG/limited, matching ordinary decoded NV12 video.
fn yuv_to_rgb_rows(
    space: ffmpeg::color::Space,
    range: ffmpeg::color::Range,
    height: u32,
) -> [[f32; 4]; 3] {
    let (kr, kb) = match space {
        ffmpeg::color::Space::BT709 => (0.2126f32, 0.0722f32),
        ffmpeg::color::Space::BT2020NCL | ffmpeg::color::Space::BT2020CL => (0.2627f32, 0.0593f32),
        ffmpeg::color::Space::FCC => (0.30f32, 0.11f32),
        ffmpeg::color::Space::SMPTE240M => (0.212f32, 0.087f32),
        ffmpeg::color::Space::Unspecified if height > 576 => (0.2126f32, 0.0722f32),
        _ => (0.299f32, 0.114f32),
    };
    let kg = 1.0 - kr - kb;
    let (y_offset, y_scale, chroma_scale) = match range {
        ffmpeg::color::Range::JPEG => (0.0, 1.0, 1.0),
        ffmpeg::color::Range::MPEG | ffmpeg::color::Range::Unspecified => {
            (16.0 / 255.0, 255.0 / 219.0, 255.0 / 224.0)
        }
    };
    let chroma_offset = 128.0 / 255.0;
    let red_cr = 2.0 * (1.0 - kr) * chroma_scale;
    let blue_cb = 2.0 * (1.0 - kb) * chroma_scale;
    let green_cb = -2.0 * kb * (1.0 - kb) / kg * chroma_scale;
    let green_cr = -2.0 * kr * (1.0 - kr) / kg * chroma_scale;
    let offset = |cb: f32, cr: f32| -y_scale * y_offset - cb * chroma_offset - cr * chroma_offset;

    [
        [y_scale, 0.0, red_cr, offset(0.0, red_cr)],
        [y_scale, green_cb, green_cr, offset(green_cb, green_cr)],
        [y_scale, blue_cb, 0.0, offset(blue_cb, 0.0)],
    ]
}

/// Turns pixel-space [`LayerGeometry`] into a D3D11 viewport (where the
/// whole scaled image is drawn) plus a scissor rect (what's actually kept,
/// clipped to the layer's own rect *and* the canvas) — the GPU equivalent
/// of the CPU `SwVideoCompositor`'s `blend_bgra` clipping math, computed
/// once instead of per output pixel. `None` if the clipped area is empty
/// (nothing to draw).
fn clipped_viewport(
    geometry: &LayerGeometry,
    canvas_width: u32,
    canvas_height: u32,
) -> Option<(D3D11_VIEWPORT, RECT)> {
    let output_width = i64::from(canvas_width);
    let output_height = i64::from(canvas_height);
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
        return None;
    }

    let viewport = D3D11_VIEWPORT {
        TopLeftX: geometry.image_x as f32,
        TopLeftY: geometry.image_y as f32,
        Width: geometry.image_width as f32,
        Height: geometry.image_height as f32,
        MinDepth: 0.0,
        MaxDepth: 1.0,
    };
    let scissor = RECT {
        left: left as i32,
        top: top as i32,
        right: right as i32,
        bottom: bottom as i32,
    };
    Some((viewport, scissor))
}

/// Builds a single-slice `Texture2DArray` SRV description — see
/// `render_common::d3d11_window_renderer`'s own `plane_srv_desc` (same
/// shape, duplicated here since that one is private to an example crate,
/// not something `lib` can depend on).
fn plane_srv_desc(format: DXGI_FORMAT, array_index: u32) -> D3D11_SHADER_RESOURCE_VIEW_DESC {
    D3D11_SHADER_RESOURCE_VIEW_DESC {
        Format: format,
        ViewDimension: D3D_SRV_DIMENSION_TEXTURE2DARRAY,
        Anonymous: D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2DArray: D3D11_TEX2D_ARRAY_SRV {
                MostDetailedMip: 0,
                MipLevels: 1,
                FirstArraySlice: array_index,
                ArraySize: 1,
            },
        },
    }
}

#[allow(clippy::type_complexity)]
unsafe fn build_pipeline_state(
    device: &ID3D11Device,
) -> windows::core::Result<(
    ID3D11VertexShader,
    ID3D11PixelShader,
    ID3D11PixelShader,
    ID3D11SamplerState,
    ID3D11BlendState,
    ID3D11RasterizerState,
    ID3D11Buffer,
)> {
    // SAFETY: the device is live; compiled blobs retain their bytecode while
    // shader creation reads it, all state descriptors are initialized, and
    // every interface slot is a live out-parameter. No Rust pointer is kept.
    unsafe {
        let vertex_bytecode = compile_shader(
            BGRA_SHADER_SOURCE,
            s!("composite_bgra.hlsl"),
            s!("vs_main"),
            s!("vs_5_0"),
        )?;
        let bgra_bytecode = compile_shader(
            BGRA_SHADER_SOURCE,
            s!("composite_bgra.hlsl"),
            s!("ps_bgra"),
            s!("ps_5_0"),
        )?;
        let nv12_bytecode = compile_shader(
            NV12_SHADER_SOURCE,
            s!("composite_nv12.hlsl"),
            s!("ps_nv12"),
            s!("ps_5_0"),
        )?;

        let mut vertex_shader = None;
        device.CreateVertexShader(
            std::slice::from_raw_parts(
                vertex_bytecode.GetBufferPointer().cast::<u8>(),
                vertex_bytecode.GetBufferSize(),
            ),
            None,
            Some(&mut vertex_shader),
        )?;
        let vertex_shader = vertex_shader.unwrap();

        let mut bgra_pixel_shader = None;
        device.CreatePixelShader(
            std::slice::from_raw_parts(
                bgra_bytecode.GetBufferPointer().cast::<u8>(),
                bgra_bytecode.GetBufferSize(),
            ),
            None,
            Some(&mut bgra_pixel_shader),
        )?;
        let bgra_pixel_shader = bgra_pixel_shader.unwrap();

        let mut nv12_pixel_shader = None;
        device.CreatePixelShader(
            std::slice::from_raw_parts(
                nv12_bytecode.GetBufferPointer().cast::<u8>(),
                nv12_bytecode.GetBufferSize(),
            ),
            None,
            Some(&mut nv12_pixel_shader),
        )?;
        let nv12_pixel_shader = nv12_pixel_shader.unwrap();

        let sampler_desc = D3D11_SAMPLER_DESC {
            Filter: D3D11_FILTER_MIN_MAG_MIP_LINEAR,
            AddressU: D3D11_TEXTURE_ADDRESS_CLAMP,
            AddressV: D3D11_TEXTURE_ADDRESS_CLAMP,
            AddressW: D3D11_TEXTURE_ADDRESS_CLAMP,
            ComparisonFunc: D3D11_COMPARISON_NEVER,
            MaxLOD: f32::MAX,
            ..Default::default()
        };
        let mut sampler = None;
        device.CreateSamplerState(&sampler_desc, Some(&mut sampler))?;
        let sampler = sampler.unwrap();

        // Standard "over" alpha compositing for RGB. Destination alpha is
        // deliberately left untouched by every draw (`SrcBlendAlpha =
        // ZERO`, `DestBlendAlpha = ONE`) so it stays at whatever the
        // initial `ClearRenderTargetView` set (1.0) — same "output is
        // always opaque" contract the CPU `SwVideoCompositor`'s
        // `blend_bgra` enforces by always writing 255 to the destination
        // alpha byte.
        let mut blend_desc = D3D11_BLEND_DESC::default();
        blend_desc.RenderTarget[0] = D3D11_RENDER_TARGET_BLEND_DESC {
            BlendEnable: true.into(),
            SrcBlend: D3D11_BLEND_SRC_ALPHA,
            DestBlend: D3D11_BLEND_INV_SRC_ALPHA,
            BlendOp: D3D11_BLEND_OP_ADD,
            SrcBlendAlpha: D3D11_BLEND_ZERO,
            DestBlendAlpha: D3D11_BLEND_ONE,
            BlendOpAlpha: D3D11_BLEND_OP_ADD,
            RenderTargetWriteMask: D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8,
        };
        let mut blend_state = None;
        device.CreateBlendState(&blend_desc, Some(&mut blend_state))?;
        let blend_state = blend_state.unwrap();

        // `ScissorEnable = TRUE`: every layer draw needs its own scissor
        // rect to crop overflow (e.g. `VideoFit::Cover`) or an
        // off-canvas rect — see `clipped_viewport`'s own docs. Left bound
        // on the shared context afterward; every renderer sharing this
        // context must explicitly select its own rasterizer state before
        // drawing, just as this compositor does.
        let rasterizer_desc = D3D11_RASTERIZER_DESC {
            FillMode: D3D11_FILL_SOLID,
            CullMode: D3D11_CULL_NONE,
            ScissorEnable: true.into(),
            DepthClipEnable: true.into(),
            ..Default::default()
        };
        let mut rasterizer_state = None;
        device.CreateRasterizerState(&rasterizer_desc, Some(&mut rasterizer_state))?;
        let rasterizer_state = rasterizer_state.unwrap();

        let buffer_desc = D3D11_BUFFER_DESC {
            ByteWidth: std::mem::size_of::<LayerConstants>() as u32,
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
            StructureByteStride: 0,
        };
        let mut layer_buffer = None;
        device.CreateBuffer(&buffer_desc, None, Some(&mut layer_buffer))?;
        let layer_buffer = layer_buffer.unwrap();

        Ok((
            vertex_shader,
            bgra_pixel_shader,
            nv12_pixel_shader,
            sampler,
            blend_state,
            rasterizer_state,
            layer_buffer,
        ))
    }
}

unsafe fn create_output_target(
    device: &ID3D11Device,
    width: u32,
    height: u32,
) -> windows::core::Result<OutputTarget> {
    // SAFETY: `device` is live, dimensions were validated by the compositor,
    // texture/view descriptors are fully initialized, and both optional
    // interface slots are live out-parameters.
    unsafe {
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: (D3D11_BIND_RENDER_TARGET.0 | D3D11_BIND_SHADER_RESOURCE.0) as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        let mut texture = None;
        device.CreateTexture2D(&desc, None, Some(&mut texture))?;
        let texture = texture.unwrap();

        let mut render_target_view = None;
        device.CreateRenderTargetView(&texture, None, Some(&mut render_target_view))?;
        let render_target_view = render_target_view.unwrap();

        Ok(OutputTarget {
            texture,
            render_target_view,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::test_support::try_d3d11_device as try_device;
    use crate::{
        color::Color,
        elements::{D3d11Download, VideoRect},
    };

    struct CapturingSink {
        pp_log: PpLog,
        received: Arc<Mutex<Vec<MediaBuffer>>>,
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

    fn bgra_texture(
        device: &ID3D11Device,
        width: u32,
        height: u32,
        bgra: [u8; 4],
    ) -> ID3D11Texture2D {
        let pixels: Vec<u8> = (0..width * height).flat_map(|_| bgra).collect();
        bgra_texture_from_pixels(device, width, height, &pixels)
    }

    fn bgra_texture_from_pixels(
        device: &ID3D11Device,
        width: u32,
        height: u32,
        pixels: &[u8],
    ) -> ID3D11Texture2D {
        assert_eq!(pixels.len(), (width * height * 4) as usize);
        // SAFETY: `pixels` is live and exactly matches the BGRA description's
        // dimensions and pitch; the output interface slot is live.
        unsafe {
            let desc = D3D11_TEXTURE2D_DESC {
                Width: width,
                Height: height,
                MipLevels: 1,
                ArraySize: 1,
                Format: DXGI_FORMAT_B8G8R8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Usage: D3D11_USAGE_DEFAULT,
                BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
                CPUAccessFlags: 0,
                MiscFlags: 0,
            };
            let initial = D3D11_SUBRESOURCE_DATA {
                pSysMem: pixels.as_ptr().cast::<c_void>(),
                SysMemPitch: width * 4,
                SysMemSlicePitch: 0,
            };
            let mut texture = None;
            device
                .CreateTexture2D(&desc, Some(&initial), Some(&mut texture))
                .expect("CreateTexture2D failed");
            texture.expect("CreateTexture2D succeeded without producing a texture")
        }
    }

    fn nv12_texture(
        device: &ID3D11Device,
        width: u32,
        height: u32,
        y: u8,
        cb: u8,
        cr: u8,
    ) -> ID3D11Texture2D {
        let row_bytes = width as usize;
        let luma_size = row_bytes * height as usize;
        let mut pixels = vec![y; luma_size + row_bytes * height.div_ceil(2) as usize];
        for pair in pixels[luma_size..].as_chunks_mut::<2>().0 {
            *pair = [cb, cr];
        }
        // SAFETY: the contiguous NV12 buffer is live and sized for the padded
        // row pitch and both planes described below; the output slot is live.
        unsafe {
            let desc = D3D11_TEXTURE2D_DESC {
                Width: width,
                Height: height,
                MipLevels: 1,
                ArraySize: 1,
                Format: DXGI_FORMAT_NV12,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Usage: D3D11_USAGE_DEFAULT,
                BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
                CPUAccessFlags: 0,
                MiscFlags: 0,
            };
            let initial = D3D11_SUBRESOURCE_DATA {
                pSysMem: pixels.as_ptr().cast::<c_void>(),
                SysMemPitch: width,
                SysMemSlicePitch: 0,
            };
            let mut texture = None;
            device
                .CreateTexture2D(&desc, Some(&initial), Some(&mut texture))
                .expect("CreateTexture2D(NV12) failed");
            texture.expect("CreateTexture2D succeeded without producing a texture")
        }
    }

    fn texture_key(frame: &ffmpeg::frame::Video) -> usize {
        d3d11va_texture(frame).expect("expected a D3D11 frame").0 as usize
    }

    fn apply_color_rows(rows: [[f32; 4]; 3], y: f32, cb: f32, cr: f32) -> [f32; 3] {
        rows.map(|row| row[0] * y + row[1] * cb + row[2] * cr + row[3])
    }

    fn pooled_video(frame: ffmpeg::frame::Video) -> MediaBuffer {
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut pooled = pool.get();
        *pooled = frame;
        MediaBuffer::Video(Arc::new(pooled))
    }

    /// A `Bus` with its receiver immediately dropped, for tests that don't
    /// care about per-layer error reporting — `Bus::post` no-ops once the
    /// receiving end is gone.
    fn test_bus() -> Bus {
        Bus::new().0
    }

    fn download_frame(
        device: &ID3D11Device,
        context: Arc<Mutex<ID3D11DeviceContext>>,
        composed: UnboundObjectPoolRef<ffmpeg::frame::Video>,
    ) -> Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        let (width, height) = (composed.width(), composed.height());
        let mut download = D3d11Download::new("download", device, context, width, height)
            .expect("D3d11Download::new should succeed");
        let received = Arc::new(Mutex::new(Vec::new()));
        download.src_pads()[0].link(Box::new(CapturingSink {
            received: received.clone(),
            pp_log: element_pp_log(ElementType::Other, "capture", None),
        }));
        download
            .consume(MediaBuffer::Video(Arc::new(composed)))
            .expect("download consume should succeed");
        let mut received = received.lock().unwrap();
        let MediaBuffer::Video(frame) = received.remove(0) else {
            panic!("expected a Video buffer");
        };
        frame
    }

    fn pixel(frame: &ffmpeg::frame::Video, x: usize, y: usize) -> [u8; 4] {
        let offset = y * frame.stride(0) + x * 4;
        frame.data(0)[offset..offset + 4].try_into().unwrap()
    }

    #[test]
    fn invalid_text_layer_does_not_replace_an_existing_registration() {
        let Some((device, context)) = try_device() else {
            return;
        };
        let options = VideoCompositorOptions {
            width: 4,
            height: 4,
            frame_rate: ffmpeg::Rational::new(30, 1),
            background: Color::BLACK,
        };
        let (_compositor, handle) =
            D3d11VideoCompositor::new("compositor", &device, context, options).unwrap();
        let existing = handle
            .add_layer("overlay", VideoLayer::new(VideoRect::new(0, 0, 1, 1)))
            .unwrap()
            .unwrap();

        let result = handle.add_text_layer("overlay", TextLayer::new(vec![0, 1, 2, 3]));

        assert!(matches!(result, Err(D3d11TextLayerError::InvalidFont(_))));
        assert_eq!(handle.source_count(), 1);
        assert!(existing.layer().is_some());
    }

    #[test]
    fn composes_gpu_inputs_in_z_order_and_preserves_output_contract() {
        let Some((device, context)) = try_device() else {
            return;
        };
        let options = VideoCompositorOptions {
            width: 4,
            height: 4,
            frame_rate: ffmpeg::Rational::new(30, 1),
            background: Color::BLACK,
        };
        let (mut compositor, handle) =
            D3d11VideoCompositor::new("compositor", &device, context.clone(), options)
                .expect("D3d11VideoCompositor::new should succeed");

        let mut background_layer = VideoLayer::new(VideoRect::new(0, 0, 4, 4));
        background_layer.fit = video_layer::VideoFit::Stretch;
        let mut red_sink = handle
            .add_source("red", background_layer)
            .unwrap()
            .unwrap()
            .sink;

        let mut overlay_layer = VideoLayer::new(VideoRect::new(1, 1, 2, 2));
        overlay_layer.z_index = 1;
        overlay_layer.fit = video_layer::VideoFit::Stretch;
        let mut blue_sink = handle
            .add_source("blue", overlay_layer)
            .unwrap()
            .unwrap()
            .sink;

        // BGRA byte order: [blue, green, red, alpha].
        let red_texture = bgra_texture(&device, 4, 4, [0, 0, 255, 255]);
        let blue_texture = bgra_texture(&device, 2, 2, [255, 0, 0, 255]);
        red_sink
            .consume(pooled_video(wrap_d3d11_texture(red_texture, 4, 4).unwrap()))
            .unwrap();
        blue_sink
            .consume(pooled_video(
                wrap_d3d11_texture(blue_texture, 2, 2).unwrap(),
            ))
            .unwrap();

        let composed = compositor
            .compose_frame(&test_bus())
            .expect("compose_frame failed");
        assert_eq!(composed.format(), ffmpeg::format::Pixel::D3D11);
        assert_eq!((composed.width(), composed.height()), (4, 4));
        assert_eq!(composed.pts(), Some(0));

        let downloaded = download_frame(&device, context, composed);
        assert_eq!(pixel(&downloaded, 0, 0), [0, 0, 255, 255], "red background");
        assert_eq!(pixel(&downloaded, 1, 1), [255, 0, 0, 255], "blue overlay");
    }

    #[test]
    fn ignores_rows_outside_the_frame_visible_dimensions() {
        let Some((device, context)) = try_device() else {
            return;
        };
        let options = VideoCompositorOptions {
            width: 4,
            height: 3,
            frame_rate: ffmpeg::Rational::new(30, 1),
            background: Color::BLACK,
        };
        let (mut compositor, handle) =
            D3d11VideoCompositor::new("compositor", &device, context.clone(), options)
                .expect("D3d11VideoCompositor::new should succeed");
        let mut layer = VideoLayer::new(VideoRect::new(0, 0, 4, 3));
        layer.fit = video_layer::VideoFit::Stretch;
        let mut sink = handle.add_source("input", layer).unwrap().unwrap().sink;

        // The frame exposes only the top three red rows of a four-row
        // texture. Sampling the full texture would blend the blue padding
        // row into the last visible output row.
        let mut pixels = Vec::with_capacity(4 * 4 * 4);
        for y in 0..4 {
            let color = if y < 3 {
                [0, 0, 255, 255]
            } else {
                [255, 0, 0, 255]
            };
            pixels.extend((0..4).flat_map(|_| color));
        }
        let texture = bgra_texture_from_pixels(&device, 4, 4, &pixels);
        sink.consume(pooled_video(wrap_d3d11_texture(texture, 4, 3).unwrap()))
            .unwrap();

        let composed = compositor
            .compose_frame(&test_bus())
            .expect("compose_frame failed");
        let downloaded = download_frame(&device, context, composed);
        for y in 0..3 {
            for x in 0..4 {
                assert_eq!(pixel(&downloaded, x, y), [0, 0, 255, 255]);
            }
        }
    }

    #[test]
    fn rejects_frame_dimensions_larger_than_the_backing_texture() {
        let error = visible_uv_scale(1920, 1088, 1920, 1080).unwrap_err();
        assert!(matches!(
            error,
            D3d11VideoCompositorError::FrameExceedsTexture {
                frame_width: 1920,
                frame_height: 1088,
                texture_width: 1920,
                texture_height: 1080,
            }
        ));
    }

    #[test]
    fn live_output_frames_keep_distinct_textures_until_the_last_arc_drops() {
        let Some((device, context)) = try_device() else {
            return;
        };
        let options = VideoCompositorOptions {
            width: 1,
            height: 1,
            frame_rate: ffmpeg::Rational::new(30, 1),
            background: Color::BLACK,
        };
        let (mut compositor, handle) =
            D3d11VideoCompositor::new("compositor", &device, context.clone(), options)
                .expect("D3d11VideoCompositor::new should succeed");
        let mut layer = VideoLayer::new(VideoRect::new(0, 0, 1, 1));
        layer.fit = video_layer::VideoFit::Stretch;
        let mut sink = handle.add_source("input", layer).unwrap().unwrap().sink;

        sink.consume(pooled_video(
            wrap_d3d11_texture(bgra_texture(&device, 1, 1, [0, 0, 255, 255]), 1, 1).unwrap(),
        ))
        .unwrap();
        let first = compositor
            .compose_frame(&test_bus())
            .expect("first compose failed");

        sink.consume(pooled_video(
            wrap_d3d11_texture(bgra_texture(&device, 1, 1, [255, 0, 0, 255]), 1, 1).unwrap(),
        ))
        .unwrap();
        let mut later = Vec::new();
        for _ in 0..OUTPUT_POOL_SIZE {
            later.push(
                compositor
                    .compose_frame(&test_bus())
                    .expect("later compose failed"),
            );
        }

        let mut keys = HashSet::new();
        keys.insert(texture_key(&first));
        keys.extend(later.iter().map(|frame| texture_key(frame)));
        assert_eq!(
            keys.len(),
            OUTPUT_POOL_SIZE + 1,
            "simultaneously-live output frames must never alias one texture"
        );

        let downloaded = download_frame(&device, context, first);
        assert_eq!(
            pixel(&downloaded, 0, 0),
            [0, 0, 255, 255],
            "later compositions must not overwrite a queued first frame"
        );
    }

    #[test]
    fn nv12_conversion_uses_frame_color_space_and_range() {
        let Some((device, context)) = try_device() else {
            return;
        };
        let options = VideoCompositorOptions {
            width: 2,
            height: 2,
            frame_rate: ffmpeg::Rational::new(30, 1),
            background: Color::BLACK,
        };
        let (mut compositor, handle) =
            D3d11VideoCompositor::new("compositor", &device, context.clone(), options)
                .expect("D3d11VideoCompositor::new should succeed");
        let mut layer = VideoLayer::new(VideoRect::new(0, 0, 2, 2));
        layer.fit = video_layer::VideoFit::Stretch;
        let mut sink = handle.add_source("input", layer).unwrap().unwrap().sink;
        let texture = nv12_texture(&device, 2, 2, 81, 90, 240);

        let mut bt601 = wrap_d3d11_texture(texture.clone(), 2, 2).unwrap();
        bt601.set_color_space(ffmpeg::color::Space::SMPTE170M);
        bt601.set_color_range(ffmpeg::color::Range::MPEG);
        sink.consume(pooled_video(bt601)).unwrap();
        let bt601 = compositor
            .compose_frame(&test_bus())
            .expect("BT.601 compose failed");
        let bt601 = download_frame(&device, context.clone(), bt601);

        let mut bt709 = wrap_d3d11_texture(texture, 2, 2).unwrap();
        bt709.set_color_space(ffmpeg::color::Space::BT709);
        bt709.set_color_range(ffmpeg::color::Range::MPEG);
        sink.consume(pooled_video(bt709)).unwrap();
        let bt709 = compositor
            .compose_frame(&test_bus())
            .expect("BT.709 compose failed");
        let bt709 = download_frame(&device, context, bt709);

        let pixel_601 = pixel(&bt601, 0, 0);
        let pixel_709 = pixel(&bt709, 0, 0);
        assert!(
            pixel_601[1].abs_diff(pixel_709[1]) >= 20,
            "the same NV12 sample should use different 601/709 matrices: {pixel_601:?} vs {pixel_709:?}"
        );
        assert_eq!(pixel_601[3], 255);
        assert_eq!(pixel_709[3], 255);
    }

    #[test]
    fn nv12_conversion_distinguishes_limited_and_full_range() {
        let limited = yuv_to_rgb_rows(
            ffmpeg::color::Space::BT709,
            ffmpeg::color::Range::MPEG,
            1080,
        );
        let full = yuv_to_rgb_rows(
            ffmpeg::color::Space::BT709,
            ffmpeg::color::Range::JPEG,
            1080,
        );
        let neutral = 128.0 / 255.0;
        let limited_black = apply_color_rows(limited, 16.0 / 255.0, neutral, neutral);
        let limited_white = apply_color_rows(limited, 235.0 / 255.0, neutral, neutral);
        let full_black = apply_color_rows(full, 0.0, neutral, neutral);
        let full_white = apply_color_rows(full, 1.0, neutral, neutral);

        for channel in limited_black.into_iter().chain(full_black) {
            assert!(channel.abs() < 1e-5, "black mapped to {channel}");
        }
        for channel in limited_white.into_iter().chain(full_white) {
            assert!((channel - 1.0).abs() < 1e-5, "white mapped to {channel}");
        }
    }

    #[test]
    fn layer_handle_moves_blends_and_hides_a_live_source() {
        let Some((device, context)) = try_device() else {
            return;
        };
        let options = VideoCompositorOptions {
            width: 3,
            height: 1,
            frame_rate: ffmpeg::Rational::new(30, 1),
            background: Color::BLACK,
        };
        let (mut compositor, handle) =
            D3d11VideoCompositor::new("compositor", &device, context.clone(), options)
                .expect("D3d11VideoCompositor::new should succeed");

        let layer = VideoLayer::new(VideoRect::new(0, 0, 1, 1));
        let input = handle.add_source("white", layer).unwrap().unwrap();
        let mut sink = input.sink;
        let layer_handle = input.layer;

        let white_texture = bgra_texture(&device, 1, 1, [255, 255, 255, 255]);
        sink.consume(pooled_video(
            wrap_d3d11_texture(white_texture, 1, 1).unwrap(),
        ))
        .unwrap();

        layer_handle.set_rect(VideoRect::new(1, 0, 1, 1)).unwrap();
        layer_handle.set_opacity(0.5).unwrap();
        let blended = compositor
            .compose_frame(&test_bus())
            .expect("compose_frame failed");
        let downloaded = download_frame(&device, context.clone(), blended);
        assert_eq!(pixel(&downloaded, 0, 0), [0, 0, 0, 255], "background only");
        // 255 * 0.5 = 127.5 — the CPU SwVideoCompositor's software blend
        // rounds this to 128 (`f32::round`), but D3D11's fixed-function
        // blend hardware truncates instead, landing on 127. Both are
        // legitimate roundings of the same exact half-way value; this
        // one-off discrepancy is an inherent CPU-vs-GPU-blend-unit
        // difference, not a bug in either path.
        let blended_pixel = pixel(&downloaded, 1, 0);
        assert_eq!(
            blended_pixel[3], 255,
            "50% white over black: {blended_pixel:?}"
        );
        for channel in &blended_pixel[..3] {
            assert!(
                (127..=128).contains(channel),
                "50% white over black: {blended_pixel:?}"
            );
        }

        layer_handle.set_visible(false).unwrap();
        let hidden = compositor
            .compose_frame(&test_bus())
            .expect("compose_frame failed");
        assert_eq!(hidden.pts(), Some(1));
        let downloaded = download_frame(&device, context, hidden);
        assert_eq!(pixel(&downloaded, 1, 0), [0, 0, 0, 255], "hidden layer");
    }

    #[test]
    fn skips_a_mismatched_device_texture_and_reports_it_on_the_bus() {
        let Some((device_a, context_a)) = try_device() else {
            return;
        };
        let Some((device_b, _context_b)) = try_device() else {
            return;
        };
        let options = VideoCompositorOptions {
            width: 1,
            height: 1,
            frame_rate: ffmpeg::Rational::new(30, 1),
            background: Color::BLACK,
        };
        let (mut compositor, handle) =
            D3d11VideoCompositor::new("compositor", &device_a, context_a.clone(), options)
                .expect("D3d11VideoCompositor::new should succeed");
        let mut sink = handle
            .add_source("mismatched", VideoLayer::new(VideoRect::new(0, 0, 1, 1)))
            .unwrap()
            .unwrap()
            .sink;

        let foreign_texture = bgra_texture(&device_b, 1, 1, [255, 255, 255, 255]);
        sink.consume(pooled_video(
            wrap_d3d11_texture(foreign_texture, 1, 1).unwrap(),
        ))
        .unwrap();

        let (bus, bus_rx) = Bus::new();
        let composed = compositor
            .compose_frame(&bus)
            .expect("a mismatched-device layer must be skipped, not fail the whole frame");

        let error = match bus_rx
            .try_recv()
            .expect("the skipped layer should be reported on the bus")
        {
            BusEvent::Error { error, .. } => error,
            other => panic!("expected a BusEvent::Error, got {other:?}"),
        };
        assert!(matches!(
            error,
            crate::error::Error::D3d11VideoCompositorError(
                D3d11VideoCompositorError::DeviceMismatch
            )
        ));

        let downloaded = download_frame(&device_a, context_a, composed);
        assert_eq!(
            pixel(&downloaded, 0, 0),
            [0, 0, 0, 255],
            "mismatched-device layer must not be drawn — background only"
        );
    }

    struct TimestampSink {
        pp_log: PpLog,
        tx: crossbeam_channel::Sender<Instant>,
    }

    impl Element for TimestampSink {
        fn name(&self) -> Arc<str> {
            "timestamp-recorder".into()
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

    /// Same regression as `SwVideoCompositor`'s
    /// `resuming_after_a_pause_preserves_output_phase` — see that test's
    /// docs for the full rationale. `D3d11VideoCompositor::run` had the
    /// identical bug (never folding `paused_for` back into `next_due`).
    #[test]
    fn resuming_after_a_pause_preserves_output_phase() {
        use crate::pipeline::Pipeline;

        let Some((device, context)) = try_device() else {
            return;
        };
        let (tx, rx) = crossbeam_channel::unbounded();
        let sink = TimestampSink {
            tx,
            pp_log: element_pp_log(ElementType::Other, "timestamp-recorder", None),
        };
        let options = VideoCompositorOptions {
            width: 2,
            height: 2,
            frame_rate: ffmpeg::Rational::new(10, 1),
            background: Color::BLACK,
        };
        let (compositor, _handle) =
            D3d11VideoCompositor::new("compositor", &device, context, options)
                .expect("D3d11VideoCompositor::new should succeed");

        let pipeline = Pipeline::new("phase-test", compositor, |source, ctx| {
            let branch = ctx.branch().to(Box::new(sink))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("test pipeline wiring must succeed");

        pipeline.run().unwrap();
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
