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

use arc_swap::ArcSwapOption;
use ffmpeg_next as ffmpeg;
use rust_hlog::{HLog, hinfo};
use thiserror::Error as ThisError;
use windows::{
    Win32::Foundation::RECT,
    Win32::Graphics::{
        Direct3D::{
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, D3D_SRV_DIMENSION_TEXTURE2DARRAY, Fxc::*,
            ID3DBlob, ID3DInclude,
        },
        Direct3D11::*,
        Dxgi::Common::{
            DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_NV12, DXGI_FORMAT_R8_UNORM,
            DXGI_FORMAT_R8G8_UNORM, DXGI_SAMPLE_DESC,
        },
    },
    core::{Interface, s},
};

use super::video_layer::{self, LayerGeometry, MAX_DIMENSION, VideoLayer, VideoLayerError};
use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    control::{ControlMsg, ControlReceiver, drain_control},
    element::{Element, ElementType, Sink, Source, SourceElement, element_hlog},
    elements::{
        VideoCompositorOptions,
        filter::decoder::d3d11va_decoder::{d3d11va_texture, wrap_d3d11_texture},
    },
    error::Result,
    pad::SrcPad,
    pool::{UnboundObjectPool, UnboundObjectPoolRef},
};

const OUTPUT_POOL_SIZE: usize = 4;
const CONTROL_POLL_INTERVAL: Duration = Duration::from_millis(5);

const BGRA_SHADER_SOURCE: &[u8] = include_bytes!("../../../shaders/d3d11/composite_bgra.hlsl");
const NV12_SHADER_SOURCE: &[u8] = include_bytes!("../../../shaders/d3d11/composite_nv12.hlsl");

/// Errors specific to [`D3d11VideoCompositor`].
#[derive(Debug, ThisError)]
pub enum D3d11VideoCompositorError {
    #[error("windows error: {0}")]
    Windows(#[from] windows::core::Error),

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
        "D3d11VideoCompositorInputSink only accepts Pixel::D3D11 frames, got {0:?}; \
         upload/decode/capture to GPU first"
    )]
    UnsupportedFormat(ffmpeg::format::Pixel),

    #[error(
        "frame claimed the D3D11 pixel format but carries no texture — must come from \
         D3d11Upload/D3d11Decoder/DxgiCaptureSource's GPU mode"
    )]
    InvalidD3d11Frame,

    #[error(
        "D3d11VideoCompositor only draws DXGI_FORMAT_B8G8R8A8_UNORM or DXGI_FORMAT_NV12 input \
         textures, got {0:?}"
    )]
    UnsupportedTextureFormat(DXGI_FORMAT),

    #[error("D3D11 texture array index {index} is outside ArraySize {array_size}")]
    InvalidArrayIndex { index: isize, array_size: u32 },

    #[error(
        "frame dimensions {frame_width}x{frame_height} exceed the backing D3D11 texture's \
         {texture_width}x{texture_height} dimensions"
    )]
    FrameExceedsTexture {
        frame_width: u32,
        frame_height: u32,
        texture_width: u32,
        texture_height: u32,
    },

    #[error(
        "a Pixel::D3D11 frame's texture lives on a different ID3D11Device than this \
         D3d11VideoCompositor was created with — every D3D11 element in one pipeline must share \
         exactly one device for zero-copy to be valid"
    )]
    DeviceMismatch,

    #[error(
        "D3d11VideoCompositorInputSink only accepts decoded Video frames, got a {0}; link it \
         after a decoder or video source"
    )]
    UnsupportedBuffer(&'static str),

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
    /// Same shape as the CPU `VideoCompositor`'s own `VideoInput` field —
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
}

/// A cheaply cloneable handle for adding and removing
/// [`D3d11VideoCompositor`] inputs — the GPU sibling of
/// [`crate::elements::VideoCompositorHandle`], same shape and behavior.
#[derive(Clone)]
pub struct D3d11VideoCompositorHandle {
    shared: Weak<D3d11CompositorShared>,
}

/// The two endpoints created for one compositor input registration.
pub struct D3d11VideoCompositorInput {
    pub sink: Box<dyn Sink>,
    pub layer: D3d11VideoLayerHandle,
}

impl D3d11VideoCompositorHandle {
    /// Registers an input and returns its terminal Sink plus independent
    /// runtime layer control — see
    /// [`crate::elements::VideoCompositorHandle::add_source`]'s own docs
    /// (identical contract: reusing `name` replaces the old registration,
    /// old sinks/layer handles become harmlessly stale).
    pub fn add_source(
        &self,
        name: impl Into<String>,
        layer: VideoLayer,
    ) -> std::result::Result<Option<D3d11VideoCompositorInput>, D3d11VideoCompositorError> {
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

        Ok(Some(D3d11VideoCompositorInput {
            sink: Box::new(D3d11VideoCompositorInputSink {
                name: name.clone(),
                hlog: element_hlog(ElementType::D3d11VideoCompositor, &name, None),
                shared: self.shared.clone(),
                input: Arc::downgrade(&input),
            }),
            layer: D3d11VideoLayerHandle {
                id,
                name,
                input: Arc::downgrade(&input),
            },
        }))
    }

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

/// Thread-safe runtime placement control for one [`D3d11VideoCompositor`]
/// input — the GPU sibling of [`crate::elements::VideoLayerHandle`].
#[derive(Clone)]
pub struct D3d11VideoLayerHandle {
    id: video_layer::VideoInputId,
    name: Arc<str>,
    input: Weak<GpuVideoInput>,
}

impl D3d11VideoLayerHandle {
    pub fn id(&self) -> video_layer::VideoInputId {
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
    ) -> std::result::Result<(), D3d11VideoCompositorError> {
        video_layer::validate_layer(layer).map_err(map_layer_error)?;
        self.update(|current| *current = layer)
    }

    pub fn set_rect(
        &self,
        rect: video_layer::VideoRect,
    ) -> std::result::Result<(), D3d11VideoCompositorError> {
        video_layer::validate_rect(rect).map_err(map_layer_error)?;
        self.update(|layer| layer.rect = rect)
    }

    pub fn set_opacity(&self, opacity: f32) -> std::result::Result<(), D3d11VideoCompositorError> {
        video_layer::validate_opacity(opacity).map_err(map_layer_error)?;
        self.update(|layer| layer.opacity = opacity)
    }

    pub fn set_z_index(&self, z_index: i32) -> std::result::Result<(), D3d11VideoCompositorError> {
        self.update(|layer| layer.z_index = z_index)
    }

    pub fn set_visible(&self, visible: bool) -> std::result::Result<(), D3d11VideoCompositorError> {
        self.update(|layer| layer.visible = visible)
    }

    pub fn set_fit(
        &self,
        fit: video_layer::VideoFit,
    ) -> std::result::Result<(), D3d11VideoCompositorError> {
        self.update(|layer| layer.fit = fit)
    }

    fn update(
        &self,
        update: impl FnOnce(&mut VideoLayer),
    ) -> std::result::Result<(), D3d11VideoCompositorError> {
        let input = self
            .input
            .upgrade()
            .ok_or(D3d11VideoCompositorError::SourceRemoved)?;
        update(&mut input.layer.lock().unwrap());
        Ok(())
    }
}

/// One terminal video input returned by
/// [`D3d11VideoCompositorHandle::add_source`] — the GPU sibling of
/// [`crate::elements::VideoCompositorInputSink`], same behavior (stores
/// only the latest frame; a fast producer can't build an unbounded queue
/// behind a slower compositor output rate).
#[rust_hlog::hlog]
pub struct D3d11VideoCompositorInputSink {
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

    fn hlog(&self) -> &HLog {
        &self.hlog
    }

    fn hlog_mut(&mut self) -> &mut HLog {
        &mut self.hlog
    }
}

impl Sink for D3d11VideoCompositorInputSink {
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
/// sibling of [`crate::elements::VideoCompositor`] (which does the same
/// job on the CPU via `libswscale`). Same [`VideoLayer`]/[`video_layer::VideoRect`]/
/// [`video_layer::VideoFit`] API — a caller's layer-control code doesn't
/// change shape when switching between the two.
///
/// Every input must already be a `Pixel::D3D11` frame (from
/// [`crate::elements::D3d11Upload`], [`crate::elements::D3d11Decoder`], or
/// [`crate::elements::DxgiCaptureSource`]'s GPU mode) on the exact same
/// `ID3D11Device`/shared context this compositor was built with — see
/// [`D3d11VideoCompositor::new`]'s own docs on why `context` specifically
/// must be shared, not just the device.
///
/// Like [`crate::elements::VideoCompositor`], this is a [`SourceElement`],
/// not a conventional one-input filter: upstream pipelines terminate at
/// the sinks returned by [`D3d11VideoCompositorHandle::add_source`], while
/// this element's own pipeline drives output on its independent clock.
#[rust_hlog::hlog]
pub struct D3d11VideoCompositor {
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
        let hlog = element_hlog(ElementType::D3d11VideoCompositor, &name, None);
        let shared = Arc::new(D3d11CompositorShared {
            inputs: Mutex::new(HashMap::new()),
            next_input_id: AtomicU64::new(1),
        });
        let frame_interval = Duration::from_secs_f64(
            options.frame_rate.denominator() as f64 / options.frame_rate.numerator() as f64,
        );

        let (
            vertex_shader,
            bgra_pixel_shader,
            nv12_pixel_shader,
            sampler,
            blend_state,
            rasterizer_state,
            layer_buffer,
        ) = unsafe { build_pipeline_state(device)? };

        hinfo!(
            hlog: &hlog,
            "created: {}x{}, frame_rate={}, format=D3D11(BGRA)",
            options.width,
            options.height,
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
                pad: SrcPad::new(format!("{name}_src")),
            },
            D3d11VideoCompositorHandle {
                shared: Arc::downgrade(&shared),
            },
        ))
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
            let target =
                unsafe { create_output_target(&self.device, canvas_width, canvas_height)? };
            let key = target.texture.as_raw() as usize;
            self.output_views
                .insert(key, target.render_target_view.clone());
            *output_frame = wrap_d3d11_texture(target.texture, canvas_width, canvas_height);
        }
        let (output_raw, _) = d3d11va_texture(&output_frame)
            .expect("output pool frames are initialized immediately after checkout");
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
        let draw_result = unsafe {
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

            let result = snapshots.iter().try_for_each(|snapshot| {
                if !snapshot.layer.visible || snapshot.layer.opacity == 0.0 {
                    return Ok(());
                }
                let Some(frame) = &snapshot.frame else {
                    return Ok(());
                };
                self.draw_layer(
                    &context,
                    frame,
                    &snapshot.layer,
                    canvas_width,
                    canvas_height,
                )
            });

            // Always release resource bindings, including when one layer
            // failed after earlier layers had already drawn.
            context.PSSetShaderResources(0, Some(&[None, None]));
            context.OMSetRenderTargets(None, None);
            result
        };
        drop(context);
        draw_result?;

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
        // Safety: `texture_raw` is a borrowed raw `ID3D11Texture2D*` — see
        // `d3d11va_texture`'s own docs; `.clone()` (`AddRef`) gives an
        // independently ref-counted handle valid for this draw call.
        let texture = unsafe {
            ID3D11Texture2D::from_raw_borrowed(&texture_raw)
                .expect("D3d11 frame's texture pointer must not be null")
                .clone()
        };

        let texture_device =
            unsafe { texture.GetDevice() }.map_err(D3d11VideoCompositorError::from)?;
        if texture_device.as_raw() != self.device.as_raw() {
            return Err(D3d11VideoCompositorError::DeviceMismatch);
        }

        let mut desc = D3D11_TEXTURE2D_DESC::default();
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
        let output = self.compose_frame()?;
        if let Err(error) = self.pad.push(MediaBuffer::Video(Arc::new(output))) {
            bus.post(
                &self.hlog,
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

    fn hlog(&self) -> &HLog {
        &self.hlog
    }

    fn hlog_mut(&mut self) -> &mut HLog {
        &mut self.hlog
    }
}

impl Source for D3d11VideoCompositor {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl SourceElement for D3d11VideoCompositor {
    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        hinfo!(self, "run: starting");
        let mut next_due = Instant::now();
        loop {
            if drain_control(control, self, bus)?.stopped {
                hinfo!(self, "run: stopped");
                return Ok(());
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
                next_due = now + self.frame_interval;
            }
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
/// of the CPU `VideoCompositor`'s `blend_bgra` clipping math, computed
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
    unsafe {
        let vertex_bytecode = compile_shader(BGRA_SHADER_SOURCE, s!("vs_main"), s!("vs_5_0"))?;
        let bgra_bytecode = compile_shader(BGRA_SHADER_SOURCE, s!("ps_bgra"), s!("ps_5_0"))?;
        let nv12_bytecode = compile_shader(NV12_SHADER_SOURCE, s!("ps_nv12"), s!("ps_5_0"))?;

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
        // always opaque" contract the CPU `VideoCompositor`'s
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

unsafe fn compile_shader(
    source: &[u8],
    entry: windows::core::PCSTR,
    target: windows::core::PCSTR,
) -> windows::core::Result<ID3DBlob> {
    let mut shader = None;
    let mut errors = None;
    let flags = if cfg!(debug_assertions) {
        D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION
    } else {
        D3DCOMPILE_OPTIMIZATION_LEVEL3
    };
    let result = unsafe {
        D3DCompile(
            source.as_ptr().cast::<c_void>(),
            source.len(),
            s!("d3d11_video_compositor.hlsl"),
            None,
            None::<&ID3DInclude>,
            entry,
            target,
            flags,
            0,
            &mut shader,
            Some(&mut errors),
        )
    };
    if let Err(error) = result {
        let message = errors
            .map(|blob| unsafe {
                let bytes = std::slice::from_raw_parts(
                    blob.GetBufferPointer().cast::<u8>(),
                    blob.GetBufferSize(),
                );
                String::from_utf8_lossy(bytes).into_owned()
            })
            .unwrap_or_else(|| error.message());
        return Err(windows::core::Error::new(
            windows::Win32::Foundation::E_FAIL,
            message,
        ));
    }
    Ok(shader.unwrap())
}

unsafe fn create_output_target(
    device: &ID3D11Device,
    width: u32,
    height: u32,
) -> windows::core::Result<OutputTarget> {
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

    use windows::Win32::Graphics::Direct3D::D3D_DRIVER_TYPE_HARDWARE;

    use super::*;
    use crate::elements::{D3d11Download, VideoRect};

    #[rust_hlog::hlog]
    struct CapturingSink {
        received: Arc<Mutex<Vec<MediaBuffer>>>,
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

    /// `None` if this machine has no working D3D11 hardware device —
    /// every test here skips gracefully in that case, same convention as
    /// `D3d11Upload`/`D3d11Decoder`/`D3d11Download`'s own hardware tests.
    fn try_device() -> Option<(ID3D11Device, Arc<Mutex<ID3D11DeviceContext>>)> {
        let mut device = None;
        let mut context = None;
        let result = unsafe {
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                Default::default(),
                Default::default(),
                None,
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut context),
            )
        };
        if result.is_err() {
            eprintln!("skipping: D3D11CreateDevice failed on this machine: {result:?}");
            return None;
        }
        Some((
            device.expect("D3D11CreateDevice succeeded without producing a device"),
            Arc::new(Mutex::new(context.expect(
                "D3D11CreateDevice succeeded without producing a context",
            ))),
        ))
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
        for pair in pixels[luma_size..].chunks_exact_mut(2) {
            pair.copy_from_slice(&[cb, cr]);
        }
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
            hlog: element_hlog(ElementType::Other, "capture", None),
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
    fn composes_gpu_inputs_in_z_order_and_preserves_output_contract() {
        let Some((device, context)) = try_device() else {
            return;
        };
        let options = VideoCompositorOptions {
            width: 4,
            height: 4,
            frame_rate: ffmpeg::Rational::new(30, 1),
            background: video_layer::VideoColor::BLACK,
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
            .consume(pooled_video(wrap_d3d11_texture(red_texture, 4, 4)))
            .unwrap();
        blue_sink
            .consume(pooled_video(wrap_d3d11_texture(blue_texture, 2, 2)))
            .unwrap();

        let composed = compositor.compose_frame().expect("compose_frame failed");
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
            background: video_layer::VideoColor::BLACK,
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
        sink.consume(pooled_video(wrap_d3d11_texture(texture, 4, 3)))
            .unwrap();

        let composed = compositor.compose_frame().expect("compose_frame failed");
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
            background: video_layer::VideoColor::BLACK,
        };
        let (mut compositor, handle) =
            D3d11VideoCompositor::new("compositor", &device, context.clone(), options)
                .expect("D3d11VideoCompositor::new should succeed");
        let mut layer = VideoLayer::new(VideoRect::new(0, 0, 1, 1));
        layer.fit = video_layer::VideoFit::Stretch;
        let mut sink = handle.add_source("input", layer).unwrap().unwrap().sink;

        sink.consume(pooled_video(wrap_d3d11_texture(
            bgra_texture(&device, 1, 1, [0, 0, 255, 255]),
            1,
            1,
        )))
        .unwrap();
        let first = compositor.compose_frame().expect("first compose failed");

        sink.consume(pooled_video(wrap_d3d11_texture(
            bgra_texture(&device, 1, 1, [255, 0, 0, 255]),
            1,
            1,
        )))
        .unwrap();
        let mut later = Vec::new();
        for _ in 0..OUTPUT_POOL_SIZE {
            later.push(compositor.compose_frame().expect("later compose failed"));
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
            background: video_layer::VideoColor::BLACK,
        };
        let (mut compositor, handle) =
            D3d11VideoCompositor::new("compositor", &device, context.clone(), options)
                .expect("D3d11VideoCompositor::new should succeed");
        let mut layer = VideoLayer::new(VideoRect::new(0, 0, 2, 2));
        layer.fit = video_layer::VideoFit::Stretch;
        let mut sink = handle.add_source("input", layer).unwrap().unwrap().sink;
        let texture = nv12_texture(&device, 2, 2, 81, 90, 240);

        let mut bt601 = wrap_d3d11_texture(texture.clone(), 2, 2);
        bt601.set_color_space(ffmpeg::color::Space::SMPTE170M);
        bt601.set_color_range(ffmpeg::color::Range::MPEG);
        sink.consume(pooled_video(bt601)).unwrap();
        let bt601 = compositor.compose_frame().expect("BT.601 compose failed");
        let bt601 = download_frame(&device, context.clone(), bt601);

        let mut bt709 = wrap_d3d11_texture(texture, 2, 2);
        bt709.set_color_space(ffmpeg::color::Space::BT709);
        bt709.set_color_range(ffmpeg::color::Range::MPEG);
        sink.consume(pooled_video(bt709)).unwrap();
        let bt709 = compositor.compose_frame().expect("BT.709 compose failed");
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
            background: video_layer::VideoColor::BLACK,
        };
        let (mut compositor, handle) =
            D3d11VideoCompositor::new("compositor", &device, context.clone(), options)
                .expect("D3d11VideoCompositor::new should succeed");

        let layer = VideoLayer::new(VideoRect::new(0, 0, 1, 1));
        let input = handle.add_source("white", layer).unwrap().unwrap();
        let mut sink = input.sink;
        let layer_handle = input.layer;

        let white_texture = bgra_texture(&device, 1, 1, [255, 255, 255, 255]);
        sink.consume(pooled_video(wrap_d3d11_texture(white_texture, 1, 1)))
            .unwrap();

        layer_handle.set_rect(VideoRect::new(1, 0, 1, 1)).unwrap();
        layer_handle.set_opacity(0.5).unwrap();
        let blended = compositor.compose_frame().expect("compose_frame failed");
        let downloaded = download_frame(&device, context.clone(), blended);
        assert_eq!(pixel(&downloaded, 0, 0), [0, 0, 0, 255], "background only");
        // 255 * 0.5 = 127.5 — the CPU VideoCompositor's software blend
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
        let hidden = compositor.compose_frame().expect("compose_frame failed");
        assert_eq!(hidden.pts(), Some(1));
        let downloaded = download_frame(&device, context, hidden);
        assert_eq!(pixel(&downloaded, 1, 0), [0, 0, 0, 255], "hidden layer");
    }

    #[test]
    fn rejects_a_texture_from_a_different_device() {
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
            background: video_layer::VideoColor::BLACK,
        };
        let (mut compositor, handle) =
            D3d11VideoCompositor::new("compositor", &device_a, context_a, options)
                .expect("D3d11VideoCompositor::new should succeed");
        let mut sink = handle
            .add_source("mismatched", VideoLayer::new(VideoRect::new(0, 0, 1, 1)))
            .unwrap()
            .unwrap()
            .sink;

        let foreign_texture = bgra_texture(&device_b, 1, 1, [255, 255, 255, 255]);
        sink.consume(pooled_video(wrap_d3d11_texture(foreign_texture, 1, 1)))
            .unwrap();

        let error = match compositor.compose_frame() {
            Ok(_) => panic!("expected a device-mismatch error"),
            Err(error) => error,
        };
        assert!(matches!(error, D3d11VideoCompositorError::DeviceMismatch));
    }
}
