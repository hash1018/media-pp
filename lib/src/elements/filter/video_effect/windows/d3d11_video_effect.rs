use std::{
    ffi::c_void,
    sync::{Arc, Mutex},
};

use crate::pp_log::{PpLog, pp_debug, pp_error, pp_info};
use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;
use windows::{
    Win32::Graphics::{
        Direct3D::{D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, D3D_SRV_DIMENSION_TEXTURE2DARRAY},
        Direct3D11::*,
        Dxgi::Common::{DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_SAMPLE_DESC},
    },
    core::{Interface, s},
};

use super::super::handle::{VideoEffectControl, VideoEffectHandle};
use super::super::options::{EffectParams, VideoEffect};
use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    error::{D3d11SharedDeviceError, Result},
    pad::SrcPad,
    platform::windows::d3d11::{compile_shader, protect_shared_device},
    platform::windows::d3d11va::{d3d11va_texture, wrap_d3d11_texture},
    pool::{UnboundObjectPool, UnboundObjectPoolRef},
    repeat::{PerFrameTransform, RepeatedOutput},
};

const SHADER_SOURCE: &[u8] = include_bytes!("../../../../shaders/d3d11/video_effect_bgra.hlsl");

/// Errors specific to [`D3d11VideoEffect`]. Converts into the crate-wide
/// `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum D3d11VideoEffectError {
    /// A Direct3D resource, shader, or draw operation failed.
    #[error("windows error: {0}")]
    Windows(#[from] windows::core::Error),

    /// The device cannot be shared across a pipeline's threads.
    #[error(transparent)]
    SharedDevice(#[from] D3d11SharedDeviceError),

    /// FFmpeg could not take a second reference to the frame already in
    /// hand, which is how an unchanged input texture is answered.
    #[error("failed to reference the previous frame (code {0})")]
    FrameRef(i32),

    /// The input frame is not backed by a D3D11 texture.
    #[error("D3d11VideoEffect only takes Pixel::D3D11 frames, got {0:?}")]
    UnsupportedFormat(ffmpeg::format::Pixel),

    /// A frame tagged as D3D11 carries no valid texture.
    #[error(
        "frame claimed the D3D11 pixel format but carries no texture — must \
         come from D3d11Upload/D3d11Decoder/DxgiCaptureSource's GPU mode/D3d11VideoCompositor"
    )]
    InvalidD3d11Frame,

    /// The texture is not BGRA.
    #[error(
        "D3d11VideoEffect only takes DXGI_FORMAT_B8G8R8A8_UNORM textures, got {0:?}; \
         convert an NV12 surface with D3d11Scaler and D3d11ScalerFormat::Bgra first"
    )]
    UnsupportedTextureFormat(DXGI_FORMAT),

    /// The input texture belongs to another D3D11 device.
    #[error(
        "a Pixel::D3D11 frame's texture lives on a different ID3D11Device \
         than this D3d11VideoEffect was created with — every D3D11 element in one \
         pipeline must share exactly one device for zero-copy to be valid"
    )]
    DeviceMismatch,

    /// The supplied immediate context belongs to another D3D11 device.
    #[error(
        "the supplied ID3D11DeviceContext belongs to a different ID3D11Device than this \
         D3d11VideoEffect"
    )]
    ContextDeviceMismatch,

    /// The backing texture is smaller than the visible input frame.
    #[error(
        "D3D11 texture is {actual_width}x{actual_height}, smaller than the \
         frame's {expected_width}x{expected_height} visible size"
    )]
    TextureTooSmall {
        /// Backing texture width in pixels.
        actual_width: u32,
        /// Backing texture height in pixels.
        actual_height: u32,
        /// Visible frame width in pixels.
        expected_width: u32,
        /// Visible frame height in pixels.
        expected_height: u32,
    },

    /// The frame selects a texture-array slice outside the resource bounds.
    #[error("D3D11 texture array index {index} is outside ArraySize {array_size}")]
    InvalidArrayIndex {
        /// Invalid texture-array index.
        index: isize,
        /// Number of slices in the texture array.
        array_size: u32,
    },

    /// The texture is multisampled and cannot be sampled by this path.
    #[error("D3d11VideoEffect does not accept multisampled textures (SampleDesc.Count={0})")]
    MultisampledTexture(u32),

    /// The texture was not created with shader-resource binding.
    #[error(
        "the input texture was created without D3D11_BIND_SHADER_RESOURCE (BindFlags={0:#x}), \
         so this element's pixel shader cannot read it"
    )]
    MissingShaderResourceBind(u32),

    /// The visible input dimensions are zero.
    #[error("frame has invalid dimensions {width}x{height}")]
    InvalidFrameDimensions {
        /// Invalid frame width in pixels.
        width: u32,
        /// Invalid frame height in pixels.
        height: u32,
    },

    /// The sink received a buffer other than decoded video or end-of-stream.
    #[error("D3d11VideoEffect only accepts Video and Eos buffers, got a {0}")]
    UnsupportedBuffer(&'static str),
}

/// The per-draw constant buffer `video_effect_bgra.hlsl` reads, in the
/// order and padding HLSL's `cbuffer` packing expects — see the shader's
/// `EffectBuffer`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct EffectConstants {
    rows: [[f32; 4]; 3],
    exponent: f32,
    opacity: f32,
    uv_scale: [f32; 2],
    luma: [f32; 4],
}

impl EffectConstants {
    fn new(params: &EffectParams, uv_scale: [f32; 2]) -> Self {
        Self {
            rows: params.rows,
            exponent: params.exponent,
            opacity: params.opacity,
            uv_scale,
            luma: [
                params.luma_low,
                params.luma_low_inv,
                params.luma_high,
                params.luma_high_inv,
            ],
        }
    }
}

/// Applies a [`VideoEffect`] — a colour correction or a luma key — to a
/// GPU-resident `Pixel::D3D11` BGRA frame through a pixel shader, without
/// the frame leaving video memory. The D3D11 member of the family whose
/// software member is [`crate::elements::SwVideoEffect`]; both evaluate the
/// same resolved numbers per pixel.
///
/// Everything else is `D3d11ChromaKey`'s, and for its reasons: BGRA in and
/// BGRA out (a decoded frame reaches it through
/// [`crate::elements::D3d11Scaler`] with
/// [`crate::elements::D3d11ScalerFormat::Bgra`]); a fresh output texture the
/// visible size of the input, since a published frame is never changed in
/// place; `device` and `context` the ones every other D3D11 element in the
/// pipeline shares; state re-selected on every draw; and a `Flush` after
/// each, which is what keeps the three objects a frame creates from piling
/// up on the device.
///
/// An effect that changes nothing, and an element that is turned off, hand
/// each frame straight through — the same texture, not a copy of it.
pub struct D3d11VideoEffect {
    pp_log: PpLog,
    name: Arc<str>,
    device: ID3D11Device,
    context: Arc<Mutex<ID3D11DeviceContext>>,
    /// The effect in force and what it resolves to, refreshed from `control`
    /// once per frame.
    effect: VideoEffect,
    params: EffectParams,
    control: Arc<VideoEffectControl>,
    enabled: bool,

    vertex_shader: ID3D11VertexShader,
    pixel_shader: ID3D11PixelShader,
    sampler: ID3D11SamplerState,
    blend_state: ID3D11BlendState,
    rasterizer_state: ID3D11RasterizerState,
    constant_buffer: ID3D11Buffer,
    pad: SrcPad,
    /// Only the CPU-side `AVFrame` wrapper is reused; each output texture is
    /// new, since downstream may still hold the last one.
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
    /// The last output and the texture it was made from — see
    /// [`RepeatedOutput`]. Cleared when the effect changes.
    repeated: RepeatedOutput,
}

// SAFETY: every field is either a `windows-rs` COM interface wrapper (the
// device-level ones free-threaded, the context-level ones behind `context`'s
// own `Mutex`) or plain data, and `&mut self` on every method that touches the
// rest rules out concurrent access — the reasoning `D3d11ChromaKey` gives.
unsafe impl Send for D3d11VideoEffect {}

/// One validated input: the texture to sample, which slice of it, the
/// visible size to draw, and the fraction of the texture that size covers.
struct ValidatedInput {
    texture: ID3D11Texture2D,
    array_slice: u32,
    width: u32,
    height: u32,
    uv_scale: [f32; 2],
}

impl D3d11VideoEffect {
    /// `device` must be the same `ID3D11Device`, and `context` the same
    /// shared immediate context, every other D3D11 element in this pipeline
    /// uses.
    ///
    /// Output dimensions are not a parameter: an effect is per pixel, so
    /// every frame comes out the size it went in.
    pub fn new(
        name: impl Into<String>,
        device: &ID3D11Device,
        context: Arc<Mutex<ID3D11DeviceContext>>,
        effect: VideoEffect,
    ) -> std::result::Result<(Self, VideoEffectHandle), D3d11VideoEffectError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::D3d11VideoEffect, &name, None);
        // Usually behind a `Queue`, so the device has to be usable from
        // another thread before any command is issued.
        protect_shared_device(device)?;
        {
            let context = context
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            // SAFETY: `context` is a live immediate-context interface;
            // `GetDevice` returns an owned reference to its creating device.
            let context_device = unsafe { context.GetDevice() }?;
            if context_device.as_raw() != device.as_raw() {
                return Err(D3d11VideoEffectError::ContextDeviceMismatch);
            }
        }

        // SAFETY: `device` is live; the helper creates state only from static
        // shader bytes and fully initialized descriptors, returning owned COM
        // references without retaining borrowed pointers.
        let (vertex_shader, pixel_shader, sampler, blend_state, rasterizer_state, constant_buffer) =
            unsafe { build_pipeline_state(device) }?;

        let pad = SrcPad::with_contract(
            format!("{name}_src"),
            OutputContract::Fixed(PortContract::frame(
                MediaKind::VideoFrame,
                MemoryDomain::D3d11,
            )),
        );
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        pp_info!(pp_log: &pp_log, "created: {} {effect:?}", effect.name());
        let control = Arc::new(VideoEffectControl::new(effect));
        let handle = VideoEffectHandle::new(control.clone());
        let element = Self {
            name,
            pp_log,
            device: device.clone(),
            context,
            effect,
            params: effect.params(),
            control,
            enabled: true,
            vertex_shader,
            pixel_shader,
            sampler,
            blend_state,
            rasterizer_state,
            constant_buffer,
            pad,
            pool,
            repeated: RepeatedOutput::new(),
        };
        Ok((element, handle))
    }

    /// Picks up whatever the handle has been set to, once per frame.
    fn refresh(&mut self) {
        let effect = self.control.get();
        if effect != self.effect {
            pp_debug!(self, "retuned: {} {effect:?}", effect.name());
            self.effect = effect;
            self.params = effect.params();
            self.repeated.clear();
        }
        let enabled = self.control.enabled();
        if enabled != self.enabled {
            pp_debug!(self, "effect {}", if enabled { "on" } else { "off" });
            self.enabled = enabled;
            self.repeated.clear();
        }
    }

    /// Rejects anything this element cannot draw from, so a bad frame fails
    /// here naming the problem rather than as a `Draw` sampling nothing.
    fn validate(
        &self,
        frame: &ffmpeg::frame::Video,
    ) -> std::result::Result<ValidatedInput, D3d11VideoEffectError> {
        if frame.format() != ffmpeg::format::Pixel::D3D11 {
            return Err(D3d11VideoEffectError::UnsupportedFormat(frame.format()));
        }
        if frame.width() == 0 || frame.height() == 0 {
            return Err(D3d11VideoEffectError::InvalidFrameDimensions {
                width: frame.width(),
                height: frame.height(),
            });
        }
        let (texture_raw, index) =
            d3d11va_texture(frame).ok_or(D3d11VideoEffectError::InvalidD3d11Frame)?;
        if texture_raw.is_null() {
            return Err(D3d11VideoEffectError::InvalidD3d11Frame);
        }
        // SAFETY: `texture_raw` is a borrowed raw `ID3D11Texture2D*` still
        // owned by `frame`'s buffer; `.clone()` (`AddRef`) gives an
        // independently ref-counted handle.
        let texture = unsafe {
            ID3D11Texture2D::from_raw_borrowed(&texture_raw)
                .ok_or(D3d11VideoEffectError::InvalidD3d11Frame)?
                .clone()
        };

        // SAFETY: `texture` is a live cloned COM interface; `GetDevice`
        // returns an owned reference to its creating device.
        let texture_device = unsafe { texture.GetDevice() }?;
        if texture_device.as_raw() != self.device.as_raw() {
            return Err(D3d11VideoEffectError::DeviceMismatch);
        }

        let mut desc = D3D11_TEXTURE2D_DESC::default();
        // SAFETY: `desc` is a live out-parameter for the live texture.
        unsafe { texture.GetDesc(&mut desc) };
        if desc.Format != DXGI_FORMAT_B8G8R8A8_UNORM {
            return Err(D3d11VideoEffectError::UnsupportedTextureFormat(desc.Format));
        }
        if desc.Width < frame.width() || desc.Height < frame.height() {
            return Err(D3d11VideoEffectError::TextureTooSmall {
                actual_width: desc.Width,
                actual_height: desc.Height,
                expected_width: frame.width(),
                expected_height: frame.height(),
            });
        }
        if index < 0 || index as u64 >= u64::from(desc.ArraySize) {
            return Err(D3d11VideoEffectError::InvalidArrayIndex {
                index,
                array_size: desc.ArraySize,
            });
        }
        if desc.SampleDesc.Count != 1 {
            return Err(D3d11VideoEffectError::MultisampledTexture(
                desc.SampleDesc.Count,
            ));
        }
        if desc.BindFlags & D3D11_BIND_SHADER_RESOURCE.0 as u32 == 0 {
            return Err(D3d11VideoEffectError::MissingShaderResourceBind(
                desc.BindFlags,
            ));
        }

        Ok(ValidatedInput {
            texture,
            array_slice: index as u32,
            // The visible size, not the texture's: a decoder pads its
            // surfaces, and the padding is not part of the picture.
            width: frame.width(),
            height: frame.height(),
            uv_scale: [
                frame.width() as f32 / desc.Width as f32,
                frame.height() as f32 / desc.Height as f32,
            ],
        })
    }

    fn draw(
        &mut self,
        frame: &ffmpeg::frame::Video,
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        let input = self
            .validate(frame)
            .inspect_err(|error| pp_error!(self, "{error}"))?;

        let output = create_output_texture(&self.device, input.width, input.height)
            .inspect_err(|error| pp_error!(self, "failed to allocate the output texture: {error}"))
            .map_err(D3d11VideoEffectError::from)?;
        let mut output_view = None;
        // SAFETY: `output` is a live render-target-capable texture and
        // `output_view` is the correctly typed live out-parameter.
        unsafe {
            self.device
                .CreateRenderTargetView(&output, None, Some(&mut output_view))
                .inspect_err(|error| pp_error!(self, "failed to create the output RTV: {error}"))
                .map_err(D3d11VideoEffectError::from)?;
        }
        let output_view = output_view.expect("CreateRenderTargetView succeeded without a view");

        let srv_desc = D3D11_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            ViewDimension: D3D_SRV_DIMENSION_TEXTURE2DARRAY,
            Anonymous: D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2DArray: D3D11_TEX2D_ARRAY_SRV {
                    MostDetailedMip: 0,
                    MipLevels: 1,
                    FirstArraySlice: input.array_slice,
                    ArraySize: 1,
                },
            },
        };
        let mut srv = None;
        // SAFETY: validation established the texture format and bounded the
        // array slice; `srv_desc` selects exactly that slice and `srv` is a
        // live out-parameter.
        unsafe {
            self.device
                .CreateShaderResourceView(&input.texture, Some(&srv_desc), Some(&mut srv))
                .inspect_err(|error| pp_error!(self, "failed to create the input SRV: {error}"))
                .map_err(D3d11VideoEffectError::from)?;
        }

        let constants = EffectConstants::new(&self.params, input.uv_scale);
        let viewport = D3D11_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: input.width as f32,
            Height: input.height as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };

        {
            let context = self
                .context
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            // SAFETY: all state objects and views are live and belong to the
            // context's device; the constants pointer is readable for the
            // buffer's declared size. Bindings are cleared before references
            // drop, and the immediate context is serialized by its mutex.
            unsafe {
                context.UpdateSubresource(
                    &self.constant_buffer,
                    0,
                    None,
                    (&raw const constants).cast::<c_void>(),
                    0,
                    0,
                );
                // Every piece of state re-selected: the context is shared,
                // and whatever drew last left its own bound.
                context.OMSetRenderTargets(Some(&[Some(output_view)]), None);
                context.OMSetBlendState(&self.blend_state, None, 0xffff_ffff);
                context.RSSetState(&self.rasterizer_state);
                context.RSSetViewports(Some(&[viewport]));
                context.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                context.IASetInputLayout(None);
                context.VSSetShader(&self.vertex_shader, None);
                context.PSSetShader(&self.pixel_shader, None);
                context.PSSetSamplers(0, Some(&[Some(self.sampler.clone())]));
                context.PSSetConstantBuffers(0, Some(&[Some(self.constant_buffer.clone())]));
                context.PSSetShaderResources(0, Some(&[srv]));
                context.Draw(3, 0);

                // Released on the way out, always, so this frame's views do
                // not live on inside the shared context's state.
                context.PSSetShaderResources(0, Some(&[None]));
                context.OMSetRenderTargets(None, None);
                // D3D11 destroys a released object only once the context is
                // flushed, and this creates three per frame — see
                // `D3d11ChromaKey::key`, which measured it.
                context.Flush();
            }
        }

        let mut drawn = self.pool.get();
        *drawn = wrap_d3d11_texture(output, input.width, input.height)?;
        drawn.set_pts(frame.pts());
        drawn.set_color_space(frame.color_space());
        drawn.set_color_range(frame.color_range());
        Ok(drawn)
    }
}

impl PerFrameTransform for D3d11VideoEffect {
    fn repeated(&mut self) -> &mut RepeatedOutput {
        &mut self.repeated
    }

    fn frame_ref_failed(&self, code: i32) -> crate::error::Error {
        pp_error!(self, "av_frame_ref failed: {code}");
        D3d11VideoEffectError::FrameRef(code).into()
    }

    fn produce(
        &mut self,
        frame: &Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        self.draw(frame)
    }
}

impl Element for D3d11VideoEffect {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::D3d11VideoEffect
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for D3d11VideoEffect {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for D3d11VideoEffect {
    /// Drawn on the GPU; a system-memory frame belongs in SwVideoEffect.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::frame(
            MediaKind::VideoFrame,
            MemoryDomain::D3d11,
        ))
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Video(frame) => {
                self.refresh();
                if !self.enabled || self.params.is_identity() {
                    // Straight through: the same picture, not a copy of it.
                    return self.pad.push(MediaBuffer::Video(frame));
                }
                let drawn = self.transform(&frame)?;
                self.pad.push(MediaBuffer::Video(drawn))
            }
            // One `Draw` per frame, pushed before `consume` returns, so
            // there is nothing to drain.
            MediaBuffer::Eos => self.pad.push(MediaBuffer::Eos),
            other => {
                let kind = other.kind();
                pp_error!(self, "unsupported buffer: {kind}");
                Err(D3d11VideoEffectError::UnsupportedBuffer(kind).into())
            }
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        if matches!(msg, ControlMsg::Flush | ControlMsg::Stop) {
            self.repeated.clear();
        }
        self.pad.control(msg)
    }
}

/// Every piece of D3D11 state this element re-selects per draw, built once.
#[allow(clippy::type_complexity)]
unsafe fn build_pipeline_state(
    device: &ID3D11Device,
) -> windows::core::Result<(
    ID3D11VertexShader,
    ID3D11PixelShader,
    ID3D11SamplerState,
    ID3D11BlendState,
    ID3D11RasterizerState,
    ID3D11Buffer,
)> {
    // SAFETY: the device is live; compiler blobs retain their bytecode while
    // shader creation reads it, every descriptor is fully initialized, and
    // each optional interface slot is a live out-parameter. No call retains a
    // borrowed Rust pointer after returning.
    unsafe {
        let vertex_bytecode = compile_shader(
            SHADER_SOURCE,
            s!("video_effect_bgra.hlsl"),
            s!("vs_main"),
            s!("vs_5_0"),
        )?;
        let pixel_bytecode = compile_shader(
            SHADER_SOURCE,
            s!("video_effect_bgra.hlsl"),
            s!("ps_effect"),
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
        let vertex_shader = vertex_shader.expect("CreateVertexShader succeeded without a shader");

        let mut pixel_shader = None;
        device.CreatePixelShader(
            std::slice::from_raw_parts(
                pixel_bytecode.GetBufferPointer().cast::<u8>(),
                pixel_bytecode.GetBufferSize(),
            ),
            None,
            Some(&mut pixel_shader),
        )?;
        let pixel_shader = pixel_shader.expect("CreatePixelShader succeeded without a shader");

        // Point sampling: input and output are the same size, so every
        // output pixel is exactly one input texel.
        let sampler_desc = D3D11_SAMPLER_DESC {
            Filter: D3D11_FILTER_MIN_MAG_MIP_POINT,
            AddressU: D3D11_TEXTURE_ADDRESS_CLAMP,
            AddressV: D3D11_TEXTURE_ADDRESS_CLAMP,
            AddressW: D3D11_TEXTURE_ADDRESS_CLAMP,
            ComparisonFunc: D3D11_COMPARISON_NEVER,
            MaxLOD: f32::MAX,
            ..Default::default()
        };
        let mut sampler = None;
        device.CreateSamplerState(&sampler_desc, Some(&mut sampler))?;
        let sampler = sampler.expect("CreateSamplerState succeeded without a state");

        // Blending off: the alpha this computes is the output, not something
        // to blend against the render target.
        let mut blend_desc = D3D11_BLEND_DESC::default();
        blend_desc.RenderTarget[0] = D3D11_RENDER_TARGET_BLEND_DESC {
            BlendEnable: false.into(),
            RenderTargetWriteMask: D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8,
            ..Default::default()
        };
        let mut blend_state = None;
        device.CreateBlendState(&blend_desc, Some(&mut blend_state))?;
        let blend_state = blend_state.expect("CreateBlendState succeeded without a state");

        // Scissoring off: this draw covers the whole target, and the shared
        // context may carry someone else's scissor rect.
        let rasterizer_desc = D3D11_RASTERIZER_DESC {
            FillMode: D3D11_FILL_SOLID,
            CullMode: D3D11_CULL_NONE,
            ScissorEnable: false.into(),
            DepthClipEnable: true.into(),
            ..Default::default()
        };
        let mut rasterizer_state = None;
        device.CreateRasterizerState(&rasterizer_desc, Some(&mut rasterizer_state))?;
        let rasterizer_state =
            rasterizer_state.expect("CreateRasterizerState succeeded without a state");

        let buffer_desc = D3D11_BUFFER_DESC {
            ByteWidth: std::mem::size_of::<EffectConstants>() as u32,
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
            StructureByteStride: 0,
        };
        let mut constant_buffer = None;
        device.CreateBuffer(&buffer_desc, None, Some(&mut constant_buffer))?;
        let constant_buffer = constant_buffer.expect("CreateBuffer succeeded without a buffer");

        Ok((
            vertex_shader,
            pixel_shader,
            sampler,
            blend_state,
            rasterizer_state,
            constant_buffer,
        ))
    }
}

/// Render target and shader resource both: drawn into here, then sampled by
/// whatever comes next.
fn create_output_texture(
    device: &ID3D11Device,
    width: u32,
    height: u32,
) -> std::result::Result<ID3D11Texture2D, windows::core::Error> {
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
    // SAFETY: `desc` fully describes a render-target texture, no initial data
    // is supplied, and `texture` is a live out-parameter.
    unsafe {
        device.CreateTexture2D(&desc, None, Some(&mut texture))?;
    }
    Ok(texture.expect("CreateTexture2D succeeded without producing a texture"))
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use windows::Win32::Graphics::{
        Direct3D11::{
            D3D11_CREATE_DEVICE_DEBUG, D3D11_RLDO_DETAIL, D3D11_SDK_VERSION, D3D11CreateDevice,
            ID3D11Debug, ID3D11InfoQueue,
        },
        Dxgi::Common::DXGI_FORMAT_NV12,
    };

    use super::super::super::options::apply;
    use super::*;
    use crate::elements::{ColorCorrection, D3d11Download, LumaKey};
    use crate::test_support::try_d3d11_device as try_device;

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

    fn capture(element: &mut dyn Source) -> Arc<Mutex<Vec<MediaBuffer>>> {
        let received = Arc::new(Mutex::new(Vec::new()));
        element.src_pads()[0].link(Box::new(CapturingSink {
            received: received.clone(),
            pp_log: element_pp_log(ElementType::Other, "capture", None),
        }));
        received
    }

    /// A BGRA texture whose every texel of the last slice is `pixel`, and of
    /// any earlier slice opaque white.
    fn bgra_texture(
        device: &ID3D11Device,
        width: u32,
        height: u32,
        pixel: [u8; 4],
        slices: u32,
    ) -> ID3D11Texture2D {
        let planes: Vec<Vec<u8>> = (0..slices)
            .map(|slice| {
                let texel = if slice == slices - 1 {
                    pixel
                } else {
                    [255, 255, 255, 255]
                };
                texel.repeat((width * height) as usize)
            })
            .collect();
        let initial: Vec<D3D11_SUBRESOURCE_DATA> = planes
            .iter()
            .map(|plane| D3D11_SUBRESOURCE_DATA {
                pSysMem: plane.as_ptr().cast::<c_void>(),
                SysMemPitch: width * 4,
                SysMemSlicePitch: 0,
            })
            .collect();
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: slices,
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
        let mut texture = None;
        // SAFETY: `initial` has one entry per declared array slice and every
        // pointer addresses a live, correctly pitched plane through the call.
        unsafe {
            device
                .CreateTexture2D(&desc, Some(initial.as_ptr()), Some(&mut texture))
                .expect("CreateTexture2D(BGRA) failed");
        }
        texture.expect("CreateTexture2D succeeded without producing a texture")
    }

    fn frame(texture: ID3D11Texture2D, width: u32, height: u32, pts: i64) -> MediaBuffer {
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut slot = pool.get();
        *slot = wrap_d3d11_texture(texture, width, height).unwrap();
        slot.set_pts(Some(pts));
        MediaBuffer::Video(Arc::new(slot))
    }

    fn texture_of(buffer: &MediaBuffer) -> *mut c_void {
        let MediaBuffer::Video(frame) = buffer else {
            panic!("expected a Video buffer");
        };
        d3d11va_texture(frame).expect("a D3D11 frame").0
    }

    /// Draws one flat frame and hands back the downloaded result's first
    /// pixel, so a test asserts on pixels rather than on a `Draw` returning.
    fn drawn_pixel(
        device: &ID3D11Device,
        context: &Arc<Mutex<ID3D11DeviceContext>>,
        effect: VideoEffect,
        pixel: [u8; 4],
    ) -> [u8; 4] {
        let (mut element, _) = D3d11VideoEffect::new("effect", device, context.clone(), effect)
            .expect("D3d11VideoEffect::new");
        let mut download = D3d11Download::new("download", device, context.clone(), 8, 8)
            .expect("D3d11Download::new");
        let received = capture(&mut download);
        element.src_pads()[0].link(Box::new(download));

        element
            .consume(frame(bgra_texture(device, 8, 8, pixel, 1), 8, 8, 7))
            .expect("draw");

        let received = received.lock().unwrap();
        let MediaBuffer::Video(drawn) = &received[0] else {
            panic!("expected a Video buffer, got {}", received[0].kind());
        };
        drawn.data(0)[0..4].try_into().unwrap()
    }

    fn effects() -> Vec<VideoEffect> {
        vec![
            VideoEffect::ColorCorrection(ColorCorrection {
                brightness: 0.1,
                contrast: 1.4,
                saturation: 0.6,
                hue_degrees: 35.0,
                gamma: 1.6,
                opacity: 0.8,
            }),
            VideoEffect::ColorCorrection(ColorCorrection {
                saturation: 0.0,
                ..ColorCorrection::default()
            }),
            VideoEffect::LumaKey(LumaKey {
                min: 0.25,
                min_smoothing: 0.1,
                max: 0.75,
                max_smoothing: 0.1,
            }),
        ]
    }

    /// The shader is the shared definition: every effect, on a spread of
    /// pixels, comes out within one step of what the CPU computes — the
    /// difference a GPU's own `pow` and rounding are allowed.
    #[test]
    fn every_effect_draws_what_the_shared_definition_says() {
        let Some((device, context)) = try_device() else {
            return;
        };
        for effect in effects() {
            for pixel in [
                [0u8, 0, 0, 255],
                [255, 255, 255, 255],
                [30, 140, 220, 255],
                [200, 60, 10, 160],
                [100, 110, 120, 255],
            ] {
                let expected = apply(&effect.params(), pixel);
                let drawn = drawn_pixel(&device, &context, effect, pixel);
                for channel in 0..4 {
                    assert!(
                        drawn[channel].abs_diff(expected[channel]) <= 1,
                        "{effect:?} on {pixel:?}: drew {drawn:?}, expected {expected:?}"
                    );
                }
            }
        }
    }

    /// An effect that changes nothing hands on the texture that arrived.
    #[test]
    fn a_neutral_effect_hands_the_same_texture_through() {
        let Some((device, context)) = try_device() else {
            return;
        };
        let source = frame(bgra_texture(&device, 4, 4, [10, 20, 30, 255], 1), 4, 4, 1);
        let input = texture_of(&source);
        let (mut element, _) = D3d11VideoEffect::new(
            "effect",
            &device,
            context,
            VideoEffect::LumaKey(LumaKey::default()),
        )
        .expect("D3d11VideoEffect::new");
        let received = capture(&mut element);

        element.consume(source).expect("pass through");

        assert_eq!(texture_of(&received.lock().unwrap()[0]), input);
    }

    /// A repeat of an unchanged texture is answered with the output already
    /// in hand — until the effect changes, when it is drawn again.
    #[test]
    fn a_repeat_is_drawn_once_and_again_after_a_retune() {
        let Some((device, context)) = try_device() else {
            return;
        };
        let texture = bgra_texture(&device, 4, 4, [128, 128, 128, 255], 1);
        let (mut element, handle) = D3d11VideoEffect::new(
            "effect",
            &device,
            context,
            VideoEffect::ColorCorrection(ColorCorrection {
                brightness: 0.2,
                ..ColorCorrection::default()
            }),
        )
        .expect("D3d11VideoEffect::new");
        let received = capture(&mut element);
        let source = frame(texture, 4, 4, 1);
        let MediaBuffer::Video(inner) = &source else {
            unreachable!();
        };
        let repeat = {
            let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
            let mut slot = pool.get();
            // SAFETY: both are live `AVFrame`s and distinct.
            unsafe {
                assert!(ffmpeg::ffi::av_frame_ref(slot.as_mut_ptr(), inner.as_ptr()) >= 0);
            }
            MediaBuffer::Video(Arc::new(slot))
        };

        element.consume(source.clone()).expect("first");
        element.consume(repeat).expect("repeat");
        handle.set_effect(VideoEffect::ColorCorrection(ColorCorrection {
            brightness: -0.2,
            ..ColorCorrection::default()
        }));
        element.consume(source).expect("after a retune");

        let received = received.lock().unwrap();
        assert_eq!(texture_of(&received[1]), texture_of(&received[0]));
        assert_ne!(texture_of(&received[2]), texture_of(&received[0]));
    }

    /// Only the visible region of a padded texture is drawn, from the slice
    /// the frame names.
    #[test]
    fn the_visible_region_of_the_named_slice_is_what_is_drawn() {
        let Some((device, context)) = try_device() else {
            return;
        };
        // A 16x16 two-slice array: slice 0 white, slice 1 black; the frame
        // names slice 0 and an 8x8 picture.
        let array = bgra_texture(&device, 16, 16, [0, 0, 0, 255], 2);
        let darker = VideoEffect::ColorCorrection(ColorCorrection {
            brightness: -0.4,
            ..ColorCorrection::default()
        });
        let (mut element, _) = D3d11VideoEffect::new("effect", &device, context.clone(), darker)
            .expect("D3d11VideoEffect::new");
        let mut download =
            D3d11Download::new("download", &device, context, 8, 8).expect("D3d11Download::new");
        let received = capture(&mut download);
        element.src_pads()[0].link(Box::new(download));

        element.consume(frame(array, 8, 8, 1)).expect("draw");

        let received = received.lock().unwrap();
        let MediaBuffer::Video(drawn) = &received[0] else {
            panic!("expected a Video buffer");
        };
        assert_eq!((drawn.width(), drawn.height()), (8, 8));
        // White less 0.4 is 153 — slice 0, not the black slice 1.
        assert_eq!(drawn.data(0)[0..4], [153, 153, 153, 255]);
    }

    /// A debug device and the two interfaces that count what it still owns,
    /// or `None` without the D3D11 SDK debug layer.
    fn try_debug_device() -> Option<(
        ID3D11Device,
        Arc<Mutex<ID3D11DeviceContext>>,
        ID3D11Debug,
        ID3D11InfoQueue,
    )> {
        use windows::Win32::Graphics::Direct3D::D3D_DRIVER_TYPE_HARDWARE;

        let mut device = None;
        let mut context = None;
        // SAFETY: null adapter/software pointers select the hardware driver,
        // feature-level defaults are requested, and `device`/`context` are
        // live correctly typed out-parameters.
        let result = unsafe {
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                Default::default(),
                D3D11_CREATE_DEVICE_DEBUG,
                None,
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut context),
            )
        };
        if result.is_err() {
            eprintln!("skipping: no D3D11 debug device on this machine: {result:?}");
            return None;
        }
        let device = device?;
        let context = context?;
        let debug = device.cast::<ID3D11Debug>().ok()?;
        let info = device.cast::<ID3D11InfoQueue>().ok()?;
        // SAFETY: `info` is the live debug info queue for this device.
        unsafe { info.SetMessageCountLimit(u64::MAX) }.ok()?;
        Some((device, Arc::new(Mutex::new(context)), debug, info))
    }

    fn live_objects(debug: &ID3D11Debug, info: &ID3D11InfoQueue) -> u64 {
        // SAFETY: both interfaces belong to the same live debug device; the
        // report appends its messages before they are counted.
        unsafe {
            info.ClearStoredMessages();
            debug
                .ReportLiveDeviceObjects(D3D11_RLDO_DETAIL)
                .expect("ReportLiveDeviceObjects");
            info.GetNumStoredMessages()
        }
    }

    /// Each draw creates an output texture and two views, which the device
    /// destroys only once the context is flushed — `D3d11ChromaKey` leaked
    /// three objects a frame before it flushed. A flat count across a
    /// hundred frames is what shows this one does not.
    #[test]
    fn drawing_frames_does_not_accumulate_d3d11_objects() {
        let Some((device, context, debug, info)) = try_debug_device() else {
            return;
        };
        let texture = bgra_texture(&device, 64, 64, [30, 140, 220, 255], 1);
        let (mut element, _) = D3d11VideoEffect::new(
            "effect",
            &device,
            context,
            VideoEffect::ColorCorrection(ColorCorrection {
                saturation: 0.3,
                ..ColorCorrection::default()
            }),
        )
        .expect("D3d11VideoEffect::new");
        let received = capture(&mut element);
        let mut push = |pts: i64| {
            element
                .consume(frame(texture.clone(), 64, 64, pts))
                .expect("draw");
            received.lock().unwrap().clear();
        };

        for pts in 0..20 {
            push(pts);
        }
        let baseline = live_objects(&debug, &info);
        for pts in 20..120 {
            push(pts);
        }
        let after = live_objects(&debug, &info);

        assert_eq!(
            after,
            baseline,
            "100 more frames left {} extra D3D11 objects on the device",
            after as i64 - baseline as i64
        );
    }

    #[test]
    fn what_cannot_be_drawn_is_refused_by_name() {
        let Some((device, context)) = try_device() else {
            return;
        };
        let effect = VideoEffect::ColorCorrection(ColorCorrection {
            brightness: 0.1,
            ..ColorCorrection::default()
        });
        let (mut element, _) = D3d11VideoEffect::new("effect", &device, context.clone(), effect)
            .expect("D3d11VideoEffect::new");

        let cpu = UnboundObjectPool::new(
            0,
            || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, 8, 8),
            |_| {},
        );
        assert!(matches!(
            element.consume(MediaBuffer::Video(Arc::new(cpu.get()))),
            Err(crate::error::Error::D3d11VideoEffectError(
                D3d11VideoEffectError::UnsupportedFormat(ffmpeg::format::Pixel::BGRA)
            ))
        ));

        let desc = D3D11_TEXTURE2D_DESC {
            Width: 16,
            Height: 16,
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
        let mut nv12 = None;
        // SAFETY: `desc` is fully initialized, no initial pixels are
        // supplied, and `nv12` is a live out-parameter.
        unsafe {
            device
                .CreateTexture2D(&desc, None, Some(&mut nv12))
                .expect("CreateTexture2D(NV12)");
        }
        assert!(matches!(
            element.consume(frame(nv12.unwrap(), 16, 16, 1)),
            Err(crate::error::Error::D3d11VideoEffectError(
                D3d11VideoEffectError::UnsupportedTextureFormat(_)
            ))
        ));

        if let Some((other, other_context)) = try_device() {
            let foreign = bgra_texture(&other, 8, 8, [0, 0, 0, 255], 1);
            assert!(matches!(
                element.consume(frame(foreign, 8, 8, 1)),
                Err(crate::error::Error::D3d11VideoEffectError(
                    D3d11VideoEffectError::DeviceMismatch
                ))
            ));
            assert!(matches!(
                D3d11VideoEffect::new("effect", &device, other_context, effect),
                Err(D3d11VideoEffectError::ContextDeviceMismatch)
            ));
        }

        assert!(matches!(
            element.consume(MediaBuffer::Audio(Arc::new(ffmpeg::frame::Audio::empty()))),
            Err(crate::error::Error::D3d11VideoEffectError(
                D3d11VideoEffectError::UnsupportedBuffer("Audio")
            ))
        ));
    }
}
