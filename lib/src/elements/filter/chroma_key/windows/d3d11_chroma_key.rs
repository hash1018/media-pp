use std::{
    ffi::c_void,
    sync::{Arc, Mutex},
};

use crate::pp_log::{PpLog, pp_error, pp_info};
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

use super::super::options::ChromaKeyOptions;
use crate::{
    buffer::MediaBuffer,
    color::Color,
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    error::Result,
    pad::SrcPad,
    platform::windows::d3d11::compile_shader,
    platform::windows::d3d11va::{d3d11va_texture, wrap_d3d11_texture},
    pool::UnboundObjectPool,
};

const SHADER_SOURCE: &[u8] = include_bytes!("../../../../shaders/d3d11/chroma_key_bgra.hlsl");

/// Errors specific to [`D3d11ChromaKey`]. Converts into the crate-wide
/// `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum D3d11ChromaKeyError {
    #[error("windows error: {0}")]
    Windows(#[from] windows::core::Error),

    #[error("D3d11ChromaKey only keys Pixel::D3D11 frames, got {0:?}")]
    UnsupportedFormat(ffmpeg::format::Pixel),

    #[error(
        "frame claimed the D3D11 pixel format but carries no texture — must \
         come from D3d11Upload/D3d11Decoder/DxgiCaptureSource's GPU mode/D3d11VideoCompositor"
    )]
    InvalidD3d11Frame,

    #[error(
        "D3d11ChromaKey only keys DXGI_FORMAT_B8G8R8A8_UNORM textures, got {0:?}; \
         keying writes per-pixel alpha, which an NV12 surface has nowhere to store"
    )]
    UnsupportedTextureFormat(DXGI_FORMAT),

    #[error(
        "a Pixel::D3D11 frame's texture lives on a different ID3D11Device \
         than this D3d11ChromaKey was created with — every D3D11 element in one \
         pipeline must share exactly one device for zero-copy to be valid"
    )]
    DeviceMismatch,

    #[error(
        "the supplied ID3D11DeviceContext belongs to a different ID3D11Device than this \
         D3d11ChromaKey"
    )]
    ContextDeviceMismatch,

    #[error(
        "D3D11 texture is {actual_width}x{actual_height}, smaller than the \
         frame's {expected_width}x{expected_height} visible size"
    )]
    TextureTooSmall {
        actual_width: u32,
        actual_height: u32,
        expected_width: u32,
        expected_height: u32,
    },

    #[error("D3D11 texture array index {index} is outside ArraySize {array_size}")]
    InvalidArrayIndex { index: isize, array_size: u32 },

    #[error("D3d11ChromaKey does not accept multisampled textures (SampleDesc.Count={0})")]
    MultisampledTexture(u32),

    #[error(
        "the input texture was created without D3D11_BIND_SHADER_RESOURCE (BindFlags={0:#x}), \
         so this element's pixel shader cannot read it"
    )]
    MissingShaderResourceBind(u32),

    #[error("frame has invalid dimensions {width}x{height}")]
    InvalidFrameDimensions { width: u32, height: u32 },

    #[error("D3d11ChromaKey only accepts Video and Eos buffers, got a {0}")]
    UnsupportedBuffer(&'static str),
}

/// The per-draw constant buffer `chroma_key_bgra.hlsl` reads. Field order
/// and padding are what HLSL's own `cbuffer` packing rules expect — see the
/// shader's `ChromaKeyBuffer`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ChromaKeyConstants {
    key_color: [f32; 3],
    band_low: f32,
    inv_band_width: f32,
    _padding: [f32; 3],
    uv_scale: [f32; 2],
    _uv_padding: [f32; 2],
}

impl ChromaKeyConstants {
    /// Resolves `threshold`/`smoothing` into the feather band the shader
    /// actually evaluates: `saturate((distance - band_low) *
    /// inv_band_width)`.
    ///
    /// A hard key (`smoothing <= 0.0`) is a band of no width, which that
    /// expression cannot represent directly — so it is given a `band_low`
    /// of exactly `threshold` and an `inv_band_width` large enough that any
    /// distance above `threshold`, by however little, saturates to 1.0
    /// while `threshold` itself still lands on 0.0. That is precisely
    /// [`crate::elements::SwChromaKey`]'s own step, and it costs the shader
    /// neither a branch nor a division by zero.
    fn new(key_color: Color, threshold: f32, smoothing: f32, uv_scale: [f32; 2]) -> Self {
        let smoothing = smoothing.max(0.0);
        let (band_low, inv_band_width) = if smoothing > 0.0 {
            (threshold - smoothing / 2.0, 1.0 / smoothing)
        } else {
            (threshold, f32::MAX)
        };
        Self {
            key_color: [
                f32::from(key_color.red) / 255.0,
                f32::from(key_color.green) / 255.0,
                f32::from(key_color.blue) / 255.0,
            ],
            band_low,
            inv_band_width,
            _padding: [0.0; 3],
            uv_scale,
            _uv_padding: [0.0; 2],
        }
    }
}

/// Keys a solid background color out of a GPU-resident `Pixel::D3D11` BGRA
/// frame into its alpha channel without ever touching the CPU — the D3D11
/// half of this crate's chroma-key support, doing through a pixel shader
/// exactly what [`crate::elements::SwChromaKey`] does per pixel on the CPU.
/// A `Filter`: receives via `Sink`, pushes the keyed frame on through its
/// own single src pad.
///
/// The point is placement: `DxgiCaptureSource -> D3d11ChromaKey ->
/// D3d11VideoCompositorHandle::add_source` keys a live layer with the frame
/// staying in video memory the whole way, where the software element would
/// have cost a [`crate::elements::D3d11Download`] and a
/// [`crate::elements::D3d11Upload`] — two PCIe crossings per frame — around
/// it.
///
/// # BGRA in, BGRA out
///
/// Input must be a `DXGI_FORMAT_B8G8R8A8_UNORM` texture, the same
/// constraint [`crate::elements::SwChromaKey`] has for the same reason: the
/// result of keying *is* an alpha channel, and NV12 has nowhere to put one.
/// [`crate::elements::D3d11Upload`] (from a `Pixel::BGRA` frame),
/// [`crate::elements::DxgiCaptureSource`]'s GPU mode, and a compositor's
/// output all produce that directly. A decoder does not — its surfaces are
/// NV12 — so a decoded green screen reaches this element through
/// [`crate::elements::D3d11Scaler`] with
/// [`crate::elements::D3d11ScalerFormat::Bgra`], which converts on the same
/// `VideoProcessorBlt` it would resize with. RGB is written through
/// untouched; only alpha changes.
///
/// Output is a fresh texture rather than a keyed-in-place input: the frame
/// arriving here is `Arc`-shared and its upstream owner may still be
/// reading it, and this crate's contract is that a published frame is never
/// mutated. Every output frame is the *visible* size of its input, so a
/// decoder's alignment padding is dropped here rather than travelling on as
/// garbage in the keyed layer.
///
/// # Device and context
///
/// `device` must be the same `ID3D11Device`, and `context` the same shared
/// immediate context, every other D3D11 element in the pipeline uses; the
/// context lock is held for one configure-and-`Draw` sequence per frame,
/// for the same reason [`crate::elements::D3d11Scaler`] holds it across its
/// `Blt`. The shader/sampler/blend/rasterizer state is built once at
/// construction and re-selected on every draw, because the context is
/// shared: whatever drew last left its own state bound.
///
/// Each draw ends with a `Flush`. D3D11 defers destroying an object until
/// the context is flushed, and this element creates three per frame — the
/// output texture and the two views around it — so without one the device
/// accumulates them for the life of the pipeline. That makes this element
/// the flush point for everything else queued on the shared context, which
/// is the trade it makes: batching across elements in exchange for a bound
/// on what the device holds. See `key`'s own comment for the measurements.
pub struct D3d11ChromaKey {
    pp_log: PpLog,
    name: Arc<str>,
    device: ID3D11Device,
    context: Arc<Mutex<ID3D11DeviceContext>>,
    key_color: Color,
    threshold: f32,
    smoothing: f32,
    vertex_shader: ID3D11VertexShader,
    pixel_shader: ID3D11PixelShader,
    sampler: ID3D11SamplerState,
    blend_state: ID3D11BlendState,
    rasterizer_state: ID3D11RasterizerState,
    constant_buffer: ID3D11Buffer,
    pad: SrcPad,
    /// Reused across every keyed frame — see [`UnboundObjectPool`]'s docs.
    /// Only the small CPU-side `AVFrame` wrapper is actually reused; the
    /// output texture itself is a fresh allocation per frame, since
    /// downstream may still hold `Arc` clones of the previous one.
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
}

// SAFETY: every field is either a `windows-rs` COM interface wrapper (the
// device-level ones free-threaded, the context-level ones behind `context`'s
// own `Mutex`) or plain data. `&mut self` on every method that touches
// non-`Arc`/`Mutex` state already rules out concurrent access to those parts
// from multiple threads — same reasoning as `D3d11Scaler`.
unsafe impl Send for D3d11ChromaKey {}

/// One validated input: the texture to sample, which slice of it, the
/// visible size to draw, and the fraction of the texture that size covers.
struct ValidatedInput {
    texture: ID3D11Texture2D,
    array_slice: u32,
    width: u32,
    height: u32,
    uv_scale: [f32; 2],
}

impl D3d11ChromaKey {
    /// `device` must be the same `ID3D11Device`, and `context` the same
    /// shared immediate context, every other D3D11 element in this pipeline
    /// uses — see this type's own docs on why.
    ///
    /// Output dimensions are not a parameter: keying is a per-pixel
    /// transform, so every frame comes out the size it went in.
    pub fn new(
        name: impl Into<String>,
        device: &ID3D11Device,
        context: Arc<Mutex<ID3D11DeviceContext>>,
        options: ChromaKeyOptions,
    ) -> std::result::Result<Self, D3d11ChromaKeyError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::D3d11ChromaKey, &name, None);
        // The caller's own mistake is diagnosed before any GPU resource is
        // built: a context belonging to another device would otherwise only
        // surface later, as a `Draw` quietly reading nothing.
        {
            let context = context
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            // SAFETY: `context` is a live immediate-context interface;
            // `GetDevice` returns an owned reference to its creating device.
            let context_device = unsafe { context.GetDevice() }?;
            if context_device.as_raw() != device.as_raw() {
                return Err(D3d11ChromaKeyError::ContextDeviceMismatch);
            }
        }

        // SAFETY: `device` is live; the helper creates state only from static
        // shader bytes and fully initialized descriptors, returning owned COM
        // references without retaining borrowed pointers.
        let (vertex_shader, pixel_shader, sampler, blend_state, rasterizer_state, constant_buffer) =
            unsafe { build_pipeline_state(device) }?;

        let key_color = options.method.key_color();
        let pad = SrcPad::new(format!("{name}_src"));
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        pp_info!(
            pp_log: &pp_log,
            "created: key_color={key_color:?}, threshold={}, smoothing={}",
            options.threshold,
            options.smoothing
        );
        Ok(Self {
            name,
            pp_log,
            device: device.clone(),
            context,
            key_color,
            threshold: options.threshold,
            smoothing: options.smoothing,
            vertex_shader,
            pixel_shader,
            sampler,
            blend_state,
            rasterizer_state,
            constant_buffer,
            pad,
            pool,
        })
    }

    /// Rejects anything this element cannot key, so a bad frame fails here
    /// with a message naming the actual problem rather than as a `Draw`
    /// silently sampling a texture that belongs to another device.
    fn validate(
        &self,
        frame: &ffmpeg::frame::Video,
    ) -> std::result::Result<ValidatedInput, D3d11ChromaKeyError> {
        if frame.format() != ffmpeg::format::Pixel::D3D11 {
            return Err(D3d11ChromaKeyError::UnsupportedFormat(frame.format()));
        }
        if frame.width() == 0 || frame.height() == 0 {
            return Err(D3d11ChromaKeyError::InvalidFrameDimensions {
                width: frame.width(),
                height: frame.height(),
            });
        }
        let (texture_raw, index) =
            d3d11va_texture(frame).ok_or(D3d11ChromaKeyError::InvalidD3d11Frame)?;
        if texture_raw.is_null() {
            return Err(D3d11ChromaKeyError::InvalidD3d11Frame);
        }
        // SAFETY: `texture_raw` is a borrowed raw `ID3D11Texture2D*` — still
        // owned by `frame`'s own buffer reference, not by us. `.clone()`
        // (`AddRef`) gives an independently ref-counted handle, valid for as
        // long as the caller keeps it.
        let texture = unsafe {
            ID3D11Texture2D::from_raw_borrowed(&texture_raw)
                .ok_or(D3d11ChromaKeyError::InvalidD3d11Frame)?
                .clone()
        };

        // SAFETY: `texture` is a live cloned COM interface; `GetDevice`
        // returns an owned reference to its creating device.
        let texture_device = unsafe { texture.GetDevice() }?;
        if texture_device.as_raw() != self.device.as_raw() {
            return Err(D3d11ChromaKeyError::DeviceMismatch);
        }

        let mut desc = D3D11_TEXTURE2D_DESC::default();
        // SAFETY: `desc` is a live out-parameter for the live texture.
        unsafe { texture.GetDesc(&mut desc) };
        if desc.Format != DXGI_FORMAT_B8G8R8A8_UNORM {
            return Err(D3d11ChromaKeyError::UnsupportedTextureFormat(desc.Format));
        }
        if desc.Width < frame.width() || desc.Height < frame.height() {
            return Err(D3d11ChromaKeyError::TextureTooSmall {
                actual_width: desc.Width,
                actual_height: desc.Height,
                expected_width: frame.width(),
                expected_height: frame.height(),
            });
        }
        if index < 0 || index as u64 >= u64::from(desc.ArraySize) {
            return Err(D3d11ChromaKeyError::InvalidArrayIndex {
                index,
                array_size: desc.ArraySize,
            });
        }
        if desc.SampleDesc.Count != 1 {
            return Err(D3d11ChromaKeyError::MultisampledTexture(
                desc.SampleDesc.Count,
            ));
        }
        // Checked here rather than left to `CreateShaderResourceView`: the
        // API's own failure is a bare `E_INVALIDARG` that says nothing
        // about which of this element's inputs was built wrong.
        if desc.BindFlags & D3D11_BIND_SHADER_RESOURCE.0 as u32 == 0 {
            return Err(D3d11ChromaKeyError::MissingShaderResourceBind(
                desc.BindFlags,
            ));
        }

        Ok(ValidatedInput {
            texture,
            array_slice: index as u32,
            // The visible size, not the texture's: a decoder pads its
            // surfaces up to its own alignment, and the padding is not
            // part of the picture.
            width: frame.width(),
            height: frame.height(),
            uv_scale: [
                frame.width() as f32 / desc.Width as f32,
                frame.height() as f32 / desc.Height as f32,
            ],
        })
    }

    fn key(&mut self, frame: &ffmpeg::frame::Video) -> Result<()> {
        let input = self
            .validate(frame)
            .inspect_err(|error| pp_error!(self, "{error}"))?;

        let output = create_output_texture(&self.device, input.width, input.height)
            .inspect_err(|error| pp_error!(self, "failed to allocate the output texture: {error}"))
            .map_err(D3d11ChromaKeyError::from)?;
        let mut output_view = None;
        // SAFETY: `output` is a live render-target-capable texture and
        // `output_view` is the correctly typed live out-parameter.
        unsafe {
            self.device
                .CreateRenderTargetView(&output, None, Some(&mut output_view))
                .inspect_err(|error| pp_error!(self, "failed to create the output RTV: {error}"))
                .map_err(D3d11ChromaKeyError::from)?;
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
        // SAFETY: input validation established the texture format and bounded
        // array slice; `srv_desc` selects exactly that slice and `srv` is a
        // live out-parameter.
        unsafe {
            self.device
                .CreateShaderResourceView(&input.texture, Some(&srv_desc), Some(&mut srv))
                .inspect_err(|error| pp_error!(self, "failed to create the input SRV: {error}"))
                .map_err(D3d11ChromaKeyError::from)?;
        }

        let constants = ChromaKeyConstants::new(
            self.key_color,
            self.threshold,
            self.smoothing,
            input.uv_scale,
        );
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
                // Every piece of state is re-selected rather than assumed:
                // this context is shared with every other D3D11 element in
                // the pipeline, and whatever drew last left its own bound.
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

                // Released on the way out, always: leaving this frame's
                // input SRV and output RTV bound would keep them alive
                // inside the shared context's state until something else
                // happened to overwrite the same slots.
                context.PSSetShaderResources(0, Some(&[None]));
                context.OMSetRenderTargets(None, None);
                // Dropping the last reference to a D3D11 object does not
                // destroy it: destruction is deferred until the context is
                // flushed. This element creates three objects per frame —
                // the output texture, its render-target view, and the
                // input's shader-resource view — so without this the device
                // accumulates them for as long as the pipeline runs.
                // Measured directly: 500 frames through one element grew
                // the debug layer's live-object count by exactly 3 per
                // frame, and stayed flat at its starting value with this
                // call in place.
                //
                // The two views cannot be cached instead. The output's RTV
                // could be, by pooling output textures the way
                // `D3d11VideoCompositor` pools its own, but the input's SRV
                // could not: it is built over a texture this element does
                // not own, so a cache keyed by that texture's pointer goes
                // stale the moment upstream frees it and reuses the
                // address. Flushing is what D3D11 offers for this, and the
                // cost is the batching given up by submitting once per
                // frame on the shared context.
                context.Flush();
            }
        }

        let mut keyed = self.pool.get();
        // Overwrites the pooled slot's previous contents in place —
        // `ffmpeg::frame::Video`'s own `Drop` runs on whatever was there
        // before, releasing that frame's GPU texture right here.
        *keyed = wrap_d3d11_texture(output, input.width, input.height);
        keyed.set_pts(frame.pts());
        keyed.set_color_space(frame.color_space());
        keyed.set_color_range(frame.color_range());

        self.pad.push(MediaBuffer::Video(Arc::new(keyed)))
    }
}

impl Element for D3d11ChromaKey {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::D3d11ChromaKey
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for D3d11ChromaKey {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for D3d11ChromaKey {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Video(frame) => self.key(&frame),
            // Nothing is buffered here — one `Draw` per frame, pushed
            // before `consume` returns — so there is nothing to drain.
            MediaBuffer::Eos => self.pad.push(MediaBuffer::Eos),
            other => {
                let kind = other.kind();
                pp_error!(self, "unsupported buffer: {kind}");
                Err(D3d11ChromaKeyError::UnsupportedBuffer(kind).into())
            }
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        // A pure per-pixel transform, same as `D3d11Scaler` — nothing local
        // buffered or ordered to flush on any `ControlMsg`.
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
            s!("chroma_key_bgra.hlsl"),
            s!("vs_main"),
            s!("vs_5_0"),
        )?;
        let pixel_bytecode = compile_shader(
            SHADER_SOURCE,
            s!("chroma_key_bgra.hlsl"),
            s!("ps_chroma_key"),
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

        // Point sampling, not linear: input and output are the same size,
        // so every output pixel maps to exactly one input texel. Filtering
        // would only blend neighbours across the key edge, softening it in
        // a way `smoothing` is supposed to control explicitly.
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

        // Blending is explicitly *off*. The alpha this shader computes is
        // the element's whole output; letting it blend against the render
        // target would consume it instead of storing it.
        let mut blend_desc = D3D11_BLEND_DESC::default();
        blend_desc.RenderTarget[0] = D3D11_RENDER_TARGET_BLEND_DESC {
            BlendEnable: false.into(),
            RenderTargetWriteMask: D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8,
            ..Default::default()
        };
        let mut blend_state = None;
        device.CreateBlendState(&blend_desc, Some(&mut blend_state))?;
        let blend_state = blend_state.expect("CreateBlendState succeeded without a state");

        // Scissoring off, unlike the compositor's: this draw covers the
        // whole render target by construction, and the shared context may
        // arrive carrying someone else's scissor rect.
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
            ByteWidth: std::mem::size_of::<ChromaKeyConstants>() as u32,
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

/// `SHADER_RESOURCE` alongside `RENDER_TARGET` because the keyed frame is
/// drawn into here and then sampled by whatever comes next — a compositor
/// layer, a renderer, or another shader-based filter.
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
        Direct3D::D3D_DRIVER_TYPE_HARDWARE,
        Direct3D11::{
            D3D11_CREATE_DEVICE_DEBUG, D3D11_RLDO_DETAIL, D3D11_SDK_VERSION,
            D3D11_SUBRESOURCE_DATA, D3D11CreateDevice, ID3D11Debug, ID3D11InfoQueue,
        },
        Dxgi::Common::DXGI_FORMAT_NV12,
    };

    use super::{super::super::options::ChromaKeyMethod, *};
    use crate::elements::D3d11Download;
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

    fn default_options() -> ChromaKeyOptions {
        ChromaKeyOptions {
            method: ChromaKeyMethod::Green,
            threshold: 0.15,
            smoothing: 0.1,
        }
    }

    /// A BGRA texture whose every texel of the last slice is `color`;
    /// earlier slices get a contrasting one, so a test that names a slice
    /// proves the array index was honored rather than defaulted to zero.
    fn bgra_texture(
        device: &ID3D11Device,
        width: u32,
        height: u32,
        color: [u8; 4],
        slices: u32,
    ) -> ID3D11Texture2D {
        let other = [255u8, 255, 255, 255];
        let planes: Vec<Vec<u8>> = (0..slices)
            .map(|slice| {
                let pixel = if slice == slices - 1 { color } else { other };
                pixel.repeat((width * height) as usize)
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

    /// Wraps a texture as the pooled `Pixel::D3D11` `MediaBuffer` an
    /// upstream element would push. `width`/`height` are the *visible*
    /// size, which a caller may deliberately set smaller than the texture.
    fn frame(texture: ID3D11Texture2D, width: u32, height: u32, pts: i64) -> MediaBuffer {
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut slot = pool.get();
        *slot = wrap_d3d11_texture(texture, width, height);
        slot.set_pts(Some(pts));
        MediaBuffer::Video(Arc::new(slot))
    }

    /// Keys one flat `color` frame and hands back the downloaded BGRA
    /// result, so a test can assert on actual pixels rather than on the
    /// fact that a `Draw` returned without complaint.
    fn keyed_pixel(
        device: &ID3D11Device,
        context: &Arc<Mutex<ID3D11DeviceContext>>,
        options: ChromaKeyOptions,
        color: [u8; 4],
    ) -> [u8; 4] {
        let mut key = D3d11ChromaKey::new("key", device, context.clone(), options)
            .expect("D3d11ChromaKey::new should succeed");
        let mut download = D3d11Download::new("download", device, context.clone(), 8, 8)
            .expect("D3d11Download::new should succeed");
        let received = capture(&mut download);
        key.src_pads()[0].link(Box::new(download));

        key.consume(frame(bgra_texture(device, 8, 8, color, 1), 8, 8, 7))
            .expect("keying must succeed");

        let received = received.lock().unwrap();
        let MediaBuffer::Video(keyed) = &received[0] else {
            panic!("expected a Video buffer, got {}", received[0].kind());
        };
        keyed.data(0)[0..4]
            .try_into()
            .expect("a BGRA pixel is four bytes")
    }

    #[test]
    fn a_pixel_matching_the_key_color_becomes_fully_transparent() {
        let Some((device, context)) = try_device() else {
            return;
        };
        let green = [0u8, 255, 0, 255]; // BGRA: pure green
        assert_eq!(
            keyed_pixel(&device, &context, default_options(), green),
            [0, 255, 0, 0],
            "the key color must come out with alpha 0 and its RGB untouched"
        );
    }

    #[test]
    fn a_clearly_different_pixel_stays_fully_opaque_and_unchanged() {
        let Some((device, context)) = try_device() else {
            return;
        };
        let red = [0u8, 0, 255, 255]; // BGRA: pure red, far from the green key
        assert_eq!(
            keyed_pixel(&device, &context, default_options(), red),
            [0, 0, 255, 255]
        );
    }

    /// The same off-green `SwChromaKey`'s own feather test uses: only the
    /// red channel differs, by 60/255, so the distance is
    /// `(60/255)/sqrt(3) ~= 0.136` — inside the 0.10..0.20 band around
    /// threshold 0.15.
    #[test]
    fn a_pixel_inside_the_smoothing_band_gets_a_partial_alpha() {
        let Some((device, context)) = try_device() else {
            return;
        };
        let almost_green = [0u8, 255, 60, 255];
        let alpha = keyed_pixel(&device, &context, default_options(), almost_green)[3];
        assert!(
            alpha > 0 && alpha < 255,
            "expected a feathered mid-range alpha, got {alpha}"
        );
    }

    #[test]
    fn a_custom_key_color_replaces_the_green_default() {
        let Some((device, context)) = try_device() else {
            return;
        };
        let options = ChromaKeyOptions {
            method: ChromaKeyMethod::Custom(Color::new(10, 20, 30)),
            threshold: 0.05,
            smoothing: 0.0,
        };
        // BGRA, so this is exactly Color::new(10, 20, 30).
        let alpha = keyed_pixel(&device, &context, options, [30, 20, 10, 255])[3];
        assert_eq!(alpha, 0);
    }

    /// A hard key has no band to interpolate across, and both backends
    /// resolve that the same way: at the threshold itself the pixel is
    /// still background, past it foreground. See `ChromaKeyConstants::new`.
    #[test]
    fn a_zero_smoothing_key_is_a_hard_step_with_no_partial_alpha() {
        let Some((device, context)) = try_device() else {
            return;
        };
        let options = ChromaKeyOptions {
            method: ChromaKeyMethod::Green,
            threshold: 0.15,
            smoothing: 0.0,
        };
        // Distance ~0.136 — below the threshold, so fully keyed out.
        assert_eq!(
            keyed_pixel(&device, &context, options, [0, 255, 60, 255])[3],
            0
        );
        // Distance (100/255)/sqrt(3) ~= 0.226 — above it, so fully opaque.
        assert_eq!(
            keyed_pixel(&device, &context, options, [0, 255, 100, 255])[3],
            255
        );
    }

    /// The whole point of doing this on the GPU: the result stays a
    /// `Pixel::D3D11` frame, at the input's size, still carrying its
    /// timestamp and colorimetry.
    #[test]
    fn the_output_stays_on_the_gpu_and_keeps_its_pts_and_colorimetry() {
        let Some((device, context)) = try_device() else {
            return;
        };
        let texture = bgra_texture(&device, 16, 16, [0, 255, 0, 255], 1);
        let mut source = frame(texture, 16, 16, 777);
        let MediaBuffer::Video(source_frame) = &mut source else {
            unreachable!("frame always returns a Video buffer");
        };
        let source_frame = Arc::get_mut(source_frame).expect("the frame is not shared yet");
        source_frame.set_color_space(ffmpeg::color::Space::RGB);
        source_frame.set_color_range(ffmpeg::color::Range::JPEG);

        let mut key = D3d11ChromaKey::new("key", &device, context, default_options())
            .expect("D3d11ChromaKey::new should succeed");
        let received = capture(&mut key);
        key.consume(source).expect("keying must succeed");
        key.consume(MediaBuffer::Eos).expect("eos");

        let received = received.lock().unwrap();
        let MediaBuffer::Video(keyed) = &received[0] else {
            panic!("expected a Video buffer, got {}", received[0].kind());
        };
        assert_eq!(keyed.format(), ffmpeg::format::Pixel::D3D11);
        assert_eq!(keyed.width(), 16);
        assert_eq!(keyed.height(), 16);
        assert_eq!(keyed.pts(), Some(777), "the chroma key dropped the pts");
        assert_eq!(keyed.color_space(), ffmpeg::color::Space::RGB);
        assert_eq!(keyed.color_range(), ffmpeg::color::Range::JPEG);
        assert!(
            d3d11va_texture(keyed).is_some(),
            "the keyed frame must carry a texture"
        );
        assert!(
            received.last().is_some_and(MediaBuffer::is_eos),
            "Eos was not forwarded"
        );
    }

    /// A decoder pads its surfaces up to its own alignment. Only the
    /// visible region is the picture, so the output is that size and every
    /// sample comes from inside it.
    #[test]
    fn only_the_visible_region_of_a_padded_texture_is_keyed() {
        let Some((device, context)) = try_device() else {
            return;
        };
        // A 16x16 texture carrying an 8x8 visible picture, every texel the
        // key color: sampling past the visible region would clamp to the
        // same color, so what this pins down is the output's size and that
        // the uv_scale did not shrink the picture into a corner.
        let padded = bgra_texture(&device, 16, 16, [0, 255, 0, 255], 1);

        let mut key = D3d11ChromaKey::new("key", &device, context.clone(), default_options())
            .expect("D3d11ChromaKey::new should succeed");
        let mut download = D3d11Download::new("download", &device, context, 8, 8)
            .expect("D3d11Download::new should succeed");
        let received = capture(&mut download);
        key.src_pads()[0].link(Box::new(download));

        key.consume(frame(padded, 8, 8, 1)).expect("keying");

        let received = received.lock().unwrap();
        let MediaBuffer::Video(keyed) = &received[0] else {
            panic!("expected a Video buffer, got {}", received[0].kind());
        };
        assert_eq!(keyed.width(), 8);
        assert_eq!(keyed.height(), 8);
        let stride = keyed.stride(0);
        for row in 0..8usize {
            for column in 0..8usize {
                let offset = row * stride + column * 4;
                assert_eq!(
                    &keyed.data(0)[offset..offset + 4],
                    [0, 255, 0, 0],
                    "row {row}, column {column} was not keyed out"
                );
            }
        }
    }

    /// The array slice a frame names has to be the one sampled — a texture
    /// array is how D3D11VA hands out decode surfaces, and reading slice 0
    /// regardless would silently key the wrong picture.
    #[test]
    fn the_frames_own_array_slice_is_the_one_keyed() {
        let Some((device, context)) = try_device() else {
            return;
        };
        // Slice 1 holds the key color, slice 0 opaque white. `frame`
        // stores array index 0, so the result must come out white and
        // fully opaque rather than keyed.
        let array = bgra_texture(&device, 8, 8, [0, 255, 0, 255], 2);

        let mut key = D3d11ChromaKey::new("key", &device, context.clone(), default_options())
            .expect("D3d11ChromaKey::new should succeed");
        let mut download = D3d11Download::new("download", &device, context, 8, 8)
            .expect("D3d11Download::new should succeed");
        let received = capture(&mut download);
        key.src_pads()[0].link(Box::new(download));

        key.consume(frame(array, 8, 8, 1)).expect("keying");

        let received = received.lock().unwrap();
        let MediaBuffer::Video(keyed) = &received[0] else {
            panic!("expected a Video buffer, got {}", received[0].kind());
        };
        assert_eq!(
            keyed.data(0)[0..4],
            [255, 255, 255, 255],
            "slice 0 is white, so nothing should have been keyed out"
        );
    }

    #[test]
    fn a_cpu_frame_is_a_typed_error_not_a_panic() {
        let Some((device, context)) = try_device() else {
            return;
        };
        let mut key = D3d11ChromaKey::new("key", &device, context, default_options())
            .expect("D3d11ChromaKey::new should succeed");
        let pool = UnboundObjectPool::new(
            0,
            || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, 8, 8),
            |_| {},
        );

        let error = key
            .consume(MediaBuffer::Video(Arc::new(pool.get())))
            .expect_err("a CPU BGRA frame must be rejected");
        assert!(
            matches!(
                error,
                crate::error::Error::D3d11ChromaKeyError(D3d11ChromaKeyError::UnsupportedFormat(
                    ffmpeg::format::Pixel::BGRA
                ))
            ),
            "unexpected error: {error}"
        );
    }

    /// Keying writes per-pixel alpha, which an NV12 surface cannot hold —
    /// so a decoder's own output has to be rejected with a message that
    /// says why, not drawn into a garbage result.
    #[test]
    fn an_nv12_texture_is_rejected_with_the_format_named() {
        let Some((device, context)) = try_device() else {
            return;
        };
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
        let mut texture = None;
        // SAFETY: `desc` is fully initialized, no initial pixels are supplied,
        // and `texture` is a live out-parameter.
        unsafe {
            device
                .CreateTexture2D(&desc, None, Some(&mut texture))
                .expect("CreateTexture2D(NV12) failed");
        }
        let texture = texture.expect("CreateTexture2D succeeded without producing a texture");

        let mut key = D3d11ChromaKey::new("key", &device, context, default_options())
            .expect("D3d11ChromaKey::new should succeed");
        let error = key
            .consume(frame(texture, 16, 16, 1))
            .expect_err("an NV12 texture must be rejected");
        assert!(
            matches!(
                error,
                crate::error::Error::D3d11ChromaKeyError(
                    D3d11ChromaKeyError::UnsupportedTextureFormat(_)
                )
            ),
            "unexpected error: {error}"
        );
    }

    /// A texture the pixel shader is not allowed to read is caught here
    /// rather than surfacing as an unexplained `CreateShaderResourceView`
    /// failure.
    #[test]
    fn a_texture_without_the_shader_resource_bind_is_rejected() {
        let Some((device, context)) = try_device() else {
            return;
        };
        let desc = D3D11_TEXTURE2D_DESC {
            Width: 8,
            Height: 8,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_RENDER_TARGET.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        let mut texture = None;
        // SAFETY: `desc` is fully initialized for a render target, no initial
        // pixels are supplied, and `texture` is a live out-parameter.
        unsafe {
            device
                .CreateTexture2D(&desc, None, Some(&mut texture))
                .expect("CreateTexture2D(BGRA render target) failed");
        }
        let texture = texture.expect("CreateTexture2D succeeded without producing a texture");

        let mut key = D3d11ChromaKey::new("key", &device, context, default_options())
            .expect("D3d11ChromaKey::new should succeed");
        let error = key
            .consume(frame(texture, 8, 8, 1))
            .expect_err("a non-sampleable texture must be rejected");
        assert!(
            matches!(
                error,
                crate::error::Error::D3d11ChromaKeyError(
                    D3d11ChromaKeyError::MissingShaderResourceBind(_)
                )
            ),
            "unexpected error: {error}"
        );
    }

    /// Zero-copy is only valid within one device, so a foreign texture is
    /// named as such rather than drawn from.
    #[test]
    fn a_texture_from_another_device_is_rejected() {
        let Some((device, context)) = try_device() else {
            return;
        };
        let Some((other_device, _other_context)) = try_device() else {
            return;
        };
        let foreign = bgra_texture(&other_device, 8, 8, [0, 255, 0, 255], 1);

        let mut key = D3d11ChromaKey::new("key", &device, context, default_options())
            .expect("D3d11ChromaKey::new should succeed");
        let error = key
            .consume(frame(foreign, 8, 8, 1))
            .expect_err("a foreign texture must be rejected");
        assert!(
            matches!(
                error,
                crate::error::Error::D3d11ChromaKeyError(D3d11ChromaKeyError::DeviceMismatch)
            ),
            "unexpected error: {error}"
        );
    }

    /// Caught at construction, where the caller can still act on it —
    /// not once per frame.
    #[test]
    fn a_context_from_another_device_is_rejected_at_construction() {
        let Some((device, _context)) = try_device() else {
            return;
        };
        let Some((_other_device, other_context)) = try_device() else {
            return;
        };

        assert!(matches!(
            D3d11ChromaKey::new("key", &device, other_context, default_options()),
            Err(D3d11ChromaKeyError::ContextDeviceMismatch)
        ));
    }

    #[test]
    fn audio_buffers_are_rejected_not_silently_dropped() {
        let Some((device, context)) = try_device() else {
            return;
        };
        let mut key = D3d11ChromaKey::new("key", &device, context, default_options())
            .expect("D3d11ChromaKey::new should succeed");

        let error = key
            .consume(MediaBuffer::Audio(Arc::new(ffmpeg::frame::Audio::empty())))
            .expect_err("an Audio buffer must be rejected");
        assert!(
            matches!(
                error,
                crate::error::Error::D3d11ChromaKeyError(D3d11ChromaKeyError::UnsupportedBuffer(
                    "Audio"
                ))
            ),
            "unexpected error: {error}"
        );
    }

    /// The band the shader evaluates, checked without a GPU: a feathered
    /// key becomes a real interval around the threshold, and a hard one an
    /// interval of no width that still puts the threshold itself on the
    /// background side.
    #[test]
    fn the_constant_buffer_resolves_the_band_the_shader_needs() {
        let feathered = ChromaKeyConstants::new(Color::new(0, 255, 0), 0.15, 0.1, [1.0, 1.0]);
        assert!((feathered.band_low - 0.10).abs() < 1e-6);
        assert!((feathered.inv_band_width - 10.0).abs() < 1e-6);
        assert_eq!(feathered.key_color, [0.0, 1.0, 0.0]);

        // Negative smoothing means the same thing as none, exactly as
        // `SwChromaKey::alpha_for` clamps it.
        for smoothing in [0.0, -1.0] {
            let hard = ChromaKeyConstants::new(Color::BLACK, 0.15, smoothing, [1.0, 1.0]);
            assert_eq!(hard.band_low, 0.15);
            // At the threshold itself the shader's `saturate((d - low) *
            // inv)` is exactly 0; anything above it saturates to 1.
            let at = ((0.15f32 - hard.band_low) * hard.inv_band_width).clamp(0.0, 1.0);
            let above =
                ((0.15f32 + f32::EPSILON - hard.band_low) * hard.inv_band_width).clamp(0.0, 1.0);
            assert_eq!(at, 0.0);
            assert_eq!(above, 1.0);
        }
    }

    /// A debug device plus the two interfaces that can count what it still
    /// owns, or `None` on a machine without the D3D11 SDK debug layer —
    /// which is most machines that are not set up for graphics development,
    /// so this skips rather than fails there.
    fn try_debug_device() -> Option<(
        ID3D11Device,
        Arc<Mutex<ID3D11DeviceContext>>,
        ID3D11Debug,
        ID3D11InfoQueue,
    )> {
        let mut device = None;
        let mut context = None;
        // SAFETY: adapter/software pointers are intentionally null for the
        // hardware driver path, feature-level defaults are requested, and
        // `device`/`context` are live correctly typed out-parameters.
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
        let device = device.expect("D3D11CreateDevice succeeded without producing a device");
        let context = context.expect("D3D11CreateDevice succeeded without producing a context");
        let debug = device.cast::<ID3D11Debug>().ok()?;
        let info = device.cast::<ID3D11InfoQueue>().ok()?;
        // The report is one message per live object and can exceed the
        // queue's default limit on its own.
        // SAFETY: `info` is the live debug info queue for this device and the
        // count is an unrestricted scalar limit.
        unsafe { info.SetMessageCountLimit(u64::MAX) }.ok()?;
        Some((device, Arc::new(Mutex::new(context)), debug, info))
    }

    /// How many objects the device still owns. Only the trend across
    /// identical work means anything — the device, its context, and this
    /// element's own construction-time state are always counted.
    fn live_objects(debug: &ID3D11Debug, info: &ID3D11InfoQueue) -> u64 {
        // SAFETY: both interfaces belong to the same live debug device; the
        // report synchronously appends messages before their count is read.
        unsafe {
            info.ClearStoredMessages();
            debug
                .ReportLiveDeviceObjects(D3D11_RLDO_DETAIL)
                .expect("ReportLiveDeviceObjects");
            info.GetNumStoredMessages()
        }
    }

    /// Regression test. Dropping the last reference to a D3D11 object only
    /// queues it for destruction; the device keeps it until the context is
    /// flushed. This element builds three objects per frame — an output
    /// texture, its RTV, and the input's SRV — so before `key` flushed, a
    /// long-running pipeline grew the device's object count by exactly
    /// three per frame and never gave any of it back.
    ///
    /// Asserting the count is flat, rather than merely bounded, is what
    /// makes this catch a regression: the leak it guards against is
    /// per-frame, so any reintroduction shows up as a slope no threshold
    /// would hide.
    #[test]
    fn keying_frames_does_not_accumulate_d3d11_objects() {
        let Some((device, context, debug, info)) = try_debug_device() else {
            return;
        };
        let texture = bgra_texture(&device, 64, 64, [0, 255, 0, 255], 1);
        let mut key = D3d11ChromaKey::new("key", &device, context, default_options())
            .expect("D3d11ChromaKey::new should succeed");
        let received = capture(&mut key);

        let mut push = |index: i64| {
            key.consume(frame(texture.clone(), 64, 64, index))
                .expect("keying must succeed");
            // The pushed frame owns the output texture; releasing it here
            // is what an ordinary downstream element does per frame.
            received.lock().unwrap().clear();
        };

        // The first frames also build whatever the driver lazily creates on
        // first draw, so the baseline is taken after them rather than
        // before.
        for index in 0..20 {
            push(index);
        }
        let baseline = live_objects(&debug, &info);

        for index in 20..120 {
            push(index);
        }
        let after = live_objects(&debug, &info);

        assert_eq!(
            after,
            baseline,
            "100 more keyed frames left {} extra D3D11 objects on the device; \
             per-frame views are not being destroyed",
            after as i64 - baseline as i64
        );
    }
}
