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
        Dxgi::Common::{
            DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_NV12, DXGI_FORMAT_P010,
            DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8G8_UNORM, DXGI_FORMAT_R16_UNORM,
            DXGI_FORMAT_R16G16_UNORM, DXGI_SAMPLE_DESC,
        },
    },
    core::{Interface, s},
};

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
    tone_map::ToneMap,
};

const SHADER_SOURCE: &[u8] = include_bytes!("../../../../shaders/d3d11/tone_map.hlsl");

/// Errors specific to [`D3d11ToneMap`]. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum D3d11ToneMapError {
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
    #[error("D3d11ToneMap only takes Pixel::D3D11 frames, got {0:?}")]
    UnsupportedFormat(ffmpeg::format::Pixel),

    /// A frame tagged as D3D11 carries no valid texture.
    #[error("frame claimed the D3D11 pixel format but carries no texture")]
    InvalidD3d11Frame,

    /// The frame does not say it is PQ or HLG, so there is nothing to say
    /// what its numbers mean in light.
    #[error("D3d11ToneMap brings PQ or HLG video to SDR, and this frame is tagged {0:?}")]
    NotHdr(ffmpeg::color::TransferCharacteristic),

    /// The texture is neither P010 nor NV12.
    #[error("D3d11ToneMap takes DXGI_FORMAT_P010 or DXGI_FORMAT_NV12 textures, got {0:?}")]
    UnsupportedTextureFormat(DXGI_FORMAT),

    /// The input texture belongs to another D3D11 device.
    #[error(
        "a Pixel::D3D11 frame's texture lives on a different ID3D11Device than this \
         D3d11ToneMap was created with — every D3D11 element in one pipeline must share \
         exactly one device"
    )]
    DeviceMismatch,

    /// The supplied immediate context belongs to another D3D11 device.
    #[error(
        "the supplied ID3D11DeviceContext belongs to a different ID3D11Device than this \
         D3d11ToneMap"
    )]
    ContextDeviceMismatch,

    /// The backing texture is smaller than the visible input frame.
    #[error(
        "D3D11 texture is {actual_width}x{actual_height}, smaller than the frame's \
         {expected_width}x{expected_height} visible size"
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

    /// The texture was not created with shader-resource binding.
    #[error(
        "the input texture was created without D3D11_BIND_SHADER_RESOURCE (BindFlags={0:#x}), \
         so this element's pixel shader cannot read it"
    )]
    MissingShaderResourceBind(u32),

    /// The visible input dimensions are zero or odd, which 4:2:0 cannot be.
    #[error("frame has invalid 4:2:0 dimensions {width}x{height}")]
    InvalidFrameDimensions {
        /// Invalid frame width in pixels.
        width: u32,
        /// Invalid frame height in pixels.
        height: u32,
    },

    /// The sink received a buffer other than decoded video or end-of-stream.
    #[error("D3d11ToneMap only accepts Video and Eos buffers, got a {0}")]
    UnsupportedBuffer(&'static str),
}

/// The per-draw constant buffer `tone_map.hlsl` reads: [`ToneMap`], which is
/// laid out for exactly this, and the visible fraction of the texture.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ToneMapConstants {
    tone_map: ToneMap,
    uv_scale: [f32; 2],
    _padding: [f32; 2],
}

/// Brings a GPU-resident `Pixel::D3D11` HDR frame — P010, or NV12, tagged PQ
/// or HLG — into SDR BT.709 BGRA through a pixel shader, without the frame
/// leaving video memory.
///
/// What it does to each pixel is `core/tone_map.rs`'s definition:
/// BT.2020's matrix, the frame's transfer into nits, BT.2020's primaries
/// into BT.709's, BT.2390's EETF from 1000 nits down to 203 on the largest
/// channel, and 203 nits as SDR white, encoded with gamma 2.2. A D3D11 video
/// processor would be the obvious place for this and is not one: the RTX
/// 3050's reports no conversion from PQ or HLG to SDR RGB at all.
///
/// A frame is read by its own transfer tag, and one that says neither PQ nor
/// HLG is refused with [`D3d11ToneMapError::NotHdr`] rather than guessed at.
/// The output is the visible size of the input, tagged full-range RGB with
/// BT.709 primaries, with the input's PTS. `device` and `context` are the
/// ones every other D3D11 element in the pipeline shares; state is
/// re-selected on every draw and the context flushed after it, for
/// `D3d11VideoEffect`'s reasons. A repeated input texture is answered with
/// the output already made from it.
pub struct D3d11ToneMap {
    pp_log: PpLog,
    name: Arc<str>,
    device: ID3D11Device,
    context: Arc<Mutex<ID3D11DeviceContext>>,
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
    /// [`RepeatedOutput`].
    repeated: RepeatedOutput,
}

// SAFETY: every field is either a `windows-rs` COM interface wrapper (the
// device-level ones free-threaded, the context-level ones behind `context`'s
// own `Mutex`) or plain data, and `&mut self` on every method that touches the
// rest rules out concurrent access — the reasoning `D3d11VideoEffect` gives.
unsafe impl Send for D3d11ToneMap {}

/// One validated input: the texture, which slice of it, the view formats
/// of its two planes, the visible size, the fraction of the texture that
/// covers, and how to bring it down.
struct ValidatedInput {
    texture: ID3D11Texture2D,
    array_slice: u32,
    planes: (DXGI_FORMAT, DXGI_FORMAT),
    width: u32,
    height: u32,
    uv_scale: [f32; 2],
    tone_map: ToneMap,
}

impl D3d11ToneMap {
    /// `device` must be the same `ID3D11Device`, and `context` the same
    /// shared immediate context, every other D3D11 element in this pipeline
    /// uses. Every frame comes out the size it went in.
    pub fn new(
        name: impl Into<String>,
        device: &ID3D11Device,
        context: Arc<Mutex<ID3D11DeviceContext>>,
    ) -> std::result::Result<Self, D3d11ToneMapError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::D3d11ToneMap, &name, None);
        protect_shared_device(device)?;
        {
            let context = context
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            // SAFETY: `context` is a live immediate-context interface;
            // `GetDevice` returns an owned reference to its creating device.
            let context_device = unsafe { context.GetDevice() }?;
            if context_device.as_raw() != device.as_raw() {
                return Err(D3d11ToneMapError::ContextDeviceMismatch);
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
        pp_info!(pp_log: &pp_log, "created");
        Ok(Self {
            name,
            pp_log,
            device: device.clone(),
            context,
            vertex_shader,
            pixel_shader,
            sampler,
            blend_state,
            rasterizer_state,
            constant_buffer,
            pad,
            pool: UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {}),
            repeated: RepeatedOutput::new(),
        })
    }

    /// Rejects anything this element cannot draw from, so a bad frame fails
    /// here naming the problem rather than as a `Draw` sampling nothing.
    fn validate(
        &self,
        frame: &ffmpeg::frame::Video,
    ) -> std::result::Result<ValidatedInput, D3d11ToneMapError> {
        if frame.format() != ffmpeg::format::Pixel::D3D11 {
            return Err(D3d11ToneMapError::UnsupportedFormat(frame.format()));
        }
        let tone_map = ToneMap::of_frame(frame).ok_or(D3d11ToneMapError::NotHdr(
            frame.color_transfer_characteristic(),
        ))?;
        let (width, height) = (frame.width(), frame.height());
        if width == 0 || height == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Err(D3d11ToneMapError::InvalidFrameDimensions { width, height });
        }
        let (texture_raw, index) =
            d3d11va_texture(frame).ok_or(D3d11ToneMapError::InvalidD3d11Frame)?;
        // SAFETY: `texture_raw` is a borrowed raw `ID3D11Texture2D*` still
        // owned by `frame`'s buffer; `.clone()` (`AddRef`) gives an
        // independently ref-counted handle.
        let texture = unsafe {
            ID3D11Texture2D::from_raw_borrowed(&texture_raw)
                .ok_or(D3d11ToneMapError::InvalidD3d11Frame)?
                .clone()
        };
        // SAFETY: `texture` is a live cloned COM interface; `GetDevice`
        // returns an owned reference to its creating device.
        let texture_device = unsafe { texture.GetDevice() }?;
        if texture_device.as_raw() != self.device.as_raw() {
            return Err(D3d11ToneMapError::DeviceMismatch);
        }

        let mut desc = D3D11_TEXTURE2D_DESC::default();
        // SAFETY: `desc` is a live out-parameter for the live texture.
        unsafe { texture.GetDesc(&mut desc) };
        let planes = match desc.Format {
            DXGI_FORMAT_P010 => (DXGI_FORMAT_R16_UNORM, DXGI_FORMAT_R16G16_UNORM),
            DXGI_FORMAT_NV12 => (DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8G8_UNORM),
            other => return Err(D3d11ToneMapError::UnsupportedTextureFormat(other)),
        };
        if desc.Width < width || desc.Height < height {
            return Err(D3d11ToneMapError::TextureTooSmall {
                actual_width: desc.Width,
                actual_height: desc.Height,
                expected_width: width,
                expected_height: height,
            });
        }
        if index < 0 || index as u64 >= u64::from(desc.ArraySize) {
            return Err(D3d11ToneMapError::InvalidArrayIndex {
                index,
                array_size: desc.ArraySize,
            });
        }
        if desc.BindFlags & D3D11_BIND_SHADER_RESOURCE.0 as u32 == 0 {
            return Err(D3d11ToneMapError::MissingShaderResourceBind(desc.BindFlags));
        }

        Ok(ValidatedInput {
            texture,
            array_slice: index as u32,
            planes,
            width,
            height,
            uv_scale: [
                width as f32 / desc.Width as f32,
                height as f32 / desc.Height as f32,
            ],
            tone_map,
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
            .map_err(D3d11ToneMapError::from)?;
        let mut output_view = None;
        // SAFETY: `output` is a live render-target-capable texture and
        // `output_view` is the correctly typed live out-parameter.
        unsafe {
            self.device
                .CreateRenderTargetView(&output, None, Some(&mut output_view))
                .inspect_err(|error| pp_error!(self, "failed to create the output RTV: {error}"))
                .map_err(D3d11ToneMapError::from)?;
        }
        let output_view = output_view.expect("CreateRenderTargetView succeeded without a view");

        let mut views = [None, None];
        for (view, format) in views.iter_mut().zip([input.planes.0, input.planes.1]) {
            let desc = D3D11_SHADER_RESOURCE_VIEW_DESC {
                Format: format,
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
            // SAFETY: validation established the texture format, which is
            // what these plane formats view, and bounded the array slice;
            // `view` is a live out-parameter.
            unsafe {
                self.device
                    .CreateShaderResourceView(&input.texture, Some(&desc), Some(view))
                    .inspect_err(|error| pp_error!(self, "failed to create a plane SRV: {error}"))
                    .map_err(D3d11ToneMapError::from)?;
            }
        }

        let constants = ToneMapConstants {
            tone_map: input.tone_map,
            uv_scale: input.uv_scale,
            _padding: [0.0; 2],
        };
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
                context.PSSetShaderResources(0, Some(&views));
                context.Draw(3, 0);

                context.PSSetShaderResources(0, Some(&[None, None]));
                context.OMSetRenderTargets(None, None);
                context.Flush();
            }
        }

        let mut drawn = self.pool.get();
        *drawn = wrap_d3d11_texture(output, input.width, input.height)?;
        drawn.set_pts(frame.pts());
        drawn.set_color_space(ffmpeg::color::Space::RGB);
        drawn.set_color_range(ffmpeg::color::Range::JPEG);
        drawn.set_color_primaries(ffmpeg::color::Primaries::BT709);
        Ok(drawn)
    }
}

impl PerFrameTransform for D3d11ToneMap {
    fn repeated(&mut self) -> &mut RepeatedOutput {
        &mut self.repeated
    }

    fn frame_ref_failed(&self, code: i32) -> crate::error::Error {
        pp_error!(self, "av_frame_ref failed: {code}");
        D3d11ToneMapError::FrameRef(code).into()
    }

    fn produce(
        &mut self,
        frame: &Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        self.draw(frame)
    }
}

impl Element for D3d11ToneMap {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::D3d11ToneMap
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for D3d11ToneMap {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for D3d11ToneMap {
    /// Drawn on the GPU from a D3D11 texture.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::frame(
            MediaKind::VideoFrame,
            MemoryDomain::D3d11,
        ))
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Video(frame) => {
                let drawn = self.transform(&frame)?;
                self.pad.push(MediaBuffer::Video(drawn))
            }
            // One `Draw` per frame, pushed before `consume` returns, so
            // there is nothing to drain.
            MediaBuffer::Eos => self.pad.push(MediaBuffer::Eos),
            other => {
                let kind = other.kind();
                pp_error!(self, "unsupported buffer: {kind}");
                Err(D3d11ToneMapError::UnsupportedBuffer(kind).into())
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
            s!("tone_map.hlsl"),
            s!("vs_main"),
            s!("vs_5_0"),
        )?;
        let pixel_bytecode = compile_shader(
            SHADER_SOURCE,
            s!("tone_map.hlsl"),
            s!("ps_tone_map"),
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

        // Point sampling: output and input are the same size, and chroma is
        // read at the sample that covers each pixel, as the NV12 kernels do.
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

        let mut blend_desc = D3D11_BLEND_DESC::default();
        blend_desc.RenderTarget[0] = D3D11_RENDER_TARGET_BLEND_DESC {
            BlendEnable: false.into(),
            RenderTargetWriteMask: D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8,
            ..Default::default()
        };
        let mut blend_state = None;
        device.CreateBlendState(&blend_desc, Some(&mut blend_state))?;
        let blend_state = blend_state.expect("CreateBlendState succeeded without a state");

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
            ByteWidth: std::mem::size_of::<ToneMapConstants>() as u32,
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
    use windows::Win32::Graphics::Direct3D11::{
        D3D11_CREATE_DEVICE_DEBUG, D3D11_RLDO_DETAIL, D3D11_SDK_VERSION, D3D11_SUBRESOURCE_DATA,
        D3D11CreateDevice, ID3D11Debug, ID3D11InfoQueue,
    };

    use super::*;
    use crate::elements::D3d11Download;
    use crate::test_support::try_d3d11_device as try_device;
    use crate::tone_map::HdrTransfer;

    /// A flat P010 texture: every luma sample `luma` and every chroma pair
    /// `(cb, cr)`, as 10-bit codes.
    fn p010_texture(device: &ID3D11Device, luma: u16, cb: u16, cr: u16) -> ID3D11Texture2D {
        let (width, height) = (32u32, 32u32);
        let samples: Vec<u16> = std::iter::repeat_n(luma << 6, (width * height) as usize)
            .chain(std::iter::repeat_n([cb << 6, cr << 6], (width * height / 4) as usize).flatten())
            .collect();
        let initial = D3D11_SUBRESOURCE_DATA {
            pSysMem: samples.as_ptr().cast::<c_void>(),
            SysMemPitch: width * 2,
            SysMemSlicePitch: 0,
        };
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_P010,
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
        // SAFETY: `initial` points at `samples`, both planes at the declared
        // pitch, alive for the call; `texture` is a live out-parameter.
        unsafe {
            device
                .CreateTexture2D(&desc, Some(&initial), Some(&mut texture))
                .expect("CreateTexture2D(P010)");
        }
        texture.expect("a texture")
    }

    /// Draws one such texture tagged with `transfer` and reads back the
    /// pixel in the middle as R, G, B.
    fn drawn(
        transfer: ffmpeg::color::TransferCharacteristic,
        (luma, cb, cr): (u16, u16, u16),
    ) -> Option<std::result::Result<[u8; 3], crate::error::Error>> {
        let (device, context) = try_device()?;
        let mut frame =
            wrap_d3d11_texture(p010_texture(&device, luma, cb, cr), 32, 32).expect("wrap");
        frame.set_color_space(ffmpeg::color::Space::BT2020NCL);
        frame.set_color_range(ffmpeg::color::Range::MPEG);
        frame.set_color_transfer_characteristic(transfer);
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut slot = pool.get();
        *slot = frame;

        let mut tone_map = D3d11ToneMap::new("tone-map", &device, context.clone()).expect("new");
        let download = D3d11Download::new("download", &device, context, 32, 32).expect("download");
        let received = Arc::new(Mutex::new(Vec::new()));
        let collected = received.clone();
        let mut download = download;
        download.src_pads()[0].link(Box::new(crate::elements::AppSink::new(
            "out",
            move |buffer| {
                collected.lock().unwrap().push(buffer);
                Ok(())
            },
        )));
        tone_map.src_pads()[0].link(Box::new(download));
        if let Err(error) = tone_map.consume(MediaBuffer::Video(Arc::new(slot))) {
            return Some(Err(error));
        }
        let received = received.lock().unwrap();
        let MediaBuffer::Video(out) = &received[0] else {
            panic!("a Video buffer");
        };
        assert_eq!(out.color_space(), ffmpeg::color::Space::RGB);
        let [b, g, r, _] = out.data(0)[out.stride(0) * 16 + 16 * 4..][..4] else {
            unreachable!("four bytes were sliced");
        };
        Some(Ok([r, g, b]))
    }

    /// What the shared definition makes of those codes, sampled as a
    /// shader samples a 16-bit plane.
    fn expected(transfer: HdrTransfer, (luma, cb, cr): (u16, u16, u16)) -> [u8; 3] {
        let unit = |code: u16| f32::from(code << 6) / 65535.0;
        ToneMap::new(transfer, ffmpeg::color::Range::MPEG, None).apply(
            unit(luma),
            unit(cb),
            unit(cr),
        )
    }

    fn assert_near(got: [u8; 3], want: [u8; 3], what: &str) {
        assert!(
            got.iter()
                .zip(want)
                .all(|(got, want)| got.abs_diff(want) <= 3),
            "{what}: got {got:?}, want {want:?}"
        );
    }

    /// PQ greys and a colour, and HLG, each land where `core/tone_map.rs`
    /// says: the shader and the Rust definition are one definition.
    #[test]
    fn hdr_is_drawn_as_the_shared_definition_says() {
        use ffmpeg::color::TransferCharacteristic::{ARIB_STD_B67, SMPTE2084};
        for (transfer, tag, codes, what) in [
            // About 100 nits, below the knee.
            (HdrTransfer::Pq, SMPTE2084, (500, 512, 512), "PQ grey"),
            // About 1000 nits, the peak.
            (HdrTransfer::Pq, SMPTE2084, (720, 512, 512), "PQ peak"),
            (HdrTransfer::Pq, SMPTE2084, (520, 420, 700), "PQ colour"),
            (HdrTransfer::Hlg, ARIB_STD_B67, (720, 512, 512), "HLG grey"),
            (
                HdrTransfer::Hlg,
                ARIB_STD_B67,
                (600, 450, 650),
                "HLG colour",
            ),
        ] {
            let Some(got) = drawn(tag, codes) else {
                return;
            };
            assert_near(got.expect(what), expected(transfer, codes), what);
        }
    }

    /// A frame that says nothing of PQ or HLG is refused by name rather than
    /// read as either.
    #[test]
    fn a_frame_that_is_not_hdr_is_refused() {
        let Some(result) = drawn(
            ffmpeg::color::TransferCharacteristic::BT709,
            (500, 512, 512),
        ) else {
            return;
        };
        let error = result.expect_err("an SDR frame is not tone mapped");
        assert!(
            matches!(
                error,
                crate::error::Error::D3d11ToneMapError(D3d11ToneMapError::NotHdr(
                    ffmpeg::color::TransferCharacteristic::BT709
                ))
            ),
            "{error}"
        );
    }

    /// A debug device and the two interfaces that count what it still owns,
    /// or `None` without the D3D11 SDK debug layer — `D3d11VideoEffect`'s
    /// own test's helper.
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

    /// Each draw creates an output texture, its render target view and two
    /// plane views, which the device destroys only once the context is
    /// flushed — `D3d11ChromaKey` leaked three objects a frame before it
    /// flushed. Two textures in turn, so every frame is drawn rather than
    /// answered as a repeat, and a flat count across a hundred frames.
    #[test]
    fn drawing_frames_does_not_accumulate_d3d11_objects() {
        let Some((device, context, debug, info)) = try_debug_device() else {
            return;
        };
        let textures = [
            p010_texture(&device, 500, 512, 512),
            p010_texture(&device, 600, 450, 650),
        ];
        let mut element = D3d11ToneMap::new("tone-map", &device, context).expect("new");
        let drawn = Arc::new(Mutex::new(0usize));
        let counted = drawn.clone();
        element.src_pads()[0].link(Box::new(crate::elements::AppSink::new(
            "out",
            move |buffer| {
                if matches!(buffer, MediaBuffer::Video(_)) {
                    *counted.lock().unwrap() += 1;
                }
                Ok(())
            },
        )));
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut push = |pts: i64| {
            let mut frame =
                wrap_d3d11_texture(textures[pts as usize % 2].clone(), 32, 32).expect("wrap");
            frame.set_pts(Some(pts));
            frame.set_color_space(ffmpeg::color::Space::BT2020NCL);
            frame.set_color_range(ffmpeg::color::Range::MPEG);
            frame.set_color_transfer_characteristic(
                ffmpeg::color::TransferCharacteristic::SMPTE2084,
            );
            let mut slot = pool.get();
            *slot = frame;
            element
                .consume(MediaBuffer::Video(Arc::new(slot)))
                .expect("draw");
        };

        for pts in 0..20 {
            push(pts);
        }
        let baseline = live_objects(&debug, &info);
        for pts in 20..120 {
            push(pts);
        }
        let after = live_objects(&debug, &info);

        assert_eq!(*drawn.lock().unwrap(), 120, "every frame came out");
        assert_eq!(
            after,
            baseline,
            "100 more frames left {} extra D3D11 objects on the device",
            after as i64 - baseline as i64
        );
    }
}
