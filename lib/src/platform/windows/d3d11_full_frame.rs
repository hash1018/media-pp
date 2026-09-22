//! One pixel shader drawn over a whole BGRA frame — what `D3d11ChromaKey`,
//! `D3d11VideoEffect` and `D3d11ToneMap` each are, apart from their shader
//! and the constants it reads.
//!
//! Each shader file brings its own `vs_main`, a screen-covering triangle
//! from `SV_VertexID` with no vertex buffer, and a pixel shader reading
//! `register(b0)` for its constants, `register(s0)` for a point sampler and
//! `register(t0)` onward for its inputs.

use std::{ffi::c_void, marker::PhantomData};

use windows::{
    Win32::Graphics::{
        Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
        Direct3D11::*,
        Dxgi::Common::{DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_SAMPLE_DESC},
    },
    core::{PCSTR, s},
};

use super::d3d11::compile_shader;

/// Every piece of D3D11 state a full-frame draw re-selects, built once,
/// with a constant buffer sized for `C`.
pub(crate) struct FullFramePass<C> {
    vertex_shader: ID3D11VertexShader,
    pixel_shader: ID3D11PixelShader,
    sampler: ID3D11SamplerState,
    blend_state: ID3D11BlendState,
    rasterizer_state: ID3D11RasterizerState,
    constant_buffer: ID3D11Buffer,
    constants: PhantomData<C>,
}

impl<C: Copy> FullFramePass<C> {
    /// Compiles `source`'s `vs_main` and `pixel_entry`, naming `file_name`
    /// in any compiler diagnostic, and builds the state the draw uses:
    ///
    /// - point sampling, clamped — input and output are the same size, so
    ///   every output pixel is exactly one input texel, and filtering would
    ///   only blend neighbours an element such as a key means to keep apart;
    /// - blending off — what the shader computes, alpha included, is the
    ///   output, not something to mix with the render target;
    /// - scissoring off — the draw covers the whole target by construction,
    ///   and the shared context may carry someone else's scissor rect.
    pub(crate) fn new(
        device: &ID3D11Device,
        source: &[u8],
        file_name: PCSTR,
        pixel_entry: PCSTR,
    ) -> windows::core::Result<Self> {
        // SAFETY: the device is live; compiler blobs retain their bytecode
        // while shader creation reads it, every descriptor is fully
        // initialized, and each optional interface slot is a live
        // out-parameter. No call retains a borrowed Rust pointer after
        // returning.
        unsafe {
            let vertex_bytecode = compile_shader(source, file_name, s!("vs_main"), s!("vs_5_0"))?;
            let pixel_bytecode = compile_shader(source, file_name, pixel_entry, s!("ps_5_0"))?;

            let mut vertex_shader = None;
            device.CreateVertexShader(
                std::slice::from_raw_parts(
                    vertex_bytecode.GetBufferPointer().cast::<u8>(),
                    vertex_bytecode.GetBufferSize(),
                ),
                None,
                Some(&mut vertex_shader),
            )?;
            let mut pixel_shader = None;
            device.CreatePixelShader(
                std::slice::from_raw_parts(
                    pixel_bytecode.GetBufferPointer().cast::<u8>(),
                    pixel_bytecode.GetBufferSize(),
                ),
                None,
                Some(&mut pixel_shader),
            )?;

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

            let mut blend_desc = D3D11_BLEND_DESC::default();
            blend_desc.RenderTarget[0] = D3D11_RENDER_TARGET_BLEND_DESC {
                BlendEnable: false.into(),
                RenderTargetWriteMask: D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8,
                ..Default::default()
            };
            let mut blend_state = None;
            device.CreateBlendState(&blend_desc, Some(&mut blend_state))?;

            let rasterizer_desc = D3D11_RASTERIZER_DESC {
                FillMode: D3D11_FILL_SOLID,
                CullMode: D3D11_CULL_NONE,
                ScissorEnable: false.into(),
                DepthClipEnable: true.into(),
                ..Default::default()
            };
            let mut rasterizer_state = None;
            device.CreateRasterizerState(&rasterizer_desc, Some(&mut rasterizer_state))?;

            let buffer_desc = D3D11_BUFFER_DESC {
                ByteWidth: std::mem::size_of::<C>() as u32,
                Usage: D3D11_USAGE_DEFAULT,
                BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
                CPUAccessFlags: 0,
                MiscFlags: 0,
                StructureByteStride: 0,
            };
            let mut constant_buffer = None;
            device.CreateBuffer(&buffer_desc, None, Some(&mut constant_buffer))?;

            Ok(Self {
                vertex_shader: vertex_shader
                    .expect("CreateVertexShader succeeded without a shader"),
                pixel_shader: pixel_shader.expect("CreatePixelShader succeeded without a shader"),
                sampler: sampler.expect("CreateSamplerState succeeded without a state"),
                blend_state: blend_state.expect("CreateBlendState succeeded without a state"),
                rasterizer_state: rasterizer_state
                    .expect("CreateRasterizerState succeeded without a state"),
                constant_buffer: constant_buffer.expect("CreateBuffer succeeded without a buffer"),
                constants: PhantomData,
            })
        }
    }

    /// Draws the shader over all of `target`, `width`x`height`, reading
    /// `constants` and `inputs` from `t0` on, then unbinds what it bound and
    /// flushes.
    ///
    /// Every piece of state is re-selected rather than assumed: `context` is
    /// the immediate context every D3D11 element in the pipeline shares, and
    /// whatever drew last left its own bound. What this bound is released
    /// on the way out, so this frame's views do not live on inside that
    /// shared state.
    ///
    /// The flush is what keeps the device from filling up. Dropping the last
    /// reference to a D3D11 object does not destroy it — destruction waits
    /// for the context to be flushed — and an element drawing this way makes
    /// an output texture and its views every frame. `D3d11ChromaKey`
    /// measured three objects a frame left on the device without it, and a
    /// flat count with it. The views cannot be cached instead: an input's is
    /// built over a texture the element does not own, whose address upstream
    /// frees and reuses.
    ///
    /// # Safety
    ///
    /// `context` must be the immediate context of the device this pass and
    /// every view were made on, held exclusively for the call — the lock an
    /// element takes on the shared context.
    pub(crate) unsafe fn draw(
        &self,
        context: &ID3D11DeviceContext,
        constants: &C,
        target: &ID3D11RenderTargetView,
        inputs: &[Option<ID3D11ShaderResourceView>],
        width: u32,
        height: u32,
    ) {
        let viewport = D3D11_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: width as f32,
            Height: height as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };
        let unbound: Vec<Option<ID3D11ShaderResourceView>> = vec![None; inputs.len()];
        // SAFETY: the caller's promise — a live immediate context of this
        // pass's device, held exclusively. Every state object and view is
        // live and on that device, and `constants` is readable for the
        // buffer's size, which `new` made `size_of::<C>()`.
        unsafe {
            context.UpdateSubresource(
                &self.constant_buffer,
                0,
                None,
                (constants as *const C).cast::<c_void>(),
                0,
                0,
            );
            context.OMSetRenderTargets(Some(&[Some(target.clone())]), None);
            context.OMSetBlendState(&self.blend_state, None, 0xffff_ffff);
            context.RSSetState(&self.rasterizer_state);
            context.RSSetViewports(Some(&[viewport]));
            context.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            context.IASetInputLayout(None);
            context.VSSetShader(&self.vertex_shader, None);
            context.PSSetShader(&self.pixel_shader, None);
            context.PSSetSamplers(0, Some(&[Some(self.sampler.clone())]));
            context.PSSetConstantBuffers(0, Some(&[Some(self.constant_buffer.clone())]));
            context.PSSetShaderResources(0, Some(inputs));
            context.Draw(3, 0);

            context.PSSetShaderResources(0, Some(&unbound));
            context.OMSetRenderTargets(None, None);
            context.Flush();
        }
    }
}

/// A BGRA texture of `width`x`height` to draw into, and a render target view
/// of it. Shader resource as well as render target: what is drawn here is
/// sampled by whatever comes next — a compositor layer, a renderer, another
/// shader-based filter.
pub(crate) fn create_bgra_target(
    device: &ID3D11Device,
    width: u32,
    height: u32,
) -> windows::core::Result<(ID3D11Texture2D, ID3D11RenderTargetView)> {
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
    let mut view = None;
    // SAFETY: `desc` fully describes a render-target texture, no initial data
    // is supplied, and both slots are live out-parameters; the view is made
    // over the texture just created.
    unsafe {
        device.CreateTexture2D(&desc, None, Some(&mut texture))?;
        let texture = texture
            .as_ref()
            .expect("CreateTexture2D succeeded without producing a texture");
        device.CreateRenderTargetView(texture, None, Some(&mut view))?;
    }
    Ok((
        texture.expect("CreateTexture2D succeeded without producing a texture"),
        view.expect("CreateRenderTargetView succeeded without a view"),
    ))
}
