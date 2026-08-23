//! Process-wide D3D12 state shared by every window: the device, the
//! command queue every window presents through, and the root
//! signature/pipeline state objects `d3d12_window_renderer` draws with.
//! Kept separate from any one window so opening a second window doesn't
//! pay for shader compilation again — mirrors why the `renderer-engine`
//! git dependency this replaces split things the same way.

use std::{ffi::c_void, mem::ManuallyDrop};

use windows::{
    Win32::Graphics::{
        Direct3D::{D3D_FEATURE_LEVEL_11_0, Fxc::*, ID3DBlob, ID3DInclude},
        Direct3D12::*,
        Dxgi::{
            CreateDXGIFactory2, DXGI_CREATE_FACTORY_DEBUG, DXGI_CREATE_FACTORY_FLAGS, IDXGIFactory4,
        },
    },
    core::{Error, Result, s},
};

const FRAME_SHADER_SOURCE: &[u8] = include_bytes!("shaders/d3d12/frame.hlsl");
const NV12_SHADER_SOURCE: &[u8] = include_bytes!("shaders/d3d12/nv12.hlsl");

/// Owns the `ID3D12Device` and everything derived from it that every
/// window's renderer shares read-only: the DXGI factory (used to create
/// each window's own swap chain), the single `DIRECT` command queue every
/// window presents through (a process/driver has a limited number of
/// these — one shared queue, not one per window), and the root
/// signature/PSOs for the two pixel formats [`crate::D3d12WindowRenderer`]
/// knows how to draw (`Pixel::YUV420P` CPU-uploaded planes,
/// `Pixel::D3D12`/NV12 zero-copy textures).
pub struct D3d12GpuContext {
    pub(crate) factory: IDXGIFactory4,
    pub(crate) device: ID3D12Device,
    pub(crate) command_queue: ID3D12CommandQueue,
    pub(crate) root_signature: ID3D12RootSignature,
    pub(crate) nv12_pipeline: ID3D12PipelineState,
}

impl D3d12GpuContext {
    /// The shared D3D12 device — pass into
    /// [`media_pp::elements::D3d12Decoder::new`]/
    /// [`media_pp::elements::D3d12Upload::new`] so decoded/uploaded frames
    /// land on the same device [`crate::d3d12_window_renderer`] draws with,
    /// required for their zero-copy path to be valid at all.
    pub fn device(&self) -> &ID3D12Device {
        &self.device
    }

    /// Creates the D3D12 device and shared shader pipeline in dependency
    /// order. One `D3d12GpuContext` per process is enough — every window's
    /// renderer clones (COM ref-count bump, not a deep copy) what it
    /// needs from this.
    pub fn new() -> Result<Self> {
        unsafe {
            #[cfg(debug_assertions)]
            {
                let mut debug: Option<ID3D12Debug> = None;
                D3D12GetDebugInterface(&mut debug)?;
                debug.unwrap().EnableDebugLayer();
            }

            let factory_flags = if cfg!(debug_assertions) {
                DXGI_CREATE_FACTORY_DEBUG
            } else {
                DXGI_CREATE_FACTORY_FLAGS(0)
            };
            let factory: IDXGIFactory4 = CreateDXGIFactory2(factory_flags)?;

            let mut device = None;
            D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut device)?;
            let device: ID3D12Device = device.unwrap();

            let queue_desc = D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            };
            let command_queue: ID3D12CommandQueue = device.CreateCommandQueue(&queue_desc)?;

            let vertex_shader = compile_shader(FRAME_SHADER_SOURCE, s!("vs_main"), s!("vs_5_1"))?;
            // Its own source file, so its texture registers are declared
            // exactly once — see d3d12/nv12.hlsl for why.
            let nv12_shader = compile_shader(NV12_SHADER_SOURCE, s!("ps_nv12"), s!("ps_5_1"))?;

            // Extract the serialized root signature from HLSL's
            // [RootSignature] attribute.
            let root_blob = D3DGetBlobPart(
                vertex_shader.GetBufferPointer(),
                vertex_shader.GetBufferSize(),
                D3D_BLOB_ROOT_SIGNATURE,
                0,
            )?;
            let root_bytes = std::slice::from_raw_parts(
                root_blob.GetBufferPointer().cast::<u8>(),
                root_blob.GetBufferSize(),
            );
            let root_signature: ID3D12RootSignature = device.CreateRootSignature(0, root_bytes)?;

            let nv12_pipeline =
                create_pipeline(&device, &root_signature, &vertex_shader, &nv12_shader)?;

            Ok(Self {
                factory,
                device,
                command_queue,
                root_signature,
                nv12_pipeline,
            })
        }
    }
}

unsafe fn compile_shader(
    source: &[u8],
    entry: windows::core::PCSTR,
    target: windows::core::PCSTR,
) -> Result<ID3DBlob> {
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
            s!("d3d12_shader.hlsl"),
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
        return Err(Error::new(windows::Win32::Foundation::E_FAIL, message));
    }
    Ok(shader.unwrap())
}

unsafe fn create_pipeline(
    device: &ID3D12Device,
    root_signature: &ID3D12RootSignature,
    vertex_shader: &ID3DBlob,
    pixel_shader: &ID3DBlob,
) -> Result<ID3D12PipelineState> {
    let render_target_blend = D3D12_RENDER_TARGET_BLEND_DESC {
        SrcBlend: D3D12_BLEND_ONE,
        DestBlend: D3D12_BLEND_ZERO,
        BlendOp: D3D12_BLEND_OP_ADD,
        SrcBlendAlpha: D3D12_BLEND_ONE,
        DestBlendAlpha: D3D12_BLEND_ZERO,
        BlendOpAlpha: D3D12_BLEND_OP_ADD,
        LogicOp: D3D12_LOGIC_OP_NOOP,
        RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
        ..Default::default()
    };
    let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: ManuallyDrop::new(Some(root_signature.clone())),
        VS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: unsafe { vertex_shader.GetBufferPointer() },
            BytecodeLength: unsafe { vertex_shader.GetBufferSize() },
        },
        PS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: unsafe { pixel_shader.GetBufferPointer() },
            BytecodeLength: unsafe { pixel_shader.GetBufferSize() },
        },
        BlendState: D3D12_BLEND_DESC {
            RenderTarget: [render_target_blend; 8],
            ..Default::default()
        },
        SampleMask: u32::MAX,
        RasterizerState: D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID,
            CullMode: D3D12_CULL_MODE_NONE,
            DepthClipEnable: true.into(),
            ..Default::default()
        },
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 1,
        RTVFormats: {
            let mut formats = [Default::default(); 8];
            formats[0] = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM;
            formats
        },
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        ..Default::default()
    };
    let pipeline = unsafe { device.CreateGraphicsPipelineState(&desc) };
    unsafe { ManuallyDrop::drop(&mut desc.pRootSignature) };
    pipeline
}
