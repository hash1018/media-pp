//! Process-wide D3D11 state shared by every window *and* every non-render
//! D3D11 element (`DxgiCaptureSource`'s GPU capture mode, `D3d11Upload`,
//! `D3d11Decoder`, `D3d11Scaler`) — the device, its **one** immediate
//! context, and the shader objects [`crate::D3d11WindowRenderer`] draws with.
//! Sharing one device+context across the whole pipeline (not just one per
//! window) is load-bearing, not just convenient: see
//! [`media_pp::elements::D3d11Renderer`]'s
//! own docs on why that's what lets this whole stack skip explicit
//! GPU-side fences entirely, unlike the D3D12 side.

use std::{ffi::c_void, sync::Arc, sync::Mutex};

use windows::{
    Win32::Graphics::{
        Direct3D::{D3D_DRIVER_TYPE_HARDWARE, Fxc::*, ID3DBlob, ID3DInclude},
        Direct3D11::*,
        Dxgi::{CreateDXGIFactory1, IDXGIFactory2},
    },
    core::{Error, Interface, Result, s},
};

const BGRA_SHADER_SOURCE: &[u8] = include_bytes!("shaders/d3d11/bgra.hlsl");
const NV12_SHADER_SOURCE: &[u8] = include_bytes!("shaders/d3d11/nv12.hlsl");

/// Owns the `ID3D11Device`, its one `ID3D11DeviceContext`, and the shader
/// objects every window's renderer shares read-only.
///
/// `context` is behind an `Arc<Mutex<_>>`, not a bare clone, even though
/// individual immediate-context calls are only safe across threads after
/// `ID3D11Multithread` protection is enabled below. What that protection does
/// *not* cover is a multi-call *sequence*: bind-RTV → bind-shaders →
/// bind-SRVs → set-viewport → `Draw` is several calls against context
/// state that's global to the device, not scoped to a command list the
/// way D3D12 has it. Two windows racing their own submit sequences on the
/// same context could interleave and draw window A's frame into window
/// B's render target. The mutex makes each window's whole bind+draw+present
/// sequence atomic with respect to every other window sharing this context.
pub struct D3d11GpuContext {
    pub(crate) factory: IDXGIFactory2,
    pub(crate) device: ID3D11Device,
    pub(crate) context: Arc<Mutex<ID3D11DeviceContext>>,
    pub(crate) vertex_shader: ID3D11VertexShader,
    pub(crate) bgra_pixel_shader: ID3D11PixelShader,
    pub(crate) nv12_pixel_shader: ID3D11PixelShader,
    pub(crate) sampler: ID3D11SamplerState,
}

impl D3d11GpuContext {
    /// The shared D3D11 device — pass into
    /// [`media_pp::elements::D3d11Upload::new`]/
    /// [`media_pp::elements::D3d11Decoder::new`]/
    /// [`media_pp::elements::D3d11Scaler::new`] so every producer/filter
    /// lands on the same device (and, transitively, the same immediate context
    /// — see this type's own docs) [`crate::d3d11_window_renderer`] draws
    /// with. Capture sources can use the same value through
    /// `DxgiCaptureSource::open_with_device` or
    /// `WgcCaptureSource::open_with_device`.
    pub fn device(&self) -> &ID3D11Device {
        &self.device
    }

    /// The shared immediate context, behind the lock every D3D11 window
    /// renderer's submit sequence holds for its full bind+draw+present —
    /// see this type's own docs on why. Pass the same value to
    /// [`media_pp::elements::D3d11Scaler::new`]; every other GPU-resident
    /// capture/upload path that issues `CopySubresourceRegion`/`CopyResource`
    /// against this context must acquire this lock around the copy too.
    pub fn context(&self) -> Arc<Mutex<ID3D11DeviceContext>> {
        self.context.clone()
    }

    /// Creates the shared shader/sampler state on top of a D3D11
    /// device+context, and compiles the shared shaders in dependency
    /// order. One `D3d11GpuContext` per process is enough — every
    /// window's renderer clones (COM ref-count bump, not a deep copy)
    /// what it needs from this.
    ///
    /// `device`: `None` creates a fresh device on the OS's default
    /// adapter (the original behavior — fine when nothing else in the
    /// pipeline cares which adapter that is, e.g. `d3d11_upload`/
    /// `d3d11_decode_render`, which only ever talk to this one device).
    /// `Some(device)` reuses an already-created device as-is instead of
    /// creating a new one — e.g. a device already pinned to the adapter of a
    /// captured monitor.
    /// Reusing that exact device instead of separately creating a
    /// same-adapter one and relying on the two matching is what lets this
    /// whole stack skip a device-mismatch check entirely — there's only
    /// ever one device in play, not two independently resolved ones that
    /// could disagree.
    pub fn new(device: Option<ID3D11Device>) -> Result<Self> {
        unsafe {
            let factory: IDXGIFactory2 = CreateDXGIFactory1()?;

            let (device, context) = match device {
                Some(device) => {
                    let context = device.GetImmediateContext()?;
                    (device, context)
                }
                None => {
                    // BGRA support is harmless for the other examples and is
                    // required when this shared device is injected into WGC.
                    let flags = if cfg!(debug_assertions) {
                        D3D11_CREATE_DEVICE_DEBUG | D3D11_CREATE_DEVICE_BGRA_SUPPORT
                    } else {
                        D3D11_CREATE_DEVICE_BGRA_SUPPORT
                    };
                    let mut device = None;
                    let mut context = None;
                    D3D11CreateDevice(
                        None,
                        D3D_DRIVER_TYPE_HARDWARE,
                        Default::default(),
                        flags,
                        None,
                        D3D11_SDK_VERSION,
                        Some(&mut device),
                        None,
                        Some(&mut context),
                    )?;
                    (device.unwrap(), context.unwrap())
                }
            };
            // `screen_preview_gpu` really
            // does drive this context from two threads: this crate's
            // window renderer and `DxgiCaptureSource`'s own capture thread,
            // via its own `ID3D11DeviceContext` handle obtained from
            // `GetImmediateContext()` on the same shared device — see
            // `CaptureMode::Gpu`'s own docs — and crashed without runtime
            // multithread protection. Setting it explicitly made that stable.
            let _ = context
                .cast::<windows::Win32::Graphics::Direct3D11::ID3D11Multithread>()?
                .SetMultithreadProtected(true);
            let context: Arc<Mutex<ID3D11DeviceContext>> = Arc::new(Mutex::new(context));

            let vertex_bytecode = compile_shader(BGRA_SHADER_SOURCE, s!("vs_main"), s!("vs_5_0"))?;
            let bgra_bytecode = compile_shader(BGRA_SHADER_SOURCE, s!("ps_bgra"), s!("ps_5_0"))?;
            // Compiled from its own source file (not `BGRA_SHADER_SOURCE`)
            // so its `t0`/`t1` register declarations don't collide with
            // `ps_bgra`'s `t0` — see shaders/d3d11/nv12.hlsl for why this
            // doesn't actually matter for D3D11 the way it did for the
            // D3D12 side, kept as separate files anyway for the same
            // readability reasons.
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

            Ok(Self {
                factory,
                device,
                context,
                vertex_shader,
                bgra_pixel_shader,
                nv12_pixel_shader,
                sampler,
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
            s!("d3d11_shader.hlsl"),
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
