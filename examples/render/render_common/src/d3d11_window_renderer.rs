//! A single window's D3D11 renderer — swap chain, the two zero-copy draw
//! paths [`media_pp::elements::D3d11FrameRenderer`] needs, and resize.
//! Reuses [`crate::D3d11GpuContext`]'s device/context/shaders/sampler;
//! everything else here (swap chain, render target view) is this window's
//! own.
//!
//! No fence, no `keep_alive`, no descriptor heaps, no resource-state
//! barriers — see [`media_pp::elements::D3d11Renderer`]'s own docs on why
//! D3D11's single shared immediate context (auto-serialized by the
//! runtime) and driver-deferred resource destruction make all of that
//! D3D12-only machinery unnecessary here. What *does* still need explicit
//! synchronization, and why, is documented on [`crate::D3d11GpuContext`]'s
//! `context` field: every submit here locks that shared context for its
//! whole bind+draw+present sequence so two windows' draws can't interleave.

use std::{
    ffi::c_void,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use media_pp::elements::{D3d11FrameRenderer, SubmitError};
use windows::{
    Win32::{
        Foundation::{HWND, RECT},
        Graphics::{
            Direct3D::{D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, D3D_SRV_DIMENSION_TEXTURE2DARRAY},
            Direct3D11::*,
            Dxgi::{
                Common::{
                    DXGI_ALPHA_MODE_UNSPECIFIED, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R8_UNORM,
                    DXGI_FORMAT_R8G8_UNORM, DXGI_SAMPLE_DESC,
                },
                DXGI_ERROR_DEVICE_HUNG, DXGI_ERROR_DEVICE_REMOVED, DXGI_ERROR_DEVICE_RESET,
                DXGI_MWA_NO_ALT_ENTER, DXGI_PRESENT, DXGI_SWAP_CHAIN_DESC1, DXGI_SWAP_CHAIN_FLAG,
                DXGI_SWAP_EFFECT_FLIP_DISCARD, DXGI_USAGE_RENDER_TARGET_OUTPUT, IDXGISwapChain1,
            },
        },
    },
    core::Error,
};

use crate::d3d11_gpu_context::D3d11GpuContext;

/// Double-buffered, same as the D3D12 side.
const FRAME_COUNT: u32 = 2;

struct RendererState {
    swap_chain: IDXGISwapChain1,
    /// `None` only ever momentarily, inside `resize` — dropped (releasing
    /// its reference to the old backbuffer) before `ResizeBuffers`, then
    /// immediately replaced with a view of the new one.
    render_target_view: Option<ID3D11RenderTargetView>,
    width: u32,
    height: u32,
}

pub struct D3d11WindowRenderer {
    device: ID3D11Device,
    /// Shared with every other window built from the same
    /// [`D3d11GpuContext`] — see that type's own docs on why every submit
    /// below locks it for its whole bind+draw+present sequence.
    context: Arc<Mutex<ID3D11DeviceContext>>,
    vertex_shader: ID3D11VertexShader,
    bgra_pixel_shader: ID3D11PixelShader,
    nv12_pixel_shader: ID3D11PixelShader,
    sampler: ID3D11SamplerState,
    /// Set once a DXGI/D3D11 call comes back with a device-removed-class
    /// error. After that every further call fails fast with
    /// `SubmitError::DeviceRemoved` instead of touching the GPU again —
    /// recovery means recreating the whole `D3d11GpuContext`, not retrying.
    device_lost: AtomicBool,
    state: Mutex<RendererState>,
}

// SAFETY: every field is either a `windows-rs` COM interface wrapper
// (free-threaded by contract for the methods used here, or behind the
// `Mutex`/`AtomicBool` above) — the mutexes are what actually serialize
// the non-thread-safe parts (swap chain `Present`/`ResizeBuffers`, and the
// shared context's own bind+draw sequence).
unsafe impl Send for D3d11WindowRenderer {}
unsafe impl Sync for D3d11WindowRenderer {}

impl D3d11WindowRenderer {
    pub fn new(
        gpu: &D3d11GpuContext,
        hwnd_value: isize,
        width: u32,
        height: u32,
    ) -> Result<Self, SubmitError> {
        if hwnd_value == 0 || width == 0 || height == 0 {
            return Err(SubmitError::InvalidFrame);
        }

        unsafe {
            let hwnd = HWND(hwnd_value as *mut c_void);

            let swap_chain_desc = DXGI_SWAP_CHAIN_DESC1 {
                Width: width,
                Height: height,
                Format: DXGI_FORMAT_B8G8R8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount: FRAME_COUNT,
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                AlphaMode: DXGI_ALPHA_MODE_UNSPECIFIED,
                ..Default::default()
            };
            let swap_chain: IDXGISwapChain1 = gpu
                .factory
                .CreateSwapChainForHwnd(&gpu.device, hwnd, &swap_chain_desc, None, None)
                .map_err(window_err)?;
            gpu.factory
                .MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER)
                .map_err(window_err)?;

            let render_target_view =
                create_render_target_view(&gpu.device, &swap_chain).map_err(window_err)?;

            Ok(Self {
                device: gpu.device.clone(),
                context: gpu.context.clone(),
                vertex_shader: gpu.vertex_shader.clone(),
                bgra_pixel_shader: gpu.bgra_pixel_shader.clone(),
                nv12_pixel_shader: gpu.nv12_pixel_shader.clone(),
                sampler: gpu.sampler.clone(),
                device_lost: AtomicBool::new(false),
                state: Mutex::new(RendererState {
                    swap_chain,
                    render_target_view: Some(render_target_view),
                    width,
                    height,
                }),
            })
        }
    }

    fn check_device_lost<T>(&self, result: windows::core::Result<T>) -> Result<T, SubmitError> {
        result.map_err(|error| {
            if is_device_lost(&error) {
                self.device_lost.store(true, Ordering::Relaxed);
            }
            window_err(error)
        })
    }
}

impl D3d11FrameRenderer for D3d11WindowRenderer {
    fn device(&self) -> ID3D11Device {
        self.device.clone()
    }

    unsafe fn submit_bgra_texture(
        &self,
        texture: ID3D11Texture2D,
        array_index: u32,
        width: u32,
        height: u32,
    ) -> Result<(), SubmitError> {
        if self.device_lost.load(Ordering::Relaxed) {
            return Err(SubmitError::DeviceRemoved);
        }
        if width == 0 || height == 0 {
            return Err(SubmitError::InvalidFrame);
        }

        let desc = plane_srv_desc(DXGI_FORMAT_B8G8R8A8_UNORM, array_index);
        let mut srv = None;
        self.check_device_lost(unsafe {
            self.device
                .CreateShaderResourceView(&texture, Some(&desc), Some(&mut srv))
        })?;

        self.record_and_present(&self.bgra_pixel_shader, &[srv], width, height)
    }

    unsafe fn submit_nv12_texture(
        &self,
        texture: ID3D11Texture2D,
        array_index: u32,
        width: u32,
        height: u32,
    ) -> Result<(), SubmitError> {
        if self.device_lost.load(Ordering::Relaxed) {
            return Err(SubmitError::DeviceRemoved);
        }
        if width == 0 || height == 0 {
            return Err(SubmitError::InvalidFrame);
        }

        let luma_desc = plane_srv_desc(DXGI_FORMAT_R8_UNORM, array_index);
        let chroma_desc = plane_srv_desc(DXGI_FORMAT_R8G8_UNORM, array_index);
        let mut luma_srv = None;
        self.check_device_lost(unsafe {
            self.device
                .CreateShaderResourceView(&texture, Some(&luma_desc), Some(&mut luma_srv))
        })?;
        let mut chroma_srv = None;
        self.check_device_lost(unsafe {
            self.device.CreateShaderResourceView(
                &texture,
                Some(&chroma_desc),
                Some(&mut chroma_srv),
            )
        })?;

        self.record_and_present(
            &self.nv12_pixel_shader,
            &[luma_srv, chroma_srv],
            width,
            height,
        )
    }

    fn resize(&self, width: u32, height: u32) -> Result<(), SubmitError> {
        if self.device_lost.load(Ordering::Relaxed) {
            return Err(SubmitError::DeviceRemoved);
        }
        if width == 0 || height == 0 {
            return Err(SubmitError::InvalidFrame);
        }

        let mut state = self
            .state
            .lock()
            .map_err(|_| SubmitError::RendererStopped)?;
        let _context = self
            .context
            .lock()
            .map_err(|_| SubmitError::RendererStopped)?;

        // Drop the old RTV's reference to the backbuffer before
        // `ResizeBuffers` — same reasoning as the D3D12 side clearing
        // `render_targets` first.
        state.render_target_view = None;
        unsafe {
            self.check_device_lost(state.swap_chain.ResizeBuffers(
                FRAME_COUNT,
                width,
                height,
                DXGI_FORMAT_B8G8R8A8_UNORM,
                DXGI_SWAP_CHAIN_FLAG(0),
            ))?;
        }
        state.render_target_view = Some(self.check_device_lost(unsafe {
            create_render_target_view(&self.device, &state.swap_chain)
        })?);
        state.width = width;
        state.height = height;
        Ok(())
    }
}

impl D3d11WindowRenderer {
    /// The shared tail of both `submit_*` methods, under `self.context`'s
    /// lock for its entire duration — bind the shaders/SRVs/sampler/
    /// viewport, clear, draw a full-screen triangle, present. No barriers,
    /// no fence: the caller's `texture` was already fully written by
    /// whichever GPU commands produced it, and because every producer in
    /// this crate's D3D11 stack shares this same context, those commands
    /// are guaranteed to have been submitted (though not necessarily
    /// completed on the GPU — that's fine, the driver still executes
    /// everything in submission order) before this draw.
    fn record_and_present(
        &self,
        pixel_shader: &ID3D11PixelShader,
        shader_resources: &[Option<ID3D11ShaderResourceView>],
        frame_width: u32,
        frame_height: u32,
    ) -> Result<(), SubmitError> {
        let state = self
            .state
            .lock()
            .map_err(|_| SubmitError::RendererStopped)?;
        let context = self
            .context
            .lock()
            .map_err(|_| SubmitError::RendererStopped)?;

        let (window_width, window_height) = (state.width, state.height);
        let viewport = letterbox_viewport(frame_width, frame_height, window_width, window_height);
        let scissor = RECT {
            left: 0,
            top: 0,
            right: window_width as i32,
            bottom: window_height as i32,
        };
        let render_target_view = state
            .render_target_view
            .clone()
            .expect("render_target_view is only None transiently inside resize()");

        unsafe {
            context.ClearRenderTargetView(&render_target_view, &[0.0, 0.0, 0.0, 1.0]);
            context.OMSetRenderTargets(Some(&[Some(render_target_view)]), None);
            context.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            context.RSSetViewports(Some(&[viewport]));
            context.RSSetScissorRects(Some(&[scissor]));
            context.VSSetShader(&self.vertex_shader, None);
            context.PSSetShader(pixel_shader, None);
            context.PSSetShaderResources(0, Some(shader_resources));
            context.PSSetSamplers(0, Some(&[Some(self.sampler.clone())]));
            context.Draw(3, 0);

            // Unbind the SRVs before returning — leaving `texture` bound
            // as an input would keep it referenced by the context's
            // current state until the next draw overwrites the slot,
            // which would be surprising given `D3d11FrameRenderer`'s own
            // docs promise no lifetime-keeping is needed here.
            let clear_srvs = vec![None; shader_resources.len()];
            context.PSSetShaderResources(0, Some(&clear_srvs));
        }

        self.check_device_lost(unsafe { state.swap_chain.Present(1, DXGI_PRESENT(0)) }.ok())?;
        drop(context);
        drop(state);
        Ok(())
    }
}

fn window_err(error: windows::core::Error) -> SubmitError {
    if is_device_lost(&error) {
        SubmitError::DeviceRemoved
    } else {
        SubmitError::RenderFailed
    }
}

fn is_device_lost(error: &Error) -> bool {
    matches!(
        error.code(),
        DXGI_ERROR_DEVICE_REMOVED | DXGI_ERROR_DEVICE_RESET | DXGI_ERROR_DEVICE_HUNG
    )
}

unsafe fn create_render_target_view(
    device: &ID3D11Device,
    swap_chain: &IDXGISwapChain1,
) -> windows::core::Result<ID3D11RenderTargetView> {
    unsafe {
        let back_buffer: ID3D11Texture2D = swap_chain.GetBuffer(0)?;
        let mut rtv = None;
        device.CreateRenderTargetView(&back_buffer, None, Some(&mut rtv))?;
        Ok(rtv.unwrap())
    }
}

/// Builds a single-slice `Texture2DArray` SRV description — used for every
/// plane/format this renderer draws, not just the ones that actually come
/// from an array texture. A `Texture2DArray` view describing one slice
/// (`FirstArraySlice: array_index, ArraySize: 1`) works identically to a
/// plain `Texture2D` view when the underlying resource happens to have
/// `ArraySize == 1` (`DxgiCaptureSource`'s GPU mode, `D3d11Upload` — always
/// `array_index == 0` there) — so using this one shape unconditionally
/// avoids branching on where the texture came from, while still handling
/// [`crate::D3d11WindowRenderer::submit_nv12_texture`]'s real case: a
/// `D3d11Decoder` frame is one slice of libavcodec's own shared array-pooled
/// decode texture (see [`media_pp::elements::D3d11FrameRenderer::submit_nv12_texture`]'s
/// own docs).
fn plane_srv_desc(
    format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
    array_index: u32,
) -> D3D11_SHADER_RESOURCE_VIEW_DESC {
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

fn letterbox_viewport(
    frame_width: u32,
    frame_height: u32,
    window_width: u32,
    window_height: u32,
) -> D3D11_VIEWPORT {
    let scale =
        (window_width as f32 / frame_width as f32).min(window_height as f32 / frame_height as f32);
    let width = frame_width as f32 * scale;
    let height = frame_height as f32 * scale;
    D3D11_VIEWPORT {
        TopLeftX: (window_width as f32 - width) * 0.5,
        TopLeftY: (window_height as f32 - height) * 0.5,
        Width: width,
        Height: height,
        MinDepth: 0.0,
        MaxDepth: 1.0,
    }
}
