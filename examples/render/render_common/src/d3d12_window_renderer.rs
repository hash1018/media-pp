//! A single window's D3D12 renderer — swap chain, the two draw paths
//! [`media_pp::elements::D3d12FrameRenderer`] needs (`submit_yuv420p`
//! CPU-upload, `submit_nv12_texture` zero-copy), and resize. Reuses
//! [`crate::D3d12GpuContext`]'s device/queue/root-signature/PSOs; everything
//! else here (swap chain, RTV heap, SRV heap, command allocator/list,
//! fence) is this window's own.
//!
//! Deliberately synchronous, single frame in flight: every `submit_*`
//! call first waits for the *previous* frame this window submitted to
//! finish on the GPU, then records/executes/presents the new one on the
//! calling thread. The upstream `renderer-engine` crate this replaces ran
//! a background thread with a 2-deep upload ring so a slow render never
//! blocked the submitting thread; that pipelining isn't worth the extra
//! state machine here; the calling thread is already whatever `Queue`
//! worker media-pp put in front of the sink (see `ChainBuilder::queue`
//! in every example that uses this).

use std::{
    any::Any,
    ffi::c_void,
    mem::ManuallyDrop,
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use media_pp::elements::{D3d12FrameRenderer, SubmitError};
use windows::{
    Win32::{
        Foundation::{CloseHandle, HANDLE, HWND, RECT},
        Graphics::{
            Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            Direct3D12::*,
            Dxgi::{
                Common::{
                    DXGI_ALPHA_MODE_UNSPECIFIED, DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM,
                    DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8G8_UNORM, DXGI_SAMPLE_DESC,
                },
                DXGI_ERROR_DEVICE_HUNG, DXGI_ERROR_DEVICE_REMOVED, DXGI_ERROR_DEVICE_RESET,
                DXGI_MWA_NO_ALT_ENTER, DXGI_PRESENT, DXGI_SWAP_CHAIN_DESC1, DXGI_SWAP_CHAIN_FLAG,
                DXGI_SWAP_EFFECT_FLIP_DISCARD, DXGI_USAGE_RENDER_TARGET_OUTPUT, IDXGISwapChain3,
            },
        },
        System::Threading::{CreateEventW, INFINITE, WaitForSingleObject},
    },
    core::{Error, Interface},
};

use crate::d3d12_gpu_context::D3d12GpuContext;

/// Double-buffered, same as the `renderer-engine` crate this replaces.
const FRAME_COUNT: usize = 2;

struct RendererState {
    swap_chain: IDXGISwapChain3,
    command_allocator: ID3D12CommandAllocator,
    command_list: ID3D12GraphicsCommandList,
    rtv_heap: ID3D12DescriptorHeap,
    rtv_descriptor_size: usize,
    render_targets: Vec<ID3D12Resource>,
    srv_heap: ID3D12DescriptorHeap,
    srv_size: usize,
    /// The fence value signaled by the last frame *this window* submitted
    /// — waited on at the start of the next `submit_*`/`resize` call
    /// before touching any resource that frame might still be using.
    last_submitted: u64,
    /// Whatever kept the last `submit_nv12_texture` call's external
    /// resource alive — held until the wait above confirms the GPU is
    /// actually done reading it, then dropped (replaced by the next
    /// frame's, or by `None` on `resize`/`Drop`).
    pending_keep_alive: Option<Box<dyn Any + Send>>,
    width: u32,
    height: u32,
}

pub struct D3d12WindowRenderer {
    device: ID3D12Device,
    command_queue: ID3D12CommandQueue,
    root_signature: ID3D12RootSignature,
    nv12_pipeline: ID3D12PipelineState,
    fence: ID3D12Fence,
    fence_event: HANDLE,
    /// Set once a DXGI/D3D12 call comes back with a device-removed-class
    /// error. After that every further call fails fast with
    /// `SubmitError::DeviceRemoved` instead of touching the GPU again —
    /// recovery means recreating the whole `D3d12GpuContext`, not retrying.
    device_lost: AtomicBool,
    state: Mutex<RendererState>,
}

// SAFETY: every field is either a `windows-rs` COM interface wrapper
// (free-threaded by contract for the methods used here) or behind the
// `Mutex`/`AtomicBool` above, which is what actually serializes access to
// the non-thread-safe parts (command allocator/list `Reset`, swap chain
// `Present`/`ResizeBuffers`).
unsafe impl Send for D3d12WindowRenderer {}
unsafe impl Sync for D3d12WindowRenderer {}

impl D3d12WindowRenderer {
    pub fn new(
        gpu: &D3d12GpuContext,
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
                BufferCount: FRAME_COUNT as u32,
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                AlphaMode: DXGI_ALPHA_MODE_UNSPECIFIED,
                ..Default::default()
            };
            let swap_chain1 = gpu
                .factory
                .CreateSwapChainForHwnd(&gpu.command_queue, hwnd, &swap_chain_desc, None, None)
                .map_err(window_err)?;
            gpu.factory
                .MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER)
                .map_err(window_err)?;
            let swap_chain: IDXGISwapChain3 = swap_chain1.cast().map_err(window_err)?;

            let rtv_heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: FRAME_COUNT as u32,
                ..Default::default()
            };
            let rtv_heap: ID3D12DescriptorHeap = gpu
                .device
                .CreateDescriptorHeap(&rtv_heap_desc)
                .map_err(window_err)?;
            let rtv_descriptor_size = gpu
                .device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV)
                as usize;
            let render_targets =
                create_render_targets(&gpu.device, &swap_chain, &rtv_heap, rtv_descriptor_size)
                    .map_err(window_err)?;

            let srv_heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 2,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                ..Default::default()
            };
            let srv_heap: ID3D12DescriptorHeap = gpu
                .device
                .CreateDescriptorHeap(&srv_heap_desc)
                .map_err(window_err)?;
            let srv_size = gpu
                .device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
                as usize;

            let command_allocator: ID3D12CommandAllocator = gpu
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .map_err(window_err)?;
            let command_list: ID3D12GraphicsCommandList = gpu
                .device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &command_allocator, None)
                .map_err(window_err)?;
            // Right after CreateCommandList it's in a recording state;
            // must be closed once before the first submit's Reset.
            command_list.Close().map_err(window_err)?;

            let fence: ID3D12Fence = gpu
                .device
                .CreateFence(0, D3D12_FENCE_FLAG_NONE)
                .map_err(window_err)?;
            let fence_event = CreateEventW(None, false, false, None).map_err(window_err)?;

            Ok(Self {
                device: gpu.device.clone(),
                command_queue: gpu.command_queue.clone(),
                root_signature: gpu.root_signature.clone(),
                nv12_pipeline: gpu.nv12_pipeline.clone(),
                fence,
                fence_event,
                device_lost: AtomicBool::new(false),
                state: Mutex::new(RendererState {
                    swap_chain,
                    command_allocator,
                    command_list,
                    rtv_heap,
                    rtv_descriptor_size,
                    render_targets,
                    srv_heap,
                    srv_size,
                    last_submitted: 0,
                    pending_keep_alive: None,
                    width,
                    height,
                }),
            })
        }
    }

    /// Waits until `value` (a value previously returned by
    /// `command_queue.Signal(&self.fence, value)`) has been reached —
    /// i.e. every GPU command submitted before that `Signal` has finished.
    /// A no-op if `value == 0` (nothing submitted yet).
    fn wait_for_value(&self, value: u64) -> windows::core::Result<()> {
        if value == 0 {
            return Ok(());
        }
        unsafe {
            if self.fence.GetCompletedValue() < value {
                self.fence.SetEventOnCompletion(value, self.fence_event)?;
                WaitForSingleObject(self.fence_event, INFINITE);
            }
        }
        Ok(())
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

impl D3d12FrameRenderer for D3d12WindowRenderer {
    fn device(&self) -> ID3D12Device {
        self.device.clone()
    }

    unsafe fn submit_nv12_texture(
        &self,
        texture: ID3D12Resource,
        fence: ID3D12Fence,
        fence_value: u64,
        width: u32,
        height: u32,
        keep_alive: Box<dyn Any + Send>,
    ) -> Result<(), SubmitError> {
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
        self.check_device_lost(self.wait_for_value(state.last_submitted))?;
        state.pending_keep_alive = None;

        unsafe {
            let luma_srv = plane_srv_desc(DXGI_FORMAT_R8_UNORM, 0);
            let chroma_srv = plane_srv_desc(DXGI_FORMAT_R8G8_UNORM, 1);
            let handle0 = srv_cpu_handle(&state.srv_heap, state.srv_size, 0);
            let handle1 = srv_cpu_handle(&state.srv_heap, state.srv_size, 1);
            self.device
                .CreateShaderResourceView(&texture, Some(&luma_srv), handle0);
            self.device
                .CreateShaderResourceView(&texture, Some(&chroma_srv), handle1);
        }

        self.check_device_lost(unsafe { state.command_allocator.Reset() })?;
        self.check_device_lost(unsafe {
            state.command_list.Reset(&state.command_allocator, None)
        })?;

        unsafe {
            record_transition(
                &state.command_list,
                &texture,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            );
        }

        let (window_width, window_height) = (state.width, state.height);
        self.check_device_lost(unsafe {
            self.record_and_present(
                &mut state,
                self.nv12_pipeline.clone(),
                (width, height),
                (window_width, window_height),
                Some((&fence, fence_value)),
                Some(&texture),
            )
        })?;

        state.pending_keep_alive = Some(keep_alive);
        Ok(())
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
        self.check_device_lost(self.wait_for_value(state.last_submitted))?;
        state.pending_keep_alive = None;

        unsafe {
            state.render_targets.clear();
            self.check_device_lost(state.swap_chain.ResizeBuffers(
                FRAME_COUNT as u32,
                width,
                height,
                DXGI_FORMAT_B8G8R8A8_UNORM,
                DXGI_SWAP_CHAIN_FLAG(0),
            ))?;
            state.render_targets = self.check_device_lost(create_render_targets(
                &self.device,
                &state.swap_chain,
                &state.rtv_heap,
                state.rtv_descriptor_size,
            ))?;
        }
        state.width = width;
        state.height = height;
        Ok(())
    }
}

impl D3d12WindowRenderer {
    /// The shared tail of both `submit_*` methods: clear + draw a
    /// full-screen triangle with `pipeline` sampling whatever's currently
    /// in `srv_heap`, restore `restore_to_common` (if any) back to
    /// `D3D12_RESOURCE_STATE_COMMON`, present, and signal this window's
    /// own completion fence. `wait_on` is an *external* `(fence, value)`
    /// to `Wait()` the GPU queue on before this command list executes
    /// (the caller's own decode/upload completion, for the zero-copy path
    /// only) — this is a GPU-side wait (the queue stalls, not the CPU),
    /// matching the `AVD3D12VAFrame` sync contract every `Pixel::D3D12`
    /// producer in this crate follows.
    ///
    /// `restore_to_common` (the zero-copy path's externally-owned
    /// texture) must be transitioned back to `COMMON` *before* `Close()`
    /// — recording it after this method returns would mean recording
    /// onto an already-closed, already-executed command list, which is
    /// invalid, not merely deferred to the next frame.
    unsafe fn record_and_present(
        &self,
        state: &mut RendererState,
        pipeline: ID3D12PipelineState,
        frame_size: (u32, u32),
        window_size: (u32, u32),
        wait_on: Option<(&ID3D12Fence, u64)>,
        restore_to_common: Option<&ID3D12Resource>,
    ) -> windows::core::Result<()> {
        let (frame_width, frame_height) = frame_size;
        let (window_width, window_height) = window_size;
        unsafe {
            let frame_index = state.swap_chain.GetCurrentBackBufferIndex() as usize;
            let render_target = state.render_targets[frame_index].clone();
            let rtv_start = state.rtv_heap.GetCPUDescriptorHandleForHeapStart();
            let rtv_handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: rtv_start.ptr + frame_index * state.rtv_descriptor_size,
            };

            record_transition(
                &state.command_list,
                &render_target,
                D3D12_RESOURCE_STATE_PRESENT,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            );
            state
                .command_list
                .OMSetRenderTargets(1, Some(&rtv_handle), false, None);
            state
                .command_list
                .ClearRenderTargetView(rtv_handle, &[0.0, 0.0, 0.0, 1.0], None);

            let viewport =
                letterbox_viewport(frame_width, frame_height, window_width, window_height);
            let scissor = RECT {
                left: 0,
                top: 0,
                right: window_width as i32,
                bottom: window_height as i32,
            };
            state.command_list.SetPipelineState(&pipeline);
            state
                .command_list
                .SetGraphicsRootSignature(&self.root_signature);
            state
                .command_list
                .SetDescriptorHeaps(&[Some(state.srv_heap.clone())]);
            state.command_list.SetGraphicsRootDescriptorTable(
                0,
                state.srv_heap.GetGPUDescriptorHandleForHeapStart(),
            );
            state.command_list.RSSetViewports(&[viewport]);
            state.command_list.RSSetScissorRects(&[scissor]);
            state
                .command_list
                .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            state.command_list.DrawInstanced(3, 1, 0, 0);

            if let Some(resource) = restore_to_common {
                record_transition(
                    &state.command_list,
                    resource,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                );
            }

            record_transition(
                &state.command_list,
                &render_target,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_PRESENT,
            );

            state.command_list.Close()?;
            let base_command_list: ID3D12CommandList = state.command_list.cast()?;
            if let Some((wait_fence, wait_value)) = wait_on {
                // GPU-side wait: this window's queue is the same shared
                // `DIRECT` queue every window presents through, but the
                // decode/upload work that produced `texture` was a
                // separate, unrelated submission — only sharing a device
                // doesn't serialize the two, so this command list must
                // wait explicitly before its own commands (which read
                // `texture`) are allowed to start.
                self.command_queue.Wait(wait_fence, wait_value)?;
            }
            self.command_queue
                .ExecuteCommandLists(&[Some(base_command_list)]);
            state.swap_chain.Present(1, DXGI_PRESENT(0)).ok()?;

            let fence_value = {
                let value = state.last_submitted + 1;
                self.command_queue.Signal(&self.fence, value)?;
                value
            };
            state.last_submitted = fence_value;
            Ok(())
        }
    }
}

impl Drop for D3d12WindowRenderer {
    fn drop(&mut self) {
        if let Ok(state) = self.state.lock() {
            let _ = self.wait_for_value(state.last_submitted);
        }
        unsafe {
            let _ = CloseHandle(self.fence_event);
        }
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

unsafe fn create_render_targets(
    device: &ID3D12Device,
    swap_chain: &IDXGISwapChain3,
    rtv_heap: &ID3D12DescriptorHeap,
    descriptor_size: usize,
) -> windows::core::Result<Vec<ID3D12Resource>> {
    let mut targets = Vec::with_capacity(FRAME_COUNT);
    let heap_start = unsafe { rtv_heap.GetCPUDescriptorHandleForHeapStart() };
    for index in 0..FRAME_COUNT {
        let resource: ID3D12Resource = unsafe { swap_chain.GetBuffer(index as u32)? };
        let handle = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: heap_start.ptr + index * descriptor_size,
        };
        unsafe { device.CreateRenderTargetView(&resource, None, handle) };
        targets.push(resource);
    }
    Ok(targets)
}

fn srv_cpu_handle(
    heap: &ID3D12DescriptorHeap,
    size: usize,
    index: usize,
) -> D3D12_CPU_DESCRIPTOR_HANDLE {
    unsafe {
        let start = heap.GetCPUDescriptorHandleForHeapStart();
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: start.ptr + index * size,
        }
    }
}

fn plane_srv_desc(format: DXGI_FORMAT, plane_slice: u32) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: format,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MostDetailedMip: 0,
                MipLevels: 1,
                PlaneSlice: plane_slice,
                ResourceMinLODClamp: 0.0,
            },
        },
    }
}
#[allow(clippy::explicit_auto_deref)]
unsafe fn record_transition(
    command_list: &ID3D12GraphicsCommandList,
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) {
    let mut barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: ManuallyDrop::new(Some(resource.clone())),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    };
    unsafe {
        command_list.ResourceBarrier(std::slice::from_ref(&barrier));
        let transition: &mut D3D12_RESOURCE_TRANSITION_BARRIER = &mut *barrier.Anonymous.Transition;
        ManuallyDrop::drop(&mut transition.pResource);
    }
}

fn letterbox_viewport(
    frame_width: u32,
    frame_height: u32,
    window_width: u32,
    window_height: u32,
) -> D3D12_VIEWPORT {
    let scale =
        (window_width as f32 / frame_width as f32).min(window_height as f32 / frame_height as f32);
    let width = frame_width as f32 * scale;
    let height = frame_height as f32 * scale;
    D3D12_VIEWPORT {
        TopLeftX: (window_width as f32 - width) * 0.5,
        TopLeftY: (window_height as f32 - height) * 0.5,
        Width: width,
        Height: height,
        MinDepth: 0.0,
        MaxDepth: 1.0,
    }
}
