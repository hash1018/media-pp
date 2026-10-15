//! What this crate's D3D12 compute shaders run with: a shader compiled with
//! the root signature it declares, the command list it is recorded into and
//! submitted from, and the buffers, textures, views and barriers around it —
//! shared by the D3D12 detectors, the overlay and the cut detector, each of
//! which runs its own shaders on a [`D3d12Gpu`](super::d3d12_gpu::D3d12Gpu)'s
//! queue.

use std::mem::ManuallyDrop;

use windows::{
    Win32::{
        Foundation::{CloseHandle, HANDLE},
        Graphics::{
            Direct3D::{
                Fxc::{D3D_BLOB_ROOT_SIGNATURE, D3DGetBlobPart},
                ID3DBlob,
            },
            Direct3D12::*,
            Dxgi::Common::{DXGI_FORMAT, DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC},
        },
        System::Threading::{CreateEventW, INFINITE, WaitForSingleObject},
    },
    core::{Interface, PCSTR, s},
};

use super::hlsl::compile_shader;

/// A compute shader and the root signature it declares.
pub(crate) struct Kernel {
    pub(crate) root_signature: ID3D12RootSignature,
    pub(crate) pipeline: ID3D12PipelineState,
}

impl Kernel {
    /// `entry` of `source`, compiled and made a pipeline on `device`.
    ///
    /// # Safety
    ///
    /// `device` is live.
    pub(crate) unsafe fn new(
        device: &ID3D12Device,
        source: &[u8],
        file: PCSTR,
        entry: PCSTR,
    ) -> windows::core::Result<Self> {
        // SAFETY: a live device and descriptions built here; the shader's
        // blob outlives the calls that read it.
        unsafe {
            let shader = compile_shader(source, file, entry, s!("cs_5_1"))?;
            let root = D3DGetBlobPart(
                shader.GetBufferPointer(),
                shader.GetBufferSize(),
                D3D_BLOB_ROOT_SIGNATURE,
                0,
            )?;
            let root_signature: ID3D12RootSignature =
                device.CreateRootSignature(0, blob_bytes(&root))?;
            let pipeline = pipeline(device, &shader, &root_signature)?;
            Ok(Self {
                root_signature,
                pipeline,
            })
        }
    }

    /// `entry` of `source` made a pipeline under this kernel's root
    /// signature, which it declares alike, so a list can change from one to
    /// the other keeping what is bound.
    ///
    /// # Safety
    ///
    /// `device` is live.
    pub(crate) unsafe fn sibling(
        &self,
        device: &ID3D12Device,
        source: &[u8],
        file: PCSTR,
        entry: PCSTR,
    ) -> windows::core::Result<ID3D12PipelineState> {
        // SAFETY: as in `new`.
        unsafe {
            let shader = compile_shader(source, file, entry, s!("cs_5_1"))?;
            pipeline(device, &shader, &self.root_signature)
        }
    }
}

/// `shader` made a compute pipeline under `root_signature`.
unsafe fn pipeline(
    device: &ID3D12Device,
    shader: &ID3DBlob,
    root_signature: &ID3D12RootSignature,
) -> windows::core::Result<ID3D12PipelineState> {
    // SAFETY: a live device and blob; the description's reference to the
    // root signature is released once the pipeline is made.
    unsafe {
        let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: ManuallyDrop::new(Some(root_signature.clone())),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: shader.GetBufferPointer(),
                BytecodeLength: shader.GetBufferSize(),
            },
            ..Default::default()
        };
        let pipeline = device.CreateComputePipelineState(&desc);
        ManuallyDrop::drop(&mut desc.pRootSignature);
        pipeline
    }
}

/// One command list, recorded and submitted to a queue a submission at a
/// time and waited for through its own fence.
///
/// Dropping it waits for what it submitted; an owner whose resources the
/// GPU may still be using calls [`Self::wait_idle`] in its own `Drop`, as
/// fields are dropped in order.
pub(crate) struct Commands {
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    /// Signalled at `submitted` once the last submission has run.
    fence: ID3D12Fence,
    event: HANDLE,
    submitted: u64,
}

// SAFETY: the D3D12 objects are free-threaded; the list and allocator are
// only touched through `&mut self`.
unsafe impl Send for Commands {}

impl Commands {
    /// A list on `device`, submitted to `queue`.
    pub(crate) fn new(
        device: &ID3D12Device,
        queue: &ID3D12CommandQueue,
    ) -> windows::core::Result<Self> {
        // SAFETY: a live device; the event is closed by `Drop`, or here
        // where what follows it fails.
        unsafe {
            let allocator: ID3D12CommandAllocator =
                device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList =
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
            // Created recording; closed so the first `begin` finds it as
            // `Reset` expects.
            list.Close()?;
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            let event = CreateEventW(None, false, false, None)?;
            Ok(Self {
                queue: queue.clone(),
                allocator,
                list,
                fence,
                event,
                submitted: 0,
            })
        }
    }

    /// Waits for the last submission to have run, then opens the list with
    /// `pipeline` set.
    pub(crate) fn begin(
        &mut self,
        pipeline: Option<&ID3D12PipelineState>,
    ) -> windows::core::Result<ID3D12GraphicsCommandList> {
        self.wait_idle()?;
        // SAFETY: the list and allocator are idle, waited for above, and
        // recorded by this thread alone.
        unsafe {
            self.allocator.Reset()?;
            self.list.Reset(&self.allocator, pipeline)?;
        }
        Ok(self.list.clone())
    }

    /// Closes the list and submits it, after a GPU-side wait for each of
    /// `waits` — a fence and the value whatever wrote a resource it reads
    /// signals it at: sharing a device does not order this queue after
    /// another's. It returns once the work is queued.
    pub(crate) fn submit(&mut self, waits: &[(&ID3D12Fence, u64)]) -> windows::core::Result<()> {
        // SAFETY: the list is this one's, open since `begin`; the fences are
        // live.
        unsafe {
            self.list.Close()?;
            for &(fence, value) in waits {
                self.queue.Wait(fence, value)?;
            }
            self.queue
                .ExecuteCommandLists(&[Some(self.list.cast::<ID3D12CommandList>()?)]);
            self.submitted += 1;
            self.queue.Signal(&self.fence, self.submitted)
        }
    }

    /// Waits for the last submission to have run.
    pub(crate) fn wait_idle(&self) -> windows::core::Result<()> {
        // SAFETY: a live fence and event; the wait is for a value this
        // signals.
        unsafe {
            if self.fence.GetCompletedValue() < self.submitted {
                self.fence
                    .SetEventOnCompletion(self.submitted, self.event)?;
                WaitForSingleObject(self.event, INFINITE);
            }
        }
        Ok(())
    }
}

impl Drop for Commands {
    fn drop(&mut self) {
        let _ = self.wait_idle();
        // SAFETY: the event was made by `new` and is closed once, here.
        unsafe {
            let _ = CloseHandle(self.event);
        }
    }
}

/// A buffer of `bytes` on `heap`.
///
/// # Safety
///
/// `device` is live.
pub(crate) unsafe fn buffer(
    device: &ID3D12Device,
    bytes: u64,
    heap: D3D12_HEAP_TYPE,
    flags: D3D12_RESOURCE_FLAGS,
    state: D3D12_RESOURCE_STATES,
) -> windows::core::Result<ID3D12Resource> {
    let mut resource: Option<ID3D12Resource> = None;
    // SAFETY: descriptions built here, and a correctly typed out-parameter.
    unsafe {
        device.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES {
                Type: heap,
                ..Default::default()
            },
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Alignment: 0,
                Width: bytes.max(1),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_UNKNOWN,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: flags,
            },
            state,
            None,
            &mut resource,
        )?;
    }
    Ok(resource.expect("CreateCommittedResource succeeded without a resource"))
}

/// A texture of `format`, `width` by `height`, that a shader writes, made
/// in `UNORDERED_ACCESS`.
///
/// # Safety
///
/// `device` is live.
#[cfg(feature = "ort-directml")]
pub(crate) unsafe fn writable_texture(
    device: &ID3D12Device,
    format: DXGI_FORMAT,
    (width, height): (u32, u32),
) -> windows::core::Result<ID3D12Resource> {
    let mut resource: Option<ID3D12Resource> = None;
    // SAFETY: descriptions built here, and a correctly typed out-parameter.
    unsafe {
        device.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            },
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Alignment: 0,
                Width: u64::from(width),
                Height: height,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            },
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            None,
            &mut resource,
        )?;
    }
    Ok(resource.expect("CreateCommittedResource succeeded without a resource"))
}

/// One plane of a texture read as `format`: the format picks which of a
/// two-plane one, the plane slice says where.
pub(crate) fn plane_view(format: DXGI_FORMAT, plane_slice: u32) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
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

/// One plane of a texture written as `format`.
#[cfg(feature = "ort-directml")]
pub(crate) fn plane_write_view(
    format: DXGI_FORMAT,
    plane_slice: u32,
) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: format,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_UAV {
                MipSlice: 0,
                PlaneSlice: plane_slice,
            },
        },
    }
}

/// Records `resource` going from `before` to `after`.
///
/// # Safety
///
/// `list` is open and `resource` live.
pub(crate) unsafe fn transition(
    list: &ID3D12GraphicsCommandList,
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
    // SAFETY: records one barrier on an open list, then releases the
    // resource reference the barrier description took.
    unsafe {
        list.ResourceBarrier(std::slice::from_ref(&barrier));
        let transition = &mut *barrier.Anonymous.Transition;
        ManuallyDrop::drop(&mut transition.pResource);
    }
}

/// Records a barrier between a shader's writes to `resource` and what reads
/// it after them.
///
/// # Safety
///
/// `list` is open and `resource` live.
pub(crate) unsafe fn uav_barrier(list: &ID3D12GraphicsCommandList, resource: &ID3D12Resource) {
    let mut barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: ManuallyDrop::new(Some(resource.clone())),
            }),
        },
    };
    // SAFETY: records one barrier on an open list, then releases the
    // resource reference the barrier description took.
    unsafe {
        list.ResourceBarrier(std::slice::from_ref(&barrier));
        let uav = &mut *barrier.Anonymous.UAV;
        ManuallyDrop::drop(&mut uav.pResource);
    }
}

/// The bytes a blob holds.
///
/// # Safety
///
/// `blob` is live.
unsafe fn blob_bytes(blob: &ID3DBlob) -> &[u8] {
    // SAFETY: a blob owns `GetBufferSize()` readable bytes at
    // `GetBufferPointer()` for as long as it lives.
    unsafe {
        std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
    }
}
