//! Copies a D3D12VA picture into another of the same shape on the GPU — what
//! [`super::D3d12Decoder`] holds while playing backwards, since the surfaces
//! it decodes to are a pool the rest of the stretch is decoded into.

use std::mem::ManuallyDrop;

use ffmpeg_next as ffmpeg;
use windows::{
    Win32::{
        Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0},
        Graphics::Direct3D12::{
            D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC, D3D12_FENCE_FLAG_NONE,
            D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0,
            D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_BARRIER_FLAG_NONE,
            D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATES, D3D12_RESOURCE_TRANSITION_BARRIER, D3D12_TEXTURE_COPY_LOCATION,
            D3D12_TEXTURE_COPY_LOCATION_0, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            ID3D12CommandAllocator, ID3D12CommandList, ID3D12CommandQueue, ID3D12Device,
            ID3D12Fence, ID3D12GraphicsCommandList, ID3D12Resource,
        },
        System::Threading::{CreateEventW, INFINITE, WaitForSingleObject},
    },
    core::Interface,
};

use crate::platform::windows::d3d12va::{d3d12va_texture, set_d3d12va_fence_value};

/// A queue, a list and a fence of its own, for one copy at a time: each is
/// waited for before the next is recorded, so the picture copied can go
/// back to the decoder's pool as soon as `copy` returns.
pub(super) struct D3d12Copier {
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    next: u64,
    event: HANDLE,
}

// SAFETY: D3D12 devices, queues, allocators, lists and fences are
// free-threaded COM objects, and `event` is an owned kernel handle; `&mut
// self` on every method rules out recording from two threads at once.
unsafe impl Send for D3d12Copier {}

impl D3d12Copier {
    pub(super) fn new(device: &ID3D12Device) -> windows::core::Result<Self> {
        // SAFETY: plain creation calls on a live device with local
        // descriptions; the event handle is closed on the one path that
        // fails after it exists, and otherwise by `Drop`.
        unsafe {
            let queue: ID3D12CommandQueue =
                device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })?;
            let allocator: ID3D12CommandAllocator =
                device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList =
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
            list.Close()?;
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            let event = CreateEventW(None, false, false, None)?;
            Ok(Self {
                queue,
                allocator,
                list,
                fence,
                next: 1,
                event,
            })
        }
    }

    /// Copies `source`'s picture into `destination`, a D3D12VA frame of the
    /// same frames-context shape, and waits for the copy to be done. Waits
    /// on the GPU for whatever wrote either before, and signals
    /// `destination`'s own fence after, as a consumer of it waits on.
    pub(super) fn copy(
        &mut self,
        source: &ffmpeg::frame::Video,
        destination: &mut ffmpeg::frame::Video,
    ) -> windows::core::Result<()> {
        let invalid = || windows::core::Error::from(windows::Win32::Foundation::E_INVALIDARG);
        let (from, from_fence, from_value) = d3d12va_texture(source).ok_or_else(invalid)?;
        let (to, to_fence, to_value) = d3d12va_texture(destination).ok_or_else(invalid)?;
        // SAFETY: a live D3D12VA frame's texture and fence are live COM
        // objects for as long as the frame is; each borrow is cloned into
        // an owned reference before either frame can go.
        let (from, from_fence, to, to_fence) = unsafe {
            (
                ID3D12Resource::from_raw_borrowed(&from).cloned(),
                ID3D12Fence::from_raw_borrowed(&from_fence).cloned(),
                ID3D12Resource::from_raw_borrowed(&to).cloned(),
                ID3D12Fence::from_raw_borrowed(&to_fence).cloned(),
            )
        };
        let (Some(from), Some(from_fence), Some(to), Some(to_fence)) =
            (from, from_fence, to, to_fence)
        else {
            return Err(invalid());
        };
        let signalled = to_value + 1;
        // SAFETY: every object here is live and on this copier's device.
        // The list is closed and its allocator idle — the last copy was
        // waited for — so both may be reset. FFmpeg's frames rest in the
        // common state, which the barriers leave them in again. Both frames
        // are the same frames-context shape, so each plane subresource of
        // one fits the other.
        unsafe {
            self.queue.Wait(&from_fence, from_value)?;
            self.queue.Wait(&to_fence, to_value)?;
            self.allocator.Reset()?;
            self.list.Reset(&self.allocator, None)?;
            let mut before = [
                transition(
                    &from,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                ),
                transition(
                    &to,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                ),
            ];
            self.list.ResourceBarrier(&before);
            release(&mut before);
            // Luma and chroma, the two planes of NV12 and P010.
            for plane in 0..2 {
                let mut into = location(&to, plane);
                let mut out_of = location(&from, plane);
                self.list.CopyTextureRegion(&into, 0, 0, 0, &out_of, None);
                ManuallyDrop::drop(&mut into.pResource);
                ManuallyDrop::drop(&mut out_of.pResource);
            }
            let mut after = [
                transition(
                    &from,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                ),
                transition(
                    &to,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_COMMON,
                ),
            ];
            self.list.ResourceBarrier(&after);
            release(&mut after);
            self.list.Close()?;
            self.queue
                .ExecuteCommandLists(&[Some(self.list.cast::<ID3D12CommandList>()?)]);
            self.queue.Signal(&to_fence, signalled)?;
            let done = self.next;
            self.next += 1;
            self.queue.Signal(&self.fence, done)?;
            if self.fence.GetCompletedValue() < done {
                self.fence.SetEventOnCompletion(done, self.event)?;
                if WaitForSingleObject(self.event, INFINITE) != WAIT_OBJECT_0 {
                    return Err(windows::core::Error::from_thread());
                }
            }
        }
        set_d3d12va_fence_value(destination, signalled);
        Ok(())
    }
}

impl Drop for D3d12Copier {
    fn drop(&mut self) {
        // SAFETY: every copy was waited for before `copy` returned, so
        // nothing on the GPU still uses what this owns; `event` is the handle
        // `new` created, closed once.
        unsafe { CloseHandle(self.event).ok() };
    }
}

fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
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
    }
}

/// Drops the resource references `transition` put in `barriers`.
fn release(barriers: &mut [D3D12_RESOURCE_BARRIER]) {
    for barrier in barriers {
        // SAFETY: `transition` initialized this union arm with one cloned
        // reference; this is its single matching drop.
        unsafe {
            let transition = &mut barrier.Anonymous.Transition;
            ManuallyDrop::drop(&mut transition.pResource);
        }
    }
}

/// One plane of `resource`, as a copy's end: plane `n` of a planar texture
/// with one mip and one slice is subresource `n`.
fn location(resource: &ID3D12Resource, plane: u32) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        pResource: ManuallyDrop::new(Some(resource.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: plane,
        },
    }
}
