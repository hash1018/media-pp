//! What a graphics API needs from CUDA to show a CUDA frame: which GPU the
//! frames are on, and a way to write into memory the graphics API allocated.
//!
//! The direction is fixed by what works. CUDA cannot export the surfaces
//! NVDEC decodes into, so the graphics side allocates exportable memory of
//! its own, hands CUDA a file descriptor for it, and each frame is copied
//! device to device into that — see [`crate::elements::CudaFrameRenderer`]'s
//! own notes on why this is a copy and not an import.
//!
//! Separate from [`super::CudaDriver`] because none of this needs a kernel:
//! that type JIT-compiles its modules at construction, which a renderer has
//! no use for.

use std::{
    ffi::c_int,
    os::fd::{IntoRawFd, OwnedFd},
    sync::Arc,
};

use super::{
    CU_MEMORYTYPE_DEVICE, CUcontext, CUdevice, CUdeviceptr, CUresult, CudaDriverError,
    CudaMemcpy2D, check, cuCtxPopCurrent_v2, cuCtxPushCurrent_v2, cuCtxSynchronize, cuDeviceGet,
    cuDevicePrimaryCtxRelease_v2, cuDevicePrimaryCtxRetain, cuInit, cuMemFree_v2, cuMemcpy2D_v2,
};

type CUexternalMemory = *mut std::ffi::c_void;

/// `CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD` — a POSIX file descriptor
/// another API exported, which is what Vulkan's `VK_KHR_external_memory_fd`
/// hands out.
const CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD: std::ffi::c_uint = 1;

/// `CUDA_EXTERNAL_MEMORY_DEDICATED` — the memory is a dedicated allocation,
/// which is how the Vulkan side makes it.
const CUDA_EXTERNAL_MEMORY_DEDICATED: std::ffi::c_uint = 1;

/// `CUDA_EXTERNAL_MEMORY_HANDLE_DESC`, versioned by name and unchanged since
/// CUDA 10.
#[repr(C)]
#[derive(Clone, Copy)]
struct ExternalMemoryHandleDesc {
    type_: std::ffi::c_uint,
    handle: ExternalMemoryHandle,
    size: u64,
    flags: std::ffi::c_uint,
    reserved: [std::ffi::c_uint; 16],
}

/// The handle union inside it. Only `fd` is ever written here; the other two
/// members are declared because they are what size the union.
#[repr(C)]
#[derive(Clone, Copy)]
union ExternalMemoryHandle {
    fd: c_int,
    win32: [*mut std::ffi::c_void; 2],
    nv_sci_buf_object: *const std::ffi::c_void,
}

/// `CUDA_EXTERNAL_MEMORY_BUFFER_DESC`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ExternalMemoryBufferDesc {
    offset: u64,
    size: u64,
    flags: std::ffi::c_uint,
    reserved: [std::ffi::c_uint; 16],
}

#[link(name = "cuda")]
unsafe extern "C" {
    /// The 16 bytes Vulkan reports as `VkPhysicalDeviceIDProperties::deviceUUID`
    /// for the same GPU — what proves a Vulkan device is the one the frames
    /// are on, rather than the first one listed.
    fn cuDeviceGetUuid(uuid: *mut [u8; 16], device: CUdevice) -> CUresult;
    fn cuImportExternalMemory(
        memory: *mut CUexternalMemory,
        desc: *const ExternalMemoryHandleDesc,
    ) -> CUresult;
    fn cuExternalMemoryGetMappedBuffer(
        pointer: *mut CUdeviceptr,
        memory: CUexternalMemory,
        desc: *const ExternalMemoryBufferDesc,
    ) -> CUresult;
    fn cuDestroyExternalMemory(memory: CUexternalMemory) -> CUresult;
}

/// One retained reference to device 0's primary context — the one
/// [`crate::elements::CudaDevice`] makes FFmpeg decode on, so a frame's
/// device pointers are valid in it — plus the few calls interop needs.
///
/// Made once, up front, and kept: retaining and releasing a primary context
/// while another thread has NVDEC work in flight has crashed inside the
/// driver (see `CudaDevice`'s own notes), which rules out doing either per
/// frame.
pub(crate) struct CudaInterop {
    device: CUdevice,
    ctx: CUcontext,
}

// SAFETY: a `CUcontext` is not thread-affine — every call below pushes it
// onto the calling thread and pops it again — and nothing here is mutated
// after construction.
unsafe impl Send for CudaInterop {}

// SAFETY: as above; the driver allows one context to be current on several
// threads at once, and every method takes `&self`.
unsafe impl Sync for CudaInterop {}

impl CudaInterop {
    /// Retains device 0's primary context: deliberately the device
    /// `CudaDevice` opens, which takes no ordinal for the same reason.
    pub(crate) fn retain_primary() -> Result<Self, CudaDriverError> {
        // SAFETY: the driver calls run in the order it requires, `cuInit`
        // first, and each result is checked before the next call relies on it.
        // Both out-params are live locals.
        unsafe {
            check("cuInit", cuInit(0))?;
            let mut device: CUdevice = 0;
            check("cuDeviceGet", cuDeviceGet(&mut device, 0))?;
            let mut ctx: CUcontext = std::ptr::null_mut();
            check(
                "cuDevicePrimaryCtxRetain",
                cuDevicePrimaryCtxRetain(&mut ctx, device),
            )?;
            Ok(Self { device, ctx })
        }
    }

    /// The device's UUID, as Vulkan reports it for the same GPU.
    pub(crate) fn uuid(&self) -> Result<[u8; 16], CudaDriverError> {
        let mut uuid = [0u8; 16];
        // SAFETY: `uuid` is a live 16-byte local, the layout of `CUuuid`, and
        // `self.device` is the device this was retained on.
        unsafe { check("cuDeviceGetUuid", cuDeviceGetUuid(&mut uuid, self.device))? };
        Ok(uuid)
    }

    /// Takes `fd` — memory another API allocated and exported, `size` bytes
    /// of it — and maps it into this context as one buffer.
    ///
    /// The descriptor is CUDA's once this succeeds, and is closed with the
    /// import; on failure it is closed here.
    pub(crate) fn import(
        self: &Arc<Self>,
        fd: OwnedFd,
        size: u64,
    ) -> Result<ImportedMemory, CudaDriverError> {
        let raw = fd.into_raw_fd();
        let desc = ExternalMemoryHandleDesc {
            type_: CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD,
            handle: ExternalMemoryHandle { fd: raw },
            size,
            flags: CUDA_EXTERNAL_MEMORY_DEDICATED,
            reserved: [0; 16],
        };
        self.with_context(|| {
            let mut memory: CUexternalMemory = std::ptr::null_mut();
            // SAFETY: `desc` is a fully initialized handle description for an
            // opaque fd, and `memory` a live out-param.
            let imported = unsafe {
                check(
                    "cuImportExternalMemory",
                    cuImportExternalMemory(&mut memory, &desc),
                )
            };
            if let Err(error) = imported {
                // SAFETY: the import failed, so the descriptor is still this
                // process's and nothing else will close it.
                drop(unsafe { <OwnedFd as std::os::fd::FromRawFd>::from_raw_fd(raw) });
                return Err(error);
            }
            let buffer = ExternalMemoryBufferDesc {
                offset: 0,
                size,
                ..Default::default()
            };
            let mut ptr: CUdeviceptr = 0;
            // SAFETY: `memory` was just imported and `buffer` asks for all of
            // it; `ptr` is a live out-param.
            let mapped = unsafe {
                check(
                    "cuExternalMemoryGetMappedBuffer",
                    cuExternalMemoryGetMappedBuffer(&mut ptr, memory, &buffer),
                )
            };
            if let Err(error) = mapped {
                // SAFETY: nothing was mapped from `memory`, which this owns.
                unsafe { cuDestroyExternalMemory(memory) };
                return Err(error);
            }
            Ok(ImportedMemory {
                interop: Arc::clone(self),
                memory,
                ptr,
            })
        })
    }

    /// Copies each of `rows` device to device, then waits for the copies to
    /// finish.
    ///
    /// The wait is what makes the result safe to hand to another API: it
    /// cannot see CUDA's work, so a copy still in flight would be read half
    /// done. It waits on the whole context, which is heavier than a shared
    /// semaphore would be — measured cost is part of what a later change to
    /// one would have to beat.
    ///
    /// # Safety
    /// Each source must be a device pointer in this context, readable for
    /// `rows` rows of `src_pitch` bytes, and `dst` must be a mapping from
    /// [`Self::import`] large enough for every destination written.
    pub(crate) unsafe fn copy_rows(
        &self,
        dst: CUdeviceptr,
        planes: &[DeviceRows],
    ) -> Result<(), CudaDriverError> {
        self.with_context(|| {
            for plane in planes {
                let copy = CudaMemcpy2D {
                    src_memory_type: CU_MEMORYTYPE_DEVICE,
                    src_device: plane.src,
                    src_pitch: plane.src_pitch,
                    dst_memory_type: CU_MEMORYTYPE_DEVICE,
                    dst_device: dst + plane.dst_offset,
                    dst_pitch: plane.dst_pitch,
                    width_in_bytes: plane.width_bytes,
                    height: plane.rows,
                    ..Default::default()
                };
                // SAFETY: the caller's contract above covers both ends of the
                // copy, and `copy` describes exactly the rows it named.
                unsafe { check("cuMemcpy2D", cuMemcpy2D_v2(&copy))? };
            }
            // SAFETY: a plain wait on the context `with_context` made current.
            unsafe { check("cuCtxSynchronize", cuCtxSynchronize()) }
        })
    }

    fn with_context<T>(
        &self,
        f: impl FnOnce() -> Result<T, CudaDriverError>,
    ) -> Result<T, CudaDriverError> {
        // SAFETY: `self.ctx` is the primary context this retained and still
        // holds a reference to.
        unsafe { check("cuCtxPushCurrent", cuCtxPushCurrent_v2(self.ctx))? };
        let value = f();
        let mut popped: CUcontext = std::ptr::null_mut();
        // SAFETY: balances the push above on this same thread.
        unsafe { check("cuCtxPopCurrent", cuCtxPopCurrent_v2(&mut popped))? };
        value
    }
}

impl Drop for CudaInterop {
    fn drop(&mut self) {
        // SAFETY: balances the retain in `retain_primary`; `Drop` runs once.
        unsafe { cuDevicePrimaryCtxRelease_v2(self.device) };
    }
}

/// One plane's rows, as [`CudaInterop::copy_rows`] copies them.
pub(crate) struct DeviceRows {
    /// Where the rows start, in the frame.
    pub(crate) src: CUdeviceptr,
    /// Bytes from one source row to the next.
    pub(crate) src_pitch: usize,
    /// Bytes of each row that are picture.
    pub(crate) width_bytes: usize,
    /// How many rows.
    pub(crate) rows: usize,
    /// Where they go, from the start of the destination.
    pub(crate) dst_offset: u64,
    /// Bytes from one destination row to the next.
    pub(crate) dst_pitch: usize,
}

/// Another API's memory, mapped into CUDA — freed from CUDA when dropped.
///
/// Must be dropped before the memory it maps is freed on the other side:
/// the mapping refers to that allocation.
pub(crate) struct ImportedMemory {
    interop: Arc<CudaInterop>,
    memory: CUexternalMemory,
    ptr: CUdeviceptr,
}

// SAFETY: the handle and pointer are only used through `interop`, which
// pushes its context around every call, on whichever thread drops this.
unsafe impl Send for ImportedMemory {}

impl ImportedMemory {
    /// The device pointer the whole buffer is mapped at.
    pub(crate) fn ptr(&self) -> CUdeviceptr {
        self.ptr
    }
}

impl Drop for ImportedMemory {
    fn drop(&mut self) {
        let (ptr, memory) = (self.ptr, self.memory);
        let _ = self.interop.with_context(|| {
            // SAFETY: `ptr` was mapped from `memory` by `import` and is freed
            // once, before the import it belongs to is destroyed, as the
            // driver requires.
            unsafe {
                cuMemFree_v2(ptr);
                cuDestroyExternalMemory(memory);
            }
            Ok(())
        });
    }
}
