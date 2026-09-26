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
    cuDevicePrimaryCtxRelease_v2, cuDevicePrimaryCtxRetain, cuInit, cuMemcpy2D_v2,
};

type CUexternalMemory = *mut std::ffi::c_void;
type CUmipmappedArray = *mut std::ffi::c_void;
pub(crate) type CUarray = *mut std::ffi::c_void;

/// `CU_MEMORYTYPE_ARRAY` — a copy's end is a CUDA array, laid out however
/// the driver lays arrays out, rather than pitched rows.
const CU_MEMORYTYPE_ARRAY: std::ffi::c_uint = 3;

/// `CU_AD_FORMAT_UNSIGNED_INT8`: each channel of an array element one byte.
const CU_AD_FORMAT_UNSIGNED_INT8: std::ffi::c_uint = 0x01;

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

/// `CUDA_ARRAY3D_DESCRIPTOR`, versioned by name (`_v2`) and unchanged since
/// CUDA 3.2. A depth of zero is a 2D array.
#[repr(C)]
#[derive(Clone, Copy)]
struct Array3dDescriptor {
    width: usize,
    height: usize,
    depth: usize,
    format: std::ffi::c_uint,
    num_channels: std::ffi::c_uint,
    flags: std::ffi::c_uint,
}

/// `CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC`.
#[repr(C)]
#[derive(Clone, Copy)]
struct ExternalMemoryMipmappedArrayDesc {
    offset: u64,
    array_desc: Array3dDescriptor,
    num_levels: std::ffi::c_uint,
    reserved: [std::ffi::c_uint; 16],
}

super::load::cuda_driver! {
    /// The 16 bytes Vulkan reports as `VkPhysicalDeviceIDProperties::deviceUUID`
    /// for the same GPU — what proves a Vulkan device is the one the frames
    /// are on, rather than the first one listed.
    fn cuDeviceGetUuid(uuid: *mut [u8; 16], device: CUdevice) -> CUresult;
    fn cuImportExternalMemory(
        memory: *mut CUexternalMemory,
        desc: *const ExternalMemoryHandleDesc,
    ) -> CUresult;
    fn cuExternalMemoryGetMappedMipmappedArray(
        array: *mut CUmipmappedArray,
        memory: CUexternalMemory,
        desc: *const ExternalMemoryMipmappedArrayDesc,
    ) -> CUresult;
    fn cuMipmappedArrayGetLevel(
        level_array: *mut CUarray,
        array: CUmipmappedArray,
        level: std::ffi::c_uint,
    ) -> CUresult;
    fn cuMipmappedArrayDestroy(array: CUmipmappedArray) -> CUresult;
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

    /// Takes `fd` — an image another API allocated and exported, `size`
    /// bytes of memory dedicated to it — and maps it into this context as a
    /// 2D array of `width` by `height` elements, each `channels` bytes.
    ///
    /// The shape must be the image's own: CUDA takes the layout of the
    /// memory from it, and an image of another shape or element size is
    /// read and written as garbage.
    ///
    /// The descriptor is CUDA's once this succeeds, and is closed with the
    /// import; on failure it is closed here.
    pub(crate) fn import_image(
        self: &Arc<Self>,
        fd: OwnedFd,
        size: u64,
        width: u32,
        height: u32,
        channels: u32,
    ) -> Result<ImportedArray, CudaDriverError> {
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
            let desc = ExternalMemoryMipmappedArrayDesc {
                offset: 0,
                array_desc: Array3dDescriptor {
                    width: width as usize,
                    height: height as usize,
                    depth: 0,
                    format: CU_AD_FORMAT_UNSIGNED_INT8,
                    num_channels: channels,
                    flags: 0,
                },
                num_levels: 1,
                reserved: [0; 16],
            };
            let mut mipmapped: CUmipmappedArray = std::ptr::null_mut();
            // SAFETY: `memory` was just imported and `desc` describes one level
            // at its start; `mipmapped` is a live out-param.
            let mapped = unsafe {
                check(
                    "cuExternalMemoryGetMappedMipmappedArray",
                    cuExternalMemoryGetMappedMipmappedArray(&mut mipmapped, memory, &desc),
                )
            };
            if let Err(error) = mapped {
                // SAFETY: nothing was mapped from `memory`, which this owns.
                unsafe { cuDestroyExternalMemory(memory) };
                return Err(error);
            }
            let mut array: CUarray = std::ptr::null_mut();
            // SAFETY: level 0 of the array just mapped, which has one level;
            // `array` is a live out-param.
            let level = unsafe {
                check(
                    "cuMipmappedArrayGetLevel",
                    cuMipmappedArrayGetLevel(&mut array, mipmapped, 0),
                )
            };
            if let Err(error) = level {
                // SAFETY: both were made above and nothing else refers to them;
                // the array goes first, as the driver requires.
                unsafe {
                    cuMipmappedArrayDestroy(mipmapped);
                    cuDestroyExternalMemory(memory);
                }
                return Err(error);
            }
            Ok(ImportedArray {
                interop: Arc::clone(self),
                memory,
                mipmapped,
                array,
            })
        })
    }

    /// Copies each of `planes` from device memory into an array, then waits
    /// for the copies to finish.
    ///
    /// The wait is what makes the result safe to hand to another API: it
    /// cannot see CUDA's work, so a copy still in flight would be read half
    /// done. It waits on the whole context, which is heavier than a shared
    /// semaphore would be — measured cost is part of what a later change to
    /// one would have to beat.
    ///
    /// # Safety
    /// Each source must be a device pointer in this context, readable for
    /// `rows` rows of `src_pitch` bytes, and each destination an array from
    /// [`Self::import_image`], still imported, at least `width_bytes` by
    /// `rows`.
    pub(crate) unsafe fn copy_into(&self, planes: &[ArrayRows]) -> Result<(), CudaDriverError> {
        self.with_context(|| {
            for plane in planes {
                let copy = CudaMemcpy2D {
                    src_memory_type: CU_MEMORYTYPE_DEVICE,
                    src_device: plane.src,
                    src_pitch: plane.src_pitch,
                    dst_memory_type: CU_MEMORYTYPE_ARRAY,
                    dst_array: plane.dst,
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

/// One plane's rows, as [`CudaInterop::copy_into`] copies them.
pub(crate) struct ArrayRows {
    /// Where the rows start, in the frame.
    pub(crate) src: CUdeviceptr,
    /// Bytes from one source row to the next.
    pub(crate) src_pitch: usize,
    /// Bytes of each row that are picture.
    pub(crate) width_bytes: usize,
    /// How many rows.
    pub(crate) rows: usize,
    /// Where they go: [`ImportedArray::array`].
    pub(crate) dst: CUarray,
}

/// Another API's image, mapped into CUDA as an array — let go of by CUDA
/// when dropped.
///
/// Must be dropped before the memory it maps is freed on the other side:
/// the mapping refers to that allocation.
pub(crate) struct ImportedArray {
    interop: Arc<CudaInterop>,
    memory: CUexternalMemory,
    mipmapped: CUmipmappedArray,
    array: CUarray,
}

// SAFETY: the handles are only used through `interop`, which pushes its
// context around every call, on whichever thread drops this.
unsafe impl Send for ImportedArray {}

impl ImportedArray {
    /// The image's one level, which is what a copy writes.
    pub(crate) fn array(&self) -> CUarray {
        self.array
    }
}

impl Drop for ImportedArray {
    fn drop(&mut self) {
        let (mipmapped, memory) = (self.mipmapped, self.memory);
        let _ = self.interop.with_context(|| {
            // SAFETY: `mipmapped` was mapped from `memory` by `import_image`
            // and is destroyed once, before the import it belongs to, as the
            // driver requires; the level taken from it goes with it.
            unsafe {
                cuMipmappedArrayDestroy(mipmapped);
                cuDestroyExternalMemory(memory);
            }
            Ok(())
        });
    }
}
