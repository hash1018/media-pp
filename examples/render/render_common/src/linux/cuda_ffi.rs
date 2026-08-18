//! The handful of CUDA driver API entry points this renderer needs, declared
//! directly rather than pulled in through a CUDA binding crate.
//!
//! Everything here is the *driver* API (`cu*`, `libcuda.so.1`), which the
//! NVIDIA driver ships — no CUDA toolkit install required to build or run.
//! The signatures below come from `cuda.h`; they are plain C functions with
//! scalar/pointer arguments, not structs whose layout has to be mirrored,
//! except `CUDA_EXTERNAL_MEMORY_*` and `CUDA_MEMCPY2D`, which are versioned
//! by name and stable since CUDA 10.

use std::ffi::{c_int, c_uint, c_void};

pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUcontext = *mut c_void;
pub type CUdeviceptr = u64;
pub type CUexternalMemory = *mut c_void;

pub const CUDA_SUCCESS: CUresult = 0;

/// `CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD` — a POSIX file descriptor
/// exported by another API. This is the only handle type that matters here:
/// Vulkan exports its memory as an opaque fd and CUDA imports it.
pub const CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD: c_uint = 1;

/// `CU_MEMORYTYPE_DEVICE`.
pub const CU_MEMORYTYPE_DEVICE: c_uint = 2;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CudaExternalMemoryHandleDesc {
    pub type_: c_uint,
    pub handle: CudaExternalMemoryHandle,
    pub size: u64,
    pub flags: c_uint,
    pub reserved: [c_uint; 16],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union CudaExternalMemoryHandle {
    pub fd: c_int,
    pub win32: CudaExternalMemoryHandleWin32,
    pub nv_sci_buf_object: *const c_void,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CudaExternalMemoryHandleWin32 {
    pub handle: *mut c_void,
    pub name: *const c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CudaExternalMemoryBufferDesc {
    pub offset: u64,
    pub size: u64,
    pub flags: c_uint,
    pub reserved: [c_uint; 16],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CudaMemcpy2D {
    pub src_x_in_bytes: usize,
    pub src_y: usize,
    pub src_memory_type: c_uint,
    pub src_host: *const c_void,
    pub src_device: CUdeviceptr,
    pub src_array: *mut c_void,
    pub src_pitch: usize,

    pub dst_x_in_bytes: usize,
    pub dst_y: usize,
    pub dst_memory_type: c_uint,
    pub dst_host: *mut c_void,
    pub dst_device: CUdeviceptr,
    pub dst_array: *mut c_void,
    pub dst_pitch: usize,

    pub width_in_bytes: usize,
    pub height: usize,
}

// SAFETY of the block: these are the driver's own C ABI declarations. Every
// call site below checks the returned `CUresult`.
#[link(name = "cuda")]
unsafe extern "C" {
    pub fn cuInit(flags: c_uint) -> CUresult;
    pub fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;
    /// Fills `uuid` with the 16-byte device UUID — the same value Vulkan
    /// reports as `VkPhysicalDeviceIDProperties::deviceUUID`, which is what
    /// lets this renderer prove its Vulkan device is the GPU the frames were
    /// decoded on instead of assuming it.
    pub fn cuDeviceGetUuid(uuid: *mut [u8; 16], device: CUdevice) -> CUresult;
    /// Retains the device's *primary* context — the same context
    /// `media_pp::elements::CudaDevice` makes FFmpeg use, so no FFmpeg
    /// struct has to be mirrored to reach it.
    pub fn cuDevicePrimaryCtxRetain(ctx: *mut CUcontext, device: CUdevice) -> CUresult;
    pub fn cuDevicePrimaryCtxRelease_v2(device: CUdevice) -> CUresult;
    pub fn cuCtxPushCurrent_v2(ctx: CUcontext) -> CUresult;
    pub fn cuCtxPopCurrent_v2(ctx: *mut CUcontext) -> CUresult;
    pub fn cuCtxSynchronize() -> CUresult;
    pub fn cuImportExternalMemory(
        ext_mem: *mut CUexternalMemory,
        desc: *const CudaExternalMemoryHandleDesc,
    ) -> CUresult;
    pub fn cuExternalMemoryGetMappedBuffer(
        dev_ptr: *mut CUdeviceptr,
        ext_mem: CUexternalMemory,
        desc: *const CudaExternalMemoryBufferDesc,
    ) -> CUresult;
    pub fn cuDestroyExternalMemory(ext_mem: CUexternalMemory) -> CUresult;
    pub fn cuMemFree_v2(dev_ptr: CUdeviceptr) -> CUresult;
    pub fn cuMemcpy2D_v2(copy: *const CudaMemcpy2D) -> CUresult;
    pub fn cuGetErrorString(error: CUresult, str_: *mut *const std::ffi::c_char) -> CUresult;
}

/// Turns a non-success `CUresult` into a readable message.
pub fn cuda_error(what: &str, result: CUresult) -> String {
    let mut raw: *const std::ffi::c_char = std::ptr::null();
    let name = unsafe {
        if cuGetErrorString(result, &mut raw) == CUDA_SUCCESS && !raw.is_null() {
            std::ffi::CStr::from_ptr(raw).to_string_lossy().into_owned()
        } else {
            format!("unknown CUDA error {result}")
        }
    };
    format!("{what} failed: {name}")
}

/// Runs `f` with `ctx` current, restoring the previous context afterwards.
///
/// Every CUDA call this renderer makes goes through here: the pipeline calls
/// into `submit_nv12` from whichever thread the sink runs on, and a CUDA
/// context is per-thread state, so it cannot be pushed once at startup.
pub fn with_context<T>(ctx: CUcontext, f: impl FnOnce() -> T) -> Result<T, String> {
    let result = unsafe { cuCtxPushCurrent_v2(ctx) };
    if result != CUDA_SUCCESS {
        return Err(cuda_error("cuCtxPushCurrent", result));
    }
    let value = f();
    let mut popped: CUcontext = std::ptr::null_mut();
    let result = unsafe { cuCtxPopCurrent_v2(&mut popped) };
    if result != CUDA_SUCCESS {
        return Err(cuda_error("cuCtxPopCurrent", result));
    }
    Ok(value)
}
