//! The handful of CUDA driver API entry points this crate calls directly,
//! declared here rather than pulled in through a CUDA binding crate.
//!
//! Everything here is the *driver* API (`cu*`), which the NVIDIA driver
//! itself ships — no CUDA toolkit is needed to build or run, the same
//! property `render_common`'s own `cuda_ffi` relies on. The signatures come
//! from `cuda.h`; they are plain C functions with scalar/pointer arguments
//! rather than structs whose layout has to be mirrored, except
//! `CUDA_MEMCPY2D`, which is versioned by name (`cuMemcpy2D_v2`) and has been
//! stable since CUDA 4.
//!
//! # Why this exists at all
//!
//! [`crate::elements::CudaVideoCompositor`] has to place one surface inside
//! another at an arbitrary offset, and libavfilter offers no CUDA filter that
//! can: `overlay_cuda` cannot crop, so `VideoFit::Cover` is not expressible,
//! and it ignores runtime commands, so moving a layer would mean rebuilding a
//! filter graph. A 2D device-to-device copy does all of it directly.

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};

use thiserror::Error as ThisError;

use crate::color::Color;

pub(crate) type CUresult = c_int;
pub(crate) type CUdevice = c_int;
pub(crate) type CUcontext = *mut c_void;
pub(crate) type CUdeviceptr = u64;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;

const CUDA_SUCCESS: CUresult = 0;
/// `CU_MEMORYTYPE_DEVICE`.
const CU_MEMORYTYPE_DEVICE: c_uint = 2;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CudaMemcpy2D {
    src_x_in_bytes: usize,
    src_y: usize,
    src_memory_type: c_uint,
    src_host: *const c_void,
    src_device: CUdeviceptr,
    src_array: *mut c_void,
    src_pitch: usize,

    dst_x_in_bytes: usize,
    dst_y: usize,
    dst_memory_type: c_uint,
    dst_host: *mut c_void,
    dst_device: CUdeviceptr,
    dst_array: *mut c_void,
    dst_pitch: usize,

    width_in_bytes: usize,
    height: usize,
}

/// The BGRA-to-NV12 conversion nothing else on this crate's CUDA path can do.
///
/// `scale_cuda` resizes but does not convert — FFmpeg 8.1 answers a BGRA
/// input with "Unsupported conversion: bgra -> semiplanar8" — and
/// `colorspace_cuda` only moves YUV between ranges. Without this kernel a
/// BGRA surface can reach [`crate::elements::CudaEncoder`], which ingests it
/// directly, and nothing else: [`crate::elements::CudaVideoCompositor`] and
/// [`crate::elements::CudaRenderer`] both work in NV12.
///
/// Two entry points rather than one: luma is a thread per pixel, chroma a
/// thread per 2x2 block, and splitting them keeps each a straight line of
/// loads, arithmetic, and one store. The colour maths is BT.709 limited
/// range, deliberately the same definition [`rgb_to_bt709_limited`] uses for
/// compositor backgrounds so a converted capture and a filled background
/// agree, and written in the same operation order so a test can compare every
/// byte against that expression instead of a tolerance.
const CONVERT_PTX: &str = r#"
.version 6.0
.target sm_50
.address_size 64

.visible .entry bgra_to_luma(
    .param .u64 dst,
    .param .u32 dst_pitch,
    .param .u64 src,
    .param .u32 src_pitch,
    .param .u32 width,
    .param .u32 height
)
{
    .reg .pred  %p<4>;
    .reg .b16   %rs<8>;
    .reg .b32   %r<24>;
    .reg .f32   %f<20>;
    .reg .b64   %rd<12>;

    ld.param.u64    %rd1, [dst];
    ld.param.u32    %r1, [dst_pitch];
    ld.param.u64    %rd2, [src];
    ld.param.u32    %r2, [src_pitch];
    ld.param.u32    %r3, [width];
    ld.param.u32    %r4, [height];

    mov.u32         %r5, %ctaid.x;
    mov.u32         %r6, %ntid.x;
    mov.u32         %r7, %tid.x;
    mad.lo.s32      %r8, %r5, %r6, %r7;
    mov.u32         %r9, %ctaid.y;
    mov.u32         %r10, %ntid.y;
    mov.u32         %r11, %tid.y;
    mad.lo.s32      %r12, %r9, %r10, %r11;

    setp.ge.u32     %p1, %r8, %r3;
    @%p1 bra        LUMA_DONE;
    setp.ge.u32     %p2, %r12, %r4;
    @%p2 bra        LUMA_DONE;

    mul.lo.s32      %r13, %r12, %r2;
    shl.b32         %r14, %r8, 2;
    add.s32         %r15, %r13, %r14;
    cvt.u64.u32     %rd3, %r15;
    add.s64         %rd4, %rd2, %rd3;

    ld.global.u8    %rs1, [%rd4];
    ld.global.u8    %rs2, [%rd4+1];
    ld.global.u8    %rs3, [%rd4+2];

    cvt.u32.u16     %r16, %rs1;
    cvt.rn.f32.u32  %f1, %r16;
    cvt.u32.u16     %r17, %rs2;
    cvt.rn.f32.u32  %f2, %r17;
    cvt.u32.u16     %r18, %rs3;
    cvt.rn.f32.u32  %f3, %r18;

    mul.f32         %f4, %f3, 0f3E59B3D0;
    mul.f32         %f5, %f2, 0f3F371759;
    add.f32         %f6, %f4, %f5;
    mul.f32         %f7, %f1, 0f3D93DD98;
    add.f32         %f8, %f6, %f7;

    mul.f32         %f9, %f8, 0f435B0000;
    div.rn.f32      %f10, %f9, 0f437F0000;
    add.f32         %f11, %f10, 0f41800000;
    max.f32         %f12, %f11, 0f00000000;
    min.f32         %f13, %f12, 0f437F0000;
    add.f32         %f14, %f13, 0f3F000000;
    cvt.rzi.u32.f32 %r19, %f14;
    cvt.u16.u32     %rs4, %r19;

    mul.lo.s32      %r20, %r12, %r1;
    add.s32         %r21, %r20, %r8;
    cvt.u64.u32     %rd5, %r21;
    add.s64         %rd6, %rd1, %rd5;
    st.global.u8    [%rd6], %rs4;

LUMA_DONE:
    ret;
}

.visible .entry bgra_to_chroma(
    .param .u64 dst,
    .param .u32 dst_pitch,
    .param .u64 src,
    .param .u32 src_pitch,
    .param .u32 half_width,
    .param .u32 half_height
)
{
    .reg .pred  %p<4>;
    .reg .b16   %rs<20>;
    .reg .b32   %r<48>;
    .reg .f32   %f<32>;
    .reg .b64   %rd<16>;

    ld.param.u64    %rd1, [dst];
    ld.param.u32    %r1, [dst_pitch];
    ld.param.u64    %rd2, [src];
    ld.param.u32    %r2, [src_pitch];
    ld.param.u32    %r3, [half_width];
    ld.param.u32    %r4, [half_height];

    mov.u32         %r5, %ctaid.x;
    mov.u32         %r6, %ntid.x;
    mov.u32         %r7, %tid.x;
    mad.lo.s32      %r8, %r5, %r6, %r7;
    mov.u32         %r9, %ctaid.y;
    mov.u32         %r10, %ntid.y;
    mov.u32         %r11, %tid.y;
    mad.lo.s32      %r12, %r9, %r10, %r11;

    setp.ge.u32     %p1, %r8, %r3;
    @%p1 bra        CHROMA_DONE;
    setp.ge.u32     %p2, %r12, %r4;
    @%p2 bra        CHROMA_DONE;

    shl.b32         %r13, %r12, 1;
    shl.b32         %r14, %r8, 1;
    mul.lo.s32      %r15, %r13, %r2;
    shl.b32         %r16, %r14, 2;
    add.s32         %r17, %r15, %r16;
    cvt.u64.u32     %rd3, %r17;
    add.s64         %rd4, %rd2, %rd3;
    cvt.u64.u32     %rd5, %r2;
    add.s64         %rd6, %rd4, %rd5;

    ld.global.u8    %rs1, [%rd4];
    ld.global.u8    %rs2, [%rd4+1];
    ld.global.u8    %rs3, [%rd4+2];
    ld.global.u8    %rs4, [%rd4+4];
    ld.global.u8    %rs5, [%rd4+5];
    ld.global.u8    %rs6, [%rd4+6];
    ld.global.u8    %rs7, [%rd6];
    ld.global.u8    %rs8, [%rd6+1];
    ld.global.u8    %rs9, [%rd6+2];
    ld.global.u8    %rs10, [%rd6+4];
    ld.global.u8    %rs11, [%rd6+5];
    ld.global.u8    %rs12, [%rd6+6];

    cvt.u32.u16     %r18, %rs1;
    cvt.u32.u16     %r19, %rs4;
    add.s32         %r20, %r18, %r19;
    cvt.u32.u16     %r21, %rs7;
    add.s32         %r22, %r20, %r21;
    cvt.u32.u16     %r23, %rs10;
    add.s32         %r24, %r22, %r23;
    cvt.rn.f32.u32  %f1, %r24;
    mul.f32         %f2, %f1, 0f3E800000;

    cvt.u32.u16     %r25, %rs2;
    cvt.u32.u16     %r26, %rs5;
    add.s32         %r27, %r25, %r26;
    cvt.u32.u16     %r28, %rs8;
    add.s32         %r29, %r27, %r28;
    cvt.u32.u16     %r30, %rs11;
    add.s32         %r31, %r29, %r30;
    cvt.rn.f32.u32  %f3, %r31;
    mul.f32         %f4, %f3, 0f3E800000;

    cvt.u32.u16     %r32, %rs3;
    cvt.u32.u16     %r33, %rs6;
    add.s32         %r34, %r32, %r33;
    cvt.u32.u16     %r35, %rs9;
    add.s32         %r36, %r34, %r35;
    cvt.u32.u16     %r37, %rs12;
    add.s32         %r38, %r36, %r37;
    cvt.rn.f32.u32  %f5, %r38;
    mul.f32         %f6, %f5, 0f3E800000;

    mul.f32         %f7, %f6, 0f3E59B3D0;
    mul.f32         %f8, %f4, 0f3F371759;
    add.f32         %f9, %f7, %f8;
    mul.f32         %f10, %f2, 0f3D93DD98;
    add.f32         %f11, %f9, %f10;

    sub.f32         %f12, %f2, %f11;
    div.rn.f32      %f13, %f12, 0f3FED844D;
    mul.f32         %f14, %f13, 0f43600000;
    div.rn.f32      %f15, %f14, 0f437F0000;
    add.f32         %f16, %f15, 0f43000000;
    max.f32         %f17, %f16, 0f00000000;
    min.f32         %f18, %f17, 0f437F0000;
    add.f32         %f19, %f18, 0f3F000000;
    cvt.rzi.u32.f32 %r39, %f19;
    cvt.u16.u32     %rs13, %r39;

    sub.f32         %f20, %f6, %f11;
    div.rn.f32      %f21, %f20, 0f3FC9930C;
    mul.f32         %f22, %f21, 0f43600000;
    div.rn.f32      %f23, %f22, 0f437F0000;
    add.f32         %f24, %f23, 0f43000000;
    max.f32         %f25, %f24, 0f00000000;
    min.f32         %f26, %f25, 0f437F0000;
    add.f32         %f27, %f26, 0f3F000000;
    cvt.rzi.u32.f32 %r40, %f27;
    cvt.u16.u32     %rs14, %r40;

    mul.lo.s32      %r41, %r12, %r1;
    shl.b32         %r42, %r8, 1;
    add.s32         %r43, %r41, %r42;
    cvt.u64.u32     %rd7, %r43;
    add.s64         %rd8, %rd1, %rd7;
    st.global.u8    [%rd8], %rs13;
    st.global.u8    [%rd8+1], %rs14;

CHROMA_DONE:
    ret;
}
"#;

// SAFETY of the block: these are the driver's own C ABI declarations, and
// every call site below checks the returned `CUresult`. On Windows the
// driver's `nvcuda.dll` is linked by name — `raw-dylib` needs no import
// library, so no CUDA toolkit install is required there either.
#[cfg_attr(windows, link(name = "nvcuda", kind = "raw-dylib"))]
#[cfg_attr(not(windows), link(name = "cuda"))]
unsafe extern "C" {
    fn cuInit(flags: c_uint) -> CUresult;
    fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;
    /// Retains the device's *primary* context — the same one
    /// [`crate::elements::CudaDevice`] makes FFmpeg use, so the frames these
    /// calls touch are reachable without mirroring any FFmpeg struct.
    fn cuDevicePrimaryCtxRetain(ctx: *mut CUcontext, device: CUdevice) -> CUresult;
    fn cuDevicePrimaryCtxRelease_v2(device: CUdevice) -> CUresult;
    fn cuCtxPushCurrent_v2(ctx: CUcontext) -> CUresult;
    fn cuCtxPopCurrent_v2(ctx: *mut CUcontext) -> CUresult;
    fn cuMemcpy2D_v2(copy: *const CudaMemcpy2D) -> CUresult;
    fn cuMemAlloc_v2(ptr: *mut CUdeviceptr, size: usize) -> CUresult;
    fn cuMemFree_v2(ptr: CUdeviceptr) -> CUresult;
    fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const c_void, size: usize) -> CUresult;
    fn cuMemsetD2D8_v2(
        dst: CUdeviceptr,
        dst_pitch: usize,
        value: u8,
        width: usize,
        height: usize,
    ) -> CUresult;
    fn cuMemsetD2D16_v2(
        dst: CUdeviceptr,
        dst_pitch: usize,
        value: u16,
        width: usize,
        height: usize,
    ) -> CUresult;
    fn cuGetErrorString(error: CUresult, str_: *mut *const c_char) -> CUresult;
    /// Takes PTX *text* as well as a compiled cubin: the driver carries its
    /// own JIT, which is what lets this crate ship a kernel as a string
    /// without a CUDA toolchain anywhere in the build.
    fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;
    fn cuModuleUnload(module: CUmodule) -> CUresult;
    fn cuModuleGetFunction(
        func: *mut CUfunction,
        module: CUmodule,
        name: *const c_char,
    ) -> CUresult;
    fn cuLaunchKernel(
        f: CUfunction,
        grid_x: c_uint,
        grid_y: c_uint,
        grid_z: c_uint,
        block_x: c_uint,
        block_y: c_uint,
        block_z: c_uint,
        shared_bytes: c_uint,
        stream: *mut c_void,
        params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult;
    fn cuCtxSynchronize() -> CUresult;
}

/// The one kernel this crate runs, as PTX the driver JIT-compiles at load.
///
/// # Why PTX rather than CUDA C
///
/// Compiling CUDA C needs `nvcc` or NVRTC, both of which ship with the CUDA
/// *toolkit* — a build requirement this crate deliberately does not impose,
/// since everything else it does needs only the driver. PTX is the driver's
/// own input format, so a kernel written here is a plain string constant:
/// nothing to compile, nothing to generate, nothing to check in. `.target
/// sm_50` is a floor, not a pin — the JIT recompiles it for whatever GPU is
/// actually present.
///
/// # What it computes
///
/// `dst = (src * alpha + dst * (255 - alpha) + 127) / 255`, one byte at a
/// time over a 2D region. Every term is non-negative, so the rounding is
/// symmetric and the division is unsigned — the reference implementation in
/// this module's tests is the same expression, and they are compared pixel
/// for pixel.
///
/// One kernel covers both NV12 planes. Luma is a byte per pixel; chroma is
/// interleaved `(U, V)` bytes, and blending each byte independently is
/// exactly right for both, so the chroma pass is the same call with the
/// plane's own byte width and half the rows.
///
/// `blend_masked` is the same mix with a per-pixel alpha and a constant
/// color, which is what a text layer needs: glyph coverage varies, the color
/// does not. Two things differ between its passes rather than one — chroma
/// alternates `(U, V)` by byte parity, hence `value_even`/`value_odd`, and
/// its mask is half resolution, hence `mask_shift`.
const BLEND_PTX: &str = r#"
.version 6.0
.target sm_50
.address_size 64

.visible .entry blend_plane(
    .param .u64 dst,
    .param .u32 dst_pitch,
    .param .u64 src,
    .param .u32 src_pitch,
    .param .u32 width,
    .param .u32 height,
    .param .u32 alpha
)
{
    .reg .pred  %p<4>;
    .reg .b16   %rs<4>;
    .reg .b32   %r<32>;
    .reg .b64   %rd<16>;

    ld.param.u64    %rd1, [dst];
    ld.param.u32    %r1, [dst_pitch];
    ld.param.u64    %rd2, [src];
    ld.param.u32    %r2, [src_pitch];
    ld.param.u32    %r3, [width];
    ld.param.u32    %r4, [height];
    ld.param.u32    %r5, [alpha];

    mov.u32         %r6, %ctaid.x;
    mov.u32         %r7, %ntid.x;
    mov.u32         %r8, %tid.x;
    mad.lo.s32      %r9, %r6, %r7, %r8;
    mov.u32         %r10, %ctaid.y;
    mov.u32         %r11, %ntid.y;
    mov.u32         %r12, %tid.y;
    mad.lo.s32      %r13, %r10, %r11, %r12;

    setp.ge.u32     %p1, %r9, %r3;
    @%p1 bra        DONE;
    setp.ge.u32     %p2, %r13, %r4;
    @%p2 bra        DONE;

    mad.lo.s32      %r14, %r13, %r1, %r9;
    cvt.u64.u32     %rd3, %r14;
    add.s64         %rd4, %rd1, %rd3;
    mad.lo.s32      %r15, %r13, %r2, %r9;
    cvt.u64.u32     %rd5, %r15;
    add.s64         %rd6, %rd2, %rd5;

    ld.global.u8    %r16, [%rd4];
    ld.global.u8    %r17, [%rd6];
    mul.lo.s32      %r18, %r17, %r5;
    sub.s32         %r19, 255, %r5;
    mul.lo.s32      %r20, %r16, %r19;
    add.s32         %r21, %r18, %r20;
    add.s32         %r22, %r21, 127;
    div.u32         %r23, %r22, 255;

    cvt.u16.u32     %rs1, %r23;
    st.global.u8    [%rd4], %rs1;
DONE:
    ret;
}

.visible .entry blend_masked(
    .param .u64 dst,
    .param .u32 dst_pitch,
    .param .u64 mask,
    .param .u32 mask_pitch,
    .param .u32 width,
    .param .u32 height,
    .param .u32 value_even,
    .param .u32 value_odd,
    .param .u32 opacity,
    .param .u32 mask_shift
)
{
    .reg .pred  %p<4>;
    .reg .b16   %rs<4>;
    .reg .b32   %r<40>;
    .reg .b64   %rd<16>;

    ld.param.u64    %rd1, [dst];
    ld.param.u32    %r1, [dst_pitch];
    ld.param.u64    %rd2, [mask];
    ld.param.u32    %r2, [mask_pitch];
    ld.param.u32    %r3, [width];
    ld.param.u32    %r4, [height];
    ld.param.u32    %r5, [value_even];
    ld.param.u32    %r6, [value_odd];
    ld.param.u32    %r7, [opacity];
    ld.param.u32    %r31, [mask_shift];

    mov.u32         %r8, %ctaid.x;
    mov.u32         %r9, %ntid.x;
    mov.u32         %r10, %tid.x;
    mad.lo.s32      %r11, %r8, %r9, %r10;
    mov.u32         %r12, %ctaid.y;
    mov.u32         %r13, %ntid.y;
    mov.u32         %r14, %tid.y;
    mad.lo.s32      %r15, %r12, %r13, %r14;

    setp.ge.u32     %p1, %r11, %r3;
    @%p1 bra        MDONE;
    setp.ge.u32     %p2, %r15, %r4;
    @%p2 bra        MDONE;

    mad.lo.s32      %r16, %r15, %r1, %r11;
    cvt.u64.u32     %rd3, %r16;
    add.s64         %rd4, %rd1, %rd3;

    shr.u32         %r32, %r11, %r31;
    mad.lo.s32      %r17, %r15, %r2, %r32;
    cvt.u64.u32     %rd5, %r17;
    add.s64         %rd6, %rd2, %rd5;

    ld.global.u8    %r18, [%rd4];
    ld.global.u8    %r19, [%rd6];

    mul.lo.s32      %r20, %r19, %r7;
    add.s32         %r21, %r20, 127;
    div.u32         %r22, %r21, 255;

    and.b32         %r23, %r11, 1;
    setp.eq.u32     %p3, %r23, 0;
    selp.b32        %r24, %r5, %r6, %p3;

    mul.lo.s32      %r25, %r24, %r22;
    sub.s32         %r26, 255, %r22;
    mul.lo.s32      %r27, %r18, %r26;
    add.s32         %r28, %r25, %r27;
    add.s32         %r29, %r28, 127;
    div.u32         %r30, %r29, 255;

    cvt.u16.u32     %rs2, %r30;
    st.global.u8    [%rd4], %rs2;
MDONE:
    ret;
}
"#;

/// Errors from the CUDA driver calls this crate makes directly.
#[derive(Debug, ThisError)]
pub enum CudaDriverError {
    /// A named CUDA Driver API call returned an error.
    #[error("{call} failed: {message}")]
    Call {
        /// CUDA Driver API function that failed.
        call: &'static str,
        /// Driver-provided error description.
        message: String,
    },
    /// Loading or launching the built-in blend kernel failed.

    #[error("the CUDA driver rejected this crate's blend kernel: {0}")]
    KernelRejected(String),
    /// A text mask is too small to contain an NV12 chroma sample.

    #[error("a coverage mask smaller than 2x2 has no chroma samples to blend into")]
    EmptyMask,
}

fn check(call: &'static str, result: CUresult) -> Result<(), CudaDriverError> {
    if result == CUDA_SUCCESS {
        return Ok(());
    }
    let mut raw: *const c_char = std::ptr::null();
    // SAFETY: `cuGetErrorString` writes a pointer to a string the driver owns
    // and keeps for the process, so it stays valid for the copy. A code it does
    // not recognize leaves `raw` as the null it was initialized to, which is
    // what the check beside it is for.
    let message = unsafe {
        if cuGetErrorString(result, &mut raw) == CUDA_SUCCESS && !raw.is_null() {
            CStr::from_ptr(raw).to_string_lossy().into_owned()
        } else {
            format!("CUDA error {result}")
        }
    };
    Err(CudaDriverError::Call { call, message })
}

/// One retained reference to the device's primary CUDA context, plus the 2D
/// memory operations this crate issues against it.
///
/// A CUDA context is per-thread state, so every operation here pushes and
/// pops it rather than assuming it is current — the same reason
/// `render_common`'s `with_context` exists. Owning this keeps the primary
/// context alive for as long as the element that holds it, which is what
/// makes the pointers inside a frame it composites remain valid.
pub(crate) struct CudaDriver {
    device: CUdevice,
    ctx: CUcontext,
    /// The JIT-compiled [`BLEND_PTX`] module and its entry points. Loaded
    /// once at construction: the JIT costs milliseconds, and a compositor
    /// would otherwise pay it per frame.
    module: CUmodule,
    blend: CUfunction,
    blend_masked: CUfunction,
    /// [`CONVERT_PTX`] and its entry points, loaded alongside the blend
    /// module: one JIT at construction rather than one on the first frame
    /// that needs a conversion.
    convert_module: CUmodule,
    bgra_to_luma: CUfunction,
    bgra_to_chroma: CUfunction,
}

// SAFETY: a `CUcontext` is not thread-affine — it is pushed onto whichever
// thread uses it, which is exactly what `with_context` does around every
// call, and the CUDA driver allows one context to be current on several
// threads at once.
unsafe impl Send for CudaDriver {}

// SAFETY: nothing here is mutated through `&self`: the context, the modules,
// and the function handles are set at construction and only read afterwards,
// so concurrent calls are the driver's own thread-safe operations rather than
// shared mutable state. This is what lets a compositor and a text layer
// handle on another thread share one driver.
unsafe impl Sync for CudaDriver {}

impl CudaDriver {
    /// Retains the primary context of CUDA device 0 — deliberately the same
    /// device [`crate::elements::CudaDevice`] opens, and for the same reason
    /// it takes no ordinal: a composite mixing surfaces from two GPUs is not
    /// expressible here anyway.
    pub(crate) fn retain_primary() -> Result<Self, CudaDriverError> {
        // SAFETY: the driver API calls run in the order it requires — `cuInit`
        // before anything else, and a context current before a module is loaded into
        // it — and every result is checked before the next call depends on it. All
        // out-params are live locals. Both failure paths give back what they had
        // taken: the first module is unloaded by hand, since no `Self` owns it yet,
        // and the primary-context reference is released before returning.
        unsafe {
            check("cuInit", cuInit(0))?;
            let mut device: CUdevice = 0;
            check("cuDeviceGet", cuDeviceGet(&mut device, 0))?;
            let mut ctx: CUcontext = std::ptr::null_mut();
            check(
                "cuDevicePrimaryCtxRetain",
                cuDevicePrimaryCtxRetain(&mut ctx, device),
            )?;

            // Loading a module needs a current context, and this is before
            // there is a `Self` to push it through.
            check("cuCtxPushCurrent", cuCtxPushCurrent_v2(ctx))?;
            let loaded =
                load_module(BLEND_PTX, ["blend_plane", "blend_masked"]).and_then(|blend| {
                    match load_module(CONVERT_PTX, ["bgra_to_luma", "bgra_to_chroma"]) {
                        Ok(convert) => Ok((blend, convert)),
                        Err(error) => {
                            // The first module has no owner yet, so nothing else
                            // would ever unload it.
                            cuModuleUnload(blend.0);
                            Err(error)
                        }
                    }
                });
            let mut popped: CUcontext = std::ptr::null_mut();
            check("cuCtxPopCurrent", cuCtxPopCurrent_v2(&mut popped))?;
            let ((module, blend, blend_masked), (convert_module, bgra_to_luma, bgra_to_chroma)) =
                match loaded {
                    Ok(modules) => modules,
                    Err(error) => {
                        cuDevicePrimaryCtxRelease_v2(device);
                        return Err(error);
                    }
                };

            Ok(Self {
                device,
                ctx,
                module,
                blend,
                blend_masked,
                convert_module,
                bgra_to_luma,
                bgra_to_chroma,
            })
        }
    }

    fn with_context<T>(
        &self,
        f: impl FnOnce() -> Result<T, CudaDriverError>,
    ) -> Result<T, CudaDriverError> {
        // SAFETY: `self.ctx` is the primary context this retained at construction
        // and still holds a reference to, so it can be made current for as long as
        // `self` lives.
        unsafe { check("cuCtxPushCurrent", cuCtxPushCurrent_v2(self.ctx))? };
        let value = f();
        let mut popped: CUcontext = std::ptr::null_mut();
        // SAFETY: balances the push above on this same thread, which is what
        // `cuCtxPopCurrent` requires; `popped` is a live local for the context it
        // removes.
        unsafe { check("cuCtxPopCurrent", cuCtxPopCurrent_v2(&mut popped))? };
        value
    }

    /// Fills an NV12 surface with one opaque color.
    ///
    /// Two operations rather than one because NV12 is planar: luma is a byte
    /// per pixel, chroma is a `(U, V)` byte pair per 2x2 block — which is
    /// exactly a 16-bit pattern, so `cuMemsetD2D16` writes it without a
    /// kernel of its own.
    pub(crate) fn fill_nv12(
        &self,
        surface: Nv12Surface,
        width: u32,
        height: u32,
        color: Color,
    ) -> Result<(), CudaDriverError> {
        let (y, u, v) = rgb_to_bt709_limited(color);
        // Little-endian: the low byte lands at the lower address, which in an
        // interleaved NV12 chroma plane is U.
        let chroma = u16::from(u) | (u16::from(v) << 8);
        // SAFETY: `with_context` has this driver's context current. Both plane
        // pointers and pitches come from a frame the caller has already validated,
        // and `width` x `height` is the caller's to keep within them — `cuMemsetD2D*`
        // writes exactly that rectangle at the given pitch.
        self.with_context(|| unsafe {
            check(
                "cuMemsetD2D8",
                cuMemsetD2D8_v2(
                    surface.luma,
                    surface.luma_pitch,
                    y,
                    width as usize,
                    height as usize,
                ),
            )?;
            check(
                "cuMemsetD2D16",
                cuMemsetD2D16_v2(
                    surface.chroma,
                    surface.chroma_pitch,
                    chroma,
                    (width / 2) as usize,
                    (height / 2) as usize,
                ),
            )
        })
    }

    /// Copies a rectangle of one NV12 surface into another, device to device.
    pub(crate) fn blit_nv12(
        &self,
        source: Nv12Surface,
        destination: Nv12Surface,
        region: Nv12Region,
    ) -> Result<(), CudaDriverError> {
        let Nv12Region {
            source_x,
            source_y,
            destination_x,
            destination_y,
            width,
            height,
        } = region;
        debug_assert!(
            [
                source_x,
                source_y,
                destination_x,
                destination_y,
                width,
                height
            ]
            .iter()
            .all(|value| value.is_multiple_of(2)),
            "NV12 blits must be aligned to the 2x2 chroma grid"
        );
        if width == 0 || height == 0 {
            return Ok(());
        }
        // SAFETY: `with_context` has the context current. Each `CudaMemcpy2D` is a
        // live local with both memory types set to device, so the driver reads the
        // device pointers and ignores the host and array fields `Default` left null.
        // Keeping the rectangle inside both surfaces is the caller's contract; the
        // `debug_assert` above only checks its 2x2 alignment.
        self.with_context(|| unsafe {
            let luma = CudaMemcpy2D {
                src_memory_type: CU_MEMORYTYPE_DEVICE,
                src_device: source.luma,
                src_pitch: source.luma_pitch,
                src_x_in_bytes: source_x as usize,
                src_y: source_y as usize,
                dst_memory_type: CU_MEMORYTYPE_DEVICE,
                dst_device: destination.luma,
                dst_pitch: destination.luma_pitch,
                dst_x_in_bytes: destination_x as usize,
                dst_y: destination_y as usize,
                width_in_bytes: width as usize,
                height: height as usize,
                ..CudaMemcpy2D::default()
            };
            check("cuMemcpy2D", cuMemcpy2D_v2(&luma))?;

            // Half the resolution in both axes, but two bytes per sample, so
            // the byte width stays `width` while the row count halves.
            let chroma = CudaMemcpy2D {
                src_memory_type: CU_MEMORYTYPE_DEVICE,
                src_device: source.chroma,
                src_pitch: source.chroma_pitch,
                src_x_in_bytes: source_x as usize,
                src_y: (source_y / 2) as usize,
                dst_memory_type: CU_MEMORYTYPE_DEVICE,
                dst_device: destination.chroma,
                dst_pitch: destination.chroma_pitch,
                dst_x_in_bytes: destination_x as usize,
                dst_y: (destination_y / 2) as usize,
                width_in_bytes: width as usize,
                height: (height / 2) as usize,
                ..CudaMemcpy2D::default()
            };
            check("cuMemcpy2D", cuMemcpy2D_v2(&chroma))
        })
    }

    /// Blends a rectangle of one NV12 surface into another with a uniform
    /// `alpha`, on the GPU — what [`CudaDriver::blit_nv12`] cannot do, since
    /// a copy has no way to mix with what is already there.
    ///
    /// `alpha` is 0 (leave the destination alone) to 255 (replace it). A
    /// caller with 255 should use `blit_nv12` instead: a copy moves whole
    /// rows at the memory system's own rate, where this reads, mixes, and
    /// writes every byte.
    ///
    /// Launches asynchronously. [`CudaDriver::synchronize`] is what makes the
    /// result visible to anything outside this context's stream ordering.
    pub(crate) fn blend_nv12(
        &self,
        source: Nv12Surface,
        destination: Nv12Surface,
        region: Nv12Region,
        alpha: u8,
    ) -> Result<(), CudaDriverError> {
        let Nv12Region {
            source_x,
            source_y,
            destination_x,
            destination_y,
            width,
            height,
        } = region;
        debug_assert!(
            [
                source_x,
                source_y,
                destination_x,
                destination_y,
                width,
                height
            ]
            .iter()
            .all(|value| value.is_multiple_of(2)),
            "NV12 blends must be aligned to the 2x2 chroma grid"
        );
        if width == 0 || height == 0 {
            return Ok(());
        }
        self.with_context(|| {
            // Luma: one byte per pixel.
            self.launch_blend(
                destination.luma
                    + u64::from(destination_y) * destination.luma_pitch as u64
                    + u64::from(destination_x),
                destination.luma_pitch,
                source.luma + u64::from(source_y) * source.luma_pitch as u64 + u64::from(source_x),
                source.luma_pitch,
                width,
                height,
                alpha,
            )?;
            // Chroma: interleaved (U, V) at half resolution, so the same byte
            // width covers half as many samples over half as many rows.
            self.launch_blend(
                destination.chroma
                    + u64::from(destination_y / 2) * destination.chroma_pitch as u64
                    + u64::from(destination_x),
                destination.chroma_pitch,
                source.chroma
                    + u64::from(source_y / 2) * source.chroma_pitch as u64
                    + u64::from(source_x),
                source.chroma_pitch,
                width,
                height / 2,
                alpha,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_blend(
        &self,
        mut dst: CUdeviceptr,
        dst_pitch: usize,
        mut src: CUdeviceptr,
        src_pitch: usize,
        width: u32,
        height: u32,
        alpha: u8,
    ) -> Result<(), CudaDriverError> {
        // 16x16 threads: one warp wide in x, which keeps the byte loads of a
        // row coalesced.
        const BLOCK: u32 = 16;
        let mut dst_pitch = dst_pitch as u32;
        let mut src_pitch = src_pitch as u32;
        let mut width = width;
        let mut height = height;
        let mut alpha = u32::from(alpha);
        let mut params: [*mut c_void; 7] = [
            (&mut dst) as *mut _ as *mut c_void,
            (&mut dst_pitch) as *mut _ as *mut c_void,
            (&mut src) as *mut _ as *mut c_void,
            (&mut src_pitch) as *mut _ as *mut c_void,
            (&mut width) as *mut _ as *mut c_void,
            (&mut height) as *mut _ as *mut c_void,
            (&mut alpha) as *mut _ as *mut c_void,
        ];
        // SAFETY: `params` holds one pointer per parameter `blend_plane` declares,
        // in that order, each pointing at a live local that outlives the launch —
        // `cuLaunchKernel` reads the argument values before it returns. The context
        // is current: every caller reaches this inside `with_context`.
        unsafe {
            check(
                "cuLaunchKernel",
                cuLaunchKernel(
                    self.blend,
                    width.div_ceil(BLOCK),
                    height.div_ceil(BLOCK),
                    1,
                    BLOCK,
                    BLOCK,
                    1,
                    0,
                    std::ptr::null_mut(),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
            )
        }
    }

    /// Converts a BGRA surface into an NV12 one, both CUDA-resident and both
    /// `width` x `height`.
    ///
    /// Dimensions must be even: NV12 chroma is 2x2 subsampled, so an odd
    /// extent has no whole chroma sample to write. The caller validates that
    /// once at construction — see [`crate::elements::CudaConverter`] — rather
    /// than this rejecting a frame per call.
    ///
    /// Two launches, one per plane. Nothing is synchronized here: a caller
    /// that needs the result on the host calls [`Self::synchronize`], the
    /// same split the blends use.
    pub(crate) fn bgra_to_nv12(
        &self,
        source: BgraSurface,
        destination: Nv12Surface,
        width: u32,
        height: u32,
    ) -> Result<(), CudaDriverError> {
        // 16x16 threads: one warp wide in x, which keeps a row's byte loads
        // coalesced, the same shape the blends use.
        const BLOCK: u32 = 16;
        // SAFETY: `with_context` has the context current, and each `params` array
        // holds one pointer per parameter its kernel declares, in that order,
        // pointing at locals that outlive the launch. The surfaces are the caller's
        // already-validated frames, and the even dimensions the chroma launch
        // divides by are established once at construction rather than here.
        self.with_context(|| unsafe {
            let mut luma = destination.luma;
            let mut luma_pitch = destination.luma_pitch as u32;
            let mut pixels = source.pixels;
            let mut source_pitch = source.pitch as u32;
            let mut width = width;
            let mut height = height;
            let mut luma_params: [*mut c_void; 6] = [
                (&mut luma) as *mut _ as *mut c_void,
                (&mut luma_pitch) as *mut _ as *mut c_void,
                (&mut pixels) as *mut _ as *mut c_void,
                (&mut source_pitch) as *mut _ as *mut c_void,
                (&mut width) as *mut _ as *mut c_void,
                (&mut height) as *mut _ as *mut c_void,
            ];
            check(
                "cuLaunchKernel",
                cuLaunchKernel(
                    self.bgra_to_luma,
                    width.div_ceil(BLOCK),
                    height.div_ceil(BLOCK),
                    1,
                    BLOCK,
                    BLOCK,
                    1,
                    0,
                    std::ptr::null_mut(),
                    luma_params.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
            )?;

            let mut chroma = destination.chroma;
            let mut chroma_pitch = destination.chroma_pitch as u32;
            let mut half_width = width / 2;
            let mut half_height = height / 2;
            let mut chroma_params: [*mut c_void; 6] = [
                (&mut chroma) as *mut _ as *mut c_void,
                (&mut chroma_pitch) as *mut _ as *mut c_void,
                (&mut pixels) as *mut _ as *mut c_void,
                (&mut source_pitch) as *mut _ as *mut c_void,
                (&mut half_width) as *mut _ as *mut c_void,
                (&mut half_height) as *mut _ as *mut c_void,
            ];
            check(
                "cuLaunchKernel",
                cuLaunchKernel(
                    self.bgra_to_chroma,
                    half_width.div_ceil(BLOCK),
                    half_height.div_ceil(BLOCK),
                    1,
                    BLOCK,
                    BLOCK,
                    1,
                    0,
                    std::ptr::null_mut(),
                    chroma_params.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
            )
        })
    }

    /// Waits for everything issued on this context, which a caller does once
    /// after a frame's blends rather than after each one.
    pub(crate) fn synchronize(&self) -> Result<(), CudaDriverError> {
        // SAFETY: `with_context` has this driver's context current, which is the
        // context `cuCtxSynchronize` waits on.
        self.with_context(|| unsafe { check("cuCtxSynchronize", cuCtxSynchronize()) })
    }
}

/// A rasterized glyph coverage mask living in device memory, at both the
/// resolutions an NV12 blend needs.
///
/// The half-resolution copy is built once here rather than sampled 2x2 in
/// the kernel: it changes only when the text does, and doing it on the CPU
/// keeps the kernel to one load per byte.
pub(crate) struct CudaMask {
    ctx: CUcontext,
    full: CUdeviceptr,
    half: CUdeviceptr,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

// SAFETY: the pointers are plain device allocations with no thread affinity,
// and `Drop` pushes the context it captured before freeing them, so a mask
// released on another thread is still released in the context it was
// allocated in.
unsafe impl Send for CudaMask {}

// SAFETY: nothing is mutated through `&self` — a mask is uploaded once and
// only read by the kernel afterwards. This is what lets one published mask be
// read by the compositor thread while the handle that made it lives on
// another.
unsafe impl Sync for CudaMask {}

impl Drop for CudaMask {
    fn drop(&mut self) {
        // SAFETY: both pointers came from `cuMemAlloc` in `upload_mask` and are
        // freed once, in the context they were allocated in — hence the push, and
        // hence the frees only running when it succeeded. That context is still
        // retained: a mask is owned by the compositor that owns the driver which
        // allocated it.
        unsafe {
            if cuCtxPushCurrent_v2(self.ctx) == CUDA_SUCCESS {
                cuMemFree_v2(self.full);
                cuMemFree_v2(self.half);
                let mut popped: CUcontext = std::ptr::null_mut();
                cuCtxPopCurrent_v2(&mut popped);
            }
        }
    }
}

impl CudaDriver {
    /// Uploads one coverage mask, tightly packed `width * height` bytes.
    ///
    /// Dimensions are rounded down to even: a text layer is blended into an
    /// NV12 surface, whose chroma covers 2x2 blocks, so an odd trailing row
    /// or column has nowhere to land.
    pub(crate) fn upload_mask(
        &self,
        coverage: &[u8],
        width: u32,
        height: u32,
    ) -> Result<CudaMask, CudaDriverError> {
        let full_width = width & !1;
        let full_height = height & !1;
        debug_assert_eq!(coverage.len(), (width * height) as usize);
        let (half_width, half_height) = (full_width / 2, full_height / 2);
        if full_width == 0 || full_height == 0 {
            return Err(CudaDriverError::EmptyMask);
        }

        // Average each 2x2 block, so a half-covered chroma sample is
        // half-covered rather than snapped to one of its four luma pixels.
        let mut half = vec![0u8; (half_width * half_height) as usize];
        for y in 0..half_height as usize {
            for x in 0..half_width as usize {
                let at = |dy: usize, dx: usize| {
                    u32::from(coverage[(y * 2 + dy) * width as usize + x * 2 + dx])
                };
                half[y * half_width as usize + x] =
                    ((at(0, 0) + at(0, 1) + at(1, 0) + at(1, 1) + 2) / 4) as u8;
            }
        }

        // SAFETY: `with_context` has the context current. Every out-param is a live
        // local, each row copy reads `full_width <= width` bytes from within
        // `coverage`'s own row, and the half-resolution upload is bounded by the
        // length of the buffer built for it just above.
        self.with_context(|| unsafe {
            let mut full = 0;
            check(
                "cuMemAlloc",
                cuMemAlloc_v2(&mut full, (full_width * full_height) as usize),
            )?;
            let mut half_ptr = 0;
            check("cuMemAlloc", cuMemAlloc_v2(&mut half_ptr, half.len()))?;

            // Row by row: the source rows are `width` apart, the destination
            // rows `full_width`, which differ whenever an odd column was
            // dropped.
            for y in 0..full_height as usize {
                let row = &coverage[y * width as usize..y * width as usize + full_width as usize];
                check(
                    "cuMemcpyHtoD",
                    cuMemcpyHtoD_v2(
                        full + (y * full_width as usize) as u64,
                        row.as_ptr().cast(),
                        full_width as usize,
                    ),
                )?;
            }
            check(
                "cuMemcpyHtoD",
                cuMemcpyHtoD_v2(half_ptr, half.as_ptr().cast(), half.len()),
            )?;
            Ok(CudaMask {
                ctx: self.ctx,
                full,
                half: half_ptr,
                width: full_width,
                height: full_height,
            })
        })
    }

    /// Draws `mask` into an NV12 surface in one flat `color`, weighted by
    /// coverage and by `opacity`.
    ///
    /// `x`/`y` are where the mask's top-left corner lands on the surface and
    /// must be even; `width`/`height` are the already-clipped extent.
    /// Launches asynchronously, like [`CudaDriver::blend_nv12`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn blend_mask_nv12(
        &self,
        destination: Nv12Surface,
        x: u32,
        y: u32,
        mask: &CudaMask,
        mask_x: u32,
        mask_y: u32,
        width: u32,
        height: u32,
        color: Color,
        opacity: u8,
    ) -> Result<(), CudaDriverError> {
        if width == 0 || height == 0 {
            return Ok(());
        }
        let (luma, u, v) = rgb_to_bt709_limited(color);
        self.with_context(|| {
            self.launch_masked(
                destination.luma + u64::from(y) * destination.luma_pitch as u64 + u64::from(x),
                destination.luma_pitch as u32,
                mask.full + u64::from(mask_y) * u64::from(mask.width) + u64::from(mask_x),
                mask.width,
                width,
                height,
                (u32::from(luma), u32::from(luma)),
                opacity,
                0,
            )?;
            self.launch_masked(
                destination.chroma
                    + u64::from(y / 2) * destination.chroma_pitch as u64
                    + u64::from(x),
                destination.chroma_pitch as u32,
                mask.half
                    + u64::from(mask_y / 2) * u64::from(mask.width / 2)
                    + u64::from(mask_x / 2),
                mask.width / 2,
                width,
                height / 2,
                (u32::from(u), u32::from(v)),
                opacity,
                1,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_masked(
        &self,
        mut dst: CUdeviceptr,
        mut dst_pitch: u32,
        mut mask: CUdeviceptr,
        mut mask_pitch: u32,
        width: u32,
        height: u32,
        values: (u32, u32),
        opacity: u8,
        shift: u32,
    ) -> Result<(), CudaDriverError> {
        const BLOCK: u32 = 16;
        let mut width = width;
        let mut height = height;
        let (mut value_even, mut value_odd) = values;
        let mut opacity = u32::from(opacity);
        let mut shift = shift;
        let mut params: [*mut c_void; 10] = [
            (&mut dst) as *mut _ as *mut c_void,
            (&mut dst_pitch) as *mut _ as *mut c_void,
            (&mut mask) as *mut _ as *mut c_void,
            (&mut mask_pitch) as *mut _ as *mut c_void,
            (&mut width) as *mut _ as *mut c_void,
            (&mut height) as *mut _ as *mut c_void,
            (&mut value_even) as *mut _ as *mut c_void,
            (&mut value_odd) as *mut _ as *mut c_void,
            (&mut opacity) as *mut _ as *mut c_void,
            (&mut shift) as *mut _ as *mut c_void,
        ];
        // SAFETY: as `launch_blend` — one pointer per parameter `blend_masked`
        // declares, in that order, each at a live local, and the context is current
        // because every caller reaches this inside `with_context`.
        unsafe {
            check(
                "cuLaunchKernel",
                cuLaunchKernel(
                    self.blend_masked,
                    width.div_ceil(BLOCK),
                    height.div_ceil(BLOCK),
                    1,
                    BLOCK,
                    BLOCK,
                    1,
                    0,
                    std::ptr::null_mut(),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
            )
        }
    }
}

/// Hands the driver one PTX module and looks up its two entry points. The
/// context must already be current.
unsafe fn load_module(
    ptx: &str,
    entry_names: [&str; 2],
) -> Result<(CUmodule, CUfunction, CUfunction), CudaDriverError> {
    // SAFETY: the caller has a context current, which is this function's own
    // documented contract. `image` and each `name` are live NUL-terminated
    // `CString`s, and a failed lookup unloads the module before returning, so no
    // path leaks it.
    unsafe {
        let image = std::ffi::CString::new(ptx)
            .map_err(|error| CudaDriverError::KernelRejected(error.to_string()))?;
        let mut module: CUmodule = std::ptr::null_mut();
        check(
            "cuModuleLoadData",
            cuModuleLoadData(&mut module, image.as_ptr().cast()),
        )
        .map_err(|error| CudaDriverError::KernelRejected(error.to_string()))?;
        let mut entries = [std::ptr::null_mut(); 2];
        for (entry, name) in entries.iter_mut().zip(entry_names) {
            let name = std::ffi::CString::new(name).expect("a literal without a nul");
            if let Err(error) = check(
                "cuModuleGetFunction",
                cuModuleGetFunction(entry, module, name.as_ptr()),
            ) {
                cuModuleUnload(module);
                return Err(CudaDriverError::KernelRejected(error.to_string()));
            }
        }
        Ok((module, entries[0], entries[1]))
    }
}

impl Drop for CudaDriver {
    fn drop(&mut self) {
        // Nothing useful to do with a failure here, and the process is
        // usually on its way out; the retain count is what matters.
        // SAFETY: both modules were loaded into `self.ctx`, so they are unloaded in
        // that same context — hence the push, and hence the unloads only running when
        // it succeeded. The release balances the retain in `retain_primary` and has
        // to happen either way, so it sits outside that branch.
        unsafe {
            if cuCtxPushCurrent_v2(self.ctx) == CUDA_SUCCESS {
                cuModuleUnload(self.module);
                cuModuleUnload(self.convert_module);
                let mut popped: CUcontext = std::ptr::null_mut();
                cuCtxPopCurrent_v2(&mut popped);
            }
            cuDevicePrimaryCtxRelease_v2(self.device);
        }
    }
}

/// Which rectangle [`CudaDriver::blit_nv12`] moves, in luma pixels. Every
/// field must be even — chroma is subsampled 2x2, so an odd offset or extent
/// has no corresponding chroma rectangle. Callers align before calling; see
/// [`crate::elements::CudaVideoCompositor`]'s own notes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Nv12Region {
    pub(crate) source_x: u32,
    pub(crate) source_y: u32,
    pub(crate) destination_x: u32,
    pub(crate) destination_y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

/// The device pointers and pitches of one NV12 CUDA surface — what an
/// `AVFrame` carries in `data[0..2]`/`linesize[0..2]`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Nv12Surface {
    pub(crate) luma: CUdeviceptr,
    pub(crate) luma_pitch: usize,
    pub(crate) chroma: CUdeviceptr,
    pub(crate) chroma_pitch: usize,
}

impl Nv12Surface {
    /// Reads the planes out of a CUDA-resident frame. The caller has already
    /// validated that this *is* one — format, frames context, and device —
    /// so the only thing left to reject is a frame with no pointers at all.
    pub(crate) fn from_frame(frame: &ffmpeg_next::frame::Video) -> Option<Self> {
        // SAFETY: `frame` is a live `frame::Video`, so `as_ptr` yields an
        // initialized `AVFrame`. `data` and `linesize` are plain arrays in it and
        // indices 0 and 1 exist for every pixel format; whether the values describe
        // a usable surface is what the check below decides.
        let (luma, chroma, luma_pitch, chroma_pitch) = unsafe {
            let ptr = frame.as_ptr();
            (
                (*ptr).data[0],
                (*ptr).data[1],
                (*ptr).linesize[0],
                (*ptr).linesize[1],
            )
        };
        if luma.is_null() || chroma.is_null() || luma_pitch <= 0 || chroma_pitch <= 0 {
            return None;
        }
        Some(Self {
            luma: luma as CUdeviceptr,
            luma_pitch: luma_pitch as usize,
            chroma: chroma as CUdeviceptr,
            chroma_pitch: chroma_pitch as usize,
        })
    }
}

/// The device pointer and pitch of one packed BGRA CUDA surface — what an
/// `AVFrame` carries in `data[0]`/`linesize[0]`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BgraSurface {
    pub(crate) pixels: CUdeviceptr,
    pub(crate) pitch: usize,
}

impl BgraSurface {
    /// Reads the surface out of a CUDA-resident BGRA frame. The caller has
    /// already established that this *is* one — format, frames context, and
    /// device — so the only thing left to reject is a frame with no pointer.
    pub(crate) fn from_frame(frame: &ffmpeg_next::frame::Video) -> Option<Self> {
        // SAFETY: as `Nv12Surface::from_frame` — a live `AVFrame`'s own `data` and
        // `linesize` arrays, checked below rather than trusted here.
        let (pixels, pitch) = unsafe {
            let ptr = frame.as_ptr();
            ((*ptr).data[0], (*ptr).linesize[0])
        };
        (!pixels.is_null() && pitch > 0).then_some(Self {
            pixels: pixels as CUdeviceptr,
            pitch: pitch as usize,
        })
    }
}

/// BT.709 limited-range Y'CbCr, matching what NVDEC produces and what NVENC
/// expects for HD content — a background filled with anything else would not
/// match the layers composited on top of it.
fn rgb_to_bt709_limited(color: Color) -> (u8, u8, u8) {
    bt709_limited(
        f32::from(color.red),
        f32::from(color.green),
        f32::from(color.blue),
    )
}

/// The same conversion over channels that need not be whole numbers, which is
/// what a chroma sample averaged over a 2x2 block is.
///
/// [`CONVERT_PTX`] performs exactly this, in this operation order, so a test
/// can hold the kernel to this expression byte for byte instead of to a
/// tolerance.
pub(crate) fn bt709_limited(r: f32, g: f32, b: f32) -> (u8, u8, u8) {
    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let u = (b - y) / 1.8556;
    let v = (r - y) / 1.5748;
    (
        (16.0 + y * 219.0 / 255.0).round().clamp(0.0, 255.0) as u8,
        (128.0 + u * 224.0 / 255.0).round().clamp(0.0, 255.0) as u8,
        (128.0 + v * 224.0 / 255.0).round().clamp(0.0, 255.0) as u8,
    )
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use ffmpeg_next::{self as ffmpeg};

    use super::*;
    use crate::{
        buffer::MediaBuffer,
        control::ControlMsg,
        element::{Element, ElementType, Sink, Source, element_pp_log},
        elements::{CudaDownload, CudaFrameFormat, CudaUpload},
        pool::UnboundObjectPool,
        pp_log::PpLog,
        test_support::try_cuda_device,
    };

    struct CapturingSink {
        pp_log: PpLog,
        received: Arc<Mutex<Vec<MediaBuffer>>>,
    }

    impl Element for CapturingSink {
        fn name(&self) -> Arc<str> {
            "capture".into()
        }
        fn element_type(&self) -> ElementType {
            ElementType::Other
        }
        fn pp_log(&self) -> &PpLog {
            &self.pp_log
        }
        fn pp_log_mut(&mut self) -> &mut PpLog {
            &mut self.pp_log
        }
    }

    impl Sink for CapturingSink {
        fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
            self.received.lock().unwrap().push(buf);
            Ok(())
        }
        fn control(&mut self, _msg: ControlMsg) -> crate::error::Result<()> {
            Ok(())
        }
    }

    fn capture(element: &mut dyn Source) -> Arc<Mutex<Vec<MediaBuffer>>> {
        let received = Arc::new(Mutex::new(Vec::new()));
        element.src_pads()[0].link(Box::new(CapturingSink {
            received: received.clone(),
            pp_log: element_pp_log(ElementType::Other, "capture", None),
        }));
        received
    }

    /// Uploads one NV12 frame whose luma is `luma` everywhere and hands back
    /// the CUDA-resident result, so a test has a real surface to operate on.
    fn cuda_surface(
        device: &crate::elements::CudaDevice,
        width: u32,
        height: u32,
        luma: u8,
    ) -> Option<MediaBuffer> {
        let Ok(mut upload) =
            CudaUpload::new("upload", device, CudaFrameFormat::Nv12, width, height)
        else {
            eprintln!("skipping: this machine has no usable CUDA frames context");
            return None;
        };
        let uploaded = capture(&mut upload);
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, width, height);
        let y_stride = frame.stride(0);
        frame.data_mut(0)[..y_stride * height as usize].fill(luma);
        let uv_stride = frame.stride(1);
        frame.data_mut(1)[..uv_stride * (height / 2) as usize].fill(128);
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut slot = pool.get();
        *slot = frame;
        upload
            .consume(MediaBuffer::Video(Arc::new(slot)))
            .expect("upload");
        Some(uploaded.lock().unwrap().remove(0))
    }

    fn download(
        device: &crate::elements::CudaDevice,
        frame: MediaBuffer,
        width: u32,
        height: u32,
    ) -> Arc<crate::pool::UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        let mut download =
            CudaDownload::new("download", device, CudaFrameFormat::Nv12, width, height);
        let received = capture(&mut download);
        download.consume(frame).expect("download");
        let buf = received.lock().unwrap().remove(0);
        match buf {
            MediaBuffer::Video(frame) => frame,
            other => panic!("expected a Video buffer, got {}", other.kind()),
        }
    }

    /// The driver layer's whole contract in one pass: a fill covers the
    /// surface, and a blit moves exactly the requested rectangle to exactly
    /// the requested place, leaving everything else as the fill left it.
    #[test]
    fn fill_then_blit_writes_the_expected_rectangles() {
        let Some((device, _cuda_lock)) = try_cuda_device() else {
            return;
        };
        let driver = match CudaDriver::retain_primary() {
            Ok(driver) => driver,
            Err(error) => {
                eprintln!("skipping: no usable CUDA driver context ({error})");
                return;
            }
        };
        let (width, height) = (64u32, 64u32);
        let Some(canvas) = cuda_surface(&device, width, height, 0) else {
            return;
        };
        let Some(layer) = cuda_surface(&device, 32, 32, 200) else {
            return;
        };

        let (MediaBuffer::Video(canvas_frame), MediaBuffer::Video(layer_frame)) = (&canvas, &layer)
        else {
            panic!("expected Video buffers");
        };
        let canvas_surface = Nv12Surface::from_frame(canvas_frame).expect("canvas planes");
        let layer_surface = Nv12Surface::from_frame(layer_frame).expect("layer planes");

        driver
            .fill_nv12(canvas_surface, width, height, Color::WHITE)
            .expect("fill");
        driver
            .blit_nv12(
                layer_surface,
                canvas_surface,
                Nv12Region {
                    source_x: 0,
                    source_y: 0,
                    destination_x: 16,
                    destination_y: 8,
                    width: 32,
                    height: 32,
                },
            )
            .expect("blit");

        let out = download(&device, canvas.clone(), width, height);
        let stride = out.stride(0);
        let at = |x: usize, y: usize| out.data(0)[y * stride + x];
        assert_eq!(at(0, 0), 235, "the fill did not cover the top-left corner");
        assert_eq!(at(63, 63), 235, "the fill did not cover the bottom-right");
        assert_eq!(at(16, 8), 200, "the blit missed its top-left corner");
        assert_eq!(at(47, 39), 200, "the blit missed its bottom-right corner");
        assert_eq!(at(15, 8), 235, "the blit wrote left of its rectangle");
        assert_eq!(at(48, 8), 235, "the blit wrote right of its rectangle");
        assert_eq!(at(16, 7), 235, "the blit wrote above its rectangle");
        assert_eq!(at(16, 40), 235, "the blit wrote below its rectangle");

        let uv_stride = out.stride(1);
        assert_eq!(
            out.data(1)[uv_stride * 20 + 4],
            128,
            "chroma did not survive the fill"
        );
    }

    /// A CUDA surface whose luma is `f(x, y)`, so an indexing or pitch
    /// mistake in the kernel shows up as a wrong *position*, not just a
    /// wrong value.
    /// Uploads one BGRA frame built by `pixel`, so a conversion test starts
    /// from a CUDA-resident source with content it can predict.
    fn cuda_bgra_surface(
        device: &crate::elements::CudaDevice,
        width: u32,
        height: u32,
        pixel: impl Fn(u32, u32) -> [u8; 4],
    ) -> Option<MediaBuffer> {
        let Ok(mut upload) =
            CudaUpload::new("upload", device, CudaFrameFormat::Bgra, width, height)
        else {
            eprintln!("skipping: this machine has no usable CUDA frames context");
            return None;
        };
        let uploaded = capture(&mut upload);
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, width, height);
        let stride = frame.stride(0);
        for y in 0..height {
            let row = &mut frame.data_mut(0)[y as usize * stride..];
            for x in 0..width {
                row[x as usize * 4..x as usize * 4 + 4].copy_from_slice(&pixel(x, y));
            }
        }
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut slot = pool.get();
        *slot = frame;
        upload
            .consume(MediaBuffer::Video(Arc::new(slot)))
            .expect("upload");
        Some(uploaded.lock().unwrap().remove(0))
    }

    /// The conversion nothing else on the CUDA path can do, checked against
    /// the definition the compositor fills backgrounds with rather than
    /// against a tolerance: every luma byte is that pixel's own conversion,
    /// and every chroma pair is the conversion of its 2x2 block's average.
    #[test]
    fn bgra_converts_to_nv12_exactly_as_the_shared_definition_says() {
        let Some((device, _cuda_lock)) = crate::test_support::try_cuda_device() else {
            return;
        };
        let Ok(driver) = CudaDriver::retain_primary() else {
            eprintln!("skipping: no usable CUDA driver on this machine");
            return;
        };
        const WIDTH: u32 = 64;
        const HEIGHT: u32 = 32;
        // A pattern with no symmetry between channels, so a kernel that mixed
        // two of them up could not still match.
        let pixel = |x: u32, y: u32| {
            [
                (x * 4 % 256) as u8,
                (y * 8 % 256) as u8,
                ((x + y) * 3 % 256) as u8,
                255,
            ]
        };
        let Some(source) = cuda_bgra_surface(&device, WIDTH, HEIGHT, pixel) else {
            return;
        };
        let Some(destination) = cuda_surface(&device, WIDTH, HEIGHT, 0) else {
            return;
        };

        let (MediaBuffer::Video(source_frame), MediaBuffer::Video(destination_frame)) =
            (&source, &destination)
        else {
            panic!("both uploads produce Video buffers");
        };
        driver
            .bgra_to_nv12(
                BgraSurface::from_frame(source_frame).expect("a BGRA surface"),
                Nv12Surface::from_frame(destination_frame).expect("an NV12 surface"),
                WIDTH,
                HEIGHT,
            )
            .expect("convert");
        driver.synchronize().expect("synchronize");

        let converted = download(&device, destination, WIDTH, HEIGHT);
        let luma_stride = converted.stride(0);
        let chroma_stride = converted.stride(1);
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let [b, g, r, _] = pixel(x, y);
                let (expected, _, _) = bt709_limited(f32::from(r), f32::from(g), f32::from(b));
                assert_eq!(
                    converted.data(0)[y as usize * luma_stride + x as usize],
                    expected,
                    "luma at {x},{y}"
                );
            }
        }
        for cy in 0..HEIGHT / 2 {
            for cx in 0..WIDTH / 2 {
                let mut sums = [0u32; 3];
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let [b, g, r, _] = pixel(cx * 2 + dx, cy * 2 + dy);
                    sums[0] += u32::from(b);
                    sums[1] += u32::from(g);
                    sums[2] += u32::from(r);
                }
                let (_, expected_u, expected_v) = bt709_limited(
                    sums[2] as f32 / 4.0,
                    sums[1] as f32 / 4.0,
                    sums[0] as f32 / 4.0,
                );
                let at = cy as usize * chroma_stride + cx as usize * 2;
                assert_eq!(converted.data(1)[at], expected_u, "u at {cx},{cy}");
                assert_eq!(converted.data(1)[at + 1], expected_v, "v at {cx},{cy}");
            }
        }
    }

    fn cuda_surface_with(
        device: &crate::elements::CudaDevice,
        width: u32,
        height: u32,
        luma: impl Fn(u32, u32) -> u8,
        chroma: u8,
    ) -> Option<MediaBuffer> {
        let Ok(mut upload) =
            CudaUpload::new("upload", device, CudaFrameFormat::Nv12, width, height)
        else {
            eprintln!("skipping: this machine has no usable CUDA frames context");
            return None;
        };
        let uploaded = capture(&mut upload);
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, width, height);
        let y_stride = frame.stride(0);
        let plane = frame.data_mut(0);
        for y in 0..height {
            for x in 0..width {
                plane[y as usize * y_stride + x as usize] = luma(x, y);
            }
        }
        let uv_stride = frame.stride(1);
        frame.data_mut(1)[..uv_stride * (height / 2) as usize].fill(chroma);
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut slot = pool.get();
        *slot = frame;
        upload
            .consume(MediaBuffer::Video(Arc::new(slot)))
            .expect("upload");
        Some(uploaded.lock().unwrap().remove(0))
    }

    /// The kernel's whole contract: every blended byte matches the same
    /// expression evaluated on the CPU. Hand-written PTX is only defensible
    /// because this can be checked exactly rather than eyeballed.
    #[test]
    fn the_blend_kernel_matches_a_cpu_reference_byte_for_byte() {
        let Some((device, _cuda_lock)) = try_cuda_device() else {
            return;
        };
        let driver = match CudaDriver::retain_primary() {
            Ok(driver) => driver,
            Err(error) => {
                eprintln!("skipping: no usable CUDA driver context ({error})");
                return;
            }
        };
        let (width, height) = (64u32, 64u32);
        // Ramps along different axes, so a swapped coordinate cannot pass.
        let Some(destination) = cuda_surface_with(&device, width, height, |_, y| (y * 3) as u8, 90)
        else {
            return;
        };
        let Some(source) = cuda_surface_with(&device, width, height, |x, _| (x * 4) as u8, 200)
        else {
            return;
        };
        let (MediaBuffer::Video(dst_frame), MediaBuffer::Video(src_frame)) =
            (&destination, &source)
        else {
            panic!("expected Video buffers");
        };
        let dst_surface = Nv12Surface::from_frame(dst_frame).expect("destination planes");
        let src_surface = Nv12Surface::from_frame(src_frame).expect("source planes");

        let alpha = 77u8;
        driver
            .blend_nv12(
                src_surface,
                dst_surface,
                Nv12Region {
                    source_x: 0,
                    source_y: 0,
                    destination_x: 0,
                    destination_y: 0,
                    width,
                    height,
                },
                alpha,
            )
            .expect("blend");
        driver.synchronize().expect("synchronize");

        let out = download(&device, destination.clone(), width, height);
        let blend = |dst: u32, src: u32| {
            ((src * u32::from(alpha) + dst * (255 - u32::from(alpha)) + 127) / 255) as u8
        };
        let stride = out.stride(0);
        for y in 0..height {
            for x in 0..width {
                let expected = blend(u32::from((y * 3) as u8), u32::from((x * 4) as u8));
                let actual = out.data(0)[y as usize * stride + x as usize];
                assert_eq!(
                    actual, expected,
                    "luma mismatch at ({x}, {y}): kernel {actual} != cpu {expected}"
                );
            }
        }
        let uv_stride = out.stride(1);
        let expected_chroma = blend(90, 200);
        for y in 0..height / 2 {
            for x in 0..width {
                let actual = out.data(1)[y as usize * uv_stride + x as usize];
                assert_eq!(
                    actual, expected_chroma,
                    "chroma mismatch at ({x}, {y}): kernel {actual} != cpu {expected_chroma}"
                );
            }
        }
    }

    /// The endpoints have to be exact, not merely close: a fully opaque
    /// blend must equal the source, and a fully transparent one must leave
    /// the destination untouched.
    #[test]
    fn alpha_endpoints_replace_and_preserve_exactly() {
        let Some((device, _cuda_lock)) = try_cuda_device() else {
            return;
        };
        let Ok(driver) = CudaDriver::retain_primary() else {
            eprintln!("skipping: no usable CUDA driver context");
            return;
        };
        let (width, height) = (32u32, 32u32);
        for (alpha, expected) in [(255u8, 200u8), (0, 60)] {
            let Some(destination) = cuda_surface_with(&device, width, height, |_, _| 60, 128)
            else {
                return;
            };
            let Some(source) = cuda_surface_with(&device, width, height, |_, _| 200, 128) else {
                return;
            };
            let (MediaBuffer::Video(dst_frame), MediaBuffer::Video(src_frame)) =
                (&destination, &source)
            else {
                panic!("expected Video buffers");
            };
            driver
                .blend_nv12(
                    Nv12Surface::from_frame(src_frame).expect("source planes"),
                    Nv12Surface::from_frame(dst_frame).expect("destination planes"),
                    Nv12Region {
                        source_x: 0,
                        source_y: 0,
                        destination_x: 0,
                        destination_y: 0,
                        width,
                        height,
                    },
                    alpha,
                )
                .expect("blend");
            driver.synchronize().expect("synchronize");

            let out = download(&device, destination.clone(), width, height);
            assert_eq!(
                out.data(0)[out.stride(0) * 5 + 5],
                expected,
                "alpha {alpha} must produce {expected}"
            );
        }
    }

    /// The two colors every background test in this crate uses, checked
    /// against the BT.709 limited-range values they are defined to produce.
    #[test]
    fn black_and_white_map_to_limited_range_endpoints() {
        assert_eq!(rgb_to_bt709_limited(Color::BLACK), (16, 128, 128));
        let (y, u, v) = rgb_to_bt709_limited(Color::WHITE);
        assert_eq!(y, 235);
        assert!(
            u.abs_diff(128) <= 1 && v.abs_diff(128) <= 1,
            "white must be chroma-neutral, got ({u}, {v})"
        );
    }
}
