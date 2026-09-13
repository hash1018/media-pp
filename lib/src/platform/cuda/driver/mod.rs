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

mod ptx;

use ptx::{BLEND_PTX, CONVERT_PTX};

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
    /// A layer blended with a per-pixel alpha *and* per-pixel colour, which
    /// is what an overlay carrying its own transparency needs. Loaded but not
    /// yet called: the surfaces it reads from have no owner until the
    /// compositor grows a path for a layer that brings its own alpha.
    #[allow(dead_code)]
    blend_plane_masked: CUfunction,
    /// [`CONVERT_PTX`] and its entry points, loaded alongside the blend
    /// module: one JIT at construction rather than one on the first frame
    /// that needs a conversion.
    convert_module: CUmodule,
    bgra_to_luma: CUfunction,
    bgra_to_chroma: CUfunction,
    /// Lifts a BGRA surface's alpha into a plane of its own, so a blend can
    /// read it per pixel. Loaded but not yet called — see
    /// `blend_plane_masked`.
    #[allow(dead_code)]
    extract_alpha: CUfunction,
    /// The same at half resolution, averaging each 2x2 block — what the
    /// chroma pass reads, for the reason `upload_mask` averages too. Loaded
    /// but not yet called — see `blend_plane_masked`.
    #[allow(dead_code)]
    extract_alpha_half: CUfunction,
    /// Writes a BGRA surface's alpha from each pixel's distance to a key
    /// colour — what [`crate::elements::CudaChromaKey`] is.
    key_bgra: CUfunction,
    /// Turns an NV12 surface into a BGRA one, which is what a filter
    /// wanting BGRA needs in front of a camera.
    nv12_to_bgra: CUfunction,
    /// Runs a colour matrix, an exponent, an opacity and a luma mask over a
    /// BGRA surface — what [`crate::elements::CudaVideoEffect`] is.
    effect_bgra: CUfunction,
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
            let loaded = load_module(
                BLEND_PTX,
                ["blend_plane", "blend_masked", "blend_plane_masked"],
            )
            .and_then(|blend| {
                match load_module(
                    CONVERT_PTX,
                    [
                        "bgra_to_luma",
                        "bgra_to_chroma",
                        "extract_alpha",
                        "extract_alpha_half",
                        "key_bgra",
                        "nv12_to_bgra",
                        "effect_bgra",
                    ],
                ) {
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
            let (
                (module, [blend, blend_masked, blend_plane_masked]),
                (
                    convert_module,
                    [
                        bgra_to_luma,
                        bgra_to_chroma,
                        extract_alpha,
                        extract_alpha_half,
                        key_bgra,
                        nv12_to_bgra,
                        effect_bgra,
                    ],
                ),
            ) = match loaded {
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
                blend_plane_masked,
                convert_module,
                bgra_to_luma,
                bgra_to_chroma,
                extract_alpha,
                extract_alpha_half,
                key_bgra,
                nv12_to_bgra,
                effect_bgra,
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

    /// Copies a rectangle of one BGRA surface into the start of another,
    /// device to device.
    ///
    /// One plane and four bytes a pixel, so there is no chroma grid to keep
    /// to and no alignment to respect: a 2D copy addresses bytes.
    pub(crate) fn blit_bgra(
        &self,
        source: BgraSurface,
        destination: BgraSurface,
        source_x: u32,
        source_y: u32,
        width: u32,
        height: u32,
    ) -> Result<(), CudaDriverError> {
        if width == 0 || height == 0 {
            return Ok(());
        }
        // SAFETY: `with_context` has the context current, and the descriptor is
        // a live local with both memory types set to device — the host and
        // array fields `Default` left null are ignored. Keeping the rectangle
        // inside both surfaces is the caller's contract.
        self.with_context(|| unsafe {
            let copy = CudaMemcpy2D {
                src_memory_type: CU_MEMORYTYPE_DEVICE,
                src_device: source.pixels,
                src_pitch: source.pitch,
                src_x_in_bytes: source_x as usize * 4,
                src_y: source_y as usize,
                dst_memory_type: CU_MEMORYTYPE_DEVICE,
                dst_device: destination.pixels,
                dst_pitch: destination.pitch,
                width_in_bytes: width as usize * 4,
                height: height as usize,
                ..CudaMemcpy2D::default()
            };
            check("cuMemcpy2D", cuMemcpy2D_v2(&copy))
        })
    }

    /// Writes `destination`'s alpha from each pixel's distance to
    /// `key_color`, times the alpha `source` already had, copying BGR through
    /// unchanged — the GPU-resident half of what
    /// [`crate::elements::SwChromaKey`] does per pixel on the CPU.
    ///
    /// `band_low`/`inv_band_width` come from the chroma-key module's own
    /// `feather_band`, so this and the D3D11 shader evaluate one definition
    /// of the ramp rather than two.
    ///
    /// Both surfaces must be BGRA at `width` x `height`; the caller has
    /// already established that, since a frame this size in another format
    /// is not something a pitch can express.
    ///
    /// Launches asynchronously. [`CudaDriver::synchronize`] is what makes the
    /// result visible to anything outside this context's stream ordering.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn key_bgra(
        &self,
        source: BgraSurface,
        destination: BgraSurface,
        width: u32,
        height: u32,
        key_color: Color,
        band_low: f32,
        inv_band_width: f32,
    ) -> Result<(), CudaDriverError> {
        if width == 0 || height == 0 {
            return Ok(());
        }
        const BLOCK: u32 = 16;
        let (grid_x, grid_y) = (width.div_ceil(BLOCK), height.div_ceil(BLOCK));

        let mut dst = destination.pixels;
        let mut dst_pitch = destination.pitch as u32;
        let mut src = source.pixels;
        let mut src_pitch = source.pitch as u32;
        let mut width = width;
        let mut height = height;
        // Normalized here rather than by the caller: the kernel compares a
        // pixel scaled to 0..1 against these, and one place to divide is one
        // place to be wrong.
        let mut key_b = f32::from(key_color.blue) / 255.0;
        let mut key_g = f32::from(key_color.green) / 255.0;
        let mut key_r = f32::from(key_color.red) / 255.0;
        let mut band_low = band_low;
        let mut inv_band_width = inv_band_width;

        self.with_context(|| {
            let mut params: [*mut c_void; 11] = [
                (&mut dst) as *mut _ as *mut c_void,
                (&mut dst_pitch) as *mut _ as *mut c_void,
                (&mut src) as *mut _ as *mut c_void,
                (&mut src_pitch) as *mut _ as *mut c_void,
                (&mut width) as *mut _ as *mut c_void,
                (&mut height) as *mut _ as *mut c_void,
                (&mut key_b) as *mut _ as *mut c_void,
                (&mut key_g) as *mut _ as *mut c_void,
                (&mut key_r) as *mut _ as *mut c_void,
                (&mut band_low) as *mut _ as *mut c_void,
                (&mut inv_band_width) as *mut _ as *mut c_void,
            ];
            // SAFETY: one pointer per parameter `key_bgra` declares, in that
            // order, at a live local; the context is current inside
            // `with_context`.
            unsafe {
                check(
                    "cuLaunchKernel",
                    cuLaunchKernel(
                        self.key_bgra,
                        grid_x,
                        grid_y,
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
        })
    }

    /// Runs one video effect over a BGRA surface: every pixel through a
    /// colour matrix, an exponent, an opacity and a luma mask, into
    /// `destination`.
    ///
    /// The numbers arrive resolved — `rows` one per output channel red,
    /// green, blue as `[from_red, from_green, from_blue, offset]`, `luma` as
    /// `[low, low_inv, high, high_inv]` — and the kernel evaluates exactly the
    /// definition the video-effect module's `EffectParams` documents, which
    /// is also what the D3D11 shader and the software element evaluate.
    ///
    /// Both surfaces must be BGRA at `width` x `height`. Launches
    /// asynchronously; [`CudaDriver::synchronize`] is what makes the result
    /// visible outside this context's stream ordering.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn effect_bgra(
        &self,
        source: BgraSurface,
        destination: BgraSurface,
        width: u32,
        height: u32,
        rows: &[[f32; 4]; 3],
        exponent: f32,
        opacity: f32,
        luma: [f32; 4],
    ) -> Result<(), CudaDriverError> {
        if width == 0 || height == 0 {
            return Ok(());
        }
        const BLOCK: u32 = 16;
        let (grid_x, grid_y) = (width.div_ceil(BLOCK), height.div_ceil(BLOCK));

        let mut dst = destination.pixels;
        let mut dst_pitch = destination.pitch as u32;
        let mut src = source.pixels;
        let mut src_pitch = source.pitch as u32;
        let mut width = width;
        let mut height = height;
        let mut matrix: [f32; 12] = [
            rows[0][0], rows[0][1], rows[0][2], rows[0][3], rows[1][0], rows[1][1], rows[1][2],
            rows[1][3], rows[2][0], rows[2][1], rows[2][2], rows[2][3],
        ];
        let mut exponent = exponent;
        let mut opacity = opacity;
        let mut luma = luma;

        self.with_context(|| {
            let mut params: [*mut c_void; 24] = [std::ptr::null_mut(); 24];
            params[0] = (&mut dst) as *mut _ as *mut c_void;
            params[1] = (&mut dst_pitch) as *mut _ as *mut c_void;
            params[2] = (&mut src) as *mut _ as *mut c_void;
            params[3] = (&mut src_pitch) as *mut _ as *mut c_void;
            params[4] = (&mut width) as *mut _ as *mut c_void;
            params[5] = (&mut height) as *mut _ as *mut c_void;
            for (slot, value) in params[6..18].iter_mut().zip(matrix.iter_mut()) {
                *slot = value as *mut f32 as *mut c_void;
            }
            params[18] = (&mut exponent) as *mut _ as *mut c_void;
            params[19] = (&mut opacity) as *mut _ as *mut c_void;
            for (slot, value) in params[20..24].iter_mut().zip(luma.iter_mut()) {
                *slot = value as *mut f32 as *mut c_void;
            }
            // SAFETY: one pointer per parameter `effect_bgra` declares, in
            // that order, each at a live local that outlives the launch call;
            // the context is current inside `with_context`.
            unsafe {
                check(
                    "cuLaunchKernel",
                    cuLaunchKernel(
                        self.effect_bgra,
                        grid_x,
                        grid_y,
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
        })
    }

    /// Converts an NV12 surface into a BGRA one, on the GPU.
    ///
    /// The direction `scale_cuda` refuses and
    /// [`CudaScaler`](crate::elements::CudaScaler) documents as impossible:
    /// its module carries no kernel for a YUV/RGB pair either way, so this
    /// one does.
    ///
    /// The colour maths is the exact inverse of [`bt709_limited`], written
    /// in the order that undoes it, so a round trip through both is off by
    /// rounding and by chroma subsampling and by nothing else.
    ///
    /// Launches asynchronously. [`CudaDriver::synchronize`] is what makes
    /// the result visible to anything outside this context's stream
    /// ordering.
    pub(crate) fn nv12_to_bgra(
        &self,
        source: Nv12Surface,
        destination: BgraSurface,
        width: u32,
        height: u32,
    ) -> Result<(), CudaDriverError> {
        if width == 0 || height == 0 {
            return Ok(());
        }
        const BLOCK: u32 = 16;
        let (grid_x, grid_y) = (width.div_ceil(BLOCK), height.div_ceil(BLOCK));

        let mut dst = destination.pixels;
        let mut dst_pitch = destination.pitch as u32;
        let mut luma = source.luma;
        let mut luma_pitch = source.luma_pitch as u32;
        let mut chroma = source.chroma;
        let mut chroma_pitch = source.chroma_pitch as u32;
        let mut width = width;
        let mut height = height;

        self.with_context(|| {
            let mut params: [*mut c_void; 8] = [
                (&mut dst) as *mut _ as *mut c_void,
                (&mut dst_pitch) as *mut _ as *mut c_void,
                (&mut luma) as *mut _ as *mut c_void,
                (&mut luma_pitch) as *mut _ as *mut c_void,
                (&mut chroma) as *mut _ as *mut c_void,
                (&mut chroma_pitch) as *mut _ as *mut c_void,
                (&mut width) as *mut _ as *mut c_void,
                (&mut height) as *mut _ as *mut c_void,
            ];
            // SAFETY: one pointer per parameter `nv12_to_bgra` declares, in
            // that order, at a live local; the context is current inside
            // `with_context`.
            unsafe {
                check(
                    "cuLaunchKernel",
                    cuLaunchKernel(
                        self.nv12_to_bgra,
                        grid_x,
                        grid_y,
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

/// The scratch a BGRA layer needs on its way into an NV12 canvas.
///
/// A layer that brings its own transparency cannot be blitted: its colour has
/// to be converted into the canvas's own space, and its alpha has to be read
/// per pixel while that happens. Both need somewhere to land, and that is
/// this — an NV12 pair for the converted colour and an alpha plane at each
/// resolution the two passes read.
///
/// About 2.75 bytes a pixel, so a canvas-sized overlay is a few megabytes.
/// Held by whoever draws the layer and reused for as long as its size is
/// unchanged, because the alternative is allocating that per frame.
pub(crate) struct CudaOverlayScratch {
    ctx: CUcontext,
    luma: CUdeviceptr,
    chroma: CUdeviceptr,
    alpha_full: CUdeviceptr,
    alpha_half: CUdeviceptr,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

// SAFETY: plain device allocations with no thread affinity, and `Drop` pushes
// the context it captured before freeing them — the same contract `CudaMask`
// documents just below.
unsafe impl Send for CudaOverlayScratch {}

impl Drop for CudaOverlayScratch {
    fn drop(&mut self) {
        // SAFETY: every pointer came from `cuMemAlloc` in `overlay_scratch` and
        // is freed once, in the context it was allocated in — hence the push,
        // and hence the frees only running when it succeeded.
        unsafe {
            if cuCtxPushCurrent_v2(self.ctx) == CUDA_SUCCESS {
                cuMemFree_v2(self.luma);
                cuMemFree_v2(self.chroma);
                cuMemFree_v2(self.alpha_full);
                cuMemFree_v2(self.alpha_half);
                let mut popped: CUcontext = std::ptr::null_mut();
                cuCtxPopCurrent_v2(&mut popped);
            }
        }
    }
}

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
    /// Allocates the scratch a BGRA layer of this size needs — see
    /// [`CudaOverlayScratch`].
    ///
    /// The dimensions are rounded down to even, as every NV12 size in this
    /// module is: chroma is 2x2 subsampled and a half-sample has nowhere to
    /// go.
    #[allow(dead_code)]
    pub(crate) fn overlay_scratch(
        &self,
        width: u32,
        height: u32,
    ) -> Result<CudaOverlayScratch, CudaDriverError> {
        let width = width & !1;
        let height = height & !1;
        if width == 0 || height == 0 {
            return Err(CudaDriverError::EmptyMask);
        }
        let pixels = (width * height) as usize;
        // SAFETY: `with_context` has the context current, every out-param is a
        // live local, and a failure frees whatever was already taken before
        // returning, so no path leaks an allocation.
        self.with_context(|| unsafe {
            let mut taken = Vec::with_capacity(4);
            let mut alloc = |bytes: usize| -> Result<CUdeviceptr, CudaDriverError> {
                let mut pointer = 0;
                match check("cuMemAlloc", cuMemAlloc_v2(&mut pointer, bytes)) {
                    Ok(()) => {
                        taken.push(pointer);
                        Ok(pointer)
                    }
                    Err(error) => {
                        for pointer in taken.drain(..) {
                            cuMemFree_v2(pointer);
                        }
                        Err(error)
                    }
                }
            };
            let luma = alloc(pixels)?;
            let chroma = alloc(pixels / 2)?;
            let alpha_full = alloc(pixels)?;
            let alpha_half = alloc(pixels / 4)?;
            Ok(CudaOverlayScratch {
                ctx: self.ctx,
                luma,
                chroma,
                alpha_full,
                alpha_half,
                width,
                height,
            })
        })
    }

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
    /// Draws a BGRA layer into an NV12 canvas, honouring the layer's own
    /// per-pixel alpha.
    ///
    /// This is what a `blit` cannot do. A blit moves bytes, so a layer that is
    /// transparent in places arrives opaque in all of them; and the scalar
    /// blend fades a whole layer evenly rather than pixel by pixel. An overlay
    /// — anything drawn over a capture rather than beside it — needs the third
    /// thing, and this is it.
    ///
    /// # Three passes, because the colour has to change space first
    ///
    /// The canvas is NV12 and the layer is BGRA, so its colour is converted
    /// into `scratch`'s own NV12 pair with the same kernels
    /// [`CudaDriver::bgra_to_nv12`] uses. Its alpha is lifted out separately,
    /// at both resolutions, because the luma pass reads one sample per pixel
    /// and the chroma pass one per 2x2. Then each plane is mixed under that
    /// alpha.
    ///
    /// `scratch` has to be at least `width` x `height`; anything larger is
    /// fine and is what lets one allocation serve a layer whose size is
    /// unchanged between frames.
    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) fn blend_bgra_nv12(
        &self,
        source: BgraSurface,
        scratch: &CudaOverlayScratch,
        destination: Nv12Surface,
        region: Nv12Region,
        opacity: u8,
    ) -> Result<(), CudaDriverError> {
        let Nv12Region {
            source_x,
            source_y,
            destination_x,
            destination_y,
            width,
            height,
        } = region;
        if width == 0 || height == 0 {
            return Ok(());
        }
        debug_assert!(
            width <= scratch.width && height <= scratch.height,
            "scratch is smaller than the region it is asked to hold"
        );
        // The layer's own top-left, since everything below works in the
        // scratch's coordinates rather than the source's.
        let pixels =
            source.pixels + u64::from(source_y) * source.pitch as u64 + u64::from(source_x) * 4;
        let scratch_pitch = scratch.width;
        let half_pitch = scratch.width / 2;

        self.with_context(|| {
            self.launch_plane(
                self.bgra_to_luma,
                scratch.luma,
                scratch_pitch,
                pixels,
                source.pitch as u32,
                width,
                height,
            )?;
            self.launch_plane(
                self.bgra_to_chroma,
                scratch.chroma,
                scratch_pitch,
                pixels,
                source.pitch as u32,
                width / 2,
                height / 2,
            )?;
            self.launch_plane(
                self.extract_alpha,
                scratch.alpha_full,
                scratch_pitch,
                pixels,
                source.pitch as u32,
                width,
                height,
            )?;
            self.launch_plane(
                self.extract_alpha_half,
                scratch.alpha_half,
                half_pitch,
                pixels,
                source.pitch as u32,
                width / 2,
                height / 2,
            )?;
            self.launch_plane_masked(
                destination.luma
                    + u64::from(destination_y) * destination.luma_pitch as u64
                    + u64::from(destination_x),
                destination.luma_pitch as u32,
                scratch.luma,
                scratch_pitch,
                scratch.alpha_full,
                scratch_pitch,
                width,
                height,
                opacity,
                0,
            )?;
            // Chroma is interleaved `(U, V)`, so its width in bytes is the
            // layer's own and each byte's alpha comes from the half-resolution
            // plane one sample to the right of every pair — the `1` shift.
            self.launch_plane_masked(
                destination.chroma
                    + u64::from(destination_y / 2) * destination.chroma_pitch as u64
                    + u64::from(destination_x),
                destination.chroma_pitch as u32,
                scratch.chroma,
                scratch_pitch,
                scratch.alpha_half,
                half_pitch,
                width,
                height / 2,
                opacity,
                1,
            )
        })
    }

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

    /// One 6-parameter convert-shaped launch: `(dst, dst_pitch, src,
    /// src_pitch, width, height)`, which is the shape both alpha extractions
    /// and both colour conversions share.
    #[allow(clippy::too_many_arguments)]
    fn launch_plane(
        &self,
        entry: CUfunction,
        mut dst: CUdeviceptr,
        mut dst_pitch: u32,
        mut src: CUdeviceptr,
        mut src_pitch: u32,
        mut width: u32,
        mut height: u32,
    ) -> Result<(), CudaDriverError> {
        const BLOCK: u32 = 16;
        let (grid_x, grid_y) = (width.div_ceil(BLOCK), height.div_ceil(BLOCK));
        let mut params: [*mut c_void; 6] = [
            (&mut dst) as *mut _ as *mut c_void,
            (&mut dst_pitch) as *mut _ as *mut c_void,
            (&mut src) as *mut _ as *mut c_void,
            (&mut src_pitch) as *mut _ as *mut c_void,
            (&mut width) as *mut _ as *mut c_void,
            (&mut height) as *mut _ as *mut c_void,
        ];
        // SAFETY: one pointer per parameter each of these kernels declares, in
        // that order, at a live local; the context is current because every
        // caller reaches this inside `with_context`.
        unsafe {
            check(
                "cuLaunchKernel",
                cuLaunchKernel(
                    entry,
                    grid_x,
                    grid_y,
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

    /// `blend_plane_masked`: the mix `launch_masked` does, with the colour
    /// read per pixel from `src` instead of taken from a constant.
    #[allow(clippy::too_many_arguments)]
    fn launch_plane_masked(
        &self,
        mut dst: CUdeviceptr,
        mut dst_pitch: u32,
        mut src: CUdeviceptr,
        mut src_pitch: u32,
        mut mask: CUdeviceptr,
        mut mask_pitch: u32,
        mut width: u32,
        mut height: u32,
        opacity: u8,
        mut shift: u32,
    ) -> Result<(), CudaDriverError> {
        const BLOCK: u32 = 16;
        let (grid_x, grid_y) = (width.div_ceil(BLOCK), height.div_ceil(BLOCK));
        let mut opacity = u32::from(opacity);
        let mut params: [*mut c_void; 10] = [
            (&mut dst) as *mut _ as *mut c_void,
            (&mut dst_pitch) as *mut _ as *mut c_void,
            (&mut src) as *mut _ as *mut c_void,
            (&mut src_pitch) as *mut _ as *mut c_void,
            (&mut mask) as *mut _ as *mut c_void,
            (&mut mask_pitch) as *mut _ as *mut c_void,
            (&mut width) as *mut _ as *mut c_void,
            (&mut height) as *mut _ as *mut c_void,
            (&mut opacity) as *mut _ as *mut c_void,
            (&mut shift) as *mut _ as *mut c_void,
        ];
        // SAFETY: as `launch_masked`, with the one extra pair its own kernel
        // declares.
        unsafe {
            check(
                "cuLaunchKernel",
                cuLaunchKernel(
                    self.blend_plane_masked,
                    grid_x,
                    grid_y,
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
unsafe fn load_module<const N: usize>(
    ptx: &str,
    entry_names: [&str; N],
) -> Result<(CUmodule, [CUfunction; N]), CudaDriverError> {
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
        let mut entries = [std::ptr::null_mut(); N];
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
        Ok((module, entries))
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
mod tests;
