//! What [`crate::elements::CudaOrtDetector`] asks of the driver: its own
//! kernels, loaded only by it — see [`super::ptx::FIT_PTX`] — and the
//! device buffer it fits each picture into.

use std::ffi::c_void;

use super::ptx::FIT_PTX;
use super::{
    BgraSurface, CUcontext, CUdeviceptr, CUfunction, CUmodule, CudaDriver, CudaDriverError,
    Nv12Surface, YuvToBgra, arg, check, cuCtxPopCurrent_v2, cuCtxPushCurrent_v2, cuMemAlloc_v2,
    cuMemFree_v2, cuModuleUnload, launch, load_module,
};

/// The fitting kernels, JIT-compiled into the driver's context. Dropped
/// before the driver that made them, which is what keeps their context
/// alive while they unload.
pub(crate) struct FitKernels {
    ctx: CUcontext,
    module: CUmodule,
    nv12: CUfunction,
    bgra: CUfunction,
}

// SAFETY: a module and two function handles in a context that is pushed
// around every use; nothing here is thread-affine — see `CudaDriver`.
unsafe impl Send for FitKernels {}

impl Drop for FitKernels {
    fn drop(&mut self) {
        // SAFETY: the module was loaded into `self.ctx`, still retained by the
        // driver this outlives by contract, and is unloaded once.
        unsafe {
            if cuCtxPushCurrent_v2(self.ctx) == 0 {
                cuModuleUnload(self.module);
                let mut popped: CUcontext = std::ptr::null_mut();
                cuCtxPopCurrent_v2(&mut popped);
            }
        }
    }
}

/// A device buffer of `f32`s a detector's input is fitted into: three
/// planes, R, G and B, of the model's width by height.
pub(crate) struct CudaTensor {
    ctx: CUcontext,
    pointer: CUdeviceptr,
    floats: usize,
}

impl CudaTensor {
    /// Where it is on the device.
    pub(crate) fn pointer(&self) -> CUdeviceptr {
        self.pointer
    }

    /// How many `f32`s it holds.
    pub(crate) fn floats(&self) -> usize {
        self.floats
    }
}

// SAFETY: a plain device allocation, freed in the context it was made in —
// the contract `CudaBgraScratch` documents.
unsafe impl Send for CudaTensor {}

impl Drop for CudaTensor {
    fn drop(&mut self) {
        // SAFETY: the context this was allocated in is still retained by the
        // driver that made it, and the pointer is freed exactly once.
        unsafe {
            if cuCtxPushCurrent_v2(self.ctx) == 0 {
                cuMemFree_v2(self.pointer);
                let mut popped: CUcontext = std::ptr::null_mut();
                cuCtxPopCurrent_v2(&mut popped);
            }
        }
    }
}

/// Where a picture sits in the model's input: the input's size, the scaled
/// picture's corner and size — a detector's letterbox, in the numbers the
/// kernels take.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Fit {
    pub(crate) model: (u32, u32),
    pub(crate) offset: (u32, u32),
    pub(crate) scaled: (u32, u32),
}

impl CudaDriver {
    /// Loads the fitting kernels into this driver's context.
    pub(crate) fn fit_kernels(&self) -> Result<FitKernels, CudaDriverError> {
        self.with_context(|| {
            // SAFETY: the context is current inside `with_context`, which is
            // `load_module`'s whole contract.
            let (module, [nv12, bgra]) = unsafe { load_module(FIT_PTX, ["fit_nv12", "fit_bgra"])? };
            Ok(FitKernels {
                ctx: self.ctx,
                module,
                nv12,
                bgra,
            })
        })
    }

    /// A device buffer for a `width` by `height` model's input.
    pub(crate) fn tensor(&self, width: u32, height: u32) -> Result<CudaTensor, CudaDriverError> {
        let floats = 3 * (width as usize) * (height as usize);
        // SAFETY: `with_context` has the context current and the out-param is
        // a live local; a failure allocates nothing to leak.
        self.with_context(|| unsafe {
            let mut pointer = 0;
            check(
                "cuMemAlloc",
                cuMemAlloc_v2(&mut pointer, floats * size_of::<f32>()),
            )?;
            Ok(CudaTensor {
                ctx: self.ctx,
                pointer,
                floats,
            })
        })
    }

    /// Fits an NV12 picture, `size` in pixels, into `tensor` as `fit` says,
    /// made RGB by `colour`'s rows.
    ///
    /// Launches asynchronously: [`CudaDriver::synchronize`] is what makes
    /// the tensor ready for anything outside this context's stream.
    pub(crate) fn fit_nv12(
        &self,
        kernels: &FitKernels,
        tensor: &CudaTensor,
        fit: Fit,
        source: Nv12Surface,
        size: (u32, u32),
        colour: &YuvToBgra,
    ) -> Result<(), CudaDriverError> {
        let (grid, block) = grid(fit, tensor)?;
        let mut dst = tensor.pointer;
        let (mut model_w, mut model_h) = fit.model;
        let (mut offset_x, mut offset_y) = fit.offset;
        let (mut scaled_w, mut scaled_h) = fit.scaled;
        let mut luma = source.luma;
        let mut luma_pitch = source.luma_pitch as u32;
        let mut chroma = source.chroma;
        let mut chroma_pitch = source.chroma_pitch as u32;
        let (mut src_w, mut src_h) = size;
        let mut rows = colour.rows;
        self.with_context(|| {
            let mut params: Vec<*mut c_void> = vec![
                arg(&mut dst),
                arg(&mut model_w),
                arg(&mut model_h),
                arg(&mut offset_x),
                arg(&mut offset_y),
                arg(&mut scaled_w),
                arg(&mut scaled_h),
                arg(&mut luma),
                arg(&mut luma_pitch),
                arg(&mut chroma),
                arg(&mut chroma_pitch),
                arg(&mut src_w),
                arg(&mut src_h),
            ];
            params.extend(rows.iter_mut().flatten().map(arg));
            // SAFETY: one pointer per parameter `fit_nv12` declares, in that
            // order — thirteen buffer, size and placement values, then twelve
            // row coefficients — each at a live local; the context is current
            // inside `with_context`, and the kernel is this context's.
            unsafe { launch(kernels.nv12, grid, block, &mut params) }
        })
    }

    /// The same for a BGRA picture, read as it is.
    pub(crate) fn fit_bgra(
        &self,
        kernels: &FitKernels,
        tensor: &CudaTensor,
        fit: Fit,
        source: BgraSurface,
        size: (u32, u32),
    ) -> Result<(), CudaDriverError> {
        let (grid, block) = grid(fit, tensor)?;
        let mut dst = tensor.pointer;
        let (mut model_w, mut model_h) = fit.model;
        let (mut offset_x, mut offset_y) = fit.offset;
        let (mut scaled_w, mut scaled_h) = fit.scaled;
        let mut src = source.pixels;
        let mut src_pitch = source.pitch as u32;
        let (mut src_w, mut src_h) = size;
        self.with_context(|| {
            let mut params: Vec<*mut c_void> = vec![
                arg(&mut dst),
                arg(&mut model_w),
                arg(&mut model_h),
                arg(&mut offset_x),
                arg(&mut offset_y),
                arg(&mut scaled_w),
                arg(&mut scaled_h),
                arg(&mut src),
                arg(&mut src_pitch),
                arg(&mut src_w),
                arg(&mut src_h),
            ];
            // SAFETY: one pointer per parameter `fit_bgra` declares, in that
            // order, each at a live local; the context is current inside
            // `with_context`, and the kernel is this context's.
            unsafe { launch(kernels.bgra, grid, block, &mut params) }
        })
    }
}

/// The launch shape for `fit` — a thread per input pixel — after checking
/// `tensor` holds that many, which is what keeps the kernel's stores inside
/// it.
fn grid(fit: Fit, tensor: &CudaTensor) -> Result<((u32, u32), u32), CudaDriverError> {
    const BLOCK: u32 = 16;
    let (width, height) = fit.model;
    if 3 * (width as usize) * (height as usize) > tensor.floats {
        return Err(CudaDriverError::KernelRejected(format!(
            "a {width}x{height} input does not fit a tensor of {} floats",
            tensor.floats
        )));
    }
    Ok(((width.div_ceil(BLOCK), height.div_ceil(BLOCK)), BLOCK))
}
