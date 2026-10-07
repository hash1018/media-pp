//! What [`crate::elements::CudaOrtDetector`] and
//! [`crate::elements::CudaOrtClassifier`] ask of the driver: their own
//! kernels, loaded only by them — see [`super::ptx::FIT_PTX`] — and the
//! device buffer they fit pictures into.

use std::ffi::c_void;

#[cfg(test)]
use super::cuMemcpyHtoD_v2;
use super::ptx::FIT_PTX;
use super::{
    BgraSurface, CUcontext, CUdeviceptr, CUfunction, CUmodule, CudaDriver, CudaDriverError,
    Nv12Surface, YuvToBgra, arg, check, cuCtxPopCurrent_v2, cuCtxPushCurrent_v2, cuMemAlloc_v2,
    cuMemFree_v2, cuMemcpyDtoH_v2, cuModuleUnload, launch, load_module,
};
use crate::orientation::Orientation;

/// The fitting kernels, JIT-compiled into the driver's context. Dropped
/// before the driver that made them, which is what keeps their context
/// alive while they unload.
pub(crate) struct FitKernels {
    ctx: CUcontext,
    module: CUmodule,
    nv12: CUfunction,
    bgra: CUfunction,
    scale: CUfunction,
    swap: CUfunction,
    best: CUfunction,
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

/// A device buffer of `f32`s a model's input is fitted into: three planes,
/// R, G and B, of the model's width by height, for each picture of a batch.
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
/// kernels take — and how the picture is turned to be shown, which is how
/// it is fitted: the corner and size are of it turned.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Fit {
    pub(crate) model: (u32, u32),
    pub(crate) offset: (u32, u32),
    pub(crate) scaled: (u32, u32),
    pub(crate) orientation: Orientation,
}

impl Fit {
    /// The source's size as shown, and where each shown pixel of it is
    /// stored, for a source stored `size`: what the kernels read it by.
    fn reading(self, size: (u32, u32)) -> ((u32, u32), [i32; 6]) {
        (
            self.orientation.display_size(size.0, size.1),
            self.orientation.sampling(size.0, size.1),
        )
    }
}

impl CudaDriver {
    /// Loads the fitting kernels into this driver's context.
    pub(crate) fn fit_kernels(&self) -> Result<FitKernels, CudaDriverError> {
        self.with_context(|| {
            // SAFETY: the context is current inside `with_context`, which is
            // `load_module`'s whole contract.
            let (module, [nv12, bgra, scale, swap, best]) = unsafe {
                load_module(
                    FIT_PTX,
                    [
                        "fit_nv12",
                        "fit_bgra",
                        "scale_planes",
                        "swap_planes",
                        "best_class",
                    ],
                )?
            };
            Ok(FitKernels {
                ctx: self.ctx,
                module,
                nv12,
                bgra,
                scale,
                swap,
                best,
            })
        })
    }

    /// A device buffer for `pictures` inputs of a `width` by `height` model,
    /// one after another.
    pub(crate) fn batch_tensor(
        &self,
        width: u32,
        height: u32,
        pictures: usize,
    ) -> Result<CudaTensor, CudaDriverError> {
        self.buffer(3 * (width as usize) * (height as usize) * pictures.max(1))
    }

    /// A device buffer of `floats` `f32`s.
    pub(crate) fn buffer(&self, floats: usize) -> Result<CudaTensor, CudaDriverError> {
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

    /// Fits an NV12 picture, stored `size` in pixels, into the `slot`th
    /// input of `tensor` as `fit` says, made RGB by `colour`'s rows.
    #[allow(clippy::too_many_arguments)]
    ///
    /// Launches asynchronously: [`CudaDriver::synchronize`] is what makes
    /// the tensor ready for anything outside this context's stream.
    pub(crate) fn fit_nv12(
        &self,
        kernels: &FitKernels,
        tensor: &CudaTensor,
        slot: usize,
        fit: Fit,
        source: Nv12Surface,
        size: (u32, u32),
        colour: &YuvToBgra,
    ) -> Result<(), CudaDriverError> {
        let (grid, block, mut dst) = grid(fit, tensor, slot)?;
        let (mut model_w, mut model_h) = fit.model;
        let (mut offset_x, mut offset_y) = fit.offset;
        let (mut scaled_w, mut scaled_h) = fit.scaled;
        let mut luma = source.luma;
        let mut luma_pitch = source.luma_pitch as u32;
        let mut chroma = source.chroma;
        let mut chroma_pitch = source.chroma_pitch as u32;
        let ((mut src_w, mut src_h), mut sampling) = fit.reading(size);
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
            params.extend(sampling.iter_mut().map(arg));
            params.extend(rows.iter_mut().flatten().map(arg));
            // SAFETY: one pointer per parameter `fit_nv12` declares, in that
            // order — thirteen buffer, size and placement values, six of
            // sampling, then twelve row coefficients — each at a live local;
            // the context is current
            // inside `with_context`, and the kernel is this context's.
            unsafe { launch(kernels.nv12, grid, block, &mut params) }
        })
    }

    /// The same for a BGRA picture, read as it is.
    pub(crate) fn fit_bgra(
        &self,
        kernels: &FitKernels,
        tensor: &CudaTensor,
        slot: usize,
        fit: Fit,
        source: BgraSurface,
        size: (u32, u32),
    ) -> Result<(), CudaDriverError> {
        let (grid, block, mut dst) = grid(fit, tensor, slot)?;
        let (mut model_w, mut model_h) = fit.model;
        let (mut offset_x, mut offset_y) = fit.offset;
        let (mut scaled_w, mut scaled_h) = fit.scaled;
        let mut src = source.pixels;
        let mut src_pitch = source.pitch as u32;
        let ((mut src_w, mut src_h), mut sampling) = fit.reading(size);
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
            params.extend(sampling.iter_mut().map(arg));
            // SAFETY: one pointer per parameter `fit_bgra` declares, in that
            // order, each at a live local; the context is current inside
            // `with_context`, and the kernel is this context's.
            unsafe { launch(kernels.bgra, grid, block, &mut params) }
        })
    }
}

/// The launch shape for `fit` — a thread per input pixel — and where the
/// `slot`th input starts, after checking `tensor` holds it, which is what
/// keeps the kernel's stores inside it.
fn grid(
    fit: Fit,
    tensor: &CudaTensor,
    slot: usize,
) -> Result<((u32, u32), u32, CUdeviceptr), CudaDriverError> {
    const BLOCK: u32 = 16;
    let (width, height) = fit.model;
    let input = 3 * (width as usize) * (height as usize);
    if (slot + 1) * input > tensor.floats {
        return Err(CudaDriverError::KernelRejected(format!(
            "input {slot} of {width}x{height} does not fit a tensor of {} floats",
            tensor.floats
        )));
    }
    let start = tensor.pointer + (slot * input * size_of::<f32>()) as CUdeviceptr;
    Ok((
        (width.div_ceil(BLOCK), height.div_ceil(BLOCK)),
        BLOCK,
        start,
    ))
}

impl CudaDriver {
    /// Makes each float of the first `pictures` inputs of a `width` by
    /// `height` model in `tensor` `x · scale + bias`, by its plane's
    /// channel — a model's normalisation, applied after the fits.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn scale_planes(
        &self,
        kernels: &FitKernels,
        tensor: &CudaTensor,
        (width, height): (u32, u32),
        pictures: usize,
        scale: [f32; 3],
        bias: [f32; 3],
    ) -> Result<(), CudaDriverError> {
        let plane = (width as usize) * (height as usize);
        let total = 3 * plane * pictures;
        if total > tensor.floats || total > u32::MAX as usize {
            return Err(CudaDriverError::KernelRejected(format!(
                "{pictures} inputs of {width}x{height} do not fit a tensor of {} floats",
                tensor.floats
            )));
        }
        let mut data = tensor.pointer;
        let (mut plane, mut total) = (plane as u32, total as u32);
        let [mut sr, mut sg, mut sb] = scale;
        let [mut br, mut bg, mut bb] = bias;
        self.with_context(|| {
            let mut params: Vec<*mut c_void> = vec![
                arg(&mut data),
                arg(&mut plane),
                arg(&mut total),
                arg(&mut sr),
                arg(&mut sg),
                arg(&mut sb),
                arg(&mut br),
                arg(&mut bg),
                arg(&mut bb),
            ];
            // SAFETY: one pointer per parameter `scale_planes` declares, in
            // that order, each at a live local; the kernel reads and writes the
            // first `total` floats of `tensor`, which the check above holds it
            // to, and the context is current inside `with_context`.
            unsafe { launch(kernels.scale, (total.div_ceil(256), 1), 16, &mut params) }
        })
    }

    /// Swaps the first and third planes of each of the first `pictures`
    /// inputs of a `width` by `height` model in `tensor`: R G B made B G R,
    /// for a model that reads OpenCV's order — before [`Self::scale_planes`],
    /// whose channels are then the model's.
    pub(crate) fn swap_planes(
        &self,
        kernels: &FitKernels,
        tensor: &CudaTensor,
        (width, height): (u32, u32),
        pictures: usize,
    ) -> Result<(), CudaDriverError> {
        let plane = (width as usize) * (height as usize);
        let total = plane * pictures;
        if 3 * total > tensor.floats || 3 * total > u32::MAX as usize {
            return Err(CudaDriverError::KernelRejected(format!(
                "{pictures} inputs of {width}x{height} do not fit a tensor of {} floats",
                tensor.floats
            )));
        }
        let mut data = tensor.pointer;
        let (mut plane, mut total) = (plane as u32, total as u32);
        self.with_context(|| {
            let mut params: Vec<*mut c_void> =
                vec![arg(&mut data), arg(&mut plane), arg(&mut total)];
            // SAFETY: one pointer per parameter `swap_planes` declares, in
            // that order, each at a live local; a thread a pixel of the
            // first `total` of `tensor`'s pictures, reading and writing two
            // of its three planes, which the check above holds to the
            // tensor; the context is current inside `with_context`.
            unsafe { launch(kernels.swap, (total.div_ceil(256), 1), 16, &mut params) }
        })
    }

    /// Copies `from` up into the start of `into`.
    #[cfg(test)]
    pub(crate) fn upload_floats(
        &self,
        from: &[f32],
        into: &CudaTensor,
    ) -> Result<(), CudaDriverError> {
        assert!(
            from.len() <= into.floats,
            "{} into {}",
            from.len(),
            into.floats
        );
        // SAFETY: the context is current; the copy writes no more than the
        // tensor holds, checked above.
        self.with_context(|| unsafe {
            check(
                "cuMemcpyHtoD",
                cuMemcpyHtoD_v2(
                    into.pointer,
                    from.as_ptr().cast(),
                    std::mem::size_of_val(from),
                ),
            )
        })
    }
}

impl CudaDriver {
    /// Reads the `pictures` outputs of a YOLOv8-layout model at `src` on the
    /// device — `rows` (four, then a score per class) by `boxes` floats each
    /// — into `dst`, six floats a box: its centre, width and height, its
    /// best class's score and that class.
    ///
    /// # Safety
    ///
    /// `src` must hold `pictures · rows · boxes` floats in this driver's
    /// context, written by work this call is ordered after — ONNX Runtime's
    /// run, which has finished when it returns — and stay allocated until
    /// the next synchronous call on this context, the copy down, returns.
    pub(crate) unsafe fn best_class(
        &self,
        kernels: &FitKernels,
        src: CUdeviceptr,
        dst: &CudaTensor,
        (pictures, rows, boxes): (usize, usize, usize),
    ) -> Result<(), CudaDriverError> {
        let total = pictures * boxes;
        if rows < 5
            || total * 6 > dst.floats
            || total > u32::MAX as usize
            || rows * boxes > u32::MAX as usize
        {
            return Err(CudaDriverError::KernelRejected(format!(
                "{pictures} outputs of {rows} by {boxes} do not fit {} floats of six a box",
                dst.floats
            )));
        }
        let (mut src, mut data) = (src, dst.pointer);
        let (mut rows, mut boxes, mut total) = (rows as u32, boxes as u32, total as u32);
        self.with_context(|| {
            let mut params: Vec<*mut c_void> = vec![
                arg(&mut src),
                arg(&mut data),
                arg(&mut rows),
                arg(&mut boxes),
                arg(&mut total),
            ];
            // SAFETY: one pointer per parameter `best_class` declares, in that
            // order, each at a live local. It reads `pictures · rows · boxes`
            // floats of `src`, the caller's promise, and writes six floats for
            // each of `total` boxes into `dst`, which the check above holds it
            // to; the context is current inside `with_context`.
            unsafe { launch(kernels.best, (total.div_ceil(256), 1), 16, &mut params) }
        })
    }

    /// Copies the first `into.len()` floats of `from` down, after the work
    /// queued before it on the default stream.
    pub(crate) fn download(
        &self,
        from: &CudaTensor,
        into: &mut [f32],
    ) -> Result<(), CudaDriverError> {
        if into.len() > from.floats {
            return Err(CudaDriverError::KernelRejected(format!(
                "{} floats out of {}",
                into.len(),
                from.floats
            )));
        }
        // SAFETY: a copy of `into`'s own length from no further into `from`
        // than the check above holds it to.
        unsafe { self.download_from(from.pointer, into) }
    }

    /// Copies `into.len()` floats down from `from`, after the work queued
    /// before it on the default stream.
    ///
    /// # Safety
    ///
    /// `from` must hold `into.len()` floats in this driver's context.
    pub(crate) unsafe fn download_from(
        &self,
        from: CUdeviceptr,
        into: &mut [f32],
    ) -> Result<(), CudaDriverError> {
        // SAFETY: the context is current; the copy writes `into`'s own length,
        // read from an allocation the caller promises holds as much.
        self.with_context(|| unsafe {
            check(
                "cuMemcpyDtoH",
                cuMemcpyDtoH_v2(into.as_mut_ptr().cast(), from, std::mem::size_of_val(into)),
            )
        })
    }
}
