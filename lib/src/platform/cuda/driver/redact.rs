//! What [`crate::elements::CudaDetectionOverlay`] asks of the driver to hide
//! a box: its own kernels, loaded only by an overlay that hides — see
//! [`super::ptx::REDACT_PTX`] — and the buffer a box's cell means go
//! through.

use std::ffi::c_void;

use super::ptx::REDACT_PTX;
use super::{
    CUcontext, CUdeviceptr, CUfunction, CUmodule, CudaDriver, CudaDriverError, arg, check,
    cuCtxPopCurrent_v2, cuCtxPushCurrent_v2, cuMemAlloc_v2, cuMemFree_v2, cuMemcpyDtoH_v2,
    cuMemcpyHtoD_v2, cuModuleUnload, launch, load_module,
};

/// The hiding kernels, JIT-compiled into the driver's context. Dropped
/// before the driver that made them, which is what keeps their context
/// alive while they unload.
pub(crate) struct RedactKernels {
    ctx: CUcontext,
    module: CUmodule,
    means: CUfunction,
    paint: CUfunction,
    /// The same for two-byte samples, P010's.
    means16: CUfunction,
    paint16: CUfunction,
}

// SAFETY: a module and two function handles in a context that is pushed
// around every use; nothing here is thread-affine — see `CudaDriver`.
unsafe impl Send for RedactKernels {}

impl Drop for RedactKernels {
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

/// A device buffer of `f32`s a box's cell means are written to and read
/// back from, grown as a larger box needs.
pub(crate) struct CellMeans {
    ctx: CUcontext,
    pointer: CUdeviceptr,
    floats: usize,
}

// SAFETY: a plain device allocation, freed in the context it was made in —
// the contract `CudaBgraScratch` documents.
unsafe impl Send for CellMeans {}

impl Drop for CellMeans {
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

impl CellMeans {
    /// How many `f32`s it holds.
    pub(crate) fn floats(&self) -> usize {
        self.floats
    }
}

/// One plane of a box to hide: `width` by `height` samples of `channels`
/// channels each from `plane`, rows `pitch` bytes apart, cut into `cells` —
/// across, down. A channel is a byte, or with `wide` two, ten bits at the
/// top, as P010 lays them.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CellPlane {
    pub(crate) plane: CUdeviceptr,
    pub(crate) pitch: usize,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) channels: u32,
    pub(crate) cells: (u32, u32),
    pub(crate) wide: bool,
}

impl CellPlane {
    /// The bytes a row of its samples takes.
    fn row_bytes(&self) -> usize {
        self.width as usize * self.channels as usize * if self.wide { 2 } else { 1 }
    }
}

impl CudaDriver {
    /// Loads the hiding kernels into this driver's context.
    pub(crate) fn redact_kernels(&self) -> Result<RedactKernels, CudaDriverError> {
        self.with_context(|| {
            // SAFETY: the context is current inside `with_context`, which is
            // `load_module`'s whole contract.
            let (module, [means, paint, means16, paint16]) = unsafe {
                load_module(
                    REDACT_PTX,
                    ["cell_means", "cell_paint", "cell_means16", "cell_paint16"],
                )?
            };
            Ok(RedactKernels {
                ctx: self.ctx,
                module,
                means,
                paint,
                means16,
                paint16,
            })
        })
    }

    /// A buffer of `floats` `f32`s for cell means.
    pub(crate) fn cell_means(&self, floats: usize) -> Result<CellMeans, CudaDriverError> {
        let floats = floats.max(1);
        // SAFETY: `with_context` has the context current and the out-param is
        // a live local; a failure allocates nothing to leak.
        self.with_context(|| unsafe {
            let mut pointer = 0;
            check(
                "cuMemAlloc",
                cuMemAlloc_v2(&mut pointer, floats * size_of::<f32>()),
            )?;
            Ok(CellMeans {
                ctx: self.ctx,
                pointer,
                floats,
            })
        })
    }

    /// Writes the mean of each cell of `plane` to `means`, from float
    /// `offset` on: `cells` rows of cells of `channels` floats, as the CPU's
    /// `cut::cell_means` makes them. Launched on the default stream and not
    /// waited for; [`Self::read_means`] waits.
    pub(crate) fn measure_cells(
        &self,
        kernels: &RedactKernels,
        means: &CellMeans,
        offset: usize,
        plane: CellPlane,
    ) -> Result<(), CudaDriverError> {
        let (cells_x, cells_y) = plane.cells;
        let cells = cells_x as usize * cells_y as usize;
        if cells_x == 0
            || cells_y == 0
            || cells_x > plane.width
            || cells_y > plane.height
            || !(1..=4).contains(&plane.channels)
            || offset + cells * plane.channels as usize > means.floats
            || plane.pitch < plane.row_bytes()
            || plane.pitch > u32::MAX as usize
        {
            return Err(CudaDriverError::KernelRejected(format!(
                "{}x{} samples of {} bytes in {cells_x}x{cells_y} cells, pitch {}, from float {offset} of {}",
                plane.width, plane.height, plane.channels, plane.pitch, means.floats
            )));
        }
        let mut source = plane.plane;
        let mut pitch = plane.pitch as u32;
        let (mut width, mut height) = (plane.width, plane.height);
        let mut channels = plane.channels;
        let (mut across, mut down) = (cells_x, cells_y);
        let mut data = means.pointer + (offset * size_of::<f32>()) as CUdeviceptr;
        self.with_context(|| {
            let mut params: Vec<*mut c_void> = vec![
                arg(&mut source),
                arg(&mut pitch),
                arg(&mut width),
                arg(&mut height),
                arg(&mut channels),
                arg(&mut across),
                arg(&mut down),
                arg(&mut data),
            ];
            // SAFETY: one pointer per parameter `cell_means` declares, in that
            // order, each at a live local. It reads the plane's `width` by
            // `height` samples, which the surface the caller took `plane`
            // from holds, and writes `cells · channels` floats from `offset`,
            // which the check above keeps inside `means`; the context is
            // current.
            unsafe {
                launch(
                    if plane.wide {
                        kernels.means16
                    } else {
                        kernels.means
                    },
                    ((cells as u32).div_ceil(256), 1),
                    16,
                    &mut params,
                )
            }
        })
    }

    /// The first `into.len()` floats of `means`, once everything launched
    /// before on the default stream has finished.
    pub(crate) fn read_means(
        &self,
        means: &CellMeans,
        into: &mut [f32],
    ) -> Result<(), CudaDriverError> {
        if into.len() > means.floats {
            return Err(CudaDriverError::KernelRejected(format!(
                "{} floats read from {}",
                into.len(),
                means.floats
            )));
        }
        // SAFETY: `with_context` has the buffer's context current; `into` is a
        // live slice no longer than the allocation, and a synchronous copy on
        // the legacy default stream waits for the kernels before it.
        self.with_context(|| unsafe {
            check(
                "cuMemcpyDtoH",
                cuMemcpyDtoH_v2(
                    into.as_mut_ptr().cast(),
                    means.pointer,
                    std::mem::size_of_val(into),
                ),
            )
        })
    }

    /// Paints each cell of `plane` its mean — or, with `smooth`, its mean
    /// blended into its neighbours' — through `means`; with `ellipse`, only
    /// the samples inside the ellipse touching the plane's sides, and with
    /// `feather`, faded toward the edge (`overlay::cover`). With
    /// `fill`, the means are not taken: the plane, as one cell, is painted
    /// those values — a fill cut to an ellipse. Launches on the default
    /// stream, each reading what the one before wrote; nothing is waited
    /// for.
    pub(crate) fn hide_cells(
        &self,
        kernels: &RedactKernels,
        means: &CellMeans,
        plane: CellPlane,
        (smooth, ellipse, feather): (bool, bool, f32),
        fill: Option<&[f32]>,
    ) -> Result<(), CudaDriverError> {
        let (cells_x, cells_y) = plane.cells;
        let cells = cells_x as usize * cells_y as usize;
        if cells_x == 0
            || cells_y == 0
            || cells_x > plane.width
            || cells_y > plane.height
            || !(1..=4).contains(&plane.channels)
            || cells * plane.channels as usize > means.floats
            || plane.pitch < plane.row_bytes()
            || plane.pitch > u32::MAX as usize
            || fill.is_some_and(|values| values.len() != cells * plane.channels as usize)
        {
            return Err(CudaDriverError::KernelRejected(format!(
                "{}x{} samples of {} bytes in {cells_x}x{cells_y} cells, pitch {}, means for {} floats",
                plane.width, plane.height, plane.channels, plane.pitch, means.floats
            )));
        }
        let mut source = plane.plane;
        let mut pitch = plane.pitch as u32;
        let (mut width, mut height) = (plane.width, plane.height);
        let mut channels = plane.channels;
        let (mut across, mut down) = (cells_x, cells_y);
        let mut data = means.pointer;
        let mut smooth = u32::from(smooth);
        let mut ellipse = u32::from(ellipse);
        let mut feather = feather;
        self.with_context(|| {
            let mut params: Vec<*mut c_void> = vec![
                arg(&mut source),
                arg(&mut pitch),
                arg(&mut width),
                arg(&mut height),
                arg(&mut channels),
                arg(&mut across),
                arg(&mut down),
                arg(&mut data),
            ];
            match fill {
                // SAFETY: the values, checked above to be the plane's one
                // cell's channels, into the first floats of `means`, which
                // hold them; a synchronous copy on the default stream, before
                // the launch that reads them.
                Some(values) => unsafe {
                    check(
                        "cuMemcpyHtoD",
                        cuMemcpyHtoD_v2(
                            data,
                            values.as_ptr().cast(),
                            std::mem::size_of_val(values),
                        ),
                    )?
                },
                // SAFETY: one pointer per parameter `cell_means` declares, in
                // that order, each at a live local. It reads the plane's
                // `width` by `height` samples, which the surface the caller
                // took `plane` from holds, and writes `cells · channels` floats
                // of `means`, which the check above holds it to; the context
                // is current.
                None => unsafe {
                    launch(
                        if plane.wide {
                            kernels.means16
                        } else {
                            kernels.means
                        },
                        ((cells as u32).div_ceil(256), 1),
                        16,
                        &mut params,
                    )?
                },
            }
            params.push(arg(&mut smooth));
            params.push(arg(&mut ellipse));
            params.push(arg(&mut feather));
            // SAFETY: as above, with `smooth`, `ellipse` and `feather` last
            // as `cell_paint` declares them; a thread a sample of the same
            // plane, each writing its own.
            unsafe {
                launch(
                    if plane.wide {
                        kernels.paint16
                    } else {
                        kernels.paint
                    },
                    (width.div_ceil(16), height.div_ceil(16)),
                    16,
                    &mut params,
                )
            }
        })
    }
}
