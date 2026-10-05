//! What `ObjectTracker` asks of the driver to follow objects by how they
//! look on CUDA pictures — see [`super::ptx::DCF_PTX`]: its kernels, the
//! transforms cuFFT makes between them, and the device memory every
//! object's correlation filter and every picture's samples live in.
//!
//! cuFFT is linked, by build.rs, with the `cuda-visual-tracking` feature:
//! the program does not start without it, as it does not without FFmpeg.

use std::collections::HashMap;
use std::ffi::c_void;

use super::ptx::DCF_PTX;
use super::{
    CUcontext, CUdeviceptr, CUfunction, CUmodule, CudaDriver, CudaDriverError, arg, check,
    cuCtxPopCurrent_v2, cuCtxPushCurrent_v2, cuMemAlloc_v2, cuMemFree_v2, cuMemcpyDtoH_v2,
    cuMemcpyHtoD_v2, cuMemsetD8_v2, cuModuleUnload, launch, launch_layers, load_module,
};

/// A cuFFT plan.
type CufftHandle = i32;
/// `CUFFT_C2C`: complex to complex, single precision.
const CUFFT_C2C: i32 = 0x29;
/// `CUFFT_FORWARD`.
const CUFFT_FORWARD: i32 = -1;
/// `CUFFT_INVERSE`.
const CUFFT_INVERSE: i32 = 1;

// cuFFT's C ABI, from cufft.h, linked by build.rs. Every call checks the
// `cufftResult` it returns.
unsafe extern "C" {
    fn cufftPlanMany(
        plan: *mut CufftHandle,
        rank: i32,
        n: *mut i32,
        inembed: *mut i32,
        istride: i32,
        idist: i32,
        onembed: *mut i32,
        ostride: i32,
        odist: i32,
        kind: i32,
        batch: i32,
    ) -> i32;
    fn cufftExecC2C(
        plan: CufftHandle,
        idata: *mut c_void,
        odata: *mut c_void,
        direction: i32,
    ) -> i32;
    fn cufftDestroy(plan: CufftHandle) -> i32;
    fn cufftGetVersion(version: *mut i32) -> i32;
}

/// The side of a neighbourhood's square of samples.
const SIDE: u32 = 64;
/// Samples in one.
const CELLS: usize = (SIDE * SIDE) as usize;

fn cufft(call: &'static str, result: i32) -> Result<(), CudaDriverError> {
    if result == 0 {
        Ok(())
    } else {
        Err(CudaDriverError::KernelRejected(format!(
            "{call} failed: cufftResult {result}"
        )))
    }
}

/// A device allocation, freed in the context it was made in.
struct Buffer {
    ctx: CUcontext,
    pointer: CUdeviceptr,
    bytes: usize,
}

// SAFETY: a plain device allocation, freed in the context it was made in —
// the contract `CudaTensor` documents.
unsafe impl Send for Buffer {}

impl Buffer {
    fn new(driver: &CudaDriver, bytes: usize) -> Result<Self, CudaDriverError> {
        // SAFETY: `with_context` has the context current and the out-params are
        // live locals; the fresh allocation is zeroed, its own whole length.
        driver.with_context(|| unsafe {
            let mut pointer = 0;
            check("cuMemAlloc", cuMemAlloc_v2(&mut pointer, bytes.max(1)))?;
            check("cuMemsetD8", cuMemsetD8_v2(pointer, 0, bytes.max(1)))?;
            Ok(Self {
                ctx: driver.ctx,
                pointer,
                bytes,
            })
        })
    }

    fn upload(&self, driver: &CudaDriver, data: &[u8]) -> Result<(), CudaDriverError> {
        assert!(
            data.len() <= self.bytes,
            "{} bytes into {}",
            data.len(),
            self.bytes
        );
        // SAFETY: the context is current; the copy reads `data`'s own length
        // and writes no further into this allocation, which the assertion
        // holds to be at least that long.
        driver.with_context(|| unsafe {
            check(
                "cuMemcpyHtoD",
                cuMemcpyHtoD_v2(self.pointer, data.as_ptr().cast(), data.len()),
            )
        })
    }

    fn download(&self, driver: &CudaDriver, into: &mut [u8]) -> Result<(), CudaDriverError> {
        assert!(
            into.len() <= self.bytes,
            "{} bytes out of {}",
            into.len(),
            self.bytes
        );
        // SAFETY: the context is current; the copy writes `into`'s own length,
        // read from no further into this allocation than the assertion holds.
        // It waits for the work queued before it on the default stream.
        driver.with_context(|| unsafe {
            check(
                "cuMemcpyDtoH",
                cuMemcpyDtoH_v2(into.as_mut_ptr().cast(), self.pointer, into.len()),
            )
        })
    }
}

impl Drop for Buffer {
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

/// A picture's brightness on the device: an NV12 surface's luma plane, or
/// a BGRA surface.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DcfSource {
    pub(crate) pixels: CUdeviceptr,
    pub(crate) pitch: usize,
    pub(crate) size: (u32, u32),
    pub(crate) bgra: bool,
}

/// One neighbourhood to sample: its centre, width and height, in pixels.
pub(crate) type DcfJob = [f32; 4];

/// Where a response peaks, in samples from the neighbourhood's corner, how
/// high, and how it stands out of the rest.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct DcfPeak {
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) top: f32,
    pub(crate) psr: f32,
}

/// The kernels, cuFFT's plans and the device memory of every followed
/// object's filter and of one batch of samples.
///
/// The buffers are freed in the driver's context, which the driver that
/// made this keeps retained for as long as it lives.
pub(crate) struct CudaDcf {
    ctx: CUcontext,
    module: CUmodule,
    sample: CUfunction,
    normalize: CUfunction,
    correlate: CUfunction,
    peak: CUfunction,
    learn: CUfunction,
    window: Buffer,
    target: Buffer,
    /// Each filter slot's numerator, `CELLS` complex values.
    numerator: Buffer,
    /// Each filter slot's denominator, `CELLS` real values.
    denominator: Buffer,
    slots: usize,
    /// One batch of samples, then spectra, then responses.
    spectra: Buffer,
    jobs: Buffer,
    job_slots: Buffer,
    peaks: Buffer,
    capacity: usize,
    /// How many jobs `spectra` holds now.
    sampled: usize,
    plans: HashMap<usize, CufftHandle>,
}

// SAFETY: a module, function handles, plans and device allocations, used by
// the one thread transforming at a time, in a context pushed around every
// use — see `CudaDriver`.
unsafe impl Send for CudaDcf {}

impl Drop for CudaDcf {
    fn drop(&mut self) {
        // SAFETY: the plans and the module were made in `self.ctx`, still
        // retained by the driver this outlives by contract; each is destroyed
        // once. The buffers drop after, each in the same context.
        unsafe {
            if cuCtxPushCurrent_v2(self.ctx) == 0 {
                for plan in self.plans.values() {
                    cufftDestroy(*plan);
                }
                cuModuleUnload(self.module);
                let mut popped: CUcontext = std::ptr::null_mut();
                cuCtxPopCurrent_v2(&mut popped);
            }
        }
    }
}

/// cuFFT's version, `1000 × major + 100 × minor + patch` — what calling
/// into it for keeps the link to it.
pub(crate) fn cufft_version() -> i32 {
    let mut version = 0;
    // SAFETY: `cufftGetVersion` writes one int through a pointer to a live
    // local, and needs no context.
    unsafe {
        cufftGetVersion(&mut version);
    }
    version
}

impl CudaDriver {
    /// Loads the kernels into this driver's context, and puts `window` and
    /// the wanted response's spectrum `target`, both `CELLS` long, on the
    /// device.
    pub(crate) fn dcf(
        &self,
        window: &[f32],
        target: &[[f32; 2]],
    ) -> Result<CudaDcf, CudaDriverError> {
        assert_eq!(window.len(), CELLS);
        assert_eq!(target.len(), CELLS);
        let (module, [sample, normalize, correlate, peak, learn]) = self.with_context(|| {
            // SAFETY: the context is current inside `with_context`, which is
            // `load_module`'s whole contract.
            unsafe {
                load_module(
                    DCF_PTX,
                    [
                        "dcf_sample",
                        "dcf_normalize",
                        "dcf_correlate",
                        "dcf_peak",
                        "dcf_learn",
                    ],
                )
            }
        })?;
        let window_buffer = Buffer::new(self, CELLS * 4)?;
        window_buffer.upload(self, bytes(window))?;
        let target_buffer = Buffer::new(self, CELLS * 8)?;
        target_buffer.upload(self, bytes(target))?;
        Ok(CudaDcf {
            ctx: self.ctx,
            module,
            sample,
            normalize,
            correlate,
            peak,
            learn,
            window: window_buffer,
            target: target_buffer,
            numerator: Buffer::new(self, 0)?,
            denominator: Buffer::new(self, 0)?,
            slots: 0,
            spectra: Buffer::new(self, 0)?,
            jobs: Buffer::new(self, 0)?,
            job_slots: Buffer::new(self, 0)?,
            peaks: Buffer::new(self, 0)?,
            capacity: 0,
            sampled: 0,
            plans: HashMap::new(),
        })
    }
}

/// `values` as their bytes.
fn bytes<T: Copy>(values: &[T]) -> &[u8] {
    // SAFETY: `T` here is `f32`, `u32` or an array of `f32`: plain data with
    // no padding, every byte of which is initialized, viewed for as long as
    // the slice is borrowed.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

impl CudaDcf {
    /// Room for filters in `slots` slots, keeping those already learned.
    pub(crate) fn reserve_slots(
        &mut self,
        driver: &CudaDriver,
        slots: usize,
    ) -> Result<(), CudaDriverError> {
        if slots <= self.slots {
            return Ok(());
        }
        let grown = slots.next_power_of_two().max(16);
        let numerator = Buffer::new(driver, grown * CELLS * 8)?;
        let denominator = Buffer::new(driver, grown * CELLS * 4)?;
        if self.slots > 0 {
            // Rare — when more objects are followed at once than ever before
            // — so through the host rather than a copy of the driver's own.
            let mut kept = vec![0u8; self.slots * CELLS * 8];
            self.numerator.download(driver, &mut kept)?;
            numerator.upload(driver, &kept)?;
            let mut kept = vec![0u8; self.slots * CELLS * 4];
            self.denominator.download(driver, &mut kept)?;
            denominator.upload(driver, &kept)?;
        }
        self.numerator = numerator;
        self.denominator = denominator;
        self.slots = grown;
        Ok(())
    }

    fn reserve_jobs(&mut self, driver: &CudaDriver, jobs: usize) -> Result<(), CudaDriverError> {
        if jobs <= self.capacity {
            return Ok(());
        }
        let grown = jobs.next_power_of_two().max(16);
        self.spectra = Buffer::new(driver, grown * CELLS * 8)?;
        self.jobs = Buffer::new(driver, grown * 16)?;
        self.job_slots = Buffer::new(driver, grown * 4)?;
        self.peaks = Buffer::new(driver, grown * 16)?;
        self.capacity = grown;
        Ok(())
    }

    /// Transforms the first `count` neighbourhoods of `spectra` in place.
    fn transform(
        &mut self,
        driver: &CudaDriver,
        count: usize,
        direction: i32,
    ) -> Result<(), CudaDriverError> {
        let plan = match self.plans.get(&count) {
            Some(plan) => *plan,
            None => {
                let mut plan = 0;
                let mut n = [SIDE as i32, SIDE as i32];
                // SAFETY: the context is current inside `with_context`; the
                // out-param and the dimensions are live locals, and the null
                // layouts ask for the basic one — `count` contiguous squares.
                driver.with_context(|| unsafe {
                    cufft(
                        "cufftPlanMany",
                        cufftPlanMany(
                            &mut plan,
                            2,
                            n.as_mut_ptr(),
                            std::ptr::null_mut(),
                            1,
                            CELLS as i32,
                            std::ptr::null_mut(),
                            1,
                            CELLS as i32,
                            CUFFT_C2C,
                            count as i32,
                        ),
                    )
                })?;
                self.plans.insert(count, plan);
                plan
            }
        };
        let data = self.spectra.pointer as usize as *mut c_void;
        // SAFETY: the plan transforms `count` squares of `CELLS` complex
        // values in place, which `spectra` holds — `reserve_jobs` made it at
        // least that long — in the context current inside `with_context`, on
        // the default stream the kernels around it run on.
        driver.with_context(|| unsafe {
            cufft("cufftExecC2C", cufftExecC2C(plan, data, data, direction))
        })
    }

    /// Samples each of `jobs` from `source`, readies the samples to be
    /// correlated, and transforms them, for [`Self::find`] or
    /// [`Self::learn`] to use.
    pub(crate) fn sample(
        &mut self,
        driver: &CudaDriver,
        source: DcfSource,
        jobs: &[DcfJob],
    ) -> Result<(), CudaDriverError> {
        self.sampled = 0;
        if jobs.is_empty() {
            return Ok(());
        }
        self.reserve_jobs(driver, jobs.len())?;
        self.jobs.upload(driver, bytes(jobs))?;
        let count = jobs.len() as u32;
        let mut src = source.pixels;
        let mut pitch = source.pitch as u32;
        let (mut width, mut height) = source.size;
        let mut bgra = u32::from(source.bgra);
        let mut job_list = self.jobs.pointer;
        let mut out = self.spectra.pointer;
        let mut window = self.window.pointer;
        driver.with_context(|| {
            let mut params: Vec<*mut c_void> = vec![
                arg(&mut src),
                arg(&mut pitch),
                arg(&mut width),
                arg(&mut height),
                arg(&mut bgra),
                arg(&mut job_list),
                arg(&mut out),
            ];
            // SAFETY: one pointer per parameter `dcf_sample` declares, in that
            // order, each at a live local. It reads `source` within its size
            // and pitch — the caller's surface — and writes `count` squares of
            // `spectra`, which holds at least that many; the context is current.
            unsafe { launch_layers(self.sample, (SIDE / 16, SIDE / 16, count), 16, &mut params)? };
            let mut params: Vec<*mut c_void> = vec![arg(&mut out), arg(&mut window)];
            // SAFETY: the same squares, a block each, and the window of `CELLS`
            // floats `dcf` put on the device.
            unsafe { launch(self.normalize, (count, 1), 16, &mut params) }
        })?;
        self.transform(driver, jobs.len(), CUFFT_FORWARD)?;
        self.sampled = jobs.len();
        Ok(())
    }

    fn upload_slots(&mut self, driver: &CudaDriver, slots: &[u32]) -> Result<(), CudaDriverError> {
        assert_eq!(slots.len(), self.sampled, "a slot for each job sampled");
        assert!(
            slots.iter().all(|&slot| (slot as usize) < self.slots),
            "slots within the {} reserved",
            self.slots
        );
        self.job_slots.upload(driver, bytes(slots))
    }

    /// Correlates each job last sampled with the filter in its slot of
    /// `slots`, and says where each response peaks. Uses the samples up.
    pub(crate) fn find(
        &mut self,
        driver: &CudaDriver,
        slots: &[u32],
        regulariser: f32,
    ) -> Result<Vec<DcfPeak>, CudaDriverError> {
        if self.sampled == 0 {
            return Ok(Vec::new());
        }
        self.upload_slots(driver, slots)?;
        let count = self.sampled;
        let mut spectra = self.spectra.pointer;
        let mut job_slots = self.job_slots.pointer;
        let mut numerator = self.numerator.pointer;
        let mut denominator = self.denominator.pointer;
        let mut total = (count * CELLS) as u32;
        let mut lambda = regulariser;
        driver.with_context(|| {
            let mut params: Vec<*mut c_void> = vec![
                arg(&mut spectra),
                arg(&mut job_slots),
                arg(&mut numerator),
                arg(&mut denominator),
                arg(&mut total),
                arg(&mut lambda),
            ];
            // SAFETY: one pointer per parameter `dcf_correlate` declares, in
            // that order, each at a live local; it touches `total` values of
            // `spectra` and, through slots `upload_slots` checked, the filters'
            // reserved slots; the context is current.
            unsafe { launch(self.correlate, ((count * 16) as u32, 1), 16, &mut params) }
        })?;
        self.transform(driver, count, CUFFT_INVERSE)?;
        let mut peaks_out = self.peaks.pointer;
        driver.with_context(|| {
            let mut params: Vec<*mut c_void> = vec![arg(&mut spectra), arg(&mut peaks_out)];
            // SAFETY: one pointer per parameter `dcf_peak` declares, each at a
            // live local; it reads `count` responses and writes four floats
            // each into `peaks`, which holds as many as `spectra`.
            unsafe { launch(self.peak, (count as u32, 1), 16, &mut params) }
        })?;
        let mut found = vec![[0f32; 4]; count];
        // SAFETY: `[f32; 4]` is plain data, every byte of which the copy writes.
        let raw =
            unsafe { std::slice::from_raw_parts_mut(found.as_mut_ptr().cast::<u8>(), count * 16) };
        self.peaks.download(driver, raw)?;
        self.sampled = 0;
        Ok(found
            .into_iter()
            .map(|[x, y, top, psr]| DcfPeak { x, y, top, psr })
            .collect())
    }

    /// Takes each job last sampled into the filter in its slot of `slots`,
    /// at `rate`; no two jobs may share a slot. Leaves the samples as they
    /// were.
    pub(crate) fn learn(
        &mut self,
        driver: &CudaDriver,
        slots: &[u32],
        rate: f32,
    ) -> Result<(), CudaDriverError> {
        if self.sampled == 0 {
            return Ok(());
        }
        self.upload_slots(driver, slots)?;
        let count = self.sampled;
        let mut spectra = self.spectra.pointer;
        let mut job_slots = self.job_slots.pointer;
        let mut numerator = self.numerator.pointer;
        let mut denominator = self.denominator.pointer;
        let mut target = self.target.pointer;
        let mut total = (count * CELLS) as u32;
        let mut rate = rate;
        driver.with_context(|| {
            let mut params: Vec<*mut c_void> = vec![
                arg(&mut spectra),
                arg(&mut job_slots),
                arg(&mut numerator),
                arg(&mut denominator),
                arg(&mut target),
                arg(&mut total),
                arg(&mut rate),
            ];
            // SAFETY: one pointer per parameter `dcf_learn` declares, in that
            // order, each at a live local; it reads `total` values of `spectra`
            // and the target, and writes the checked slots of the filters; the
            // context is current.
            unsafe { launch(self.learn, ((count * 16) as u32, 1), 16, &mut params) }
        })
    }
}
