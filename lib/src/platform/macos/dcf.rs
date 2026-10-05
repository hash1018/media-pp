//! What `ObjectTracker` asks of Metal to follow objects by how they look on
//! VideoToolbox pictures — see `shaders/metal/dcf.metal`: its kernels, and
//! the memory every object's correlation filter and every picture's
//! samples live in, shared by the CPU and GPU. The Metal counterpart of
//! `platform::cuda::driver::CudaDcf`, with the transform written in the
//! shader where CUDA's is cuFFT's: Metal ships with macOS, so nothing more
//! is linked.

use objc2_metal::{MTLBuffer, MTLPixelFormat, MTLTextureUsage};

use crate::ffmpeg;

use super::metal::{Buffer, Kernel, MetalError, MetalGpu, Pass, Texture};
use super::pixel_buffer::PixelBuffer;
use super::videotoolbox::sw_format_of;

const SHADER: &str = include_str!("../../shaders/metal/dcf.metal");

/// The side of a neighbourhood's square of samples.
const SIDE: usize = 64;
/// Samples in one.
const CELLS: usize = SIDE * SIDE;
/// Threads in a threadgroup that reduces over one neighbourhood.
const REDUCING: usize = 256;
/// How many batches of jobs, and of their slots, one pass may be asked of
/// before it is run: each is written where the GPU reads it when the pass
/// runs, so none may be written over before then.
const BATCHES: usize = 8;

/// A VideoToolbox picture whose brightness the kernels read: its pixel
/// buffer, held, and whether it is BGRA rather than NV12.
pub(crate) struct MetalDcfSource {
    buffer: PixelBuffer,
    bgra: bool,
    size: (u32, u32),
}

impl MetalDcfSource {
    /// `frame`, where it is an NV12 or BGRA VideoToolbox picture no larger
    /// than its pixel buffer.
    pub(crate) fn of(frame: &ffmpeg::frame::Video) -> Option<Self> {
        let bgra = match sw_format_of(frame).ok()? {
            ffmpeg::format::Pixel::NV12 => false,
            ffmpeg::format::Pixel::BGRA => true,
            _ => return None,
        };
        let buffer = PixelBuffer::of_frame(frame)?;
        let size = (frame.width(), frame.height());
        let surface = buffer.size();
        (size.0 <= surface.0 && size.1 <= surface.1 && size.0 > 0 && size.1 > 0).then_some(Self {
            buffer,
            bgra,
            size,
        })
    }
}

/// The kernels, and the memory of every followed object's filter and of one
/// batch of samples, all in memory the CPU and GPU share.
///
/// What is asked of it is encoded into one pass, run when an answer is
/// needed — [`Self::find`] — or when [`Self::finish`] says the picture is
/// about to be let go of: a round trip to the GPU costs more than the
/// sampling and learning of a few objects, and a picture the tracker learns
/// from on a detection is otherwise nine of them.
pub(crate) struct MetalDcf {
    gpu: MetalGpu,
    sample_nv12: Kernel,
    sample_bgra: Kernel,
    normalize: Kernel,
    fft: Kernel,
    correlate: Kernel,
    peak: Kernel,
    learn: Kernel,
    window: Buffer,
    target: Buffer,
    twiddles: Buffer,
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
    /// What has been asked and not yet run.
    pending: Option<Pass>,
    /// How many jobs, and how many slots, the pending pass has written.
    jobs_used: usize,
    slots_used: usize,
}

// SAFETY: Metal's device, pipelines and buffers are thread-safe objects;
// the buffers are written by the CPU only between passes this waits for, by
// the one thread tracking at a time, through `&mut self`.
unsafe impl Send for MetalDcf {}

/// A buffer of `bytes` bytes holding `values`, the rest zero.
fn filled<T: Copy>(gpu: &MetalGpu, values: &[T], bytes: usize) -> Result<Buffer, MetalError> {
    let buffer = gpu.shared_buffer(bytes.max(16))?;
    // SAFETY: a fresh shared buffer of at least `bytes` bytes, no pass using
    // it; `values` is plain data no longer than that, checked by the callers'
    // sizes, and the rest is zeroed.
    unsafe {
        let base = buffer.contents().as_ptr().cast::<u8>();
        std::ptr::write_bytes(base, 0, bytes.max(16));
        std::ptr::copy_nonoverlapping(
            values.as_ptr().cast::<u8>(),
            base,
            std::mem::size_of_val(values),
        );
    }
    Ok(buffer)
}

/// `values` written `at` bytes into `buffer`, which holds that many more,
/// in a part no pass encoded so far reads.
fn write<T: Copy>(buffer: &Buffer, at: usize, values: &[T]) {
    assert!(at + std::mem::size_of_val(values) <= buffer.length());
    // SAFETY: the buffer is shared memory long enough, checked above, and the
    // part written is one no encoded pass reads — the callers write each
    // batch past the last until the pass has run.
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr().cast::<u8>(),
            buffer.contents().as_ptr().cast::<u8>().add(at),
            std::mem::size_of_val(values),
        );
    }
}

/// The kernel parameters `words` as the bytes the shader reads.
fn words(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_ne_bytes()).collect()
}

impl MetalDcf {
    /// The kernels compiled, and `window` and the wanted response's spectrum
    /// `target`, both `CELLS` long, put where the GPU reads them.
    pub(crate) fn new(window: &[f32], target: &[[f32; 2]]) -> Result<Self, MetalError> {
        assert_eq!(window.len(), CELLS);
        assert_eq!(target.len(), CELLS);
        let gpu = MetalGpu::new()?;
        let [
            sample_nv12,
            sample_bgra,
            normalize,
            fft,
            correlate,
            peak,
            learn,
        ] = <[Kernel; 7]>::try_from(gpu.kernels(
            SHADER,
            &[
                "dcf_sample_nv12",
                "dcf_sample_bgra",
                "dcf_normalize",
                "dcf_fft",
                "dcf_correlate",
                "dcf_peak",
                "dcf_learn",
            ],
        )?)
        .unwrap_or_else(|_| unreachable!("seven kernels for seven names"));
        let twiddles: Vec<[f32; 2]> = (0..SIDE / 2)
            .map(|k| {
                let angle = -2.0 * std::f64::consts::PI * k as f64 / SIDE as f64;
                [angle.cos() as f32, angle.sin() as f32]
            })
            .collect();
        Ok(Self {
            window: filled(&gpu, window, CELLS * 4)?,
            target: filled(&gpu, target, CELLS * 8)?,
            twiddles: filled(&gpu, &twiddles, SIDE / 2 * 8)?,
            numerator: filled::<u8>(&gpu, &[], 0)?,
            denominator: filled::<u8>(&gpu, &[], 0)?,
            slots: 0,
            spectra: filled::<u8>(&gpu, &[], 0)?,
            jobs: filled::<u8>(&gpu, &[], 0)?,
            job_slots: filled::<u8>(&gpu, &[], 0)?,
            peaks: filled::<u8>(&gpu, &[], 0)?,
            capacity: 0,
            sampled: 0,
            pending: None,
            jobs_used: 0,
            slots_used: 0,
            gpu,
            sample_nv12,
            sample_bgra,
            normalize,
            fft,
            correlate,
            peak,
            learn,
        })
    }

    /// Runs what has been asked and waits for it: then the picture it
    /// sampled may be let go of, and every filter holds what it learned.
    pub(crate) fn finish(&mut self) -> Result<(), MetalError> {
        self.jobs_used = 0;
        self.slots_used = 0;
        match self.pending.take() {
            Some(pass) => pass.finish(),
            None => Ok(()),
        }
    }

    /// The pending pass, begun where there is none.
    fn pass(&mut self) -> Result<&mut Pass, MetalError> {
        if self.pending.is_none() {
            self.pending = Some(self.gpu.pass()?);
        }
        Ok(self.pending.as_mut().expect("begun above"))
    }

    /// Room for filters in `slots` slots, keeping those already learned.
    pub(crate) fn reserve_slots(&mut self, slots: usize) -> Result<(), MetalError> {
        if slots <= self.slots {
            return Ok(());
        }
        // The filters are copied by the CPU, after what the GPU was asked.
        self.finish()?;
        let grown = slots.next_power_of_two().max(16);
        let numerator = self.gpu.shared_buffer(grown * CELLS * 8)?;
        let denominator = self.gpu.shared_buffer(grown * CELLS * 4)?;
        // SAFETY: both new buffers are shared memory of `grown` slots, the old
        // ones of `self.slots`, fewer; no pass is using any of them. The kept
        // slots are copied, the rest zeroed.
        unsafe {
            for (to, from, cell) in [
                (&numerator, &self.numerator, 8),
                (&denominator, &self.denominator, 4),
            ] {
                let kept = self.slots * CELLS * cell;
                let base = to.contents().as_ptr().cast::<u8>();
                std::ptr::copy_nonoverlapping(from.contents().as_ptr().cast::<u8>(), base, kept);
                std::ptr::write_bytes(base.add(kept), 0, grown * CELLS * cell - kept);
            }
        }
        self.numerator = numerator;
        self.denominator = denominator;
        self.slots = grown;
        Ok(())
    }

    fn reserve_jobs(&mut self, jobs: usize) -> Result<(), MetalError> {
        if jobs <= self.capacity {
            return Ok(());
        }
        // The samples of what is pending are in the buffers being replaced.
        self.finish()?;
        let grown = jobs.next_power_of_two().max(16);
        self.spectra = self.gpu.shared_buffer(grown * CELLS * 8)?;
        self.jobs = self.gpu.shared_buffer(BATCHES * grown * 16)?;
        self.job_slots = self.gpu.shared_buffer(BATCHES * grown * 4)?;
        self.peaks = self.gpu.shared_buffer(grown * 16)?;
        self.capacity = grown;
        Ok(())
    }

    /// Encodes the transform of the first `count` neighbourhoods of
    /// `spectra`, rows then columns.
    fn transform(
        pass: &mut Pass,
        fft: &Kernel,
        spectra: &Buffer,
        twiddles: &Buffer,
        count: usize,
        inverse: bool,
    ) {
        for columns in [0, 1] {
            pass.dispatch_groups(
                fft,
                &[],
                &[(spectra, 0), (twiddles, 0)],
                Some(&words(&[count as u32, columns, u32::from(inverse)])),
                (1, count, 1),
                (SIDE, 1, 1),
            );
        }
    }

    /// Samples each of `jobs` — centre and size in pixels — from `source`,
    /// readies the samples to be correlated, and transforms them, for
    /// [`Self::find`] or [`Self::learn`] to use.
    pub(crate) fn sample(
        &mut self,
        source: &MetalDcfSource,
        jobs: &[[f32; 4]],
    ) -> Result<(), MetalError> {
        self.sampled = 0;
        if jobs.is_empty() {
            return Ok(());
        }
        self.reserve_jobs(jobs.len())?;
        if self.jobs_used + jobs.len() > BATCHES * self.capacity {
            self.finish()?;
        }
        let at = self.jobs_used * 16;
        write(&self.jobs, at, jobs);
        self.jobs_used += jobs.len();
        let read = MTLTextureUsage::ShaderRead;
        let (kernel, picture): (&Kernel, Texture) = if source.bgra {
            (
                &self.sample_bgra,
                self.gpu
                    .plane(&source.buffer, 0, MTLPixelFormat::BGRA8Unorm, read)?,
            )
        } else {
            (
                &self.sample_nv12,
                self.gpu
                    .plane(&source.buffer, 0, MTLPixelFormat::R8Unorm, read)?,
            )
        };
        let count = jobs.len();
        let kernel = kernel.clone();
        let (jobs_buffer, spectra, window, normalize, fft, twiddles) = (
            self.jobs.clone(),
            self.spectra.clone(),
            self.window.clone(),
            self.normalize.clone(),
            self.fft.clone(),
            self.twiddles.clone(),
        );
        let pass = self.pass()?;
        pass.dispatch_groups(
            &kernel,
            &[&picture],
            &[(&jobs_buffer, at), (&spectra, 0)],
            Some(&words(&[source.size.0, source.size.1, count as u32, 0])),
            (SIDE / 16, SIDE / 16, count),
            (16, 16, 1),
        );
        pass.dispatch_groups(
            &normalize,
            &[],
            &[(&spectra, 0), (&window, 0)],
            None,
            (count, 1, 1),
            (REDUCING, 1, 1),
        );
        Self::transform(pass, &fft, &spectra, &twiddles, count, false);
        self.sampled = count;
        Ok(())
    }

    /// `slots`, one for each job sampled, written where the pending pass
    /// reads them: their byte offset.
    fn write_slots(&mut self, slots: &[u32]) -> Result<usize, MetalError> {
        assert_eq!(slots.len(), self.sampled, "a slot for each job sampled");
        assert!(
            slots.iter().all(|&slot| (slot as usize) < self.slots),
            "slots within the {} reserved",
            self.slots
        );
        if self.slots_used + slots.len() > BATCHES * self.capacity {
            // Every batch of slots follows a batch of jobs, which would have
            // run the pass first.
            unreachable!("no more slots than jobs");
        }
        let at = self.slots_used * 4;
        write(&self.job_slots, at, slots);
        self.slots_used += slots.len();
        Ok(at)
    }

    /// Correlates each job last sampled with the filter in its slot of
    /// `slots`, and says where each response peaks: `x, y, top, psr`, the
    /// peak in samples from the neighbourhood's corner. Uses the samples up.
    pub(crate) fn find(
        &mut self,
        slots: &[u32],
        regulariser: f32,
    ) -> Result<Vec<[f32; 4]>, MetalError> {
        if self.sampled == 0 {
            return Ok(Vec::new());
        }
        let at = self.write_slots(slots)?;
        let count = self.sampled;
        let total = count * CELLS;
        let mut parameters = words(&[total as u32]);
        parameters.extend(regulariser.to_ne_bytes());
        let (correlate, peak, fft) = (self.correlate.clone(), self.peak.clone(), self.fft.clone());
        let (spectra, job_slots, numerator, denominator, peaks, twiddles) = (
            self.spectra.clone(),
            self.job_slots.clone(),
            self.numerator.clone(),
            self.denominator.clone(),
            self.peaks.clone(),
            self.twiddles.clone(),
        );
        let pass = self.pass()?;
        pass.dispatch_groups(
            &correlate,
            &[],
            &[
                (&spectra, 0),
                (&job_slots, at),
                (&numerator, 0),
                (&denominator, 0),
            ],
            Some(&parameters),
            (total.div_ceil(REDUCING), 1, 1),
            (REDUCING, 1, 1),
        );
        Self::transform(pass, &fft, &spectra, &twiddles, count, true);
        pass.dispatch_groups(
            &peak,
            &[],
            &[(&spectra, 0), (&peaks, 0)],
            None,
            (count, 1, 1),
            (REDUCING, 1, 1),
        );
        // The answer is wanted now.
        self.finish()?;
        self.sampled = 0;
        // SAFETY: the pass that wrote `count` peaks of four floats into the
        // shared buffer has finished, and nothing writes it until the next.
        let peaks = unsafe {
            std::slice::from_raw_parts(self.peaks.contents().as_ptr().cast::<[f32; 4]>(), count)
        };
        Ok(peaks.to_vec())
    }

    /// Takes each job last sampled into the filter in its slot of `slots`,
    /// at `rate`; no two jobs may share a slot. Leaves the samples as they
    /// were. Asked, not yet run — see [`Self::finish`].
    pub(crate) fn learn(&mut self, slots: &[u32], rate: f32) -> Result<(), MetalError> {
        if self.sampled == 0 {
            return Ok(());
        }
        let at = self.write_slots(slots)?;
        let total = self.sampled * CELLS;
        let mut parameters = words(&[total as u32]);
        parameters.extend(rate.to_ne_bytes());
        let learn = self.learn.clone();
        let (spectra, job_slots, numerator, denominator, target) = (
            self.spectra.clone(),
            self.job_slots.clone(),
            self.numerator.clone(),
            self.denominator.clone(),
            self.target.clone(),
        );
        self.pass()?.dispatch_groups(
            &learn,
            &[],
            &[
                (&spectra, 0),
                (&job_slots, at),
                (&numerator, 0),
                (&denominator, 0),
                (&target, 0),
            ],
            Some(&parameters),
            (total.div_ceil(REDUCING), 1, 1),
            (REDUCING, 1, 1),
        );
        Ok(())
    }
}
