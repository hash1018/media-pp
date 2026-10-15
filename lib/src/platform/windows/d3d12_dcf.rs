//! What `ObjectTracker` asks of D3D12 to follow objects by how they look on
//! D3D12 pictures — see `shaders/d3d12/dcf.hlsl`: its kernels, and the
//! buffers every object's correlation filter and every picture's samples
//! live in. The D3D12 counterpart of `platform::macos::dcf::MetalDcf`, the
//! transform written in the shader as Metal's is.

use windows::{
    Win32::{
        Foundation::E_INVALIDARG,
        Graphics::{
            Direct3D12::*,
            Dxgi::Common::{
                DXGI_FORMAT, DXGI_FORMAT_NV12, DXGI_FORMAT_P010, DXGI_FORMAT_R8_UNORM,
                DXGI_FORMAT_R16_UNORM,
            },
        },
    },
    core::{Interface, s},
};

use crate::ffmpeg;

use super::d3d12_compute::{Commands, Kernel, buffer, plane_view, transition, uav_barrier};
use super::d3d12va::d3d12va_texture;

const SHADER: &[u8] = include_bytes!("../../shaders/d3d12/dcf.hlsl");

/// The side of a neighbourhood's square of samples.
const SIDE: usize = 64;
/// Samples in one.
const CELLS: usize = SIDE * SIDE;
/// Threads in a group that reduces over one neighbourhood.
const REDUCING: usize = 256;
/// How many batches of jobs, and of their slots, may be asked for before
/// what was asked is run: each is written where the GPU reads it when it
/// runs, so none may be written over before then.
const BATCHES: usize = 8;
/// How many pictures' views the heap holds before what was asked is run.
const VIEWS: usize = 16;

/// A D3D12 picture whose luma the kernels read: its texture and the fence
/// that says it is written, the view format of its luma plane, and its
/// size.
pub(crate) struct D3d12DcfSource {
    texture: ID3D12Resource,
    fence: ID3D12Fence,
    fence_value: u64,
    luma: DXGI_FORMAT,
    size: (u32, u32),
}

// SAFETY: free-threaded COM objects, only read through `&self`.
unsafe impl Send for D3d12DcfSource {}

impl D3d12DcfSource {
    /// `frame`, where it is an NV12 or P010 D3D12 picture no larger than
    /// its texture.
    pub(crate) fn of(frame: &ffmpeg::frame::Video) -> Option<Self> {
        let (texture, fence, fence_value) = d3d12va_texture(frame)
            .filter(|(texture, fence, _)| !texture.is_null() && !fence.is_null())?;
        // SAFETY: the live frame owns these non-null objects; cloning the
        // borrowed wrappers takes references of the frame's own.
        let (texture, fence) = unsafe {
            (
                ID3D12Resource::from_raw_borrowed(&texture)?.clone(),
                ID3D12Fence::from_raw_borrowed(&fence)?.clone(),
            )
        };
        // SAFETY: a live resource's plain description, returned by value.
        let desc = unsafe { texture.GetDesc() };
        let luma = match desc.Format {
            DXGI_FORMAT_NV12 => DXGI_FORMAT_R8_UNORM,
            DXGI_FORMAT_P010 => DXGI_FORMAT_R16_UNORM,
            _ => return None,
        };
        let size = (frame.width(), frame.height());
        (size.0 > 0 && size.1 > 0 && size.0 <= desc.Width as u32 && size.1 <= desc.Height)
            .then_some(Self {
                texture,
                fence,
                fence_value,
                luma,
                size,
            })
    }

    /// The device the picture is on.
    fn device(&self) -> windows::core::Result<ID3D12Device> {
        let mut device: Option<ID3D12Device> = None;
        // SAFETY: a live resource and a correctly typed out-parameter.
        unsafe { self.texture.GetDevice(&mut device) }?;
        device.ok_or_else(|| windows::core::Error::from(E_INVALIDARG))
    }
}

/// The kernels by name, all under one root signature.
struct Kernels {
    sample: Kernel,
    normalize: ID3D12PipelineState,
    fft: ID3D12PipelineState,
    correlate: ID3D12PipelineState,
    peak: ID3D12PipelineState,
    learn: ID3D12PipelineState,
}

/// The kernels, and the buffers of every followed object's filter and of
/// one batch of samples, on the device of the pictures followed, with a
/// queue of its own.
///
/// What is asked of it is recorded into one list, run when an answer is
/// needed — [`Self::find`] — or when [`Self::finish`] says the picture is
/// about to be let go of: a round trip to the GPU costs more than the
/// sampling and learning of a few objects.
pub(crate) struct D3d12Dcf {
    device: ID3D12Device,
    kernels: Kernels,
    commands: Commands,
    /// The list what is asked is recorded into, open since the first ask.
    list: Option<ID3D12GraphicsCommandList>,
    /// What the GPU waits for before running it — whatever wrote each
    /// picture sampled — and the pictures, held until it has run.
    waits: Vec<(ID3D12Fence, u64)>,
    held: Vec<ID3D12Resource>,
    heap: ID3D12DescriptorHeap,
    heap_step: usize,
    views_used: usize,
    window: ID3D12Resource,
    target: ID3D12Resource,
    twiddles: ID3D12Resource,
    /// Each filter slot's numerator, `CELLS` complex values, and its
    /// denominator, `CELLS` real ones.
    numerator: ID3D12Resource,
    denominator: ID3D12Resource,
    slots: usize,
    /// One batch of samples, then spectra, then responses.
    spectra: ID3D12Resource,
    /// The jobs and their slots, written by the CPU in batches the GPU reads.
    jobs: ID3D12Resource,
    job_slots: ID3D12Resource,
    peaks: ID3D12Resource,
    readback: ID3D12Resource,
    capacity: usize,
    /// How many jobs `spectra` holds now.
    sampled: usize,
    /// How many jobs, and how many slots, the open list has written.
    jobs_used: usize,
    slots_used: usize,
}

// SAFETY: the D3D12 objects are free-threaded; the list, heap and the
// buffers the CPU writes are touched only through `&mut self`, by the one
// thread tracking at a time, between runs it waits for.
unsafe impl Send for D3d12Dcf {}

/// An upload buffer of `bytes` holding `values`, the rest zero.
fn uploaded<T: Copy>(
    device: &ID3D12Device,
    values: &[T],
    bytes: usize,
) -> windows::core::Result<ID3D12Resource> {
    // SAFETY: a live device; the buffer is mapped, written within its
    // length and unmapped before anything reads it.
    unsafe {
        let made = buffer(
            device,
            bytes.max(16) as u64,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let mut mapped = std::ptr::null_mut();
        made.Map(0, None, Some(&mut mapped))?;
        std::ptr::write_bytes(mapped.cast::<u8>(), 0, bytes.max(16));
        std::ptr::copy_nonoverlapping(
            values.as_ptr().cast::<u8>(),
            mapped.cast::<u8>(),
            std::mem::size_of_val(values),
        );
        made.Unmap(0, None);
        Ok(made)
    }
}

/// `values` written `at` bytes into the upload buffer `buffer`.
fn write<T: Copy>(buffer: &ID3D12Resource, at: usize, values: &[T]) -> windows::core::Result<()> {
    // SAFETY: an upload buffer the callers size for every batch, written in
    // a part no recorded work reads until it runs; mapped for the copy.
    unsafe {
        assert!(at as u64 + std::mem::size_of_val(values) as u64 <= buffer.GetDesc().Width);
        let mut mapped = std::ptr::null_mut();
        buffer.Map(0, None, Some(&mut mapped))?;
        std::ptr::copy_nonoverlapping(
            values.as_ptr().cast::<u8>(),
            mapped.cast::<u8>().add(at),
            std::mem::size_of_val(values),
        );
        buffer.Unmap(0, None);
    }
    Ok(())
}

/// A buffer of `bytes` the shaders write, zeroed as a committed resource
/// is made, in `UNORDERED_ACCESS`.
fn writable(device: &ID3D12Device, bytes: usize) -> windows::core::Result<ID3D12Resource> {
    // SAFETY: a live device.
    unsafe {
        buffer(
            device,
            bytes.max(16) as u64,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )
    }
}

impl D3d12Dcf {
    /// The kernels compiled on `source`'s device, and `window` and the
    /// wanted response's spectrum `target`, both `CELLS` long, put where the
    /// GPU reads them.
    pub(crate) fn new(
        window: &[f32],
        target: &[[f32; 2]],
        source: &D3d12DcfSource,
    ) -> windows::core::Result<Self> {
        assert_eq!(window.len(), CELLS);
        assert_eq!(target.len(), CELLS);
        let device = source.device()?;
        // SAFETY: a live device and descriptions built here.
        let (kernels, queue, heap, heap_step) = unsafe {
            let sample = Kernel::new(&device, SHADER, s!("dcf.hlsl"), s!("dcf_sample"))?;
            let sibling = |entry| sample.sibling(&device, SHADER, s!("dcf.hlsl"), entry);
            let (normalize, fft, correlate, peak, learn) = (
                sibling(s!("dcf_normalize"))?,
                sibling(s!("dcf_fft"))?,
                sibling(s!("dcf_correlate"))?,
                sibling(s!("dcf_peak"))?,
                sibling(s!("dcf_learn"))?,
            );
            let queue: ID3D12CommandQueue =
                device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })?;
            let heap: ID3D12DescriptorHeap =
                device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: VIEWS as u32,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })?;
            let heap_step = device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
                as usize;
            (
                Kernels {
                    sample,
                    normalize,
                    fft,
                    correlate,
                    peak,
                    learn,
                },
                queue,
                heap,
                heap_step,
            )
        };
        let twiddles: Vec<[f32; 2]> = (0..SIDE / 2)
            .map(|k| {
                let angle = -2.0 * std::f64::consts::PI * k as f64 / SIDE as f64;
                [angle.cos() as f32, angle.sin() as f32]
            })
            .collect();
        Ok(Self {
            commands: Commands::new(&device, &queue)?,
            list: None,
            waits: Vec::new(),
            held: Vec::new(),
            heap,
            heap_step,
            views_used: 0,
            window: uploaded(&device, window, CELLS * 4)?,
            target: uploaded(&device, target, CELLS * 8)?,
            twiddles: uploaded(&device, &twiddles, SIDE / 2 * 8)?,
            numerator: writable(&device, 16)?,
            denominator: writable(&device, 16)?,
            slots: 0,
            spectra: writable(&device, 16)?,
            jobs: uploaded::<u8>(&device, &[], 16)?,
            job_slots: uploaded::<u8>(&device, &[], 16)?,
            peaks: writable(&device, 16)?,
            // SAFETY: a live device.
            readback: unsafe {
                buffer(
                    &device,
                    16,
                    D3D12_HEAP_TYPE_READBACK,
                    D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                )?
            },
            capacity: 0,
            sampled: 0,
            jobs_used: 0,
            slots_used: 0,
            kernels,
            device,
        })
    }

    /// Runs what has been asked and waits for it: then the pictures it
    /// sampled may be let go of, and every filter holds what it learned.
    pub(crate) fn finish(&mut self) -> windows::core::Result<()> {
        self.jobs_used = 0;
        self.slots_used = 0;
        self.views_used = 0;
        if self.list.take().is_some() {
            let waits: Vec<(&ID3D12Fence, u64)> = self
                .waits
                .iter()
                .map(|(fence, value)| (fence, *value))
                .collect();
            let submitted = self.commands.submit(&waits, &[]);
            let waited = self.commands.wait_idle();
            self.waits.clear();
            self.held.clear();
            submitted?;
            waited?;
        }
        Ok(())
    }

    /// The open list, begun where there is none, its root signature, heap
    /// and every root view set.
    fn list(&mut self) -> windows::core::Result<ID3D12GraphicsCommandList> {
        if let Some(list) = &self.list {
            return Ok(list.clone());
        }
        let list = self.commands.begin(None)?;
        // SAFETY: the list is open, and every buffer named lives as long as
        // this, which waits for the list before replacing any.
        unsafe {
            list.SetComputeRootSignature(&self.kernels.sample.root_signature);
            list.SetDescriptorHeaps(&[Some(self.heap.clone())]);
            list.SetComputeRootShaderResourceView(2, self.jobs.GetGPUVirtualAddress());
            list.SetComputeRootShaderResourceView(3, self.job_slots.GetGPUVirtualAddress());
            list.SetComputeRootShaderResourceView(4, self.window.GetGPUVirtualAddress());
            list.SetComputeRootShaderResourceView(5, self.target.GetGPUVirtualAddress());
            list.SetComputeRootShaderResourceView(6, self.twiddles.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(7, self.spectra.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(8, self.numerator.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(9, self.denominator.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(10, self.peaks.GetGPUVirtualAddress());
            list.SetComputeRootDescriptorTable(1, self.heap.GetGPUDescriptorHandleForHeapStart());
        }
        self.list = Some(list.clone());
        Ok(list)
    }

    /// Room for filters in `slots` slots, keeping those already learned.
    pub(crate) fn reserve_slots(&mut self, slots: usize) -> windows::core::Result<()> {
        if slots <= self.slots {
            return Ok(());
        }
        // The filters are copied after what was asked of them.
        self.finish()?;
        let grown = slots.next_power_of_two().max(16);
        let numerator = writable(&self.device, grown * CELLS * 8)?;
        let denominator = writable(&self.device, grown * CELLS * 4)?;
        if self.slots > 0 {
            let list = self.commands.begin(None)?;
            // SAFETY: the list is open and the GPU idle on these buffers;
            // the kept slots are copied, the rest left zero as made.
            unsafe {
                for (to, from, cell) in [
                    (&numerator, &self.numerator, 8),
                    (&denominator, &self.denominator, 4),
                ] {
                    transition(
                        &list,
                        from,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    );
                    transition(
                        &list,
                        to,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                    );
                    list.CopyBufferRegion(to, 0, from, 0, (self.slots * CELLS * cell) as u64);
                    transition(
                        &list,
                        to,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    );
                }
            }
            self.commands.submit(&[], &[])?;
            self.commands.wait_idle()?;
        }
        self.numerator = numerator;
        self.denominator = denominator;
        self.slots = grown;
        Ok(())
    }

    fn reserve_jobs(&mut self, jobs: usize) -> windows::core::Result<()> {
        if jobs <= self.capacity {
            return Ok(());
        }
        // The samples of what is asked are in the buffers being replaced.
        self.finish()?;
        let grown = jobs.next_power_of_two().max(16);
        self.spectra = writable(&self.device, grown * CELLS * 8)?;
        self.jobs = uploaded::<u8>(&self.device, &[], BATCHES * grown * 16)?;
        self.job_slots = uploaded::<u8>(&self.device, &[], BATCHES * grown * 4)?;
        self.peaks = writable(&self.device, grown * 16)?;
        // SAFETY: a live device.
        self.readback = unsafe {
            buffer(
                &self.device,
                (grown * 16) as u64,
                D3D12_HEAP_TYPE_READBACK,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )?
        };
        self.capacity = grown;
        Ok(())
    }

    /// Records the transform of the first `count` neighbourhoods of the
    /// spectra, rows then columns.
    fn transform(&self, list: &ID3D12GraphicsCommandList, count: usize, inverse: bool) {
        // SAFETY: the open list, its root signature and views set.
        unsafe {
            list.SetPipelineState(&self.kernels.fft);
            for columns in [0u32, 1] {
                let constants = [count as u32, columns, u32::from(inverse), 0];
                list.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
                list.Dispatch(1, count as u32, 1);
                uav_barrier(list, &self.spectra);
            }
        }
    }

    /// Samples each of `jobs` — centre and size in pixels — from `source`,
    /// readies the samples to be correlated, and transforms them, for
    /// [`Self::find`] or [`Self::learn`] to use.
    pub(crate) fn sample(
        &mut self,
        source: &D3d12DcfSource,
        jobs: &[[f32; 4]],
    ) -> windows::core::Result<()> {
        self.sampled = 0;
        if jobs.is_empty() {
            return Ok(());
        }
        if source.device()?.as_raw() != self.device.as_raw() {
            return Err(windows::core::Error::new(
                E_INVALIDARG,
                "a picture of another D3D12 device than the filters'",
            ));
        }
        self.reserve_jobs(jobs.len())?;
        if self.jobs_used + jobs.len() > BATCHES * self.capacity || self.views_used == VIEWS {
            self.finish()?;
        }
        let at = self.jobs_used * 16;
        write(&self.jobs, at, jobs)?;
        self.jobs_used += jobs.len();
        let view = self.views_used;
        self.views_used += 1;
        // SAFETY: a live heap and texture, the view inside the heap and not
        // read by anything recorded but unrun — each sample takes the next.
        unsafe {
            let start = self.heap.GetCPUDescriptorHandleForHeapStart();
            self.device.CreateShaderResourceView(
                &source.texture,
                Some(&plane_view(source.luma, 0)),
                D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: start.ptr + view * self.heap_step,
                },
            );
        }
        self.waits.push((source.fence.clone(), source.fence_value));
        self.held.push(source.texture.clone());
        let count = jobs.len();
        let list = self.list()?;
        // SAFETY: the open list, its root signature and views set; the
        // picture is held until the list has run.
        unsafe {
            let gpu_start = self.heap.GetGPUDescriptorHandleForHeapStart();
            list.SetComputeRootDescriptorTable(
                1,
                D3D12_GPU_DESCRIPTOR_HANDLE {
                    ptr: gpu_start.ptr + (view * self.heap_step) as u64,
                },
            );
            list.SetComputeRootShaderResourceView(2, self.jobs.GetGPUVirtualAddress() + at as u64);
            transition(
                &list,
                &source.texture,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            );
            list.SetPipelineState(&self.kernels.sample.pipeline);
            let constants = [source.size.0, source.size.1, count as u32, 0];
            list.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
            list.Dispatch((SIDE / 16) as u32, (SIDE / 16) as u32, count as u32);
            transition(
                &list,
                &source.texture,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            );
            uav_barrier(&list, &self.spectra);
            list.SetPipelineState(&self.kernels.normalize);
            list.Dispatch(count as u32, 1, 1);
            uav_barrier(&list, &self.spectra);
        }
        self.transform(&list, count, false);
        self.sampled = count;
        Ok(())
    }

    /// `slots`, one for each job sampled, written where the open list reads
    /// them, and that list told to read them there.
    fn write_slots(
        &mut self,
        list: &ID3D12GraphicsCommandList,
        slots: &[u32],
    ) -> windows::core::Result<()> {
        assert_eq!(slots.len(), self.sampled, "a slot for each job sampled");
        assert!(
            slots.iter().all(|&slot| (slot as usize) < self.slots),
            "slots within the {} reserved",
            self.slots
        );
        // Every batch of slots follows a batch of jobs, which would have run
        // the list first.
        assert!(self.slots_used + slots.len() <= BATCHES * self.capacity);
        let at = self.slots_used * 4;
        write(&self.job_slots, at, slots)?;
        self.slots_used += slots.len();
        // SAFETY: the open list, its root signature set.
        unsafe {
            list.SetComputeRootShaderResourceView(
                3,
                self.job_slots.GetGPUVirtualAddress() + at as u64,
            );
        }
        Ok(())
    }

    /// Correlates each job last sampled with the filter in its slot of
    /// `slots`, and says where each response peaks: `x, y, top, psr`, the
    /// peak in samples from the neighbourhood's corner. Uses the samples up.
    pub(crate) fn find(
        &mut self,
        slots: &[u32],
        regulariser: f32,
    ) -> windows::core::Result<Vec<[f32; 4]>> {
        if self.sampled == 0 {
            return Ok(Vec::new());
        }
        let list = self.list()?;
        self.write_slots(&list, slots)?;
        let count = self.sampled;
        let total = (count * CELLS) as u32;
        // SAFETY: the open list, its root signature and views set.
        unsafe {
            list.SetPipelineState(&self.kernels.correlate);
            let constants = [total, regulariser.to_bits(), 0, 0];
            list.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
            list.Dispatch(total.div_ceil(REDUCING as u32), 1, 1);
            uav_barrier(&list, &self.spectra);
        }
        self.transform(&list, count, true);
        // SAFETY: as above; the peaks are copied out for the CPU once found.
        unsafe {
            list.SetPipelineState(&self.kernels.peak);
            list.Dispatch(count as u32, 1, 1);
            uav_barrier(&list, &self.peaks);
            transition(
                &list,
                &self.peaks,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            );
            list.CopyBufferRegion(&self.readback, 0, &self.peaks, 0, (count * 16) as u64);
            transition(
                &list,
                &self.peaks,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            );
        }
        // The answer is wanted now.
        self.finish()?;
        self.sampled = 0;
        // SAFETY: the read-back buffer holds `capacity` peaks, of which the
        // list just waited for wrote `count`; unmapped before this returns.
        unsafe {
            let mut mapped = std::ptr::null_mut();
            self.readback.Map(0, None, Some(&mut mapped))?;
            let peaks = std::slice::from_raw_parts(mapped.cast::<[f32; 4]>(), count).to_vec();
            self.readback.Unmap(0, None);
            Ok(peaks)
        }
    }

    /// Takes each job last sampled into the filter in its slot of `slots`,
    /// at `rate`; no two jobs may share a slot. Leaves the samples as they
    /// were. Asked, not yet run — see [`Self::finish`].
    pub(crate) fn learn(&mut self, slots: &[u32], rate: f32) -> windows::core::Result<()> {
        if self.sampled == 0 {
            return Ok(());
        }
        let list = self.list()?;
        self.write_slots(&list, slots)?;
        let total = (self.sampled * CELLS) as u32;
        // SAFETY: the open list, its root signature and views set.
        unsafe {
            list.SetPipelineState(&self.kernels.learn);
            let constants = [total, rate.to_bits(), 0, 0];
            list.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
            list.Dispatch(total.div_ceil(REDUCING as u32), 1, 1);
            uav_barrier(&list, &self.numerator);
            uav_barrier(&list, &self.denominator);
        }
        Ok(())
    }
}

impl Drop for D3d12Dcf {
    fn drop(&mut self) {
        // What was asked and not run is let go of; what runs is waited for
        // before its buffers go.
        let _ = self.finish();
    }
}
