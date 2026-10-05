//! Following by look on the GPU: the same correlation filters as
//! [`super::dcf`], every followed object's sampled, transformed and matched
//! at once on the device — CUDA pictures with the `cuda-visual-tracking`
//! feature (`platform::cuda::driver::CudaDcf`), VideoToolbox pictures with
//! `metal` (`platform::macos::dcf::MetalDcf`). Only four numbers an object
//! come back: where its response peaks, how high, and how sure.
//!
//! What every device shares — which slot each object's filter is in, the
//! jobs to sample and what is made of what comes back — is [`GpuLooks`];
//! what a device does is a [`DcfDevice`].

use std::collections::HashMap;

use ffmpeg_next as ffmpeg;

use super::byte_track::{ByteTrack, Expected};
use super::dcf::{LEARNING_RATE, REGULARISER, SCALES, SIZE, target, window, window_of};

/// The looks a new object's filter is first learned from, and the rate
/// each is taken in at — 1, a half, a third: their average, as the CPU's
/// filter learns its first.
const FIRST_LOOKS: [(f64, f32); 3] = [(0.95, 1.0), (1.0, 0.5), (1.05, 1.0 / 3.0)];

/// One neighbourhood to sample: its centre, width and height, in pixels.
pub(super) type DcfJob = [f32; 4];

/// A GPU's half of following by look: the filters, in slots, and the
/// kernels that sample a picture, correlate and learn.
pub(super) trait DcfDevice: Sized {
    /// Where a picture's brightness is for the device.
    type Source;
    /// Why the device would not.
    type Error: std::fmt::Display;

    /// The device, with the window and the wanted response on it.
    fn open(window: &[f32], target: &[[f32; 2]]) -> Result<Self, Self::Error>;
    /// What it is, for the log.
    fn describe(&self) -> String;
    /// Where `frame`'s brightness is, where the device reads it.
    fn source(frame: &ffmpeg::frame::Video) -> Option<Self::Source>;
    /// Waits for whatever wrote the picture, where that ran apart.
    fn ready(&self) -> Result<(), Self::Error>;
    /// Room for filters in `slots` slots, keeping those already learned.
    fn reserve_slots(&mut self, slots: usize) -> Result<(), Self::Error>;
    /// Samples each of `jobs` from `source`, readies and transforms them.
    fn sample(&mut self, source: &Self::Source, jobs: &[DcfJob]) -> Result<(), Self::Error>;
    /// Correlates each job sampled with the filter in its slot: where each
    /// response peaks — `x, y, top, psr`, in samples.
    fn find(&mut self, slots: &[u32], regulariser: f32) -> Result<Vec<[f32; 4]>, Self::Error>;
    /// Takes each job sampled into the filter in its slot, at `rate`.
    fn learn(&mut self, slots: &[u32], rate: f32) -> Result<(), Self::Error>;
    /// Has done what was asked of it, where it waits to be asked more: the
    /// picture may be let go of after.
    fn finish(&mut self) -> Result<(), Self::Error>;
}

/// The filters of the objects followed on one GPU.
pub(super) struct GpuLooks<D> {
    device: D,
    /// Each followed object's filter slot, and the box size it was last
    /// learned at.
    slots: HashMap<u64, (u32, (f64, f64))>,
    free: Vec<u32>,
    next: u32,
}

impl<D: DcfDevice> GpuLooks<D> {
    /// The device, with the window and wanted response on it.
    pub(super) fn open() -> Result<Self, D::Error> {
        let target: Vec<[f32; 2]> = target().iter().map(|c| [c.re, c.im]).collect();
        Ok(Self {
            device: D::open(window(), &target)?,
            slots: HashMap::new(),
            free: Vec::new(),
            next: 0,
        })
    }

    /// What the device is, for the log.
    pub(super) fn describe(&self) -> String {
        self.device.describe()
    }

    /// A slot for object `id`'s filter.
    fn slot(&mut self, id: u64) -> Result<u32, D::Error> {
        let slot = self.free.pop().unwrap_or_else(|| {
            self.next += 1;
            self.next - 1
        });
        self.device.reserve_slots(slot as usize + 1)?;
        self.slots.insert(id, (slot, (0.0, 0.0)));
        Ok(slot)
    }

    /// Learns how each of `boxes` looks, into the filters already slotted,
    /// at `rate`, each looked at `scale` times its window.
    fn learn(
        &mut self,
        source: &D::Source,
        boxes: &[(u64, [f64; 4])],
        scale: f64,
        rate: f32,
    ) -> Result<(), D::Error> {
        let mut jobs: Vec<DcfJob> = Vec::with_capacity(boxes.len());
        let mut slots = Vec::with_capacity(boxes.len());
        for (id, [x, y, w, h]) in boxes {
            let Some((slot, size)) = self.slots.get_mut(id) else {
                continue;
            };
            let (ww, wh) = window_of((*w, *h));
            jobs.push([
                (x + w / 2.0) as f32,
                (y + h / 2.0) as f32,
                (ww * scale) as f32,
                (wh * scale) as f32,
            ]);
            slots.push(*slot);
            *size = (*w, *h);
        }
        self.device.sample(source, &jobs)?;
        self.device.learn(&slots, rate)
    }

    /// Moves each followed object to where its filter finds it on the
    /// picture `source` is, where the filter is at least `min_psr` sure,
    /// weighed against the motion at `doubt`, and learns how it looks
    /// there: [`super::object_tracker`]'s `follow_by_look`, for every
    /// object at once.
    pub(super) fn follow(
        &mut self,
        tracker: &mut ByteTrack,
        source: &D::Source,
        min_psr: f32,
        doubt: f64,
    ) -> Result<Vec<Expected>, D::Error> {
        self.device.ready()?;
        let mut expected = tracker.expected();
        let mut followed = Vec::new();
        let mut jobs: Vec<DcfJob> = Vec::new();
        let mut slots = Vec::new();
        for (index, object) in expected.iter().enumerate() {
            let Some(&(slot, size)) = self.slots.get(&object.id) else {
                continue;
            };
            let [x, y, w, h] = object.tlwh;
            let (ww, wh) = window_of(size);
            for scale in SCALES {
                jobs.push([
                    (x + w / 2.0) as f32,
                    (y + h / 2.0) as f32,
                    (ww * scale) as f32,
                    (wh * scale) as f32,
                ]);
                slots.push(slot);
            }
            followed.push((index, size));
        }
        if followed.is_empty() {
            return Ok(expected);
        }
        self.device.sample(source, &jobs)?;
        let peaks = self.device.find(&slots, REGULARISER)?;

        let half = (SIZE / 2) as f32;
        let shift = |at: f32| {
            let d = at - half;
            if d > half { d - SIZE as f32 } else { d }
        };
        let mut corrected = Vec::new();
        for (n, (index, size)) in followed.into_iter().enumerate() {
            let object = &expected[index];
            let [x, y, w, h] = object.tlwh;
            let centre = (x + w / 2.0, y + h / 2.0);
            let (ww, wh) = window_of(size);
            let (scale, [peak_x, peak_y, _, psr]) = SCALES
                .iter()
                .zip(&peaks[n * SCALES.len()..(n + 1) * SCALES.len()])
                .max_by(|a, b| a.1[2].total_cmp(&b.1[2]))
                .expect("three scales");
            if *psr < min_psr {
                continue;
            }
            let (sw, sh) = (ww * scale, wh * scale);
            let moved = (
                f64::from(shift(*peak_x)) * sw / SIZE as f64,
                f64::from(shift(*peak_y)) * sh / SIZE as f64,
            );
            let found_size = (size.0 * scale, size.1 * scale);
            let found = [
                centre.0 + moved.0 - found_size.0 / 2.0,
                centre.1 + moved.1 - found_size.1 / 2.0,
                found_size.0,
                found_size.1,
            ];
            if let Some(tlwh) = tracker.correct(object.id, found, doubt) {
                corrected.push((index, object.id, tlwh));
            }
        }
        let boxes: Vec<(u64, [f64; 4])> =
            corrected.iter().map(|&(_, id, tlwh)| (id, tlwh)).collect();
        self.learn(source, &boxes, 1.0, LEARNING_RATE)?;
        self.device.finish()?;
        for (index, _, tlwh) in corrected {
            expected[index].tlwh = tlwh;
        }
        Ok(expected)
    }

    /// Learns how each object a detection was matched to looks — a new one
    /// from three looks, nearer and further, as the CPU's filter starts —
    /// and forgets the filters of objects no longer followed, `alive`
    /// being those that are.
    pub(super) fn detected(
        &mut self,
        source: &D::Source,
        matched: &[(u64, [f64; 4])],
        alive: &[u64],
    ) -> Result<(), D::Error> {
        let gone: Vec<u64> = self
            .slots
            .keys()
            .copied()
            .filter(|id| !alive.contains(id))
            .collect();
        for id in gone {
            if let Some((slot, _)) = self.slots.remove(&id) {
                self.free.push(slot);
            }
        }
        self.device.ready()?;
        let (known, new): (Vec<_>, Vec<_>) = matched
            .iter()
            .copied()
            .partition(|(id, _)| self.slots.contains_key(id));
        self.learn(source, &known, 1.0, LEARNING_RATE)?;
        for (id, _) in &new {
            self.slot(*id)?;
        }
        for (scale, rate) in FIRST_LOOKS {
            self.learn(source, &new, scale, rate)?;
        }
        self.device.finish()
    }

    /// Forgets every filter: after a seek, the numbers are a new tracker's.
    pub(super) fn clear(&mut self) {
        self.slots.clear();
        self.free.clear();
        self.next = 0;
    }
}

/// CUDA's: the driver's kernels and cuFFT.
///
/// The device's filters come before the driver: fields drop in order, and
/// they are freed in the driver's context, which the driver releases.
#[cfg(feature = "cuda-visual-tracking")]
pub(super) struct CudaLooks {
    dcf: crate::platform::cuda::driver::CudaDcf,
    driver: crate::platform::cuda::driver::CudaDriver,
}

#[cfg(feature = "cuda-visual-tracking")]
impl DcfDevice for CudaLooks {
    type Source = crate::platform::cuda::driver::DcfSource;
    type Error = crate::platform::cuda::driver::CudaDriverError;

    fn open(window: &[f32], target: &[[f32; 2]]) -> Result<Self, Self::Error> {
        let driver = crate::platform::cuda::driver::CudaDriver::retain_primary()?;
        let dcf = driver.dcf(window, target)?;
        Ok(Self { dcf, driver })
    }

    fn describe(&self) -> String {
        format!("cuFFT {}", crate::platform::cuda::driver::cufft_version())
    }

    /// Where `frame`'s brightness is on the device, where it is an NV12 or
    /// BGRA CUDA picture.
    fn source(frame: &ffmpeg::frame::Video) -> Option<Self::Source> {
        use crate::platform::cuda::driver::{BgraSurface, DcfSource, Nv12Surface};
        let size = (frame.width(), frame.height());
        match crate::platform::cuda::frame::surface_layout(frame)? {
            ffmpeg::format::Pixel::NV12 => {
                let surface = Nv12Surface::from_frame(frame)?;
                Some(DcfSource {
                    pixels: surface.luma,
                    pitch: surface.luma_pitch,
                    size,
                    bgra: false,
                })
            }
            ffmpeg::format::Pixel::BGRA => {
                let surface = BgraSurface::from_frame(frame)?;
                Some(DcfSource {
                    pixels: surface.pixels,
                    pitch: surface.pitch,
                    size,
                    bgra: true,
                })
            }
            _ => None,
        }
    }

    /// What wrote the picture ran on other streams of the context.
    fn ready(&self) -> Result<(), Self::Error> {
        self.driver.synchronize()
    }

    fn reserve_slots(&mut self, slots: usize) -> Result<(), Self::Error> {
        self.dcf.reserve_slots(&self.driver, slots)
    }

    fn sample(&mut self, source: &Self::Source, jobs: &[DcfJob]) -> Result<(), Self::Error> {
        self.dcf.sample(&self.driver, *source, jobs)
    }

    fn find(&mut self, slots: &[u32], regulariser: f32) -> Result<Vec<[f32; 4]>, Self::Error> {
        Ok(self
            .dcf
            .find(&self.driver, slots, regulariser)?
            .into_iter()
            .map(|peak| [peak.x, peak.y, peak.top, peak.psr])
            .collect())
    }

    fn learn(&mut self, slots: &[u32], rate: f32) -> Result<(), Self::Error> {
        self.dcf.learn(&self.driver, slots, rate)
    }

    /// Each call ran as it was made.
    fn finish(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// Metal's: kernels of this crate's own, the transform among them.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) struct MetalLooks(crate::platform::macos::dcf::MetalDcf);

#[cfg(all(target_os = "macos", feature = "metal"))]
impl DcfDevice for MetalLooks {
    type Source = crate::platform::macos::dcf::MetalDcfSource;
    type Error = crate::platform::macos::metal::MetalError;

    fn open(window: &[f32], target: &[[f32; 2]]) -> Result<Self, Self::Error> {
        crate::platform::macos::dcf::MetalDcf::new(window, target).map(Self)
    }

    fn describe(&self) -> String {
        "Metal".to_owned()
    }

    fn source(frame: &ffmpeg::frame::Video) -> Option<Self::Source> {
        crate::platform::macos::dcf::MetalDcfSource::of(frame)
    }

    /// What wrote a VideoToolbox picture finished before handing it on.
    fn ready(&self) -> Result<(), Self::Error> {
        Ok(())
    }

    fn reserve_slots(&mut self, slots: usize) -> Result<(), Self::Error> {
        self.0.reserve_slots(slots)
    }

    fn sample(&mut self, source: &Self::Source, jobs: &[DcfJob]) -> Result<(), Self::Error> {
        self.0.sample(source, jobs)
    }

    fn find(&mut self, slots: &[u32], regulariser: f32) -> Result<Vec<[f32; 4]>, Self::Error> {
        self.0.find(slots, regulariser)
    }

    fn learn(&mut self, slots: &[u32], rate: f32) -> Result<(), Self::Error> {
        self.0.learn(slots, rate)
    }

    fn finish(&mut self) -> Result<(), Self::Error> {
        self.0.finish()
    }
}

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::buffer::MediaBuffer;
    use crate::element::{RawSink, SrcPads};
    use crate::elements::{AppSink, VideoToolboxUpload};
    use crate::test_support::try_videotoolbox_device;

    use super::super::dcf::Dcf;
    use super::super::luma::SystemLuma;

    fn noise(x: u32, y: u32, seed: u32) -> u8 {
        let mut h = x.wrapping_mul(374_761_393) ^ y.wrapping_mul(668_265_263) ^ seed;
        h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
        ((h ^ (h >> 16)) & 0xff) as u8
    }

    /// A 320 by 240 picture of noise with a brighter textured block at
    /// `block`, its texture moving with it: in system memory as grey, and
    /// uploaded as an NV12 VideoToolbox picture.
    fn scene(
        device: &crate::elements::VideoToolboxDevice,
        block: [u32; 4],
    ) -> (ffmpeg::frame::Video, MediaBuffer) {
        let (w, h) = (320u32, 240u32);
        let mut grey = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::GRAY8, w, h);
        let mut nv12 = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, w, h);
        let [bx, by, bw, bh] = block;
        let (gs, ns) = (grey.stride(0), nv12.stride(0));
        for y in 0..h {
            for x in 0..w {
                let inside = x >= bx && x < bx + bw && y >= by && y < by + bh;
                let value = if inside {
                    100 + noise((x - bx) / 3, (y - by) / 3, 7) / 2
                } else {
                    noise(x, y, 1) / 4
                };
                grey.data_mut(0)[y as usize * gs + x as usize] = value;
                nv12.data_mut(0)[y as usize * ns + x as usize] = value;
            }
        }
        nv12.data_mut(1).fill(128);
        let mut upload = VideoToolboxUpload::new("upload", device);
        let kept = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&kept);
        upload.src_pads()[0].link(Box::new(AppSink::new("kept", move |buf| {
            sink.lock().unwrap().push(buf);
            Ok(())
        })));
        upload.consume(MediaBuffer::video(nv12)).expect("uploaded");
        let on_gpu = kept.lock().unwrap().remove(0);
        (grey, on_gpu)
    }

    /// Learned on one picture and looked for on the next, where the block
    /// moved 5 pixels right and 3 down, Metal's filters find it where the
    /// CPU's do — the same filter, sampled, transformed and correlated on
    /// the GPU instead — and are as sure of it. Proves the kernels compile
    /// and run, which the tracker would otherwise quietly fall back from.
    #[test]
    fn metal_filters_find_what_the_cpu_filters_find() {
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let target: Vec<[f32; 2]> = target().iter().map(|c| [c.re, c.im]).collect();
        let mut gpu = MetalLooks::open(window(), &target).expect("the kernels compile");
        let block = [100u32, 80, 40, 60];
        let moved = [105u32, 83, 40, 60];
        let (first_grey, first) = scene(&device, block);
        let (second_grey, second) = scene(&device, moved);
        let (MediaBuffer::Video(first), MediaBuffer::Video(second)) = (&first, &second) else {
            panic!("pictures");
        };

        let tlwh = block.map(f64::from);
        let centre = (tlwh[0] + tlwh[2] / 2.0, tlwh[1] + tlwh[3] / 2.0);
        let (ww, wh) = window_of((tlwh[2], tlwh[3]));
        let source = MetalLooks::source(first).expect("an NV12 VideoToolbox picture");
        gpu.reserve_slots(1).expect("a slot");
        for (scale, rate) in FIRST_LOOKS {
            let job = [
                centre.0 as f32,
                centre.1 as f32,
                (ww * scale) as f32,
                (wh * scale) as f32,
            ];
            gpu.sample(&source, &[job]).expect("sampled");
            gpu.learn(&[0], rate).expect("learned");
        }
        let source = MetalLooks::source(second).expect("an NV12 VideoToolbox picture");
        let jobs: Vec<DcfJob> = SCALES
            .iter()
            .map(|scale| {
                [
                    centre.0 as f32,
                    centre.1 as f32,
                    (ww * scale) as f32,
                    (wh * scale) as f32,
                ]
            })
            .collect();
        gpu.sample(&source, &jobs).expect("sampled");
        let peaks = gpu.find(&[0, 0, 0], REGULARISER).expect("found");
        let (scale, [px, py, _, psr]) = SCALES
            .iter()
            .zip(&peaks)
            .max_by(|a, b| a.1[2].total_cmp(&b.1[2]))
            .expect("three scales");
        let half = (SIZE / 2) as f32;
        let shift = |at: f32| {
            let d = at - half;
            if d > half { d - SIZE as f32 } else { d }
        };
        let found = (
            centre.0 + f64::from(shift(*px)) * ww * scale / SIZE as f64,
            centre.1 + f64::from(shift(*py)) * wh * scale / SIZE as f64,
        );

        let cpu = Dcf::new(&mut SystemLuma::of(&first_grey).unwrap(), tlwh).expect("a filter");
        let on_cpu = cpu
            .find(&mut SystemLuma::of(&second_grey).unwrap(), centre)
            .expect("found");
        let cpu_centre = (
            on_cpu.tlwh[0] + on_cpu.tlwh[2] / 2.0,
            on_cpu.tlwh[1] + on_cpu.tlwh[3] / 2.0,
        );
        eprintln!(
            "metal {found:?} psr {psr}, cpu {cpu_centre:?} psr {}",
            on_cpu.psr
        );
        assert!(
            (found.0 - 125.0).abs() < 1.5 && (found.1 - 113.0).abs() < 1.5,
            "found at {found:?}"
        );
        assert!(
            (found.0 - cpu_centre.0).abs() < 0.5 && (found.1 - cpu_centre.1).abs() < 0.5,
            "metal {found:?}, cpu {cpu_centre:?}"
        );
        assert!(
            *psr > 8.0 && (psr - on_cpu.psr).abs() < 1.0,
            "psr {psr} against {}",
            on_cpu.psr
        );
    }
}
