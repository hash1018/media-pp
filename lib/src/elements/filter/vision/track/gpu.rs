//! Following by look on the GPU, for CUDA pictures, with the
//! `cuda-visual-tracking` feature: the same correlation filters as
//! [`super::dcf`], every followed object's sampled, transformed and matched
//! at once on the device — see `platform::cuda::driver::CudaDcf`. Only
//! four numbers an object come back: where its response peaks, how high,
//! and how sure.

use std::collections::HashMap;

use ffmpeg_next::{self as ffmpeg, format::Pixel};

use crate::platform::cuda::driver::{
    BgraSurface, CudaDcf, CudaDriver, CudaDriverError, DcfJob, DcfSource, Nv12Surface,
    cufft_version,
};

use super::byte_track::{ByteTrack, Expected};
use super::dcf::{LEARNING_RATE, REGULARISER, SCALES, SIZE, target, window, window_of};

/// The looks a new object's filter is first learned from, and the rate
/// each is taken in at — 1, a half, a third: their average, as the CPU's
/// filter learns its first.
const FIRST_LOOKS: [(f64, f32); 3] = [(0.95, 1.0), (1.0, 0.5), (1.05, 1.0 / 3.0)];

/// The filters of the objects followed on CUDA pictures.
///
/// The device's filters come before the driver: fields drop in order, and
/// they are freed in the driver's context, which the driver releases.
pub(super) struct GpuLooks {
    dcf: CudaDcf,
    driver: CudaDriver,
    /// Each followed object's filter slot, and the box size it was last
    /// learned at.
    slots: HashMap<u64, (u32, (f64, f64))>,
    free: Vec<u32>,
    next: u32,
}

impl GpuLooks {
    /// The driver, the kernels and the window and wanted response on the
    /// device; `None` where the driver cannot be opened.
    pub(super) fn new() -> Result<Self, CudaDriverError> {
        let driver = CudaDriver::retain_primary()?;
        let target: Vec<[f32; 2]> = target().iter().map(|c| [c.re, c.im]).collect();
        let dcf = driver.dcf(window(), &target)?;
        Ok(Self {
            dcf,
            driver,
            slots: HashMap::new(),
            free: Vec::new(),
            next: 0,
        })
    }

    /// cuFFT's version, as it says it.
    pub(super) fn cufft() -> i32 {
        cufft_version()
    }

    /// Where `frame`'s brightness is on the device, where it is an NV12 or
    /// BGRA CUDA picture.
    pub(super) fn source(frame: &ffmpeg::frame::Video) -> Option<DcfSource> {
        let size = (frame.width(), frame.height());
        match crate::platform::cuda::frame::surface_layout(frame)? {
            Pixel::NV12 => {
                let surface = Nv12Surface::from_frame(frame)?;
                Some(DcfSource {
                    pixels: surface.luma,
                    pitch: surface.luma_pitch,
                    size,
                    bgra: false,
                })
            }
            Pixel::BGRA => {
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

    /// A slot for object `id`'s filter.
    fn slot(&mut self, id: u64) -> Result<u32, CudaDriverError> {
        let slot = self.free.pop().unwrap_or_else(|| {
            self.next += 1;
            self.next - 1
        });
        self.dcf.reserve_slots(&self.driver, slot as usize + 1)?;
        self.slots.insert(id, (slot, (0.0, 0.0)));
        Ok(slot)
    }

    /// Learns how each of `boxes` looks, into the filters already slotted,
    /// at `rate`, each looked at `scale` times its window.
    fn learn(
        &mut self,
        source: DcfSource,
        boxes: &[(u64, [f64; 4])],
        scale: f64,
        rate: f32,
    ) -> Result<(), CudaDriverError> {
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
        self.dcf.sample(&self.driver, source, &jobs)?;
        self.dcf.learn(&self.driver, &slots, rate)
    }

    /// Moves each followed object to where its filter finds it on the
    /// picture `source` is, where the filter is at least `min_psr` sure,
    /// weighed against the motion at `doubt`, and learns how it looks
    /// there: [`super::object_tracker`]'s `follow_by_look`, for every
    /// object at once.
    pub(super) fn follow(
        &mut self,
        tracker: &mut ByteTrack,
        source: DcfSource,
        min_psr: f32,
        doubt: f64,
    ) -> Result<Vec<Expected>, CudaDriverError> {
        // What wrote the picture ran on other streams of the context.
        self.driver.synchronize()?;
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
        self.dcf.sample(&self.driver, source, &jobs)?;
        let peaks = self.dcf.find(&self.driver, &slots, REGULARISER)?;

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
            let (scale, peak) = SCALES
                .iter()
                .zip(&peaks[n * SCALES.len()..(n + 1) * SCALES.len()])
                .max_by(|a, b| a.1.top.total_cmp(&b.1.top))
                .expect("three scales");
            if peak.psr < min_psr {
                continue;
            }
            let (sw, sh) = (ww * scale, wh * scale);
            let moved = (
                f64::from(shift(peak.x)) * sw / SIZE as f64,
                f64::from(shift(peak.y)) * sh / SIZE as f64,
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
        source: DcfSource,
        matched: &[(u64, [f64; 4])],
        alive: &[u64],
    ) -> Result<(), CudaDriverError> {
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
        self.driver.synchronize()?;
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
        Ok(())
    }

    /// Forgets every filter: after a seek, the numbers are a new tracker's.
    pub(super) fn clear(&mut self) {
        self.slots.clear();
        self.free.clear();
        self.next = 0;
    }
}
