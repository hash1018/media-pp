//! [`ObjectTracker`]: what a detector found, numbered across pictures.

use std::collections::HashMap;
use std::sync::Arc;

use ffmpeg_next as ffmpeg;

use crate::pp_log::{PpLog, pp_info, pp_warn};
use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, OutputContract, PortContract},
    element::{Element, ElementType, element_pp_log},
    elements::{Detection, Detections},
    error::Result,
    transform::{Filter, FilterStage, Output, filter_stage},
};

use super::TrackerOptions;
use super::byte_track::{ByteTrack, Expected, Seen};
use super::dcf::{Dcf, LEARNING_RATE};
use super::luma::{Luma, SystemLuma};

/// How sure a correlation filter must be of where it found an object for
/// the track to be moved there: a peak-to-sidelobe ratio Bolme found to
/// tell a found object from a lost one.
const MIN_PSR: f32 = 7.0;
/// How much less a filter's find is trusted than a detection, as the
/// factor on a detection's uncertainty. The find is weighed against the
/// motion rather than taken as it is: over a short gap the motion is the
/// better guess, and a filter taken at its word made the boxes worse there
/// than motion alone — measured on 12 fps walking people, 0.71 mean overlap
/// with the detector's against 0.84 a picture after a detection. At 3 the
/// two agree over short gaps, and the look wins over long ones: 0.61
/// against 0.47 ten pictures on. 2 and 5 measured a little worse.
const LOOK_DOUBT: f64 = 3.0;

/// The most pictures one gap between two is taken to span: past it, a gap
/// is a jump rather than pictures missed, and the motion is not run on
/// through it.
const MAX_STEPS: u64 = 120;

/// Numbers the [`Detections`] each picture carries so that one object keeps
/// one number — [`Detection::track_id`] — and puts on each picture a
/// detector let by where it expects the objects to be, marked
/// [`Detections::predicted`]: ByteTrack, a Kalman filter per object and
/// matching by overlap, confident detections first and then the
/// unconfident a partly hidden object still gets.
///
/// It goes after a detector: `Detector -> ObjectTracker -> Overlay`. It
/// reads and writes metadata, and hands each picture on as it came, so it
/// takes pictures wherever they live — system memory, CUDA, VideoToolbox,
/// any GPU. With
/// [`TrackerOptions::visual`] it also follows each object by how it looks,
/// reading the pixels around it where it can. A picture before any
/// detection has been seen goes on carrying nothing.
///
/// Pictures missed between two — a dropping queue before the detector —
/// are told from the timestamps, and the motion is carried through them,
/// so an object that moved further than one picture's worth is still
/// matched.
pub struct ObjectTracker(FilterStage<Tracking>);

filter_stage!(ObjectTracker);

/// How many pictures each one handed in is on from the last, from the
/// timestamps: the gap over the smallest gap yet seen.
#[derive(Debug, Clone, Copy, Default)]
struct Pace {
    last: Option<i64>,
    unit: Option<i64>,
}

impl Pace {
    fn steps(&mut self, pts: Option<i64>) -> u64 {
        let Some(pts) = pts else {
            return 1;
        };
        let steps = match self.last {
            Some(last) if pts > last => {
                let gap = pts - last;
                let unit = self.unit.map_or(gap, |unit| unit.min(gap));
                self.unit = Some(unit);
                ((gap as f64 / unit as f64).round() as u64).clamp(1, MAX_STEPS)
            }
            _ => 1,
        };
        self.last = Some(pts);
        steps
    }
}

/// The detector detections came from, and its class names.
#[derive(Debug, Clone)]
struct Origin {
    detector: Arc<str>,
    labels: Arc<[Arc<str>]>,
}

/// What an [`ObjectTracker`] does with each picture.
struct Tracking {
    name: Arc<str>,
    pp_log: PpLog,
    options: TrackerOptions,
    tracker: ByteTrack,
    pace: Pace,
    /// Whose the last detections seen were, which the expected ones on a
    /// picture let by say too.
    last: Option<Origin>,
    /// Each followed object's correlation filter, by its number, where the
    /// tracker follows by look.
    filters: HashMap<u64, Dcf>,
    /// The driver CUDA pictures are read through, opened at the first.
    #[cfg(feature = "cuda")]
    driver: Option<crate::platform::cuda::driver::CudaDriver>,
    /// Whether it has said that it cannot read a picture's pixels.
    unreadable_said: bool,
    /// The filters of objects followed on CUDA pictures, on the GPU, made at
    /// the first such picture.
    #[cfg(feature = "cuda-visual-tracking")]
    gpu: Option<super::gpu::GpuLooks>,
    /// Whether the GPU's filters could not be made, and CUDA pictures are
    /// read down to the CPU's instead.
    #[cfg(feature = "cuda-visual-tracking")]
    gpu_failed: bool,
}

#[cfg(feature = "cuda-visual-tracking")]
impl Tracking {
    /// Where `frame`'s brightness is on the device, with the GPU's filters
    /// ready, where it is a CUDA picture they can follow on.
    fn on_gpu(
        &mut self,
        frame: &ffmpeg::frame::Video,
    ) -> Option<crate::platform::cuda::driver::DcfSource> {
        let source = super::gpu::GpuLooks::source(frame)?;
        if self.gpu.is_none() && !self.gpu_failed {
            match super::gpu::GpuLooks::new() {
                Ok(gpu) => {
                    pp_info!(
                        pp_log: &self.pp_log,
                        "following by look on the GPU, cuFFT {}",
                        super::gpu::GpuLooks::cufft()
                    );
                    self.gpu = Some(gpu);
                }
                Err(error) => {
                    pp_warn!(
                        pp_log: &self.pp_log,
                        "cannot follow by look on the GPU ({error}): reading CUDA pictures down instead"
                    );
                    self.gpu_failed = true;
                }
            }
        }
        self.gpu.as_ref().map(|_| source)
    }
}

/// `frame`'s brightness, where it can be read.
fn luma<'a>(
    frame: &'a ffmpeg::frame::Video,
    #[cfg(feature = "cuda")] driver: &'a mut Option<crate::platform::cuda::driver::CudaDriver>,
) -> Option<Box<dyn Luma + 'a>> {
    if let Some(system) = SystemLuma::of(frame) {
        return Some(Box::new(system));
    }
    #[cfg(feature = "cuda")]
    if frame.format() == ffmpeg::format::Pixel::CUDA {
        use crate::platform::cuda::driver::CudaDriver;
        if driver.is_none() {
            *driver = CudaDriver::retain_primary().ok();
        }
        let driver = driver.as_ref()?;
        // What wrote the picture — a decoder, a scaler — ran on other
        // streams of the context; the copies wait for all of it.
        driver.synchronize().ok()?;
        return super::luma::CudaLuma::of(driver, frame)
            .map(|luma| Box::new(luma) as Box<dyn Luma>);
    }
    #[cfg(all(target_os = "macos", feature = "metal"))]
    if frame.format() == ffmpeg::format::Pixel::VIDEOTOOLBOX {
        return super::luma::VideoToolboxLuma::of(frame)
            .map(|luma| Box::new(luma) as Box<dyn Luma>);
    }
    None
}

/// Moves each followed object to where its filter finds it, where the
/// filter is sure, and learns how it looks there.
fn follow_by_look(
    tracker: &mut ByteTrack,
    filters: &mut HashMap<u64, Dcf>,
    luma: &mut dyn Luma,
) -> Vec<Expected> {
    let mut expected = tracker.expected();
    for object in &mut expected {
        let Some(filter) = filters.get_mut(&object.id) else {
            continue;
        };
        let [x, y, w, h] = object.tlwh;
        let Some(found) = filter.find(luma, (x + w / 2.0, y + h / 2.0)) else {
            continue;
        };
        if found.psr < MIN_PSR {
            continue;
        }
        let Some(tlwh) = tracker.correct(object.id, found.tlwh, LOOK_DOUBT) else {
            continue;
        };
        filter.learn(luma, tlwh, LEARNING_RATE);
        object.tlwh = tlwh;
    }
    expected
}

impl ObjectTracker {
    /// A tracker deciding as `options` say.
    pub fn new(name: impl Into<String>, options: TrackerOptions) -> Self {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::ObjectTracker, &name, None);
        pp_info!(pp_log: &pp_log, "opened: {options:?}");
        Self(FilterStage::new(Tracking {
            name,
            pp_log,
            options,
            tracker: ByteTrack::new(options),
            pace: Pace::default(),
            last: None,
            filters: HashMap::new(),
            #[cfg(feature = "cuda")]
            driver: None,
            unreadable_said: false,
            #[cfg(feature = "cuda-visual-tracking")]
            gpu: None,
            #[cfg(feature = "cuda-visual-tracking")]
            gpu_failed: false,
        }))
    }
}

impl Element for Tracking {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::ObjectTracker
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Filter for Tracking {
    /// Pictures wherever they live: only their metadata is read.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::any_frame(MediaKind::VideoFrame))
    }

    fn output_contract(&self) -> OutputContract {
        OutputContract::Passthrough
    }

    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        let MediaBuffer::Video(frame) = &buf else {
            out.push(buf);
            return Ok(());
        };
        let (width, height) = (f64::from(frame.width()), f64::from(frame.height()));
        let steps = self.pace.steps(frame.pts());
        let found = buf
            .metadata()
            .and_then(|metadata| metadata.get::<Detections>())
            .cloned();
        if found.is_none() && self.last.is_none() {
            out.push(buf);
            return Ok(());
        }

        self.tracker.advance(steps);
        #[cfg(feature = "cuda-visual-tracking")]
        let gpu_source = if self.options.visual {
            self.on_gpu(frame)
        } else {
            None
        };
        #[cfg(not(feature = "cuda-visual-tracking"))]
        let gpu_source: Option<()> = None;
        let mut pixels = if self.options.visual && gpu_source.is_none() {
            luma(
                frame,
                #[cfg(feature = "cuda")]
                &mut self.driver,
            )
        } else {
            None
        };
        if self.options.visual && gpu_source.is_none() && pixels.is_none() && !self.unreadable_said
        {
            pp_warn!(
                pp_log: &self.pp_log,
                "cannot read the pixels of {:?} pictures: following by motion alone",
                frame.format()
            );
            self.unreadable_said = true;
        }
        // Where each object is now, by its look where it can be seen: on a
        // picture let by, that is the answer; on a detected one, it is where
        // the matching starts from.
        let expected = match pixels.as_deref_mut() {
            Some(luma) => follow_by_look(&mut self.tracker, &mut self.filters, luma),
            None => self.tracker.expected(),
        };
        #[cfg(feature = "cuda-visual-tracking")]
        let expected = match (gpu_source, self.gpu.as_mut()) {
            (Some(source), Some(gpu)) => gpu
                .follow(&mut self.tracker, source, MIN_PSR, LOOK_DOUBT)
                .unwrap_or_else(|error| {
                    pp_warn!(pp_log: &self.pp_log, "following by look on the GPU failed: {error}");
                    expected
                }),
            _ => expected,
        };

        let Some(mut found) = found else {
            let Some(Origin { detector, labels }) = self.last.clone() else {
                unreachable!("returned above without one");
            };
            let items = expected
                .into_iter()
                .filter_map(|expected| {
                    let [x, y, w, h] = expected.tlwh;
                    let (left, top) = ((x / width).max(0.0), (y / height).max(0.0));
                    let (right, bottom) = (((x + w) / width).min(1.0), ((y + h) / height).min(1.0));
                    (right > left && bottom > top).then(|| Detection {
                        track_id: Some(expected.id),
                        ..Detection::new(
                            expected.class_id,
                            expected.score,
                            left as f32,
                            top as f32,
                            (right - left) as f32,
                            (bottom - top) as f32,
                        )
                    })
                })
                .collect();
            let expected = Detections {
                predicted: true,
                ..Detections::new(detector, labels, items)
            };
            drop(pixels);
            out.push(expected.attach_to(buf));
            return Ok(());
        };

        let seen: Vec<Seen> = found
            .items
            .iter()
            .map(|item| Seen {
                class_id: item.class_id,
                score: item.score,
                tlwh: [
                    f64::from(item.x) * width,
                    f64::from(item.y) * height,
                    f64::from(item.width) * width,
                    f64::from(item.height) * height,
                ],
            })
            .collect();
        let ids = self.tracker.associate(&seen);
        if let Some(luma) = pixels.as_deref_mut() {
            // Each matched object as the detector saw it is how it looks now.
            for (seen, id) in seen.iter().zip(&ids) {
                let Some(id) = id else {
                    continue;
                };
                match self.filters.get_mut(id) {
                    Some(filter) => filter.learn(luma, seen.tlwh, LEARNING_RATE),
                    None => {
                        if let Some(filter) = Dcf::new(luma, seen.tlwh) {
                            self.filters.insert(*id, filter);
                        }
                    }
                }
            }
            let alive = self.tracker.ids();
            self.filters.retain(|id, _| alive.contains(id));
        }
        #[cfg(feature = "cuda-visual-tracking")]
        if let (Some(source), Some(gpu)) = (gpu_source, self.gpu.as_mut()) {
            let matched: Vec<(u64, [f64; 4])> = seen
                .iter()
                .zip(&ids)
                .filter_map(|(seen, id)| Some(((*id)?, seen.tlwh)))
                .collect();
            if let Err(error) = gpu.detected(source, &matched, &self.tracker.ids()) {
                pp_warn!(pp_log: &self.pp_log, "learning on the GPU failed: {error}");
            }
        }
        drop(pixels);
        for (item, id) in found.items.iter_mut().zip(ids) {
            item.track_id = id;
        }
        self.last = Some(Origin {
            detector: Arc::clone(&found.detector),
            labels: Arc::clone(&found.labels),
        });
        out.push(found.attach_to(buf));
        Ok(())
    }

    /// A seek or a flush: what follows is not where these objects went.
    fn reset(&mut self) {
        self.tracker = ByteTrack::new(self.options);
        self.pace = Pace::default();
        self.last = None;
        self.filters.clear();
        #[cfg(feature = "cuda-visual-tracking")]
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::element::{RawSink, SrcPads};
    use crate::elements::AppSink;
    use crate::ffmpeg;

    fn capture(stage: &mut dyn SrcPads) -> Arc<Mutex<Vec<MediaBuffer>>> {
        let kept = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&kept);
        stage.src_pads()[0].link(Box::new(AppSink::new("kept", move |buf| {
            sink.lock().unwrap().push(buf);
            Ok(())
        })));
        kept
    }

    /// A 200 by 100 picture at `pts`, carrying `found` if anything.
    fn picture(pts: i64, found: Option<Vec<Detection>>) -> MediaBuffer {
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::RGB24, 200, 100);
        frame.set_pts(Some(pts));
        let buf = MediaBuffer::video(frame);
        match found {
            Some(items) => {
                Detections::new("detector", Arc::from([Arc::from("thing")]), items).attach_to(buf)
            }
            None => buf,
        }
    }

    fn detections(buf: &MediaBuffer) -> Option<&Detections> {
        buf.metadata()?.get::<Detections>()
    }

    /// An object moving right a fraction a picture, detected on every third
    /// picture: every detection has one number, and the pictures between
    /// carry it where the motion takes it.
    #[test]
    fn detections_are_numbered_and_the_pictures_between_filled_in() {
        let mut tracker = ObjectTracker::new("tracker", TrackerOptions::default());
        let kept = capture(&mut tracker);
        tracker.consume(picture(0, None)).expect("before anything");
        for n in 0..30 {
            let x = 0.1 + 0.01 * n as f32;
            let found = (n % 3 == 0).then(|| vec![Detection::new(0, 0.9, x, 0.2, 0.1, 0.5)]);
            tracker
                .consume(picture(1000 * (n + 1), found))
                .expect("tracked");
        }
        let kept = kept.lock().unwrap();
        assert!(
            detections(&kept[0]).is_none(),
            "nothing before anything was seen"
        );
        let ids: Vec<Option<u64>> = kept[1..]
            .iter()
            .map(|buf| detections(buf).unwrap().items[0].track_id)
            .collect();
        assert!(
            ids.iter().all(|id| *id == ids[0] && id.is_some()),
            "{ids:?}"
        );
        for (n, buf) in kept[1..].iter().enumerate() {
            let found = detections(buf).unwrap();
            assert_eq!(found.predicted, n % 3 != 0, "picture {n}");
            assert_eq!(found.label(&found.items[0]), Some("thing"));
            if n > 9 {
                let x = 0.1 + 0.01 * n as f32;
                assert!(
                    (found.items[0].x - x).abs() < 0.01,
                    "picture {n}: {} for {x}",
                    found.items[0].x
                );
            }
        }
    }

    /// A grey 240 by 160 picture at `pts` with a textured block at `block`
    /// on a textured background, made a buffer by `place` — put where it is
    /// to be read from — and carrying a detection of the block where
    /// `detected`.
    fn scene(
        pts: i64,
        block: [u32; 4],
        detected: bool,
        place: &dyn Fn(ffmpeg::frame::Video) -> MediaBuffer,
    ) -> MediaBuffer {
        let (w, h) = (240u32, 160u32);
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::GRAY8, w, h);
        frame.set_pts(Some(pts));
        let stride = frame.stride(0);
        let hash = |x: u32, y: u32, seed: u32| {
            let mut v = x.wrapping_mul(374_761_393) ^ y.wrapping_mul(668_265_263) ^ seed;
            v = (v ^ (v >> 13)).wrapping_mul(1_274_126_177);
            ((v ^ (v >> 16)) & 0xff) as u8
        };
        let [bx, by, bw, bh] = block;
        for y in 0..h {
            for x in 0..w {
                let inside = x >= bx && x < bx + bw && y >= by && y < by + bh;
                frame.data_mut(0)[y as usize * stride + x as usize] = if inside {
                    100 + hash((x - bx) / 3, (y - by) / 3, 7) / 2
                } else {
                    hash(x, y, 1) / 4
                };
            }
        }
        let buf = place(frame);
        if !detected {
            return buf;
        }
        let fraction = |v: u32, of: u32| v as f32 / of as f32;
        Detections::new(
            "detector",
            Arc::from([Arc::from("thing")]),
            vec![Detection::new(
                0,
                0.9,
                fraction(bx, w),
                fraction(by, h),
                fraction(bw, w),
                fraction(bh, h),
            )],
        )
        .attach_to(buf)
    }

    /// Where the block is at picture `n`: right for twenty pictures, then
    /// down — a turn, between two detections.
    fn turning(n: u32) -> [u32; 4] {
        let (x, y) = if n <= 20 {
            (20 + 3 * n, 30)
        } else {
            (80, 30 + 3 * (n - 20))
        };
        [x, y, 24, 32]
    }

    /// How far the boxes the tracker put on the pictures let by are from
    /// the block, at worst, in pixels, detecting every tenth picture.
    fn worst_miss(visual: bool) -> f32 {
        worst_miss_on(visual, &MediaBuffer::video)
    }

    /// [`worst_miss`], each picture put where `place` puts it.
    fn worst_miss_on(visual: bool, place: &dyn Fn(ffmpeg::frame::Video) -> MediaBuffer) -> f32 {
        let options = TrackerOptions {
            visual,
            ..TrackerOptions::default()
        };
        let mut tracker = ObjectTracker::new("tracker", options);
        let kept = capture(&mut tracker);
        for n in 0..=34 {
            tracker
                .consume(scene(i64::from(n) * 100, turning(n), n % 10 == 0, place))
                .expect("tracked");
        }
        let kept = kept.lock().unwrap();
        (0..=34u32)
            .filter(|n| n % 10 != 0)
            .map(|n| {
                let found = detections(&kept[n as usize]).expect("filled in");
                // An object no longer followed is as lost as it can be.
                let Some(item) = found.items.first() else {
                    return f32::INFINITY;
                };
                let [bx, by, _, _] = turning(n);
                let dx = item.x * 240.0 - bx as f32;
                let dy = item.y * 160.0 - by as f32;
                (dx * dx + dy * dy).sqrt()
            })
            .fold(0.0, f32::max)
    }

    /// Detected every tenth picture, a block that turns sharply between two
    /// detections is kept by its look, where its motion alone carries the
    /// box on past the turn and the next detection is not matched to it.
    /// The look is weighed against the motion, so the box lags the turn for
    /// a few pictures — about half the block's width at worst — before the
    /// look pulls it round.
    #[test]
    fn following_by_look_holds_through_a_turn_motion_alone_loses_it() {
        let by_look = worst_miss(true);
        let by_motion = worst_miss(false);
        eprintln!("worst miss: by look {by_look}, by motion {by_motion}");
        assert!(by_look < 16.0, "by look, {by_look} pixels at worst");
        assert!(
            by_motion.is_infinite(),
            "by motion, {by_motion} pixels at worst"
        );
    }

    /// The same turn, on VideoToolbox pictures: the look is read where the
    /// pictures are, and holds the block as it does in system memory.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn following_by_look_reads_videotoolbox_pictures() {
        let Some(device) = crate::test_support::try_videotoolbox_device() else {
            return;
        };
        let upload = |grey: ffmpeg::frame::Video| {
            let (w, h) = (grey.width(), grey.height());
            let mut nv12 = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, w, h);
            nv12.set_pts(grey.pts());
            let (from, to) = (grey.stride(0), nv12.stride(0));
            for y in 0..h as usize {
                let row = grey.data(0)[y * from..][..w as usize].to_vec();
                nv12.data_mut(0)[y * to..][..w as usize].copy_from_slice(&row);
            }
            nv12.data_mut(1).fill(128);
            let mut upload = crate::elements::VideoToolboxUpload::new("upload", &device);
            let uploaded = capture(&mut upload);
            upload.consume(MediaBuffer::video(nv12)).expect("uploaded");
            uploaded.lock().unwrap().remove(0)
        };
        let by_look = worst_miss_on(true, &upload);
        eprintln!("worst miss on VideoToolbox: by look {by_look}");
        assert!(by_look < 16.0, "by look, {by_look} pixels at worst");
    }

    /// The same turn on NV12 CUDA pictures: the look is read where they
    /// are — copied down region by region, or with `cuda-visual-tracking`
    /// followed on the GPU — and holds the block as the CPU's does, to
    /// within a pixel or two.
    #[cfg(feature = "cuda")]
    #[test]
    fn following_by_look_reads_cuda_pictures() {
        let Some((device, _serial)) = crate::test_support::try_cuda_device() else {
            return;
        };
        let upload = |grey: ffmpeg::frame::Video| {
            let (w, h) = (grey.width(), grey.height());
            let mut nv12 = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, w, h);
            nv12.set_pts(grey.pts());
            let (from, to) = (grey.stride(0), nv12.stride(0));
            for y in 0..h as usize {
                let row = grey.data(0)[y * from..][..w as usize].to_vec();
                nv12.data_mut(0)[y * to..][..w as usize].copy_from_slice(&row);
            }
            nv12.data_mut(1).fill(128);
            let mut upload = crate::elements::CudaUpload::new(
                "upload",
                &device,
                crate::platform::cuda::CudaFrameFormat::Nv12,
            );
            let uploaded = capture(&mut upload);
            upload.consume(MediaBuffer::video(nv12)).expect("uploaded");
            uploaded.lock().unwrap().remove(0)
        };
        let on_cuda = worst_miss_on(true, &upload);
        let in_memory = worst_miss(true);
        eprintln!("worst miss by look: on CUDA {on_cuda}, in system memory {in_memory}");
        assert!(on_cuda < 16.0, "on CUDA, {on_cuda} pixels at worst");
        assert!(
            (on_cuda - in_memory).abs() < 2.0,
            "on CUDA {on_cuda} against {in_memory} in system memory"
        );
    }

    /// A gap of three pictures' time between two is three pictures of
    /// motion: the object is matched where it got to.
    #[test]
    fn missed_pictures_are_counted_from_the_timestamps() {
        let mut pace = Pace::default();
        assert_eq!(pace.steps(Some(0)), 1);
        assert_eq!(pace.steps(Some(40)), 1);
        assert_eq!(pace.steps(Some(160)), 3);
        assert_eq!(pace.steps(Some(160)), 1, "the same time again");
        assert_eq!(pace.steps(None), 1);
        assert_eq!(pace.steps(Some(1_000_000)), MAX_STEPS);
    }

    #[test]
    fn what_is_not_a_picture_goes_through() {
        let mut tracker = ObjectTracker::new("tracker", TrackerOptions::default());
        let kept = capture(&mut tracker);
        let audio = MediaBuffer::audio(ffmpeg::frame::Audio::empty());
        tracker.consume(audio).expect("passed");
        assert_eq!(kept.lock().unwrap().len(), 1);
    }
}
