//! [`ObjectTracker`]: what a detector found, numbered across pictures.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use ffmpeg_next as ffmpeg;

use crate::pp_log::{PpLog, pp_info, pp_warn};
use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, OutputContract, PortContract},
    element::{Element, ElementType, element_pp_log},
    elements::vision::batch::PerStream,
    elements::{Detection, Detections, SceneCut},
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
///
/// After a [`StreamMux`](crate::elements::StreamMux) each stream is followed
/// on its own, by the [`StreamOrigin`](crate::elements::StreamOrigin) its
/// pictures carry, and its objects numbered from one count shared by all, so
/// that no two streams' objects ever share a number. A picture carrying a
/// [`SceneCut`] — the first of a new shot, after a cut detector such as
/// [`SwCutDetector`](crate::elements::SwCutDetector) — starts its stream
/// over: nothing of the new shot is where anything was. A seek of one stream —
/// its origin's `generation` moved — starts that stream over and leaves the
/// others as they were.
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

/// What an [`ObjectTracker`] follows of one stream.
struct Stream {
    tracker: ByteTrack,
    pace: Pace,
    /// Whose the last detections seen were, which the expected ones on a
    /// picture let by say too.
    last: Option<Origin>,
}

/// What an [`ObjectTracker`] does with each picture.
struct Tracking {
    name: Arc<str>,
    pp_log: PpLog,
    options: TrackerOptions,
    /// Each stream's objects, apart: the pictures of a mux's streams come
    /// one after another, and an object of one is never another's.
    streams: PerStream<Stream>,
    /// The last number given out, to an object of any stream.
    numbers: Arc<AtomicU64>,
    /// Each followed object's correlation filter, by its number — unique
    /// across the streams — where the tracker follows by look.
    filters: HashMap<u64, Dcf>,
    /// The driver CUDA pictures are read through, opened at the first.
    #[cfg(feature = "cuda")]
    driver: Option<crate::platform::cuda::driver::CudaDriver>,
    /// Whether it has said that it cannot read a picture's pixels.
    unreadable_said: bool,
    /// The filters of objects followed on a GPU.
    gpus: Gpus,
}

/// The GPUs' filters, by the pictures each follows on.
#[derive(Default)]
struct Gpus {
    /// CUDA pictures', made at the first such picture.
    #[cfg(feature = "cuda-visual-tracking")]
    cuda: OnGpu<super::gpu::CudaLooks>,
    /// VideoToolbox pictures', with Metal.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    metal: OnGpu<super::gpu::MetalLooks>,
}

/// One GPU's filters, made at the first picture they can follow on — or,
/// where they cannot be made, said once and not tried again, the pictures
/// read down to the CPU's filters instead.
#[cfg(any(
    feature = "cuda-visual-tracking",
    all(target_os = "macos", feature = "metal")
))]
struct OnGpu<D> {
    looks: Option<super::gpu::GpuLooks<D>>,
    failed: bool,
}

#[cfg(any(
    feature = "cuda-visual-tracking",
    all(target_os = "macos", feature = "metal")
))]
impl<D> Default for OnGpu<D> {
    fn default() -> Self {
        Self {
            looks: None,
            failed: false,
        }
    }
}

#[cfg(any(
    feature = "cuda-visual-tracking",
    all(target_os = "macos", feature = "metal")
))]
impl<D: super::gpu::DcfDevice> OnGpu<D> {
    /// The filters, made now if they are not yet and can be.
    fn open(&mut self, pp_log: &PpLog) -> Option<&mut super::gpu::GpuLooks<D>> {
        if self.looks.is_none() && !self.failed {
            match super::gpu::GpuLooks::<D>::open() {
                Ok(looks) => {
                    pp_info!(
                        pp_log: pp_log,
                        "following by look on the GPU, {}",
                        looks.describe()
                    );
                    self.looks = Some(looks);
                }
                Err(error) => {
                    pp_warn!(
                        pp_log: pp_log,
                        "cannot follow by look on the GPU ({error}): reading the pictures down instead"
                    );
                    self.failed = true;
                }
            }
        }
        self.looks.as_mut()
    }
}

/// A picture the GPU's filters follow on, and which GPU's.
enum OnGpuSource {
    #[cfg(feature = "cuda-visual-tracking")]
    Cuda(<super::gpu::CudaLooks as super::gpu::DcfDevice>::Source),
    #[cfg(all(target_os = "macos", feature = "metal"))]
    Metal(<super::gpu::MetalLooks as super::gpu::DcfDevice>::Source),
}

#[cfg_attr(
    not(any(
        feature = "cuda-visual-tracking",
        all(target_os = "macos", feature = "metal")
    )),
    allow(
        unused_variables,
        clippy::unused_self,
        clippy::needless_pass_by_ref_mut
    )
)]
impl Gpus {
    /// Where `frame`'s brightness is for a GPU's filters, with them made,
    /// where it is a picture one can follow on.
    fn source(&mut self, frame: &ffmpeg::frame::Video, pp_log: &PpLog) -> Option<OnGpuSource> {
        #[cfg(feature = "cuda-visual-tracking")]
        if let Some(source) = <super::gpu::CudaLooks as super::gpu::DcfDevice>::source(frame) {
            return self.cuda.open(pp_log).map(|_| OnGpuSource::Cuda(source));
        }
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Some(source) = <super::gpu::MetalLooks as super::gpu::DcfDevice>::source(frame) {
            return self.metal.open(pp_log).map(|_| OnGpuSource::Metal(source));
        }
        None
    }

    /// [`follow_by_look`], on the GPU `source` is for.
    fn follow(
        &mut self,
        tracker: &mut ByteTrack,
        source: &OnGpuSource,
    ) -> std::result::Result<Vec<Expected>, String> {
        match *source {
            #[cfg(feature = "cuda-visual-tracking")]
            OnGpuSource::Cuda(ref source) => self
                .cuda
                .looks
                .as_mut()
                .ok_or("no filters on the GPU for this picture")?
                .follow(tracker, source, MIN_PSR, LOOK_DOUBT)
                .map_err(|error| error.to_string()),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            OnGpuSource::Metal(ref source) => self
                .metal
                .looks
                .as_mut()
                .ok_or("no filters on the GPU for this picture")?
                .follow(tracker, source, MIN_PSR, LOOK_DOUBT)
                .map_err(|error| error.to_string()),
        }
    }

    /// Learns `matched` on the GPU `source` is for, forgetting the objects
    /// not among `alive`.
    fn detected(
        &mut self,
        source: &OnGpuSource,
        matched: &[(u64, [f64; 4])],
        alive: &[u64],
    ) -> std::result::Result<(), String> {
        match *source {
            #[cfg(feature = "cuda-visual-tracking")]
            OnGpuSource::Cuda(ref source) => self
                .cuda
                .looks
                .as_mut()
                .ok_or("no filters on the GPU for this picture")?
                .detected(source, matched, alive)
                .map_err(|error| error.to_string()),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            OnGpuSource::Metal(ref source) => self
                .metal
                .looks
                .as_mut()
                .ok_or("no filters on the GPU for this picture")?
                .detected(source, matched, alive)
                .map_err(|error| error.to_string()),
        }
    }

    /// Forgets every filter.
    fn clear(&mut self) {
        #[cfg(feature = "cuda-visual-tracking")]
        if let Some(looks) = self.cuda.looks.as_mut() {
            looks.clear();
        }
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Some(looks) = self.metal.looks.as_mut() {
            looks.clear();
        }
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
        object.look = Some(found.psr);
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
            streams: PerStream::default(),
            numbers: Arc::default(),
            filters: HashMap::new(),
            #[cfg(feature = "cuda")]
            driver: None,
            unreadable_said: false,
            gpus: Gpus::default(),
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
        let (options, numbers) = (self.options, &self.numbers);
        let fresh = || Stream {
            tracker: ByteTrack::new(options, Arc::clone(numbers)),
            pace: Pace::default(),
            last: None,
        };
        let (stream, moved) = self.streams.get(&buf, |_| fresh());
        let cut = buf
            .metadata()
            .is_some_and(|metadata| metadata.get::<SceneCut>().is_some());
        if moved || cut {
            // That stream was sought, or its picture begins a new shot: its
            // objects are not where they were. Their filters go at the next
            // detection, with every object no longer followed; what the new
            // shot holds is numbered anew.
            *stream = fresh();
        }
        let steps = stream.pace.steps(frame.pts());
        let found = buf
            .metadata()
            .and_then(|metadata| metadata.get::<Detections>())
            .cloned();
        // What this picture carries, or, on one a detector let by, which
        // detector the expected objects are put on it as; before any
        // detection there is nothing to expect, and it goes on as it came.
        let found = match (found, &stream.last) {
            (Some(found), _) => Ok(found),
            (None, Some(last)) => Err(last.clone()),
            (None, None) => {
                out.push(buf);
                return Ok(());
            }
        };

        stream.tracker.advance(steps);
        let gpu_source = if self.options.visual {
            self.gpus.source(frame, &self.pp_log)
        } else {
            None
        };
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
            Some(luma) => follow_by_look(&mut stream.tracker, &mut self.filters, luma),
            None => stream.tracker.expected(),
        };
        let expected = match &gpu_source {
            Some(source) => self
                .gpus
                .follow(&mut stream.tracker, source)
                .unwrap_or_else(|error| {
                    pp_warn!(pp_log: &self.pp_log, "following by look on the GPU failed: {error}");
                    expected
                }),
            None => expected,
        };

        let mut found = match found {
            Ok(found) => found,
            Err(Origin { detector, labels }) => {
                let items = expected
                    .into_iter()
                    .filter_map(|expected| {
                        let [x, y, w, h] = expected.tlwh;
                        let (left, top) = ((x / width).max(0.0), (y / height).max(0.0));
                        let (right, bottom) =
                            (((x + w) / width).min(1.0), ((y + h) / height).min(1.0));
                        (right > left && bottom > top).then(|| Detection {
                            track_id: Some(expected.id),
                            look: expected.look,
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
            }
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
        let ids = stream.tracker.associate(&seen);
        stream.last = Some(Origin {
            detector: Arc::clone(&found.detector),
            labels: Arc::clone(&found.labels),
        });
        // The filters are of every stream's objects.
        let alive: HashSet<u64> = self
            .streams
            .values()
            .flat_map(|stream| stream.tracker.ids())
            .collect();
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
            self.filters.retain(|id, _| alive.contains(id));
        }
        if let Some(source) = &gpu_source {
            let matched: Vec<(u64, [f64; 4])> = seen
                .iter()
                .zip(&ids)
                .filter_map(|(seen, id)| Some(((*id)?, seen.tlwh)))
                .collect();
            let alive: Vec<u64> = alive.into_iter().collect();
            if let Err(error) = self.gpus.detected(source, &matched, &alive) {
                pp_warn!(pp_log: &self.pp_log, "learning on the GPU failed: {error}");
            }
        }
        drop(pixels);
        for (item, id) in found.items.iter_mut().zip(ids) {
            item.track_id = id;
        }
        out.push(found.attach_to(buf));
        Ok(())
    }

    /// A seek or a flush: what follows is not where these objects went.
    /// The numbers go on from where they were, so that an object after it
    /// is never taken for one before.
    fn reset(&mut self) {
        self.streams.clear();
        self.filters.clear();
        self.gpus.clear();
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

    /// On a picture carrying a [`SceneCut`], what was followed is let go
    /// of: an object of the new shot where one of the old shot was takes a
    /// new number, and the pictures after the cut that the detector let by
    /// carry nothing of the old shot's.
    #[test]
    fn a_new_shot_starts_over() {
        let mut tracker = ObjectTracker::new(
            "tracker",
            TrackerOptions {
                confirm_after: 1,
                ..TrackerOptions::default()
            },
        );
        let kept = capture(&mut tracker);
        let there = || Some(vec![Detection::new(0, 0.9, 0.4, 0.2, 0.1, 0.5)]);
        for n in 0..5 {
            tracker.consume(picture(n, there())).expect("tracked");
        }
        let cut = picture(5, there());
        let metadata = cut.metadata().cloned().unwrap_or_default().with(SceneCut {
            detector: "cuts".into(),
            score: 100.0,
        });
        tracker
            .consume(cut.with_metadata(metadata))
            .expect("tracked");
        tracker.consume(picture(6, None)).expect("let by");
        let kept = kept.lock().unwrap();
        let id = |n: usize| detections(&kept[n]).unwrap().items[0].track_id;
        assert!(id(4).is_some());
        assert_ne!(id(5), id(4), "numbered anew after the cut");
        assert!(id(5).is_some());
        // The picture after carries what the new shot was seen to hold.
        let after = detections(&kept[6]).unwrap();
        assert!(after.items.iter().all(|item| item.track_id == id(5)));
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

    /// How sure following by look was on each of eleven pictures of a block
    /// detected on the first two, there for five more, then gone — each
    /// picture put where `place` puts it.
    fn looks_on(
        visual: bool,
        place: &dyn Fn(ffmpeg::frame::Video) -> MediaBuffer,
    ) -> Vec<Option<f32>> {
        let mut tracker = ObjectTracker::new(
            "tracker",
            TrackerOptions {
                visual,
                ..TrackerOptions::default()
            },
        );
        let kept = capture(&mut tracker);
        for n in 0..=10u32 {
            let block = if n <= 6 {
                [20 + 2 * n, 30, 24, 32]
            } else {
                [300, 30, 24, 32]
            };
            tracker
                .consume(scene(i64::from(n) * 100, block, n < 2, place))
                .expect("tracked");
        }
        let kept = kept.lock().unwrap();
        (0..=10usize)
            .map(|n| {
                detections(&kept[n])
                    .and_then(|found| found.items.first().map(|item| item.look))
                    .flatten()
            })
            .collect()
    }

    /// What [`looks_on`] gives where the look worked: nothing said on the
    /// pictures detected, sure while the block is there, unsure once gone.
    fn assert_looks(looks: &[Option<f32>]) {
        assert_eq!(looks[..2], [None, None], "detected: no look");
        assert!(
            looks[2..=6].iter().all(|l| l.is_some_and(|l| l > MIN_PSR)),
            "{looks:?}"
        );
        assert!(
            looks[8..].iter().all(|l| l.is_some_and(|l| l < MIN_PSR)),
            "{looks:?}"
        );
    }

    /// A block followed by look says how sure the look was; following by
    /// motion alone says nothing.
    #[test]
    fn following_by_look_says_how_sure_it_was() {
        let by_look = looks_on(true, &MediaBuffer::video);
        eprintln!("looks: {by_look:?}");
        assert_looks(&by_look);
        assert!(
            looks_on(false, &MediaBuffer::video)
                .iter()
                .all(Option::is_none),
            "by motion alone"
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
    /// within a pixel or two, saying how sure it was as the CPU's does.
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
        let looks = looks_on(true, &upload);
        eprintln!("looks on CUDA: {looks:?}");
        assert_looks(&looks);
        let on_cuda = worst_miss_on(true, &upload);
        let in_memory = worst_miss(true);
        eprintln!("worst miss by look: on CUDA {on_cuda}, in system memory {in_memory}");
        assert!(on_cuda < 16.0, "on CUDA, {on_cuda} pixels at worst");
        assert!(
            (on_cuda - in_memory).abs() < 2.0,
            "on CUDA {on_cuda} against {in_memory} in system memory"
        );
    }

    /// [`picture`], of `stream` at `generation` as a mux hands it on.
    fn of_stream(stream: u64, generation: u64, buf: MediaBuffer) -> MediaBuffer {
        use crate::elements::vision::batch::{StreamId, StreamOrigin};
        let metadata = buf
            .metadata()
            .cloned()
            .unwrap_or_default()
            .with(StreamOrigin {
                id: StreamId(stream),
                name: Arc::from(format!("camera {stream}")),
                generation,
            });
        buf.with_metadata(metadata)
    }

    /// The number of the one object `buf` carries.
    fn number(buf: &MediaBuffer) -> Option<u64> {
        detections(buf)?.items.first()?.track_id
    }

    /// Two streams' pictures in turn, the same object at the same place on
    /// each, and a third stream's at a time the others' never reach: each
    /// stream's object has a number of its own, held across its pictures,
    /// and the pictures of one do not count as missed pictures of another.
    /// A seek of one stream numbers its object anew and leaves the others'.
    #[test]
    fn each_stream_is_followed_on_its_own() {
        let mut tracker = ObjectTracker::new("tracker", TrackerOptions::default());
        let kept = capture(&mut tracker);
        let thing = || Some(vec![Detection::new(0, 0.9, 0.4, 0.2, 0.1, 0.5)]);
        for n in 0..6 {
            for stream in [1, 2] {
                tracker
                    .consume(of_stream(stream, 0, picture(1000 * n, thing())))
                    .expect("tracked");
            }
            tracker
                .consume(of_stream(3, 0, picture(1_000_000 + 1000 * n, thing())))
                .expect("tracked");
        }
        let numbers = |kept: &[MediaBuffer], from: usize| -> Vec<u64> {
            kept[from..]
                .iter()
                .map(|buf| number(buf).expect("numbered"))
                .collect()
        };
        let first = numbers(&kept.lock().unwrap(), 3);
        let (one, two, three) = (first[0], first[1], first[2]);
        assert!(one != two && two != three && one != three, "{first:?}");
        assert!(
            first.chunks(3).all(|turn| turn == [one, two, three]),
            "{first:?}"
        );

        let before = kept.lock().unwrap().len();
        for n in 6..9 {
            tracker
                .consume(of_stream(1, 1, picture(1000 * n, thing())))
                .expect("tracked");
            tracker
                .consume(of_stream(2, 0, picture(1000 * n, thing())))
                .expect("tracked");
        }
        let kept = kept.lock().unwrap();
        let after: Vec<Option<u64>> = kept[before..].iter().map(number).collect();
        let renumbered = after[0].expect("numbered at once, as a first sighting is");
        assert_eq!(after[2], Some(renumbered), "{after:?}");
        assert!(![one, two, three].contains(&renumbered), "{after:?}");
        assert_eq!(
            after[1..].iter().step_by(2).collect::<Vec<_>>(),
            [&Some(two); 3]
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
