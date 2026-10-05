//! [`ObjectTracker`]: what a detector found, numbered across pictures.

use std::sync::Arc;

use crate::pp_log::{PpLog, pp_info};
use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, OutputContract, PortContract},
    element::{Element, ElementType, element_pp_log},
    elements::{Detection, Detections},
    error::Result,
    transform::{Filter, FilterStage, Output, filter_stage},
};

use super::TrackerOptions;
use super::byte_track::{ByteTrack, Seen};

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
/// reads and writes metadata alone, never a pixel, so it takes pictures
/// wherever they live — system memory, CUDA, any GPU — and hands each on
/// as it came. A picture before any detection has been seen goes on
/// carrying nothing.
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

        let Some(mut found) = found else {
            // Let by unlooked-at: where the objects are expected, once there
            // are any.
            let Some(Origin { detector, labels }) = self.last.clone() else {
                out.push(buf);
                return Ok(());
            };
            let items = self
                .tracker
                .expect(steps)
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
        let ids = self.tracker.update(&seen, steps);
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
