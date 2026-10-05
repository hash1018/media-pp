//! [`ObjectAnalytics`]: objects counted in zones and across lines.

use std::collections::HashMap;
use std::sync::Arc;

use crate::pp_log::{PpLog, pp_info};
use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, OutputContract, PortContract},
    element::{Element, ElementType, element_pp_log},
    elements::{Analytics, Crossing, Detection, Detections, LineCount, ZoneCount},
    error::Result,
    transform::{Filter, FilterStage, Output, filter_stage},
};

use super::{AnalyticsOptions, Line, ObjectAnalyticsError, Zone};

/// How many pictures an object may go unseen before its last place is
/// forgotten — a tracker forgets it sooner, so this only bounds memory.
const FORGET_AFTER: u64 = 300;

/// Counts the objects each picture's [`Detections`] place inside zones of
/// the picture, and the objects crossing lines across it, and hands the
/// picture on carrying what it counted as [`Analytics`] beside them —
/// DeepStream's `nvdsanalytics`, for region-of-interest counts,
/// overcrowding and line crossing.
///
/// It goes after a detector, and for lines after an
/// [`ObjectTracker`](crate::elements::ObjectTracker):
/// `Detector -> ObjectTracker -> ObjectAnalytics`. A line is crossed by an
/// object followed from one side of it to the other, which only a tracker's
/// numbers say; a zone counts whatever is in it. The pictures a tracker
/// filled in count as the detector's do, so an object crossing between two
/// detections is counted on the picture it crossed on.
///
/// It reads metadata alone, never a pixel, so it takes pictures wherever
/// they live and hands each on as it came. A picture carrying no
/// `Detections` goes on carrying no `Analytics`. A seek forgets where each
/// object was, so a jump is not counted as a crossing; the totals carry on.
pub struct ObjectAnalytics(FilterStage<Analysing>);

filter_stage!(ObjectAnalytics);

/// A zone, ready to be tested against.
struct ZoneRule {
    name: Arc<str>,
    corners: Vec<(f32, f32)>,
    classes: Option<Vec<usize>>,
    crowded_at: Option<usize>,
}

/// A line, and what has crossed it.
struct LineRule {
    name: Arc<str>,
    start: (f32, f32),
    end: (f32, f32),
    classes: Option<Vec<usize>>,
    margin: f32,
    forward: u64,
    backward: u64,
}

/// The side of a line an object was last clearly on — at least the
/// line's margin away — and where it was then, in pixels.
#[derive(Debug, Clone, Copy)]
struct Settled {
    right: bool,
    at: (f32, f32),
}

/// What is remembered of an object: the picture it was last seen on, and
/// for each line the side it settled on.
#[derive(Debug, Clone)]
struct Last {
    picture: u64,
    sides: Vec<Option<Settled>>,
}

/// What an [`ObjectAnalytics`] does with each picture.
struct Analysing {
    name: Arc<str>,
    pp_log: PpLog,
    zones: Vec<ZoneRule>,
    lines: Vec<LineRule>,
    last: HashMap<u64, Last>,
    /// Pictures carrying `Detections` seen so far.
    pictures: u64,
}

fn finite(point: (f32, f32)) -> bool {
    point.0.is_finite() && point.1.is_finite()
}

impl ObjectAnalytics {
    /// Analytics over the zones and lines `options` give.
    ///
    /// # Errors
    ///
    /// A zone of fewer than three corners or a corner that is not a
    /// number, and a line of no length or an end that is not a number.
    pub fn new(
        name: impl Into<String>,
        options: AnalyticsOptions,
    ) -> std::result::Result<Self, ObjectAnalyticsError> {
        let zones = options
            .zones
            .into_iter()
            .map(
                |Zone {
                     name,
                     corners,
                     classes,
                     crowded_at,
                 }| {
                    let invalid = |reason| ObjectAnalyticsError::InvalidZone {
                        zone: name.clone(),
                        reason,
                    };
                    if corners.len() < 3 {
                        return Err(invalid("a zone needs three corners or more"));
                    }
                    if !corners.iter().copied().all(finite) {
                        return Err(invalid("a corner is not a number"));
                    }
                    Ok(ZoneRule {
                        name: name.into(),
                        corners,
                        classes,
                        crowded_at,
                    })
                },
            )
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let lines = options
            .lines
            .into_iter()
            .map(
                |Line {
                     name,
                     start,
                     end,
                     classes,
                     margin,
                 }| {
                    let invalid = |reason| ObjectAnalyticsError::InvalidLine {
                        line: name.clone(),
                        reason,
                    };
                    if !finite(start) || !finite(end) {
                        return Err(invalid("an end is not a number"));
                    }
                    if start == end {
                        return Err(invalid("its start and end are the same point"));
                    }
                    if !margin.is_finite() || margin < 0.0 {
                        return Err(invalid("its margin is not a number of zero or more"));
                    }
                    Ok(LineRule {
                        name: name.into(),
                        start,
                        end,
                        classes,
                        margin,
                        forward: 0,
                        backward: 0,
                    })
                },
            )
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::ObjectAnalytics, &name, None);
        pp_info!(
            pp_log: &pp_log,
            "opened: {} zones, {} lines",
            zones.len(),
            lines.len()
        );
        Ok(Self(FilterStage::new(Analysing {
            name,
            pp_log,
            zones,
            lines,
            last: HashMap::new(),
            pictures: 0,
        })))
    }
}

/// Where an object is: the bottom middle of its box.
fn anchor(detection: &Detection) -> (f32, f32) {
    (
        detection.x + detection.width / 2.0,
        detection.y + detection.height,
    )
}

fn counts(classes: &Option<Vec<usize>>, class_id: usize) -> bool {
    classes
        .as_ref()
        .is_none_or(|classes| classes.contains(&class_id))
}

/// Whether `point` is inside the polygon `corners` — the even-odd rule.
fn inside(corners: &[(f32, f32)], point: (f32, f32)) -> bool {
    let (x, y) = point;
    let mut inside = false;
    let mut previous = corners[corners.len() - 1];
    for &corner in corners {
        let ((x1, y1), (x2, y2)) = (previous, corner);
        if (y1 > y) != (y2 > y) && x < x1 + (y - y1) * (x2 - x1) / (y2 - y1) {
            inside = !inside;
        }
        previous = corner;
    }
    inside
}

/// Which side of the line from `start` to `end` `point` is on: positive to
/// its right as seen going from start to end, in a picture whose `y` grows
/// downward; zero on it. Over the line's length, the distance from it.
fn side(start: (f32, f32), end: (f32, f32), point: (f32, f32)) -> f32 {
    (end.0 - start.0) * (point.1 - start.1) - (end.1 - start.1) * (point.0 - start.0)
}

/// Whether the path from `from` to `to` passes the line from `start` to
/// `end` within its length rather than beyond an end: the ends are on
/// either side of the path, or on it.
fn within(start: (f32, f32), end: (f32, f32), from: (f32, f32), to: (f32, f32)) -> bool {
    side(from, to, start) * side(from, to, end) <= 0.0
}

impl Analysing {
    /// What `found`, on a picture of `size` pixels, counts as.
    fn analyse(&mut self, found: &Detections, size: (u32, u32)) -> Analytics {
        self.pictures += 1;
        let zones = self
            .zones
            .iter()
            .map(|zone| {
                let objects: Vec<usize> = found
                    .items
                    .iter()
                    .enumerate()
                    .filter(|(_, item)| {
                        counts(&zone.classes, item.class_id) && inside(&zone.corners, anchor(item))
                    })
                    .map(|(index, _)| index)
                    .collect();
                ZoneCount {
                    name: Arc::clone(&zone.name),
                    crowded: zone.crowded_at.is_some_and(|at| objects.len() >= at),
                    objects,
                }
            })
            .collect();

        // In pixels, so that a margin is a distance whichever way a line
        // runs across a picture that is not square.
        let (width, height) = (size.0 as f32, size.1 as f32);
        let pixels = |(x, y): (f32, f32)| (x * width, y * height);
        let mut crossed: Vec<Vec<Crossing>> = vec![Vec::new(); self.lines.len()];
        for item in &found.items {
            let Some(track_id) = item.track_id else {
                continue;
            };
            let at = pixels(anchor(item));
            let last = self.last.entry(track_id).or_insert_with(|| Last {
                picture: 0,
                sides: vec![None; self.lines.len()],
            });
            last.picture = self.pictures;
            for ((line, crossed), settled) in
                self.lines.iter_mut().zip(&mut crossed).zip(&mut last.sides)
            {
                if !counts(&line.classes, item.class_id) {
                    continue;
                }
                let (start, end) = (pixels(line.start), pixels(line.end));
                let length = ((end.0 - start.0).powi(2) + (end.1 - start.1).powi(2)).sqrt();
                let distance = side(start, end, at) / length;
                // Within the margin the object is on neither side yet: a
                // box trembling about the line moves nothing.
                if distance.abs() < line.margin * item.height * height {
                    continue;
                }
                let right = distance >= 0.0;
                if let Some(was) = *settled
                    && was.right != right
                    && within(start, end, was.at, at)
                {
                    if right {
                        line.forward += 1;
                    } else {
                        line.backward += 1;
                    }
                    crossed.push(Crossing {
                        track_id,
                        class_id: item.class_id,
                        forward: right,
                    });
                }
                *settled = Some(Settled { right, at });
            }
        }
        let now = self.pictures;
        self.last
            .retain(|_, last| now - last.picture <= FORGET_AFTER);

        let lines = self
            .lines
            .iter()
            .zip(crossed)
            .map(|(line, crossed)| LineCount {
                name: Arc::clone(&line.name),
                forward: line.forward,
                backward: line.backward,
                crossed,
            })
            .collect();
        Analytics { zones, lines }
    }
}

impl Element for Analysing {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::ObjectAnalytics
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Filter for Analysing {
    /// Pictures wherever they live: only their metadata is read.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::any_frame(MediaKind::VideoFrame))
    }

    fn output_contract(&self) -> OutputContract {
        OutputContract::Passthrough
    }

    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        let (MediaBuffer::Video(frame), Some(found)) = (
            &buf,
            buf.metadata()
                .and_then(|metadata| metadata.get::<Detections>()),
        ) else {
            out.push(buf);
            return Ok(());
        };
        let analytics = self.analyse(found, (frame.width(), frame.height()));
        let metadata = buf.metadata().cloned().unwrap_or_default().with(analytics);
        out.push(buf.with_metadata(metadata));
        Ok(())
    }

    /// A seek or a flush: where each object was is not where it comes from
    /// next, so a jump across a line is not a crossing.
    fn reset(&mut self) {
        self.last.clear();
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

    /// A picture carrying `items`, or nothing.
    fn picture(items: Option<Vec<Detection>>) -> MediaBuffer {
        let frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::RGB24, 64, 64);
        let buf = MediaBuffer::video(frame);
        match items {
            Some(items) => Detections::new("detector", Arc::from([]), items).attach_to(buf),
            None => buf,
        }
    }

    /// An object of `class_id`, numbered `track_id`, its box 0.1 wide and
    /// 0.2 high standing on (`x`, `y`).
    fn standing(class_id: usize, track_id: Option<u64>, x: f32, y: f32) -> Detection {
        Detection {
            track_id,
            ..Detection::new(class_id, 0.9, x - 0.05, y - 0.2, 0.1, 0.2)
        }
    }

    fn analytics(buf: &MediaBuffer) -> &Analytics {
        buf.metadata()
            .and_then(|metadata| metadata.get::<Analytics>())
            .expect("carries Analytics")
    }

    /// Run `items` through, a picture each, and what each came out with.
    fn run(mut element: ObjectAnalytics, pictures: Vec<Vec<Detection>>) -> Vec<Analytics> {
        let kept = capture(&mut element);
        for items in pictures {
            element.consume(picture(Some(items))).expect("analysed");
        }
        kept.lock()
            .unwrap()
            .iter()
            .map(|buf| analytics(buf).clone())
            .collect()
    }

    /// A zone counts what stands in it, of its classes, and says when it is
    /// crowded; the detections go on beside the analytics.
    #[test]
    fn a_zone_counts_what_stands_in_it() {
        let zone = Zone {
            classes: Some(vec![0]),
            crowded_at: Some(2),
            ..Zone::new("left", vec![(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)])
        };
        let mut element = ObjectAnalytics::new(
            "analytics",
            AnalyticsOptions {
                zones: vec![zone],
                lines: vec![],
            },
        )
        .expect("a zone");
        let kept = capture(&mut element);
        element
            .consume(picture(Some(vec![
                standing(0, None, 0.2, 0.5),
                standing(0, None, 0.8, 0.5),
                standing(1, None, 0.3, 0.5),
            ])))
            .expect("analysed");
        element
            .consume(picture(Some(vec![
                standing(0, None, 0.2, 0.5),
                standing(0, None, 0.4, 0.9),
            ])))
            .expect("analysed");
        let kept = kept.lock().unwrap();
        let first = &analytics(&kept[0]).zones[0];
        assert_eq!(&*first.name, "left");
        assert_eq!(first.objects, vec![0], "inside and of the class only");
        assert!(!first.crowded);
        let second = &analytics(&kept[1]).zones[0];
        assert_eq!(second.objects, vec![0, 1]);
        assert!(second.crowded, "two make it crowded");
        assert_eq!(
            kept[1]
                .metadata()
                .and_then(|m| m.get::<Detections>())
                .map(|found| found.items.len()),
            Some(2),
            "the detections go on"
        );
    }

    /// A line counts each crossing once, each way apart, on the picture it
    /// happens; totals carry on from picture to picture.
    #[test]
    fn a_line_counts_crossings_each_way() {
        let line = Line::new("middle", (0.0, 0.5), (1.0, 0.5));
        let analytics = ObjectAnalytics::new(
            "analytics",
            AnalyticsOptions {
                zones: vec![],
                lines: vec![line],
            },
        )
        .expect("a line");
        let down = |y| standing(0, Some(7), 0.5, y);
        let seen = run(
            analytics,
            vec![
                vec![down(0.3)],
                vec![down(0.45)],
                vec![down(0.55)],
                vec![down(0.7)],
                vec![down(0.4)],
            ],
        );
        let lines: Vec<&LineCount> = seen.iter().map(|a| &a.lines[0]).collect();
        assert!(lines[1].crossed.is_empty(), "not across yet");
        assert_eq!(
            lines[2].crossed,
            vec![Crossing {
                track_id: 7,
                class_id: 0,
                forward: true
            }],
            "drawn left to right, moving down is forward"
        );
        assert!(lines[3].crossed.is_empty(), "counted once");
        assert_eq!((lines[3].forward, lines[3].backward), (1, 0));
        assert!(!lines[4].crossed[0].forward);
        assert_eq!((lines[4].forward, lines[4].backward), (1, 1));
    }

    /// Only a numbered object crosses, only within the line's length, and
    /// only of the line's classes; one stopping on the line has crossed
    /// once it is past it.
    #[test]
    fn a_line_counts_only_what_crosses_it() {
        let line = Line {
            classes: Some(vec![0]),
            ..Line::new("short", (0.4, 0.5), (0.6, 0.5))
        };
        let analytics = ObjectAnalytics::new(
            "analytics",
            AnalyticsOptions {
                zones: vec![],
                lines: vec![line],
            },
        )
        .expect("a line");
        let seen = run(
            analytics,
            vec![
                vec![
                    standing(0, None, 0.5, 0.3),
                    standing(0, Some(1), 0.9, 0.3),
                    standing(1, Some(2), 0.5, 0.3),
                    standing(0, Some(3), 0.5, 0.3),
                ],
                vec![
                    standing(0, None, 0.5, 0.7),
                    standing(0, Some(1), 0.9, 0.7),
                    standing(1, Some(2), 0.5, 0.7),
                    standing(0, Some(3), 0.5, 0.5),
                ],
                vec![standing(0, Some(3), 0.5, 0.7)],
            ],
        );
        assert!(
            seen[1].lines[0].crossed.is_empty(),
            "not the unnumbered, nor past its end, nor another class, nor one \
             on the line: {:?}",
            seen[1].lines[0].crossed
        );
        assert_eq!(
            seen[2].lines[0]
                .crossed
                .iter()
                .map(|crossing| crossing.track_id)
                .collect::<Vec<_>>(),
            vec![3],
            "it has crossed once past it"
        );
    }

    /// A box trembling about a line within its margin crosses nothing, and
    /// counts once it is clear of it; without a margin every tremble is a
    /// crossing.
    #[test]
    fn a_box_trembling_about_a_line_does_not_cross_it() {
        // Boxes 0.2 of the 64-pixel picture high: a margin of a tenth is
        // 1.28 pixels, 0.02 of the picture.
        let trembling: Vec<Vec<Detection>> = [0.3, 0.51, 0.49, 0.51, 0.49, 0.7]
            .iter()
            .map(|&y| vec![standing(0, Some(1), 0.5, y)])
            .collect();
        let count = |margin| {
            let line = Line {
                margin,
                ..Line::new("middle", (0.0, 0.5), (1.0, 0.5))
            };
            let analytics = ObjectAnalytics::new(
                "analytics",
                AnalyticsOptions {
                    zones: vec![],
                    lines: vec![line],
                },
            )
            .expect("a line");
            let seen = run(analytics, trembling.clone());
            let last = &seen[seen.len() - 1].lines[0];
            (last.forward, last.backward)
        };
        assert_eq!(count(0.1), (1, 0), "once, as it gets clear below");
        assert_eq!(count(0.0), (3, 2), "every tremble");
    }

    /// A seek forgets where each object was: the jump across the line is
    /// not a crossing.
    #[test]
    fn a_seek_is_not_a_crossing() {
        let line = Line::new("middle", (0.0, 0.5), (1.0, 0.5));
        let mut element = ObjectAnalytics::new(
            "analytics",
            AnalyticsOptions {
                zones: vec![],
                lines: vec![line],
            },
        )
        .expect("a line");
        let kept = capture(&mut element);
        element
            .consume(picture(Some(vec![standing(0, Some(1), 0.5, 0.3)])))
            .expect("analysed");
        element.0.inner_mut().reset();
        element
            .consume(picture(Some(vec![standing(0, Some(1), 0.5, 0.7)])))
            .expect("analysed");
        assert_eq!(analytics(&kept.lock().unwrap()[1]).lines[0].forward, 0);
    }

    /// A picture nobody looked at goes on as it came, and so does what is
    /// not a picture.
    #[test]
    fn what_carries_no_detections_goes_through() {
        let mut element =
            ObjectAnalytics::new("analytics", AnalyticsOptions::default()).expect("nothing");
        let kept = capture(&mut element);
        element.consume(picture(None)).expect("passed");
        element
            .consume(MediaBuffer::audio(ffmpeg::frame::Audio::empty()))
            .expect("passed");
        let kept = kept.lock().unwrap();
        assert_eq!(kept.len(), 2);
        assert!(
            kept[0]
                .metadata()
                .is_none_or(|m| m.get::<Analytics>().is_none())
        );
    }

    #[test]
    fn a_zone_or_line_that_cannot_be_counted_is_refused() {
        let zone = Zone::new("flat", vec![(0.0, 0.0), (1.0, 1.0)]);
        assert!(matches!(
            ObjectAnalytics::new(
                "analytics",
                AnalyticsOptions {
                    zones: vec![zone],
                    lines: vec![],
                },
            ),
            Err(ObjectAnalyticsError::InvalidZone { zone, .. }) if zone == "flat"
        ));
        let line = Line::new("dot", (0.5, 0.5), (0.5, 0.5));
        assert!(matches!(
            ObjectAnalytics::new(
                "analytics",
                AnalyticsOptions {
                    zones: vec![],
                    lines: vec![line],
                },
            ),
            Err(ObjectAnalyticsError::InvalidLine { line, .. }) if line == "dot"
        ));
        let below = Line {
            margin: -0.1,
            ..Line::new("below", (0.0, 0.5), (1.0, 0.5))
        };
        assert!(
            ObjectAnalytics::new(
                "analytics",
                AnalyticsOptions {
                    zones: vec![],
                    lines: vec![below],
                },
            )
            .is_err()
        );
        let nan = Line::new("nan", (f32::NAN, 0.5), (0.5, 0.5));
        assert!(
            ObjectAnalytics::new(
                "analytics",
                AnalyticsOptions {
                    zones: vec![],
                    lines: vec![nan],
                },
            )
            .is_err()
        );
    }
}
