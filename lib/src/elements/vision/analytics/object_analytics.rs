//! [`ObjectAnalytics`]: objects counted in zones and across lines.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Arc;

use crate::pp_log::{PpLog, pp_info};
use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, OutputContract, PortContract},
    element::{Element, ElementType, element_pp_log},
    elements::vision::batch::PerStream,
    elements::{Analytics, Crossing, Detection, Detections, LineCount, ZoneCount},
    error::Result,
    transform::{Filter, FilterStage, Output, filter_stage},
};

use super::{AnalyticsOptions, Line, ObjectAnalyticsError, StreamAnalyticsOptions, Zone};

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
///
/// After a [`StreamMux`](crate::elements::StreamMux) each stream is watched
/// on its own, by the [`StreamOrigin`](crate::elements::StreamOrigin) its
/// pictures carry — with the zones and lines
/// [`AnalyticsOptions::streams`] gives it, or the shared ones — and its
/// lines' totals are its own. A seek of one stream forgets where that
/// stream's objects were and no other's.
pub struct ObjectAnalytics(FilterStage<Analysing>);

filter_stage!(ObjectAnalytics);

/// A zone, ready to be tested against.
struct ZoneRule {
    name: Arc<str>,
    corners: Vec<(f32, f32)>,
    classes: Option<Vec<usize>>,
    crowded_at: Option<usize>,
}

/// A line, ready to be tested against.
struct LineRule {
    name: Arc<str>,
    start: (f32, f32),
    end: (f32, f32),
    classes: Option<Vec<usize>>,
    margin: f32,
}

/// The zones and lines one stream, or every stream not given its own, is
/// watched with.
struct Rules {
    zones: Vec<ZoneRule>,
    lines: Vec<LineRule>,
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

/// What is kept of one stream: what has crossed its lines, and where its
/// objects were.
struct Watch {
    rules: Arc<Rules>,
    /// For each line, the crossings forward and backward so far.
    totals: Vec<(u64, u64)>,
    last: HashMap<u64, Last>,
    /// Pictures of the stream carrying `Detections` seen so far.
    pictures: u64,
}

/// What an [`ObjectAnalytics`] does with each picture.
struct Analysing {
    name: Arc<str>,
    pp_log: PpLog,
    /// The zones and lines of every stream not given its own.
    rules: Arc<Rules>,
    /// Those of the streams given their own, by the streams' names.
    by_stream: HashMap<Arc<str>, Arc<Rules>>,
    streams: PerStream<Watch>,
}

fn finite(point: (f32, f32)) -> bool {
    point.0.is_finite() && point.1.is_finite()
}

impl Rules {
    /// `zones` and `lines` ready to be tested against.
    ///
    /// # Errors
    ///
    /// As [`ObjectAnalytics::new`]'s.
    fn new(zones: Vec<Zone>, lines: Vec<Line>) -> std::result::Result<Self, ObjectAnalyticsError> {
        let zones = zones
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
        let lines = lines
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
                    })
                },
            )
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(Self { zones, lines })
    }
}

impl ObjectAnalytics {
    /// Analytics over the zones and lines `options` give.
    ///
    /// # Errors
    ///
    /// A zone of fewer than three corners or a corner that is not a
    /// number, a line of no length or an end that is not a number, and a
    /// stream given zones and lines of its own twice.
    pub fn new(
        name: impl Into<String>,
        options: AnalyticsOptions,
    ) -> std::result::Result<Self, ObjectAnalyticsError> {
        let AnalyticsOptions {
            zones,
            lines,
            streams,
        } = options;
        let rules = Rules::new(zones, lines)?;
        let mut by_stream = HashMap::new();
        for StreamAnalyticsOptions {
            stream,
            zones,
            lines,
        } in streams
        {
            let rules = Rules::new(zones, lines)?;
            match by_stream.entry(Arc::<str>::from(stream)) {
                Entry::Occupied(taken) => {
                    return Err(ObjectAnalyticsError::DuplicateStream(
                        taken.key().to_string(),
                    ));
                }
                Entry::Vacant(free) => {
                    free.insert(Arc::new(rules));
                }
            }
        }

        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::ObjectAnalytics, &name, None);
        pp_info!(
            pp_log: &pp_log,
            "opened: {} zones, {} lines, {} streams with their own",
            rules.zones.len(),
            rules.lines.len(),
            by_stream.len()
        );
        Ok(Self(FilterStage::new(Analysing {
            name,
            pp_log,
            rules: Arc::new(rules),
            by_stream,
            streams: PerStream::default(),
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

impl Watch {
    fn new(rules: Arc<Rules>) -> Self {
        Self {
            totals: vec![(0, 0); rules.lines.len()],
            rules,
            last: HashMap::new(),
            pictures: 0,
        }
    }

    /// What `found`, on a picture of `size` pixels, counts as.
    fn analyse(&mut self, found: &Detections, size: (u32, u32)) -> Analytics {
        self.pictures += 1;
        let rules = &*self.rules;
        let zones = rules
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
        let mut crossed: Vec<Vec<Crossing>> = vec![Vec::new(); rules.lines.len()];
        for item in &found.items {
            let Some(track_id) = item.track_id else {
                continue;
            };
            let at = pixels(anchor(item));
            let last = self.last.entry(track_id).or_insert_with(|| Last {
                picture: 0,
                sides: vec![None; rules.lines.len()],
            });
            last.picture = self.pictures;
            for (((line, total), crossed), settled) in rules
                .lines
                .iter()
                .zip(&mut self.totals)
                .zip(&mut crossed)
                .zip(&mut last.sides)
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
                        total.0 += 1;
                    } else {
                        total.1 += 1;
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

        let lines = rules
            .lines
            .iter()
            .zip(&self.totals)
            .zip(crossed)
            .map(|((line, &(forward, backward)), crossed)| LineCount {
                name: Arc::clone(&line.name),
                forward,
                backward,
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
        let (rules, by_stream) = (&self.rules, &self.by_stream);
        let (watch, moved) = self.streams.get(&buf, |origin| {
            let rules = origin
                .and_then(|origin| by_stream.get(&*origin.name))
                .unwrap_or(rules);
            Watch::new(Arc::clone(rules))
        });
        if moved {
            // That stream was sought, as `reset` is for all of them.
            watch.last.clear();
        }
        let analytics = watch.analyse(found, (frame.width(), frame.height()));
        let metadata = buf.metadata().cloned().unwrap_or_default().with(analytics);
        out.push(buf.with_metadata(metadata));
        Ok(())
    }

    /// A seek or a flush: where each object was is not where it comes from
    /// next, so a jump across a line is not a crossing.
    fn reset(&mut self) {
        for watch in self.streams.values_mut() {
            watch.last.clear();
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
                ..AnalyticsOptions::default()
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
                ..AnalyticsOptions::default()
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
                ..AnalyticsOptions::default()
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
                    ..AnalyticsOptions::default()
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
                ..AnalyticsOptions::default()
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

    /// [`picture`] carrying `items`, of `stream` at `generation` as a mux
    /// hands it on.
    fn of_stream(stream: u64, generation: u64, items: Vec<Detection>) -> MediaBuffer {
        use crate::elements::vision::batch::{StreamId, StreamOrigin};
        let buf = picture(Some(items));
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

    /// Two streams' pictures in turn, one object numbered 1 on each. The
    /// first stream's crosses its line, and that stream alone counts it;
    /// the second is watched with a line of its own, which the first does
    /// not have. A seek of the first is not a crossing back, and leaves the
    /// second's object where it was: it crosses its line after the seek.
    #[test]
    fn each_stream_is_watched_on_its_own() {
        let middle = Line::new("middle", (0.0, 0.5), (1.0, 0.5));
        let mut element = ObjectAnalytics::new(
            "analytics",
            AnalyticsOptions {
                lines: vec![middle],
                streams: vec![StreamAnalyticsOptions {
                    stream: "camera 2".into(),
                    lines: vec![Line::new("upright", (0.5, 0.0), (0.5, 1.0))],
                    ..StreamAnalyticsOptions::default()
                }],
                ..AnalyticsOptions::default()
            },
        )
        .expect("lines");
        let kept = capture(&mut element);
        let at = |x, y| vec![standing(0, Some(1), x, y)];
        for y in [0.3, 0.7, 0.7] {
            element
                .consume(of_stream(1, 0, at(0.3, y)))
                .expect("analysed");
            element
                .consume(of_stream(2, 0, at(0.3, 0.3)))
                .expect("analysed");
        }
        // Stream 1 sought back above its line; stream 2 goes on, across its.
        element
            .consume(of_stream(1, 1, at(0.3, 0.3)))
            .expect("analysed");
        element
            .consume(of_stream(2, 0, at(0.7, 0.3)))
            .expect("analysed");

        let kept = kept.lock().unwrap();
        let line = |n: usize| analytics(&kept[n]).lines[0].clone();
        assert_eq!(&*line(0).name, "middle");
        assert_eq!(
            &*line(1).name,
            "upright",
            "its own line in place of the shared"
        );
        assert_eq!(line(2).crossed.len(), 1, "stream 1 crossed");
        assert_eq!(line(4).forward, 1, "and its total stays");
        for n in [1, 3, 5] {
            assert_eq!(
                (line(n).forward, line(n).backward),
                (0, 0),
                "stream 2 has crossed nothing yet, picture {n}"
            );
        }
        assert!(line(6).crossed.is_empty(), "a seek is not a crossing back");
        assert_eq!((line(6).forward, line(6).backward), (1, 0));
        assert_eq!(line(7).crossed.len(), 1, "stream 2 kept where it was");
    }

    /// One stream given zones and lines of its own twice is refused.
    #[test]
    fn a_stream_given_its_own_twice_is_refused() {
        let own = StreamAnalyticsOptions {
            stream: "door".into(),
            ..StreamAnalyticsOptions::default()
        };
        assert!(matches!(
            ObjectAnalytics::new(
                "analytics",
                AnalyticsOptions {
                    streams: vec![own.clone(), own],
                    ..AnalyticsOptions::default()
                },
            ),
            Err(ObjectAnalyticsError::DuplicateStream(stream)) if stream == "door"
        ));
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
                    ..AnalyticsOptions::default()
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
                    ..AnalyticsOptions::default()
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
                    ..AnalyticsOptions::default()
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
                    ..AnalyticsOptions::default()
                },
            )
            .is_err()
        );
    }
}
