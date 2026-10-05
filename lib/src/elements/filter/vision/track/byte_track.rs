//! ByteTrack (Zhang et al., 2022): following objects from picture to
//! picture by where a Kalman filter expects each to be, matching confident
//! detections first and then the unconfident ones a partly hidden object
//! still gets, which other trackers throw away and lose it with.

use super::TrackerOptions;
use super::assign::assign;
use super::kalman::{Kalman, Measurement};

/// A detection as the tracker reads it: a box in pixels, by its top-left
/// corner, width and height.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Seen {
    pub(super) class_id: usize,
    pub(super) score: f32,
    pub(super) tlwh: [f64; 4],
}

/// Where a followed object is expected, on a picture nothing was detected
/// in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Expected {
    pub(super) id: u64,
    pub(super) class_id: usize,
    pub(super) score: f32,
    pub(super) tlwh: [f64; 4],
}

/// Matched cost a first-stage pair may have, as `1 - IoU · score`:
/// ByteTrack's `match_thresh`.
const FIRST_LIMIT: f64 = 0.8;
/// The same for the second stage, unconfident detections, by IoU alone.
const SECOND_LIMIT: f64 = 0.5;
/// The same for a track not yet confirmed, against the confident left over.
const UNCONFIRMED_LIMIT: f64 = 0.7;
/// Overlap past which a followed and a lost track are the same object, the
/// younger a duplicate.
const DUPLICATE_IOU: f64 = 0.85;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Matched on the last picture it could have been.
    Tracked,
    /// Unmatched lately, kept a while in case it comes back.
    Lost,
}

#[derive(Debug, Clone)]
struct Track {
    /// Its number once confirmed; until then, `None`.
    id: Option<u64>,
    kalman: Kalman,
    state: State,
    class_id: usize,
    score: f32,
    /// The picture it was last matched on.
    seen_at: u64,
    /// The picture it started on.
    started_at: u64,
    /// How many pictures it has been detected on.
    sightings: u32,
}

impl Track {
    fn tlwh(&self) -> [f64; 4] {
        to_tlwh(self.kalman.measurement())
    }
}

fn to_xywh([x, y, w, h]: [f64; 4]) -> Measurement {
    [x + w / 2.0, y + h / 2.0, w, h]
}

fn to_tlwh([x, y, w, h]: Measurement) -> [f64; 4] {
    [x - w / 2.0, y - h / 2.0, w, h]
}

/// Intersection over union of two boxes by their top-left corner and size.
pub(super) fn iou(a: [f64; 4], b: [f64; 4]) -> f64 {
    let w = (a[0] + a[2]).min(b[0] + b[2]) - a[0].max(b[0]);
    let h = (a[1] + a[3]).min(b[1] + b[3]) - a[1].max(b[1]);
    let inter = w.max(0.0) * h.max(0.0);
    let union = a[2] * a[3] + b[2] * b[3] - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

/// Follows objects through the pictures it is handed.
#[derive(Debug, Clone)]
pub(super) struct ByteTrack {
    options: TrackerOptions,
    tracks: Vec<Track>,
    /// Pictures so far, counting those detected on and those not.
    picture: u64,
    /// Whether it has been handed a detection at all: the first picture's
    /// objects are confirmed at once, as ByteTrack's are.
    started: bool,
    next_id: u64,
}

impl ByteTrack {
    pub(super) fn new(options: TrackerOptions) -> Self {
        Self {
            options,
            tracks: Vec::new(),
            picture: 0,
            started: false,
            next_id: 1,
        }
    }

    /// Moves every track on by `steps` pictures.
    fn advance(&mut self, steps: u64) {
        self.picture += steps;
        for track in &mut self.tracks {
            for _ in 0..steps {
                if track.state != State::Tracked {
                    track.kalman.hold_size();
                }
                track.kalman.predict();
            }
        }
    }

    /// A cost matrix of `tracks` against `seen`, `1 - IoU` — times the
    /// detection's score where `fuse` — and refused across classes where
    /// the options keep classes apart.
    fn costs(&self, tracks: &[usize], seen: &[Seen], from: &[usize], fuse: bool) -> Vec<f64> {
        let mut cost = Vec::with_capacity(tracks.len() * from.len());
        for &t in tracks {
            let track = &self.tracks[t];
            let tlwh = track.tlwh();
            for &d in from {
                let detection = &seen[d];
                if self.options.per_class && detection.class_id != track.class_id {
                    cost.push(f64::INFINITY);
                    continue;
                }
                let mut similarity = iou(tlwh, detection.tlwh);
                if fuse {
                    similarity *= f64::from(detection.score);
                }
                cost.push(1.0 - similarity);
            }
        }
        cost
    }

    /// Pairs `tracks` with `from`, returning the pairs as indices into
    /// `self.tracks` and `seen`, and what of each was left.
    fn pair(
        &self,
        tracks: &[usize],
        seen: &[Seen],
        from: &[usize],
        fuse: bool,
        limit: f64,
    ) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
        let cost = self.costs(tracks, seen, from, fuse);
        let pairs: Vec<(usize, usize)> = assign(&cost, tracks.len(), from.len(), limit)
            .into_iter()
            .map(|(t, d)| (tracks[t], from[d]))
            .collect();
        let left_tracks = tracks
            .iter()
            .copied()
            .filter(|t| !pairs.iter().any(|(pt, _)| pt == t))
            .collect();
        let left_seen = from
            .iter()
            .copied()
            .filter(|d| !pairs.iter().any(|(_, pd)| pd == d))
            .collect();
        (pairs, left_tracks, left_seen)
    }

    fn confirm(&mut self, track: usize) -> u64 {
        if let Some(id) = self.tracks[track].id {
            return id;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.tracks[track].id = Some(id);
        id
    }

    fn matched(&mut self, track: usize, seen: &Seen) {
        let picture = self.picture;
        let track = &mut self.tracks[track];
        track.kalman.update(to_xywh(seen.tlwh));
        track.state = State::Tracked;
        track.score = seen.score;
        track.seen_at = picture;
        track.sightings += 1;
    }

    /// A picture `steps` on from the last, with what was detected on it:
    /// the number of the object each detection is, where it is one this
    /// follows.
    pub(super) fn update(&mut self, seen: &[Seen], steps: u64) -> Vec<Option<u64>> {
        self.advance(steps);
        let options = self.options;
        let mut ids = vec![None; seen.len()];

        let high: Vec<usize> = (0..seen.len())
            .filter(|&d| seen[d].score >= options.high_score)
            .collect();
        let low: Vec<usize> = (0..seen.len())
            .filter(|&d| seen[d].score >= options.low_score && seen[d].score < options.high_score)
            .collect();
        let (confirmed, unconfirmed): (Vec<usize>, Vec<usize>) =
            (0..self.tracks.len()).partition(|&t| self.tracks[t].id.is_some());

        // First, confirmed tracks — followed or lost — against the confident
        // detections, by overlap weighed by score.
        let (pairs, left, left_high) = self.pair(&confirmed, seen, &high, true, FIRST_LIMIT);
        for (t, d) in pairs {
            self.matched(t, &seen[d]);
            ids[d] = Some(self.confirm(t));
        }

        // Then those still followed against the unconfident: an object
        // partly hidden, whose score has dropped, is still that object.
        let followed: Vec<usize> = left
            .iter()
            .copied()
            .filter(|&t| self.tracks[t].state == State::Tracked)
            .collect();
        let (pairs, unmatched, _) = self.pair(&followed, seen, &low, false, SECOND_LIMIT);
        for (t, d) in pairs {
            self.matched(t, &seen[d]);
            ids[d] = Some(self.confirm(t));
        }
        for t in unmatched {
            self.tracks[t].state = State::Lost;
        }

        // Tracks begun lately, against the confident left over: matched
        // often enough, they are confirmed; unmatched, they were noise.
        let (pairs, stale, left_high) =
            self.pair(&unconfirmed, seen, &left_high, true, UNCONFIRMED_LIMIT);
        for (t, d) in pairs {
            self.matched(t, &seen[d]);
            if self.tracks[t].sightings >= options.confirm_after {
                ids[d] = Some(self.confirm(t));
            }
        }

        // Whatever confident detection is left begins a track — confirmed at
        // once on the first picture, as everything is there, and where one
        // sighting is enough.
        let first = !self.started || options.confirm_after <= 1;
        for d in left_high {
            if seen[d].score < options.new_track_score {
                continue;
            }
            self.tracks.push(Track {
                id: None,
                kalman: Kalman::new(to_xywh(seen[d].tlwh)),
                state: State::Tracked,
                class_id: seen[d].class_id,
                score: seen[d].score,
                seen_at: self.picture,
                started_at: self.picture,
                sightings: 1,
            });
            if first {
                let t = self.tracks.len() - 1;
                ids[d] = Some(self.confirm(t));
            }
        }
        self.started = true;

        // Forget the noise, and what has been lost too long.
        let picture = self.picture;
        let lost_for = u64::from(options.lost_pictures);
        let stale: Vec<usize> = stale;
        let mut index = 0;
        self.tracks.retain(|track| {
            let keep = !stale.contains(&index)
                && !(track.state == State::Lost && picture - track.seen_at > lost_for);
            index += 1;
            keep
        });
        self.drop_duplicates();
        ids
    }

    /// Where a lost track and a followed one are the same object, keeps
    /// the one followed longer.
    fn drop_duplicates(&mut self) {
        let mut gone = Vec::new();
        for a in 0..self.tracks.len() {
            for b in 0..self.tracks.len() {
                let (ta, tb) = (&self.tracks[a], &self.tracks[b]);
                if a == b || ta.state != State::Tracked || tb.state != State::Lost {
                    continue;
                }
                if self.options.per_class && ta.class_id != tb.class_id {
                    continue;
                }
                if iou(ta.tlwh(), tb.tlwh()) > DUPLICATE_IOU {
                    let younger = |t: &Track| self.picture - t.started_at;
                    gone.push(if younger(ta) > younger(tb) { b } else { a });
                }
            }
        }
        let mut index = 0;
        self.tracks.retain(|_| {
            let keep = !gone.contains(&index);
            index += 1;
            keep
        });
    }

    /// A picture `steps` on with nothing detected on it: where each
    /// followed, confirmed object is expected to be.
    pub(super) fn expect(&mut self, steps: u64) -> Vec<Expected> {
        self.advance(steps);
        self.tracks
            .iter()
            .filter(|track| track.state == State::Tracked)
            .filter_map(|track| {
                Some(Expected {
                    id: track.id?,
                    class_id: track.class_id,
                    score: track.score,
                    tlwh: track.tlwh(),
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seen(class_id: usize, score: f32, x: f64, y: f64) -> Seen {
        Seen {
            class_id,
            score,
            tlwh: [x, y, 40.0, 80.0],
        }
    }

    fn tracker() -> ByteTrack {
        ByteTrack::new(TrackerOptions::default())
    }

    #[test]
    fn a_moving_object_keeps_its_number() {
        let mut tracker = tracker();
        let first = tracker.update(&[seen(0, 0.9, 100.0, 100.0)], 1);
        let id = first[0].expect("confirmed on the first picture");
        for step in 1..50 {
            let ids = tracker.update(&[seen(0, 0.9, 100.0 + 4.0 * f64::from(step), 100.0)], 1);
            assert_eq!(ids, vec![Some(id)], "picture {step}");
        }
    }

    /// Two people walking past each other, one a little further back, keep
    /// their own numbers.
    #[test]
    fn objects_crossing_keep_their_own_numbers() {
        let mut tracker = tracker();
        let ids = tracker.update(&[seen(0, 0.9, 100.0, 100.0), seen(0, 0.9, 300.0, 110.0)], 1);
        let (left, right) = (ids[0].unwrap(), ids[1].unwrap());
        for step in 1..=50 {
            let s = f64::from(step) * 4.0;
            let ids = tracker.update(
                &[
                    seen(0, 0.9, 100.0 + s, 100.0),
                    seen(0, 0.9, 300.0 - s, 110.0),
                ],
                1,
            );
            assert_eq!(ids, vec![Some(left), Some(right)], "picture {step}");
        }
    }

    /// Partly hidden, an object's score drops below the confident line for a
    /// few pictures; it is matched through them, and keeps its number.
    #[test]
    fn an_unconfident_detection_keeps_a_followed_object() {
        let mut tracker = tracker();
        let id = tracker.update(&[seen(0, 0.9, 100.0, 100.0)], 1)[0].unwrap();
        for step in 1..=5 {
            let ids = tracker.update(&[seen(0, 0.3, 100.0 + f64::from(step), 100.0)], 1);
            assert_eq!(ids, vec![Some(id)], "hidden picture {step}");
        }
        let ids = tracker.update(&[seen(0, 0.9, 106.0, 100.0)], 1);
        assert_eq!(ids, vec![Some(id)]);
    }

    /// An unconfident detection of nothing followed starts nothing, and a
    /// confident one seen once is not numbered until it is seen again.
    #[test]
    fn noise_is_not_numbered() {
        let mut tracker = tracker();
        tracker.update(&[seen(0, 0.9, 0.0, 0.0)], 1);
        assert_eq!(tracker.update(&[seen(0, 0.3, 500.0, 500.0)], 1), vec![None]);
        let ids = tracker.update(&[seen(0, 0.3, 500.0, 500.0), seen(0, 0.9, 900.0, 100.0)], 1);
        assert_eq!(ids, vec![None, None], "once is not yet an object");
        let ids = tracker.update(&[seen(0, 0.9, 902.0, 100.0)], 1);
        assert!(ids[0].is_some(), "twice is");
    }

    /// Where one sighting is enough, an object is numbered when first seen;
    /// where three are wanted, on the third.
    #[test]
    fn how_many_sightings_confirm_an_object_is_a_choice() {
        let mut once = ByteTrack::new(TrackerOptions {
            confirm_after: 1,
            ..TrackerOptions::default()
        });
        once.update(&[seen(0, 0.9, 0.0, 0.0)], 1);
        assert!(once.update(&[seen(0, 0.9, 900.0, 100.0)], 1)[0].is_some());

        let mut thrice = ByteTrack::new(TrackerOptions {
            confirm_after: 3,
            ..TrackerOptions::default()
        });
        thrice.update(&[seen(0, 0.9, 0.0, 0.0)], 1);
        let ids: Vec<bool> = (0..3)
            .map(|n| thrice.update(&[seen(0, 0.9, 900.0 + f64::from(n), 100.0)], 1)[0].is_some())
            .collect();
        assert_eq!(ids, [false, false, true]);
    }

    #[test]
    fn a_lost_object_comes_back_with_its_number_within_the_while() {
        let mut tracker = tracker();
        let id = tracker.update(&[seen(0, 0.9, 100.0, 100.0)], 1)[0].unwrap();
        tracker.update(&[seen(0, 0.9, 100.0, 100.0)], 1);
        for _ in 0..10 {
            tracker.update(&[], 1);
        }
        assert_eq!(
            tracker.update(&[seen(0, 0.9, 100.0, 100.0)], 1),
            vec![Some(id)]
        );
        for _ in 0..40 {
            tracker.update(&[], 1);
        }
        let ids = tracker.update(&[seen(0, 0.9, 100.0, 100.0)], 1);
        assert_ne!(ids, vec![Some(id)], "forgotten after the while");
    }

    #[test]
    fn classes_are_kept_apart() {
        let mut tracker = tracker();
        let id = tracker.update(&[seen(0, 0.9, 100.0, 100.0)], 1)[0].unwrap();
        let ids = tracker.update(&[seen(2, 0.9, 100.0, 100.0)], 1);
        assert_ne!(ids, vec![Some(id)]);
    }

    /// Between detections, the expectation carries the motion on: three
    /// pictures without a detection move the box three steps of its speed.
    #[test]
    fn between_detections_objects_are_expected_where_their_motion_takes_them() {
        let mut tracker = tracker();
        for step in 0..20 {
            tracker.update(&[seen(0, 0.9, 100.0 + 6.0 * f64::from(step), 100.0)], 1);
        }
        // Last seen at x = 214; three pictures on it should be near 232.
        let expected = tracker.expect(3);
        assert_eq!(expected.len(), 1);
        assert!(
            (expected[0].tlwh[0] - 232.0).abs() < 3.0,
            "{:?}",
            expected[0].tlwh
        );
        // And the next detection is matched to it.
        let ids = tracker.update(&[seen(0, 0.9, 238.0, 100.0)], 1);
        assert_eq!(ids[0], Some(expected[0].id));
    }
}
