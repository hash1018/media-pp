//! What a Low-Latency HLS stream holds and says: its runs' init sections,
//! its segments and their parts, and the media playlist that lists them —
//! the state behind [`super::LlHlsStream`], kept under its lock.

use std::collections::VecDeque;
use std::fmt::Write as _;
use std::sync::Arc;
use std::task::Waker;
use std::time::Duration;

/// One part of a segment: a fragment of every track.
pub(super) struct Part {
    pub(super) data: Arc<[u8]>,
    /// In seconds.
    pub(super) duration: f64,
    /// Whether it begins at a keyframe.
    pub(super) independent: bool,
}

/// One segment: the parts of a run between two of its keyframes.
pub(super) struct Segment {
    pub(super) msn: u64,
    pub(super) run: u64,
    /// Whether it begins a run after another: what comes before it is on
    /// another timeline, and maybe of other tracks.
    pub(super) discontinuity: bool,
    pub(super) parts: Vec<Part>,
    /// Set once its last part is in.
    pub(super) complete: bool,
}

impl Segment {
    fn duration(&self) -> f64 {
        self.parts.iter().map(|part| part.duration).sum()
    }
}

/// Where a player asks to be answered from — `_HLS_msn` and `_HLS_part`:
/// once segment `msn` is complete, or once its part `part` is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlHlsPosition {
    /// The media sequence number of the segment.
    pub msn: u64,
    /// The part of it, counted from zero; `None` for the whole segment.
    pub part: Option<u32>,
}

/// What a stream holds, as one lock guards it.
pub(super) struct State {
    pub(super) part_target: Duration,
    pub(super) segment_target: Duration,
    /// How many complete segments are kept.
    pub(super) keep: usize,
    pub(super) segments: VecDeque<Segment>,
    /// Each run's init section, while a segment of it is kept.
    pub(super) inits: VecDeque<(u64, Arc<[u8]>)>,
    /// The number the next segment is given.
    pub(super) next_msn: u64,
    /// The number the next run is given.
    pub(super) next_run: u64,
    /// The run under way, while one is.
    pub(super) writing: Option<u64>,
    /// Whether the next segment begins a run after another.
    pub(super) discontinuity_next: bool,
    /// Where the last run ended on the stream's timeline, in seconds: the
    /// next goes on from there.
    pub(super) timeline_end: Option<f64>,
    /// How many discontinuities have left the playlist.
    pub(super) discontinuity_sequence: u64,
    /// The longest segment yet, in whole seconds rounded — the playlist's
    /// target duration, which never goes down.
    pub(super) target_duration: u64,
    pub(super) closed: bool,
    /// What waits for the stream to change.
    pub(super) waiting: Vec<Waker>,
}

impl State {
    pub(super) fn new(part_target: Duration, segment_target: Duration, keep: usize) -> Self {
        Self {
            part_target,
            segment_target,
            keep,
            segments: VecDeque::new(),
            inits: VecDeque::new(),
            next_msn: 0,
            next_run: 0,
            writing: None,
            discontinuity_next: false,
            timeline_end: None,
            discontinuity_sequence: 0,
            target_duration: segment_target.as_secs_f64().ceil().max(1.0) as u64,
            closed: false,
            waiting: Vec::new(),
        }
    }

    /// Wakes whatever waits for a change.
    pub(super) fn changed(&mut self) {
        for waker in self.waiting.drain(..) {
            waker.wake();
        }
    }

    pub(super) fn wait(&mut self, waker: &Waker) {
        if !self.waiting.iter().any(|waiting| waiting.will_wake(waker)) {
            self.waiting.push(waker.clone());
        }
    }

    /// Begins a run whose init section is `init`: the number of the init
    /// section its segments are listed under.
    ///
    /// A run's timestamps go on where the last ended — see
    /// [`State::timeline_end`] — so where its init section is the last
    /// run's, byte for byte, as a camera connected again makes it, it is the
    /// same stream going on: listed under the same init section, with no
    /// discontinuity, so a player plays on as if nothing happened. hls.js
    /// stopped at a camera's reconnection told any other way — behind a
    /// discontinuity, or under a new init section without one. A run of
    /// other tracks or other pictures follows behind a discontinuity, with
    /// an init section of its own.
    pub(super) fn begin_run(&mut self, init: Arc<[u8]>) -> u64 {
        let last = self.segments.back().map(|segment| segment.run);
        if let Some((run, _)) = self
            .inits
            .iter()
            .find(|(run, same)| Some(*run) == last && **same == *init)
        {
            let run = *run;
            self.writing = Some(run);
            return run;
        }
        let run = self.next_run;
        self.next_run += 1;
        self.inits.push_back((run, init));
        self.writing = Some(run);
        self.discontinuity_next = !self.segments.is_empty();
        run
    }

    /// Opens a segment of `run`: its number.
    pub(super) fn open_segment(&mut self, run: u64) -> u64 {
        let msn = self.next_msn;
        self.next_msn += 1;
        self.segments.push_back(Segment {
            msn,
            run,
            discontinuity: std::mem::take(&mut self.discontinuity_next),
            parts: Vec::new(),
            complete: false,
        });
        self.changed();
        msn
    }

    pub(super) fn add_part(&mut self, msn: u64, part: Part) {
        if let Some(segment) = self.segment_mut(msn) {
            segment.parts.push(part);
            self.changed();
        }
    }

    /// Completes segment `msn`, letting go of what falls out of the window.
    /// One with no part in it — opened at a keyframe as its run ended — is
    /// taken away instead, and its number is the next segment's: a player
    /// numbers the segments by where they stand in the playlist, so one
    /// skipped would have put every name after it a number off what the
    /// player asks for.
    pub(super) fn close_segment(&mut self, msn: u64) {
        let Some(segment) = self.segment_mut(msn) else {
            return;
        };
        if segment.parts.is_empty() {
            let discontinuity = segment.discontinuity;
            self.segments.retain(|segment| segment.msn != msn);
            if self.next_msn == msn + 1 {
                self.next_msn = msn;
            }
            // What it began, the next segment begins.
            self.discontinuity_next |= discontinuity;
            self.changed();
            return;
        }
        segment.complete = true;
        // A segment counts at its duration rounded to the nearest second,
        // as the protocol measures it against the target: rounded up, a
        // segment of two seconds summed from its parts to a hair over
        // would have raised a target a player takes as fixed.
        let seconds = segment.duration().round() as u64;
        self.target_duration = self.target_duration.max(seconds);
        let mut complete = self.segments.iter().filter(|s| s.complete).count();
        while complete > self.keep {
            let Some(gone) = self.segments.pop_front() else {
                break;
            };
            if gone.discontinuity {
                self.discontinuity_sequence += 1;
            }
            complete -= usize::from(gone.complete);
        }
        let first_run = self.segments.front().map(|segment| segment.run);
        self.inits.retain(|(run, _)| {
            first_run.is_some_and(|first| *run >= first) || Some(*run) == self.writing
        });
        self.changed();
    }

    /// The run under way ended.
    pub(super) fn end_run(&mut self, run: u64) {
        if self.writing == Some(run) {
            self.writing = None;
            self.changed();
        }
    }

    pub(super) fn segment(&self, msn: u64) -> Option<&Segment> {
        self.segments.iter().find(|segment| segment.msn == msn)
    }

    fn segment_mut(&mut self, msn: u64) -> Option<&mut Segment> {
        self.segments.iter_mut().find(|segment| segment.msn == msn)
    }

    /// The segment parts are being added to, while one is.
    fn open(&self) -> Option<&Segment> {
        self.segments.back().filter(|segment| !segment.complete)
    }

    /// Whether a player could begin with what has been written: a whole
    /// segment, so the target duration the playlist says — which a player
    /// takes as said once and for all — is what the segments are.
    pub(super) fn has_media(&self) -> bool {
        self.segments.iter().any(|segment| segment.complete)
    }

    /// Whether `at` is in the playlist yet.
    pub(super) fn reached(&self, at: LlHlsPosition) -> bool {
        let last_complete = self
            .segments
            .iter()
            .rev()
            .find(|segment| segment.complete)
            .map(|segment| segment.msn);
        if last_complete.is_some_and(|msn| msn >= at.msn) {
            return true;
        }
        match (at.part, self.open()) {
            (Some(part), Some(open)) => {
                open.msn > at.msn || (open.msn == at.msn && open.parts.len() > part as usize)
            }
            _ => false,
        }
    }

    /// Whether `at` is further ahead than a player may ask to be answered
    /// from: more than two segments past the one being written.
    pub(super) fn too_far(&self, at: LlHlsPosition) -> bool {
        at.msn > self.next_msn + 1
    }

    /// The part a player is told to ask for next, while one is being made.
    fn hint(&self) -> Option<(u64, usize)> {
        self.writing?;
        let open = self.open()?;
        Some((open.msn, open.parts.len()))
    }

    /// The media playlist.
    pub(super) fn playlist(&self) -> String {
        let part_target = self.part_target.as_secs_f64();
        let mut out = String::with_capacity(2048);
        out.push_str("#EXTM3U\n#EXT-X-VERSION:9\n");
        let _ = writeln!(out, "#EXT-X-TARGETDURATION:{}", self.target_duration);
        let _ = writeln!(
            out,
            "#EXT-X-SERVER-CONTROL:CAN-BLOCK-RELOAD=YES,PART-HOLD-BACK={:.3}",
            part_target * 3.0
        );
        let _ = writeln!(out, "#EXT-X-PART-INF:PART-TARGET={part_target:.3}");
        let first = self.segments.front().map_or(self.next_msn, |s| s.msn);
        let _ = writeln!(out, "#EXT-X-MEDIA-SEQUENCE:{first}");
        // Always said, and counted from one: a playlist that says none, or
        // zero, has its timelines' numbers taken by hls.js as unreliable and
        // renumbered from what it held before — which, across a run's
        // discontinuity, numbered a segment two ways and stopped the player.
        // Where the count starts is the server's to say.
        let _ = writeln!(
            out,
            "#EXT-X-DISCONTINUITY-SEQUENCE:{}",
            self.discontinuity_sequence + 1
        );
        // Parts are listed for what is within three target durations of the
        // live edge, as players joining near it use them.
        let listed_from = self.segments.len().saturating_sub(4);
        let mut run = None;
        for (index, segment) in self.segments.iter().enumerate() {
            // Said before its segment wherever it stands, the first in the
            // window too: a player numbers each segment's timeline by the
            // discontinuities before it, counted from the sequence, which
            // goes up only as the segment itself leaves. Left out at the
            // top, the segment's number fell back by one and a player that
            // had it refused the playlist.
            if segment.discontinuity {
                out.push_str("#EXT-X-DISCONTINUITY\n");
            }
            if run != Some(segment.run) {
                let _ = writeln!(out, "#EXT-X-MAP:URI=\"init-{}.mp4\"", segment.run);
                run = Some(segment.run);
            }
            if index >= listed_from {
                for (n, part) in segment.parts.iter().enumerate() {
                    let _ = write!(
                        out,
                        "#EXT-X-PART:DURATION={:.5},URI=\"part-{}-{n}.mp4\"",
                        part.duration, segment.msn
                    );
                    out.push_str(if part.independent {
                        ",INDEPENDENT=YES\n"
                    } else {
                        "\n"
                    });
                }
            }
            if segment.complete {
                let _ = writeln!(
                    out,
                    "#EXTINF:{:.5},\nseg-{}.mp4",
                    segment.duration(),
                    segment.msn
                );
            }
        }
        if let Some((msn, part)) = self.hint() {
            let _ = writeln!(
                out,
                "#EXT-X-PRELOAD-HINT:TYPE=PART,URI=\"part-{msn}-{part}.mp4\""
            );
        }
        out
    }

    /// A file the playlist names, where it is held.
    pub(super) fn file(&self, name: &File) -> Option<Arc<[u8]>> {
        match *name {
            File::Init(run) => self
                .inits
                .iter()
                .find(|(id, _)| *id == run)
                .map(|(_, data)| Arc::clone(data)),
            File::Part(msn, part) => self
                .segment(msn)?
                .parts
                .get(part as usize)
                .map(|part| Arc::clone(&part.data)),
            File::Segment(msn) => {
                let segment = self.segment(msn).filter(|segment| segment.complete)?;
                let mut whole =
                    Vec::with_capacity(segment.parts.iter().map(|p| p.data.len()).sum());
                for part in &segment.parts {
                    whole.extend_from_slice(&part.data);
                }
                Some(whole.into())
            }
        }
    }

    /// Whether `name` may still come: the part being made, or the one after.
    pub(super) fn may_come(&self, name: &File) -> bool {
        match *name {
            File::Part(msn, part) => self.hint().is_some_and(|(open, next)| {
                (msn == open && part as usize >= next && part as usize <= next + 1)
                    || (msn == open + 1 && part == 0)
            }),
            _ => false,
        }
    }
}

/// A file a playlist names, by what its name says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum File {
    Init(u64),
    Part(u64, u32),
    Segment(u64),
}

impl File {
    /// The file `name` is, where it is one of a stream's.
    pub(super) fn parse(name: &str) -> Option<Self> {
        let stem = name.strip_suffix(".mp4")?;
        if let Some(run) = stem.strip_prefix("init-") {
            return run.parse().ok().map(File::Init);
        }
        if let Some(rest) = stem.strip_prefix("part-") {
            let (msn, part) = rest.split_once('-')?;
            return Some(File::Part(msn.parse().ok()?, part.parse().ok()?));
        }
        let msn = stem.strip_prefix("seg-")?;
        msn.parse().ok().map(File::Segment)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> State {
        State::new(Duration::from_millis(200), Duration::from_secs(1), 3)
    }

    fn part(independent: bool) -> Part {
        Part {
            data: Arc::from(vec![1u8; 4]),
            duration: 0.2,
            independent,
        }
    }

    /// A segment of `parts` parts in `run`, completed: its number.
    fn segment(state: &mut State, run: u64, parts: usize) -> u64 {
        let msn = state.open_segment(run);
        for n in 0..parts {
            state.add_part(msn, part(n == 0));
        }
        state.close_segment(msn);
        msn
    }

    #[test]
    fn a_playlist_lists_parts_segments_and_the_part_coming() {
        let mut state = state();
        let run = state.begin_run(Arc::from(vec![0u8; 8]));
        segment(&mut state, run, 5);
        let open = state.open_segment(run);
        state.add_part(open, part(true));
        let playlist = state.playlist();
        for line in [
            "#EXT-X-VERSION:9",
            "#EXT-X-SERVER-CONTROL:CAN-BLOCK-RELOAD=YES,PART-HOLD-BACK=0.600",
            "#EXT-X-PART-INF:PART-TARGET=0.200",
            "#EXT-X-MEDIA-SEQUENCE:0",
            "#EXT-X-MAP:URI=\"init-0.mp4\"",
            "#EXT-X-PART:DURATION=0.20000,URI=\"part-0-0.mp4\",INDEPENDENT=YES",
            "#EXT-X-PART:DURATION=0.20000,URI=\"part-0-4.mp4\"",
            "#EXTINF:1.00000,",
            "seg-0.mp4",
            "#EXT-X-PART:DURATION=0.20000,URI=\"part-1-0.mp4\",INDEPENDENT=YES",
            "#EXT-X-PRELOAD-HINT:TYPE=PART,URI=\"part-1-1.mp4\"",
        ] {
            assert!(playlist.lines().any(|l| l == line), "{line}\n{playlist}");
        }
        assert!(
            !playlist.contains("seg-1.mp4"),
            "the open segment is parts alone"
        );
    }

    /// Ten parts of a hair over a fifth of a second make a segment of a hair
    /// over two seconds; the target is two, not three.
    #[test]
    fn the_target_is_the_longest_segment_rounded() {
        let mut state = state();
        let run = state.begin_run(Arc::from(vec![0u8; 8]));
        let msn = state.open_segment(run);
        for n in 0..10 {
            state.add_part(
                msn,
                Part {
                    duration: 0.200_001,
                    ..part(n == 0)
                },
            );
        }
        state.close_segment(msn);
        assert!(state.segments[0].duration() > 2.0, "summed past two");
        assert!(state.playlist().contains("#EXT-X-TARGETDURATION:2\n"));
    }

    #[test]
    fn a_new_run_is_a_discontinuity_with_its_own_init_and_the_numbers_go_on() {
        let mut state = state();
        let first = state.begin_run(Arc::from(vec![0u8; 8]));
        segment(&mut state, first, 5);
        state.end_run(first);
        let second = state.begin_run(Arc::from(vec![1u8; 8]));
        let msn = segment(&mut state, second, 5);
        assert_eq!(msn, 1);
        let playlist = state.playlist();
        let discontinuity = playlist.find("#EXT-X-DISCONTINUITY\n").expect(&playlist);
        let second_map = playlist.find("init-1.mp4").expect(&playlist);
        assert!(discontinuity < second_map, "{playlist}");
        assert!(playlist.contains("seg-1.mp4"));
        assert_eq!(state.file(&File::Init(1)).as_deref(), Some(&[1u8; 8][..]));
    }

    /// A run whose init section is the last run's goes on under it, with no
    /// discontinuity.
    #[test]
    fn a_run_of_the_same_init_goes_on_as_the_same_stream() {
        let mut state = state();
        let first = state.begin_run(Arc::from(vec![0u8; 8]));
        segment(&mut state, first, 5);
        state.end_run(first);
        let again = state.begin_run(Arc::from(vec![0u8; 8]));
        assert_eq!(again, first);
        segment(&mut state, again, 5);
        let playlist = state.playlist();
        assert!(!playlist.contains("#EXT-X-DISCONTINUITY\n"), "{playlist}");
        assert_eq!(playlist.matches("#EXT-X-MAP").count(), 1, "{playlist}");
        assert!(playlist.contains("seg-1.mp4"), "{playlist}");
    }

    #[test]
    fn what_falls_out_of_the_window_is_let_go_of_and_counted() {
        let mut state = state();
        let first = state.begin_run(Arc::from(vec![0u8; 8]));
        segment(&mut state, first, 5);
        state.end_run(first);
        let second = state.begin_run(Arc::from(vec![1u8; 8]));
        for _ in 0..4 {
            segment(&mut state, second, 5);
        }
        assert_eq!(state.segments.len(), 3);
        let playlist = state.playlist();
        assert!(playlist.contains("#EXT-X-MEDIA-SEQUENCE:2"), "{playlist}");
        assert!(
            playlist.contains("#EXT-X-DISCONTINUITY-SEQUENCE:2"),
            "{playlist}"
        );
        assert!(
            state.file(&File::Init(0)).is_none(),
            "its run's segments are gone"
        );
        assert!(state.file(&File::Segment(0)).is_none());
        assert_eq!(state.file(&File::Segment(4)).map(|d| d.len()), Some(20));
    }

    /// A run that ends with a segment opened at its last keyframe and
    /// nothing in it leaves no gap in the numbers: the next run's first
    /// segment takes the number, behind a discontinuity.
    #[test]
    fn an_empty_segment_a_run_ended_on_leaves_no_gap_in_the_numbers() {
        let mut state = state();
        let first = state.begin_run(Arc::from(vec![0u8; 8]));
        segment(&mut state, first, 5);
        let empty = state.open_segment(first);
        state.close_segment(empty);
        state.end_run(first);
        let second = state.begin_run(Arc::from(vec![1u8; 8]));
        assert_eq!(segment(&mut state, second, 5), empty);
        let numbers: Vec<u64> = state.segments.iter().map(|s| s.msn).collect();
        assert_eq!(numbers, [0, 1]);
        let playlist = state.playlist();
        assert!(playlist.contains("#EXT-X-DISCONTINUITY\n"), "{playlist}");
        assert!(playlist.contains("seg-1.mp4"), "{playlist}");
    }

    /// The segment a run begins with keeps the discontinuity before it as
    /// the segments before it leave the window: the tag stays at the top
    /// until the segment itself leaves, and the sequence counts it then.
    #[test]
    fn a_discontinuity_keeps_its_place_as_the_window_slides_past_it() {
        let mut state = State::new(Duration::from_millis(200), Duration::from_secs(1), 2);
        let first = state.begin_run(Arc::from(vec![0u8; 8]));
        segment(&mut state, first, 5);
        state.end_run(first);
        let second = state.begin_run(Arc::from(vec![1u8; 8]));
        segment(&mut state, second, 5);
        segment(&mut state, second, 5);
        // The first run's segment is gone; the second run's first is on top.
        let playlist = state.playlist();
        assert!(playlist.contains("#EXT-X-MEDIA-SEQUENCE:1"), "{playlist}");
        assert!(
            playlist.contains("#EXT-X-DISCONTINUITY-SEQUENCE:1\n"),
            "{playlist}"
        );
        let tag = playlist.find("#EXT-X-DISCONTINUITY\n").expect(&playlist);
        assert!(tag < playlist.find("seg-1.mp4").unwrap(), "{playlist}");
        // Once it leaves, the sequence counts it.
        segment(&mut state, second, 5);
        let playlist = state.playlist();
        assert!(
            playlist.contains("#EXT-X-DISCONTINUITY-SEQUENCE:2"),
            "{playlist}"
        );
        assert!(!playlist.contains("#EXT-X-DISCONTINUITY\n"), "{playlist}");
    }

    #[test]
    fn a_position_is_reached_by_its_segment_or_its_part() {
        let mut state = state();
        let run = state.begin_run(Arc::from(vec![0u8; 8]));
        let at = |msn, part| LlHlsPosition { msn, part };
        assert!(!state.reached(at(0, Some(0))));
        let open = state.open_segment(run);
        state.add_part(open, part(true));
        assert!(state.reached(at(0, Some(0))));
        assert!(!state.reached(at(0, Some(1))));
        assert!(
            !state.reached(at(0, None)),
            "the whole segment is not in yet"
        );
        state.close_segment(open);
        assert!(state.reached(at(0, None)));
        assert!(!state.too_far(at(2, None)));
        assert!(state.too_far(at(3, None)));
    }

    #[test]
    fn a_files_name_says_which_it_is() {
        assert_eq!(File::parse("init-3.mp4"), Some(File::Init(3)));
        assert_eq!(File::parse("part-12-4.mp4"), Some(File::Part(12, 4)));
        assert_eq!(File::parse("seg-12.mp4"), Some(File::Segment(12)));
        assert_eq!(File::parse("seg-12.ts"), None);
        assert_eq!(File::parse("../seg-1.mp4"), None);
    }
}
