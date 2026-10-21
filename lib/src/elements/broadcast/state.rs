//! What a broadcast holds: the run going now, and the packets it keeps.
//!
//! Kept apart from the handles so that the rules — which packets are kept,
//! where a reader starts, when a run ends — are one place, under one lock.

use std::collections::VecDeque;
use std::task::Waker;
use std::time::{Duration, Instant};

use crossbeam_channel::{Sender, TrySendError};

use crate::buffer::PacketBuffer;
use crate::elements::StreamInfo;

/// The most a broadcast holds, keyframe or not — see [`super::Broadcast`].
#[cfg(not(test))]
pub(crate) const MAX_BYTES: usize = 256 * 1024 * 1024;
/// Small enough for a test to go over, large enough for a few seconds of
/// 1080p.
#[cfg(test)]
pub(crate) const MAX_BYTES: usize = 8 * 1024 * 1024;

/// One packet as the broadcast keeps it.
pub(crate) struct Entry {
    pub(crate) track: usize,
    pub(crate) packet: PacketBuffer,
    /// Whether a reader may start here: a keyframe of the run's first video
    /// track, or any packet of a run with none.
    pub(crate) start: bool,
    /// Where it is on the run's timeline, in seconds — its own timestamp on
    /// the timeline track, the time it arrived since the run began
    /// elsewhere. What is kept is measured in this.
    pub(crate) at: f64,
    /// Where it is on its own track's timeline, in seconds, where it is
    /// stamped — which a reader starting at a keyframe compares other
    /// tracks' packets against.
    pub(crate) ts: Option<f64>,
    pub(crate) arrived: Instant,
}

/// Where a reader joining starts.
pub(crate) struct Join {
    /// The entry it reads first.
    pub(crate) next: u64,
    /// Whether it passes over what it reads until a place to start.
    pub(crate) waiting: bool,
    /// How far back from the place it starts it looks for other tracks'
    /// packets written before it and stamped after it.
    pub(crate) back_from: u64,
}

/// Why a write was refused.
pub(crate) enum Refused {
    Closed,
    Superseded,
}

/// Everything about one broadcast that changes, under its lock.
#[derive(Default)]
pub(crate) struct State {
    /// Which run is going, from one; 0 before the first. A reader that sees
    /// it change knows one run ended and another began.
    pub(crate) run: u64,
    /// Which publication the sinks of the run going now belong to. Not the
    /// run: a seek in the pipeline writing begins a run of the same
    /// publication, whose sinks go on.
    pub(crate) publication: u64,
    pub(crate) tracks: Option<Vec<StreamInfo>>,
    /// The track whose keyframes a reader starts at — the run's first video
    /// track — or `None`, where any packet will do.
    pub(crate) key_track: Option<usize>,
    /// Whether the run's sinks have not all ended.
    pub(crate) live: bool,
    /// Which of the run's tracks have not ended.
    open: Vec<bool>,
    /// Which tracks a seek's flush has reached, while one is on its way
    /// through them all: the first begins a run of the same tracks, and
    /// what a track not yet reached writes belongs to the timeline left.
    flushed: Vec<bool>,
    run_began: Option<Instant>,
    pub(crate) entries: VecDeque<Entry>,
    /// The number of `entries[0]`; numbers go on rising across runs, so a
    /// reader's position is never mistaken for one in another run.
    pub(crate) first_seq: u64,
    pub(crate) next_seq: u64,
    /// The numbers of the entries a reader may start at, oldest first.
    pub(crate) starts: VecDeque<u64>,
    pub(crate) bytes: usize,
    /// Where the newest entry on the timeline track is.
    pub(crate) newest: Option<f64>,
    pub(crate) closed: bool,
    pub(crate) subscribers: usize,
    wakers: Vec<Waker>,
    bells: Vec<Sender<()>>,
}

impl State {
    /// Begins a run of `tracks` for a new publication, letting go of what
    /// the last run kept; answers the publication's number.
    pub(crate) fn publish(&mut self, tracks: Vec<StreamInfo>) -> u64 {
        self.publication += 1;
        self.key_track = tracks
            .iter()
            .position(|track| track.kind == ffmpeg_next::media::Type::Video);
        self.open = vec![true; tracks.len()];
        self.flushed = vec![false; tracks.len()];
        self.tracks = Some(tracks);
        self.begin_run();
        self.publication
    }

    fn begin_run(&mut self) {
        self.run += 1;
        self.live = true;
        self.run_began = None;
        self.entries.clear();
        self.first_seq = self.next_seq;
        self.starts.clear();
        self.bytes = 0;
        self.newest = None;
    }

    /// Keeps `packet` of `track`, as the sink of `publication` hands it
    /// over, and lets go of what `retain` no longer asks for. Answers
    /// whether it had to let go of more than `retain` would for the bound on
    /// what a broadcast holds.
    pub(crate) fn write(
        &mut self,
        publication: u64,
        track: usize,
        packet: PacketBuffer,
        retain: Duration,
    ) -> Result<bool, Refused> {
        if self.closed {
            return Err(Refused::Closed);
        }
        if publication != self.publication || !self.open.get(track).copied().unwrap_or(false) {
            return Err(Refused::Superseded);
        }
        if self.flushed.iter().any(|flushed| *flushed) && !self.flushed[track] {
            // Left over from before a seek this track has not seen yet.
            return Ok(false);
        }
        let now = Instant::now();
        let began = *self.run_began.get_or_insert(now);
        let timeline = self.key_track.unwrap_or(0);
        let ts = seconds(&packet, self.time_base(track, &packet));
        let start = match self.key_track {
            Some(key) => track == key && packet.is_key(),
            None => true,
        };
        let at = if track == timeline {
            ts.unwrap_or_else(|| (now - began).as_secs_f64())
        } else {
            self.newest.unwrap_or(0.0)
        };
        if track == timeline {
            self.newest = Some(self.newest.map_or(at, |newest| newest.max(at)));
        }
        self.bytes += packet.size();
        if start {
            self.starts.push_back(self.next_seq);
        }
        self.entries.push_back(Entry {
            track,
            packet,
            start,
            at,
            ts,
            arrived: now,
        });
        self.next_seq += 1;
        self.let_go(retain);
        Ok(self.bound())
    }

    /// The unit `packet`'s timestamps count in: its own, or its track's
    /// where it carries none.
    pub(crate) fn time_base(
        &self,
        track: usize,
        packet: &ffmpeg_next::Packet,
    ) -> ffmpeg_next::Rational {
        time_base(
            packet,
            self.tracks.as_ref().map(|tracks| tracks[track].time_base),
        )
    }

    /// Lets go of whole groups of pictures from the front while what is
    /// left still spans `retain`.
    fn let_go(&mut self, retain: Duration) {
        let Some(newest) = self.newest else {
            return;
        };
        let retain = retain.as_secs_f64();
        loop {
            // The next place what is kept could begin: past whatever comes
            // before the first start, or past the oldest group.
            let boundary = match self.starts.front() {
                Some(&first) if self.first_seq < first => first,
                Some(_) => match self.starts.get(1) {
                    Some(&second) => second,
                    None => return,
                },
                None => return,
            };
            let at = self.entries[(boundary - self.first_seq) as usize].at;
            if newest - at < retain {
                return;
            }
            self.drop_front_until(boundary);
        }
    }

    /// Lets go of everything before entry `seq`.
    fn drop_front_until(&mut self, seq: u64) {
        while self.first_seq < seq {
            self.drop_front();
        }
    }

    fn drop_front(&mut self) {
        if let Some(entry) = self.entries.pop_front() {
            self.bytes -= entry.packet.size();
            if self.starts.front() == Some(&self.first_seq) {
                self.starts.pop_front();
            }
            self.first_seq += 1;
        }
    }

    /// Keeps what is held under [`MAX_BYTES`], keyframe or not; whether
    /// that let go of anything.
    fn bound(&mut self) -> bool {
        let mut over = false;
        while self.bytes > MAX_BYTES && self.entries.len() > 1 {
            self.drop_front();
            over = true;
        }
        over
    }

    /// Track `track` of `publication` has ended; whether that ended the
    /// run.
    pub(crate) fn end_track(&mut self, publication: u64, track: usize) -> bool {
        if publication != self.publication || !self.live {
            return false;
        }
        if let Some(open) = self.open.get_mut(track) {
            *open = false;
        }
        if self.open.iter().any(|open| *open) {
            return false;
        }
        self.live = false;
        true
    }

    /// A seek's flush reached the sink of `track` of `publication`: what
    /// comes next is a run of its own, which the first track the flush
    /// reaches begins. The track goes on, even where it had ended — a file
    /// sought back from its end. Whether that began a run.
    pub(crate) fn flush(&mut self, publication: u64, track: usize) -> bool {
        if publication != self.publication || track >= self.flushed.len() {
            return false;
        }
        let under_way = self.flushed.iter().any(|flushed| *flushed);
        let began = !under_way || self.flushed[track];
        if began {
            self.begin_run();
            self.flushed.fill(false);
        }
        self.flushed[track] = true;
        self.open[track] = true;
        if self.flushed.iter().all(|flushed| *flushed) {
            self.flushed.fill(false);
        }
        began
    }

    /// Where a reader joining now starts: the newest start held, or — with
    /// none held — wherever the next one is written.
    pub(crate) fn join_at(&self) -> Join {
        match self.starts.back() {
            Some(&start) => Join {
                next: start,
                waiting: false,
                back_from: self
                    .starts
                    .iter()
                    .rev()
                    .nth(1)
                    .copied()
                    .unwrap_or(self.first_seq),
            },
            None => Join {
                next: self.next_seq,
                waiting: true,
                back_from: self.next_seq,
            },
        }
    }

    pub(crate) fn add_waker(&mut self, waker: &Waker) {
        if !self.wakers.iter().any(|known| known.will_wake(waker)) {
            self.wakers.push(waker.clone());
        }
    }

    pub(crate) fn add_bell(&mut self, bell: Sender<()>) {
        self.bells.push(bell);
    }

    /// Wakes every reader waiting by a waker or a bell.
    pub(crate) fn wake(&mut self) {
        for waker in self.wakers.drain(..) {
            waker.wake();
        }
        self.bells
            .retain(|bell| !matches!(bell.try_send(()), Err(TrySendError::Disconnected(_))));
    }
}

/// Where `packet` is on its timeline, in seconds: its decoding time where
/// it has one, its presentation time where not.
pub(crate) fn seconds(
    packet: &ffmpeg_next::Packet,
    time_base: ffmpeg_next::Rational,
) -> Option<f64> {
    if time_base.numerator() <= 0 || time_base.denominator() <= 0 {
        return None;
    }
    let ts = packet.dts().or(packet.pts())?;
    Some(ts as f64 * f64::from(time_base))
}

/// The unit `packet`'s timestamps count in: its own, or `track`'s where it
/// carries none.
pub(crate) fn time_base(
    packet: &ffmpeg_next::Packet,
    track: Option<ffmpeg_next::Rational>,
) -> ffmpeg_next::Rational {
    let own = packet.time_base();
    if own.numerator() > 0 && own.denominator() > 0 {
        return own;
    }
    track.unwrap_or(ffmpeg_next::Rational(0, 1))
}
