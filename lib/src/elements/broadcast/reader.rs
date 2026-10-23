//! Reading a broadcast from code of one's own: [`BroadcastReader`].

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::Ordering;
use std::sync::{Arc, MutexGuard};
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, bounded};

use crate::buffer::PacketBuffer;
use crate::elements::StreamInfo;

use super::Shared;
use super::state::{Entry, State};

/// What a [`BroadcastReader`] is handed next.
#[derive(Clone)]
pub enum BroadcastItem {
    /// A run begins: these tracks, numbered as a packet's `track` numbers
    /// them, and from here timestamps on a timeline of their own —
    /// an input opened again has nothing to do with the last one's. Always
    /// the first thing a reader is handed of a run.
    Started(Vec<StreamInfo>),
    /// A packet of track `track`, as it was written: shared, not copied.
    Packet {
        /// Which of the run's tracks.
        track: usize,
        /// The packet, its timestamps those it was written with.
        packet: PacketBuffer,
        /// When it was written into the broadcast.
        arrived: Instant,
    },
    /// The reader fell so far behind that what it was to read next has been
    /// let go of. It goes on from a keyframe — the newest held, or the next
    /// written; what lay between is lost to it alone.
    Skipped,
    /// The run ended — its input finished, stopped or dropped, or the
    /// broadcast was published again. Another may begin.
    Ended,
    /// The broadcast is closed: nothing more will come. Handed once, after
    /// everything before it.
    Closed,
}

impl std::fmt::Debug for BroadcastItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Started(tracks) => write!(f, "Started({} tracks)", tracks.len()),
            Self::Packet { track, packet, .. } => f
                .debug_struct("Packet")
                .field("track", track)
                .field("pts", &packet.pts())
                .field("key", &packet.is_key())
                .finish(),
            Self::Skipped => f.write_str("Skipped"),
            Self::Ended => f.write_str("Ended"),
            Self::Closed => f.write_str("Closed"),
        }
    }
}

/// One reader of a [`Broadcast`](super::Broadcast), from code of one's own:
/// a server's viewer, a counter, anything not a pipeline.
///
/// It is a position in what the broadcast keeps and nothing more, so it
/// costs nothing to hold and readers never wait for one another. Made with
/// [`Broadcast::subscribe`](super::Broadcast::subscribe), it starts at the
/// newest keyframe the broadcast holds — or, with none held, at the next
/// one written — and hands on everything after, every track in the order it
/// was written. Another track's packets stamped from that keyframe on are
/// handed on even where they were written a little before it, just after
/// it: tracks written from threads of their own reach a broadcast out of
/// step. See [`BroadcastItem`] for what it is handed.
///
/// Three ways to read, for three kinds of caller: [`Self::try_next`], which
/// never waits; [`Self::next_timeout`], which waits on the calling thread;
/// and [`Self::recv`] (or [`Self::poll_next`]), which waits as a future does,
/// on whatever runtime polls it. Not `Clone`: two readers are two
/// subscriptions.
pub struct BroadcastReader {
    shared: Arc<Shared>,
    /// The run it is reading, 0 before the first.
    run: u64,
    /// The number of the next entry it reads.
    next: u64,
    /// Whether it passes over what it reads until a place to start.
    waiting_for_start: bool,
    /// Whether it has been handed [`BroadcastItem::Ended`] for its run.
    ended: bool,
    /// Whether it has been handed [`BroadcastItem::Closed`].
    closed: bool,
    /// Where it looks back from once it reads the place it starts at — see
    /// [`Back`].
    back_from: Option<u64>,
    back: Option<Back>,
    /// The waker it left to be woken at the next write, and which of the
    /// broadcast's wakes that next one is — so that it is left once, not
    /// each time it is polled before then.
    waiting: Option<(u64, Waker)>,
    /// How far behind the newest packet a keyframe it joins at may be;
    /// further, and it waits for the next one. `None` joins at the newest
    /// keyframe however far behind.
    join_within: Option<Duration>,
}

/// Other tracks' packets written before the keyframe a reader starts at
/// and stamped from it on, which it is handed after that keyframe.
///
/// Tracks written from threads of their own — a queue each — are not
/// written in the order of their timestamps: one track's packets of the
/// keyframe's moment can be written well before the keyframe. Starting
/// strictly at the keyframe would lose them.
struct Back {
    /// The next entry to look at.
    cursor: u64,
    /// The keyframe's entry, where looking back ends.
    end: u64,
    /// The keyframe's time on its own timeline, in seconds.
    from: f64,
    /// The keyframe's track.
    key_track: usize,
}

impl BroadcastReader {
    pub(super) fn new(shared: Arc<Shared>) -> Self {
        Self::joining_within(shared, None)
    }

    /// A reader that joins at the newest keyframe only where it is at most
    /// `within` behind the newest packet, and at the next keyframe else.
    pub(super) fn joining_within(shared: Arc<Shared>, within: Option<Duration>) -> Self {
        shared
            .state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .subscribers += 1;
        Self {
            shared,
            run: 0,
            next: 0,
            waiting_for_start: false,
            ended: false,
            closed: false,
            back_from: None,
            back: None,
            waiting: None,
            join_within: within,
        }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.shared.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// What comes next, if anything has: `None` where it is up to date, or
    /// once it has been handed [`BroadcastItem::Closed`]. Never waits.
    pub fn try_next(&mut self) -> Option<BroadcastItem> {
        let shared = Arc::clone(&self.shared);
        let state = shared.state.lock().unwrap_or_else(|p| p.into_inner());
        self.next_in(&state)
    }

    /// What comes next, waiting up to `timeout` for it: `None` where
    /// nothing came in time, or once it has been handed
    /// [`BroadcastItem::Closed`].
    pub fn next_timeout(&mut self, timeout: Duration) -> Option<BroadcastItem> {
        let deadline = Instant::now() + timeout;
        let shared = Arc::clone(&self.shared);
        let mut state = shared.state.lock().unwrap_or_else(|p| p.into_inner());
        loop {
            if let Some(item) = self.next_in(&state) {
                return Some(item);
            }
            if self.closed {
                return None;
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return None;
            }
            state = shared
                .changed
                .wait_timeout(state, left)
                .unwrap_or_else(|p| p.into_inner())
                .0;
        }
    }

    /// What comes next, as a stream polls it: ready with the next item, or
    /// with `None` once [`BroadcastItem::Closed`] has been handed; pending,
    /// with `cx`'s waker woken when something is written, otherwise.
    pub fn poll_next(&mut self, cx: &mut Context<'_>) -> Poll<Option<BroadcastItem>> {
        let shared = Arc::clone(&self.shared);
        let mut state = shared.state.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(item) = self.next_in(&state) {
            return Poll::Ready(Some(item));
        }
        if self.closed {
            return Poll::Ready(None);
        }
        let left = self
            .waiting
            .as_ref()
            .is_some_and(|(wake, waker)| *wake == state.wakes && waker.will_wake(cx.waker()));
        if !left {
            state.add_waker(cx.waker());
            self.waiting = Some((state.wakes, cx.waker().clone()));
        }
        Poll::Pending
    }

    /// What comes next, as a future: [`Self::poll_next`] until it is ready.
    pub fn recv(&mut self) -> Recv<'_> {
        Recv(self)
    }

    /// A channel rung whenever something is written, for a reader that waits
    /// alongside other things — a [`super::BroadcastSource`] waits on it and
    /// on its pipeline together.
    pub(super) fn bell(&mut self) -> Receiver<()> {
        let (tx, rx) = bounded(1);
        self.lock().add_bell(tx);
        rx
    }

    /// The next item from `state`, moving this reader on.
    fn next_in(&mut self, state: &State) -> Option<BroadcastItem> {
        if state.run != self.run {
            if self.run != 0 && !self.ended {
                self.ended = true;
                return Some(BroadcastItem::Ended);
            }
            if state.run == 0 {
                return self.closed_in(state);
            }
            self.run = state.run;
            self.ended = false;
            self.join(state);
            let tracks = state.tracks.clone()?;
            return Some(BroadcastItem::Started(tracks));
        }
        if self.run == 0 {
            return self.closed_in(state);
        }
        if self.next < state.first_seq {
            self.shared.fell_behind.fetch_add(1, Ordering::Relaxed);
            self.join(state);
            return Some(BroadcastItem::Skipped);
        }
        if let Some(item) = self.look_back(state) {
            return Some(item);
        }
        while self.next < state.next_seq {
            let seq = self.next;
            let entry = &state.entries[(seq - state.first_seq) as usize];
            self.next += 1;
            if self.waiting_for_start {
                if !entry.start {
                    continue;
                }
                self.waiting_for_start = false;
            }
            if entry.start
                && let Some(cursor) = self.back_from.take()
                && let Some(from) = entry.ts
            {
                self.back = Some(Back {
                    cursor,
                    end: seq,
                    from,
                    key_track: entry.track,
                });
            }
            return Some(item(entry));
        }
        if !state.live && !self.ended {
            self.ended = true;
            return Some(BroadcastItem::Ended);
        }
        self.closed_in(state)
    }

    /// Starts reading at where a reader joining now starts.
    fn join(&mut self, state: &State) {
        let join = state.join_at(self.join_within);
        self.next = join.next;
        self.waiting_for_start = join.waiting;
        self.back_from = Some(join.back_from);
        self.back = None;
    }

    /// The next packet of another track written before the keyframe this
    /// reader started at and stamped from it on — see [`Back`].
    fn look_back(&mut self, state: &State) -> Option<BroadcastItem> {
        let back = self.back.as_mut()?;
        back.cursor = back.cursor.max(state.first_seq);
        while back.cursor < back.end {
            let entry = &state.entries[(back.cursor - state.first_seq) as usize];
            back.cursor += 1;
            if entry.track != back.key_track && entry.ts.is_some_and(|ts| ts >= back.from) {
                return Some(item(entry));
            }
        }
        self.back = None;
        None
    }

    fn closed_in(&mut self, state: &State) -> Option<BroadcastItem> {
        if state.closed && !self.closed {
            self.closed = true;
            return Some(BroadcastItem::Closed);
        }
        None
    }
}

fn item(entry: &Entry) -> BroadcastItem {
    BroadcastItem::Packet {
        track: entry.track,
        packet: entry.packet.clone(),
        arrived: entry.arrived,
    }
}

impl Drop for BroadcastReader {
    fn drop(&mut self) {
        let mut state = self.lock();
        state.subscribers = state.subscribers.saturating_sub(1);
    }
}

/// [`BroadcastReader::recv`]'s future.
pub struct Recv<'a>(&'a mut BroadcastReader);

impl Future for Recv<'_> {
    type Output = Option<BroadcastItem>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.0.poll_next(cx)
    }
}
