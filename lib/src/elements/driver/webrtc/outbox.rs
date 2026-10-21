//! What a connection's track sinks hand it to send: [`Outbox`].

use std::{
    collections::{HashSet, VecDeque},
    sync::{Arc, Mutex, MutexGuard},
};

use str0m::format::Codec;

use crate::buffer::PacketBuffer;

use super::command::TrackId;

/// One packet a track sink has made ready for its connection: in the
/// codec it was declared in, in the form RTP carries, its time from zero.
pub(super) struct Outgoing {
    pub(super) track: TrackId,
    pub(super) codec: Codec,
    pub(super) packet: PacketBuffer,
}

/// What a connection is to send, in order, from every one of its track
/// sinks — whichever thread they are on. The connection's owner takes it
/// out: a [`WebRtcPeer`](super::WebRtcPeer) all at once, a server session
/// as each packet is due.
///
/// It keeps `capacity` packets at most. One that comes when it is full
/// means the connection has fallen behind as far as it keeps: what it holds
/// is let go of, and each track goes on from its next keyframe, rather than
/// sending what is late and broken by the gap after it.
#[derive(Clone)]
pub(super) struct Outbox(Arc<Mutex<State>>);

struct State {
    queue: VecDeque<Outgoing>,
    capacity: usize,
    /// The tracks letting go of what they are handed until a keyframe.
    skipping: HashSet<TrackId>,
    /// Whether the connection is gone, so that nothing more is taken.
    closed: bool,
}

/// What [`Outbox::push`] did with a packet.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Pushed {
    Taken,
    /// Let go of, on its way to its track's next keyframe.
    Skipped,
    /// It found the outbox full: what it held is let go of, and so is
    /// this, unless it is a keyframe.
    FellBehind,
    /// The connection is gone.
    Closed,
}

impl Outbox {
    pub(super) fn new(capacity: usize) -> Self {
        Self(Arc::new(Mutex::new(State {
            queue: VecDeque::new(),
            capacity: capacity.max(1),
            skipping: HashSet::new(),
            closed: false,
        })))
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub(super) fn push(&self, outgoing: Outgoing) -> Pushed {
        let mut state = self.lock();
        if state.closed {
            return Pushed::Closed;
        }
        let key = outgoing.packet.is_key();
        if state.skipping.contains(&outgoing.track) {
            if !key {
                return Pushed::Skipped;
            }
            state.skipping.remove(&outgoing.track);
        }
        if state.queue.len() < state.capacity {
            state.queue.push_back(outgoing);
            return Pushed::Taken;
        }
        let mut behind: HashSet<TrackId> = state.queue.drain(..).map(|held| held.track).collect();
        behind.insert(outgoing.track);
        if key {
            behind.remove(&outgoing.track);
            state.queue.push_back(outgoing);
        }
        state.skipping = behind;
        Pushed::FellBehind
    }

    /// Everything it holds, in order.
    pub(super) fn take_all(&self) -> VecDeque<Outgoing> {
        std::mem::take(&mut self.lock().queue)
    }

    /// The connection is gone: what is held is let go of, and nothing more
    /// is taken.
    pub(super) fn close(&self) {
        let mut state = self.lock();
        state.closed = true;
        state.queue.clear();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ffmpeg_next as ffmpeg;

    use super::*;

    fn outgoing(track: u64, key: bool) -> Outgoing {
        let mut packet = ffmpeg::Packet::copy(&[1]);
        if key {
            packet.set_flags(ffmpeg::packet::Flags::KEY);
        }
        Outgoing {
            track: TrackId(track),
            codec: Codec::H264,
            packet: Arc::new(packet).into(),
        }
    }

    /// Full, it lets go of what it holds, and each track goes on from its
    /// next keyframe — a sound's packets are each one.
    #[test]
    fn full_it_lets_go_and_each_track_goes_on_from_a_keyframe() {
        let outbox = Outbox::new(2);
        assert_eq!(outbox.push(outgoing(0, true)), Pushed::Taken);
        assert_eq!(outbox.push(outgoing(1, true)), Pushed::Taken);
        assert_eq!(outbox.push(outgoing(0, false)), Pushed::FellBehind);
        assert!(outbox.take_all().is_empty());
        assert_eq!(outbox.push(outgoing(0, false)), Pushed::Skipped);
        assert_eq!(outbox.push(outgoing(1, true)), Pushed::Taken);
        assert_eq!(outbox.push(outgoing(0, true)), Pushed::Taken);
        assert_eq!(outbox.push(outgoing(0, false)), Pushed::FellBehind);
        let held: Vec<_> = outbox.take_all().iter().map(|held| held.track).collect();
        assert!(held.is_empty(), "{held:?}");
    }

    #[test]
    fn closed_it_takes_nothing() {
        let outbox = Outbox::new(4);
        assert_eq!(outbox.push(outgoing(0, true)), Pushed::Taken);
        outbox.close();
        assert!(outbox.take_all().is_empty());
        assert_eq!(outbox.push(outgoing(0, true)), Pushed::Closed);
    }
}
