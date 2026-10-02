//! What a source with several outputs holds back for an output that cannot
//! take a buffer yet — see [`Parking`].

use std::{collections::VecDeque, time::Duration};

use ffmpeg_next::{self as ffmpeg, Rescale};

use crate::{buffer::MediaBuffer, error::Error, pad::SrcPad};

/// How far the read cursor may run ahead of what a blocked output still
/// owes, in the source's own time, to keep another output fed — see
/// [`Parking::may_park`]. Covers how far apart a container's streams are
/// muxed in practice, typically a second or less, with room to spare; past
/// it, the blocked output is waited on.
const MAX_INTERLEAVE: Duration = Duration::from_secs(5);

/// The most buffers held back at once, and the most bytes of packets: not a
/// tuning knob — see [`Parking::blocked`].
const MAX_PARKED: usize = 4_096;
const MAX_PARKED_BYTES: usize = 64 * 1024 * 1024;

/// Buffers made for an output that could not take them yet, held back so
/// the others go on being fed.
///
/// A source with several outputs makes them one at a time, in the order
/// one read cursor meets them: a container interleaves every stream into
/// one. Refusing to read on while a single output is blocked stalls the
/// outputs that *are* ready — during a preroll that starves whichever
/// branch has not yet taken its sample, and the seek times out waiting for
/// it; while playing, a file whose picture is muxed ahead of its sound
/// fills the picture's queue with frames the sound has not reached, and
/// waiting on it stops the reading that would let the sound reach them.
/// Holding the blocked output's buffers keeps the cursor moving. What
/// matters is each output's own order, and each output's buffers stay in
/// the order they were made.
///
/// This is the one read cursor's multiqueue, kept on the source's own
/// thread: bounded by how far the cursor may run ahead of what an output
/// owes, and by how much may be held at all.
#[derive(Default)]
pub(crate) struct Parking {
    /// Buffers made but not yet handed on, in the order they were made,
    /// each with its output.
    pending: VecDeque<(usize, MediaBuffer)>,
    /// Total payload of the packets in `pending`.
    bytes: usize,
    /// How many buffers each output owes, so a freshly made one never
    /// overtakes them.
    per_output: Vec<usize>,
    /// When the oldest buffer each output owes is due, in nanoseconds of
    /// the source's time — how far behind the read cursor that output has
    /// been left.
    since_ns: Vec<Option<i64>>,
    /// When the buffer made last is due, in the same units.
    read_ns: Option<i64>,
}

/// When `buffer` is due, in nanoseconds of its source's time: a packet's
/// decode time, which only moves forward in the order a file is read, else
/// its presentation time.
fn due_ns(buffer: &MediaBuffer) -> Option<i64> {
    let (ts, time_base) = match buffer {
        MediaBuffer::Packet(packet) => (packet.dts().or(packet.pts())?, packet.time_base()),
        MediaBuffer::Video(frame) => (frame.pts()?, crate::buffer::time_base(frame)?),
        MediaBuffer::Audio(frame) => (frame.pts()?, crate::buffer::time_base(frame)?),
    };
    (time_base.numerator() > 0 && time_base.denominator() > 0)
        .then(|| ts.rescale(time_base, ffmpeg::Rational::new(1, 1_000_000_000)))
}

fn size_of(buffer: &MediaBuffer) -> usize {
    match buffer {
        MediaBuffer::Packet(packet) => packet.size(),
        MediaBuffer::Video(_) | MediaBuffer::Audio(_) => 0,
    }
}

impl Parking {
    /// Holds nothing yet, for `outputs` outputs.
    pub(crate) fn new(outputs: usize) -> Self {
        Self {
            per_output: vec![0; outputs],
            since_ns: vec![None; outputs],
            ..Self::default()
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Whether output `index` still owes a buffer held back for it.
    pub(crate) fn owes(&self, index: usize) -> bool {
        self.per_output.get(index).is_some_and(|&owed| owed > 0)
    }

    /// Hands `buf`, just made for output `index`, on through its pad if the
    /// pad can take one now, and otherwise holds it back — while
    /// `prerolling` always, and outside one while another output is fed by
    /// the cursor moving on (see [`Self::may_park`]). Past that, the pad is
    /// waited on inside the push, what it already owes going first.
    ///
    /// A downstream failure drops just that one buffer, passed to `report`:
    /// one buffer's failure is not the source's end.
    pub(crate) fn hand_on(
        &mut self,
        pads: &mut [SrcPad],
        index: usize,
        buf: MediaBuffer,
        prerolling: bool,
        report: &mut dyn FnMut(Error),
    ) {
        if let Some(ns) = due_ns(&buf) {
            self.read_ns = Some(ns);
        }
        // Anything already held for this output was made first and must
        // stay first. Overtaking it would hand a decoder its stream out of
        // decode order.
        let blocked = self.owes(index) || !pads[index].ready_consume();
        if blocked && self.may_park(prerolling) {
            self.park(index, buf);
            return;
        }
        if self.owes(index) {
            self.deliver_owed(pads, index, report);
        }
        if let Err(error) = pads[index].push(buf) {
            report(error);
        }
    }

    /// Whether a buffer for a blocked output may be held back so the read
    /// cursor can move on, rather than waited on.
    ///
    /// Always during a preroll, which needs every branch's first sample.
    /// Otherwise within [`MAX_INTERLEAVE`] of what the outputs already owe.
    ///
    /// That second case is a file whose picture is muxed ahead of its sound.
    /// The picture's queue fills with frames the sound has not reached;
    /// waiting on it stops the thread before the sound's packets that would
    /// let the sound reach them, and the picture waits on the sound for
    /// good. Holding the picture's packets back keeps both fed.
    ///
    /// Whether anything else can take the cursor now is not asked here: it
    /// is what [`Self::blocked`] asks before each read, which is what keeps
    /// a source from running away from playback — holding without any
    /// bound once buffered 67 MB within 1.5 s. Asked here, it held the
    /// picture back only while the sound's own queue had room at that very
    /// moment; a packet read while both were full was pushed into the
    /// picture's and waited on there, and once the sound drained the thread
    /// was still waiting on the picture that waited on the sound — a player
    /// froze a few seconds into a file whose picture was muxed a second
    /// ahead.
    fn may_park(&self, prerolling: bool) -> bool {
        prerolling || !self.interleave_exceeded()
    }

    /// Whether the read cursor has got [`MAX_INTERLEAVE`] away from the
    /// oldest buffer any output still owes — ahead of it, or, reading
    /// backwards, behind it.
    fn interleave_exceeded(&self) -> bool {
        let Some(read_ns) = self.read_ns else {
            return false;
        };
        let bound = MAX_INTERLEAVE.as_nanos() as u64;
        self.since_ns
            .iter()
            .flatten()
            .any(|&since| read_ns.abs_diff(since) > bound)
    }

    /// Holds one buffer back, keeping the totals that bound the backlog in
    /// step.
    fn park(&mut self, index: usize, buf: MediaBuffer) {
        self.bytes = self.bytes.saturating_add(size_of(&buf));
        if self.per_output[index] == 0 {
            self.since_ns[index] = due_ns(&buf).or(self.read_ns);
        }
        self.per_output[index] += 1;
        self.pending.push_back((index, buf));
    }

    /// Hands on every buffer output `index` owes, oldest first, waiting on
    /// its pad as any push does.
    fn deliver_owed(&mut self, pads: &mut [SrcPad], index: usize, report: &mut dyn FnMut(Error)) {
        let mut rest = VecDeque::with_capacity(self.pending.len());
        while let Some((owner, buf)) = self.pending.pop_front() {
            if owner != index {
                rest.push_back((owner, buf));
                continue;
            }
            self.bytes = self.bytes.saturating_sub(size_of(&buf));
            self.per_output[index] -= 1;
            if let Err(error) = pads[index].push(buf) {
                report(error);
            }
        }
        self.pending = rest;
        self.since_ns[index] = None;
    }

    /// Hands on every held buffer whose pad can take one, oldest first,
    /// leaving the rest in place.
    ///
    /// Once an output has held one buffer back, every later buffer of *that*
    /// output is held too, even if its pad reports itself ready again a
    /// moment later. Readiness here is a `Queue`'s "not full", which its
    /// worker changes on another thread while this runs — so asking again
    /// per buffer would let the second overtake the first the instant a
    /// slot opened, and a decoder handed its stream out of order produces
    /// garbage and a flood of `co located POCs unavailable`. Other outputs
    /// are unaffected: keeping the cursor moving for them is the whole
    /// reason for holding anything back.
    pub(crate) fn drain(&mut self, pads: &mut [SrcPad], report: &mut dyn FnMut(Error)) {
        // The ordinary case — nothing is held back: this runs once per
        // buffer made, so it must not allocate to find nothing.
        if self.pending.is_empty() {
            return;
        }
        let mut deferred = VecDeque::with_capacity(self.pending.len());
        let mut blocked = vec![false; pads.len()];
        while let Some((index, buf)) = self.pending.pop_front() {
            // Still held while its pad cannot take it: pushing would block
            // this thread on that pad, which is what holding back is there
            // to avoid. When reading has to stop instead, `blocked` says
            // so, and this hands on as the pad frees up.
            if blocked[index] || !pads[index].ready_consume() {
                blocked[index] = true;
                deferred.push_back((index, buf));
                continue;
            }
            self.bytes = self.bytes.saturating_sub(size_of(&buf));
            self.per_output[index] -= 1;
            // The next it owes, if any, is the one that bounds it now.
            self.since_ns[index] = None;
            if let Err(error) = pads[index].push(buf) {
                report(error);
            }
        }
        for (index, buf) in &deferred {
            if self.since_ns[*index].is_none() {
                self.since_ns[*index] = due_ns(buf).or(self.read_ns);
            }
        }
        self.pending = deferred;
    }

    /// Hands on everything held, in order, waiting on each pad — for a
    /// finish, which plays on to all of it, while the pipeline's interrupt
    /// lets a push past a full queue.
    pub(crate) fn deliver_all(&mut self, pads: &mut [SrcPad], report: &mut dyn FnMut(Error)) {
        for (index, buf) in std::mem::take(&mut self.pending) {
            if let Err(error) = pads[index].push(buf) {
                report(error);
            }
        }
        self.forget();
    }

    /// Whether making another buffer would only deepen the backlog.
    ///
    /// Not a tuning knob: an output that is briefly behind has a handful of
    /// buffers held for it and clears them within milliseconds. These
    /// ceilings only bound an output that has stopped taking altogether, so
    /// the read cursor cannot pull an arbitrary amount of its source into
    /// memory waiting for it. Both are needed — a few large keyframes reach
    /// the byte limit at a count that would never trip on its own.
    pub(crate) fn blocked(&self, pads: &mut [SrcPad], prerolling: bool) -> bool {
        if self.pending.is_empty() {
            return false;
        }
        // Outside a preroll, an output held back past the interleave bound
        // is waited on: reading on would only hold back more for it.
        if !prerolling && self.interleave_exceeded() {
            return true;
        }
        self.pending.len() >= MAX_PARKED
            || self.bytes >= MAX_PARKED_BYTES
            || !pads
                .iter_mut()
                .any(|pad| pad.is_linked() && pad.ready_consume())
    }

    /// Forgets everything held, for a new position to be read from: what a
    /// `Flush` lets go of. Those buffers were made from the timeline being
    /// left behind, and this is the only place they exist — every stage
    /// after the source drops its own on the same `Flush`, so handing these
    /// on afterwards would be the one way old media could reach a decoder
    /// that had already reset for the new position.
    pub(crate) fn forget(&mut self) {
        self.pending.clear();
        self.bytes = 0;
        self.per_output.fill(0);
        self.since_ns.fill(None);
        self.read_ns = None;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    };

    use super::*;
    use crate::element::{Element, ElementType, Sink, element_pp_log};
    use crate::pp_log::PpLog;

    /// A packet due at `ms`, in milliseconds, carrying its time base.
    fn packet_at(ms: i64) -> MediaBuffer {
        let mut packet = ffmpeg::Packet::empty();
        packet.set_pts(Some(ms));
        packet.set_dts(Some(ms));
        packet.set_time_base(ffmpeg::Rational::new(1, 1000));
        MediaBuffer::Packet(Arc::new(packet))
    }

    fn pads(count: usize) -> Vec<SrcPad> {
        (0..count)
            .map(|index| SrcPad::new(format!("src_{index}")))
            .collect()
    }

    /// Takes nothing while its gate is shut, whatever is pushed — an output
    /// reporting backpressure without a `Queue`'s blocking behind it, so a
    /// test sees what the source would have waited on — and writes down
    /// the time of each packet it is handed.
    struct Gated {
        open: Arc<AtomicBool>,
        /// Refusals still to answer before the gate has its say — a `Queue`
        /// whose worker frees a slot partway through a drain.
        refusals: Arc<AtomicUsize>,
        /// Room left, taken by each packet — a `Queue` filling up.
        room: Arc<AtomicUsize>,
        seen: Arc<Mutex<Vec<i64>>>,
        pp_log: PpLog,
    }

    /// What a [`Gated`] linked to an output shows and is steered by.
    struct Gate {
        open: Arc<AtomicBool>,
        refusals: Arc<AtomicUsize>,
        room: Arc<AtomicUsize>,
        seen: Arc<Mutex<Vec<i64>>>,
    }

    impl Gate {
        fn link(pad: &mut SrcPad, open: bool) -> Self {
            let gate = Self {
                open: Arc::new(AtomicBool::new(open)),
                refusals: Arc::default(),
                room: Arc::new(AtomicUsize::new(usize::MAX)),
                seen: Arc::default(),
            };
            pad.link(Box::new(Gated {
                open: Arc::clone(&gate.open),
                refusals: Arc::clone(&gate.refusals),
                room: Arc::clone(&gate.room),
                seen: Arc::clone(&gate.seen),
                pp_log: element_pp_log(ElementType::Other, "gated", None),
            }));
            gate
        }

        fn seen(&self) -> Vec<i64> {
            self.seen.lock().unwrap().clone()
        }
    }

    impl Element for Gated {
        fn name(&self) -> Arc<str> {
            "gated".into()
        }
        fn element_type(&self) -> ElementType {
            ElementType::Other
        }
        fn pp_log(&self) -> &PpLog {
            &self.pp_log
        }
        fn pp_log_mut(&mut self) -> &mut PpLog {
            &mut self.pp_log
        }
    }

    impl Sink for Gated {
        fn ready_consume(&mut self) -> bool {
            let refusals = self.refusals.load(Ordering::SeqCst);
            if refusals > 0 {
                self.refusals.store(refusals - 1, Ordering::SeqCst);
                return false;
            }
            self.open.load(Ordering::SeqCst) && self.room.load(Ordering::SeqCst) > 0
        }

        fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
            if let MediaBuffer::Packet(packet) = &buf
                && let Some(pts) = packet.pts()
            {
                self.seen.lock().unwrap().push(pts);
            }
            crate::test_support::take_one(&self.room);
            Ok(())
        }
    }

    fn no_errors() -> impl FnMut(Error) {
        |error| panic!("nothing downstream fails here: {error}")
    }

    /// An output that says "full" for the first packet of a drain and
    /// "ready" for the next must not let the second overtake the first.
    ///
    /// This is what a `Queue` does: its worker frees a slot on another
    /// thread while the drain runs, so asking again per packet gives a
    /// different answer mid-pass. Asking again let the later packet through
    /// first, handing the decoder its stream out of decode order —
    /// libavcodec answers with a flood of `co located POCs unavailable`,
    /// and the picture is wrong. Reproduced by launching `av_playback` with
    /// no seek at all: 45 warnings against 0 before the fix.
    #[test]
    fn an_output_that_frees_up_mid_drain_keeps_its_order() {
        let mut pads = pads(1);
        let gate = Gate::link(&mut pads[0], true);
        gate.refusals.store(1, Ordering::SeqCst);
        let mut parking = Parking::new(1);
        for ms in [0, 40, 80, 120] {
            parking.park(0, packet_at(ms));
        }
        parking.drain(&mut pads, &mut no_errors());
        parking.drain(&mut pads, &mut no_errors());
        assert_eq!(gate.seen(), [0, 40, 80, 120]);
    }

    /// An output's packets reach it in the order they were made, however
    /// often it fills up partway through a drain.
    #[test]
    fn what_is_held_keeps_its_order_when_the_output_fills_mid_drain() {
        let mut pads = pads(1);
        let gate = Gate::link(&mut pads[0], true);
        gate.room.store(1, Ordering::SeqCst);
        let mut parking = Parking::new(1);
        let made = [0, 40, 80, 120, 160];
        for ms in made {
            parking.park(0, packet_at(ms));
        }
        for _ in made {
            parking.drain(&mut pads, &mut no_errors());
            gate.room.fetch_add(1, Ordering::SeqCst);
        }
        parking.drain(&mut pads, &mut no_errors());
        assert!(parking.is_empty());
        assert_eq!(gate.seen(), made);
    }

    /// With nothing else to take the read cursor, a blocked output is
    /// waited on: its packet is held for it and reading stops, rather than
    /// holding more behind it — or pushing into an output that cannot take
    /// it, where nothing could come back for another that drains meanwhile.
    /// A preroll holds its packet back too. And once the preroll is over,
    /// what it held waits for its output, in order — reading stops
    /// meanwhile, since nothing else needs it.
    #[test]
    fn a_lone_blocked_output_is_waited_on() {
        let mut pads = pads(1);
        let gate = Gate::link(&mut pads[0], false);
        let mut parking = Parking::new(1);

        parking.hand_on(&mut pads, 0, packet_at(0), false, &mut no_errors());
        assert!(parking.owes(0), "held for the output");
        assert!(gate.seen().is_empty(), "not pushed into it");
        assert!(
            parking.blocked(&mut pads, false),
            "and reading waits for it, rather than holding more behind it"
        );
        gate.open.store(true, Ordering::SeqCst);
        parking.drain(&mut pads, &mut no_errors());
        assert_eq!(gate.seen(), [0]);
        gate.open.store(false, Ordering::SeqCst);

        parking.hand_on(&mut pads, 0, packet_at(40), true, &mut no_errors());
        assert!(
            parking.owes(0),
            "a preroll holds a blocked output's packet, so its siblings keep flowing"
        );
        parking.drain(&mut pads, &mut no_errors());
        assert!(parking.owes(0), "still owed to an output that is full");
        assert!(
            parking.blocked(&mut pads, false),
            "and reading waits for it"
        );

        gate.open.store(true, Ordering::SeqCst);
        parking.drain(&mut pads, &mut no_errors());
        assert!(parking.is_empty());
        assert_eq!(gate.seen(), [0, 40]);
        assert!(!parking.blocked(&mut pads, false));
    }

    /// A file whose picture is muxed ahead of its sound: the picture's
    /// queue is full of frames the sound has not reached, and the sound's
    /// packets that would let it reach them lie further on. Waiting on the
    /// picture there stopped the thread before them, and the picture waited
    /// on the sound for good. The picture's packets are held back instead
    /// while the sound takes its own — up to `MAX_INTERLEAVE`, past which
    /// the picture is waited on after all — and go out in order once it
    /// has room.
    #[test]
    fn a_blocked_output_is_held_back_while_another_is_fed() {
        let (picture, sound) = (0, 1);
        let mut pads = pads(2);
        let picture_gate = Gate::link(&mut pads[picture], false);
        let sound_gate = Gate::link(&mut pads[sound], true);
        let mut parking = Parking::new(2);
        let hand_on = |parking: &mut Parking, pads: &mut [SrcPad], index, ms| {
            parking.hand_on(pads, index, packet_at(ms), false, &mut no_errors());
        };

        // The picture a second ahead: picture at 1 s, then sound from 0 on.
        hand_on(&mut parking, &mut pads, picture, 1_000);
        for ms in (0..=4_000).step_by(100) {
            hand_on(&mut parking, &mut pads, sound, ms);
        }
        assert!(
            picture_gate.seen().is_empty(),
            "the full output is not pushed into"
        );
        assert!(parking.owes(picture), "its packet is held back instead");
        assert_eq!(sound_gate.seen().len(), 41, "while the sound keeps flowing");
        assert!(
            !parking.blocked(&mut pads, false),
            "within the interleave bound, reading goes on"
        );

        // More picture than any file interleaves: past the bound, the
        // picture is waited on rather than held back without end.
        hand_on(&mut parking, &mut pads, picture, 1_040);
        hand_on(&mut parking, &mut pads, sound, 7_000);
        assert!(
            parking.blocked(&mut pads, false),
            "past the bound, reading waits for it"
        );

        picture_gate.open.store(true, Ordering::SeqCst);
        parking.drain(&mut pads, &mut no_errors());
        assert!(parking.is_empty());
        assert_eq!(picture_gate.seen(), [1_000, 1_040], "both, in order");
        assert!(!parking.blocked(&mut pads, false));
    }

    /// The picture muxed ahead and both outputs full at once: the
    /// picture's packet is held back all the same, and reading waits while
    /// neither can take anything; once the sound drains, the cursor goes on
    /// to its packets. Pushed into the full picture instead — as when a
    /// packet was held back only while the sound had room at that moment —
    /// the thread waited there, and the picture waited on sound nothing was
    /// reading.
    #[test]
    fn a_blocked_output_is_held_back_while_the_other_is_full_too() {
        let (picture, sound) = (0, 1);
        let mut pads = pads(2);
        let picture_gate = Gate::link(&mut pads[picture], false);
        let sound_gate = Gate::link(&mut pads[sound], false);
        let mut parking = Parking::new(2);

        parking.hand_on(
            &mut pads,
            picture,
            packet_at(1_000),
            false,
            &mut no_errors(),
        );
        assert!(picture_gate.seen().is_empty(), "not pushed into");
        assert!(parking.owes(picture), "held back");
        assert!(
            parking.blocked(&mut pads, false),
            "neither can take anything"
        );

        sound_gate.open.store(true, Ordering::SeqCst);
        assert!(!parking.blocked(&mut pads, false), "the sound can: read on");
        parking.hand_on(&mut pads, sound, packet_at(0), false, &mut no_errors());
        assert_eq!(sound_gate.seen(), [0]);
    }

    /// Read backwards, the cursor goes back from what an output owes: past
    /// the bound that way too, the output is waited on rather than held
    /// back for without end.
    #[test]
    fn reading_backwards_away_from_what_is_owed_is_bounded_too() {
        let (picture, sound) = (0, 1);
        let mut pads = pads(2);
        let picture_gate = Gate::link(&mut pads[picture], false);
        let _sound = Gate::link(&mut pads[sound], true);
        let mut parking = Parking::new(2);
        parking.hand_on(
            &mut pads,
            picture,
            packet_at(9_000),
            false,
            &mut no_errors(),
        );
        parking.hand_on(
            &mut pads,
            picture,
            packet_at(6_000),
            false,
            &mut no_errors(),
        );
        assert!(parking.owes(picture), "within the bound, held back");
        assert!(picture_gate.seen().is_empty());
        parking.hand_on(
            &mut pads,
            picture,
            packet_at(3_000),
            false,
            &mut no_errors(),
        );
        assert!(
            parking.is_empty(),
            "past it, going back, the output is waited on"
        );
        assert_eq!(picture_gate.seen(), [9_000, 6_000, 3_000], "in order");
    }

    /// A flush lets go of everything held: it is from the timeline being
    /// left, and handed on after the seek it would reach a decoder that had
    /// reset for the new position.
    #[test]
    fn a_flush_forgets_everything_held() {
        let mut pads = pads(1);
        let gate = Gate::link(&mut pads[0], false);
        let mut parking = Parking::new(1);
        for ms in [0, 40, 80] {
            parking.hand_on(&mut pads, 0, packet_at(ms), true, &mut no_errors());
        }
        assert!(parking.owes(0));
        parking.forget();
        assert!(parking.is_empty() && !parking.owes(0));
        gate.open.store(true, Ordering::SeqCst);
        parking.drain(&mut pads, &mut no_errors());
        assert!(gate.seen().is_empty(), "nothing from before it went on");
    }
}
