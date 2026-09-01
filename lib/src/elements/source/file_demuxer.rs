use std::{
    collections::VecDeque,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicI64, Ordering},
    },
    time::Duration,
};

use crate::pp_log::{PpLog, pp_debug, pp_error, pp_info};
use ffmpeg_next::{self as ffmpeg, Rescale};
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    contract::{MediaKind, OutputContract, PortContract},
    control::{ControlReceiver, drain_control},
    element::{Element, ElementType, Source, SourceElement, element_pp_log},
    pad::SrcPad,
};

/// Errors specific to `FileDemuxer`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum FileDemuxError {
    /// FFmpeg rejected opening, reading, or seeking the input container.
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),
}

/// Metadata about one stream in an opened container, reported up front so
/// callers can decide what to build downstream before the pipeline runs.
#[derive(Debug, Clone, Copy)]
pub struct StreamInfo {
    /// Zero-based stream index used by the matching source pad.
    pub index: usize,
    /// Media kind reported by the container, such as audio or video.
    pub kind: ffmpeg::media::Type,
}

/// Runtime control for a [`FileDemuxer`], taken with
/// [`FileDemuxer::looping_handle`] before the demuxer is moved into its
/// pipeline.
///
/// Cheap to clone and safe to share: it holds one atomic flag and nothing
/// else, so it keeps neither the demuxer, its file, nor its pipeline alive.
/// No call blocks or does any work beyond that store. A call after the
/// source has finished is simply never read.
#[derive(Clone)]
pub struct FileDemuxerHandle {
    looping: Arc<AtomicBool>,
    published_offset: Arc<AtomicI64>,
}

impl FileDemuxerHandle {
    /// Whether reaching the end of the file starts it again instead of
    /// ending the stream. Off unless this says otherwise.
    ///
    /// Read once per lap, at the end of the file — never mid-file. So
    /// turning it off part way through means "play this lap out and then
    /// finish", not "stop now", and the stream still ends with a real
    /// `Eos` rather than being abandoned the way [`ControlMsg::Stop`]
    /// abandons it. Turning it on part way through takes effect at the end
    /// the source was already heading for.
    ///
    /// [`ControlMsg::Stop`]: crate::control::ControlMsg::Stop
    pub fn set_looping(&self, looping: bool) {
        self.looping.store(looping, Ordering::Relaxed);
    }

    /// What [`FileDemuxerHandle::set_looping`] last set.
    pub fn is_looping(&self) -> bool {
        self.looping.load(Ordering::Relaxed)
    }

    /// How far this source's output timeline has been carried past the
    /// file's own — the sum of every lap already played.
    ///
    /// Zero until the first wrap, so a source that never loops never needs
    /// this. What it is for is reading a timestamp *back*: subtract it from
    /// a packet's or frame's timestamp and the result is a position in the
    /// file, which is what a progress bar means by one. See
    /// [`FileDemuxer`]'s own docs on why the two are not the same number.
    ///
    /// It moves once per lap, so a reader that samples it beside a timestamp
    /// from the same moment can be one lap out for the instant either side of
    /// a wrap. Nothing here can close that window, and a progress bar that is
    /// wrong for one frame at the moment it jumps back to zero is not wrong
    /// in a way anyone can see.
    pub fn lap_offset(&self) -> Duration {
        Duration::from_micros(self.published_offset.load(Ordering::Relaxed).max(0) as u64)
    }
}

/// Demuxes a file, exposing one src pad per container stream (indexed the
/// same way as `StreamInfo::index`). Linking a pad "selects" that stream;
/// leaving it unlinked just drops its packets. Real demuxer I/O is
/// blocking, so this is meant to be run as the pipeline's source thread.
///
/// Fan-out (e.g. routing video and audio to separate branches) needs no
/// separate "Tee" element here — it's just a matter of linking more than
/// one of these pads.
///
/// Set to loop through [`FileDemuxer::looping_handle`] and the end of the
/// file rewinds to the start instead of ending the stream. Timestamps then
/// keep climbing across the join rather than restarting: what a lap already
/// reached is added to every later one, so a `Pacer` still paces, a muxer
/// still sees its timestamps advance, and nothing downstream has to know a
/// join happened. The consequence is that a looping source's timestamps are
/// no longer positions *in the file* — one second into the third lap is at
/// twice the file's length plus a second — and [`FileDemuxer::seek`] stays
/// the way to speak in the file's own timeline.
///
/// [`FileDemuxer::seek`]: crate::element::SourceElement::seek
pub struct FileDemuxer {
    pp_log: PpLog,
    name: Arc<str>,
    input: ffmpeg::format::context::Input,
    pads: Vec<SrcPad>,
    /// Packets read but not yet delivered, in file order.
    ///
    /// `seek` puts one here: peeking a packet right after `Input::seek` is how
    /// it learns where playback actually landed (see `seek`'s docs), and that
    /// packet is real data that still has to be delivered.
    ///
    /// `run` also parks a packet here when its own pad cannot accept one yet.
    /// A container interleaves every stream into one read cursor, so refusing
    /// to read at all while a single pad is blocked stalls the streams that
    /// *are* ready — during preroll that starves whichever branch has not yet
    /// taken its sample, and the seek times out waiting for it. Holding the
    /// blocked pad's packets keeps the cursor moving; per-pad order is what
    /// matters and each pad's packets stay in the order they were read.
    pending: VecDeque<(usize, ffmpeg::Rational, ffmpeg::Packet)>,
    /// Total payload parked in `pending`.
    pending_bytes: usize,
    /// How many parked packets each pad owes, so a freshly read one never
    /// overtakes them.
    parked_per_pad: Vec<usize>,
    /// Whether a preroll is running. Parking is only correct then: outside
    /// one, a blocked pad is ordinary backpressure this source must wait on.
    prerolling: bool,
    /// Set through [`FileDemuxerHandle::set_looping`], read only where the
    /// container runs out.
    looping: Arc<AtomicBool>,
    /// `loop_offset`, published for [`FileDemuxerHandle::lap_offset`].
    ///
    /// A copy rather than the field itself: the offset is read and written
    /// once per packet on this thread, and making that an atomic to serve a
    /// reader that looks a few times a second is the wrong way round. This
    /// is stored only where the offset moves, which is once per lap.
    published_offset: Arc<AtomicI64>,
    /// How far this source's output timeline has been carried past the
    /// file's own, in microseconds: the sum of every lap already played.
    /// Zero until the first wrap, so a source that never loops emits the
    /// file's timestamps untouched.
    ///
    /// Microseconds because one lap has to be one length for every stream.
    /// Measuring each stream's own end separately would let audio and video
    /// restart at different points and drift apart by that difference on
    /// every lap.
    loop_offset: i64,
    /// The furthest into the file, in the same units, any packet read this
    /// lap reaches — what `loop_offset` grows by at the next wrap.
    ///
    /// A running maximum that only the wrap resets. A seek backwards does
    /// not un-deliver what already went downstream, so the lap stays as long
    /// as its furthest packet; growing the offset by what was *played*
    /// instead would drop the next lap on top of timestamps a muxer has
    /// already written.
    lap_end: i64,
}

impl FileDemuxer {
    /// Opens the file and returns it alongside every stream it contains,
    /// so the caller can inspect them (count, media type, ...) before
    /// deciding which of `src_pads()` to link.
    pub fn open(
        name: impl Into<String>,
        path: impl AsRef<Path>,
    ) -> Result<(Self, Vec<StreamInfo>), FileDemuxError> {
        let input = ffmpeg::format::input(&path)?;

        let streams: Vec<StreamInfo> = input
            .streams()
            .map(|s| StreamInfo {
                index: s.index(),
                kind: s.parameters().medium(),
            })
            .collect();

        let pads: Vec<SrcPad> = streams
            .iter()
            .map(|s| {
                // Per stream, from the medium the container announced:
                // both pads emit `MediaBuffer::Packet`, so only this tells
                // an audio stream apart from a video one. A medium this
                // crate does not model (subtitles, data) declares nothing
                // and is left to the runtime check.
                match MediaKind::packet_for(s.kind) {
                    Some(kind) => SrcPad::with_contract(
                        format!("src_{}", s.index),
                        OutputContract::Fixed(PortContract::packet(kind)),
                    ),
                    None => SrcPad::new(format!("src_{}", s.index)),
                }
            })
            .collect();

        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::FileDemuxer, &name, None);
        pp_info!(
            pp_log: &pp_log,
            "opened: path={}, {} stream(s)",
            path.as_ref().display(),
            streams.len()
        );
        Ok((
            Self {
                name,
                pp_log,
                input,
                parked_per_pad: vec![0; pads.len()],
                prerolling: false,
                pads,
                pending: VecDeque::new(),
                pending_bytes: 0,
                looping: Arc::new(AtomicBool::new(false)),
                published_offset: Arc::new(AtomicI64::new(0)),
                loop_offset: 0,
                lap_end: 0,
            },
            streams,
        ))
    }

    /// The control endpoint for looping this file, valid for as long as the
    /// demuxer runs — take it here, before moving the demuxer into its
    /// pipeline, and keep it for as long as the loop is meant to be
    /// switchable. See [`FileDemuxerHandle::set_looping`].
    pub fn looping_handle(&self) -> FileDemuxerHandle {
        FileDemuxerHandle {
            looping: self.looping.clone(),
            published_offset: self.published_offset.clone(),
        }
    }

    /// Codec parameters for one of this file's streams — what you need to
    /// construct a matching [`crate::elements::SwDecoder`] for it.
    pub fn stream_parameters(&self, index: usize) -> Option<ffmpeg::codec::Parameters> {
        self.stream(index).map(|s| s.parameters())
    }

    /// The unit decoded frame timestamps for this stream are expressed in —
    /// what you need to construct a matching [`crate::elements::Pacer`] for
    /// it.
    pub fn stream_time_base(&self, index: usize) -> Option<ffmpeg::Rational> {
        self.stream(index).map(|s| s.time_base())
    }

    fn stream(&self, index: usize) -> Option<ffmpeg::format::stream::Stream<'_>> {
        self.input.streams().find(|s| s.index() == index)
    }

    /// Puts a freshly read packet on this source's output timeline, and
    /// records how far into the file this lap has now reached.
    ///
    /// Called at each of the two places a packet is read out of the
    /// container — `run`'s cursor and `seek`'s read-ahead — rather than
    /// where they are delivered. Stamping a time base is idempotent and
    /// `deliver_or_park` can do it to the same packet twice; shifting a
    /// timestamp is not.
    ///
    /// Only a linked pad's stream counts towards the lap's length. An
    /// unlinked one is dropped rather than delivered, so letting a longer
    /// audio track nobody selected decide where the video restarts would
    /// only open a gap at every join.
    fn stamp_lap(
        &mut self,
        index: usize,
        time_base: ffmpeg::Rational,
        packet: &mut ffmpeg::Packet,
    ) {
        if let Some(start) = packet.pts().or_else(|| packet.dts())
            && self.pads.get(index).is_some_and(SrcPad::is_linked)
        {
            // A packet carrying no duration of its own still ends after it
            // starts, and one tick is the least that keeps the next lap's
            // first timestamp past this one's rather than equal to it.
            let end = start.saturating_add(packet.duration().max(1));
            self.lap_end = self.lap_end.max(end.rescale(time_base, microseconds()));
        }
        if self.loop_offset == 0 {
            return;
        }
        let shift = self.loop_offset.rescale(microseconds(), time_base);
        packet.set_pts(packet.pts().map(|pts| pts.saturating_add(shift)));
        packet.set_dts(packet.dts().map(|dts| dts.saturating_add(shift)));
    }

    /// Starts the file again: carries the output timeline past the lap that
    /// just ended, then rewinds the container.
    ///
    /// Ordering matters. The offset moves first so that the read-ahead
    /// packet `seek` parks — the new lap's first — is stamped onto the new
    /// timeline like every packet after it.
    fn wrap(&mut self) -> crate::error::Result<()> {
        self.loop_offset = self.loop_offset.saturating_add(self.lap_end);
        self.published_offset
            .store(self.loop_offset, Ordering::Relaxed);
        self.lap_end = 0;
        let landed = self.seek(Duration::ZERO)?;
        pp_debug!(
            self,
            "looped: restarted at {landed:?}, timeline now {}us past the file's own",
            self.loop_offset
        );
        Ok(())
    }

    /// Pushes `item` if its pad can take one now, otherwise parks it.
    ///
    /// A downstream failure drops just that one packet — the same
    /// "report, don't die" contract `Queue`'s worker gives a failing `Sink` —
    /// rather than ending this whole source thread over it. `Pipeline::stop`
    /// is how a caller who decides an error is fatal actually ends things.
    fn deliver_or_park(
        &mut self,
        item: (usize, ffmpeg::Rational, ffmpeg::Packet),
        bus: &Bus,
    ) -> crate::error::Result<()> {
        let (index, time_base, mut packet) = item;
        // `AVCodecParameters` does not carry the container stream's timestamp
        // unit, and FFmpeg does not guarantee that demuxers populate
        // `AVPacket::time_base`. Stamping it here — the one place every packet
        // leaves this source through, parked or not — means a packet held for
        // a blocked pad arrives downstream describing itself the same way an
        // immediately delivered one does.
        packet.set_time_base(time_base);
        if self.pads.get(index).is_none() {
            // No pad for this stream: nobody selected it, so it is dropped
            // rather than parked. Parking it would grow without bound.
            return Ok(());
        }
        // Anything already parked for this pad was read first and must stay
        // first. Overtaking it would hand a decoder its stream out of decode
        // order.
        let blocked = self.parked_per_pad[index] > 0 || !self.pads[index].ready_consume();
        if blocked && self.prerolling {
            self.park((index, time_base, packet));
            return Ok(());
        }
        // Outside a preroll, a blocked pad is ordinary backpressure and the
        // right answer is to wait on it: the push below blocks until the
        // downstream `Queue` has room, which paces this whole source to the
        // branch that is furthest behind. Holding the packet and reading on
        // instead would let the source run away from playback and buffer the
        // file here — measured at 67 MB parked within 1.5 s of paced playback,
        // after which the backlog ceiling stopped the read cursor for *every*
        // pad and starved the branches that were still keeping up.
        self.push_to_pad(index, packet, bus);
        Ok(())
    }

    fn push_to_pad(&mut self, index: usize, packet: ffmpeg::Packet, bus: &Bus) {
        if let Err(error) = self.pads[index].push(MediaBuffer::Packet(Arc::new(packet))) {
            bus.post(
                &self.pp_log,
                BusEvent::Error {
                    element_type: ElementType::FileDemuxer,
                    name: self.name.clone(),
                    error,
                },
            );
        }
    }

    /// Holds one packet back, keeping the totals that bound the backlog in
    /// step with the queue. The one place anything enters `pending`, so it is
    /// also where the stream time base is guaranteed onto a packet that will
    /// be delivered later — `seek` parks its read-ahead packet directly.
    fn park(&mut self, item: (usize, ffmpeg::Rational, ffmpeg::Packet)) {
        let (index, time_base, mut packet) = item;
        packet.set_time_base(time_base);
        self.pending_bytes = self.pending_bytes.saturating_add(packet.size());
        self.parked_per_pad[index] += 1;
        self.pending.push_back((index, time_base, packet));
    }

    /// Delivers every parked packet whose pad can take one, oldest first,
    /// leaving the rest in place.
    ///
    /// Once a pad has held one packet back, every later packet of *that* pad
    /// is held too, even if the pad reports itself ready again a moment later.
    /// Readiness here is a `Queue`'s "not full", which its worker changes on
    /// another thread while this loop runs — so re-asking per packet would let
    /// the second overtake the first the instant a slot opened, and a decoder
    /// handed its stream out of order produces garbage and a flood of
    /// `co located POCs unavailable`. Other pads are unaffected: keeping the
    /// read cursor moving for them is the whole reason parking exists.
    fn drain_pending(&mut self, bus: &Bus) -> crate::error::Result<()> {
        // The ordinary case, now that holding back is scoped to preroll: this
        // runs once per packet read, so it must not allocate to find nothing.
        if self.pending.is_empty() {
            return Ok(());
        }
        let mut deferred = VecDeque::with_capacity(self.pending.len());
        let mut blocked = vec![false; self.pads.len()];
        while let Some((index, time_base, packet)) = self.pending.pop_front() {
            // Once the preroll that justified holding these is over, waiting
            // on the pad is what empties the backlog; deferring again would
            // leave it parked for as long as playback keeps that pad busy.
            let hold = self.prerolling && (blocked[index] || !self.pads[index].ready_consume());
            if hold {
                blocked[index] = true;
                deferred.push_back((index, time_base, packet));
                continue;
            }
            self.pending_bytes = self.pending_bytes.saturating_sub(packet.size());
            self.parked_per_pad[index] -= 1;
            self.push_to_pad(index, packet, bus);
        }
        self.pending = deferred;
        Ok(())
    }

    /// Whether reading another packet would only deepen the parked backlog.
    ///
    /// Not a tuning knob: a branch that is briefly behind parks a handful of
    /// packets and clears them within milliseconds. These ceilings only bound
    /// a pad that has stopped accepting altogether, so the read cursor cannot
    /// pull an arbitrary amount of the file into memory waiting for it. Both
    /// are needed — a few large keyframes reach the byte limit at a packet
    /// count that would never trip on its own.
    fn pending_blocked(&mut self) -> bool {
        if !self.prerolling {
            // Nothing is parked outside a preroll, and a blocked pad is
            // waited on rather than skipped.
            return false;
        }
        const MAX_PENDING_PACKETS: usize = 4_096;
        const MAX_PENDING_BYTES: usize = 64 * 1024 * 1024;

        self.pending.len() >= MAX_PENDING_PACKETS
            || self.pending_bytes >= MAX_PENDING_BYTES
            || !self.pads.iter_mut().any(SrcPad::ready_consume)
    }
}

impl Element for FileDemuxer {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::FileDemuxer
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for FileDemuxer {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        &mut self.pads
    }
}

impl SourceElement for FileDemuxer {
    fn is_live(&self) -> bool {
        false
    }

    fn is_seekable(&self) -> bool {
        true
    }

    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> crate::error::Result<()> {
        pp_info!(self, "started");
        // Deliberately re-creates `self.input.packets()` fresh every
        // iteration (cheap — it's just a short-lived wrapper, not a
        // stateful cursor of its own) instead of holding one `for` loop's
        // iterator across the whole function, the way this used to read.
        // That iterator borrows `input` for as long as it's alive; `Seek`
        // needs `drain_control` to be able to call `self.seek()` — a
        // *second* mutable borrow of `input` — in between reads, which a
        // single loop-spanning iterator would rule out.
        loop {
            if drain_control(control, self, bus)?.stopped {
                // Stop: abandon in place, no final Eos.
                pp_info!(self, "stopped");
                return Ok(());
            }
            // Deliver whatever is parked and now accepted, oldest first.
            // Skipping a still-blocked pad's entry to reach a later one is
            // safe: only each pad's own order has to hold, and this preserves
            // it because entries for one pad are never reordered against each
            // other.
            self.drain_pending(bus)?;
            // Read on unless everything is blocked, or the parked backlog has
            // grown past what one branch briefly falling behind can explain.
            // Sleeping is the only option then: the container cannot hand out
            // a different stream's packet without reading this one.
            if self.pending_blocked() {
                std::thread::sleep(Duration::from_millis(1));
                continue;
            }
            let next = self
                .input
                .packets()
                .next()
                .map(|(s, p)| (s.index(), s.time_base(), p));
            let Some((index, time_base, mut packet)) = next else {
                if !self.pending.is_empty() {
                    // Nothing left to read, but a blocked pad still owes
                    // delivery.
                    std::thread::sleep(Duration::from_millis(1));
                    continue;
                }
                // The end of the file, and the one place the loop flag is
                // read: a change made mid-file lands here, at the end the
                // source was already heading for.
                if self.looping.load(Ordering::Relaxed) {
                    match self.wrap() {
                        Ok(()) => continue,
                        // A file that cannot be rewound cannot be looped,
                        // but it has been fully read — so report why the
                        // loop stopped and end the stream properly, rather
                        // than failing a source that delivered everything
                        // it was asked for.
                        Err(error) => bus.post(
                            &self.pp_log,
                            BusEvent::Error {
                                element_type: ElementType::FileDemuxer,
                                name: self.name.clone(),
                                error,
                            },
                        ),
                    }
                }
                break;
            };
            self.stamp_lap(index, time_base, &mut packet);
            // `deliver_or_park` stamps the stream time base; every packet
            // leaves this source through it, parked or not.
            self.deliver_or_park((index, time_base, packet), bus)?;
        }
        for pad in self.pads.iter_mut() {
            pad.push_eos(&self.pp_log)?;
        }
        pp_info!(self, "event=eos phase=source_completed outcome=ok");
        Ok(())
    }

    fn on_control(&mut self, msg: &crate::control::ControlMsg) {
        use crate::control::ControlMsg;
        match msg {
            // Those packets were read from the timeline being left behind,
            // and this source is the only place they exist — every downstream
            // stage discards its own on the same `Flush`, so releasing these
            // afterwards would be the one way old media could reach a decoder
            // that had already reset for the new position.
            ControlMsg::Flush => {
                self.pending.clear();
                self.pending_bytes = 0;
                self.parked_per_pad.fill(0);
            }
            // Holding a blocked pad's packets is only correct while a preroll
            // is running; see `deliver_or_park`.
            ControlMsg::Preroll(_) => self.prerolling = true,
            ControlMsg::Pause | ControlMsg::Resume | ControlMsg::Stop => self.prerolling = false,
            ControlMsg::CheckSeek(_) | ControlMsg::Seek(_) => {}
        }
    }

    fn seek(&mut self, target: Duration) -> crate::error::Result<Duration> {
        // `Input::seek` takes microseconds (`AV_TIME_BASE` units) when
        // seeking the whole container (stream index -1, which is what it
        // uses internally) rather than one specific stream — an unbounded
        // range (`..`) just means "as close to `ts` as ffmpeg can manage",
        // no extra min/max constraint. In practice that means *backward*
        // to the nearest keyframe at or before `target`: never forward,
        // and never onto a non-keyframe, since either would leave nothing
        // downstream can decode/remux from. A sparse-keyframe file can
        // make that keyframe well before `target` — e.g. a single
        // 10-second file with keyframes only at 0s and 8.3s means every
        // `target` under 8.3s lands back at 0s.
        let ts = target.as_micros().min(i64::MAX as u128) as i64;
        self.input.seek(ts, ..).inspect_err(|error| {
            pp_error!(self, "seek to {target:?} failed: {error}");
        })?;

        // `avformat_seek_file` only reports success/failure, not where it
        // landed — the one way to find out is to read the next packet and
        // look at its own timestamp. That packet is real data (not a
        // probe to throw away), so it's stashed in `pending` for `run`'s
        // next iteration instead of being dropped here.
        let landed_packet = self
            .input
            .packets()
            .next()
            .map(|(stream, packet)| (stream.index(), stream.time_base(), packet));
        match landed_packet {
            Some((index, time_base, mut packet)) => {
                // Read before `stamp_lap` moves it: where a seek landed is a
                // position in the *file*, which a loop's accumulated offset
                // must not be added to.
                let landed = packet
                    .pts()
                    .or_else(|| packet.dts())
                    .map(|ts| ts_to_duration(ts, time_base))
                    .unwrap_or(Duration::ZERO);
                self.stamp_lap(index, time_base, &mut packet);
                self.park((index, time_base, packet));
                Ok(landed)
            }
            // Nothing left to read right after seeking (`target` at/past
            // EOF) — there's no packet to learn a real position from, so
            // just report the request back as-is.
            None => Ok(target),
        }
    }
}

/// The unit a lap's length is kept in, so it can be one length for every
/// stream. Also what `Input::seek` takes, which is why the wrap needs no
/// conversion of its own.
///
/// A hardcoded constant, not external input, so there is nothing to
/// validate.
fn microseconds() -> ffmpeg::Rational {
    ffmpeg::Rational::new(1, 1_000_000)
}

fn ts_to_duration(ts: i64, time_base: ffmpeg::Rational) -> Duration {
    let secs = ts as f64 * f64::from(time_base.numerator()) / f64::from(time_base.denominator());
    Duration::from_secs_f64(secs.max(0.0))
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use super::*;
    use crate::control;
    use crate::test_support::try_test_video;

    struct CountingSink {
        pp_log: PpLog,
        count: Arc<AtomicUsize>,
        saw_eos: Arc<AtomicBool>,
        expected_time_base: ffmpeg::Rational,
        time_base_matches: Arc<AtomicBool>,
    }

    impl Element for CountingSink {
        fn name(&self) -> Arc<str> {
            "counting-sink".into()
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

    impl crate::element::Sink for CountingSink {
        fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
            match buf {
                MediaBuffer::Eos => self.saw_eos.store(true, Ordering::SeqCst),
                MediaBuffer::Packet(packet) => {
                    if packet.time_base() != self.expected_time_base {
                        self.time_base_matches.store(false, Ordering::SeqCst);
                    }
                    self.count.fetch_add(1, Ordering::SeqCst);
                }
                _ => {}
            }
            Ok(())
        }

        fn control(&mut self, _msg: crate::control::ControlMsg) -> crate::error::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn open_reports_stream_parameters_for_a_valid_index_and_none_out_of_range() {
        let Some(path) = try_test_video() else { return };
        let (demuxer, streams) = FileDemuxer::open("demux", &path).expect("open test video");
        let video = streams
            .iter()
            .find(|s| s.kind == ffmpeg::media::Type::Video)
            .expect("test video has a video stream");

        assert!(demuxer.stream_parameters(video.index).is_some());
        assert!(demuxer.stream_time_base(video.index).is_some());

        let out_of_range = streams.len() + 1;
        assert!(
            demuxer.stream_parameters(out_of_range).is_none(),
            "an out-of-range stream index must report nothing, not panic"
        );
        assert!(demuxer.stream_time_base(out_of_range).is_none());
    }

    /// Follows the output timeline across a loop's join, and switches the
    /// loop off once the file has been through once so `run` finishes
    /// instead of going round forever.
    struct LoopSink {
        pp_log: PpLog,
        count: Arc<AtomicUsize>,
        saw_eos: Arc<AtomicBool>,
        /// Cleared the first time a decode timestamp goes backwards.
        climbing: Arc<AtomicBool>,
        last_dts: Arc<Mutex<Option<i64>>>,
        /// How many packets one pass of this stream delivers.
        lap: usize,
        handle: FileDemuxerHandle,
    }

    impl Element for LoopSink {
        fn name(&self) -> Arc<str> {
            "loop-sink".into()
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

    impl crate::element::Sink for LoopSink {
        fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
            match buf {
                MediaBuffer::Eos => self.saw_eos.store(true, Ordering::SeqCst),
                MediaBuffer::Packet(packet) => {
                    if let Some(dts) = packet.dts().or_else(|| packet.pts()) {
                        let mut last = self.last_dts.lock().unwrap();
                        if last.is_some_and(|last| dts < last) {
                            self.climbing.store(false, Ordering::SeqCst);
                        }
                        *last = Some(dts);
                    }
                    // Past a whole pass of the file: the source is into its
                    // second lap, so let that one play out and then end.
                    if self.count.fetch_add(1, Ordering::SeqCst) + 1 > self.lap {
                        self.handle.set_looping(false);
                    }
                }
                _ => {}
            }
            Ok(())
        }

        fn control(&mut self, _msg: crate::control::ControlMsg) -> crate::error::Result<()> {
            Ok(())
        }
    }

    /// The index of this file's video stream, and the time base its packets
    /// carry.
    fn video_stream(demuxer: &FileDemuxer, streams: &[StreamInfo]) -> (usize, ffmpeg::Rational) {
        let index = streams
            .iter()
            .find(|s| s.kind == ffmpeg::media::Type::Video)
            .expect("test video has a video stream")
            .index;
        let time_base = demuxer
            .stream_time_base(index)
            .expect("video stream has a time base");
        (index, time_base)
    }

    /// How many packets one pass of this file's video stream delivers, so a
    /// test can tell a second lap has begun without assuming anything about
    /// the fixture.
    fn packets_in_one_pass(path: impl AsRef<Path>) -> usize {
        let (mut demuxer, streams) = FileDemuxer::open("demux", path).expect("open test video");
        let (index, expected_time_base) = video_stream(&demuxer, &streams);
        let count = Arc::new(AtomicUsize::new(0));
        demuxer.src_pads()[index].link(Box::new(CountingSink {
            count: count.clone(),
            saw_eos: Arc::new(AtomicBool::new(false)),
            expected_time_base,
            time_base_matches: Arc::new(AtomicBool::new(true)),
            pp_log: element_pp_log(ElementType::Other, "counting-sink", None),
        }));
        let (bus, _bus_rx) = Bus::new();
        let (_tx, rx) = control::channel();
        demuxer.run(&rx, &bus).expect("run must reach eos cleanly");
        count.load(Ordering::SeqCst)
    }

    /// Looping puts the start of the file where its end was, and the
    /// timestamps that come out keep climbing across that join instead of
    /// restarting at zero.
    ///
    /// That is the whole point of the offset. A `Pacer` downstream anchors
    /// on the first timestamp it sees and waits for each later one to come
    /// due; hand it a second lap starting back at zero and every frame of it
    /// is already overdue, so the lap is emitted as fast as it can be read
    /// rather than played. A muxer refuses it outright.
    ///
    /// Also covers when the flag is read: it is switched off part way into
    /// the second lap, and that lap still plays out and ends with a real
    /// `Eos`.
    #[test]
    fn looping_restarts_the_file_and_carries_the_timeline_past_the_join() {
        let Some(path) = try_test_video() else { return };
        let lap = packets_in_one_pass(&path);
        assert!(lap > 0, "the fixture must deliver something to loop");

        let (mut demuxer, streams) = FileDemuxer::open("demux", &path).expect("open test video");
        let (index, _) = video_stream(&demuxer, &streams);
        let handle = demuxer.looping_handle();
        assert!(
            !handle.is_looping(),
            "a file plays once unless asked not to"
        );
        handle.set_looping(true);

        let count = Arc::new(AtomicUsize::new(0));
        let saw_eos = Arc::new(AtomicBool::new(false));
        let climbing = Arc::new(AtomicBool::new(true));
        demuxer.src_pads()[index].link(Box::new(LoopSink {
            count: count.clone(),
            saw_eos: saw_eos.clone(),
            climbing: climbing.clone(),
            last_dts: Arc::new(Mutex::new(None)),
            lap,
            handle: handle.clone(),
            pp_log: element_pp_log(ElementType::Other, "loop-sink", None),
        }));

        let (bus, bus_rx) = Bus::new();
        let (_tx, rx) = control::channel();
        demuxer
            .run(&rx, &bus)
            .expect("run must reach eos cleanly, not error");

        assert!(
            count.load(Ordering::SeqCst) > lap,
            "the end of the file must start it again, not end the stream"
        );
        assert!(
            climbing.load(Ordering::SeqCst),
            "timestamps must not fall back to the file's own at the join"
        );
        assert!(
            saw_eos.load(Ordering::SeqCst),
            "switching looping off must end the lap it is in with an Eos"
        );
        // What a reader needs to turn one of those climbing timestamps back
        // into a position in the file. It only moves at a wrap, so a run that
        // wrapped has one and a run that did not has zero.
        assert!(
            handle.lap_offset() > Duration::ZERO,
            "a lap that has been stepped over must be reported"
        );
        drop(bus);
        assert!(
            bus_rx.iter().all(|e| !matches!(e, BusEvent::Error { .. })),
            "looping a well-formed file must not report any errors"
        );
    }

    /// Regression test for how far a wrap carries the timeline. It has to be
    /// how far into the file the lap *reached*, not how much of it was
    /// played: a seek backwards does not un-deliver the packets that already
    /// went downstream, so a shorter step would drop the next lap on top of
    /// timestamps a muxer has already written.
    #[test]
    fn a_backward_seek_leaves_the_lap_as_long_as_its_furthest_packet() {
        let Some(path) = try_test_video() else { return };
        let (mut demuxer, streams) = FileDemuxer::open("demux", &path).expect("open test video");

        // Every pad linked, because only a linked pad's stream counts
        // towards the lap — and `seek` parks whichever stream's packet it
        // happens to read first.
        let time_bases: Vec<ffmpeg::Rational> = (0..streams.len())
            .map(|index| {
                demuxer
                    .stream_time_base(index)
                    .expect("every stream has a time base")
            })
            .collect();
        for (index, expected_time_base) in time_bases.into_iter().enumerate() {
            demuxer.src_pads()[index].link(Box::new(CountingSink {
                count: Arc::new(AtomicUsize::new(0)),
                saw_eos: Arc::new(AtomicBool::new(false)),
                expected_time_base,
                time_base_matches: Arc::new(AtomicBool::new(true)),
                pp_log: element_pp_log(ElementType::Other, "counting-sink", None),
            }));
        }

        demuxer
            .seek(Duration::ZERO)
            .expect("seek to the start of the test video");
        let at_start = demuxer.lap_end;

        // Half way in, so the keyframe this lands on is somewhere past the
        // first one for any file that has more than one — which is what the
        // guard below checks rather than assumes.
        let half = Duration::from_micros((demuxer.input.duration().max(0) / 2) as u64);
        demuxer
            .seek(half)
            .expect("seek half way into the test video");
        let reached = demuxer.lap_end;
        if reached <= at_start {
            // Nothing in this fixture is reachable past its own start, so
            // there is no reach for a seek back to lose.
            return;
        }

        demuxer
            .seek(Duration::ZERO)
            .expect("seek back to the start of the test video");

        assert_eq!(
            demuxer.lap_end, reached,
            "going back must not shorten the lap the next wrap steps over"
        );
    }

    /// Drives `FileDemuxer::run` directly (no `Pipeline`) to prove the
    /// basic contract on its own: every packet on a linked pad's stream
    /// arrives, and running off the end of the file delivers a final
    /// `Eos` rather than just stopping silently.
    #[test]
    fn run_delivers_every_packet_on_a_linked_pad_then_eos() {
        let Some(path) = try_test_video() else { return };
        let (mut demuxer, streams) = FileDemuxer::open("demux", &path).expect("open test video");
        let video = streams
            .iter()
            .find(|s| s.kind == ffmpeg::media::Type::Video)
            .expect("test video has a video stream");

        let count = Arc::new(AtomicUsize::new(0));
        let saw_eos = Arc::new(AtomicBool::new(false));
        let time_base_matches = Arc::new(AtomicBool::new(true));
        let expected_time_base = demuxer
            .stream_time_base(video.index)
            .expect("video stream has a time base");
        demuxer.src_pads()[video.index].link(Box::new(CountingSink {
            count: count.clone(),
            saw_eos: saw_eos.clone(),
            expected_time_base,
            time_base_matches: time_base_matches.clone(),
            pp_log: element_pp_log(ElementType::Other, "counting-sink", None),
        }));

        let (bus, bus_rx) = Bus::new();
        let (_tx, rx) = control::channel();
        demuxer
            .run(&rx, &bus)
            .expect("run must reach eos cleanly, not error");

        assert!(
            count.load(Ordering::SeqCst) > 0,
            "expected at least one packet delivered to the linked pad"
        );
        assert!(
            saw_eos.load(Ordering::SeqCst),
            "expected an Eos once the file is exhausted"
        );
        assert!(
            time_base_matches.load(Ordering::SeqCst),
            "every delivered packet must carry its stream time base"
        );
        drop(bus);
        assert!(
            bus_rx.iter().all(|e| !matches!(e, BusEvent::Error { .. })),
            "run must not report any errors demuxing a well-formed file"
        );
    }

    #[test]
    fn seek_read_ahead_packet_carries_its_stream_time_base_when_delivered() {
        let Some(path) = try_test_video() else { return };
        let (mut demuxer, _) = FileDemuxer::open("demux", &path).expect("open test video");

        demuxer
            .seek(Duration::from_secs(1))
            .expect("seek within the test video");
        let (pending_index, expected_time_base, _) = demuxer
            .pending
            .front()
            .expect("seek must retain the first packet at or after the target");
        let pending_index = *pending_index;
        let expected_time_base = *expected_time_base;

        let count = Arc::new(AtomicUsize::new(0));
        let saw_eos = Arc::new(AtomicBool::new(false));
        let time_base_matches = Arc::new(AtomicBool::new(true));
        demuxer.src_pads()[pending_index].link(Box::new(CountingSink {
            count: count.clone(),
            saw_eos,
            expected_time_base,
            time_base_matches: time_base_matches.clone(),
            pp_log: element_pp_log(ElementType::Other, "counting-sink", None),
        }));

        let (bus, _bus_rx) = Bus::new();
        let (_tx, rx) = control::channel();
        demuxer
            .run(&rx, &bus)
            .expect("run after seek must reach eos cleanly");

        assert!(
            count.load(Ordering::SeqCst) > 0,
            "the packet retained by seek must be delivered"
        );
        assert!(
            time_base_matches.load(Ordering::SeqCst),
            "the packet retained by seek must carry its stream time base"
        );
    }

    /// A seek starts a new timeline. Anything the demuxer had already read and
    /// parked for a blocked pad belongs to the old one, so delivering it after the
    /// reposition feeds pre-seek packets to a decoder whose reference state was
    /// just flushed — corrupt output, and a stream of `co located POCs
    /// unavailable` from libavcodec.
    #[test]
    fn seeking_discards_packets_parked_before_the_jump() {
        let Some(path) = try_test_video() else { return };
        let (mut demuxer, streams) = FileDemuxer::open("demux", &path).expect("open test video");
        let video = streams
            .iter()
            .find(|stream| stream.kind == ffmpeg::media::Type::Video)
            .expect("test video has a video stream");
        let time_base = demuxer
            .stream_time_base(video.index)
            .expect("video stream disappeared");

        // Read a little of the start into the parked queue by hand, the way `run`
        // does when a pad cannot accept a packet yet.
        let mut input = ffmpeg::format::input(&path).expect("second handle");
        let mut parked = 0usize;
        for (stream, packet) in input.packets() {
            if stream.index() != video.index {
                continue;
            }
            demuxer.park((stream.index(), time_base, packet));
            parked += 1;
            if parked == 4 {
                break;
            }
        }
        assert_eq!(parked, 4, "the fixture must have packets to park");

        // The order a pipeline uses: Flush marks the timeline boundary, Seek
        // then moves the cursor and reads one packet ahead to learn where it
        // landed. Counting rather than comparing timestamps, because a seek
        // that lands back at the start legitimately re-reads a packet with
        // the same pts as one of the discarded ones.
        demuxer.on_control(&crate::control::ControlMsg::Flush);
        demuxer
            .seek(Duration::from_secs(3))
            .expect("seek within the test video");

        assert_eq!(
            demuxer.pending.len(),
            1,
            "only the seek's own read-ahead packet may survive the flush"
        );
        let counted: usize = demuxer
            .pending
            .iter()
            .map(|(_, _, packet)| packet.size())
            .sum();
        assert_eq!(
            demuxer.pending_bytes, counted,
            "the parked byte total must stay in step with the queue"
        );
    }

    /// Reports "full" until asked a given number of times, then reports room
    /// again — a `Queue` whose worker frees a slot partway through a drain.
    /// Readiness flipping *during* the loop is the real behaviour: the worker
    /// runs on its own thread, so the answer is not stable across the packets
    /// of one pass.
    struct FlipToReadySink {
        refusals_left: Arc<AtomicUsize>,
        seen: Arc<Mutex<Vec<i64>>>,
        pp_log: PpLog,
    }

    impl Element for FlipToReadySink {
        fn name(&self) -> Arc<str> {
            "flip-to-ready".into()
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

    impl crate::element::Sink for FlipToReadySink {
        fn ready_consume(&mut self) -> bool {
            let left = self.refusals_left.load(Ordering::SeqCst);
            if left > 0 {
                self.refusals_left.store(left - 1, Ordering::SeqCst);
                return false;
            }
            true
        }

        fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
            if let MediaBuffer::Packet(packet) = &buf
                && let Some(pts) = packet.pts()
            {
                self.seen.lock().unwrap().push(pts);
            }
            Ok(())
        }

        fn control(&mut self, _msg: crate::control::ControlMsg) -> crate::error::Result<()> {
            Ok(())
        }
    }

    /// A pad that says "full" for the first packet of a drain and "ready" for
    /// the next must not let the second overtake the first.
    ///
    /// This is what a `Queue` does: its worker frees a slot on another thread
    /// while the drain loop is running, so asking again per packet gives a
    /// different answer mid-pass. Re-asking let the later packet through
    /// first, handing the decoder its stream out of decode order — libavcodec
    /// answers with a flood of `co located POCs unavailable`, and the picture
    /// is wrong. Reproduced by launching `av_playback` with no seek at all:
    /// 45 warnings against 0 before this branch.
    #[test]
    fn a_pad_that_becomes_ready_mid_drain_does_not_reorder_its_stream() {
        let Some(path) = try_test_video() else { return };
        let (mut demuxer, streams) = FileDemuxer::open("demux", &path).expect("open");
        let video = streams
            .iter()
            .find(|stream| stream.kind == ffmpeg::media::Type::Video)
            .expect("video stream");
        let time_base = demuxer.stream_time_base(video.index).expect("time base");

        // Refuse once: the first packet of the drain is held back, and every
        // later one finds the pad ready again.
        let seen = Arc::new(Mutex::new(Vec::new()));
        demuxer.pads[video.index].link(Box::new(FlipToReadySink {
            refusals_left: Arc::new(AtomicUsize::new(1)),
            seen: Arc::clone(&seen),
            pp_log: element_pp_log(ElementType::Other, "flip-to-ready", None),
        }));

        let mut input = ffmpeg::format::input(&path).expect("second handle");
        let mut order = Vec::new();
        for (stream, packet) in input.packets() {
            if stream.index() != video.index {
                continue;
            }
            order.push(packet.pts().expect("fixture packets carry a pts"));
            demuxer.park((stream.index(), time_base, packet));
            if order.len() == 4 {
                break;
            }
        }

        let (bus, _bus_rx) = Bus::new();
        demuxer.drain_pending(&bus).expect("first drain");
        demuxer.drain_pending(&bus).expect("second drain");

        assert_eq!(
            *seen.lock().unwrap(),
            order,
            "a packet overtook one still parked for the same pad"
        );
    }

    /// Accepts a bounded number of buffers, then blocks — the shape of a
    /// `Queue` filling up mid-drain.
    struct BoundedSink {
        remaining: Arc<AtomicUsize>,
        seen: Arc<Mutex<Vec<i64>>>,
        pp_log: PpLog,
    }

    impl Element for BoundedSink {
        fn name(&self) -> Arc<str> {
            "bounded".into()
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

    impl crate::element::Sink for BoundedSink {
        fn ready_consume(&mut self) -> bool {
            self.remaining.load(Ordering::SeqCst) > 0
        }

        fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
            if let MediaBuffer::Packet(packet) = &buf
                && let Some(pts) = packet.pts()
            {
                self.seen.lock().unwrap().push(pts);
            }
            self.remaining.fetch_sub(1, Ordering::SeqCst);
            Ok(())
        }

        fn control(&mut self, _msg: crate::control::ControlMsg) -> crate::error::Result<()> {
            Ok(())
        }
    }

    /// A stream's packets must reach its pad in the order they were read.
    /// Parking exists so a blocked pad does not stall the read cursor; it must
    /// not reorder that pad's own packets, or the decoder is handed frames out
    /// of decode order and produces `co located POCs unavailable` and garbage.
    #[test]
    fn parked_packets_keep_their_per_pad_order_when_the_pad_blocks_mid_drain() {
        let Some(path) = try_test_video() else { return };
        let (mut demuxer, streams) = FileDemuxer::open("demux", &path).expect("open");
        let video = streams
            .iter()
            .find(|stream| stream.kind == ffmpeg::media::Type::Video)
            .expect("video stream");
        let time_base = demuxer.stream_time_base(video.index).expect("time base");

        // Room for one buffer only, so the pad blocks partway through the
        // drain and the rest have to be parked again.
        let remaining = Arc::new(AtomicUsize::new(1));
        let seen = Arc::new(Mutex::new(Vec::new()));
        demuxer.pads[video.index].link(Box::new(BoundedSink {
            remaining: Arc::clone(&remaining),
            seen: Arc::clone(&seen),
            pp_log: element_pp_log(ElementType::Other, "bounded", None),
        }));

        let mut input = ffmpeg::format::input(&path).expect("second handle");
        let mut order = Vec::new();
        for (stream, packet) in input.packets() {
            if stream.index() != video.index {
                continue;
            }
            order.push(packet.pts().expect("fixture packets carry a pts"));
            demuxer.park((stream.index(), time_base, packet));
            if order.len() == 5 {
                break;
            }
        }

        let (bus, _bus_rx) = Bus::new();
        // Drain repeatedly, opening one slot at a time, until everything has
        // been delivered.
        for _ in 0..order.len() {
            demuxer.drain_pending(&bus).expect("drain");
            remaining.fetch_add(1, Ordering::SeqCst);
        }
        demuxer.drain_pending(&bus).expect("final drain");

        assert_eq!(
            *seen.lock().unwrap(),
            order,
            "packets reached the pad out of the order they were read"
        );
    }

    /// Never has room, but accepts what is pushed — a pad reporting
    /// backpressure without a `Queue`'s blocking behind it.
    struct NeverReadyRecorder {
        seen: Arc<AtomicUsize>,
        pp_log: PpLog,
    }

    impl Element for NeverReadyRecorder {
        fn name(&self) -> Arc<str> {
            "never-ready".into()
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

    impl crate::element::Sink for NeverReadyRecorder {
        fn ready_consume(&mut self) -> bool {
            false
        }
        fn consume(&mut self, _buf: MediaBuffer) -> crate::error::Result<()> {
            self.seen.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn control(&mut self, _msg: crate::control::ControlMsg) -> crate::error::Result<()> {
            Ok(())
        }
    }

    /// Holding a blocked pad's packets is a preroll measure, not a playback
    /// one.
    ///
    /// Preroll needs it: a terminal closes after its one sample, and refusing
    /// to read while that pad is shut would starve the branches still owed
    /// theirs. Playback must not have it — a blocked pad there is ordinary
    /// backpressure, and waiting on it is what paces this source to the
    /// slowest branch. Holding instead let the source run away from playback
    /// and buffer the file: 67 MB parked within 1.5 s, after which the backlog
    /// ceiling stopped the read cursor for *every* pad. Attaching an audio
    /// branch then never received a packet, so it never primed the playback
    /// clock, and the picture froze.
    #[test]
    fn packets_are_only_held_back_while_a_preroll_is_running() {
        let Some(path) = try_test_video() else { return };
        let (mut demuxer, streams) = FileDemuxer::open("demux", &path).expect("open");
        let index = streams.first().expect("at least one stream").index;
        let time_base = demuxer.stream_time_base(index).expect("time base");
        let seen = Arc::new(AtomicUsize::new(0));
        demuxer.pads[index].link(Box::new(NeverReadyRecorder {
            seen: Arc::clone(&seen),
            pp_log: element_pp_log(ElementType::Other, "never-ready", None),
        }));

        let (bus, _bus_rx) = Bus::new();
        let packet = || {
            let mut packet = ffmpeg::Packet::empty();
            packet.set_pts(Some(0));
            (index, time_base, packet)
        };

        demuxer
            .deliver_or_park(packet(), &bus)
            .expect("playback delivery");
        assert!(
            demuxer.pending.is_empty(),
            "playback must wait on a blocked pad, not buffer behind it"
        );
        assert_eq!(seen.load(Ordering::SeqCst), 1);

        demuxer.on_control(&crate::control::ControlMsg::Preroll(Arc::new(
            crate::control::PrerollContext::new([]),
        )));
        demuxer
            .deliver_or_park(packet(), &bus)
            .expect("preroll delivery");
        assert_eq!(
            demuxer.pending.len(),
            1,
            "preroll must hold a blocked pad's packet so its siblings keep flowing"
        );
        assert_eq!(seen.load(Ordering::SeqCst), 1);

        demuxer.on_control(&crate::control::ControlMsg::Resume);
        demuxer.drain_pending(&bus).expect("drain after preroll");
        assert!(
            demuxer.pending.is_empty(),
            "the backlog must not outlive the preroll that justified it"
        );
        assert_eq!(seen.load(Ordering::SeqCst), 2);
    }
}
