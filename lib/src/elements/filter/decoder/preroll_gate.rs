use ffmpeg_next::{self as ffmpeg, Rescale};

use std::sync::Arc;

use crate::buffer::MediaBuffer;
use crate::pad::SrcPad;
use crate::playback_state::PlaybackState;
use crate::stash::OutputStash;
use crate::stream::Segment;

const NANOS: ffmpeg::Rational = ffmpeg::Rational(1, 1_000_000_000);
#[cfg(any(
    test,
    feature = "cuda",
    feature = "vulkan",
    all(target_os = "windows", feature = "d3d11")
))]
pub(super) const ACCURATE_SEEK_CANDIDATE_FRAMES: i32 = 1;

#[cfg(any(
    test,
    feature = "cuda",
    feature = "vulkan",
    all(target_os = "windows", feature = "d3d11")
))]
pub(super) fn hw_surface_budget(downstream_frames: i32) -> Option<i32> {
    (downstream_frames >= 0)
        .then(|| downstream_frames.checked_add(ACCURATE_SEEK_CANDIDATE_FRAMES))
        .flatten()
}

/// Suppresses decoded media that precedes where the segment shows from, and
/// keeps what the decoder's pad cannot take while a preroll holds the graph.
///
/// A seek lands on the keyframe at or before the requested position, so a
/// decoder has to decode forward from there to rebuild reference state. The
/// segment the source begins after an accurate seek says where what is shown
/// begins ([`Segment::show_from`]); what is decoded before it exists purely to
/// warm the codec up. Video holds one preceding frame until the next PTS
/// proves which frame covers the requested instant; audio uses its sample
/// count and rate for the same decision. The segment's, not a preroll's: what
/// precedes the target is dropped until a sample reaches it, whatever the
/// phase — a preroll that ends before this decoder got there, a step back's
/// sound, leaves nothing to carry over.
///
/// # Why this lives in the decoder
///
/// Deciding "is this sample before the target" needs a PTS *and* the time base
/// to read it in, and every decoded branch has a decoder while not every one
/// has anything after it that paces. The gate first lived in
/// [`Pacer`](crate::elements::Pacer) and
/// [`VideoSynchronizer`](crate::elements::VideoSynchronizer), but an audio
/// branch has neither (an audio renderer paces itself against its own device
/// clock), so that branch went ungated and delivered its whole pre-target
/// span. The decoder is also where a decoded frame first gets its time base
/// at all — see `describe` — which is what lets everything downstream read
/// one off the frame instead of being told it.
///
/// # What this deliberately does not do
///
/// It does not decide when preroll is *complete*, nor stop after one sample.
/// A terminal counts what it takes and, once it has what the preroll asks
/// of it, answers not ready (see [`PrerollContext::mark_ready`](crate::control::PrerollContext::mark_ready)); what the
/// decoder makes after that goes into whatever between them can still take
/// it, and what its own pad cannot take waits in the stash
/// ([`OutputStash`]) — a decoder can answer one packet with several frames,
/// all pushed from one call, and nothing asks the pad between them. The
/// decoder answers not ready while the stash keeps something, so its packets
/// wait in turn.
#[derive(Default)]
pub(super) struct PrerollGate {
    /// Whether a sample has gone on that shows the segment's
    /// [`show_from`](Segment::show_from), after which nothing more is
    /// dropped for coming before it.
    reached: bool,
    /// Learned from the packets being fed in. FFmpeg's decoders give a
    /// decoded frame a `pts` in this unit but not the unit itself, so the
    /// gate stamps it on every frame it lets through.
    time_base: Option<ffmpeg::Rational>,
    /// Last decoded sample before the target. Video needs one-sample
    /// lookahead to select the frame that actually covers the requested
    /// instant, and the end of the stream uses the same candidate as its
    /// last-presentable fallback.
    candidate: Option<MediaBuffer>,
    /// The segment the stream is in, which says where what is shown begins
    /// — on the timeline the samples are stamped on, a looping file's a lap
    /// further on for every lap played. `None` for a decoder handed none,
    /// which has no target.
    segment: Option<Arc<Segment>>,
    /// What the decoder made that its pad could not take while a preroll
    /// held the graph.
    stash: OutputStash,
}

impl PrerollGate {
    /// Reads from `state` whether a preroll holds the graph — what a
    /// decoder's `attach_context` hands on.
    pub(super) fn attach(&mut self, state: &Arc<PlaybackState>) {
        self.stash.attach(state);
    }

    /// Whether the decoder can take its next packet — what its
    /// `ready_consume` answers: not while the stash keeps what `pad` cannot
    /// take yet, since anything decoded now would only be kept behind it.
    ///
    /// It went on taking them once, and a branch whose preroll finished well
    /// before its sibling's — the sound, while a slow picture caught up — was
    /// fed and emptied its whole stream in that time: nothing of it was left
    /// to play once the seek was done.
    pub(super) fn ready(&mut self, pad: &mut SrcPad) -> bool {
        self.stash.ready(pad)
    }

    /// Forgets the learned time base along with the segment and what is
    /// kept, for a `Flush` that begins a new timeline.
    pub(super) fn reset(&mut self) {
        self.reached = false;
        self.time_base = None;
        self.candidate = None;
        self.segment = None;
        self.stash.clear();
    }

    /// The segment the stream is in from here — see `segment`. What is kept
    /// goes ahead of it through `pad`, since it came before it: the graph
    /// passes the segment on once this returns.
    pub(super) fn begin_segment(
        &mut self,
        segment: &Arc<Segment>,
        pad: &mut SrcPad,
    ) -> crate::error::Result<()> {
        let kept = self.stash.release_all(pad);
        self.segment = Some(Arc::clone(segment));
        self.reached = false;
        self.candidate = None;
        kept
    }

    /// Where what is shown begins, in nanoseconds on the samples' timeline,
    /// while no sample has reached it yet — see `reached`.
    ///
    /// None played backwards: there it is where the stretches start from,
    /// and what each shows is cut by the source — see
    /// `crate::element::ReversibleSource` — so every sample is below it.
    fn target_ns(&self) -> Option<i64> {
        if self.reached {
            return None;
        }
        let segment = self.segment.as_ref().filter(|segment| !segment.backwards)?;
        let show_from = segment.show_from?;
        Some(show_from.as_nanos().min(i64::MAX as u128) as i64)
    }

    /// Records the unit this decoder's `pts` values will be expressed in.
    /// `FileDemuxer` stamps every packet with its stream's time base, so this
    /// is available before the first frame comes back out.
    pub(super) fn observe_packet(&mut self, packet: &ffmpeg::Packet) {
        let time_base = packet.time_base();
        if time_base.numerator() > 0 && time_base.denominator() > 0 {
            self.time_base = Some(time_base);
        }
    }

    /// Whether this decoded sample precedes the target and must not be
    /// forwarded.
    ///
    /// Fails open — a missing target, time base, or PTS all pass the sample
    /// through. Suppressing on a guess would freeze the branch outright, and a
    /// branch that shows slightly early media is recoverable where one that
    /// shows none is not.
    #[cfg(test)]
    pub(super) fn suppresses(&self, pts: Option<i64>) -> bool {
        let (Some(target_ns), Some(time_base), Some(pts)) = (self.target_ns(), self.time_base, pts)
        else {
            return false;
        };
        pts.rescale(time_base, NANOS) < target_ns
    }

    /// Admits one decoded sample: what goes on because of it, in order — the
    /// held-back frame that turns out to cover the target, then this one —
    /// retaining at most one pre-target candidate.
    fn admit(&mut self, buffer: MediaBuffer) -> [Option<MediaBuffer>; 2] {
        let Some(target_ns) = self.target_ns() else {
            return [None, Some(buffer)];
        };
        let Some(time_base) = self.time_base else {
            return [None, Some(buffer)];
        };
        let (pts, audio_end_ns) = match &buffer {
            MediaBuffer::Video(frame) => (frame.pts(), None),
            MediaBuffer::Audio(frame) => {
                let end = frame.pts().and_then(|pts| {
                    let rate = u128::from(frame.rate());
                    (rate > 0).then(|| {
                        let start = pts.rescale(time_base, NANOS);
                        let duration =
                            (frame.samples() as u128).saturating_mul(1_000_000_000) / rate;
                        start.saturating_add(duration.min(i64::MAX as u128) as i64)
                    })
                });
                (frame.pts(), end)
            }
            _ => return [None, Some(buffer)],
        };
        let Some(pts) = pts else {
            return [None, Some(buffer)];
        };
        let start_ns = pts.rescale(time_base, NANOS);

        // An audio frame crossing the target is the audio that exists at the
        // requested instant; do not discard the whole frame just because its
        // first sample precedes the target.
        if audio_end_ns.is_some_and(|end| start_ns <= target_ns && end > target_ns) {
            self.reached = true;
            self.candidate = None;
            return [None, Some(buffer)];
        }
        if start_ns < target_ns {
            self.candidate = Some(buffer);
            return [None, None];
        }
        self.reached = true;
        let candidate = self.candidate.take();
        let covering =
            candidate.filter(|_| start_ns > target_ns && matches!(&buffer, MediaBuffer::Video(_)));
        [covering, Some(buffer)]
    }

    /// Hands one decoded sample on through `pad` as the target and the stash
    /// say: nothing, while it precedes the target; the frame that turns out
    /// to cover the target, then this one; and what `pad` cannot take while
    /// a preroll holds the graph kept, to go first once it can. Every one
    /// is pushed however the one before fared, and the first failure
    /// answered.
    pub(super) fn push_admitted(
        &mut self,
        mut buffer: MediaBuffer,
        pad: &mut SrcPad,
    ) -> crate::error::Result<()> {
        self.describe(&mut buffer);
        let mut first = Ok(());
        for buffer in self.admit(buffer).into_iter().flatten() {
            let pushed = self.stash.push(pad, buffer);
            if first.is_ok() {
                first = pushed;
            }
        }
        first
    }

    /// Says on a decoded frame what unit its `pts` is in — the packets' own,
    /// which FFmpeg's decoders do not copy onto what they produce. Every
    /// decoder here sends its output through this gate, so this is the one
    /// place each decoded frame is described; see [`crate::buffer::time_base`]
    /// for what reads it.
    ///
    /// The frame arrives freshly wrapped, so its `Arc` is still unshared and
    /// can be written through; one that is not is left as it is rather than
    /// copied.
    fn describe(&self, buffer: &mut MediaBuffer) {
        let Some(time_base) = self.time_base else {
            return;
        };
        match buffer {
            MediaBuffer::Video(frame) => {
                if let Some(frame) = std::sync::Arc::get_mut(frame) {
                    crate::buffer::set_time_base(frame, time_base);
                }
            }
            MediaBuffer::Audio(frame) => {
                if let Some(frame) = std::sync::Arc::get_mut(frame) {
                    crate::buffer::set_time_base(frame, time_base);
                }
            }
            MediaBuffer::Packet(_) | MediaBuffer::Eos => {}
        }
    }

    /// Ahead of the end of the stream, through `pad`: everything kept,
    /// whatever `pad` says, since after the end nothing would take it; and
    /// where no sample reached the target, the last one before it — the
    /// stream's last presentable, which is what a seek past it shows.
    pub(super) fn push_eos_candidate(&mut self, pad: &mut SrcPad) -> crate::error::Result<()> {
        let kept = self.stash.release_all(pad);
        let candidate = self.candidate.take().filter(|_| !self.reached);
        self.reached = true;
        match candidate {
            Some(candidate) => kept.and(pad.push(candidate)),
            None => kept,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    };
    use std::time::Duration;

    use super::*;
    use crate::control::{ControlMsg, PrerollContext};
    use crate::element::{Element, ElementType, Sink, element_pp_log};
    use crate::pool::UnboundObjectPool;
    use crate::pp_log::PpLog;

    fn millis(time_base: ffmpeg::Rational) -> ffmpeg::Packet {
        let mut packet = ffmpeg::Packet::empty();
        packet.set_time_base(time_base);
        packet
    }

    fn video(pts: i64) -> MediaBuffer {
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut frame = pool.get();
        frame.set_pts(Some(pts));
        MediaBuffer::Video(std::sync::Arc::new(frame))
    }

    /// The segment an accurate seek to `at` begins: what is shown begins at
    /// `at`.
    fn showing(at: Duration) -> Arc<Segment> {
        Arc::new(Segment {
            id: 2,
            flushed: true,
            position: at,
            start: at,
            show_from: Some(at),
            backwards: false,
        })
    }

    /// What a decoder's pad leads to, as a test sees it: the pictures it
    /// took by `pts`, taking one only while open — and, like a terminal that
    /// has its preroll sample, shutting after each where asked to — and
    /// refusing the first `refusals`.
    struct Screen {
        pp_log: PpLog,
        open: Arc<AtomicBool>,
        shuts: bool,
        refusals: Arc<AtomicUsize>,
        taken: Arc<Mutex<Vec<i64>>>,
    }

    impl Element for Screen {
        fn name(&self) -> Arc<str> {
            "screen".into()
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

    impl Sink for Screen {
        fn ready_consume(&mut self) -> bool {
            self.open.load(Ordering::SeqCst)
        }
        fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
            if self
                .refusals
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |left| {
                    left.checked_sub(1)
                })
                .is_ok()
            {
                return Err(ffmpeg::Error::InvalidData.into());
            }
            if let MediaBuffer::Video(frame) = &buf {
                self.taken.lock().unwrap().push(frame.pts().unwrap_or(-1));
            }
            if self.shuts {
                self.open.store(false, Ordering::SeqCst);
            }
            Ok(())
        }
    }

    struct Linked {
        pad: SrcPad,
        open: Arc<AtomicBool>,
        refusals: Arc<AtomicUsize>,
        taken: Arc<Mutex<Vec<i64>>>,
    }

    /// A pad linked to a [`Screen`], open, that shuts after each picture
    /// where `shuts`.
    fn screen(shuts: bool) -> Linked {
        let open = Arc::new(AtomicBool::new(true));
        let refusals = Arc::new(AtomicUsize::new(0));
        let taken = Arc::new(Mutex::new(Vec::new()));
        let mut pad = SrcPad::new("src");
        pad.link(Box::new(Screen {
            pp_log: element_pp_log(ElementType::Other, "screen", None),
            open: Arc::clone(&open),
            shuts,
            refusals: Arc::clone(&refusals),
            taken: Arc::clone(&taken),
        }));
        Linked {
            pad,
            open,
            refusals,
            taken,
        }
    }

    /// A gate reading its pipeline's state, in a preroll, and the state.
    fn prerolling() -> (PrerollGate, Arc<PlaybackState>) {
        let state = PlaybackState::new();
        let mut gate = PrerollGate::default();
        gate.attach(&state);
        state.observe(&ControlMsg::Preroll(Arc::new(PrerollContext::new([]))));
        (gate, state)
    }

    #[test]
    fn hardware_surface_budget_includes_the_internal_seek_candidate() {
        assert_eq!(hw_surface_budget(8), Some(9));
        assert_eq!(hw_surface_budget(-1), None);
        assert_eq!(hw_surface_budget(i32::MAX), None);
    }

    #[test]
    fn suppresses_only_what_precedes_the_target() {
        let mut gate = PrerollGate::default();
        let mut out = screen(false);
        gate.observe_packet(&millis(ffmpeg::Rational(1, 1_000)));
        gate.begin_segment(&showing(Duration::from_secs(2)), &mut out.pad)
            .unwrap();

        assert!(gate.suppresses(Some(1_999)));
        assert!(!gate.suppresses(Some(2_000)), "the target itself passes");
        assert!(!gate.suppresses(Some(5_000)));
    }

    /// The unit matters, not the raw number: 1999 ticks is before a 2s target
    /// at millisecond resolution and long after it at microsecond resolution.
    #[test]
    fn reads_the_pts_in_the_packets_own_time_base() {
        let mut gate = PrerollGate::default();
        let mut out = screen(false);
        gate.begin_segment(&showing(Duration::from_secs(2)), &mut out.pad)
            .unwrap();

        gate.observe_packet(&millis(ffmpeg::Rational(1, 1_000_000)));
        assert!(gate.suppresses(Some(1_999_999)));
        assert!(!gate.suppresses(Some(2_000_000)));
    }

    /// A stream starting, or a seek to a keyframe, shows everything from
    /// wherever the source landed.
    #[test]
    fn a_segment_without_a_target_cuts_nothing() {
        let mut gate = PrerollGate::default();
        let mut out = screen(false);
        gate.observe_packet(&millis(ffmpeg::Rational(1, 1_000)));
        assert!(!gate.suppresses(Some(0)), "no segment at all");
        let start = Arc::new(Segment {
            show_from: None,
            ..Arc::into_inner(showing(Duration::ZERO)).expect("only one")
        });
        gate.begin_segment(&start, &mut out.pad).unwrap();
        assert!(!gate.suppresses(Some(0)));
    }

    /// Freezing a branch is worse than briefly showing early media, so every
    /// missing input passes through instead of guessing.
    #[test]
    fn missing_time_base_or_pts_fails_open() {
        let mut gate = PrerollGate::default();
        let mut out = screen(false);
        gate.begin_segment(&showing(Duration::from_secs(2)), &mut out.pad)
            .unwrap();
        assert!(!gate.suppresses(Some(0)), "no time base yet");

        gate.observe_packet(&millis(ffmpeg::Rational(1, 1_000)));
        assert!(!gate.suppresses(None), "no pts");
        assert!(gate.suppresses(Some(0)), "both known again");
    }

    /// Whatever the phase — a preroll that ended before this decoder got
    /// there, a step back's sound — what precedes the target is dropped
    /// until a sample reaches it, then everything passes; a `Flush` forgets
    /// the target and the unit with it.
    #[test]
    fn the_target_is_the_segments_until_reached_and_a_flush_forgets_it() {
        let mut gate = PrerollGate::default();
        let mut out = screen(false);
        gate.observe_packet(&millis(ffmpeg::Rational(1, 1_000)));
        gate.begin_segment(&showing(Duration::from_secs(2)), &mut out.pad)
            .unwrap();
        for pts in [1_000, 1_500, 2_000, 2_033] {
            gate.push_admitted(video(pts), &mut out.pad).unwrap();
        }
        assert_eq!(*out.taken.lock().unwrap(), [2_000, 2_033]);
        assert!(!gate.suppresses(Some(0)), "reached: nothing more cut");

        gate.begin_segment(&showing(Duration::from_secs(3)), &mut out.pad)
            .unwrap();
        gate.reset();
        assert!(gate.time_base.is_none());
        assert!(gate.target_ns().is_none(), "nor is the target kept");
    }

    /// A segment begun anew owes its own target, whatever the one before
    /// reached.
    #[test]
    fn each_segment_owes_its_own_target() {
        let mut gate = PrerollGate::default();
        let mut out = screen(false);
        gate.observe_packet(&millis(ffmpeg::Rational(1, 1_000)));
        gate.begin_segment(&showing(Duration::from_secs(2)), &mut out.pad)
            .unwrap();
        gate.push_admitted(video(2_000), &mut out.pad).unwrap();
        gate.push_admitted(video(1_000), &mut out.pad).unwrap();

        gate.begin_segment(&showing(Duration::from_secs(3)), &mut out.pad)
            .unwrap();
        gate.push_admitted(video(2_500), &mut out.pad).unwrap();
        gate.push_admitted(video(3_000), &mut out.pad).unwrap();
        assert_eq!(*out.taken.lock().unwrap(), [2_000, 1_000, 3_000]);
    }

    /// Played backwards the target is where the stretches start from, and
    /// every sample is below it: none is dropped for being so. What the
    /// conformance matrix found when the target first came from the
    /// segment — a backwards accurate seek showed nothing at all.
    #[test]
    fn a_backwards_segment_drops_nothing_below_its_target() {
        let mut gate = PrerollGate::default();
        let mut out = screen(false);
        gate.observe_packet(&millis(ffmpeg::Rational(1, 1_000)));
        let backwards = Arc::new(Segment {
            backwards: true,
            ..Arc::into_inner(showing(Duration::from_secs(2))).expect("only one")
        });
        gate.begin_segment(&backwards, &mut out.pad).unwrap();
        gate.push_admitted(video(1_000), &mut out.pad).unwrap();
        assert_eq!(*out.taken.lock().unwrap(), [1_000]);
        assert!(!gate.suppresses(Some(0)));
    }

    /// The frame shown at the target is the one covering it: the last
    /// before it, once the next proves it covers the instant — then that
    /// next one.
    #[test]
    fn video_shows_the_frame_covering_the_target_first() {
        let mut gate = PrerollGate::default();
        let mut out = screen(false);
        gate.observe_packet(&millis(ffmpeg::Rational(1, 1_000)));
        gate.begin_segment(&showing(Duration::from_secs(2)), &mut out.pad)
            .unwrap();

        gate.push_admitted(video(1_967), &mut out.pad).unwrap();
        assert!(
            out.taken.lock().unwrap().is_empty(),
            "not known to cover yet"
        );
        gate.push_admitted(video(2_033), &mut out.pad).unwrap();
        assert_eq!(*out.taken.lock().unwrap(), [1_967, 2_033]);
    }

    /// In a preroll a terminal that has its sample takes nothing more: what
    /// is decoded after it — the picture that proved which one covered the
    /// target, and the rest of a decoder's burst — is kept rather than
    /// pushed into it, and the decoder is not fed meanwhile. Once playback
    /// goes on it goes first, in order: it is what comes next, what a step
    /// forward after a seek shows and where playing on starts.
    #[test]
    fn in_a_preroll_what_the_terminal_will_not_take_is_kept_in_order() {
        let (mut gate, state) = prerolling();
        let mut out = screen(true);
        gate.observe_packet(&millis(ffmpeg::Rational(1, 1_000)));
        gate.begin_segment(&showing(Duration::from_secs(2)), &mut out.pad)
            .unwrap();

        for pts in [1_967, 2_033, 2_067] {
            gate.push_admitted(video(pts), &mut out.pad).unwrap();
        }
        assert_eq!(*out.taken.lock().unwrap(), [1_967], "the sample alone");
        assert!(!gate.ready(&mut out.pad), "and the decoder is not fed");

        state.observe(&ControlMsg::Pause);
        state.observe(&ControlMsg::Resume);
        out.open.store(true, Ordering::SeqCst);
        assert!(gate.ready(&mut out.pad));
        gate.push_admitted(video(2_100), &mut out.pad).unwrap();
        assert_eq!(
            *out.taken.lock().unwrap(),
            [1_967, 2_033, 2_067, 2_100],
            "nothing lost, nothing out of order"
        );
    }

    /// A seek to a keyframe shows the first sample decoded, and a preroll
    /// keeps what follows it the same way.
    #[test]
    fn a_keyframe_preroll_shows_the_first_sample_and_keeps_the_rest() {
        let (mut gate, _state) = prerolling();
        let mut out = screen(true);
        gate.push_admitted(video(1_000), &mut out.pad).unwrap();
        gate.push_admitted(video(1_033), &mut out.pad).unwrap();
        assert_eq!(*out.taken.lock().unwrap(), [1_000]);
        assert!(!gate.ready(&mut out.pad));
    }

    /// A push the pad refuses is answered; the next sample still goes.
    #[test]
    fn a_refused_push_is_answered_and_the_next_sample_goes() {
        let mut gate = PrerollGate::default();
        let mut out = screen(false);
        gate.observe_packet(&millis(ffmpeg::Rational(1, 1_000)));
        gate.begin_segment(&showing(Duration::from_secs(2)), &mut out.pad)
            .unwrap();
        out.refusals.store(1, Ordering::SeqCst);

        assert!(gate.push_admitted(video(2_000), &mut out.pad).is_err());
        gate.push_admitted(video(2_033), &mut out.pad).unwrap();
        assert_eq!(*out.taken.lock().unwrap(), [2_033]);
    }

    /// A seek past the last picture shows the last there is, handed on at
    /// the end of the stream.
    #[test]
    fn the_end_shows_the_last_presentable_frame() {
        let mut gate = PrerollGate::default();
        let mut out = screen(false);
        gate.observe_packet(&millis(ffmpeg::Rational(1, 1_000)));
        gate.begin_segment(&showing(Duration::from_secs(2)), &mut out.pad)
            .unwrap();

        gate.push_admitted(video(1_933), &mut out.pad).unwrap();
        gate.push_admitted(video(1_967), &mut out.pad).unwrap();
        gate.push_eos_candidate(&mut out.pad).unwrap();
        assert_eq!(*out.taken.lock().unwrap(), [1_967]);
    }

    /// Ahead of the end, and ahead of a segment, what is kept goes
    /// whatever the pad says: nothing would take it after the end, and an
    /// event must not overtake what came before it.
    #[test]
    fn what_is_kept_goes_ahead_of_the_end_and_of_a_segment() {
        let (mut gate, _state) = prerolling();
        let mut out = screen(true);
        gate.push_admitted(video(1_000), &mut out.pad).unwrap();
        gate.push_admitted(video(1_033), &mut out.pad).unwrap();
        gate.push_eos_candidate(&mut out.pad).unwrap();
        assert_eq!(*out.taken.lock().unwrap(), [1_000, 1_033]);

        gate.push_admitted(video(1_067), &mut out.pad).unwrap();
        let lap = Arc::new(Segment {
            flushed: false,
            show_from: None,
            ..Arc::into_inner(showing(Duration::ZERO)).expect("only one")
        });
        gate.begin_segment(&lap, &mut out.pad).unwrap();
        assert_eq!(*out.taken.lock().unwrap(), [1_000, 1_033, 1_067]);
    }

    #[test]
    fn audio_frame_crossing_the_target_is_not_discarded() {
        let mut gate = PrerollGate::default();
        let mut out = screen(false);
        gate.observe_packet(&millis(ffmpeg::Rational(1, 48_000)));
        gate.begin_segment(&showing(Duration::from_millis(10)), &mut out.pad)
            .unwrap();
        let mut frame = ffmpeg::frame::Audio::new(
            ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed),
            1_024,
            ffmpeg::ChannelLayout::MONO,
        );
        frame.set_rate(48_000);
        frame.set_pts(Some(0));

        assert!(
            matches!(
                gate.admit(MediaBuffer::Audio(std::sync::Arc::new(frame))),
                [None, Some(MediaBuffer::Audio(_))]
            ),
            "0..21.3ms audio covers a 10ms target"
        );
    }
}
