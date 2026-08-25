use ffmpeg_next::{self as ffmpeg, Rescale};

use crate::buffer::MediaBuffer;
use crate::control::PrerollContext;

const NANOS: ffmpeg::Rational = ffmpeg::Rational(1, 1_000_000_000);

/// Suppresses decoded media that precedes a seek target.
///
/// A seek lands on the keyframe at or before the requested position, so a
/// decoder has to decode forward from there to rebuild reference state. Video
/// holds one preceding frame until the next PTS proves which frame covers the
/// requested instant; audio uses its sample count and rate for the same
/// decision. Earlier output exists purely to warm the codec up.
///
/// # Why this lives in the decoder
///
/// Deciding "is this sample before the target" needs a PTS *and* the time base
/// to read it in. A decoder is the only stage that has both on every decoded
/// branch: `MediaBuffer::Video`/`Audio` carry a raw `pts` with no time base
/// attached, so a downstream stage can only interpret it if it was separately
/// constructed with one. [`Pacer`](crate::elements::Pacer) and
/// [`VideoSynchronizer`](crate::elements::VideoSynchronizer) were, which is
/// why the gate first lived there — but an audio branch has neither (an audio
/// renderer paces itself against its own device clock), so that branch went
/// ungated and delivered its whole pre-target span.
///
/// # What this deliberately does not do
///
/// It does not decide when preroll is *complete*. That belongs to the terminal
/// which reports its first accepted sample (see
/// [`PrerollContext::mark_ready`]). Keeping the two apart is what lets this
/// stay a plain filter over one stream: no shared completion state, no
/// coordination between the decoders on sibling branches, and no notion of
/// which stream "owns" the seek.
#[derive(Default)]
pub(super) struct PrerollGate {
    /// `None` outside a seek preroll, which is the pass-everything state.
    target_ns: Option<i64>,
    /// Learned from the packets being fed in; decoded frames carry a `pts`
    /// in this unit but not the unit itself.
    time_base: Option<ffmpeg::Rational>,
    /// Last decoded sample before the target. Video needs one-sample
    /// lookahead to select the frame that actually covers the requested
    /// instant, and EOF uses the same candidate as its last-presentable
    /// fallback.
    candidate: Option<MediaBuffer>,
}

impl PrerollGate {
    /// Arms the gate for `context`, or disarms it when that preroll carries no
    /// target (a first-sample preroll has nothing to suppress).
    pub(super) fn begin(&mut self, context: &PrerollContext) {
        self.candidate = None;
        self.target_ns = context
            .target()
            .map(|target| target.as_nanos().min(i64::MAX as u128) as i64);
    }

    /// Ends suppression. Pause and Resume both restore ordinary delivery: the
    /// preroll they belong to is over either way.
    pub(super) fn clear(&mut self) {
        self.target_ns = None;
        self.candidate = None;
    }

    /// Forgets the learned time base along with any target, for a `Flush` that
    /// begins a new timeline.
    pub(super) fn reset(&mut self) {
        self.target_ns = None;
        self.time_base = None;
        self.candidate = None;
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

    /// Whether this decoded sample precedes the seek target and must not be
    /// forwarded.
    ///
    /// Fails open — a missing target, time base, or PTS all pass the sample
    /// through. Suppressing on a guess would freeze the branch outright, and a
    /// branch that shows slightly early media is recoverable where one that
    /// shows none is not.
    #[cfg(test)]
    pub(super) fn suppresses(&self, pts: Option<i64>) -> bool {
        let (Some(target_ns), Some(time_base), Some(pts)) = (self.target_ns, self.time_base, pts)
        else {
            return false;
        };
        pts.rescale(time_base, NANOS) < target_ns
    }

    /// Admits one decoded sample, retaining at most one pre-target candidate.
    pub(super) fn admit(&mut self, buffer: MediaBuffer) -> Option<MediaBuffer> {
        let Some(target_ns) = self.target_ns else {
            return Some(buffer);
        };
        let Some(time_base) = self.time_base else {
            self.clear();
            return Some(buffer);
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
            _ => {
                self.clear();
                return Some(buffer);
            }
        };
        let Some(pts) = pts else {
            self.clear();
            return Some(buffer);
        };
        let start_ns = pts.rescale(time_base, NANOS);

        // An audio frame crossing the target is the audio that exists at the
        // requested instant; do not discard the whole frame just because its
        // first sample precedes the target.
        if audio_end_ns.is_some_and(|end| start_ns <= target_ns && end > target_ns) {
            self.clear();
            return Some(buffer);
        }

        if start_ns < target_ns {
            self.candidate = Some(buffer);
            return None;
        }

        let selected = if start_ns > target_ns && matches!(&buffer, MediaBuffer::Video(_)) {
            self.candidate.take().unwrap_or(buffer)
        } else {
            buffer
        };
        self.clear();
        Some(selected)
    }

    /// Selects the last decoded pre-target sample when EOS proves no later
    /// sample can cover the requested instant.
    pub(super) fn finish_on_eos(&mut self) -> Option<MediaBuffer> {
        self.target_ns = None;
        self.candidate.take()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::pool::UnboundObjectPool;

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

    #[test]
    fn suppresses_only_what_precedes_the_target() {
        let mut gate = PrerollGate::default();
        gate.observe_packet(&millis(ffmpeg::Rational(1, 1_000)));
        gate.begin(&PrerollContext::for_seek([], Duration::from_secs(2)));

        assert!(gate.suppresses(Some(1_999)));
        assert!(!gate.suppresses(Some(2_000)), "the target itself passes");
        assert!(!gate.suppresses(Some(5_000)));
    }

    /// The unit matters, not the raw number: 1999 ticks is before a 2s target
    /// at millisecond resolution and long after it at microsecond resolution.
    #[test]
    fn reads_the_pts_in_the_packets_own_time_base() {
        let mut gate = PrerollGate::default();
        gate.begin(&PrerollContext::for_seek([], Duration::from_secs(2)));

        gate.observe_packet(&millis(ffmpeg::Rational(1, 1_000_000)));
        assert!(gate.suppresses(Some(1_999_999)));
        assert!(!gate.suppresses(Some(2_000_000)));
    }

    #[test]
    fn a_preroll_without_a_target_suppresses_nothing() {
        let mut gate = PrerollGate::default();
        gate.observe_packet(&millis(ffmpeg::Rational(1, 1_000)));
        gate.begin(&PrerollContext::new([]));

        assert!(!gate.suppresses(Some(0)));
    }

    /// Freezing a branch is worse than briefly showing early media, so every
    /// missing input passes through instead of guessing.
    #[test]
    fn missing_time_base_or_pts_fails_open() {
        let mut gate = PrerollGate::default();
        gate.begin(&PrerollContext::for_seek([], Duration::from_secs(2)));
        assert!(!gate.suppresses(Some(0)), "no time base yet");

        gate.observe_packet(&millis(ffmpeg::Rational(1, 1_000)));
        assert!(!gate.suppresses(None), "no pts");
        assert!(gate.suppresses(Some(0)), "both known again");
    }

    #[test]
    fn pause_and_resume_end_suppression_but_flush_also_forgets_the_unit() {
        let mut gate = PrerollGate::default();
        gate.observe_packet(&millis(ffmpeg::Rational(1, 1_000)));
        gate.begin(&PrerollContext::for_seek([], Duration::from_secs(2)));

        gate.clear();
        assert!(!gate.suppresses(Some(0)));
        assert!(gate.time_base.is_some(), "the unit outlives one preroll");

        gate.begin(&PrerollContext::for_seek([], Duration::from_secs(2)));
        gate.reset();
        assert!(gate.time_base.is_none());
    }

    #[test]
    fn video_selects_the_frame_covering_the_target() {
        let mut gate = PrerollGate::default();
        gate.observe_packet(&millis(ffmpeg::Rational(1, 1_000)));
        gate.begin(&PrerollContext::for_seek([], Duration::from_secs(2)));

        assert!(gate.admit(video(1_967)).is_none());
        let selected = gate.admit(video(2_033)).expect("one frame selected");
        let MediaBuffer::Video(selected) = selected else {
            panic!("selected a non-video buffer");
        };
        assert_eq!(selected.pts(), Some(1_967));
    }

    #[test]
    fn eos_selects_the_last_presentable_frame() {
        let mut gate = PrerollGate::default();
        gate.observe_packet(&millis(ffmpeg::Rational(1, 1_000)));
        gate.begin(&PrerollContext::for_seek([], Duration::from_secs(2)));

        assert!(gate.admit(video(1_967)).is_none());
        let selected = gate.finish_on_eos().expect("last frame selected");
        let MediaBuffer::Video(selected) = selected else {
            panic!("selected a non-video buffer");
        };
        assert_eq!(selected.pts(), Some(1_967));
    }

    #[test]
    fn audio_frame_crossing_the_target_is_not_discarded() {
        let mut gate = PrerollGate::default();
        gate.observe_packet(&millis(ffmpeg::Rational(1, 48_000)));
        gate.begin(&PrerollContext::for_seek([], Duration::from_millis(10)));
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
                Some(MediaBuffer::Audio(_))
            ),
            "0..21.3ms audio covers a 10ms target"
        );
    }
}
