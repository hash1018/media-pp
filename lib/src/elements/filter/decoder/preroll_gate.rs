use ffmpeg_next::{self as ffmpeg, Rescale};

use crate::control::PrerollContext;

const NANOS: ffmpeg::Rational = ffmpeg::Rational(1, 1_000_000_000);

/// Suppresses decoded media that precedes a seek target.
///
/// A seek lands on the keyframe at or before the requested position, so a
/// decoder has to decode forward from there to rebuild reference state. Only
/// the samples at or after the target may be delivered; the rest exist purely
/// to warm the codec up.
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
}

impl PrerollGate {
    /// Arms the gate for `context`, or disarms it when that preroll carries no
    /// target (a first-sample preroll has nothing to suppress).
    pub(super) fn begin(&mut self, context: &PrerollContext) {
        self.target_ns = context
            .target()
            .map(|target| target.as_nanos().min(i64::MAX as u128) as i64);
    }

    /// Ends suppression. Pause and Resume both restore ordinary delivery: the
    /// preroll they belong to is over either way.
    pub(super) fn clear(&mut self) {
        self.target_ns = None;
    }

    /// Forgets the learned time base along with any target, for a `Flush` that
    /// begins a new timeline.
    pub(super) fn reset(&mut self) {
        self.target_ns = None;
        self.time_base = None;
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
    pub(super) fn suppresses(&self, pts: Option<i64>) -> bool {
        let (Some(target_ns), Some(time_base), Some(pts)) = (self.target_ns, self.time_base, pts)
        else {
            return false;
        };
        pts.rescale(time_base, NANOS) < target_ns
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn millis(time_base: ffmpeg::Rational) -> ffmpeg::Packet {
        let mut packet = ffmpeg::Packet::empty();
        packet.set_time_base(time_base);
        packet
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
}
