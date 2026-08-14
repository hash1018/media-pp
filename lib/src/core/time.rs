//! A validated `(pts, time_base)` pair — [`MediaTimestamp`] — plus the
//! [`TimeBase`] it's built on. Exists because the pieces it replaces kept
//! going wrong in the same specific way: an `i64` `pts` traveling
//! separately from the `ffmpeg::Rational` it's expressed in, with no
//! single place enforcing that the pair is even valid (`numerator`/
//! `denominator` both positive) before something downstream divides by
//! it. Two real bugs came from exactly that — [`Pacer`] converting `pts`
//! to seconds via `elapsed_ticks as f64 * f64::from(time_base)` (silently
//! imprecise for a long-running stream), and `WebRtcPeer::write_track`
//! building `str0m`'s `MediaTime` from `pts / time_base.denominator()`
//! (silently wrong for any non-unit numerator, e.g. NTSC's `1001/30_000`).
//!
//! Deliberately thin: rescaling itself is delegated to
//! [`ffmpeg_next::Rescale`], which wraps FFmpeg's own `av_rescale_q_rnd`
//! rather than reimplementing overflow-safe rational arithmetic here.
//! Backend-specific conversions (e.g. to `str0m`'s `MediaTime`) stay in
//! their own backend module, not here — see
//! `crate::elements::driver::webrtc::peer::to_str0m_media_time`.
//!
//! [`Pacer`]: crate::elements::Pacer

use ffmpeg_next as ffmpeg;
use ffmpeg_next::Rescale;
use thiserror::Error as ThisError;

/// `numerator`/`denominator` were not both positive — `0/1` (FFmpeg's own
/// "unset" sentinel), a zero denominator, or a negative value are all
/// rejected here rather than left to produce `inf`/`NaN`/a division panic
/// somewhere downstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ThisError)]
#[error(
    "invalid time base {numerator}/{denominator}: both numerator and denominator must be positive"
)]
pub struct InvalidTimeBase {
    pub numerator: i32,
    pub denominator: i32,
}

/// An [`ffmpeg::Rational`] known to have a positive numerator and
/// denominator — the only two values [`InvalidTimeBase::try_new`]
/// rejects, `try_new` is the only way to build one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeBase(ffmpeg::Rational);

impl TimeBase {
    pub fn try_new(value: ffmpeg::Rational) -> Result<Self, InvalidTimeBase> {
        if value.numerator() <= 0 || value.denominator() <= 0 {
            return Err(InvalidTimeBase {
                numerator: value.numerator(),
                denominator: value.denominator(),
            });
        }
        Ok(Self(value))
    }

    /// For a `value` the caller already knows is valid — e.g. a hardcoded
    /// constant like nanoseconds (`1/1_000_000_000`) — where re-deriving
    /// that via `try_new` on every call would be pure overhead. Prefer
    /// `try_new` whenever `value` originates outside this crate's own
    /// control. `pub(crate)` rather than `pub`: this whole module is
    /// crate-private already (see `lib.rs`), but marking the unvalidated
    /// escape hatch itself is worth the redundancy in case that ever
    /// changes without this callsite being revisited.
    pub(crate) fn new_unchecked(value: ffmpeg::Rational) -> Self {
        Self(value)
    }

    /// Only [`crate::elements::driver::webrtc::peer::to_str0m_media_time`]
    /// needs the raw `ffmpeg::Rational` back out today — `#[cfg]`-gated
    /// rather than left for a hypothetical future caller (this module has
    /// no `#[allow(dead_code)]` precedent elsewhere in this crate).
    #[cfg(feature = "webrtc")]
    pub fn get(self) -> ffmpeg::Rational {
        self.0
    }
}

/// A `pts` paired with the [`TimeBase`] it's expressed in — inseparable
/// once constructed, unlike passing the two around as independent `i64`/
/// `ffmpeg::Rational` values.
#[derive(Debug, Clone, Copy)]
pub struct MediaTimestamp {
    pts: i64,
    time_base: TimeBase,
}

impl MediaTimestamp {
    /// Only [`crate::elements::driver::webrtc::peer::packet_rtp_time`]
    /// builds a `MediaTimestamp` straight from an unvalidated
    /// `ffmpeg::Rational` today; [`Pacer`] already holds a validated
    /// [`TimeBase`] and uses [`MediaTimestamp::new_unchecked`] instead. See
    /// [`TimeBase::get`]'s own doc for why this is `#[cfg]`-gated rather
    /// than kept for a hypothetical future caller.
    ///
    /// [`Pacer`]: crate::elements::Pacer
    #[cfg(feature = "webrtc")]
    pub fn try_new(pts: i64, time_base: ffmpeg::Rational) -> Result<Self, InvalidTimeBase> {
        Ok(Self {
            pts,
            time_base: TimeBase::try_new(time_base)?,
        })
    }

    /// For a caller already holding a proven-valid [`TimeBase`] (from an
    /// earlier [`TimeBase::try_new`]) that wants to pair a new `pts` with
    /// it without re-validating. `pub(crate)` for the same reason as
    /// [`TimeBase::new_unchecked`].
    pub(crate) fn new_unchecked(pts: i64, time_base: TimeBase) -> Self {
        Self { pts, time_base }
    }

    #[cfg(feature = "webrtc")]
    pub fn pts(self) -> i64 {
        self.pts
    }

    #[cfg(feature = "webrtc")]
    pub fn time_base(self) -> TimeBase {
        self.time_base
    }

    /// Rescales `pts` into `target`'s units using FFmpeg's default
    /// rounding (nearest, ties away from zero). A caller where early vs.
    /// late matters more than exact nearness — a packet timestamp, a
    /// duration, a presentation deadline can each have a different
    /// tolerance — should reach for [`ffmpeg_next::Rescale::rescale_with`]
    /// directly (`self.pts().rescale_with(self.time_base().get(),
    /// target.get(), rounding)`) and pick a rounding explicitly instead;
    /// not wrapped here since nothing in this crate needs it yet.
    pub fn rescale(self, target: TimeBase) -> i64 {
        self.pts.rescale(self.time_base.0, target.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_new_rejects_non_positive_numerator_or_denominator() {
        for (numerator, denominator) in [(0, 1), (1, 0), (-1, 1), (1, -1), (0, 0)] {
            let rational = ffmpeg::Rational::new(numerator, denominator);
            assert_eq!(
                TimeBase::try_new(rational),
                Err(InvalidTimeBase {
                    numerator,
                    denominator
                }),
                "expected {numerator}/{denominator} to be rejected"
            );
        }
    }

    #[cfg(feature = "webrtc")]
    #[test]
    fn media_timestamp_try_new_rejects_the_same_invalid_time_bases() {
        for (numerator, denominator) in [(0, 1), (1, 0), (-1, 1), (1, -1), (0, 0)] {
            let rational = ffmpeg::Rational::new(numerator, denominator);
            assert!(MediaTimestamp::try_new(1, rational).is_err());
        }
    }

    #[test]
    fn rescale_matches_hand_computed_seconds() {
        // 30 ticks of 1001/30_000s each = 1.001s = 30_030/30_000.
        let time_base = TimeBase::try_new(ffmpeg::Rational::new(1001, 30_000)).unwrap();
        let timestamp = MediaTimestamp::new_unchecked(30, time_base);
        let seconds = TimeBase::try_new(ffmpeg::Rational::new(1, 30_000)).unwrap();

        assert_eq!(timestamp.rescale(seconds), 30_030);
    }

    #[test]
    fn rescale_is_exact_for_a_unit_numerator_time_base() {
        let time_base = TimeBase::try_new(ffmpeg::Rational::new(1, 90_000)).unwrap();
        let timestamp = MediaTimestamp::new_unchecked(3_000, time_base);
        let same_base = TimeBase::try_new(ffmpeg::Rational::new(1, 90_000)).unwrap();

        assert_eq!(timestamp.rescale(same_base), 3_000);
    }
}
