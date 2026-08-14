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
use ffmpeg_next::{Rescale, Rounding};
use thiserror::Error as ThisError;

/// `numerator`/`denominator` were not both positive — `0/1` (FFmpeg's own
/// "unset" sentinel), a zero denominator, or a negative value are all
/// rejected here rather than left to produce `inf`/`NaN`/a division panic
/// somewhere downstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ThisError)]
#[error("invalid time base {numerator}/{denominator}: both numerator and denominator must be positive")]
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
    /// control.
    pub fn new_unchecked(value: ffmpeg::Rational) -> Self {
        Self(value)
    }

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
    pub fn try_new(pts: i64, time_base: ffmpeg::Rational) -> Result<Self, InvalidTimeBase> {
        Ok(Self {
            pts,
            time_base: TimeBase::try_new(time_base)?,
        })
    }

    /// For a caller already holding a proven-valid [`TimeBase`] (from an
    /// earlier [`TimeBase::try_new`]) that wants to pair a new `pts` with
    /// it without re-validating.
    pub fn new_unchecked(pts: i64, time_base: TimeBase) -> Self {
        Self { pts, time_base }
    }

    pub fn pts(self) -> i64 {
        self.pts
    }

    pub fn time_base(self) -> TimeBase {
        self.time_base
    }

    /// Rescales `pts` into `target`'s units using FFmpeg's default
    /// rounding (nearest, ties away from zero). Callers where early vs.
    /// late matters more than exact nearness — a packet timestamp, a
    /// duration, a presentation deadline can each have a different
    /// tolerance — should use [`MediaTimestamp::rescale_with`] and pick a
    /// rounding explicitly instead.
    pub fn rescale(self, target: TimeBase) -> i64 {
        self.pts.rescale(self.time_base.0, target.0)
    }

    /// Same as [`MediaTimestamp::rescale`], with an explicit rounding
    /// policy.
    pub fn rescale_with(self, target: TimeBase, rounding: Rounding) -> i64 {
        self.pts.rescale_with(self.time_base.0, target.0, rounding)
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
            assert!(MediaTimestamp::try_new(1, rational).is_err());
        }
    }

    #[test]
    fn rescale_matches_hand_computed_seconds() {
        // 30 ticks of 1001/30_000s each = 1.001s = 30_030/30_000.
        let timestamp = MediaTimestamp::try_new(30, ffmpeg::Rational::new(1001, 30_000)).unwrap();
        let seconds = TimeBase::try_new(ffmpeg::Rational::new(1, 30_000)).unwrap();

        assert_eq!(timestamp.rescale(seconds), 30_030);
    }

    #[test]
    fn rescale_is_exact_for_a_unit_numerator_time_base() {
        let timestamp = MediaTimestamp::try_new(3_000, ffmpeg::Rational::new(1, 90_000)).unwrap();
        let same_base = TimeBase::try_new(ffmpeg::Rational::new(1, 90_000)).unwrap();

        assert_eq!(timestamp.rescale(same_base), 3_000);
    }
}
