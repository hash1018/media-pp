//! Whether an element that makes a stream out of its inputs keeps to the
//! wall clock or to the timestamps it is given.

use std::{cmp::Ordering, time::Duration};

use ffmpeg_next as ffmpeg;

/// Whether a compositor makes its output as time passes or as its inputs'
/// timestamps say.
///
/// A choice made at construction, not a switch: a live element and an
/// offline one differ in what a pipeline may do with them — whether it can
/// be sought, whether it can preroll — and a pipeline settles that when it
/// is wired (see [`crate::element::SourceElement::is_live`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RenderMode {
    /// Emits at its own rate by the wall clock, drawing whatever each input
    /// last handed over. What a preview, a recording or a broadcast wants:
    /// an input that is late is shown late rather than waited for.
    #[default]
    Live,
    /// Emits the frame for each output time as soon as every input has
    /// said what it shows at that time, and goes straight on to the next —
    /// faster than real time where decoding allows, slower where it does
    /// not, and the same output either way. What an export wants.
    ///
    /// An input's frames are placed by their own timestamps, read in their
    /// own time base, so the inputs need not share one. An input is shown
    /// from each frame's timestamp until its duration runs out or the next
    /// frame begins; it holds back whatever feeds it once it is a frame
    /// ahead of the output, which needs a [`crate::queue::Queue`] somewhere
    /// upstream in its pipeline to wait on.
    Offline {
        /// The output time the stream ends at. `None` ends it once every
        /// input fed through a sink has ended and been shown to its last
        /// frame's end; `Some` renders exactly up to this time, filling what
        /// no input covers with the background.
        end: Option<Duration>,
    },
}

impl RenderMode {
    /// Whether this is [`RenderMode::Live`].
    pub fn is_live(self) -> bool {
        matches!(self, Self::Live)
    }
}

/// A point in media time, as a count of some time base's units, compared
/// exactly with any other whatever their time bases.
///
/// Two inputs rarely share a time base — 1/90000 from one container, 1/1000
/// or 1/30000 from another — and the output's is its own frame interval.
/// Comparing them through floating-point seconds rounds at exactly the
/// boundaries that decide which frame is shown, and drifts over an hour; so
/// the comparison cross-multiplies instead.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MediaTime {
    value: i64,
    base: ffmpeg::Rational,
}

impl MediaTime {
    /// `value` units of `base`, or `None` for a time base that is not
    /// positive, which no comparison could be made in.
    pub(crate) fn new(value: i64, base: ffmpeg::Rational) -> Option<Self> {
        (base.numerator() > 0 && base.denominator() > 0).then_some(Self { value, base })
    }

    /// `duration`, in nanoseconds.
    pub(crate) fn from_duration(duration: Duration) -> Self {
        Self {
            value: i64::try_from(duration.as_nanos()).unwrap_or(i64::MAX),
            base: ffmpeg::Rational::new(1, 1_000_000_000),
        }
    }

    /// This time and `duration` units of the same base later.
    pub(crate) fn plus(self, duration: i64) -> Self {
        Self {
            value: self.value.saturating_add(duration),
            base: self.base,
        }
    }

    /// The numerator of this time over a common denominator with `other`.
    fn scaled_against(self, other: Self) -> i128 {
        i128::from(self.value)
            * i128::from(self.base.numerator())
            * i128::from(other.base.denominator())
    }
}

impl PartialEq for MediaTime {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for MediaTime {}

impl PartialOrd for MediaTime {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MediaTime {
    fn cmp(&self, other: &Self) -> Ordering {
        // value·num/den against value'·num'/den', both denominators positive:
        // value·num·den' against value'·num'·den. An i64 value times two i32
        // factors fits an i128 with room to spare.
        self.scaled_against(*other)
            .cmp(&other.scaled_against(*self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(value: i64, num: i32, den: i32) -> MediaTime {
        MediaTime::new(value, ffmpeg::Rational::new(num, den)).unwrap()
    }

    /// The boundary cases floating point gets wrong: a 30000/1001 rate's
    /// frames against a millisecond clock and a 90 kHz one.
    #[test]
    fn times_in_different_bases_compare_exactly() {
        // Frame 3 at 30000/1001 fps is 3003/30000 s: exactly 0.1001 s, and
        // exactly 9009 ticks of a 90 kHz clock.
        let frame = at(3, 1001, 30000);
        assert_eq!(frame, at(100_100, 1, 1_000_000));
        assert_eq!(frame, at(9009, 1, 90_000));
        assert!(frame < at(9010, 1, 90_000));
        assert!(frame > at(9008, 1, 90_000));
        assert_eq!(
            MediaTime::from_duration(Duration::from_millis(1500)),
            at(45, 1, 30)
        );
        assert_eq!(at(10, 1, 30).plus(5), at(15, 1, 30));
        assert!(MediaTime::new(1, ffmpeg::Rational::new(0, 1)).is_none());
    }
}
