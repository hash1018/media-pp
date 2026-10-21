//! One reader's own timeline across a broadcast's runs: [`Timeline`].

use std::time::Instant;

use ffmpeg_next::{self as ffmpeg, Rational, Rescale};

use crate::elements::StreamInfo;

/// The unit a [`Timeline`] keeps where runs begin and end in, between
/// tracks.
const MICROSECONDS: Rational = Rational(1, 1_000_000);

/// Puts what a reader reads of a broadcast on a timeline of its own,
/// beginning at zero: each run after the last, as far after it as the time
/// between them, so a recording or a viewer of an input that dropped and came
/// back sees a gap rather than timestamps starting over.
///
/// A run's origin is the first packet put on it — the keyframe a reader
/// starts at — and every track is moved by the same amount, so the tracks
/// stay together. What would come before zero is passed over.
#[derive(Default)]
pub(crate) struct Timeline {
    /// Where the run being read began on the broadcast's timeline: the
    /// first packet put on it, and that packet's unit.
    origin: Option<(i64, Rational)>,
    /// Where the run being read begins on this timeline, in
    /// [`MICROSECONDS`].
    offset: i64,
    /// Where what has been put on it so far ends, in [`MICROSECONDS`].
    end: i64,
    /// When the last packet put on it had arrived, for how far after it the
    /// next run is put.
    last_arrival: Option<Instant>,
}

impl Timeline {
    /// Another run begins: its first packet goes after what has been.
    pub(crate) fn next_run(&mut self) {
        self.origin = None;
    }

    /// Where the run being read begins on this timeline, in seconds — once
    /// its first packet has been put on it.
    pub(crate) fn run_begins_at(&self) -> Option<f64> {
        self.origin.map(|_| self.offset as f64 / 1e6)
    }

    /// A copy of `packet`, whose timestamps count in `unit`, stamped on this
    /// timeline: `None` where it would come before zero, or carries no
    /// timestamp at all.
    pub(crate) fn restamp(
        &mut self,
        packet: &ffmpeg::Packet,
        unit: Rational,
        arrived: Instant,
    ) -> Option<ffmpeg::Packet> {
        if unit.numerator() <= 0 || unit.denominator() <= 0 {
            return None;
        }
        let at = packet.dts().or(packet.pts())?;
        let (origin, origin_unit) = match self.origin {
            Some(origin) => origin,
            None => {
                // The first packet of a run: it goes as far after the last
                // run's last packet as it arrived after it.
                if let Some(last) = self.last_arrival {
                    let gap = arrived.saturating_duration_since(last).as_micros();
                    self.offset = self
                        .end
                        .saturating_add(i64::try_from(gap).unwrap_or(i64::MAX));
                }
                *self.origin.insert((at, unit))
            }
        };
        let shift = self.offset.rescale(MICROSECONDS, unit) - origin.rescale(origin_unit, unit);
        let pts = packet.pts().map(|pts| pts + shift);
        let dts = packet.dts().map(|dts| dts + shift);
        if dts.or(pts).is_some_and(|ts| ts < 0) {
            return None;
        }
        let mut stamped = packet.clone();
        stamped.set_pts(pts);
        stamped.set_dts(dts);
        if let Some(last) = pts.or(dts) {
            let ends = (last + stamped.duration().max(0)).rescale(unit, MICROSECONDS);
            self.end = self.end.max(ends);
        }
        self.last_arrival = Some(arrived);
        Some(stamped)
    }
}

/// Whether a run of `new` tracks can go on where one of `old` was being
/// read: as many, each of the same medium and codec.
pub(crate) fn same_shape(old: &[StreamInfo], new: &[StreamInfo]) -> bool {
    old.len() == new.len()
        && old
            .iter()
            .zip(new)
            .all(|(old, new)| old.kind == new.kind && old.parameters.id() == new.parameters.id())
}
