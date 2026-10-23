//! Decoding timestamps for pictures RTP gives only the presentation time
//! of.
//!
//! RTP stamps each picture with when it is shown and says nothing of when
//! it is decoded, which for a stream that reorders its pictures — B-frames
//! — is another time. A decoder needs no decoding time, but a muxer does,
//! and refuses times that go back. [`Dts`] works it out the way a decoder
//! puts pictures out: its sequence parameter set says how many pictures
//! may wait to be shown (`N`); the pictures arrive in decoding order; and
//! once more than `N` wait, the earliest is the one shown next. Each
//! picture is decoded at the time of the one that leaves the wait as it
//! arrives:
//!
//! ```text
//! arrives  I0  P3  B1  B2  P6  B4  B5        N = 1
//! waiting  I0  I0  P3  P3  P3  P6  P6
//!              P3  B1  B2  P6  B4  B5
//! leaves   —   I0  B1  B2  P3  B4  B5
//! dts     -1   0   1   2   3   4   5
//! ```
//!
//! — times that never go back, and none after the picture's own `pts`,
//! read from the times themselves: nothing is estimated once the wait is
//! full, nothing drifts, and the pictures go on at once — only their times
//! wait. Until it is full, the first pictures are decoded as far ahead of
//! the earliest as there are places left, a picture's duration apart.
//!
//! A stream that does not reorder — most cameras' — is decoded when it is
//! shown, and nothing is waited for.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use ffmpeg_next as ffmpeg;

use crate::elements::parameter_sets::{annex_b, h264_sequence, hevc_sequence};

/// The decoding times of one picture track.
pub(crate) struct Dts {
    hevc: bool,
    /// How many pictures may wait to be shown: the parameter set's reorder
    /// count — and for H.265, its whole picture buffer, so that a
    /// keyframe's leading pictures, shown before it, find room.
    wait: usize,
    reorder: u32,
    /// The `pts` of the pictures waiting.
    waiting: BinaryHeap<Reverse<i64>>,
    /// The RTP clock's ticks a picture lasts where nothing better is known:
    /// the VUI's rate, or a 25th of a second.
    duration: i64,
    last: Option<i64>,
}

impl Dts {
    /// The decoding times of an H.264 or H.265 track whose configuration —
    /// its parameter sets, Annex B — is `extradata`, timed on a clock of
    /// `clock_rate`; `None` for any other codec, which is decoded when it
    /// is shown.
    pub(crate) fn new(codec: ffmpeg::codec::Id, extradata: &[u8], clock_rate: u32) -> Option<Self> {
        let hevc = match codec {
            ffmpeg::codec::Id::H264 => false,
            ffmpeg::codec::Id::HEVC => true,
            _ => return None,
        };
        let mut dts = Self {
            hevc,
            wait: 0,
            reorder: 0,
            waiting: BinaryHeap::new(),
            duration: i64::from(clock_rate.max(1)) / 25,
            last: None,
        };
        for nal in annex_b(extradata) {
            dts.parameter_set(nal, clock_rate);
        }
        Some(dts)
    }

    /// The RTP clock's ticks a picture lasts as its parameter sets say: the
    /// VUI's rate, or a 25th of a second where they say none.
    pub(crate) fn picture_duration(&self) -> i64 {
        self.duration
    }

    #[cfg(test)]
    /// How many pictures may be decoded ahead of one shown before them —
    /// FFmpeg's `video_delay` — as the sequence parameter set says.
    pub(crate) fn reorder(&self) -> u32 {
        self.reorder
    }

    /// The decoding time of the picture `unit` — one access unit, Annex B —
    /// shown at `pts`, in the order pictures are decoded. A keyframe's
    /// parameter sets, where it carries them, are read for what they say
    /// of reordering; another picture is not looked into.
    pub(crate) fn stamp(&mut self, unit: &[u8], key: bool, pts: i64, clock_rate: u32) -> i64 {
        if key {
            for nal in annex_b(unit) {
                self.parameter_set(nal, clock_rate);
            }
        }
        if self.wait == 0 {
            self.waiting.clear();
            self.last = Some(pts);
            return pts;
        }
        self.waiting.push(Reverse(pts));
        let dts = if self.waiting.len() > self.wait {
            let Reverse(leaves) = self.waiting.pop().unwrap_or(Reverse(pts));
            leaves
        } else {
            // Not full yet: as far ahead of the earliest as places are left.
            let Reverse(earliest) = *self.waiting.peek().unwrap_or(&Reverse(pts));
            let left = (self.wait + 1 - self.waiting.len()) as i64;
            earliest - left * self.duration
        };
        let dts = match self.last {
            Some(last) if dts <= last => last + 1,
            _ => dts,
        };
        self.last = Some(dts);
        dts.min(pts)
    }

    /// Takes in what `nal` says where it is a sequence parameter set.
    fn parameter_set(&mut self, nal: &[u8], clock_rate: u32) {
        let (reorder, wait) = if self.hevc {
            if nal.first().map(|byte| (byte >> 1) & 0x3f) != Some(33) {
                return;
            }
            let Some(sequence) = hevc_sequence(nal) else {
                return;
            };
            let buffer = sequence.buffer.saturating_sub(1);
            (
                sequence.reorder,
                if sequence.reorder == 0 {
                    0
                } else {
                    sequence.reorder.max(buffer)
                },
            )
        } else {
            if nal.first().map(|byte| byte & 0x1f) != Some(7) {
                return;
            }
            let Some(sequence) = h264_sequence(nal) else {
                return;
            };
            if let Some((units, scale)) = sequence.timing {
                // A frame is two ticks of a field.
                let ticks = i64::from(clock_rate) * 2 * i64::from(units) / i64::from(scale);
                if ticks > 0 {
                    self.duration = ticks;
                }
            }
            (sequence.reorder, sequence.reorder)
        };
        self.reorder = reorder;
        self.wait = wait as usize;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bit writer for parameter sets made by hand.
    struct Bits {
        bytes: Vec<u8>,
        bit: usize,
    }

    impl Bits {
        fn new() -> Self {
            Self {
                bytes: Vec::new(),
                bit: 0,
            }
        }

        fn put(&mut self, value: u32, count: usize) {
            for index in (0..count).rev() {
                if self.bit.is_multiple_of(8) {
                    self.bytes.push(0);
                }
                if value >> index & 1 == 1 {
                    *self.bytes.last_mut().unwrap() |= 0x80 >> (self.bit % 8);
                }
                self.bit += 1;
            }
        }

        fn ue(&mut self, value: u32) {
            let coded = value + 1;
            let length = 32 - coded.leading_zeros() as usize;
            self.put(0, length - 1);
            self.put(coded, length);
        }

        /// Finished with a stop bit, and no emulation to prevent: the
        /// values here make no run of zeros that needs it.
        fn done(mut self) -> Vec<u8> {
            self.put(1, 1);
            self.bytes
        }
    }

    /// An H.264 SPS, Main profile, 320x240, and a VUI saying 25 frames a
    /// second and `reorder`.
    fn h264_sps(reorder: u32) -> Vec<u8> {
        let mut b = Bits::new();
        b.put(77, 8);
        b.put(0, 8);
        b.put(30, 8);
        b.ue(0); // sps id
        b.ue(0); // log2_max_frame_num - 4
        b.ue(0); // poc type
        b.ue(4); // log2_max_poc_lsb - 4
        b.ue(4); // max_num_ref_frames
        b.put(0, 1);
        b.ue(19); // 20 macroblocks wide
        b.ue(14); // 15 high
        b.put(1, 1); // frame_mbs_only
        b.put(1, 1); // direct_8x8
        b.put(0, 1); // no cropping
        b.put(1, 1); // VUI
        b.put(0, 1); // aspect
        b.put(0, 1); // overscan
        b.put(0, 1); // video signal
        b.put(0, 1); // chroma location
        b.put(1, 1); // timing
        b.put(1, 32);
        b.put(50, 32); // 25 frames a second
        b.put(1, 1);
        b.put(0, 1); // nal hrd
        b.put(0, 1); // vcl hrd
        b.put(0, 1); // pic struct
        b.put(1, 1); // bitstream restriction
        b.put(1, 1);
        b.ue(0);
        b.ue(0);
        b.ue(16);
        b.ue(16);
        b.ue(reorder);
        b.ue(4);
        let mut nal = vec![0x67];
        nal.extend(b.done());
        nal
    }

    fn annex(nals: &[Vec<u8>]) -> Vec<u8> {
        nals.iter()
            .flat_map(|nal| {
                let mut unit = vec![0, 0, 0, 1];
                unit.extend_from_slice(nal);
                unit
            })
            .collect()
    }

    /// A picture's access unit: a slice of no consequence.
    fn picture() -> Vec<u8> {
        annex(&[vec![0x41, 0x9a]])
    }

    #[test]
    fn the_sequence_says_how_far_pictures_are_reordered() {
        let sequence = h264_sequence(&h264_sps(2)).unwrap();
        assert_eq!((sequence.shape.width, sequence.shape.height), (320, 240));
        assert_eq!(sequence.reorder, 2);
        assert_eq!(sequence.timing, Some((1, 50)));
    }

    /// The decoding times of pictures shown at `shown` — in frames, in the
    /// order they arrive — with `reorder`.
    fn decoded(reorder: u32, shown: &[i64]) -> Vec<i64> {
        let frame = 3600;
        let mut dts = Dts::new(
            ffmpeg::codec::Id::H264,
            &annex(&[h264_sps(reorder)]),
            90_000,
        )
        .unwrap();
        assert_eq!(dts.reorder(), reorder);
        shown
            .iter()
            .map(|&shown| {
                let pts = 1000 + shown * frame;
                let got = dts.stamp(&picture(), false, pts, 90_000);
                assert!(got <= pts, "decoded at {got}, shown at {pts}");
                got
            })
            .collect()
    }

    /// I P B B P B B, as x264 sends them without a pyramid: evenly spaced,
    /// one picture early.
    #[test]
    fn reordered_pictures_are_decoded_in_time_and_in_order() {
        assert_eq!(
            decoded(1, &[0, 3, 1, 2, 6, 4, 5]),
            (0..7).map(|n| 1000 + (n - 1) * 3600).collect::<Vec<_>>()
        );
    }

    /// x264's pyramid: I P B b b, two pictures early.
    #[test]
    fn a_pyramid_is_decoded_two_early() {
        assert_eq!(
            decoded(2, &[0, 4, 2, 1, 3, 8, 6, 5, 7]),
            (0..9).map(|n| 1000 + (n - 2) * 3600).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_stream_that_does_not_reorder_is_decoded_when_shown() {
        assert_eq!(decoded(0, &[0, 1, 2]), [1000, 4600, 8200]);
        assert!(Dts::new(ffmpeg::codec::Id::AAC, &[], 48_000).is_none());
    }

    /// A picture lost leaves a gap in what was shown, not a time that goes
    /// back.
    #[test]
    fn a_picture_lost_breaks_nothing() {
        let times = decoded(2, &[0, 4, 2, 3, 8, 6, 5, 7, 12, 10, 9]);
        assert!(times.windows(2).all(|w| w[1] > w[0]), "{times:?}");
    }
}
