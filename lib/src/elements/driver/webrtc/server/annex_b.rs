//! H.264 and H.265 as WebRTC's packetizers take them: Annex B, with the
//! parameter sets ahead of every keyframe.
//!
//! What a broadcast carries is what its input made: Annex B from an RTSP
//! camera or an encoder, but length-prefixed NAL units from an MP4 — the
//! parameter sets then kept apart, in the track's `avcC` or `hvcC`. A
//! player that joins at a keyframe needs those sets in band, ahead of it.

/// How one track's packets are turned into Annex B.
pub(crate) enum AnnexB {
    /// Already Annex B: as they are.
    Already,
    /// Length-prefixed, `length` bytes a prefix, with `sets` — the
    /// parameter sets, in Annex B — put ahead of each keyframe.
    Prefixed { length: usize, sets: Vec<u8> },
}

impl AnnexB {
    /// How packets of a track with `extradata` are turned into Annex B:
    /// length-prefixed where it is an `avcC` (`hevc` false) or `hvcC`
    /// record, as they are where not.
    pub(crate) fn for_track(extradata: &[u8], hevc: bool) -> Self {
        let parsed = if hevc {
            hvcc(extradata)
        } else {
            avcc(extradata)
        };
        match parsed {
            Some((length, sets)) => Self::Prefixed { length, sets },
            None => Self::Already,
        }
    }

    /// `data` as Annex B — with the parameter sets ahead where it is a
    /// keyframe — or `None` where it is not what its track said.
    pub(crate) fn convert(&self, data: &[u8], key: bool) -> Option<Vec<u8>> {
        let Self::Prefixed { length, sets } = self else {
            return Some(data.to_vec());
        };
        let mut out = Vec::with_capacity(data.len() + if key { sets.len() } else { 0 } + 16);
        if key {
            out.extend_from_slice(sets);
        }
        let mut rest = data;
        while !rest.is_empty() {
            if rest.len() < *length {
                return None;
            }
            let size = rest[..*length]
                .iter()
                .fold(0usize, |size, byte| (size << 8) | usize::from(*byte));
            rest = &rest[*length..];
            if rest.len() < size {
                return None;
            }
            out.extend_from_slice(&[0, 0, 0, 1]);
            out.extend_from_slice(&rest[..size]);
            rest = &rest[size..];
        }
        Some(out)
    }
}

/// An `avcC` record's NAL length size and its parameter sets in Annex B.
fn avcc(record: &[u8]) -> Option<(usize, Vec<u8>)> {
    if record.len() < 7 || record[0] != 1 {
        return None;
    }
    let length = usize::from(record[4] & 0x03) + 1;
    let mut sets = Vec::new();
    let mut at = 5;
    for mask in [0x1f, 0xff] {
        // The SPS count's high bits are reserved; the PPS count's are not.
        let count = record.get(at)? & mask;
        at += 1;
        for _ in 0..count {
            let size = usize::from(u16::from_be_bytes([*record.get(at)?, *record.get(at + 1)?]));
            at += 2;
            sets.extend_from_slice(&[0, 0, 0, 1]);
            sets.extend_from_slice(record.get(at..at + size)?);
            at += size;
        }
    }
    Some((length, sets))
}

/// An `hvcC` record's NAL length size and its parameter sets in Annex B.
fn hvcc(record: &[u8]) -> Option<(usize, Vec<u8>)> {
    if record.len() < 23 || record[0] != 1 {
        return None;
    }
    let length = usize::from(record[21] & 0x03) + 1;
    let arrays = record[22];
    let mut sets = Vec::new();
    let mut at = 23;
    for _ in 0..arrays {
        at += 1; // array_completeness, reserved, NAL unit type
        let count = u16::from_be_bytes([*record.get(at)?, *record.get(at + 1)?]);
        at += 2;
        for _ in 0..count {
            let size = usize::from(u16::from_be_bytes([*record.get(at)?, *record.get(at + 1)?]));
            at += 2;
            sets.extend_from_slice(&[0, 0, 0, 1]);
            sets.extend_from_slice(record.get(at..at + size)?);
            at += size;
        }
    }
    Some((length, sets))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An `avcC` with one SPS and one PPS, four-byte lengths.
    const AVCC: &[u8] = &[
        1, 0x42, 0xc0, 0x1e, 0xff, 0xe1, 0, 3, 0x67, 1, 2, 1, 0, 2, 0x68, 3,
    ];

    #[test]
    fn length_prefixed_units_become_annex_b_with_the_sets_ahead_of_a_keyframe() {
        let annex_b = AnnexB::for_track(AVCC, false);
        let packet = [0, 0, 0, 2, 0x65, 9, 0, 0, 0, 1, 0x06];
        let key = annex_b.convert(&packet, true).unwrap();
        assert_eq!(
            key,
            [
                0, 0, 0, 1, 0x67, 1, 2, 0, 0, 0, 1, 0x68, 3, 0, 0, 0, 1, 0x65, 9, 0, 0, 0, 1, 0x06
            ]
        );
        let other = annex_b.convert(&[0, 0, 0, 1, 0x41], false).unwrap();
        assert_eq!(other, [0, 0, 0, 1, 0x41]);
        assert!(
            annex_b.convert(&[0, 0, 0, 9, 0x41], false).is_none(),
            "a length past the end"
        );
    }

    #[test]
    fn annex_b_extradata_leaves_packets_as_they_are() {
        let annex_b = AnnexB::for_track(&[0, 0, 0, 1, 0x67, 1], false);
        assert!(matches!(annex_b, AnnexB::Already));
        assert_eq!(
            annex_b.convert(&[0, 0, 1, 0x65], true).unwrap(),
            [0, 0, 1, 0x65]
        );
    }

    #[test]
    fn an_hvcc_gives_its_sets() {
        let mut record = vec![1u8; 22];
        record[21] = 0x03;
        record.push(1); // one array
        record.extend_from_slice(&[0x20, 0, 1, 0, 2, 0x40, 1]); // a VPS
        let AnnexB::Prefixed { length, sets } = AnnexB::for_track(&record, true) else {
            panic!("an hvcC");
        };
        assert_eq!(length, 4);
        assert_eq!(sets, [0, 0, 0, 1, 0x40, 1]);
    }
}
