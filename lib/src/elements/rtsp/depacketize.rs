//! A publisher's RTP taken apart into the packets it was cut from.

use ffmpeg_next as ffmpeg;

use super::sdp::Payload;

/// One RTP packet's header, read.
struct Header {
    marker: bool,
    sequence: u16,
    timestamp: u32,
}

/// `packet`'s header and payload — `None` for what is not RTP version 2.
fn split(packet: &[u8]) -> Option<(Header, &[u8])> {
    if packet.len() < 12 || packet[0] >> 6 != 2 {
        return None;
    }
    let padding = packet[0] & 0x20 != 0;
    let extension = packet[0] & 0x10 != 0;
    let csrcs = usize::from(packet[0] & 0x0f);
    let mut start = 12 + 4 * csrcs;
    if extension {
        let words = usize::from(u16::from_be_bytes([
            *packet.get(start + 2)?,
            *packet.get(start + 3)?,
        ]));
        start += 4 + 4 * words;
    }
    let mut end = packet.len();
    if padding {
        end = end.checked_sub(usize::from(*packet.last()?))?;
    }
    if start > end {
        return None;
    }
    Some((
        Header {
            marker: packet[1] & 0x80 != 0,
            sequence: u16::from_be_bytes([packet[2], packet[3]]),
            timestamp: u32::from_be_bytes([packet[4], packet[5], packet[6], packet[7]]),
        },
        &packet[start..end],
    ))
}

/// Takes one track's RTP packets apart, in the order they arrive.
pub(crate) struct Depacketizer {
    payload: Payload,
    /// The last timestamp seen, and where it is unwrapped to.
    last: Option<(u32, i64)>,
    /// The next sequence number expected.
    expected: Option<u16>,
    /// The access unit being gathered, in Annex B, and its timestamp.
    unit: Vec<u8>,
    unit_at: i64,
    /// A NAL unit or AAC frame arriving in fragments: what has come of it.
    fragment: Option<Vec<u8>>,
    /// An AAC frame in fragments: how long it is.
    fragment_size: usize,
}

impl Depacketizer {
    pub(crate) fn new(payload: Payload) -> Self {
        Self {
            payload,
            last: None,
            expected: None,
            unit: Vec::new(),
            unit_at: 0,
            fragment: None,
            fragment_size: 0,
        }
    }

    /// What `packet` completes: whole packets, stamped in the track's RTP
    /// clock from zero.
    pub(crate) fn push(&mut self, packet: &[u8]) -> Vec<ffmpeg::Packet> {
        let Some((header, payload)) = split(packet) else {
            return Vec::new();
        };
        if let Some(expected) = self.expected
            && expected != header.sequence
        {
            // Something was lost: a fragment in hand can no longer be whole.
            self.fragment = None;
        }
        self.expected = Some(header.sequence.wrapping_add(1));
        let at = self.unwrap(header.timestamp);
        let mut done = Vec::new();
        match self.payload {
            Payload::H264 | Payload::H265 => {
                if !self.unit.is_empty() && at != self.unit_at {
                    done.extend(self.finish_unit());
                }
                self.unit_at = at;
                self.nal_units(payload);
                if header.marker {
                    done.extend(self.finish_unit());
                }
            }
            Payload::Aac {
                size_length,
                index_length,
            } => done.extend(self.aac(payload, at, header.marker, size_length, index_length)),
            Payload::Whole => {
                if !payload.is_empty() {
                    done.push(stamped(payload, at, true));
                }
            }
        }
        done
    }

    fn unwrap(&mut self, timestamp: u32) -> i64 {
        let at = match self.last {
            None => 0,
            Some((last, at)) => at + i64::from(timestamp.wrapping_sub(last) as i32),
        };
        self.last = Some((timestamp, at));
        at
    }

    fn finish_unit(&mut self) -> Option<ffmpeg::Packet> {
        if self.unit.is_empty() {
            return None;
        }
        let key = key_unit(&self.unit, self.payload == Payload::H265);
        let packet = stamped(&self.unit, self.unit_at, key);
        self.unit.clear();
        Some(packet)
    }

    fn push_nal(&mut self, nal: &[u8]) {
        if !nal.is_empty() {
            self.unit.extend_from_slice(&[0, 0, 0, 1]);
            self.unit.extend_from_slice(nal);
        }
    }

    /// The NAL units of one H.264 or H.265 payload into the unit gathered.
    fn nal_units(&mut self, payload: &[u8]) {
        let h265 = self.payload == Payload::H265;
        let header_len = if h265 { 2 } else { 1 };
        if payload.len() < header_len {
            return;
        }
        let kind = if h265 {
            (payload[0] >> 1) & 0x3f
        } else {
            payload[0] & 0x1f
        };
        match (h265, kind) {
            // Aggregation packets: STAP-A, AP.
            (false, 24) | (true, 48) => {
                let mut rest = &payload[header_len..];
                while rest.len() >= 2 {
                    let size = usize::from(u16::from_be_bytes([rest[0], rest[1]]));
                    if rest.len() < 2 + size {
                        break;
                    }
                    self.push_nal(&rest[2..2 + size]);
                    rest = &rest[2 + size..];
                }
            }
            // Fragmentation units: FU-A, FU.
            (false, 28) | (true, 49) => {
                let Some(&fu) = payload.get(header_len) else {
                    return;
                };
                let (start, end) = (fu & 0x80 != 0, fu & 0x40 != 0);
                let data = &payload[header_len + 1..];
                if start {
                    let mut nal = Vec::with_capacity(data.len() * 4);
                    if h265 {
                        nal.push((payload[0] & 0x81) | ((fu & 0x3f) << 1));
                        nal.push(payload[1]);
                    } else {
                        nal.push((payload[0] & 0xe0) | (fu & 0x1f));
                    }
                    self.fragment = Some(nal);
                }
                if let Some(nal) = self.fragment.as_mut() {
                    nal.extend_from_slice(data);
                }
                if end && let Some(nal) = self.fragment.take() {
                    self.push_nal(&nal);
                }
            }
            _ => self.push_nal(payload),
        }
    }

    /// The AAC frames of one `mpeg4-generic` payload: each its own packet,
    /// 1024 samples after the one before.
    fn aac(
        &mut self,
        payload: &[u8],
        at: i64,
        marker: bool,
        size_length: u8,
        index_length: u8,
    ) -> Vec<ffmpeg::Packet> {
        let mut done = Vec::new();
        if payload.len() < 2 {
            return done;
        }
        let header_bits = usize::from(u16::from_be_bytes([payload[0], payload[1]]));
        let header_bytes = header_bits.div_ceil(8);
        let Some(headers) = payload.get(2..2 + header_bytes) else {
            return done;
        };
        let mut data = &payload[2 + header_bytes..];
        let each = usize::from(size_length) + usize::from(index_length);
        if each == 0 {
            return done;
        }
        let sizes: Vec<usize> = (0..header_bits / each)
            .map(|n| read_bits(headers, n * each, usize::from(size_length)))
            .collect();
        // One frame larger than a packet comes in fragments, every one with
        // the whole frame's size, the last with the marker.
        if sizes.len() == 1 && sizes[0] > data.len() {
            let fragment = self.fragment.get_or_insert_with(Vec::new);
            if fragment.is_empty() {
                self.fragment_size = sizes[0];
            }
            fragment.extend_from_slice(data);
            if (marker || fragment.len() >= self.fragment_size)
                && let Some(frame) = self.fragment.take()
                && frame.len() == self.fragment_size
            {
                done.push(stamped(&frame, at, true));
            }
            return done;
        }
        self.fragment = None;
        for (n, size) in sizes.into_iter().enumerate() {
            if data.len() < size {
                break;
            }
            done.push(stamped(&data[..size], at + 1024 * n as i64, true));
            data = &data[size..];
        }
        done
    }
}

/// `count` bits of `bytes` from bit `from`, most significant first.
fn read_bits(bytes: &[u8], from: usize, count: usize) -> usize {
    (from..from + count).fold(0, |value, bit| {
        let set = bytes
            .get(bit / 8)
            .is_some_and(|byte| byte & (0x80 >> (bit % 8)) != 0);
        (value << 1) | usize::from(set)
    })
}

fn stamped(data: &[u8], at: i64, key: bool) -> ffmpeg::Packet {
    let mut packet = ffmpeg::Packet::copy(data);
    packet.set_pts(Some(at));
    if key {
        packet.set_flags(ffmpeg::packet::Flags::KEY);
    }
    packet
}

/// Whether an Annex B access unit holds a picture a decoder can start at.
pub(crate) fn key_unit(unit: &[u8], h265: bool) -> bool {
    nal_units(unit).any(|nal| match (h265, nal.first()) {
        (false, Some(&byte)) => byte & 0x1f == 5,
        (true, Some(&byte)) => (16..=21).contains(&((byte >> 1) & 0x3f)),
        _ => false,
    })
}

/// The NAL units of an Annex B stream, start codes left out.
pub(crate) fn nal_units(stream: &[u8]) -> impl Iterator<Item = &[u8]> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 <= stream.len() {
        if stream[i] == 0 && stream[i + 1] == 0 && stream[i + 2] == 1 {
            starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    let ends: Vec<usize> = starts
        .iter()
        .skip(1)
        .map(|&next| {
            let mut end = next - 3;
            while end > 0 && stream[end - 1] == 0 {
                end -= 1;
            }
            end
        })
        .chain(std::iter::once(stream.len()))
        .collect();
    starts
        .into_iter()
        .zip(ends)
        .filter(|(start, end)| start < end)
        .map(move |(start, end)| &stream[start..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rtp(sequence: u16, timestamp: u32, marker: bool, payload: &[u8]) -> Vec<u8> {
        let mut packet = vec![0x80, if marker { 0xe0 } else { 0x60 }];
        packet.extend_from_slice(&sequence.to_be_bytes());
        packet.extend_from_slice(&timestamp.to_be_bytes());
        packet.extend_from_slice(&[1, 2, 3, 4]);
        packet.extend_from_slice(payload);
        packet
    }

    #[test]
    fn h264_fragments_and_aggregates_come_back_whole() {
        let mut d = Depacketizer::new(Payload::H264);
        // STAP-A with an SPS and a PPS, then an IDR slice in three FU-As.
        let stap = [24, 0, 2, 0x67, 0xAA, 0, 2, 0x68, 0xBB];
        assert!(d.push(&rtp(1, 1000, false, &stap)).is_empty());
        assert!(d.push(&rtp(2, 1000, false, &[0x7c, 0x85, 1, 2])).is_empty());
        assert!(d.push(&rtp(3, 1000, false, &[0x7c, 0x05, 3])).is_empty());
        let done = d.push(&rtp(4, 1000, true, &[0x7c, 0x45, 4, 5]));
        assert_eq!(done.len(), 1);
        let unit = done[0].data().unwrap();
        assert_eq!(
            unit,
            &[
                0, 0, 0, 1, 0x67, 0xAA, 0, 0, 0, 1, 0x68, 0xBB, 0, 0, 0, 1, 0x65, 1, 2, 3, 4, 5
            ]
        );
        assert!(done[0].is_key());
        assert_eq!(done[0].pts(), Some(0));

        // A picture that is not a keyframe, 3000 ticks on.
        let done = d.push(&rtp(5, 4000, true, &[0x41, 9]));
        assert_eq!(done[0].pts(), Some(3000));
        assert!(!done[0].is_key());
    }

    #[test]
    fn a_lost_fragment_drops_its_nal_unit_alone() {
        let mut d = Depacketizer::new(Payload::H264);
        assert!(d.push(&rtp(1, 0, false, &[0x7c, 0x85, 1])).is_empty());
        // Sequence 2 is lost.
        let done = d.push(&rtp(3, 0, true, &[0x7c, 0x45, 3]));
        assert!(done.is_empty(), "nothing whole");
        let done = d.push(&rtp(4, 3000, true, &[0x41, 9]));
        assert_eq!(done.len(), 1);
    }

    #[test]
    fn timestamps_unwrap() {
        let mut d = Depacketizer::new(Payload::Whole);
        assert_eq!(
            d.push(&rtp(1, u32::MAX - 959, true, &[1]))[0].pts(),
            Some(0)
        );
        assert_eq!(d.push(&rtp(2, 0, true, &[1]))[0].pts(), Some(960));
        assert_eq!(d.push(&rtp(3, 960, true, &[1]))[0].pts(), Some(1920));
    }

    #[test]
    fn aac_frames_are_cut_by_their_headers() {
        let mut d = Depacketizer::new(Payload::Aac {
            size_length: 13,
            index_length: 3,
        });
        // Two AU headers (32 bits): sizes 3 and 2.
        let mut payload = vec![0, 32, (3 >> 5) as u8, (3 << 3) as u8, 0, 2 << 3];
        payload.extend_from_slice(&[1, 2, 3, 4, 5]);
        let done = d.push(&rtp(1, 0, true, &payload));
        assert_eq!(done.len(), 2);
        assert_eq!(done[0].data().unwrap(), &[1, 2, 3]);
        assert_eq!(done[1].data().unwrap(), &[4, 5]);
        assert_eq!(done[1].pts(), Some(1024));
    }

    #[test]
    fn h265_fragments_come_back_whole() {
        let mut d = Depacketizer::new(Payload::H265);
        // An IDR_W_RADL (19) in two FUs: payload header type 49.
        let header = [49 << 1, 1];
        let mut first = header.to_vec();
        first.extend_from_slice(&[0x80 | 19, 1, 2]);
        let mut last = header.to_vec();
        last.extend_from_slice(&[0x40 | 19, 3]);
        assert!(d.push(&rtp(1, 0, false, &first)).is_empty());
        let done = d.push(&rtp(2, 0, true, &last));
        assert_eq!(done[0].data().unwrap(), &[0, 0, 0, 1, 19 << 1, 1, 1, 2, 3]);
        assert!(done[0].is_key());
    }
}
