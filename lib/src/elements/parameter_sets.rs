//! What an H.264 or H.265 stream's sequence parameter sets say its pictures
//! are — size, profile, chroma, bit depth — and a watch over a stream's
//! packets for the moment that changes partway ([`ShapeWatch`]).
//!
//! A [`StreamInfo`](crate::elements::StreamInfo) is the stream as it
//! began. A camera reconfigured, or a WebRTC sender that lowers its picture
//! size for its connection, goes on with new parameter sets in band: a
//! decoder reading them follows, but a source has described the stream as
//! it was, and an MP4 keeps the first sets in its header for the whole
//! file. Only what the picture is counts as a change here: a sender that
//! writes its sets again with another level or other timing information has
//! not changed what it sends.

use ffmpeg_next as ffmpeg;

use crate::{buffer::MediaBuffer, elements::StreamInfo};

/// What a sequence parameter set says the pictures are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Shape {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) profile: u8,
    pub(crate) chroma_format: u32,
    pub(crate) bit_depth: u32,
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}x{}, profile {}, chroma format {}, {} bits",
            self.width, self.height, self.profile, self.chroma_format, self.bit_depth
        )
    }
}

/// The two codecs whose parameter sets this reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Codec {
    H264,
    Hevc,
}

impl Codec {
    fn of(id: ffmpeg::codec::Id) -> Option<Self> {
        match id {
            ffmpeg::codec::Id::H264 => Some(Self::H264),
            ffmpeg::codec::Id::HEVC => Some(Self::Hevc),
            _ => None,
        }
    }

    fn is_sps(self, nal: &[u8]) -> bool {
        match self {
            Self::H264 => nal.first().is_some_and(|byte| byte & 0x1f == 7),
            Self::Hevc => nal.first().is_some_and(|byte| (byte >> 1) & 0x3f == 33),
        }
    }

    fn shape(self, sps: &[u8]) -> Option<Shape> {
        match self {
            Self::H264 => h264_shape(sps),
            Self::Hevc => hevc_shape(sps),
        }
    }
}

/// Watches one stream's keyframes for parameter sets that say its pictures
/// are no longer what they began as.
pub(crate) struct ShapeWatch {
    codec: Codec,
    /// The size of a length-prefixed packet's prefixes, as the stream's
    /// `avcC` or `hvcC` says; four where it says nothing.
    length_size: usize,
    /// What the stream began as: its configuration's, or else the first
    /// set seen in band.
    began: Option<Shape>,
}

impl ShapeWatch {
    /// A watch over a stream of `parameters`, or `None` for a codec whose
    /// sets this does not read.
    pub(crate) fn new(parameters: &ffmpeg::codec::Parameters) -> Option<Self> {
        let codec = Codec::of(parameters.id())?;
        // SAFETY: plain fields of parameters the caller owns; FFmpeg keeps
        // `extradata_size` bytes at `extradata` where it is not null.
        let config = unsafe {
            let raw = parameters.as_ptr();
            let size = usize::try_from((*raw).extradata_size).unwrap_or(0);
            if (*raw).extradata.is_null() || size == 0 {
                &[][..]
            } else {
                std::slice::from_raw_parts((*raw).extradata, size)
            }
        };
        let (sets, length_size) = config_sets(codec, config);
        let began = sets
            .into_iter()
            .find(|nal| codec.is_sps(nal))
            .and_then(|sps| codec.shape(sps));
        Some(Self {
            codec,
            length_size,
            began,
        })
    }

    /// A watch over a stream of codec `id` with no configuration to read:
    /// what it began as is the first set it carries.
    #[cfg(any(feature = "webrtc", test))]
    pub(crate) fn learning(id: ffmpeg::codec::Id) -> Option<Self> {
        Some(Self {
            codec: Codec::of(id)?,
            length_size: 4,
            began: None,
        })
    }

    /// What the stream began as, where that is known yet.
    #[cfg(test)]
    pub(crate) fn began(&self) -> Option<Shape> {
        self.began
    }

    /// [`Self::changed`] for a buffer as a source hands it out: only a
    /// keyframe's packet is read, since that is where a sender's new sets
    /// go.
    pub(crate) fn changed_in(&mut self, buf: &MediaBuffer) -> Option<(Shape, Shape)> {
        let MediaBuffer::Packet(packet) = buf else {
            return None;
        };
        if !packet.is_key() {
            return None;
        }
        self.changed(packet.data()?)
    }

    /// The shape a keyframe's packet says its pictures now are, where its
    /// sets say other than the stream began as; `None` for one that says
    /// the same, says nothing, or cannot be read. The first set a stream
    /// with none in its configuration carries is what it began as.
    pub(crate) fn changed(&mut self, data: &[u8]) -> Option<(Shape, Shape)> {
        let now = nal_units(data, self.length_size)
            .into_iter()
            .find(|nal| self.codec.is_sps(nal))
            .and_then(|sps| self.codec.shape(sps))?;
        match self.began {
            None => {
                self.began = Some(now);
                None
            }
            Some(began) => (began != now).then_some((began, now)),
        }
    }
}

/// A source's watch over each of its outputs, saying of each once: a
/// source's own part of [`crate::bus::BusEvent::FormatChanged`].
pub(crate) struct OutputWatches(Vec<Option<ShapeWatch>>);

impl OutputWatches {
    /// One watch for each of `streams`, in their order — each its output's.
    pub(crate) fn new<'a>(streams: impl IntoIterator<Item = &'a StreamInfo>) -> Self {
        Self(
            streams
                .into_iter()
                .map(|stream| ShapeWatch::new(stream.parameters()))
                .collect(),
        )
    }

    /// What output `output` changed from and to, the first time a keyframe
    /// of it says so; never again for that output after.
    pub(crate) fn changed(&mut self, output: usize, buf: &MediaBuffer) -> Option<(Shape, Shape)> {
        let watch = self.0.get_mut(output)?;
        let change = watch.as_mut()?.changed_in(buf)?;
        *watch = None;
        Some(change)
    }
}

/// Says on `bus` that output `output` of the source `name` changed from
/// `was` to `now` — logged with what changed, where the event says only
/// which output.
pub(crate) fn report_change(
    bus: Option<&crate::bus::Bus>,
    pp_log: &crate::pp_log::PpLog,
    element_type: crate::element::ElementType,
    name: &str,
    output: usize,
    (was, now): (Shape, Shape),
) {
    crate::pp_log::pp_info!(
        pp_log: pp_log,
        "output {output} changed its pictures from {was} to {now}"
    );
    if let Some(bus) = bus {
        bus.post(
            pp_log,
            crate::bus::BusEvent::FormatChanged {
                element_type,
                name: name.into(),
                output,
            },
        );
    }
}

/// The parameter sets a stream's configuration holds — `avcC`, `hvcC`, or
/// Annex-B as an encoder writes it — and the prefix size its packets use.
fn config_sets(codec: Codec, config: &[u8]) -> (Vec<&[u8]>, usize) {
    if starts_annex_b(config) {
        return (annex_b(config), 4);
    }
    match codec {
        Codec::H264 => avcc_sets(config).unwrap_or((Vec::new(), 4)),
        Codec::Hevc => hvcc_sets(config).unwrap_or((Vec::new(), 4)),
    }
}

fn avcc_sets(config: &[u8]) -> Option<(Vec<&[u8]>, usize)> {
    if config.len() < 7 || config[0] != 1 {
        return None;
    }
    let length_size = usize::from(config[4] & 3) + 1;
    let mut sets = Vec::new();
    let mut at = 5;
    for count_mask in [0x1f, 0xff] {
        let count = usize::from(*config.get(at)? & count_mask);
        at += 1;
        for _ in 0..count {
            let length = usize::from(u16::from_be_bytes([*config.get(at)?, *config.get(at + 1)?]));
            sets.push(config.get(at + 2..at + 2 + length)?);
            at += 2 + length;
        }
    }
    Some((sets, length_size))
}

fn hvcc_sets(config: &[u8]) -> Option<(Vec<&[u8]>, usize)> {
    if config.len() < 23 || config[0] != 1 {
        return None;
    }
    let length_size = usize::from(config[21] & 3) + 1;
    let mut sets = Vec::new();
    let mut at = 23;
    for _ in 0..config[22] {
        let count = usize::from(u16::from_be_bytes([
            *config.get(at + 1)?,
            *config.get(at + 2)?,
        ]));
        at += 3;
        for _ in 0..count {
            let length = usize::from(u16::from_be_bytes([*config.get(at)?, *config.get(at + 1)?]));
            sets.push(config.get(at + 2..at + 2 + length)?);
            at += 2 + length;
        }
    }
    Some((sets, length_size))
}

fn starts_annex_b(data: &[u8]) -> bool {
    data.starts_with(&[0, 0, 1]) || data.starts_with(&[0, 0, 0, 1])
}

/// A packet's NAL units: Annex-B where it begins with a start code, and
/// otherwise each behind a `length_size`-byte length.
fn nal_units(data: &[u8], length_size: usize) -> Vec<&[u8]> {
    if starts_annex_b(data) {
        return annex_b(data);
    }
    let mut units = Vec::new();
    let mut at = 0;
    while at + length_size <= data.len() {
        let length = data[at..at + length_size]
            .iter()
            .fold(0usize, |length, &byte| (length << 8) | usize::from(byte));
        at += length_size;
        let Some(unit) = data.get(at..at + length) else {
            break;
        };
        units.push(unit);
        at += length;
    }
    units
}

/// The NAL units of an Annex-B stream, each without its start code.
pub(crate) fn annex_b(data: &[u8]) -> Vec<&[u8]> {
    let mut starts = Vec::new();
    let mut offset = 0;
    while offset + 3 <= data.len() {
        let length = if data[offset..].starts_with(&[0, 0, 0, 1]) {
            4
        } else if data[offset..].starts_with(&[0, 0, 1]) {
            3
        } else {
            offset += 1;
            continue;
        };
        starts.push((offset, length));
        offset += length;
    }
    starts
        .iter()
        .enumerate()
        .filter_map(|(index, (start, length))| {
            let unit_start = start + length;
            let unit_end = starts
                .get(index + 1)
                .map(|(next, _)| *next)
                .unwrap_or(data.len());
            (unit_start < unit_end).then_some(&data[unit_start..unit_end])
        })
        .collect()
}

/// A NAL unit's payload with the emulation prevention bytes taken out.
fn rbsp(payload: &[u8]) -> Vec<u8> {
    let mut rbsp = Vec::with_capacity(payload.len());
    let mut zeros = 0;
    for &byte in payload {
        if zeros >= 2 && byte == 3 {
            zeros = 0;
            continue;
        }
        rbsp.push(byte);
        zeros = if byte == 0 { zeros + 1 } else { 0 };
    }
    rbsp
}

/// The picture size after cropping, from the coded size, the crop offsets
/// and the units they are in.
fn cropped(
    (width, height): (u32, u32),
    (left, right, top, bottom): (u32, u32, u32, u32),
    (unit_x, unit_y): (u32, u32),
) -> Option<(u32, u32)> {
    let width = width.checked_sub(left.checked_add(right)?.checked_mul(unit_x)?)?;
    let height = height.checked_sub(top.checked_add(bottom)?.checked_mul(unit_y)?)?;
    (width > 0 && height > 0 && width <= i32::MAX as u32 && height <= i32::MAX as u32)
        .then_some((width, height))
}

/// What an H.264 sequence parameter set — the NAL unit, its header byte
/// first — says the pictures are.
pub(crate) fn h264_shape(sps: &[u8]) -> Option<Shape> {
    if sps.len() < 4 || sps[0] & 0x1f != 7 {
        return None;
    }
    let rbsp = rbsp(&sps[1..]);
    let mut bits = BitReader::new(&rbsp);
    let profile = bits.read_bits(8)? as u8;
    bits.read_bits(8)?;
    bits.read_bits(8)?;
    bits.read_ue()?;

    let mut chroma_format = 1;
    let mut separate_colour_plane = false;
    let mut bit_depth = 8;
    if matches!(
        profile,
        44 | 83 | 86 | 100 | 110 | 118 | 122 | 128 | 134 | 135 | 138 | 139 | 244
    ) {
        chroma_format = bits.read_ue()?;
        if chroma_format > 3 {
            return None;
        }
        if chroma_format == 3 {
            separate_colour_plane = bits.read_bit()?;
        }
        bit_depth = bits.read_ue()?.checked_add(8)?;
        bits.read_ue()?;
        bits.read_bit()?;
        if bits.read_bit()? {
            let count = if chroma_format == 3 { 12 } else { 8 };
            for index in 0..count {
                if bits.read_bit()? {
                    skip_scaling_list(&mut bits, if index < 6 { 16 } else { 64 })?;
                }
            }
        }
    }

    bits.read_ue()?;
    match bits.read_ue()? {
        0 => {
            bits.read_ue()?;
        }
        1 => {
            bits.read_bit()?;
            bits.read_se()?;
            bits.read_se()?;
            for _ in 0..bits.read_ue()? {
                bits.read_se()?;
            }
        }
        2 => {}
        _ => return None,
    }
    bits.read_ue()?;
    bits.read_bit()?;
    let width_in_mbs = bits.read_ue()?.checked_add(1)?;
    let height_in_map_units = bits.read_ue()?.checked_add(1)?;
    let frame_mbs_only = bits.read_bit()?;
    if !frame_mbs_only {
        bits.read_bit()?;
    }
    bits.read_bit()?;
    let crop = if bits.read_bit()? {
        (
            bits.read_ue()?,
            bits.read_ue()?,
            bits.read_ue()?,
            bits.read_ue()?,
        )
    } else {
        (0, 0, 0, 0)
    };

    let frame_factor = if frame_mbs_only { 1 } else { 2 };
    let chroma_array_type = if separate_colour_plane {
        0
    } else {
        chroma_format
    };
    let (sub_width_c, sub_height_c) = match chroma_array_type {
        0 | 3 => (1, 1),
        1 => (2, 2),
        2 => (2, 1),
        _ => return None,
    };
    let unit = if chroma_array_type == 0 {
        (1, frame_factor)
    } else {
        (sub_width_c, sub_height_c * frame_factor)
    };
    let coded = (
        width_in_mbs.checked_mul(16)?,
        height_in_map_units
            .checked_mul(16)?
            .checked_mul(frame_factor)?,
    );
    let (width, height) = cropped(coded, crop, unit)?;
    Some(Shape {
        width,
        height,
        profile,
        chroma_format,
        bit_depth,
    })
}

/// What an H.265 sequence parameter set — the NAL unit, its two header
/// bytes first — says the pictures are.
pub(crate) fn hevc_shape(sps: &[u8]) -> Option<Shape> {
    if sps.len() < 4 || (sps[0] >> 1) & 0x3f != 33 {
        return None;
    }
    let rbsp = rbsp(&sps[2..]);
    let mut bits = BitReader::new(&rbsp);
    bits.read_bits(4)?;
    let max_sub_layers_minus1 = bits.read_bits(3)? as usize;
    bits.read_bit()?;

    // profile_tier_level(1, max_sub_layers_minus1)
    bits.read_bits(3)?;
    let profile = bits.read_bits(5)? as u8;
    bits.skip(32 + 4 + 43 + 1)?;
    bits.read_bits(8)?;
    let mut present = Vec::with_capacity(max_sub_layers_minus1);
    for _ in 0..max_sub_layers_minus1 {
        present.push((bits.read_bit()?, bits.read_bit()?));
    }
    if max_sub_layers_minus1 > 0 {
        bits.skip(2 * (8 - max_sub_layers_minus1))?;
    }
    for (profile_present, level_present) in present {
        if profile_present {
            bits.skip(88)?;
        }
        if level_present {
            bits.skip(8)?;
        }
    }

    bits.read_ue()?;
    let chroma_format = bits.read_ue()?;
    if chroma_format > 3 {
        return None;
    }
    let separate_colour_plane = chroma_format == 3 && bits.read_bit()?;
    let coded = (bits.read_ue()?, bits.read_ue()?);
    let crop = if bits.read_bit()? {
        (
            bits.read_ue()?,
            bits.read_ue()?,
            bits.read_ue()?,
            bits.read_ue()?,
        )
    } else {
        (0, 0, 0, 0)
    };
    let bit_depth = bits.read_ue()?.checked_add(8)?;
    let unit = match (separate_colour_plane, chroma_format) {
        (false, 1) => (2, 2),
        (false, 2) => (2, 1),
        _ => (1, 1),
    };
    let (width, height) = cropped(coded, crop, unit)?;
    Some(Shape {
        width,
        height,
        profile,
        chroma_format,
        bit_depth,
    })
}

fn skip_scaling_list(bits: &mut BitReader<'_>, size: usize) -> Option<()> {
    let mut last_scale = 8i32;
    let mut next_scale = 8i32;
    for _ in 0..size {
        if next_scale != 0 {
            next_scale = (last_scale + bits.read_se()? + 256) % 256;
        }
        if next_scale != 0 {
            last_scale = next_scale;
        }
    }
    Some(())
}

struct BitReader<'a> {
    data: &'a [u8],
    bit: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, bit: 0 }
    }

    fn skip(&mut self, count: usize) -> Option<()> {
        let to = self.bit.checked_add(count)?;
        (to <= self.data.len().checked_mul(8)?).then(|| self.bit = to)
    }

    fn read_bit(&mut self) -> Option<bool> {
        Some(self.read_bits(1)? != 0)
    }

    fn read_bits(&mut self, count: usize) -> Option<u32> {
        if count > 32 || self.bit.checked_add(count)? > self.data.len().checked_mul(8)? {
            return None;
        }
        let mut value = 0u32;
        for _ in 0..count {
            let byte = self.data[self.bit / 8];
            value = (value << 1) | u32::from((byte >> (7 - self.bit % 8)) & 1);
            self.bit += 1;
        }
        Some(value)
    }

    fn read_ue(&mut self) -> Option<u32> {
        let mut leading_zeroes = 0usize;
        while !self.read_bit()? {
            leading_zeroes += 1;
            if leading_zeroes > 31 {
                return None;
            }
        }
        let suffix = self.read_bits(leading_zeroes)?;
        ((1u32 << leading_zeroes) - 1).checked_add(suffix)
    }

    fn read_se(&mut self) -> Option<i32> {
        let code_num = self.read_ue()?;
        let magnitude = i32::try_from(code_num.checked_add(1)? / 2).ok()?;
        Some(if code_num % 2 == 0 {
            -magnitude
        } else {
            magnitude
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One 1920x1080 picture's sequence parameter set, as NVENC wrote it
    /// for an RGB picture: the range extensions profile, 4:4:4, coding 1088
    /// rows and cropping eight.
    const HEVC_1080P_SPS: &[u8] = &[
        0x42, 0x01, 0x01, 0x04, 0x08, 0x00, 0x00, 0x03, 0x00, 0x9e, 0x08, 0x00, 0x00, 0x03, 0x00,
        0x00, 0x78, 0x90, 0x00, 0x78, 0x10, 0x02, 0x20, 0xf8, 0x9c, 0xb3, 0x3a, 0x52, 0x10, 0x8c,
        0x8d, 0xfa, 0x86, 0x02, 0xdc, 0x08, 0x08, 0x00, 0x10, 0x00, 0x00, 0x03, 0x00, 0x10, 0x00,
        0x00, 0x03, 0x01, 0xe6, 0x00, 0x5d, 0xe5, 0x10, 0x00, 0x0b, 0x71, 0xb0, 0x00, 0x03, 0xd0,
        0x90, 0x10,
    ];

    /// One 1280x720 picture's, NVENC's again, for a 4:2:0 picture: Main
    /// profile, coding 736 rows and cropping sixteen, in chroma rows of two.
    const HEVC_720P_SPS: &[u8] = &[
        0x42, 0x01, 0x01, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0x90, 0x00, 0x00, 0x03, 0x00, 0x00,
        0x03, 0x00, 0x5d, 0xa0, 0x02, 0x80, 0x80, 0x2e, 0x1f, 0x13, 0x96, 0x67, 0x4a, 0x42, 0x11,
        0x91, 0xbf, 0x50, 0xc0, 0x40, 0x40, 0x00, 0x00, 0x03, 0x00, 0x40, 0x00, 0x00, 0x07, 0x98,
        0x01, 0x77, 0x94, 0x40, 0x00, 0x2d, 0xc6, 0xc0, 0x00, 0x0f, 0x42, 0x40, 0x40,
    ];

    /// Five pictures of `size` from OpenH264, which writes its parameter
    /// sets in band, in front of its keyframe.
    fn h264(size: (u32, u32)) -> Vec<ffmpeg::Packet> {
        crate::test_support::try_encoded_packets(
            "libopenh264",
            ffmpeg::format::Pixel::YUV420P,
            size,
            128,
        )
        .expect("OpenH264 is always present")
        .1
    }

    fn sizes(change: Option<(Shape, Shape)>) -> Option<((u32, u32), (u32, u32))> {
        change.map(|(was, now)| ((was.width, was.height), (now.width, now.height)))
    }

    #[test]
    fn an_hevc_sequence_parameter_set_says_its_size_cropped() {
        assert_eq!(
            hevc_shape(HEVC_1080P_SPS),
            Some(Shape {
                width: 1920,
                height: 1080,
                profile: 4,
                chroma_format: 3,
                bit_depth: 8,
            })
        );
        assert_eq!(
            hevc_shape(HEVC_720P_SPS),
            Some(Shape {
                width: 1280,
                height: 720,
                profile: 1,
                chroma_format: 1,
                bit_depth: 8,
            })
        );
    }

    #[test]
    fn an_h264_sequence_parameter_set_says_its_size_cropped() {
        // 360 rows are coded as 368, and cropped.
        let packets = h264((640, 360));
        let sps = annex_b(packets[0].data().unwrap())
            .into_iter()
            .find(|nal| Codec::H264.is_sps(nal))
            .expect("the keyframe carries its parameter sets");
        let shape = h264_shape(sps).expect("read");
        assert_eq!((shape.width, shape.height), (640, 360));
        assert_eq!((shape.chroma_format, shape.bit_depth), (1, 8));
    }

    /// The first set a stream carries is what it began as, the same set
    /// again changes nothing, and another size is said.
    #[test]
    fn a_watch_says_when_the_pictures_are_no_longer_what_they_began_as() {
        let (small, large) = (h264((320, 240)), h264((640, 360)));
        let mut watch = ShapeWatch::learning(ffmpeg::codec::Id::H264).unwrap();
        for packet in &small {
            assert_eq!(watch.changed(packet.data().unwrap()), None);
        }
        assert_eq!(
            watch.began().map(|shape| (shape.width, shape.height)),
            Some((320, 240))
        );
        assert_eq!(
            sizes(watch.changed(large[0].data().unwrap())),
            Some(((320, 240), (640, 360)))
        );
        assert_eq!(
            ShapeWatch::learning(ffmpeg::codec::Id::VP8).map(|_| ()),
            None,
            "a codec whose sets are not read is not watched"
        );
    }

    /// What a stream began as comes from its configuration where it has
    /// one, `avcC` or Annex-B, and a packet is read the way the
    /// configuration says: length-prefixed after an `avcC`.
    #[test]
    fn a_configuration_says_what_the_stream_began_as() {
        let small = h264((320, 240));
        let units = annex_b(small[0].data().unwrap());
        let sps = *units.iter().find(|nal| nal[0] & 0x1f == 7).unwrap();
        let pps = *units.iter().find(|nal| nal[0] & 0x1f == 8).unwrap();
        let mut avcc = vec![1, sps[1], sps[2], sps[3], 0xff, 0xe1];
        avcc.extend_from_slice(&(sps.len() as u16).to_be_bytes());
        avcc.extend_from_slice(sps);
        avcc.push(1);
        avcc.extend_from_slice(&(pps.len() as u16).to_be_bytes());
        avcc.extend_from_slice(pps);
        let mut annex = vec![0, 0, 0, 1];
        annex.extend_from_slice(sps);
        annex.extend_from_slice(&[0, 0, 0, 1]);
        annex.extend_from_slice(pps);

        let large = h264((640, 360));
        let prefixed: Vec<u8> = annex_b(large[0].data().unwrap())
            .into_iter()
            .flat_map(|nal| {
                let mut unit = (nal.len() as u32).to_be_bytes().to_vec();
                unit.extend_from_slice(nal);
                unit
            })
            .collect();

        for (config, packet) in [(&avcc, &prefixed[..]), (&annex, large[0].data().unwrap())] {
            let mut parameters = ffmpeg::codec::Parameters::new();
            // SAFETY: parameters this test owns alone; the configuration is
            // copied into a buffer FFmpeg allocates, padded as it requires,
            // and frees with the parameters.
            unsafe {
                let raw = parameters.as_mut_ptr();
                (*raw).codec_type = ffmpeg::ffi::AVMediaType::AVMEDIA_TYPE_VIDEO;
                (*raw).codec_id = ffmpeg::codec::Id::H264.into();
                let padding = ffmpeg::ffi::AV_INPUT_BUFFER_PADDING_SIZE as usize;
                let data = ffmpeg::ffi::av_mallocz(config.len() + padding).cast::<u8>();
                std::ptr::copy_nonoverlapping(config.as_ptr(), data, config.len());
                (*raw).extradata = data;
                (*raw).extradata_size = config.len() as i32;
            }
            let mut watch = ShapeWatch::new(&parameters).unwrap();
            assert_eq!(
                watch.began().map(|shape| (shape.width, shape.height)),
                Some((320, 240)),
                "from the configuration, before any packet"
            );
            assert_eq!(sizes(watch.changed(packet)), Some(((320, 240), (640, 360))));
        }
    }
}
