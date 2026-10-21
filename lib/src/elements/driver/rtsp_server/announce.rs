//! What a publisher announces: its SDP read into tracks this server can
//! take.

use std::collections::HashMap;

use ffmpeg_next::{self as ffmpeg, ffi};

/// What a track's payloads are, as far as taking them apart goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Payload {
    H264,
    H265,
    /// `mpeg4-generic` AAC: AU headers of `size_length` and `index_length`
    /// bits ahead of the frames.
    Aac {
        size_length: u8,
        index_length: u8,
    },
    /// One packet a payload: Opus, G.711, and the like.
    Whole,
}

/// One announced track.
#[derive(Debug, Clone)]
pub(crate) struct Announced {
    /// Its control URL, as the description gives it — relative or whole.
    pub(crate) control: String,
    pub(crate) medium: ffmpeg::media::Type,
    pub(crate) codec: ffmpeg::codec::Id,
    pub(crate) payload: Payload,
    pub(crate) clock_rate: u32,
    pub(crate) channels: u32,
    /// What the decoder needs ahead of the stream: H.264's and H.265's
    /// parameter sets in Annex B, AAC's AudioSpecificConfig.
    pub(crate) extradata: Vec<u8>,
}

impl Announced {
    /// Its codec parameters, as far as the description says them — a
    /// picture's size is read from its first keyframe.
    pub(crate) fn parameters(&self) -> ffmpeg::codec::Parameters {
        let mut parameters = ffmpeg::codec::Parameters::new();
        // SAFETY: parameters made here and owned alone; the extradata is
        // allocated with FFmpeg's allocator and its padding, as it frees it.
        unsafe {
            let par = parameters.as_mut_ptr();
            (*par).codec_type = self.medium.into();
            (*par).codec_id = self.codec.into();
            if self.medium == ffmpeg::media::Type::Audio {
                (*par).sample_rate = i32::try_from(self.clock_rate).unwrap_or(0);
                ffi::av_channel_layout_default(
                    &mut (*par).ch_layout,
                    i32::try_from(self.channels.max(1)).unwrap_or(1),
                );
                if self.codec == ffmpeg::codec::Id::AAC {
                    (*par).frame_size = 1024;
                }
            }
            if !self.extradata.is_empty() {
                let padded = self.extradata.len() + ffi::AV_INPUT_BUFFER_PADDING_SIZE as usize;
                let data = ffi::av_mallocz(padded).cast::<u8>();
                if !data.is_null() {
                    std::ptr::copy_nonoverlapping(
                        self.extradata.as_ptr(),
                        data,
                        self.extradata.len(),
                    );
                    (*par).extradata = data;
                    (*par).extradata_size = self.extradata.len() as i32;
                }
            }
        }
        parameters
    }

    /// The unit its timestamps count in: its RTP clock's.
    pub(crate) fn time_base(&self) -> ffmpeg::Rational {
        ffmpeg::Rational(1, i32::try_from(self.clock_rate).unwrap_or(90_000).max(1))
    }
}

/// Why an announcement cannot be taken.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Refused {
    /// Not an SDP with a media section.
    Malformed,
    /// A track in a codec this server cannot take apart.
    Unsupported(String),
}

/// The tracks `sdp` announces, in its order.
pub(crate) fn parse(sdp: &str) -> Result<Vec<Announced>, Refused> {
    let mut tracks = Vec::new();
    let mut current: Option<Section> = None;
    for line in sdp.lines().map(str::trim) {
        if let Some(media) = line.strip_prefix("m=") {
            if let Some(section) = current.take() {
                tracks.push(section.finish()?);
            }
            current = Some(Section::new(media)?);
        } else if let (Some(section), Some(attribute)) = (current.as_mut(), line.strip_prefix("a="))
        {
            section.attribute(attribute);
        }
    }
    if let Some(section) = current.take() {
        tracks.push(section.finish()?);
    }
    if tracks.is_empty() {
        return Err(Refused::Malformed);
    }
    Ok(tracks)
}

/// A media section as it is read.
struct Section {
    medium: ffmpeg::media::Type,
    payload_type: String,
    control: Option<String>,
    rtpmap: Option<String>,
    fmtp: HashMap<String, String>,
}

impl Section {
    fn new(media: &str) -> Result<Self, Refused> {
        let mut fields = media.split_whitespace();
        let medium = match fields.next() {
            Some("video") => ffmpeg::media::Type::Video,
            Some("audio") => ffmpeg::media::Type::Audio,
            Some(other) => return Err(Refused::Unsupported(format!("{other} media"))),
            None => return Err(Refused::Malformed),
        };
        let payload_type = fields.nth(2).ok_or(Refused::Malformed)?.to_owned();
        Ok(Self {
            medium,
            payload_type,
            control: None,
            rtpmap: None,
            fmtp: HashMap::new(),
        })
    }

    fn attribute(&mut self, attribute: &str) {
        if let Some(control) = attribute.strip_prefix("control:") {
            self.control = Some(control.trim().to_owned());
        } else if let Some(rtpmap) = attribute.strip_prefix("rtpmap:") {
            if let Some((pt, map)) = rtpmap.split_once(' ')
                && pt == self.payload_type
            {
                self.rtpmap = Some(map.trim().to_owned());
            }
        } else if let Some(fmtp) = attribute.strip_prefix("fmtp:")
            && let Some((pt, params)) = fmtp.split_once(' ')
            && pt == self.payload_type
        {
            for param in params.split(';') {
                if let Some((key, value)) = param.split_once('=') {
                    self.fmtp
                        .insert(key.trim().to_ascii_lowercase(), value.trim().to_owned());
                }
            }
        }
    }

    fn finish(self) -> Result<Announced, Refused> {
        // A static payload type says its codec without an rtpmap.
        let rtpmap = match (self.rtpmap.as_deref(), self.payload_type.as_str()) {
            (Some(map), _) => map.to_owned(),
            (None, "0") => "PCMU/8000/1".to_owned(),
            (None, "8") => "PCMA/8000/1".to_owned(),
            (None, other) => {
                return Err(Refused::Unsupported(format!(
                    "payload type {other} with no rtpmap"
                )));
            }
        };
        let mut parts = rtpmap.split('/');
        let name = parts.next().unwrap_or("").to_ascii_uppercase();
        let clock_rate: u32 = parts
            .next()
            .and_then(|rate| rate.parse().ok())
            .ok_or(Refused::Malformed)?;
        let channels: u32 = parts.next().and_then(|n| n.parse().ok()).unwrap_or(1);
        let fmtp = &self.fmtp;
        let (codec, payload, extradata) = match name.as_str() {
            "H264" => (
                ffmpeg::codec::Id::H264,
                Payload::H264,
                annex_b(
                    fmtp.get("sprop-parameter-sets")
                        .map(String::as_str)
                        .into_iter()
                        .flat_map(|sets| sets.split(',')),
                ),
            ),
            "H265" => (
                ffmpeg::codec::Id::HEVC,
                Payload::H265,
                annex_b(
                    ["sprop-vps", "sprop-sps", "sprop-pps"]
                        .iter()
                        .filter_map(|key| fmtp.get(*key))
                        .flat_map(|sets| sets.split(',')),
                ),
            ),
            "MPEG4-GENERIC" => {
                let mode = fmtp.get("mode").map(|m| m.to_ascii_lowercase());
                if mode
                    .as_deref()
                    .is_some_and(|mode| mode != "aac-hbr" && mode != "aac-lbr")
                {
                    return Err(Refused::Unsupported(format!("mpeg4-generic mode {mode:?}")));
                }
                let size_length = fmtp
                    .get("sizelength")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(13);
                let index_length = fmtp
                    .get("indexlength")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(3);
                let config = fmtp
                    .get("config")
                    .map(|hex| from_hex(hex))
                    .unwrap_or_default();
                (
                    ffmpeg::codec::Id::AAC,
                    Payload::Aac {
                        size_length,
                        index_length,
                    },
                    config,
                )
            }
            "OPUS" => (ffmpeg::codec::Id::OPUS, Payload::Whole, Vec::new()),
            "PCMU" => (ffmpeg::codec::Id::PCM_MULAW, Payload::Whole, Vec::new()),
            "PCMA" => (ffmpeg::codec::Id::PCM_ALAW, Payload::Whole, Vec::new()),
            other => return Err(Refused::Unsupported(other.to_owned())),
        };
        Ok(Announced {
            control: self.control.unwrap_or_default(),
            medium: self.medium,
            codec,
            payload,
            clock_rate,
            channels,
            extradata,
        })
    }
}

/// Base64 parameter sets as Annex B: each behind a start code.
fn annex_b<'a>(sets: impl Iterator<Item = &'a str>) -> Vec<u8> {
    let mut out = Vec::new();
    for set in sets {
        if let Some(nal) = base64(set.trim()) {
            if nal.is_empty() {
                continue;
            }
            out.extend_from_slice(&[0, 0, 0, 1]);
            out.extend_from_slice(&nal);
        }
    }
    out
}

fn base64(text: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len() * 3 / 4);
    let mut bits = 0u32;
    let mut count = 0;
    for byte in text.bytes() {
        let value = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' => break,
            _ => return None,
        };
        bits = (bits << 6) | u32::from(value);
        count += 6;
        if count >= 8 {
            count -= 8;
            out.push((bits >> count) as u8);
        }
    }
    Some(out)
}

fn from_hex(hex: &str) -> Vec<u8> {
    hex.as_bytes()
        .chunks(2)
        .filter_map(|pair| u8::from_str_radix(std::str::from_utf8(pair).ok()?, 16).ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffmpegs_announcement_is_read() {
        let sdp = "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=No Name\r\nc=IN IP4 127.0.0.1\r\nt=0 0\r\n\
                   m=video 0 RTP/AVP 96\r\na=rtpmap:96 H264/90000\r\n\
                   a=fmtp:96 packetization-mode=1; sprop-parameter-sets=Z0LAHtkDxWhAAAADAEAAAAwDxYuS,aMuMsg==; profile-level-id=42C01E\r\n\
                   a=control:streamid=0\r\n\
                   m=audio 0 RTP/AVP 97\r\nb=AS:128\r\na=rtpmap:97 MPEG4-GENERIC/44100/2\r\n\
                   a=fmtp:97 profile-level-id=1;mode=AAC-hbr;sizelength=13;indexlength=3;indexdeltalength=3; config=1210\r\n\
                   a=control:streamid=1\r\n";
        let tracks = parse(sdp).unwrap();
        assert_eq!(tracks.len(), 2);
        assert_eq!(tracks[0].codec, ffmpeg::codec::Id::H264);
        assert_eq!(tracks[0].control, "streamid=0");
        assert_eq!(
            &tracks[0].extradata[..5],
            &[0, 0, 0, 1, 0x67],
            "the SPS behind a start code"
        );
        assert!(
            tracks[0]
                .extradata
                .windows(4)
                .filter(|w| w == &[0, 0, 0, 1])
                .count()
                == 2,
            "the SPS and the PPS"
        );
        assert_eq!(tracks[1].codec, ffmpeg::codec::Id::AAC);
        assert_eq!((tracks[1].clock_rate, tracks[1].channels), (44_100, 2));
        assert_eq!(tracks[1].extradata, vec![0x12, 0x10]);
        assert_eq!(
            tracks[1].payload,
            Payload::Aac {
                size_length: 13,
                index_length: 3
            }
        );
    }

    #[test]
    fn what_cannot_be_taken_apart_is_refused() {
        let sdp = "v=0\r\nm=video 0 RTP/AVP 96\r\na=rtpmap:96 JPEG2000/90000\r\n";
        assert_eq!(
            parse(sdp).unwrap_err(),
            Refused::Unsupported("JPEG2000".into())
        );
        assert_eq!(parse("v=0\r\ns=x\r\n").unwrap_err(), Refused::Malformed);
        let g711 = parse("v=0\r\nm=audio 0 RTP/AVP 0\r\n").unwrap();
        assert_eq!(g711[0].codec, ffmpeg::codec::Id::PCM_MULAW);
    }

    #[test]
    fn base64_is_read() {
        assert_eq!(base64("aMuMsg==").unwrap(), vec![0x68, 0xcb, 0x8c, 0xb2]);
        assert_eq!(base64("TWFu").unwrap(), b"Man");
    }
}
