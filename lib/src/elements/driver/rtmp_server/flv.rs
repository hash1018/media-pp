//! What an RTMP audio or video message holds: an FLV tag's body, legacy or
//! as Enhanced RTMP extends it.

use ffmpeg_next::codec::Id;

/// What one audio or video message says.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Tag<'a> {
    /// A track's configuration — its codec and the record FFmpeg takes as
    /// extradata (`avcC`, `hvcC`, `av1C`, `vpcC`, an AudioSpecificConfig, an
    /// `OpusHead`).
    Config { codec: Id, extradata: &'a [u8] },
    /// One coded frame.
    Frame {
        codec: Id,
        key: bool,
        /// Composition time: how far its presentation follows its decoding,
        /// in milliseconds.
        cts: i32,
        data: &'a [u8],
    },
    /// The end of a track's sequence.
    End,
    /// Something a server passes over, and why.
    Ignored(&'static str),
}

fn video_fourcc(fourcc: &[u8]) -> Option<Id> {
    Some(match fourcc {
        b"avc1" => Id::H264,
        b"hvc1" => Id::HEVC,
        b"av01" => Id::AV1,
        b"vp09" => Id::VP9,
        _ => return None,
    })
}

fn audio_fourcc(fourcc: &[u8]) -> Option<Id> {
    Some(match fourcc {
        b"mp4a" => Id::AAC,
        b"Opus" => Id::OPUS,
        _ => return None,
    })
}

/// A signed 24-bit big-endian number.
fn si24(bytes: &[u8]) -> i32 {
    (i32::from_be_bytes([bytes[0], bytes[1], bytes[2], 0])) >> 8
}

/// A video message's body.
pub(crate) fn video(body: &[u8]) -> Result<Tag<'_>, &'static str> {
    let Some(&first) = body.first() else {
        return Ok(Tag::Ignored("an empty video message"));
    };
    if first & 0x80 != 0 {
        return enhanced_video(body);
    }
    let key = first >> 4 == 1;
    if first >> 4 == 5 {
        return Ok(Tag::Ignored("a video command frame"));
    }
    let codec = match first & 0x0f {
        7 => Id::H264,
        // Not in the FLV specification, but what encoders sent before
        // Enhanced RTMP gave H.265 a FourCC.
        12 => Id::HEVC,
        _ => return Ok(Tag::Ignored("a video codec other than H.264 and H.265")),
    };
    if body.len() < 5 {
        return Err("a video message cut short");
    }
    match body[1] {
        0 => Ok(Tag::Config {
            codec,
            extradata: &body[5..],
        }),
        1 => Ok(Tag::Frame {
            codec,
            key,
            cts: si24(&body[2..5]),
            data: &body[5..],
        }),
        2 => Ok(Tag::End),
        _ => Err("an unknown AVC packet type"),
    }
}

fn enhanced_video(body: &[u8]) -> Result<Tag<'_>, &'static str> {
    if body.len() < 5 {
        return Err("a video message cut short");
    }
    let frame_type = (body[0] >> 4) & 0x07;
    let packet_type = body[0] & 0x0f;
    if frame_type == 5 {
        return Ok(Tag::Ignored("a video command frame"));
    }
    let Some(codec) = video_fourcc(&body[1..5]) else {
        return Ok(Tag::Ignored(match packet_type {
            6 => "multiple video tracks in one message",
            _ => "a video codec this server does not take",
        }));
    };
    let key = frame_type == 1;
    let rest = &body[5..];
    match packet_type {
        0 => Ok(Tag::Config {
            codec,
            extradata: rest,
        }),
        // H.264 and H.265 frames carry a composition time; the others, and
        // CodedFramesX, have none.
        1 if matches!(codec, Id::H264 | Id::HEVC) => {
            if rest.len() < 3 {
                return Err("a video message cut short");
            }
            Ok(Tag::Frame {
                codec,
                key,
                cts: si24(&rest[..3]),
                data: &rest[3..],
            })
        }
        1 | 3 => Ok(Tag::Frame {
            codec,
            key,
            cts: 0,
            data: rest,
        }),
        2 => Ok(Tag::End),
        4 => Ok(Tag::Ignored("video metadata")),
        5 => Ok(Tag::Ignored("an MPEG-2 TS sequence start")),
        _ => Ok(Tag::Ignored(
            "a video packet type this server does not take",
        )),
    }
}

/// An audio message's body.
pub(crate) fn audio(body: &[u8]) -> Result<Tag<'_>, &'static str> {
    let Some(&first) = body.first() else {
        return Ok(Tag::Ignored("an empty audio message"));
    };
    match first >> 4 {
        10 => {
            if body.len() < 2 {
                return Err("an audio message cut short");
            }
            match body[1] {
                0 => Ok(Tag::Config {
                    codec: Id::AAC,
                    extradata: &body[2..],
                }),
                1 => Ok(Tag::Frame {
                    codec: Id::AAC,
                    key: true,
                    cts: 0,
                    data: &body[2..],
                }),
                _ => Err("an unknown AAC packet type"),
            }
        }
        9 => {
            if body.len() < 5 {
                return Err("an audio message cut short");
            }
            let packet_type = first & 0x0f;
            let Some(codec) = audio_fourcc(&body[1..5]) else {
                return Ok(Tag::Ignored(match packet_type {
                    5 => "multiple audio tracks in one message",
                    _ => "an audio codec this server does not take",
                }));
            };
            let rest = &body[5..];
            match packet_type {
                0 => Ok(Tag::Config {
                    codec,
                    extradata: rest,
                }),
                1 => Ok(Tag::Frame {
                    codec,
                    key: true,
                    cts: 0,
                    data: rest,
                }),
                2 => Ok(Tag::End),
                _ => Ok(Tag::Ignored(
                    "an audio packet type this server does not take",
                )),
            }
        }
        _ => Ok(Tag::Ignored("an audio codec other than AAC and Opus")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_h264_and_aac_are_read() {
        let config = [0x17, 0, 0, 0, 0, 1, 0x64, 0, 0x1f];
        assert_eq!(
            video(&config),
            Ok(Tag::Config {
                codec: Id::H264,
                extradata: &[1, 0x64, 0, 0x1f]
            })
        );
        // An inter frame shown 80 ms after it is decoded.
        let frame = [0x27, 1, 0, 0, 80, 0, 0, 0, 2, 0x41, 0x9a];
        assert_eq!(
            video(&frame),
            Ok(Tag::Frame {
                codec: Id::H264,
                key: false,
                cts: 80,
                data: &[0, 0, 0, 2, 0x41, 0x9a]
            })
        );
        // A negative composition time reads as one.
        assert!(matches!(
            video(&[0x17, 1, 0xff, 0xff, 0xd8, 1]),
            Ok(Tag::Frame {
                key: true,
                cts: -40,
                ..
            })
        ));
        assert_eq!(
            audio(&[0xaf, 0, 0x12, 0x10]),
            Ok(Tag::Config {
                codec: Id::AAC,
                extradata: &[0x12, 0x10]
            })
        );
        assert!(matches!(
            audio(&[0xaf, 1, 0x21]),
            Ok(Tag::Frame {
                codec: Id::AAC,
                key: true,
                ..
            })
        ));
    }

    #[test]
    fn enhanced_rtmp_is_read() {
        // H.265: a sequence start, a keyframe with its composition time, and
        // CodedFramesX without one.
        assert_eq!(
            video(b"\x90hvc1\x01\x02"),
            Ok(Tag::Config {
                codec: Id::HEVC,
                extradata: &[1, 2]
            })
        );
        assert_eq!(
            video(b"\x91hvc1\x00\x00\x28\x07"),
            Ok(Tag::Frame {
                codec: Id::HEVC,
                key: true,
                cts: 40,
                data: &[7]
            })
        );
        assert_eq!(
            video(b"\xa3hvc1\x07"),
            Ok(Tag::Frame {
                codec: Id::HEVC,
                key: false,
                cts: 0,
                data: &[7]
            })
        );
        // AV1 frames have no composition time.
        assert_eq!(
            video(b"\x91av01\x12\x00"),
            Ok(Tag::Frame {
                codec: Id::AV1,
                key: true,
                cts: 0,
                data: &[0x12, 0]
            })
        );
        assert_eq!(
            audio(b"\x90OpusOpusHead"),
            Ok(Tag::Config {
                codec: Id::OPUS,
                extradata: b"OpusHead"
            })
        );
        assert!(
            matches!(video(b"\x96avc1"), Ok(Tag::Ignored(_))),
            "multitrack"
        );
        assert!(matches!(video(b"\x90xxxx"), Ok(Tag::Ignored(_))));
        assert!(video(b"\x91hvc1\x00").is_err(), "cut short");
    }
}
