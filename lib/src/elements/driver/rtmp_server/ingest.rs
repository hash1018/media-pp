//! A publisher's audio and video messages handed out as a source:
//! [`Ingest`].

use std::time::{Duration, Instant};

use ffmpeg_next::{self as ffmpeg, codec::Id, ffi};

use crossbeam_channel::TrySendError;

use crate::core::bell::Ringing;
use crate::elements::StreamInfo;
use crate::elements::driver::publishers::Arrivals;
use crate::elements::parameters::{picture_size, set_extradata};
use crate::pp_log::{PpLog, pp_warn};

use super::amf::Amf;
use super::flv::Tag;
use super::source::RtmpSource;

/// RTMP counts time in milliseconds.
const TIME_BASE: ffmpeg::Rational = ffmpeg::Rational(1, 1000);

/// How long an ingest without metadata waits, from its first frame, for
/// the other track's configuration — a publisher sends both up front.
const WAIT_FOR_TRACKS: Duration = Duration::from_secs(1);

/// How long, and how much, an ingest holds while it waits for what it
/// needs before it publishes what it has.
const HOLD_FOR: Duration = Duration::from_secs(5);
const HOLD_AT_MOST: usize = 4096;

/// Why an ingest cannot go on.
pub(crate) enum Stopped {
    /// Nothing takes the publisher: its path's `Publishers` has gone, or
    /// the source it was handed out as was dropped.
    Gone,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Medium {
    Video,
    Audio,
}

/// Where an [`RtmpSource`] is fed: each packet with the output it goes out
/// on.
type Feed = Ringing<(usize, ffmpeg::Packet)>;

/// One track's configuration, as the publisher last sent it.
struct Track {
    codec: Id,
    extradata: Vec<u8>,
    /// A picture's size, once known: from the metadata, or read from the
    /// first keyframe.
    size: Option<(i32, i32)>,
}

/// A publisher's stream on its way out as an [`RtmpSource`].
///
/// The source is handed out once its tracks are known — those the
/// metadata names, or, without metadata, those configured within a second
/// of the first frame — each with its configuration and a picture its
/// size, so that what reads it has them; what came before is held, and
/// handed on then. A configuration that changes later ends that source,
/// and hands out another.
pub(crate) struct Ingest {
    arrivals: Arrivals<RtmpSource>,
    path: String,
    video: Option<Track>,
    audio: Option<Track>,
    /// The tracks the metadata names, where there was metadata.
    expected: Option<(bool, bool)>,
    /// The picture size the metadata gives.
    metadata_size: Option<(i32, i32)>,
    held: Vec<(Medium, ffmpeg::Packet)>,
    first_frame: Option<Instant>,
    /// Where the source handed out is fed, and which medium each of its
    /// outputs is.
    feed: Option<(Vec<Medium>, Feed)>,
    /// Whether the pictures lost a packet the source had no room for, and
    /// go on from their next keyframe.
    losing: bool,
    /// Timestamps unwrapped past 2^32 milliseconds.
    last_timestamp: Option<u32>,
    epoch: i64,
}

// SAFETY: every field is `Send` but the packets held, which are this
// value's own and used only through `&mut self`.
unsafe impl Send for Ingest {}

impl Ingest {
    pub(crate) fn new(arrivals: Arrivals<RtmpSource>, path: &str) -> Self {
        Self {
            arrivals,
            path: path.to_owned(),
            video: None,
            audio: None,
            expected: None,
            metadata_size: None,
            held: Vec::new(),
            first_frame: None,
            feed: None,
            losing: false,
            last_timestamp: None,
            epoch: 0,
        }
    }

    /// Takes `@setDataFrame`'s or `onMetaData`'s object.
    pub(crate) fn metadata(&mut self, metadata: &Amf) {
        let present = |name| {
            metadata
                .get(name)
                .is_some_and(|value| !matches!(value, Amf::Null | Amf::Undefined))
        };
        let (video, audio) = (present("videocodecid"), present("audiocodecid"));
        if video || audio {
            self.expected = Some((video, audio));
        }
        let number = |name| metadata.get(name).and_then(Amf::as_number);
        if let (Some(width), Some(height)) = (number("width"), number("height"))
            && width >= 1.0
            && height >= 1.0
            && width <= 16384.0
            && height <= 16384.0
        {
            self.metadata_size = Some((width as i32, height as i32));
        }
    }

    /// Takes one audio or video message's tag, sent at `timestamp`.
    pub(crate) fn take(
        &mut self,
        medium: Medium,
        tag: Tag<'_>,
        timestamp: u32,
        pp_log: &PpLog,
    ) -> Result<(), Stopped> {
        let dts = self.unwrap(timestamp);
        match tag {
            Tag::Config { codec, extradata } => self.configure(medium, codec, extradata),
            Tag::Frame {
                codec,
                key,
                cts,
                data,
            } => {
                let configured = match medium {
                    Medium::Video => self.video.as_ref(),
                    Medium::Audio => self.audio.as_ref(),
                }
                .is_some_and(|track| track.codec == codec);
                if !configured {
                    // A frame of a track not configured, or of another
                    // codec than its configuration: nothing could read it.
                    return Ok(());
                }
                let mut packet = ffmpeg::Packet::copy(data);
                packet.set_dts(Some(dts));
                packet.set_pts(Some(dts + i64::from(cts)));
                packet.set_time_base(TIME_BASE);
                if key {
                    packet.set_flags(ffmpeg::packet::Flags::KEY);
                }
                if self.feed.is_some() {
                    return self.write(medium, packet, pp_log);
                }
                self.first_frame.get_or_insert_with(Instant::now);
                if medium == Medium::Video
                    && key
                    && let Some(track) = &mut self.video
                    && track.size.is_none()
                {
                    track.size = self
                        .metadata_size
                        .or_else(|| picture_size(track.codec, &track.extradata, data));
                }
                self.held.push((medium, packet));
                if self.ready() {
                    self.hand_out(pp_log)?;
                }
                Ok(())
            }
            Tag::End | Tag::Ignored(_) => Ok(()),
        }
    }

    fn unwrap(&mut self, timestamp: u32) -> i64 {
        if let Some(last) = self.last_timestamp
            && timestamp < last
            && last - timestamp > u32::MAX / 2
        {
            self.epoch += 1 << 32;
        }
        self.last_timestamp = Some(timestamp);
        self.epoch + i64::from(timestamp)
    }

    fn configure(&mut self, medium: Medium, codec: Id, extradata: &[u8]) -> Result<(), Stopped> {
        let slot = match medium {
            Medium::Video => &mut self.video,
            Medium::Audio => &mut self.audio,
        };
        if slot
            .as_ref()
            .is_some_and(|track| track.codec == codec && track.extradata == extradata)
        {
            // The same again, as an encoder sends at each keyframe.
            return Ok(());
        }
        let changed = slot.is_some();
        *slot = Some(Track {
            codec,
            extradata: extradata.to_vec(),
            size: None,
        });
        if changed && self.feed.is_some() {
            // A new configuration mid-stream — a new resolution: the source
            // ends, and another is handed out from its next keyframe, which
            // is what readers start at.
            self.feed = None;
            self.losing = false;
            self.first_frame = None;
            if let Some(video) = &mut self.video {
                video.size = None;
            }
            self.metadata_size = None;
        }
        Ok(())
    }

    /// Whether what is held says enough to publish — or has waited as
    /// long, or grown as large, as it may.
    fn ready(&self) -> bool {
        let Some(first) = self.first_frame else {
            return false;
        };
        if self.held.len() >= HOLD_AT_MOST || first.elapsed() >= HOLD_FOR {
            return self.video.is_some() || self.audio.is_some();
        }
        let video_ready = self
            .video
            .as_ref()
            .is_some_and(|video| video.size.is_some());
        let audio_ready = self.audio.is_some();
        let (want_video, want_audio) = match self.expected {
            Some(expected) => expected,
            None if first.elapsed() >= WAIT_FOR_TRACKS => {
                (self.video.is_some(), self.audio.is_some())
            }
            None => (true, true),
        };
        (!want_video || video_ready) && (!want_audio || audio_ready) && (want_video || want_audio)
    }

    /// Hands the publisher out as a source of its tracks, and feeds it what
    /// was held.
    fn hand_out(&mut self, pp_log: &PpLog) -> Result<(), Stopped> {
        let mut media = Vec::new();
        let mut streams = Vec::new();
        for (medium, track) in [(Medium::Video, &self.video), (Medium::Audio, &self.audio)] {
            if let Some(track) = track {
                streams.push(
                    StreamInfo::new(parameters(medium, track), TIME_BASE).with_index(media.len()),
                );
                media.push(medium);
            }
        }
        let (source, feed) = RtmpSource::published(&self.path, &streams);
        if !self.arrivals.hand(source, streams) {
            return Err(Stopped::Gone);
        }
        self.feed = Some((media, feed));
        for (medium, packet) in std::mem::take(&mut self.held) {
            self.write(medium, packet, pp_log)?;
        }
        Ok(())
    }

    fn write(
        &mut self,
        medium: Medium,
        packet: ffmpeg::Packet,
        pp_log: &PpLog,
    ) -> Result<(), Stopped> {
        let Some((media, feed)) = &self.feed else {
            return Ok(());
        };
        let Some(index) = media.iter().position(|of| *of == medium) else {
            return Ok(());
        };
        let video = medium == Medium::Video;
        if video && self.losing && !packet.is_key() {
            return Ok(());
        }
        match feed.try_send((index, packet)) {
            Ok(()) => {
                if video {
                    self.losing = false;
                }
                Ok(())
            }
            Err(TrySendError::Full(_)) => {
                // A picture lost leaves those after it to the next keyframe;
                // a sound lost is only itself.
                if video && !self.losing {
                    pp_warn!(
                        pp_log: pp_log,
                        "{}: its source is not reading; the pictures go on from the next keyframe",
                        self.path
                    );
                    self.losing = true;
                }
                Ok(())
            }
            Err(TrySendError::Disconnected(_)) => Err(Stopped::Gone),
        }
    }
}

/// A track's codec parameters, from its configuration.
fn parameters(medium: Medium, track: &Track) -> ffmpeg::codec::Parameters {
    let mut parameters = ffmpeg::codec::Parameters::new();
    set_extradata(&mut parameters, &track.extradata);
    // SAFETY: parameters made here and owned alone; plain fields, and the
    // channel layout FFmpeg fills in itself.
    unsafe {
        let par = parameters.as_mut_ptr();
        (*par).codec_id = track.codec.into();
        match medium {
            Medium::Video => {
                (*par).codec_type = ffmpeg::media::Type::Video.into();
                if let Some((width, height)) = track.size {
                    (*par).width = width;
                    (*par).height = height;
                }
            }
            Medium::Audio => {
                (*par).codec_type = ffmpeg::media::Type::Audio.into();
                if let Some((rate, channels)) = audio_format(track.codec, &track.extradata) {
                    (*par).sample_rate = rate;
                    ffi::av_channel_layout_default(&mut (*par).ch_layout, channels);
                }
                if track.codec == Id::AAC {
                    (*par).frame_size = 1024;
                }
            }
        }
    }
    parameters
}

/// The sample rate and channels a configuration says.
fn audio_format(codec: Id, extradata: &[u8]) -> Option<(i32, i32)> {
    match codec {
        Id::AAC => {
            // AudioSpecificConfig: object type (5 bits, or 6 more after 31),
            // frequency index (4, or 24 bits of rate after 15), channels (4).
            let mut bits = BitReader::new(extradata);
            if bits.read(5)? == 31 {
                bits.read(6)?;
            }
            let rate = match bits.read(4)? {
                15 => bits.read(24)? as i32,
                index => *[
                    96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025,
                    8000, 7350,
                ]
                .get(index as usize)?,
            };
            let channels = match bits.read(4)? {
                0 => return None,
                7 => 8,
                channels => channels as i32,
            };
            Some((rate, channels))
        }
        // OpusHead: its channel count at byte 9; Opus always decodes at 48k.
        Id::OPUS => extradata
            .get(9)
            .filter(|_| extradata.starts_with(b"OpusHead"))
            .map(|&channels| (48_000, i32::from(channels.max(1)))),
        _ => None,
    }
}

struct BitReader<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }

    fn read(&mut self, count: usize) -> Option<u32> {
        let mut value = 0u32;
        for _ in 0..count {
            let byte = *self.data.get(self.position / 8)?;
            let bit = (byte >> (7 - self.position % 8)) & 1;
            value = (value << 1) | u32::from(bit);
            self.position += 1;
        }
        Some(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_audio_configuration_says_its_rate_and_channels() {
        // AAC-LC, 44.1 kHz, stereo; and 48 kHz mono.
        assert_eq!(audio_format(Id::AAC, &[0x12, 0x10]), Some((44_100, 2)));
        assert_eq!(audio_format(Id::AAC, &[0x11, 0x88]), Some((48_000, 1)));
        let mut head = b"OpusHead".to_vec();
        head.extend_from_slice(&[1, 2, 0, 0, 0x80, 0xbb, 0, 0, 0, 0, 0]);
        assert_eq!(audio_format(Id::OPUS, &head), Some((48_000, 2)));
        assert_eq!(audio_format(Id::AAC, &[]), None);
    }

    #[test]
    fn timestamps_go_on_past_their_wrap() {
        let (arrivals, _publishers) =
            crate::elements::driver::publishers::publishers("live/wrap", ());
        let mut ingest = Ingest::new(arrivals, "live/wrap");
        assert_eq!(ingest.unwrap(u32::MAX - 10), i64::from(u32::MAX - 10));
        assert_eq!(ingest.unwrap(20), (1i64 << 32) + 20);
        // A small step back is a reordered message, not a wrap.
        assert_eq!(ingest.unwrap(15), (1i64 << 32) + 15);
    }
}
