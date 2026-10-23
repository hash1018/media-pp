use std::{
    borrow::Cow,
    ffi::CString,
    path::{Path, PathBuf},
    ptr,
    time::Duration,
};

use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use super::track_sink::{Muxer, PendingStream, TrackOptions, open_tracks};
use super::tracks::{MuxerId, MuxerSinks, MuxerTrack};
use crate::elements::StreamInfo;

use crate::{
    element::ElementType,
    error::{Error, Result},
};

/// How the media playlist grows and which completed segments remain
/// referenced by it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HlsMode {
    /// A sliding live playlist. `window_size` is the maximum number of
    /// entries kept in the manifest; `delete_old_segments` also removes
    /// segment files after they fall outside that window.
    Live {
        /// Maximum number of completed media segments retained in the playlist.
        window_size: usize,
        /// Whether FFmpeg deletes segment files after they leave the window.
        delete_old_segments: bool,
    },
    /// An append-only event playlist. Every segment remains listed and the
    /// playlist receives `#EXT-X-ENDLIST` when every track finishes.
    Event,
    /// A complete video-on-demand playlist. Every segment remains listed
    /// and the playlist is finalized when every track finishes.
    Vod,
}

/// Container used for each HLS media segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HlsSegmentFormat {
    /// MPEG transport stream segments, conventionally named `*.ts`.
    MpegTs,
    /// Fragmented MP4 segments, conventionally named `*.m4s`, plus the
    /// initialization file selected by [`HlsOptions::init_filename`].
    Fmp4,
}

/// Where an [`HlsMuxer`] writes, and how it cuts and keeps segments.
///
/// [`HlsOptions::new`] gives a live fMP4 playlist; change a field to
/// differ from it.
#[derive(Debug, Clone)]
pub struct HlsOptions {
    /// Media playlist written by FFmpeg, normally ending in `.m3u8`.
    pub playlist_path: PathBuf,
    /// `printf`-style media segment path containing one integer conversion,
    /// for example `output/segment_%05d.m4s` — or, with
    /// [`HlsOptions::single_file`], the one file every segment goes into,
    /// which needs none.
    pub segment_pattern: PathBuf,
    /// Target segment duration. FFmpeg cuts on the next video keyframe, so
    /// the actual duration can be longer when keyframes are sparse.
    pub segment_duration: Duration,
    /// Playlist retention and finalization policy.
    pub mode: HlsMode,
    /// Container format used for media segments.
    pub segment_format: HlsSegmentFormat,
    /// fMP4 initialization filename written beside the playlist and used
    /// in `#EXT-X-MAP`. Ignored for [`HlsSegmentFormat::MpegTs`].
    pub init_filename: String,
    /// Optional URI prefix written before segment references in the media
    /// playlist, for example `https://cdn.example.com/live/`.
    pub base_url: Option<String>,
    /// Every segment written one after another into the one file
    /// `segment_pattern` names, the playlist listing each by its byte range
    /// in it (`#EXT-X-BYTERANGE`). An fMP4 one's initialization section is
    /// the head of that file, listed by its range too, and
    /// [`HlsOptions::init_filename`] goes unused.
    ///
    /// For a recording kept for days: short segments, for seeking and for
    /// following it while it is written, without a file for each — an hour
    /// of four-second segments is one file rather than nine hundred. A
    /// server answers each segment with a range request on that file. The
    /// file is written under its own name as it grows, the playlist
    /// replaced whole after each segment as ever. Not with
    /// [`HlsMode::Live`]'s deletion, which removes whole files.
    pub single_file: bool,
}

impl HlsOptions {
    /// Creates a live fMP4 configuration with two-second segments, a
    /// six-segment sliding window, and automatic deletion of old segments.
    pub fn new(playlist_path: impl Into<PathBuf>, segment_pattern: impl Into<PathBuf>) -> Self {
        crate::ensure_ffmpeg();
        Self {
            playlist_path: playlist_path.into(),
            segment_pattern: segment_pattern.into(),
            segment_duration: Duration::from_secs(2),
            mode: HlsMode::Live {
                window_size: 6,
                delete_old_segments: true,
            },
            segment_format: HlsSegmentFormat::Fmp4,
            init_filename: "init.mp4".into(),
            base_url: None,
            single_file: false,
        }
    }

    fn validate(&self) -> std::result::Result<(), HlsMuxerError> {
        if self.segment_duration.is_zero() {
            return Err(HlsMuxerError::ZeroSegmentDuration);
        }
        if let HlsMode::Live { window_size, .. } = self.mode
            && (window_size == 0 || window_size > i32::MAX as usize)
        {
            return Err(HlsMuxerError::InvalidWindowSize(window_size));
        }
        if self.segment_format == HlsSegmentFormat::Fmp4 && self.init_filename.is_empty() {
            return Err(HlsMuxerError::EmptyInitFilename);
        }

        let pattern = path_as_utf8(&self.segment_pattern, "segment_pattern")?;
        if !self.single_file && !has_integer_conversion(pattern) {
            return Err(HlsMuxerError::MissingSegmentIndex(
                self.segment_pattern.clone(),
            ));
        }
        path_as_utf8(&self.playlist_path, "playlist_path")?;
        reject_nul(pattern, "segment_pattern")?;
        reject_nul(&self.init_filename, "init_filename")?;
        if let Some(base_url) = &self.base_url {
            reject_nul(base_url, "base_url")?;
        }
        Ok(())
    }

    fn header_options(&self) -> std::result::Result<ffmpeg::Dictionary<'static>, HlsMuxerError> {
        let mut options = ffmpeg::Dictionary::new();
        options.set("hls_time", &self.segment_duration.as_secs_f64().to_string());
        let segment_pattern = path_for_ffmpeg(&self.segment_pattern, "segment_pattern")?;
        options.set("hls_segment_filename", &segment_pattern);

        let mut flags = vec!["independent_segments"];
        if self.single_file {
            // Not with `temp_file`, which keeps the one file under a `.tmp`
            // name — the name the playlist then gives — until the end.
            flags.push("single_file");
        } else {
            flags.push("temp_file");
        }
        match self.mode {
            HlsMode::Live {
                window_size,
                delete_old_segments,
            } => {
                options.set("hls_list_size", &window_size.to_string());
                if delete_old_segments {
                    flags.push("delete_segments");
                }
            }
            HlsMode::Event => {
                options.set("hls_playlist_type", "event");
                options.set("hls_list_size", "0");
            }
            HlsMode::Vod => {
                options.set("hls_playlist_type", "vod");
                options.set("hls_list_size", "0");
            }
        }

        match self.segment_format {
            HlsSegmentFormat::MpegTs => options.set("hls_segment_type", "mpegts"),
            HlsSegmentFormat::Fmp4 => {
                options.set("hls_segment_type", "fmp4");
                options.set("hls_fmp4_init_filename", &self.init_filename);
            }
        }
        options.set("hls_flags", &flags.join("+"));
        if let Some(base_url) = &self.base_url {
            options.set("hls_base_url", base_url);
        }
        Ok(options)
    }
}

fn path_as_utf8<'a>(
    path: &'a Path,
    field: &'static str,
) -> std::result::Result<&'a str, HlsMuxerError> {
    path.to_str().ok_or_else(|| HlsMuxerError::NonUtf8Path {
        field,
        path: path.to_path_buf(),
    })
}

/// FFmpeg's HLS path splitting recognizes `/` on Windows. Native `\`
/// separators otherwise make a relative fMP4 init filename land in the
/// process working directory instead of beside the playlist.
fn path_for_ffmpeg<'a>(
    path: &'a Path,
    field: &'static str,
) -> std::result::Result<Cow<'a, str>, HlsMuxerError> {
    let path = path_as_utf8(path, field)?;
    #[cfg(windows)]
    {
        Ok(Cow::Owned(path.replace('\\', "/")))
    }
    #[cfg(not(windows))]
    {
        Ok(Cow::Borrowed(path))
    }
}

fn reject_nul(value: &str, field: &'static str) -> std::result::Result<(), HlsMuxerError> {
    if value.contains('\0') {
        Err(HlsMuxerError::EmbeddedNul { field })
    } else {
        Ok(())
    }
}

/// Allocates the `AVFMT_NOFILE` HLS muxer without opening `playlist_path`
/// as a normal `AVIOContext`. HLS owns that file and atomically replaces
/// it; pre-opening it prevents the rename on Windows.
fn allocate_output(
    options: &HlsOptions,
) -> std::result::Result<ffmpeg::format::context::Output, HlsMuxerError> {
    let path = path_for_ffmpeg(&options.playlist_path, "playlist_path")?;
    let path = CString::new(path.as_ref()).map_err(|_| HlsMuxerError::EmbeddedNul {
        field: "playlist_path",
    })?;
    let format = CString::new("hls").expect("static HLS format name contains no NUL");
    let mut context = ptr::null_mut();
    // SAFETY: `format` and `path` are live NUL-terminated `CString`s, and
    // `context` is a live local FFmpeg writes the allocated context into. A
    // failure leaves it null, which is checked before it is used.
    let result = unsafe {
        ffmpeg::ffi::avformat_alloc_output_context2(
            &mut context,
            ptr::null_mut(),
            format.as_ptr(),
            path.as_ptr(),
        )
    };
    if result < 0 {
        return Err(HlsMuxerError::Ffmpeg(ffmpeg::Error::from(result)));
    }
    if context.is_null() {
        return Err(HlsMuxerError::Ffmpeg(ffmpeg::Error::Unknown));
    }
    // SAFETY: `context` is non-null — checked just above — and was allocated by
    // `avformat_alloc_output_context2` and never handed anywhere else, so
    // `Output::wrap` is taking sole ownership as it requires.
    Ok(unsafe { ffmpeg::format::context::Output::wrap(context) })
}

/// Accepts `%d`, `%03d`, and the other width variants understood by the
/// HLS muxer's `printf`-style filename expansion. `%%` is a literal `%`.
fn has_integer_conversion(pattern: &str) -> bool {
    let bytes = pattern.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'%' {
            index += 1;
            continue;
        }
        index += 1;
        if index < bytes.len() && bytes[index] == b'%' {
            index += 1;
            continue;
        }
        while index < bytes.len() && (bytes[index] == b'0' || bytes[index].is_ascii_digit()) {
            index += 1;
        }
        if index < bytes.len() && bytes[index] == b'd' {
            return true;
        }
    }
    false
}

/// What can go wrong configuring or writing an [`HlsMuxer`].
#[derive(Debug, ThisError)]
pub enum HlsMuxerError {
    /// A stream sink received a buffer other than a packet or end-of-stream.
    #[error("HlsMuxer stream sinks only accept Packet buffers, got {0}")]
    UnsupportedBuffer(&'static str),

    /// [`HlsOptions::segment_duration`] is zero.
    #[error("HLS segment duration must be greater than zero")]
    ZeroSegmentDuration,

    /// A live playlist window cannot be represented by FFmpeg.
    #[error(
        "HLS live window size must be between 1 and {max}, got {0}",
        max = i32::MAX
    )]
    InvalidWindowSize(usize),

    /// fMP4 output was selected without an initialization filename.
    #[error("HLS fMP4 init filename must not be empty")]
    EmptyInitFilename,

    /// The segment pattern has no integer conversion for unique filenames.
    #[error("HLS segment pattern must contain an integer conversion such as %05d: {0:?}")]
    MissingSegmentIndex(PathBuf),

    /// A path option cannot be passed through FFmpeg's UTF-8 API.
    #[error("HLS {field} must be valid UTF-8: {path:?}")]
    NonUtf8Path {
        /// Name of the invalid option field.
        field: &'static str,
        /// Non-UTF-8 path supplied by the caller.
        path: PathBuf,
    },

    /// A string option contains an interior NUL byte rejected by FFmpeg's C API.
    #[error("HLS {field} must not contain a NUL byte")]
    EmbeddedNul {
        /// Name of the invalid option field.
        field: &'static str,
    },

    /// The muxer was opened before any media stream was registered.
    #[error("HLS muxer requires at least one stream")]
    NoStreams,

    /// FFmpeg rejected muxer creation, packet writing, or finalization.
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),
}

/// Writes encoded tracks as an HLS playlist and its segment files, for
/// players and browsers to stream over plain HTTP.
///
/// It takes packets, one sink per track, and encodes nothing. [`HlsOptions`]
/// says where the playlist and segments go, how long a segment is, and
/// whether the playlist is a sliding live window, a growing event or a
/// finished video-on-demand one ([`HlsMode`]). Serving the files is left to
/// any web server. Each sink ends a chain behind its encoder, as in
/// [`FileMuxer`](crate::elements::FileMuxer)'s example.
///
/// # Example
///
/// ```no_run
/// # use media_pp::ffmpeg;
/// # use media_pp::elements::{SwEncoder, SwEncoderOptions, VideoCodec};
/// use media_pp::elements::{HlsMuxer, HlsOptions};
/// # fn main() -> media_pp::Result<()> {
/// # let encoder = SwEncoder::new("encoder", SwEncoderOptions {
/// #     codec: VideoCodec::OpenH264,
/// #     width: 640,
/// #     height: 360,
/// #     pixel_format: ffmpeg::format::Pixel::YUV420P,
/// #     frame_rate: ffmpeg::Rational(30, 1),
/// #     bit_rate: 2_000_000,
/// #     gop_size: 60,
/// #     max_b_frames: None,
/// # })?;
/// // A live fMP4 playlist: two-second segments, the last six kept.
/// let options = HlsOptions::new("hls/stream.m3u8", "hls/segment_%05d.m4s");
/// let mut muxer = HlsMuxer::create(options)?;
/// let video = muxer.add_stream("video", &encoder)?;
/// let mut sinks = muxer.open()?; // writes the header
/// let video_sink = sinks.take(video)?; // ends the chain behind `encoder`
/// # Ok(())
/// # }
/// ```
///
/// # How it works
///
/// It builds one HLS media playlist and returns one
/// [`RawSink`](crate::element::RawSink) per registered track. FFmpeg owns
/// segment boundary selection, fMP4/MPEG-TS creation, atomic playlist
/// replacement, live-window trimming, and final `#EXT-X-ENDLIST` generation.
///
/// This deliberately has the same two-phase shape as
/// [`crate::elements::FileMuxer`]: call [`HlsMuxer::add_stream`] for every
/// track before [`HlsMuxer::open`] writes the header. The returned sinks
/// share one output lock and finalize the playlist only after every track
/// reports `Eos` or a stop.
pub struct HlsMuxer {
    id: MuxerId,
    output: ffmpeg::format::context::Output,
    streams: Vec<PendingStream>,
    options: HlsOptions,
}

impl HlsMuxer {
    /// Whether segments in `format` hold `track`: in fMP4, what FFmpeg's MP4
    /// muxer says it writes, which [`Self::add_stream`] of anything else is
    /// refused by when the muxer opens; in TS, the codecs FFmpeg's MPEG-TS
    /// muxer gives a stream type of their own — H.264, H.265, MPEG-1, -2
    /// and -4 pictures, AAC, MP2, MP3, AC-3, E-AC-3 and Opus sound — since
    /// it writes anything else as data no player reads. Which of them an
    /// HLS player plays is the player's: AAC is the sound every one does.
    pub fn carries(track: &StreamInfo, format: HlsSegmentFormat) -> bool {
        use ffmpeg::codec::Id;
        if format == HlsSegmentFormat::MpegTs {
            return matches!(
                track.parameters().id(),
                Id::H264
                    | Id::HEVC
                    | Id::MPEG1VIDEO
                    | Id::MPEG2VIDEO
                    | Id::MPEG4
                    | Id::AAC
                    | Id::MP2
                    | Id::MP3
                    | Id::AC3
                    | Id::EAC3
                    | Id::OPUS
            );
        }
        crate::ensure_ffmpeg();
        // SAFETY: a lookup by name, and a query of the format it found —
        // both FFmpeg's own static data; a null format is not queried.
        unsafe {
            let muxer =
                ffmpeg::ffi::av_guess_format(c"mp4".as_ptr(), std::ptr::null(), std::ptr::null());
            !muxer.is_null()
                && ffmpeg::ffi::avformat_query_codec(
                    muxer,
                    track.parameters().id().into(),
                    ffmpeg::ffi::FF_COMPLIANCE_NORMAL,
                ) == 1
        }
    }

    /// Allocates FFmpeg's HLS output context. Parent directories for the
    /// playlist, segment pattern, and fMP4 init file must already exist.
    pub fn create(options: HlsOptions) -> Result<Self> {
        crate::ensure_ffmpeg();
        options.validate()?;
        let output = allocate_output(&options)?;
        Ok(Self {
            id: MuxerId::next(),
            output,
            streams: Vec::new(),
            options,
        })
    }

    /// Registers one encoded packet stream, described by `format` (see
    /// [`StreamInfo`]). Its time base must match the timestamps carried by
    /// packets arriving at the returned track sink.
    ///
    /// The returned [`MuxerTrack`] is how this track's sink is taken out of
    /// what [`HlsMuxer::open`] returns.
    pub fn add_stream(
        &mut self,
        name: impl Into<String>,
        format: impl Into<StreamInfo>,
    ) -> Result<MuxerTrack> {
        let format = format.into();
        let time_base = format.time_base();
        let parameters = format.into_parameters();
        let mut stream = self
            .output
            .add_stream(parameters.id())
            .map_err(HlsMuxerError::from)?;
        let pending = PendingStream::new(name, &parameters, time_base);
        stream.set_time_base(time_base);
        stream.set_parameters(parameters);
        let track = self.id.track(self.streams.len());
        self.streams.push(pending);
        Ok(track)
    }

    /// Writes the HLS header and returns one sink per stream, each taken
    /// out by the [`MuxerTrack`] its [`HlsMuxer::add_stream`] returned. The
    /// playlist is finalized only after every returned sink has received
    /// `Eos` or a stop.
    pub fn open(mut self) -> Result<MuxerSinks> {
        crate::ensure_ffmpeg();
        if self.streams.is_empty() {
            return Err(HlsMuxerError::NoStreams.into());
        }
        let options = self.options.header_options()?;
        let unused = self
            .output
            .write_header_with(options)
            .map_err(HlsMuxerError::from)?;
        drop(unused);
        Ok(open_tracks::<Self>(
            self.id,
            self.output,
            self.streams,
            TrackOptions {
                finish_on_stop: true,
                destination: None,
                timeline: None,
                prune: None,
                keeps_parameter_sets: self.options.segment_format == HlsSegmentFormat::Fmp4,
            },
        ))
    }
}

impl Muxer for HlsMuxer {
    const ELEMENT_TYPE: ElementType = ElementType::HlsMuxer;

    fn ffmpeg_error(error: ffmpeg::Error) -> Error {
        HlsMuxerError::Ffmpeg(error).into()
    }

    fn unsupported_buffer(kind: &'static str) -> Error {
        HlsMuxerError::UnsupportedBuffer(kind).into()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use crate::{
        buffer::MediaBuffer,
        element::{RawSink, SrcPads},
        elements::{AudioCodec, SwAudioEncoder, SwAudioEncoderOptions},
    };

    fn open_aac_encoder(sample_rate: u32, channels: u16) -> SwAudioEncoder {
        SwAudioEncoder::new(
            "encoder",
            SwAudioEncoderOptions {
                codec: AudioCodec::Aac,
                sample_rate,
                channels,
                bit_rate: 64_000,
            },
        )
        .expect("aac encoder must be available")
    }

    fn silent_frame(
        sample_rate: u32,
        channels: u16,
        samples: usize,
        pts: i64,
    ) -> ffmpeg::frame::Audio {
        let mut frame = ffmpeg::frame::Audio::new(
            ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed),
            samples,
            ffmpeg::ChannelLayout::default(channels as i32),
        );
        frame.set_rate(sample_rate);
        frame.set_pts(Some(pts));
        frame.data_mut(0).fill(0);
        frame
    }

    fn unique_test_dir(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("media_pp_{label}_{}_{}", std::process::id(), nonce))
    }

    fn encode_silence(options: HlsOptions, ticks: i64) {
        let mut encoder = open_aac_encoder(48_000, 1);
        let mut muxer = HlsMuxer::create(options).expect("HLS muxer must open");
        let audio = muxer
            .add_stream(
                "audio",
                StreamInfo::new(encoder.parameters(), ffmpeg::Rational::new(1, 48_000)),
            )
            .expect("add_stream must succeed");
        let mut sinks = muxer.open().expect("HLS header must be written");
        encoder.src_pads()[0].link(sinks.take(audio).expect("the muxer's own track").into_raw());

        for tick in 0..ticks {
            encoder
                .consume(MediaBuffer::Audio(
                    Arc::new(silent_frame(48_000, 1, 960, tick * 960)).into(),
                ))
                .expect("encoding and muxing must succeed");
        }
        crate::stream::deliver(&mut encoder, &crate::stream::StreamEvent::Eos)
            .expect("EOS must finalize the HLS playlist");
    }

    #[test]
    fn segment_pattern_requires_an_integer_conversion() {
        assert!(has_integer_conversion("segment_%d.m4s"));
        assert!(has_integer_conversion("segment_%05d.m4s"));
        assert!(!has_integer_conversion("segment_%%05d.m4s"));
        assert!(!has_integer_conversion("segment.m4s"));
    }

    #[test]
    fn writes_a_playable_fmp4_vod_playlist() {
        let dir = unique_test_dir("hls_vod");
        std::fs::create_dir_all(&dir).unwrap();
        let playlist_path = dir.join("index.m3u8");
        let mut options = HlsOptions::new(&playlist_path, dir.join("segment_%03d.m4s"));
        options.segment_duration = Duration::from_secs(1);
        options.mode = HlsMode::Vod;

        encode_silence(options, 80);

        let playlist = std::fs::read_to_string(&playlist_path).unwrap();
        assert!(playlist.starts_with("#EXTM3U"));
        assert!(playlist.contains("#EXT-X-PLAYLIST-TYPE:VOD"));
        assert!(playlist.contains("#EXT-X-MAP:URI=\"init.mp4\""));
        assert!(playlist.contains("#EXT-X-ENDLIST"));

        let segment_uris: Vec<_> = playlist
            .lines()
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .collect();
        assert!(
            segment_uris.len() >= 2,
            "expected multiple media segments, got {segment_uris:?}\n{playlist}"
        );
        assert!(dir.join("init.mp4").metadata().unwrap().len() > 0);
        for uri in segment_uris {
            let path = PathBuf::from(uri);
            let path = if path.is_absolute() {
                path
            } else {
                dir.join(path)
            };
            assert!(
                path.metadata().unwrap().len() > 0,
                "empty segment: {path:?}"
            );
        }

        let mut input = ffmpeg::format::input(&playlist_path)
            .expect("FFmpeg must be able to read the generated playlist");
        assert_eq!(input.streams().count(), 1);
        let mut packet = ffmpeg::Packet::empty();
        packet
            .read(&mut input)
            .expect("the generated playlist must contain media packets");
        drop(input);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A recording's playlist: every segment in one fMP4 file, each listed
    /// by its byte range, the event playlist ended as the stream ends — and
    /// read back by FFmpeg through those ranges.
    #[test]
    fn writes_every_segment_into_one_file_by_byte_range() {
        let dir = unique_test_dir("hls_single_file");
        std::fs::create_dir_all(&dir).unwrap();
        let playlist_path = dir.join("index.m3u8");
        let mut options = HlsOptions::new(&playlist_path, dir.join("media.m4s"));
        options.segment_duration = Duration::from_secs(1);
        options.mode = HlsMode::Event;
        options.single_file = true;

        encode_silence(options, 150);

        let playlist = std::fs::read_to_string(&playlist_path).unwrap();
        assert!(
            playlist.contains("#EXT-X-PLAYLIST-TYPE:EVENT"),
            "{playlist}"
        );
        assert!(
            playlist.contains("#EXT-X-MAP:URI=\"media.m4s\",BYTERANGE="),
            "the initialization section heads the one file:\n{playlist}"
        );
        assert!(playlist.contains("#EXT-X-ENDLIST"), "{playlist}");
        let ranges = playlist
            .lines()
            .filter(|line| line.starts_with("#EXT-X-BYTERANGE:"))
            .count();
        assert!(ranges >= 2, "a range for each segment:\n{playlist}");
        let uris: Vec<_> = playlist
            .lines()
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .collect();
        assert_eq!(uris.len(), ranges, "{playlist}");
        assert!(uris.iter().all(|uri| *uri == "media.m4s"), "{uris:?}");
        let files: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        assert_eq!(files.len(), 2, "the playlist and media alone: {files:?}");

        let mut input = ffmpeg::format::input(&playlist_path)
            .expect("FFmpeg must be able to read the byte-range playlist");
        let mut packets = 0;
        let mut packet = ffmpeg::Packet::empty();
        while packet.read(&mut input).is_ok() {
            packets += 1;
        }
        assert!(packets > 100, "read {packets} packets back");
        drop(input);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_single_file_needs_no_index_in_its_name() {
        let mut options = HlsOptions::new("index.m3u8", "media.m4s");
        assert!(matches!(
            options.validate(),
            Err(HlsMuxerError::MissingSegmentIndex(_))
        ));
        options.single_file = true;
        assert!(options.validate().is_ok());
    }

    #[test]
    fn writes_a_playable_mpegts_vod_playlist() {
        let dir = unique_test_dir("hls_mpegts");
        std::fs::create_dir_all(&dir).unwrap();
        let playlist_path = dir.join("index.m3u8");
        let mut options = HlsOptions::new(&playlist_path, dir.join("segment_%03d.ts"));
        options.segment_duration = Duration::from_secs(1);
        options.mode = HlsMode::Vod;
        options.segment_format = HlsSegmentFormat::MpegTs;

        encode_silence(options, 80);

        let playlist = std::fs::read_to_string(&playlist_path).unwrap();
        assert!(playlist.contains("#EXT-X-PLAYLIST-TYPE:VOD"));
        assert!(!playlist.contains("#EXT-X-MAP"));
        assert!(playlist.contains("#EXT-X-ENDLIST"));
        assert!(
            playlist
                .lines()
                .filter(|line| !line.starts_with('#'))
                .any(|line| line.ends_with(".ts"))
        );

        let mut input = ffmpeg::format::input(&playlist_path)
            .expect("FFmpeg must be able to read the generated MPEG-TS playlist");
        assert_eq!(input.streams().count(), 1);
        let mut packet = ffmpeg::Packet::empty();
        packet.read(&mut input).unwrap();
        drop(input);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn live_playlist_keeps_its_window_and_deletes_old_segments() {
        let dir = unique_test_dir("hls_live");
        std::fs::create_dir_all(&dir).unwrap();
        let playlist_path = dir.join("index.m3u8");
        let mut options = HlsOptions::new(&playlist_path, dir.join("segment_%03d.m4s"));
        options.segment_duration = Duration::from_secs(1);
        options.mode = HlsMode::Live {
            window_size: 2,
            delete_old_segments: true,
        };

        encode_silence(options, 200);

        let playlist = std::fs::read_to_string(&playlist_path).unwrap();
        let segment_uris: Vec<_> = playlist
            .lines()
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .collect();
        assert_eq!(segment_uris.len(), 2, "{playlist}");
        let media_sequence = playlist
            .lines()
            .find_map(|line| line.strip_prefix("#EXT-X-MEDIA-SEQUENCE:"))
            .unwrap()
            .parse::<u64>()
            .unwrap();
        assert!(media_sequence > 0, "{playlist}");
        assert!(playlist.contains("#EXT-X-ENDLIST"));

        let media_files = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "m4s"))
            .count();
        assert!(
            (2..=3).contains(&media_files),
            "the two listed segments plus at most one deletion-threshold file should remain; \
             found {media_files}"
        );
        assert!(
            std::fs::read_dir(&dir)
                .unwrap()
                .filter_map(|entry| entry.ok())
                .all(|entry| entry.path().extension().is_none_or(|ext| ext != "tmp")),
            "atomic temp files must not remain after finalization"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn playlist_finalizes_only_after_every_track_finishes() {
        let dir = unique_test_dir("hls_tracks");
        std::fs::create_dir_all(&dir).unwrap();
        let playlist_path = dir.join("index.m3u8");
        let mut options = HlsOptions::new(&playlist_path, dir.join("segment_%03d.m4s"));
        options.segment_duration = Duration::from_secs(1);
        options.mode = HlsMode::Vod;

        let mut encoder_a = open_aac_encoder(48_000, 1);
        let mut encoder_b = open_aac_encoder(48_000, 1);
        let mut muxer = HlsMuxer::create(options).unwrap();
        let a = muxer
            .add_stream(
                "audio-a",
                StreamInfo::new(encoder_a.parameters(), ffmpeg::Rational::new(1, 48_000)),
            )
            .unwrap();
        let b = muxer
            .add_stream(
                "audio-b",
                StreamInfo::new(encoder_b.parameters(), ffmpeg::Rational::new(1, 48_000)),
            )
            .unwrap();
        let mut sinks = muxer.open().unwrap();
        let sink_b = sinks.take(b).unwrap();
        let sink_a = sinks.take(a).unwrap();
        encoder_a.src_pads()[0].link(sink_a.into_raw());
        encoder_b.src_pads()[0].link(sink_b.into_raw());

        for tick in 0..60i64 {
            encoder_a
                .consume(MediaBuffer::Audio(
                    Arc::new(silent_frame(48_000, 1, 960, tick * 960)).into(),
                ))
                .unwrap();
            encoder_b
                .consume(MediaBuffer::Audio(
                    Arc::new(silent_frame(48_000, 1, 960, tick * 960)).into(),
                ))
                .unwrap();
        }

        crate::stream::deliver(&mut encoder_a, &crate::stream::StreamEvent::Eos).unwrap();
        if let Ok(unfinished) = std::fs::read_to_string(&playlist_path) {
            assert!(
                !unfinished.contains("#EXT-X-ENDLIST"),
                "the first finished track must not finalize the shared playlist"
            );
        }

        for tick in 60..80i64 {
            encoder_b
                .consume(MediaBuffer::Audio(
                    Arc::new(silent_frame(48_000, 1, 960, tick * 960)).into(),
                ))
                .unwrap();
        }
        crate::stream::deliver(&mut encoder_b, &crate::stream::StreamEvent::Eos).unwrap();
        let finished = std::fs::read_to_string(&playlist_path).unwrap();
        assert!(finished.contains("#EXT-X-ENDLIST"));

        drop(encoder_a);
        drop(encoder_b);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Each container's own say: fMP4 holds Opus and not G.711, TS holds
    /// AAC and not VP8.
    #[test]
    fn what_a_segment_holds_is_its_containers_say() {
        use ffmpeg::codec::Id;
        let unit = ffmpeg::Rational::new(1, 90_000);
        let video = |codec| StreamInfo::video(codec, &[], (320, 240), unit).unwrap();
        let audio = |codec| StreamInfo::audio(codec, &[], 48_000, 2, unit).unwrap();
        let fmp4 = HlsSegmentFormat::Fmp4;
        let ts = HlsSegmentFormat::MpegTs;
        for (track, format) in [
            (video(Id::H264), fmp4),
            (video(Id::HEVC), fmp4),
            (audio(Id::AAC), fmp4),
            (audio(Id::OPUS), fmp4),
            (video(Id::H264), ts),
            (audio(Id::AAC), ts),
        ] {
            assert!(
                HlsMuxer::carries(&track, format),
                "{:?} in {format:?}",
                track.parameters().id()
            );
        }
        for (track, format) in [(audio(Id::PCM_MULAW), fmp4), (video(Id::VP8), ts)] {
            assert!(
                !HlsMuxer::carries(&track, format),
                "{:?} in {format:?}",
                track.parameters().id()
            );
        }
    }
}
