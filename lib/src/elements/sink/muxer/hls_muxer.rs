use std::{
    borrow::Cow,
    ffi::CString,
    path::{Path, PathBuf},
    ptr,
    sync::{Arc, Mutex},
    time::Duration,
};

use crate::clog::{CLog, cerror};
use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Element, ElementType, Sink, element_clog},
    error::Result,
};

/// How the media playlist grows and which completed segments remain
/// referenced by it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HlsMode {
    /// A sliding live playlist. `window_size` is the maximum number of
    /// entries kept in the manifest; `delete_old_segments` also removes
    /// segment files after they fall outside that window.
    Live {
        window_size: usize,
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

/// Construction-time options for [`HlsMuxer`].
#[derive(Debug, Clone)]
pub struct HlsOptions {
    /// Media playlist written by FFmpeg, normally ending in `.m3u8`.
    pub playlist_path: PathBuf,
    /// `printf`-style media segment path containing one integer conversion,
    /// for example `output/segment_%05d.m4s`.
    pub segment_pattern: PathBuf,
    /// Target segment duration. FFmpeg cuts on the next video keyframe, so
    /// the actual duration can be longer when keyframes are sparse.
    pub segment_duration: Duration,
    pub mode: HlsMode,
    pub segment_format: HlsSegmentFormat,
    /// fMP4 initialization filename written beside the playlist and used
    /// in `#EXT-X-MAP`. Ignored for [`HlsSegmentFormat::MpegTs`].
    pub init_filename: String,
    /// Optional URI prefix written before segment references in the media
    /// playlist, for example `https://cdn.example.com/live/`.
    pub base_url: Option<String>,
}

impl HlsOptions {
    /// Creates a live fMP4 configuration with two-second segments, a
    /// six-segment sliding window, and automatic deletion of old segments.
    pub fn new(playlist_path: impl Into<PathBuf>, segment_pattern: impl Into<PathBuf>) -> Self {
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
        if !has_integer_conversion(pattern) {
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

        let mut flags = vec!["temp_file", "independent_segments"];
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

/// Errors specific to [`HlsMuxer`].
#[derive(Debug, ThisError)]
pub enum HlsMuxerError {
    #[error("HlsMuxer stream sinks only accept Packet or Eos buffers, got {0}")]
    UnsupportedBuffer(&'static str),

    #[error("HLS segment duration must be greater than zero")]
    ZeroSegmentDuration,

    #[error(
        "HLS live window size must be between 1 and {max}, got {0}",
        max = i32::MAX
    )]
    InvalidWindowSize(usize),

    #[error("HLS fMP4 init filename must not be empty")]
    EmptyInitFilename,

    #[error("HLS segment pattern must contain an integer conversion such as %05d: {0:?}")]
    MissingSegmentIndex(PathBuf),

    #[error("HLS {field} must be valid UTF-8: {path:?}")]
    NonUtf8Path { field: &'static str, path: PathBuf },

    #[error("HLS {field} must not contain a NUL byte")]
    EmbeddedNul { field: &'static str },

    #[error("HLS muxer requires at least one stream")]
    NoStreams,

    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),
}

struct PendingStream {
    name: Arc<str>,
    input_time_base: ffmpeg::Rational,
}

/// Builds one HLS media playlist and returns one [`Sink`] per registered
/// track. FFmpeg owns segment boundary selection, fMP4/MPEG-TS creation,
/// atomic playlist replacement, live-window trimming, and final
/// `#EXT-X-ENDLIST` generation.
///
/// This deliberately has the same two-phase shape as
/// [`crate::elements::Mp4Muxer`]: call [`HlsMuxer::add_stream`] for every
/// track before [`HlsMuxer::open`] writes the header. The returned sinks
/// share one output lock and finalize the playlist only after every track
/// reports `Eos` or [`ControlMsg::Stop`].
pub struct HlsMuxer {
    output: ffmpeg::format::context::Output,
    streams: Vec<PendingStream>,
    options: HlsOptions,
}

impl HlsMuxer {
    /// Allocates FFmpeg's HLS output context. Parent directories for the
    /// playlist, segment pattern, and fMP4 init file must already exist.
    pub fn create(options: HlsOptions) -> Result<Self> {
        options.validate()?;
        let output = allocate_output(&options)?;
        Ok(Self {
            output,
            streams: Vec::new(),
            options,
        })
    }

    /// Registers one encoded packet stream. `time_base` must match the
    /// timestamps carried by packets arriving at the returned track sink.
    pub fn add_stream(
        &mut self,
        name: impl Into<String>,
        parameters: ffmpeg::codec::Parameters,
        time_base: ffmpeg::Rational,
    ) -> Result<()> {
        let mut stream = self
            .output
            .add_stream(parameters.id())
            .map_err(HlsMuxerError::from)?;
        stream.set_time_base(time_base);
        stream.set_parameters(parameters);
        self.streams.push(PendingStream {
            name: name.into().into(),
            input_time_base: time_base,
        });
        Ok(())
    }

    /// Writes the HLS header and returns one sink per stream, in registration
    /// order. The playlist is finalized only after every returned sink has
    /// received `Eos` or [`ControlMsg::Stop`].
    pub fn open(mut self) -> Result<Vec<Box<dyn Sink>>> {
        if self.streams.is_empty() {
            return Err(HlsMuxerError::NoStreams.into());
        }
        let options = self.options.header_options()?;
        let unused = self
            .output
            .write_header_with(options)
            .map_err(HlsMuxerError::from)?;
        drop(unused);
        let total = self.streams.len();
        let shared = Arc::new(HlsMuxerShared {
            state: Mutex::new(MuxerState {
                output: self.output,
                done: 0,
                finished: false,
            }),
            total,
        });
        Ok(self
            .streams
            .into_iter()
            .enumerate()
            .map(|(index, stream)| -> Box<dyn Sink> {
                Box::new(HlsMuxerStreamSink {
                    clog: element_clog(ElementType::HlsMuxer, &stream.name, None),
                    name: stream.name,
                    shared: shared.clone(),
                    stream_index: index,
                    input_time_base: stream.input_time_base,
                    done: false,
                })
            })
            .collect())
    }
}

struct MuxerState {
    output: ffmpeg::format::context::Output,
    done: usize,
    finished: bool,
}

struct HlsMuxerShared {
    state: Mutex<MuxerState>,
    total: usize,
}

impl HlsMuxerShared {
    fn write_packet(
        &self,
        stream_index: usize,
        input_time_base: ffmpeg::Rational,
        packet: &ffmpeg::Packet,
    ) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        if state.finished {
            return Ok(());
        }
        let mut packet = packet.clone();
        let output_time_base = state
            .output
            .stream(stream_index)
            .expect("stream was added in HlsMuxer::add_stream")
            .time_base();
        packet.rescale_ts(input_time_base, output_time_base);
        packet.set_stream(stream_index);
        packet.set_position(-1);
        packet
            .write_interleaved(&mut state.output)
            .map_err(HlsMuxerError::from)?;
        Ok(())
    }

    fn finish_track(&self) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        state.done += 1;
        if state.finished || state.done < self.total {
            return Ok(());
        }
        state.finished = true;
        state.output.write_trailer().map_err(HlsMuxerError::from)?;
        Ok(())
    }
}

/// One registered HLS track's sink. Instances returned by the same
/// [`HlsMuxer::open`] share their muxer and finalization state.
pub struct HlsMuxerStreamSink {
    clog: CLog,
    name: Arc<str>,
    shared: Arc<HlsMuxerShared>,
    stream_index: usize,
    input_time_base: ffmpeg::Rational,
    done: bool,
}

impl HlsMuxerStreamSink {
    fn finish(&mut self) -> Result<()> {
        if self.done {
            return Ok(());
        }
        self.done = true;
        self.shared
            .finish_track()
            .inspect_err(|error| cerror!(self, "write_trailer failed: {error}"))
    }
}

impl Element for HlsMuxerStreamSink {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::HlsMuxer
    }

    fn clog(&self) -> &CLog {
        &self.clog
    }

    fn clog_mut(&mut self) -> &mut CLog {
        &mut self.clog
    }
}

impl Sink for HlsMuxerStreamSink {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Packet(packet) => self
                .shared
                .write_packet(self.stream_index, self.input_time_base, &packet)
                .inspect_err(|error| cerror!(self, "write_interleaved failed: {error}")),
            MediaBuffer::Eos => self.finish(),
            other => Err(HlsMuxerError::UnsupportedBuffer(other.kind()).into()),
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        if msg == ControlMsg::Stop {
            self.finish()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use crate::{
        element::Source,
        elements::{AudioCodec, SwAudioEncoder, SwAudioEncoderOptions},
    };

    fn open_aac_encoder(sample_rate: u32, channels: u16) -> SwAudioEncoder {
        SwAudioEncoder::new(
            "encoder",
            SwAudioEncoderOptions {
                codec: AudioCodec::Aac,
                sample_rate,
                channels,
                time_base: ffmpeg::Rational::new(1, sample_rate as i32),
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
        muxer
            .add_stream(
                "audio",
                encoder.parameters(),
                ffmpeg::Rational::new(1, 48_000),
            )
            .expect("add_stream must succeed");
        let mut sinks = muxer.open().expect("HLS header must be written");
        encoder.src_pads()[0].link(sinks.pop().unwrap());

        for tick in 0..ticks {
            encoder
                .consume(MediaBuffer::Audio(Arc::new(silent_frame(
                    48_000,
                    1,
                    960,
                    tick * 960,
                ))))
                .expect("encoding and muxing must succeed");
        }
        encoder
            .consume(MediaBuffer::Eos)
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
        muxer
            .add_stream(
                "audio-a",
                encoder_a.parameters(),
                ffmpeg::Rational::new(1, 48_000),
            )
            .unwrap();
        muxer
            .add_stream(
                "audio-b",
                encoder_b.parameters(),
                ffmpeg::Rational::new(1, 48_000),
            )
            .unwrap();
        let mut sinks = muxer.open().unwrap();
        let sink_b = sinks.pop().unwrap();
        let sink_a = sinks.pop().unwrap();
        encoder_a.src_pads()[0].link(sink_a);
        encoder_b.src_pads()[0].link(sink_b);

        for tick in 0..60i64 {
            encoder_a
                .consume(MediaBuffer::Audio(Arc::new(silent_frame(
                    48_000,
                    1,
                    960,
                    tick * 960,
                ))))
                .unwrap();
            encoder_b
                .consume(MediaBuffer::Audio(Arc::new(silent_frame(
                    48_000,
                    1,
                    960,
                    tick * 960,
                ))))
                .unwrap();
        }

        encoder_a.consume(MediaBuffer::Eos).unwrap();
        if let Ok(unfinished) = std::fs::read_to_string(&playlist_path) {
            assert!(
                !unfinished.contains("#EXT-X-ENDLIST"),
                "the first finished track must not finalize the shared playlist"
            );
        }

        for tick in 60..80i64 {
            encoder_b
                .consume(MediaBuffer::Audio(Arc::new(silent_frame(
                    48_000,
                    1,
                    960,
                    tick * 960,
                ))))
                .unwrap();
        }
        encoder_b.consume(MediaBuffer::Eos).unwrap();
        let finished = std::fs::read_to_string(&playlist_path).unwrap();
        assert!(finished.contains("#EXT-X-ENDLIST"));

        drop(encoder_a);
        drop(encoder_b);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
