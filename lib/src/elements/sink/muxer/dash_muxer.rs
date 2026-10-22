use std::{
    borrow::Cow,
    ffi::CString,
    path::{Path, PathBuf},
    ptr,
    time::Duration,
};

use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use super::prune::SegmentPruner;
use super::track_sink::{Muxer, PendingStream, TrackOptions, open_tracks};
use super::tracks::{MuxerId, MuxerSinks, MuxerTrack, TrackFormat};

use crate::{
    element::ElementType,
    error::{Error, Result},
};

/// How many segments a live manifest's window leaves on disk after they
/// leave it, for a player still fetching one — as an HLS playlist keeps one
/// past its window.
const KEPT_PAST_THE_WINDOW: usize = 1;

/// Which segments a [`DashMuxer`]'s manifest lists, and how it ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DashMode {
    /// A sliding live manifest of the last `window_size` segments; older
    /// segment files are deleted once they have left it. FFmpeg deletes
    /// them where it can; where it was built with MSVC, as vcpkg builds it
    /// for Windows, its file protocol cannot, and the muxer deletes them
    /// itself.
    Live {
        /// How many completed segments the manifest lists.
        window_size: usize,
    },
    /// Every segment kept and listed: a live manifest (`type="dynamic"`) as
    /// it grows — a player may join and seek back to its start — and a
    /// finished video-on-demand one (`type="static"`) once every track has
    /// ended. What a recording that is also watched is.
    Event,
}

/// Where a [`DashMuxer`] writes, and how it cuts and keeps segments.
///
/// [`DashOptions::new`] gives a live manifest; change a field to differ
/// from it.
#[derive(Debug, Clone)]
pub struct DashOptions {
    /// The manifest, normally ending in `.mpd`. The initialization and
    /// media segments — `init-stream<N>.m4s`, `chunk-stream<N>-<number>.m4s`,
    /// a track each — are written beside it.
    pub manifest_path: PathBuf,
    /// Target segment duration. FFmpeg cuts on the next video keyframe, so
    /// the actual duration can be longer when keyframes are sparse.
    pub segment_duration: Duration,
    /// Which segments the manifest lists, and how it ends.
    pub mode: DashMode,
    /// Also writes HLS playlists over the very same fMP4 segments —
    /// `master.m3u8` beside the manifest, and a media playlist for each
    /// track — so that one set of segments plays in DASH players and HLS
    /// players (Safari, iOS) alike. CMAF, as it is called.
    pub hls_playlist: bool,
}

impl DashOptions {
    /// A live manifest at `manifest_path`: two-second segments, the last six
    /// listed, no HLS playlists.
    pub fn new(manifest_path: impl Into<PathBuf>) -> Self {
        crate::ensure_ffmpeg();
        Self {
            manifest_path: manifest_path.into(),
            segment_duration: Duration::from_secs(2),
            mode: DashMode::Live { window_size: 6 },
            hls_playlist: false,
        }
    }

    fn validate(&self) -> std::result::Result<(), DashMuxerError> {
        if self.segment_duration.is_zero() {
            return Err(DashMuxerError::ZeroSegmentDuration);
        }
        if let DashMode::Live { window_size } = self.mode
            && (window_size == 0 || window_size > i32::MAX as usize)
        {
            return Err(DashMuxerError::InvalidWindowSize(window_size));
        }
        manifest_for_ffmpeg(&self.manifest_path)?;
        Ok(())
    }

    fn header_options(&self) -> ffmpeg::Dictionary<'static> {
        let mut options = ffmpeg::Dictionary::new();
        options.set(
            "seg_duration",
            &self.segment_duration.as_secs_f64().to_string(),
        );
        options.set("dash_segment_type", "mp4");
        match self.mode {
            DashMode::Live { window_size } => {
                options.set("window_size", &window_size.to_string());
                options.set("extra_window_size", &KEPT_PAST_THE_WINDOW.to_string());
            }
            DashMode::Event => options.set("window_size", "0"),
        }
        if self.hls_playlist {
            options.set("hls_playlist", "1");
        }
        options
    }
}

/// The manifest's path as FFmpeg takes it: UTF-8 with no NUL, and `/` for
/// separators — the DASH muxer puts its segments in the directory it finds
/// before the last `/`, so a Windows `\` would put them in the working
/// directory instead.
fn manifest_for_ffmpeg(path: &Path) -> std::result::Result<Cow<'_, str>, DashMuxerError> {
    let utf8 = path
        .to_str()
        .ok_or_else(|| DashMuxerError::NonUtf8Path(path.to_path_buf()))?;
    if utf8.contains('\0') {
        return Err(DashMuxerError::EmbeddedNul);
    }
    #[cfg(windows)]
    {
        Ok(Cow::Owned(utf8.replace('\\', "/")))
    }
    #[cfg(not(windows))]
    {
        Ok(Cow::Borrowed(utf8))
    }
}

/// Allocates the `AVFMT_NOFILE` DASH muxer without opening the manifest as
/// a plain `AVIOContext`: the muxer owns that file and replaces it whole
/// each time it writes it.
fn allocate_output(
    options: &DashOptions,
) -> std::result::Result<ffmpeg::format::context::Output, DashMuxerError> {
    let path = manifest_for_ffmpeg(&options.manifest_path)?;
    let path = CString::new(path.as_ref()).map_err(|_| DashMuxerError::EmbeddedNul)?;
    let format = CString::new("dash").expect("static DASH format name contains no NUL");
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
        return Err(DashMuxerError::Ffmpeg(ffmpeg::Error::from(result)));
    }
    if context.is_null() {
        return Err(DashMuxerError::Ffmpeg(ffmpeg::Error::Unknown));
    }
    // SAFETY: `context` is non-null — checked just above — and was allocated by
    // `avformat_alloc_output_context2` and never handed anywhere else, so
    // `Output::wrap` is taking sole ownership as it requires.
    Ok(unsafe { ffmpeg::format::context::Output::wrap(context) })
}

/// What can go wrong configuring or writing a [`DashMuxer`].
#[derive(Debug, ThisError)]
pub enum DashMuxerError {
    /// A stream sink received a buffer other than a packet or end-of-stream.
    #[error("DashMuxer stream sinks only accept Packet buffers, got {0}")]
    UnsupportedBuffer(&'static str),

    /// [`DashOptions::segment_duration`] is zero.
    #[error("DASH segment duration must be greater than zero")]
    ZeroSegmentDuration,

    /// A live manifest's window cannot be represented by FFmpeg.
    #[error(
        "DASH live window size must be between 1 and {max}, got {0}",
        max = i32::MAX
    )]
    InvalidWindowSize(usize),

    /// The manifest path cannot be passed through FFmpeg's UTF-8 API.
    #[error("DASH manifest path must be valid UTF-8: {0:?}")]
    NonUtf8Path(PathBuf),

    /// The manifest path contains an interior NUL byte, which FFmpeg's C
    /// API cannot take.
    #[error("DASH manifest path must not contain a NUL byte")]
    EmbeddedNul,

    /// The muxer was opened before any media stream was registered.
    #[error("DASH muxer requires at least one stream")]
    NoStreams,

    /// FFmpeg rejected muxer creation, packet writing, or finalization.
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),
}

/// Writes encoded tracks as an MPEG-DASH manifest and its fMP4 segments, for
/// players and browsers to stream over plain HTTP — and, with
/// [`DashOptions::hls_playlist`], HLS playlists over the same segments.
///
/// It takes packets, one sink per track, and encodes nothing, as
/// [`HlsMuxer`](crate::elements::HlsMuxer) does: [`DashOptions`] says where
/// the manifest goes, how long a segment is, and whether the manifest is a
/// sliding live window or every segment ([`DashMode`]). Serving the files
/// is left to any web server; a browser plays DASH through a player such as
/// dash.js, and Safari plays none — which is what the HLS playlists are for.
///
/// Reading one back — [`FileDemuxer`](crate::elements::FileDemuxer) on the
/// `.mpd` — takes an FFmpeg built with its DASH demuxer, which needs
/// libxml2: where the FFmpeg this crate links has none, opening it fails
/// as any unreadable input does, and the HLS playlists still read.
///
/// # Example
///
/// ```no_run
/// # use media_pp::ffmpeg;
/// # use media_pp::elements::{SwEncoder, SwEncoderOptions, VideoCodec};
/// use media_pp::elements::{DashMode, DashMuxer, DashOptions};
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
/// // A recording watched as it grows, playable in DASH and HLS players alike.
/// let mut options = DashOptions::new("recording/stream.mpd");
/// options.mode = DashMode::Event;
/// options.hls_playlist = true;
/// let mut muxer = DashMuxer::create(options)?;
/// let video = muxer.add_stream("video", &encoder)?;
/// let mut sinks = muxer.open()?; // writes the header
/// let video_sink = sinks.take(video)?; // ends the chain behind `encoder`
/// # Ok(())
/// # }
/// ```
///
/// # How it works
///
/// FFmpeg's `dash` muxer, writing each track as fMP4 — an initialization
/// segment and media segments cut at keyframes — and rewriting the manifest
/// as each segment completes. The manifest is finished once every track's
/// sink has had its end or a stop, as `HlsMuxer`'s playlist is: a recording
/// stopped part way is still a whole one.
pub struct DashMuxer {
    id: MuxerId,
    output: ffmpeg::format::context::Output,
    streams: Vec<PendingStream>,
    options: DashOptions,
}

impl DashMuxer {
    /// Allocates FFmpeg's DASH output context. The manifest's directory
    /// must already exist.
    pub fn create(options: DashOptions) -> Result<Self> {
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
    /// [`TrackFormat`]). Its time base must match the timestamps carried by
    /// packets arriving at the returned track sink.
    ///
    /// The returned [`MuxerTrack`] is how this track's sink is taken out of
    /// what [`DashMuxer::open`] returns.
    pub fn add_stream(
        &mut self,
        name: impl Into<String>,
        format: impl Into<TrackFormat>,
    ) -> Result<MuxerTrack> {
        let TrackFormat {
            parameters,
            time_base,
            ..
        } = format.into();
        let mut stream = self
            .output
            .add_stream(parameters.id())
            .map_err(DashMuxerError::from)?;
        let pending = PendingStream::new(name, &parameters, time_base);
        stream.set_time_base(time_base);
        stream.set_parameters(parameters);
        let track = self.id.track(self.streams.len());
        self.streams.push(pending);
        Ok(track)
    }

    /// Writes the DASH header and returns one sink per stream, each taken
    /// out by the [`MuxerTrack`] its [`DashMuxer::add_stream`] returned. The
    /// manifest is finished only after every returned sink has received
    /// `Eos` or a stop.
    pub fn open(mut self) -> Result<MuxerSinks> {
        crate::ensure_ffmpeg();
        if self.streams.is_empty() {
            return Err(DashMuxerError::NoStreams.into());
        }
        let unused = self
            .output
            .write_header_with(self.options.header_options())
            .map_err(DashMuxerError::from)?;
        drop(unused);
        // FFmpeg keeps the window and one past it; one more again, so that
        // where FFmpeg deletes, this finds nothing to.
        let prune = match self.options.mode {
            DashMode::Live { window_size } => Some(SegmentPruner::new(
                SegmentPruner::dir_of(&self.options.manifest_path),
                "chunk-stream",
                ".m4s",
                window_size + KEPT_PAST_THE_WINDOW + 1,
            )),
            DashMode::Event => None,
        };
        Ok(open_tracks::<Self>(
            self.id,
            self.output,
            self.streams,
            TrackOptions {
                finish_on_stop: true,
                destination: None,
                timeline: None,
                prune,
            },
        ))
    }
}

impl Muxer for DashMuxer {
    const ELEMENT_TYPE: ElementType = ElementType::DashMuxer;

    fn ffmpeg_error(error: ffmpeg::Error) -> Error {
        DashMuxerError::Ffmpeg(error).into()
    }

    fn unsupported_buffer(kind: &'static str) -> Error {
        DashMuxerError::UnsupportedBuffer(kind).into()
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

    fn silent_frame(samples: usize, pts: i64) -> ffmpeg::frame::Audio {
        let mut frame = ffmpeg::frame::Audio::new(
            ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed),
            samples,
            ffmpeg::ChannelLayout::default(1),
        );
        frame.set_rate(48_000);
        frame.set_pts(Some(pts));
        frame.data_mut(0).fill(0);
        frame
    }

    fn unique_test_dir(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("media_pp_{label}_{}_{nonce}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// `ticks` of 20 ms of AAC silence through a muxer of `options`.
    fn encode_silence(options: DashOptions, ticks: i64) {
        let mut encoder = SwAudioEncoder::new(
            "encoder",
            SwAudioEncoderOptions {
                codec: AudioCodec::Aac,
                sample_rate: 48_000,
                channels: 1,
                bit_rate: 64_000,
            },
        )
        .expect("aac encoder must be available");
        let mut muxer = DashMuxer::create(options).expect("DASH muxer must open");
        let audio = muxer
            .add_stream(
                "audio",
                TrackFormat::new(encoder.parameters(), ffmpeg::Rational::new(1, 48_000)),
            )
            .expect("add_stream must succeed");
        let mut sinks = muxer.open().expect("DASH header must be written");
        encoder.src_pads()[0].link(sinks.take(audio).expect("the muxer's own track").into_raw());
        for tick in 0..ticks {
            encoder
                .consume(MediaBuffer::Audio(
                    Arc::new(silent_frame(960, tick * 960)).into(),
                ))
                .expect("encoding and muxing must succeed");
        }
        crate::stream::deliver(&mut encoder, &crate::stream::StreamEvent::Eos)
            .expect("EOS must finalize the DASH manifest");
    }

    fn dash_demuxer_built() -> bool {
        // SAFETY: a static NUL-terminated name; FFmpeg returns a pointer to
        // its own static demuxer table, or null.
        !unsafe { ffmpeg::ffi::av_find_input_format(c"dash".as_ptr()) }.is_null()
    }

    fn files_ending(dir: &Path, suffix: &str) -> Vec<PathBuf> {
        std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.ends_with(suffix))
            })
            .collect()
    }

    /// An event recording ends as a finished, static manifest over every
    /// segment, and with HLS playlists asked for, those play the very same
    /// segments. Both are read back through FFmpeg — the manifest only where
    /// its DASH demuxer was built, which takes libxml2; the HLS one always is.
    #[test]
    fn an_event_recording_ends_a_static_manifest_playable_as_hls_too() {
        let dir = unique_test_dir("dash_event");
        let manifest = dir.join("stream.mpd");
        let mut options = DashOptions::new(&manifest);
        options.segment_duration = Duration::from_secs(1);
        options.mode = DashMode::Event;
        options.hls_playlist = true;

        encode_silence(options, 200);

        let mpd = std::fs::read_to_string(&manifest).unwrap();
        assert!(mpd.contains("<MPD"), "{mpd}");
        assert!(mpd.contains("type=\"static\""), "finished: {mpd}");
        assert!(
            !files_ending(&dir, ".m4s").is_empty(),
            "segments beside the manifest"
        );
        let chunks = files_ending(&dir, ".m4s")
            .into_iter()
            .filter(|path| path.to_string_lossy().contains("chunk"))
            .count();
        assert!(chunks >= 3, "four seconds in one-second segments: {chunks}");

        let master = dir.join("master.m3u8");
        let playlist = std::fs::read_to_string(&master).unwrap();
        assert!(playlist.starts_with("#EXTM3U"), "{playlist}");
        let mut input = ffmpeg::format::input(&master)
            .expect("FFmpeg must be able to read the HLS playlists over the DASH segments");
        let mut packet = ffmpeg::Packet::empty();
        packet
            .read(&mut input)
            .expect("the playlists must lead to media packets");
        drop(input);
        if dash_demuxer_built() {
            let mut input = ffmpeg::format::input(&manifest)
                .expect("FFmpeg's DASH demuxer must be able to read the manifest");
            packet
                .read(&mut input)
                .expect("the manifest must lead to media packets");
        } else {
            eprintln!(
                "skipped reading the manifest back: FFmpeg was built without its DASH demuxer"
            );
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A live manifest lists its window alone, and segments that left it go
    /// from the disk, but one or two for a player still fetching them —
    /// whether FFmpeg can delete a file or the muxer does it for it.
    #[test]
    fn a_live_manifest_keeps_its_window_and_lets_old_segments_go() {
        let dir = unique_test_dir("dash_live");
        let manifest = dir.join("stream.mpd");
        let mut options = DashOptions::new(&manifest);
        options.segment_duration = Duration::from_secs(1);
        options.mode = DashMode::Live { window_size: 2 };
        // Six segments of a representation FFmpeg writes none of here, so
        // never deletes: what goes of them is the muxer's own doing, on an
        // FFmpeg that deletes as on one that cannot — the window, the one
        // FFmpeg keeps past it and one more stay. The seconds go by faster
        // than they play, as written here.
        let foreign = |number: u32| dir.join(format!("chunk-stream9-{number:05}.m4s"));
        for number in 1..=6 {
            std::fs::write(foreign(number), b"x").unwrap();
        }

        encode_silence(options, 400);

        assert!(
            !foreign(1).exists() && !foreign(2).exists(),
            "the muxer deletes what left the window"
        );
        assert!((3..=6).all(|number| foreign(number).exists()));
        for number in 3..=6 {
            std::fs::remove_file(foreign(number)).unwrap();
        }
        let mpd = std::fs::read_to_string(&manifest).unwrap();
        let start = mpd
            .split("startNumber=\"")
            .nth(1)
            .and_then(|rest| rest.split('"').next())
            .and_then(|number| number.parse::<u32>().ok())
            .unwrap_or_else(|| panic!("a numbered template: {mpd}"));
        assert!(
            start >= 5,
            "eight one-second segments, a window of two: listed from {start}"
        );
        let chunks = files_ending(&dir, ".m4s")
            .into_iter()
            .filter(|path| path.to_string_lossy().contains("chunk"))
            .count();
        // FFmpeg keeps the window and one more; where it cannot delete — built
        // with MSVC, as on Windows CI — the muxer keeps one more again.
        assert!(
            (2..=2 + KEPT_PAST_THE_WINDOW + 1).contains(&chunks),
            "eight seconds in one-second segments, a window of two kept: {chunks} on disk"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn what_cannot_be_written_is_refused_at_create() {
        let mut options = DashOptions::new(std::env::temp_dir().join("x.mpd"));
        options.segment_duration = Duration::ZERO;
        assert!(matches!(
            DashMuxer::create(options).err(),
            Some(Error::DashMuxerError(DashMuxerError::ZeroSegmentDuration))
        ));
        let mut options = DashOptions::new(std::env::temp_dir().join("x.mpd"));
        options.mode = DashMode::Live { window_size: 0 };
        assert!(matches!(
            DashMuxer::create(options).err(),
            Some(Error::DashMuxerError(DashMuxerError::InvalidWindowSize(0)))
        ));
    }
}
