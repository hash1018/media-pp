use std::{
    ffi::CString,
    path::{Path, PathBuf},
    ptr,
};

use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use super::track_sink::{Muxer, PendingStream, TrackOptions, open_tracks};
use super::tracks::{MuxerId, MuxerSinks, MuxerTrack, TrackFormat};

use crate::{
    element::ElementType,
    error::{Error, Result},
};

/// What can go wrong creating or writing a [`FileMuxer`]'s file.
///
/// Converts into the crate-wide `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum FileMuxerError {
    /// A stream sink received a buffer other than a packet or end-of-stream.
    #[error("FileMuxer stream sinks only accept Packet buffers, got {0}")]
    UnsupportedBuffer(&'static str),

    /// FFmpeg rejected packet writing or finalization.
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),

    /// The path's name picks no container FFmpeg knows. Its extension is
    /// what picks one — `.mp4`, `.mkv`, `.mov`, `.ts` — so a name with none,
    /// or one FFmpeg has no muxer for, is refused before anything is
    /// created.
    #[error(
        "no container is named by {}: its extension picks one, such as .mp4 or .mkv",
        path.display()
    )]
    UnknownContainer {
        /// The file asked for.
        path: PathBuf,
    },

    /// [`FileMuxer::create`] asked a file whose name picks a
    /// container other than MP4 or MOV for a layout only those have.
    #[error(
        "{} is not an MP4 or MOV name: only those are laid out fast-start or in fragments",
        path.display()
    )]
    NotMp4 {
        /// The file asked for.
        path: PathBuf,
    },

    /// The file could not be created: its directory is not there, or not
    /// writable.
    #[error("could not create {}: {source}", path.display())]
    Create {
        /// The file asked for.
        path: PathBuf,
        /// What FFmpeg said.
        #[source]
        source: ffmpeg::Error,
    },
}

/// Writes encoded video, sound and subtitle tracks into one file — MP4,
/// MKV, MOV or any container FFmpeg knows by the file's extension.
///
/// It takes packets, one sink per track, and encodes nothing: put an encoder in
/// front of each track, or copy a demuxer's packets through unchanged. Declare
/// every track with [`FileMuxer::add_stream`] first, then [`FileMuxer::open`]
/// writes the header and hands back the tracks' sinks, each of which ends a
/// chain. The file is finished once every track has reached the end of its
/// stream — [`Pipeline::finish`](crate::pipeline::Pipeline::finish) — or the
/// pipeline is stopped, which finishes it with what it was handed so far. A
/// recording added to a running [`Tee`](crate::elements::Tee) starts at zero
/// with a [`TimestampOrigin`](crate::elements::TimestampOrigin) in front of
/// each track. For a recording cut into several files see
/// [`SegmentedFileMuxer`](crate::elements::SegmentedFileMuxer); for one written
/// only when asked, [`ReplayBuffer`](crate::elements::ReplayBuffer).
///
/// # Example
///
/// ```no_run
/// # fn main() -> media_pp::Result<()> {
/// # use media_pp::ffmpeg;
/// # use media_pp::elements::{SwEncoderOptions, VideoCodec};
/// use media_pp::elements::{FileMuxer, Mp4Layout, SwEncoder, TestVideoOptions, TestVideoSource};
/// use media_pp::pipeline::Pipeline;
///
/// let source = TestVideoSource::new("source", TestVideoOptions::default());
/// # let options = SwEncoderOptions {
/// #     codec: VideoCodec::OpenH264,
/// #     width: 640,
/// #     height: 480,
/// #     pixel_format: ffmpeg::format::Pixel::YUV420P,
/// #     frame_rate: ffmpeg::Rational(30, 1),
/// #     bit_rate: 2_000_000,
/// #     gop_size: 60,
/// #     max_b_frames: None,
/// # };
/// let encoder = SwEncoder::new("encoder", options)?; // H.264, 640x480 at 30 fps
///
/// let mut muxer = FileMuxer::create("out.mp4", Mp4Layout::Standard)?;
/// let video = muxer.add_stream("video", &encoder)?; // more tracks go here
/// let mut sinks = muxer.open()?; // writes the header
/// let video_sink = sinks.take(video)?;
///
/// let (pipeline, ()) = Pipeline::new("record", source, |source, ctx| {
///     let branch = ctx.branch().pipe(encoder).to(video_sink)?;
///     ctx.attach(source, 0, branch)?;
///     Ok(())
/// })?;
/// pipeline.run()?;
/// // ... later: drain the encoder and write the trailer.
/// pipeline.finish();
/// # Ok(())
/// # }
/// ```
///
/// # How it works
///
/// It builds one container file with one or more tracks, then opens it into
/// one [`RawSink`](crate::element::RawSink) per track.
///
/// Which container is the path's own: `format::output` asks FFmpeg to guess
/// a muxer from the file name, so `.mp4` gets MP4 and `.mkv` gets Matroska
/// out of the same type. Nothing here is MP4-specific.
///
/// Two-phase on purpose: a container's header has to describe
/// every stream's codec parameters up front — `avformat_write_header`
/// can't run until every [`FileMuxer::add_stream`] this file will ever hold
/// has already happened — so there's no way to make this a single
/// long-lived `RawSink` that tracks attach to one at a time as their encoders
/// come online (contrast [`crate::elements::AudioMixer`], whose inputs
/// *can* attach at any time — it has no "known shape before the first
/// byte" constraint the way a container header does).
pub struct FileMuxer {
    id: MuxerId,
    output: ffmpeg::format::context::Output,
    streams: Vec<PendingStream>,
    layout: Mp4Layout,
}

/// Where an MP4 or MOV file keeps its index, chosen in [`FileMuxer::create`].
///
/// The index is the `moov` box that says where every sample is. The samples are
/// the same in all three; what differs is what can read the file before it
/// is finished, and how soon one streamed starts to play.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Mp4Layout {
    /// The index after the samples, written by the trailer: the quickest to
    /// finish. A file cut off before its trailer — a process that dies, a
    /// machine that loses power — opens not at all, and one streamed plays
    /// only once its end has been fetched.
    #[default]
    Standard,
    /// The index moved in front of the samples as the file is finished, so
    /// one streamed — in a browser, a messenger's preview, a cloud drive's
    /// player — starts as its first bytes arrive. Finishing reads and writes
    /// the whole file once more, which takes about as long as copying it;
    /// for a file handed to someone else when it is done, not a recording
    /// that may be cut off.
    FastStart,
    /// A header that lists the tracks and no samples, then one fragment
    /// (`moof` + `mdat`) after another, each beginning at a video keyframe
    /// — or, with no video track, every second — and each handed to the
    /// file as it is cut.
    ///
    /// What it is for is a file cut off: every fragment already on disk
    /// plays, and what is lost is the fragment under way, one keyframe
    /// interval at most. The same shape is what a browser streams from
    /// (Media Source Extensions, or WebCodecs behind `mp4box.js`) without
    /// reading the whole file first. The cost is a slightly larger file,
    /// and a seek that reads through the fragments rather than one index;
    /// the trailer still writes an index of the fragments (`mfra`) for the
    /// players that look for it.
    Fragmented,
}

impl FileMuxer {
    /// Allocates the output file. No header is written yet — nothing is on
    /// disk in a readable shape until [`FileMuxer::open`] runs.
    ///
    /// An MP4 or MOV file is laid out as `layout` says; a file of any other
    /// container takes [`Mp4Layout::Standard`] alone, and asked for another
    /// is refused as [`FileMuxerError::NotMp4`] before anything is created.
    pub fn create(path: impl AsRef<Path>, layout: Mp4Layout) -> Result<Self> {
        crate::ensure_ffmpeg();
        let path = path.as_ref();
        // Asked first, as FFmpeg's own guess would be, so a name with no
        // container in it says so rather than FFmpeg's `Invalid argument`.
        let named = container_named(path);
        if named.as_ref().is_some_and(Option::is_none) {
            return Err(FileMuxerError::UnknownContainer {
                path: path.to_path_buf(),
            }
            .into());
        }
        if layout != Mp4Layout::Standard && !named.flatten().is_some_and(|name| quicktime(&name)) {
            return Err(FileMuxerError::NotMp4 {
                path: path.to_path_buf(),
            }
            .into());
        }
        let output = ffmpeg::format::output(&path).map_err(|source| FileMuxerError::Create {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(Self {
            id: MuxerId::next(),
            output,
            streams: Vec::new(),
            layout,
        })
    }

    /// Registers one more track this file will hold. `format` describes it
    /// — typically `&encoder` for the encoder that feeds it, or the
    /// demuxed [`&StreamInfo`](crate::elements::StreamInfo) for a stream
    /// copied through unchanged; see [`TrackFormat`]. `name` becomes
    /// this track's own [`Element::name`](crate::element::Element::name)/`pp_log` identity once
    /// [`FileMuxer::open`] turns it into a `RawSink` — pick something that
    /// tells multiple tracks apart in logs/[`crate::bus::BusEvent`]s,
    /// e.g. `"video"`/`"audio"`.
    ///
    /// The returned [`MuxerTrack`] is how this track's sink is taken out of
    /// what [`FileMuxer::open`] returns.
    pub fn add_stream(
        &mut self,
        name: impl Into<String>,
        format: impl Into<TrackFormat>,
    ) -> Result<MuxerTrack> {
        let TrackFormat {
            parameters,
            time_base,
            fixed_parameter_sets,
        } = format.into();
        let hvc1 = self.hvc1(&parameters, fixed_parameter_sets);
        let mut stream = self
            .output
            .add_stream(parameters.id())
            .map_err(FileMuxerError::from)?;
        let pending = PendingStream::new(name, &parameters, time_base);
        stream.set_time_base(time_base);
        stream.set_parameters(parameters);
        if hvc1 {
            // SAFETY: the stream's own parameters, just set; the tag is a
            // plain field the muxer reads when it writes the header.
            unsafe {
                (*stream.parameters().as_mut_ptr()).codec_tag = u32::from_le_bytes(*b"hvc1");
            }
        }
        let track = self.id.track(self.streams.len());
        self.streams.push(pending);
        Ok(track)
    }

    /// Whether a track of `parameters` goes in as `hvc1`: HEVC, into MP4 or
    /// MOV, saying no tag of its own, its parameter sets out of band — as
    /// an encoder of this crate's are. FFmpeg writes `hev1` unless told,
    /// and an iPhone, QuickTime and every other Apple player refuse to play
    /// HEVC in MP4 or MOV under it; `hvc1` is the same stream with its
    /// parameter sets in the sample entry, which every other player plays
    /// too.
    ///
    /// Or in band, from an encoder whose sets never change
    /// (`fixed_parameter_sets`): FFmpeg's MP4 muxer then takes them from the
    /// first packet into the sample entry, and out of every sample, as
    /// `hvc1` requires. A stream copied from elsewhere with its sets in band
    /// stays `hev1`, since they may change partway and only `hev1` lets a
    /// sample carry new ones.
    fn hvc1(&self, parameters: &ffmpeg::codec::Parameters, fixed_parameter_sets: bool) -> bool {
        // SAFETY: plain fields of parameters the caller owns.
        let (tag, extradata) = unsafe {
            let raw = parameters.as_ptr();
            ((*raw).codec_tag, (*raw).extradata_size)
        };
        parameters.id() == ffmpeg::codec::Id::HEVC
            && tag == 0
            && (extradata > 0 || fixed_parameter_sets)
            && quicktime(self.output.format().name())
    }

    /// Writes the container header — every [`FileMuxer::add_stream`] call
    /// this file will ever get must already have happened — and returns
    /// one [`RawSink`](crate::element::RawSink) per track, each taken out by the [`MuxerTrack`] its
    /// [`FileMuxer::add_stream`] returned.
    ///
    /// All returned `RawSink`s write into the same underlying file behind a
    /// shared lock: packets from independently-threaded branches (e.g. a
    /// video encode chain and an audio encode chain, each on their own
    /// [`crate::queue::Queue`]) can arrive concurrently, and neither
    /// `av_interleaved_write_frame` nor `av_write_trailer` is safe to call
    /// from multiple threads against the same file at once. They also
    /// share one trailer: it's written once every track has reported
    /// itself done — via `Eos` *or* a stop, either meaning
    /// "this track is finished" rather than "abandon the whole file" —
    /// not on whichever track finishes first, which would silently
    /// truncate whatever the other track(s) still had left to write. A
    /// single-track file (e.g. `screen_record_software`/`audio_record`) degenerates
    /// to finalizing on that one track's own `Eos`/`Stop`, same as before
    /// this type supported more than one.
    ///
    /// Tracks fed by several sources belong in one pipeline, through
    /// [`crate::pipeline::PipelineBuilder::add_source`] — a live video
    /// capture and a live audio capture together — so that one
    /// [`crate::pipeline::Pipeline::finish`] ends them all. A caller feeding
    /// tracks from separate pipelines instead has to end every one of them:
    /// the trailer is written only once every track has reported done, so
    /// ending one pipeline while another keeps running leaves the file
    /// un-finalized (and unplayable) until the rest catch up too.
    pub fn open(mut self) -> Result<MuxerSinks> {
        crate::ensure_ffmpeg();
        if let Some(options) = self.layout_options() {
            let unused = self
                .output
                .write_header_with(options)
                .map_err(FileMuxerError::from)?;
            debug_assert!(
                unused.iter().next().is_none(),
                "the mov muxer took no option named {:?}",
                unused.iter().map(|(key, _)| key).collect::<Vec<_>>()
            );
        } else {
            self.output.write_header().map_err(FileMuxerError::from)?;
        }
        Ok(open_tracks::<Self>(
            self.id,
            self.output,
            self.streams,
            TrackOptions {
                finish_on_stop: true,
                destination: None,
                timeline: None,
            },
        ))
    }
}

impl FileMuxer {
    /// What the mov muxer is told for the layout asked; `None` for the one
    /// it writes untold.
    ///
    /// `faststart` has the trailer move the index in front of the samples,
    /// reading the file back to do it.
    ///
    /// For fragments, `empty_moov` puts the track list up front with no samples in it, and
    /// `delay_moov` writes it with the first fragment rather than with the
    /// header, so a track whose codec configuration comes only in its first
    /// packets — a stream copied from a transport stream — still gets one.
    /// `default_base_moof` makes each fragment's offsets its own, as Media
    /// Source Extensions asks.
    ///
    /// A fragment ends at a video keyframe, so each begins with a picture
    /// that decodes on its own. `frag_keyframe` looks at video tracks
    /// alone, so a file of sound only would be one fragment the trailer
    /// writes — nothing on disk until the end, the very thing fragments
    /// are for; such a file is cut every second instead.
    ///
    /// `flush_packets` hands each fragment to the file as it is written
    /// rather than leaving up to 32 KB of it in FFmpeg's own buffer, which a
    /// process that dies takes with it. The mov muxer holds a fragment's
    /// packets itself until it cuts one, so between fragments there is
    /// nothing to hand over.
    fn layout_options(&self) -> Option<ffmpeg::Dictionary<'static>> {
        let mut options = ffmpeg::Dictionary::new();
        match self.layout {
            Mp4Layout::Standard => return None,
            Mp4Layout::FastStart => {
                options.set("movflags", "faststart");
                return Some(options);
            }
            Mp4Layout::Fragmented => {}
        }
        let video = self
            .output
            .streams()
            .any(|stream| stream.parameters().medium() == ffmpeg::media::Type::Video);
        options.set("flush_packets", "1");
        if video {
            options.set(
                "movflags",
                "frag_keyframe+empty_moov+delay_moov+default_base_moof",
            );
        } else {
            options.set("movflags", "empty_moov+delay_moov+default_base_moof");
            options.set("frag_duration", "1000000");
        }
        Some(options)
    }
}

/// The muxer FFmpeg would pick for `path` by its name: `Some(None)` for a
/// name that picks none, `None` for a name that is not text, which FFmpeg
/// is left to try.
fn container_named(path: &Path) -> Option<Option<String>> {
    crate::ensure_ffmpeg();
    path.to_str()
        .and_then(|name| CString::new(name).ok())
        .map(|name| {
            // SAFETY: `name` is a live NUL-terminated string; the other two
            // arguments may be null. A format returned is a static FFmpeg
            // owns, whose name is a static string.
            unsafe {
                let format = ffmpeg::ffi::av_guess_format(ptr::null(), name.as_ptr(), ptr::null());
                (!format.is_null()).then(|| {
                    std::ffi::CStr::from_ptr((*format).name)
                        .to_string_lossy()
                        .into_owned()
                })
            }
        })
}

/// Whether a file at `path` would be an MP4 or MOV, and so takes every
/// [`Mp4Layout`].
pub(crate) fn names_mp4(path: &Path) -> bool {
    container_named(path)
        .flatten()
        .is_some_and(|name| quicktime(&name))
}

/// Whether FFmpeg's muxer `name` — a comma-separated list for the mov
/// family — writes MP4 or MOV boxes.
fn quicktime(name: &str) -> bool {
    name.split(',')
        .any(|name| matches!(name, "mp4" | "mov" | "ipod" | "ismv" | "3gp" | "3g2"))
}

impl Muxer for FileMuxer {
    const ELEMENT_TYPE: ElementType = ElementType::FileMuxer;

    fn ffmpeg_error(error: ffmpeg::Error) -> Error {
        FileMuxerError::Ffmpeg(error).into()
    }

    fn unsupported_buffer(kind: &'static str) -> Error {
        FileMuxerError::UnsupportedBuffer(kind).into()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::buffer::MediaBuffer;
    use crate::contract::{InputContract, MediaKind, PortContract};
    use crate::control::ControlMsg;
    use crate::element::RawSinkExt;
    use crate::element::{RawSink, SrcPads};
    use crate::elements::{AudioCodec, SwAudioEncoder, SwAudioEncoderOptions};

    /// A name that picks no container is refused as that, naming the file
    /// — not as FFmpeg's bare `Invalid argument`.
    #[test]
    fn a_name_that_picks_no_container_is_refused_as_one() {
        let path = std::env::temp_dir().join("media-pp-muxer-test.unknownext");
        let Err(error) = FileMuxer::create(&path, Mp4Layout::Standard) else {
            panic!("a name with no container in it was taken");
        };
        assert!(
            matches!(
                error,
                Error::FileMuxerError(FileMuxerError::UnknownContainer { .. })
            ),
            "{error:?}"
        );
        assert!(
            error.to_string().contains("media-pp-muxer-test.unknownext"),
            "{error}"
        );
        assert!(!path.exists(), "nothing is created for it");
    }

    /// A file that cannot be created says which — a directory that is not
    /// there reads as only `No such file or directory` otherwise.
    #[test]
    fn a_file_that_cannot_be_created_says_which() {
        let path = std::env::temp_dir()
            .join("media-pp-no-such-directory")
            .join("out.mp4");
        let Err(error) = FileMuxer::create(&path, Mp4Layout::Standard) else {
            panic!("a file in a directory that is not there was created");
        };
        assert!(
            matches!(error, Error::FileMuxerError(FileMuxerError::Create { .. })),
            "{error:?}"
        );
        assert!(error.to_string().contains("out.mp4"), "{error}");
    }

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

    /// `None` where the build has no `libopus`, so a stripped FFmpeg skips
    /// rather than failing on something it was never going to have.
    fn open_opus_encoder(sample_rate: u32, channels: u16) -> Option<SwAudioEncoder> {
        SwAudioEncoder::new(
            "encoder",
            SwAudioEncoderOptions {
                codec: AudioCodec::Opus,
                sample_rate,
                channels,
                bit_rate: 64_000,
            },
        )
        .ok()
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
        // `frame::Audio::new` doesn't zero its buffer — leaving it
        // uninitialized risks the encoder reading garbage bytes as NaN/Inf
        // floats (`avcodec_send_frame` then rejects the frame outright).
        frame.data_mut(0).fill(0);
        frame
    }

    #[test]
    fn seek_check_rejects_a_muxer_track() {
        let encoder = open_aac_encoder(48000, 1);
        let path = std::env::temp_dir().join(format!(
            "file_muxer_seek_check_test_{}.mp4",
            std::process::id()
        ));
        let mut muxer = FileMuxer::create(&path, Mp4Layout::Standard).expect("create muxer");
        let audio = muxer
            .add_stream(
                "audio",
                TrackFormat::new(encoder.parameters(), ffmpeg::Rational::new(1, 48000)),
            )
            .expect("add stream");
        let mut sink = muxer
            .open()
            .expect("open muxer")
            .take(audio)
            .expect("the muxer's own track");
        assert!(!sink.accepts_seek(), "muxer must reject seek");
        sink.control(&ControlMsg::Stop).expect("finalize muxer");
        std::fs::remove_file(path).ok();
    }

    /// One track, driven end to end (encode -> mux -> write_trailer on
    /// `Eos`), still produces a real, playable file — the single-track
    /// case `FileMuxer` degenerates to.
    #[test]
    fn single_track_still_produces_a_playable_file() {
        let mut encoder = open_aac_encoder(48000, 1);

        let dir = std::env::temp_dir();
        let path = dir.join(format!("file_muxer_single_test_{}.mp4", std::process::id()));

        let mut muxer = FileMuxer::create(&path, Mp4Layout::Standard).expect("the muxer must open");
        let audio = muxer
            .add_stream(
                "audio",
                TrackFormat::new(encoder.parameters(), ffmpeg::Rational::new(1, 48000)),
            )
            .expect("add_stream must succeed");
        let mut sinks = muxer.open().expect("open must write the header");
        encoder.src_pads()[0].link(sinks.take(audio).expect("the muxer's own track").into_raw());

        for tick in 0..20i64 {
            encoder
                .consume(MediaBuffer::Audio(
                    Arc::new(silent_frame(48000, 1, 960, tick * 960)).into(),
                ))
                .expect("consume must succeed");
        }
        crate::stream::deliver(&mut encoder, &crate::stream::StreamEvent::Eos)
            .expect("eos must flush cleanly");
        drop(encoder);

        let input = ffmpeg::format::input(&path).expect("muxed file must be readable back");
        assert_eq!(input.streams().count(), 1);
        std::fs::remove_file(&path).ok();
    }

    /// Regression test against a leaked file handle: dropping every track
    /// `RawSink` without ever sending `Eos`/`Stop` (simulating a `Pipeline`
    /// just getting dropped mid-recording, e.g. the process is tearing
    /// down) must still release the underlying file — no stray clone of
    /// the shared `Arc` (or the `ffmpeg::format::context::Output` it
    /// guards) left holding it open. Windows won't let an open file be
    /// deleted, so a successful `remove_file` here is direct proof
    /// nothing lingered; on a build where that isn't already guaranteed
    /// by construction, this would instead hang or fail with a sharing
    /// violation.
    #[test]
    fn dropping_every_sink_without_eos_or_stop_still_releases_the_file() {
        let encoder = open_aac_encoder(48000, 1);

        let dir = std::env::temp_dir();
        let path = dir.join(format!("file_muxer_drop_test_{}.mp4", std::process::id()));

        let mut muxer = FileMuxer::create(&path, Mp4Layout::Standard).expect("the muxer must open");
        // Never taken: its sink goes down with the rest, untouched.
        let _audio = muxer
            .add_stream(
                "audio",
                TrackFormat::new(encoder.parameters(), ffmpeg::Rational::new(1, 48000)),
            )
            .expect("add_stream must succeed");
        let sinks = muxer.open().expect("open must write the header");

        // No `Eos`/`Stop`, no trailer — just drop everything, on purpose.
        drop(sinks);
        drop(encoder);

        std::fs::remove_file(&path)
            .expect("file handle must be released once every sink is dropped");
    }

    /// Two independent tracks (standing in for a real video+audio pair —
    /// `FileMuxer` treats every stream as an opaque `codec::Parameters`, so
    /// two AAC tracks at different sample rates exercise the same
    /// stream-index/trailer-timing machinery a real video+audio pair
    /// would) muxed into one file. Track `a` reaches `Eos` well before
    /// track `b` does — proving the trailer isn't written until *both*
    /// report done, not on whichever finishes first (which would
    /// silently truncate whichever track was still running).
    #[test]
    fn muxes_two_independent_tracks_without_finalizing_early() {
        let mut encoder_a = open_aac_encoder(48000, 2);
        let mut encoder_b = open_aac_encoder(44100, 1);

        let dir = std::env::temp_dir();
        let path = dir.join(format!("file_muxer_multi_test_{}.mp4", std::process::id()));

        let mut muxer = FileMuxer::create(&path, Mp4Layout::Standard).expect("the muxer must open");
        let a = muxer
            .add_stream(
                "a",
                TrackFormat::new(encoder_a.parameters(), ffmpeg::Rational::new(1, 48000)),
            )
            .expect("add_stream a");
        let b = muxer
            .add_stream(
                "b",
                TrackFormat::new(encoder_b.parameters(), ffmpeg::Rational::new(1, 44100)),
            )
            .expect("add_stream b");
        let mut sinks = muxer.open().expect("open must write the header");
        let sink_b = sinks.take(b).expect("the muxer's own track");
        let sink_a = sinks.take(a).expect("the muxer's own track");
        encoder_a.src_pads()[0].link(sink_a.into_raw());
        encoder_b.src_pads()[0].link(sink_b.into_raw());

        for tick in 0..10i64 {
            encoder_a
                .consume(MediaBuffer::Audio(
                    Arc::new(silent_frame(48000, 2, 960, tick * 960)).into(),
                ))
                .expect("consume must succeed");
        }
        // Track `a` finishes here — well before track `b` has written
        // anything at all.
        crate::stream::deliver(&mut encoder_a, &crate::stream::StreamEvent::Eos)
            .expect("eos must flush cleanly");

        for tick in 0..10i64 {
            encoder_b
                .consume(MediaBuffer::Audio(
                    Arc::new(silent_frame(44100, 1, 882, tick * 882)).into(),
                ))
                .expect("consume must succeed");
        }
        crate::stream::deliver(&mut encoder_b, &crate::stream::StreamEvent::Eos)
            .expect("eos must flush cleanly");

        drop(encoder_a);
        drop(encoder_b);

        let mut input = ffmpeg::format::input(&path).expect("muxed file must be readable back");
        assert_eq!(input.streams().count(), 2, "expected two tracks");

        let mut counts = [0usize; 2];
        let mut packet = ffmpeg::Packet::empty();
        while packet.read(&mut input).is_ok() {
            counts[packet.stream()] += 1;
            packet = ffmpeg::Packet::empty();
        }
        assert!(counts[0] > 0, "track a has no packets: {counts:?}");
        assert!(
            counts[1] > 0,
            "track b has no packets: {counts:?} — trailer was written before track b finished"
        );
        std::fs::remove_file(&path).ok();
    }

    /// The container comes from the path's own extension, not from this
    /// type's name.
    ///
    /// `format::output` asks FFmpeg to guess a muxer from the filename, so
    /// this writes Matroska for a `.mkv` as readily as MP4 for a `.mp4` — the
    /// name says what it was written for, not what it is limited to. Worth a
    /// test rather than a comment: it is the difference between "we would
    /// have to add a muxer" and "name the file .mkv", and nothing else here
    /// would have caught the day it stopped being true.
    ///
    /// Written with real packets rather than a header and a trailer alone: a
    /// Matroska file with no cluster in it is not something FFmpeg will read
    /// back, so an empty one would fail here for a reason that has nothing to
    /// do with which muxer was picked.
    #[test]
    fn the_container_follows_the_path_extension() {
        let mut encoder = open_aac_encoder(48000, 1);

        let dir = std::env::temp_dir();
        let path = dir.join(format!("muxer_container_test_{}.mkv", std::process::id()));
        let _ = std::fs::remove_file(&path);

        let mut muxer =
            FileMuxer::create(&path, Mp4Layout::Standard).expect("the muxer must open a .mkv path");
        let audio = muxer
            .add_stream(
                "audio",
                TrackFormat::new(encoder.parameters(), ffmpeg::Rational::new(1, 48000)),
            )
            .expect("add_stream must succeed");
        let mut sinks = muxer.open().expect("open must write the header");
        encoder.src_pads()[0].link(sinks.take(audio).expect("the muxer's own track").into_raw());

        for tick in 0..20i64 {
            encoder
                .consume(MediaBuffer::Audio(
                    Arc::new(silent_frame(48000, 1, 960, tick * 960)).into(),
                ))
                .expect("consume must succeed");
        }
        crate::stream::deliver(&mut encoder, &crate::stream::StreamEvent::Eos)
            .expect("eos must flush cleanly");
        drop(encoder);

        let input = ffmpeg::format::input(&path).expect("the file must be readable");
        let format = input.format();
        assert!(
            format.name().contains("matroska"),
            "a .mkv path should have produced Matroska, got {:?}",
            format.name()
        );
        assert_eq!(input.streams().count(), 1);
        std::fs::remove_file(&path).ok();
    }

    /// A video track's codec extradata has to be in the container's header
    /// for Matroska, which writes `CodecPrivate` up front. MP4 does not need
    /// it there — `avcC` is written in the trailer, and the mov muxer will
    /// take the Annex-B SPS/PPS out of the packets themselves — so a
    /// video encoder opened without `AV_CODEC_FLAG_GLOBAL_HEADER` produces
    /// a working `.mp4` and an `avformat_write_header` that fails with
    /// `INVALIDDATA` for `.mkv`.
    ///
    /// Which is the whole reason this test exists at the muxer rather than at
    /// the encoder: the flag is the encoder's, and nothing but a non-MP4
    /// container ever notices it is missing.
    #[test]
    fn a_video_track_carries_its_extradata_into_a_matroska_header() {
        use crate::elements::{SwEncoder, SwEncoderOptions, VideoCodec};

        let options = |codec| SwEncoderOptions {
            codec,
            width: 320,
            height: 180,
            pixel_format: ffmpeg::format::Pixel::YUV420P,
            frame_rate: ffmpeg::Rational::new(30, 1),
            bit_rate: 400_000,
            gop_size: 30,
            max_b_frames: None,
        };
        // Either software H.264 will do; a build carrying neither is one this
        // cannot be asked about, so it skips the way a hardware test does.
        let Some(encoder) = [VideoCodec::OpenH264, VideoCodec::H264]
            .into_iter()
            .find_map(|codec| SwEncoder::new("video", options(codec)).ok())
        else {
            eprintln!("skipping: this FFmpeg build has no libopenh264 or libx264");
            return;
        };

        let dir = std::env::temp_dir();
        let path = dir.join(format!("file_muxer_mkv_video_{}.mkv", std::process::id()));
        let _ = std::fs::remove_file(&path);

        let mut muxer = FileMuxer::create(&path, Mp4Layout::Standard).expect("the muxer must open");
        let _video = muxer
            .add_stream("video", &encoder)
            .expect("add_stream must succeed");
        // `open` is the whole assertion: it is `avformat_write_header`, and
        // that is what refuses a video track it has no `CodecPrivate` for.
        // Nothing is read back afterwards because a Matroska file with no
        // cluster in it is not readable at all — proving that would mean
        // encoding frames, which is a different element's contract.
        let sinks = muxer
            .open()
            .expect("Matroska must accept a video track's header");
        drop(sinks);
        std::fs::remove_file(&path).ok();
    }

    /// Opus in Matroska reads back as Opus.
    ///
    /// Not a guard on the global-header flag, which is what it was written to
    /// check: FFmpeg's `libopus` wrapper writes `OpusHead` into `extradata`
    /// whether or not the flag is set, so this passes with the flag reverted.
    /// Unlike the video track above, Opus was never at risk here.
    ///
    /// Kept because the pairing is worth an actual check rather than an
    /// assumption from the AAC one: `libopus` is an external library, the
    /// only audio codec here that a build can be missing, and Matroska is the
    /// container that writes a `CodecPrivate` for it up front.
    ///
    /// Skips where the build has no `libopus`, which is a real configuration
    /// — which is also why the application probes for it rather than
    /// offering it blind.
    #[test]
    fn opus_carries_its_own_header_into_matroska() {
        // 48 kHz because libopus takes nothing else.
        let Some(mut encoder) = open_opus_encoder(48_000, 2) else {
            eprintln!("skipping: this FFmpeg build has no libopus");
            return;
        };

        let dir = std::env::temp_dir();
        let path = dir.join(format!("file_muxer_opus_mkv_{}.mkv", std::process::id()));
        let _ = std::fs::remove_file(&path);

        let mut muxer = FileMuxer::create(&path, Mp4Layout::Standard).expect("the muxer must open");
        let audio = muxer
            .add_stream(
                "audio",
                TrackFormat::new(encoder.parameters(), ffmpeg::Rational::new(1, 48_000)),
            )
            .expect("add_stream must succeed");
        let mut sinks = muxer
            .open()
            .expect("Matroska must accept an Opus track's header");
        encoder.src_pads()[0].link(sinks.take(audio).expect("the muxer's own track").into_raw());

        for tick in 0..20i64 {
            encoder
                .consume(MediaBuffer::Audio(
                    Arc::new(silent_frame(48_000, 2, 960, tick * 960)).into(),
                ))
                .expect("consume must succeed");
        }
        crate::stream::deliver(&mut encoder, &crate::stream::StreamEvent::Eos)
            .expect("eos must flush cleanly");
        drop(encoder);

        let input = ffmpeg::format::input(&path).expect("the file must be readable");
        assert!(
            input.format().name().contains("matroska"),
            "got {:?}",
            input.format().name()
        );
        let stream = input
            .streams()
            .best(ffmpeg::media::Type::Audio)
            .expect("the file must hold the audio track it was given");
        assert_eq!(
            stream.parameters().id(),
            ffmpeg::codec::Id::OPUS,
            "the track must read back as Opus, not as whatever the header defaulted to"
        );
        std::fs::remove_file(&path).ok();
    }

    /// A copied stream given an orientation comes out of the file saying
    /// so — to the demuxer, and on every picture decoded from it — and one
    /// given upright again says nothing: what a re-encoded portrait
    /// recording needs to be shown the right way up.
    #[test]
    fn a_track_carries_the_orientation_it_is_given() {
        use crate::elements::{FileDemuxer, SwDecoder};
        use crate::orientation::{Orientation, Rotation};
        use crate::pipeline::Pipeline;

        let Some(source) = crate::test_support::try_test_video() else {
            return;
        };
        let turned = Orientation::rotated(Rotation::Clockwise90);
        for (name, orientation) in [("turned", turned), ("upright", Orientation::UPRIGHT)] {
            let path = std::env::temp_dir().join(format!("media-pp-orientation-{name}.mp4"));
            let _ = std::fs::remove_file(&path);
            let (demuxer, _) = FileDemuxer::open("demuxer", &source).expect("open the fixture");
            let video = demuxer.best(ffmpeg::media::Type::Video).expect("video");
            // Turned first, then set again: the last one given is the one kept.
            let format = TrackFormat::from(&video)
                .with_orientation(Orientation::rotated(Rotation::Half))
                .with_orientation(orientation);
            let mut muxer = FileMuxer::create(&path, Mp4Layout::Standard).expect("create");
            let track = muxer.add_stream("video", format).expect("add");
            let sink = muxer.open().expect("header").take(track).expect("track");
            let index = video.index;
            let (pipeline, ()) = Pipeline::new("remux", demuxer, move |source, context| {
                let branch = context.branch().to(sink)?;
                context.attach(source, index, branch)?;
                Ok(())
            })
            .expect("wire");
            pipeline.run().expect("run");
            for event in pipeline.bus().iter() {
                if matches!(event, crate::bus::BusEvent::Eos { .. }) {
                    break;
                }
            }
            pipeline.stop();

            let (written, _) =
                FileDemuxer::open("written", path.to_str().unwrap()).expect("reopen");
            let stream = written.best(ffmpeg::media::Type::Video).expect("video");
            assert_eq!(stream.orientation(), Ok(orientation), "{name}: the stream");
            let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            let kept = seen.clone();
            let decoder = SwDecoder::new("decoder", stream.parameters.clone()).expect("decoder");
            let sink = crate::elements::AppSink::new("sink", move |buf| {
                if let MediaBuffer::Video(frame) = &buf {
                    kept.lock().unwrap().push(Orientation::of(frame));
                }
                Ok(())
            });
            let index = stream.index;
            let (pipeline, ()) = Pipeline::new("decode", written, move |source, context| {
                let branch = context.branch().pipe(decoder).to(sink)?;
                context.attach(source, index, branch)?;
                Ok(())
            })
            .expect("wire");
            pipeline.run().expect("run");
            for event in pipeline.bus().iter() {
                if matches!(event, crate::bus::BusEvent::Finished) {
                    break;
                }
            }
            pipeline.stop();
            let seen = seen.lock().unwrap();
            assert!(!seen.is_empty(), "{name}: pictures were decoded");
            assert!(
                seen.iter().all(|read| *read == Ok(orientation)),
                "{name}: every picture"
            );
            let _ = std::fs::remove_file(&path);
        }
    }

    /// Packets whose `dts` and `pts` differ, remuxed, still differ.
    ///
    /// A B-frame is coded from frames on both sides of it, so a container
    /// holding any carries its packets in decode order and their timestamps
    /// stop being the same number. A muxer that wrote `pts` into both — or
    /// that reordered on the way through — produces a file whose decode order
    /// no longer matches its timestamps, and every player of it either stalls
    /// or shows the frames in the wrong order.
    ///
    /// The ordinary fixture cannot show this: `libopenh264` emits no
    /// B-frames. This one is MPEG-4 Part 2 for that reason alone — see
    /// `test_support::synthesize_reordered`.
    #[test]
    fn a_reordered_stream_keeps_its_decode_order_through_the_muxer() {
        use crate::elements::FileDemuxer;
        use crate::pipeline::Pipeline;

        let fixture = crate::test_support::synthesize_reordered("reordered", 3.0);
        let source = fixture.path.to_string_lossy().into_owned();

        // The fixture has to actually be reordered, or the rest of this
        // asserts nothing. Measured rather than assumed: whether an encoder
        // honours `max_b_frames` is the encoder's business, not this crate's.
        let arriving = timestamps(&source);
        assert!(
            arriving.iter().any(|(dts, pts)| dts != pts),
            "the fixture carries no reordering to preserve: {:?}",
            &arriving[..arriving.len().min(8)]
        );

        let path = std::env::temp_dir().join("media-pp-reordered-remux.mp4");
        let _ = std::fs::remove_file(&path);
        let (demuxer, _) = FileDemuxer::open("demuxer", &source).expect("open the fixture");
        let video = demuxer
            .best(ffmpeg::media::Type::Video)
            .expect("the fixture has video");
        let parameters = video.parameters.clone();
        let time_base = video.time_base;
        let video = video.index;

        let mut muxer = FileMuxer::create(&path, Mp4Layout::Standard).expect("create the remux");
        let track = muxer
            .add_stream("video", TrackFormat::new(parameters, time_base))
            .expect("add the video stream");
        let sink = muxer
            .open()
            .expect("write the header")
            .take(track)
            .expect("the muxer's own track");

        let (pipeline, ()) = Pipeline::new("remux", demuxer, move |source, context| {
            let branch = context.branch().to(sink)?;
            context.attach(source, video, branch)?;
            Ok(())
        })
        .expect("wire the remux");
        pipeline.run().expect("run the remux");
        for event in pipeline.bus().iter() {
            if matches!(event, crate::bus::BusEvent::Eos { .. }) {
                break;
            }
        }
        pipeline.stop();

        let written = timestamps(&path.to_string_lossy());
        assert_eq!(
            written.len(),
            arriving.len(),
            "every packet that came in has to come out"
        );
        assert!(
            written.iter().any(|(dts, pts)| dts != pts),
            "the reordering did not survive being written"
        );
        assert!(
            written.windows(2).all(|pair| pair[0].0 <= pair[1].0),
            "decode order is what `dts` is for and it has to keep rising: {:?}",
            &written[..written.len().min(8)]
        );
        assert_eq!(
            written, arriving,
            "a remux copies packets; it does not restamp them"
        );
        std::fs::remove_file(&path).ok();
    }

    /// Every video packet's `(dts, pts)`, in the order the container holds
    /// them.
    fn timestamps(path: &str) -> Vec<(i64, i64)> {
        let mut input = ffmpeg::format::input(path).expect("the file opens");
        let video = input
            .streams()
            .find(|stream| stream.parameters().medium() == ffmpeg::media::Type::Video)
            .expect("it has video")
            .index();
        input
            .packets()
            .filter(|(stream, _)| stream.index() == video)
            .filter_map(|(_, packet)| Some((packet.dts()?, packet.pts()?)))
            .collect()
    }

    /// What a written file's timeline looks like from outside it.
    #[derive(Debug, Clone, Copy)]
    struct Shape {
        video_frames: usize,
        video_seconds: f64,
        audio_seconds: f64,
        video_start: f64,
        audio_start: f64,
    }

    impl Shape {
        fn video_fps(self) -> f64 {
            self.video_frames as f64 / self.video_seconds
        }
    }

    /// Reads one back, the way anything that plays it would.
    fn shape_of(path: &str) -> Shape {
        let input = ffmpeg::format::input(path).expect("the file opens");
        let stream = |medium| {
            input
                .streams()
                .find(|stream| stream.parameters().medium() == medium)
                .unwrap_or_else(|| panic!("{path} carries no {medium:?}"))
        };
        let seconds = |stream: &ffmpeg::format::stream::Stream<'_>| {
            stream.duration() as f64 * f64::from(stream.time_base())
        };
        let start = |stream: &ffmpeg::format::stream::Stream<'_>| {
            stream.start_time() as f64 * f64::from(stream.time_base())
        };
        let video = stream(ffmpeg::media::Type::Video);
        let audio = stream(ffmpeg::media::Type::Audio);
        Shape {
            video_frames: video.frames() as usize,
            video_seconds: seconds(&video),
            audio_seconds: seconds(&audio),
            video_start: start(&video),
            audio_start: start(&audio),
        }
    }

    /// A file decoded, re-encoded and written back keeps the shape it had.
    ///
    /// The path a recording really takes, and the one every defect found in
    /// this crate's timeline handling has lived on. What makes it worth a
    /// test of its own is that each of those was invisible from inside: every
    /// element returned `Ok`, every buffer went where it was sent, and the
    /// file that came out was the wrong length, or its sound no longer sat
    /// against its picture. None of that can be seen without reading the
    /// result back.
    ///
    /// The tolerances are deliberately loose. What this is watching for is a
    /// stream that lost or gained *time* — an encoder dropping samples, a
    /// muxer restamping a track, a time base that means something different
    /// at each end — not the frame or two an encoder is entitled to hold.
    #[test]
    fn a_transcoded_file_keeps_the_shape_of_what_went_in() {
        use crate::elements::{FileDemuxer, SwDecoder, SwEncoder, SwEncoderOptions, VideoCodec};
        use crate::pipeline::Pipeline;

        let fixture = crate::test_support::synthesize("transcode-shape", 4.0, 44_100);
        let source_path = fixture.path.to_string_lossy().into_owned();
        let arriving = shape_of(&source_path);

        let path = std::env::temp_dir().join("media-pp-transcode-shape.mp4");
        let _ = std::fs::remove_file(&path);

        let (demuxer, _) = FileDemuxer::open("demuxer", &source_path).expect("open");
        let stream = |medium| {
            demuxer
                .best(medium)
                .unwrap_or_else(|_| panic!("the fixture carries no {medium:?}"))
        };
        let (video, audio) = (
            stream(ffmpeg::media::Type::Video),
            stream(ffmpeg::media::Type::Audio),
        );
        let video_decoder = SwDecoder::new("video-decoder", video.parameters.clone())
            .expect("open the video decoder");
        let audio_decoder = SwDecoder::new("audio-decoder", audio.parameters.clone())
            .expect("open the audio decoder");
        let (video, audio) = (video.index, audio.index);

        // Re-encoded at the same rates it arrived with, so a difference in
        // the result is this crate's doing rather than a conversion's.
        let width = 320;
        let height = 240;
        // The container's own unit, not `1/frame_rate`: what the decoder
        // hands over is stamped in whatever the container counts in, and an
        // encoder told a different unit writes those same numbers meaning
        // something else. Which is not hypothetical — the first version of
        // this test said `1/30` and produced 121 frames across 34 minutes.
        let video_encoder = SwEncoder::new(
            "video-encoder",
            SwEncoderOptions {
                codec: VideoCodec::OpenH264,
                width,
                height,
                pixel_format: ffmpeg::format::Pixel::YUV420P,
                frame_rate: ffmpeg::Rational::new(30, 1),
                bit_rate: 800_000,
                gop_size: 30,
                max_b_frames: None,
            },
        )
        .expect("open the video encoder");
        // Written at 48kHz from a 44.1kHz source, which is what a recording
        // really does — a file's own rate is rarely the one everything else
        // in the graph runs at. It also puts a rate conversion inside what
        // this measures, and an audio path losing a fraction of every frame
        // to one is a defect this crate has actually had.
        let audio_encoder = open_aac_encoder(48_000, 2);

        let mut muxer = FileMuxer::create(&path, Mp4Layout::Standard).expect("create the output");
        let video_track = muxer
            .add_stream("video", &video_encoder)
            .expect("add the video stream");
        let audio_track = muxer
            .add_stream("audio", &audio_encoder)
            .expect("add the audio stream");
        let mut sinks = muxer.open().expect("write the header");
        let video_sink = sinks.take(video_track).expect("the muxer's own track");
        let audio_sink = sinks.take(audio_track).expect("the muxer's own track");

        let scaler = crate::elements::SwScaler::new(
            "to-yuv",
            ffmpeg::format::Pixel::YUV420P,
            width,
            height,
            ffmpeg::software::scaling::Flags::BILINEAR,
        );

        let (pipeline, ()) = Pipeline::new("transcode", demuxer, move |source, context| {
            let picture = context
                .branch()
                .pipe(video_decoder)
                .pipe(scaler)
                .pipe(video_encoder)
                .to(video_sink)?;
            context.attach(source, video, picture)?;
            let sound = context
                .branch()
                .pipe(audio_decoder)
                .pipe(audio_encoder)
                .to(audio_sink)?;
            context.attach(source, audio, sound)?;
            Ok(())
        })
        .expect("wire the transcode");
        pipeline.run().expect("run the transcode");
        for event in pipeline.bus().iter() {
            if matches!(event, crate::bus::BusEvent::Eos { .. }) {
                break;
            }
        }
        pipeline.stop();

        let written = shape_of(&path.to_string_lossy());
        eprintln!("SHAPE in : {arriving:?} fps={:.3}", arriving.video_fps());
        eprintln!("SHAPE out: {written:?} fps={:.3}", written.video_fps());

        assert!(
            (written.video_fps() - arriving.video_fps()).abs() < 0.5,
            "the picture came out at a different rate: {:.3} in, {:.3} out",
            arriving.video_fps(),
            written.video_fps()
        );
        assert!(
            (written.video_seconds - arriving.video_seconds).abs() < 0.25,
            "the picture came out a different length: {:.3}s in, {:.3}s out",
            arriving.video_seconds,
            written.video_seconds
        );
        assert!(
            (written.audio_seconds - written.video_seconds).abs() < 0.25,
            "the sound and the picture came out different lengths: \
             {:.3}s of sound against {:.3}s of picture",
            written.audio_seconds,
            written.video_seconds
        );
        assert!(
            (written.audio_start - written.video_start).abs()
                <= (arriving.audio_start - arriving.video_start).abs() + 0.05,
            "the two tracks no longer start where they did: \
             in {:.3}s/{:.3}s, out {:.3}s/{:.3}s",
            arriving.video_start,
            arriving.audio_start,
            written.video_start,
            written.audio_start
        );
        std::fs::remove_file(&path).ok();
    }
    /// Three tracks in one file, with the subtitles arriving after the
    /// video and audio have already finished.
    ///
    /// This is the shape a transcriber forces. Inference lags the audio it
    /// reads by seconds, so a subtitle for the fifth second is handed over
    /// when the fortieth is being written — and here, to put the question
    /// past doubt, every video and audio packet is written and ended before
    /// the first line of text arrives at all.
    ///
    /// What makes that legal is that a packet's place in the file is its
    /// timestamp and not its arrival: the muxer interleaves by `dts`, and
    /// MP4's index is written at the end, so a sample landing late in
    /// `mdat` is still indexed at the moment it belongs to. The assertion
    /// is that the timestamps survive and the other two tracks are
    /// undamaged by the wait.
    #[test]
    fn subtitles_arriving_after_everything_else_still_land_on_their_own_timestamps() {
        let fixture = crate::test_support::synthesize("late-subtitles", 4.0, 44100);
        let mut input = ffmpeg::format::input(&fixture.path).expect("open the fixture");
        let video = input
            .streams()
            .best(ffmpeg::media::Type::Video)
            .expect("the fixture has video");
        let audio = input
            .streams()
            .best(ffmpeg::media::Type::Audio)
            .expect("the fixture has audio");
        let (video_index, video_params, video_tb) =
            (video.index(), video.parameters(), video.time_base());
        let (audio_index, audio_params, audio_tb) =
            (audio.index(), audio.parameters(), audio.time_base());

        let path = std::env::temp_dir().join(format!(
            "file_muxer_subtitles_{}_{:?}.mp4",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_file(&path);

        // Milliseconds for the text track, which is its own timeline and
        // has no reason to share the video's or the audio's.
        let text_tb = ffmpeg::Rational::new(1, 1000);
        let mut muxer = FileMuxer::create(&path, Mp4Layout::Standard).expect("the muxer must open");
        let video_track = muxer
            .add_stream("video", TrackFormat::new(video_params, video_tb))
            .expect("add_stream video");
        let audio_track = muxer
            .add_stream("audio", TrackFormat::new(audio_params, audio_tb))
            .expect("add_stream audio");
        let text_track = muxer
            .add_stream(
                "text",
                TrackFormat::new(crate::subtitle::Codec::MovText.parameters(), text_tb),
            )
            .expect("add_stream text");
        let mut sinks = muxer.open().expect("open must write the header");
        // Taken in neither the order they were added nor its reverse: which
        // sink is which is the track's to say, not the position's.
        let mut audio_sink = sinks.take(audio_track).expect("the muxer's own track");
        let mut text_sink = sinks.take(text_track).expect("the muxer's own track");
        let mut video_sink = sinks.take(video_track).expect("the muxer's own track");

        assert_eq!(
            text_sink.input_contract(),
            InputContract::Fixed(PortContract::packet(MediaKind::SubtitlePacket)),
            "a subtitle track states what it takes, like every other track"
        );

        for (stream, packet) in input.packets() {
            let buffer = MediaBuffer::Packet(Arc::new(packet).into());
            if stream.index() == video_index {
                video_sink.consume(buffer).expect("video packet");
            } else if stream.index() == audio_index {
                audio_sink.consume(buffer).expect("audio packet");
            }
        }
        video_sink
            .stream_event(&crate::stream::StreamEvent::Eos)
            .expect("video eos");
        audio_sink
            .stream_event(&crate::stream::StreamEvent::Eos)
            .expect("audio eos");

        // Only now, with the rest of the file already written.
        text_sink
            .consume(crate::subtitle::Codec::MovText.packet("첫 번째 자막", 1_000, 1_000))
            .expect("first line");
        text_sink
            .consume(crate::subtitle::Codec::MovText.packet("second line", 2_500, 1_500))
            .expect("second line");
        text_sink
            .stream_event(&crate::stream::StreamEvent::Eos)
            .expect("text eos");

        let written = ffmpeg::format::input(&path).expect("the file must be readable");
        let kinds: Vec<_> = written
            .streams()
            .map(|stream| stream.parameters().medium())
            .collect();
        assert_eq!(
            kinds,
            vec![
                ffmpeg::media::Type::Video,
                ffmpeg::media::Type::Audio,
                ffmpeg::media::Type::Subtitle
            ],
            "three tracks, in the order they were registered"
        );

        let text_stream = written
            .streams()
            .best(ffmpeg::media::Type::Subtitle)
            .expect("a subtitle stream");
        assert_eq!(text_stream.parameters().id(), ffmpeg::codec::Id::MOV_TEXT);
        let text_index = text_stream.index();
        let written_tb = text_stream.time_base();

        // Read the lines back and check where they landed. The muxer fills
        // the gaps between them with empty samples of its own, so this
        // takes only the ones carrying text.
        let mut written = written;
        let mut lines: Vec<(i64, String)> = Vec::new();
        for (stream, packet) in written.packets() {
            if stream.index() != text_index {
                continue;
            }
            let data = packet.data().expect("a sample has bytes");
            let length = u16::from_be_bytes([data[0], data[1]]) as usize;
            if length == 0 {
                continue;
            }
            let text = std::str::from_utf8(&data[2..2 + length])
                .expect("valid utf-8")
                .to_owned();
            let pts = packet.pts().expect("a sample carries its time");
            // Back into milliseconds, whatever base the file chose.
            let millis =
                pts * 1000 * written_tb.numerator() as i64 / written_tb.denominator() as i64;
            lines.push((millis, text));
        }

        assert_eq!(
            lines,
            vec![
                (1_000, "첫 번째 자막".to_owned()),
                (2_500, "second line".to_owned())
            ],
            "the text and its timing survived arriving last"
        );

        std::fs::remove_file(&path).ok();
    }

    /// A software H.264 encoder of a small picture, a keyframe every
    /// `gop` pictures; `None` where the build has neither, as above.
    fn open_h264_encoder(gop: u32) -> Option<crate::elements::SwEncoder> {
        use crate::elements::{SwEncoder, SwEncoderOptions, VideoCodec};
        [VideoCodec::OpenH264, VideoCodec::H264]
            .into_iter()
            .find_map(|codec| {
                SwEncoder::new(
                    "video",
                    SwEncoderOptions {
                        codec,
                        width: 160,
                        height: 96,
                        pixel_format: ffmpeg::format::Pixel::YUV420P,
                        frame_rate: ffmpeg::Rational::new(30, 1),
                        bit_rate: 200_000,
                        gop_size: gop,
                        max_b_frames: None,
                    },
                )
                .ok()
            })
    }

    /// Picture `index` of a 30-a-second stream, each a different shade.
    fn picture(index: i64) -> MediaBuffer {
        let pool = crate::pool::UnboundObjectPool::new(
            0,
            || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::YUV420P, 160, 96),
            |_| {},
        );
        let mut frame = pool.get();
        for plane in 0..3 {
            frame.data_mut(plane).fill((index * 7 % 200) as u8 + 16);
        }
        frame.set_pts(Some(index));
        crate::buffer::set_time_base(&mut frame, ffmpeg::Rational::new(1, 30));
        MediaBuffer::Video(Arc::new(frame).into())
    }

    /// The four-letter types of a file's top-level boxes, in order.
    fn top_level_boxes(path: &Path) -> Vec<String> {
        let bytes = std::fs::read(path).expect("the file must be there");
        let mut boxes = Vec::new();
        let mut at = 0usize;
        while at + 8 <= bytes.len() {
            let mut size = u32::from_be_bytes(bytes[at..at + 4].try_into().unwrap()) as u64;
            boxes.push(String::from_utf8_lossy(&bytes[at + 4..at + 8]).into_owned());
            if size == 1 && at + 16 <= bytes.len() {
                size = u64::from_be_bytes(bytes[at + 8..at + 16].try_into().unwrap());
            }
            if size < 8 {
                break;
            }
            at += size as usize;
        }
        boxes
    }

    /// How many packets of each track a file reads back with; `None` where
    /// it does not open at all.
    fn packets_read_back(path: &Path) -> Option<Vec<usize>> {
        let mut input = ffmpeg::format::input(path).ok()?;
        let mut counts = vec![0usize; input.streams().count()];
        let mut packet = ffmpeg::Packet::empty();
        while packet.read(&mut input).is_ok() {
            counts[packet.stream()] += 1;
            packet = ffmpeg::Packet::empty();
        }
        Some(counts)
    }

    /// Only an MP4 or MOV name takes a layout other than the standard one;
    /// any other is refused as that, before a file is made.
    #[test]
    fn only_an_mp4_or_mov_name_takes_a_layout() {
        for layout in [Mp4Layout::FastStart, Mp4Layout::Fragmented] {
            let path = std::env::temp_dir().join(format!(
                "file_muxer_layout_refused_{layout:?}_{}.mkv",
                std::process::id()
            ));
            let _ = std::fs::remove_file(&path);
            let Err(error) = FileMuxer::create(&path, layout) else {
                panic!("a .mkv was taken to be laid out {layout:?}");
            };
            assert!(
                matches!(error, Error::FileMuxerError(FileMuxerError::NotMp4 { .. })),
                "{error:?}"
            );
            assert!(!path.exists(), "nothing is created for it");

            let mov = std::env::temp_dir().join(format!(
                "file_muxer_layout_mov_{layout:?}_{}.mov",
                std::process::id()
            ));
            drop(FileMuxer::create(&mov, layout).expect("a .mov takes the layout"));
            std::fs::remove_file(&mov).ok();
        }
    }

    /// Laid out fast-start, the index of a finished file comes before its
    /// samples, where the standard layout puts it after them; every packet
    /// reads back from both.
    #[test]
    fn a_fast_start_file_has_its_index_before_its_samples() {
        for (layout, index_first) in [(Mp4Layout::FastStart, true), (Mp4Layout::Standard, false)] {
            let Some(mut video) = open_h264_encoder(15) else {
                eprintln!("skipping: this FFmpeg build has no libopenh264 or libx264");
                return;
            };
            let path = std::env::temp_dir().join(format!(
                "file_muxer_layout_{layout:?}_{}.mp4",
                std::process::id()
            ));
            let _ = std::fs::remove_file(&path);
            let mut muxer = FileMuxer::create(&path, layout).expect("the muxer must open");
            let track = muxer.add_stream("video", &video).expect("add video");
            let mut sinks = muxer.open().expect("open must write the header");
            video.src_pads()[0].link(sinks.take(track).unwrap().into_raw());
            for index in 0..60i64 {
                video.consume(picture(index)).expect("encode a picture");
            }
            crate::stream::deliver(&mut video, &crate::stream::StreamEvent::Eos).unwrap();
            drop(video);

            let boxes = top_level_boxes(&path);
            let moov = boxes.iter().position(|b| b == "moov").expect("an index");
            let mdat = boxes.iter().position(|b| b == "mdat").expect("samples");
            assert_eq!(moov < mdat, index_first, "{layout:?}: {boxes:?}");
            assert_eq!(
                packets_read_back(&path),
                Some(vec![60]),
                "{layout:?}: every picture reads back"
            );
            std::fs::remove_file(&path).ok();
        }
    }

    /// Video and sound written to the end in fragments: the track list
    /// first, then a fragment at each keyframe, and every packet reads
    /// back.
    #[test]
    fn a_fragmented_file_finished_holds_every_packet_in_fragments() {
        let Some(mut video) = open_h264_encoder(15) else {
            eprintln!("skipping: this FFmpeg build has no libopenh264 or libx264");
            return;
        };
        let mut audio = open_aac_encoder(48_000, 1);
        let path =
            std::env::temp_dir().join(format!("file_muxer_fragmented_{}.mp4", std::process::id()));
        let _ = std::fs::remove_file(&path);

        let mut muxer =
            FileMuxer::create(&path, Mp4Layout::Fragmented).expect("the muxer must open");
        let video_track = muxer.add_stream("video", &video).expect("add video");
        let audio_track = muxer
            .add_stream(
                "audio",
                TrackFormat::new(audio.parameters(), ffmpeg::Rational::new(1, 48_000)),
            )
            .expect("add audio");
        let mut sinks = muxer.open().expect("open must write the header");
        video.src_pads()[0].link(sinks.take(video_track).unwrap().into_raw());
        audio.src_pads()[0].link(sinks.take(audio_track).unwrap().into_raw());

        // Two seconds: four keyframes, and the sound beside them.
        for index in 0..60i64 {
            video.consume(picture(index)).expect("encode a picture");
            audio
                .consume(MediaBuffer::Audio(
                    Arc::new(silent_frame(48_000, 1, 1600, index * 1600)).into(),
                ))
                .expect("encode its sound");
        }
        crate::stream::deliver(&mut video, &crate::stream::StreamEvent::Eos).unwrap();
        crate::stream::deliver(&mut audio, &crate::stream::StreamEvent::Eos).unwrap();
        drop((video, audio));

        let boxes = top_level_boxes(&path);
        let moov = boxes
            .iter()
            .position(|b| b == "moov")
            .expect("a track list");
        let first_moof = boxes.iter().position(|b| b == "moof").expect("a fragment");
        assert!(moov < first_moof, "the track list comes first: {boxes:?}");
        assert!(
            boxes.iter().filter(|b| *b == "moof").count() >= 4,
            "a fragment at each keyframe: {boxes:?}"
        );
        let counts = packets_read_back(&path).expect("the file must open");
        assert_eq!(counts[0], 60, "every picture reads back");
        assert!(counts[1] > 0, "the sound reads back: {counts:?}");
        std::fs::remove_file(&path).ok();
    }

    /// The point of fragments: a recording never finished still opens, with
    /// the pictures of every fragment before the one under way. What is
    /// read is a copy of the file taken while it is still being written —
    /// what a process that dies leaves on disk, FFmpeg's own buffer never
    /// handed over — and then the file itself once its sinks are dropped
    /// with no `Eos` and no `Stop`. The same recording written plainly
    /// opens in neither, or this would prove nothing.
    #[test]
    fn a_fragmented_recording_cut_off_still_plays() {
        let cut_off = |fragmented: bool| {
            let mut video = open_h264_encoder(15)?;
            let path = std::env::temp_dir().join(format!(
                "file_muxer_cut_off_{fragmented}_{}.mp4",
                std::process::id()
            ));
            let _ = std::fs::remove_file(&path);
            let mut muxer = if fragmented {
                FileMuxer::create(&path, Mp4Layout::Fragmented)
            } else {
                FileMuxer::create(&path, Mp4Layout::Standard)
            }
            .expect("the muxer must open");
            let track = muxer.add_stream("video", &video).expect("add video");
            let mut sinks = muxer.open().expect("open must write the header");
            video.src_pads()[0].link(sinks.take(track).unwrap().into_raw());
            // Four keyframe intervals and a half: the last fragment is
            // still being gathered when the recording is cut off.
            for index in 0..68i64 {
                video.consume(picture(index)).expect("encode a picture");
            }
            let snapshot = path.with_extension("snapshot.mp4");
            std::fs::copy(&path, &snapshot).expect("copy the file being written");
            drop(sinks);
            drop(video);
            let read = (packets_read_back(&snapshot), packets_read_back(&path));
            std::fs::remove_file(&snapshot).ok();
            std::fs::remove_file(&path).ok();
            Some(read)
        };
        let Some((snapshot, dropped)) = cut_off(true) else {
            eprintln!("skipping: this FFmpeg build has no libopenh264 or libx264");
            return;
        };
        for (what, counts) in [
            ("a copy taken while writing", snapshot),
            ("dropped", dropped),
        ] {
            let counts =
                counts.unwrap_or_else(|| panic!("{what}: a fragmented recording must open"));
            assert!(
                counts[0] >= 60,
                "{what}: every fragment before the one under way plays: {counts:?}"
            );
        }
        assert_eq!(
            cut_off(false).unwrap(),
            (None, None),
            "a plain MP4 cut off does not open, or this proves nothing"
        );
    }

    /// Sound alone has no keyframes to cut at, so it is cut every second:
    /// cut off, it still plays up to its last whole second.
    #[test]
    fn a_fragmented_recording_of_sound_alone_cut_off_still_plays() {
        let mut audio = open_aac_encoder(48_000, 1);
        let path = std::env::temp_dir().join(format!(
            "file_muxer_cut_off_sound_{}.mp4",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let mut muxer =
            FileMuxer::create(&path, Mp4Layout::Fragmented).expect("the muxer must open");
        let track = muxer
            .add_stream(
                "audio",
                TrackFormat::new(audio.parameters(), ffmpeg::Rational::new(1, 48_000)),
            )
            .expect("add audio");
        let mut sinks = muxer.open().expect("open must write the header");
        audio.src_pads()[0].link(sinks.take(track).unwrap().into_raw());
        // Three seconds and a half.
        for tick in 0..175i64 {
            audio
                .consume(MediaBuffer::Audio(
                    Arc::new(silent_frame(48_000, 1, 960, tick * 960)).into(),
                ))
                .expect("encode sound");
        }
        drop(sinks);
        drop(audio);

        let counts = packets_read_back(&path).expect("sound cut off must open");
        // 1024 samples a packet: three whole seconds are over 140.
        assert!(counts[0] >= 140, "every whole second plays: {counts:?}");
        std::fs::remove_file(&path).ok();
    }
}
