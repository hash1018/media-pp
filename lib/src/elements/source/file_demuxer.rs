use std::{
    collections::VecDeque,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicI64, Ordering},
    },
    time::Duration,
};

use crate::pp_log::{PpLog, pp_debug, pp_error, pp_info};
use ffmpeg_next::{self as ffmpeg, Rescale};
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    contract::{MediaKind, OutputContract, PortContract},
    element::{
        Element, ElementType, Produced, ReversibleSource, SeekableSource, Source, SourceStage,
        Wait, element_pp_log,
    },
    pad::SrcPad,
    produce::source_stage,
};

/// Errors specific to `FileDemuxer`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum FileDemuxerError {
    /// The file could not be opened as a media file: it is not there, not
    /// readable, or not a container FFmpeg knows.
    #[error("could not open {}: {source}", path.display())]
    Open {
        /// The file asked for.
        path: std::path::PathBuf,
        /// What FFmpeg said.
        #[source]
        source: ffmpeg::Error,
    },

    /// The file opened and holds no streams at all — a recording that was
    /// finished before anything was written into it, say.
    #[error("{} has no streams to play", path.display())]
    NoStreams {
        /// The file asked for.
        path: std::path::PathBuf,
    },

    /// FFmpeg rejected reading or seeking the input container.
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),

    /// The file has no stream of the kind asked for — see
    /// [`FileDemuxer::best`].
    #[error("the file has no {0:?} stream")]
    NoStream(ffmpeg::media::Type),
}

/// Metadata about one stream in an opened container, reported up front so
/// callers can decide what to build downstream before the pipeline runs —
/// everything a branch for it is built from, without asking again by index.
#[derive(Clone)]
pub struct StreamInfo {
    /// Zero-based stream index used by the matching source pad.
    pub index: usize,
    /// Media kind reported by the container, such as audio or video.
    pub kind: ffmpeg::media::Type,
    /// What a decoder for it is built from — [`crate::elements::SwDecoder`],
    /// [`crate::elements::VideoDecodeBin`], and the rest.
    pub parameters: ffmpeg::codec::Parameters,
    /// The unit its packets' timestamps are in — what a
    /// [`crate::elements::Pacer`] for it is built with.
    pub time_base: ffmpeg::Rational,
    /// How many pictures a second a video stream has on average, as the
    /// container says — what an encoder re-encoding it is opened at. `None`
    /// for sound, and for a video stream that does not say.
    pub frame_rate: Option<ffmpeg::Rational>,
}

impl StreamInfo {
    /// What one of a container's streams says about itself.
    pub(crate) fn of(stream: &ffmpeg::format::stream::Stream<'_>) -> Self {
        // A copy of its own. What a stream hands out shares the whole
        // input's ownership through an `Rc` — so it kept the file (or the
        // RTSP connection) open for as long as the caller held this, and,
        // this going to one thread and the source to another, counted that
        // `Rc` from two threads at once.
        let parameters = stream.parameters().clone();
        let kind = parameters.medium();
        let rate = stream.avg_frame_rate();
        Self {
            index: stream.index(),
            kind,
            frame_rate: (kind == ffmpeg::media::Type::Video
                && rate.numerator() > 0
                && rate.denominator() > 0)
                .then_some(rate),
            parameters,
            time_base: stream.time_base(),
        }
    }

    /// What a video stream says its Y'CbCr is — matrix, range, primaries
    /// and transfer — for an encoder re-encoding its decoded pictures to
    /// write into its own stream. `None` for sound, and for a video stream
    /// that names no matrix.
    pub fn color(&self) -> Option<crate::color::ColorDescription> {
        // SAFETY: plain fields of parameters this value owns.
        let raw = unsafe { &*self.parameters.as_ptr() };
        let space = ffmpeg::color::Space::from(raw.color_space);
        (self.kind == ffmpeg::media::Type::Video && space != ffmpeg::color::Space::Unspecified)
            .then(|| crate::color::ColorDescription {
                space,
                range: raw.color_range.into(),
                primaries: raw.color_primaries.into(),
                transfer: raw.color_trc.into(),
            })
    }

    /// Which way up a video stream's pictures are shown — the display
    /// matrix its container keeps beside them, which a phone's portrait
    /// recording carries — and upright where it keeps none, or for sound.
    /// Each decoded picture carries it too, read by
    /// [`Orientation::of`](crate::orientation::Orientation::of); what a file
    /// its pictures are re-encoded to is to say is this, given to
    /// [`TrackFormat::with_orientation`](crate::elements::TrackFormat::with_orientation).
    ///
    /// # Errors
    ///
    /// Where the matrix turns the pictures by other than quarter turns.
    pub fn orientation(
        &self,
    ) -> Result<crate::orientation::Orientation, crate::orientation::UnsupportedOrientation> {
        // SAFETY: plain fields of parameters this value owns; the side data
        // is FFmpeg's own, a display matrix being nine `i32`s by its
        // definition, which the size check makes sure of.
        unsafe {
            let raw = self.parameters.as_ptr();
            let data = ffmpeg::ffi::av_packet_side_data_get(
                (*raw).coded_side_data,
                (*raw).nb_coded_side_data,
                ffmpeg::ffi::AVPacketSideDataType::AV_PKT_DATA_DISPLAYMATRIX,
            );
            if data.is_null() || (*data).size < 9 * size_of::<i32>() {
                return Ok(crate::orientation::Orientation::UPRIGHT);
            }
            crate::orientation::Orientation::of_matrix(&*((*data).data as *const [i32; 9]))
        }
    }

    /// What a video stream's pictures are in, decoded in software — 4:2:0
    /// at eight bits or ten, say — as its parameters say. A hardware
    /// decoder puts the same in its own layout: a 10-bit stream is P010 out
    /// of NVDEC, which [`crate::contract::PixelLayoutSet::decoded_from`]
    /// says. `None` for sound, and for a video stream that does not say.
    pub fn pixel_format(&self) -> Option<ffmpeg::format::Pixel> {
        // SAFETY: a plain field of parameters this value owns.
        let format = unsafe { (*self.parameters.as_ptr()).format };
        let known = 0..ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_NB as i32;
        if self.kind != ffmpeg::media::Type::Video || !known.contains(&format) {
            return None;
        }
        // SAFETY: every value from naught to `AV_PIX_FMT_NB` names one of
        // FFmpeg's pixel formats, which the enum has a variant for each of.
        let format = unsafe { std::mem::transmute::<i32, ffmpeg::ffi::AVPixelFormat>(format) };
        Some(format.into())
    }

    /// A video stream's picture size, width then height, as its parameters
    /// say — what an encoder re-encoding it is opened at. `None` for sound,
    /// and for a video stream that does not say.
    pub fn size(&self) -> Option<(u32, u32)> {
        // SAFETY: plain fields of parameters this value owns.
        let (width, height) = unsafe {
            let raw = self.parameters.as_ptr();
            ((*raw).width, (*raw).height)
        };
        (self.kind == ffmpeg::media::Type::Video && width > 0 && height > 0)
            .then_some((width as u32, height as u32))
    }
}

/// By hand, since FFmpeg's parameters have no `Debug` of their own: the
/// codec stands in for them.
impl std::fmt::Debug for StreamInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamInfo")
            .field("index", &self.index)
            .field("kind", &self.kind)
            .field("codec", &self.parameters.id())
            .field("time_base", &self.time_base)
            .field("frame_rate", &self.frame_rate)
            .field("size", &self.size())
            .finish()
    }
}

/// Runtime control for a [`FileDemuxer`], taken with
/// [`FileDemuxer::looping_handle`] before the demuxer is moved into its
/// pipeline.
///
/// Cheap to clone and safe to share: it holds a few atomics and nothing
/// else, so it keeps neither the demuxer, its file, nor its pipeline alive.
/// No call blocks or does any work beyond that store. A call after the
/// source has finished is simply never read.
#[derive(Clone)]
pub struct FileDemuxerHandle {
    looping: Arc<AtomicBool>,
    published_offset: Arc<AtomicI64>,
    published_lap: Arc<AtomicI64>,
}

impl FileDemuxerHandle {
    /// Whether reaching the end of the file starts it again instead of
    /// ending the stream. Off unless this says otherwise.
    ///
    /// Read once per lap, at the end of the file — never mid-file. So
    /// turning it off part way through means "play this lap out and then
    /// finish", not "stop now", and the stream still ends with a real
    /// `Eos` rather than being abandoned the way a stop
    /// abandons it. Turning it on part way through takes effect at the end
    /// the source was already heading for.
    pub fn set_looping(&self, looping: bool) {
        self.looping.store(looping, Ordering::Relaxed);
    }

    /// What [`FileDemuxerHandle::set_looping`] last set.
    pub fn is_looping(&self) -> bool {
        self.looping.load(Ordering::Relaxed)
    }

    /// How far this source's output timeline has been carried past the
    /// file's own — the sum of every lap already played.
    ///
    /// Zero until the first wrap, so a source that never loops never needs
    /// this. What it is for is reading a timestamp *back*: subtract it from
    /// a packet's or frame's timestamp and the result is a position in the
    /// file, which is what a progress bar means by one. See
    /// [`FileDemuxer`]'s own docs on why the two are not the same number.
    ///
    /// It is the lap the demuxer is *reading*, which is ahead of what is
    /// shown by as much as the queues and a decoder hold: the next lap for
    /// the last of one played forwards, the lap before for the first of one
    /// played backwards, where a decoder holds whole stretches. Played fast
    /// that is seconds of the file, so a timestamp is turned into a position
    /// with [`Self::in_lap`], which reads the lap off the timestamp itself.
    pub fn lap_offset(&self) -> Duration {
        Duration::from_micros(self.published_offset.load(Ordering::Relaxed).max(0) as u64)
    }

    /// `timeline` — a frame's timestamp, or the pipeline's position — as a
    /// position in the file, on whichever lap it is.
    ///
    /// The timeline unchanged until the file has looped; after that, how far
    /// into its lap it is. A lap is as long as the one the timeline was last
    /// carried over, so the start of the next lap is the start of the file.
    pub fn in_lap(&self, timeline: Duration) -> Duration {
        let lap = self.published_lap.load(Ordering::Relaxed);
        if lap <= 0 {
            return timeline;
        }
        let lap = u128::from(lap.unsigned_abs()) * 1_000;
        Duration::from_nanos((timeline.as_nanos() % lap) as u64)
    }
}

/// Demuxes a file, exposing one src pad per container stream (indexed the
/// same way as `StreamInfo::index`). Linking a pad "selects" that stream;
/// leaving it unlinked just drops its packets. Real demuxer I/O is
/// blocking, so this is meant to be run as the pipeline's source thread.
///
/// Fan-out (e.g. routing video and audio to separate branches) needs no
/// separate "Tee" element here — it's just a matter of linking more than
/// one of these pads. A pad that cannot take a packet yet has what is read
/// for it held back while the others go on being fed, within bounds — see
/// the framework's handling of a source with several outputs.
///
/// Set to loop through [`FileDemuxer::looping_handle`] and the end of the
/// file rewinds to the start instead of ending the stream. Timestamps then
/// keep climbing across the join rather than restarting: what a lap already
/// reached is added to every later one, so a `Pacer` still paces, a muxer
/// still sees its timestamps advance, and nothing downstream has to know a
/// join happened. The consequence is that a looping source's timestamps are
/// no longer positions *in the file* — one second into the third lap is at
/// twice the file's length plus a second — and [`FileDemuxer::seek`] stays
/// the way to speak in the file's own timeline.
///
/// At the end of the file it sends `Eos` down every pad and stays, passing
/// `Pause`, `Resume` and the rest on to its branches — whose queues still
/// hold what was read, a second or more of it — until it is stopped or
/// finished; a seek from there reads on from where it lands. So a pipeline
/// reading a file does not end when the file does: it posts
/// [`BusEvent::Finished`](crate::bus::BusEvent::Finished) once everything
/// read has reached its terminals, and it is the caller's to stop it then.
/// Run by hand, [`run`](crate::element::RawSource::run) returns at that
/// point only once its control sender is gone.
///
/// [`FileDemuxer::seek`]: crate::element::SeekableSource::seek
pub struct FileDemuxer(SourceStage<Demuxing>);

source_stage!(FileDemuxer);

/// What a [`FileDemuxer`] reads, a packet at a time: all of its work, which
/// the framework makes the source.
struct Demuxing {
    pp_log: PpLog,
    name: Arc<str>,
    input: ffmpeg::format::context::Input,
    /// What each stream's output declares, by stream index.
    contracts: Vec<OutputContract>,
    /// Packets read but not yet handed on, in file order: what a seek read
    /// to learn where it landed — see [`Demuxing::read_to_landing`] — which
    /// is real data that still has to go on.
    ahead: VecDeque<(usize, ffmpeg::Rational, ffmpeg::Packet)>,
    /// Which streams something reads, as the framework first says; every
    /// one until then.
    linked: Option<Vec<bool>>,
    /// Its pipeline's, for what goes wrong that is not the source's end —
    /// `None` driven by hand.
    bus: Option<Bus>,
    /// Set through [`FileDemuxerHandle::set_looping`], read only where the
    /// container runs out.
    looping: Arc<AtomicBool>,
    /// `loop_offset`, published for [`FileDemuxerHandle::lap_offset`].
    ///
    /// A copy rather than the field itself: the offset is read and written
    /// once per packet on this thread, and making that an atomic to serve a
    /// reader that looks a few times a second is the wrong way round. This
    /// is stored only where the offset moves, which is once per lap.
    published_offset: Arc<AtomicI64>,
    /// `lap_length`, published for [`FileDemuxerHandle::in_lap`], as the
    /// offset is and for the same reason.
    published_lap: Arc<AtomicI64>,
    /// How far this source's output timeline has been carried past the
    /// file's own, in microseconds: the sum of every lap already played.
    /// Zero until the first wrap, so a source that never loops emits the
    /// file's timestamps untouched.
    ///
    /// Microseconds because one lap has to be one length for every stream.
    /// Measuring each stream's own end separately would let audio and video
    /// restart at different points and drift apart by that difference on
    /// every lap.
    loop_offset: i64,
    /// The furthest into the file, in the same units, any packet read this
    /// lap reaches — what `loop_offset` grows by at the next wrap.
    ///
    /// A running maximum that only the wrap resets. A seek backwards does
    /// not un-deliver what already went downstream, so the lap stays as long
    /// as its furthest packet; growing the offset by what was *played*
    /// instead would drop the next lap on top of timestamps a muxer has
    /// already written.
    lap_end: i64,
    /// How far one lap carries the timeline, in the same units: the
    /// `lap_end` the last wrap stepped over. Zero until the first wrap.
    ///
    /// What puts a position on the timeline back into its lap — see
    /// [`Demuxing::locate_in_lap`] — and what reading backwards steps the
    /// offset down by, back over the start of a lap into the one before.
    lap_length: i64,
    /// Where reading backwards has got to, while it is.
    backwards: Option<Backwards>,
}

/// Reading the picture backwards: a stretch at a time from the end, each
/// stretch in its own order, for a decoder to hand on last picture first.
///
/// A stretch is the pictures shown in `[start, end)`. It is read from the
/// keyframe at or before `start` — a picture is only decoded from its
/// keyframe on — up to the first packet decoded at or after `end`: every
/// picture shown before `end` is decoded before it. What is read only for
/// its pictures' references, shown before `start` or at `end` and after, is
/// flagged `AV_PKT_FLAG_DISCARD`, which the decoder decodes and does not
/// hand on. Then the stretch before it, whose `end` is this one's `start`.
///
/// A stretch begins at a keyframe where it can: the one the last picture
/// before `end` is decoded from, so what is decoded for it is decoded once.
/// It is as long as the decoder can hold, though — see [`BACKWARDS_BUDGET`]
/// — and a group of pictures longer than that is read in stretches of that
/// length from its end, each decoding the group's start again. Only the
/// picture is read; the sound is not played backwards, and its packets are
/// passed over.
struct Backwards {
    /// The picture's stream.
    stream: usize,
    /// The longest a stretch may be, in nanoseconds — see [`BACKWARDS_BUDGET`].
    longest: i64,
    /// This stretch, in nanoseconds of the file.
    start: i64,
    end: i64,
    /// Whether this stretch has still to be sought to.
    unread: bool,
}

/// How much a decoder may hold of a stretch read backwards — see
/// [`Backwards`] — in bytes of decoded pictures. What sets how long a
/// stretch may be: at 1080p and 30 pictures a second about five seconds,
/// the whole of a group of pictures of the length recordings commonly have,
/// which is then decoded once; at 4K and 60 a second two thirds of one, and
/// a group of pictures longer than a stretch is decoded from its start
/// again for each. A hardware decoder holds copies on the device, so this
/// is memory there.
const BACKWARDS_BUDGET: u64 = 512 * 1024 * 1024;

/// The shortest and longest a stretch is, whatever the budget says: a
/// picture too large for it still has to be played, and one small enough
/// for minutes of it gains nothing past a group's length.
const BACKWARDS_STRETCH_LIMITS: (Duration, Duration) =
    (Duration::from_millis(250), Duration::from_secs(20));

/// A stretch's length where the stream does not say its size or rate.
const BACKWARDS_STRETCH_UNKNOWN: Duration = Duration::from_secs(1);

/// When `packet` is due, in nanoseconds of file time — its decode time, which
/// only moves forward in file order, else its presentation time.
fn packet_ns(packet: &ffmpeg::Packet, time_base: ffmpeg::Rational) -> Option<i64> {
    let ts = packet.dts().or(packet.pts())?;
    (time_base.denominator() > 0)
        .then(|| ts.rescale(time_base, ffmpeg::Rational::new(1, 1_000_000_000)))
}

fn duration_ns(duration: Duration) -> i64 {
    duration.as_nanos().min(i64::MAX as u128) as i64
}

impl FileDemuxer {
    /// Opens the file and returns it alongside every stream it contains,
    /// so the caller can inspect them (count, media type, ...) before
    /// deciding which to use — each at its own `index`, which is the pad
    /// [`Context::attach`](crate::element::Context::attach) takes.
    pub fn open(
        name: impl Into<String>,
        path: impl AsRef<Path>,
    ) -> Result<(Self, Vec<StreamInfo>), FileDemuxerError> {
        crate::ensure_ffmpeg();
        let input = ffmpeg::format::input(&path).map_err(|source| FileDemuxerError::Open {
            path: path.as_ref().to_path_buf(),
            source,
        })?;

        let streams: Vec<StreamInfo> = input.streams().map(|s| StreamInfo::of(&s)).collect();
        if streams.is_empty() {
            return Err(FileDemuxerError::NoStreams {
                path: path.as_ref().to_path_buf(),
            });
        }

        // Per stream, from the medium the container announced: every output
        // hands on `MediaBuffer::Packet`, so only this tells an audio stream
        // apart from a video one. A medium this crate does not model
        // (subtitles, data) declares nothing and is left to the runtime
        // check.
        let contracts = streams
            .iter()
            .map(|s| match MediaKind::packet_for(s.kind) {
                Some(kind) => OutputContract::Fixed(PortContract::packet(kind)),
                None => OutputContract::Unknown,
            })
            .collect();

        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::FileDemuxer, &name, None);
        pp_info!(
            pp_log: &pp_log,
            "opened: path={}, {} stream(s)",
            path.as_ref().display(),
            streams.len()
        );
        Ok((
            Self(SourceStage::new(Demuxing {
                name,
                pp_log,
                input,
                contracts,
                ahead: VecDeque::new(),
                linked: None,
                bus: None,
                looping: Arc::new(AtomicBool::new(false)),
                backwards: None,
                published_offset: Arc::new(AtomicI64::new(0)),
                published_lap: Arc::new(AtomicI64::new(0)),
                loop_offset: 0,
                lap_end: 0,
                lap_length: 0,
            })),
            streams,
        ))
    }

    /// The control endpoint for looping this file, valid for as long as the
    /// demuxer runs — take it here, before moving the demuxer into its
    /// pipeline, and keep it for as long as the loop is meant to be
    /// switchable. See [`FileDemuxerHandle::set_looping`].
    pub fn looping_handle(&self) -> FileDemuxerHandle {
        let demuxing = self.0.inner();
        FileDemuxerHandle {
            looping: demuxing.looping.clone(),
            published_offset: demuxing.published_offset.clone(),
            published_lap: demuxing.published_lap.clone(),
        }
    }

    /// The stream of `kind` FFmpeg judges the one to play, as everything a
    /// branch for it is built from — or a [`FileDemuxerError::NoStream`] naming
    /// the kind the file lacks, so finding the video to play is one `?`:
    ///
    /// ```ignore
    /// let (source, _streams) = FileDemuxer::open("demux", path)?;
    /// let video = source.best(media::Type::Video)?;
    /// let decoder = SwDecoder::new("decoder", video.parameters.clone())?;
    /// ```
    ///
    /// The first stream of a kind is not always it. A file can carry a still
    /// picture as a video stream — cover art, a thumbnail — ahead of its
    /// moving picture, and taking the first video stream plays the still.
    /// FFmpeg's own choice (`av_find_best_stream`) passes over a stream marked
    /// as an attached picture and prefers one with more than a single frame.
    pub fn best(&self, kind: ffmpeg::media::Type) -> Result<StreamInfo, FileDemuxerError> {
        self.0
            .inner()
            .input
            .streams()
            .best(kind)
            .map(|stream| StreamInfo::of(&stream))
            .ok_or(FileDemuxerError::NoStream(kind))
    }

    /// How long the file plays, as its container says — `None` where it
    /// says nothing, as a live capture or a truncated recording may not.
    pub fn duration(&self) -> Option<Duration> {
        let micros = self.0.inner().input.duration();
        (micros > 0).then(|| Duration::from_micros(micros as u64))
    }
}

impl Element for Demuxing {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::FileDemuxer
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }

    fn attach_context(&mut self, context: &std::sync::Arc<crate::element::Context>) {
        self.bus = Some(context.bus.for_element(context.source_id));
    }
}

impl Source for Demuxing {
    fn is_live(&self) -> bool {
        false
    }

    /// One per stream in the file, at the stream's own index.
    fn outputs(&self) -> Vec<SrcPad> {
        self.contracts
            .iter()
            .enumerate()
            .map(|(index, contract)| SrcPad::with_contract(format!("src_{index}"), *contract))
            .collect()
    }

    /// The next packet of the file, on its stream's output — read forwards,
    /// or a stretch at a time backwards (see [`Backwards`]). At the end of
    /// the file a looping one begins its next lap, as a segment of its own;
    /// one that does not ends its stream.
    fn produce(&mut self, wait: &mut Wait<'_>) -> crate::error::Result<Produced> {
        if self.linked.is_none() && !wait.linked().is_empty() {
            self.linked = Some(wait.linked().to_vec());
        }
        if self.backwards.is_some() {
            return self.read_backwards();
        }
        let next = self.ahead.pop_front().or_else(|| {
            let (index, time_base, mut packet) = self.next_packet()?;
            self.stamp_lap(index, time_base, &mut packet);
            Some((index, time_base, packet))
        });
        if let Some(item) = next {
            return Ok(Self::on_its_output(item));
        }
        // The end of the file, and the one place the loop flag is read: a
        // change made mid-file lands here, at the end the source was
        // already heading for.
        if self.looping.load(Ordering::Relaxed) {
            match self.wrap() {
                // The lap just wrapped to begins as a segment of its own:
                // on the same timeline — no seek began it, and nothing is
                // flushed — but starting a lap further on, at the start of
                // the file. What is shown and what a caller seeks in are a
                // lap apart across the join, and the segment is where
                // downstream can tell which lap it has; see
                // `crate::stream`. It goes out behind what the lap before
                // still has held back.
                Ok(()) => {
                    return Ok(Produced::Segment {
                        flushed: false,
                        position: Duration::ZERO,
                        start: Duration::from_micros(self.loop_offset.max(0).unsigned_abs()),
                    });
                }
                // A file that cannot be rewound cannot be looped, but it has
                // been fully read — so report why the loop stopped and end
                // the stream properly, rather than failing a source that
                // delivered everything it was asked for.
                Err(error) => {
                    pp_error!(self, "could not start the file again: {error}");
                    if let Some(bus) = &self.bus {
                        bus.post(
                            &self.pp_log,
                            BusEvent::Error {
                                element_type: ElementType::FileDemuxer,
                                name: self.name.clone(),
                                error,
                            },
                        );
                    }
                }
            }
        }
        Ok(Produced::End)
    }

    fn as_seekable(&mut self) -> Option<&mut dyn SeekableSource> {
        Some(self)
    }

    /// A file with a picture: that is what is read backwards.
    fn as_reversible(&mut self) -> Option<&mut dyn ReversibleSource> {
        if self.picture_stream().is_some() {
            Some(self)
        } else {
            None
        }
    }
}

impl SeekableSource for Demuxing {
    fn seek(&mut self, target: Duration) -> crate::error::Result<Duration> {
        self.backwards = None;
        let target = self.locate_in_lap(target);
        self.reposition(target)
    }

    /// On the lap the seek left the timeline on — see
    /// [`Demuxing::locate_in_lap`]: a position in the file a lap further on
    /// for every lap played, and one already past a lap's end where it is.
    fn on_timeline(&self, position: crate::stream::Position) -> Duration {
        let lap = self.lap_length.saturating_mul(1000);
        if lap > 0 && duration_ns(position.0) > lap {
            return position.0;
        }
        position.0 + Duration::from_micros(self.loop_offset.max(0).unsigned_abs())
    }
}

/// The picture's stream, read back a stretch at a time. What nothing is
/// wired to is read all the same and handed to no one, down to the start
/// of the file, where the stream ends.
impl ReversibleSource for Demuxing {
    fn seek_backwards(&mut self, target: Duration) -> crate::error::Result<Duration> {
        let stream = self
            .picture_stream()
            .ok_or(FileDemuxerError::NoStream(ffmpeg::media::Type::Video))?;
        let target = self.locate_in_lap(target);
        // The picture at `target` is the last of the first stretch.
        let end = duration_ns(target).saturating_add(1);
        let longest = duration_ns(self.stretch_limit(stream));
        self.backwards = Some(Backwards {
            stream,
            longest,
            start: end,
            end,
            unread: true,
        });
        self.start_stretch()?;
        Ok(target)
    }

    /// Once the next stretch is marked to be sought to: the one just read
    /// was the last of its own.
    fn stretch_complete(&self) -> bool {
        self.backwards
            .as_ref()
            .is_none_or(|backwards| backwards.unread)
    }
}

impl Demuxing {
    #[cfg(test)]
    fn stream(&self, index: usize) -> Option<ffmpeg::format::stream::Stream<'_>> {
        self.input.streams().find(|s| s.index() == index)
    }

    /// The next packet of a stream this source has an output for, with its
    /// stream's number and time base; `None` at the end of the file.
    ///
    /// The outputs are the streams the file was opened with. A container
    /// that has no header to list them — MPEG-TS, FLV — may announce
    /// another one later, which has no output to go on: its packets are
    /// passed over, as nothing linked to them would read them either.
    fn next_packet(&mut self) -> Option<(usize, ffmpeg::Rational, ffmpeg::Packet)> {
        let outputs = self.contracts.len();
        self.input
            .packets()
            .map(|(stream, packet)| (stream.index(), stream.time_base(), packet))
            .find(|&(index, _, _)| index < outputs)
    }

    /// A packet read, stamped with its stream's time base, on its stream's
    /// output.
    ///
    /// `AVCodecParameters` does not carry the container stream's timestamp
    /// unit, and FFmpeg does not guarantee that demuxers populate
    /// `AVPacket::time_base`. Stamping it here — the one place every packet
    /// leaves this source through, read ahead or not — means a packet held
    /// back for a blocked output arrives downstream describing itself the
    /// same way an immediately delivered one does.
    fn on_its_output(
        (index, time_base, mut packet): (usize, ffmpeg::Rational, ffmpeg::Packet),
    ) -> Produced {
        packet.set_time_base(time_base);
        Produced::On(index, MediaBuffer::Packet(Arc::new(packet).into()))
    }

    /// Puts a freshly read packet on this source's output timeline, and
    /// records how far into the file this lap has now reached.
    ///
    /// Called at each of the two places a packet is read out of the
    /// container — `produce`'s cursor and `seek`'s read-ahead — rather than
    /// where they are handed on. Stamping a time base is idempotent;
    /// shifting a timestamp is not.
    ///
    /// Only a linked output's stream counts towards the lap's length. An
    /// unlinked one is dropped rather than delivered, so letting a longer
    /// audio track nobody selected decide where the video restarts would
    /// only open a gap at every join.
    fn stamp_lap(
        &mut self,
        index: usize,
        time_base: ffmpeg::Rational,
        packet: &mut ffmpeg::Packet,
    ) {
        let linked = self
            .linked
            .as_ref()
            .is_none_or(|linked| linked.get(index).copied().unwrap_or(false));
        if let Some(start) = packet.pts().or_else(|| packet.dts())
            && linked
        {
            // A packet carrying no duration of its own still ends after it
            // starts, and one tick is the least that keeps the next lap's
            // first timestamp past this one's rather than equal to it.
            let end = start.saturating_add(packet.duration().max(1));
            self.lap_end = self.lap_end.max(end.rescale(time_base, microseconds()));
        }
        self.shift_to_lap(time_base, packet);
    }

    /// Carries `packet` from the file's own timeline onto the lap this is
    /// reading, forwards or backwards.
    fn shift_to_lap(&self, time_base: ffmpeg::Rational, packet: &mut ffmpeg::Packet) {
        if self.loop_offset == 0 {
            return;
        }
        let shift = self.loop_offset.rescale(microseconds(), time_base);
        packet.set_pts(packet.pts().map(|pts| pts.saturating_add(shift)));
        packet.set_dts(packet.dts().map(|dts| dts.saturating_add(shift)));
    }

    /// Puts the timeline on the lap that starts `offset` past the file's
    /// own, as a whole one: the next wrap steps over exactly one lap from
    /// there.
    fn move_to_lap(&mut self, offset: i64) {
        self.loop_offset = offset.max(0);
        self.lap_end = self.lap_length;
        self.published_offset
            .store(self.loop_offset, Ordering::Relaxed);
    }

    /// `target` on the output timeline, as a position in the file — the
    /// timeline moved to the lap it falls in.
    ///
    /// A caller's seek is a position in the file, never past its end; but
    /// where playback turns round is the picture shown, which is on the
    /// timeline looping carries past the end, a lap further on for every
    /// lap played. Read as a position in the file it is past the end, and
    /// what was read from there was stamped a lap or more short of it, so
    /// the picture stopped until the clock came down to it. Only such a
    /// position is past the end of a lap, which is what tells the two apart.
    fn locate_in_lap(&mut self, target: Duration) -> Duration {
        let (lap, at) = (self.lap_length.saturating_mul(1000), duration_ns(target));
        if lap <= 0 || at <= lap {
            return target;
        }
        let laps = (at - 1) / lap;
        self.move_to_lap(laps * self.lap_length);
        pp_debug!(
            self,
            "{target:?} is on lap {}, {}us past the file's own",
            laps + 1,
            self.loop_offset
        );
        Duration::from_nanos((at - laps * lap) as u64)
    }

    /// Starts the file again: carries the output timeline past the lap that
    /// just ended, then rewinds the container.
    ///
    /// Ordering matters. The offset moves first so that the read-ahead
    /// packets `seek` keeps — the new lap's first — are stamped onto the new
    /// timeline like every packet after them.
    fn wrap(&mut self) -> crate::error::Result<()> {
        self.lap_length = self.lap_end;
        self.published_lap.store(self.lap_length, Ordering::Relaxed);
        self.loop_offset = self.loop_offset.saturating_add(self.lap_end);
        self.published_offset
            .store(self.loop_offset, Ordering::Relaxed);
        self.lap_end = 0;
        let landed = self.seek(Duration::ZERO)?;
        pp_debug!(
            self,
            "looped: restarted at {landed:?}, timeline now {}us past the file's own",
            self.loop_offset
        );
        Ok(())
    }

    /// The stream FFmpeg seeks the file by — its default one, the picture
    /// where there is one — and that stream's time base. `None` where it
    /// has none with a time base to read.
    fn seek_stream(&self) -> Option<(usize, ffmpeg::Rational)> {
        // SAFETY: `input` is an open format context for as long as `self`
        // is, and this only reads its stream list.
        let index =
            unsafe { ffmpeg::ffi::av_find_default_stream_index(self.input.as_ptr().cast_mut()) };
        let index = usize::try_from(index).ok()?;
        let time_base = self.input.stream(index)?.time_base();
        (time_base.numerator() > 0 && time_base.denominator() > 0).then_some((index, time_base))
    }

    /// Reads on from where a seek left the file up to the first packet of
    /// the stream it seeks by, keeping all of it to be handed on, and
    /// answers where the first packet of all starts and when that stream's
    /// first starts and is decoded — `None` for either past the end.
    ///
    /// Where a seek landed is only known by reading: `avformat_seek_file`
    /// says nothing but whether it failed. What is read is real data, not a
    /// probe to throw away, which is why it is kept rather than dropped.
    fn read_to_landing(
        &mut self,
        by: Option<(usize, ffmpeg::Rational)>,
    ) -> (Option<Duration>, Option<(Duration, Duration)>) {
        // Enough for any interleave; a stream this does not reach within it
        // is one to stop looking for.
        const READ_LIMIT: usize = 1_024;
        let nanos = ffmpeg::Rational::new(1, 1_000_000_000);
        let mut first = None;
        for _ in 0..READ_LIMIT {
            let Some((index, time_base, mut packet)) = self.next_packet() else {
                break;
            };
            // Read before `stamp_lap` moves it: where a seek landed is a
            // position in the *file*, which a loop's accumulated offset must
            // not be added to.
            first.get_or_insert_with(|| {
                packet
                    .pts()
                    .or_else(|| packet.dts())
                    .map(|ts| ts_to_duration(ts, time_base))
                    .unwrap_or(Duration::ZERO)
            });
            let at = |ts: Option<i64>| {
                ts.and_then(|ts| u64::try_from(ts.rescale(time_base, nanos)).ok())
                    .map(Duration::from_nanos)
            };
            let seeking_by = by.is_none_or(|(by, _)| by == index);
            let landing = at(packet.pts().or_else(|| packet.dts()))
                .zip(at(packet.dts().or_else(|| packet.pts())));
            // Backwards, what is kept is read back as it is read from the
            // file, and carried onto the lap only as it is handed on — see
            // `read_backwards`, which compares it with the stretch in the
            // file's own time first.
            if self.backwards.is_none() {
                self.stamp_lap(index, time_base, &mut packet);
            }
            self.ahead.push_back((index, time_base, packet));
            if seeking_by {
                return (first, landing);
            }
        }
        (first, None)
    }

    /// `target` as the microseconds to hand `Input::seek`, such that the seek
    /// never lands on a keyframe after it.
    ///
    /// FFmpeg reads those microseconds in the time base of the stream it
    /// seeks by — the one `av_find_default_stream_index` picks — rounding
    /// to the nearest tick. Rounded up, a target just before a keyframe was
    /// that keyframe, and the seek landed on it: an accurate seek to the
    /// instant before a keyframe showed the keyframe, and a frame step back
    /// from one stayed where it was. Here the target is floored to that
    /// stream's tick and then to a microsecond whose nearest tick is that
    /// one, or a later microsecond's is not reached.
    fn seek_micros(target: Duration, by: Option<(usize, ffmpeg::Rational)>) -> i64 {
        let whole = target.as_micros().min(i64::MAX as u128) as i64;
        let Some((_, time_base)) = by else {
            return whole;
        };
        let (num, den) = (
            i128::from(time_base.numerator()),
            i128::from(time_base.denominator()),
        );
        let ticks = target.as_nanos() as i128 * den / (1_000_000_000 * num);
        let micros = ticks * 1_000_000 * num / den;
        i64::try_from(micros).unwrap_or(whole).min(whole)
    }

    /// The picture's stream, where the file has one to play backwards.
    fn picture_stream(&self) -> Option<usize> {
        self.input
            .streams()
            .best(ffmpeg::media::Type::Video)
            .map(|stream| stream.index())
            .filter(|&index| index < self.contracts.len())
    }

    /// Seeks to the stretch before the one just read — see [`Backwards`].
    fn start_stretch(&mut self) -> crate::error::Result<()> {
        let Some(backwards) = self.backwards.as_mut() else {
            return Ok(());
        };
        backwards.unread = false;
        let (end, longest) = (backwards.end, backwards.longest);
        self.ahead.clear();
        // The keyframe the last picture before `end` is decoded from: a
        // seek to just before `end` lands on it, and it is where this
        // stretch is read from whatever its length.
        let before_end = Duration::from_nanos(end.saturating_sub(1).max(0) as u64);
        let (_, keyframe) = self.reposition_landing(before_end)?;
        let shortest_start = end.saturating_sub(longest).max(0);
        let start = keyframe
            .map(duration_ns)
            .filter(|&keyframe| keyframe < end)
            .map_or(shortest_start, |keyframe| keyframe.max(shortest_start));
        if let Some(backwards) = self.backwards.as_mut() {
            backwards.start = start;
        }
        Ok(())
    }

    /// How long a stretch of `stream` may be: as many of its pictures as
    /// [`BACKWARDS_BUDGET`] holds, reckoned from the size, depth and rate
    /// its parameters say, within [`BACKWARDS_STRETCH_LIMITS`].
    fn stretch_limit(&self, stream: usize) -> Duration {
        let Some(stream) = self.input.stream(stream) else {
            return BACKWARDS_STRETCH_UNKNOWN;
        };
        let rate = stream.avg_frame_rate();
        let rate = if rate.numerator() > 0 && rate.denominator() > 0 {
            rate
        } else {
            stream.rate()
        };
        let parameters = stream.parameters();
        // SAFETY: `parameters` is the stream's live codec parameters, read
        // for three plain fields.
        let (width, height, depth) = unsafe {
            let parameters = parameters.as_ptr();
            (
                (*parameters).width,
                (*parameters).height,
                (*parameters).bits_per_raw_sample,
            )
        };
        let (Ok(width), Ok(height)) = (u64::try_from(width), u64::try_from(height)) else {
            return BACKWARDS_STRETCH_UNKNOWN;
        };
        if width == 0 || height == 0 || rate.numerator() <= 0 || rate.denominator() <= 0 {
            return BACKWARDS_STRETCH_UNKNOWN;
        }
        // Chroma at a quarter the resolution of luma, as decoders hand
        // most pictures on, and two bytes a sample past eight bits.
        let sample_bytes = if depth > 8 { 2 } else { 1 };
        let picture_bytes = (width * height * 3 / 2 * sample_bytes).max(1);
        let pictures = (BACKWARDS_BUDGET / picture_bytes) as f64;
        let seconds = pictures * f64::from(rate.denominator()) / f64::from(rate.numerator());
        let (shortest, longest) = BACKWARDS_STRETCH_LIMITS;
        Duration::from_secs_f64(seconds).clamp(shortest, longest)
    }

    /// Reads one step backwards — see [`Backwards`]: the next packet of the
    /// stretch under way, on the picture's output; nothing, where the step
    /// sought to a stretch or passed over a packet it does not hand on; or
    /// the end of the stream, once the stretch that begins the file has
    /// been read.
    fn read_backwards(&mut self) -> crate::error::Result<Produced> {
        if self
            .backwards
            .as_ref()
            .is_some_and(|backwards| backwards.unread)
        {
            self.start_stretch()?;
            return Ok(Produced::Nothing);
        }
        let next = self.ahead.pop_front().or_else(|| self.next_packet());
        let Some(backwards) = self.backwards.as_mut() else {
            return Ok(Produced::End);
        };
        let Some((index, time_base, mut packet)) = next else {
            // The file ran out inside the stretch: it is read.
            return Ok(self.stretch_read());
        };
        if index != backwards.stream {
            return Ok(Produced::Nothing);
        }
        let nanos = ffmpeg::Rational::new(1, 1_000_000_000);
        let decoded = packet_ns(&packet, time_base).unwrap_or(i64::MIN);
        if decoded >= backwards.end {
            return Ok(self.stretch_read());
        }
        let shown = packet
            .pts()
            .or_else(|| packet.dts())
            .map(|ts| ts.rescale(time_base, nanos));
        if shown.is_none_or(|shown| shown < backwards.start || shown >= backwards.end) {
            use ffmpeg::packet::Mut;
            // SAFETY: `packet` is this function's own live packet; `flags`
            // is a plain field FFmpeg reads as it decodes it.
            unsafe {
                (*packet.as_mut_ptr()).flags |= ffmpeg::ffi::AV_PKT_FLAG_DISCARD;
            }
        }
        self.shift_to_lap(time_base, &mut packet);
        Ok(Self::on_its_output((index, time_base, packet)))
    }

    /// The stretch just read is done: on to the one before — the end of the
    /// lap before, where this one began the file and looping put a lap
    /// before it — unless it began the timeline, which ends the stream.
    ///
    /// Back into a lap that was played, whether or not the file still
    /// loops: that lap is on the timeline either way. Only the first lap's
    /// start ends the stream, as the start of a file that never looped does.
    fn stretch_read(&mut self) -> Produced {
        let (offset, lap) = (self.loop_offset, self.lap_length);
        let Some(backwards) = self.backwards.as_mut() else {
            return Produced::End;
        };
        if backwards.start <= 0 {
            if offset <= 0 || lap <= 0 {
                return Produced::End;
            }
            backwards.start = lap.saturating_mul(1000);
            backwards.end = backwards.start;
            backwards.unread = true;
            self.move_to_lap(offset - lap);
            pp_debug!(
                self,
                "back over the start of a lap: timeline now {}us past the file's own",
                self.loop_offset
            );
            return Produced::Nothing;
        }
        backwards.end = backwards.start;
        backwards.unread = true;
        Produced::Nothing
    }

    /// Moves the read cursor to `target`, forwards — see
    /// [`SeekableSource::seek`].
    fn reposition(&mut self, target: Duration) -> crate::error::Result<Duration> {
        self.reposition_landing(target).map(|(landed, _)| landed)
    }

    /// As [`Self::reposition`], answering besides where the first picture
    /// of the stream it seeks by starts, where the file has one after it —
    /// the keyframe it landed on.
    fn reposition_landing(
        &mut self,
        target: Duration,
    ) -> crate::error::Result<(Duration, Option<Duration>)> {
        // `Input::seek` takes microseconds (`AV_TIME_BASE` units) when
        // seeking the whole container (stream index -1, which is what it
        // uses internally) rather than one specific stream — an unbounded
        // range (`..`) just means "as close to `ts` as ffmpeg can manage",
        // no extra min/max constraint. In practice that means *backward*
        // to the nearest keyframe at or before `target`: never forward,
        // and never onto a non-keyframe, since either would leave nothing
        // downstream can decode/remux from. A sparse-keyframe file can
        // make that keyframe well before `target` — e.g. a single
        // 10-second file with keyframes only at 0s and 8.3s means every
        // `target` under 8.3s lands back at 0s.
        //
        // Where it lands is by when a keyframe is *decoded*, which with
        // B-frames is before it is shown: the pictures shown just before a
        // keyframe belong to the group before it, and a target among them
        // landed on the keyframe, whose pictures all come after the target.
        // Nothing then covered it — an accurate seek there showed the
        // keyframe, a frame step back from it stayed on it. So a landing
        // whose first picture starts after the target is sought again from
        // before that picture is decoded, which is the group before.
        const ATTEMPTS: usize = 4;
        let by = self.seek_stream();
        let mut aim = target;
        let mut landed = None;
        let mut last_picture = None;
        // What an earlier seek read ahead is from where it left the file,
        // which this one is leaving.
        self.ahead.clear();
        for _ in 0..ATTEMPTS {
            let ts = Self::seek_micros(aim, by);
            self.input.seek(ts, ..).inspect_err(|error| {
                pp_error!(self, "seek to {target:?} failed: {error}");
            })?;
            let (first, picture) = self.read_to_landing(by);
            landed = first;
            // Landed where it did last time too: the stream's pictures start
            // after the target, and there is nothing earlier to land on.
            if std::mem::replace(&mut last_picture, picture) == picture {
                break;
            }
            let Some((starts, decoded)) = picture.filter(|(starts, _)| *starts > target) else {
                break;
            };
            let tick = by.map_or(Duration::from_micros(1), |(_, base)| {
                Duration::from_nanos(
                    (1_000_000_000 * u64::try_from(base.numerator()).unwrap_or(1))
                        .div_ceil(u64::try_from(base.denominator()).unwrap_or(1)),
                )
            });
            let earlier = decoded.min(aim).saturating_sub(tick);
            if earlier >= aim {
                // The start of the file: nothing before it to land on.
                break;
            }
            pp_debug!(
                self,
                "seek to {target:?} landed on a picture at {starts:?}; seeking again from {earlier:?}"
            );
            aim = earlier;
            self.ahead.clear();
        }
        // Nothing left to read (`target` at/past EOF): no packet to learn a
        // real position from, so the request is reported back as it is.
        Ok((
            landed.unwrap_or(target),
            last_picture.map(|(starts, _)| starts),
        ))
    }
}

/// The unit a lap's length is kept in, so it can be one length for every
/// stream. Also what `Input::seek` takes, which is why the wrap needs no
/// conversion of its own.
///
/// A hardcoded constant, not external input, so there is nothing to
/// validate.
fn microseconds() -> ffmpeg::Rational {
    ffmpeg::Rational::new(1, 1_000_000)
}

fn ts_to_duration(ts: i64, time_base: ffmpeg::Rational) -> Duration {
    let secs = ts as f64 * f64::from(time_base.numerator()) / f64::from(time_base.denominator());
    Duration::from_secs_f64(secs.max(0.0))
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    /// A video stream says its size and rate, what re-encoding it needs; a
    /// sound stream says neither.
    #[test]
    fn a_video_stream_says_its_size_and_rate() {
        let Some(path) = try_test_video() else { return };
        let (_, streams) = FileDemuxer::open("demux", &path).unwrap();
        let video = streams
            .iter()
            .find(|stream| stream.kind == ffmpeg::media::Type::Video)
            .unwrap();
        assert_eq!(video.size(), Some((320, 240)));
        assert_eq!(video.frame_rate, Some(ffmpeg::Rational::new(30, 1)));
        let sound = streams
            .iter()
            .find(|stream| stream.kind == ffmpeg::media::Type::Audio)
            .unwrap();
        assert_eq!((sound.size(), sound.frame_rate), (None, None));
    }

    /// What `open` says of the streams is the caller's own: it does not
    /// keep the file open behind the source. It did — the parameters shared
    /// the input's ownership — so a file could not be deleted, on Windows,
    /// while anything held the stream list, even with the source gone.
    #[test]
    fn the_streams_open_describes_do_not_hold_the_file() {
        let Some(fixture) = crate::test_support::try_test_video() else {
            return;
        };
        let path = std::env::temp_dir().join(format!(
            "media-pp-streams-hold-{}-{:?}.mp4",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::copy(&fixture, &path).expect("copy the fixture");
        let (demuxer, streams) = FileDemuxer::open("demux", &path).expect("it opens");
        drop(demuxer);
        let removed = std::fs::remove_file(&path);
        assert!(removed.is_ok(), "the file is still held: {removed:?}");
        assert!(
            streams
                .iter()
                .any(|stream| stream.parameters.id() != ffmpeg::codec::Id::None),
            "and what was said of it is still there to read"
        );
    }

    /// A file that cannot be opened says which file it was.
    #[test]
    fn a_file_that_cannot_be_opened_says_which() {
        let path = std::env::temp_dir().join("media-pp-no-such-file.mp4");
        let Err(error) = FileDemuxer::open("demux", &path) else {
            panic!("a file that is not there opens");
        };
        assert!(matches!(error, FileDemuxerError::Open { .. }), "{error:?}");
        assert!(
            error.to_string().contains("media-pp-no-such-file.mp4"),
            "{error}"
        );
    }

    /// A recording finished before anything was written into it is a valid
    /// file with nothing in it, and is refused as one rather than opened
    /// with no stream for a caller to index into.
    #[test]
    fn a_file_with_no_streams_is_refused() {
        use crate::elements::{FileMuxer, SwEncoder, SwEncoderOptions, VideoCodec};

        let path = std::env::temp_dir().join(format!(
            "media-pp-empty-recording-{}.mp4",
            std::process::id()
        ));
        let encoder = SwEncoder::new(
            "encoder",
            SwEncoderOptions {
                codec: VideoCodec::OpenH264,
                width: 320,
                height: 240,
                pixel_format: ffmpeg::format::Pixel::YUV420P,
                frame_rate: ffmpeg::Rational::new(30, 1),
                bit_rate: 1_000_000,
                gop_size: 30,
                max_b_frames: None,
            },
        )
        .expect("the always-present encoder opens");
        let mut muxer = FileMuxer::create(&path).unwrap();
        let track = muxer.add_stream("video", &encoder).unwrap();
        let mut sinks = muxer.open().unwrap();
        let mut sink = sinks.take(track).unwrap();
        sink.stream_event(&crate::stream::StreamEvent::Eos).unwrap();
        drop(sink);
        drop(sinks);

        let opened = FileDemuxer::open("demux", &path);
        let _ = std::fs::remove_file(&path);
        assert!(
            matches!(opened, Err(FileDemuxerError::NoStreams { .. })),
            "{:?}",
            opened.map(|(_, streams)| streams.len())
        );
    }

    use super::*;

    /// A video stream says what its pictures decode to in software; a
    /// sound stream says nothing.
    #[test]
    fn a_video_stream_says_its_pixel_format() {
        let Some(video) = crate::test_support::try_test_video() else {
            return;
        };
        let (demuxer, _) = FileDemuxer::open("demux", &video).expect("opens");
        let picture = demuxer.best(ffmpeg::media::Type::Video).expect("video");
        assert_eq!(picture.pixel_format(), Some(ffmpeg::format::Pixel::YUV420P));
        if let Ok(sound) = demuxer.best(ffmpeg::media::Type::Audio) {
            assert_eq!(sound.pixel_format(), None);
        }
    }
    use crate::control;
    use crate::element::{RawSource, SrcPads};
    use crate::test_support::try_test_video;

    struct CountingSink {
        pp_log: PpLog,
        count: Arc<AtomicUsize>,
        saw_eos: Arc<AtomicBool>,
        expected_time_base: ffmpeg::Rational,
        time_base_matches: Arc<AtomicBool>,
    }

    impl Element for CountingSink {
        fn name(&self) -> Arc<str> {
            "counting-sink".into()
        }

        fn element_type(&self) -> ElementType {
            ElementType::Other
        }

        fn pp_log(&self) -> &PpLog {
            &self.pp_log
        }

        fn pp_log_mut(&mut self) -> &mut PpLog {
            &mut self.pp_log
        }
    }

    impl crate::element::RawSink for CountingSink {
        fn stream_event(&mut self, event: &crate::stream::StreamEvent) -> crate::error::Result<()> {
            if let crate::stream::StreamEvent::Eos = event {
                self.saw_eos.store(true, Ordering::SeqCst);
            }
            Ok(())
        }

        fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
            if let MediaBuffer::Packet(packet) = buf {
                if packet.time_base() != self.expected_time_base {
                    self.time_base_matches.store(false, Ordering::SeqCst);
                }
                self.count.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        }
    }

    /// `open` lists every stream, each at its own index — which is also its
    /// pad's — so a caller can attach by the `index` it was handed.
    #[test]
    fn open_lists_every_stream_at_its_own_index() {
        let Some(path) = try_test_video() else { return };
        let (mut demuxer, streams) = FileDemuxer::open("demux", &path).expect("open test video");

        assert_eq!(streams.len(), demuxer.src_pads().len());
        for (position, stream) in streams.iter().enumerate() {
            assert_eq!(stream.index, position);
            assert!(stream.time_base.numerator() > 0 && stream.time_base.denominator() > 0);
        }
    }

    /// Follows the output timeline across a loop's join, and switches the
    /// loop off once the file has been through once so `run` finishes
    /// instead of going round forever.
    struct LoopSink {
        pp_log: PpLog,
        count: Arc<AtomicUsize>,
        saw_eos: Arc<AtomicBool>,
        /// Cleared the first time a decode timestamp goes backwards.
        climbing: Arc<AtomicBool>,
        last_dts: Arc<Mutex<Option<i64>>>,
        /// How many packets one pass of this stream delivers.
        lap: usize,
        handle: FileDemuxerHandle,
    }

    impl Element for LoopSink {
        fn name(&self) -> Arc<str> {
            "loop-sink".into()
        }

        fn element_type(&self) -> ElementType {
            ElementType::Other
        }

        fn pp_log(&self) -> &PpLog {
            &self.pp_log
        }

        fn pp_log_mut(&mut self) -> &mut PpLog {
            &mut self.pp_log
        }
    }

    impl crate::element::RawSink for LoopSink {
        fn stream_event(&mut self, event: &crate::stream::StreamEvent) -> crate::error::Result<()> {
            if let crate::stream::StreamEvent::Eos = event {
                self.saw_eos.store(true, Ordering::SeqCst);
            }
            Ok(())
        }

        fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
            if let MediaBuffer::Packet(packet) = buf {
                if let Some(dts) = packet.dts().or_else(|| packet.pts()) {
                    let mut last = self.last_dts.lock().unwrap();
                    if last.is_some_and(|last| dts < last) {
                        self.climbing.store(false, Ordering::SeqCst);
                    }
                    *last = Some(dts);
                }
                // Past a whole pass of the file: the source is into its
                // second lap, so let that one play out and then end.
                if self.count.fetch_add(1, Ordering::SeqCst) + 1 > self.lap {
                    self.handle.set_looping(false);
                }
            }
            Ok(())
        }
    }

    /// The index of this file's video stream, and the time base its packets
    /// carry.
    fn video_stream(demuxer: &FileDemuxer) -> (usize, ffmpeg::Rational) {
        let video = demuxer
            .best(ffmpeg::media::Type::Video)
            .expect("test video has a video stream");
        (video.index, video.time_base)
    }

    /// A seek made while paused at the end of the file waits, as any paused
    /// seek does, for playback to go on — it does not read on at once.
    ///
    /// It did: the wait at the end passed a `Pause` on without pausing
    /// itself, so a seek arriving after it sent the file straight into the
    /// paused queue behind — which, once full, held this source handing it a
    /// packet while the seek waited on a `Preroll` the source could no longer
    /// take. A player seeking the moment it heard the file had ended hung,
    /// on a machine slow enough for the queue to fill first.
    #[test]
    fn a_seek_while_paused_at_the_end_reads_nothing_until_resumed() {
        let Some(path) = try_test_video() else { return };
        let (mut demuxer, _) = FileDemuxer::open("demux", &path).expect("open test video");
        let (index, expected_time_base) = video_stream(&demuxer);
        let count = Arc::new(AtomicUsize::new(0));
        let saw_eos = Arc::new(AtomicBool::new(false));
        demuxer.src_pads()[index].link(Box::new(CountingSink {
            count: count.clone(),
            saw_eos: saw_eos.clone(),
            expected_time_base,
            time_base_matches: Arc::new(AtomicBool::new(true)),
            pp_log: element_pp_log(ElementType::Other, "counting-sink", None),
        }));
        let (bus, _bus_rx) = Bus::new();
        let (tx, rx) = control::channel();
        let runner = std::thread::spawn(move || demuxer.run(&rx, &bus));
        let waited = std::time::Instant::now();
        while !saw_eos.load(Ordering::SeqCst) {
            assert!(
                waited.elapsed() < Duration::from_secs(10),
                "the file never ended"
            );
            std::thread::sleep(Duration::from_millis(10));
        }

        let at_the_end = count.load(Ordering::SeqCst);
        tx.send(crate::control::ControlMsg::Pause);
        tx.send(crate::control::ControlMsg::Seek(Duration::ZERO));
        std::thread::sleep(Duration::from_millis(300));
        assert_eq!(
            count.load(Ordering::SeqCst),
            at_the_end,
            "paused, it read the file on from where the seek put it"
        );

        tx.send(crate::control::ControlMsg::Resume);
        let waited = std::time::Instant::now();
        while count.load(Ordering::SeqCst) == at_the_end {
            assert!(
                waited.elapsed() < Duration::from_secs(10),
                "resumed, it never read on"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        tx.send(crate::control::ControlMsg::Stop);
        runner
            .join()
            .expect("the source thread")
            .expect("it stops cleanly");
    }

    /// How many packets one pass of this file's video stream delivers, so a
    /// test can tell a second lap has begun without assuming anything about
    /// the fixture.
    fn packets_in_one_pass(path: impl AsRef<Path>) -> usize {
        let (mut demuxer, _) = FileDemuxer::open("demux", path).expect("open test video");
        let (index, expected_time_base) = video_stream(&demuxer);
        let count = Arc::new(AtomicUsize::new(0));
        demuxer.src_pads()[index].link(Box::new(CountingSink {
            count: count.clone(),
            saw_eos: Arc::new(AtomicBool::new(false)),
            expected_time_base,
            time_base_matches: Arc::new(AtomicBool::new(true)),
            pp_log: element_pp_log(ElementType::Other, "counting-sink", None),
        }));
        let (bus, _bus_rx) = Bus::new();
        let (_, rx) = control::channel();
        demuxer.run(&rx, &bus).expect("run must reach eos cleanly");
        count.load(Ordering::SeqCst)
    }

    /// Looping puts the start of the file where its end was, and the
    /// timestamps that come out keep climbing across that join instead of
    /// restarting at zero.
    ///
    /// That is the whole point of the offset. A `Pacer` downstream anchors
    /// on the first timestamp it sees and waits for each later one to come
    /// due; hand it a second lap starting back at zero and every frame of it
    /// is already overdue, so the lap is emitted as fast as it can be read
    /// rather than played. A muxer refuses it outright.
    ///
    /// Also covers when the flag is read: it is switched off part way into
    /// the second lap, and that lap still plays out and ends with a real
    /// `Eos`.
    #[test]
    fn looping_restarts_the_file_and_carries_the_timeline_past_the_join() {
        let Some(path) = try_test_video() else { return };
        let lap = packets_in_one_pass(&path);
        assert!(lap > 0, "the fixture must deliver something to loop");

        let (mut demuxer, _) = FileDemuxer::open("demux", &path).expect("open test video");
        let (index, _) = video_stream(&demuxer);
        let handle = demuxer.looping_handle();
        assert!(
            !handle.is_looping(),
            "a file plays once unless asked not to"
        );
        handle.set_looping(true);

        let count = Arc::new(AtomicUsize::new(0));
        let saw_eos = Arc::new(AtomicBool::new(false));
        let climbing = Arc::new(AtomicBool::new(true));
        demuxer.src_pads()[index].link(Box::new(LoopSink {
            count: count.clone(),
            saw_eos: saw_eos.clone(),
            climbing: climbing.clone(),
            last_dts: Arc::new(Mutex::new(None)),
            lap,
            handle: handle.clone(),
            pp_log: element_pp_log(ElementType::Other, "loop-sink", None),
        }));

        let (bus, bus_rx) = Bus::new();
        let (_, rx) = control::channel();
        demuxer
            .run(&rx, &bus)
            .expect("run must reach eos cleanly, not error");

        assert!(
            count.load(Ordering::SeqCst) > lap,
            "the end of the file must start it again, not end the stream"
        );
        assert!(
            climbing.load(Ordering::SeqCst),
            "timestamps must not fall back to the file's own at the join"
        );
        assert!(
            saw_eos.load(Ordering::SeqCst),
            "switching looping off must end the lap it is in with an Eos"
        );
        // What a reader needs to turn one of those climbing timestamps back
        // into a position in the file. It only moves at a wrap, so a run that
        // wrapped has one and a run that did not has zero.
        assert!(
            handle.lap_offset() > Duration::ZERO,
            "a lap that has been stepped over must be reported"
        );
        drop(bus);
        assert!(
            bus_rx.iter().all(|e| !matches!(e, BusEvent::Error { .. })),
            "looping a well-formed file must not report any errors"
        );
    }

    /// Regression test for how far a wrap carries the timeline. It has to be
    /// how far into the file the lap *reached*, not how much of it was
    /// played: a seek backwards does not un-deliver the packets that already
    /// went downstream, so a shorter step would drop the next lap on top of
    /// timestamps a muxer has already written.
    #[test]
    fn a_backward_seek_leaves_the_lap_as_long_as_its_furthest_packet() {
        let Some(path) = try_test_video() else { return };
        let (mut demuxer, streams) = FileDemuxer::open("demux", &path).expect("open test video");

        // Every pad linked, because only a linked pad's stream counts
        // towards the lap — and `seek` parks whichever stream's packet it
        // happens to read first.
        let time_bases: Vec<ffmpeg::Rational> =
            streams.iter().map(|stream| stream.time_base).collect();
        for (index, expected_time_base) in time_bases.into_iter().enumerate() {
            demuxer.src_pads()[index].link(Box::new(CountingSink {
                count: Arc::new(AtomicUsize::new(0)),
                saw_eos: Arc::new(AtomicBool::new(false)),
                expected_time_base,
                time_base_matches: Arc::new(AtomicBool::new(true)),
                pp_log: element_pp_log(ElementType::Other, "counting-sink", None),
            }));
        }

        demuxer
            .0
            .inner_mut()
            .seek(Duration::ZERO)
            .expect("seek to the start of the test video");
        let at_start = demuxer.0.inner().lap_end;

        // Half way in, so the keyframe this lands on is somewhere past the
        // first one for any file that has more than one — which is what the
        // guard below checks rather than assumes.
        let half = Duration::from_micros((demuxer.0.inner().input.duration().max(0) / 2) as u64);
        demuxer
            .0
            .inner_mut()
            .seek(half)
            .expect("seek half way into the test video");
        let reached = demuxer.0.inner().lap_end;
        if reached <= at_start {
            // Nothing in this fixture is reachable past its own start, so
            // there is no reach for a seek back to lose.
            return;
        }

        demuxer
            .0
            .inner_mut()
            .seek(Duration::ZERO)
            .expect("seek back to the start of the test video");

        assert_eq!(
            demuxer.0.inner().lap_end,
            reached,
            "going back must not shorten the lap the next wrap steps over"
        );
    }

    /// Where a pipeline turns round is the picture shown, on the timeline
    /// looping carries past the end of the file; a caller's seek is in the
    /// file. A position past the end of a lap can only be the first, and is
    /// put back into its lap, with the timeline moved to that lap.
    #[test]
    fn a_position_past_the_end_of_a_lap_is_put_back_into_its_lap() {
        let Some(path) = try_test_video() else { return };
        let (mut demuxer, _) = FileDemuxer::open("demux", &path).expect("open test video");
        let lap = 8_000_000;
        demuxer.0.inner_mut().lap_length = lap;
        // Reading the fourth lap, as a demuxer ahead of what is shown is.
        demuxer.0.inner_mut().move_to_lap(3 * lap);

        let seconds = |s: u64| Duration::from_secs(s);
        assert_eq!(
            demuxer.0.inner_mut().locate_in_lap(seconds(3)),
            seconds(3),
            "a seek in the file"
        );
        assert_eq!(
            demuxer.0.inner().loop_offset,
            3 * lap,
            "and the lap left alone"
        );

        assert_eq!(
            demuxer.0.inner_mut().locate_in_lap(seconds(10)),
            seconds(2),
            "two seconds into the second lap"
        );
        assert_eq!(demuxer.0.inner().loop_offset, lap);
        assert_eq!(
            demuxer.0.inner().lap_end,
            lap,
            "a whole lap for the next wrap to step over"
        );
        assert_eq!(
            demuxer.looping_handle().lap_offset(),
            Duration::from_micros(lap as u64)
        );

        assert_eq!(
            demuxer.0.inner_mut().locate_in_lap(seconds(16)),
            seconds(8),
            "the end of the second lap is the end of the file, not the start of the third"
        );
        assert_eq!(demuxer.0.inner().loop_offset, lap);
    }

    /// Read backwards on a lap after the first, what comes out is stamped on
    /// that lap; at its start it goes on from the end of the lap before, and
    /// only the start of the first lap ends it.
    ///
    /// It read the file's own time as a position on the timeline: turned
    /// round a lap in, it went back from the end of the file, stamped a lap
    /// short, and the picture stopped until the clock came down to it.
    #[test]
    fn reading_backwards_goes_back_over_the_start_of_a_lap_that_was_played() {
        let Some(path) = try_test_video() else { return };
        let (mut demuxer, _) = FileDemuxer::open("demux", &path).expect("open test video");
        let (_, time_base) = video_stream(&demuxer);
        // As one wrap leaves it.
        let lap = demuxer.0.inner().input.duration().max(0);
        assert!(lap > 0, "the fixture says how long it is");
        demuxer.0.inner_mut().lap_length = lap;
        demuxer
            .0
            .inner()
            .published_lap
            .store(lap, Ordering::Relaxed);
        demuxer.0.inner_mut().move_to_lap(lap);

        let quarter = lap / 4;
        let landed = demuxer
            .0
            .inner_mut()
            .seek_backwards(Duration::from_micros((lap + quarter) as u64))
            .expect("turn round in the second lap");
        assert_eq!(
            landed,
            Duration::from_micros(quarter as u64),
            "a quarter into the file"
        );
        let mut seen = Vec::new();
        loop {
            match demuxer
                .0
                .inner_mut()
                .read_backwards()
                .expect("read backwards")
            {
                Produced::On(_, MediaBuffer::Packet(packet)) => {
                    seen.extend(packet.pts().or_else(|| packet.dts()));
                }
                Produced::End => break,
                _ => {}
            }
        }

        let micros = |ts: i64| ts.rescale(time_base, microseconds());
        let seen: Vec<i64> = seen.into_iter().map(micros).collect();
        assert!(
            seen.iter().any(|&at| (lap..=lap + quarter).contains(&at)),
            "the second lap, where it turned: {seen:?}"
        );
        assert!(
            seen.iter().any(|&at| (lap - quarter..lap).contains(&at)),
            "the end of the first lap, after the start of the second: {seen:?}"
        );
        assert!(
            seen.iter().all(|&at| (0..=lap + quarter).contains(&at)),
            "{seen:?}"
        );
        assert_eq!(
            demuxer.0.inner().loop_offset,
            0,
            "down to the first lap, where it ended"
        );
        let handle = demuxer.looping_handle();
        assert_eq!(handle.lap_offset(), Duration::ZERO);
        // Each read back on its own lap.
        assert_eq!(
            handle.in_lap(Duration::from_micros((lap + quarter) as u64)),
            Duration::from_micros(quarter as u64)
        );
    }

    /// A timestamp is read back on its own lap, not the one the demuxer is
    /// reading — which is ahead of what is shown by whatever the queues and
    /// the decoder hold.
    #[test]
    fn a_timestamp_is_read_back_on_its_own_lap() {
        let Some(path) = try_test_video() else { return };
        let (mut demuxer, _) = FileDemuxer::open("demux", &path).expect("open test video");
        let handle = demuxer.looping_handle();
        let seconds = |s: f64| Duration::from_secs_f64(s);
        assert_eq!(
            handle.in_lap(seconds(12.5)),
            seconds(12.5),
            "before any wrap, the file's own"
        );

        let lap = 8_000_000;
        demuxer.0.inner_mut().lap_length = lap;
        demuxer
            .0
            .inner()
            .published_lap
            .store(lap, Ordering::Relaxed);
        // Already reading the third lap.
        demuxer.0.inner_mut().move_to_lap(2 * lap);
        assert_eq!(
            handle.in_lap(seconds(15.5)),
            seconds(7.5),
            "the end of the second, still shown"
        );
        assert_eq!(handle.in_lap(seconds(17.0)), seconds(1.0), "the third");
        assert_eq!(
            handle.in_lap(seconds(16.0)),
            Duration::ZERO,
            "the start of a lap is the start of the file"
        );
    }

    /// Drives `FileDemuxer::run` directly (no `Pipeline`) to prove the
    /// basic contract on its own: every packet on a linked pad's stream
    /// arrives, and running off the end of the file delivers a final
    /// `Eos` rather than just stopping silently.
    /// A stream the file announces after it was opened — as MPEG-TS and
    /// FLV may — has no output, and its packets are passed over rather
    /// than ending the source. Played here by forgetting the last stream's
    /// output, as if it had not been there at the start.
    #[test]
    fn a_stream_with_no_output_is_passed_over() {
        let Some(path) = try_test_video() else { return };
        let (mut demuxer, streams) = FileDemuxer::open("demux", &path).expect("open test video");
        let late = streams.len() - 1;
        let mut read_late = 0;
        while let Some((index, _, _)) = demuxer.0.inner_mut().next_packet() {
            read_late += usize::from(index == late);
        }
        assert!(
            late > 0 && read_late > 0,
            "the fixture has a second stream, with packets"
        );
        let (mut demuxer, _) = FileDemuxer::open("demux", &path).expect("open test video");
        demuxer.0.inner_mut().contracts.truncate(late);
        let mut read = 0;
        while let Some((index, _, _)) = demuxer.0.inner_mut().next_packet() {
            assert!(
                index < late,
                "a packet of stream {index}, which has no output"
            );
            read += 1;
        }
        assert!(read > 0, "the other streams are still read");
    }

    #[test]
    fn run_delivers_every_packet_on_a_linked_pad_then_eos() {
        let Some(path) = try_test_video() else { return };
        let (mut demuxer, streams) = FileDemuxer::open("demux", &path).expect("open test video");
        let video = streams
            .iter()
            .find(|s| s.kind == ffmpeg::media::Type::Video)
            .expect("test video has a video stream");

        let count = Arc::new(AtomicUsize::new(0));
        let saw_eos = Arc::new(AtomicBool::new(false));
        let time_base_matches = Arc::new(AtomicBool::new(true));
        let expected_time_base = video.time_base;
        demuxer.src_pads()[video.index].link(Box::new(CountingSink {
            count: count.clone(),
            saw_eos: saw_eos.clone(),
            expected_time_base,
            time_base_matches: time_base_matches.clone(),
            pp_log: element_pp_log(ElementType::Other, "counting-sink", None),
        }));

        let (bus, bus_rx) = Bus::new();
        let (_, rx) = control::channel();
        demuxer
            .run(&rx, &bus)
            .expect("run must reach eos cleanly, not error");

        assert!(
            count.load(Ordering::SeqCst) > 0,
            "expected at least one packet delivered to the linked pad"
        );
        assert!(
            saw_eos.load(Ordering::SeqCst),
            "expected an Eos once the file is exhausted"
        );
        assert!(
            time_base_matches.load(Ordering::SeqCst),
            "every delivered packet must carry its stream time base"
        );
        drop(bus);
        assert!(
            bus_rx.iter().all(|e| !matches!(e, BusEvent::Error { .. })),
            "run must not report any errors demuxing a well-formed file"
        );
    }

    #[test]
    fn seek_read_ahead_packet_carries_its_stream_time_base_when_delivered() {
        let Some(path) = try_test_video() else { return };
        let (mut demuxer, _) = FileDemuxer::open("demux", &path).expect("open test video");

        demuxer
            .0
            .inner_mut()
            .seek(Duration::from_secs(1))
            .expect("seek within the test video");
        let (pending_index, expected_time_base, _) = demuxer
            .0
            .inner()
            .ahead
            .front()
            .expect("seek must retain the first packet at or after the target");
        let pending_index = *pending_index;
        let expected_time_base = *expected_time_base;

        let count = Arc::new(AtomicUsize::new(0));
        let saw_eos = Arc::new(AtomicBool::new(false));
        let time_base_matches = Arc::new(AtomicBool::new(true));
        demuxer.src_pads()[pending_index].link(Box::new(CountingSink {
            count: count.clone(),
            saw_eos,
            expected_time_base,
            time_base_matches: time_base_matches.clone(),
            pp_log: element_pp_log(ElementType::Other, "counting-sink", None),
        }));

        let (bus, _bus_rx) = Bus::new();
        let (_, rx) = control::channel();
        demuxer
            .run(&rx, &bus)
            .expect("run after seek must reach eos cleanly");

        assert!(
            count.load(Ordering::SeqCst) > 0,
            "the packet retained by seek must be delivered"
        );
        assert!(
            time_base_matches.load(Ordering::SeqCst),
            "the packet retained by seek must carry its stream time base"
        );
    }

    /// A seek starts a new timeline. What an earlier seek read ahead belongs
    /// to the old one, so handing it on after the reposition feeds pre-seek
    /// packets to a decoder whose reference state was just flushed —
    /// corrupt output, and a stream of `co located POCs unavailable` from
    /// libavcodec. What was held back for a blocked output goes with the
    /// `Flush` before the seek; see the framework's parking.
    #[test]
    fn a_seek_keeps_only_what_it_read_from_where_it_landed() {
        let Some(path) = try_test_video() else { return };
        let (mut demuxer, streams) = FileDemuxer::open("demux", &path).expect("open test video");
        let video = streams
            .iter()
            .find(|stream| stream.kind == ffmpeg::media::Type::Video)
            .expect("test video has a video stream");

        let demuxing = demuxer.0.inner_mut();
        demuxing
            .seek(Duration::ZERO)
            .expect("seek to the start of the test video");
        assert!(!demuxing.ahead.is_empty(), "the first seek's read-ahead");
        // Three seconds in lands well past what the first read, which is
        // the file's start.
        let landed = demuxing
            .seek(Duration::from_secs(3))
            .expect("seek within the test video");
        assert!(landed > Duration::from_secs(1), "landed at {landed:?}");

        assert!(!demuxing.ahead.is_empty(), "the seek's own read-ahead");
        for (_, time_base, packet) in &demuxing.ahead {
            let at = ts_to_duration(packet.pts().or(packet.dts()).unwrap_or(0), *time_base);
            assert!(
                at > Duration::from_secs(1),
                "a packet from before the seek survived it: {at:?}"
            );
        }
        assert_eq!(
            demuxing.ahead.back().map(|(index, _, _)| *index),
            Some(video.index),
            "read ahead as far as the first picture, and no further"
        );
    }

    /// Records what reaches a pad — how many packets, and whether its end
    /// did — and takes nothing while its gate is shut.
    struct EndRecorder {
        open: Arc<AtomicBool>,
        packets: Arc<AtomicUsize>,
        ended: Arc<AtomicBool>,
        pp_log: PpLog,
    }

    /// What an [`EndRecorder`] linked to a pad shows: its gate, what it
    /// counted, whether it ended.
    type EndRecord = (Arc<AtomicBool>, Arc<AtomicUsize>, Arc<AtomicBool>);

    impl EndRecorder {
        fn link(pad: &mut SrcPad, name: &str, open: bool) -> EndRecord {
            let (gate, packets, ended) = (
                Arc::new(AtomicBool::new(open)),
                Arc::new(AtomicUsize::new(0)),
                Arc::new(AtomicBool::new(false)),
            );
            pad.link(Box::new(EndRecorder {
                open: Arc::clone(&gate),
                packets: Arc::clone(&packets),
                ended: Arc::clone(&ended),
                pp_log: element_pp_log(ElementType::Other, name, None),
            }));
            (gate, packets, ended)
        }
    }

    impl Element for EndRecorder {
        fn name(&self) -> Arc<str> {
            "ends".into()
        }
        fn element_type(&self) -> ElementType {
            ElementType::Other
        }
        fn pp_log(&self) -> &PpLog {
            &self.pp_log
        }
        fn pp_log_mut(&mut self) -> &mut PpLog {
            &mut self.pp_log
        }
    }

    impl crate::element::RawSink for EndRecorder {
        fn ready_consume(&mut self) -> bool {
            self.open.load(Ordering::SeqCst)
        }
        fn consume(&mut self, _buf: MediaBuffer) -> crate::error::Result<()> {
            self.packets.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn stream_event(&mut self, event: &crate::stream::StreamEvent) -> crate::error::Result<()> {
            if let crate::stream::StreamEvent::Eos = event {
                self.ended.store(true, Ordering::SeqCst);
            }
            Ok(())
        }
    }

    /// At the end of the file a pad that owes nothing parked has its end at
    /// once, while another's parked packets still wait for room — and those
    /// go out in order, then that pad's end, once it has some.
    ///
    /// What the conformance matrix found: a seek to the last picture, past
    /// the last of the sound, has the picture's branch take its sample and
    /// nothing more, so the rest of its packets are parked; the sound's
    /// branch has nothing at or after the target and prerolls only on its
    /// end. Reading waited at the end of the file until every parked packet
    /// had gone, which was until the preroll ended, so the sound's end never
    /// came and the seek timed out.
    #[test]
    fn a_pad_owing_nothing_is_ended_while_another_still_owes_packets() {
        let Some(path) = try_test_video() else { return };
        let (mut demuxer, streams) = FileDemuxer::open("demux", &path).expect("open");
        let [held, ended] = [ffmpeg::media::Type::Video, ffmpeg::media::Type::Audio].map(|kind| {
            streams
                .iter()
                .find(|stream| stream.kind == kind)
                .expect("the fixture has a picture and a sound")
                .index
        });
        let (room, held_packets, held_ended) =
            EndRecorder::link(&mut demuxer.src_pads()[held], "held", false);
        let (_, _, sound_ended) = EndRecorder::link(&mut demuxer.src_pads()[ended], "ended", true);
        // In a preroll, as a seek's: a pad that takes nothing more has its
        // packets parked, and reading goes on for the other.
        let context = Arc::new(crate::element::Context::for_test_with_clock(
            Bus::new().0,
            "test",
            crate::graph::PipelineGraph::new(),
            crate::graph::ElementId::for_test(1),
            Arc::new(crate::clock::Clock::new()),
        ));
        demuxer.attach_context(&context);
        context
            .state
            .observe(&crate::control::ControlMsg::Preroll(Arc::new(
                crate::control::PrerollContext::new([]),
            )));
        let (requests, control) = crate::control::channel_in(&context.state);
        let (bus, _bus_rx) = Bus::new();
        // Answers rather than asserting: a demuxer left running would keep
        // the scope below from ever ending, so it is stopped first.
        let happens = |done: &dyn Fn() -> bool| {
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            while !done() {
                if std::time::Instant::now() >= deadline {
                    return false;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            true
        };

        let (sound_first, still_parked, picture_after, packets_first) =
            std::thread::scope(|scope| {
                let running = scope.spawn(|| demuxer.run(&control, &bus));
                let sound_first = happens(&|| sound_ended.load(Ordering::SeqCst));
                let still_parked =
                    held_packets.load(Ordering::SeqCst) == 0 && !held_ended.load(Ordering::SeqCst);
                room.store(true, Ordering::SeqCst);
                let picture_after = happens(&|| held_ended.load(Ordering::SeqCst));
                let packets_first = held_packets.load(Ordering::SeqCst) > 0;
                requests.send(crate::control::ControlMsg::Stop);
                running.join().expect("no panic").expect("no error");
                (sound_first, still_parked, picture_after, packets_first)
            });
        assert!(
            sound_first,
            "the sound was not ended while the picture still owed packets"
        );
        assert!(
            still_parked,
            "the picture took nothing, nor its end, meanwhile"
        );
        assert!(
            picture_after,
            "the picture's packets and end did not follow"
        );
        assert!(packets_first, "its packets went ahead of its end");
    }

    /// Writes a Matroska file whose first stream is one still picture and
    /// whose second is five moving ones — a cover image ahead of the video,
    /// in the shape a first-of-its-kind search trips over.
    fn still_ahead_of_video(path: &std::path::Path) -> Option<()> {
        let encode = |fill| {
            crate::test_support::try_encoded_packets(
                "mjpeg",
                ffmpeg::format::Pixel::YUVJ420P,
                (64, 48),
                fill,
            )
        };
        let (still_params, still_packets) = encode(40)?;
        let (video_params, video_packets) = encode(200)?;
        let mut output = ffmpeg::format::output(&path).ok()?;
        for params in [still_params, video_params] {
            let mut stream = output.add_stream(ffmpeg::encoder::find(params.id())).ok()?;
            stream.set_parameters(params);
            // SAFETY: a stream of an output this test owns, before its
            // header is written.
            unsafe {
                (*(*stream.as_mut_ptr()).codecpar).codec_tag = 0;
            }
        }
        output.write_header().ok()?;
        let source = ffmpeg::Rational::new(1, 30);
        let packets = still_packets
            .into_iter()
            .take(1)
            .map(|packet| (0, packet))
            .chain(video_packets.into_iter().map(|packet| (1, packet)));
        for (index, mut packet) in packets {
            let target = output.stream(index)?.time_base();
            packet.rescale_ts(source, target);
            packet.set_stream(index);
            packet.write_interleaved(&mut output).ok()?;
        }
        output.write_trailer().ok()?;
        Some(())
    }

    /// The first video stream is the still; the one to play is the video,
    /// and that is the one `best` answers.
    #[test]
    fn best_passes_over_a_still_ahead_of_the_video() {
        let path = std::env::temp_dir().join("media-pp-still-ahead-of-video.mkv");
        if still_ahead_of_video(&path).is_none() {
            eprintln!("skipping: could not write a file with a still ahead of its video");
            return;
        }
        let (demuxer, streams) = FileDemuxer::open("still-ahead", &path).unwrap();
        let first_video = streams
            .iter()
            .find(|stream| stream.kind == ffmpeg::media::Type::Video)
            .map(|stream| stream.index);
        assert_eq!(first_video, Some(0), "the still comes first");

        // `best` passes over it, as the stream's whole description, and says
        // which kind is missing rather than answering nothing.
        let video = demuxer
            .best(ffmpeg::media::Type::Video)
            .expect("the file has a video stream");
        assert_eq!(video.index, 1, "the video is the one to play");
        assert_eq!(video.time_base, streams[1].time_base);
        assert_eq!(video.parameters.id(), streams[1].parameters.id());
        let missing = demuxer
            .best(ffmpeg::media::Type::Audio)
            .expect_err("the file has no audio");
        assert!(matches!(
            missing,
            FileDemuxerError::NoStream(ffmpeg::media::Type::Audio)
        ));
        assert_eq!(missing.to_string(), "the file has no Audio stream");
    }

    /// What `open` reports for a stream is what asking by its index answers:
    /// the same codec, the same parameters, the same time base.
    #[test]
    fn stream_info_carries_what_the_stream_would_answer() {
        let Some(path) = try_test_video() else {
            return;
        };
        let (demuxer, streams) = FileDemuxer::open("info", &path).unwrap();
        assert!(!streams.is_empty());
        for info in &streams {
            let stream = demuxer
                .0
                .inner()
                .stream(info.index)
                .expect("a listed stream");
            let asked = stream.parameters();
            assert_eq!(info.parameters.id(), asked.id());
            assert_eq!(info.parameters.medium(), info.kind);
            assert_eq!(info.time_base, stream.time_base());
            assert!(
                format!("{info:?}").contains(&format!("{:?}", asked.id())),
                "its Debug names the codec"
            );
        }
    }
}
