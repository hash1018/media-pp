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
    control::{ControlReceiver, RequestKind, drain_control, handle_request},
    element::{
        Element, ElementType, ReversibleSource, SeekableSource, Source, SourceElement,
        element_pp_log,
    },
    pad::SrcPad,
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
/// Cheap to clone and safe to share: it holds one atomic flag and nothing
/// else, so it keeps neither the demuxer, its file, nor its pipeline alive.
/// No call blocks or does any work beyond that store. A call after the
/// source has finished is simply never read.
#[derive(Clone)]
pub struct FileDemuxerHandle {
    looping: Arc<AtomicBool>,
    published_offset: Arc<AtomicI64>,
}

impl FileDemuxerHandle {
    /// Whether reaching the end of the file starts it again instead of
    /// ending the stream. Off unless this says otherwise.
    ///
    /// Read once per lap, at the end of the file — never mid-file. So
    /// turning it off part way through means "play this lap out and then
    /// finish", not "stop now", and the stream still ends with a real
    /// `Eos` rather than being abandoned the way [`ControlMsg::Stop`]
    /// abandons it. Turning it on part way through takes effect at the end
    /// the source was already heading for.
    ///
    /// [`ControlMsg::Stop`]: crate::control::ControlMsg::Stop
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
    /// It moves once per lap, so a reader that samples it beside a timestamp
    /// from the same moment can be one lap out for the instant either side of
    /// a wrap. Nothing here can close that window, and a progress bar that is
    /// wrong for one frame at the moment it jumps back to zero is not wrong
    /// in a way anyone can see.
    pub fn lap_offset(&self) -> Duration {
        Duration::from_micros(self.published_offset.load(Ordering::Relaxed).max(0) as u64)
    }
}

/// Demuxes a file, exposing one src pad per container stream (indexed the
/// same way as `StreamInfo::index`). Linking a pad "selects" that stream;
/// leaving it unlinked just drops its packets. Real demuxer I/O is
/// blocking, so this is meant to be run as the pipeline's source thread.
///
/// Fan-out (e.g. routing video and audio to separate branches) needs no
/// separate "Tee" element here — it's just a matter of linking more than
/// one of these pads.
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
/// Run by hand, [`run`](crate::element::SourceElement::run) returns at that
/// point only once its control sender is gone.
///
/// [`FileDemuxer::seek`]: crate::element::SeekableSource::seek
pub struct FileDemuxer {
    pp_log: PpLog,
    name: Arc<str>,
    input: ffmpeg::format::context::Input,
    pads: Vec<SrcPad>,
    /// Packets read but not yet delivered, in file order.
    ///
    /// `seek` puts one here: peeking a packet right after `Input::seek` is how
    /// it learns where playback actually landed (see `seek`'s docs), and that
    /// packet is real data that still has to be delivered.
    ///
    /// `run` also parks a packet here when its own pad cannot accept one yet.
    /// A container interleaves every stream into one read cursor, so refusing
    /// to read at all while a single pad is blocked stalls the streams that
    /// *are* ready — during preroll that starves whichever branch has not yet
    /// taken its sample, and the seek times out waiting for it. Holding the
    /// blocked pad's packets keeps the cursor moving; per-pad order is what
    /// matters and each pad's packets stay in the order they were read.
    pending: VecDeque<(usize, ffmpeg::Rational, ffmpeg::Packet)>,
    /// Total payload parked in `pending`.
    pending_bytes: usize,
    /// How many parked packets each pad owes, so a freshly read one never
    /// overtakes them.
    parked_per_pad: Vec<usize>,
    /// When the oldest packet each pad owes is due, in nanoseconds of file
    /// time — how far behind the read cursor that pad has been left. See
    /// [`MAX_INTERLEAVE`].
    parked_since_ns: Vec<Option<i64>>,
    /// When the packet last read is due, in nanoseconds of file time.
    read_ns: Option<i64>,
    /// The pipeline's playback state, which says whether a preroll is
    /// running — see [`FileDemuxer::prerolling`].
    state: Option<std::sync::Arc<crate::playback_state::PlaybackState>>,
    /// Whether a `Seek` has repositioned it since [`Self::linger`] began
    /// waiting at the end of the file.
    sought: bool,
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
    /// The pipeline's playback clock, whose rate says which way to read —
    /// see [`Backwards`].
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
/// A stretch is [`BACKWARDS_STRETCH`] long, not a whole group of pictures:
/// the decoder holds a stretch's pictures to hand them on backwards, and a
/// group can be seconds long. The price is decoding the group's start again
/// for each stretch in it. Only the picture is read; the sound is not
/// played backwards, and its packets are passed over.
struct Backwards {
    /// The picture's stream.
    stream: usize,
    /// This stretch, in nanoseconds of the file.
    start: i64,
    end: i64,
    /// Whether this stretch has still to be sought to.
    unread: bool,
}

/// How much of the picture one stretch read backwards covers — see
/// [`Backwards`]. What the decoder holds at once, and how often a group of
/// pictures is decoded from its start again: a second of 1080p is about
/// 90 MB, and a five-second group is decoded three times over.
const BACKWARDS_STRETCH: Duration = Duration::from_secs(1);

/// How far the read cursor may run ahead of what a blocked pad still owes,
/// in file time, to keep another branch fed — see
/// [`FileDemuxer::may_park`]. Covers how far apart a container's streams are
/// muxed in practice, typically a second or less, with room to spare; past
/// it, the blocked pad is waited on.
const MAX_INTERLEAVE: Duration = Duration::from_secs(5);

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
    /// Whether a preroll is running. Any blocked pad is parked for then;
    /// outside one, only while another branch is waiting on the read cursor
    /// — see [`FileDemuxer::may_park`]. Never, for a demuxer no pipeline
    /// has wired.
    fn prerolling(&self) -> bool {
        // Including the pause that ends one, until it reaches this source:
        // what it parked stays parked — see `crate::playback_state`.
        self.state
            .as_ref()
            .is_some_and(|state| state.preroll().is_some())
    }

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

        let pads: Vec<SrcPad> = streams
            .iter()
            .map(|s| {
                // Per stream, from the medium the container announced:
                // both pads emit `MediaBuffer::Packet`, so only this tells
                // an audio stream apart from a video one. A medium this
                // crate does not model (subtitles, data) declares nothing
                // and is left to the runtime check.
                match MediaKind::packet_for(s.kind) {
                    Some(kind) => SrcPad::with_contract(
                        format!("src_{}", s.index),
                        OutputContract::Fixed(PortContract::packet(kind)),
                    ),
                    None => SrcPad::new(format!("src_{}", s.index)),
                }
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
            Self {
                name,
                pp_log,
                input,
                parked_per_pad: vec![0; pads.len()],
                parked_since_ns: vec![None; pads.len()],
                read_ns: None,
                state: None,
                sought: false,
                pads,
                pending: VecDeque::new(),
                pending_bytes: 0,
                looping: Arc::new(AtomicBool::new(false)),
                backwards: None,
                published_offset: Arc::new(AtomicI64::new(0)),
                loop_offset: 0,
                lap_end: 0,
            },
            streams,
        ))
    }

    /// The control endpoint for looping this file, valid for as long as the
    /// demuxer runs — take it here, before moving the demuxer into its
    /// pipeline, and keep it for as long as the loop is meant to be
    /// switchable. See [`FileDemuxerHandle::set_looping`].
    pub fn looping_handle(&self) -> FileDemuxerHandle {
        FileDemuxerHandle {
            looping: self.looping.clone(),
            published_offset: self.published_offset.clone(),
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
        self.input
            .streams()
            .best(kind)
            .map(|stream| StreamInfo::of(&stream))
            .ok_or(FileDemuxerError::NoStream(kind))
    }

    /// How long the file plays, as its container says — `None` where it
    /// says nothing, as a live capture or a truncated recording may not.
    pub fn duration(&self) -> Option<Duration> {
        let micros = self.input.duration();
        (micros > 0).then(|| Duration::from_micros(micros as u64))
    }

    #[cfg(test)]
    fn stream(&self, index: usize) -> Option<ffmpeg::format::stream::Stream<'_>> {
        self.input.streams().find(|s| s.index() == index)
    }

    /// Puts a freshly read packet on this source's output timeline, and
    /// records how far into the file this lap has now reached.
    ///
    /// Called at each of the two places a packet is read out of the
    /// container — `run`'s cursor and `seek`'s read-ahead — rather than
    /// where they are delivered. Stamping a time base is idempotent and
    /// `deliver_or_park` can do it to the same packet twice; shifting a
    /// timestamp is not.
    ///
    /// Only a linked pad's stream counts towards the lap's length. An
    /// unlinked one is dropped rather than delivered, so letting a longer
    /// audio track nobody selected decide where the video restarts would
    /// only open a gap at every join.
    fn stamp_lap(
        &mut self,
        index: usize,
        time_base: ffmpeg::Rational,
        packet: &mut ffmpeg::Packet,
    ) {
        if let Some(start) = packet.pts().or_else(|| packet.dts())
            && self.pads.get(index).is_some_and(SrcPad::is_linked)
        {
            // A packet carrying no duration of its own still ends after it
            // starts, and one tick is the least that keeps the next lap's
            // first timestamp past this one's rather than equal to it.
            let end = start.saturating_add(packet.duration().max(1));
            self.lap_end = self.lap_end.max(end.rescale(time_base, microseconds()));
        }
        if self.loop_offset == 0 {
            return;
        }
        let shift = self.loop_offset.rescale(microseconds(), time_base);
        packet.set_pts(packet.pts().map(|pts| pts.saturating_add(shift)));
        packet.set_dts(packet.dts().map(|dts| dts.saturating_add(shift)));
    }

    /// Starts the file again: carries the output timeline past the lap that
    /// just ended, then rewinds the container.
    ///
    /// Ordering matters. The offset moves first so that the read-ahead
    /// packet `seek` parks — the new lap's first — is stamped onto the new
    /// timeline like every packet after it.
    fn wrap(&mut self) -> crate::error::Result<()> {
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

    /// Pushes `item` if its pad can take one now, otherwise parks it.
    ///
    /// A downstream failure drops just that one packet — the same
    /// "report, don't die" contract `Queue`'s worker gives a failing `Sink` —
    /// rather than ending this whole source thread over it. `Pipeline::stop`
    /// is how a caller who decides an error is fatal actually ends things.
    fn deliver_or_park(
        &mut self,
        item: (usize, ffmpeg::Rational, ffmpeg::Packet),
        bus: &Bus,
    ) -> crate::error::Result<()> {
        let (index, time_base, mut packet) = item;
        // `AVCodecParameters` does not carry the container stream's timestamp
        // unit, and FFmpeg does not guarantee that demuxers populate
        // `AVPacket::time_base`. Stamping it here — the one place every packet
        // leaves this source through, parked or not — means a packet held for
        // a blocked pad arrives downstream describing itself the same way an
        // immediately delivered one does.
        packet.set_time_base(time_base);
        if self.pads.get(index).is_none() {
            // No pad for this stream: nobody selected it, so it is dropped
            // rather than parked. Parking it would grow without bound.
            return Ok(());
        }
        // Anything already parked for this pad was read first and must stay
        // first. Overtaking it would hand a decoder its stream out of decode
        // order.
        let blocked = self.parked_per_pad[index] > 0 || !self.pads[index].ready_consume();
        if blocked && self.may_park(index) {
            self.park((index, time_base, packet));
            return Ok(());
        }
        // Otherwise the pad is past the interleave bound and is waited on:
        // the push below blocks until the downstream `Queue` has room. What
        // it already owes goes first, having been read before this.
        if self.parked_per_pad[index] > 0 {
            self.deliver_parked(index, bus);
        }
        self.push_to_pad(index, packet, bus);
        Ok(())
    }

    /// Delivers every packet pad `index` owes, oldest first, waiting on the
    /// pad as any push does.
    fn deliver_parked(&mut self, index: usize, bus: &Bus) {
        let mut rest = VecDeque::with_capacity(self.pending.len());
        while let Some((owner, time_base, packet)) = self.pending.pop_front() {
            if owner != index {
                rest.push_back((owner, time_base, packet));
                continue;
            }
            self.pending_bytes = self.pending_bytes.saturating_sub(packet.size());
            self.parked_per_pad[index] -= 1;
            self.push_to_pad(index, packet, bus);
        }
        self.pending = rest;
        self.parked_since_ns[index] = None;
    }

    /// Whether a packet for blocked pad `index` may be held back so the read
    /// cursor can move on, rather than waited on.
    ///
    /// Always during a preroll, which needs every branch's first sample.
    /// Otherwise within [`MAX_INTERLEAVE`] of what the pads already owe.
    ///
    /// That second case is a file whose picture is muxed ahead of its sound.
    /// The video queue fills with frames the audio has not reached; waiting
    /// on it stops this thread before the audio packets that would let the
    /// audio reach them, and the video waits on the audio for good. Holding
    /// the video packets back keeps both fed.
    ///
    /// Whether anything else can take the cursor now is not asked here: it is
    /// what `pending_blocked` asks before each read, which is what keeps this
    /// source from running away from playback — holding without any bound
    /// once buffered 67 MB within 1.5 s. Asked here, it held the video back
    /// only while the sound's own queue had room at that very moment; a
    /// packet read while both were full was pushed into the video's and
    /// waited on there, and once the sound drained this thread was still
    /// waiting on the video that waited on the sound — a player froze a
    /// few seconds into a file whose picture was muxed a second ahead.
    fn may_park(&mut self, _index: usize) -> bool {
        self.prerolling() || !self.interleave_exceeded()
    }

    /// Whether the read cursor has got [`MAX_INTERLEAVE`] ahead of the oldest
    /// packet any pad still owes.
    fn interleave_exceeded(&self) -> bool {
        let Some(read_ns) = self.read_ns else {
            return false;
        };
        self.parked_since_ns
            .iter()
            .flatten()
            .any(|&since| read_ns.saturating_sub(since) > duration_ns(MAX_INTERLEAVE))
    }

    fn push_to_pad(&mut self, index: usize, packet: ffmpeg::Packet, bus: &Bus) {
        if let Err(error) = self.pads[index].push(MediaBuffer::Packet(Arc::new(packet))) {
            bus.post_downstream_error(
                &self.pp_log,
                ElementType::FileDemuxer,
                self.name.clone(),
                error,
            );
        }
    }

    /// Holds one packet back, keeping the totals that bound the backlog in
    /// step with the queue. The one place anything enters `pending`, so it is
    /// also where the stream time base is guaranteed onto a packet that will
    /// be delivered later — `seek` parks its read-ahead packet directly.
    fn park(&mut self, item: (usize, ffmpeg::Rational, ffmpeg::Packet)) {
        let (index, time_base, mut packet) = item;
        packet.set_time_base(time_base);
        self.pending_bytes = self.pending_bytes.saturating_add(packet.size());
        if self.parked_per_pad[index] == 0 {
            self.parked_since_ns[index] = packet_ns(&packet, time_base).or(self.read_ns);
        }
        self.parked_per_pad[index] += 1;
        self.pending.push_back((index, time_base, packet));
    }

    /// Delivers every parked packet whose pad can take one, oldest first,
    /// leaving the rest in place.
    ///
    /// Once a pad has held one packet back, every later packet of *that* pad
    /// is held too, even if the pad reports itself ready again a moment later.
    /// Readiness here is a `Queue`'s "not full", which its worker changes on
    /// another thread while this loop runs — so re-asking per packet would let
    /// the second overtake the first the instant a slot opened, and a decoder
    /// handed its stream out of order produces garbage and a flood of
    /// `co located POCs unavailable`. Other pads are unaffected: keeping the
    /// read cursor moving for them is the whole reason parking exists.
    fn drain_pending(&mut self, bus: &Bus) -> crate::error::Result<()> {
        // The ordinary case — nothing is held back: this runs once per
        // packet read, so it must not allocate to find nothing.
        if self.pending.is_empty() {
            return Ok(());
        }
        let mut deferred = VecDeque::with_capacity(self.pending.len());
        let mut blocked = vec![false; self.pads.len()];
        while let Some((index, time_base, packet)) = self.pending.pop_front() {
            // Still held while its pad cannot take it: pushing would block
            // this thread on that pad, which is what parking is there to
            // avoid. When reading has to stop instead, `pending_blocked` says
            // so, and this delivers as the pad frees up.
            let hold = blocked[index] || !self.pads[index].ready_consume();
            if hold {
                blocked[index] = true;
                deferred.push_back((index, time_base, packet));
                continue;
            }
            self.pending_bytes = self.pending_bytes.saturating_sub(packet.size());
            self.parked_per_pad[index] -= 1;
            // The next it owes, if any, is the one that bounds it now.
            self.parked_since_ns[index] = None;
            self.push_to_pad(index, packet, bus);
        }
        for (index, time_base, packet) in &deferred {
            if self.parked_since_ns[*index].is_none() {
                self.parked_since_ns[*index] = packet_ns(packet, *time_base).or(self.read_ns);
            }
        }
        self.pending = deferred;
        Ok(())
    }

    /// Waits at the end of the file, passing every control request on to
    /// the branches — pausing and resuming them as the pipeline does — until
    /// it is stopped or finished, `Ok(false)`, or a seek moves it back into
    /// the file, `Ok(true)`, to read on from there.
    fn linger(&mut self, control: &ControlReceiver, bus: &Bus) -> crate::error::Result<bool> {
        self.sought = false;
        loop {
            let Some((request, ack)) = control.recv() else {
                return Ok(false);
            };
            if matches!(request, RequestKind::Finish) {
                // A graceful finish: the `Eos` it would push is already out.
                let _ = ack.send(());
                return Ok(false);
            }
            // Handled as a source handles one anywhere, a `Pause` included:
            // nothing is read until a `Resume`, or a seek's `Preroll` asking
            // for its picture. Passed on without pausing, as it once was
            // here, a seek from the end read on into the paused queue after
            // this and never took the `Preroll` it then waited on.
            if handle_request(control, self, bus, request, ack)?.stopped {
                return Ok(false);
            }
            if self.sought {
                return Ok(true);
            }
        }
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
    /// the stream it seeks by, parking all of it for `run`, and answers
    /// where the first packet of all starts and when that stream's first
    /// starts and is decoded — `None` for either past the end.
    ///
    /// Where a seek landed is only known by reading: `avformat_seek_file`
    /// says nothing but whether it failed. What is read is real data, not a
    /// probe to throw away, which is why it is parked rather than dropped.
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
            let Some((index, time_base, mut packet)) = self
                .input
                .packets()
                .next()
                .map(|(stream, packet)| (stream.index(), stream.time_base(), packet))
            else {
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
            self.stamp_lap(index, time_base, &mut packet);
            self.park((index, time_base, packet));
            if seeking_by {
                return (first, landing);
            }
        }
        (first, None)
    }

    /// Forgets every packet read and parked, for a new position to be read
    /// from — a `Flush`'s, or a seek's that landed too late.
    fn forget_parked(&mut self) {
        self.pending.clear();
        self.pending_bytes = 0;
        self.parked_per_pad.fill(0);
        self.parked_since_ns.fill(None);
        self.read_ns = None;
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

    /// Whether reading another packet would only deepen the parked backlog.
    ///
    /// Not a tuning knob: a branch that is briefly behind parks a handful of
    /// packets and clears them within milliseconds. These ceilings only bound
    /// a pad that has stopped accepting altogether, so the read cursor cannot
    /// pull an arbitrary amount of the file into memory waiting for it. Both
    /// are needed — a few large keyframes reach the byte limit at a packet
    /// count that would never trip on its own.
    fn pending_blocked(&mut self) -> bool {
        if self.pending.is_empty() {
            return false;
        }
        // Outside a preroll, a pad held back past the interleave bound is
        // waited on: reading on would only park more for it.
        if !self.prerolling() && self.interleave_exceeded() {
            return true;
        }
        const MAX_PENDING_PACKETS: usize = 4_096;
        const MAX_PENDING_BYTES: usize = 64 * 1024 * 1024;

        self.pending.len() >= MAX_PENDING_PACKETS
            || self.pending_bytes >= MAX_PENDING_BYTES
            || !self
                .pads
                .iter_mut()
                .any(|pad| pad.is_linked() && pad.ready_consume())
    }
}

impl Element for FileDemuxer {
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
        self.state = Some(std::sync::Arc::clone(&context.state));
    }
}

impl Source for FileDemuxer {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        &mut self.pads
    }
}

impl SourceElement for FileDemuxer {
    fn is_live(&self) -> bool {
        false
    }

    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> crate::error::Result<()> {
        pp_info!(self, "started");
        // Deliberately re-creates `self.input.packets()` fresh every
        // iteration (cheap — it's just a short-lived wrapper, not a
        // stateful cursor of its own) instead of holding one `for` loop's
        // iterator across the whole function, the way this used to read.
        // That iterator borrows `input` for as long as it's alive; `Seek`
        // needs `drain_control` to be able to call `self.seek()` — a
        // *second* mutable borrow of `input` — in between reads, which a
        // single loop-spanning iterator would rule out.
        loop {
            loop {
                if drain_control(control, self, bus)?.stopped {
                    // Stop: abandon in place, no final Eos.
                    pp_info!(self, "stopped");
                    return Ok(());
                }
                if self.backwards.is_some() {
                    if self.read_backwards(bus)? {
                        continue;
                    }
                    // The start of the file: the end of the stream, backwards.
                    break;
                }
                // Deliver whatever is parked and now accepted, oldest first.
                // Skipping a still-blocked pad's entry to reach a later one is
                // safe: only each pad's own order has to hold, and this preserves
                // it because entries for one pad are never reordered against each
                // other.
                self.drain_pending(bus)?;
                // Read on unless everything is blocked, or the parked backlog has
                // grown past what one branch briefly falling behind can explain.
                // Sleeping is the only option then: the container cannot hand out
                // a different stream's packet without reading this one.
                if self.pending_blocked() {
                    std::thread::sleep(Duration::from_millis(1));
                    continue;
                }
                let next = self
                    .input
                    .packets()
                    .next()
                    .map(|(s, p)| (s.index(), s.time_base(), p));
                let Some((index, time_base, mut packet)) = next else {
                    if !self.pending.is_empty() {
                        // Nothing left to read, but a blocked pad still owes
                        // delivery.
                        std::thread::sleep(Duration::from_millis(1));
                        continue;
                    }
                    // The end of the file, and the one place the loop flag is
                    // read: a change made mid-file lands here, at the end the
                    // source was already heading for.
                    if self.looping.load(Ordering::Relaxed) {
                        match self.wrap() {
                            Ok(()) => continue,
                            // A file that cannot be rewound cannot be looped,
                            // but it has been fully read — so report why the
                            // loop stopped and end the stream properly, rather
                            // than failing a source that delivered everything
                            // it was asked for.
                            Err(error) => bus.post(
                                &self.pp_log,
                                BusEvent::Error {
                                    element_type: ElementType::FileDemuxer,
                                    name: self.name.clone(),
                                    error,
                                },
                            ),
                        }
                    }
                    break;
                };
                self.stamp_lap(index, time_base, &mut packet);
                if let Some(ns) = packet_ns(&packet, time_base) {
                    self.read_ns = Some(ns);
                }
                // `deliver_or_park` stamps the stream time base; every packet
                // leaves this source through it, parked or not.
                self.deliver_or_park((index, time_base, packet), bus)?;
            }
            for pad in self.pads.iter_mut() {
                pad.push_eos(&self.pp_log)?;
            }
            pp_info!(self, "event=eos phase=source_completed outcome=ok");
            // The end of the file is not the end of playback: the queues after
            // this still hold what was read, a second of it or more, and they
            // belong to this thread — ending it would drop them with it, and a
            // `Pause` or `Resume` sent from then on would reach nothing. In a
            // paused seek's preroll that is the `Eos` itself, held in a paused
            // queue, and the file would never finish. So it stays, passing
            // control on, until it is stopped — or sought back into the file.
            if !self.linger(control, bus)? {
                return Ok(());
            }
        }
    }

    fn on_control(&mut self, msg: &crate::control::ControlMsg) {
        use crate::control::ControlMsg;
        match msg {
            // Those packets were read from the timeline being left behind,
            // and this source is the only place they exist — every downstream
            // stage discards its own on the same `Flush`, so releasing these
            // afterwards would be the one way old media could reach a decoder
            // that had already reset for the new position.
            ControlMsg::Flush => self.forget_parked(),
            ControlMsg::Seek(_) => self.sought = true,
            ControlMsg::Pause | ControlMsg::Resume | ControlMsg::Preroll(_) | ControlMsg::Stop => {}
        }
    }

    /// A file with a picture: that is what is read backwards.
    fn as_reversible(&mut self) -> Option<&mut dyn ReversibleSource> {
        if self.picture_stream().is_some() {
            Some(self)
        } else {
            None
        }
    }

    fn as_seekable(&mut self) -> Option<&mut dyn SeekableSource> {
        Some(self)
    }
}

impl SeekableSource for FileDemuxer {
    fn seek(&mut self, target: Duration) -> crate::error::Result<Duration> {
        self.backwards = None;
        self.reposition(target)
    }
}

/// The picture's stream, read back a second at a time. What
/// nothing is wired to is read all the same and handed to no one, down to
/// the start of the file, where the stream ends.
impl ReversibleSource for FileDemuxer {
    fn seek_backwards(&mut self, target: Duration) -> crate::error::Result<Duration> {
        let stream = self
            .picture_stream()
            .ok_or(FileDemuxerError::NoStream(ffmpeg::media::Type::Video))?;
        // The picture at `target` is the last of the first stretch.
        let end = duration_ns(target).saturating_add(1);
        self.backwards = Some(Backwards {
            stream,
            start: end,
            end,
            unread: true,
        });
        self.start_stretch()?;
        Ok(target)
    }

    /// Not the one after: `unread` marks the one just read as the last.
    fn finish_stretch(&mut self, bus: &Bus) -> crate::error::Result<()> {
        while self
            .backwards
            .as_ref()
            .is_some_and(|backwards| !backwards.unread)
        {
            if !self.read_backwards(bus)? {
                break;
            }
        }
        Ok(())
    }
}

impl FileDemuxer {
    /// The picture's stream, where the file has one to play backwards.
    fn picture_stream(&self) -> Option<usize> {
        self.input
            .streams()
            .best(ffmpeg::media::Type::Video)
            .map(|stream| stream.index())
    }

    /// Seeks to the stretch before the one just read — see [`Backwards`].
    fn start_stretch(&mut self) -> crate::error::Result<()> {
        let Some(backwards) = self.backwards.as_mut() else {
            return Ok(());
        };
        backwards.start = backwards
            .end
            .saturating_sub(duration_ns(BACKWARDS_STRETCH))
            .max(0);
        backwards.unread = false;
        let start = Duration::from_nanos(backwards.start.max(0) as u64);
        self.forget_parked();
        self.reposition(start)?;
        Ok(())
    }

    /// Reads one packet backwards — see [`Backwards`] — and answers
    /// whether there is more to read: `false` once the stretch that begins
    /// the file has been read.
    fn read_backwards(&mut self, bus: &Bus) -> crate::error::Result<bool> {
        if self
            .backwards
            .as_ref()
            .is_some_and(|backwards| backwards.unread)
        {
            self.start_stretch()?;
            return Ok(true);
        }
        let next = self.take_parked().or_else(|| {
            self.input
                .packets()
                .next()
                .map(|(stream, packet)| (stream.index(), stream.time_base(), packet))
        });
        let Some(backwards) = self.backwards.as_mut() else {
            return Ok(false);
        };
        let Some((index, time_base, mut packet)) = next else {
            // The file ran out inside the stretch: it is read.
            return Ok(self.stretch_read());
        };
        if index != backwards.stream {
            return Ok(true);
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
        packet.set_time_base(time_base);
        self.push_to_pad(index, packet, bus);
        Ok(true)
    }

    /// The stretch just read is done: on to the one before, unless it began
    /// the file.
    fn stretch_read(&mut self) -> bool {
        let Some(backwards) = self.backwards.as_mut() else {
            return false;
        };
        if backwards.start <= 0 {
            return false;
        }
        backwards.end = backwards.start;
        backwards.unread = true;
        true
    }

    /// The oldest parked packet, taken out of the parked accounts.
    fn take_parked(&mut self) -> Option<(usize, ffmpeg::Rational, ffmpeg::Packet)> {
        let (index, time_base, packet) = self.pending.pop_front()?;
        self.pending_bytes = self.pending_bytes.saturating_sub(packet.size());
        if let Some(parked) = self.parked_per_pad.get_mut(index) {
            *parked = parked.saturating_sub(1);
            if *parked == 0 {
                self.parked_since_ns[index] = None;
            }
        }
        Some((index, time_base, packet))
    }

    /// Moves the read cursor to `target`, forwards — see
    /// [`SourceElement::seek`].
    fn reposition(&mut self, target: Duration) -> crate::error::Result<Duration> {
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
            self.forget_parked();
        }
        // Nothing left to read (`target` at/past EOF): no packet to learn a
        // real position from, so the request is reported back as it is.
        Ok(landed.unwrap_or(target))
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
        use crate::element::Sink;
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
        sink.consume(MediaBuffer::Eos).unwrap();
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
    use crate::control;
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

    impl crate::element::Sink for CountingSink {
        fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
            match buf {
                MediaBuffer::Eos => self.saw_eos.store(true, Ordering::SeqCst),
                MediaBuffer::Packet(packet) => {
                    if packet.time_base() != self.expected_time_base {
                        self.time_base_matches.store(false, Ordering::SeqCst);
                    }
                    self.count.fetch_add(1, Ordering::SeqCst);
                }
                _ => {}
            }
            Ok(())
        }
    }

    /// `open` lists every stream, each at its own index — which is also its
    /// pad's — so a caller can attach by the `index` it was handed.
    #[test]
    fn open_lists_every_stream_at_its_own_index() {
        let Some(path) = try_test_video() else { return };
        let (demuxer, streams) = FileDemuxer::open("demux", &path).expect("open test video");

        assert_eq!(streams.len(), demuxer.pads.len());
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

    impl crate::element::Sink for LoopSink {
        fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
            match buf {
                MediaBuffer::Eos => self.saw_eos.store(true, Ordering::SeqCst),
                MediaBuffer::Packet(packet) => {
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
                _ => {}
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
            .seek(Duration::ZERO)
            .expect("seek to the start of the test video");
        let at_start = demuxer.lap_end;

        // Half way in, so the keyframe this lands on is somewhere past the
        // first one for any file that has more than one — which is what the
        // guard below checks rather than assumes.
        let half = Duration::from_micros((demuxer.input.duration().max(0) / 2) as u64);
        demuxer
            .seek(half)
            .expect("seek half way into the test video");
        let reached = demuxer.lap_end;
        if reached <= at_start {
            // Nothing in this fixture is reachable past its own start, so
            // there is no reach for a seek back to lose.
            return;
        }

        demuxer
            .seek(Duration::ZERO)
            .expect("seek back to the start of the test video");

        assert_eq!(
            demuxer.lap_end, reached,
            "going back must not shorten the lap the next wrap steps over"
        );
    }

    /// Drives `FileDemuxer::run` directly (no `Pipeline`) to prove the
    /// basic contract on its own: every packet on a linked pad's stream
    /// arrives, and running off the end of the file delivers a final
    /// `Eos` rather than just stopping silently.
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
            .seek(Duration::from_secs(1))
            .expect("seek within the test video");
        let (pending_index, expected_time_base, _) = demuxer
            .pending
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

    /// A seek starts a new timeline. Anything the demuxer had already read and
    /// parked for a blocked pad belongs to the old one, so delivering it after the
    /// reposition feeds pre-seek packets to a decoder whose reference state was
    /// just flushed — corrupt output, and a stream of `co located POCs
    /// unavailable` from libavcodec.
    #[test]
    fn seeking_discards_packets_parked_before_the_jump() {
        let Some(path) = try_test_video() else { return };
        let (mut demuxer, streams) = FileDemuxer::open("demux", &path).expect("open test video");
        let video = streams
            .iter()
            .find(|stream| stream.kind == ffmpeg::media::Type::Video)
            .expect("test video has a video stream");
        let time_base = video.time_base;

        // Read a little of the start into the parked queue by hand, the way `run`
        // does when a pad cannot accept a packet yet.
        let mut input = ffmpeg::format::input(&path).expect("second handle");
        let mut parked = 0usize;
        for (stream, packet) in input.packets() {
            if stream.index() != video.index {
                continue;
            }
            demuxer.park((stream.index(), time_base, packet));
            parked += 1;
            if parked == 4 {
                break;
            }
        }
        assert_eq!(parked, 4, "the fixture must have packets to park");

        // The order a pipeline uses: Flush marks the timeline boundary, Seek
        // then moves the cursor and reads ahead, up to its first picture, to
        // learn where it landed. Three seconds in lands well past the four
        // parked, which are the file's first.
        demuxer.on_control(&crate::control::ControlMsg::Flush);
        let landed = demuxer
            .seek(Duration::from_secs(3))
            .expect("seek within the test video");
        assert!(landed > Duration::from_secs(1), "landed at {landed:?}");

        assert!(!demuxer.pending.is_empty(), "the seek's own read-ahead");
        for (_, time_base, packet) in &demuxer.pending {
            let at = ts_to_duration(packet.pts().or(packet.dts()).unwrap_or(0), *time_base);
            assert!(
                at > Duration::from_secs(1),
                "a packet from before the seek survived the flush: {at:?}"
            );
        }
        assert_eq!(
            demuxer.pending.back().map(|(index, _, _)| *index),
            Some(video.index),
            "read ahead as far as the first picture, and no further"
        );
        let counted: usize = demuxer
            .pending
            .iter()
            .map(|(_, _, packet)| packet.size())
            .sum();
        assert_eq!(
            demuxer.pending_bytes, counted,
            "the parked byte total must stay in step with the queue"
        );
    }

    /// Reports "full" until asked a given number of times, then reports room
    /// again — a `Queue` whose worker frees a slot partway through a drain.
    /// Readiness flipping *during* the loop is the real behaviour: the worker
    /// runs on its own thread, so the answer is not stable across the packets
    /// of one pass.
    struct FlipToReadySink {
        refusals_left: Arc<AtomicUsize>,
        seen: Arc<Mutex<Vec<i64>>>,
        pp_log: PpLog,
    }

    impl Element for FlipToReadySink {
        fn name(&self) -> Arc<str> {
            "flip-to-ready".into()
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

    impl crate::element::Sink for FlipToReadySink {
        fn ready_consume(&mut self) -> bool {
            let left = self.refusals_left.load(Ordering::SeqCst);
            if left > 0 {
                self.refusals_left.store(left - 1, Ordering::SeqCst);
                return false;
            }
            true
        }

        fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
            if let MediaBuffer::Packet(packet) = &buf
                && let Some(pts) = packet.pts()
            {
                self.seen.lock().unwrap().push(pts);
            }
            Ok(())
        }
    }

    /// A pad that says "full" for the first packet of a drain and "ready" for
    /// the next must not let the second overtake the first.
    ///
    /// This is what a `Queue` does: its worker frees a slot on another thread
    /// while the drain loop is running, so asking again per packet gives a
    /// different answer mid-pass. Re-asking let the later packet through
    /// first, handing the decoder its stream out of decode order — libavcodec
    /// answers with a flood of `co located POCs unavailable`, and the picture
    /// is wrong. Reproduced by launching `av_playback` with no seek at all:
    /// 45 warnings against 0 before this branch.
    #[test]
    fn a_pad_that_becomes_ready_mid_drain_does_not_reorder_its_stream() {
        let Some(path) = try_test_video() else { return };
        let (mut demuxer, streams) = FileDemuxer::open("demux", &path).expect("open");
        let video = streams
            .iter()
            .find(|stream| stream.kind == ffmpeg::media::Type::Video)
            .expect("video stream");
        let time_base = video.time_base;

        // Refuse once: the first packet of the drain is held back, and every
        // later one finds the pad ready again.
        let seen = Arc::new(Mutex::new(Vec::new()));
        demuxer.pads[video.index].link(Box::new(FlipToReadySink {
            refusals_left: Arc::new(AtomicUsize::new(1)),
            seen: Arc::clone(&seen),
            pp_log: element_pp_log(ElementType::Other, "flip-to-ready", None),
        }));

        let mut input = ffmpeg::format::input(&path).expect("second handle");
        let mut order = Vec::new();
        for (stream, packet) in input.packets() {
            if stream.index() != video.index {
                continue;
            }
            order.push(packet.pts().expect("fixture packets carry a pts"));
            demuxer.park((stream.index(), time_base, packet));
            if order.len() == 4 {
                break;
            }
        }

        let (bus, _bus_rx) = Bus::new();
        demuxer.drain_pending(&bus).expect("first drain");
        demuxer.drain_pending(&bus).expect("second drain");

        assert_eq!(
            *seen.lock().unwrap(),
            order,
            "a packet overtook one still parked for the same pad"
        );
    }

    /// Accepts a bounded number of buffers, then blocks — the shape of a
    /// `Queue` filling up mid-drain.
    struct BoundedSink {
        remaining: Arc<AtomicUsize>,
        seen: Arc<Mutex<Vec<i64>>>,
        pp_log: PpLog,
    }

    impl Element for BoundedSink {
        fn name(&self) -> Arc<str> {
            "bounded".into()
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

    impl crate::element::Sink for BoundedSink {
        fn ready_consume(&mut self) -> bool {
            self.remaining.load(Ordering::SeqCst) > 0
        }

        fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
            if let MediaBuffer::Packet(packet) = &buf
                && let Some(pts) = packet.pts()
            {
                self.seen.lock().unwrap().push(pts);
            }
            self.remaining.fetch_sub(1, Ordering::SeqCst);
            Ok(())
        }
    }

    /// A stream's packets must reach its pad in the order they were read.
    /// Parking exists so a blocked pad does not stall the read cursor; it must
    /// not reorder that pad's own packets, or the decoder is handed frames out
    /// of decode order and produces `co located POCs unavailable` and garbage.
    #[test]
    fn parked_packets_keep_their_per_pad_order_when_the_pad_blocks_mid_drain() {
        let Some(path) = try_test_video() else { return };
        let (mut demuxer, streams) = FileDemuxer::open("demux", &path).expect("open");
        let video = streams
            .iter()
            .find(|stream| stream.kind == ffmpeg::media::Type::Video)
            .expect("video stream");
        let time_base = video.time_base;

        // Room for one buffer only, so the pad blocks partway through the
        // drain and the rest have to be parked again.
        let remaining = Arc::new(AtomicUsize::new(1));
        let seen = Arc::new(Mutex::new(Vec::new()));
        demuxer.pads[video.index].link(Box::new(BoundedSink {
            remaining: Arc::clone(&remaining),
            seen: Arc::clone(&seen),
            pp_log: element_pp_log(ElementType::Other, "bounded", None),
        }));

        let mut input = ffmpeg::format::input(&path).expect("second handle");
        let mut order = Vec::new();
        for (stream, packet) in input.packets() {
            if stream.index() != video.index {
                continue;
            }
            order.push(packet.pts().expect("fixture packets carry a pts"));
            demuxer.park((stream.index(), time_base, packet));
            if order.len() == 5 {
                break;
            }
        }

        let (bus, _bus_rx) = Bus::new();
        // Drain repeatedly, opening one slot at a time, until everything has
        // been delivered.
        for _ in 0..order.len() {
            demuxer.drain_pending(&bus).expect("drain");
            remaining.fetch_add(1, Ordering::SeqCst);
        }
        demuxer.drain_pending(&bus).expect("final drain");

        assert_eq!(
            *seen.lock().unwrap(),
            order,
            "packets reached the pad out of the order they were read"
        );
    }

    /// Has room only while its gate is open, but accepts whatever is pushed —
    /// a pad reporting backpressure without a `Queue`'s blocking behind it,
    /// so a test sees what the source would have waited on.
    struct GatedRecorder {
        seen: Arc<AtomicUsize>,
        open: Arc<AtomicBool>,
        pp_log: PpLog,
    }

    impl GatedRecorder {
        fn link(pad: &mut SrcPad, name: &str, open: bool) -> (Arc<AtomicUsize>, Arc<AtomicBool>) {
            let seen = Arc::new(AtomicUsize::new(0));
            let gate = Arc::new(AtomicBool::new(open));
            pad.link(Box::new(GatedRecorder {
                seen: Arc::clone(&seen),
                open: Arc::clone(&gate),
                pp_log: element_pp_log(ElementType::Other, name, None),
            }));
            (seen, gate)
        }
    }

    impl Element for GatedRecorder {
        fn name(&self) -> Arc<str> {
            "gated".into()
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

    impl crate::element::Sink for GatedRecorder {
        fn ready_consume(&mut self) -> bool {
            self.open.load(Ordering::SeqCst)
        }
        fn consume(&mut self, _buf: MediaBuffer) -> crate::error::Result<()> {
            self.seen.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    /// A packet for `index` due at `ms`, in milliseconds.
    fn packet_at(index: usize, ms: i64) -> (usize, ffmpeg::Rational, ffmpeg::Packet) {
        let mut packet = ffmpeg::Packet::empty();
        packet.set_pts(Some(ms));
        packet.set_dts(Some(ms));
        (index, ffmpeg::Rational::new(1, 1000), packet)
    }

    /// With nothing else to take the read cursor, a blocked pad is waited
    /// on: its packet is held for it and reading stops, rather than
    /// buffering behind it — or pushing into a pad that cannot take it,
    /// where nothing could come back for another pad that drains meanwhile.
    /// A preroll holds its packet back too. And once the preroll is over,
    /// what it held waits for its pad, in order — reading stops meanwhile,
    /// since nothing else needs it.
    #[test]
    fn a_lone_blocked_pad_is_waited_on_outside_a_preroll() {
        let Some(path) = try_test_video() else { return };
        let (mut demuxer, streams) = FileDemuxer::open("demux", &path).expect("open");
        let index = streams.first().expect("at least one stream").index;
        let (seen, gate) = GatedRecorder::link(&mut demuxer.pads[index], "blocked", false);
        let (bus, _bus_rx) = Bus::new();

        demuxer
            .deliver_or_park(packet_at(index, 0), &bus)
            .expect("playback delivery");
        assert_eq!(demuxer.pending.len(), 1, "held for the pad");
        assert_eq!(seen.load(Ordering::SeqCst), 0, "not pushed into it");
        assert!(
            demuxer.pending_blocked(),
            "and reading waits for it, rather than buffering behind it"
        );
        gate.store(true, Ordering::SeqCst);
        demuxer.drain_pending(&bus).expect("drain once it has room");
        assert_eq!(seen.load(Ordering::SeqCst), 1);
        gate.store(false, Ordering::SeqCst);

        // As a pipeline does it: its playback state first, then the message.
        let context = Arc::new(crate::element::Context::for_test_with_clock(
            Bus::new().0,
            "test",
            crate::graph::PipelineGraph::new(),
            crate::graph::ElementId::for_test(1),
            Arc::new(crate::clock::Clock::new()),
        ));
        demuxer.attach_context(&context);
        let preroll =
            crate::control::ControlMsg::Preroll(Arc::new(crate::control::PrerollContext::new([])));
        context.state.observe(&preroll);
        demuxer.on_control(&preroll);
        demuxer
            .deliver_or_park(packet_at(index, 40), &bus)
            .expect("preroll delivery");
        assert_eq!(
            demuxer.pending.len(),
            1,
            "preroll must hold a blocked pad's packet so its siblings keep flowing"
        );

        context.state.observe(&crate::control::ControlMsg::Resume);
        demuxer.on_control(&crate::control::ControlMsg::Resume);
        demuxer.drain_pending(&bus).expect("drain while blocked");
        assert_eq!(demuxer.pending.len(), 1, "still owed to a pad that is full");
        assert_eq!(seen.load(Ordering::SeqCst), 1, "nothing pushed into it");
        assert!(demuxer.pending_blocked(), "and reading waits for it");

        gate.store(true, Ordering::SeqCst);
        demuxer.drain_pending(&bus).expect("drain once it has room");
        assert!(demuxer.pending.is_empty());
        assert_eq!(seen.load(Ordering::SeqCst), 2);
        assert!(!demuxer.pending_blocked());
    }

    /// A file whose picture is muxed ahead of its sound: the video pad's queue
    /// is full of frames the audio has not reached, and the audio packets
    /// that would let it reach them lie further on in the file. Waiting on
    /// the video there stopped this thread before them, and the video waited
    /// on the audio for good. The video packets are held back instead while
    /// the audio is taking its own — up to `MAX_INTERLEAVE` of file time,
    /// past which the video is waited on after all — and go out in order
    /// once it has room.
    #[test]
    fn a_blocked_pad_is_held_back_while_another_branch_waits_on_the_cursor() {
        let Some(path) = try_test_video() else { return };
        let (mut demuxer, streams) = FileDemuxer::open("demux", &path).expect("open");
        let find = |kind| {
            streams
                .iter()
                .find(|stream| stream.kind == kind)
                .map(|stream| stream.index)
        };
        let (Some(video), Some(audio)) = (
            find(ffmpeg::media::Type::Video),
            find(ffmpeg::media::Type::Audio),
        ) else {
            eprintln!("skipping: the fixture has no audio stream");
            return;
        };
        let (video_seen, video_gate) =
            GatedRecorder::link(&mut demuxer.pads[video], "video", false);
        let (audio_seen, _) = GatedRecorder::link(&mut demuxer.pads[audio], "audio", true);
        let (bus, _bus_rx) = Bus::new();
        let read = |demuxer: &mut FileDemuxer, index: usize, ms: i64| {
            demuxer.read_ns = Some(ms * 1_000_000);
            demuxer
                .deliver_or_park(packet_at(index, ms), &bus)
                .expect("delivery");
        };

        // The picture a second ahead: video at 1 s, then audio from 0 on.
        read(&mut demuxer, video, 1_000);
        for ms in (0..=4_000).step_by(100) {
            read(&mut demuxer, audio, ms);
        }
        assert_eq!(
            video_seen.load(Ordering::SeqCst),
            0,
            "the full pad is not pushed into"
        );
        assert_eq!(demuxer.pending.len(), 1, "its packet is held back instead");
        assert_eq!(
            audio_seen.load(Ordering::SeqCst),
            41,
            "while the sound keeps flowing"
        );
        assert!(
            !demuxer.pending_blocked(),
            "within the interleave bound, reading goes on"
        );

        // More picture than any file interleaves: past the bound, the video
        // is waited on rather than held back without end.
        read(&mut demuxer, video, 1_040);
        demuxer.read_ns = Some(7_000 * 1_000_000);
        assert!(
            demuxer.pending_blocked(),
            "past the bound, reading waits for it"
        );

        video_gate.store(true, Ordering::SeqCst);
        demuxer.drain_pending(&bus).expect("drain once it has room");
        assert!(demuxer.pending.is_empty());
        assert_eq!(video_seen.load(Ordering::SeqCst), 2, "both, in order");
        assert!(!demuxer.pending_blocked());
    }

    /// The picture muxed ahead and both branches full at once: the picture's
    /// packet is held back all the same, and reading waits while neither can
    /// take anything; once the sound drains, the cursor goes on to its
    /// packets. Pushed into the full picture instead — as when a packet was
    /// held back only while the sound had room at that moment — this thread
    /// waited there, and the picture waited on sound nothing was reading.
    #[test]
    fn a_blocked_pad_is_held_back_while_the_other_is_full_too() {
        let Some(path) = try_test_video() else { return };
        let (mut demuxer, streams) = FileDemuxer::open("demux", &path).expect("open");
        let find = |kind| {
            streams
                .iter()
                .find(|stream| stream.kind == kind)
                .map(|stream| stream.index)
        };
        let (Some(video), Some(audio)) = (
            find(ffmpeg::media::Type::Video),
            find(ffmpeg::media::Type::Audio),
        ) else {
            eprintln!("skipping: the fixture has no audio stream");
            return;
        };
        let (video_seen, _) = GatedRecorder::link(&mut demuxer.pads[video], "video", false);
        let (audio_seen, audio_gate) =
            GatedRecorder::link(&mut demuxer.pads[audio], "audio", false);
        let (bus, _bus_rx) = Bus::new();

        demuxer.read_ns = Some(1_000_000_000);
        demuxer
            .deliver_or_park(packet_at(video, 1_000), &bus)
            .expect("delivery");
        assert_eq!(video_seen.load(Ordering::SeqCst), 0, "not pushed into");
        assert_eq!(demuxer.pending.len(), 1, "held back");
        assert!(demuxer.pending_blocked(), "neither can take anything");

        audio_gate.store(true, Ordering::SeqCst);
        assert!(!demuxer.pending_blocked(), "the sound can: read on");
        demuxer
            .deliver_or_park(packet_at(audio, 0), &bus)
            .expect("delivery");
        assert_eq!(audio_seen.load(Ordering::SeqCst), 1);
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
            let stream = demuxer.stream(info.index).expect("a listed stream");
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
