//! Low-Latency HLS: a stream cut into parts of a fifth of a second as it is
//! written, kept in memory, and handed out as a player asks — the playlist
//! held until what it asks for is in, the part being made held until it is.
//!
//! [`LlHlsStream`] is what is served, for as long as it is; each
//! [`LlHlsMuxer`] it makes writes one run of it — a camera's connection —
//! which a player goes on across: the next run follows a discontinuity
//! with its own init section, and the segments go on being numbered.

mod fragmenter;
mod playlist;

use std::future::poll_fn;
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::Poll;
use std::time::Duration;

use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use self::fragmenter::Fragmenter;
pub use self::playlist::LlHlsPosition;
use self::playlist::{File, Part, State};
use super::track_sink::PendingStream;
use super::tracks::{MuxerId, MuxerSinks, MuxerTrack, MuxerTrackError};
use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, PortContract},
    control::ControlMsg,
    element::{Element, ElementType, Flow, RawSink, element_pp_log},
    elements::{HlsMuxer, HlsSegmentFormat, StreamInfo, parameter_sets::ShapeWatch},
    error::{Error, Result},
    pp_log::{PpLog, pp_error, pp_info},
    stream::StreamEvent,
};

/// How a Low-Latency HLS stream is cut and how much of it is kept.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlHlsOptions {
    /// How long a part lasts at most: what a player waits for at a time,
    /// and so about a third of how far behind the camera it plays. A fifth
    /// of a second by default.
    pub part_duration: Duration,
    /// How long a segment lasts at least: it ends at the first keyframe
    /// after this, so a camera that sends one every two seconds makes
    /// segments of two. One second by default.
    pub segment_duration: Duration,
    /// How many whole segments the playlist keeps. Seven by default.
    pub segments: usize,
}

impl Default for LlHlsOptions {
    fn default() -> Self {
        Self {
            part_duration: Duration::from_millis(200),
            segment_duration: Duration::from_secs(1),
            segments: 7,
        }
    }
}

/// What can go wrong making or writing a Low-Latency HLS stream.
#[derive(Debug, ThisError)]
#[non_exhaustive]
pub enum LlHlsMuxerError {
    /// A stream sink received a buffer other than a packet.
    #[error("LlHlsMuxer stream sinks only accept Packet buffers, got {0}")]
    UnsupportedBuffer(&'static str),
    /// A part that lasts no time, or longer than a segment.
    #[error(
        "a part must last some time, and no longer than a segment: part {part:?}, segment {segment:?}"
    )]
    InvalidDurations {
        /// [`LlHlsOptions::part_duration`].
        part: Duration,
        /// [`LlHlsOptions::segment_duration`].
        segment: Duration,
    },
    /// Fewer than two segments kept: a player joins two back.
    #[error("at least two segments must be kept, got {0}")]
    TooFewSegments(usize),
    /// A track fragmented MP4 does not carry — see [`LlHlsStream::carries`].
    #[error("track {name} is {codec}, which fragmented MP4 does not carry")]
    UnsupportedTrack {
        /// The track's name, as its `add_stream` gave it.
        name: String,
        /// Its codec.
        codec: String,
    },
    /// `open` with no stream added.
    #[error("an LlHlsMuxer needs at least one stream")]
    NoStreams,
    /// The stream is closed: nothing more is written to it.
    #[error("the Low-Latency HLS stream {0} is closed")]
    Closed(Arc<str>),
    /// FFmpeg's MP4 muxer failed.
    #[error(transparent)]
    Ffmpeg(#[from] ffmpeg::Error),
}

/// Why a playlist is not answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ThisError)]
#[non_exhaustive]
pub enum LlHlsPlaylistError {
    /// Asked to be answered from more than two segments past the one being
    /// written — HTTP's 400, as the protocol says.
    #[error("asked for a segment more than two past the one being written")]
    TooFarAhead,
    /// The stream is closed.
    #[error("the stream is closed")]
    Closed,
}

/// A Low-Latency HLS stream, as its players are served it: its media
/// playlist and the files it names, kept in memory — whole segments for
/// [`LlHlsOptions::segments`], and parts for the newest.
///
/// What is served is written by the muxers [`Self::muxer`] makes, a run each,
/// one after another. A clone is another handle to the same stream; it lives
/// until the last handle and the last muxer's run are gone. Its waits —
/// [`Self::playlist`] and [`Self::file`] — may be awaited on any async
/// runtime; what writes the stream wakes them, and [`Self::close`] ends them.
#[derive(Clone)]
pub struct LlHlsStream {
    shared: Arc<Shared>,
}

struct Shared {
    name: Arc<str>,
    state: Mutex<State>,
    pp_log: PpLog,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }
}

impl LlHlsStream {
    /// A stream with nothing written to it yet, named `name` in its log.
    ///
    /// # Errors
    ///
    /// [`LlHlsMuxerError::InvalidDurations`] for a part that lasts no time
    /// or longer than a segment, [`LlHlsMuxerError::TooFewSegments`] for
    /// fewer than two segments kept.
    pub fn new(name: impl Into<String>, options: LlHlsOptions) -> Result<Self> {
        if options.part_duration.is_zero() || options.part_duration > options.segment_duration {
            return Err(LlHlsMuxerError::InvalidDurations {
                part: options.part_duration,
                segment: options.segment_duration,
            }
            .into());
        }
        if options.segments < 2 {
            return Err(LlHlsMuxerError::TooFewSegments(options.segments).into());
        }
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::LlHlsMuxer, &name, None);
        pp_info!(
            pp_log: &pp_log,
            "created: part={:?} segment={:?} segments={}",
            options.part_duration,
            options.segment_duration,
            options.segments
        );
        Ok(Self {
            shared: Arc::new(Shared {
                name,
                state: Mutex::new(State::new(
                    options.part_duration,
                    options.segment_duration,
                    options.segments,
                )),
                pp_log,
            }),
        })
    }

    /// Whether a track goes into the stream as it is: what fragmented MP4
    /// carries — as [`HlsMuxer::carries`] says of it.
    pub fn carries(track: &StreamInfo) -> bool {
        HlsMuxer::carries(track, HlsSegmentFormat::Fmp4)
    }

    /// A muxer of a run of the stream: given its tracks with
    /// [`LlHlsMuxer::add_stream`] and opened, a sink per track. The run
    /// begins with its first keyframe — what comes before it is dropped —
    /// and ends once every track has ended, or been stopped, or its sinks
    /// dropped.
    pub fn muxer(&self) -> LlHlsMuxer {
        LlHlsMuxer {
            stream: self.clone(),
            id: MuxerId::next(),
            tracks: Vec::new(),
        }
    }

    /// The media playlist — at once, or where `at` is asked for, once what
    /// it names is in: the part or the whole segment a player's
    /// `_HLS_msn` and `_HLS_part` say. Until a first segment is whole, it
    /// waits for it: what a player is first told the target duration is
    /// must be what the segments are. A caller bounds the wait.
    ///
    /// # Errors
    ///
    /// [`LlHlsPlaylistError::TooFarAhead`] at once for `at` more than two
    /// segments past the one being written, and
    /// [`LlHlsPlaylistError::Closed`] once the stream is closed.
    pub async fn playlist(
        &self,
        at: Option<LlHlsPosition>,
    ) -> std::result::Result<String, LlHlsPlaylistError> {
        poll_fn(|cx| {
            let mut state = self.shared.lock();
            if state.closed {
                return Poll::Ready(Err(LlHlsPlaylistError::Closed));
            }
            let ready = match at {
                Some(at) if state.too_far(at) => {
                    return Poll::Ready(Err(LlHlsPlaylistError::TooFarAhead));
                }
                Some(at) => state.reached(at),
                None => state.has_media(),
            };
            if ready {
                return Poll::Ready(Ok(state.playlist()));
            }
            state.wait(cx.waker());
            Poll::Pending
        })
        .await
    }

    /// A file the playlist names — an init section, a part or a whole
    /// segment — by the name it gives it: at once where it is held, and the
    /// part a player is told is coming once it is in. `None` for a name
    /// that is none of its, one let go of, or once the stream is closed.
    pub async fn file(&self, name: &str) -> Option<Arc<[u8]>> {
        let file = File::parse(name)?;
        poll_fn(|cx| {
            let mut state = self.shared.lock();
            if let Some(data) = state.file(&file) {
                return Poll::Ready(Some(data));
            }
            if state.closed || !state.may_come(&file) {
                return Poll::Ready(None);
            }
            state.wait(cx.waker());
            Poll::Pending
        })
        .await
    }

    /// Closes it: whatever waits is answered that it is closed, and its
    /// muxers write nothing more.
    pub fn close(&self) {
        let mut state = self.shared.lock();
        if !state.closed {
            state.closed = true;
            state.changed();
            drop(state);
            pp_info!(pp_log: &self.shared.pp_log, "closed");
        }
    }

    /// Its name, as its log records give it.
    pub fn name(&self) -> Arc<str> {
        Arc::clone(&self.shared.name)
    }
}

impl std::fmt::Debug for LlHlsStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlHlsStream")
            .field("name", &self.shared.name)
            .finish()
    }
}

/// A muxer of one run of an [`LlHlsStream`] — see [`LlHlsStream::muxer`].
///
/// Has the shape of the other muxers here: every track added with
/// [`Self::add_stream`] before [`Self::open`], whose sinks are taken out
/// with the [`MuxerTrack`]s it gave. Its sinks share one lock; the parts are
/// cut on the first video track's timeline — the first track's where there
/// is none — and a segment ends at its first keyframe past
/// [`LlHlsOptions::segment_duration`]. Each track holds its newest packet
/// back until the next says how long it lasts, so a part ends a picture
/// behind the camera.
pub struct LlHlsMuxer {
    stream: LlHlsStream,
    id: MuxerId,
    tracks: Vec<(PendingStream, StreamInfo)>,
}

impl LlHlsMuxer {
    /// Registers one encoded packet stream, described by `format` (see
    /// [`StreamInfo`]).
    ///
    /// # Errors
    ///
    /// [`LlHlsMuxerError::UnsupportedTrack`] for a track fragmented MP4 does
    /// not carry.
    pub fn add_stream(
        &mut self,
        name: impl Into<String>,
        format: impl Into<StreamInfo>,
    ) -> Result<MuxerTrack> {
        let name = name.into();
        let format = format.into();
        if !LlHlsStream::carries(&format) {
            return Err(LlHlsMuxerError::UnsupportedTrack {
                name,
                codec: format!("{:?}", format.parameters().id()),
            }
            .into());
        }
        let pending = PendingStream::new(name, format.parameters(), format.time_base());
        let track = self.id.track(self.tracks.len());
        self.tracks.push((pending, format));
        Ok(track)
    }

    /// Opens the run: a sink per track, each taken out by the
    /// [`MuxerTrack`] its [`Self::add_stream`] returned.
    ///
    /// # Errors
    ///
    /// [`LlHlsMuxerError::NoStreams`] with no track added,
    /// [`LlHlsMuxerError::Closed`] once the stream is closed, and
    /// [`LlHlsMuxerError::Ffmpeg`] where FFmpeg's MP4 muxer will not take
    /// the tracks.
    pub fn open(self) -> Result<MuxerSinks> {
        crate::ensure_ffmpeg();
        if self.tracks.is_empty() {
            return Err(LlHlsMuxerError::NoStreams.into());
        }
        let shared = Arc::clone(&self.stream.shared);
        if shared.lock().closed {
            return Err(LlHlsMuxerError::Closed(Arc::clone(&shared.name)).into());
        }
        let infos: Vec<StreamInfo> = self.tracks.iter().map(|(_, info)| info.clone()).collect();
        let fragmenter = Fragmenter::new(&infos).map_err(LlHlsMuxerError::from)?;
        let reference = infos
            .iter()
            .position(|info| info.kind() == ffmpeg::media::Type::Video)
            .unwrap_or(0);
        let total = infos.len();
        let run = Arc::new(Mutex::new(Run {
            shared,
            fragmenter,
            offset: 0.0,
            shifts: vec![None; total],
            reference,
            run: None,
            segment: None,
            started: false,
            part_start: 0.0,
            segment_start: 0.0,
            last_at: 0.0,
            step: 0.0,
            independent: true,
            held: (0..total).map(|_| None).collect(),
            steps: vec![0; total],
            remaining: total,
            ended: false,
        }));
        let sinks = self
            .tracks
            .into_iter()
            .enumerate()
            .map(|(index, (pending, _))| -> Box<dyn RawSink> {
                Box::new(LlHlsTrackSink {
                    pp_log: element_pp_log(ElementType::LlHlsMuxer, &pending.name, None),
                    name: pending.name,
                    index,
                    kind: pending.kind,
                    watch: pending.watch,
                    run: Arc::clone(&run),
                    refused: false,
                    done: false,
                })
            })
            .collect();
        Ok(self.id.sinks(sinks))
    }
}

/// One run being written: what its tracks' sinks share.
struct Run {
    shared: Arc<Shared>,
    fragmenter: Fragmenter,
    /// What its timestamps are moved by, in seconds, to go on where the
    /// last run ended.
    offset: f64,
    /// That, in each track's unit in the muxer, once worked out.
    shifts: Vec<Option<i64>>,
    /// The track parts are cut on.
    reference: usize,
    /// Its number in the stream, once its first part is in.
    run: Option<u64>,
    /// The segment being written, once one is.
    segment: Option<u64>,
    /// Set at the reference track's first keyframe.
    started: bool,
    /// Where the part and the segment being written began, and where the
    /// reference track's newest packet is, in seconds on its timeline.
    part_start: f64,
    segment_start: f64,
    last_at: f64,
    /// The reference track's last step, in seconds.
    step: f64,
    /// Whether the part being written begins at a keyframe.
    independent: bool,
    /// Each track's newest packet, in the muxer's unit, until the next says
    /// how long it lasts.
    held: Vec<Option<ffmpeg::Packet>>,
    /// Each track's last step, in the muxer's unit, for its last packet.
    steps: Vec<i64>,
    /// How many tracks have not ended.
    remaining: usize,
    ended: bool,
}

impl Run {
    /// Takes `packet` of track `index`: writes the one before it, now that
    /// it says how long that one lasted, and cuts a part or a segment
    /// before it where it is time.
    fn take(
        &mut self,
        index: usize,
        packet: &ffmpeg::Packet,
    ) -> std::result::Result<(), LlHlsMuxerError> {
        if self.ended {
            return Ok(());
        }
        let unit = self.fragmenter.unit_in(index, packet);
        let Some(ts) = packet.dts().or(packet.pts()) else {
            return Ok(());
        };
        let at = ts as f64 * f64::from(unit);
        let reference = index == self.reference;
        if !self.started {
            if !reference || !packet.is_key() {
                return Ok(());
            }
            self.started = true;
            self.part_start = at;
            self.segment_start = at;
            self.last_at = at;
            self.independent = true;
            // On from where the last run ended, a picture's step after it:
            // what a player sees of a camera connected again, where its
            // init section is the same — see `State::begin_run`. The time
            // it was gone is not put between them: nothing in the playlist
            // could say where a player's timeline had a hole.
            if let Some(end) = self.shared.lock().timeline_end {
                self.offset = end - at;
            }
        }
        let out = self.fragmenter.unit_out(index);
        let mut packet = crate::buffer::shared_packet(packet);
        packet.rescale_ts(unit, out);
        let offset = self.offset;
        let shift =
            *self.shifts[index].get_or_insert_with(|| (offset / f64::from(out)).round() as i64);
        if shift != 0 {
            packet.set_pts(packet.pts().map(|pts| pts + shift));
            packet.set_dts(packet.dts().map(|dts| dts + shift));
        }
        self.write_held(index, packet.dts().or(packet.pts()))?;
        if reference {
            if at > self.last_at {
                self.step = at - self.last_at;
            }
            let (part_target, segment_target) = {
                let state = self.shared.lock();
                (
                    state.part_target.as_secs_f64(),
                    state.segment_target.as_secs_f64(),
                )
            };
            const SLACK: f64 = 0.001;
            let key = packet.is_key();
            if key && at > self.segment_start && at - self.segment_start + SLACK >= segment_target {
                self.cut(at)?;
                // The next segment is opened with its first part, not here:
                // a player told that part is coming takes its segment as on
                // this run's timeline, and one the run ended before would
                // have been the next run's, behind a discontinuity the player
                // had not counted.
                if let Some(msn) = self.segment.take() {
                    self.shared.lock().close_segment(msn);
                }
                self.segment_start = at;
                self.independent = true;
            } else if at > self.part_start && at - self.part_start + self.step > part_target + SLACK
            {
                self.cut(at)?;
                self.independent = key;
            }
            self.last_at = at;
        }
        self.held[index] = Some(packet);
        Ok(())
    }

    /// Writes track `index`'s held packet, its duration up to `next` where
    /// that is after it, its track's last step else.
    fn write_held(
        &mut self,
        index: usize,
        next: Option<i64>,
    ) -> std::result::Result<(), LlHlsMuxerError> {
        let Some(mut held) = self.held[index].take() else {
            return Ok(());
        };
        if let (Some(next), Some(at)) = (next, held.dts().or(held.pts()))
            && next > at
        {
            self.steps[index] = next - at;
        }
        if held.duration() <= 0 {
            held.set_duration(self.steps[index]);
        }
        self.fragmenter.write(index, held)?;
        Ok(())
    }

    /// Cuts what was written since the last part into one, ending at `at`.
    fn cut(&mut self, at: f64) -> std::result::Result<(), LlHlsMuxerError> {
        let data = self.fragmenter.cut()?;
        if data.is_empty() {
            return Ok(());
        }
        let mut state = self.shared.lock();
        if state.closed {
            self.ended = true;
            return Ok(());
        }
        let run = match self.run {
            Some(run) => run,
            None => {
                let init = self.fragmenter.take_init().unwrap_or_default();
                let run = state.begin_run(init.into());
                self.run = Some(run);
                pp_info!(pp_log: &self.shared.pp_log, "run {run} began");
                run
            }
        };
        let msn = *self.segment.get_or_insert_with(|| state.open_segment(run));
        state.add_part(
            msn,
            Part {
                data: data.into(),
                duration: (at - self.part_start).max(0.0),
                independent: self.independent,
            },
        );
        drop(state);
        self.part_start = at;
        Ok(())
    }

    /// Ends the run: what is held written and cut, the segment closed.
    fn end(&mut self) {
        if self.ended {
            return;
        }
        if self.started {
            let end = self.last_at + self.step;
            let result = (0..self.held.len())
                .try_for_each(|index| self.write_held(index, None))
                .and_then(|()| self.cut(end));
            if let Err(error) = result {
                pp_error!(pp_log: &self.shared.pp_log, "the run's last part was not written: {error}");
            }
        }
        self.ended = true;
        let mut state = self.shared.lock();
        if let Some(msn) = self.segment.take() {
            state.close_segment(msn);
        }
        if self.started {
            state.timeline_end = Some(self.last_at + self.step + self.offset);
        }
        if let Some(run) = self.run {
            state.end_run(run);
            drop(state);
            pp_info!(pp_log: &self.shared.pp_log, "run {run} ended");
        }
    }
}

impl Drop for Run {
    fn drop(&mut self) {
        self.end();
    }
}

/// One track's sink of a run.
struct LlHlsTrackSink {
    pp_log: PpLog,
    name: Arc<str>,
    index: usize,
    kind: Option<MediaKind>,
    watch: Option<ShapeWatch>,
    run: Arc<Mutex<Run>>,
    /// Set once its pictures changed: what follows is not written.
    refused: bool,
    done: bool,
}

impl LlHlsTrackSink {
    fn run(&self) -> MutexGuard<'_, Run> {
        self.run.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn finish(&mut self) {
        if std::mem::replace(&mut self.done, true) {
            return;
        }
        let mut run = self.run();
        run.remaining = run.remaining.saturating_sub(1);
        if run.remaining == 0 {
            run.end();
        }
    }
}

impl Element for LlHlsTrackSink {
    fn name(&self) -> Arc<str> {
        Arc::clone(&self.name)
    }

    fn element_type(&self) -> ElementType {
        ElementType::LlHlsMuxer
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl RawSink for LlHlsTrackSink {
    fn input_contract(&self) -> InputContract {
        match self.kind {
            Some(kind) => InputContract::Fixed(PortContract::packet(kind)),
            None => InputContract::Unknown,
        }
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        let MediaBuffer::Packet(packet) = buf else {
            pp_error!(self, "unsupported buffer: {}", buf.kind());
            return Err(LlHlsMuxerError::UnsupportedBuffer(buf.kind()).into());
        };
        if self.refused {
            return Ok(());
        }
        // The init section says what the pictures are; pictures that changed
        // partway are refused, and a new run describes them.
        if packet.is_key()
            && let (Some(watch), Some(data)) = (&mut self.watch, packet.data())
            && let Some((was, now)) = watch.changed(data)
        {
            self.refused = true;
            pp_error!(
                self,
                "the pictures changed from {was} to {now}; nothing more is written"
            );
            return Err(MuxerTrackError::FormatChanged {
                track: self.name.to_string(),
                was: was.to_string(),
                now: now.to_string(),
            }
            .into());
        }
        let index = self.index;
        self.run()
            .take(index, &packet)
            .inspect_err(|error| pp_error!(self, "writing failed: {error}"))
            .map_err(Error::from)
    }

    fn stream_event(&mut self, event: &StreamEvent) -> Result<()> {
        if matches!(event, StreamEvent::Eos) {
            self.finish();
        }
        Ok(())
    }

    /// What it writes is the stream as it ran; a seek would put another
    /// timeline in the middle of a run.
    fn accepts_seek(&self) -> bool {
        false
    }

    fn flow(&mut self, Flow(msg): Flow<'_>) -> Result<()> {
        if let ControlMsg::Stop = msg {
            self.finish();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
