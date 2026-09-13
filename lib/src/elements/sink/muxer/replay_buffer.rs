//! The last stretch of what was encoded, kept in memory and written to a
//! file only when asked — what OBS calls a replay buffer.
//!
//! # A muxer that writes nothing until asked
//!
//! Its tracks are declared and taken the way every muxer's are —
//! `add_stream`, `open`, [`MuxerSinks::take`] — so it goes wherever a
//! [`FileMuxer`] would. What differs is what a packet does on arrival: it is
//! kept rather than written, and whatever has fallen out of the window is let
//! go. [`ReplayBufferHandle::save`] then writes what is held as a file of its
//! own, and can be asked again for the next one.
//!
//! # Where a clip can start
//!
//! On a keyframe of the track the window is measured on — the first video
//! track, or the first track of any kind when there is no video — since a
//! file that opens mid-GOP shows nothing until the next one. So the window is
//! let go of a GOP at a time: it holds at most its length, and at least that
//! less one keyframe interval. A two-second interval under a thirty-second
//! window keeps between 28 and 30 seconds.
//!
//! A window shorter than one keyframe interval still holds one GOP, which is
//! the least a clip can be. The window only moves on when a keyframe comes,
//! so an encoder that stopped sending them would have it grow without bound —
//! every encoder in this crate sends them at its `gop_size`.
//!
//! # One timeline for every track
//!
//! The other tracks are kept from the moment the picture starts: anything of
//! theirs from before the first keyframe held is let go with it. A saved clip
//! is moved to start at zero by one origin — that keyframe, converted into
//! each track's own time base — the same rule [`SegmentedFileMuxer`] rebases
//! segments by, and for its reason: zeroing each track on its own first
//! packet would pull them apart by however far they were interleaved.
//!
//! [`SegmentedFileMuxer`]: super::SegmentedFileMuxer

use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use ffmpeg_next::{self as ffmpeg, Rescale};
use thiserror::Error as ThisError;

use super::file_muxer::FileMuxer;
use super::tracks::{MuxerId, MuxerSinks, MuxerTrack};
use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, PortContract},
    control::{ControlMsg, SeekRejectReason},
    element::{Element, ElementType, Sink, element_pp_log},
    error::Result,
    pp_log::{PpLog, pp_error, pp_info},
};

/// Errors specific to [`ReplayBuffer`]. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum ReplayBufferError {
    /// [`ReplayBuffer::open`] was given a window of no length at all.
    #[error("a replay buffer's length must be more than zero")]
    ZeroLength,

    /// [`ReplayBuffer::open`] was called before any track was added.
    #[error("a replay buffer needs at least one track")]
    NoTracks,

    /// [`ReplayBufferHandle::save`] was asked for a clip before the first
    /// keyframe had arrived — there is nothing a file could start on.
    #[error("nothing is buffered yet: no keyframe has arrived")]
    Empty,

    /// [`ReplayBufferHandle::save`] was called after every track's sink was
    /// dropped, which lets go of what they held.
    #[error("the replay buffer has stopped: its tracks are gone")]
    Stopped,

    /// A track's sink received a buffer other than a packet or
    /// end-of-stream.
    #[error("replay buffer tracks only accept Packet or Eos buffers, got {0}")]
    UnsupportedBuffer(&'static str),
}

/// Keeps the last `length` of what its tracks are handed, and writes it to
/// a file when [`ReplayBufferHandle::save`] asks — see this module's docs.
///
/// Built the two-phase way every muxer is, because a saved clip has to
/// describe each track's codec in its header: `add_stream` for each track,
/// `open` for one sink per track and the handle that saves.
///
/// ```no_run
/// # use std::time::Duration;
/// # use media_pp::ffmpeg;
/// # use media_pp::elements::{
/// #     AudioCodec, ReplayBuffer, SwAudioEncoder, SwAudioEncoderOptions, SwEncoder,
/// #     SwEncoderOptions, VideoCodec,
/// # };
/// # fn main() -> media_pp::Result<()> {
/// # let video_time_base = ffmpeg::Rational(1, 30);
/// # let audio_time_base = ffmpeg::Rational(1, 48_000);
/// # let video_encoder = SwEncoder::new("video", SwEncoderOptions {
/// #     codec: VideoCodec::H264,
/// #     width: 640,
/// #     height: 360,
/// #     time_base: video_time_base,
/// #     frame_rate: ffmpeg::Rational(30, 1),
/// #     bit_rate: 2_000_000,
/// #     gop_size: 60,
/// #     max_b_frames: None,
/// # })?;
/// # let audio_encoder = SwAudioEncoder::new("audio", SwAudioEncoderOptions {
/// #     codec: AudioCodec::Aac,
/// #     sample_rate: 48_000,
/// #     channels: 2,
/// #     time_base: audio_time_base,
/// #     bit_rate: 128_000,
/// # })?;
/// let mut replay = ReplayBuffer::create(Duration::from_secs(30));
/// let video = replay.add_stream("video", video_encoder.parameters(), video_time_base);
/// let audio = replay.add_stream("audio", audio_encoder.parameters(), audio_time_base);
/// let (mut sinks, handle) = replay.open()?;
/// let video_sink = sinks.take(video)?;
/// let audio_sink = sinks.take(audio)?;
/// // ...wire both sinks behind their encoders and run; then, when asked:
/// let length = handle.save("replay.mp4")?;
/// # Ok(())
/// # }
/// ```
pub struct ReplayBuffer {
    id: MuxerId,
    length: Duration,
    streams: Vec<StreamDef>,
}

/// One track as it was added: what a saved clip's header declares for it.
struct StreamDef {
    name: Arc<str>,
    parameters: ffmpeg::codec::Parameters,
    time_base: ffmpeg::Rational,
}

impl ReplayBuffer {
    /// A buffer that keeps `length` of what it is handed — validated by
    /// [`ReplayBuffer::open`], which is where a zero is refused.
    pub fn create(length: Duration) -> Self {
        Self {
            id: MuxerId::next(),
            length,
            streams: Vec::new(),
        }
    }

    /// Registers one more track, on the terms [`FileMuxer::add_stream`]
    /// takes one: `time_base` is what its packets are stamped in, and
    /// `name` is its sink's identity in logs and bus events.
    ///
    /// Cannot fail — nothing is opened until a clip is saved, and each save
    /// declares every track to a file of its own.
    pub fn add_stream(
        &mut self,
        name: impl Into<String>,
        parameters: ffmpeg::codec::Parameters,
        time_base: ffmpeg::Rational,
    ) -> MuxerTrack {
        let track = self.id.track(self.streams.len());
        self.streams.push(StreamDef {
            name: name.into().into(),
            parameters,
            time_base,
        });
        track
    }

    /// One sink per track, each taken out by the [`MuxerTrack`] its
    /// [`ReplayBuffer::add_stream`] returned, and the handle that saves what
    /// they hold.
    ///
    /// The window is measured on the first track whose parameters are
    /// video, or on the first track when none is.
    pub fn open(self) -> Result<(MuxerSinks, ReplayBufferHandle)> {
        if self.length.is_zero() {
            return Err(ReplayBufferError::ZeroLength.into());
        }
        if self.streams.is_empty() {
            return Err(ReplayBufferError::NoTracks.into());
        }
        let anchor = self
            .streams
            .iter()
            .position(|stream| stream.parameters.medium() == ffmpeg::media::Type::Video)
            .unwrap_or(0);
        let tracks: Vec<(Arc<str>, Option<MediaKind>)> = self
            .streams
            .iter()
            .map(|stream| {
                (
                    stream.name.clone(),
                    MediaKind::packet_for(stream.parameters.medium()),
                )
            })
            .collect();
        let shared = Arc::new(Shared {
            length: i64::try_from(self.length.as_micros()).unwrap_or(i64::MAX),
            pp_log: element_pp_log(ElementType::ReplayBuffer, &self.streams[anchor].name, None),
            window: Mutex::new(Window {
                held: self.streams.iter().map(|_| VecDeque::new()).collect(),
                streams: self.streams,
                anchor,
                keyframes: 0,
            }),
        });
        let handle = ReplayBufferHandle {
            shared: Arc::downgrade(&shared),
        };
        let sinks = self.id.sinks(
            tracks
                .into_iter()
                .enumerate()
                .map(|(track, (name, kind))| -> Box<dyn Sink> {
                    Box::new(ReplayTrackSink {
                        pp_log: element_pp_log(ElementType::ReplayBuffer, &name, None),
                        name,
                        track,
                        kind,
                        shared: Arc::clone(&shared),
                    })
                })
                .collect(),
        );
        Ok((sinks, handle))
    }
}

/// What every track's sink writes into and the handle reads from.
struct Shared {
    /// The window's length, in microseconds — the unit every held packet's
    /// time is kept in, so tracks in different time bases compare directly.
    length: i64,
    /// The measured track's identity, for what a save logs.
    pp_log: PpLog,
    /// One lock over every track, because trimming one is decided by
    /// another: the others are let go of where the measured one starts.
    window: Mutex<Window>,
}

/// What is held, track by track.
///
/// Also where the track descriptions live: [`ffmpeg::codec::Parameters`] is
/// `Send` but not `Sync`, so it has to be inside the `Mutex` for the `Arc`
/// the sinks share to cross threads — see `SegmentedFileMuxer`'s
/// `GroupState`, which is arranged the same way for the same reason.
struct Window {
    streams: Vec<StreamDef>,
    /// Each track's packets in the order they arrived, which for one track
    /// is decode order.
    held: Vec<VecDeque<Held>>,
    /// The track the window is measured on and a clip starts on.
    anchor: usize,
    /// How many keyframes the measured track holds. The front of it is
    /// always one, and a GOP is let go only while another remains behind it
    /// — so a clip always has somewhere to start.
    keyframes: usize,
}

/// One packet, and when it is, in microseconds on its track's timeline.
struct Held {
    packet: Arc<ffmpeg::Packet>,
    at: i64,
}

impl Window {
    /// Takes one packet, and lets go of whatever that moves out of the
    /// window.
    fn push(&mut self, track: usize, packet: Arc<ffmpeg::Packet>, length: i64) {
        // Decode order is what a track arrives in, and `dts` is what counts
        // it; a packet with neither timestamp has nowhere on the timeline to
        // be kept, and no encoder here sends one.
        let Some(stamp) = packet.dts().or(packet.pts()) else {
            return;
        };
        let at = stamp.rescale(self.streams[track].time_base, MICROSECONDS);
        if track != self.anchor {
            self.held[track].push_back(Held { packet, at });
            self.trim_follower(track, length);
            return;
        }
        // A timeline that went backwards is a new one — a restarted source,
        // an upstream reset. What is held belongs to the old one, and
        // measuring the window across the jump would never let it go.
        if self.held[track].back().is_some_and(|last| at < last.at) {
            self.clear();
        }
        // Nothing can start before a keyframe, so what comes ahead of the
        // first one is not worth holding.
        if self.held[track].is_empty() && !packet.is_key() {
            return;
        }
        if packet.is_key() {
            self.keyframes += 1;
        }
        self.held[track].push_back(Held { packet, at });
        self.trim(length);
    }

    /// Lets go of the measured track a GOP at a time while it is longer
    /// than the window, then of everything the others hold from before it
    /// now starts.
    fn trim(&mut self, length: i64) {
        let anchor = &mut self.held[self.anchor];
        let Some(newest) = anchor.back().map(|held| held.at) else {
            return;
        };
        while self.keyframes > 1 && anchor.front().is_some_and(|held| newest - held.at > length) {
            // The front is a keyframe — see `keyframes` — and the GOP it
            // opens runs up to the next one.
            anchor.pop_front();
            self.keyframes -= 1;
            while anchor.front().is_some_and(|held| !held.packet.is_key()) {
                anchor.pop_front();
            }
        }
        let Some(start) = anchor.front().map(|held| held.at) else {
            return;
        };
        for (track, held) in self.held.iter_mut().enumerate() {
            if track != self.anchor {
                while held.front().is_some_and(|packet| packet.at < start) {
                    held.pop_front();
                }
            }
        }
    }

    /// Lets go of what one of the other tracks holds from before the
    /// picture starts — or, while there is no picture yet, from before its
    /// own window.
    fn trim_follower(&mut self, track: usize, length: i64) {
        let start = match self.held[self.anchor].front() {
            Some(front) => front.at,
            None => match self.held[track].back() {
                Some(newest) => newest.at.saturating_sub(length),
                None => return,
            },
        };
        let held = &mut self.held[track];
        while held.front().is_some_and(|packet| packet.at < start) {
            held.pop_front();
        }
    }

    fn clear(&mut self) {
        for held in &mut self.held {
            held.clear();
        }
        self.keyframes = 0;
    }

    /// How much the measured track holds, from its first packet to its last.
    fn span(&self) -> Duration {
        let anchor = &self.held[self.anchor];
        match (anchor.front(), anchor.back()) {
            (Some(first), Some(last)) => micros(last.at - first.at),
            _ => Duration::ZERO,
        }
    }

    /// What a save writes: every track's description and every packet held
    /// from the first keyframe on, as references rather than copies — this
    /// runs under the lock the tracks write through.
    fn clip(&self) -> std::result::Result<Clip, ReplayBufferError> {
        let anchor = &self.held[self.anchor];
        let (Some(first), Some(last)) = (anchor.front(), anchor.back()) else {
            return Err(ReplayBufferError::Empty);
        };
        let origin_stamp = first
            .packet
            .dts()
            .or(first.packet.pts())
            .expect("only a stamped packet is ever held");
        let anchor_base = self.streams[self.anchor].time_base;
        let mut packets: Vec<(i64, usize, Arc<ffmpeg::Packet>)> = self
            .held
            .iter()
            .enumerate()
            .flat_map(|(track, held)| {
                held.iter()
                    .filter(move |packet| packet.at >= first.at)
                    .map(move |packet| (packet.at, track, Arc::clone(&packet.packet)))
            })
            .collect();
        // Interleaved by time, as a file is read back. Stable, so a track's
        // own packets keep the order they arrived in where two share a time.
        packets.sort_by_key(|(at, _, _)| *at);
        let last_duration = last.packet.duration().rescale(anchor_base, MICROSECONDS);
        Ok(Clip {
            streams: self
                .streams
                .iter()
                .map(|stream| StreamDef {
                    name: stream.name.clone(),
                    parameters: stream.parameters.clone(),
                    time_base: stream.time_base,
                })
                .collect(),
            origins: self
                .streams
                .iter()
                .map(|stream| origin_stamp.rescale(anchor_base, stream.time_base))
                .collect(),
            packets,
            duration: micros(last.at - first.at + last_duration.max(0)),
        })
    }
}

/// One save's worth of what was held, taken out from under the lock.
struct Clip {
    streams: Vec<StreamDef>,
    /// Where the clip starts, in each track's own time base.
    origins: Vec<i64>,
    packets: Vec<(i64, usize, Arc<ffmpeg::Packet>)>,
    duration: Duration,
}

impl Clip {
    /// Writes the clip to `path`, starting at zero — see this module's docs
    /// on why every track is moved by the one origin.
    fn write(self, path: &Path) -> Result<()> {
        let mut muxer = FileMuxer::create(path)?;
        let tracks = self
            .streams
            .iter()
            .map(|stream| {
                muxer.add_stream(
                    stream.name.to_string(),
                    stream.parameters.clone(),
                    stream.time_base,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let mut opened = muxer.open()?;
        let mut sinks = tracks
            .into_iter()
            .map(|track| opened.take(track))
            .collect::<Result<Vec<_>>>()?;
        for (_, track, packet) in self.packets {
            let origin = self.origins[track];
            // Clamped rather than left negative, which no container takes:
            // a track interleaved just ahead of the keyframe can carry a
            // packet a fraction older than it.
            let moved = |value: i64| (value - origin).max(0);
            let mut rebased = (*packet).clone();
            rebased.set_pts(packet.pts().map(moved));
            rebased.set_dts(packet.dts().map(moved));
            sinks[track].consume(MediaBuffer::Packet(Arc::new(rebased)))?;
        }
        // Every track, whatever the first one says: the trailer is written
        // once the last of them reports done.
        let mut finished = Ok(());
        for sink in &mut sinks {
            let done = sink.consume(MediaBuffer::Eos);
            if finished.is_ok() {
                finished = done;
            }
        }
        finished
    }
}

/// Saves what a [`ReplayBuffer`] holds.
///
/// Cheap to clone, and holds the buffer only weakly: what is buffered lives
/// as long as its tracks' sinks, and a handle kept after those are dropped
/// answers [`ReplayBufferError::Stopped`] rather than keeping the window's
/// memory alive.
#[derive(Clone)]
pub struct ReplayBufferHandle {
    shared: Weak<Shared>,
}

impl ReplayBufferHandle {
    /// Writes what is held now to `path` as a file of its own, starting on
    /// its first keyframe and at zero, and answers how long it is.
    ///
    /// The container is the path's, as it is for [`FileMuxer`] — which is
    /// what writes it — and so is the rule that a file already at `path` is
    /// replaced. A save that fails part-way removes what it wrote.
    ///
    /// Blocks for the whole write — every packet of the clip onto the disk —
    /// so it belongs off any thread that has something else to keep up
    /// with. The tracks are held up only while the clip's packets are
    /// gathered, which copies references rather than the packets; they go
    /// on filling the window while the file is written, and the next save
    /// takes the window as it then stands.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<Duration> {
        let path = path.as_ref();
        let shared = self.shared.upgrade().ok_or(ReplayBufferError::Stopped)?;
        let clip = shared.window.lock().unwrap().clip()?;
        let (duration, packets) = (clip.duration, clip.packets.len());
        match clip.write(path) {
            Ok(()) => {
                pp_info!(
                    pp_log: &shared.pp_log,
                    "event=save outcome=ok path={} duration_ms={} packets={packets}",
                    path.display(),
                    duration.as_millis()
                );
                Ok(duration)
            }
            Err(error) => {
                pp_error!(
                    pp_log: &shared.pp_log,
                    "event=save outcome=error path={} error={error}",
                    path.display()
                );
                let _ = std::fs::remove_file(path);
                Err(error)
            }
        }
    }

    /// How much is held now, measured on the track a clip starts on — what
    /// a save would write, give or take the last packet's duration. Zero
    /// before the first keyframe and once the buffer has stopped.
    pub fn buffered(&self) -> Duration {
        self.shared.upgrade().map_or(Duration::ZERO, |shared| {
            shared.window.lock().unwrap().span()
        })
    }
}

/// One track's own sink: a handle on the window every track shares.
struct ReplayTrackSink {
    pp_log: PpLog,
    name: Arc<str>,
    track: usize,
    /// The medium this track was registered for; `None` for one this crate
    /// does not model, which then declares nothing.
    kind: Option<MediaKind>,
    shared: Arc<Shared>,
}

impl Element for ReplayTrackSink {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::ReplayBuffer
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Sink for ReplayTrackSink {
    /// Encoded packets of this track's own medium, as a muxer's track takes.
    fn input_contract(&self) -> InputContract {
        match self.kind {
            Some(kind) => InputContract::Fixed(PortContract::packet(kind)),
            None => InputContract::Unknown,
        }
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Packet(packet) => {
                self.shared
                    .window
                    .lock()
                    .unwrap()
                    .push(self.track, packet, self.shared.length);
                Ok(())
            }
            // What is held stays saveable after its stream has ended: the
            // last stretch of something that finished is still the last
            // stretch of it.
            MediaBuffer::Eos => Ok(()),
            other => {
                pp_error!(self, "unsupported buffer: {}", other.kind());
                Err(ReplayBufferError::UnsupportedBuffer(other.kind()).into())
            }
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        // Terminal, so nothing is forwarded.
        match msg {
            // A window measured across a jump in the timeline would hold
            // the wrong stretch and let go of it at the wrong time.
            ControlMsg::CheckSeek(context) => context.reject(
                self.element_type(),
                self.name(),
                SeekRejectReason::ElementNotSeekable,
            ),
            // What is held belongs to the timeline being discarded.
            ControlMsg::Flush => self.shared.window.lock().unwrap().clear(),
            // Nothing to finalize and nothing to pause: what is held stays
            // as it is until the sinks are dropped.
            ControlMsg::Stop
            | ControlMsg::Pause
            | ControlMsg::Resume
            | ControlMsg::Preroll(_)
            | ControlMsg::Seek(_) => {}
        }
        Ok(())
    }
}

const MICROSECONDS: ffmpeg::Rational = ffmpeg::Rational(1, 1_000_000);

/// A span in microseconds as a `Duration`, a negative one as none.
fn micros(value: i64) -> Duration {
    Duration::from_micros(u64::try_from(value).unwrap_or(0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;

    /// Every packet of a file, with the tracks it declares — the file stands
    /// in for two encoders, and its packets are real ones: keyframes where
    /// an encoder put them, stamped in the file's own time bases.
    struct Recorded {
        streams: Vec<(ffmpeg::codec::Parameters, ffmpeg::Rational)>,
        packets: Vec<(usize, ffmpeg::Packet)>,
    }

    fn read(path: &std::path::Path) -> Recorded {
        let mut input = ffmpeg::format::input(&path).expect("the fixture opens");
        let streams = input
            .streams()
            .map(|stream| (stream.parameters(), stream.time_base()))
            .collect();
        let packets = input
            .packets()
            .map(|(stream, packet)| (stream.index(), packet))
            .collect();
        Recorded { streams, packets }
    }

    /// A buffer over `recorded`'s tracks, with every track's sink.
    fn buffer_over(
        recorded: &Recorded,
        length: Duration,
    ) -> (Vec<Box<dyn Sink>>, ReplayBufferHandle) {
        let mut replay = ReplayBuffer::create(length);
        let tracks: Vec<_> = recorded
            .streams
            .iter()
            .enumerate()
            .map(|(index, (parameters, time_base))| {
                replay.add_stream(format!("track-{index}"), parameters.clone(), *time_base)
            })
            .collect();
        let (mut sinks, handle) = replay.open().expect("the buffer opens");
        let sinks = tracks
            .into_iter()
            .map(|track| sinks.take(track).expect("the buffer's own track"))
            .collect();
        (sinks, handle)
    }

    fn feed<'a>(
        sinks: &mut [Box<dyn Sink>],
        packets: impl IntoIterator<Item = &'a (usize, ffmpeg::Packet)>,
    ) {
        for (track, packet) in packets {
            sinks[*track]
                .consume(MediaBuffer::Packet(Arc::new(packet.clone())))
                .expect("a packet is taken");
        }
    }

    fn fixture() -> Option<Recorded> {
        crate::init().ok()?;
        let path = crate::test_support::try_test_video()?;
        Some(read(std::path::Path::new(&path)))
    }

    fn clip_path(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "replay_buffer_{label}_{}_{:?}.mp4",
            std::process::id(),
            std::thread::current().id()
        ))
    }

    /// Which track is the picture, and each track's first and last packet
    /// of a saved clip, as seconds.
    struct Readback {
        video: usize,
        first: Vec<ffmpeg::Packet>,
        span: Vec<(f64, f64)>,
        streams: usize,
    }

    fn read_back(path: &std::path::Path) -> Readback {
        let recorded = read(path);
        let video = recorded
            .streams
            .iter()
            .position(|(parameters, _)| parameters.medium() == ffmpeg::media::Type::Video)
            .expect("the clip has its picture");
        let seconds = |track: usize, value: i64| {
            let base = recorded.streams[track].1;
            value as f64 * f64::from(base.numerator()) / f64::from(base.denominator())
        };
        let mut first: Vec<Option<ffmpeg::Packet>> = vec![None; recorded.streams.len()];
        let mut span = vec![(f64::MAX, f64::MIN); recorded.streams.len()];
        for (track, packet) in &recorded.packets {
            let pts = packet.pts().expect("a written packet is stamped");
            assert!(
                packet.dts().is_none_or(|dts| dts >= 0),
                "no track may be written before zero"
            );
            let at = seconds(*track, pts);
            span[*track] = (span[*track].0.min(at), span[*track].1.max(at));
            first[*track].get_or_insert_with(|| packet.clone());
        }
        Readback {
            video,
            first: first
                .into_iter()
                .map(|packet| packet.expect("every track has packets"))
                .collect(),
            span,
            streams: recorded.streams.len(),
        }
    }

    /// Where the window over `recorded` should start and end, in
    /// microseconds of its picture's decode timeline: the earliest keyframe
    /// no further back than `length` from the last packet, or the last
    /// keyframe when every one is further — what this module's docs promise,
    /// worked out from the file's own keyframes rather than from any
    /// spacing a particular fixture happens to have.
    fn expected_window(recorded: &Recorded, length: Duration) -> (i64, i64) {
        let video = recorded
            .streams
            .iter()
            .position(|(parameters, _)| parameters.medium() == ffmpeg::media::Type::Video)
            .expect("the fixture has a picture");
        let base = recorded.streams[video].1;
        let at = |packet: &ffmpeg::Packet| {
            packet
                .dts()
                .or(packet.pts())
                .expect("a demuxed packet is stamped")
                .rescale(base, MICROSECONDS)
        };
        let pictures: Vec<_> = recorded
            .packets
            .iter()
            .filter(|(track, _)| *track == video)
            .map(|(_, packet)| packet)
            .collect();
        let newest = at(pictures.last().expect("the fixture has frames"));
        let keys: Vec<i64> = pictures
            .iter()
            .filter(|packet| packet.is_key())
            .map(|packet| at(packet))
            .collect();
        let length = i64::try_from(length.as_micros()).expect("a short window");
        let start = keys
            .iter()
            .copied()
            .find(|key| newest - key <= length)
            .unwrap_or(*keys.last().expect("the fixture has a keyframe"));
        (start, newest)
    }

    /// The contract this exists for: a buffer fed more than its window
    /// saves the stretch that ends at the last packet and starts on the
    /// earliest keyframe inside the window — opening on that keyframe, at
    /// zero, with the sound kept from where the picture starts.
    #[test]
    fn a_saved_clip_is_the_last_window_from_a_keyframe_at_zero() {
        let Some(recorded) = fixture() else { return };
        let window = Duration::from_secs(3);
        let (mut sinks, handle) = buffer_over(&recorded, window);
        feed(&mut sinks, &recorded.packets);

        let (start, newest) = expected_window(&recorded, window);
        assert_eq!(handle.buffered(), micros(newest - start));
        assert!(handle.buffered() <= window);

        let path = clip_path("window");
        let length = handle.save(&path).expect("the clip saves");
        let clip = read_back(&path);

        assert_eq!(
            clip.streams,
            recorded.streams.len(),
            "every track is declared"
        );
        assert!(
            clip.first[clip.video].is_key(),
            "the picture opens on a keyframe"
        );
        assert!(
            length >= handle.buffered() && length - handle.buffered() < Duration::from_millis(100),
            "the clip is what was held, and its last frame: {length:?}"
        );
        let (first, last) = clip.span[clip.video];
        assert!(first.abs() < 0.1, "the picture starts at zero: {first}");
        assert!(
            (last - micros(newest - start).as_secs_f64()).abs() < 0.1,
            "and ends where the window did: {last}"
        );
        for (track, (first, _)) in clip.span.iter().enumerate() {
            assert!(
                *first >= 0.0 && *first < 0.1,
                "track {track} starts with the picture: {first}"
            );
        }
        assert!(
            first_picture_decodes(&path),
            "a player can show the clip's first frame on its own"
        );
        std::fs::remove_file(&path).ok();
    }

    /// Whether the clip's first video packet decodes to a picture with
    /// nothing before it — which is what opening on a keyframe is for.
    fn first_picture_decodes(path: &std::path::Path) -> bool {
        let mut input = ffmpeg::format::input(&path).expect("the clip opens");
        let stream = input
            .streams()
            .best(ffmpeg::media::Type::Video)
            .expect("the clip has its picture");
        let index = stream.index();
        let mut decoder = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
            .and_then(|context| context.decoder().video())
            .expect("the picture's decoder opens");
        let Some((_, packet)) = input.packets().find(|(stream, _)| stream.index() == index) else {
            return false;
        };
        decoder
            .send_packet(&packet)
            .expect("the first packet is taken");
        decoder.send_eof().expect("and the stream ends there");
        let mut picture = ffmpeg::frame::Video::empty();
        decoder.receive_frame(&mut picture).is_ok()
    }

    /// What arrives before the first keyframe cannot start a clip, so it is
    /// not kept — a buffer attached mid-GOP still saves a clip that opens on
    /// one.
    #[test]
    fn packets_before_the_first_keyframe_are_not_kept() {
        let Some(recorded) = fixture() else { return };
        let video = recorded
            .streams
            .iter()
            .position(|(parameters, _)| parameters.medium() == ffmpeg::media::Type::Video)
            .expect("the fixture has a picture");
        // From just past the first keyframe, so the stream begins mid-GOP.
        let joined = recorded
            .packets
            .iter()
            .skip_while(|(track, packet)| !(*track == video && packet.is_key()))
            .skip(1);
        let (mut sinks, handle) = buffer_over(&recorded, Duration::from_secs(30));
        feed(&mut sinks, joined);

        let path = clip_path("mid_gop");
        handle.save(&path).expect("the clip saves");
        let clip = read_back(&path);
        assert!(clip.first[clip.video].is_key());
        std::fs::remove_file(&path).ok();
    }

    /// A window shorter than one keyframe interval cannot be cut inside it,
    /// so it keeps the one GOP it has rather than nothing.
    #[test]
    fn a_window_shorter_than_a_gop_still_holds_one() {
        let Some(recorded) = fixture() else { return };
        // Shorter than any keyframe interval an encoder would be asked for.
        let window = Duration::from_millis(1);
        let (mut sinks, handle) = buffer_over(&recorded, window);
        feed(&mut sinks, &recorded.packets);

        let (start, newest) = expected_window(&recorded, window);
        assert_eq!(
            handle.buffered(),
            micros(newest - start),
            "the last GOP is held"
        );
        let path = clip_path("short");
        handle.save(&path).expect("the clip saves");
        let clip = read_back(&path);
        assert!(clip.first[clip.video].is_key());
        std::fs::remove_file(&path).ok();
    }

    /// B-frames put `dts` behind `pts`. A clip starting on the keyframe's
    /// decode time keeps every frame's order and writes nothing before zero
    /// — `read_back` refuses a negative `dts` for every packet.
    #[test]
    fn reordered_frames_are_saved_in_order_from_zero() {
        crate::init().expect("ffmpeg initializes");
        let fixture = crate::test_support::synthesize_reordered("replay-reordered", 4.0);
        let recorded = read(&fixture.path);
        let (mut sinks, handle) = buffer_over(&recorded, Duration::from_secs(2));
        feed(&mut sinks, &recorded.packets);

        let path = clip_path("reordered");
        handle.save(&path).expect("the clip saves");
        let clip = read_back(&path);
        let first = &clip.first[clip.video];
        assert!(first.is_key());
        assert!(
            first.pts() >= first.dts(),
            "presentation never precedes decode"
        );
        std::fs::remove_file(&path).ok();
    }

    /// Nothing held is an error to act on, not an empty file left behind.
    #[test]
    fn saving_before_a_keyframe_is_refused_and_writes_nothing() {
        let Some(recorded) = fixture() else { return };
        let (_sinks, handle) = buffer_over(&recorded, Duration::from_secs(3));
        let path = clip_path("empty");

        let refused = handle.save(&path);
        assert!(matches!(
            refused,
            Err(Error::ReplayBufferError(ReplayBufferError::Empty))
        ));
        assert!(
            !path.exists(),
            "no file is started for a clip that has nothing"
        );
        assert_eq!(handle.buffered(), Duration::ZERO);
    }

    /// The handle holds the window weakly: once the sinks are gone so is
    /// what they held, and saving says so.
    #[test]
    fn a_handle_outliving_its_sinks_answers_stopped() {
        let Some(recorded) = fixture() else { return };
        let (mut sinks, handle) = buffer_over(&recorded, Duration::from_secs(3));
        feed(&mut sinks, &recorded.packets);
        assert!(handle.buffered() > Duration::ZERO);

        drop(sinks);
        assert!(matches!(
            handle.save(clip_path("stopped")),
            Err(Error::ReplayBufferError(ReplayBufferError::Stopped))
        ));
        assert_eq!(handle.buffered(), Duration::ZERO);
    }

    /// A flush discards the timeline what is held belongs to, and a stream
    /// that has ended leaves its last stretch saveable.
    #[test]
    fn a_flush_empties_the_window_and_an_end_of_stream_does_not() {
        let Some(recorded) = fixture() else { return };
        let (mut sinks, handle) = buffer_over(&recorded, Duration::from_secs(3));
        feed(&mut sinks, &recorded.packets);

        for sink in &mut sinks {
            sink.consume(MediaBuffer::Eos).expect("eos");
        }
        assert!(
            handle.buffered() > Duration::ZERO,
            "an ended stream is still held"
        );

        sinks[0].control(ControlMsg::Flush).expect("flush");
        assert_eq!(handle.buffered(), Duration::ZERO);
    }

    /// A timeline that goes backwards is a new one: what was held is let
    /// go rather than measured across the jump, which would never trim.
    #[test]
    fn a_timeline_that_goes_backwards_starts_the_window_again() {
        let Some(recorded) = fixture() else { return };
        // Longer than the file, so nothing is let go of by the window itself.
        let window = Duration::from_secs(3_600);
        let (mut sinks, handle) = buffer_over(&recorded, window);
        feed(&mut sinks, &recorded.packets);
        let (start, newest) = expected_window(&recorded, window);
        assert_eq!(handle.buffered(), micros(newest - start));

        // The first half again, stamped exactly as before.
        let again = Recorded {
            streams: recorded.streams.clone(),
            packets: recorded.packets[..recorded.packets.len() / 2].to_vec(),
        };
        feed(&mut sinks, &again.packets);

        let (start, newest) = expected_window(&again, window);
        assert_eq!(
            handle.buffered(),
            micros(newest - start),
            "only the second run is held"
        );
    }

    /// Asking for nothing is refused before any sink exists.
    #[test]
    fn opening_with_no_length_or_no_tracks_is_refused() {
        assert!(matches!(
            ReplayBuffer::create(Duration::from_secs(3)).open(),
            Err(Error::ReplayBufferError(ReplayBufferError::NoTracks))
        ));
        let Some(recorded) = fixture() else { return };
        let mut replay = ReplayBuffer::create(Duration::ZERO);
        let (parameters, time_base) = recorded.streams[0].clone();
        let _track = replay.add_stream("video", parameters, time_base);
        assert!(matches!(
            replay.open(),
            Err(Error::ReplayBufferError(ReplayBufferError::ZeroLength))
        ));
    }

    /// A track takes its own medium's packets, as a muxer's does — and a
    /// decoded frame is refused, since nothing here encodes.
    #[test]
    fn a_track_declares_the_packets_of_its_own_medium() {
        let Some(recorded) = fixture() else { return };
        let (mut sinks, _handle) = buffer_over(&recorded, Duration::from_secs(3));
        for (sink, (parameters, _)) in sinks.iter().zip(&recorded.streams) {
            let kind = MediaKind::packet_for(parameters.medium()).expect("audio or video");
            assert_eq!(
                sink.input_contract(),
                InputContract::Fixed(PortContract::packet(kind))
            );
        }
        let frame = crate::buffer::MediaBuffer::Audio(Arc::new(ffmpeg::frame::Audio::empty()));
        assert!(matches!(
            sinks[0].consume(frame),
            Err(Error::ReplayBufferError(
                ReplayBufferError::UnsupportedBuffer(_)
            ))
        ));
    }
}
