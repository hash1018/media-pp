//! One stream of encoded packets, written once and read by many: a
//! [`Broadcast`].
//!
//! A live input — a camera pulled with [`RtspSource`](crate::elements::RtspSource),
//! a publisher received with `RtspSource::listen` — is written into a
//! broadcast once, through the [`BroadcastSink`]s its
//! [`Broadcast::publish`] hands out, and read by as many as want it: a
//! server's viewers through a [`BroadcastReader`] each, a recording through
//! a [`BroadcastSource`] at the start of a pipeline of its own. Nothing is
//! decoded or encoded again for any of them.
//!
//! - The writer never waits. A reader that cannot keep up is the only one
//!   that loses anything: it is moved on to a keyframe, and told.
//! - A broadcast keeps only the newest [`BroadcastOptions::retain`] of
//!   what it is written — room for a reader to fall behind, a second by
//!   default — and no group of pictures for one to start at. A reader starts
//!   at a keyframe: the newest one held, or the next one written.
//! - A broadcast outlives what writes into it. An input that drops and is
//!   opened again publishes again into the same broadcast, and its readers
//!   go on: they are told that one run ended and another began.

mod reader;
mod sink;
mod source;
mod state;
mod timeline;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use thiserror::Error as ThisError;

use crate::element::{ElementType, element_pp_log};
use crate::elements::StreamInfo;
use crate::pp_log::{PpLog, pp_info};

pub use reader::{BroadcastItem, BroadcastReader};
pub use sink::BroadcastSink;
pub use source::BroadcastSource;
#[cfg(feature = "rtsp-server")]
pub(crate) use state::time_base;
#[cfg(feature = "rtsp-server")]
pub(crate) use timeline::same_shape;

/// How far behind the newest packet a server's player may join at a
/// keyframe; further, and it starts at the next one. What it joins with is
/// sent at twice the stream's speed, and a second of a large picture's
/// packets is as much as a burst can carry without losing what cannot be
/// sent again — so a player starts up to a group of pictures later, but
/// whole.
#[cfg(any(feature = "rtsp-server", feature = "webrtc"))]
pub(crate) const PLAYER_JOIN_WITHIN: Duration = Duration::from_secs(1);

use state::State;

/// Why a [`Broadcast`] or what it hands out refused something.
///
/// Converts into the crate-wide `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum BroadcastError {
    /// [`Broadcast::publish`] was given no tracks.
    #[error("a broadcast is published with at least one track")]
    NoTracks,
    /// A track's codec parameters say neither video, sound nor text.
    #[error("track {index} is neither video, audio nor subtitles")]
    UnsupportedTrack {
        /// Where the track was in the list published.
        index: usize,
    },
    /// The broadcast has been closed: nothing is written to it or read from
    /// it any more.
    #[error("broadcast {0} is closed")]
    Closed(Arc<str>),
    /// The sink belongs to a publication that a later one replaced: what it
    /// is handed goes nowhere.
    #[error("broadcast {0} has been published again; this sink's input is no longer its")]
    Superseded(Arc<str>),
    /// A [`BroadcastSink`] was handed something other than a packet.
    #[error("a BroadcastSink takes encoded packets, got {0}")]
    UnsupportedBuffer(&'static str),
    /// A [`BroadcastSource`] was made for a broadcast nothing has been
    /// published to: there are no tracks yet to make outputs of.
    #[error("broadcast {0} has not been published to yet")]
    NotPublished(Arc<str>),
}

/// How a [`Broadcast`] keeps what it is written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BroadcastOptions {
    /// How much of the newest stretch to keep, packet by packet: how far a
    /// reader may fall behind — busy for a moment, its connection slow for
    /// one — before it is moved on to a keyframe. One second by default.
    ///
    /// Nothing more is kept, keyframe or not, which is what a broadcast
    /// costs: a second of a 4 Mbit/s camera is half a megabyte. So a reader
    /// that joins starts at the newest keyframe held where one is, and
    /// otherwise waits for the next one written — up to a keyframe interval
    /// before its first picture, as a server that keeps no group of
    /// pictures has its players wait. A `retain` longer than the keyframe
    /// interval always holds one, and so starts every reader at once, for
    /// what it holds.
    ///
    /// It is kept while the broadcast has a reader. With none — a camera a
    /// server pulls that nobody plays — what is written is let go of as the
    /// next is, since nothing can fall behind by it: the first reader to
    /// come starts at the next keyframe, and what is kept from then on is
    /// kept for it and those after.
    pub retain: Duration,
}

impl Default for BroadcastOptions {
    fn default() -> Self {
        Self {
            retain: Duration::from_secs(1),
        }
    }
}

/// One stream of encoded packets, written once and read by many.
///
/// See the [module](self) for what it is for. A cheap handle: clones share
/// the one broadcast. The broadcast is closed — its readers told so, its
/// sinks refused — once [`Self::close`] is called or the last clone is
/// dropped; readers and sinks alone do not keep it open.
///
/// # Example
///
/// A camera's stream written once and recorded, while a reader of its own
/// counts what it is handed:
///
/// ```no_run
/// # fn main() -> media_pp::Result<()> {
/// use media_pp::elements::{
///     Broadcast, BroadcastItem, BroadcastOptions, BroadcastSource, FileMuxer, Mp4Layout,
///     RtspOptions, RtspSource, StreamInfo,
/// };
/// use media_pp::pipeline::Pipeline;
///
/// let camera = Broadcast::new("camera", BroadcastOptions::default());
///
/// // The input: every stream of the camera into the broadcast.
/// let (source, streams) = RtspSource::open("rtsp", "rtsp://192.168.0.10/stream", RtspOptions::default())?;
/// let tracks: Vec<StreamInfo> = streams.iter().map(StreamInfo::from).collect();
/// let sinks = camera.publish(&tracks)?;
/// let (input, ()) = Pipeline::new("input", source, |source, ctx| {
///     for (stream, sink) in streams.iter().zip(sinks) {
///         let branch = ctx.branch().queue("packets", 64).to(sink)?;
///         ctx.attach(source, stream.index(), branch)?;
///     }
///     Ok(())
/// })?;
/// input.run()?;
///
/// // A recording, from the broadcast's newest keyframe on.
/// let (recorded, streams) = BroadcastSource::new("recorder", &camera)?;
/// let mut muxer = FileMuxer::create("camera.mp4", Mp4Layout::Standard)?;
/// let tracks: Vec<_> = streams.iter().map(|stream| muxer.add_stream("track", stream)).collect::<Result<_, _>>()?;
/// let mut files = muxer.open()?;
/// let mut ends = Vec::new();
/// for track in tracks {
///     ends.push(files.take(track)?);
/// }
/// let (recording, ()) = Pipeline::new("recording", recorded, |source, ctx| {
///     for (index, end) in ends.into_iter().enumerate() {
///         let branch = ctx.branch().to(end)?;
///         ctx.attach(source, index, branch)?;
///     }
///     Ok(())
/// })?;
/// recording.run()?;
///
/// // A reader of one's own.
/// let mut reader = camera.subscribe();
/// while let Some(item) = reader.next_timeout(std::time::Duration::from_secs(1)) {
///     if let BroadcastItem::Packet { track, .. } = item {
///         println!("a packet of track {track}");
///     }
/// }
/// # Ok(())
/// # }
/// ```
///
/// # How it works
///
/// What is written is kept in order in one buffer, every track together,
/// with where each group of pictures begins: the keyframes of its first
/// video track, or every packet where it has none. A reader is only a
/// position in that buffer, so a thousand of them cost a thousand
/// positions, and the packets they are handed are the ones written, shared
/// and never copied.
///
/// Each [`Self::publish`] begins a run: the tracks it is given, and what
/// its sinks are handed after. A run ends when every one of its sinks has
/// been handed the end of its stream, or stopped, or dropped, or when the
/// broadcast is published again; what a run kept is let go of when the next
/// one begins, since a new input's timestamps have nothing to do with the
/// last one's. A seek in the pipeline writing — a file played into a
/// broadcast — begins a run of the same tracks.
///
/// Above [`BroadcastOptions::retain`] there is one more bound: a broadcast
/// never holds more than 256 MiB, so a `retain` long enough to hold more
/// cannot fill the memory.
#[derive(Clone)]
pub struct Broadcast {
    shared: Arc<Shared>,
    /// Closes the broadcast as the last clone of this handle goes.
    _owner: Arc<Owner>,
}

/// What every handle, sink and reader of one broadcast share.
pub(crate) struct Shared {
    pub(crate) name: Arc<str>,
    pub(crate) options: BroadcastOptions,
    pub(crate) pp_log: PpLog,
    pub(crate) state: Mutex<State>,
    /// Rung whenever something is written or the broadcast changes, for a
    /// reader that waits on a thread of its own.
    pub(crate) changed: Condvar,
    /// How many times a reader has been moved on for falling behind.
    pub(crate) fell_behind: AtomicU64,
}

impl Shared {
    /// Wakes every reader waiting on this broadcast, however it waits.
    pub(crate) fn wake(&self, state: &mut State) {
        state.wake();
        self.changed.notify_all();
    }
}

/// Closes its broadcast when the last [`Broadcast`] handle goes.
struct Owner(Arc<Shared>);

impl Drop for Owner {
    fn drop(&mut self) {
        close(&self.0);
    }
}

fn close(shared: &Shared) {
    let mut state = shared
        .state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if !state.closed {
        state.closed = true;
        pp_info!(pp_log: &shared.pp_log, "closed");
        shared.wake(&mut state);
    }
}

impl Broadcast {
    /// A broadcast with nothing published to it yet.
    pub fn new(name: impl Into<String>, options: BroadcastOptions) -> Self {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::Broadcast, &name, None);
        pp_info!(pp_log: &pp_log, "created: retain={:?}", options.retain);
        let shared = Arc::new(Shared {
            name,
            options,
            pp_log,
            state: Mutex::new(State::default()),
            changed: Condvar::new(),
            fell_behind: AtomicU64::new(0),
        });
        Self {
            _owner: Arc::new(Owner(Arc::clone(&shared))),
            shared,
        }
    }

    /// Its name, as its log records and its sinks' errors give it.
    pub fn name(&self) -> Arc<str> {
        Arc::clone(&self.shared.name)
    }

    /// Begins a run of `tracks`, and answers a sink for each, in the same
    /// order: the end of the branch carrying that track's packets.
    ///
    /// A run already going ends here — its readers are told — and its
    /// sinks are refused from now on with [`BroadcastError::Superseded`].
    /// This is what an input opened again after it dropped does.
    ///
    /// # Errors
    ///
    /// [`BroadcastError::NoTracks`] for an empty list,
    /// [`BroadcastError::UnsupportedTrack`] for a track that is neither
    /// video, audio nor subtitles, and [`BroadcastError::Closed`] once the
    /// broadcast is closed. Nothing changes on an error.
    pub fn publish(&self, tracks: &[StreamInfo]) -> Result<Vec<BroadcastSink>, BroadcastError> {
        if tracks.is_empty() {
            return Err(BroadcastError::NoTracks);
        }
        let mut infos = Vec::with_capacity(tracks.len());
        for (index, track) in tracks.iter().enumerate() {
            if crate::contract::MediaKind::packet_for(track.kind()).is_none() {
                return Err(BroadcastError::UnsupportedTrack { index });
            }
            infos.push(track.clone().with_index(index));
        }
        let publication = {
            let mut state = self.shared.state.lock().unwrap_or_else(|p| p.into_inner());
            if state.closed {
                return Err(BroadcastError::Closed(self.name()));
            }
            let publication = state.publish(infos.clone());
            pp_info!(
                pp_log: &self.shared.pp_log,
                "published: run={} tracks=[{}]",
                state.run,
                describe(&infos)
            );
            self.shared.wake(&mut state);
            publication
        };
        Ok(infos
            .iter()
            .map(|info| BroadcastSink::new(Arc::clone(&self.shared), publication, info))
            .collect())
    }

    /// A reader of everything written from now on, starting at the newest
    /// keyframe the broadcast holds. See [`BroadcastReader`].
    pub fn subscribe(&self) -> BroadcastReader {
        BroadcastReader::new(Arc::clone(&self.shared))
    }

    /// A reader as [`Self::subscribe`] makes one, but one that joins at the
    /// newest keyframe only where it is at most `within` behind the newest
    /// packet, and at the next keyframe else: for a player across a network,
    /// to which what it joins with is sent faster than the stream goes.
    #[cfg(any(feature = "rtsp-server", feature = "webrtc"))]
    pub(crate) fn subscribe_within(&self, within: Duration) -> BroadcastReader {
        BroadcastReader::joining_within(Arc::clone(&self.shared), Some(within))
    }

    /// The tracks of the run going now, or of the last one where none is —
    /// `None` before anything has been published.
    pub fn tracks(&self) -> Option<Vec<StreamInfo>> {
        let state = self.shared.state.lock().unwrap_or_else(|p| p.into_inner());
        state.tracks.as_ref().map(|tracks| tracks.to_vec())
    }

    /// Whether a run is going: something has been published and its sinks
    /// have not all ended.
    pub fn is_live(&self) -> bool {
        let state = self.shared.state.lock().unwrap_or_else(|p| p.into_inner());
        state.live
    }

    /// Whether it has been closed.
    pub fn is_closed(&self) -> bool {
        let state = self.shared.state.lock().unwrap_or_else(|p| p.into_inner());
        state.closed
    }

    /// How many readers it has — [`BroadcastReader`]s and
    /// [`BroadcastSource`]s alike.
    pub fn subscribers(&self) -> usize {
        let state = self.shared.state.lock().unwrap_or_else(|p| p.into_inner());
        state.subscribers
    }

    /// Waits until it has other than `from` readers, and answers how many
    /// it has: at once where that is already so.
    #[cfg(any(test, feature = "rtsp-server", feature = "webrtc"))]
    pub(crate) async fn readers_other_than(&self, from: usize) -> usize {
        std::future::poll_fn(|cx| {
            let mut state = self.shared.state.lock().unwrap_or_else(|p| p.into_inner());
            if state.subscribers != from {
                return std::task::Poll::Ready(state.subscribers);
            }
            state.add_counting(cx.waker());
            std::task::Poll::Pending
        })
        .await
    }

    /// What it is doing and has done, for a dashboard or a list of what a
    /// server serves. See [`BroadcastStats`].
    pub fn stats(&self) -> BroadcastStats {
        let state = self.shared.state.lock().unwrap_or_else(|p| p.into_inner());
        BroadcastStats {
            live: state.live,
            live_since: state.published_at.filter(|_| state.live),
            publications: state.publication,
            readers: state.subscribers,
            bytes: state.bytes_written,
            packets: state.packets_written,
            last_written: state.last_written,
            fell_behind: self.shared.fell_behind.load(Ordering::Relaxed),
        }
    }

    /// Closes it now rather than when the last handle goes: readers are told
    /// it is closed once they have read what it holds, and sinks are refused
    /// with [`BroadcastError::Closed`].
    pub fn close(&self) {
        close(&self.shared);
    }
}

impl std::fmt::Debug for Broadcast {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Broadcast")
            .field("name", &self.shared.name)
            .finish()
    }
}

/// What a [`Broadcast`] is doing and has done, read with
/// [`Broadcast::stats`]: a snapshot, taken at once under its lock.
///
/// The counts only rise, from the broadcast's making: a rate is two
/// snapshots apart — the input's bitrate is the difference in [`bytes`]
/// over the time between them, times eight.
///
/// [`bytes`]: Self::bytes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct BroadcastStats {
    /// Whether a run is going — as [`Broadcast::is_live`].
    pub live: bool,
    /// Since when the publication going now was published, while one is;
    /// a seek in the pipeline writing it goes on from the same time.
    pub live_since: Option<Instant>,
    /// How many times it has been published: an input that dropped and came
    /// back — a publisher reconnecting — counts once more each time.
    pub publications: u64,
    /// How many readers it has now — as [`Broadcast::subscribers`]: every
    /// player of every server serving it, and every recording of it.
    pub readers: usize,
    /// The packets' bytes written into it.
    pub bytes: u64,
    /// The packets written into it.
    pub packets: u64,
    /// When the last packet was written: a live broadcast written nothing
    /// for a while has an input that stalled without leaving.
    pub last_written: Option<Instant>,
    /// How many times a reader fell behind as far as it keeps and was
    /// moved on to its newest keyframe — a picture its player skipped.
    pub fell_behind: u64,
}

/// The tracks, for a log record.
fn describe(tracks: &[StreamInfo]) -> String {
    tracks
        .iter()
        .map(|track| format!("{:?}/{:?}", track.kind(), track.parameters().id()))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests;
