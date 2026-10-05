//! Many streams through one model: [`StreamMux`] gathers a picture from each
//! of its inputs into a batch, for an element that runs a model to take at
//! once — DeepStream's `nvstreammux`. See docs/stream-mux.md.
//!
//! A batch is not a buffer of its own. The mux hands its pictures on one
//! after another, each carrying [`StreamOrigin`] and [`BatchSlot`]; every
//! element that does not care passes them on as it would any picture.

mod input;
mod meta;

use std::collections::VecDeque;
use std::sync::{
    Arc, Mutex, Weak,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::time::{Duration, Instant};

use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_info};
use crate::{
    buffer::MediaBuffer,
    element::{BoxSink, Element, ElementType, Produced, Source, SourceStage, Wait, element_pp_log},
    elements::source::render_mode::RenderMode,
    error::Result,
    playback_state::{Bell, PlaybackState},
    produce::source_stage,
};

pub use input::StreamMuxInput;
pub use meta::{BatchSlot, StreamId, StreamOrigin};

/// Errors specific to [`StreamMux`]. Converts into the crate-wide `Error`
/// via `?`.
#[derive(Debug, ThisError)]
pub enum StreamMuxError {
    /// An input was handed something other than a decoded picture.
    #[error("StreamMux inputs only take video frames, got {0}")]
    UnsupportedBuffer(&'static str),
    /// The mux this handle belongs to has stopped.
    #[error("the stream mux has stopped")]
    Stopped,
    /// [`StreamMuxOptions`] that no mux can run with.
    #[error("invalid stream mux options: {0}")]
    InvalidOptions(&'static str),
}

/// How a [`StreamMux`] makes its batches.
#[derive(Debug, Clone, Copy)]
pub struct StreamMuxOptions {
    /// Live, a batch goes out when every input has a picture or after
    /// [`Self::batch_timeout`]; offline, once every input that has not ended
    /// has one — see [`StreamMux`].
    pub mode: RenderMode,
    /// The most pictures in one batch. With more inputs than this, they take
    /// turns. At least 1.
    pub max_batch: usize,
    /// Live, how long the first picture of a batch waits for the inputs that
    /// have none before the batch goes out without them.
    pub batch_timeout: Duration,
    /// How many pictures each input holds. At least 1.
    pub queue: usize,
}

impl Default for StreamMuxOptions {
    /// Live; batches of up to 8; 40 ms, a little more than a frame at 30 fps;
    /// four pictures an input.
    fn default() -> Self {
        Self {
            mode: RenderMode::Live,
            max_batch: 8,
            batch_timeout: Duration::from_millis(40),
            queue: 4,
        }
    }
}

/// One input, as the mux holds it.
struct Input {
    /// Its registration, which is also its [`StreamId`].
    id: u64,
    name: Arc<str>,
    pictures: VecDeque<MediaBuffer>,
    ended: bool,
    generation: u64,
    /// The last batch a picture of it went out in.
    last_batch: Option<u64>,
    dropped: u64,
    /// The pipeline feeding it, woken as an offline mux makes room.
    upstream: Weak<PlaybackState>,
}

/// A stream that has ended, or been removed, and the last batch it had a
/// picture in — after which nothing more of it comes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Ended {
    pub(crate) id: StreamId,
    pub(crate) after: Option<u64>,
}

/// What the mux, its inputs on their own threads, and its handle share.
pub(super) struct MuxShared {
    inputs: Mutex<Vec<Input>>,
    /// Streams that ended or were removed, for whatever splits the streams
    /// out again to finish theirs. Taken by [`MuxShared::take_ended`].
    ended: Mutex<Vec<Ended>>,
    next_id: AtomicU64,
    options: StreamMuxOptions,
    /// Rung as anything a batch waits for changes.
    arrived: Bell,
    /// Whether an input has ever been added: an offline mux with none has
    /// not ended, it has not begun.
    fed: AtomicBool,
}

impl MuxShared {
    /// Runs `change` on input `id`, if it is still registered.
    fn with_input<T>(&self, id: u64, change: impl FnOnce(&mut Input) -> T) -> Option<T> {
        let mut inputs = self.inputs.lock().unwrap();
        inputs.iter_mut().find(|input| input.id == id).map(change)
    }

    /// Takes input `id` out, recording that its stream is over.
    fn remove(&self, id: u64) {
        let mut inputs = self.inputs.lock().unwrap();
        if let Some(index) = inputs.iter().position(|input| input.id == id) {
            let input = inputs.remove(index);
            self.ended.lock().unwrap().push(Ended {
                id: StreamId(input.id),
                after: input.last_batch,
            });
        }
    }

    /// The streams that have ended since the last call.
    #[allow(dead_code)] // read by the demux, which comes next
    pub(crate) fn take_ended(&self) -> Vec<Ended> {
        std::mem::take(&mut *self.ended.lock().unwrap())
    }
}

/// Adds and removes a [`StreamMux`]'s inputs while it runs, from any thread.
/// Cheap to clone; keeps only a [`Weak`] reference, so a handle kept after
/// the mux's pipeline is gone keeps nothing of it alive, and every call then
/// answers [`StreamMuxError::Stopped`] or does nothing.
#[derive(Clone)]
pub struct StreamMuxHandle {
    shared: Weak<MuxShared>,
}

impl StreamMuxHandle {
    /// Registers an input under `name`, returning the sink to end its
    /// pipeline with and the [`StreamId`] its pictures will carry.
    ///
    /// Each call is a new stream, a name already in use included: the input
    /// it replaces is removed — whatever it held let go of — and a sink of
    /// the replaced one changes nothing from then on.
    pub fn add_source(
        &self,
        name: impl Into<String>,
    ) -> std::result::Result<(BoxSink, StreamId), StreamMuxError> {
        let shared = self.shared.upgrade().ok_or(StreamMuxError::Stopped)?;
        let name: Arc<str> = name.into().into();
        let id = shared.next_id.fetch_add(1, Ordering::Relaxed);
        let replaced = shared
            .inputs
            .lock()
            .unwrap()
            .iter()
            .find(|input| input.name == name)
            .map(|input| input.id);
        if let Some(replaced) = replaced {
            shared.remove(replaced);
        }
        shared.inputs.lock().unwrap().push(Input {
            id,
            name: name.clone(),
            pictures: VecDeque::new(),
            ended: false,
            generation: 0,
            last_batch: None,
            dropped: 0,
            upstream: Weak::new(),
        });
        shared.fed.store(true, Ordering::Release);
        shared.arrived.ring();
        let sink = StreamMuxInput {
            pp_log: element_pp_log(ElementType::StreamMuxInput, &name, None),
            name,
            id,
            shared: self.shared.clone(),
        };
        Ok((BoxSink::new(sink), StreamId(id)))
    }

    /// Removes `name`'s input at once, letting go of what it holds — nothing
    /// if no input is registered under it, or the mux is gone.
    pub fn remove_source(&self, name: &str) {
        let Some(shared) = self.shared.upgrade() else {
            return;
        };
        let id = shared
            .inputs
            .lock()
            .unwrap()
            .iter()
            .find(|input| *input.name == *name)
            .map(|input| input.id);
        if let Some(id) = id {
            shared.remove(id);
            shared.arrived.ring();
        }
    }

    /// How many inputs are registered; zero once the mux is gone.
    pub fn source_count(&self) -> usize {
        self.shared
            .upgrade()
            .map_or(0, |shared| shared.inputs.lock().unwrap().len())
    }
}

/// Gathers a picture from each of its inputs into a batch, for an element
/// that runs a model to take at once — DeepStream's `nvstreammux`.
///
/// Inputs come from other pipelines, each ending in the sink
/// [`StreamMuxHandle::add_source`] returns, and come and go while it runs.
/// Every picture it hands on carries [`StreamOrigin`] — which input — and
/// [`BatchSlot`] — which batch, and where in it — beside whatever metadata it
/// came with, and keeps its own timestamps: the mux's output interleaves
/// several timelines, which an element keeping state per stream reads per
/// stream. The pictures of a batch go out one after another, in slot order.
///
/// **Live** ([`RenderMode::Live`]), a batch goes out as soon as every input
/// has a picture — or enough to fill [`StreamMuxOptions::max_batch`] — or
/// once its first picture has waited [`StreamMuxOptions::batch_timeout`],
/// with whichever inputs have one: an input that stalls holds the others
/// back by that much at most. An input whose queue is full lets go of its
/// oldest picture. It runs until stopped.
///
/// **Offline** ([`RenderMode::Offline`]), a batch goes out once every input
/// that has not ended has a picture, with no timeout, and a full queue holds
/// its feeding pipeline back — which needs a [`crate::queue::Queue`] in that
/// pipeline to wait in. It ends once every input it was given has ended and
/// been handed on.
///
/// Each input gives at most one picture to a batch. With more inputs than
/// `max_batch`, the ones left out go first in the next.
///
/// It decodes nothing and copies no picture, so it takes pictures wherever
/// they live; what reads them — a detector — says where it needs them.
pub struct StreamMux(SourceStage<Muxing>);

source_stage!(StreamMux);

/// What a [`StreamMux`] does when asked: the next picture of a batch.
struct Muxing {
    pp_log: PpLog,
    name: Arc<str>,
    shared: Arc<MuxShared>,
    /// The rest of the batch being handed on.
    pending: VecDeque<MediaBuffer>,
    /// The next batch's number.
    next_batch: u64,
    /// Where the next batch starts taking from, among the inputs.
    cursor: usize,
    /// Live, when the batch being gathered got its first picture.
    gathering_since: Option<Instant>,
}

impl StreamMux {
    /// A mux with no inputs yet; add them through the returned handle
    /// before or after its pipeline starts. Fails with
    /// [`StreamMuxError::InvalidOptions`] for a `max_batch` or `queue` of 0.
    pub fn new(
        name: impl Into<String>,
        options: StreamMuxOptions,
    ) -> std::result::Result<(Self, StreamMuxHandle), StreamMuxError> {
        if options.max_batch == 0 {
            return Err(StreamMuxError::InvalidOptions(
                "max_batch must be at least 1",
            ));
        }
        if options.queue == 0 {
            return Err(StreamMuxError::InvalidOptions("queue must be at least 1"));
        }
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::StreamMux, &name, None);
        pp_info!(
            pp_log: &pp_log,
            "created: max_batch={}, batch_timeout={:?}, queue={}, mode={:?}",
            options.max_batch,
            options.batch_timeout,
            options.queue,
            options.mode
        );
        let shared = Arc::new(MuxShared {
            inputs: Mutex::new(Vec::new()),
            ended: Mutex::new(Vec::new()),
            next_id: AtomicU64::new(0),
            options,
            arrived: Bell::new(),
            fed: AtomicBool::new(false),
        });
        let handle = StreamMuxHandle {
            shared: Arc::downgrade(&shared),
        };
        Ok((
            Self(SourceStage::new(Muxing {
                pp_log,
                name,
                shared,
                pending: VecDeque::new(),
                next_batch: 0,
                cursor: 0,
                gathering_since: None,
            })),
            handle,
        ))
    }
}

impl Element for Muxing {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::StreamMux
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for Muxing {
    fn is_live(&self) -> bool {
        self.shared.options.mode.is_live()
    }

    fn produce(&mut self, wait: &mut Wait<'_>) -> Result<Produced> {
        if let Some(picture) = self.pending.pop_front() {
            return Ok(Produced::Buffer(picture));
        }
        let live = self.shared.options.mode.is_live();
        let shared = Arc::clone(&self.shared);
        let mut inputs = shared.inputs.lock().unwrap();
        self.retire(&mut inputs);
        let holding = inputs
            .iter()
            .filter(|input| !input.pictures.is_empty())
            .count();
        let ready = if live {
            let full = holding > 0 && holding >= inputs.len().min(self.shared.options.max_batch);
            if holding == 0 {
                self.gathering_since = None;
                false
            } else if full {
                true
            } else {
                let since = *self.gathering_since.get_or_insert_with(|| wait.now());
                let deadline = since + self.shared.options.batch_timeout;
                if wait.now() >= deadline {
                    true
                } else {
                    drop(inputs);
                    let _ = wait.recv_until(self.shared.arrived.rings(), deadline);
                    return Ok(Produced::Nothing);
                }
            }
        } else {
            let fed = self.shared.fed.load(Ordering::Acquire);
            if fed && inputs.is_empty() {
                pp_info!(self, "finished: {} batches", self.next_batch);
                return Ok(Produced::End);
            }
            // Every input that may still give one has one.
            let settled = inputs
                .iter()
                .all(|input| input.ended || !input.pictures.is_empty());
            fed && holding > 0 && settled
        };
        if !ready {
            drop(inputs);
            let _ = wait.recv(self.shared.arrived.rings());
            return Ok(Produced::Nothing);
        }
        let room = self.take_batch(&mut inputs);
        self.retire(&mut inputs);
        drop(inputs);
        self.gathering_since = None;
        // Offline, the inputs that gave a picture have room again.
        if !live {
            for upstream in room {
                if let Some(upstream) = upstream.upgrade() {
                    upstream.wake();
                }
            }
        }
        Ok(self
            .pending
            .pop_front()
            .map_or(Produced::Nothing, Produced::Buffer))
    }

    /// A stop abandons the batch being handed on.
    fn stopping(&mut self) {
        self.pending.clear();
    }
}

impl Muxing {
    /// Takes the next batch into `pending`: a picture from each input that
    /// has one, up to `max_batch`, starting where the last left off. Returns
    /// the pipelines of the inputs it took from.
    fn take_batch(&mut self, inputs: &mut [Input]) -> Vec<Weak<PlaybackState>> {
        let count = inputs.len();
        let max = self.shared.options.max_batch;
        let start = self.cursor % count.max(1);
        let mut taken = Vec::new();
        for step in 0..count {
            if taken.len() == max {
                break;
            }
            let index = (start + step) % count;
            if let Some(picture) = inputs[index].pictures.pop_front() {
                taken.push((index, picture));
            }
        }
        if let Some(&(last, _)) = taken.last() {
            self.cursor = last + 1;
        }
        let batch = self.next_batch;
        self.next_batch += 1;
        let size = taken.len();
        let mut room = Vec::with_capacity(size);
        for (slot, (index, picture)) in taken.into_iter().enumerate() {
            let input = &mut inputs[index];
            input.last_batch = Some(batch);
            room.push(input.upstream.clone());
            let metadata = picture
                .metadata()
                .cloned()
                .unwrap_or_default()
                .with(StreamOrigin {
                    id: StreamId(input.id),
                    name: input.name.clone(),
                    generation: input.generation,
                })
                .with(BatchSlot {
                    batch,
                    index: slot,
                    size,
                });
            self.pending.push_back(picture.with_metadata(metadata));
        }
        room
    }

    /// Lets go of the inputs that have ended and been handed on to their
    /// last picture, recording that their streams are over.
    fn retire(&self, inputs: &mut Vec<Input>) {
        let mut ended = Vec::new();
        inputs.retain(|input| {
            let over = input.ended && input.pictures.is_empty();
            if over {
                ended.push(Ended {
                    id: StreamId(input.id),
                    after: input.last_batch,
                });
            }
            !over
        });
        if !ended.is_empty() {
            self.shared.ended.lock().unwrap().extend(ended);
        }
    }
}

#[cfg(test)]
mod tests;
