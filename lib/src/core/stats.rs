//! What each element of a running pipeline is doing, counted as it happens.
//!
//! The graph says what a pipeline looks like; this says whether anything is
//! moving through it. A capture that has stopped delivering, a queue that
//! is full and dropping, a branch still draining after it was finished —
//! each of these used to be found by reading a log after the fact, and each
//! is a number here that can be read while it happens, through
//! [`Pipeline::stats`](crate::pipeline::Pipeline::stats).
//!
//! # Counted where the pipeline already stands
//!
//! Nothing is added to the graph to do this. Every stage a chain builds is
//! already wrapped — to trace its EOS and control, and to name it when it
//! fails — and every output port is a [`SrcPad`](crate::pad::SrcPad); the
//! counting happens in those, so an element written outside this crate is
//! counted exactly like one written inside it.
//!
//! # Counters, not rates
//!
//! What is kept is running totals and the moment of the last buffer. A rate
//! is two readings apart, which is the reader's to take: this has no window
//! to choose and no averaging to get wrong, and two readings a second apart
//! give the rate over exactly that second.
//!
//! `busy` is the time spent inside an element's `consume`, and a chain runs
//! its stages as nested calls on one thread — so it includes everything
//! downstream of that stage on the same thread, up to the next
//! [`Queue`](crate::queue::Queue). An element's own share is its `busy`
//! less that of the stage it feeds directly, which the graph's edges say.
//!
//! # A branch that has been removed
//!
//! A branch detached from a [`Tee`](crate::elements::Tee) is gone from the
//! graph at once. One *finished* is not gone from the world: its `Eos` is
//! still draining through it — an encoder flushing, a muxer writing its
//! trailer — on a thread of the `Tee`'s own. Its elements go on being
//! reported, as [`ElementState::Finishing`], until they have been dropped,
//! and then they are not reported at all. The registry holds each element's
//! counters weakly for exactly this: it sees an element for as long as the
//! element exists, and no longer.
//!
//! # Cost
//!
//! A few relaxed atomic additions and three clock readings per buffer per
//! stage — entering it, leaving it, and the pad it pushes through. On a
//! desktop CPU that is about a tenth of a microsecond, nearly all of it the
//! clock: nothing beside a frame's own work, and a pipeline has no stage
//! that does none. Nothing here takes a lock on the path a buffer travels;
//! the registry is locked only when a branch is attached or detached and
//! when a snapshot is taken.

use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::time::{Duration, Instant};

use crossbeam_channel::Sender;

use crate::{
    buffer::MediaBuffer,
    element::ElementType,
    graph::{BranchId, ElementId},
};

/// Nanoseconds since this process first asked, plus one — so that zero can
/// stand for "never".
fn now_ns() -> u64 {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    let epoch = *EPOCH.get_or_init(Instant::now);
    u64::try_from(epoch.elapsed().as_nanos())
        .unwrap_or(u64::MAX - 1)
        .saturating_add(1)
}

/// How long ago a moment recorded by [`now_ns`] was, or `None` for never.
fn since(recorded: u64, now: u64) -> Option<Duration> {
    (recorded != 0).then(|| Duration::from_nanos(now.saturating_sub(recorded)))
}

fn nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

/// One element's running totals, written by whichever thread drives it.
///
/// Shared by the element's wrapper, which holds it strongly, and the
/// pipeline's registry, which holds it weakly — see the [module](self).
#[derive(Default)]
pub(crate) struct ElementCounters {
    buffers_in: AtomicU64,
    busy_ns: AtomicU64,
    last_buffer_at: AtomicU64,
    errors: AtomicU64,
    eos: AtomicBool,
    /// This element's output ports, registered once when it is wired.
    pads: Mutex<Vec<Arc<PadCounters>>>,
    queue: Option<QueueCounters>,
}

impl ElementCounters {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// The counters of a [`Queue`](crate::queue::Queue) holding at most
    /// `capacity` buffers.
    pub(crate) fn for_queue(capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            queue: Some(QueueCounters {
                capacity,
                ..QueueCounters::default()
            }),
            ..Self::default()
        })
    }

    /// Adds output ports to what is reported for this element.
    pub(crate) fn add_pads(&self, pads: impl IntoIterator<Item = Arc<PadCounters>>) {
        self.pads
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .extend(pads);
    }

    /// Starts timing one call into the element. `Eos` is a marker rather
    /// than media, so it is not counted as a buffer — only noted, once the
    /// call returns, if the element took it.
    pub(crate) fn begin(&self, is_eos: bool) -> Call<'_> {
        Call {
            counters: self,
            started: now_ns(),
            is_eos,
            forwarding: false,
        }
    }

    /// Starts timing a [`Queue`](crate::queue::Queue)'s worker handing on
    /// a buffer that already arrived — counted once, when it arrived, and
    /// not again here. A failure is the downstream element's, and counted
    /// there.
    pub(crate) fn begin_forwarding(&self, is_eos: bool) -> Call<'_> {
        Call {
            forwarding: true,
            ..self.begin(is_eos)
        }
    }

    /// A buffer arrived at a queue, whatever then became of it.
    pub(crate) fn arrived(&self) {
        self.buffers_in.fetch_add(1, Ordering::Relaxed);
        self.last_buffer_at.store(now_ns(), Ordering::Relaxed);
    }

    /// A call into the element failed without going through [`Call`].
    pub(crate) fn failed(&self) {
        self.errors.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn queue(&self) -> Option<&QueueCounters> {
        self.queue.as_ref()
    }

    fn read(&self, now: u64) -> Reading {
        let pads: Vec<PadStats> = self
            .pads
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .map(|pad| pad.read(now))
            .collect();
        let last_buffer_at = self.last_buffer_at.load(Ordering::Relaxed);
        // What an element last did, whichever way it did it: a source only
        // ever pushes, a terminal only ever takes.
        let idle_for = std::iter::once(since(last_buffer_at, now))
            .chain(pads.iter().map(|pad| pad.idle_for))
            .flatten()
            .min();
        Reading {
            buffers_in: self.buffers_in.load(Ordering::Relaxed),
            busy: Duration::from_nanos(self.busy_ns.load(Ordering::Relaxed)),
            idle_for,
            errors: self.errors.load(Ordering::Relaxed),
            eos: self.eos.load(Ordering::Relaxed),
            pads,
            queue: self.queue.as_ref().map(QueueCounters::read),
        }
    }
}

/// One call into an element, being timed. Ends with [`Call::end`].
pub(crate) struct Call<'a> {
    counters: &'a ElementCounters,
    started: u64,
    is_eos: bool,
    forwarding: bool,
}

impl Call<'_> {
    /// Records the call as having returned `ok` or not.
    pub(crate) fn end(self, ok: bool) {
        let counters = self.counters;
        let now = now_ns();
        counters
            .busy_ns
            .fetch_add(now.saturating_sub(self.started), Ordering::Relaxed);
        if self.is_eos {
            if ok {
                counters.eos.store(true, Ordering::Relaxed);
            }
        } else if !self.forwarding {
            counters.buffers_in.fetch_add(1, Ordering::Relaxed);
            counters.last_buffer_at.store(now, Ordering::Relaxed);
        }
        if !ok && !self.forwarding {
            counters.errors.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// One output port's running totals. Every [`SrcPad`](crate::pad::SrcPad)
/// has one, whether or not anything ever reads it.
pub(crate) struct PadCounters {
    name: Arc<str>,
    buffers: AtomicU64,
    push_errors: AtomicU64,
    last_push_at: AtomicU64,
}

impl PadCounters {
    pub(crate) fn new(name: &str) -> Arc<Self> {
        Arc::new(Self {
            name: name.into(),
            buffers: AtomicU64::new(0),
            push_errors: AtomicU64::new(0),
            last_push_at: AtomicU64::new(0),
        })
    }

    /// Records one buffer handed to whatever the pad is linked to.
    pub(crate) fn pushed(&self, ok: bool) {
        self.buffers.fetch_add(1, Ordering::Relaxed);
        self.last_push_at.store(now_ns(), Ordering::Relaxed);
        if !ok {
            self.push_errors.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn read(&self, now: u64) -> PadStats {
        PadStats {
            name: self.name.clone(),
            buffers: self.buffers.load(Ordering::Relaxed),
            push_errors: self.push_errors.load(Ordering::Relaxed),
            idle_for: since(self.last_push_at.load(Ordering::Relaxed), now),
        }
    }
}

/// What a [`Queue`](crate::queue::Queue) adds: how full it is, what it
/// threw away, and how long its upstream waited on it.
#[derive(Default)]
pub(crate) struct QueueCounters {
    capacity: usize,
    /// A second sending end of the queue's channel, for its length.
    ///
    /// Counting in and out would drift: a `Flush` empties the channel
    /// wholesale. The channel knows its own length exactly, and a sender is
    /// the end that can ask without changing anything — a spare receiver
    /// would keep the channel open after the worker left, and a `send`
    /// that should fail would wait for ever instead. Set once, when the
    /// queue is spawned.
    ///
    /// The worker holds these counters too, so its end of the channel never
    /// sees the sending side close while it runs. It never relied on that:
    /// the queue's own sender lives exactly as long as the `Queue`, whose
    /// drop stops the worker by its flag and joins it first.
    channel: OnceLock<Sender<MediaBuffer>>,
    dropped: AtomicU64,
    blocked_ns: AtomicU64,
}

impl QueueCounters {
    pub(crate) fn watch(&self, channel: Sender<MediaBuffer>) {
        let _ = self.channel.set(channel);
    }

    pub(crate) fn dropped(&self) {
        self.dropped.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn blocked(&self, for_how_long: Duration) {
        self.blocked_ns
            .fetch_add(nanos(for_how_long), Ordering::Relaxed);
    }

    fn read(&self) -> QueueStats {
        QueueStats {
            len: self.channel.get().map_or(0, Sender::len),
            capacity: self.capacity,
            dropped: self.dropped.load(Ordering::Relaxed),
            blocked: Duration::from_nanos(self.blocked_ns.load(Ordering::Relaxed)),
        }
    }
}

struct Reading {
    buffers_in: u64,
    busy: Duration,
    idle_for: Option<Duration>,
    errors: u64,
    eos: bool,
    pads: Vec<PadStats>,
    queue: Option<QueueStats>,
}

/// One element's entry in the registry: who it is, and its counters, held
/// weakly — see the [module](self).
#[derive(Clone)]
pub(crate) struct Registered {
    pub(crate) id: ElementId,
    pub(crate) element_type: ElementType,
    pub(crate) name: Arc<str>,
    pub(crate) branch: Option<BranchId>,
    pub(crate) counters: std::sync::Weak<ElementCounters>,
}

/// Reads what the registry held, outside its lock. `attached` says which
/// of them the graph still has.
pub(crate) fn read(
    registered: Vec<Registered>,
    attached: impl Fn(ElementId) -> bool,
) -> Vec<ElementStats> {
    let now = now_ns();
    registered
        .into_iter()
        .filter_map(|entry| {
            let reading = entry.counters.upgrade()?.read(now);
            Some(ElementStats {
                id: entry.id,
                element_type: entry.element_type,
                name: entry.name,
                branch: entry.branch,
                state: if attached(entry.id) {
                    ElementState::Attached
                } else {
                    ElementState::Finishing
                },
                buffers_in: reading.buffers_in,
                busy: reading.busy,
                idle_for: reading.idle_for,
                errors: reading.errors,
                eos: reading.eos,
                pads: reading.pads,
                queue: reading.queue,
            })
        })
        .collect()
}

/// Every element of one pipeline, read at one moment — see
/// [`Pipeline::stats`](crate::pipeline::Pipeline::stats).
#[derive(Debug, Clone)]
pub struct PipelineStats {
    /// The graph revision this was read at: the same number
    /// [`GraphSnapshot::revision`](crate::graph::GraphSnapshot::revision)
    /// carries, so the two can be matched. It moves when a branch is
    /// attached or detached, not when a finishing branch finally goes.
    pub revision: u64,
    /// Whether the pipeline is paused — which is the first thing to know
    /// about an element that has stopped delivering.
    pub paused: bool,
    /// One entry per element that exists, attached or finishing.
    pub elements: Vec<ElementStats>,
}

/// Whether an element is in the graph, or out of it and still at work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementState {
    /// In the graph.
    Attached,
    /// Removed from the graph with its branch, and still alive: a branch
    /// finished with [`TeeHandle::finish_branch`](crate::elements::TeeHandle::finish_branch)
    /// draining its `Eos`. Reported until the element is dropped.
    Finishing,
}

/// One element's totals, read at one moment.
#[derive(Debug, Clone)]
pub struct ElementStats {
    /// The element's stable identity — the same one the graph uses. It is
    /// never given to another element, so it is what two readings are
    /// matched by: a branch attached again under the same name is new.
    pub id: ElementId,
    /// Its type.
    pub element_type: ElementType,
    /// The name its caller gave it.
    pub name: Arc<str>,
    /// The branch it arrived with, or `None` for one of the pipeline's
    /// sources and the stages wired with them.
    pub branch: Option<BranchId>,
    /// In the graph, or finishing outside it.
    pub state: ElementState,
    /// Buffers handed to it. `Eos` is not one.
    pub buffers_in: u64,
    /// Time spent inside its `consume` — including every stage it feeds
    /// on the same thread; see the [module](self). For a queue, what its
    /// worker spent handing buffers on, which is how busy that thread is.
    pub busy: Duration,
    /// How long since it last took or gave a buffer, or `None` if it never
    /// has.
    pub idle_for: Option<Duration>,
    /// Calls into it that failed.
    pub errors: u64,
    /// Whether it has taken an `Eos`.
    pub eos: bool,
    /// Its output ports.
    pub pads: Vec<PadStats>,
    /// Set for a [`Queue`](crate::queue::Queue), and only for one.
    pub queue: Option<QueueStats>,
}

/// One output port's totals.
#[derive(Debug, Clone)]
pub struct PadStats {
    /// The port's name.
    pub name: Arc<str>,
    /// Buffers pushed through it, whether or not anything was linked.
    pub buffers: u64,
    /// Pushes that failed downstream.
    pub push_errors: u64,
    /// How long since the last push, or `None` if there never was one.
    pub idle_for: Option<Duration>,
}

/// A queue's own state.
#[derive(Debug, Clone, Copy)]
pub struct QueueStats {
    /// Buffers waiting for its worker.
    pub len: usize,
    /// The most that may wait.
    pub capacity: usize,
    /// Buffers thrown away because it was full — under
    /// [`OverflowPolicy::DropNewest`](crate::queue::OverflowPolicy::DropNewest).
    pub dropped: u64,
    /// Time its upstream spent waiting for room — under
    /// [`OverflowPolicy::Block`](crate::queue::OverflowPolicy::Block).
    pub blocked: Duration,
}
