//! Pause, Resume, Stop, Flush, Seek, and Finish — and the channel they travel
//! through.
//!
//! Control follows the same pad-to-pad path as data but on a dedicated
//! channel, because unlike [`Eos`](crate::buffer::MediaBuffer::Eos) it has to
//! reach elements mid-stream and, at a [`Queue`](crate::queue::Queue), jump
//! ahead of whatever is already backed up instead of queueing behind it.
//!
//! A [`SourceElement`](crate::element::SourceElement) loop stays responsive by
//! calling [`drain_control`] every iteration. The returned [`ControlOutcome`]
//! is not only a "should I stop" flag: a source that schedules against the
//! wall clock must add `paused_for` back into its own timing, or resuming will
//! look like a burst of catch-up work owed all at once.

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Condvar, Mutex},
    time::{Duration, Instant},
};

use thiserror::Error;

use crate::pp_log::pp_trace;
use crossbeam_channel::{Receiver, Sender, unbounded};

use crate::{
    bus::{Bus, BusEvent},
    element::{ElementType, Filter, SourceElement},
    error::Result,
    graph::{ElementId, NodeInfo},
    pad::SrcPad,
    playback_state::PlaybackState,
};

/// A command that can be sent down a running [`crate::pipeline::Pipeline`]
/// — travels the same pad-to-pad path `MediaBuffer` does (see
/// [`crate::element::Sink::control`]), but through a dedicated channel
/// instead of riding along as data: unlike `Eos`, it has to be able to
/// reach every element even mid-stream, and (for `Queue`) jump ahead of
/// whatever data is already backed up rather than wait in line behind it.
#[derive(Debug, Clone)]
pub enum ControlMsg {
    /// Freeze in place. Every [`crate::queue::Queue`] downstream stops
    /// pulling from its data channel until `Resume`/`Stop` — which also
    /// backpressures anything feeding it, since a full queue blocks the
    /// sender. Pairs with [`crate::clock::Clock::pause`], which
    /// [`crate::pipeline::Pipeline::pause`] calls at the same time so
    /// paced elements don't see a jump once resumed.
    Pause,
    /// Undoes `Pause`.
    Resume,
    /// Abandon immediately rather than draining to a natural `Eos` —
    /// whatever's in flight is dropped, not flushed. The pipeline isn't
    /// reusable afterward; build a new one for the next run.
    Stop,
    /// Discard buffered data and reset state that belongs to the current
    /// timeline without changing the source position. Pipelines issue this
    /// before `Seek`; keeping the two controls separate lets paused preroll
    /// and future timeline operations compose the same flush boundary.
    Flush,
    /// Temporarily lets paused source and queue workers process data until
    /// every expected terminal reports its first new-timeline sample.
    Preroll(Arc<PrerollContext>),
    /// Jump to an absolute position from the start of the media.
    /// The source repositions via [`crate::element::SourceElement::seek`]
    /// before this is forwarded downstream. Timeline state is discarded by
    /// the preceding `Flush`, not implicitly by this message.
    Seek(Duration),
}

impl PartialEq for ControlMsg {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Pause, Self::Pause)
            | (Self::Resume, Self::Resume)
            | (Self::Stop, Self::Stop)
            | (Self::Flush, Self::Flush) => true,
            (Self::Seek(left), Self::Seek(right)) => left == right,
            (Self::Preroll(left), Self::Preroll(right)) => Arc::ptr_eq(left, right),
            _ => false,
        }
    }
}

impl Eq for ControlMsg {}

/// Why one element refused a pipeline-wide seek check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeekRejectReason {
    /// The source follows an external timeline that cannot be repositioned.
    LiveSource,
    /// The source cannot reposition its input timeline.
    SourceNotSeekable,
    /// A downstream element cannot preserve its contract across a seek.
    ElementNotSeekable,
    /// The source cannot read its media backwards — it is not a
    /// [`crate::element::ReversibleSource`].
    SourceNotReversible,
    /// A downstream element turns a picture's packets into pictures and
    /// cannot hand them on backwards — it is not a
    /// [`crate::element::ReversibleSink`].
    ElementNotReversible,
}

/// One element that prevents a pipeline-wide seek.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeekRejection {
    /// Kind of the element that rejected the check.
    pub element_type: ElementType,
    /// Caller-selected instance name of the rejecting element.
    pub name: Arc<str>,
    /// Capability that made this element incompatible with seeking.
    pub reason: SeekRejectReason,
}

/// A pipeline-wide seek check found at least one incompatible element.
#[derive(Debug, Error)]
#[error("pipeline seek rejected by {rejections:?}")]
pub struct SeekError {
    rejections: Vec<SeekRejection>,
}

impl SeekError {
    /// `Ok` where nothing refused, and otherwise an error naming what did.
    pub(crate) fn from_rejections(rejections: Vec<SeekRejection>) -> std::result::Result<(), Self> {
        if rejections.is_empty() {
            Ok(())
        } else {
            Err(Self { rejections })
        }
    }

    /// Elements that rejected the attempted pipeline seek.
    pub fn rejections(&self) -> &[SeekRejection] {
        &self.rejections
    }
}

#[derive(Debug, Default)]
struct PrerollState {
    ready: HashSet<ElementId>,
    /// Samples each expected terminal has taken so far, where a preroll
    /// asks more than one of each — see [`PrerollContext::samples`].
    taken: HashMap<ElementId, usize>,
    cancelled: bool,
}

/// Shared completion state for one preroll pass.
#[derive(Debug)]
pub struct PrerollContext {
    expected: HashSet<ElementId>,
    /// What each expected terminal is called, for a timeout to name — given
    /// by [`crate::pipeline::Pipeline::seek`] from its graph.
    labels: HashMap<ElementId, NodeInfo>,
    target: Option<Duration>,
    /// How many samples each expected terminal takes before it is ready.
    samples: usize,
    /// A frame step's, which lets the decoders decode on as they were
    /// rather than select a sample — see [`Self::for_step`].
    step: bool,
    /// The terminals that drop what reaches them until the preroll ends —
    /// see [`Self::silencing`].
    silenced: HashSet<ElementId>,
    state: Mutex<PrerollState>,
    changed: Condvar,
}

impl PrerollContext {
    /// Creates a preroll that completes once every supplied terminal ID is
    /// marked ready (or EOS-equivalent).
    pub fn new(terminals: impl IntoIterator<Item = ElementId>) -> Self {
        Self {
            expected: terminals.into_iter().collect(),
            labels: HashMap::new(),
            target: None,
            samples: 1,
            step: false,
            silenced: HashSet::new(),
            state: Mutex::new(PrerollState::default()),
            changed: Condvar::new(),
        }
    }

    /// Creates a seek preroll whose decoded timing gates should discard
    /// samples before `target` while decoding forward from the landed
    /// keyframe.
    pub fn for_seek(terminals: impl IntoIterator<Item = ElementId>, target: Duration) -> Self {
        Self {
            expected: terminals.into_iter().collect(),
            labels: HashMap::new(),
            target: Some(target),
            samples: 1,
            step: false,
            silenced: HashSet::new(),
            state: Mutex::new(PrerollState::default()),
            changed: Condvar::new(),
        }
    }

    /// A frame step's preroll: each of `terminals` takes `frames` more
    /// pictures, from wherever its decoder is, and holds.
    ///
    /// The decoders are left to decode on as they were: no sample is being
    /// selected, and a picture decoded past the last one asked for waits in
    /// its queue to be the next step's first rather than being dropped.
    pub(crate) fn for_step(terminals: impl IntoIterator<Item = ElementId>, frames: usize) -> Self {
        Self {
            samples: frames.max(1),
            step: true,
            ..Self::new(terminals)
        }
    }

    /// This preroll, with `terminals` dropping what reaches them until it is
    /// over. A step moves the picture, and what comes meanwhile for the
    /// terminals that show none — the sound — is not played; playing on
    /// after a step puts it back in line.
    ///
    /// Named rather than read as every terminal not expected: a `Tee` is
    /// traced as the terminal of the chain in front of it, and is none of
    /// the graph's, so it is never expected — and dropping at it silenced
    /// the pictures behind it too.
    pub(crate) fn silencing(mut self, terminals: impl IntoIterator<Item = ElementId>) -> Self {
        self.silenced = terminals.into_iter().collect();
        self
    }

    /// How many samples each expected terminal takes before it is ready:
    /// one for a seek's preroll, the frames asked for in a step's.
    pub fn samples(&self) -> usize {
        self.samples
    }

    /// Whether this is a frame step's preroll — see [`Self::for_step`].
    pub(crate) fn is_step(&self) -> bool {
        self.step
    }

    /// Whether `terminal` drops what reaches it while this preroll lasts
    /// — see [`Self::silencing`].
    pub(crate) fn drops_for(&self, terminal: ElementId) -> bool {
        self.silenced.contains(&terminal)
    }

    /// Names the expected terminals, so a timeout says which ones it waited
    /// on rather than only their ids.
    pub(crate) fn labelled(mut self, nodes: impl IntoIterator<Item = NodeInfo>) -> Self {
        self.labels = nodes.into_iter().map(|node| (node.id, node)).collect();
        self
    }

    /// Exact requested position for seek preroll, or `None` for ordinary
    /// first-sample preroll.
    pub fn target(&self) -> Option<Duration> {
        self.target
    }

    /// Counts one valid sample an expected terminal has taken; it is ready
    /// once it has taken all [`Self::samples`] asks of it.
    pub fn mark_ready(&self, terminal: ElementId) {
        if !self.expected.contains(&terminal) {
            return;
        }
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.ready.contains(&terminal) {
            return;
        }
        let taken = state.taken.entry(terminal).or_insert(0);
        *taken += 1;
        if *taken >= self.samples {
            state.ready.insert(terminal);
            self.changed.notify_all();
        }
    }

    /// Takes `terminal` as done however many samples it has taken.
    fn mark_done(&self, terminal: ElementId) {
        if !self.expected.contains(&terminal) {
            return;
        }
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.ready.insert(terminal) {
            self.changed.notify_all();
        }
    }

    /// EOS means this terminal cannot produce a sample and therefore must not
    /// leave the whole preroll waiting forever — nor a step asking for
    /// more pictures than are left.
    pub fn mark_eos(&self, terminal: ElementId) {
        self.mark_done(terminal);
    }

    /// Stops expecting a terminal that has left the graph.
    ///
    /// The expected set is fixed when the seek starts, but the topology is
    /// not: detaching a `Tee` branch mid-seek removes its terminal without
    /// removing the obligation to hear from it, and the wait would run to its
    /// timeout for a sample nobody is left to produce. Like EOS, this is
    /// "cannot produce one", not "produced one".
    pub fn mark_departed(&self, terminal: ElementId) {
        self.mark_done(terminal);
    }

    /// Whether this one terminal has already taken its preroll sample.
    ///
    /// A terminal stops accepting as soon as *it* is ready, not when the whole
    /// preroll is. Waiting for the others would let a branch that reached the
    /// target first keep consuming for as long as the slowest branch takes —
    /// which is how the two streams end up at different positions when preroll
    /// finally completes.
    pub fn is_ready(&self, terminal: ElementId) -> bool {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        !state.cancelled && state.ready.contains(&terminal)
    }

    /// Whether every terminal in one downstream branch has completed.
    pub(crate) fn are_ready(&self, terminals: &[ElementId]) -> bool {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        !state.cancelled && terminals.iter().all(|id| state.ready.contains(id))
    }

    /// Returns whether every expected terminal has completed this preroll.
    pub fn is_complete(&self) -> bool {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        !state.cancelled && self.expected.is_subset(&state.ready)
    }

    /// Cancels a pending wait, used by stop and failed seek recovery.
    pub fn cancel(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.cancelled = true;
        self.changed.notify_all();
    }

    /// Waits until every expected terminal is ready, cancellation is
    /// requested, or `timeout` expires.
    pub fn wait(&self, timeout: Duration) -> std::result::Result<(), PrerollError> {
        let deadline = Instant::now() + timeout;
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        loop {
            if state.cancelled {
                return Err(PrerollError::Cancelled);
            }
            let mut pending: Vec<_> = self.expected.difference(&state.ready).copied().collect();
            pending.sort_unstable();
            if pending.is_empty() {
                return Ok(());
            }
            let now = Instant::now();
            if now >= deadline {
                let pending = pending
                    .into_iter()
                    .map(|id| {
                        self.labels.get(&id).cloned().unwrap_or_else(|| NodeInfo {
                            id,
                            element_type: crate::element::ElementType::Other,
                            name: format!("#{id}").into(),
                        })
                    })
                    .collect();
                return Err(PrerollError::TimedOut { pending });
            }
            let remaining = deadline.saturating_duration_since(now);
            let (next, _) = self
                .changed
                .wait_timeout(state, remaining)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state = next;
        }
    }
}

/// Failure while waiting for terminal preroll completion.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum PrerollError {
    /// Stop or recovery cancelled the in-flight preroll.
    #[error("preroll was cancelled")]
    Cancelled,
    /// At least one terminal did not receive a first sample in time — each
    /// named, with its type and graph id, as the pipeline's graph has it.
    #[error("preroll timed out waiting on {}", name_terminals(pending))]
    TimedOut {
        /// The terminals that had not taken their first sample.
        pending: Vec<NodeInfo>,
    },
}

/// `video-terminal (Other #4), speakers (WasapiRenderer #9)`.
fn name_terminals(pending: &[NodeInfo]) -> String {
    pending
        .iter()
        .map(|node| format!("{} ({:?} #{})", node.name, node.element_type, node.id))
        .collect::<Vec<_>>()
        .join(", ")
}

/// A request carried by a control channel. Ordinary controls cascade through
/// the graph immediately; `Finish` is source-only because graceful completion
/// must enter the graph as an ordered [`crate::buffer::MediaBuffer::Eos`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RequestKind {
    Control(ControlMsg),
    Finish,
}

/// One in-flight control request: the message plus a rendezvous channel
/// the receiver acks once it (and everything it cascaded into downstream)
/// has finished handling it — this is what makes
/// [`ControlSender::send`] synchronous. Fields are `pub(crate)` so
/// [`crate::queue::Queue`]'s worker loop can match on one directly out of
/// a `crossbeam_channel::select!` arm (which needs the raw `Receiver`,
/// not the [`ControlReceiver::try_recv`]/[`ControlReceiver::recv`]
/// wrappers used everywhere else).
pub(crate) struct Request {
    pub(crate) kind: RequestKind,
    pub(crate) ack: Sender<()>,
}

/// The sending half of a control channel — cloneable, cheap, `Send +
/// Sync`. [`crate::pipeline::Pipeline`] holds one to reach its source;
/// [`crate::queue::Queue`] holds one internally to reach its worker
/// thread across the thread boundary it owns.
#[derive(Clone)]
pub struct ControlSender {
    tx: Sender<Request>,
    /// The playback state the receiving end reads, which this moves on
    /// with each message it sends where no pipeline does — see
    /// [`channel`].
    state: Arc<PlaybackState>,
    writes_state: bool,
}

/// The receiving half — not `Clone` in spirit (only one thing should be
/// driving a given control channel at a time) but crossbeam's
/// `Receiver<T>` is a cheap shared handle under the hood, which is
/// exactly what [`crate::pipeline::Pipeline::run`] needs: it clones this
/// into a fresh worker thread on every call.
#[derive(Clone)]
pub struct ControlReceiver {
    pub(crate) rx: Receiver<Request>,
    state: Arc<PlaybackState>,
}

/// Creates a control channel.
///
/// The channel is unbounded, because a control request must never be blocked by
/// backpressure on the data path — that is the whole reason control does not
/// travel as data. [`Pipeline`](crate::pipeline::Pipeline) creates one per
/// source; [`Queue`](crate::queue::Queue) creates one to reach its own worker.
///
/// One made here stands on its own: with no pipeline to say whether a
/// paused receiver may go on, the sending half says so, moving the state
/// the receiver reads on with each message before it sends it. A
/// pipeline's own channels share the pipeline's state instead, which only
/// the pipeline moves.
pub fn channel() -> (ControlSender, ControlReceiver) {
    channel_sharing(PlaybackState::new(), true)
}

/// A channel whose receiving end reads `state`, which whoever owns it moves
/// on — a pipeline, for the channels to its sources and queues.
pub(crate) fn channel_in(state: &Arc<PlaybackState>) -> (ControlSender, ControlReceiver) {
    channel_sharing(Arc::clone(state), false)
}

fn channel_sharing(
    state: Arc<PlaybackState>,
    writes_state: bool,
) -> (ControlSender, ControlReceiver) {
    let (tx, rx) = unbounded();
    (
        ControlSender {
            tx,
            state: Arc::clone(&state),
            writes_state,
        },
        ControlReceiver { rx, state },
    )
}

impl ControlSender {
    /// Sends `msg` and blocks until the receiver — and, transitively,
    /// everything downstream of it — has finished handling it. A no-op
    /// (returns immediately) if nothing is on the other end to receive it
    /// (e.g. the pipeline already finished).
    pub fn send(&self, msg: ControlMsg) {
        self.moves_state(&msg);
        self.send_request(RequestKind::Control(msg));
    }

    /// Where no pipeline says what `msg` changes, says it — before the
    /// message goes, as a pipeline does. See [`channel`].
    fn moves_state(&self, msg: &ControlMsg) {
        if self.writes_state {
            self.state.observe(msg);
        }
    }

    /// Queues `msg` without waiting for it to be handled, and returns what
    /// its acknowledgement will arrive on — `None` if nothing is on the other
    /// end. For a message that has to be first in line before the receiver's
    /// thread exists: [`crate::pipeline::Pipeline::run`] starting paused.
    pub(crate) fn enqueue(&self, msg: ControlMsg) -> Option<Receiver<()>> {
        self.moves_state(&msg);
        let (ack_tx, ack_rx) = crossbeam_channel::bounded(0);
        self.tx
            .send(Request {
                kind: RequestKind::Control(msg),
                ack: ack_tx,
            })
            .ok()?;
        Some(ack_rx)
    }

    /// [`Self::enqueue`] for source-originated EOS, without exposing `Finish`
    /// as a downstream [`ControlMsg`]. Used only by
    /// [`crate::pipeline::Pipeline::finish`].
    pub(crate) fn enqueue_finish(&self) -> Option<Receiver<()>> {
        let (ack_tx, ack_rx) = crossbeam_channel::bounded(0);
        self.tx
            .send(Request {
                kind: RequestKind::Finish,
                ack: ack_tx,
            })
            .ok()?;
        Some(ack_rx)
    }

    fn send_request(&self, kind: RequestKind) {
        let (ack_tx, ack_rx) = crossbeam_channel::bounded(0);
        if self.tx.send(Request { kind, ack: ack_tx }).is_ok() {
            let _ = ack_rx.recv();
        }
    }
}

impl ControlReceiver {
    pub(crate) fn try_recv(&self) -> Option<(RequestKind, Sender<()>)> {
        self.rx.try_recv().ok().map(|r| (r.kind, r.ack))
    }

    pub(crate) fn recv(&self) -> Option<(RequestKind, Sender<()>)> {
        self.rx.recv().ok().map(|r| (r.kind, r.ack))
    }

    /// The playback state this end reads — a pipeline's, or the channel's
    /// own; see [`channel`].
    pub(crate) fn state(&self) -> &Arc<PlaybackState> {
        &self.state
    }
}

/// What draining pending source requests actually did — whether `Stop` or
/// source-only `Finish` ended it, and how long (if any) was spent frozen
/// inside a `Pause`/`Resume` pair. A source built on wall-clock scheduling (an elapsed-time
/// budget like [`crate::elements::TestAudioSource`]/
/// [`crate::elements::AudioMixer`], or an absolute next-tick deadline like
/// [`crate::elements::TestVideoSource`]/`DxgiCaptureSource`)
/// has to fold `paused_for` back into its own schedule after every
/// [`drain_control`] call — real (`Instant`) time keeps moving during a
/// `Pause`, but the media timeline must not, or `Resume` would look like a
/// burst of catch-up work owed all at once.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ControlOutcome {
    /// `true` if either `Stop` or source-only `Finish` was seen: the caller
    /// should return `Ok(())` immediately. `Stop` abandons without EOS;
    /// `Finish` has already pushed ordered EOS from the source boundary.
    /// Keeping this terminal flag true for both also makes existing custom
    /// source loops honor the new graceful request without continuing to emit
    /// after EOS.
    pub stopped: bool,
    /// Wall-clock time from starting the synchronous downstream `Pause`
    /// cascade through finishing the matching `Resume` (or terminating
    /// `Stop`) cascade during this call — `Duration::ZERO` if no `Pause`
    /// was seen. Still meaningful
    /// even when `stopped` is `true` (the sender simply going away while
    /// paused is treated the same as `Stop`, see `wait_out_pause`), so a
    /// caller that also tracks its own paused-time total can fold this in
    /// unconditionally rather than only on the non-stopped path.
    pub paused_for: Duration,
}

/// Call once per loop iteration in a [`SourceElement::run`] implementation,
/// right before pulling the next unit of work — mirrors how a natural
/// `Eos` is pushed into the source's own pads at the end of that same
/// loop, just for externally-triggered control instead.
///
/// Drains every pending message (see `apply_one` for what "handling
/// one" means, including `Pause`'s blocking wait). Non-blocking if
/// nothing's pending — a [`SourceElement::run`] whose own "next unit of
/// work" can't be waited on via `control`'s own channel (e.g.
/// [`crate::elements::FileDemuxer`]'s blocking file read) calls this once
/// before that blocking step; one that *can* (e.g.
/// [`crate::elements::AppSource`]'s channel receive) selects on both
/// instead, calling `apply_one`/`wait_out_pause` directly so a
/// pending `Stop`/`Finish` is never left waiting behind a slow/absent producer —
/// same reason `WasapiCaptureSource` also drives the
/// raw receiver directly, to bracket the wait with resetting/restarting
/// its capture device rather than leaving it running unread through the
/// whole pause.
///
/// See [`ControlOutcome`] for what the return value means.
pub fn drain_control<S: SourceElement>(
    control: &ControlReceiver,
    source: &mut S,
    bus: &Bus,
) -> Result<ControlOutcome> {
    let mut paused_for = Duration::ZERO;
    while let Some((request, ack)) = control.try_recv() {
        let outcome = handle_request(control, source, bus, request, ack)?;
        paused_for += outcome.paused_for;
        if outcome.stopped {
            return Ok(ControlOutcome {
                stopped: true,
                paused_for,
            });
        }
    }
    Ok(ControlOutcome {
        stopped: false,
        paused_for,
    })
}

/// Handles one request a source has taken off its control channel — the
/// one way every source does, whether it drains the channel between buffers
/// ([`drain_control`]) or selects on it beside its data: `Finish` ends the
/// stream in order, `Pause` pauses the source until it is resumed or stopped
/// (see [`SourceElement::pausing`] and [`SourceElement::resuming`]), and
/// anything else is applied and passed on.
///
/// One way, because a source that handled a request its own way got it
/// wrong: `FileDemuxer`, waiting at the end of its file, passed a `Pause` on
/// without pausing, read on into the paused queue behind it, and left a
/// seek waiting on it for good (208af56); three capture sources each kept a
/// copy of the pause wait to stop and restart their device around it.
pub(crate) fn handle_request<S: SourceElement>(
    control: &ControlReceiver,
    source: &mut S,
    bus: &Bus,
    request: RequestKind,
    ack: Sender<()>,
) -> Result<ControlOutcome> {
    let backwards = control.state().backwards();
    let RequestKind::Control(msg) = request else {
        apply_finish(source, bus, &ack, backwards);
        return Ok(ControlOutcome {
            stopped: true,
            paused_for: Duration::ZERO,
        });
    };
    if msg != ControlMsg::Pause {
        return Ok(ControlOutcome {
            stopped: apply_one(source, bus, &msg, &ack, backwards)?,
            paused_for: Duration::ZERO,
        });
    }
    // Measured from before the source stops and the `Pause` cascades: both
    // may take a while — a device to stop, a busy queue to take the pause —
    // and the source produces nothing meanwhile, so it belongs to the frozen
    // interval as much as the wait for `Resume` does.
    let pause_start = Instant::now();
    source.pausing()?;
    apply_one(source, bus, &msg, &ack, backwards)?;
    let stopped = wait_out_pause(control, source, bus)?;
    Ok(ControlOutcome {
        stopped,
        paused_for: pause_start.elapsed(),
    })
}

/// Applies one source-only graceful completion request. Unlike
/// [`apply_one`], this never calls `Sink::control`: EOS has to sit behind every
/// already-produced buffer in each data path so queues and stateful elements
/// drain in order. Playing backwards, a [`crate::element::ReversibleSource`] first reads the
/// stretch under way to its end — see [`crate::element::ReversibleSource::finish_stretch`].
pub(crate) fn apply_finish<S: SourceElement>(
    source: &mut S,
    bus: &Bus,
    ack: &Sender<()>,
    backwards: bool,
) {
    pp_trace!(
        pp_log: source.pp_log(),
        "event=finish phase=received"
    );
    let pp_log = source.pp_log().clone();
    let element_type = source.element_type();
    let name = source.name();
    let finished = match source.as_reversible() {
        Some(source) if backwards => source.finish_stretch(bus),
        _ => Ok(()),
    };
    if let Err(error) = finished {
        bus.post(
            &pp_log,
            BusEvent::Error {
                element_type,
                name: name.clone(),
                error,
            },
        );
    }
    for pad in source.src_pads() {
        if let Err(error) = pad.push_eos(&pp_log) {
            bus.post(
                &pp_log,
                BusEvent::Error {
                    element_type,
                    name: name.clone(),
                    error,
                },
            );
        }
    }
    let _ = ack.send(());
    pp_trace!(
        pp_log: source.pp_log(),
        "event=finish phase=completed outcome=ok"
    );
}

/// Applies one already-received control message to `source`: repositions
/// it first on `Seek` (see [`apply_seek`]), then forwards `msg` to every
/// one of `source`'s pads (so it cascades through the graph exactly like
/// a data buffer would), then acks. Returns `true` for `Stop` — same
/// meaning as [`drain_control`]'s own return.
pub(crate) fn apply_one<S: SourceElement>(
    source: &mut S,
    bus: &Bus,
    msg: &ControlMsg,
    ack: &Sender<()>,
    backwards: bool,
) -> Result<bool> {
    let is_stop = apply_one_unacked(source, bus, msg, backwards)?;
    let _ = ack.send(());
    Ok(is_stop)
}

/// The forwarding half of [`apply_one`], split out for a source that must
/// finish source-local state changes before the synchronous request is
/// acknowledged. [`crate::elements::WasapiCaptureSource`] uses this for
/// `Resume`: downstream is resumed first, then its capture device is
/// restarted, and only then may the caller observe the request as done.
pub(crate) fn apply_one_unacked<S: SourceElement>(
    source: &mut S,
    bus: &Bus,
    msg: &ControlMsg,
    backwards: bool,
) -> Result<bool> {
    pp_trace!(
        pp_log: source.pp_log(),
        "event=control control={msg:?} phase=received"
    );
    let result: Result<bool> = (|| {
        source.on_control(msg);
        let sought = apply_seek(source, bus, msg, backwards);
        if matches!(msg, ControlMsg::Seek(_)) {
            // What this thread reads from now on is the new position's — see
            // `crate::timeline`. Even where the seek failed: the pipeline has
            // left the old timeline all the same, and whatever this source
            // goes on to read is all it will get.
            crate::timeline::follow();
        }
        sought?;
        forward(source.src_pads(), msg)?;
        Ok(*msg == ControlMsg::Stop)
    })();
    match &result {
        Ok(_) => pp_trace!(
            pp_log: source.pp_log(),
            "event=control control={msg:?} phase=completed outcome=ok"
        ),
        Err(error) => pp_trace!(
            pp_log: source.pp_log(),
            "event=control control={msg:?} phase=completed outcome=error error={error}"
        ),
    }
    result
}

/// Blocks on `control` alone — not whatever `source.run()` itself is
/// otherwise waiting on — until `Resume`, `Stop`, or `Finish`, applying (and
/// acking) every request seen in between. Returns `true` if `Stop`/`Finish`
/// ended it (including the sender simply going away, treated the same as
/// `Stop`); `false` once `Resume` or `Preroll` arrives.
pub(crate) fn wait_out_pause<S: SourceElement>(
    control: &ControlReceiver,
    source: &mut S,
    bus: &Bus,
) -> Result<bool> {
    loop {
        let Some((request, ack)) = control.recv() else {
            return Ok(true); // sender gone — treat like Stop
        };
        let backwards = control.state().backwards();
        let RequestKind::Control(msg) = request else {
            apply_finish(source, bus, &ack, backwards);
            return Ok(true);
        };
        if !control.state().holds() {
            // Playback has moved on — to playing, or to a preroll — before
            // this message came to say so; see `crate::playback_state`.
            // Downstream goes on first, then the source itself, and only
            // then is the request done — see `SourceElement::resuming`.
            apply_one_unacked(source, bus, &msg, backwards)?;
            source.resuming()?;
            let _ = ack.send(());
            return Ok(false);
        }
        if apply_one(source, bus, &msg, &ack, backwards)? {
            return Ok(true);
        }
        // Another Pause while already paused: already forwarded above
        // (harmless no-op downstream), just keep waiting.
    }
}

/// `Seek`'s source-specific half of `drain_control` — repositions
/// `source` (see [`SourceElement::seek`]) and reports where it actually
/// landed via [`BusEvent::Seeked`], since that can differ from what was
/// requested. No-op for every other [`ControlMsg`].
/// Hands `msg` to `filter` and then on through each of its pads — what a
/// graph does for every filter in it, for code that drives one by hand: a
/// bin of your own passing control to the elements it holds, a test.
///
/// The filter reacts first ([`Sink::control`](crate::element::Sink::control)), so whatever it holds
/// already reflects the message by the time the elements after it see it.
/// The message goes on even where that reaction failed, and through every
/// pad even where one of them failed: a `Pause` or `Stop` stopped at the
/// first failure would leave the rest of the graph running. What comes back
/// is the first failure.
pub fn deliver<F: Filter + ?Sized>(filter: &mut F, msg: &ControlMsg) -> Result<()> {
    let reacted = filter.control(msg);
    let forwarded = forward(filter.src_pads(), msg);
    reacted.and(forwarded)
}

/// Passes `msg` through every one of `pads`, a failure on one keeping it
/// from none of the others, and answers the first failure.
pub(crate) fn forward(pads: &mut [SrcPad], msg: &ControlMsg) -> Result<()> {
    let mut first = Ok(());
    for pad in pads {
        let outcome = pad.control(msg);
        if first.is_ok() {
            first = outcome;
        }
    }
    first
}

fn apply_seek<S: SourceElement>(
    source: &mut S,
    bus: &Bus,
    msg: &ControlMsg,
    backwards: bool,
) -> Result<()> {
    if let ControlMsg::Seek(target) = msg {
        let landed = if backwards {
            let (element_type, name) = (source.element_type(), source.name());
            match source.as_reversible() {
                Some(source) => source.seek_backwards(*target)?,
                // Refused before a pipeline turns — see
                // `Pipeline::check_reverse` — so only a source driven by hand
                // gets here.
                None => {
                    return Err(SeekError {
                        rejections: vec![SeekRejection {
                            element_type,
                            name,
                            reason: SeekRejectReason::SourceNotReversible,
                        }],
                    }
                    .into());
                }
            }
        } else {
            source.seek(*target)?
        };
        bus.post(
            source.pp_log(),
            BusEvent::Seeked {
                element_type: source.element_type(),
                name: source.name(),
                requested: *target,
                landed,
            },
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, thread};

    use crate::pp_log::PpLog;

    use super::*;
    use crate::{
        buffer::MediaBuffer,
        element::{Element, ElementType, Sink, Source, element_pp_log},
        pad::SrcPad,
    };

    /// A `SourceElement` with no real I/O — just enough surface for
    /// `drain_control`/`wait_out_pause` to drive, since this module's own
    /// logic doesn't care what the source actually produces.
    struct DummySource {
        pp_log: PpLog,
        pad: SrcPad,
        flushes: usize,
        /// Whether it is a [`crate::element::ReversibleSource`].
        reversible: bool,
        /// Each seek it was asked for, and whether backwards.
        sought: Vec<(Duration, bool)>,
    }

    impl DummySource {
        fn new() -> Self {
            Self {
                flushes: 0,
                reversible: false,
                sought: Vec::new(),
                pp_log: element_pp_log(ElementType::Other, "dummy", None),
                pad: SrcPad::new("dummy_src"),
            }
        }
    }

    impl Element for DummySource {
        fn name(&self) -> Arc<str> {
            "dummy".into()
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

    impl Source for DummySource {
        fn src_pads(&mut self) -> &mut [SrcPad] {
            std::slice::from_mut(&mut self.pad)
        }
    }

    impl SourceElement for DummySource {
        fn is_live(&self) -> bool {
            false
        }

        fn is_seekable(&self) -> bool {
            false
        }

        fn run(&mut self, _control: &ControlReceiver, _bus: &Bus) -> Result<()> {
            unreachable!("not exercised by these tests")
        }

        fn on_control(&mut self, msg: &ControlMsg) {
            if *msg == ControlMsg::Flush {
                self.flushes += 1;
            }
        }

        fn seek(&mut self, target: Duration) -> Result<Duration> {
            self.sought.push((target, false));
            Ok(target)
        }

        fn as_reversible(&mut self) -> Option<&mut dyn crate::element::ReversibleSource> {
            if self.reversible { Some(self) } else { None }
        }
    }

    impl crate::element::ReversibleSource for DummySource {
        fn seek_backwards(&mut self, target: Duration) -> Result<Duration> {
            self.sought.push((target, true));
            Ok(target)
        }

        fn finish_stretch(&mut self, _bus: &Bus) -> Result<()> {
            Ok(())
        }
    }

    /// Playing backwards, a seek goes to the source's `seek_backwards`, and
    /// forwards to its `seek`; a source that cannot read backwards is told
    /// so with a typed error and left where it was.
    #[test]
    fn a_seek_backwards_goes_to_a_reversible_source_and_no_other() {
        let (bus, _bus_rx) = Bus::new();
        let seek = |at| ControlMsg::Seek(Duration::from_secs(at));

        let mut source = DummySource::new();
        source.reversible = true;
        apply_one_unacked(&mut source, &bus, &seek(3), true).expect("backwards");
        apply_one_unacked(&mut source, &bus, &seek(1), false).expect("forwards");
        assert_eq!(
            source.sought,
            [
                (Duration::from_secs(3), true),
                (Duration::from_secs(1), false)
            ]
        );

        let mut source = DummySource::new();
        let refused = apply_one_unacked(&mut source, &bus, &seek(3), true);
        let Err(crate::Error::SeekError(error)) = refused else {
            panic!("refused with a seek error: {refused:?}");
        };
        assert_eq!(
            error.rejections()[0].reason,
            SeekRejectReason::SourceNotReversible
        );
        assert!(source.sought.is_empty(), "and not sought at all");
    }

    /// A source that holds data of its own — `FileDemuxer` parks packets for a
    /// pad that cannot accept one yet — has to discard it on the same boundary
    /// every downstream element does. Nothing else can: the packets exist only
    /// there, so releasing them after the reposition is the one way old media
    /// reaches a decoder that has already reset for the new timeline.
    #[test]
    fn flush_reaches_the_source_itself_and_nothing_else_does() {
        let (bus, _bus_rx) = Bus::new();
        let mut source = DummySource::new();

        for msg in [
            ControlMsg::Pause,
            ControlMsg::Resume,
            ControlMsg::Seek(Duration::from_secs(1)),
        ] {
            apply_one_unacked(&mut source, &bus, &msg, false).expect("control applies");
        }
        assert_eq!(source.flushes, 0, "only Flush may discard source-held data");

        apply_one_unacked(&mut source, &bus, &ControlMsg::Flush, false).expect("flush applies");
        assert_eq!(source.flushes, 1);
    }

    /// Writes down, in one shared order, what a source's hooks and the
    /// element after it see — for checking one against the other.
    type Order = Arc<std::sync::Mutex<Vec<String>>>;

    struct HookedSource {
        pp_log: PpLog,
        pad: SrcPad,
        order: Order,
    }

    impl Element for HookedSource {
        fn name(&self) -> Arc<str> {
            "hooked".into()
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

    impl Source for HookedSource {
        fn src_pads(&mut self) -> &mut [SrcPad] {
            std::slice::from_mut(&mut self.pad)
        }
    }

    impl SourceElement for HookedSource {
        fn is_live(&self) -> bool {
            true
        }

        fn is_seekable(&self) -> bool {
            false
        }

        fn run(&mut self, _control: &ControlReceiver, _bus: &Bus) -> Result<()> {
            unreachable!("driven through drain_control directly")
        }

        fn pausing(&mut self) -> Result<()> {
            self.order.lock().unwrap().push("source stops".into());
            Ok(())
        }

        fn resuming(&mut self) -> Result<()> {
            self.order.lock().unwrap().push("source starts".into());
            Ok(())
        }

        fn seek(&mut self, target: Duration) -> Result<Duration> {
            Ok(target)
        }
    }

    struct OrderSink {
        pp_log: PpLog,
        order: Order,
    }

    impl Element for OrderSink {
        fn name(&self) -> Arc<str> {
            "order".into()
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

    impl Sink for OrderSink {
        fn consume(&mut self, _buf: MediaBuffer) -> Result<()> {
            Ok(())
        }

        fn control(&mut self, msg: &ControlMsg) -> Result<()> {
            self.order
                .lock()
                .unwrap()
                .push(format!("downstream {msg:?}"));
            Ok(())
        }
    }

    /// A source stops before its `Pause` goes downstream, and starts again
    /// only once its `Resume` has gone downstream — and before the caller
    /// hears the resume is done, so nobody sees it half resumed. What three
    /// capture sources each did in a pause loop of their own, now the one
    /// every source goes through.
    #[test]
    fn a_source_stops_before_its_pause_goes_on_and_starts_after_its_resume_has() {
        let order: Order = Arc::default();
        let mut source = HookedSource {
            pp_log: element_pp_log(ElementType::Other, "hooked", None),
            pad: SrcPad::new("hooked_src"),
            order: Arc::clone(&order),
        };
        source.pad.link(Box::new(OrderSink {
            pp_log: element_pp_log(ElementType::Other, "order", None),
            order: Arc::clone(&order),
        }));
        let (tx, rx) = channel();
        let (bus, _bus_rx) = Bus::new();
        let worker = thread::spawn(move || {
            while !drain_control(&rx, &mut source, &bus)
                .expect("the hooks succeed")
                .stopped
            {
                thread::sleep(Duration::from_millis(1));
            }
        });

        tx.send(ControlMsg::Pause);
        order.lock().unwrap().push("pause done".into());
        tx.send(ControlMsg::Resume);
        order.lock().unwrap().push("resume done".into());
        tx.send(ControlMsg::Stop);
        worker.join().expect("the source thread");

        assert_eq!(
            *order.lock().unwrap(),
            [
                "source stops",
                "downstream Pause",
                "pause done",
                "downstream Resume",
                "source starts",
                "resume done",
                "downstream Stop",
            ]
        );
    }

    struct SlowPauseSink {
        pp_log: PpLog,
        pause_delay: Duration,
    }

    impl Element for SlowPauseSink {
        fn name(&self) -> Arc<str> {
            "slow-pause".into()
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

    impl Sink for SlowPauseSink {
        fn consume(&mut self, _buf: MediaBuffer) -> Result<()> {
            Ok(())
        }

        fn control(&mut self, msg: &ControlMsg) -> Result<()> {
            if *msg == ControlMsg::Pause {
                thread::sleep(self.pause_delay);
            }
            Ok(())
        }
    }

    /// The edge case called out in `wait_out_pause`'s own docs: the
    /// `ControlSender` going away entirely (e.g. the owning `Pipeline`
    /// dropped) while paused has to be treated the same as an explicit
    /// `Stop`, not left blocking forever on a channel nothing will ever
    /// send on again.
    #[test]
    fn wait_out_pause_treats_a_dropped_sender_as_stop() {
        let (tx, rx) = channel();
        drop(tx);

        let (bus, _bus_rx) = Bus::new();
        let mut source = DummySource::new();

        let stopped = wait_out_pause(&rx, &mut source, &bus)
            .expect("no real seek/push happens on this path, so this can't fail");
        assert!(
            stopped,
            "a dropped ControlSender must be treated the same as an explicit Stop"
        );
    }

    /// A timeout names what it waited on: the terminals still owed a sample,
    /// as the pipeline's graph calls them — a seek that failed on
    /// `ElementId(8)` left the caller to work out which branch that was.
    #[test]
    fn preroll_waits_for_every_terminal_and_names_the_ones_pending() {
        let first = ElementId::for_test(1);
        let second = ElementId::for_test(2);
        let speakers = NodeInfo {
            id: second,
            element_type: ElementType::Other,
            name: "speakers".into(),
        };
        let context = PrerollContext::new([first, second]).labelled([speakers.clone()]);

        context.mark_ready(first);
        let timeout = context.wait(Duration::ZERO);
        assert_eq!(
            timeout,
            Err(PrerollError::TimedOut {
                pending: vec![speakers]
            })
        );
        assert_eq!(
            timeout.unwrap_err().to_string(),
            "preroll timed out waiting on speakers (Other #2)"
        );

        context.mark_eos(second);
        assert_eq!(context.wait(Duration::ZERO), Ok(()));
    }

    /// A context built without names still says which ids it waited on.
    #[test]
    fn an_unnamed_pending_terminal_is_given_by_its_id() {
        let context = PrerollContext::new([ElementId::for_test(7)]);
        assert_eq!(
            context.wait(Duration::ZERO).unwrap_err().to_string(),
            "preroll timed out waiting on #7 (Other #7)"
        );
    }

    #[test]
    fn preroll_wait_can_be_cancelled() {
        let context = PrerollContext::new([ElementId::for_test(1)]);
        context.cancel();
        assert_eq!(
            context.wait(Duration::from_secs(1)),
            Err(PrerollError::Cancelled)
        );
    }

    #[test]
    fn wait_out_pause_returns_when_preroll_arrives() {
        let (tx, rx) = channel();
        let (bus, _bus_rx) = Bus::new();
        let mut source = DummySource::new();
        let context = Arc::new(PrerollContext::new([]));

        let worker = thread::spawn(move || wait_out_pause(&rx, &mut source, &bus));
        tx.send(ControlMsg::Preroll(context));

        assert!(!worker.join().unwrap().unwrap());
    }

    /// `wait_out_pause` blocks past any number of redundant `Pause`s and
    /// only returns (`Ok(false)`, meaning "keep running") once `Resume`
    /// actually arrives.
    #[test]
    fn wait_out_pause_blocks_until_resume_then_returns_false() {
        let (tx, rx) = channel();
        let (bus, _bus_rx) = Bus::new();
        let mut source = DummySource::new();

        let worker = thread::spawn(move || wait_out_pause(&rx, &mut source, &bus));

        // A redundant Pause while already paused: per `wait_out_pause`'s
        // own docs, forwarded (harmless no-op downstream) and then it
        // keeps waiting rather than returning.
        tx.send(ControlMsg::Pause);
        tx.send(ControlMsg::Resume);

        let stopped = worker
            .join()
            .expect("worker must not panic")
            .expect("no real seek/push happens on this path, so this can't fail");
        assert!(
            !stopped,
            "Resume must unblock wait_out_pause with Ok(false)"
        );
    }

    /// `paused_for` starts when the source begins forwarding Pause, not
    /// only after every downstream element has finally acknowledged it.
    /// Otherwise a slow control cascade is miscounted as playable media
    /// time and an elapsed-time source catches that interval up as a burst.
    #[test]
    fn drain_control_counts_the_pause_cascade_as_paused_time() {
        let pause_delay = Duration::from_millis(80);
        let (tx, rx) = channel();
        let controller = thread::spawn(move || {
            tx.send(ControlMsg::Pause);
            tx.send(ControlMsg::Resume);
        });

        let (bus, _bus_rx) = Bus::new();
        let mut source = DummySource::new();
        source.pad.link(Box::new(SlowPauseSink {
            pause_delay,
            pp_log: element_pp_log(ElementType::Other, "slow-pause", None),
        }));

        let outcome = loop {
            let outcome = drain_control(&rx, &mut source, &bus)
                .expect("the synthetic control cascade cannot fail");
            if outcome.paused_for > Duration::ZERO {
                break outcome;
            }
            thread::yield_now();
        };
        controller.join().expect("controller must not panic");

        assert!(!outcome.stopped);
        assert!(
            outcome.paused_for >= Duration::from_millis(60),
            "the {:?} Pause cascade was omitted from paused_for: {:?}",
            pause_delay,
            outcome.paused_for
        );
    }
    /// Notes each control message it is handed, and refuses every one where
    /// told to.
    struct Noting {
        pp_log: PpLog,
        noted: Arc<std::sync::Mutex<Vec<ControlMsg>>>,
        refuses: bool,
    }

    fn noting(refuses: bool) -> (Noting, Arc<std::sync::Mutex<Vec<ControlMsg>>>) {
        let noted = Arc::default();
        let sink = Noting {
            pp_log: element_pp_log(ElementType::Other, "noting", None),
            noted: Arc::clone(&noted),
            refuses,
        };
        (sink, noted)
    }

    impl Element for Noting {
        fn name(&self) -> Arc<str> {
            "noting".into()
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

    impl Sink for Noting {
        fn consume(&mut self, _buf: MediaBuffer) -> Result<()> {
            Ok(())
        }

        fn control(&mut self, msg: &ControlMsg) -> Result<()> {
            self.noted.lock().unwrap().push(msg.clone());
            if self.refuses {
                return Err(crate::error::Error::Other("refused".into()));
            }
            Ok(())
        }
    }

    /// A filter whose own reaction to every message fails.
    struct Refusing {
        pp_log: PpLog,
        pad: SrcPad,
    }

    impl Element for Refusing {
        fn name(&self) -> Arc<str> {
            "refusing".into()
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

    impl Source for Refusing {
        fn src_pads(&mut self) -> &mut [SrcPad] {
            std::slice::from_mut(&mut self.pad)
        }
    }

    impl Sink for Refusing {
        fn consume(&mut self, _buf: MediaBuffer) -> Result<()> {
            Ok(())
        }

        fn control(&mut self, _msg: &ControlMsg) -> Result<()> {
            Err(crate::error::Error::Other("cannot pause".into()))
        }
    }

    /// An element failing to react to a message does not keep it from the
    /// elements after it: a `Pause` one element refuses still pauses the
    /// rest of the graph, and the caller still hears of the failure.
    #[test]
    fn a_filter_that_fails_to_react_still_passes_the_message_on() {
        let mut filter = Refusing {
            pp_log: element_pp_log(ElementType::Other, "refusing", None),
            pad: SrcPad::new("src"),
        };
        let (after, noted) = noting(false);
        filter.pad.link(Box::new(after));

        let delivered = deliver(&mut filter, &ControlMsg::Pause);

        assert!(delivered.is_err(), "the refusal is the answer");
        assert_eq!(
            *noted.lock().unwrap(),
            [ControlMsg::Pause],
            "and the element after it paused all the same"
        );
    }

    /// Two pads, as a file's demuxer has one for the picture and one for
    /// the sound.
    struct TwoPads {
        pp_log: PpLog,
        pads: [SrcPad; 2],
    }

    impl Element for TwoPads {
        fn name(&self) -> Arc<str> {
            "two-pads".into()
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

    impl Source for TwoPads {
        fn src_pads(&mut self) -> &mut [SrcPad] {
            &mut self.pads
        }
    }

    impl SourceElement for TwoPads {
        fn is_live(&self) -> bool {
            false
        }

        fn is_seekable(&self) -> bool {
            false
        }

        fn run(&mut self, _control: &ControlReceiver, _bus: &Bus) -> Result<()> {
            unreachable!("not exercised by these tests")
        }

        fn seek(&mut self, target: Duration) -> Result<Duration> {
            Ok(target)
        }
    }

    /// A source tells every one of its pads, whatever the first answers. It
    /// told them in turn and stopped at the first failure, so a picture
    /// branch refusing a `Pause` left the sound branch beside it playing.
    #[test]
    fn a_source_tells_every_pad_even_where_one_fails() {
        let mut source = TwoPads {
            pp_log: element_pp_log(ElementType::Other, "two-pads", None),
            pads: [SrcPad::new("video"), SrcPad::new("audio")],
        };
        let (picture, _) = noting(true);
        let (sound, sound_noted) = noting(false);
        source.pads[0].link(Box::new(picture));
        source.pads[1].link(Box::new(sound));
        let (bus, _bus_rx) = Bus::new();

        let applied = apply_one_unacked(&mut source, &bus, &ControlMsg::Pause, false);

        assert!(applied.is_err(), "the picture's refusal is the answer");
        assert_eq!(
            *sound_noted.lock().unwrap(),
            [ControlMsg::Pause],
            "and the sound paused all the same"
        );
    }
}
