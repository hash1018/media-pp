//! Where a pipeline's playback stands, in one object every element reads.
//!
//! Whether data is to flow, whether a preroll is letting it through a paused
//! graph and for which seek, and which timeline — which seek's worth of
//! media — is current: the pipeline decides all three, and it writes them
//! here. Elements are given this object when they are wired, as they are
//! given the clocks, and read it where they decide. None of them works it
//! out from the order control messages reached it in; that is how nine of
//! them each came to their own answer, and how a source that missed a
//! `Pause` went on reading.
//!
//! # Written first, then announced
//!
//! The pipeline changes this before it sends the control message that goes
//! with the change, never after. The messages still travel — they are what
//! lets a thread blocked on its channel go, and what an element with
//! something to do about a change does it on: a renderer stopping its
//! device, a decoder dropping what the old timeline left in it — but by the
//! time one reaches an element, what it announces is already true here. A
//! `Pause` therefore holds data back from the moment it is decided, rather
//! than from when it reaches each element in turn.
//!
//! The one change that cannot simply be read early is the end of a preroll
//! in a pause — a seek made while paused, done. Its sources are still
//! reading until the `Pause` reaches them, and what a preroll held back — a
//! `Tee` keeping a branch's packets once it has its sample, a decoder
//! passing nothing on after its one — has to stay held back until then.
//! So a pause that follows a preroll keeps it ([`Phase::Paused`]), and what
//! it held stays held for as long as the pause lasts.
//!
//! What lets data go, on the other hand, is never read here first. A paused
//! source or queue goes on only when the `Resume` or `Preroll` itself
//! reaches it, and passes it on downstream before anything else, so no
//! element is handed data it has not yet been told to take. A preroll lets
//! data flow without being paused, so the pipeline never ends one by
//! playing on directly: it pauses first, then resumes.
//!
//! # Interrupts
//!
//! A request the pipeline sends reaches an element only when the thread
//! carrying it is between buffers, and a thread may be anywhere else: in a
//! `Pacer` waiting for a picture's time, blocked handing a full queue a
//! buffer. So before each request the pipeline raises an interrupt here,
//! which every such wait answers by letting go, and settles it once every
//! source has handled the request — see [`PlaybackState::interrupt`].
//!
//! # Waking what waits on it
//!
//! A thread waiting in a [`Queue`](crate::queue::Queue) — for room, for the
//! interrupt to settle, for its downstream to be ready — is waiting for
//! something here to change, and each change rings it: a [`Bell`] the queue
//! lends this state when it is made. Nothing that waits on the state looks
//! again on a timer.
//!
//! # Outside a pipeline
//!
//! A [`Queue`](crate::queue::Queue) or a source driven by hand has no
//! pipeline to write its state. Its control channel then keeps one of its
//! own, which the sending half moves on with each message it sends — see
//! [`crate::control::channel`]. An element wired into no pipeline at all
//! has no state to read, and behaves as if playing: never held, never in a
//! preroll.

use std::{
    collections::HashMap,
    sync::{
        Arc, Condvar, Mutex, MutexGuard,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Duration,
};

use crossbeam_channel::{Receiver, Sender, TrySendError, bounded};

use crate::control::{ControlMsg, PrerollContext};
use crate::graph::ElementId;
use crate::timeline::UNNUMBERED;

/// Where playback stands: whether data flows, and why.
#[derive(Debug, Clone, Default)]
pub(crate) enum Phase {
    /// Data flows at the clock's pace. Where a graph starts.
    #[default]
    Playing,
    /// Nothing flows until a `Resume` or a `Preroll`. `kept` is the preroll
    /// this pause followed, if it followed one, whose holds stay in force —
    /// see the module docs.
    Paused { kept: Option<Arc<PrerollContext>> },
    /// Paused, but data flows, unpaced, until each terminal has its sample
    /// — see [`PrerollContext`].
    Prerolling(Arc<PrerollContext>),
    /// Abandoned: nothing flows again.
    Stopped,
}

impl Phase {
    /// Moves on as `msg` says — for a control channel with no pipeline to
    /// say it, see the module docs. A `Flush` or `Seek` leaves
    /// the phase as it was: a seek is a pause, a flush, a reposition and a
    /// preroll, and only the pause and the preroll change what flows.
    pub(crate) fn observe(&mut self, msg: &ControlMsg) {
        *self = match msg {
            ControlMsg::Pause => Self::paused_after(self),
            ControlMsg::Resume => Self::Playing,
            ControlMsg::Preroll(context) => Self::Prerolling(Arc::clone(context)),
            ControlMsg::Stop => Self::Stopped,
            ControlMsg::Flush | ControlMsg::Seek(_) => return,
        };
    }

    /// The pause that follows `before`, keeping the preroll it ends.
    fn paused_after(before: &Self) -> Self {
        let kept = match before {
            Self::Prerolling(context) => Some(Arc::clone(context)),
            Self::Paused { kept } => kept.clone(),
            Self::Playing | Self::Stopped => None,
        };
        Self::Paused { kept }
    }

    /// Whether nothing is to flow: paused, or stopped.
    pub(crate) fn holds(&self) -> bool {
        matches!(self, Self::Paused { .. } | Self::Stopped)
    }
}

/// What wakes a thread waiting in a `select!` on a change it cannot see
/// coming — see the module docs.
///
/// A channel of one: ringing never blocks, and a ring nobody has answered
/// yet is not lost, since the waiter finds it the next time it waits and
/// looks again. A ring that has already been answered costs one look that
/// finds nothing new.
#[derive(Clone)]
pub(crate) struct Bell {
    ringer: Sender<()>,
    rings: Receiver<()>,
}

impl Bell {
    pub(crate) fn new() -> Self {
        let (ringer, rings) = bounded(1);
        Self { ringer, rings }
    }

    /// Wakes whatever is waiting on this, or the next thing to.
    pub(crate) fn ring(&self) {
        let _ = self.ringer.try_send(());
    }

    /// What a waiter selects on.
    pub(crate) fn rings(&self) -> &Receiver<()> {
        &self.rings
    }
}

/// A pipeline's playback state, shared by everything in it — see the module
/// docs.
#[derive(Debug)]
pub(crate) struct PlaybackState {
    phase: Mutex<Phase>,
    /// The number of the timeline media is being read on — see
    /// [`crate::timeline`]. Apart from the phase, and atomic, because a
    /// queue reads it for every buffer it hands on.
    timeline: AtomicU64,
    /// Interrupts raised so far, one before each request the pipeline
    /// sends — see [`Self::interrupt`].
    interrupts: AtomicU64,
    /// The interrupt every request sent so far has been handled up to —
    /// see [`Self::interrupt_pending`].
    settled: AtomicU64,
    /// What [`Self::sleep_unless_interrupted`] waits on, notified by every
    /// interrupt. Its lock is taken around the raise, so a sleeper that has
    /// read the count cannot miss the raise that changes it.
    raised: Mutex<()>,
    raised_changed: Condvar,
    /// Rung whenever any of the above moves — see [`Bell`]. What a bell is
    /// no longer heard by, because whatever held it has gone, is dropped at
    /// the next ring.
    listeners: Mutex<Vec<Sender<()>>>,
    /// The picture each terminal that shows pictures last took — what a
    /// frame step moves from; see [`Self::picture_taken`].
    pictures: Mutex<HashMap<ElementId, Picture>>,
    /// Whether the current timeline runs backwards — see [`Self::backwards`].
    backwards: AtomicBool,
    /// How late, in nanoseconds of wall time, the last picture handed on
    /// was — see [`Self::picture_late`].
    picture_late_ns: AtomicU64,
}

/// The picture one terminal last took.
#[derive(Debug, Clone, Copy, Default)]
struct Picture {
    /// Its position in its media, or `None` once a flush has left the
    /// terminal without one.
    at: Option<Duration>,
    /// How far apart its pictures come — what a step back by more than one
    /// picture measures with. The stream's rather than the position's, so
    /// kept across a flush.
    spacing: Option<Duration>,
}

impl PlaybackState {
    /// Playing, on the first timeline.
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            phase: Mutex::new(Phase::Playing),
            timeline: AtomicU64::new(UNNUMBERED + 1),
            interrupts: AtomicU64::new(0),
            settled: AtomicU64::new(0),
            raised: Mutex::new(()),
            raised_changed: Condvar::new(),
            listeners: Mutex::new(Vec::new()),
            pictures: Mutex::new(HashMap::new()),
            backwards: AtomicBool::new(false),
            picture_late_ns: AtomicU64::new(0),
        })
    }

    fn lock(&self) -> MutexGuard<'_, Phase> {
        self.phase
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Rings `bell` whenever this state moves, from now on.
    pub(crate) fn listen(&self, bell: &Bell) {
        self.listeners
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(bell.ringer.clone());
    }

    /// Rings every bell still heard — see [`Bell`].
    fn ring(&self) {
        self.listeners
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .retain(|ringer| !matches!(ringer.try_send(()), Err(TrySendError::Disconnected(_))));
    }

    /// Moves playback to `phase` — the pipeline's to call, before it
    /// announces the change.
    pub(crate) fn enter(&self, phase: Phase) {
        *self.lock() = phase;
        self.ring();
    }

    /// Says a terminal has taken its sample for the preroll under way, which
    /// is what a branch held back for it waits on — see
    /// [`PrerollContext::mark_ready`].
    pub(crate) fn sample_taken(&self) {
        self.ring();
    }

    /// Pauses playback, keeping a preroll this ends — see the module docs.
    /// The pipeline's to call, as [`Self::enter`] is.
    pub(crate) fn pause(&self) {
        {
            let mut phase = self.lock();
            *phase = Phase::paused_after(&phase);
        }
        self.ring();
    }

    /// Moves playback on as `msg` says — what a control channel with no
    /// pipeline behind it does as it sends one.
    pub(crate) fn observe(&self, msg: &ControlMsg) {
        self.lock().observe(msg);
        self.ring();
    }

    /// Whether nothing is to flow: paused, or stopped.
    pub(crate) fn holds(&self) -> bool {
        self.lock().holds()
    }

    /// The preroll whose holds are in force: the one under way, or the one
    /// the pause that followed it keeps — see the module docs.
    pub(crate) fn preroll(&self) -> Option<Arc<PrerollContext>> {
        match &*self.lock() {
            Phase::Prerolling(context)
            | Phase::Paused {
                kept: Some(context),
            } => Some(Arc::clone(context)),
            Phase::Playing | Phase::Paused { kept: None } | Phase::Stopped => None,
        }
    }

    /// Whether a preroll is letting data through a paused graph right now,
    /// unpaced — for what asks once a buffer.
    pub(crate) fn is_prerolling(&self) -> bool {
        matches!(&*self.lock(), Phase::Prerolling(_))
    }

    fn pictures(&self) -> MutexGuard<'_, HashMap<ElementId, Picture>> {
        self.pictures
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Records that `terminal` has taken a picture at `at` in its media,
    /// lasting `lasts` where the picture says — kept by the terminal tracer
    /// for every terminal that is handed pictures, and read by a frame step:
    /// which terminals it steps, from where, and by how much.
    ///
    /// The spacing is measured between pictures taken one after the other,
    /// and taken from the picture's own duration until two have been.
    pub(crate) fn picture_taken(&self, terminal: ElementId, at: Duration, lasts: Option<Duration>) {
        let mut pictures = self.pictures();
        let picture = pictures.entry(terminal).or_default();
        let measured = picture
            .at
            .and_then(|before| at.checked_sub(before))
            .filter(|spacing| !spacing.is_zero());
        picture.spacing = measured
            .or(picture.spacing)
            .or(lasts.filter(|lasts| !lasts.is_zero()));
        picture.at = Some(at);
    }

    /// `terminal` has been flushed: where its picture is is not known again
    /// until it takes the next. Still a terminal that shows pictures, and
    /// its pictures still as far apart.
    pub(crate) fn picture_flushed(&self, terminal: ElementId) {
        if let Some(picture) = self.pictures().get_mut(&terminal) {
            picture.at = None;
        }
    }

    /// Which of `terminals` show pictures.
    pub(crate) fn picture_terminals(&self, terminals: &[ElementId]) -> Vec<ElementId> {
        let pictures = self.pictures();
        terminals
            .iter()
            .copied()
            .filter(|terminal| pictures.contains_key(terminal))
            .collect()
    }

    /// Where the picture is among `terminals` — the furthest any of them has
    /// taken, since fanned out they show the same one — and how far it came
    /// after the one before.
    pub(crate) fn picture_at(
        &self,
        terminals: &[ElementId],
    ) -> Option<(Duration, Option<Duration>)> {
        let pictures = self.pictures();
        terminals
            .iter()
            .filter_map(|terminal| pictures.get(terminal))
            .filter_map(|picture| picture.at.map(|at| (at, picture.spacing)))
            .max_by_key(|(at, _)| *at)
    }

    /// The number of the timeline media is being read on.
    pub(crate) fn timeline(&self) -> u64 {
        self.timeline.load(Ordering::Acquire)
    }

    /// Whether the timeline read now runs backwards: from the seek that
    /// turned playback round with [`crate::pipeline::Pipeline::set_rate`]
    /// until the one that turns it back. What a source is asked to seek
    /// with, and where the stretches a decoder is handed begin and end, are
    /// decided on it.
    pub(crate) fn backwards(&self) -> bool {
        self.backwards.load(Ordering::Acquire)
    }

    /// Sets which way the timeline about to begin runs — the pipeline's to
    /// call, before [`Self::begin_timeline`].
    pub(crate) fn set_backwards(&self, backwards: bool) {
        self.backwards.store(backwards, Ordering::Release);
    }

    /// Records how late a picture was as it was handed on to be shown —
    /// kept by what paces pictures, a `Pacer` or a `VideoSynchronizer`,
    /// and zero for one on time. A decoder reads it to decode less while
    /// pictures come too late to be shown in time; see the decoders' `qos`.
    /// The last one said is what is read, whichever branch said it; a seek
    /// starts again from zero.
    pub(crate) fn picture_late(&self, late: Duration) {
        let ns = u64::try_from(late.as_nanos()).unwrap_or(u64::MAX);
        self.picture_late_ns.store(ns, Ordering::Relaxed);
    }

    /// How late the last picture handed on was — see [`Self::picture_late`].
    pub(crate) fn picture_lateness(&self) -> Duration {
        Duration::from_nanos(self.picture_late_ns.load(Ordering::Relaxed))
    }

    /// Starts a new timeline, and answers its number. Everything numbered
    /// before this is behind from now on.
    pub(crate) fn begin_timeline(&self) -> u64 {
        self.timeline.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Whether a buffer made on timeline `number` is from a position the
    /// pipeline has since left.
    pub(crate) fn is_behind(&self, number: u64) -> bool {
        number != UNNUMBERED && number < self.timeline()
    }

    /// Tells whatever is waiting — a paced wait, a queue — that a request
    /// is on its way, so it lets go and the thread it holds can take it.
    /// The pipeline's to call, before each request it sends.
    pub(crate) fn interrupt(&self) {
        {
            let _raising = self
                .raised
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            self.interrupts.fetch_add(1, Ordering::Release);
        }
        self.raised_changed.notify_all();
        self.ring();
    }

    /// Sleeps for `duration`, or until the next [`Self::interrupt`] if that
    /// comes first — what a paced wait sleeps in, so a request reaches it at
    /// once rather than at the end of a polling slice.
    pub(crate) fn sleep_unless_interrupted(&self, duration: Duration) {
        let guard = self
            .raised
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let epoch = self.interrupt_epoch();
        let _ = self
            .raised_changed
            .wait_timeout_while(guard, duration, |_| self.interrupt_epoch() == epoch);
    }

    /// How many interrupts have been raised — what an element records as it
    /// takes a request, to tell a later one from it.
    pub(crate) fn interrupt_epoch(&self) -> u64 {
        self.interrupts.load(Ordering::Acquire)
    }

    /// Marks every interrupt so far as answered: the requests that followed
    /// it have been handled all the way through the graph. The pipeline's
    /// to call, once every source has acknowledged them.
    pub(crate) fn settle(&self) {
        self.settled
            .store(self.interrupt_epoch(), Ordering::Release);
        self.ring();
    }

    /// Whether an interrupt is out and the requests behind it are still on
    /// their way. A `Queue` takes nothing from its channel meanwhile: what
    /// it fed an interrupted `Pacer` would only pile up there, not play —
    /// see `Queue`.
    ///
    /// Settled by the pipeline, not by each element's own control: a
    /// request can reach an element before the interrupt that goes with it
    /// is raised, and an element that took its own control as the answer
    /// would then wait on an interrupt nothing is coming to answer.
    pub(crate) fn interrupt_pending(&self) -> bool {
        self.interrupt_epoch() != self.settled.load(Ordering::Acquire)
    }

    /// Whether a wait that last answered the interrupt numbered `answered`
    /// must let go now: a later interrupt has been raised and the pipeline
    /// has not yet settled it.
    ///
    /// Both halves, because each alone is wrong. Without the first, an
    /// element that has already taken the control behind an interrupt would
    /// keep giving up until the rest of the graph caught up. Without the
    /// second — which is how `Pacer` and `VideoSynchronizer` used to read it
    /// — only a control message of the element's own could answer an
    /// interrupt, and `Pipeline::finish` raises one and sends none
    /// downstream: its `Eos` travels as data. Every wait after it then gave
    /// up, the picture being waited on was kept for good, and the `Eos`
    /// behind it never left.
    pub(crate) fn interrupted_since(&self, answered: u64) -> bool {
        self.interrupt_epoch() != answered && self.interrupt_pending()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    /// A phase moves only on what changes what flows, and a preroll is
    /// what a pause is released into as much as a resume is.
    #[test]
    fn a_phase_follows_what_changes_what_flows() {
        let context = Arc::new(PrerollContext::new([]));
        let state = PlaybackState::new();
        assert!(!state.holds(), "a graph starts playing");

        state.observe(&ControlMsg::Pause);
        assert!(state.holds());
        for unmoved in [ControlMsg::Flush, ControlMsg::Seek(Duration::from_secs(1))] {
            state.observe(&unmoved);
            assert!(state.holds(), "{unmoved:?} leaves it paused");
        }

        state.observe(&ControlMsg::Preroll(Arc::clone(&context)));
        assert!(!state.holds(), "a preroll lets data through");
        assert!(state.is_prerolling());
        assert!(
            state
                .preroll()
                .is_some_and(|running| Arc::ptr_eq(&running, &context))
        );

        state.observe(&ControlMsg::Pause);
        assert!(!state.is_prerolling(), "a pause ends the preroll");
        assert!(state.holds());
        assert!(
            state
                .preroll()
                .is_some_and(|kept| Arc::ptr_eq(&kept, &context)),
            "and keeps what it held back"
        );
        state.observe(&ControlMsg::Resume);
        assert!(state.preroll().is_none(), "until playback goes on");
        assert!(!state.holds());
        state.observe(&ControlMsg::Stop);
        assert!(state.holds(), "a stop holds for good");
    }

    /// Every move of the state rings what listens to it, and a bell whose
    /// holder has gone is let go of rather than rung for good.
    #[test]
    fn a_bell_rings_for_every_move_and_is_dropped_with_its_holder() {
        let state = PlaybackState::new();
        let bell = Bell::new();
        state.listen(&bell);
        let heard = |bell: &Bell| bell.rings().try_recv().is_ok();

        state.interrupt();
        assert!(heard(&bell), "an interrupt");
        state.settle();
        assert!(heard(&bell), "its settling");
        state.pause();
        assert!(heard(&bell), "a pause");
        state.enter(Phase::Playing);
        assert!(heard(&bell), "a phase");
        state.sample_taken();
        assert!(heard(&bell), "a terminal's sample");
        assert!(!heard(&bell), "one ring apiece");

        drop(bell);
        state.interrupt();
        assert!(state.listeners.lock().unwrap().is_empty());
    }

    #[test]
    fn an_unnumbered_buffer_is_never_behind() {
        let state = PlaybackState::new();
        state.begin_timeline();
        state.begin_timeline();
        assert!(!state.is_behind(UNNUMBERED));
        assert!(state.is_behind(UNNUMBERED + 1));
    }
}
