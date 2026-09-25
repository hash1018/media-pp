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
    sync::{
        Arc, Condvar, Mutex, MutexGuard,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use crossbeam_channel::{Receiver, Sender, TrySendError, bounded};

use crate::control::{ControlMsg, PrerollContext};
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
    /// say it, see the module docs. A `Flush`, `CheckSeek` or `Seek` leaves
    /// the phase as it was: a seek is a pause, a flush, a reposition and a
    /// preroll, and only the pause and the preroll change what flows.
    pub(crate) fn observe(&mut self, msg: &ControlMsg) {
        *self = match msg {
            ControlMsg::Pause => Self::paused_after(self),
            ControlMsg::Resume => Self::Playing,
            ControlMsg::Preroll(context) => Self::Prerolling(Arc::clone(context)),
            ControlMsg::Stop => Self::Stopped,
            ControlMsg::Flush | ControlMsg::CheckSeek(_) | ControlMsg::Seek(_) => return,
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

    /// The number of the timeline media is being read on.
    pub(crate) fn timeline(&self) -> u64 {
        self.timeline.load(Ordering::Acquire)
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
    use crate::control::SeekCheckContext;

    /// A phase moves only on what changes what flows, and a preroll is
    /// what a pause is released into as much as a resume is.
    #[test]
    fn a_phase_follows_what_changes_what_flows() {
        let context = Arc::new(PrerollContext::new([]));
        let state = PlaybackState::new();
        assert!(!state.holds(), "a graph starts playing");

        state.observe(&ControlMsg::Pause);
        assert!(state.holds());
        for unmoved in [
            ControlMsg::Flush,
            ControlMsg::Seek(Duration::from_secs(1)),
            ControlMsg::CheckSeek(Arc::new(SeekCheckContext::new())),
        ] {
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
