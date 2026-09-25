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
//! # Outside a pipeline
//!
//! A [`Queue`](crate::queue::Queue) or a source driven by hand has no
//! pipeline to write its state. Its control channel then keeps one of its
//! own, which the sending half moves on with each message it sends — see
//! [`crate::control::channel`]. An element wired into no pipeline at all
//! has no state to read, and behaves as if playing: never held, never in a
//! preroll.

use std::sync::{
    Arc, Mutex, MutexGuard,
    atomic::{AtomicU64, Ordering},
};

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

/// A pipeline's playback state, shared by everything in it — see the module
/// docs.
#[derive(Debug)]
pub(crate) struct PlaybackState {
    phase: Mutex<Phase>,
    /// The number of the timeline media is being read on — see
    /// [`crate::timeline`]. Apart from the phase, and atomic, because a
    /// queue reads it for every buffer it hands on.
    timeline: AtomicU64,
}

impl PlaybackState {
    /// Playing, on the first timeline.
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            phase: Mutex::new(Phase::Playing),
            timeline: AtomicU64::new(UNNUMBERED + 1),
        })
    }

    fn lock(&self) -> MutexGuard<'_, Phase> {
        self.phase
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Moves playback to `phase` — the pipeline's to call, before it
    /// announces the change.
    pub(crate) fn enter(&self, phase: Phase) {
        *self.lock() = phase;
    }

    /// Pauses playback, keeping a preroll this ends — see the module docs.
    /// The pipeline's to call, as [`Self::enter`] is.
    pub(crate) fn pause(&self) {
        let mut phase = self.lock();
        *phase = Phase::paused_after(&phase);
    }

    /// Moves playback on as `msg` says — what a control channel with no
    /// pipeline behind it does as it sends one.
    pub(crate) fn observe(&self, msg: &ControlMsg) {
        self.lock().observe(msg);
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

    #[test]
    fn an_unnumbered_buffer_is_never_behind() {
        let state = PlaybackState::new();
        state.begin_timeline();
        state.begin_timeline();
        assert!(!state.is_behind(UNNUMBERED));
        assert!(state.is_behind(UNNUMBERED + 1));
    }
}
