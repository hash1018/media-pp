//! Whether a pipeline has finished — the one question a `BusEvent::Eos`
//! cannot answer on its own.
//!
//! Every element that completes end-of-stream says so, a `Queue` as much as
//! a muxer, so the first `Eos` on the bus is only the first thing to end. A
//! caller stopping on it cuts off whatever was still draining: the second
//! track of a file, the other side of a `Tee`. The pipeline knows its
//! terminals, so it is the one to say when they have all ended.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use crate::{
    bus::{Bus, BusEvent},
    graph::{ElementId, PipelineGraph},
    playback_state::PlaybackState,
    pp_log::PpLog,
};

/// Which of a pipeline's terminals have ended, shared by every source's
/// [`crate::element::Context`] so a pipeline with several sources finishes
/// once, when the last of all of them does.
///
/// Holds no [`Bus`]: a sender kept here would keep the bus open after every
/// source has stopped, and [`crate::bus::BusReceiver::iter`] would never
/// end. Whoever reports a change lends its own.
pub(crate) struct Completion {
    graph: PipelineGraph,
    /// Which timeline is current: an end counts only on it, so a seek,
    /// which moves the pipeline on to another, leaves every end before it
    /// behind without anything having to be told.
    state: Arc<PlaybackState>,
    pp_log: PpLog,
    ends: Mutex<Ends>,
}

#[derive(Default)]
struct Ends {
    /// The timeline each terminal took the end of its stream on — the id of
    /// the last segment it was handed before its `Eos`.
    ended: HashMap<ElementId, u64>,
    /// The timeline `Finished` went out for, where it has.
    posted: Option<u64>,
}

impl Completion {
    pub(crate) fn new(graph: PipelineGraph, state: Arc<PlaybackState>, pp_log: PpLog) -> Arc<Self> {
        Arc::new(Self {
            graph,
            state,
            pp_log,
            ends: Mutex::new(Ends::default()),
        })
    }

    /// `terminal` accepted the `Eos` of the stream it was on timeline `on`
    /// of. Called after its own `Eos` is on the bus, so `Finished` always
    /// follows the last one.
    pub(crate) fn terminal_ended(&self, terminal: ElementId, on: u64, bus: &Bus) {
        self.lock().ended.insert(terminal, on);
        self.check(bus);
    }

    /// Whether `terminal` has taken the end of the stream on the current
    /// timeline — a step has no more pictures to ask it for.
    pub(crate) fn has_ended(&self, terminal: ElementId) -> bool {
        let current = self.state.timeline();
        self.lock().ended.get(&terminal) == Some(&current)
    }

    /// The graph lost a branch without its terminal ending — detached, not
    /// finished — so what is left may now be all ended.
    pub(crate) fn branch_removed(&self, bus: &Bus) {
        self.check(bus);
    }

    /// Posts `Finished` once every terminal attached now has ended. A graph
    /// with no terminals has nothing to finish, and is never reported as
    /// having done so.
    ///
    /// The graph is read before this takes its own lock, and the event is
    /// posted after releasing it — posting logs, and no record is written
    /// with a lock held.
    fn check(&self, bus: &Bus) {
        let terminals = self.graph.snapshot().terminal_ids();
        let current = self.state.timeline();
        let finished = {
            let mut ends = self.lock();
            let all_ended = !terminals.is_empty()
                && terminals
                    .iter()
                    .all(|terminal| ends.ended.get(terminal) == Some(&current));
            let finished = all_ended && ends.posted != Some(current);
            if finished {
                ends.posted = Some(current);
            }
            finished
        };
        if finished {
            bus.for_pipeline().post(&self.pp_log, BusEvent::Finished);
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Ends> {
        self.ends
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}
