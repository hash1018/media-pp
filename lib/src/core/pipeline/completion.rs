//! Whether a pipeline has finished — the one question a `BusEvent::Eos`
//! cannot answer on its own.
//!
//! Every element that completes end-of-stream says so, a `Queue` as much as
//! a muxer, so the first `Eos` on the bus is only the first thing to end. A
//! caller stopping on it cuts off whatever was still draining: the second
//! track of a file, the other side of a `Tee`. The pipeline knows its
//! terminals, so it is the one to say when they have all ended.

use std::{
    collections::HashSet,
    sync::{Arc, Mutex},
};

use crate::{
    bus::{Bus, BusEvent},
    graph::{ElementId, PipelineGraph},
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
    pp_log: PpLog,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    /// Terminals that accepted `Eos` and have not been flushed since.
    ended: HashSet<ElementId>,
    /// Whether `Finished` went out for the stream these terminals ended.
    posted: bool,
}

impl Completion {
    pub(crate) fn new(graph: PipelineGraph, pp_log: PpLog) -> Arc<Self> {
        Arc::new(Self {
            graph,
            pp_log,
            state: Mutex::new(State::default()),
        })
    }

    /// `terminal` accepted `Eos`. Called after its own `Eos` is on the bus,
    /// so `Finished` always follows the last one.
    pub(crate) fn terminal_ended(&self, terminal: ElementId, bus: &Bus) {
        self.lock().ended.insert(terminal);
        self.check(bus);
    }

    /// Whether `terminal` has taken the end of its stream and not been
    /// flushed since — a step has no more pictures to ask it for.
    pub(crate) fn has_ended(&self, terminal: ElementId) -> bool {
        self.lock().ended.contains(&terminal)
    }

    /// `terminal` was flushed by a seek: what it ended is gone, and it has a
    /// new stream to end before the pipeline is finished again.
    pub(crate) fn terminal_flushed(&self, terminal: ElementId) {
        let mut state = self.lock();
        state.ended.remove(&terminal);
        state.posted = false;
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
        let finished = {
            let mut state = self.lock();
            let all_ended = !terminals.is_empty()
                && terminals
                    .iter()
                    .all(|terminal| state.ended.contains(terminal));
            let finished = all_ended && !state.posted;
            state.posted |= finished;
            finished
        };
        if finished {
            bus.for_pipeline().post(&self.pp_log, BusEvent::Finished);
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}
