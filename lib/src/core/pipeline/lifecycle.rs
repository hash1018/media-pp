//! Starting, pausing, resuming and ending a [`Pipeline`].

use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread::{self, JoinHandle},
};

use crate::pp_log::{pp_info, pp_trace, pp_warn};

use crate::{
    bus::BusEvent,
    control::{ControlMsg, ControlSender},
    error::{Result, ThreadSpawnError},
    graph::log_topology,
    playback_state::Phase,
};

use super::runtime::lock;
use super::{Pipeline, PipelineError};

impl Pipeline {
    /// Starts driving the source on a background thread and returns
    /// immediately — see the type-level docs for how to learn when it's
    /// actually done. A pipeline runs once: calling this again, whether it
    /// is still running or has ended, fails with
    /// [`PipelineError::AlreadyStarted`] and changes nothing — this type has
    /// no "reset" path; build a fresh `Pipeline` for another play-through.
    ///
    /// A pipeline [`Pipeline::pause`]d before this starts paused: every
    /// source stops before producing anything, and this returns once they
    /// all have. A [`Pipeline::seek`] then puts exactly one sample through
    /// every terminal — the one at the requested position — and returns
    /// once it has arrived, which is how to take a single frame from part
    /// way into a file without decoding everything before it.
    ///
    /// If a source worker cannot be created, any source workers already
    /// started by this call are stopped and joined before the error returns.
    /// The one-shot pipeline is not reusable after that failure.
    pub fn run(&self) -> Result<()> {
        self.run_with_spawner(|thread_name, task| {
            thread::Builder::new().name(thread_name).spawn(task)
        })
    }

    pub(super) fn run_with_spawner(
        &self,
        mut spawn: impl FnMut(
            String,
            Box<dyn FnOnce() + Send + 'static>,
        ) -> std::io::Result<JoinHandle<()>>,
    ) -> Result<()> {
        // Held throughout, so a `pause` racing this either lands before the
        // sources are taken — and this starts paused — or after, as the
        // ordinary cascade.
        let _operation = lock(&self.operation);
        let Some(sources) = lock(&self.sources).take() else {
            return Err(PipelineError::AlreadyStarted.into());
        };
        // Always `Some` in lockstep with `sources` above — all three taken
        // exactly once, on whichever `run()` call actually wins the
        // `sources` guard.
        let Some(bus) = lock(&self.bus).take() else {
            return Err(PipelineError::AlreadyStarted.into());
        };
        let Some(control_rxs) = lock(&self.control_rxs).take() else {
            return Err(PipelineError::AlreadyStarted.into());
        };

        if crate::log::enabled(crate::log::Level::Info) {
            log_topology(&self.pp_log, "run", &self.graph());
        }
        // Paused before it ran: `Pause` goes into every source's control
        // channel ahead of anything else, before its thread exists, so each
        // one's first look at its control stops it before it produces. Its
        // acknowledgements are collected once the threads are running.
        let start_paused = self.paused.load(Ordering::Acquire);
        let mut pause_acks = Vec::new();
        if start_paused {
            pp_trace!(
                pp_log: &self.pp_log,
                "event=control control=Pause phase=requested reason=start_paused"
            );
            self.state.pause();
            self.clock.pause();
            pause_acks = self
                .control_txs
                .iter()
                .filter_map(|control_tx| control_tx.enqueue(ControlMsg::Pause))
                .collect();
        }
        let source_count = sources.len();
        self.running.store(source_count, Ordering::Release);
        for (index, ((source_id, source), control_rx)) in
            sources.into_iter().zip(control_rxs).enumerate()
        {
            let bus = bus.for_element(source_id);
            let running = Arc::clone(&self.running);
            let state = Arc::clone(&self.state);
            let thread_name = "pipeline:source".to_owned();
            let spawn_result = spawn(
                thread_name.clone(),
                Box::new(move || {
                    // Keep these as locals in this order. During unwinding the
                    // guard is dropped first, then the receiver, then the
                    // source. That makes a Pipeline indirectly retained by a
                    // custom source safe to drop from this worker thread.
                    let mut source = source;
                    let control_rx = control_rx;
                    let _running = RunningSourceGuard::new(running);
                    // What this source makes is on the pipeline's timeline,
                    // and moves to each new one as it applies the `Seek`.
                    crate::timeline::enter(&state);

                    let source_name = source.name();
                    let source_type = source.element_type();
                    // `source.run()` itself already reports non-fatal,
                    // per-buffer failures to `bus` as it goes (see
                    // `SourceElement::run`'s docs) — a returned `Err` here
                    // means something genuinely ended this source, e.g.
                    // a `Seek` that failed outright.
                    let outcome = if let Err(error) = source.run(&control_rx, &bus) {
                        bus.post(
                            source.pp_log(),
                            BusEvent::Error {
                                element_type: source_type,
                                name: source_name.clone(),
                                error,
                            },
                        );
                        // Nothing has told this source's branch that it is
                        // over: `run` returned instead of being stopped, and
                        // dropping it merely tears the elements down. A muxer
                        // waiting on this track would then never write its
                        // trailer, leaving an unplayable file — so cascade the
                        // same `Stop` a deliberate shutdown would have sent.
                        // `Stop` rather than `Eos` because the source failed:
                        // there is no complete stream to drain, only state to
                        // finalize.
                        // The log identity is cloned first: `src_pads` borrows
                        // the source mutably for the whole loop.
                        let source_log = source.pp_log().clone();
                        for pad in source.src_pads() {
                            if let Err(error) = pad.control(&ControlMsg::Stop) {
                                pp_warn!(
                                    pp_log: &source_log,
                                    "failed to stop the branch after a source error: {error}"
                                );
                            }
                        }
                        "error"
                    } else {
                        "ok"
                    };
                    pp_info!(pp_log: source.pp_log(), "finished outcome={outcome}");
                }),
            );
            match spawn_result {
                Ok(handle) => lock(&self.workers).push(handle),
                Err(source) => {
                    // This pipeline is one-shot, so sources already started
                    // cannot be reconstructed for a retry. Stop and join them,
                    // then account for the failed and not-yet-started sources
                    // so callers never observe a half-running pipeline after
                    // this error returns.
                    self.running
                        .fetch_sub(source_count - index, Ordering::AcqRel);
                    // A started source waiting to hand back its `Pause`
                    // acknowledgement is let go first, so it can take the
                    // `Stop` below.
                    drop(std::mem::take(&mut pause_acks));
                    self.state.interrupt();
                    for control_tx in self.control_txs.iter().take(index) {
                        control_tx.send(ControlMsg::Stop);
                    }
                    self.join_workers();
                    return Err(ThreadSpawnError::new(thread_name, source).into());
                }
            }
        }
        if start_paused {
            for ack in pause_acks {
                let _ = ack.recv();
            }
            pp_trace!(
                pp_log: &self.pp_log,
                "event=control control=Pause phase=completed outcome=ok reason=start_paused"
            );
        }
        Ok(())
    }

    /// Blocks until every element downstream of every source has paused —
    /// see [`crate::control::drain_control`] (source side) and
    /// [`crate::queue::Queue`]'s worker loop (each thread boundary). Also
    /// pauses this pipeline's `Clock` before that synchronous cascade
    /// starts, so time spent waiting for a busy downstream element to
    /// acknowledge `Pause` is frozen too and a `Pacer` doesn't see a jump
    /// once resumed.
    ///
    /// Before [`Pipeline::run`], this makes the run start paused — see its
    /// docs. Once every source has stopped, there is nothing to pause and
    /// this does nothing.
    pub fn pause(&self) {
        let _operation = self
            .operation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.running.load(Ordering::Acquire) == 0 {
            if self.not_yet_run() {
                self.paused.store(true, Ordering::Release);
            }
            return;
        }
        self.paused.store(true, Ordering::Release);
        self.pause_runtime();
    }

    /// Whether [`Pipeline::run`] has yet to take the sources.
    fn not_yet_run(&self) -> bool {
        lock(&self.sources).is_some()
    }

    /// Puts a request on every source's control channel, wakes whatever is
    /// waiting on the clock if `wake` says to, and then waits until every
    /// source has handled its request.
    ///
    /// In that order, and for every source at once. A `Pacer` or
    /// `VideoSynchronizer` interrupted mid-wait keeps its buffer and returns,
    /// so its source can take the request that interrupted it — which has to
    /// be there already. Sent one source at a time, as this used to be, a
    /// source waiting its turn behind another's cascade was interrupted with
    /// nothing to take: it read on, each buffer handed straight back by the
    /// still-interrupted pacer, and reached the end of its file in
    /// milliseconds. Waiting for them together also keeps one slow cascade
    /// from holding up the rest.
    ///
    /// Every request interrupts, whatever it is. Only some used to — pause,
    /// stop, finish, and the question that opens a seek — on the reasoning
    /// that the rest are sent while everything is already paused, so nothing
    /// can be waiting on the clock or blocked handing data on. That held
    /// until a source was not paused when it should have been (208af56):
    /// then a `Preroll` found it blocked handing a full, paused queue a
    /// packet, nothing let it go, and the seek waited on it for good. An
    /// interrupt is what lets such a thread go — a `Queue` takes the packet
    /// past its capacity, a `Pacer` lets go of its wait — so every request raises
    /// one, and none depends on the graph already being in the state the
    /// request assumes. It costs a request sent while all is paused nothing:
    /// there is nothing waiting for it to wake.
    pub(super) fn broadcast(
        &self,
        enqueue: impl Fn(&ControlSender) -> Option<crossbeam_channel::Receiver<()>>,
    ) {
        let acks: Vec<_> = self.control_txs.iter().filter_map(enqueue).collect();
        self.state.interrupt();
        for ack in acks {
            let _ = ack.recv();
        }
        // Every source has taken the request and cascaded it, so whatever an
        // interrupt — this one or one raised just before — was holding back
        // can go on.
        self.state.settle();
    }

    pub(super) fn pause_runtime(&self) {
        let msg = ControlMsg::Pause;
        pp_trace!(
            pp_log: &self.pp_log,
            "event=control control={msg:?} phase=requested"
        );
        self.state.pause();
        self.clock.pause();
        self.broadcast(|control_tx| control_tx.enqueue(msg.clone()));
        pp_trace!(
            pp_log: &self.pp_log,
            "event=control control={msg:?} phase=completed outcome=ok"
        );
    }

    /// Undoes [`Pipeline::pause`]. Resumes the `Clock` first, so it's
    /// already shifted forward by the time `Pacer`s start receiving
    /// frames again. Before [`Pipeline::run`], this undoes a `pause` made then, so
    /// the run starts playing.
    pub fn resume(&self) {
        let _operation = self
            .operation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.running.load(Ordering::Acquire) == 0 {
            if self.not_yet_run() {
                self.paused.store(false, Ordering::Release);
            }
            return;
        }
        self.paused.store(false, Ordering::Release);
        if self.stepped.swap(false, Ordering::AcqRel) && self.play_on_from_the_picture() {
            return;
        }
        self.resume_runtime();
    }

    pub(super) fn resume_runtime(&self) {
        let msg = ControlMsg::Resume;
        pp_trace!(
            pp_log: &self.pp_log,
            "event=control control={msg:?} phase=requested"
        );
        self.clock.resume();
        self.state.enter(Phase::Playing);
        self.broadcast(|control_tx| control_tx.enqueue(msg.clone()));
        pp_trace!(
            pp_log: &self.pp_log,
            "event=control control={msg:?} phase=completed outcome=ok"
        );
    }

    /// Performs an early, full stop — abandons buffered work rather than
    /// draining to a natural `Eos`. This call is synchronous: it sends
    /// [`ControlMsg::Stop`] to every source at once and waits until each
    /// one's own cascade has finished. It therefore cannot preempt an arbitrary
    /// source read or `Sink::consume` call already blocked inside user or
    /// external-library code; the call returns only after that work gives
    /// the control cascade a turn. After it returns, watch [`Pipeline::bus`]
    /// for every source's background thread to finish. Not reusable
    /// afterward — build a new `Pipeline` for the next play-through.
    pub fn stop(&self) {
        // Before the operation lock, not after: a seek holds that lock for as
        // long as its preroll wait, and an abandoning caller must not be made
        // to sit through it. Cancelling first turns that wait into an
        // immediate return, so this only waits for the seek's own control
        // cascade to unwind.
        self.abandon_preroll();
        let _operation = self
            .operation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.running.load(Ordering::Acquire) == 0 {
            return;
        }
        self.stop_runtime();
    }

    fn stop_runtime(&self) {
        let msg = ControlMsg::Stop;
        self.paused.store(false, Ordering::Release);
        self.state.enter(Phase::Stopped);
        pp_trace!(
            pp_log: &self.pp_log,
            "event=control control={msg:?} phase=requested"
        );
        self.broadcast(|control_tx| control_tx.enqueue(msg.clone()));
        pp_trace!(
            pp_log: &self.pp_log,
            "event=control control={msg:?} phase=completed outcome=ok"
        );
    }

    /// Gracefully completes every source and waits for the whole graph to
    /// drain. Each source stops producing and places `MediaBuffer::Eos` behind
    /// its already-produced data; queues preserve that order, stateful codecs
    /// flush delayed output, and muxers finalize only after their EOS arrives.
    ///
    /// Unlike [`Pipeline::stop`], this does not abandon queued work. If the
    /// pipeline is paused, it resumes the control cascade first so a full
    /// paused queue cannot prevent its ordered EOS from being enqueued. The
    /// call returns only after every source thread (and the Queue workers each
    /// source owns) has finished. The pipeline is not reusable afterward.
    pub fn finish(&self) {
        // Same reasoning as `Pipeline::stop`: an in-flight seek's preroll wait
        // is about to be discarded by this completion, so there is nothing to
        // gain by sitting through it first.
        self.abandon_preroll();
        let _operation = self
            .operation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.running.load(Ordering::Acquire) == 0 {
            self.join_workers();
            return;
        }

        pp_trace!(
            pp_log: &self.pp_log,
            "event=finish phase=requested"
        );
        if self.paused.load(Ordering::Acquire) {
            self.paused.store(false, Ordering::Release);
            self.resume_runtime();
        }
        self.broadcast(ControlSender::enqueue_finish);
        self.join_workers();
        pp_trace!(
            pp_log: &self.pp_log,
            "event=finish phase=completed outcome=ok"
        );
    }

    pub(super) fn join_workers(&self) {
        let current_thread = thread::current().id();
        let mut workers = self
            .workers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for worker in workers.drain(..) {
            if worker.thread().id() != current_thread {
                let _ = worker.join();
            }
        }
    }
}

/// Decrements the live-source count even if a source panics while running.
struct RunningSourceGuard {
    running: Arc<AtomicUsize>,
}

impl RunningSourceGuard {
    fn new(running: Arc<AtomicUsize>) -> Self {
        Self { running }
    }
}

impl Drop for RunningSourceGuard {
    fn drop(&mut self) {
        self.running.fetch_sub(1, Ordering::AcqRel);
    }
}
