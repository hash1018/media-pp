use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
};

use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent, BusReceiver},
    clock::Clock,
    control::{self, ControlMsg, ControlReceiver, ControlSender},
    element::{Element, Filter, Sink, SourceElement},
    error::Result,
    queue::{OverflowPolicy, Queue},
};

/// Builds one chain segment (a run of elements that all execute on the same
/// thread). Call [`ChainBuilder::queue`] to close the current segment behind
/// a `Queue` and start a new one on its own worker thread.
///
/// Because each element needs a handle to *its* downstream to be
/// constructed, the chain is assembled back-to-front: elements are
/// collected in call order, then folded right-to-left starting from the
/// terminal `Sink` at [`ChainBuilder::build`] time.
pub struct ChainBuilder {
    bus: Bus,
    elements: Vec<Box<dyn StageBuilder>>,
}

trait StageBuilder: Send {
    fn wrap(self: Box<Self>, downstream: Box<dyn Sink>, bus: &Bus) -> Box<dyn Sink>;
}

struct DirectStage<T>(T);

impl<T> StageBuilder for DirectStage<T>
where
    T: Filter + 'static,
{
    fn wrap(self: Box<Self>, downstream: Box<dyn Sink>, _bus: &Bus) -> Box<dyn Sink> {
        let mut element = self.0;
        assert_eq!(
            element.src_pads().len(),
            1,
            "ChainBuilder::pipe() is for single-output elements; link a multi-pad \
             element's src_pads() by hand instead"
        );
        element.src_pads()[0].link(downstream);
        Box::new(element)
    }
}

struct QueueStage {
    name: String,
    capacity: usize,
    policy: OverflowPolicy,
}

/// Wraps a terminal `Sink`, posting a `BusEvent::Eos` (under the sink's own
/// `Element::name()`) once it sees an EOS buffer pass through — mirrors
/// what `Queue` does for its own downstream, but without introducing a
/// thread boundary. This is what lets a fully direct chain (no `queue()`
/// calls at all) still report EOS on the bus.
struct EosReporter {
    bus: Bus,
    inner: Box<dyn Sink>,
}

impl Element for EosReporter {
    fn name(&self) -> &str {
        self.inner.name()
    }
}

impl Sink for EosReporter {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        let is_eos = buf.is_eos();
        self.inner.consume(buf)?;
        if is_eos {
            self.bus.post(BusEvent::Eos {
                element: self.inner.name().to_string(),
            });
        }
        Ok(())
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        self.inner.control(msg)
    }
}

impl StageBuilder for QueueStage {
    fn wrap(self: Box<Self>, downstream: Box<dyn Sink>, bus: &Bus) -> Box<dyn Sink> {
        Box::new(Queue::spawn_with_policy(
            self.name,
            self.capacity,
            downstream,
            bus.clone(),
            self.policy,
        ))
    }
}

impl ChainBuilder {
    pub fn new(bus: Bus) -> Self {
        Self {
            bus,
            elements: Vec::new(),
        }
    }

    /// Adds a single-output `Filter` (decoder, encoder, filter, ...) that
    /// receives via `Sink` and produces through its own (single) src pad.
    /// It runs on the same thread as whatever is upstream of it — direct
    /// function call, no queue.
    pub fn pipe<T: Filter + 'static>(mut self, element: T) -> Self {
        self.elements.push(Box::new(DirectStage(element)));
        self
    }

    /// Introduces a thread boundary (blocking when full — see
    /// [`OverflowPolicy::Block`]): everything added after this runs on its
    /// own worker thread instead of the thread that feeds this queue.
    pub fn queue(self, name: impl Into<String>, capacity: usize) -> Self {
        self.queue_with_policy(name, capacity, OverflowPolicy::Block)
    }

    /// Same as [`ChainBuilder::queue`], but lets you choose what happens
    /// when the queue is full (e.g. [`OverflowPolicy::DropNewest`] for a
    /// live source that shouldn't stall upstream).
    pub fn queue_with_policy(
        mut self,
        name: impl Into<String>,
        capacity: usize,
        policy: OverflowPolicy,
    ) -> Self {
        self.elements.push(Box::new(QueueStage {
            name: name.into(),
            capacity,
            policy,
        }));
        self
    }

    /// Terminates the chain with a `Sink` (muxer, file sink, ...) and
    /// assembles everything into a single `Box<dyn Sink>` ready to be
    /// linked into a source's src pad. The terminal's own `Element::name()`
    /// is what shows up on the bus when it reports EOS.
    pub fn build(self, terminal: Box<dyn Sink>) -> Box<dyn Sink> {
        let terminal: Box<dyn Sink> = Box::new(EosReporter {
            bus: self.bus.clone(),
            inner: terminal,
        });
        self.elements
            .into_iter()
            .rev()
            .fold(terminal, |downstream, stage| {
                stage.wrap(downstream, &self.bus)
            })
    }
}

/// Top-level pipeline: a source (with everything reachable from its src
/// pads already linked) plus the bus it reports events on and the
/// [`Clock`] every [`crate::elements::Pacer`] in it shares.
///
/// `run()` is asynchronous: it starts the source on a background thread
/// and returns immediately, rather than blocking the caller for the
/// whole play-through. Always held as `Arc<Pipeline>` (that's what
/// [`Pipeline::new`] returns) — the background thread needs its own
/// owning handle to outlive the `run()` call that spawned it, and that's
/// also what lets [`Pipeline::pause`]/[`Pipeline::resume`]/
/// [`Pipeline::stop`] be called from another thread while it's running.
///
/// There's no separate "is it done yet" query or callback: watch
/// [`Pipeline::bus`] instead. [`BusReceiver::iter`]/
/// [`BusReceiver::log_events`] block until every [`Bus`] handle in the
/// whole pipeline has been dropped, which only happens once the
/// background thread (and everything reachable from the source) has
/// fully finished — so draining the bus doubles as "wait for
/// completion." A source-level failure (returned from
/// [`crate::element::SourceElement::run`] itself, as opposed to one
/// reported from inside a `Queue`) shows up there too, as a
/// [`BusEvent::Error`] under the source's own name, since there's no
/// synchronous return path left to carry it.
///
/// A `Pipeline` isn't reusable once `run()` has been called (whether it
/// finished via a natural `Eos` or [`Pipeline::stop`]) — a second `run()`
/// call is a no-op; build a fresh `Pipeline` for another play-through.
pub struct Pipeline {
    source: Mutex<Option<Box<dyn SourceElement>>>,
    /// Taken (leaving `None` behind) the moment `run()` starts, and moved
    /// into the background thread — so once a pipeline is running,
    /// `Pipeline` itself no longer holds a `Bus` sender. If it did,
    /// [`BusReceiver::iter`] could never observe every sender dropped
    /// (one would always still be sitting right here), and would block
    /// forever instead of unblocking once the pipeline actually finishes.
    bus: Mutex<Option<Bus>>,
    control_tx: ControlSender,
    control_rx: ControlReceiver,
    clock: Arc<Clock>,
    bus_rx: BusReceiver,
    running: AtomicBool,
}

impl Pipeline {
    /// `wire` is called once with the freshly created source, a `Bus`, and
    /// this pipeline's `Clock` (share it with every
    /// [`crate::elements::Pacer`] via `Clock::clone` — one clock per
    /// pipeline, so every paced branch agrees on the same t=0 and the
    /// same pause/resume timeline), so it can build one or more
    /// `ChainBuilder` chains (one per src pad that should actually be
    /// used) and link them via `source.src_pads()[i].link(...)`. Pads
    /// left unlinked just drop whatever gets pushed into them.
    pub fn new<S: SourceElement + 'static>(
        mut source: S,
        wire: impl FnOnce(&mut S, &Bus, &Arc<Clock>),
    ) -> Arc<Self> {
        let (bus, bus_rx) = Bus::new();
        let clock = Arc::new(Clock::new());
        wire(&mut source, &bus, &clock);
        let (control_tx, control_rx) = control::channel();
        Arc::new(Pipeline {
            source: Mutex::new(Some(Box::new(source))),
            bus: Mutex::new(Some(bus)),
            control_tx,
            control_rx,
            clock,
            bus_rx,
            running: AtomicBool::new(false),
        })
    }

    pub fn bus(&self) -> &BusReceiver {
        &self.bus_rx
    }

    /// The clock every `Pacer` in this pipeline paces against — see
    /// [`Pipeline::pause`] for why callers don't usually need to touch
    /// this directly.
    pub fn clock(&self) -> &Arc<Clock> {
        &self.clock
    }

    /// Starts driving the source on a background thread and returns
    /// immediately — see the type-level docs for how to learn when it's
    /// actually done. A no-op if this `Pipeline` is already running or
    /// has already finished a previous run — this type has no "reset"
    /// path; build a fresh `Pipeline` for another play-through.
    pub fn run(self: &Arc<Self>) {
        let Some(mut source) = self.source.lock().unwrap().take() else {
            return;
        };
        // Always `Some` in lockstep with `source` above — both taken
        // exactly once, on whichever `run()` call actually wins the
        // `source` guard.
        let Some(bus) = self.bus.lock().unwrap().take() else {
            return;
        };

        self.running.store(true, Ordering::Release);
        let control_rx = self.control_rx.clone();
        let this = Arc::clone(self);
        thread::Builder::new()
            .name("pipeline:source".into())
            .spawn(move || {
                let source_name = source.name().to_string();
                if let Err(e) = source.run(&control_rx) {
                    bus.post(BusEvent::Error {
                        element: source_name,
                        message: e.to_string(),
                    });
                }
                this.running.store(false, Ordering::Release);
            })
            .expect("failed to spawn pipeline source thread");
    }

    /// Blocks until every element downstream of the source has paused —
    /// see [`crate::control::drain_control`] (source side) and
    /// [`crate::queue::Queue`]'s worker loop (each thread boundary). Also
    /// pauses this pipeline's `Clock`, so a `Pacer` doesn't see a jump in
    /// elapsed time once resumed. No-op if `run()` isn't currently in
    /// progress on another thread.
    pub fn pause(&self) {
        if !self.running.load(Ordering::Acquire) {
            return;
        }
        self.control_tx.send(ControlMsg::Pause);
        self.clock.pause();
    }

    /// Undoes [`Pipeline::pause`]. Resumes the `Clock` first, so it's
    /// already shifted forward by the time `Pacer`s start receiving
    /// frames again.
    pub fn resume(&self) {
        if !self.running.load(Ordering::Acquire) {
            return;
        }
        self.clock.resume();
        self.control_tx.send(ControlMsg::Resume);
    }

    /// Requests an early, full stop — abandons whatever's in flight
    /// rather than draining to a natural `Eos`. The background thread
    /// started by `run()` exits shortly after; watch [`Pipeline::bus`] to
    /// know when. Not reusable afterward — build a new `Pipeline` for the
    /// next play-through.
    pub fn stop(&self) {
        if !self.running.load(Ordering::Acquire) {
            return;
        }
        self.control_tx.send(ControlMsg::Stop);
    }
}

#[cfg(test)]
mod tests {
    use std::{thread, time::Duration};

    use super::*;
    use crate::{element::Source, elements::FileDemuxer};

    /// Real video file, one directory up from this crate — same one every
    /// example defaults to.
    fn test_video() -> &'static str {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../test-video/h265.mp4")
    }

    /// End-to-end: `run()` (async — starts the background thread and
    /// returns right away), then `pause()`/`stop()` (skipping `resume()`)
    /// from the test's own thread — exercises the whole cascade (source's
    /// `drain_control` loop -> `Queue`'s worker) at once, not just `Queue`
    /// in isolation (see `queue::tests`). Mainly guards against the
    /// deadlock this design is built to avoid: draining the bus
    /// afterward must return promptly, not hang forever waiting on a
    /// control message — or a `Bus` handle — that never arrives/drops.
    #[test]
    fn pause_then_stop_returns_promptly() {
        let (source, streams) = FileDemuxer::open("demux", test_video()).expect("open test video");
        let video = streams
            .iter()
            .find(|s| s.kind == ffmpeg_next::media::Type::Video)
            .expect("test video has a video stream");
        let index = video.index;

        let pipeline = Pipeline::new(source, |source, bus, _clock| {
            let branch = ChainBuilder::new(bus.clone())
                .queue("q", 4)
                .build(Box::new(NoOpSink));
            source.src_pads()[index].link(branch);
        });

        pipeline.run();

        // Give the background thread a moment to actually start looping
        // so `pause()`/`stop()` land while `running` is true, not before.
        thread::sleep(Duration::from_millis(50));
        pipeline.pause();
        pipeline.stop();

        // Blocks until every `Bus` handle in the pipeline has been
        // dropped — i.e. until the background thread has actually
        // finished, not just acked `stop()`.
        let events: Vec<_> = pipeline.bus().iter().collect();
        assert!(
            !events.iter().any(|e| matches!(e, BusEvent::Error { .. })),
            "unexpected error event(s): {events:?}"
        );
    }

    struct NoOpSink;
    impl Element for NoOpSink {
        fn name(&self) -> &str {
            "noop"
        }
    }
    impl Sink for NoOpSink {
        fn consume(&mut self, _buf: MediaBuffer) -> Result<()> {
            Ok(())
        }
        fn control(&mut self, _msg: ControlMsg) -> Result<()> {
            Ok(())
        }
    }
}
