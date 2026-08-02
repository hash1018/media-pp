use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};

use rust_hlog::{HLog, hinfo};

use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent, BusReceiver},
    clock::Clock,
    control::{self, ControlMsg, ControlReceiver, ControlSender},
    element::{Element, ElementType, Filter, Sink, SourceElement, element_hlog},
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
    /// This chain's owning [`Pipeline`]'s own id — stamped as every
    /// element's `hlog` sub_id (via [`crate::element::element_hlog`]) as
    /// it's wrapped in [`ChainBuilder::pipe`]/[`ChainBuilder::queue`]/
    /// [`ChainBuilder::build`], so a log line names not just which
    /// element failed but which pipeline it belonged to.
    pipeline_id: Arc<str>,
    elements: Vec<Box<dyn StageBuilder>>,
    /// `Type(name)` for each element added so far, in call order — captured
    /// right when `.pipe()`/`.queue()` are called (before the element is
    /// boxed into an opaque `StageBuilder`, which no longer exposes its own
    /// name/type). Rendered into one line and handed to [`Bus::register_branch`]
    /// by [`ChainBuilder::build`] — see [`Pipeline::topology`].
    stage_descs: Vec<String>,
}

trait StageBuilder: Send {
    fn wrap(
        self: Box<Self>,
        downstream: Box<dyn Sink>,
        bus: &Bus,
        pipeline_id: &str,
    ) -> Box<dyn Sink>;
}

struct DirectStage<T>(T);

impl<T> StageBuilder for DirectStage<T>
where
    T: Filter + 'static,
{
    fn wrap(
        self: Box<Self>,
        downstream: Box<dyn Sink>,
        _bus: &Bus,
        pipeline_id: &str,
    ) -> Box<dyn Sink> {
        let mut element = self.0;
        assert_eq!(
            element.src_pads().len(),
            1,
            "ChainBuilder::pipe() is for single-output elements; link a multi-pad \
             element's src_pads() by hand instead"
        );
        *element.hlog_mut() =
            element_hlog(element.element_type(), &element.name(), Some(pipeline_id));
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
    fn name(&self) -> Arc<str> {
        self.inner.name()
    }

    fn element_type(&self) -> ElementType {
        self.inner.element_type()
    }

    fn hlog(&self) -> &HLog {
        self.inner.hlog()
    }

    fn hlog_mut(&mut self) -> &mut HLog {
        self.inner.hlog_mut()
    }
}

impl Sink for EosReporter {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        let is_eos = buf.is_eos();
        self.inner.consume(buf)?;
        if is_eos {
            self.bus.post(
                self.inner.hlog(),
                BusEvent::Eos {
                    element_type: self.inner.element_type(),
                    name: self.inner.name(),
                },
            );
        }
        Ok(())
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        self.inner.control(msg)
    }
}

impl StageBuilder for QueueStage {
    fn wrap(
        self: Box<Self>,
        downstream: Box<dyn Sink>,
        bus: &Bus,
        pipeline_id: &str,
    ) -> Box<dyn Sink> {
        Box::new(Queue::spawn_with_policy(
            self.name,
            self.capacity,
            downstream,
            bus.clone(),
            self.policy,
            Some(pipeline_id),
        ))
    }
}

impl ChainBuilder {
    pub fn new(bus: Bus, pipeline_id: impl Into<String>) -> Self {
        Self {
            bus,
            pipeline_id: pipeline_id.into().into(),
            elements: Vec::new(),
            stage_descs: Vec::new(),
        }
    }

    /// Adds a single-output `Filter` (decoder, encoder, filter, ...) that
    /// receives via `Sink` and produces through its own (single) src pad.
    /// It runs on the same thread as whatever is upstream of it — direct
    /// function call, no queue.
    pub fn pipe<T: Filter + 'static>(mut self, element: T) -> Self {
        self.stage_descs
            .push(format!("{:?}({})", element.element_type(), element.name()));
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
        let name = name.into();
        self.stage_descs
            .push(format!("{:?}({name})", ElementType::Queue));
        self.elements.push(Box::new(QueueStage {
            name,
            capacity,
            policy,
        }));
        self
    }

    /// Terminates the chain with a `Sink` (muxer, file sink, ...) and
    /// assembles everything into a single `Box<dyn Sink>` ready to be
    /// linked into a source's src pad. The terminal's own `Element::name()`
    /// is what shows up on the bus when it reports EOS.
    pub fn build(self, mut terminal: Box<dyn Sink>) -> Box<dyn Sink> {
        *terminal.hlog_mut() = element_hlog(
            terminal.element_type(),
            &terminal.name(),
            Some(&self.pipeline_id),
        );
        let mut descs = self.stage_descs;
        descs.push(format!(
            "{:?}({})",
            terminal.element_type(),
            terminal.name()
        ));
        self.bus.register_branch(descs.join(" - "));
        let terminal: Box<dyn Sink> = Box::new(EosReporter {
            bus: self.bus.clone(),
            inner: terminal,
        });
        self.elements
            .into_iter()
            .rev()
            .fold(terminal, |downstream, stage| {
                stage.wrap(downstream, &self.bus, &self.pipeline_id)
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
    /// This pipeline's own id — passed to [`Pipeline::new`], stamped onto
    /// the source's own `hlog` there and onto every element that passes
    /// through a [`ChainBuilder`] built with it (see [`Pipeline::id`]).
    id: Arc<str>,
    source: Mutex<Option<Box<dyn SourceElement>>>,
    /// Taken (leaving `None` behind) the moment `run()` starts, and moved
    /// into the background thread — so once a pipeline is running,
    /// `Pipeline` itself no longer holds a `Bus` sender. If it did,
    /// [`BusReceiver::iter`] could never observe every sender dropped
    /// (one would always still be sitting right here), and would block
    /// forever instead of unblocking once the pipeline actually finishes.
    bus: Mutex<Option<Bus>>,
    control_tx: ControlSender,
    /// Taken (leaving `None` behind) the moment `run()` starts, and moved
    /// into the background thread — same reasoning as `bus` above. If
    /// `Pipeline` kept its own clone alive for its whole lifetime instead,
    /// the control channel's receiver side would never fully disconnect
    /// even after the background thread has long since exited, so a
    /// [`Pipeline::stop`]/`pause`/`resume` racing that thread's own natural
    /// end (e.g. called right as `run()` finishes on its own) could
    /// enqueue a `Request` nobody will ever read *or drop* — leaving
    /// [`crate::control::ControlSender::send`]'s rendezvous ack blocked
    /// forever instead of unblocked by the disconnect, the way it is the
    /// moment the *last* `ControlReceiver` clone actually goes away.
    control_rx: Mutex<Option<ControlReceiver>>,
    clock: Arc<Clock>,
    bus_rx: BusReceiver,
    running: AtomicBool,
    /// One line per branch (`src_pads()[i]` linked through a
    /// `ChainBuilder::build` inside `wire`), each `"Type(name) - Type(name) - ..."`
    /// from the source through to that branch's terminal — see
    /// [`Pipeline::topology`].
    topology: Vec<String>,
}

impl Pipeline {
    /// `id` names this pipeline — stamped as the source's own `hlog`
    /// sub_id right away, and handed to `wire` (its last argument) so
    /// every `ChainBuilder::new(bus.clone(), id)` it builds tags whatever
    /// passes through it the same way (see [`ChainBuilder`]'s own docs).
    ///
    /// `wire` is called once with the freshly created source, a `Bus`,
    /// this pipeline's `Clock` (share it with every
    /// [`crate::elements::Pacer`] via `Clock::clone` — one clock per
    /// pipeline, so every paced branch agrees on the same t=0 and the
    /// same pause/resume timeline), and this pipeline's own `id`, so it
    /// can build one or more `ChainBuilder` chains (one per src pad that
    /// should actually be used) and link them via
    /// `source.src_pads()[i].link(...)`. Pads left unlinked just drop
    /// whatever gets pushed into them.
    pub fn new<S: SourceElement + 'static>(
        id: impl Into<String>,
        mut source: S,
        wire: impl FnOnce(&mut S, &Bus, &Arc<Clock>, &str),
    ) -> Arc<Self> {
        let id: Arc<str> = id.into().into();
        let (bus, bus_rx) = Bus::new();
        let clock = Arc::new(Clock::new());
        let source_desc = format!("{:?}({})", source.element_type(), source.name());
        *source.hlog_mut() = element_hlog(source.element_type(), &source.name(), Some(&id));
        wire(&mut source, &bus, &clock, &id);
        let topology = bus
            .topology()
            .into_iter()
            .map(|branch| format!("{source_desc} - {branch}"))
            .collect();
        let (control_tx, control_rx) = control::channel();
        Arc::new(Pipeline {
            id,
            source: Mutex::new(Some(Box::new(source))),
            bus: Mutex::new(Some(bus)),
            control_tx,
            control_rx: Mutex::new(Some(control_rx)),
            clock,
            bus_rx,
            running: AtomicBool::new(false),
            topology,
        })
    }

    /// This pipeline's own id, as passed to [`Pipeline::new`].
    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn bus(&self) -> &BusReceiver {
        &self.bus_rx
    }

    /// Human-readable rundown of every element statically wired into this
    /// pipeline: one line per branch (one per `src_pads()[i]` that got
    /// linked through a [`ChainBuilder::build`] inside `wire`), each
    /// formatted `Type(name) - Type(name) - ...` from the source through to
    /// that branch's terminal. Multiple branches (fan-out across more than
    /// one src pad) are joined by newlines.
    ///
    /// Only reflects wiring done through `ChainBuilder` at construction
    /// time — a `SrcPad::link` called by hand instead, or anything attached
    /// dynamically afterward (e.g. [`crate::elements::Tee::add_sink`]),
    /// won't show up here.
    pub fn topology(&self) -> String {
        self.topology.join("\n")
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
        // Always `Some` in lockstep with `source` above — all three taken
        // exactly once, on whichever `run()` call actually wins the
        // `source` guard.
        let Some(bus) = self.bus.lock().unwrap().take() else {
            return;
        };
        let Some(control_rx) = self.control_rx.lock().unwrap().take() else {
            return;
        };

        self.running.store(true, Ordering::Release);
        let this = Arc::clone(self);
        thread::Builder::new()
            .name("pipeline:source".into())
            .spawn(move || {
                hinfo!(main_id: &this.id, "pipeline: run starting");
                let source_name = source.name();
                let source_type = source.element_type();
                // `source.run()` itself already reports non-fatal,
                // per-buffer failures to `bus` as it goes (see
                // `SourceElement::run`'s docs) — a returned `Err` here
                // means something genuinely ended the whole source, e.g.
                // a `Seek` that failed outright.
                let outcome = if let Err(error) = source.run(&control_rx, &bus) {
                    bus.post(
                        source.hlog(),
                        BusEvent::Error {
                            element_type: source_type,
                            name: source_name,
                            error,
                        },
                    );
                    "error"
                } else {
                    "ok"
                };
                hinfo!(main_id: &this.id, "pipeline: run finished ({outcome})");
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

    /// Jumps to an absolute position from the start of the media. Blocks
    /// until the source has repositioned (see
    /// [`crate::element::SourceElement::seek`]) and every element
    /// downstream has reacted (a `Queue` drops its stale backlog, a
    /// decoder flushes, a `Pacer` re-anchors both its pts reference and
    /// this pipeline's `Clock`) — same synchronous cascade as `pause`/
    /// `resume`/`stop`. One-shot, unlike `pause`: nothing further to undo
    /// afterward, playback just continues from the new position. No-op
    /// if `run()` isn't currently in progress on another thread.
    ///
    /// Deliberately does *not* touch `Clock` directly here the way
    /// `pause`/`resume` do — see [`crate::elements::Pacer::control`] for
    /// why that would race a straggler pre-seek frame that's still being
    /// processed.
    pub fn seek(&self, target: Duration) {
        if !self.running.load(Ordering::Acquire) {
            return;
        }
        self.control_tx.send(ControlMsg::Seek(target));
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::atomic::AtomicUsize, thread, time::Duration};

    use super::*;
    use crate::{
        element::Source,
        elements::{FileDemuxer, Pacer},
    };

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

        let pipeline = Pipeline::new("test", source, |source, bus, _clock, id| {
            let branch = ChainBuilder::new(bus.clone(), id)
                .queue("q", 4)
                .build(Box::new(NoOpSink {
                    hlog: element_hlog(ElementType::Other, "noop", None),
                }));
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

    /// `seek()` mid-playback should reposition the source (no error from
    /// `Input::seek`), reset/flush everything downstream without
    /// deadlocking, and let packets keep flowing afterward.
    #[test]
    fn seek_repositions_and_playback_continues() {
        let (source, streams) = FileDemuxer::open("demux", test_video()).expect("open test video");
        let video = streams
            .iter()
            .find(|s| s.kind == ffmpeg_next::media::Type::Video)
            .expect("test video has a video stream");
        let index = video.index;
        let time_base = source.stream_time_base(index).expect("stream disappeared");

        let count = Arc::new(AtomicUsize::new(0));
        let sink = CountingSink {
            count: count.clone(),
            hlog: element_hlog(ElementType::Other, "counting-sink", None),
        };

        // A `Pacer` here isn't incidental: without it, this whole 10s/
        // 300-packet file races through in well under the 50ms sleep
        // below (no decode, no throttling), so `seek()` would land on an
        // already-finished pipeline and silently no-op — exactly the kind
        // of thing a weak `count > 0` assertion wouldn't have caught (see
        // `seek_reports_where_it_actually_landed_when_target_is_not_a_keyframe`
        // for how this was found).
        let pipeline = Pipeline::new("test", source, |source, bus, clock, id| {
            let pacer = Pacer::new("pacer", time_base, clock.clone());
            let branch = ChainBuilder::new(bus.clone(), id)
                .queue("q", 4)
                .pipe(pacer)
                .build(Box::new(sink));
            source.src_pads()[index].link(branch);
        });

        pipeline.run();
        thread::sleep(Duration::from_millis(50));
        pipeline.seek(Duration::from_secs(1));
        // Let packets flow again post-seek before tearing down.
        thread::sleep(Duration::from_millis(100));
        pipeline.stop();

        let events: Vec<_> = pipeline.bus().iter().collect();
        assert!(
            !events.iter().any(|e| matches!(e, BusEvent::Error { .. })),
            "unexpected error event(s): {events:?}"
        );
        assert!(
            count.load(Ordering::SeqCst) > 0,
            "expected at least one packet to arrive after the seek"
        );
        assert!(
            events.iter().any(|e| matches!(
                e,
                BusEvent::Seeked { requested, .. } if *requested == Duration::from_secs(1)
            )),
            "expected a Seeked event reporting the request; got {events:?}"
        );
    }

    /// Regression test for the bug found manually testing `rtsp_serve_seek`:
    /// a container seek can only land on a keyframe at or before `target`
    /// (see `FileDemuxer::seek`'s docs) — `test-video/h265.mp4` has
    /// keyframes only at 0s and ~8.33s, so any `target` between them has
    /// to land back at 0s, nowhere near what was requested. Without
    /// `BusEvent::Seeked` reporting that gap, this looked indistinguishable
    /// from `seek` silently doing nothing.
    #[test]
    fn seek_reports_where_it_actually_landed_when_target_is_not_a_keyframe() {
        let (source, streams) = FileDemuxer::open("demux", test_video()).expect("open test video");
        let video = streams
            .iter()
            .find(|s| s.kind == ffmpeg_next::media::Type::Video)
            .expect("test video has a video stream");
        let index = video.index;
        let time_base = source.stream_time_base(index).expect("stream disappeared");

        // Paced for the same reason as `seek_repositions_and_playback_continues`
        // — otherwise the file finishes before `seek()` is even called.
        let pipeline = Pipeline::new("test", source, |source, bus, clock, id| {
            let pacer = Pacer::new("pacer", time_base, clock.clone());
            let branch = ChainBuilder::new(bus.clone(), id)
                .queue("q", 4)
                .pipe(pacer)
                .build(Box::new(NoOpSink {
                    hlog: element_hlog(ElementType::Other, "noop", None),
                }));
            source.src_pads()[index].link(branch);
        });

        pipeline.run();
        thread::sleep(Duration::from_millis(50));
        // 3s falls inside the file's first (only) GOP before the 8.33s
        // keyframe — landing anywhere near 3s would mean this stream
        // unexpectedly grew more keyframes than it's known to have.
        pipeline.seek(Duration::from_secs(3));
        thread::sleep(Duration::from_millis(100));
        pipeline.stop();

        let events: Vec<_> = pipeline.bus().iter().collect();
        let seeked = events
            .iter()
            .find_map(|e| match e {
                BusEvent::Seeked {
                    requested, landed, ..
                } => Some((*requested, *landed)),
                _ => None,
            })
            .expect("expected a Seeked event");
        assert_eq!(seeked.0, Duration::from_secs(3));
        assert!(
            seeked.1 < Duration::from_secs(1),
            "expected landed to fall back to the 0s keyframe, got {:?}",
            seeked.1
        );
    }

    struct NoOpSink {
        hlog: HLog,
    }
    impl Element for NoOpSink {
        fn name(&self) -> Arc<str> {
            "noop".into()
        }

        fn element_type(&self) -> ElementType {
            ElementType::Other
        }

        fn hlog(&self) -> &HLog {
            &self.hlog
        }

        fn hlog_mut(&mut self) -> &mut HLog {
            &mut self.hlog
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

    #[rust_hlog::hlog]
    struct CountingSink {
        count: Arc<AtomicUsize>,
    }
    impl Element for CountingSink {
        fn name(&self) -> Arc<str> {
            "counting-sink".into()
        }

        fn element_type(&self) -> ElementType {
            ElementType::Other
        }

        fn hlog(&self) -> &HLog {
            &self.hlog
        }

        fn hlog_mut(&mut self) -> &mut HLog {
            &mut self.hlog
        }
    }
    impl Sink for CountingSink {
        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            if matches!(buf, MediaBuffer::Packet(_)) {
                self.count.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        }
        fn control(&mut self, _msg: ControlMsg) -> Result<()> {
            Ok(())
        }
    }

    /// `ChainBuilder::build`'s terminal — and, transitively via `.pipe()`/
    /// `.queue()`, everything upstream of it — should come out tagged with
    /// the pipeline id it was built with, not left at `sub_id: None`.
    #[test]
    fn chain_builder_stamps_pipeline_id_into_terminal_hlog() {
        let (bus, _bus_rx) = Bus::new();
        let sink = NoOpSink {
            hlog: element_hlog(ElementType::Other, "noop", None),
        };
        let built = ChainBuilder::new(bus, "my-pipeline").build(Box::new(sink));
        assert_eq!(built.hlog().log_id(), "Other(noop):Pipeline(my-pipeline)");
    }

    /// `Pipeline::new`'s `id` should come back unchanged from
    /// [`Pipeline::id`], and the `wire` closure's own `id: &str` argument
    /// (used to build a matching `ChainBuilder`) should be that same
    /// value — not, say, whatever `source.name()` happens to be.
    #[test]
    fn pipeline_id_is_whatever_new_was_given() {
        let (source, streams) = FileDemuxer::open("demux", test_video()).expect("open test video");
        let video = streams
            .iter()
            .find(|s| s.kind == ffmpeg_next::media::Type::Video)
            .expect("test video has a video stream");
        let index = video.index;

        let pipeline = Pipeline::new("my-pipeline", source, |source, bus, _clock, id| {
            let branch = ChainBuilder::new(bus.clone(), id).build(Box::new(NoOpSink {
                hlog: element_hlog(ElementType::Other, "noop", None),
            }));
            source.src_pads()[index].link(branch);
        });

        assert_eq!(pipeline.id(), "my-pipeline");
    }

    /// [`Pipeline::topology`] should render the source plus every element
    /// added via `.queue()`/`.pipe()` and the terminal, in order, joined by
    /// `" - "` — and one line per branch when more than one src pad is
    /// linked.
    #[test]
    fn topology_lists_source_through_terminal_per_branch() {
        let (source, streams) = FileDemuxer::open("demux", test_video()).expect("open test video");
        let video = streams
            .iter()
            .find(|s| s.kind == ffmpeg_next::media::Type::Video)
            .expect("test video has a video stream");
        let index = video.index;
        let time_base = source.stream_time_base(index).expect("stream disappeared");

        let pipeline = Pipeline::new("test", source, |source, bus, clock, id| {
            let pacer = Pacer::new("pacer", time_base, clock.clone());
            let branch = ChainBuilder::new(bus.clone(), id)
                .queue("q", 4)
                .pipe(pacer)
                .build(Box::new(NoOpSink {
                    hlog: element_hlog(ElementType::Other, "noop", None),
                }));
            source.src_pads()[index].link(branch);
        });

        assert_eq!(
            pipeline.topology(),
            "FileDemuxer(demux) - Queue(q) - Pacer(pacer) - Other(noop)"
        );

        // `pipeline` is dropped here without ever being `run()`, taking
        // its `.queue()`-spawned worker thread down with it — regression
        // coverage for the `Queue::drop` fix (see
        // `queue::tests::dropping_without_stop_or_eos_does_not_hang`):
        // this used to hang the test process forever.
    }
}
