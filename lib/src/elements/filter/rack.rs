use std::sync::{Arc, Mutex};

use crate::pp_log::{PpLog, pp_info};
use thiserror::Error as ThisError;

use super::line::Line;
use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, OutputContract},
    control::ControlMsg,
    element::{
        BoxFilter, Context, Element, ElementType, Flow, RawFilter, RawSink, SrcPads, element_pp_log,
    },
    error::Result,
    pad::SrcPad,
    stash::OutputStash,
    stream::StreamEvent,
};

/// Errors produced while filling a [`Rack`].
#[derive(Debug, ThisError)]
pub enum RackError {
    /// One of the elements handed over does not have exactly one output, so
    /// there is no single place for the next one to attach.
    #[error("a Rack holds a straight line, and {name} has {count} outputs rather than one")]
    NotSingleOutput {
        /// The offending element's own name.
        name: Arc<str>,
        /// How many source pads it turned out to have.
        count: usize,
    },

    /// One of the elements handed over waits on the clock, as a
    /// [`Pacer`](crate::elements::Pacer) does, and runs behind a queue of
    /// its own — which a chain gives it, and a rack, running what it holds
    /// on the thread that feeds it, cannot.
    #[error("a Rack runs what it holds on the thread feeding it, and {name} waits on the clock")]
    Waits {
        /// The offending element's own name.
        name: Arc<str>,
    },
}

/// A stretch of chain whose contents can be replaced while frames are
/// flowing through it.
///
/// Everything else in this crate settles its graph before the pipeline runs:
/// [`ChainBuilder`](crate::pipeline::ChainBuilder) wires a branch and that
/// branch keeps its shape for as long as it exists. Changing one element in
/// the middle therefore means building the branch again, which means
/// restarting whatever is at the top of it — a camera, or on Wayland a
/// screen capture whose reopening puts a portal dialog on screen.
///
/// A `Rack` is the exception, and only in one dimension. What it holds is
/// still a straight line, one buffer in and whatever that line makes out;
/// what changes is which elements are in it.
///
/// ```text
///     upstream ─▶ Rack ─▶ downstream        the two ends never notice
///                  │
///                  └ replaced through [`RackHandle`], live
/// ```
///
/// The name is an effects rack: an ordered set of processors in a signal
/// path, which is patched while the signal keeps running.
///
/// # What may go in one
///
/// Elements whose output depends on the buffer in front of them and nothing
/// else — a scaler, a converter, a chroma key. A replacement drops what the
/// outgoing elements were holding, so anything with delayed state loses it:
/// an encoder's queued frames, a resampler's partial output, a muxer's
/// everything.
///
/// This is deliberate rather than unimplemented. Draining the outgoing chain
/// would put its last frames *after* the incoming chain's first ones, and
/// putting the timeline back in order is not something a container in the
/// middle of a graph can do. Dropping is the honest answer, and the contract
/// above is what keeps it from mattering.
///
/// # Contracts are declared, not derived
///
/// A `Rack` is told what it takes and what it emits, because the caller
/// putting one between two fixed elements knows both. Deriving them from the
/// contents would mean answering `Unknown` — the contents are not knowable
/// at construction — and the wiring check would go quiet across exactly the
/// stretch a caller is most likely to get wrong.
///
/// The declaration is not verified against what goes in. Putting a download
/// element in a rack that promises device memory is a caller's mistake of the
/// same kind as declaring it anywhere else, and it surfaces the same way: at
/// runtime, in the element that receives something it cannot read.
///
/// # When a replacement takes effect
///
/// On the next buffer. A `Rack` in a pipeline that is paused, or fed by a
/// source that has stopped, keeps what it has until something arrives —
/// there is nothing on screen to be wrong meanwhile.
///
/// Control does not install one either. A Flush or a Seek addresses what is
/// running, and elements that have not seen a buffer yet have nothing to
/// flush or seek.
///
/// # What is in one is still in the pipeline
///
/// Being inside a rack changes nothing about how an element is treated. On
/// its way in it is given exactly what
/// [`ChainBuilder`](crate::pipeline::ChainBuilder) gives a stage it builds:
/// its log records name the pipeline it is in, it is offered the
/// [`Context`] so it can pace itself or claim a clock, and it is wrapped in
/// the same tracer, so a failure it raises reaches a
/// [`Queue`](crate::queue::Queue) naming *it* rather than naming the rack
/// that was holding it.
///
/// The one thing it is not is a graph node. A rack is a single element in
/// the topology diagram and what it holds does not appear there — which is
/// why filling one logs what went in, at `Info`.
pub struct Rack {
    pp_log: PpLog,
    name: Arc<str>,
    input: InputContract,
    /// What is in the rack — empty for an empty one.
    line: Line,
    control: Arc<RackControl>,
    /// The pipeline this rack was wired into, kept so that what is put in it
    /// afterwards can be given the same start a chain stage gets.
    ///
    /// `None` for a rack that is not in a pipeline — one under test, or one
    /// built and never attached. A fill then does everything except the two
    /// things that need a pipeline to be meaningful.
    context: Option<Arc<Context>>,
    pad: SrcPad,
    /// What the line made that `pad` could not take while a preroll held the
    /// graph — see [`OutputStash`]. The line hands everything it makes to
    /// the rack; this is where what follows the rack says whether it can
    /// take more.
    stash: OutputStash,
}

/// What a [`RackHandle`] leaves for its [`Rack`] to pick up.
///
/// A slot rather than the rack itself, so replacing costs the element
/// nothing until it next has a buffer to key: the lock is taken to `take()`
/// and released, never held across the work.
#[derive(Default)]
struct RackControl {
    pending: Mutex<Option<Vec<Box<dyn RawFilter>>>>,
}

impl RackControl {
    fn take(&self) -> Option<Vec<Box<dyn RawFilter>>> {
        self.pending
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
    }
}

/// Thread-safe runtime control for a [`Rack`].
///
/// Cloning is cheap, and every clone fills the same rack. Retaining one keeps
/// only the small shared slot alive — not the rack, its elements, or the
/// pipeline graph. A handle whose rack has been dropped still accepts a
/// replacement; nothing picks it up, and nothing fails.
#[derive(Clone)]
pub struct RackHandle {
    control: Arc<RackControl>,
}

impl RackHandle {
    /// Replaces everything in the rack, in the order given.
    ///
    /// Takes effect on the next buffer. An empty `elements` empties the
    /// rack, which then passes each buffer straight through — the same
    /// picture, not a copy of it.
    ///
    /// The elements are checked here rather than when they are installed, so
    /// a caller learns about a bad one at the call that handed it over
    /// instead of through a bus event one frame later.
    pub fn replace(&self, elements: Vec<BoxFilter>) -> std::result::Result<(), RackError> {
        let mut elements: Vec<Box<dyn RawFilter>> =
            elements.into_iter().map(BoxFilter::into_raw).collect();
        for element in &mut elements {
            let count = element.src_pads().len();
            if count != 1 {
                return Err(RackError::NotSingleOutput {
                    name: element.name(),
                    count,
                });
            }
            if element.own_queue().is_some() {
                return Err(RackError::Waits {
                    name: element.name(),
                });
            }
        }
        *self
            .control
            .pending
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(elements);
        Ok(())
    }
}

impl Rack {
    /// Creates an empty rack, and the handle that fills it.
    ///
    /// `input`/`output` are what this promises the elements on either side —
    /// see this type's own docs on why they are declared rather than derived.
    /// An empty rack passes buffers through untouched, so one that is never
    /// filled is a straight wire.
    pub fn new(
        name: impl Into<String>,
        input: InputContract,
        output: OutputContract,
    ) -> (Self, RackHandle) {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::Rack, &name, None);
        let control = Arc::new(RackControl::default());
        pp_info!(pp_log: &pp_log, "created: empty");
        let rack = Self {
            pad: SrcPad::with_contract(format!("{name}_src"), output),
            stash: OutputStash::default(),
            line: Line::new(ElementType::Rack, &name),
            pp_log,
            name,
            input,
            control: control.clone(),
            context: None,
        };
        (rack, RackHandle { control })
    }

    /// Hands `buf` on through the stash.
    fn hand_on(&mut self, buf: MediaBuffer) -> Result<()> {
        self.stash.push(&mut self.pad, buf)
    }

    /// Replaces what is in the rack with `elements`.
    fn fill(&mut self, elements: Vec<Box<dyn RawFilter>>) {
        // At Info because a fill is a topology change of the same kind a
        // dynamic `Tee` attach is: sparse, caller-driven, and invisible in
        // the pipeline's own diagram, since what a rack holds are not graph
        // elements. Without this line a log shows a rack and no way to know
        // what is in it.
        match self.line.fill(elements, self.context.as_ref()) {
            Some(line) => pp_info!(self, "filled: {line}"),
            None => pp_info!(self, "filled: empty, buffers pass straight through"),
        }
    }
}

impl Element for Rack {
    /// Remembers the pipeline, so that what is put in this rack afterwards
    /// can be given the same start a chain stage gets — see this type's own
    /// docs on what is in a rack still being in the pipeline.
    fn attach_context(&mut self, context: &Arc<Context>) {
        self.context = Some(context.clone());
        self.stash.attach(&context.state);
    }

    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::Rack
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl SrcPads for Rack {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl RawSink for Rack {
    /// What the first element in it says, as that element would in the
    /// rack's place: a rack that answered yes for an element that cannot take
    /// a buffer yet had the thread in front of it wait inside the push
    /// instead. A line handed to the handle since goes in first — this is
    /// asked between buffers, as `consume` is.
    fn ready_consume(&mut self) -> bool {
        if let Some(elements) = self.control.take() {
            self.fill(elements);
        }
        self.stash.ready(&mut self.pad) && self.line.ready_consume()
    }

    /// What the caller declared, which is what the elements on either side
    /// are checked against — see this type's own docs.
    fn input_contract(&self) -> InputContract {
        self.input
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        // Between buffers, which is the only moment a line can be exchanged
        // without something being halfway down it.
        if let Some(elements) = self.control.take() {
            self.fill(elements);
        }

        if self.line.is_empty() {
            // An empty rack is a wire. Not a copy of the buffer: the same
            // one, which for a pooled picture is what keeps a rack from
            // costing a slot to hold nothing.
            return self.hand_on(buf);
        }
        for buf in self.line.consume(buf)? {
            self.hand_on(buf)?;
        }
        Ok(())
    }

    /// Reaches what is installed, which is not necessarily what has been
    /// handed to the handle — see this type's own docs.
    fn flow(&mut self, Flow(msg): Flow<'_>) -> Result<()> {
        // Into the rack here, and past it once this returns, so an element
        // inside sees a Flush before the element after the rack does, exactly
        // as it would if the two were linked directly.
        if matches!(msg, ControlMsg::Flush | ControlMsg::Stop) {
            self.stash.clear();
        }
        self.line.control(msg)
    }

    /// Into the rack here, what it answers pushed ahead of it, and past it
    /// once this returns — as control goes.
    fn stream_event(&mut self, event: &StreamEvent) -> Result<()> {
        // What is kept came before the event, and goes before what the line
        // answers it with.
        let mut first = self.stash.release_all(&mut self.pad);
        for buf in self.line.stream_event(event)? {
            if let Err(error) = self.pad.push(buf) {
                first = first.and(Err(error));
            }
        }
        first
    }
}

impl Drop for Rack {
    fn drop(&mut self) {
        pp_info!(self, "dropped: releasing whatever it held");
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use ffmpeg_next as ffmpeg;

    use super::*;
    use crate::contract::{MediaKind, MemoryDomain, PortContract};
    use crate::element::RawSinkExt;
    use crate::pool::UnboundObjectPool;

    /// Records what reached the end of the branch the rack is in.
    struct CapturingSink {
        pp_log: PpLog,
        received: Arc<Mutex<Vec<MediaBuffer>>>,
        controls: Arc<Mutex<Vec<ControlMsg>>>,
    }

    impl Element for CapturingSink {
        fn name(&self) -> Arc<str> {
            "capture".into()
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

    impl RawSink for CapturingSink {
        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            self.received.lock().unwrap().push(buf);
            Ok(())
        }
        fn flow(&mut self, Flow(msg): Flow<'_>) -> Result<()> {
            self.controls.lock().unwrap().push(msg.clone());
            Ok(())
        }
    }

    /// An element that stamps every frame it passes with its own mark, so a
    /// test can read the order the rack applied things in.
    struct Marker {
        pp_log: PpLog,
        name: Arc<str>,
        mark: i64,
        pad: SrcPad,
        controls: Arc<Mutex<Vec<ControlMsg>>>,
    }

    impl Marker {
        fn new(name: &str, mark: i64, controls: Arc<Mutex<Vec<ControlMsg>>>) -> Self {
            let name: Arc<str> = name.into();
            Self {
                pp_log: element_pp_log(ElementType::Other, &name, None),
                pad: SrcPad::new(format!("{name}_src")),
                name,
                mark,
                controls,
            }
        }
    }

    impl Element for Marker {
        fn name(&self) -> Arc<str> {
            self.name.clone()
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

    impl SrcPads for Marker {
        fn src_pads(&mut self) -> &mut [SrcPad] {
            std::slice::from_mut(&mut self.pad)
        }
    }

    impl RawSink for Marker {
        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            let MediaBuffer::Video(frame) = buf else {
                return self.pad.push(buf);
            };
            // The mark rides on the timestamp, which is the one field a
            // pooled frame carries that this can write without touching
            // pixels: ten times what was there, plus this element's own
            // digit, so a chain of them reads as the order it ran in.
            let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
            let mut next = pool.get();
            *next = ffmpeg::frame::Video::empty();
            next.set_pts(Some(frame.pts().unwrap_or(0) * 10 + self.mark));
            self.pad.push(MediaBuffer::Video(Arc::new(next)))
        }

        fn flow(&mut self, Flow(msg): Flow<'_>) -> Result<()> {
            self.controls.lock().unwrap().push(msg.clone());
            Ok(())
        }
    }

    /// Takes a buffer only when the test says it can.
    struct Waiting {
        pp_log: PpLog,
        pad: SrcPad,
        ready: Arc<std::sync::atomic::AtomicBool>,
    }

    impl Element for Waiting {
        fn name(&self) -> Arc<str> {
            "waiting".into()
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

    impl SrcPads for Waiting {
        fn src_pads(&mut self) -> &mut [SrcPad] {
            std::slice::from_mut(&mut self.pad)
        }
    }

    impl RawSink for Waiting {
        fn ready_consume(&mut self) -> bool {
            self.ready.load(std::sync::atomic::Ordering::Relaxed)
        }
        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            self.pad.push(buf)
        }
    }

    /// A rack can take a buffer when what is in it can, as that element
    /// would say in the rack's place — including one put in since the last
    /// buffer, which goes in as the rack is asked.
    #[test]
    fn a_rack_is_ready_when_what_it_holds_is() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let (mut rack, handle) = Rack::new("rack", InputContract::Unknown, OutputContract::Unknown);
        assert!(rack.ready_consume(), "empty, it is a wire");
        let ready = Arc::new(AtomicBool::new(false));
        handle
            .replace(vec![crate::element::BoxFilter::new(Waiting {
                pp_log: element_pp_log(ElementType::Other, "waiting", None),
                pad: SrcPad::new("waiting_src"),
                ready: Arc::clone(&ready),
            })])
            .expect("one pad");
        assert!(!rack.ready_consume(), "what it holds cannot take one yet");
        ready.store(true, Ordering::Relaxed);
        assert!(rack.ready_consume());
    }

    /// Makes two frames of each: its own, and one a millisecond on.
    struct Doubles(PpLog);

    impl Element for Doubles {
        fn name(&self) -> Arc<str> {
            "doubles".into()
        }
        fn element_type(&self) -> ElementType {
            ElementType::Other
        }
        fn pp_log(&self) -> &PpLog {
            &self.0
        }
        fn pp_log_mut(&mut self) -> &mut PpLog {
            &mut self.0
        }
    }

    impl crate::element::Filter for Doubles {
        fn transform(&mut self, buf: MediaBuffer, out: &mut crate::element::Output) -> Result<()> {
            let pts = pts_of(&buf);
            out.push(buf);
            out.push(frame(pts + 1));
            Ok(())
        }
    }

    /// Takes a frame while open, and shuts after each — a terminal that has
    /// its preroll sample.
    struct Shuts {
        pp_log: PpLog,
        open: Arc<std::sync::atomic::AtomicBool>,
        taken: Arc<Mutex<Vec<i64>>>,
    }

    impl Element for Shuts {
        fn name(&self) -> Arc<str> {
            "shuts".into()
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

    impl RawSink for Shuts {
        fn ready_consume(&mut self) -> bool {
            self.open.load(std::sync::atomic::Ordering::SeqCst)
        }
        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            self.taken.lock().unwrap().push(pts_of(&buf));
            self.open.store(false, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }
    }

    /// A preroll's terminal after the rack is handed one frame: what the
    /// line makes goes into the rack, which takes it all, and what the
    /// rack's own pad cannot take is kept, the rack not ready meanwhile.
    /// Once playback goes on it goes first, in order.
    #[test]
    fn a_preroll_hands_the_terminal_after_the_rack_one_frame() {
        use crate::element::IntoFilter;

        let context = Arc::new(Context::for_test_with_clock(
            crate::bus::Bus::new().0,
            "test",
            crate::graph::PipelineGraph::new(),
            crate::graph::ElementId::for_test(1),
            Arc::new(crate::clock::Clock::new()),
        ));
        let (mut rack, handle) = Rack::new("rack", InputContract::Unknown, OutputContract::Unknown);
        rack.attach_context(&context);
        handle
            .replace(vec![
                Doubles(element_pp_log(ElementType::Other, "doubles", None)).into_filter(),
            ])
            .expect("one pad");
        let open = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let taken = Arc::new(Mutex::new(Vec::new()));
        rack.src_pads()[0].link(Box::new(Shuts {
            pp_log: element_pp_log(ElementType::Other, "shuts", None),
            open: Arc::clone(&open),
            taken: Arc::clone(&taken),
        }));
        context.state.observe(&ControlMsg::Preroll(Arc::new(
            crate::control::PrerollContext::new([]),
        )));

        rack.consume(frame(10)).expect("the sample, and one kept");
        assert_eq!(*taken.lock().unwrap(), [10]);
        assert!(!rack.ready_consume(), "held while it keeps one");

        context.state.observe(&ControlMsg::Pause);
        context.state.observe(&ControlMsg::Resume);
        rack.consume(frame(20)).expect("what was kept, then these");
        assert_eq!(*taken.lock().unwrap(), [10, 11, 20, 21]);
    }

    fn frame(pts: i64) -> MediaBuffer {
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut slot = pool.get();
        slot.set_pts(Some(pts));
        MediaBuffer::Video(Arc::new(slot))
    }

    fn pts_of(buf: &MediaBuffer) -> i64 {
        let MediaBuffer::Video(frame) = buf else {
            panic!("expected a Video buffer");
        };
        frame.pts().expect("every frame here carries one")
    }

    struct Rig {
        rack: Rack,
        handle: RackHandle,
        received: Arc<Mutex<Vec<MediaBuffer>>>,
        controls: Arc<Mutex<Vec<ControlMsg>>>,
    }

    fn rig() -> Rig {
        let (mut rack, handle) = Rack::new(
            "rack",
            InputContract::Fixed(PortContract::frame(
                MediaKind::VideoFrame,
                MemoryDomain::System,
            )),
            OutputContract::Fixed(PortContract::frame(
                MediaKind::VideoFrame,
                MemoryDomain::System,
            )),
        );
        let received = Arc::new(Mutex::new(Vec::new()));
        let controls = Arc::new(Mutex::new(Vec::new()));
        rack.src_pads()[0].link(Box::new(CapturingSink {
            pp_log: element_pp_log(ElementType::Other, "capture", None),
            received: received.clone(),
            controls: controls.clone(),
        }));
        Rig {
            rack,
            handle,
            received,
            controls,
        }
    }

    /// A rack nobody filled is a wire, and specifically not a copy: the
    /// buffer that arrives is the buffer that leaves.
    #[test]
    fn an_empty_rack_passes_the_same_buffer_through() {
        let mut rig = rig();
        let buf = frame(7);
        let MediaBuffer::Video(input) = &buf else {
            panic!("expected a Video buffer");
        };
        let input_id = crate::buffer::picture_id(input);

        rig.rack.consume(buf.clone()).expect("pass through");

        let received = rig.received.lock().unwrap();
        let MediaBuffer::Video(output) = &received[0] else {
            panic!("expected a Video buffer");
        };
        assert_eq!(
            crate::buffer::picture_id(output),
            input_id,
            "an empty rack must forward its input, not a copy of it"
        );
    }

    #[test]
    fn what_is_in_a_rack_runs_in_the_order_it_was_given() {
        let mut rig = rig();
        rig.handle
            .replace(vec![
                crate::element::BoxFilter::new(Marker::new("first", 1, rig.controls.clone())),
                crate::element::BoxFilter::new(Marker::new("second", 2, rig.controls.clone())),
            ])
            .expect("two ordinary filters");

        rig.rack.consume(frame(0)).expect("key the frame");

        let received = rig.received.lock().unwrap();
        assert_eq!(
            pts_of(&received[0]),
            12,
            "0 through `first` is 1, and 1 through `second` is 12"
        );
    }

    /// The whole point: the elements change while the ends do not.
    #[test]
    fn a_replacement_takes_effect_on_the_next_buffer() {
        let mut rig = rig();
        rig.handle
            .replace(vec![crate::element::BoxFilter::new(Marker::new(
                "first",
                1,
                rig.controls.clone(),
            ))])
            .expect("one filter");
        rig.rack.consume(frame(0)).expect("first line");

        rig.handle
            .replace(vec![crate::element::BoxFilter::new(Marker::new(
                "other",
                3,
                rig.controls.clone(),
            ))])
            .expect("another filter");
        rig.rack.consume(frame(0)).expect("second line");

        rig.handle.replace(Vec::new()).expect("emptying is allowed");
        rig.rack.consume(frame(9)).expect("empty line");

        let received = rig.received.lock().unwrap();
        assert_eq!(pts_of(&received[0]), 1, "the first line marked it");
        assert_eq!(pts_of(&received[1]), 3, "the second one did instead");
        assert_eq!(pts_of(&received[2]), 9, "and an emptied rack left it alone");
    }

    /// A rack is told what it carries rather than working it out, so the
    /// wiring check does not go quiet across it.
    #[test]
    fn a_rack_declares_what_it_was_told_to() {
        let (rack, _handle) = Rack::new(
            "rack",
            InputContract::Fixed(PortContract::frame(
                MediaKind::VideoFrame,
                MemoryDomain::D3d11,
            )),
            OutputContract::Fixed(PortContract::frame(
                MediaKind::VideoFrame,
                MemoryDomain::D3d11,
            )),
        );
        assert_eq!(
            rack.input_contract(),
            InputContract::Fixed(PortContract::frame(
                MediaKind::VideoFrame,
                MemoryDomain::D3d11,
            )),
            "an empty rack still promises what it was built to promise"
        );
    }

    /// Control has to reach what is inside before what is after, which is
    /// what it would do if the two were linked directly.
    #[test]
    fn control_reaches_the_elements_inside_and_then_the_one_after() {
        let mut rig = rig();
        rig.handle
            .replace(vec![crate::element::BoxFilter::new(Marker::new(
                "first",
                1,
                rig.controls.clone(),
            ))])
            .expect("one filter");
        // One buffer first: a replacement is installed on the data path, so
        // asking an unfed rack for a flush asks an empty one.
        rig.rack.consume(frame(0)).expect("install the line");

        crate::control::deliver(&mut rig.rack, &ControlMsg::Flush).expect("flush");

        let controls = rig.controls.lock().unwrap();
        assert_eq!(controls.len(), 2, "the element inside, then the one after");
        assert!(matches!(controls[0], ControlMsg::Flush));
        assert!(matches!(controls[1], ControlMsg::Flush));
    }

    #[test]
    fn control_still_reaches_past_an_empty_rack() {
        let mut rig = rig();
        crate::control::deliver(&mut rig.rack, &ControlMsg::Stop).expect("stop");

        let controls = rig.controls.lock().unwrap();
        assert_eq!(controls.len(), 1);
        assert!(matches!(controls[0], ControlMsg::Stop));
    }

    /// A rack holds a straight line, so something with two outputs has no
    /// single place for the next element to attach. Refused where it was
    /// handed over rather than a frame later.
    #[test]
    fn an_element_with_two_outputs_is_refused_at_the_handle() {
        struct TwoOut {
            pp_log: PpLog,
            pads: Vec<SrcPad>,
        }
        impl Element for TwoOut {
            fn name(&self) -> Arc<str> {
                "two-out".into()
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
        impl SrcPads for TwoOut {
            fn src_pads(&mut self) -> &mut [SrcPad] {
                &mut self.pads
            }
        }
        impl RawSink for TwoOut {
            fn consume(&mut self, _buf: MediaBuffer) -> Result<()> {
                Ok(())
            }
        }

        let rig = rig();
        let error = rig
            .handle
            .replace(vec![crate::element::BoxFilter::new(TwoOut {
                pp_log: element_pp_log(ElementType::Other, "two-out", None),
                pads: vec![SrcPad::new("a"), SrcPad::new("b")],
            })])
            .expect_err("two outputs cannot be put in a line");
        assert!(matches!(error, RackError::NotSingleOutput { count: 2, .. }));
    }

    /// What waits on the clock runs behind a queue of its own, which a
    /// chain gives it and a rack cannot: in one, its waits would hold the
    /// thread that feeds the rack. Refused at the handle, the whole
    /// replacement with it, and what the rack held stays.
    #[test]
    fn what_waits_on_the_clock_is_refused_and_what_was_held_stays() {
        let mut rig = rig();
        rig.handle
            .replace(vec![crate::element::BoxFilter::new(Marker::new(
                "first",
                1,
                rig.controls.clone(),
            ))])
            .expect("one filter");
        rig.rack.consume(frame(0)).expect("the first line");

        let error = rig
            .handle
            .replace(vec![
                crate::element::BoxFilter::new(Marker::new("other", 3, rig.controls.clone())),
                crate::element::BoxFilter::new(crate::elements::Pacer::new("pacer")),
            ])
            .expect_err("a pacer cannot go in a rack");
        assert!(
            matches!(&error, RackError::Waits { name } if &**name == "pacer"),
            "{error}"
        );
        rig.rack.consume(frame(0)).expect("still the first line");

        let received = rig.received.lock().unwrap();
        assert_eq!(pts_of(&received[1]), 1, "what the rack held marked it");
    }

    /// Dropping the rack leaves the handle usable rather than panicking on a
    /// dead reference — see [`RackHandle`]'s own docs.
    #[test]
    fn a_handle_outliving_its_rack_still_accepts_a_replacement() {
        let rig = rig();
        let handle = rig.handle.clone();
        drop(rig);

        handle
            .replace(Vec::new())
            .expect("a replacement nobody will pick up is not a failure");
    }
    /// An element that refuses every buffer, so a test can ask who gets the
    /// blame for it.
    struct Boom {
        pp_log: PpLog,
        name: Arc<str>,
        pad: SrcPad,
    }

    impl Boom {
        fn new(name: &str) -> Self {
            let name: Arc<str> = name.into();
            Self {
                pp_log: element_pp_log(ElementType::Other, &name, None),
                pad: SrcPad::new(format!("{name}_src")),
                name,
            }
        }
    }

    impl Element for Boom {
        fn name(&self) -> Arc<str> {
            self.name.clone()
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

    impl SrcPads for Boom {
        fn src_pads(&mut self) -> &mut [SrcPad] {
            std::slice::from_mut(&mut self.pad)
        }
    }

    impl RawSink for Boom {
        fn consume(&mut self, _buf: MediaBuffer) -> Result<()> {
            Err(crate::error::Error::Other("this element refuses".into()))
        }
    }

    /// What a `Queue` reports is the element that failed, and being inside a
    /// rack must not turn that back into the rack.
    ///
    /// The rack is not the first thing in the line here: `first` is, and it
    /// succeeds. Naming the head would be as wrong as naming the rack, and
    /// only the per-element wrapping tells them apart.
    #[test]
    fn a_failure_inside_a_rack_names_the_element_that_raised_it() {
        let mut rig = rig();
        rig.handle
            .replace(vec![
                crate::element::BoxFilter::new(Marker::new("first", 1, rig.controls.clone())),
                crate::element::BoxFilter::new(Boom::new("second")),
            ])
            .expect("two ordinary filters");

        let error = rig
            .rack
            .consume(frame(0))
            .expect_err("the second element refuses");

        let origin = error.origin().expect("a failure inside a rack is traced");
        assert_eq!(
            &*origin.name, "second",
            "the element that refused, not the rack and not the head"
        );
    }

    /// The rack passes a control message into what it holds, and a refusal
    /// there is traced the same way a refused buffer is.
    #[test]
    fn a_control_refused_inside_a_rack_is_traced_too() {
        struct Deaf {
            pp_log: PpLog,
            pad: SrcPad,
        }
        impl Element for Deaf {
            fn name(&self) -> Arc<str> {
                "deaf".into()
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
        impl SrcPads for Deaf {
            fn src_pads(&mut self) -> &mut [SrcPad] {
                std::slice::from_mut(&mut self.pad)
            }
        }
        impl RawSink for Deaf {
            fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
                self.pad.push(buf)
            }
            fn flow(&mut self, _flow: Flow<'_>) -> Result<()> {
                Err(crate::error::Error::Other("no controls here".into()))
            }
        }

        let mut rig = rig();
        rig.handle
            .replace(vec![crate::element::BoxFilter::new(Marker::new(
                "first",
                1,
                rig.controls.clone(),
            ))])
            .expect("one filter");
        rig.rack.consume(frame(0)).expect("install the line");

        rig.handle
            .replace(vec![crate::element::BoxFilter::new(Deaf {
                pp_log: element_pp_log(ElementType::Other, "deaf", None),
                pad: SrcPad::new("deaf_src"),
            })])
            .expect("one filter");
        rig.rack.consume(frame(0)).expect("install the new line");

        let error = rig
            .rack
            .control(&ControlMsg::Flush)
            .expect_err("the element refuses controls");
        assert_eq!(
            &*error
                .origin()
                .expect("a refusal inside a rack is traced")
                .name,
            "deaf"
        );
    }
}
