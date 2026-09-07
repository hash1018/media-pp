use std::sync::{Arc, Mutex};

use crate::pp_log::{PpLog, pp_info};
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, OutputContract},
    control::ControlMsg,
    element::{Element, ElementType, Filter, Sink, Source, element_pp_log},
    error::Result,
    pad::SrcPad,
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
pub struct Rack {
    pp_log: PpLog,
    name: Arc<str>,
    input: InputContract,
    /// The head of what is in the rack, and `None` for an empty one.
    ///
    /// Only the head is kept: filling a rack folds the elements back to
    /// front, each one linked into the next, so everything after the first
    /// is owned by the pad in front of it. It is the same fold
    /// `ChainBuilder::to` does, for the same reason — a link takes ownership.
    head: Option<Box<dyn Sink>>,
    /// Where the last element in the rack puts what it made, for this
    /// element to push onward. Empty between buffers.
    ///
    /// A `Vec` because one buffer in is not one buffer out: a scaler that
    /// answers a frame with none, or several, is a scaler this has to carry
    /// unchanged.
    made: Arc<Mutex<Vec<MediaBuffer>>>,
    control: Arc<RackControl>,
    pad: SrcPad,
}

/// What a [`RackHandle`] leaves for its [`Rack`] to pick up.
///
/// A slot rather than the rack itself, so replacing costs the element
/// nothing until it next has a buffer to key: the lock is taken to `take()`
/// and released, never held across the work.
#[derive(Default)]
struct RackControl {
    pending: Mutex<Option<Vec<Box<dyn Filter>>>>,
}

impl RackControl {
    fn take(&self) -> Option<Vec<Box<dyn Filter>>> {
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
    pub fn replace(
        &self,
        mut elements: Vec<Box<dyn Filter>>,
    ) -> std::result::Result<(), RackError> {
        for element in &mut elements {
            let count = element.src_pads().len();
            if count != 1 {
                return Err(RackError::NotSingleOutput {
                    name: element.name(),
                    count,
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
            pp_log,
            name,
            input,
            head: None,
            made: Arc::new(Mutex::new(Vec::new())),
            control: control.clone(),
        };
        (rack, RackHandle { control })
    }

    /// Wires `elements` into a line and keeps its head, dropping whatever was
    /// in the rack before.
    ///
    /// Folded back to front because a link takes ownership: the last element
    /// is linked to the collector, the one before it to that, and so on, so
    /// only the first is left to hold.
    fn fill(&mut self, elements: Vec<Box<dyn Filter>>) {
        // Read before the fold, which takes ownership of every one of them,
        // and at Info because a fill is a topology change of the same kind a
        // dynamic `Tee` attach is: sparse, caller-driven, and invisible in
        // the pipeline's own diagram, since what a rack holds are not graph
        // elements. Without this line a log shows a rack and no way to know
        // what is in it.
        let line = elements
            .iter()
            .map(|element| format!("{:?}({})", element.element_type(), element.name()))
            .collect::<Vec<_>>()
            .join(" -> ");

        // Dropped before the new line is built rather than after, so two
        // sets of pools do not exist at once on a device that may be short
        // of them.
        self.head = None;
        self.made
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();

        if elements.is_empty() {
            pp_info!(self, "filled: empty, buffers pass straight through");
            return;
        }
        let collector: Box<dyn Sink> = Box::new(Collector {
            pp_log: element_pp_log(ElementType::Rack, &self.name, None),
            made: self.made.clone(),
        });
        let head = elements
            .into_iter()
            .rev()
            .fold(collector, |downstream, mut element| {
                // One output apiece is `RackHandle::replace`'s contract, and
                // it checked. An element that changed its own pad count
                // since then would lose what came after it, so this leaves
                // the line short rather than pretending.
                if let Some(pad) = element.src_pads().first_mut() {
                    pad.link(downstream);
                }
                element
            });
        self.head = Some(head);
        pp_info!(self, "filled: {line}");
    }
}

/// The terminal of the line inside a rack, which is not a terminal at all:
/// what it receives is what the rack pushes onward.
struct Collector {
    pp_log: PpLog,
    made: Arc<Mutex<Vec<MediaBuffer>>>,
}

impl Element for Collector {
    fn name(&self) -> Arc<str> {
        "rack-collector".into()
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

impl Sink for Collector {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        self.made
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(buf);
        Ok(())
    }

    /// The end of the line inside the rack. Control has already reached
    /// every element on the way here, and what is past the rack is the
    /// rack's own pad to tell.
    fn control(&mut self, _msg: ControlMsg) -> Result<()> {
        Ok(())
    }
}

impl Element for Rack {
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

impl Source for Rack {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for Rack {
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

        let Some(head) = &mut self.head else {
            // An empty rack is a wire. Not a copy of the buffer: the same
            // one, which for a pooled picture is what keeps a rack from
            // costing a slot to hold nothing.
            return self.pad.push(buf);
        };
        head.consume(buf)?;

        // Taken before pushing rather than pushed under the lock: what is
        // downstream may be a Queue, a compositor, or anything else that
        // blocks, and none of it should be waiting on a rack's own slot.
        let made = {
            let mut made = self
                .made
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            std::mem::take(&mut *made)
        };
        for buf in made {
            self.pad.push(buf)?;
        }
        Ok(())
    }

    /// Reaches what is installed, which is not necessarily what has been
    /// handed to the handle — see this type's own docs.
    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        // Into the rack first, so an element inside sees a Flush before the
        // element after the rack does, exactly as it would if the two were
        // linked directly.
        if let Some(head) = &mut self.head {
            head.control(msg.clone())?;
        }
        self.pad.control(msg)
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

    impl Sink for CapturingSink {
        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            self.received.lock().unwrap().push(buf);
            Ok(())
        }
        fn control(&mut self, msg: ControlMsg) -> Result<()> {
            self.controls.lock().unwrap().push(msg);
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

    impl Source for Marker {
        fn src_pads(&mut self) -> &mut [SrcPad] {
            std::slice::from_mut(&mut self.pad)
        }
    }

    impl Sink for Marker {
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

        fn control(&mut self, msg: ControlMsg) -> Result<()> {
            self.controls.lock().unwrap().push(msg.clone());
            self.pad.control(msg)
        }
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
                Box::new(Marker::new("first", 1, rig.controls.clone())),
                Box::new(Marker::new("second", 2, rig.controls.clone())),
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
            .replace(vec![Box::new(Marker::new(
                "first",
                1,
                rig.controls.clone(),
            ))])
            .expect("one filter");
        rig.rack.consume(frame(0)).expect("first line");

        rig.handle
            .replace(vec![Box::new(Marker::new(
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
            .replace(vec![Box::new(Marker::new(
                "first",
                1,
                rig.controls.clone(),
            ))])
            .expect("one filter");
        // One buffer first: a replacement is installed on the data path, so
        // asking an unfed rack for a flush asks an empty one.
        rig.rack.consume(frame(0)).expect("install the line");

        rig.rack.control(ControlMsg::Flush).expect("flush");

        let controls = rig.controls.lock().unwrap();
        assert_eq!(controls.len(), 2, "the element inside, then the one after");
        assert!(matches!(controls[0], ControlMsg::Flush));
        assert!(matches!(controls[1], ControlMsg::Flush));
    }

    #[test]
    fn control_still_reaches_past_an_empty_rack() {
        let mut rig = rig();
        rig.rack.control(ControlMsg::Stop).expect("stop");

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
        impl Source for TwoOut {
            fn src_pads(&mut self) -> &mut [SrcPad] {
                &mut self.pads
            }
        }
        impl Sink for TwoOut {
            fn consume(&mut self, _buf: MediaBuffer) -> Result<()> {
                Ok(())
            }
            fn control(&mut self, _msg: ControlMsg) -> Result<()> {
                Ok(())
            }
        }

        let rig = rig();
        let error = rig
            .handle
            .replace(vec![Box::new(TwoOut {
                pp_log: element_pp_log(ElementType::Other, "two-out", None),
                pads: vec![SrcPad::new("a"), SrcPad::new("b")],
            })])
            .expect_err("two outputs cannot be put in a line");
        assert!(matches!(error, RackError::NotSingleOutput { count: 2, .. }));
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
}
