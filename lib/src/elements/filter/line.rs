//! A straight line of elements that another element runs inside itself.
//!
//! What a [`Rack`](crate::elements::Rack) holds and replaces while buffers
//! flow, and what a [`VideoDecodeBin`](crate::elements::VideoDecodeBin)
//! holds and replaces once if the hardware refuses a stream. Both need the
//! same three things of it — the elements wired into a line and given what a
//! chain stage is given, buffers run down it with what came out handed back,
//! and control sent down it — and neither is the other's to own.

use std::sync::{Arc, Mutex};

use crate::pp_log::PpLog;
use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    core::pipeline::chain::FlowTracer,
    element::{Context, Element, ElementType, Filter, Sink, element_pp_log},
    error::Result,
};

/// A straight line of elements run inside another element, which pushes on
/// what comes out of its end — see this module's own docs for whose.
///
/// Its elements are stages of the pipeline the owner is in, given what
/// [`ChainBuilder`](crate::pipeline::ChainBuilder) gives one, but not graph
/// nodes: the owner is the node, and is counted for them.
pub(crate) struct Line {
    /// Whose line this is, for the collector's own records.
    owner: ElementType,
    name: Arc<str>,
    /// The head of what is in the line, and `None` for an empty one.
    ///
    /// Only the head is kept: filling folds the elements back to front,
    /// each one linked into the next, so everything after the first is
    /// owned by the pad in front of it. It is the same fold
    /// `ChainBuilder::to` does, for the same reason — a link takes ownership.
    head: Option<Box<dyn Sink>>,
    /// Where the last element in the line puts what it made, for the owner
    /// to push onward. Empty between buffers.
    ///
    /// A `Vec` because one buffer in is not one buffer out: a scaler that
    /// answers a frame with none, or several, is a scaler this has to carry
    /// unchanged.
    made: Arc<Mutex<Vec<MediaBuffer>>>,
}

impl Line {
    pub(crate) fn new(owner: ElementType, name: &Arc<str>) -> Self {
        Self {
            owner,
            name: name.clone(),
            head: None,
            made: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Whether nothing is in the line.
    pub(crate) fn is_empty(&self) -> bool {
        self.head.is_none()
    }

    /// Wires `elements` into the line, dropping whatever was in it before,
    /// and answers what went in as one readable line — `None` where nothing
    /// did — for the owner to log.
    ///
    /// Folded back to front because a link takes ownership: the last element
    /// is linked to the collector, the one before it to that, and so on, so
    /// only the first is left to hold.
    pub(crate) fn fill(
        &mut self,
        elements: Vec<Box<dyn Filter>>,
        context: Option<&Arc<Context>>,
    ) -> Option<String> {
        // Read before the fold, which takes ownership of every one of them.
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
            return None;
        }
        let collector: Box<dyn Sink> = Box::new(Collector {
            pp_log: element_pp_log(self.owner, &self.name, None),
            owner: self.owner,
            made: self.made.clone(),
        });
        let head = elements
            .into_iter()
            .rev()
            .fold(collector, |downstream, mut element| {
                // The same three things `ChainBuilder` does to a stage it
                // builds, because an element in a line is a stage: it is in
                // a pipeline, so its records say which one; it may need the
                // context to pace itself or claim a clock; and a failure it
                // raises has to arrive naming it rather than naming the
                // element that happened to be holding it.
                if let Some(context) = context {
                    *element.pp_log_mut() = element_pp_log(
                        element.element_type(),
                        &element.name(),
                        Some(&context.pipeline_id),
                    );
                    element.attach_context(context);
                }
                // One output apiece is the filler's contract to check. An
                // element that changed its own pad count since then would
                // lose what came after it, so this leaves the line short
                // rather than pretending.
                if let Some(pad) = element.src_pads().first_mut() {
                    pad.link(downstream);
                }
                Box::new(FlowTracer::new(element)) as Box<dyn Sink>
            });
        self.head = Some(head);
        Some(line)
    }

    /// Runs `buf` down a line that is not empty, and answers what came out
    /// of its end, for the owner to push on.
    ///
    /// Taken out rather than pushed from inside: what is downstream may be
    /// a Queue, a compositor, or anything else that blocks, and none of it
    /// should be waiting on the line's own slot. On an error, what did come
    /// out stays for the next call to answer, still in order.
    pub(crate) fn consume(&mut self, buf: MediaBuffer) -> Result<Vec<MediaBuffer>> {
        if let Some(head) = &mut self.head {
            head.consume(buf)?;
        }
        let mut made = self
            .made
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Ok(std::mem::take(&mut *made))
    }

    /// Whether the line's first element can take a buffer now — which, for
    /// a decoder in it, may be no, and the owner has to say so for it.
    pub(crate) fn ready_consume(&mut self) -> bool {
        self.head.as_mut().is_none_or(|head| head.ready_consume())
    }

    /// Sends `msg` down the line, which stops at its end — what is past the
    /// owner is the owner's own pad to tell.
    pub(crate) fn control(&mut self, msg: ControlMsg) -> Result<()> {
        match &mut self.head {
            Some(head) => head.control(msg),
            None => Ok(()),
        }
    }
}

/// The terminal of a [`Line`], which is not a terminal at all: what it
/// receives is what the line's owner pushes onward.
struct Collector {
    pp_log: PpLog,
    owner: ElementType,
    made: Arc<Mutex<Vec<MediaBuffer>>>,
}

impl Element for Collector {
    fn name(&self) -> Arc<str> {
        "line-collector".into()
    }

    fn element_type(&self) -> ElementType {
        self.owner
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

    /// The end of the line. Control has already reached every element on
    /// the way here, and what is past the owner is the owner's own pad to
    /// tell.
    fn control(&mut self, _msg: ControlMsg) -> Result<()> {
        Ok(())
    }
}
