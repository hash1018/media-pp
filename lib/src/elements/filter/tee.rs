use std::sync::{Arc, Mutex};

use rust_hlog::{HLog, hinfo};

use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Context, Element, ElementType, Sink, element_hlog},
    error::Result,
    pad::SrcPad,
    pipeline::ChainBuilder,
};

/// Fans a single input out to a *dynamic* set of sinks — sinks are added
/// and removed through a [`TeeHandle`], which can be cloned and used from
/// any thread, independent of whatever thread is driving `Tee::consume`
/// (the pipeline's source/queue-worker thread). That's the whole reason
/// `Tee` doesn't implement [`crate::element::Source`] like other
/// multi-pad elements (e.g. [`crate::elements::FileDemuxer`]): its pads
/// live behind a lock instead of being a plain `&mut [SrcPad]`, so a
/// handle on another thread can add or remove one while the pipeline
/// thread is mid-`consume`.
///
/// Cheap to fan out: `MediaBuffer` wraps its payload in an `Arc`, so
/// cloning a buffer for each output is a refcount bump, not a copy of the
/// encoded/decoded data.
#[rust_hlog::hlog]
pub struct Tee {
    name: Arc<str>,
    pads: Arc<Mutex<Vec<SrcPad>>>,
}

/// A cheaply-cloneable handle for adding or removing a [`Tee`]'s sinks
/// while the pipeline is running. `Clone` is just three refcount bumps
/// (`name`, `pads`, and `context` are all `Arc`-backed) — free to hand out
/// to as many threads as want to control this `Tee`.
#[derive(Clone)]
pub struct TeeHandle {
    name: Arc<str>,
    pads: Arc<Mutex<Vec<SrcPad>>>,
    context: Arc<Context>,
}

impl Tee {
    /// Starts with no sinks — add some via the returned [`TeeHandle`]
    /// before (or any time after) wiring `Tee` itself into the pipeline.
    /// `context` should be the same one [`crate::pipeline::Pipeline::new`]'s
    /// `wire` closure was handed — `TeeHandle` keeps its own clone so
    /// [`TeeHandle::chain_builder`] can mint further branches long after
    /// `wire` has returned, without the caller needing to have kept a
    /// `Bus`/id/registry around itself.
    pub fn new(name: impl Into<String>, context: Arc<Context>) -> (Self, TeeHandle) {
        let name: Arc<str> = name.into().into();
        let hlog = element_hlog(ElementType::Tee, &name, None);
        hinfo!(hlog: &hlog, "created");
        context
            .registry
            .register(ElementType::Tee, name.clone(), None);
        let pads = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                name: name.clone(),
                hlog,
                pads: pads.clone(),
            },
            TeeHandle {
                name,
                pads,
                context,
            },
        )
    }
}

impl TeeHandle {
    /// A [`crate::pipeline::ChainBuilder`] pre-wired with this `Tee`'s own
    /// [`Context`] — lets a caller build a whole new branch (`.pipe(...)`
    /// chains, ending in `.build(...)`) at any point after the pipeline
    /// started running, then hand the result to [`TeeHandle::add_sink`],
    /// without needing to have kept a `Bus`/pipeline id/[`crate::element::ElementRegistry`]
    /// around separately.
    pub fn chain_builder(&self) -> ChainBuilder {
        ChainBuilder::new(self.context.clone())
    }

    /// Adds a new sink, live for the next buffer `Tee` consumes.
    pub fn add_sink(&self, sink: Box<dyn Sink>) {
        // Retroactively points whatever `ChainBuilder` chain built `sink`
        // (registered with an unresolved `upstream` — it had no way to
        // know at `.build()` time whether it'd end up linked straight to
        // a source pad or, as here, added to a `Tee`) at this `Tee`
        // instead of the default-to-source fallback
        // [`crate::pipeline::Pipeline::new`] would otherwise apply.
        self.context
            .registry
            .set_upstream(&sink.name(), self.name.clone());
        let mut pads = self.pads.lock().unwrap();
        let mut pad = SrcPad::new(format!("{}_src{}", self.name, pads.len()));
        hinfo!(
            hlog: &element_hlog(ElementType::Tee, &self.name, None),
            "sink added: {} total",
            pads.len() + 1
        );
        pad.link(sink);
        pads.push(pad);
    }

    /// Removes the sink at `index`, if present, and drops it — cleanup
    /// happens exactly the same way an abandoned/`Stop`'d branch's does
    /// elsewhere in this crate (e.g. `SwDecoder`/`D3d12vaDecoder` only
    /// flush on `Seek`, not `Stop`, precisely because dropping without an
    /// explicit shutdown message is already a normal, accepted way to
    /// abandon a sink here). Doesn't hand the removed `Box<dyn Sink>` back
    /// — deliberately: re-adding the same removed sink elsewhere isn't a
    /// supported way to get it back into [`crate::pipeline::Pipeline::topology`]
    /// (see [`crate::element::ElementRegistry::remove_subtree`]'s own
    /// docs), so returning it would just invite exactly that. Want a
    /// branch back? Build a fresh one.
    pub fn remove_sink(&self, index: usize) {
        let mut pads = self.pads.lock().unwrap();
        if index >= pads.len() {
            return;
        }
        hinfo!(
            hlog: &element_hlog(ElementType::Tee, &self.name, None),
            "sink removed: index={index}, {} remaining",
            pads.len() - 1
        );
        if let Some(sink) = pads.remove(index).unlink() {
            // Otherwise this branch would keep showing up under
            // `Tee(...)` in `Pipeline::topology()` forever — the registry
            // has no idea it was ever removed unless told.
            self.context.registry.remove_subtree(&sink.name());
            // `sink` drops here.
        }
    }

    pub fn sink_count(&self) -> usize {
        self.pads.lock().unwrap().len()
    }
}

impl Element for Tee {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::Tee
    }

    fn hlog(&self) -> &HLog {
        &self.hlog
    }

    fn hlog_mut(&mut self) -> &mut HLog {
        &mut self.hlog
    }
}

impl Sink for Tee {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        let mut pads = self.pads.lock().unwrap();
        let Some((last, rest)) = pads.split_last_mut() else {
            return Ok(());
        };
        for pad in rest {
            pad.push(buf.clone())?;
        }
        last.push(buf)
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        // Unlike `consume`, every branch gets the same `ControlMsg`
        // value directly (it's `Copy`, no need for the last-one-moves
        // split `consume` does for `MediaBuffer`).
        let mut pads = self.pads.lock().unwrap();
        for pad in pads.iter_mut() {
            pad.control(msg)?;
        }
        Ok(())
    }
}
