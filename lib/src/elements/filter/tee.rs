use std::sync::{Arc, Mutex};

use rust_hlog::HLog;

use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Element, ElementType, Sink, element_hlog},
    error::Result,
    pad::SrcPad,
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
/// while the pipeline is running. `Clone` is just two refcount bumps
/// (`name` and `pads` are both `Arc`) — free to hand out to as many
/// threads as want to control this `Tee`.
#[derive(Clone)]
pub struct TeeHandle {
    name: Arc<str>,
    pads: Arc<Mutex<Vec<SrcPad>>>,
}

impl Tee {
    /// Starts with no sinks — add some via the returned [`TeeHandle`]
    /// before (or any time after) wiring `Tee` itself into the pipeline.
    pub fn new(name: impl Into<String>) -> (Self, TeeHandle) {
        let name: Arc<str> = name.into().into();
        let hlog = element_hlog(ElementType::Tee, &name, None);
        let pads = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                name: name.clone(),
                hlog,
                pads: pads.clone(),
            },
            TeeHandle { name, pads },
        )
    }
}

impl TeeHandle {
    /// Adds a new sink, live for the next buffer `Tee` consumes.
    pub fn add_sink(&self, sink: Box<dyn Sink>) {
        let mut pads = self.pads.lock().unwrap();
        let mut pad = SrcPad::new(format!("{}_src{}", self.name, pads.len()));
        pad.link(sink);
        pads.push(pad);
    }

    /// Removes and returns the sink at `index`, if present.
    pub fn remove_sink(&self, index: usize) -> Option<Box<dyn Sink>> {
        let mut pads = self.pads.lock().unwrap();
        if index >= pads.len() {
            return None;
        }
        pads.remove(index).unlink()
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
