use std::sync::{Arc, Mutex, Weak};

use rust_hlog::{HLog, hinfo};

use crate::{
    buffer::MediaBuffer,
    bus::BusEvent,
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
/// handle on another thread can request an add or removal while the
/// pipeline thread is in `consume`. `consume` holds that lock while it
/// visits the current sinks, so the mutation completes after the in-flight
/// buffer has finished fan-out; the new set applies to the next buffer.
///
/// Cheap to fan out: `MediaBuffer` wraps its payload in an `Arc`, so
/// cloning a buffer for each output is a refcount bump, not a copy of the
/// encoded/decoded data.
#[rust_hlog::hlog]
pub struct Tee {
    name: Arc<str>,
    shared: Arc<TeeShared>,
}

struct TeeShared {
    pads: Mutex<Vec<SrcPad>>,
    context: Arc<Context>,
}

/// A cheaply-cloneable handle for adding or removing a [`Tee`]'s sinks
/// while the pipeline is running. It deliberately keeps only a [`Weak`]
/// reference to the `Tee`'s shared state: retaining a handle after the
/// pipeline finishes must not keep downstream sinks or the pipeline's
/// [`crate::bus::Bus`] sender alive. Once the `Tee` is gone,
/// [`TeeHandle::chain_builder`] returns `None` and the other operations are
/// harmless no-ops.
#[derive(Clone)]
pub struct TeeHandle {
    name: Arc<str>,
    shared: Weak<TeeShared>,
}

impl Tee {
    /// Starts with no sinks — add some via the returned [`TeeHandle`]
    /// before (or any time after) wiring `Tee` itself into the pipeline.
    /// `context` should be the same one [`crate::pipeline::Pipeline::new`]'s
    /// (or, for a multi-source pipeline,
    /// [`crate::pipeline::PipelineBuilder::add_source`]'s) `wire` closure
    /// was handed. The `Tee` keeps that context alive while it belongs to
    /// the pipeline; `TeeHandle` accesses it weakly so the handle itself
    /// cannot extend the pipeline's lifetime.
    pub fn new(name: impl Into<String>, context: Arc<Context>) -> (Self, TeeHandle) {
        let name: Arc<str> = name.into().into();
        let hlog = element_hlog(ElementType::Tee, &name, None);
        hinfo!(hlog: &hlog, "created");
        context.registry.register(
            ElementType::Tee,
            name.clone(),
            Some(context.default_upstream.clone()),
        );
        let shared = Arc::new(TeeShared {
            pads: Mutex::new(Vec::new()),
            context,
        });
        (
            Self {
                name: name.clone(),
                hlog,
                shared: shared.clone(),
            },
            TeeHandle {
                name,
                shared: Arc::downgrade(&shared),
            },
        )
    }

    /// Posts a branch's `push` failure to the bus under `Tee`'s own
    /// identity — mirrors how `Queue`'s worker loop attributes a failed
    /// downstream `consume()` to itself rather than the thing it was
    /// pushing into, since that's the only identity `Tee` reliably knows
    /// here (`SrcPad` doesn't expose which element it's linked to).
    fn report_branch_error(&self, error: crate::error::Error) {
        self.shared.context.bus.post(
            &self.hlog,
            BusEvent::Error {
                element_type: ElementType::Tee,
                name: self.name.clone(),
                error,
            },
        );
    }
}

impl TeeHandle {
    /// A [`crate::pipeline::ChainBuilder`] pre-wired with this `Tee`'s own
    /// [`Context`] — lets a caller build a whole new branch (`.pipe(...)`
    /// chains, ending in `.build(...)`) at any point after the pipeline
    /// started running, then hand the result to [`TeeHandle::add_sink`],
    /// without needing to have kept a `Bus`/pipeline id/[`crate::element::ElementRegistry`]
    /// around separately. Returns `None` once the `Tee` has been dropped.
    pub fn chain_builder(&self) -> Option<ChainBuilder> {
        let shared = self.shared.upgrade()?;
        Some(ChainBuilder::new(shared.context.clone()))
    }

    /// Adds a new sink, live for the next buffer `Tee` consumes.
    pub fn add_sink(&self, sink: Box<dyn Sink>) {
        let Some(shared) = self.shared.upgrade() else {
            return;
        };
        // Retroactively points whatever `ChainBuilder` chain built `sink`
        // (registered with an unresolved `upstream` — it had no way to
        // know at `.build()` time whether it'd end up linked straight to
        // a source pad or, as here, added to a `Tee`) at this `Tee`
        // instead of the default-to-source fallback
        // [`crate::pipeline::Pipeline::new`] would otherwise apply.
        shared
            .context
            .registry
            .set_upstream(&sink.name(), self.name.clone());
        let mut pads = shared.pads.lock().unwrap();
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
    /// (see `ElementRegistry::remove_subtree`'s own
    /// docs), so returning it would just invite exactly that. Want a
    /// branch back? Build a fresh one.
    pub fn remove_sink(&self, index: usize) {
        let Some(shared) = self.shared.upgrade() else {
            return;
        };
        let mut pads = shared.pads.lock().unwrap();
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
            shared.context.registry.remove_subtree(&sink.name());
            // `sink` drops here.
        }
    }

    pub fn sink_count(&self) -> usize {
        self.shared
            .upgrade()
            .map(|shared| shared.pads.lock().unwrap().len())
            .unwrap_or(0)
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
        let mut pads = self.shared.pads.lock().unwrap();
        let Some((last, rest)) = pads.split_last_mut() else {
            return Ok(());
        };
        // One branch failing must not stop the buffer from reaching its
        // siblings — same "errors never kill anything, just get reported"
        // rule `Queue`'s worker loop follows. That buffer is dropped for
        // the failing branch only; the branch itself stays wired and gets
        // retried on the next one. Whoever's watching the bus decides
        // whether to call `TeeHandle::remove_sink` over it.
        for pad in rest {
            if let Err(error) = pad.push(buf.clone()) {
                self.report_branch_error(error);
            }
        }
        if let Err(error) = last.push(buf) {
            self.report_branch_error(error);
        }
        Ok(())
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        // Unlike `consume`, every branch gets the same `ControlMsg`
        // value directly (it's `Copy`, no need for the last-one-moves
        // split `consume` does for `MediaBuffer`).
        let mut pads = self.shared.pads.lock().unwrap();
        for pad in pads.iter_mut() {
            pad.control(msg)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::{bus::Bus, clock::Clock, element::ElementRegistry};

    fn packet() -> MediaBuffer {
        MediaBuffer::Packet(Arc::new(ffmpeg_next::Packet::empty()))
    }

    #[rust_hlog::hlog]
    struct CountingSink {
        name: &'static str,
        count: Arc<AtomicUsize>,
    }

    impl Element for CountingSink {
        fn name(&self) -> Arc<str> {
            self.name.into()
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
        fn consume(&mut self, _buf: MediaBuffer) -> Result<()> {
            self.count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn control(&mut self, _msg: ControlMsg) -> Result<()> {
            Ok(())
        }
    }

    #[rust_hlog::hlog]
    struct AlwaysFailSink {}

    impl Element for AlwaysFailSink {
        fn name(&self) -> Arc<str> {
            "always-fail".into()
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

    impl Sink for AlwaysFailSink {
        fn consume(&mut self, _buf: MediaBuffer) -> Result<()> {
            Err(crate::error::Error::Other(
                "simulated branch failure".into(),
            ))
        }

        fn control(&mut self, _msg: ControlMsg) -> Result<()> {
            Ok(())
        }
    }

    /// A failing branch must not stop the same buffer from reaching its
    /// siblings, nor stop `Tee::consume` itself from returning `Ok` — same
    /// "errors get reported, nothing dies" rule `Queue`'s worker loop
    /// follows (see `queue::tests::a_failing_consume_drops_that_buffer_but_keeps_the_worker_alive`).
    /// Wires the failing branch in the *middle* (`before`/`after` on
    /// either side of it) so the test also proves a mid-`rest` failure
    /// doesn't short-circuit the fan-out to what comes after it,
    /// including `last`.
    #[test]
    fn a_failing_branch_does_not_block_its_siblings() {
        let (bus, bus_rx) = Bus::new();
        let context = Arc::new(Context {
            bus,
            pipeline_id: "test".into(),
            registry: ElementRegistry::new(),
            clock: Arc::new(Clock::new()),
            default_upstream: "source".into(),
        });
        let (mut tee, handle) = Tee::new("tee", context);

        let before_count = Arc::new(AtomicUsize::new(0));
        let after_count = Arc::new(AtomicUsize::new(0));
        handle.add_sink(Box::new(CountingSink {
            name: "before",
            count: before_count.clone(),
            hlog: element_hlog(ElementType::Other, "before", None),
        }));
        handle.add_sink(Box::new(AlwaysFailSink {
            hlog: element_hlog(ElementType::Other, "always-fail", None),
        }));
        handle.add_sink(Box::new(CountingSink {
            name: "after",
            count: after_count.clone(),
            hlog: element_hlog(ElementType::Other, "after", None),
        }));

        for _ in 0..3 {
            tee.consume(packet())
                .expect("a branch failing must not surface as an error from Tee::consume");
        }

        assert_eq!(before_count.load(Ordering::SeqCst), 3);
        assert_eq!(after_count.load(Ordering::SeqCst), 3);

        drop(tee);
        drop(handle);
        let errors = bus_rx
            .iter()
            .filter(|e| matches!(e, BusEvent::Error { .. }))
            .count();
        assert_eq!(
            errors, 3,
            "expected one Error event per failed push, not a fatal short-circuit"
        );
    }

    #[test]
    fn retained_handle_does_not_keep_tee_context_or_bus_alive() {
        let (bus, bus_rx) = Bus::new();
        let context = Arc::new(Context {
            bus,
            pipeline_id: "test".into(),
            registry: ElementRegistry::new(),
            clock: Arc::new(Clock::new()),
            default_upstream: "source".into(),
        });
        let (tee, handle) = Tee::new("tee", context.clone());

        drop(context);
        drop(tee);

        assert!(handle.chain_builder().is_none());
        assert_eq!(handle.sink_count(), 0);
        assert!(
            bus_rx.iter().next().is_none(),
            "a retained TeeHandle must not keep the Bus sender alive"
        );
    }
}
