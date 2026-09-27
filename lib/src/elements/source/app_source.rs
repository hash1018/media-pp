use std::sync::Arc;

use crate::pp_log::{PpLog, pp_info};
use crossbeam_channel::{Receiver, Sender, TrySendError, bounded};
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    contract::OutputContract,
    element::{Element, ElementType, Produce, Produced, ProducingSource, Wait, element_pp_log},
    error::Result,
    produce::{Received, produce_source},
};

/// Errors specific to `AppSource`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum AppSourceError {
    /// The source has stopped or end-of-stream was already submitted.
    #[error("AppSource has already ended (its Pipeline finished, or Eos was already pushed)")]
    Closed,
}

/// A source whose data comes from application code pushing buffers in,
/// rather than this element reading them itself — GStreamer's `appsrc`
/// equivalent, the reverse of [`crate::elements::AppSink`]. Push encoded
/// [`MediaBuffer::Packet`]s (straight into a decoder) or already-decoded
/// [`MediaBuffer::Video`]/[`MediaBuffer::Audio`] (e.g. frames from a
/// camera SDK, or synthetic test data) via [`AppSourceHandle`], from any
/// thread — a live capture callback, a network receive loop, a test.
///
/// It waits for the next buffer and for what the pipeline asks together — a
/// `Stop` (or any other control message) is handled the moment it arrives,
/// even if [`AppSourceHandle::push`] never gets called again.
///
/// Push [`MediaBuffer::Eos`] when done, or just drop every
/// [`AppSourceHandle`] clone — either ends the stream the same way, with
/// exactly one `Eos` of its own on `src_pads()`.
///
/// Has no timeline of its own, so it is not a
/// [`SeekableSource`](crate::element::SeekableSource): there is nothing to
/// reposition when the app, not a file offset, decides what comes next.
pub struct AppSource(ProducingSource<Receiving>);

produce_source!(AppSource);

/// What an [`AppSource`] does when asked: waits for the application's next
/// buffer. All of its work, which the framework makes the source.
struct Receiving {
    pp_log: PpLog,
    name: Arc<str>,
    /// Almost always [`ElementType::AppSource`]. An element built on this
    /// one — [`crate::elements::D3d11SharedTextureSource`] is the case this
    /// exists for — keeps its own identity in the log and on the bus
    /// instead, since what a reader of either wants is the element the
    /// caller actually constructed.
    element_type: ElementType,
    /// What it declares its output to be — see [`AppSource::typed`].
    contract: OutputContract,
    data_rx: Receiver<MediaBuffer>,
}

/// A cheaply-cloneable handle for pushing buffers into an [`AppSource`]
/// from any thread — `Clone` is just two refcount bumps (`name` and the
/// channel sender are both cheap to share).
#[derive(Clone)]
pub struct AppSourceHandle {
    name: Arc<str>,
    data_tx: Sender<MediaBuffer>,
}

impl AppSource {
    /// `capacity` bounds how many pushed buffers may sit unconsumed before
    /// [`AppSourceHandle::push`] blocks — same trade-off as
    /// [`crate::queue::Queue`]'s own `capacity`.
    pub fn new(name: impl Into<String>, capacity: usize) -> (Self, AppSourceHandle) {
        Self::typed(
            name,
            capacity,
            ElementType::AppSource,
            OutputContract::Unknown,
        )
    }

    /// [`Self::new`] for an element built on this one: it supplies its own
    /// [`ElementType`] and, since it knows what it pushes, the output
    /// contract a general-purpose `AppSource` cannot declare.
    pub(crate) fn typed(
        name: impl Into<String>,
        capacity: usize,
        element_type: ElementType,
        contract: OutputContract,
    ) -> (Self, AppSourceHandle) {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(element_type, &name, None);
        pp_info!(pp_log: &pp_log, "created: capacity={capacity}");
        let (data_tx, data_rx) = bounded(capacity);
        (
            Self(ProducingSource::new(Receiving {
                name: name.clone(),
                pp_log,
                element_type,
                contract,
                data_rx,
            })),
            AppSourceHandle { name, data_tx },
        )
    }
}

impl AppSourceHandle {
    /// Returns the source instance name shared by this handle.
    pub fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    /// Blocks until there's room in the channel, or `AppSource` (every
    /// clone of it, e.g. after its `Pipeline` finished) is gone.
    pub fn push(&self, buf: MediaBuffer) -> Result<()> {
        self.data_tx
            .send(buf)
            .map_err(|_| AppSourceError::Closed.into())
    }

    /// Non-blocking `push`, for a live producer where falling behind
    /// matters more than losing a buffer — e.g. a camera callback that
    /// can't afford to stall. `Ok(false)` (not an error) means the
    /// channel was full and `buf` was *not* sent; `Err` only means
    /// `AppSource` itself is gone.
    pub fn try_push(&self, buf: MediaBuffer) -> Result<bool> {
        match self.data_tx.try_send(buf) {
            Ok(()) => Ok(true),
            Err(TrySendError::Full(_)) => Ok(false),
            Err(TrySendError::Disconnected(_)) => Err(AppSourceError::Closed.into()),
        }
    }
}

impl Element for Receiving {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        self.element_type
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Produce for Receiving {
    fn is_live(&self) -> bool {
        false
    }

    fn output_contract(&self) -> OutputContract {
        self.contract
    }

    /// The application's next buffer: the end of the stream where it
    /// pushed one, or where every [`AppSourceHandle`] is gone.
    fn produce(&mut self, wait: &mut Wait<'_>) -> Result<Produced> {
        Ok(match wait.recv(&self.data_rx) {
            Received::Got(buf) if buf.is_eos() => {
                pp_info!(self, "event=eos phase=source_received");
                Produced::End
            }
            Received::Got(buf) => Produced::Buffer(buf),
            Received::LetGo => Produced::Nothing,
            Received::Gone => {
                pp_info!(self, "every AppSourceHandle dropped, ending");
                Produced::End
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::atomic::{AtomicUsize, Ordering},
        thread,
        time::Duration,
    };

    use super::*;
    use crate::{bus::BusEvent, pipeline::Pipeline};

    struct CountingSink {
        pp_log: PpLog,
        count: Arc<AtomicUsize>,
    }

    impl Element for CountingSink {
        fn name(&self) -> Arc<str> {
            "counter".into()
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

    impl crate::element::Sink for CountingSink {
        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            if !buf.is_eos() {
                self.count.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        }
    }

    fn packet() -> MediaBuffer {
        MediaBuffer::Packet(Arc::new(ffmpeg_next::Packet::empty()))
    }

    fn wire(source: AppSource, count: Arc<AtomicUsize>) -> Arc<Pipeline> {
        let sink = CountingSink {
            count,
            pp_log: element_pp_log(ElementType::Other, "counter", None),
        };
        let (pipeline, ()) = Pipeline::new("test", source, |source, ctx| {
            let branch = ctx.branch().to(sink)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("test pipeline wiring must succeed");
        pipeline
    }

    #[test]
    fn pushed_buffers_reach_downstream_then_eos_ends_it() {
        let (source, handle) = AppSource::new("app-source", 4);
        let count = Arc::new(AtomicUsize::new(0));
        let pipeline = wire(source, count.clone());
        pipeline.run().unwrap();

        for _ in 0..5 {
            handle.push(packet()).unwrap();
        }
        handle.push(MediaBuffer::Eos).unwrap();

        let events: Vec<_> = pipeline.bus().iter().collect();
        assert!(
            !events.iter().any(|e| matches!(e, BusEvent::Error { .. })),
            "unexpected error event(s): {events:?}"
        );
        assert!(events.iter().any(|e| matches!(e, BusEvent::Eos { .. })));
        assert_eq!(count.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn dropping_every_handle_without_eos_still_ends_cleanly() {
        let (source, handle) = AppSource::new("app-source", 4);
        let count = Arc::new(AtomicUsize::new(0));
        let pipeline = wire(source, count.clone());
        pipeline.run().unwrap();

        handle.push(packet()).unwrap();
        handle.push(packet()).unwrap();
        drop(handle); // no explicit Eos — the channel disconnecting must end `run` on its own

        let events: Vec<_> = pipeline.bus().iter().collect();
        assert!(
            !events.iter().any(|e| matches!(e, BusEvent::Error { .. })),
            "unexpected error event(s): {events:?}"
        );
        assert!(events.iter().any(|e| matches!(e, BusEvent::Eos { .. })));
        assert_eq!(count.load(Ordering::SeqCst), 2);
    }

    /// Regression guard for the exact reason `run` selects on `control`
    /// and its data channel together instead of just blocking on
    /// `data_rx.recv()`: with nothing ever pushed (and no `Eos`/drop
    /// either), a plain blocking recv would never wake up to see `Stop`
    /// at all — this must return promptly instead of hanging.
    #[test]
    fn stop_ends_promptly_even_with_no_producer() {
        let (source, _handle) = AppSource::new("app-source", 4);
        let count = Arc::new(AtomicUsize::new(0));
        let pipeline = wire(source, count.clone());
        pipeline.run().unwrap();

        // Give the background thread a moment to actually start looping
        // (blocked in `select!`, waiting on data that's never coming)
        // before `stop()` lands.
        thread::sleep(Duration::from_millis(50));
        pipeline.stop();

        let events: Vec<_> = pipeline.bus().iter().collect();
        assert!(
            !events.iter().any(|e| matches!(e, BusEvent::Error { .. })),
            "unexpected error event(s): {events:?}"
        );
        assert_eq!(count.load(Ordering::SeqCst), 0);
    }
}
