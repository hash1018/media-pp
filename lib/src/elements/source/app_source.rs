use std::{sync::Arc, time::Duration};

use crossbeam_channel::{Receiver, Sender, TrySendError, bounded, select};
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    control::{ControlMsg, ControlReceiver, apply_one, drain_control, wait_out_pause},
    element::{Element, ElementType, Source, SourceElement},
    error::Result,
    pad::SrcPad,
};

/// Errors specific to `AppSource`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum AppSourceError {
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
/// Unlike [`crate::elements::FileDemuxer`], whose blocking read can only
/// be checked against [`ControlMsg`](crate::control::ControlMsg) once per
/// loop iteration (see [`drain_control`]'s docs), `AppSource::run` selects
/// on its control channel and its data channel together — a `Stop` (or
/// any other control message) is handled the moment it arrives, even if
/// [`AppSourceHandle::push`] never gets called again.
///
/// Push [`MediaBuffer::Eos`] when done, or just drop every
/// [`AppSourceHandle`] clone — either ends `run` the same way, pushing
/// exactly one `Eos` of its own to `src_pads()`.
///
/// Has no timeline of its own, so [`SourceElement::seek`] is a no-op that
/// reports back whatever was requested as where it "landed" — nothing to
/// reposition when the app, not a file offset, decides what comes next.
pub struct AppSource {
    name: Arc<str>,
    pad: SrcPad,
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
        let name: Arc<str> = name.into().into();
        let pad = SrcPad::new(format!("{name}_src"));
        let (data_tx, data_rx) = bounded(capacity);
        (
            Self {
                name: name.clone(),
                pad,
                data_rx,
            },
            AppSourceHandle { name, data_tx },
        )
    }
}

impl AppSourceHandle {
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

impl Element for AppSource {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::AppSource
    }
}

impl Source for AppSource {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl SourceElement for AppSource {
    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        loop {
            // Non-blocking first: if control is already backed up, clear
            // it before the `select!` below picks an arbitrary ready arm
            // (it'd be just as correct to skip straight to `select!`, but
            // this keeps `AppSource` consistent with every other
            // `SourceElement::run` calling `drain_control` per iteration).
            if drain_control(control, self, bus)? {
                return Ok(());
            }

            select! {
                recv(control.rx) -> req => {
                    match req {
                        Ok(req) => {
                            if apply_one(self, bus, req.msg, &req.ack)? {
                                return Ok(());
                            }
                            if req.msg == ControlMsg::Pause
                                && wait_out_pause(control, self, bus)?
                            {
                                return Ok(());
                            }
                        }
                        // The Pipeline itself is gone — nothing left to drive this.
                        Err(_) => return Ok(()),
                    }
                }
                recv(self.data_rx) -> buf => {
                    match buf {
                        Ok(buf) if buf.is_eos() => break,
                        Ok(buf) => {
                            if let Err(error) = self.pad.push(buf) {
                                bus.post(BusEvent::Error {
                                    element_type: ElementType::AppSource,
                                    name: self.name.clone(),
                                    error,
                                });
                            }
                        }
                        // Every `AppSourceHandle` dropped without an explicit Eos.
                        Err(_) => break,
                    }
                }
            }
        }
        self.pad.push(MediaBuffer::Eos)
    }

    /// No-op: `AppSource` has nothing of its own to reposition — whatever
    /// comes next is whatever the app pushes next, not a position in a
    /// file. Reports `target` back as where it "landed" so downstream
    /// (e.g. a [`crate::elements::Pacer`] resetting its clock offset)
    /// still sees a consistent [`crate::bus::BusEvent::Seeked`].
    fn seek(&mut self, target: Duration) -> Result<Duration> {
        Ok(target)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::atomic::{AtomicUsize, Ordering},
        thread,
    };

    use super::*;
    use crate::pipeline::{ChainBuilder, Pipeline};

    struct CountingSink {
        count: Arc<AtomicUsize>,
    }

    impl Element for CountingSink {
        fn name(&self) -> Arc<str> {
            "counter".into()
        }

        fn element_type(&self) -> ElementType {
            ElementType::Other
        }
    }

    impl crate::element::Sink for CountingSink {
        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            if !buf.is_eos() {
                self.count.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        }

        fn control(&mut self, _msg: ControlMsg) -> Result<()> {
            Ok(())
        }
    }

    fn packet() -> MediaBuffer {
        MediaBuffer::Packet(Arc::new(ffmpeg_next::Packet::empty()))
    }

    fn wire(source: AppSource, count: Arc<AtomicUsize>) -> Arc<Pipeline> {
        let sink = CountingSink { count };
        Pipeline::new(source, |source, bus, _clock| {
            let branch = ChainBuilder::new(bus.clone()).build(Box::new(sink));
            source.src_pads()[0].link(branch);
        })
    }

    #[test]
    fn pushed_buffers_reach_downstream_then_eos_ends_it() {
        let (source, handle) = AppSource::new("app-source", 4);
        let count = Arc::new(AtomicUsize::new(0));
        let pipeline = wire(source, count.clone());
        pipeline.run();

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
        pipeline.run();

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
        pipeline.run();

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
