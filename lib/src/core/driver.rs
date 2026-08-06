use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
};

use crate::{
    bus::{Bus, BusEvent, BusReceiver},
    element::Element,
    error::Result,
};

/// Checked, not blocked on: a [`Driver`] owns a single self-contained loop
/// with no downstream dataflow graph to cascade a stop through (unlike
/// [`crate::pipeline::Pipeline`]'s `control` channel, which has to reach
/// every `Sink` a `Queue` boundary away before it can call `Stop` fully
/// handled). So [`DriverRunner::stop`] just flips a flag instead of
/// sending something that has to be received and acked — nothing here can
/// reproduce the deadlock that pattern is prone to when a receiver can
/// legitimately go away without ever looping back to check it (see
/// `Pipeline`'s own `control_rx` field docs for that history). Callers
/// that need to know `run` has actually finished watch
/// [`DriverRunner::bus`] instead, same convention as `Pipeline`.
#[derive(Clone)]
pub struct StopReceiver {
    flag: Arc<AtomicBool>,
}

impl StopReceiver {
    pub fn is_stopped(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }
}

/// A background task with no `Sink`/`Source` ports of its own — nothing to
/// push into, nothing to pull out of *this* object; whatever it produces
/// or consumes happens through other `Sink`/`Source` pairs it mints on the
/// side (e.g. [`crate::elements::WebRtcPeer`] handing out
/// `WebRtcTrackSink`/`WebRtcTrackSource`). Reach for
/// [`crate::pipeline::Pipeline`]/[`crate::element::SourceElement`] instead
/// for anything that actually has a `src_pads()` dataflow graph to wire —
/// `Driver` deliberately has no `Pause`/`Seek`/`Clock`, none of which have
/// a sensible meaning for a connection that isn't part of one.
pub trait Driver: Element {
    /// Drives this task until it ends on its own or `stop.is_stopped()`
    /// says to abandon — check it periodically, the same spirit as
    /// [`crate::control::drain_control`] for a
    /// [`crate::element::SourceElement`]. `bus` is this task's own way to
    /// report a failure without necessarily ending itself over it — see
    /// [`crate::element::SourceElement::run`]'s docs for the same
    /// convention.
    fn run(&mut self, stop: &StopReceiver, bus: &Bus) -> Result<()>;
}

/// Runs a [`Driver`] on its own background thread — the `Driver` analog of
/// [`crate::pipeline::Pipeline`], minus everything that only makes sense
/// for a dataflow graph (`Clock`, `Pause`, `Seek`, the `wire` callback).
///
/// `run()` is asynchronous, same as `Pipeline::run`: it starts the driver
/// on a background thread and returns immediately. Watch
/// [`DriverRunner::bus`] to learn when it's actually done — draining it
/// blocks until every `Bus` sender has been dropped. The built-in drivers
/// keep that sender only for the duration of their background `run` call,
/// so this normally coincides with thread completion; a custom `Driver`
/// that clones and retains `bus` extends the wait until its clone drops.
pub struct DriverRunner {
    driver: Mutex<Option<Box<dyn Driver>>>,
    bus: Mutex<Option<Bus>>,
    stop_flag: Arc<AtomicBool>,
    bus_rx: BusReceiver,
    running: AtomicBool,
}

impl DriverRunner {
    pub fn new(driver: impl Driver + 'static) -> Arc<Self> {
        let (bus, bus_rx) = Bus::new();
        Arc::new(DriverRunner {
            driver: Mutex::new(Some(Box::new(driver))),
            bus: Mutex::new(Some(bus)),
            stop_flag: Arc::new(AtomicBool::new(false)),
            bus_rx,
            running: AtomicBool::new(false),
        })
    }

    pub fn bus(&self) -> &BusReceiver {
        &self.bus_rx
    }

    /// Starts driving the task on a background thread and returns
    /// immediately. A no-op if this `DriverRunner` is already running or
    /// has already finished a previous run — same posture as
    /// [`crate::pipeline::Pipeline::run`], not reusable afterward.
    pub fn run(self: &Arc<Self>) {
        let Some(mut driver) = self.driver.lock().unwrap().take() else {
            return;
        };
        let Some(bus) = self.bus.lock().unwrap().take() else {
            return;
        };

        self.running.store(true, Ordering::Release);
        let stop = StopReceiver {
            flag: self.stop_flag.clone(),
        };
        let this = Arc::clone(self);
        thread::Builder::new()
            .name("driver".into())
            .spawn(move || {
                let name = driver.name();
                let element_type = driver.element_type();
                if let Err(error) = driver.run(&stop, &bus) {
                    bus.post(
                        driver.hlog(),
                        BusEvent::Error {
                            element_type,
                            name,
                            error,
                        },
                    );
                }
                this.running.store(false, Ordering::Release);
            })
            .expect("failed to spawn driver thread");
    }

    /// Requests an early stop — see [`StopReceiver`]'s own docs for why
    /// this never blocks. A no-op if `run()` isn't currently in progress.
    pub fn stop(&self) {
        if !self.running.load(Ordering::Acquire) {
            return;
        }
        self.stop_flag.store(true, Ordering::Release);
    }
}
