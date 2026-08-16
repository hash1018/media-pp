use std::time::{Duration, Instant};

use crate::pp_log::pp_trace;
use crossbeam_channel::{Receiver, Sender, unbounded};

use crate::{
    bus::{Bus, BusEvent},
    element::SourceElement,
    error::Result,
};

/// A command that can be sent down a running [`crate::pipeline::Pipeline`]
/// — travels the same pad-to-pad path `MediaBuffer` does (see
/// [`crate::element::Sink::control`]), but through a dedicated channel
/// instead of riding along as data: unlike `Eos`, it has to be able to
/// reach every element even mid-stream, and (for `Queue`) jump ahead of
/// whatever data is already backed up rather than wait in line behind it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlMsg {
    /// Freeze in place. Every [`crate::queue::Queue`] downstream stops
    /// pulling from its data channel until `Resume`/`Stop` — which also
    /// backpressures anything feeding it, since a full queue blocks the
    /// sender. Pairs with [`crate::clock::Clock::pause`], which
    /// [`crate::pipeline::Pipeline::pause`] calls at the same time so
    /// paced elements don't see a jump once resumed.
    Pause,
    /// Undoes `Pause`.
    Resume,
    /// Abandon immediately rather than draining to a natural `Eos` —
    /// whatever's in flight is dropped, not flushed. The pipeline isn't
    /// reusable afterward; build a new one for the next run.
    Stop,
    /// Jump to an absolute position from the start of the media.
    /// Handled in two parts, both inside [`drain_control`]: the source
    /// itself repositions via [`crate::element::SourceElement::seek`]
    /// *before* this is forwarded downstream, then the forward cascades
    /// as usual — a [`crate::queue::Queue`] drops whatever it has
    /// buffered (it predates the seek) instead of delivering it, and a
    /// decoder flushes its internal reference-frame state. Unlike
    /// `Pause`, this doesn't block waiting for anything further: it's a
    /// one-shot repositioning, not a state to later undo with `Resume`.
    Seek(Duration),
}

/// A request carried by a control channel. Ordinary controls cascade through
/// the graph immediately; `Finish` is source-only because graceful completion
/// must enter the graph as an ordered [`crate::buffer::MediaBuffer::Eos`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RequestKind {
    Control(ControlMsg),
    Finish,
}

/// One in-flight control request: the message plus a rendezvous channel
/// the receiver acks once it (and everything it cascaded into downstream)
/// has finished handling it — this is what makes
/// [`ControlSender::send`] synchronous. Fields are `pub(crate)` so
/// [`crate::queue::Queue`]'s worker loop can match on one directly out of
/// a `crossbeam_channel::select!` arm (which needs the raw `Receiver`,
/// not the [`ControlReceiver::try_recv`]/[`ControlReceiver::recv`]
/// wrappers used everywhere else).
pub(crate) struct Request {
    pub(crate) kind: RequestKind,
    pub(crate) ack: Sender<()>,
}

/// The sending half of a control channel — cloneable, cheap, `Send +
/// Sync`. [`crate::pipeline::Pipeline`] holds one to reach its source;
/// [`crate::queue::Queue`] holds one internally to reach its worker
/// thread across the thread boundary it owns.
#[derive(Clone)]
pub struct ControlSender {
    tx: Sender<Request>,
}

/// The receiving half — not `Clone` in spirit (only one thing should be
/// driving a given control channel at a time) but crossbeam's
/// `Receiver<T>` is a cheap shared handle under the hood, which is
/// exactly what [`crate::pipeline::Pipeline::run`] needs: it clones this
/// into a fresh worker thread on every call.
#[derive(Clone)]
pub struct ControlReceiver {
    pub(crate) rx: Receiver<Request>,
}

pub fn channel() -> (ControlSender, ControlReceiver) {
    let (tx, rx) = unbounded();
    (ControlSender { tx }, ControlReceiver { rx })
}

impl ControlSender {
    /// Sends `msg` and blocks until the receiver — and, transitively,
    /// everything downstream of it — has finished handling it. A no-op
    /// (returns immediately) if nothing is on the other end to receive it
    /// (e.g. the pipeline already finished).
    pub fn send(&self, msg: ControlMsg) {
        self.send_request(RequestKind::Control(msg));
    }

    /// Requests source-originated EOS without exposing `Finish` as a
    /// downstream [`ControlMsg`]. Used only by [`crate::pipeline::Pipeline`].
    pub(crate) fn finish(&self) {
        self.send_request(RequestKind::Finish);
    }

    fn send_request(&self, kind: RequestKind) {
        let (ack_tx, ack_rx) = crossbeam_channel::bounded(0);
        if self.tx.send(Request { kind, ack: ack_tx }).is_ok() {
            let _ = ack_rx.recv();
        }
    }
}

impl ControlReceiver {
    pub(crate) fn try_recv(&self) -> Option<(RequestKind, Sender<()>)> {
        self.rx.try_recv().ok().map(|r| (r.kind, r.ack))
    }

    pub(crate) fn recv(&self) -> Option<(RequestKind, Sender<()>)> {
        self.rx.recv().ok().map(|r| (r.kind, r.ack))
    }
}

/// What draining pending source requests actually did — whether `Stop` or
/// source-only `Finish` ended it, and how long (if any) was spent frozen
/// inside a `Pause`/`Resume` pair. A source built on wall-clock scheduling (an elapsed-time
/// budget like [`crate::elements::TestAudioSource`]/
/// [`crate::elements::AudioMixer`], or an absolute next-tick deadline like
/// [`crate::elements::TestVideoSource`]/`DxgiCaptureSource`)
/// has to fold `paused_for` back into its own schedule after every
/// [`drain_control`] call — real (`Instant`) time keeps moving during a
/// `Pause`, but the media timeline must not, or `Resume` would look like a
/// burst of catch-up work owed all at once.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ControlOutcome {
    /// `true` if either `Stop` or source-only `Finish` was seen: the caller
    /// should return `Ok(())` immediately. `Stop` abandons without EOS;
    /// `Finish` has already pushed ordered EOS from the source boundary.
    /// Keeping this terminal flag true for both also makes existing custom
    /// source loops honor the new graceful request without continuing to emit
    /// after EOS.
    pub stopped: bool,
    /// Wall-clock time from starting the synchronous downstream `Pause`
    /// cascade through finishing the matching `Resume` (or terminating
    /// `Stop`) cascade during this call — `Duration::ZERO` if no `Pause`
    /// was seen. Still meaningful
    /// even when `stopped` is `true` (the sender simply going away while
    /// paused is treated the same as `Stop`, see `wait_out_pause`), so a
    /// caller that also tracks its own paused-time total can fold this in
    /// unconditionally rather than only on the non-stopped path.
    pub paused_for: Duration,
}

/// Call once per loop iteration in a [`SourceElement::run`] implementation,
/// right before pulling the next unit of work — mirrors how a natural
/// `Eos` is pushed into the source's own pads at the end of that same
/// loop, just for externally-triggered control instead.
///
/// Drains every pending message (see `apply_one` for what "handling
/// one" means, including `Pause`'s blocking wait). Non-blocking if
/// nothing's pending — a [`SourceElement::run`] whose own "next unit of
/// work" can't be waited on via `control`'s own channel (e.g.
/// [`crate::elements::FileDemuxer`]'s blocking file read) calls this once
/// before that blocking step; one that *can* (e.g.
/// [`crate::elements::AppSource`]'s channel receive) selects on both
/// instead, calling `apply_one`/`wait_out_pause` directly so a
/// pending `Stop`/`Finish` is never left waiting behind a slow/absent producer —
/// same reason `WasapiCaptureSource` also drives the
/// raw receiver directly, to bracket the wait with resetting/restarting
/// its capture device rather than leaving it running unread through the
/// whole pause.
///
/// See [`ControlOutcome`] for what the return value means.
pub fn drain_control<S: SourceElement>(
    control: &ControlReceiver,
    source: &mut S,
    bus: &Bus,
) -> Result<ControlOutcome> {
    let mut paused_for = Duration::ZERO;
    while let Some((request, ack)) = control.try_recv() {
        let RequestKind::Control(msg) = request else {
            apply_finish(source, bus, &ack);
            return Ok(ControlOutcome {
                stopped: true,
                paused_for,
            });
        };
        if msg == ControlMsg::Pause {
            // Start measuring before forwarding Pause. `apply_one` is a
            // synchronous cascade and may itself spend substantial time
            // waiting for a busy Queue/Sink to become paused; the source
            // produces no media during that time, so it belongs to the
            // frozen interval just as much as the later wait for Resume.
            let pause_start = Instant::now();
            apply_one(source, bus, msg, &ack)?;
            let stopped = wait_out_pause(control, source, bus)?;
            paused_for += pause_start.elapsed();
            if stopped {
                return Ok(ControlOutcome {
                    stopped: true,
                    paused_for,
                });
            }
            continue;
        }
        if apply_one(source, bus, msg, &ack)? {
            return Ok(ControlOutcome {
                stopped: true,
                paused_for,
            });
        }
    }
    Ok(ControlOutcome {
        stopped: false,
        paused_for,
    })
}

/// Applies one source-only graceful completion request. Unlike
/// [`apply_one`], this never calls `Sink::control`: EOS has to sit behind every
/// already-produced buffer in each data path so queues and stateful elements
/// drain in order.
pub(crate) fn apply_finish<S: SourceElement>(source: &mut S, bus: &Bus, ack: &Sender<()>) {
    pp_trace!(
        pp_log: source.pp_log(),
        "event=finish phase=received"
    );
    let pp_log = source.pp_log().clone();
    let element_type = source.element_type();
    let name = source.name();
    for pad in source.src_pads() {
        if let Err(error) = pad.push_eos(&pp_log) {
            bus.post(
                &pp_log,
                BusEvent::Error {
                    element_type,
                    name: name.clone(),
                    error,
                },
            );
        }
    }
    let _ = ack.send(());
    pp_trace!(
        pp_log: source.pp_log(),
        "event=finish phase=completed outcome=ok"
    );
}

/// Applies one already-received control message to `source`: repositions
/// it first on `Seek` (see [`apply_seek`]), then forwards `msg` to every
/// one of `source`'s pads (so it cascades through the graph exactly like
/// a data buffer would), then acks. Returns `true` for `Stop` — same
/// meaning as [`drain_control`]'s own return.
pub(crate) fn apply_one<S: SourceElement>(
    source: &mut S,
    bus: &Bus,
    msg: ControlMsg,
    ack: &Sender<()>,
) -> Result<bool> {
    let is_stop = apply_one_unacked(source, bus, msg)?;
    let _ = ack.send(());
    Ok(is_stop)
}

/// The forwarding half of [`apply_one`], split out for a source that must
/// finish source-local state changes before the synchronous request is
/// acknowledged. [`crate::elements::WasapiCaptureSource`] uses this for
/// `Resume`: downstream is resumed first, then its capture device is
/// restarted, and only then may the caller observe the request as done.
pub(crate) fn apply_one_unacked<S: SourceElement>(
    source: &mut S,
    bus: &Bus,
    msg: ControlMsg,
) -> Result<bool> {
    pp_trace!(
        pp_log: source.pp_log(),
        "event=control control={msg:?} phase=received"
    );
    let result: Result<bool> = (|| {
        apply_seek(source, bus, msg)?;
        for pad in source.src_pads() {
            pad.control(msg)?;
        }
        Ok(msg == ControlMsg::Stop)
    })();
    match &result {
        Ok(_) => pp_trace!(
            pp_log: source.pp_log(),
            "event=control control={msg:?} phase=completed outcome=ok"
        ),
        Err(error) => pp_trace!(
            pp_log: source.pp_log(),
            "event=control control={msg:?} phase=completed outcome=error error={error}"
        ),
    }
    result
}

/// Blocks on `control` alone — not whatever `source.run()` itself is
/// otherwise waiting on — until `Resume`, `Stop`, or `Finish`, applying (and
/// acking) every request seen in between. Returns `true` if `Stop`/`Finish`
/// ended it (including the sender simply going away, treated the same as
/// `Stop`); `false` once `Resume` arrives.
pub(crate) fn wait_out_pause<S: SourceElement>(
    control: &ControlReceiver,
    source: &mut S,
    bus: &Bus,
) -> Result<bool> {
    loop {
        let Some((request, ack)) = control.recv() else {
            return Ok(true); // sender gone — treat like Stop
        };
        let RequestKind::Control(msg) = request else {
            apply_finish(source, bus, &ack);
            return Ok(true);
        };
        if apply_one(source, bus, msg, &ack)? {
            return Ok(true);
        }
        if msg == ControlMsg::Resume {
            return Ok(false);
        }
        // Another Pause while already paused: already forwarded above
        // (harmless no-op downstream), just keep waiting.
    }
}

/// `Seek`'s source-specific half of `drain_control` — repositions
/// `source` (see [`SourceElement::seek`]) and reports where it actually
/// landed via [`BusEvent::Seeked`], since that can differ from what was
/// requested. No-op for every other [`ControlMsg`].
fn apply_seek<S: SourceElement>(source: &mut S, bus: &Bus, msg: ControlMsg) -> Result<()> {
    if let ControlMsg::Seek(target) = msg {
        let landed = source.seek(target)?;
        bus.post(
            source.pp_log(),
            BusEvent::Seeked {
                element_type: source.element_type(),
                name: source.name(),
                requested: target,
                landed,
            },
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, thread};

    use crate::pp_log::PpLog;

    use super::*;
    use crate::{
        buffer::MediaBuffer,
        element::{Element, ElementType, Sink, Source, element_pp_log},
        pad::SrcPad,
    };

    /// A `SourceElement` with no real I/O — just enough surface for
    /// `drain_control`/`wait_out_pause` to drive, since this module's own
    /// logic doesn't care what the source actually produces.
    struct DummySource {
        pp_log: PpLog,
        pad: SrcPad,
    }

    impl DummySource {
        fn new() -> Self {
            Self {
                pp_log: element_pp_log(ElementType::Other, "dummy", None),
                pad: SrcPad::new("dummy_src"),
            }
        }
    }

    impl Element for DummySource {
        fn name(&self) -> Arc<str> {
            "dummy".into()
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

    impl Source for DummySource {
        fn src_pads(&mut self) -> &mut [SrcPad] {
            std::slice::from_mut(&mut self.pad)
        }
    }

    impl SourceElement for DummySource {
        fn run(&mut self, _control: &ControlReceiver, _bus: &Bus) -> Result<()> {
            unreachable!("not exercised by these tests")
        }

        fn seek(&mut self, target: Duration) -> Result<Duration> {
            Ok(target)
        }
    }

    struct SlowPauseSink {
        pp_log: PpLog,
        pause_delay: Duration,
    }

    impl Element for SlowPauseSink {
        fn name(&self) -> Arc<str> {
            "slow-pause".into()
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

    impl Sink for SlowPauseSink {
        fn consume(&mut self, _buf: MediaBuffer) -> Result<()> {
            Ok(())
        }

        fn control(&mut self, msg: ControlMsg) -> Result<()> {
            if msg == ControlMsg::Pause {
                thread::sleep(self.pause_delay);
            }
            Ok(())
        }
    }

    /// The edge case called out in `wait_out_pause`'s own docs: the
    /// `ControlSender` going away entirely (e.g. the owning `Pipeline`
    /// dropped) while paused has to be treated the same as an explicit
    /// `Stop`, not left blocking forever on a channel nothing will ever
    /// send on again.
    #[test]
    fn wait_out_pause_treats_a_dropped_sender_as_stop() {
        let (tx, rx) = channel();
        drop(tx);

        let (bus, _bus_rx) = Bus::new();
        let mut source = DummySource::new();

        let stopped = wait_out_pause(&rx, &mut source, &bus)
            .expect("no real seek/push happens on this path, so this can't fail");
        assert!(
            stopped,
            "a dropped ControlSender must be treated the same as an explicit Stop"
        );
    }

    /// `wait_out_pause` blocks past any number of redundant `Pause`s and
    /// only returns (`Ok(false)`, meaning "keep running") once `Resume`
    /// actually arrives.
    #[test]
    fn wait_out_pause_blocks_until_resume_then_returns_false() {
        let (tx, rx) = channel();
        let (bus, _bus_rx) = Bus::new();
        let mut source = DummySource::new();

        let worker = thread::spawn(move || wait_out_pause(&rx, &mut source, &bus));

        // A redundant Pause while already paused: per `wait_out_pause`'s
        // own docs, forwarded (harmless no-op downstream) and then it
        // keeps waiting rather than returning.
        tx.send(ControlMsg::Pause);
        tx.send(ControlMsg::Resume);

        let stopped = worker
            .join()
            .expect("worker must not panic")
            .expect("no real seek/push happens on this path, so this can't fail");
        assert!(
            !stopped,
            "Resume must unblock wait_out_pause with Ok(false)"
        );
    }

    /// `paused_for` starts when the source begins forwarding Pause, not
    /// only after every downstream element has finally acknowledged it.
    /// Otherwise a slow control cascade is miscounted as playable media
    /// time and an elapsed-time source catches that interval up as a burst.
    #[test]
    fn drain_control_counts_the_pause_cascade_as_paused_time() {
        let pause_delay = Duration::from_millis(80);
        let (tx, rx) = channel();
        let controller = thread::spawn(move || {
            tx.send(ControlMsg::Pause);
            tx.send(ControlMsg::Resume);
        });

        let (bus, _bus_rx) = Bus::new();
        let mut source = DummySource::new();
        source.pad.link(Box::new(SlowPauseSink {
            pause_delay,
            pp_log: element_pp_log(ElementType::Other, "slow-pause", None),
        }));

        let outcome = loop {
            let outcome = drain_control(&rx, &mut source, &bus)
                .expect("the synthetic control cascade cannot fail");
            if outcome.paused_for > Duration::ZERO {
                break outcome;
            }
            thread::yield_now();
        };
        controller.join().expect("controller must not panic");

        assert!(!outcome.stopped);
        assert!(
            outcome.paused_for >= Duration::from_millis(60),
            "the {:?} Pause cascade was omitted from paused_for: {:?}",
            pause_delay,
            outcome.paused_for
        );
    }
}
