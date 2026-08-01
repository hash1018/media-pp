use std::time::Duration;

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

/// One in-flight control request: the message plus a rendezvous channel
/// the receiver acks once it (and everything it cascaded into downstream)
/// has finished handling it — this is what makes
/// [`ControlSender::send`] synchronous. Fields are `pub(crate)` so
/// [`crate::queue::Queue`]'s worker loop can match on one directly out of
/// a `crossbeam_channel::select!` arm (which needs the raw `Receiver`,
/// not the [`ControlReceiver::try_recv`]/[`ControlReceiver::recv`]
/// wrappers used everywhere else).
pub(crate) struct Request {
    pub(crate) msg: ControlMsg,
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
        let (ack_tx, ack_rx) = crossbeam_channel::bounded(0);
        if self.tx.send(Request { msg, ack: ack_tx }).is_ok() {
            let _ = ack_rx.recv();
        }
    }
}

impl ControlReceiver {
    pub(crate) fn try_recv(&self) -> Option<(ControlMsg, Sender<()>)> {
        self.rx.try_recv().ok().map(|r| (r.msg, r.ack))
    }

    pub(crate) fn recv(&self) -> Option<(ControlMsg, Sender<()>)> {
        self.rx.recv().ok().map(|r| (r.msg, r.ack))
    }
}

/// Call once per loop iteration in a [`SourceElement::run`] implementation,
/// right before pulling the next unit of work — mirrors how a natural
/// `Eos` is pushed into the source's own pads at the end of that same
/// loop, just for externally-triggered control instead.
///
/// Drains every pending message (see [`apply_one`] for what "handling
/// one" means, including `Pause`'s blocking wait). Non-blocking if
/// nothing's pending — a [`SourceElement::run`] whose own "next unit of
/// work" can't be waited on via `control`'s own channel (e.g.
/// [`crate::elements::FileDemuxer`]'s blocking file read) calls this once
/// before that blocking step; one that *can* (e.g.
/// [`crate::elements::AppSource`]'s channel receive) selects on both
/// instead, calling [`apply_one`]/[`wait_out_pause`] directly so a
/// pending `Stop` is never left waiting behind a slow/absent producer.
///
/// Returns `true` if `Stop` was seen: the caller should return `Ok(())`
/// immediately, without pushing a final `Eos` (`Stop` means abandon, not
/// drain to completion).
pub fn drain_control<S: SourceElement>(
    control: &ControlReceiver,
    source: &mut S,
    bus: &Bus,
) -> Result<bool> {
    while let Some((msg, ack)) = control.try_recv() {
        if apply_one(source, bus, msg, &ack)? {
            return Ok(true);
        }
        if msg == ControlMsg::Pause && wait_out_pause(control, source, bus)? {
            return Ok(true);
        }
    }
    Ok(false)
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
    apply_seek(source, bus, msg)?;
    for pad in source.src_pads() {
        pad.control(msg)?;
    }
    let is_stop = msg == ControlMsg::Stop;
    let _ = ack.send(());
    Ok(is_stop)
}

/// Blocks on `control` alone — not whatever `source.run()` itself is
/// otherwise waiting on — until `Resume` or `Stop`, applying (and acking)
/// every message seen in between via [`apply_one`]. Returns `true` if
/// `Stop` ended it (including the sender simply going away, treated the
/// same as `Stop`); `false` once `Resume` arrives.
pub(crate) fn wait_out_pause<S: SourceElement>(
    control: &ControlReceiver,
    source: &mut S,
    bus: &Bus,
) -> Result<bool> {
    loop {
        let Some((msg, ack)) = control.recv() else {
            return Ok(true); // sender gone — treat like Stop
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
        bus.post(BusEvent::Seeked {
            element_type: source.element_type(),
            name: source.name(),
            requested: target,
            landed,
        });
    }
    Ok(())
}
