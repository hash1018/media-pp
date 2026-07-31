use std::time::Duration;

use crossbeam_channel::{Receiver, Sender, unbounded};

use crate::{element::SourceElement, error::Result};

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
/// Drains every pending message. `Seek` repositions `source` itself
/// first (see [`SourceElement::seek`]) — that's the one case a message
/// needs source-specific action, not just forwarding, which is why this
/// takes the whole source rather than just its pads. Every message is
/// then forwarded to every one of `source`'s pads (so it cascades through
/// the graph exactly like a data buffer would). `Pause` blocks right
/// here — still watching `control`, not producing anything — until
/// `Resume` or `Stop` arrives, so nothing upstream of this call keeps
/// running while paused either.
///
/// Returns `true` if `Stop` was seen: the caller should return `Ok(())`
/// immediately, without pushing a final `Eos` (`Stop` means abandon, not
/// drain to completion).
pub fn drain_control<S: SourceElement>(control: &ControlReceiver, source: &mut S) -> Result<bool> {
    while let Some((msg, ack)) = control.try_recv() {
        if let ControlMsg::Seek(target) = msg {
            source.seek(target)?;
        }
        for pad in source.src_pads() {
            pad.control(msg)?;
        }
        let is_stop = msg == ControlMsg::Stop;
        let _ = ack.send(());
        if is_stop {
            return Ok(true);
        }
        if msg == ControlMsg::Pause {
            loop {
                let Some((msg, ack)) = control.recv() else {
                    return Ok(true); // sender gone — treat like Stop
                };
                if let ControlMsg::Seek(target) = msg {
                    source.seek(target)?;
                }
                for pad in source.src_pads() {
                    pad.control(msg)?;
                }
                let is_stop = msg == ControlMsg::Stop;
                let _ = ack.send(());
                if is_stop {
                    return Ok(true);
                }
                if msg == ControlMsg::Resume {
                    break;
                }
                // Another Pause while already paused: already forwarded
                // above (harmless no-op downstream), just keep waiting.
            }
        }
    }
    Ok(false)
}
