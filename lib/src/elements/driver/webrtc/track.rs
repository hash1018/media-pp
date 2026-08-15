use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use crate::clog::{CLog, cerror, cinfo};
use crossbeam_channel::{Receiver, Sender, TrySendError, select};
use str0m::{
    change::{SdpAnswer, SdpOffer},
    format::Codec,
    media::{Direction, MediaKind, Mid},
};

use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    control::{ControlMsg, ControlReceiver, apply_one, drain_control, wait_out_pause},
    element::{Element, ElementType, Sink, Source, SourceElement, element_clog},
    error::Result,
    pad::SrcPad,
};

use super::command::{Command, TrackId, WebRtcError};

/// Cheaply-cloneable handle for requesting new tracks, completing
/// renegotiation, and picking up newly-attached tracks — same spirit as
/// [`crate::elements::AppSourceHandle`]. Cloning shares one queue of
/// pending [`WebRtcHandle::next_track`] results, same as any other
/// multi-consumer channel — only one clone's call actually receives a
/// given track, so in practice only one place in the app should be
/// draining it.
#[derive(Clone)]
pub struct WebRtcHandle {
    pub(super) next_id: Arc<AtomicU64>,
    pub(super) command_tx: Sender<Command>,
    pub(super) new_track_rx:
        Receiver<(TrackId, Mid, MediaKind, WebRtcTrackSink, WebRtcTrackSource)>,
}

impl WebRtcHandle {
    /// Requests a new track of `kind`/`direction`. Blocks only while the
    /// peer's bounded command queue is full; once the command is accepted,
    /// returns the locally assigned [`TrackId`]. This does not mean SDP
    /// negotiation has completed — receive the attached track through
    /// [`WebRtcHandle::next_track`]. Returns [`WebRtcError::Closed`] without
    /// yielding a `TrackId` if the peer loop has already stopped.
    ///
    /// `codec` is what [`WebRtcTrackSink::consume`] on the resulting track
    /// will actually be fed (an encoder's output, or a packet relayed
    /// verbatim from another track) — used to pick the matching payload
    /// type out of whatever this connection negotiates for the track,
    /// instead of guessing. If this connection never negotiates `codec` for
    /// it, pushed buffers are silently dropped, same as an unopened track.
    pub fn add_track(
        &self,
        kind: MediaKind,
        direction: Direction,
        codec: Codec,
    ) -> Result<TrackId> {
        let id = TrackId(self.next_id.fetch_add(1, Ordering::Relaxed));
        self.command_tx
            .send(Command::AddTrack(id, kind, direction, codec))
            .map_err(|_| WebRtcError::Closed)?;
        Ok(id)
    }

    /// Blocks until the next track attaches — either one requested via
    /// [`WebRtcHandle::add_track`] (on either side) once its `Mid` exists,
    /// or one the remote peer added on its own. Returns the `TrackId` (so
    /// the caller can match it against what `add_track` returned — a `Mid`
    /// alone doesn't exist yet at `add_track` time, see [`TrackId`]'s own
    /// docs) alongside the `Mid`/`MediaKind` str0m assigned it and the
    /// `WebRtcTrackSink`/`WebRtcTrackSource` pair to send/receive on it.
    /// `Err` once `WebRtcPeer` (and its `run`) is gone and every
    /// already-attached track has been drained.
    pub fn next_track(
        &self,
    ) -> Result<(TrackId, Mid, MediaKind, WebRtcTrackSink, WebRtcTrackSource)> {
        self.new_track_rx
            .recv()
            .map_err(|_| WebRtcError::Closed.into())
    }

    /// Feeds a remote answer back in, completing a renegotiation started by
    /// [`WebRtcHandle::add_track`]. A no-op if `WebRtcPeer` (and its `run`)
    /// is already gone.
    pub fn set_answer(&self, answer: SdpAnswer) {
        let _ = self.command_tx.send(Command::SetAnswer(answer));
    }

    /// Accepts a fresh offer from the *remote* peer (their own
    /// renegotiation) and returns the resulting answer for the caller to
    /// ship back over its own signaling transport. Blocks until
    /// `WebRtcPeer::run` has actually applied it.
    pub fn accept_remote_offer(&self, offer: SdpOffer) -> Result<SdpAnswer> {
        let (reply_tx, reply_rx) = crossbeam_channel::bounded(0);
        self.command_tx
            .send(Command::AcceptOffer(offer, reply_tx))
            .map_err(|_| WebRtcError::Closed)?;
        reply_rx
            .recv()
            .map_err(|_| WebRtcError::Closed)?
            .map_err(Into::into)
    }
}

/// One outbound track. A plain [`Sink`] — no bespoke push API, it links
/// into a [`crate::pipeline::ChainBuilder`] exactly like
/// [`crate::elements::RtspServer`] or any other terminal sink.
/// `consume()` only ever hands off to `WebRtcPeer::run`'s own thread via a
/// channel send; the actual str0m write happens over there.
pub struct WebRtcTrackSink {
    clog: CLog,
    id: TrackId,
    command_tx: Sender<Command>,
}

impl WebRtcTrackSink {
    pub(super) fn new(id: TrackId, command_tx: Sender<Command>) -> Self {
        Self {
            id,
            command_tx,
            clog: element_clog(
                ElementType::WebRtcPeer,
                &format!("webrtc-track-{}", id.0),
                None,
            ),
        }
    }
}

impl Element for WebRtcTrackSink {
    fn name(&self) -> Arc<str> {
        format!("webrtc-track-{}", self.id.0).into()
    }

    fn element_type(&self) -> ElementType {
        ElementType::WebRtcPeer
    }

    fn clog(&self) -> &CLog {
        &self.clog
    }

    fn clog_mut(&mut self) -> &mut CLog {
        &mut self.clog
    }
}

impl Sink for WebRtcTrackSink {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        if !matches!(buf, MediaBuffer::Packet(_) | MediaBuffer::Eos) {
            let kind = match buf {
                MediaBuffer::Video(_) => "Video",
                MediaBuffer::Audio(_) => "Audio",
                MediaBuffer::Packet(_) | MediaBuffer::Eos => unreachable!("matched above"),
            };
            cerror!(self, "unsupported buffer: {kind}");
            return Err(WebRtcError::UnsupportedBuffer(kind).into());
        }
        // `WebRtcPeer::run` gone (channel disconnected) means this track is
        // dead — surface it as `Err` rather than swallowing it, so whatever
        // pipeline this `Sink` is plugged into (its own `Queue`, its own
        // `Bus`) actually learns about it instead of silently sending into
        // a void forever. Non-fatal by the same convention as any other
        // `Sink::consume` failure (see `Queue`'s own docs) — just no longer
        // an invisible one.
        //
        // A full channel (`WebRtcPeer::run` backed up) drops the newest
        // buffer instead — same as an unopened track (see `add_track`'s
        // docs) — but isn't reported on a `Bus`: unlike `WebRtcPeer::run`,
        // which only ever borrows a `Bus` for the duration of one `run()`
        // call, `WebRtcTrackSink` is a handle the caller can keep past
        // `Driver::stop()`, so storing one here would keep that `Bus`'s
        // channel open indefinitely — including past whatever's waiting on
        // `BusReceiver::iter()` to finish once every sender is gone.
        match self.command_tx.try_send(Command::Push(self.id, buf)) {
            Ok(()) | Err(TrySendError::Full(_)) => Ok(()),
            Err(TrySendError::Disconnected(_)) => {
                cerror!(self, "WebRtcPeer::run gone — track is dead");
                Err(WebRtcError::Closed.into())
            }
        }
    }

    fn control(&mut self, _msg: ControlMsg) -> Result<()> {
        // Terminal, same as AppSink/RtspServer: nothing buffered or
        // downstream to flush/forward for any ControlMsg.
        Ok(())
    }
}

/// One inbound track — the mirror image of [`WebRtcTrackSink`]. A plain
/// [`SourceElement`], same shape as [`crate::elements::AppSource`]: it
/// links into its own [`crate::pipeline::Pipeline`] via `src_pads()` like
/// any other source. The difference from `AppSource` is only *who* feeds
/// it — instead of an [`crate::elements::AppSourceHandle`] the app calls
/// itself, [`crate::driver::Driver::run`] pushes into the sending half of this same
/// channel internally, from its own thread, for every `Event::MediaData`
/// on this track's `Mid`. Nothing here ever calls back into caller-supplied
/// code from `WebRtcPeer::run`'s own thread — that thread only ever touches
/// this crate's own types (see the module docs for why `WebRtcPeer` hands
/// tracks out through [`WebRtcHandle::next_track`] instead of a callback).
pub struct WebRtcTrackSource {
    clog: CLog,
    name: Arc<str>,
    pad: SrcPad,
    data_rx: Receiver<MediaBuffer>,
    codec: Arc<Mutex<Option<Codec>>>,
}

impl WebRtcTrackSource {
    pub(super) fn new(
        name: impl Into<String>,
        data_rx: Receiver<MediaBuffer>,
        codec: Arc<Mutex<Option<Codec>>>,
    ) -> Self {
        let name: Arc<str> = name.into().into();
        let clog = element_clog(ElementType::WebRtcPeer, &name, None);
        let pad = SrcPad::new(format!("{name}_src"));
        Self {
            name,
            clog,
            pad,
            data_rx,
            codec,
        }
    }

    /// The codec this track is actually carrying, as seen on the most
    /// recently received packet's RTP payload type — `None` until the
    /// first one arrives. Unlike [`WebRtcHandle::add_track`]'s `codec`
    /// (which the *caller* declares up front for an outbound track), an
    /// inbound track's codec isn't knowable ahead of time: SDP negotiation
    /// can accept several codecs for one `m=` line, and only the packets
    /// actually arriving say which one the remote side picked (see
    /// `Event::MediaData`'s own `params` field). Whatever's downstream
    /// (e.g. a decoder) needs a keyframe before it can do anything useful
    /// anyway, so waiting for the first packet to learn the codec isn't an
    /// extra constraint in practice.
    pub fn codec(&self) -> Option<Codec> {
        *self.codec.lock().unwrap()
    }
}

impl Element for WebRtcTrackSource {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::WebRtcPeer
    }

    fn clog(&self) -> &CLog {
        &self.clog
    }

    fn clog_mut(&mut self) -> &mut CLog {
        &mut self.clog
    }
}

impl Source for WebRtcTrackSource {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl SourceElement for WebRtcTrackSource {
    /// Identical shape to [`crate::elements::AppSource::run`]: selects on
    /// `control` and its own data channel together, so `Stop`/`Pause`
    /// never wait behind a remote peer that's gone quiet. The data channel
    /// disconnecting — `WebRtcPeer` gone, whether from `Stop` or the
    /// connection dying on its own — ends this the same way `AppSource`
    /// ends when every `AppSourceHandle` is dropped: one final `Eos`, no
    /// error.
    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        cinfo!(self, "run: starting");
        loop {
            if drain_control(control, self, bus)?.stopped {
                cinfo!(self, "run: stopped");
                return Ok(());
            }

            select! {
                recv(control.rx) -> req => {
                    match req {
                        Ok(req) => {
                            if apply_one(self, bus, req.msg, &req.ack)? {
                                cinfo!(self, "run: stopped");
                                return Ok(());
                            }
                            if req.msg == ControlMsg::Pause
                                && wait_out_pause(control, self, bus)?
                            {
                                cinfo!(self, "run: stopped");
                                return Ok(());
                            }
                        }
                        // The Pipeline itself is gone — nothing left to drive this.
                        Err(_) => {
                            cinfo!(self, "run: control channel gone, ending");
                            return Ok(());
                        }
                    }
                }
                recv(self.data_rx) -> buf => {
                    match buf {
                        Ok(buf) if buf.is_eos() => {
                            cinfo!(self, "run: reached eos");
                            break;
                        }
                        Ok(buf) => {
                            if let Err(error) = self.pad.push(buf) {
                                bus.post(
                                    &self.clog,
                                    BusEvent::Error {
                                        element_type: ElementType::WebRtcPeer,
                                        name: self.name.clone(),
                                        error,
                                    },
                                );
                            }
                        }
                        // `WebRtcPeer` gone — this track (or the whole peer) is done.
                        Err(_) => {
                            cinfo!(self, "run: WebRtcPeer gone, ending");
                            break;
                        }
                    }
                }
            }
        }
        // The data channel ending (above) can race a `Stop` sent at the
        // same moment — e.g. stopping the *upstream* `WebRtcPeer` (via its
        // `DriverRunner`) disconnects this exact channel, and a caller
        // stopping this `Pipeline` too, right after, can land its `Stop` in
        // `control`'s queue after `select!` already picked the data arm.
        // Ack it (a no-op otherwise) so `ControlSender::send`'s rendezvous
        // never blocks forever waiting for an ack this thread would
        // otherwise never get around to sending.
        while let Some((_msg, ack)) = control.try_recv() {
            let _ = ack.send(());
        }
        self.pad.push(MediaBuffer::Eos)
    }

    /// No timeline of its own — same reasoning as
    /// [`crate::elements::AppSource::seek`]: a WebRTC connection has
    /// nothing to reposition.
    fn seek(&mut self, target: Duration) -> Result<Duration> {
        Ok(target)
    }
}
