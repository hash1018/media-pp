use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use crate::pp_log::{PpLog, pp_error, pp_info};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, TrySendError, select};
use str0m::{
    change::{SdpAnswer, SdpOffer},
    format::Codec,
    media::{Direction, MediaKind, Mid},
};

use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    contract::{InputContract, OutputContract, PortContract},
    control::{
        ControlMsg, ControlReceiver, RequestKind, apply_finish, apply_one, drain_control,
        wait_out_pause,
    },
    element::{Element, ElementType, Sink, Source, SourceElement, element_pp_log},
    error::Result,
    pad::SrcPad,
};

use super::{
    command::{Command, TrackId, WebRtcError},
    stream_info::WebRtcStreamInfo,
};

/// The encoded kind a str0m track of `kind` carries. Both media flow
/// through `MediaBuffer::Packet`, so this is the only thing that tells an
/// audio track apart from a video one at wiring time.
fn packet_kind(kind: MediaKind) -> crate::contract::MediaKind {
    match kind {
        MediaKind::Audio => crate::contract::MediaKind::AudioPacket,
        MediaKind::Video => crate::contract::MediaKind::VideoPacket,
    }
}

/// The endpoints a track actually has, which is exactly what its
/// negotiated [`Direction`] allows — a `SendOnly` track carries no
/// `WebRtcTrackSource` because nothing will ever arrive on it, and a
/// `RecvOnly` one carries no [`WebRtcTrackSink`] because str0m has no
/// send capability for it.
///
/// The variant *is* the direction, so there is no separate field the two
/// could disagree with. Pushing into a sink that does not exist, or
/// waiting on a source that does not, is a compile error rather than
/// something that silently does nothing.
///
/// Fixed for the life of the track: these are handed out once, when the
/// track attaches, and a remote peer that later renegotiates a different
/// direction is reported on the [`Bus`] instead (see
/// [`WebRtcHandle::next_track`]).
pub enum TrackEndpoints {
    /// `Direction::SendOnly` — outbound only.
    Send(WebRtcTrackSink),
    /// `Direction::RecvOnly` — inbound only.
    Recv(WebRtcTrackSource),
    /// `Direction::SendRecv` — both, on the one track.
    SendRecv(WebRtcTrackSink, WebRtcTrackSource),
    /// `Direction::Inactive` — neither, for now. Still handed out: the
    /// track exists and its `mid` is negotiated, so a caller matching
    /// attachments against its own [`WebRtcHandle::add_track`] calls has
    /// to see it.
    Inactive,
}

/// One newly-attached track, from [`WebRtcHandle::next_track`].
pub struct AttachedTrack {
    /// Matches what [`WebRtcHandle::add_track`] returned for a track this
    /// side requested. A track the *remote* peer added has an id issued
    /// here that the caller has never seen before — which is how the two
    /// are told apart.
    pub id: TrackId,
    /// The `mid` str0m assigned during the SDP exchange.
    pub mid: Mid,
    /// Audio or video.
    pub kind: MediaKind,
    /// What can actually be done with this track — see [`TrackEndpoints`].
    pub endpoints: TrackEndpoints,
}

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
    pub(super) new_track_rx: Receiver<AttachedTrack>,
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
    /// instead of guessing. If this connection does not negotiate `codec`,
    /// consuming a packet returns
    /// [`WebRtcError::OutboundCodecNotNegotiated`].
    ///
    /// Declaring one codec and pushing another is not detected anywhere:
    /// str0m packetizes whatever bytes it is handed under the payload type
    /// chosen here, so the mismatch leaves as a well-formed stream that no
    /// receiver can decode. Audio is where this bites — WebRTC negotiates
    /// Opus, and there is no AAC payload type to fall back to.
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
    /// or one the remote peer added on its own. `Err` once `WebRtcPeer`
    /// (and its `run`) is gone and every already-attached track has been
    /// drained.
    ///
    /// Which track this is has to be established from
    /// [`AttachedTrack::id`]: a remote peer adding a track of its own is
    /// delivered through this same queue, so "the call right after my
    /// `add_track`" is not a guarantee of anything. Match the id against
    /// what `add_track` returned.
    ///
    /// [`AttachedTrack::endpoints`] carries only what the track's
    /// negotiated direction actually permits. That direction is read once,
    /// as the track attaches, and the endpoints are never re-issued — so a
    /// remote peer that renegotiates a different direction afterwards
    /// makes them wrong. That case is reported as
    /// [`WebRtcError::DirectionChanged`] on the [`Bus`] rather than
    /// silently tolerated; recovering from it means tearing the track down
    /// and adding a new one.
    ///
    /// Both endpoints expose their currently negotiated codec lists. A send
    /// endpoint for a track this side requested already selects the codec
    /// passed to [`WebRtcHandle::add_track`]. For a track the remote side
    /// added, choose the application's encoder output from
    /// [`WebRtcTrackSink::negotiated_codecs`] and pass it to
    /// [`WebRtcTrackSink::set_codec`] before pushing packets. The matching
    /// source separately reports the codec actually received once RTP starts.
    pub fn next_track(&self) -> Result<AttachedTrack> {
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
/// [`crate::elements::RtspSink`] or any other terminal sink.
/// `consume()` only ever hands off to `WebRtcPeer::run`'s own thread via a
/// channel send; the actual str0m write happens over there.
///
/// Its negotiated codec capabilities are available immediately through
/// [`WebRtcTrackSink::negotiated_codecs`]. The outbound selection is initialized
/// automatically for a track created by [`WebRtcHandle::add_track`]. A
/// send-capable track added by the remote peer instead requires one validated
/// [`WebRtcTrackSink::set_codec`] call before packets are consumed; omitting it
/// returns a typed error rather than guessing an RTP payload type.
pub struct WebRtcTrackSink {
    pp_log: PpLog,
    id: TrackId,
    /// What this track was negotiated to carry — audio or video.
    kind: MediaKind,
    codec: Option<Codec>,
    negotiated_codecs: Arc<Mutex<Vec<Codec>>>,
    command_tx: Sender<Command>,
    /// libavcodec may express encoder delay as a negative first PTS (Opus is
    /// a common example), while RTP media time is unsigned. The first packet
    /// establishes one track-wide shift so relative timing is preserved.
    timestamp_offset: Option<i64>,
}

impl WebRtcTrackSink {
    pub(super) fn new(
        id: TrackId,
        kind: MediaKind,
        codec: Option<Codec>,
        negotiated_codecs: Arc<Mutex<Vec<Codec>>>,
        command_tx: Sender<Command>,
    ) -> Self {
        Self {
            id,
            kind,
            codec,
            negotiated_codecs,
            command_tx,
            timestamp_offset: None,
            pp_log: element_pp_log(
                ElementType::WebRtcPeer,
                &format!("webrtc-track-{}", id.0),
                None,
            ),
        }
    }

    /// Returns the distinct codec families this track can currently send
    /// after SDP negotiation. The order is informational; select the codec
    /// produced by the application's encoder.
    ///
    /// A locally-created endpoint is handed out while its offer is still
    /// pending, so its initial value is the offered list and is narrowed when
    /// [`WebRtcHandle::set_answer`] applies the answer. A remotely-created
    /// endpoint is already negotiated when it is handed out.
    pub fn negotiated_codecs(&self) -> Vec<Codec> {
        self.negotiated_codecs.lock().unwrap().clone()
    }

    /// Declares the codec carried by packets pushed into this sink.
    ///
    /// A sink returned for this side's own [`WebRtcHandle::add_track`] call
    /// is initialized from that call's `codec`. A send-capable track added
    /// by the remote peer cannot be initialized automatically: one SDP media
    /// section can negotiate several codecs, and only this application knows
    /// which encoder feeds its outbound half. Call this before pushing a
    /// packet into such a sink; the choice is validated against
    /// [`WebRtcTrackSink::negotiated_codecs`]. If no choice is made,
    /// [`Sink::consume`] returns [`WebRtcError::OutboundCodecNotDeclared`]
    /// instead of guessing a payload type and emitting a mislabeled RTP
    /// stream.
    ///
    /// Returns [`WebRtcError::OutboundCodecNotNegotiated`] without changing
    /// the previous selection when `codec` is unavailable. Already-enqueued
    /// packets retain the declaration they were submitted with.
    pub fn set_codec(&mut self, codec: Codec) -> Result<()> {
        let negotiated = self.negotiated_codecs();
        if !negotiated.contains(&codec) {
            return Err(WebRtcError::OutboundCodecNotNegotiated {
                track_id: self.id,
                codec,
                negotiated,
            }
            .into());
        }
        self.codec = Some(codec);
        Ok(())
    }
}

impl Element for WebRtcTrackSink {
    fn name(&self) -> Arc<str> {
        format!("webrtc-track-{}", self.id.0).into()
    }

    fn element_type(&self) -> ElementType {
        ElementType::WebRtcPeer
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Sink for WebRtcTrackSink {
    /// A track carries encoded media to the peer; this sink has no
    /// encoder of its own, so a decoded frame has no route through it.
    fn input_contract(&self) -> InputContract {
        // Path-qualified: `MediaKind` in this module is str0m's own.
        InputContract::Fixed(PortContract::packet(packet_kind(self.kind)))
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        if !matches!(buf, MediaBuffer::Packet(_) | MediaBuffer::Eos) {
            let kind = match buf {
                MediaBuffer::Video(_) => "Video",
                MediaBuffer::Audio(_) => "Audio",
                MediaBuffer::Packet(_) | MediaBuffer::Eos => unreachable!("matched above"),
            };
            pp_error!(self, "unsupported buffer: {kind}");
            return Err(WebRtcError::UnsupportedBuffer(kind).into());
        }
        if matches!(buf, MediaBuffer::Packet(_)) && self.codec.is_none() {
            pp_error!(self, "outbound codec is not declared");
            return Err(WebRtcError::OutboundCodecNotDeclared(self.id).into());
        }
        if let MediaBuffer::Packet(_) = &buf {
            let codec = self.codec.expect("checked above");
            let negotiated = self.negotiated_codecs();
            if !negotiated.contains(&codec) {
                pp_error!(self, "outbound codec {codec:?} is not negotiated");
                return Err(WebRtcError::OutboundCodecNotNegotiated {
                    track_id: self.id,
                    codec,
                    negotiated,
                }
                .into());
            }
        }
        let buf = self.normalize_packet_timestamp(buf)?;
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
        match self
            .command_tx
            .try_send(Command::Push(self.id, self.codec, buf))
        {
            Ok(()) | Err(TrySendError::Full(_)) => Ok(()),
            Err(TrySendError::Disconnected(_)) => {
                pp_error!(self, "WebRtcPeer::run gone — track is dead");
                Err(WebRtcError::Closed.into())
            }
        }
    }

    fn control(&mut self, _msg: ControlMsg) -> Result<()> {
        // Terminal, same as AppSink/RtspSink: nothing buffered or
        // downstream to flush/forward for any ControlMsg.
        Ok(())
    }
}

impl WebRtcTrackSink {
    fn normalize_packet_timestamp(&mut self, buf: MediaBuffer) -> Result<MediaBuffer> {
        let MediaBuffer::Packet(packet) = buf else {
            return Ok(buf);
        };
        let Some(pts) = packet.pts() else {
            return Ok(MediaBuffer::Packet(packet));
        };
        let offset = match self.timestamp_offset {
            Some(offset) => offset,
            None if pts < 0 => {
                pts.checked_neg()
                    .ok_or(WebRtcError::PacketTimestampNormalizationOverflow {
                        value: pts,
                        offset: 0,
                    })?
            }
            None => 0,
        };
        self.timestamp_offset = Some(offset);
        if offset == 0 {
            return Ok(MediaBuffer::Packet(packet));
        }

        let shifted = |value: i64| {
            value
                .checked_add(offset)
                .ok_or(WebRtcError::PacketTimestampNormalizationOverflow { value, offset })
        };
        let mut normalized = (*packet).clone();
        normalized.set_pts(Some(shifted(pts)?));
        normalized.set_dts(packet.dts().map(shifted).transpose()?);
        Ok(MediaBuffer::Packet(Arc::new(normalized)))
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
    id: TrackId,
    pp_log: PpLog,
    name: Arc<str>,
    pad: SrcPad,
    data_rx: Receiver<MediaBuffer>,
    codec: Arc<Mutex<Option<Codec>>>,
    negotiated_codecs: Arc<Mutex<Vec<Codec>>>,
    stream_info: Mutex<StreamInfoState>,
}

struct StreamInfoState {
    rx: Receiver<WebRtcStreamInfo>,
    cached: Option<WebRtcStreamInfo>,
}

impl WebRtcTrackSource {
    pub(super) fn new(
        id: TrackId,
        kind: MediaKind,
        name: impl Into<String>,
        data_rx: Receiver<MediaBuffer>,
        codec: Arc<Mutex<Option<Codec>>>,
        negotiated_codecs: Arc<Mutex<Vec<Codec>>>,
        stream_info_rx: Receiver<WebRtcStreamInfo>,
    ) -> Self {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::WebRtcPeer, &name, None);
        // An inbound track always delivers encoded media; which codec is
        // negotiated with the peer at runtime, but the kind never varies.
        let pad = SrcPad::with_contract(
            format!("{name}_src"),
            OutputContract::Fixed(PortContract::packet(packet_kind(kind))),
        );
        Self {
            id,
            name,
            pp_log,
            pad,
            data_rx,
            codec,
            negotiated_codecs,
            stream_info: Mutex::new(StreamInfoState {
                rx: stream_info_rx,
                cached: None,
            }),
        }
    }

    /// Blocks for at most `timeout` until actual RTP media confirms enough
    /// stream parameters to construct downstream consumers. Most codecs are
    /// known from the first payload; H.264 waits until both SPS and PPS have
    /// arrived. The returned [`WebRtcStreamInfo`] can derive the RTP time base
    /// and FFmpeg parameters for a decoder or supported muxer.
    ///
    /// A timeout returns [`WebRtcError::StreamInfoTimeout`] without consuming
    /// or invalidating anything, so the caller may retry. Once confirmed, the
    /// value is cached and every later call returns it immediately. If the
    /// peer closes before the required media information arrives, this returns
    /// [`WebRtcError::Closed`]. This method does not consume media packets:
    /// they remain buffered for [`SourceElement::run`].
    pub fn wait_stream_info(&self, timeout: Duration) -> Result<WebRtcStreamInfo> {
        let mut state = self.stream_info.lock().unwrap();
        if let Some(info) = &state.cached {
            return Ok(info.clone());
        }

        match state.rx.recv_timeout(timeout) {
            Ok(info) => {
                state.cached = Some(info.clone());
                Ok(info)
            }
            Err(RecvTimeoutError::Timeout) => Err(WebRtcError::StreamInfoTimeout {
                track_id: self.id,
                timeout,
            }
            .into()),
            Err(RecvTimeoutError::Disconnected) => Err(WebRtcError::Closed.into()),
        }
    }

    /// Returns the distinct codec families this track can currently receive
    /// after SDP negotiation. The order is informational.
    ///
    /// This is available as soon as the source is created. For a source on
    /// the side that originated the media section, the initial offered list
    /// is narrowed when [`WebRtcHandle::set_answer`] applies the answer.
    /// [`WebRtcTrackSource::codec`] remains separate: it reports which codec
    /// the remote sender actually chose once media starts arriving.
    pub fn negotiated_codecs(&self) -> Vec<Codec> {
        self.negotiated_codecs.lock().unwrap().clone()
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
    /// extra constraint in practice. Use [`Self::wait_stream_info`] when the
    /// downstream graph must be configured before this source starts running.
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

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for WebRtcTrackSource {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl SourceElement for WebRtcTrackSource {
    fn is_live(&self) -> bool {
        true
    }

    /// Identical shape to [`crate::elements::AppSource::run`]: selects on
    /// `control` and its own data channel together, so `Stop`/`Pause`
    /// never wait behind a remote peer that's gone quiet. The data channel
    /// disconnecting — `WebRtcPeer` gone, whether from `Stop` or the
    /// connection dying on its own — ends this the same way `AppSource`
    /// ends when every `AppSourceHandle` is dropped: one final `Eos`, no
    /// error.
    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        pp_info!(self, "started");
        loop {
            if drain_control(control, self, bus)?.stopped {
                pp_info!(self, "stopped");
                return Ok(());
            }

            select! {
                recv(control.rx) -> req => {
                    match req {
                        Ok(req) => {
                            match req.kind {
                                RequestKind::Finish => {
                                    apply_finish(self, bus, &req.ack);
                                    pp_info!(self, "finished");
                                    return Ok(());
                                }
                                RequestKind::Control(msg) => {
                                    if apply_one(self, bus, msg, &req.ack)? {
                                        pp_info!(self, "stopped");
                                        return Ok(());
                                    }
                                    if msg == ControlMsg::Pause
                                        && wait_out_pause(control, self, bus)?
                                    {
                                        pp_info!(self, "stopped");
                                        return Ok(());
                                    }
                                }
                            }
                        }
                        // The Pipeline itself is gone — nothing left to drive this.
                        Err(_) => {
                            pp_info!(self, "run: control channel gone, ending");
                            return Ok(());
                        }
                    }
                }
                recv(self.data_rx) -> buf => {
                    match buf {
                        Ok(buf) if buf.is_eos() => {
                            pp_info!(self, "event=eos phase=source_received");
                            break;
                        }
                        Ok(buf) => {
                            if let Err(error) = self.pad.push(buf) {
                                bus.post(
                                    &self.pp_log,
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
                            pp_info!(self, "run: WebRtcPeer gone, ending");
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
        self.pad.push_eos(&self.pp_log)
    }

    /// No timeline of its own — same reasoning as
    /// [`crate::elements::AppSource::seek`]: a WebRTC connection has
    /// nothing to reposition.
    fn seek(&mut self, target: Duration) -> Result<Duration> {
        Ok(target)
    }
}
