use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use ffmpeg_next as ffmpeg;

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
    stream_info::{WebRtcStreamInfo, annex_b_nalus, str0m_codec},
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

/// The four-byte Annex-B start code. The three-byte form is equally valid
/// and is recognized on input, but nothing here has a reason to emit it.
const START_CODE: [u8; 4] = [0, 0, 0, 1];

/// Whether `data` opens an Annex-B access unit. Every Annex-B NAL unit is
/// introduced by a start code, keyframe or not, so one look at the front of
/// the first payload settles which form a track carries.
fn starts_with_start_code(data: &[u8]) -> bool {
    data.starts_with(&START_CODE) || data.starts_with(&START_CODE[1..])
}

/// Whether RTP carries `codec` as an Annex-B byte stream whose parameter
/// sets travel in the stream itself. Only these have headers to put in front
/// of a keyframe; another codec's extradata configures a decoder and has no
/// business in the bitstream.
fn annex_b_codec(codec: Codec) -> bool {
    matches!(codec, Codec::H264 | Codec::H265 | Codec::H266)
}

/// Whether an access unit already carries the parameter sets a receiver
/// needs to start decoding, or `None` for a codec whose NAL header layout
/// this does not read.
fn carries_parameter_sets(payload: &[u8], codec: Codec) -> Option<bool> {
    let (sps, pps) = match codec {
        Codec::H264 => (7, 8),
        Codec::H265 => (33, 34),
        _ => return None,
    };
    let nal_type = |nalu: &&[u8]| match codec {
        Codec::H264 => nalu.first().map(|byte| byte & 0x1f),
        // HEVC's NAL header is two bytes, six of the first being the type.
        _ => nalu.first().map(|byte| (byte >> 1) & 0x3f),
    };
    let present: Vec<u8> = annex_b_nalus(payload).iter().filter_map(nal_type).collect();
    Some(present.contains(&sps) && present.contains(&pps))
}

/// Reads an `AVCDecoderConfigurationRecord` — what a container demuxer puts
/// in `extradata` — as Annex-B parameter sets plus the NAL length prefix
/// size its packets use.
///
/// Returns `None` for anything that is not one, which is how Annex-B
/// extradata, an unrelated codec's configuration, and HEVC's differently
/// laid out `hvcC` all keep the verbatim handling they had before.
fn avcc_parameter_sets(config: &[u8]) -> Option<(Vec<u8>, usize)> {
    // configurationVersion(1) profile(3) lengthSizeMinusOne(1) numOfSPS(1),
    // then each parameter set as a 16-bit length and its bytes, SPS first.
    const HEADER: usize = 6;
    if config.len() < HEADER || config[0] != 1 || starts_with_start_code(config) {
        return None;
    }
    let nal_length_size = (config[4] & 0x03) as usize + 1;
    let mut parameter_sets = Vec::new();
    let mut offset = HEADER - 1;
    // The SPS count is five bits (the top three are reserved ones); the PPS
    // count that follows them is a whole byte.
    for count_mask in [0x1f_u8, 0xff] {
        let count = config.get(offset)? & count_mask;
        offset += 1;
        for _ in 0..count {
            let length =
                u16::from_be_bytes([*config.get(offset)?, *config.get(offset + 1)?]) as usize;
            offset += 2;
            let end = offset.checked_add(length)?;
            if end > config.len() || length == 0 {
                return None;
            }
            parameter_sets.extend_from_slice(&START_CODE);
            parameter_sets.extend_from_slice(&config[offset..end]);
            offset = end;
        }
    }
    (!parameter_sets.is_empty()).then_some((parameter_sets, nal_length_size))
}

/// Rewrites a length-prefixed access unit as Annex-B, replacing each NAL
/// unit's length with a start code.
///
/// Returns `None` when the payload does not consume exactly — a truncated
/// unit, or a prefix size that does not match the one the `avcC` record
/// declared. Guessing at either would emit a stream that looks well formed
/// and decodes to nothing.
fn length_prefixed_to_annex_b(payload: &[u8], nal_length_size: usize) -> Option<Vec<u8>> {
    let mut annex_b = Vec::with_capacity(payload.len() + START_CODE.len());
    let mut offset = 0;
    while offset < payload.len() {
        let prefix = payload.get(offset..offset + nal_length_size)?;
        let length = prefix
            .iter()
            .fold(0usize, |value, byte| (value << 8) | usize::from(*byte));
        offset += nal_length_size;
        let end = offset.checked_add(length)?;
        if length == 0 || end > payload.len() {
            return None;
        }
        annex_b.extend_from_slice(&START_CODE);
        annex_b.extend_from_slice(&payload[offset..end]);
        offset = end;
    }
    (!annex_b.is_empty()).then_some(annex_b)
}

/// Rebuilds `packet` around a new payload, carrying every field the RTP
/// write needs with it.
///
/// The time base above all: `Packet::copy` leaves it 0/0, and a packet str0m
/// cannot build a `MediaTime` from is dropped rather than refused — the
/// failure would be a peer that simply receives nothing.
fn rewritten_packet(packet: &ffmpeg::Packet, payload: &[u8]) -> MediaBuffer {
    let mut rewritten = ffmpeg::Packet::copy(payload);
    rewritten.set_time_base(packet.time_base());
    rewritten.set_pts(packet.pts());
    rewritten.set_dts(packet.dts());
    rewritten.set_stream(packet.stream());
    rewritten.set_flags(packet.flags());
    rewritten.set_duration(packet.duration());
    MediaBuffer::Packet(Arc::new(rewritten))
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
    /// The Annex-B codec headers to put back in front of every keyframe —
    /// see [`WebRtcTrackSink::set_source_parameters`].
    parameter_sets: Option<Vec<u8>>,
    /// The NAL length prefix size of incoming payloads, set when
    /// [`WebRtcTrackSink::set_source_parameters`] was given an `avcC` record
    /// instead of Annex-B headers. Its presence is what says every payload
    /// has to be rewritten before RTP.
    nal_length_size: Option<usize>,
    /// Whether the first packet's bitstream form has been examined. A track
    /// does not change form part-way through, so the check costs one
    /// comparison per track rather than one per packet.
    bitstream_checked: bool,
    /// Whether a keyframe has been seen to carry its own parameter sets.
    /// Only consulted while this sink has none of its own to prepend.
    parameter_sets_checked: bool,
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
            parameter_sets: None,
            nal_length_size: None,
            bitstream_checked: false,
            parameter_sets_checked: false,
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

    /// Declares what feeds this sink, from the parameters of whatever does —
    /// an encoder, a demuxer's stream, or another track's
    /// [`WebRtcStreamInfo::codec_parameters`].
    ///
    /// Everything this sink needs is in that one value, so nothing is asked
    /// for twice: the RTP payload type comes from the codec the parameters
    /// name, the headers to put in front of keyframes from their extradata,
    /// and whether payloads arrive length-prefixed from the shape of that
    /// extradata.
    ///
    /// # Why the headers have to travel
    ///
    /// An encoder opened with `AV_CODEC_FLAG_GLOBAL_HEADER` — which every
    /// encoder in this crate is, so that a container has a `CodecPrivate` to
    /// write — moves its SPS/PPS out of the bitstream and into
    /// `parameters()`. A file is then complete, because the container carries
    /// them; an RTP stream is not, because nothing in it does. The receiving
    /// half of this driver builds its decoder parameters by watching for
    /// SPS/PPS to go past (see `stream_info`), so without them a peer never
    /// learns what it is being sent and simply times out waiting. They go in
    /// front of every keyframe rather than once, which is what lets a peer
    /// that joins late — or that lost the first of them — start decoding at
    /// the next one.
    ///
    /// # A demuxer's parameters
    ///
    /// A container demuxer describes H.264 with an `avcC` record, and its
    /// packets are length-prefixed to match rather than Annex-B. Passing
    /// those parameters is therefore two statements at once: the parameter
    /// sets are these, and the payloads to come are length-prefixed. Both are
    /// read out of the one record, and every payload is rewritten as Annex-B
    /// on its way to RTP.
    ///
    /// # Errors
    ///
    /// [`WebRtcError::OutboundCodecNotNegotiated`] when this connection did
    /// not retain the codec the parameters name,
    /// [`WebRtcError::SourceCodecUnsupported`] when WebRTC does not carry it
    /// at all, and [`WebRtcError::ParameterSetsNotSupported`] for HEVC or VVC
    /// configuration in `hvcC`/`vvcC` form, which this sink cannot convert.
    /// A failed call changes nothing, so the previous declaration stays
    /// usable and already-enqueued packets keep the one they were sent with.
    ///
    /// Parameters carrying no extradata are accepted as they are: an encoder
    /// that still writes its headers in-band needs none prepended, and most
    /// codecs have none to prepend.
    pub fn set_source_parameters(&mut self, parameters: &ffmpeg::codec::Parameters) -> Result<()> {
        let id = parameters.id();
        let codec = str0m_codec(id).ok_or(WebRtcError::SourceCodecUnsupported(id))?;
        let negotiated = self.negotiated_codecs();
        if !negotiated.contains(&codec) {
            return Err(WebRtcError::OutboundCodecNotNegotiated {
                track_id: self.id,
                codec,
                negotiated,
            }
            .into());
        }
        // SAFETY: `parameters` is a live `AVCodecParameters`; `extradata` and
        // `extradata_size` are plain fields of it, and the slice is copied
        // out before this borrow ends.
        let bytes = unsafe {
            let raw = parameters.as_ptr();
            let size = usize::try_from((*raw).extradata_size).unwrap_or(0);
            match ((*raw).extradata.is_null() || size == 0).then_some(()) {
                Some(()) => Vec::new(),
                None => std::slice::from_raw_parts((*raw).extradata, size).to_vec(),
            }
        };
        // Decided before anything is written, so a refusal leaves the
        // previous declaration whole.
        let (parameter_sets, nal_length_size) = match () {
            // Only the Annex-B codecs prepend anything. Another codec's
            // extradata describes a decoder rather than introducing a
            // keyframe — `OpusHead` in front of every Opus packet would be
            // corruption, not configuration.
            _ if !annex_b_codec(codec) => (None, None),
            _ if bytes.is_empty() => (None, None),
            _ if starts_with_start_code(&bytes) => (Some(bytes), None),
            _ if codec == Codec::H264 => match avcc_parameter_sets(&bytes) {
                Some((annex_b, length_size)) => (Some(annex_b), Some(length_size)),
                None => {
                    return Err(WebRtcError::ParameterSetsNotSupported {
                        track_id: self.id,
                        codec,
                    }
                    .into());
                }
            },
            // `hvcC`/`vvcC`: a real configuration record this sink has no
            // conversion for. Refused rather than prepended verbatim, which
            // would put a decoder configuration into the bitstream.
            _ => {
                return Err(WebRtcError::ParameterSetsNotSupported {
                    track_id: self.id,
                    codec,
                }
                .into());
            }
        };
        self.codec = Some(codec);
        self.parameter_sets = parameter_sets;
        self.nal_length_size = nal_length_size;
        self.forget_what_was_checked();
        Ok(())
    }

    /// Declares only the codec, for a caller with no parameters to hand —
    /// one pushing packets it assembled itself rather than an encoder's or a
    /// demuxer's.
    ///
    /// Prefer [`WebRtcTrackSink::set_source_parameters`] wherever the source
    /// has `parameters()`: this leaves the sink with no headers to put in
    /// front of keyframes, which for H.264, HEVC and VVC means the packets
    /// themselves must carry their parameter sets in-band. [`Sink::consume`]
    /// checks that on the first keyframe rather than letting a peer wait for
    /// configuration that is never coming.
    ///
    /// Declaring only the codec means exactly that, including after a
    /// [`WebRtcTrackSink::set_source_parameters`] that said more: whatever
    /// that call left — headers to prepend, a length prefix to rewrite — is
    /// dropped here. Keeping it would apply one source's shape to another's
    /// packets, which for the length prefix means rejecting every Annex-B
    /// packet that follows.
    ///
    /// Returns [`WebRtcError::OutboundCodecNotNegotiated`] without changing
    /// the previous selection when `codec` is unavailable.
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
        self.parameter_sets = None;
        self.nal_length_size = None;
        self.forget_what_was_checked();
        Ok(())
    }

    /// Puts the once-per-track checks back to their unexamined state.
    ///
    /// Called by both declarations, because "once per track" is really once
    /// per source: what feeds a sink is exactly what those checks are about,
    /// and a sink told about a new one has examined nothing yet.
    fn forget_what_was_checked(&mut self) {
        self.bitstream_checked = false;
        self.parameter_sets_checked = false;
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
        let buf = self.prepare_bitstream(buf)?;
        let buf = self.prepend_parameter_sets(buf);
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
    /// Puts the codec headers in front of a keyframe, if this sink was given
    /// any — see [`WebRtcTrackSink::set_source_parameters`].
    ///
    /// Only in front of keyframes, and only when the payload does not open
    /// with them already: an encoder that was *not* opened with a global
    /// header still writes them in-band, and doubling them costs bytes on
    /// every keyframe for nothing.
    fn prepend_parameter_sets(&self, buf: MediaBuffer) -> MediaBuffer {
        let Some(headers) = self.parameter_sets.as_deref() else {
            return buf;
        };
        let MediaBuffer::Packet(packet) = &buf else {
            return buf;
        };
        let Some(payload) = packet.data() else {
            return buf;
        };
        if !packet.is_key() || payload.starts_with(headers) {
            return buf;
        }
        let mut joined = Vec::with_capacity(headers.len() + payload.len());
        joined.extend_from_slice(headers);
        joined.extend_from_slice(payload);
        rewritten_packet(packet, &joined)
    }

    /// Puts an Annex-B codec's payload into the form RTP carries, and refuses
    /// what cannot reach a decoder.
    ///
    /// Two things can be wrong, and both are invisible without this. str0m
    /// splits an outbound payload on Annex-B start codes, so a
    /// length-prefixed one is packetized as a single NAL unit whose type byte
    /// is really the first byte of a length. And a keyframe with no parameter
    /// sets — neither in-band nor prepended — is valid RTP that no receiver
    /// can configure a decoder from. Either way every packet leaves, nothing
    /// reports an error, and the symptom belongs to the far end: a wait for
    /// SPS/PPS that never ends.
    ///
    /// Each check runs once per track, since neither the bitstream form nor
    /// where an encoder keeps its headers changes part-way through.
    fn prepare_bitstream(&mut self, buf: MediaBuffer) -> Result<MediaBuffer> {
        let Some(codec) = self.codec.filter(|codec| annex_b_codec(*codec)) else {
            return Ok(buf);
        };
        let MediaBuffer::Packet(packet) = &buf else {
            return Ok(buf);
        };
        let Some(payload) = packet.data() else {
            return Ok(buf);
        };

        let converted = match self.nal_length_size {
            // Already in the form RTP carries, whatever the record said. A
            // caller can declare a demuxer's parameters — for the parameter
            // sets, which are only there — while what reaches this sink has
            // been converted on the way. Read as length-prefixed, such a
            // payload's leading start code parses as a one-byte NAL unit and
            // the packet is refused as malformed, which names neither the
            // cause nor the fix. Four bytes to rule out, per packet rather
            // than once, because this is the one thing about a payload that
            // something upstream can change without redeclaring anything.
            Some(_) if starts_with_start_code(payload) => None,
            Some(nal_length_size) => {
                let Some(annex_b) = length_prefixed_to_annex_b(payload, nal_length_size) else {
                    pp_error!(
                        self,
                        "outbound packet is not a valid length-prefixed access unit"
                    );
                    return Err(WebRtcError::MalformedLengthPrefixedPacket(self.id).into());
                };
                Some(annex_b)
            }
            None => {
                // Not marked checked on failure: a `Queue` reports this and
                // carries on with the next buffer, and refusing only the
                // first of them would put the silent failure back for every
                // packet after it. The same goes for the check below.
                if !self.bitstream_checked {
                    if !starts_with_start_code(payload) {
                        pp_error!(self, "outbound packet is not Annex-B");
                        return Err(WebRtcError::NotAnnexB(self.id).into());
                    }
                    self.bitstream_checked = true;
                }
                None
            }
        };

        if packet.is_key() && self.parameter_sets.is_none() && !self.parameter_sets_checked {
            let outgoing = converted.as_deref().unwrap_or(payload);
            if carries_parameter_sets(outgoing, codec) == Some(false) {
                pp_error!(self, "outbound keyframe carries no parameter sets");
                return Err(WebRtcError::MissingParameterSets(self.id).into());
            }
            self.parameter_sets_checked = true;
        }

        match converted {
            Some(annex_b) => Ok(rewritten_packet(packet, &annex_b)),
            None => Ok(buf),
        }
    }

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

    fn is_seekable(&self) -> bool {
        false
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
                                    if apply_one(self, bus, &msg, &req.ack)? {
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
