use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use ffmpeg_next as ffmpeg;

use crate::pp_log::{PpLog, pp_error, pp_warn};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};

use str0m::{
    change::{SdpAnswer, SdpOffer},
    format::Codec,
    media::MediaKind,
};

use crate::elements::StreamInfo;
use crate::{
    buffer::{MediaBuffer, PacketBuffer},
    contract::{InputContract, PortContract},
    element::{Element, ElementType, RawSink, element_pp_log},
    error::Result,
};

use super::{
    WebRtcMuxer, WebRtcSource,
    command::{Command, OpenTrack, TrackId, WebRtcError},
    outbox::{Outbox, Outgoing, Pushed, Rewrite},
    peer::Arriving,
    stream_info::{annex_b_nalus, str0m_codec},
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

/// Whether an access unit's NAL units already carry the parameter sets a
/// receiver needs to start decoding, or `None` for a codec whose NAL header
/// layout this does not read.
fn carries_parameter_sets(nalus: &[&[u8]], codec: Codec) -> Option<bool> {
    let (sps, pps) = match codec {
        Codec::H264 => (7, 8),
        Codec::H265 => (33, 34),
        Codec::H266 => (15, 16),
        _ => return None,
    };
    // Three codecs, three places to read one field from. H.264 keeps the
    // type in the low five bits of a one-byte header; HEVC widened the
    // header to two bytes and took six bits of the first for the type; VVC
    // kept two bytes but moved the type into the top five bits of the
    // *second*, the first now being nothing but the layer id.
    let nal_type = |nalu: &&[u8]| match codec {
        Codec::H264 => nalu.first().map(|byte| byte & 0x1f),
        Codec::H266 => nalu.get(1).map(|byte| byte >> 3),
        _ => nalu.first().map(|byte| (byte >> 1) & 0x3f),
    };
    let present: Vec<u8> = nalus.iter().filter_map(nal_type).collect();
    Some(present.contains(&sps) && present.contains(&pps))
}

/// Reads an `AVCDecoderConfigurationRecord` — what a container demuxer puts
/// in `extradata` — as Annex-B parameter sets plus the NAL length prefix
/// size its packets use.
///
/// The outer `None` is for anything that is not such a record, which is how
/// Annex-B extradata, an unrelated codec's configuration, and HEVC's
/// differently laid out `hvcC` all keep the verbatim handling they had
/// before. The inner one is a record that holds no parameter sets, which is
/// a different thing entirely and still worth having: the two facts a record
/// carries are separate fields, so one can be absent while the other is
/// exactly what a caller needs. Such a file keeps its parameter sets in the
/// bitstream, and refusing the record would leave nothing to state the
/// length prefix with — the packets are still length-prefixed either way.
pub(super) fn avcc_parameter_sets(config: &[u8]) -> Option<(Option<Vec<u8>>, usize)> {
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
    Some((
        (!parameter_sets.is_empty()).then_some(parameter_sets),
        nal_length_size,
    ))
}

/// Reads an `HEVCDecoderConfigurationRecord` — HEVC's `avcC`, what an MP4
/// demuxer puts in `extradata` — as Annex-B parameter sets plus the NAL
/// length prefix size its packets use; `None` for anything else.
pub(super) fn hvcc_parameter_sets(config: &[u8]) -> Option<(Option<Vec<u8>>, usize)> {
    // 22 bytes of profile and format, lengthSizeMinusOne in the low bits of
    // the last; then a count of arrays, each a type byte, a 16-bit count,
    // and that many 16-bit-length parameter sets.
    const HEADER: usize = 23;
    if config.len() < HEADER || config[0] != 1 || starts_with_start_code(config) {
        return None;
    }
    let nal_length_size = (config[21] & 0x03) as usize + 1;
    let mut parameter_sets = Vec::new();
    let mut offset = HEADER;
    for _ in 0..config[22] {
        offset += 1;
        let count = u16::from_be_bytes([*config.get(offset)?, *config.get(offset + 1)?]);
        offset += 2;
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
    Some((
        (!parameter_sets.is_empty()).then_some(parameter_sets),
        nal_length_size,
    ))
}

/// The NAL units of a length-prefixed access unit.
///
/// Returns `None` when the payload does not consume exactly — a truncated
/// unit, or a prefix size that does not match the one the `avcC` record
/// declared. Guessing at either would emit a stream that looks well formed
/// and decodes to nothing.
fn length_prefixed_nalus(payload: &[u8], nal_length_size: usize) -> Option<Vec<&[u8]>> {
    let mut nalus = Vec::new();
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
        nalus.push(&payload[offset..end]);
        offset = end;
    }
    (!nalus.is_empty()).then_some(nalus)
}

/// Rewrites a length-prefixed access unit as Annex-B, replacing each NAL
/// unit's length with a start code — `None` where it is not one; see
/// [`length_prefixed_nalus`].
pub(super) fn length_prefixed_to_annex_b(
    payload: &[u8],
    nal_length_size: usize,
) -> Option<Vec<u8>> {
    let nalus = length_prefixed_nalus(payload, nal_length_size)?;
    let mut annex_b = Vec::with_capacity(payload.len() + START_CODE.len());
    for nalu in nalus {
        annex_b.extend_from_slice(&START_CODE);
        annex_b.extend_from_slice(nalu);
    }
    Some(annex_b)
}

/// Rebuilds `packet` around a new payload, carrying every field the RTP
/// write needs with it.
///
/// The time base above all: `Packet::copy` leaves it 0/0, and a packet str0m
/// cannot build a `MediaTime` from is dropped rather than refused — the
/// failure would be a peer that simply receives nothing.
#[cfg(test)]
pub(super) fn rewritten_packet(packet: &ffmpeg::Packet, payload: &[u8]) -> MediaBuffer {
    let mut rewritten = ffmpeg::Packet::copy(payload);
    rewritten.set_time_base(packet.time_base());
    rewritten.set_pts(packet.pts());
    rewritten.set_dts(packet.dts());
    rewritten.set_stream(packet.stream());
    rewritten.set_flags(packet.flags());
    rewritten.set_duration(packet.duration());
    MediaBuffer::Packet(Arc::new(rewritten).into())
}

/// How long [`WebRtcHandle::open_tracks`] waits for a peer to take a
/// muxer's tracks: the peer makes their offer on its next turn, which a
/// running peer takes within milliseconds.
const OPEN_WITHIN: Duration = Duration::from_secs(10);

/// What a running [`WebRtcPeer`](crate::elements::WebRtcPeer) is reached
/// through: the tracks this side sends ([`Self::muxer`]), the ones the
/// remote peer adds ([`Self::next_source`]), and the renegotiations either
/// side starts.
///
/// [`WebRtcPeer::new`](crate::elements::WebRtcPeer::new) returns it, and
/// its example shows it in use. Cheap to clone: every clone reaches the
/// same peer and keeps nothing of it alive — once the peer has stopped,
/// what it is asked fails with [`WebRtcError::Closed`]. The clones share one
/// queue of the remote peer's tracks: each batch goes to one
/// [`Self::next_source`] call, so one place in an application should be
/// taking them.
#[derive(Clone)]
pub struct WebRtcHandle {
    pub(super) next_id: Arc<AtomicU64>,
    pub(super) command_tx: Sender<Command>,
    pub(super) sources_rx: Receiver<Vec<Arriving>>,
}

impl WebRtcHandle {
    /// A muxer of tracks this side sends over the peer's connection: made,
    /// given its tracks with `add_stream` and opened as a [`WebRtcMuxer`]
    /// to a WHIP server is, a sink per track out of what it opens.
    ///
    /// Opening it sends each of its tracks on a section the remote peer
    /// added asking to be sent to, where there is one of the track's kind
    /// that took its codec — at once, with no renegotiation — and adds a
    /// media section, sending only, for each of the others, in one
    /// renegotiation: its offer goes to the peer's `on_offer`, for the
    /// application to carry to the remote peer, and the answer comes back
    /// through [`Self::set_answer`]. Where every track found a section
    /// asked for, no offer is made. A packet a sink takes before the answer
    /// is let go of.
    pub fn muxer(&self, name: impl Into<String>) -> WebRtcMuxer {
        WebRtcMuxer::over_peer(name, self.clone())
    }

    /// Blocks until the remote peer adds tracks, and answers a source of
    /// them — the ones one offer of its adds, together — an output each,
    /// with what each carries, as [`WebRtcSource::open`] does: each track
    /// is waited for up to five seconds to say, by its first packets.
    /// Tracks it adds later come out of a later call, a source of their own.
    ///
    /// Only what the remote peer sends is received: a section it adds
    /// asking to be sent to is not one of these, but where a
    /// [`Self::muxer`]'s track of its kind is sent.
    ///
    /// # Errors
    ///
    /// [`WebRtcError::Closed`] once the peer has stopped, and
    /// [`WebRtcError::NothingDescribed`] where none of the tracks said what
    /// it carries in time.
    pub fn next_source(&self, name: impl Into<String>) -> Result<(WebRtcSource, Vec<StreamInfo>)> {
        let arriving = self.sources_rx.recv().map_err(|_| WebRtcError::Closed)?;
        WebRtcSource::arriving(name.into(), arriving)
    }

    /// Has the peer offer `tracks` to the remote peer, sending only, in one
    /// renegotiation, and answers a sink for each, in order.
    pub(super) fn open_tracks(&self, tracks: Vec<OpenTrack>) -> Result<Vec<WebRtcTrackSink>> {
        let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
        self.command_tx
            .send(Command::OpenTracks(tracks, reply_tx))
            .map_err(|_| WebRtcError::Closed)?;
        match reply_rx.recv_timeout(OPEN_WITHIN) {
            Ok(sinks) => Ok(sinks?),
            Err(RecvTimeoutError::Timeout) => Err(WebRtcError::NotConnected(OPEN_WITHIN).into()),
            Err(RecvTimeoutError::Disconnected) => Err(WebRtcError::Closed.into()),
        }
    }

    /// A number for a track this side is to send, never one the peer gives
    /// another.
    pub(super) fn track_id(&self) -> TrackId {
        TrackId(self.next_id.fetch_add(1, Ordering::Relaxed))
    }

    /// Feeds the remote peer's answer back in, completing the renegotiation
    /// a [`Self::muxer`]'s opening started. A no-op once the peer has
    /// stopped.
    pub fn set_answer(&self, answer: SdpAnswer) {
        let _ = self.command_tx.send(Command::SetAnswer(answer));
    }

    /// Accepts a fresh offer from the *remote* peer (their own
    /// renegotiation) and returns the resulting answer for the caller to
    /// ship back over its own signaling transport. Blocks until
    /// `WebRtcPeer::run` has actually applied it. The tracks it adds come
    /// out of [`Self::next_source`].
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

/// Sends the encoded packets it is handed to the remote peer, on one track:
/// the sink a [`WebRtcMuxer`] hands out for each of its tracks, and that a
/// [`WebRtcServer`](crate::elements::WebRtcServer) session sends a player
/// through.
///
/// One outbound track. `consume()` makes the packet ready for RTP and puts
/// it in its connection's outbox, shared by all the connection's tracks;
/// the str0m write happens where the connection is driven. A connection
/// that falls behind as far as its outbox keeps lets go of what it holds,
/// and each track goes on from its next keyframe.
///
/// What it sends is the codec its parameters declared
/// ([`Self::set_source_parameters`]), checked against what the connection
/// negotiated; a declaration the connection did not negotiate is refused
/// rather than sent under a payload type no receiver can decode.
pub(crate) struct WebRtcTrackSink {
    pp_log: PpLog,
    /// What it goes by: a peer's track by its number, a muxer's by the name
    /// its `add_stream` gave it.
    identity: (ElementType, Arc<str>),
    id: TrackId,
    /// What this track was negotiated to carry — audio or video.
    kind: MediaKind,
    codec: Option<Codec>,
    negotiated_codecs: Arc<Mutex<Vec<Codec>>>,
    outbox: Outbox,
    /// libavcodec may express encoder delay as a negative first PTS (Opus is
    /// a common example), while RTP media time is unsigned. The first packet
    /// establishes one track-wide shift so relative timing is preserved.
    timestamp_offset: Option<i64>,
    /// The Annex-B codec headers to put back in front of every keyframe —
    /// see [`WebRtcTrackSink::set_source_parameters`].
    parameter_sets: Option<Arc<[u8]>>,
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
    /// What the connection it writes into is kept alive by, where that is
    /// its sinks — a `WebRtcMuxer`'s, whose last sink dropped ends it.
    _keeps: Option<Arc<dyn std::any::Any + Send + Sync>>,
}

impl WebRtcTrackSink {
    pub(super) fn new(
        id: TrackId,
        kind: MediaKind,
        codec: Option<Codec>,
        negotiated_codecs: Arc<Mutex<Vec<Codec>>>,
        outbox: Outbox,
    ) -> Self {
        Self {
            id,
            kind,
            codec,
            negotiated_codecs,
            outbox,
            timestamp_offset: None,
            parameter_sets: None,
            nal_length_size: None,
            bitstream_checked: false,
            parameter_sets_checked: false,
            _keeps: None,
            identity: (
                ElementType::WebRtcPeer,
                format!("webrtc-track-{}", id.0).into(),
            ),
            pp_log: element_pp_log(
                ElementType::WebRtcPeer,
                &format!("webrtc-track-{}", id.0),
                None,
            ),
        }
    }

    /// The same, going by `name` as an element of `element_type`.
    pub(super) fn named(mut self, element_type: ElementType, name: Arc<str>) -> Self {
        self.pp_log = element_pp_log(element_type, &name, None);
        self.identity = (element_type, name);
        self
    }

    /// The same, keeping `owner` alive for as long as it lives.
    pub(super) fn keeping(mut self, owner: Arc<dyn std::any::Any + Send + Sync>) -> Self {
        self._keeps = Some(owner);
        self
    }

    /// Returns the distinct codec families this track can currently send
    /// after SDP negotiation. The order is informational; select the codec
    /// produced by the application's encoder.
    ///
    /// A peer's muxer has its sinks while their offer is still pending, so
    /// theirs is the codec offered until [`WebRtcHandle::set_answer`]
    /// applies the answer, which narrows it to what the remote peer took; a
    /// WHIP muxer's and a server's are negotiated when they are handed out.
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
    /// A container demuxer describes H.264 with an `avcC` record and HEVC
    /// with an `hvcC`, and its packets are length-prefixed to match rather
    /// than Annex-B. Passing those parameters is therefore two statements at
    /// once: the parameter sets are these, and the payloads to come are
    /// length-prefixed. Both are read out of the one record, and every
    /// payload is rewritten as Annex-B on its way to RTP.
    ///
    /// # Errors
    ///
    /// [`WebRtcError::OutboundCodecNotNegotiated`] when this connection did
    /// not retain the codec the parameters name,
    /// [`WebRtcError::SourceCodecUnsupported`] when WebRTC does not carry it
    /// at all, and [`WebRtcError::ParameterSetsNotSupported`] for
    /// configuration it cannot read — VVC's `vvcC`, or a record that is not
    /// what its codec's should be.
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
            _ if matches!(codec, Codec::H264 | Codec::H265) => match match codec {
                Codec::H264 => avcc_parameter_sets(&bytes),
                _ => hvcc_parameter_sets(&bytes),
            } {
                // A record with no parameter sets still says the packets are
                // length-prefixed. Nothing is left to prepend, so a keyframe
                // that turns out not to carry them in-band either is caught
                // by `prepare_bitstream` — at the only place that can see it.
                Some((annex_b, length_size)) => (annex_b, Some(length_size)),
                None => {
                    return Err(WebRtcError::ParameterSetsNotSupported {
                        track_id: self.id,
                        codec,
                    }
                    .into());
                }
            },
            // `vvcC`: a real configuration record this sink has no
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
        self.parameter_sets = parameter_sets.map(Arc::from);
        self.nal_length_size = nal_length_size;
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
        Arc::clone(&self.identity.1)
    }

    fn element_type(&self) -> ElementType {
        self.identity.0
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl RawSink for WebRtcTrackSink {
    /// A track carries encoded media to the peer; this sink has no
    /// encoder of its own, so a decoded frame has no route through it.
    fn input_contract(&self) -> InputContract {
        // Path-qualified: `MediaKind` in this module is str0m's own.
        InputContract::Fixed(PortContract::packet(packet_kind(self.kind)))
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        if !matches!(buf, MediaBuffer::Packet(_)) {
            let kind = match buf {
                MediaBuffer::Video(_) => "Video",
                MediaBuffer::Audio(_) => "Audio",
                MediaBuffer::Packet(_) => unreachable!("matched above"),
            };
            pp_error!(self, "unsupported buffer: {kind}");
            return Err(WebRtcError::UnsupportedBuffer(kind).into());
        }
        if matches!(buf, MediaBuffer::Packet(_)) && self.codec.is_none() {
            pp_error!(self, "outbound codec is not declared");
            return Err(WebRtcError::OutboundCodecNotDeclared(self.id).into());
        }
        let (MediaBuffer::Packet(packet), Some(codec)) = (buf, self.codec) else {
            return Ok(()); // refused above
        };
        // Looked at under the lock, not copied out: once a packet.
        if !self
            .negotiated_codecs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains(&codec)
        {
            pp_error!(self, "outbound codec {codec:?} is not negotiated");
            return Err(WebRtcError::OutboundCodecNotNegotiated {
                track_id: self.id,
                codec,
                negotiated: self.negotiated_codecs(),
            }
            .into());
        }
        let nal_length_size = self.prepare_bitstream(&packet)?;
        let packet = self.normalize_packet_timestamp(packet)?;
        // What it is rewritten into goes with it, and is made as it is
        // written: what waits in an outbox is then the packet every other
        // reader shares, not a copy of this track's own — a server's
        // player catching up holds a keyframe interval of them.
        let rewrite = Rewrite {
            nal_length_size,
            prepend: self.parameter_sets.clone(),
        };
        // The connection gone means this track is dead — surfaced as `Err`
        // rather than swallowed, so whatever pipeline this sink is in learns
        // of it instead of sending into a void forever. Non-fatal by the
        // same convention as any other `RawSink::consume` failure.
        //
        // A connection fallen behind lets go and goes on from a keyframe —
        // logged, not reported on a `Bus`: this sink is a handle the caller
        // can keep past the connection, and a `Bus` kept here would stay
        // open past whatever waits for every sender to go.
        match self.outbox.push(Outgoing {
            track: self.id,
            codec,
            packet,
            rewrite,
        }) {
            Pushed::Taken | Pushed::Skipped => Ok(()),
            Pushed::FellBehind => {
                pp_warn!(
                    self,
                    "the connection fell behind; going on from the next keyframe"
                );
                Ok(())
            }
            Pushed::Closed => {
                pp_error!(self, "the connection is gone — track is dead");
                Err(WebRtcError::Closed.into())
            }
        }
    }
}

impl WebRtcTrackSink {
    /// Says how an Annex-B codec's payload is put into the form RTP carries
    /// — the NAL length prefix to rewrite, where there is one — and refuses
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
    fn prepare_bitstream(&mut self, packet: &PacketBuffer) -> Result<Option<usize>> {
        let Some(codec) = self.codec.filter(|codec| annex_b_codec(*codec)) else {
            return Ok(None);
        };
        let Some(payload) = packet.data() else {
            return Ok(None);
        };

        let prefixed = match self.nal_length_size {
            Some(nal_length_size) => match length_prefixed_nalus(payload, nal_length_size) {
                Some(nalus) => Some((nal_length_size, nalus)),
                // Already in the form RTP carries, whatever the record said.
                // A caller can declare a demuxer's parameters — for the
                // parameter sets, which are only there — while what reaches
                // this sink has been converted on the way. Read as
                // length-prefixed, such a payload's leading start code parses
                // as a one-byte NAL unit and the packet is refused as
                // malformed, which names neither the cause nor the fix.
                // Looked at per packet rather than once, because this is the
                // one thing about a payload that something upstream can
                // change without redeclaring anything.
                //
                // Only once the payload has failed to read as what the record
                // said: a NAL unit of 256 to 511 bytes has a length that opens
                // `00 00 01`, a start code to the eye, and taken for one it
                // was sent unconverted — one picture in several of an MP4's.
                None if starts_with_start_code(payload) => None,
                None => {
                    pp_error!(
                        self,
                        "outbound packet is not a valid length-prefixed access unit"
                    );
                    return Err(WebRtcError::MalformedLengthPrefixedPacket(self.id).into());
                }
            },
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
            let carries = match &prefixed {
                Some((_, nalus)) => carries_parameter_sets(nalus, codec),
                None => carries_parameter_sets(&annex_b_nalus(payload), codec),
            };
            if carries == Some(false) {
                pp_error!(self, "outbound keyframe carries no parameter sets");
                return Err(WebRtcError::MissingParameterSets(self.id).into());
            }
            self.parameter_sets_checked = true;
        }
        Ok(prefixed.map(|(nal_length_size, _)| nal_length_size))
    }

    fn normalize_packet_timestamp(&mut self, packet: PacketBuffer) -> Result<PacketBuffer> {
        let Some(pts) = packet.pts() else {
            return Ok(packet);
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
            return Ok(packet);
        }

        let shifted = |value: i64| {
            value
                .checked_add(offset)
                .ok_or(WebRtcError::PacketTimestampNormalizationOverflow { value, offset })
        };
        let mut normalized = crate::buffer::shared_packet(&packet);
        normalized.set_pts(Some(shifted(pts)?));
        normalized.set_dts(packet.dts().map(shifted).transpose()?);
        let mut normalized: PacketBuffer = Arc::new(normalized).into();
        if let Some(metadata) = packet.metadata_arc() {
            normalized = normalized.with_metadata(Arc::clone(metadata));
        }
        Ok(normalized)
    }
}
