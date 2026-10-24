use std::sync::Arc;
use std::time::Duration;

use crossbeam_channel::Sender;
use ffmpeg_next as ffmpeg;
use str0m::{
    RtcError,
    change::{SdpAnswer, SdpOffer},
    format::Codec,
    media::{Direction, MediaKind, Mid},
};
use thiserror::Error as ThisError;

use super::track::WebRtcTrackSink;

/// Names one track a connection sends, as an error about it says which:
/// opaque, made only here.
///
/// str0m's own [`Mid`] doesn't exist until the SDP exchange that creates it
/// completes, so this is a stable name for the track before and after.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TrackId(pub(super) u64);

/// Errors specific to `WebRtcPeer`, its handle, and its track endpoints.
/// Converts into the crate-wide `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum WebRtcError {
    /// str0m rejected signaling, RTP, or connection state.
    #[error("str0m error: {0}")]
    Str0m(#[from] RtcError),
    /// Sending or receiving a network datagram failed.

    #[error("network error: {0}")]
    Io(#[from] std::io::Error),
    /// An outbound track received a buffer other than an encoded packet.

    #[error(
        "WebRtcTrackSink only accepts already-encoded Packet buffers \
         (an encoder's output), got a {0}"
    )]
    UnsupportedBuffer(&'static str),
    /// An outbound encoded packet has no presentation timestamp.

    #[error("WebRTC packet has no PTS")]
    MissingPacketPts,
    /// A packet timestamp is negative and cannot be represented as RTP media time.

    #[error("WebRTC packet has a negative PTS: {0}")]
    NegativePacketPts(i64),
    /// Applying the track's initial timestamp shift overflowed an FFmpeg
    /// packet timestamp.
    #[error("WebRTC packet timestamp normalization overflows: value={value}, offset={offset}")]
    PacketTimestampNormalizationOverflow {
        /// PTS or DTS before normalization.
        value: i64,
        /// Track-wide shift established by its first packet.
        offset: i64,
    },
    /// A packet time base has a non-positive component.

    #[error("WebRTC packet has an invalid time base: {numerator}/{denominator}")]
    InvalidPacketTimeBase {
        /// Invalid rational numerator.
        numerator: i32,
        /// Invalid rational denominator.
        denominator: i32,
    },
    /// Rescaling the packet timestamp exceeds str0m's media-time range.

    #[error(
        "WebRTC packet timestamp overflows MediaTime: pts={pts}, time_base={numerator}/{denominator}"
    )]
    PacketTimestampOverflow {
        /// Non-negative packet PTS that overflowed during rescaling.
        pts: u64,
        /// Packet time-base numerator.
        numerator: i32,
        /// Packet time-base denominator.
        denominator: i32,
    },
    /// A track's sink was pushed before what feeds it was declared. SDP can
    /// negotiate several codecs for one track, so the connection cannot
    /// infer which its packets contain without risking a payload-type
    /// mismatch; the stream a muxer's `add_stream` is given says.
    #[error("track {0:?} was sent a packet before what feeds it was declared")]
    OutboundCodecNotDeclared(TrackId),
    /// Codec parameters describe a codec WebRTC does not carry, so there is
    /// no RTP payload type this sink could send them under.
    #[error("codec parameters describe {0:?}, which WebRTC does not carry")]
    SourceCodecUnsupported(ffmpeg::codec::Id),
    /// Codec parameters carry configuration this sink cannot put in front of
    /// a keyframe. H.264's `avcC` and HEVC's `hvcC` are converted; VVC's
    /// `vvcC` is not, so it needs an encoder writing Annex-B headers.
    #[error(
        "track {track_id:?} was given {codec:?} configuration that is not Annex-B parameter sets"
    )]
    ParameterSetsNotSupported {
        /// Track whose source was being declared.
        track_id: TrackId,
        /// Codec the unusable configuration describes.
        codec: Codec,
    },
    /// The first keyframe on a track whose parameter sets live outside the
    /// bitstream carries none, and none were declared to put in front of it.
    /// RTP has no container to carry them, so a receiver would wait forever.
    #[error(
        "track {0:?} sent a keyframe with no SPS/PPS and has none to prepend; \
         give its muxer the encoder's stream, whose parameters hold them"
    )]
    MissingParameterSets(TrackId),
    /// A track was declared with, or sent, an outbound codec that its SDP
    /// negotiation did not retain. A failed declaration leaves the previous
    /// valid one unchanged.
    #[error(
        "codec {codec:?} is not negotiated for track {track_id:?}; negotiated codecs: {negotiated:?}"
    )]
    OutboundCodecNotNegotiated {
        /// Track whose outbound codec was being selected.
        track_id: TrackId,
        /// Codec rejected by the negotiated media section.
        codec: Codec,
        /// Distinct codecs currently available on that media section.
        negotiated: Vec<Codec>,
    },
    /// An outbound H.264 packet is length-prefixed — the form a container
    /// demuxer produces — and this sink has no `avcC` configuration to
    /// convert it with. RTP carries Annex-B, so such a packet would leave as
    /// a well-formed stream no receiver can decode.
    #[error(
        "track {0:?} was pushed a length-prefixed H.264 packet and has no avcC configuration; \
         push an encoder's output, or call WebRtcTrackSink::set_source_parameters with the \
         demuxer's stream parameters first"
    )]
    NotAnnexB(TrackId),
    /// A packet this sink's `avcC` configuration says is length-prefixed does
    /// not parse as one — truncated, or carrying a different prefix size.
    #[error("track {0:?} was pushed a malformed length-prefixed H.264 packet")]
    MalformedLengthPrefixedPacket(TrackId),
    /// The selected RTP payload is not media that can be represented as
    /// FFmpeg codec parameters (for example, RTX repair payload).
    #[error("WebRTC codec {0:?} cannot be converted to FFmpeg codec parameters")]
    UnsupportedCodecParameters(Codec),
    /// H.264 codec parameters were requested from an SDP-only value rather
    /// than stream info confirmed from received SPS/PPS.
    #[error("H.264 SPS/PPS have not been received yet")]
    H264ParameterSetsNotReceived,
    /// A received H.264 parameter set cannot form codec configuration.
    #[error("received H.264 {0} is invalid")]
    InvalidH264ParameterSet(&'static str),
    /// Codec configuration cannot fit FFmpeg's signed extradata length.
    #[error("codec configuration is too large: {size} bytes")]
    CodecConfigurationTooLarge {
        /// Configuration size that could not be represented.
        size: usize,
    },
    /// FFmpeg could not allocate owned codec extradata plus its required
    /// padding.
    #[error("failed to allocate {size} bytes for FFmpeg codec parameters")]
    CodecParametersAllocationFailed {
        /// Requested allocation size including FFmpeg padding.
        size: usize,
    },
    /// str0m accepts any non-zero `u32` clock rate, while FFmpeg's Rational
    /// denominator and audio sample rate are signed 32-bit values.
    #[error("WebRTC codec {codec:?} has a clock rate FFmpeg cannot represent: {clock_rate}")]
    InvalidStreamClockRate {
        /// Codec whose RTP clock rate was being converted.
        codec: Codec,
        /// Clock rate outside FFmpeg's signed range.
        clock_rate: u32,
    },
    /// An audio payload declared a channel count that FFmpeg cannot use.
    #[error("WebRTC codec {codec:?} has an invalid audio channel count: {channels}")]
    InvalidStreamChannelCount {
        /// Audio codec whose channel count was being converted.
        codec: Codec,
        /// Invalid channel count from the payload specification.
        channels: u8,
    },
    /// The remote peer renegotiated a track's direction after it attached.
    /// Each track goes one way — a muxer's sink or a source's output — and
    /// what the caller holds cannot be re-issued, so it no longer matches
    /// the connection. Reported rather than silently tolerated — the
    /// outbound half of a track that just became receive-only is dropped by
    /// str0m without an error, and the inbound half of one that just became
    /// send-only simply goes quiet.
    #[error("track {mid} renegotiated its direction from {from:?} to {to:?} after attaching")]
    DirectionChanged {
        /// The `mid` of the track whose direction changed.
        mid: Mid,
        /// The direction the track attached with.
        from: Direction,
        /// The direction the remote peer renegotiated it to.
        to: Direction,
    },

    /// The connection has ended — a peer's run loop, or the connection a
    /// `WebRtcMuxer` or `WebRtcSource` made, gone with the server — and
    /// takes nothing more.
    #[error("the WebRTC connection has ended")]
    Closed,

    /// A WHIP or WHEP URL that is not an `http` or `https` one with a host.
    #[error("{0:?} is not an http or https URL with a host")]
    InvalidUrl(String),
    /// The WHIP or WHEP request could not be made: the server could not be
    /// reached, the connection failed, or no answer came in time.
    #[error("the request to {url} failed: {reason}")]
    Request {
        /// The URL asked.
        url: String,
        /// Why.
        reason: String,
    },
    /// The WHIP or WHEP server refused the offer — 401 or 403 without the
    /// token it wants, 404 for a path it does not serve, 409 for one already
    /// published.
    #[error("{url} answered {status}: {body}")]
    Refused {
        /// The URL asked.
        url: String,
        /// The HTTP status it answered.
        status: u16,
        /// What it said, as it said it.
        body: String,
    },
    /// The offer was answered, and the connection did not come up — ICE
    /// and DTLS — in the time given.
    #[error("no connection within {0:?} of the answer")]
    NotConnected(Duration),
    /// A WHEP stream connected, and no track of it said what it is — its
    /// codec, and for H.264 its parameter sets — in the time given.
    #[error("no track said what it carries within {0:?}")]
    NothingDescribed(Duration),
    /// A `WebRtcMuxer` was opened with no track.
    #[error("a WebRTC muxer needs a track")]
    NoTracks,
}

/// One command sent from a [`super::track::WebRtcHandle`] (any thread) into
/// the thread [`super::peer::WebRtcPeer`] runs on as a driver. What its
/// track sinks send goes through its outbox instead.
pub(super) enum Command {
    /// A muxer's tracks, to be offered sending only in one renegotiation;
    /// the sink of each comes back on the `Sender`, in order, once the
    /// offer is made.
    OpenTracks(
        Vec<OpenTrack>,
        Sender<std::result::Result<Vec<WebRtcTrackSink>, WebRtcError>>,
    ),
    SetAnswer(SdpAnswer),
    /// A fresh offer from the *remote* peer (their own renegotiation, e.g.
    /// them adding a track) — out of scope to originate ourselves in v1
    /// (see the module docs), but still something this side has to be able
    /// to *accept*, or a two-way call could only ever renegotiate from one
    /// side. `Sender` here is a one-shot rendezvous, same idea as
    /// [`crate::control::ControlSender::send`]'s ack channel.
    AcceptOffer(
        SdpOffer,
        Sender<std::result::Result<SdpAnswer, WebRtcError>>,
    ),
}

/// One track a muxer sends over a peer's connection.
pub(super) struct OpenTrack {
    pub(super) id: TrackId,
    pub(super) kind: MediaKind,
    /// What its sink is fed — what its payload type is chosen by.
    pub(super) codec: Codec,
    /// The name its muxer's `add_stream` gave it: what a keyframe request
    /// for it is reported by.
    pub(super) name: Arc<str>,
}

/// Where one outbound track is in str0m's own offer/answer dance — mirrors
/// str0m's own `chat.rs` example's `TrackOutState`.
pub(super) enum TrackOutState {
    ToOpen(MediaKind, Direction),
    Negotiating(Mid),
    Open(Mid),
}

impl TrackOutState {
    pub(super) fn mid(&self) -> Option<Mid> {
        match self {
            TrackOutState::ToOpen(..) => None,
            TrackOutState::Negotiating(mid) | TrackOutState::Open(mid) => Some(*mid),
        }
    }
}
