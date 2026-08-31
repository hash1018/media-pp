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

use crate::buffer::MediaBuffer;

/// Identifies one outbound track before/after negotiation. str0m's own
/// [`Mid`] doesn't exist until the SDP exchange that creates it completes,
/// so this is a stable handle usable from the moment
/// [`super::track::WebRtcHandle::add_track`] returns.
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
    /// A sink for a track added by the remote peer was pushed before the
    /// caller declared which codec its packets contain. SDP can negotiate
    /// several codecs for one track, so the peer cannot infer this from the
    /// track itself without risking a payload-type mismatch.
    #[error(
        "track {0:?} has no outbound codec declaration; call \
         WebRtcTrackSink::set_source_parameters with what feeds it, or \
         set_codec when there are no parameters to hand"
    )]
    OutboundCodecNotDeclared(TrackId),
    /// Codec parameters describe a codec WebRTC does not carry, so there is
    /// no RTP payload type this sink could send them under.
    #[error("codec parameters describe {0:?}, which WebRTC does not carry")]
    SourceCodecUnsupported(ffmpeg::codec::Id),
    /// Codec parameters carry configuration this sink cannot put in front of
    /// a keyframe. H.264's `avcC` is converted; HEVC's `hvcC` and VVC's
    /// `vvcC` are not, so those need an encoder writing Annex-B headers.
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
         call WebRtcTrackSink::set_source_parameters with the encoder's parameters"
    )]
    MissingParameterSets(TrackId),
    /// The caller selected or tried to send an outbound codec that this
    /// track's SDP negotiation did not retain. A failed
    /// [`super::track::WebRtcTrackSink::set_codec`] leaves the previous valid
    /// selection unchanged.
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
    /// No RTP media arrived before a caller's explicit wait deadline. The
    /// source remains usable and the caller may retry with another timeout.
    #[error("timed out after {timeout:?} waiting for stream info on track {track_id:?}")]
    StreamInfoTimeout {
        /// Track whose first actual payload has not arrived yet.
        track_id: TrackId,
        /// Caller-supplied maximum wait.
        timeout: Duration,
    },
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
    /// The remote peer renegotiated a track's direction after this element
    /// had already handed out its endpoints. Those describe the direction
    /// the track attached with (see
    /// [`super::track::TrackEndpoints`]) and cannot be re-issued, so what
    /// the caller holds no longer matches the connection. Reported rather
    /// than silently tolerated — the outbound half of a track that just
    /// became receive-only is dropped by str0m without an error, and the
    /// inbound half of one that just became send-only simply goes quiet.
    #[error("track {mid} renegotiated its direction from {from:?} to {to:?} after attaching")]
    DirectionChanged {
        /// The `mid` of the track whose direction changed.
        mid: Mid,
        /// The direction the track attached with.
        from: Direction,
        /// The direction the remote peer renegotiated it to.
        to: Direction,
    },

    /// The peer run loop has ended and accepts no further commands.

    #[error("WebRtcPeer's run() has already ended")]
    Closed,
}

/// One command sent from a [`WebRtcHandle`]/[`WebRtcTrackSink`] (any
/// thread) into [`super::peer::WebRtcPeer::run`]'s own thread.
pub(super) enum Command {
    AddTrack(TrackId, MediaKind, Direction, Codec),
    Push(TrackId, Option<Codec>, MediaBuffer),
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
