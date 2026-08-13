use crossbeam_channel::Sender;
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

/// Errors specific to `WebRtcPeer`/`WebRtcHandle`/`WebRtcTrackSink`.
/// Converts into the crate-wide `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum WebRtcError {
    #[error("str0m error: {0}")]
    Str0m(#[from] RtcError),

    #[error("network error: {0}")]
    Io(#[from] std::io::Error),

    #[error(
        "WebRtcTrackSink only accepts already-encoded Packet buffers \
         (an encoder's output), got a {0}"
    )]
    UnsupportedBuffer(&'static str),

    #[error("WebRTC packet has no PTS")]
    MissingPacketPts,

    #[error("WebRTC packet has a negative PTS: {0}")]
    NegativePacketPts(i64),

    #[error("WebRTC packet has an invalid time base: {numerator}/{denominator}")]
    InvalidPacketTimeBase { numerator: i32, denominator: i32 },

    #[error(
        "WebRTC packet timestamp overflows MediaTime: pts={pts}, time_base={numerator}/{denominator}"
    )]
    PacketTimestampOverflow {
        pts: u64,
        numerator: i32,
        denominator: i32,
    },

    #[error("WebRtcPeer's run() has already ended")]
    Closed,
}

/// One command sent from a [`WebRtcHandle`]/[`WebRtcTrackSink`] (any
/// thread) into [`super::peer::WebRtcPeer::run`]'s own thread.
pub(super) enum Command {
    AddTrack(TrackId, MediaKind, Direction, Codec),
    Push(TrackId, MediaBuffer),
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
