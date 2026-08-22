//! Background tasks that are not part of a dataflow graph.
//!
//! A [`Driver`](crate::driver::Driver) has no pads of its own; it runs a loop
//! and mints ordinary `Sink`/`Source` endpoints on the side for pipelines to
//! wire. `WebRtcPeer` is the one built-in: it owns the peer connection while
//! `WebRtcTrackSink` and `WebRtcTrackSource` carry media in and out of
//! whichever pipelines those tracks belong to.

#[cfg(feature = "webrtc")]
mod webrtc;

#[cfg(feature = "webrtc")]
pub use webrtc::{
    AttachedTrack, TrackEndpoints, TrackId, WebRtcError, WebRtcHandle, WebRtcPeer, WebRtcTrackSink,
    WebRtcTrackSource,
};
