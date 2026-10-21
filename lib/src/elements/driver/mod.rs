//! Background tasks that run beside pipelines rather than in them: the
//! WebRTC peer and its tracks, and the RTSP server.
//!
//! They are not part of a dataflow graph. A [`Driver`](crate::driver::Driver)
//! has no pads of its own; it runs a loop and mints ordinary
//! `RawSink`/`SrcPads` endpoints on the side for pipelines to wire.
//! `WebRtcPeer` owns a peer connection while `WebRtcTrackSink` and
//! `WebRtcTrackSource` carry media in and out of whichever pipelines those
//! tracks belong to. `RtspServer` serves
//! [`Broadcast`](crate::elements::Broadcast)s, which pipelines write into.

#[cfg(feature = "rtsp-server")]
mod rtsp_server;
#[cfg(feature = "webrtc")]
mod webrtc;

#[cfg(feature = "rtsp-server")]
pub use rtsp_server::{
    RtspPathMode, RtspServer, RtspServerError, RtspServerHandle, RtspServerOptions,
};
#[cfg(feature = "webrtc")]
pub use webrtc::{
    AttachedTrack, SessionId, SessionRole, TrackEndpoints, TrackId, WebRtcError, WebRtcHandle,
    WebRtcHttpEndpoint, WebRtcPathMode, WebRtcPeer, WebRtcServer, WebRtcServerError,
    WebRtcServerEvent, WebRtcServerHandle, WebRtcServerOptions, WebRtcStreamInfo, WebRtcTrackSink,
    WebRtcTrackSource,
};
