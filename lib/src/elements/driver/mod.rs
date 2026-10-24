//! Background tasks that run beside pipelines rather than in them: the
//! WebRTC peer and its tracks, and the RTSP, WebRTC and RTMP servers.
//!
//! They are not part of a dataflow graph. A [`Driver`](crate::driver::Driver)
//! has no pads of its own; it runs a loop and mints ordinary
//! `RawSink`/`SrcPads` endpoints on the side for pipelines to wire.
//! `WebRtcPeer` owns a peer connection while a `WebRtcMuxer` of it and a
//! `WebRtcSource` of what the remote side adds carry media out of and into
//! whichever pipelines those tracks belong to. `RtspServer` serves
//! [`Broadcast`](crate::elements::Broadcast)s, which pipelines write into.

#[cfg(any(feature = "rtsp-server", feature = "rtmp-server", feature = "webrtc"))]
mod access;
#[cfg(any(feature = "rtsp-server", feature = "rtmp-server", feature = "webrtc"))]
pub(crate) mod publishers;
#[cfg(feature = "rtmp-server")]
mod rtmp_server;
#[cfg(feature = "rtsp-server")]
mod rtsp_server;
#[cfg(feature = "webrtc")]
mod webrtc;

#[cfg(any(feature = "rtsp-server", feature = "rtmp-server", feature = "webrtc"))]
pub use access::{Access, AccessRequest, Authorize, Credentials};
#[cfg(any(feature = "rtsp-server", feature = "rtmp-server", feature = "webrtc"))]
pub use publishers::Publishers;
#[cfg(feature = "rtmp-server")]
pub use rtmp_server::{
    RtmpServer, RtmpServerError, RtmpServerHandle, RtmpServerOptions, RtmpSource,
};
#[cfg(feature = "rtsp-server")]
pub use rtsp_server::{
    RtspDigest, RtspServer, RtspServerError, RtspServerHandle, RtspServerOptions, RtspSessionStats,
};
#[cfg(feature = "webrtc")]
pub use webrtc::{
    IceServer, SessionId, SessionRole, SessionStats, TrackId, WebRtcError, WebRtcHandle,
    WebRtcHttpEndpoint, WebRtcMuxer, WebRtcOptions, WebRtcPeer, WebRtcServer, WebRtcServerError,
    WebRtcServerEvent, WebRtcServerHandle, WebRtcServerOptions, WebRtcSource,
};
