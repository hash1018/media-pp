mod client;
mod command;
mod connection;
mod inbox;
mod outbox;
mod peer;
mod server;
mod stream_info;
mod track;
mod whep;
mod whip;

pub use client::WebRtcOptions;
pub use command::{TrackId, WebRtcError};
pub use peer::WebRtcPeer;
pub use server::{
    IceServer, SessionId, SessionRole, SessionStats, WebRtcHttpEndpoint, WebRtcServer,
    WebRtcServerError, WebRtcServerEvent, WebRtcServerHandle, WebRtcServerOptions,
    WebRtcServerPath,
};
pub use track::WebRtcHandle;
pub use whep::WebRtcSource;
pub use whip::WebRtcMuxer;

/// The fuzz targets' way to what this module reads of its peers itself —
/// see `crate::fuzzing`. The rest — SDP, RTP, RTCP, DTLS — is str0m's.
#[cfg(fuzzing)]
pub(crate) mod fuzz {
    use str0m::format::{Codec, CodecSpec, FormatParams};
    use str0m::media::Frequency;

    /// A WHIP or WHEP request's line and headers, as the server reads them.
    pub fn http(data: &[u8]) {
        let _ = super::server::read_head(&mut &data[..]);
    }

    /// What a publisher's first pictures say they are, as a server or a
    /// WHEP client works out its stream from them, and that made into a
    /// stream's parameters.
    pub fn publisher(data: &[u8]) {
        let mut probe = super::stream_info::StreamInfoProbe::new();
        let spec = CodecSpec {
            codec: Codec::H264,
            clock_rate: Frequency::NINETY_KHZ,
            channels: None,
            format: FormatParams::default(),
        };
        for frame in data.split(|byte| *byte == 0xff) {
            if let Some(info) = probe.observe(spec, frame) {
                let _ = info.stream_info();
            }
        }
    }

    /// An `avcC` or `hvcC` configuration and the length-prefixed pictures
    /// after it, as a track turns what an MP4 or RTMP carried into what
    /// WebRTC sends.
    pub fn track(data: &[u8]) {
        let [size, rest @ ..] = data else {
            return;
        };
        let _ = super::track::avcc_parameter_sets(rest);
        let _ = super::track::hvcc_parameter_sets(rest);
        let _ = super::track::length_prefixed_to_annex_b(rest, usize::from(size % 5));
    }
}

#[cfg(test)]
mod tests;
