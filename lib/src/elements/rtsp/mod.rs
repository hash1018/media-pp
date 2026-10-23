//! RTSP as this crate speaks it: the options a connection is made with,
//! and what both ends of one share — messages on the wire, descriptions,
//! RTP taken apart, authentication.

pub(crate) mod depacketize;
pub(crate) mod digest;
pub(crate) mod dts;
pub(crate) mod message;
pub(crate) mod sdp;

use std::time::Duration;

/// Lower transport used for RTP packets negotiated through RTSP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RtspTransport {
    /// RTP-over-TCP, interleaved in the RTSP control connection. This is
    /// the default because it crosses NAT and firewalls more reliably.
    #[default]
    Tcp,
    /// RTP-over-UDP. This can reduce latency on a controlled network but
    /// requires the RTP/RTCP UDP ports negotiated with the server to work.
    Udp,
}

impl RtspTransport {
    pub(crate) fn as_ffmpeg_option(self) -> &'static str {
        match self {
            Self::Tcp => "tcp",
            Self::Udp => "udp",
        }
    }
}

/// How to reach an RTSP server — for
/// [`RtspSource::open`](crate::elements::RtspSource::open), which receives
/// from one, and [`RtspMuxer::create`](crate::elements::RtspMuxer::create),
/// which publishes to one. [`RtspSource::listen`](crate::elements::RtspSource::listen),
/// which a publisher reaches instead, takes
/// [`RtspListenOptions`](crate::elements::RtspListenOptions).
#[derive(Debug, Clone, Copy)]
pub struct RtspOptions {
    /// Transport used for RTSP media delivery.
    pub transport: RtspTransport,
    /// How long the connection may take to answer — the connect, each
    /// request of the handshake — and, once [`RtspSource`] reads a
    /// session, how long the server may send no media before the session is
    /// given up as [`RtspSourceError::ServerSilent`]. For an
    /// [`RtspMuxer`](crate::elements::RtspMuxer) it is FFmpeg's own
    /// `timeout` option, which covers its connect, its handshake and every
    /// write.
    ///
    /// A connection the server refuses can take this long to report as
    /// well on some systems — Windows among them.
    ///
    /// [`RtspSource`]: crate::elements::RtspSource
    /// [`RtspSourceError::ServerSilent`]: crate::elements::RtspSourceError::ServerSilent
    pub timeout: Duration,
}

impl Default for RtspOptions {
    fn default() -> Self {
        Self {
            transport: RtspTransport::Tcp,
            timeout: Duration::from_secs(5),
        }
    }
}

impl RtspOptions {
    /// These options as FFmpeg's RTSP muxer takes them.
    pub(crate) fn to_dictionary(self) -> ffmpeg_next::Dictionary<'static> {
        let mut dict = ffmpeg_next::Dictionary::new();
        dict.set("rtsp_transport", self.transport.as_ffmpeg_option());
        dict.set("timeout", &self.timeout.as_micros().to_string());
        dict
    }
}

/// Removes credentials from an RTSP URL, leaving enough to recognize where
/// a stream was going or coming from — what reaches a log.
///
/// Only the authority can carry userinfo, so a `@` later in the path is part
/// of the path and stays. Unlike RTMP, the path itself is not a secret here.
pub(crate) fn redact(url: &str) -> String {
    let Some((scheme, rest)) = url.split_once("://") else {
        // Not a shape this understands, so nothing about it is quotable.
        return "<url>".to_string();
    };
    let authority_end = rest.find('/').unwrap_or(rest.len());
    match rest[..authority_end].rfind('@') {
        Some(at) => format!("{scheme}://<credentials>@{}", &rest[at + 1..]),
        None => url.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::{RtspTransport, redact};

    #[test]
    fn maps_transports_to_ffmpeg_options() {
        assert_eq!(RtspTransport::Tcp.as_ffmpeg_option(), "tcp");
        assert_eq!(RtspTransport::Udp.as_ffmpeg_option(), "udp");
    }

    /// Cameras and some servers are addressed with the password in the URL,
    /// and this is the only thing between it and a log file.
    #[test]
    fn redacts_credentials_from_the_authority() {
        let cases = [
            (
                "rtsp://admin:hunter2@192.168.0.10:554/stream1",
                "rtsp://<credentials>@192.168.0.10:554/stream1",
            ),
            ("rtsp://user@host/path", "rtsp://<credentials>@host/path"),
            // Nothing to remove: an RTSP path is not itself a secret, which
            // is where this differs from `RtmpMuxer`.
            (
                "rtsp://127.0.0.1:8554/stream",
                "rtsp://127.0.0.1:8554/stream",
            ),
            // A `@` in the path is part of the path, not userinfo.
            ("rtsp://127.0.0.1:8554/a@b", "rtsp://127.0.0.1:8554/a@b"),
        ];

        for (url, expected) in cases {
            assert_eq!(redact(url), expected, "redacting {url}");
        }
    }

    #[test]
    fn redacting_something_that_is_not_a_url_quotes_none_of_it() {
        assert_eq!(redact("admin:hunter2"), "<url>");
    }
}
