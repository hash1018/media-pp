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

#[cfg(test)]
mod tests {
    use super::RtspTransport;

    #[test]
    fn maps_transports_to_ffmpeg_options() {
        assert_eq!(RtspTransport::Tcp.as_ffmpeg_option(), "tcp");
        assert_eq!(RtspTransport::Udp.as_ffmpeg_option(), "udp");
    }
}
