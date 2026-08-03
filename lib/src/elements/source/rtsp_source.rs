use std::{sync::Arc, time::Duration};

use ffmpeg_next as ffmpeg;
use rust_hlog::{HLog, herror, hinfo};
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    control::{ControlReceiver, drain_control},
    element::{Element, ElementType, Source, SourceElement, element_hlog},
    error::Result,
    pad::SrcPad,
};

use super::file_demuxer::StreamInfo;

/// Errors specific to `RtspSource`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum RtspSourceError {
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),
    #[error("RtspSource doesn't support seeking a live stream")]
    SeekUnsupported,
}

/// Which RTP transport to negotiate with the RTSP server.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtspTransport {
    /// RTP-over-TCP, interleaved in the RTSP session itself — the
    /// default. Traverses NAT/firewalls that block or mangle arbitrary
    /// inbound UDP, at the cost of head-of-line blocking every stream
    /// multiplexed on the same connection behind one dropped/late segment.
    Tcp,
    /// Plain RTP-over-UDP, one port pair per stream — lower latency when
    /// nothing in the path interferes, but exactly the kind of traffic a
    /// typical firewall/NAT drops or never lets back in.
    Udp,
}

/// Construction-time options for [`RtspSource::open`].
#[derive(Debug, Clone, Copy)]
pub struct RtspOptions {
    pub transport: RtspTransport,
    /// Socket I/O timeout — ffmpeg's own `timeout` RTSP demuxer option,
    /// which covers the initial connect/handshake reads too, not just
    /// steady-state ones. Without this, ffmpeg's own default is *no
    /// timeout at all*, meaning [`RtspSource::open`] can hang forever
    /// against an unreachable or dead server.
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

/// Demuxes a live RTSP stream — the client/receive counterpart to
/// [`crate::elements::RtspServer`] (which publishes). One src pad per
/// stream the server advertises, same shape as
/// [`crate::elements::FileDemuxer`].
///
/// Deliberately does **not** retry or reconnect internally: a read failure
/// (dropped connection, camera reboot, ...) ends this source's thread with
/// `Err`, the same way any other fatal [`SourceElement::run`] failure
/// does, instead of looping forever inside `run()`. Reconnecting means
/// building a fresh `RtspSource`/[`crate::pipeline::Pipeline`] — mirrors
/// `Pipeline` itself not being reusable once it ends: watch
/// [`crate::pipeline::Pipeline::bus`], and on error, call
/// [`RtspSource::open`] again.
///
/// Uses `Packet::read` directly instead of `Input::packets()` — the
/// latter silently retries forever inside its own `next()` on any non-EOF
/// error (network timeout, connection reset, ...), which would make a
/// stuck connection un-`Stop`-able (`drain_control` never gets a turn)
/// and this element's "fail fast, don't retry" contract impossible to
/// keep.
#[rust_hlog::hlog]
pub struct RtspSource {
    name: Arc<str>,
    input: ffmpeg::format::context::Input,
    pads: Vec<SrcPad>,
}

impl RtspSource {
    /// Connects to `url` (e.g. `rtsp://host:port/path`) and returns the
    /// element alongside every stream the server advertised, so the
    /// caller can inspect them before deciding which of `src_pads()` to
    /// link — same pattern as `FileDemuxer::open`.
    pub fn open(
        name: impl Into<String>,
        url: impl AsRef<str>,
        options: RtspOptions,
    ) -> std::result::Result<(Self, Vec<StreamInfo>), RtspSourceError> {
        let mut dict = ffmpeg::Dictionary::new();
        dict.set(
            "rtsp_transport",
            match options.transport {
                RtspTransport::Tcp => "tcp",
                RtspTransport::Udp => "udp",
            },
        );
        dict.set("timeout", &options.timeout.as_micros().to_string());

        let input = ffmpeg::format::input_with_dictionary(url.as_ref(), dict)?;

        let streams: Vec<StreamInfo> = input
            .streams()
            .map(|s| StreamInfo {
                index: s.index(),
                kind: s.parameters().medium(),
            })
            .collect();

        let pads = streams
            .iter()
            .map(|s| SrcPad::new(format!("src_{}", s.index)))
            .collect();

        let name: Arc<str> = name.into().into();
        let hlog = element_hlog(ElementType::RtspSource, &name, None);
        hinfo!(
            hlog: &hlog,
            "opened: url={}, transport={:?}, {} stream(s)",
            url.as_ref(),
            options.transport,
            streams.len()
        );
        Ok((
            Self {
                name,
                hlog,
                input,
                pads,
            },
            streams,
        ))
    }

    /// Codec parameters for one of this stream's streams — what you need
    /// to construct a matching [`crate::elements::SwDecoder`] for it.
    pub fn stream_parameters(&self, index: usize) -> Option<ffmpeg::codec::Parameters> {
        self.stream(index).map(|s| s.parameters())
    }

    /// The unit decoded frame timestamps for this stream are expressed in
    /// — what you need to construct a matching [`crate::elements::Pacer`]
    /// for it.
    pub fn stream_time_base(&self, index: usize) -> Option<ffmpeg::Rational> {
        self.stream(index).map(|s| s.time_base())
    }

    fn stream(&self, index: usize) -> Option<ffmpeg::format::stream::Stream<'_>> {
        self.input.streams().find(|s| s.index() == index)
    }
}

impl Element for RtspSource {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::RtspSource
    }

    fn hlog(&self) -> &HLog {
        &self.hlog
    }

    fn hlog_mut(&mut self) -> &mut HLog {
        &mut self.hlog
    }
}

impl Source for RtspSource {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        &mut self.pads
    }
}

impl SourceElement for RtspSource {
    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        hinfo!(self, "run: starting");
        loop {
            if drain_control(control, self, bus)? {
                hinfo!(self, "run: stopped");
                return Ok(());
            }

            let mut packet = ffmpeg::Packet::empty();
            match packet.read(&mut self.input) {
                Ok(()) => {
                    let index = packet.stream();
                    if let Some(pad) = self.pads.get_mut(index) {
                        // A downstream failure drops just this one packet
                        // — same "report, don't die" contract `Queue`'s
                        // worker gives a failing `Sink` — rather than
                        // ending this whole source thread over it.
                        if let Err(error) = pad.push(MediaBuffer::Packet(Arc::new(packet))) {
                            bus.post(
                                &self.hlog,
                                BusEvent::Error {
                                    element_type: ElementType::RtspSource,
                                    name: self.name.clone(),
                                    error,
                                },
                            );
                        }
                    }
                }
                // A real on-demand RTSP stream can send a clean EOF; a
                // live camera essentially never will, but treat it the
                // same way `FileDemuxer` treats running out of packets.
                Err(ffmpeg::Error::Eof) => break,
                // Anything else (connection reset, socket timeout, ...) is
                // fatal — reported and this thread ends, rather than
                // retried. See this type's own docs on why: retrying
                // belongs to whoever's watching the bus, building a fresh
                // `RtspSource` to reconnect with.
                Err(error) => {
                    herror!(self, "read failed: {error}");
                    return Err(RtspSourceError::Ffmpeg(error).into());
                }
            }
        }
        for pad in self.pads.iter_mut() {
            pad.push(MediaBuffer::Eos)?;
        }
        hinfo!(self, "run: reached eos");
        Ok(())
    }

    fn seek(&mut self, _target: Duration) -> Result<Duration> {
        Err(RtspSourceError::SeekUnsupported.into())
    }
}
