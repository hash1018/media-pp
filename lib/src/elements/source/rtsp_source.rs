use std::{sync::Arc, time::Duration};

use crate::pp_log::{PpLog, pp_error, pp_info};
use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    contract::{MediaKind, OutputContract, PortContract},
    control::{ControlReceiver, drain_control},
    element::{Element, ElementType, Source, SourceElement, element_pp_log},
    elements::RtspTransport,
    error::Result,
    pad::SrcPad,
};

use super::file_demuxer::StreamInfo;

/// Errors specific to `RtspSource`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum RtspSourceError {
    /// FFmpeg rejected connection setup, stream reading, or shutdown.
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),
    /// Seeking was requested on a live RTSP stream.
    #[error("RtspSource doesn't support seeking a live stream")]
    SeekUnsupported,
}

/// Construction-time options for [`RtspSource::open`].
#[derive(Debug, Clone, Copy)]
pub struct RtspOptions {
    /// Transport used for RTSP media delivery.
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
/// [`crate::elements::RtspSink`] (which publishes). One src pad per
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
pub struct RtspSource {
    pp_log: PpLog,
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
        dict.set("rtsp_transport", options.transport.as_ffmpeg_option());
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
            .map(|s| {
                // Per stream, from the medium the session announced: both
                // pads emit `MediaBuffer::Packet`, so only this tells an
                // audio stream apart from a video one. A medium this crate
                // does not model declares nothing and is left to the
                // runtime check.
                match MediaKind::packet_for(s.kind) {
                    Some(kind) => SrcPad::with_contract(
                        format!("src_{}", s.index),
                        OutputContract::Fixed(PortContract::of(kind)),
                    ),
                    None => SrcPad::new(format!("src_{}", s.index)),
                }
            })
            .collect();

        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::RtspSource, &name, None);
        pp_info!(
            pp_log: &pp_log,
            "opened: url={}, transport={:?}, {} stream(s)",
            url.as_ref(),
            options.transport,
            streams.len()
        );
        Ok((
            Self {
                name,
                pp_log,
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

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for RtspSource {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        &mut self.pads
    }
}

impl SourceElement for RtspSource {
    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        pp_info!(self, "started");
        loop {
            if drain_control(control, self, bus)?.stopped {
                pp_info!(self, "stopped");
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
                                &self.pp_log,
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
                    pp_error!(self, "read failed: {error}");
                    return Err(RtspSourceError::Ffmpeg(error).into());
                }
            }
        }
        for pad in self.pads.iter_mut() {
            pad.push_eos(&self.pp_log)?;
        }
        pp_info!(self, "event=eos phase=source_completed outcome=ok");
        Ok(())
    }

    fn seek(&mut self, _target: Duration) -> Result<Duration> {
        Err(RtspSourceError::SeekUnsupported.into())
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    /// `RtspOptions::timeout`'s whole reason for existing: without it
    /// ffmpeg applies *no* timeout at all and `open` blocks forever against
    /// a server that never answers. `192.0.2.1` is RFC 5737 TEST-NET-1,
    /// reserved for documentation and guaranteed not to be routed, so this
    /// exercises the "no answer" path rather than a fast connection refusal.
    ///
    /// Only the upper bound is asserted: a network that replies with an ICMP
    /// unreachable makes this fail even sooner, which is equally correct. The
    /// bound sits well under the OS-level TCP connect timeout (~21s on
    /// Windows, far longer on Linux), so a regression that stops passing the
    /// option through is what actually trips it.
    #[test]
    fn open_gives_up_within_the_configured_timeout_instead_of_hanging() {
        let options = RtspOptions {
            timeout: Duration::from_millis(500),
            ..Default::default()
        };

        let started = Instant::now();
        let result = RtspSource::open("rtsp", "rtsp://192.0.2.1:554/none", options);
        let elapsed = started.elapsed();

        assert!(
            result.is_err(),
            "opening an unroutable address must not succeed"
        );
        assert!(
            elapsed < Duration::from_secs(10),
            "open took {elapsed:?} — the configured timeout is not reaching ffmpeg"
        );
    }

    /// A caller that never touches `RtspOptions` still has to get a bounded
    /// `open`, since the default this type supplies is the only thing
    /// standing between them and ffmpeg's unbounded one.
    #[test]
    fn the_default_options_still_bound_the_connection() {
        let options = RtspOptions::default();

        assert_eq!(options.transport, RtspTransport::Tcp);
        assert!(
            options.timeout > Duration::ZERO,
            "the default timeout must not be ffmpeg's unbounded one"
        );
    }
}
