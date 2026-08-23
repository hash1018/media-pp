use std::{ffi::CString, ptr, sync::Arc};

use crate::pp_log::{PpLog, pp_error, pp_info};
use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, PortContract},
    control::ControlMsg,
    element::{Element, ElementType, Sink, element_pp_log},
    elements::RtspTransport,
    error::Result,
};

/// Errors produced while opening or writing an [`RtspSink`].
#[derive(Debug, ThisError)]
pub enum RtspSinkError {
    /// FFmpeg rejected connection setup or packet writing.
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),

    /// The sink received a decoded frame instead of compressed packet data.
    #[error(
        "RtspSink only remuxes compressed Packets, got a decoded {0}; \
         connect an encoder or demuxer packet pad instead"
    )]
    UnsupportedBuffer(&'static str),

    /// The URL contains an interior NUL byte rejected by FFmpeg's C API.
    #[error("RTSP URL contains a NUL byte")]
    InvalidUrl,
}

/// Publishes one compressed packet stream to an already-running RTSP server.
///
/// [`RtspSink::open`] performs the RTSP `ANNOUNCE`/`SETUP`/`RECORD`
/// handshake through FFmpeg, so the server must already be listening at
/// `url` and must permit publishing to that path. The server can be
/// MediaMTX or any other implementation that accepts RTSP publishing;
/// this element does not start, stop, or otherwise depend on a particular
/// server process.
///
/// This is a remuxing sink, not an encoder. Incoming buffers must be
/// compressed [`MediaBuffer::Packet`] values whose codec parameters and
/// time base match the values passed to [`RtspSink::open`]. Place a
/// [`crate::elements::Pacer`] upstream when publishing packets from a file,
/// otherwise the file will be sent faster than real time.
///
/// The current sink publishes one stream. Build separate sinks and RTSP
/// paths when publishing independent streams.
pub struct RtspSink {
    pp_log: PpLog,
    name: Arc<str>,
    url: String,
    output: ffmpeg::format::context::Output,
    input_time_base: ffmpeg::Rational,
    last_output_dts: Option<i64>,
    last_output_pts: Option<i64>,
    pts_offset: i64,
    pending_seek: bool,
}

impl RtspSink {
    /// Connects to `url` and starts publishing.
    ///
    /// `params` and `time_base` must describe every packet subsequently
    /// passed to [`Sink::consume`]. TCP is the most reliable transport for
    /// general networks; UDP is useful when the network path and server
    /// permit the negotiated RTP/RTCP ports.
    pub fn open(
        name: impl Into<String>,
        url: impl Into<String>,
        transport: RtspTransport,
        params: ffmpeg::codec::Parameters,
        time_base: ffmpeg::Rational,
    ) -> Result<Self> {
        let url = url.into();
        let mut output = alloc_output(&url)?;

        {
            let mut stream = output
                .add_stream(ffmpeg::encoder::find(ffmpeg::codec::Id::None))
                .map_err(RtspSinkError::from)?;
            stream.set_parameters(params);
            // Avoid codec-tag incompatibilities when the input packet came
            // from a container with a different tag convention.
            // SAFETY: `as_mut_ptr` on parameters this stream owns, written before the
            // stream is handed to the muxer — see the comment beside it for why the tag
            // is cleared at all.
            unsafe {
                (*stream.parameters().as_mut_ptr()).codec_tag = 0;
            }
            stream.set_time_base(time_base);
        }

        let mut options = ffmpeg::Dictionary::new();
        options.set("rtsp_transport", transport.as_ffmpeg_option());
        output
            .write_header_with(options)
            .map_err(RtspSinkError::from)?;

        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::RtspSink, &name, None);
        pp_info!(pp_log: &pp_log, "publishing: url={url}, transport={transport:?}");

        Ok(Self {
            pp_log,
            name,
            url,
            output,
            input_time_base: time_base,
            last_output_dts: None,
            last_output_pts: None,
            pts_offset: 0,
            pending_seek: false,
        })
    }

    /// URL this sink publishes to.
    pub fn url(&self) -> &str {
        &self.url
    }
}

/// Allocates an RTSP muxer without opening a generic `AVIOContext`.
///
/// RTSP is a libavformat muxer, not a generic AVIO protocol. Its muxer
/// owns the control and RTP sockets internally during header/packet writes,
/// while `ffmpeg_next::format::output_as` attempts an incompatible generic
/// `avio_open2` first on FFmpeg builds where `rtsp` is not an AVIO protocol.
fn alloc_output(url: &str) -> Result<ffmpeg::format::context::Output> {
    let c_url = CString::new(url).map_err(|_| RtspSinkError::InvalidUrl)?;
    let c_format = CString::new("rtsp").expect("static format name contains no NUL");

    // SAFETY: `c_format` and `c_url` are live NUL-terminated `CString`s, and
    // `context` is a live local. Every path below checks it before use, and the
    // failure paths free what was allocated.
    unsafe {
        let mut context: *mut ffi::AVFormatContext = ptr::null_mut();
        let result = ffi::avformat_alloc_output_context2(
            &mut context,
            ptr::null_mut(),
            c_format.as_ptr(),
            c_url.as_ptr(),
        );
        if result < 0 {
            return Err(RtspSinkError::Ffmpeg(ffmpeg::Error::from(result)).into());
        }

        Ok(ffmpeg::format::context::Output::wrap(context))
    }
}

impl Element for RtspSink {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::RtspSink
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Sink for RtspSink {
    /// Republishes encoded data as-is; it has no encoder of its own.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::of(MediaKind::Packet))
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Packet(packet) => {
                let mut packet = (*packet).clone();
                let output_time_base = self
                    .output
                    .stream(0)
                    .expect("stream 0 was added by RtspSink::open")
                    .time_base();
                packet.rescale_ts(self.input_time_base, output_time_base);

                if let Some(raw_pts) = packet.pts() {
                    if self.pending_seek {
                        // Keep the published timeline monotonic across an
                        // upstream seek. DTS is the muxer's hard ordering
                        // requirement; PTS is the fallback for packets that
                        // carry no DTS.
                        self.pts_offset = match (self.last_output_dts, packet.dts()) {
                            (Some(last_dts), Some(raw_dts)) => last_dts + 1 - raw_dts,
                            _ => match self.last_output_pts {
                                Some(last_pts) => last_pts + 1 - raw_pts,
                                None => 0,
                            },
                        };
                        self.pending_seek = false;
                    }

                    let corrected_pts = raw_pts + self.pts_offset;
                    packet.set_pts(Some(corrected_pts));
                    if let Some(raw_dts) = packet.dts() {
                        let corrected_dts = raw_dts + self.pts_offset;
                        packet.set_dts(Some(corrected_dts));
                        self.last_output_dts = Some(corrected_dts);
                    }
                    self.last_output_pts = Some(corrected_pts);
                }

                packet.set_stream(0);
                packet.set_position(-1);
                packet
                    .write_interleaved(&mut self.output)
                    .map_err(RtspSinkError::from)
                    .map_err(Into::into)
                    .inspect_err(|error| pp_error!(self, "write_interleaved failed: {error}"))
            }
            MediaBuffer::Eos => self
                .output
                .write_trailer()
                .map_err(RtspSinkError::from)
                .map_err(Into::into)
                .inspect_err(|error| pp_error!(self, "write_trailer failed: {error}")),
            MediaBuffer::Video(_) => {
                pp_error!(self, "unsupported buffer: Video");
                Err(RtspSinkError::UnsupportedBuffer("Video").into())
            }
            MediaBuffer::Audio(_) => {
                pp_error!(self, "unsupported buffer: Audio");
                Err(RtspSinkError::UnsupportedBuffer("Audio").into())
            }
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        match msg {
            ControlMsg::Seek(_) => self.pending_seek = true,
            ControlMsg::Pause | ControlMsg::Resume | ControlMsg::Stop => {}
        }
        Ok(())
    }
}

impl Drop for RtspSink {
    fn drop(&mut self) {
        pp_info!(
            self,
            "dropped: closing publisher connection to {}",
            self.url
        );
    }
}

#[cfg(test)]
mod tests {
    use ffmpeg_next as ffmpeg;

    use super::{RtspSink, RtspSinkError};
    use crate::{elements::RtspTransport, error::Error};

    #[test]
    fn rejects_a_url_containing_a_nul_byte_before_connecting() {
        let result = RtspSink::open(
            "rtsp",
            "rtsp://127.0.0.1:8554/stream\0invalid",
            RtspTransport::Tcp,
            ffmpeg::codec::Parameters::new(),
            ffmpeg::Rational(1, 1_000),
        );

        assert!(matches!(
            result,
            Err(Error::RtspSinkError(RtspSinkError::InvalidUrl))
        ));
    }
}
