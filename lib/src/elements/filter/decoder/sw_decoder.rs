use std::sync::Arc;

use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source},
    pad::SrcPad,
};

/// Errors specific to `SwDecoder`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum SwDecoderError {
    #[error("unsupported media type: {0:?}")]
    UnsupportedMediaType(ffmpeg::media::Type),

    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),
}

enum Kind {
    Video(ffmpeg::decoder::Video),
    Audio(ffmpeg::decoder::Audio),
}

/// Decodes one stream's `Packet`s into `Frame`s in software (plain
/// libavcodec, no hardware acceleration). A `Filter`: receives via `Sink`,
/// pushes what it produces into its own (single) src pad.
///
/// One packet can turn into zero, one, or several frames (B-frame
/// reordering, decoder buffering, ...) — `consume` just drains
/// `receive_frame` in a loop after every `send_packet`/`send_eof`, pushing
/// however many frames come out.
pub struct SwDecoder {
    name: String,
    kind: Kind,
    pad: SrcPad,
}

impl SwDecoder {
    /// `params` should come from the stream you want to decode — see
    /// [`crate::elements::FileDemuxer::stream_parameters`].
    pub fn new(
        name: impl Into<String>,
        params: ffmpeg::codec::Parameters,
    ) -> Result<Self, SwDecoderError> {
        let name = name.into();
        let context = ffmpeg::codec::context::Context::from_parameters(params)?;

        let kind = match context.medium() {
            ffmpeg::media::Type::Video => Kind::Video(context.decoder().video()?),
            ffmpeg::media::Type::Audio => Kind::Audio(context.decoder().audio()?),
            other => return Err(SwDecoderError::UnsupportedMediaType(other)),
        };

        let pad = SrcPad::new(format!("{name}_src"));
        Ok(Self { name, kind, pad })
    }
}

impl Element for SwDecoder {
    fn name(&self) -> &str {
        &self.name
    }

    fn element_type(&self) -> ElementType {
        ElementType::SwDecoder
    }
}

impl Source for SwDecoder {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for SwDecoder {
    fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
        match buf {
            MediaBuffer::Packet(packet) => match &mut self.kind {
                Kind::Video(decoder) => {
                    decoder
                        .send_packet(&*packet)
                        .map_err(SwDecoderError::from)?;
                    drain_video(decoder, &mut self.pad)
                }
                Kind::Audio(decoder) => {
                    decoder
                        .send_packet(&*packet)
                        .map_err(SwDecoderError::from)?;
                    drain_audio(decoder, &mut self.pad)
                }
            },
            MediaBuffer::Eos => {
                match &mut self.kind {
                    Kind::Video(decoder) => {
                        let _ = decoder.send_eof();
                        drain_video(decoder, &mut self.pad)?;
                    }
                    Kind::Audio(decoder) => {
                        let _ = decoder.send_eof();
                        drain_audio(decoder, &mut self.pad)?;
                    }
                }
                self.pad.push(MediaBuffer::Eos)
            }
            other => {
                let _ = other;
                Ok(())
            }
        }
    }

    fn control(&mut self, msg: ControlMsg) -> crate::error::Result<()> {
        // `Stop`: no local reaction needed — abandon means there's
        // nothing to flush before this decoder's own `Drop` frees the
        // codec context.
        //
        // `Seek`: unlike `Stop`, playback continues after this from a
        // new position, so leftover reference/reordering state from
        // before the jump has to go now — otherwise the next packets
        // (from the new position) would decode against stale state and
        // produce corrupt frames.
        if let ControlMsg::Seek(_) = msg {
            match &mut self.kind {
                Kind::Video(decoder) => decoder.flush(),
                Kind::Audio(decoder) => decoder.flush(),
            }
        }
        self.pad.control(msg)
    }
}

fn drain_video(decoder: &mut ffmpeg::decoder::Video, pad: &mut SrcPad) -> crate::error::Result<()> {
    let mut frame = ffmpeg::frame::Video::empty();
    while decoder.receive_frame(&mut frame).is_ok() {
        pad.push(MediaBuffer::Video(Arc::new(frame)))?;
        frame = ffmpeg::frame::Video::empty();
    }
    Ok(())
}

fn drain_audio(decoder: &mut ffmpeg::decoder::Audio, pad: &mut SrcPad) -> crate::error::Result<()> {
    let mut frame = ffmpeg::frame::Audio::empty();
    while decoder.receive_frame(&mut frame).is_ok() {
        pad.push(MediaBuffer::Audio(Arc::new(frame)))?;
        frame = ffmpeg::frame::Audio::empty();
    }
    Ok(())
}
