use std::sync::Arc;

use crate::pp_log::{PpLog, pp_error, pp_info};
use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    pad::SrcPad,
    pool::UnboundObjectPool,
};

use crate::elements::filter::is_codec_drain_boundary;

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
    pp_log: PpLog,
    name: Arc<str>,
    kind: Kind,
    pad: SrcPad,
    /// Reused across every decoded video frame instead of allocating a
    /// fresh one each time — see [`UnboundObjectPool`]'s docs. Starts
    /// empty: decoded format/dimensions aren't known until the first
    /// frame actually comes out of the decoder, so `init` just makes an
    /// empty frame (`avcodec_receive_frame` allocates it on first use,
    /// same as before this existed) and the pool fills organically as
    /// frames get returned. Unused (harmlessly) if this turns out to be
    /// an audio decoder — `MediaBuffer::Audio` isn't pooled.
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
}

impl SwDecoder {
    /// `params` should come from the stream you want to decode — see
    /// [`crate::elements::FileDemuxer::stream_parameters`].
    pub fn new(
        name: impl Into<String>,
        params: ffmpeg::codec::Parameters,
    ) -> Result<Self, SwDecoderError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::SwDecoder, &name, None);
        let context = ffmpeg::codec::context::Context::from_parameters(params)?;

        let kind = match context.medium() {
            ffmpeg::media::Type::Video => Kind::Video(context.decoder().video()?),
            ffmpeg::media::Type::Audio => Kind::Audio(context.decoder().audio()?),
            other => return Err(SwDecoderError::UnsupportedMediaType(other)),
        };

        let pad = SrcPad::new(format!("{name}_src"));
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        pp_info!(
            pp_log: &pp_log,
            "opened: {}",
            match &kind {
                Kind::Video(d) => format!("video, codec={:?}", d.id()),
                Kind::Audio(d) => format!("audio, codec={:?}", d.id()),
            }
        );
        Ok(Self {
            name,
            pp_log,
            kind,
            pad,
            pool,
        })
    }
}

impl Element for SwDecoder {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::SwDecoder
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
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
                        .inspect_err(|error| pp_error!(self, "send_packet failed: {error}"))
                        .map_err(SwDecoderError::from)?;
                    drain_video(decoder, &mut self.pad, &self.pool)
                }
                Kind::Audio(decoder) => {
                    decoder
                        .send_packet(&*packet)
                        .inspect_err(|error| pp_error!(self, "send_packet failed: {error}"))
                        .map_err(SwDecoderError::from)?;
                    drain_audio(decoder, &mut self.pad)
                }
            },
            MediaBuffer::Eos => {
                match &mut self.kind {
                    Kind::Video(decoder) => {
                        decoder
                            .send_eof()
                            .inspect_err(|error| pp_error!(self, "send_eof failed: {error}"))
                            .map_err(SwDecoderError::from)?;
                        drain_video(decoder, &mut self.pad, &self.pool)?;
                    }
                    Kind::Audio(decoder) => {
                        decoder
                            .send_eof()
                            .inspect_err(|error| pp_error!(self, "send_eof failed: {error}"))
                            .map_err(SwDecoderError::from)?;
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

fn drain_video(
    decoder: &mut ffmpeg::decoder::Video,
    pad: &mut SrcPad,
    pool: &UnboundObjectPool<ffmpeg::frame::Video>,
) -> crate::error::Result<()> {
    let mut frame = pool.get();
    loop {
        match decoder.receive_frame(&mut frame) {
            Ok(()) => {
                pad.push(MediaBuffer::Video(Arc::new(frame)))?;
                frame = pool.get();
            }
            Err(error) if is_codec_drain_boundary(&error) => break,
            Err(error) => return Err(SwDecoderError::from(error).into()),
        }
    }
    Ok(())
}

fn drain_audio(decoder: &mut ffmpeg::decoder::Audio, pad: &mut SrcPad) -> crate::error::Result<()> {
    let mut frame = ffmpeg::frame::Audio::empty();
    loop {
        match decoder.receive_frame(&mut frame) {
            Ok(()) => {
                pad.push(MediaBuffer::Audio(Arc::new(frame)))?;
                frame = ffmpeg::frame::Audio::empty();
            }
            Err(error) if is_codec_drain_boundary(&error) => break,
            Err(error) => return Err(SwDecoderError::from(error).into()),
        }
    }
    Ok(())
}
