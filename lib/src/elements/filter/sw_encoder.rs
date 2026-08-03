use std::sync::Arc;

use ffmpeg_next as ffmpeg;
use rust_hlog::{HLog, herror, hinfo};
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_hlog},
    error::Result,
    pad::SrcPad,
};

/// Errors specific to `SwEncoder`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum SwEncoderError {
    #[error(
        "encoder {0:?} not found — this ffmpeg build wasn't compiled with it \
         (see VideoCodec's own docs: GPL-licensed ones need --enable-gpl; \
         run `ffmpeg -encoders` to see what's actually available)"
    )]
    CodecNotFound(String),

    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),
}

/// Which software encoder to open. Whatever's picked, [`SwEncoder::new`]
/// fails with [`SwEncoderError::CodecNotFound`] (not a panic) if the
/// linked ffmpeg build doesn't actually have it — this crate never
/// vendors any of these, it's whatever the local ffmpeg install was built
/// with (check with `ffmpeg -encoders`).
///
/// GPL-licensed encoders need an ffmpeg build compiled with
/// `--enable-gpl` (separate from — and unlike — FFmpeg's own native
/// `h264`/`hevc` **decoders**, which `SwDecoder` already uses and which
/// carry no such requirement; GPL only enters the picture on the encode
/// side, through these specific libraries): [`VideoCodec::H264`] and
/// [`VideoCodec::H265`] are GPL. Every other variant here is a
/// permissively-licensed (BSD/similar) alternative that needs no special
/// build flag beyond being enabled at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoCodec {
    /// `libx264` — H.264. GPL.
    H264,
    /// `libopenh264` — H.264, Cisco's BSD-2-Clause encoder (Cisco covers
    /// H.264 patent royalties for binary redistributions of it). The
    /// non-GPL alternative to [`VideoCodec::H264`].
    OpenH264,
    /// `libx265` — H.265/HEVC. GPL.
    H265,
    /// `libkvazaar` — H.265/HEVC, BSD-2-Clause. The non-GPL alternative
    /// to [`VideoCodec::H265`].
    Kvazaar,
    /// `libvpx` — VP8, BSD-3-Clause.
    Vp8,
    /// `libvpx-vp9` — VP9, BSD-3-Clause (same `libvpx` project as
    /// [`VideoCodec::Vp8`], different encoder name).
    Vp9,
    /// `libaom-av1` — AV1's reference encoder, BSD-2-Clause. Most
    /// broadly compatible AV1 output, but much slower than
    /// [`VideoCodec::Svtav1`] — prefer that one unless you specifically
    /// need `libaom`'s own encoding behavior.
    Av1,
    /// `libsvtav1` — AV1 via Intel's SVT-AV1, BSD-2-Clause-Patent. Far
    /// faster than [`VideoCodec::Av1`] (`libaom-av1`) at a given quality
    /// target — the practical default for real-time AV1 encoding.
    Svtav1,
}

impl VideoCodec {
    fn encoder_name(self) -> &'static str {
        match self {
            VideoCodec::H264 => "libx264",
            VideoCodec::OpenH264 => "libopenh264",
            VideoCodec::H265 => "libx265",
            VideoCodec::Kvazaar => "libkvazaar",
            VideoCodec::Vp8 => "libvpx",
            VideoCodec::Vp9 => "libvpx-vp9",
            VideoCodec::Av1 => "libaom-av1",
            VideoCodec::Svtav1 => "libsvtav1",
        }
    }
}

/// Construction-time options for [`SwEncoder::new`]. `width`/`height`/
/// `time_base` must already be known — same convention as
/// [`crate::elements::Scaler`]/[`crate::elements::Pacer`] — rather than
/// inferred from the first frame, since `avcodec_open2` needs them set
/// before this can be opened at all.
#[derive(Debug, Clone, Copy)]
pub struct SwEncoderOptions {
    pub codec: VideoCodec,
    pub width: u32,
    pub height: u32,
    /// Must match the `pts` unit of whatever frames this receives — e.g.
    /// [`crate::elements::TestVideoSource::time_base`] or a demuxed
    /// stream's own `time_base` if this is a transcode pipeline.
    pub time_base: ffmpeg::Rational,
    pub bit_rate: usize,
}

/// Encodes `Pixel::YUV420P` `Video` frames into `Packet`s via a software
/// encoder (see [`VideoCodec`]) — the mirror image of
/// [`crate::elements::SwDecoder`]'s decode direction. A `Filter`: receives
/// via `Sink`, pushes what it produces into its own (single) src pad.
///
/// One frame can turn into zero or one packets per `send_frame` (B-frame
/// reordering delays some frames' packets until later ones arrive, or
/// until `Eos` flushes whatever's left) — `consume` drains `receive_packet`
/// in a loop after every `send_frame`/`send_eof`, same shape as
/// `SwDecoder`'s own `receive_frame` drain loop.
#[rust_hlog::hlog]
pub struct SwEncoder {
    name: Arc<str>,
    encoder: ffmpeg::encoder::Video,
    pad: SrcPad,
}

impl SwEncoder {
    pub fn new(name: impl Into<String>, options: SwEncoderOptions) -> Result<Self> {
        let encoder_name = options.codec.encoder_name();
        let codec = ffmpeg::encoder::find_by_name(encoder_name)
            .ok_or_else(|| SwEncoderError::CodecNotFound(encoder_name.into()))?;

        let mut context = ffmpeg::codec::context::Context::new_with_codec(codec);
        context.set_time_base(options.time_base);

        let mut video = context.encoder().video().map_err(SwEncoderError::from)?;
        video.set_width(options.width);
        video.set_height(options.height);
        video.set_format(ffmpeg::format::Pixel::YUV420P);
        video.set_time_base(options.time_base);
        video.set_frame_rate(Some(ffmpeg::Rational::new(
            options.time_base.denominator(),
            options.time_base.numerator(),
        )));
        video.set_bit_rate(options.bit_rate);

        let encoder = video.open_as(codec).map_err(SwEncoderError::from)?;

        let name: Arc<str> = name.into().into();
        let hlog = element_hlog(ElementType::SwEncoder, &name, None);
        let pad = SrcPad::new(format!("{name}_src"));
        hinfo!(
            hlog: &hlog,
            "opened: codec={encoder_name}, {}x{}, bit_rate={}",
            options.width,
            options.height,
            options.bit_rate
        );
        Ok(Self {
            name,
            hlog,
            encoder,
            pad,
        })
    }

    /// This encoder's own codec parameters — what you need to construct
    /// a matching [`crate::elements::SwDecoder`] to decode the `Packet`s
    /// this produces, when there's no container/demuxer in the loop to
    /// get them from otherwise (e.g. encoding straight into a `Tee`/RTSP
    /// sink, or decoding straight back out for a round-trip smoke test).
    pub fn parameters(&self) -> ffmpeg::codec::Parameters {
        ffmpeg::codec::Parameters::from(&self.encoder)
    }

    fn drain(&mut self) -> Result<()> {
        let mut packet = ffmpeg::Packet::empty();
        while self.encoder.receive_packet(&mut packet).is_ok() {
            self.pad.push(MediaBuffer::Packet(Arc::new(packet)))?;
            packet = ffmpeg::Packet::empty();
        }
        Ok(())
    }
}

impl Element for SwEncoder {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::SwEncoder
    }

    fn hlog(&self) -> &HLog {
        &self.hlog
    }

    fn hlog_mut(&mut self) -> &mut HLog {
        &mut self.hlog
    }
}

impl Source for SwEncoder {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for SwEncoder {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Video(frame) => {
                self.encoder
                    .send_frame(&frame)
                    .inspect_err(|error| herror!(self, "send_frame failed: {error}"))
                    .map_err(SwEncoderError::from)?;
                self.drain()
            }
            MediaBuffer::Eos => {
                let _ = self.encoder.send_eof();
                self.drain()?;
                self.pad.push(MediaBuffer::Eos)
            }
            other => {
                let _ = other;
                Ok(())
            }
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        // No local state needs resetting on `Seek` — unlike a decoder,
        // this encoder has no documented "position" corruption to flush
        // against (see `SwDecoder::control`'s own reasoning, which this
        // deliberately doesn't mirror); a mid-GOP discontinuity is at
        // worst a slightly suboptimal GOP boundary, not corrupt output.
        // `Stop`: nothing to flush either — abandon means the codec
        // context just gets freed in `Drop`.
        self.pad.control(msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `SwEncoder::new` should fail cleanly (not panic) when the linked
    /// ffmpeg build wasn't compiled with the requested encoder — the
    /// GPL-only `libx264`/`libx265` in particular aren't guaranteed to be
    /// present (this crate never vendors them; it's whatever the local
    /// ffmpeg install was built with). Real regression coverage on
    /// whichever build runs this test, not a mock.
    #[test]
    fn codec_not_found_is_a_clean_error_not_a_panic() {
        for codec in [
            VideoCodec::H264,
            VideoCodec::OpenH264,
            VideoCodec::H265,
            VideoCodec::Kvazaar,
            VideoCodec::Vp8,
            VideoCodec::Vp9,
            VideoCodec::Av1,
            VideoCodec::Svtav1,
        ] {
            let result = SwEncoder::new(
                "encoder",
                SwEncoderOptions {
                    codec,
                    width: 640,
                    height: 480,
                    time_base: ffmpeg::Rational::new(1, 30),
                    bit_rate: 1_000_000,
                },
            );
            // Whether it succeeds or fails depends on how this machine's
            // ffmpeg was built — either is fine, just never a panic. If it
            // fails, it must be *this* error, not some other ffmpeg
            // failure mode.
            if let Err(error) = result {
                assert!(
                    matches!(
                        error,
                        crate::error::Error::SwEncoderError(SwEncoderError::CodecNotFound(_))
                    ),
                    "expected CodecNotFound, got {error:?}"
                );
            }
        }
    }
}
