use ffmpeg_next as ffmpeg;
use str0m::format::{Codec, CodecSpec};

use crate::error::Result;

use super::command::WebRtcError;

/// Parameters confirmed by the first actual RTP payload on an inbound
/// WebRTC track.
///
/// SDP exposes several possible codecs, while this value identifies the one
/// the remote sender actually used. It can derive the RTP [`Self::time_base`]
/// and the minimal FFmpeg parameters needed to open a decoder without asking
/// the application to duplicate codec mappings.
///
/// [`Self::decoder_parameters`] is deliberately decoder-only. Container
/// muxers generally need codec configuration that [`CodecSpec`] does not
/// contain, such as H.264 SPS/PPS-derived extradata and video dimensions,
/// before they can write a header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WebRtcStreamInfo {
    codec_spec: CodecSpec,
}

impl WebRtcStreamInfo {
    /// Returns the full str0m payload specification observed on the first
    /// media payload.
    pub fn codec_spec(&self) -> CodecSpec {
        self.codec_spec
    }

    /// Returns the selected codec family.
    pub fn codec(&self) -> Codec {
        self.codec_spec.codec
    }

    /// Returns the RTP timestamp time base derived from the codec clock rate.
    pub fn time_base(&self) -> Result<ffmpeg::Rational> {
        let clock_rate = clock_rate_i32(self.codec_spec)?;
        Ok(ffmpeg::Rational::new(1, clock_rate))
    }

    /// Builds the minimal FFmpeg codec parameters needed to open a decoder
    /// for this depayloaded WebRTC stream.
    ///
    /// Video dimensions and codec extradata are intentionally absent. str0m
    /// emits complete depayloaded frames (Annex B for H.264/H.265/H.266), so
    /// FFmpeg discovers those values from the bitstream. These parameters are
    /// not sufficient for opening a muxer that must write its container header
    /// before seeing packets.
    pub fn decoder_parameters(&self) -> Result<ffmpeg::codec::Parameters> {
        let (medium, id) = decoder_codec(self.codec_spec.codec)
            .ok_or(WebRtcError::UnsupportedDecoderCodec(self.codec_spec.codec))?;
        let audio = medium == ffmpeg::media::Type::Audio;
        let sample_rate = audio.then(|| clock_rate_i32(self.codec_spec)).transpose()?;
        let channels = if audio {
            match self.codec_spec.channels {
                Some(0) => {
                    return Err(WebRtcError::InvalidStreamChannelCount {
                        codec: self.codec_spec.codec,
                        channels: 0,
                    }
                    .into());
                }
                channels => channels,
            }
        } else {
            None
        };
        let mut parameters = ffmpeg::codec::Parameters::new();

        // SAFETY: `parameters` was just allocated and is exclusively owned
        // here. `codec_type`, `codec_id`, and the audio fields below are plain
        // AVCodecParameters fields. `av_channel_layout_default` initializes
        // the otherwise-zeroed owned channel layout without borrowing any
        // external storage.
        unsafe {
            let raw = parameters.as_mut_ptr();
            (*raw).codec_type = medium.into();
            (*raw).codec_id = id.into();
            if let Some(sample_rate) = sample_rate {
                (*raw).sample_rate = sample_rate;
                if let Some(channels) = channels {
                    ffmpeg::ffi::av_channel_layout_default(
                        &mut (*raw).ch_layout,
                        i32::from(channels),
                    );
                }
            }
        }

        Ok(parameters)
    }
}

impl From<CodecSpec> for WebRtcStreamInfo {
    fn from(codec_spec: CodecSpec) -> Self {
        Self { codec_spec }
    }
}

fn clock_rate_i32(codec_spec: CodecSpec) -> std::result::Result<i32, WebRtcError> {
    let clock_rate = codec_spec.clock_rate.get();
    i32::try_from(clock_rate).map_err(|_| WebRtcError::InvalidStreamClockRate {
        codec: codec_spec.codec,
        clock_rate,
    })
}

fn decoder_codec(codec: Codec) -> Option<(ffmpeg::media::Type, ffmpeg::codec::Id)> {
    use ffmpeg::{codec::Id, media::Type};

    match codec {
        Codec::Opus => Some((Type::Audio, Id::OPUS)),
        Codec::PCMU => Some((Type::Audio, Id::PCM_MULAW)),
        Codec::PCMA => Some((Type::Audio, Id::PCM_ALAW)),
        Codec::H264 => Some((Type::Video, Id::H264)),
        Codec::H265 => Some((Type::Video, Id::HEVC)),
        Codec::H266 => Some((Type::Video, Id::VVC)),
        Codec::Vp8 => Some((Type::Video, Id::VP8)),
        Codec::Vp9 => Some((Type::Video, Id::VP9)),
        Codec::Av1 => Some((Type::Video, Id::AV1)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use ffmpeg_next as ffmpeg;
    use str0m::{
        format::{Codec, CodecSpec, FormatParams},
        media::Frequency,
    };

    use super::WebRtcStreamInfo;
    use crate::elements::WebRtcError;

    fn spec(codec: Codec, clock_rate: Frequency, channels: Option<u8>) -> CodecSpec {
        CodecSpec {
            codec,
            clock_rate,
            channels,
            format: FormatParams::default(),
        }
    }

    #[test]
    fn h264_derives_decoder_parameters_and_rtp_time_base() {
        let info = WebRtcStreamInfo::from(spec(Codec::H264, Frequency::NINETY_KHZ, None));
        let parameters = info.decoder_parameters().expect("H.264 is supported");

        assert_eq!(info.codec(), Codec::H264);
        assert_eq!(info.time_base().unwrap(), ffmpeg::Rational::new(1, 90_000));
        assert_eq!(parameters.medium(), ffmpeg::media::Type::Video);
        assert_eq!(parameters.id(), ffmpeg::codec::Id::H264);
    }

    #[test]
    fn opus_derives_audio_rate_and_channel_layout() {
        let info = WebRtcStreamInfo::from(spec(Codec::Opus, Frequency::FORTY_EIGHT_KHZ, Some(2)));
        let parameters = info.decoder_parameters().expect("Opus is supported");

        assert_eq!(parameters.medium(), ffmpeg::media::Type::Audio);
        assert_eq!(parameters.id(), ffmpeg::codec::Id::OPUS);
        // SAFETY: read-only access to fields owned by the live `parameters`.
        unsafe {
            assert_eq!((*parameters.as_ptr()).sample_rate, 48_000);
            assert_eq!((*parameters.as_ptr()).ch_layout.nb_channels, 2);
        }
    }

    #[test]
    fn rtx_is_not_a_decodable_media_codec() {
        let info = WebRtcStreamInfo::from(spec(Codec::Rtx, Frequency::NINETY_KHZ, None));
        let error = info
            .decoder_parameters()
            .err()
            .expect("RTX must not create a decoder");

        assert!(matches!(
            error,
            crate::Error::WebRtcError(WebRtcError::UnsupportedDecoderCodec(Codec::Rtx))
        ));
    }

    #[test]
    fn an_unrepresentable_clock_rate_is_a_typed_error() {
        let clock_rate = Frequency::new(u32::MAX).unwrap();
        let info = WebRtcStreamInfo::from(spec(Codec::Opus, clock_rate, Some(2)));
        let error = info
            .time_base()
            .expect_err("FFmpeg Rational uses an i32 denominator");

        assert!(matches!(
            error,
            crate::Error::WebRtcError(WebRtcError::InvalidStreamClockRate {
                codec: Codec::Opus,
                clock_rate: u32::MAX,
            })
        ));
    }

    #[test]
    fn zero_audio_channels_are_rejected_before_ffi() {
        let info = WebRtcStreamInfo::from(spec(Codec::Opus, Frequency::FORTY_EIGHT_KHZ, Some(0)));
        let error = info
            .decoder_parameters()
            .err()
            .expect("zero channels must be rejected");

        assert!(matches!(
            error,
            crate::Error::WebRtcError(WebRtcError::InvalidStreamChannelCount {
                codec: Codec::Opus,
                channels: 0,
            })
        ));
    }
}
