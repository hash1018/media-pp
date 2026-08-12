mod audio_resampler;
mod audio_volume;
pub(crate) mod decoder;
mod download;
mod encoder;
mod pacer;
mod scaler;
mod tee;
mod upload;

pub use audio_resampler::{AudioFormat, AudioResampler, AudioResamplerError};
pub use audio_volume::{AudioVolume, AudioVolumeError, AudioVolumeHandle, AudioVolumeOptions};
#[cfg(feature = "d3d11-renderer")]
pub use decoder::{D3d11Decoder, D3d11vaDecoderError};
#[cfg(feature = "d3d12-renderer")]
pub use decoder::{D3d12vaDecoder, D3d12vaDecoderError};
pub use decoder::{SwDecoder, SwDecoderError};
#[cfg(feature = "d3d11-renderer")]
pub use download::{D3d11Download, D3d11DownloadError};
pub use encoder::{
    AudioCodec, SwAudioEncoder, SwAudioEncoderError, SwAudioEncoderOptions, SwEncoder,
    SwEncoderError, SwEncoderOptions, VideoCodec,
};
pub use pacer::Pacer;
pub use scaler::{Scaler, ScalerError};
pub use tee::{Tee, TeeBuilder, TeeHandle};
#[cfg(feature = "d3d11-renderer")]
pub use upload::{D3d11Upload, D3d11UploadError};
#[cfg(feature = "d3d12-renderer")]
pub use upload::{D3d12Upload, D3d12UploadError};

/// `avcodec_receive_frame`/`avcodec_receive_packet` use `EAGAIN` to mean
/// "drained for now" and `EOF` to mean permanently drained after flush.
/// Every other error is a real codec failure and must be propagated.
fn is_codec_drain_boundary(error: &ffmpeg_next::Error) -> bool {
    match error {
        ffmpeg_next::Error::Eof => true,
        ffmpeg_next::Error::Other { errno } => *errno == ffmpeg_next::error::EAGAIN,
        _ => false,
    }
}

#[cfg(test)]
mod codec_error_tests {
    use super::*;

    #[test]
    fn only_eagain_and_eof_are_codec_drain_boundaries() {
        assert!(is_codec_drain_boundary(&ffmpeg_next::Error::Eof));
        assert!(is_codec_drain_boundary(&ffmpeg_next::Error::Other {
            errno: ffmpeg_next::error::EAGAIN,
        }));
        assert!(!is_codec_drain_boundary(&ffmpeg_next::Error::InvalidData));
    }
}
