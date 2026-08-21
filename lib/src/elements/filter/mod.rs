mod audio_resampler;
mod audio_volume;
pub(crate) mod chroma_key;
pub(crate) mod convert;
pub(crate) mod decoder;
mod download;
mod encoder;
mod pacer;
pub(crate) mod scaler;
mod tee;
pub(crate) mod upload;
mod video_synchronizer;

pub use audio_resampler::{AudioResampler, AudioResamplerError};
pub use audio_volume::{AudioVolume, AudioVolumeError, AudioVolumeHandle, AudioVolumeOptions};
pub use chroma_key::{ChromaKeyMethod, ChromaKeyOptions, SwChromaKey, SwChromaKeyError};
#[cfg(all(target_os = "windows", feature = "d3d11"))]
pub use chroma_key::{D3d11ChromaKey, D3d11ChromaKeyError};
#[cfg(feature = "cuda")]
pub use convert::{CudaConverter, CudaConverterError};
#[cfg(feature = "cuda")]
pub use decoder::{CudaDecoder, CudaDecoderError};
#[cfg(all(target_os = "windows", feature = "d3d11"))]
pub use decoder::{D3d11Decoder, D3d11vaDecoderError};
#[cfg(all(target_os = "windows", feature = "d3d12"))]
pub use decoder::{D3d12vaDecoder, D3d12vaDecoderError};
pub use decoder::{SwDecoder, SwDecoderError};
#[cfg(feature = "cuda")]
pub use download::{CudaDownload, CudaDownloadError};
#[cfg(all(target_os = "windows", feature = "d3d11"))]
pub use download::{D3d11Download, D3d11DownloadError};
#[cfg(all(target_os = "windows", feature = "d3d12"))]
pub use download::{D3d12Download, D3d12DownloadError};
pub use encoder::{
    AudioCodec, SwAudioEncoder, SwAudioEncoderError, SwAudioEncoderOptions, SwEncoder,
    SwEncoderError, SwEncoderOptions, VideoCodec,
};
#[cfg(feature = "cuda")]
pub use encoder::{CudaCodec, CudaEncoder, CudaEncoderError, CudaEncoderOptions};
#[cfg(all(target_os = "windows", feature = "d3d11"))]
pub use encoder::{
    D3d11NvencCodec, D3d11NvencEncoder, D3d11NvencEncoderError, D3d11NvencEncoderOptions,
    D3d11NvencInputFormat,
};
pub use pacer::{Pacer, PacerError};
#[cfg(feature = "cuda")]
pub use scaler::{CudaScaler, CudaScalerError, CudaScalerInterp};
#[cfg(all(target_os = "windows", feature = "d3d11"))]
pub use scaler::{D3d11Scaler, D3d11ScalerError, D3d11ScalerFormat};
pub use scaler::{SwScaler, SwScalerError};
pub use tee::{Tee, TeeBuilder, TeeHandle};
#[cfg(feature = "cuda")]
pub use upload::{CudaUpload, CudaUploadError};
#[cfg(all(target_os = "windows", feature = "d3d11"))]
pub use upload::{D3d11Upload, D3d11UploadError};
#[cfg(all(target_os = "windows", feature = "d3d12"))]
pub use upload::{D3d12Upload, D3d12UploadError};
pub use video_synchronizer::{VideoSynchronizer, VideoSynchronizerError};

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
