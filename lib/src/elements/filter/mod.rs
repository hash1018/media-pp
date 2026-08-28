//! Elements that are both a [`Sink`](crate::element::Sink) and a
//! [`Source`](crate::element::Source).
//!
//! Codecs, scalers, pixel-format conversion, GPU upload and download, audio
//! resampling and gain, chroma keying, and the two elements that change *when*
//! rather than *what* — [`Pacer`], which holds a frame until its presentation
//! time, [`VideoSynchronizer`], and [`ChangeGate`], which forwards a picture
//! only when it is not the one it forwarded last. [`Tee`] is here too, as the one filter
//! whose fan-out can change while the pipeline runs.
//!
//! Where the same job exists on more than one backend the types are separate
//! and prefixed (`Sw*`, `Cuda*`, `D3d11*`, `D3d12*`) rather than one type with
//! a runtime switch, because the buffers they accept genuinely differ: a GPU
//! filter requires frames already resident on the device that created them.

mod audio_resampler;
mod audio_volume;
mod change_gate;
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
pub use change_gate::{ChangeGate, ChangeGateError};
pub use chroma_key::{ChromaKeyMethod, ChromaKeyOptions, SwChromaKey, SwChromaKeyError};
#[cfg(all(target_os = "windows", feature = "d3d11"))]
pub use chroma_key::{D3d11ChromaKey, D3d11ChromaKeyError};
#[cfg(feature = "cuda")]
pub use convert::{CudaConverter, CudaConverterError};
#[cfg(feature = "cuda")]
pub use decoder::{CudaDecoder, CudaDecoderError};
#[cfg(all(target_os = "windows", feature = "d3d11"))]
pub use decoder::{D3d11Decoder, D3d11DecoderError};
#[cfg(all(target_os = "windows", feature = "d3d12"))]
pub use decoder::{D3d12Decoder, D3d12DecoderError};
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
#[cfg(all(target_os = "windows", feature = "d3d12"))]
pub use scaler::{D3d12Scaler, D3d12ScalerError};
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
