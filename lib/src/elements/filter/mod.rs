mod audio_resampler;
pub(crate) mod decoder;
mod encoder;
mod pacer;
mod scaler;
mod tee;
mod upload;

pub use audio_resampler::{AudioFormat, AudioResampler, AudioResamplerError};
#[cfg(feature = "d3d11-renderer")]
pub use decoder::{D3d11Decoder, D3d11vaDecoderError};
#[cfg(feature = "d3d12-renderer")]
pub use decoder::{D3d12vaDecoder, D3d12vaDecoderError};
pub use decoder::{SwDecoder, SwDecoderError};
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
