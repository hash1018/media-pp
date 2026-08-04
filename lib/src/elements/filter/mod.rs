#[cfg(feature = "d3d12-renderer")]
mod d3d12_upload;
pub(crate) mod decoder;
mod pacer;
mod scaler;
mod sw_encoder;
mod tee;

#[cfg(feature = "d3d12-renderer")]
pub use d3d12_upload::{D3d12Upload, D3d12UploadError};
#[cfg(feature = "d3d12-renderer")]
pub use decoder::{D3d12vaDecoder, D3d12vaDecoderError};
pub use decoder::{SwDecoder, SwDecoderError};
pub use pacer::Pacer;
pub use scaler::{Scaler, ScalerError};
pub use sw_encoder::{SwEncoder, SwEncoderError, SwEncoderOptions, VideoCodec};
pub use tee::{Tee, TeeHandle};
