#[cfg(feature = "dx12-renderer")]
pub(crate) mod d3d12va_decoder;
mod sw_decoder;

#[cfg(feature = "dx12-renderer")]
pub use d3d12va_decoder::{D3d12vaDecoder, D3d12vaDecoderError};
pub use sw_decoder::{SwDecoder, SwDecoderError};
