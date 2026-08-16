#[cfg(feature = "d3d11")]
pub(crate) mod d3d11va_decoder;
#[cfg(feature = "d3d12")]
pub(crate) mod d3d12va_decoder;

#[cfg(feature = "d3d11")]
pub use d3d11va_decoder::{D3d11Decoder, D3d11vaDecoderError};
#[cfg(feature = "d3d12")]
pub use d3d12va_decoder::{D3d12vaDecoder, D3d12vaDecoderError};
