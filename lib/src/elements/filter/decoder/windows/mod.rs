#[cfg(feature = "d3d11")]
mod d3d11va_decoder;
#[cfg(feature = "d3d12")]
mod d3d12va_decoder;

#[cfg(feature = "d3d11")]
pub use d3d11va_decoder::{D3d11Decoder, D3d11DecoderError};
#[cfg(feature = "d3d12")]
pub use d3d12va_decoder::{D3d12Decoder, D3d12DecoderError};
