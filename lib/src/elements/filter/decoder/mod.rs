//! Every element that turns `Packet`s into `Video`/`Audio` frames,
//! grouped here by *what they do* rather than by which GPU API backs
//! them — [`d3d11va_decoder`]/[`d3d12va_decoder`] sit next to
//! [`sw_decoder`] instead of off in their own per-API trees, so "what
//! decoders exist" is answerable by listing this one directory.
//!
//! `d3d11va_decoder`/`d3d12va_decoder` are `pub(crate)`, not private:
//! their small `pub(crate)` helpers
//! ([`d3d11va_decoder::wrap_d3d11_texture`]/
//! [`d3d11va_decoder::create_hw_device_ctx`], and D3D12's equivalents)
//! are reused outside this module — by
//! [`crate::elements::filter::upload`]'s own D3D11/D3D12 variants, by
//! [`crate::elements::sink::renderer`]'s own D3D11/D3D12 variants (to
//! read the frames this module's decoders produce), and by
//! [`crate::elements::DxgiCaptureSource`]'s GPU capture mode
//! (which shares the D3D11 frame representation without itself decoding
//! anything).

#[cfg(feature = "d3d11-renderer")]
pub(crate) mod d3d11va_decoder;
#[cfg(feature = "d3d12-renderer")]
pub(crate) mod d3d12va_decoder;
mod sw_decoder;

#[cfg(feature = "d3d11-renderer")]
pub use d3d11va_decoder::{D3d11Decoder, D3d11vaDecoderError};
#[cfg(feature = "d3d12-renderer")]
pub use d3d12va_decoder::{D3d12vaDecoder, D3d12vaDecoderError};
pub use sw_decoder::{SwDecoder, SwDecoderError};
