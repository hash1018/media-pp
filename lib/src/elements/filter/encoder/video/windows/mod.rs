#[cfg(feature = "d3d11")]
mod d3d11_nvenc_encoder;

#[cfg(feature = "d3d11")]
pub use d3d11_nvenc_encoder::{
    D3d11NvencCodec, D3d11NvencEncoder, D3d11NvencEncoderError, D3d11NvencEncoderOptions,
    D3d11NvencInputFormat,
};
