#[cfg(feature = "d3d11")]
mod d3d11_video_encoder;
#[cfg(feature = "d3d12")]
mod d3d12_video_encoder;

#[cfg(feature = "d3d11")]
pub use d3d11_video_encoder::{
    D3d11VideoCodec, D3d11VideoEncoder, D3d11VideoEncoderError, D3d11VideoEncoderOptions,
    D3d11VideoInputFormat,
};
#[cfg(feature = "d3d12")]
pub use d3d12_video_encoder::{
    D3d12VideoCodec, D3d12VideoEncoder, D3d12VideoEncoderError, D3d12VideoEncoderOptions,
};
