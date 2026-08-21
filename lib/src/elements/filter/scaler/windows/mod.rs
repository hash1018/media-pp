#[cfg(feature = "d3d11")]
mod d3d11_scaler;
#[cfg(feature = "d3d11")]
mod d3d11_video_processor;
#[cfg(feature = "d3d12")]
mod d3d12_scaler;
#[cfg(feature = "d3d12")]
mod d3d12_video_processor;

#[cfg(feature = "d3d11")]
pub use d3d11_scaler::{D3d11Scaler, D3d11ScalerError, D3d11ScalerFormat};
#[cfg(feature = "d3d12")]
pub use d3d12_scaler::{D3d12Scaler, D3d12ScalerError};
