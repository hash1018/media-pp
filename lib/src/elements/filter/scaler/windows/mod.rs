#[cfg(feature = "d3d11")]
mod d3d11_scaler;
#[cfg(feature = "d3d11")]
mod video_processor;

#[cfg(feature = "d3d11")]
pub use d3d11_scaler::{D3d11Scaler, D3d11ScalerError};
