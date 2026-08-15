#[cfg(feature = "d3d11-renderer")]
mod d3d11_upload;
#[cfg(feature = "d3d12-renderer")]
mod d3d12_upload;

#[cfg(feature = "d3d11-renderer")]
pub use d3d11_upload::{D3d11Upload, D3d11UploadError};
#[cfg(feature = "d3d12-renderer")]
pub use d3d12_upload::{D3d12Upload, D3d12UploadError};
