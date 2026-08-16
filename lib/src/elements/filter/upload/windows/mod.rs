#[cfg(feature = "d3d11")]
mod d3d11_upload;
#[cfg(feature = "d3d12")]
mod d3d12_upload;

#[cfg(feature = "d3d11")]
pub use d3d11_upload::{D3d11Upload, D3d11UploadError};
#[cfg(feature = "d3d12")]
pub use d3d12_upload::{D3d12Upload, D3d12UploadError};
