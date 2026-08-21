#[cfg(feature = "d3d11")]
mod d3d11_download;
#[cfg(feature = "d3d12")]
mod d3d12_download;

#[cfg(feature = "d3d11")]
pub use d3d11_download::{D3d11Download, D3d11DownloadError};
#[cfg(feature = "d3d12")]
pub use d3d12_download::{D3d12Download, D3d12DownloadError};
