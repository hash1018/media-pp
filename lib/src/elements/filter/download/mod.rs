//! Every element that downloads a GPU-resident texture back to a
//! CPU-resident frame — [`d3d11_download`], the mirror of
//! [`crate::elements::filter::upload`].

#[cfg(feature = "d3d11-renderer")]
mod d3d11_download;

#[cfg(feature = "d3d11-renderer")]
pub use d3d11_download::{D3d11Download, D3d11DownloadError};
