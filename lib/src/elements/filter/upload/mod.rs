//! Every element that uploads CPU-resident frames to a GPU-resident texture.
//! The current D3D11/D3D12 implementations are Windows-specific and live
//! under [`windows`].

#[cfg(feature = "cuda")]
pub(crate) mod cuda;
#[cfg(all(target_os = "windows", any(feature = "d3d11", feature = "d3d12")))]
mod windows;

#[cfg(feature = "cuda")]
pub(crate) use cuda::cuda_upload;
#[cfg(feature = "cuda")]
pub use cuda::{CudaUpload, CudaUploadError};

#[cfg(all(target_os = "windows", any(feature = "d3d11", feature = "d3d12")))]
pub use windows::*;
