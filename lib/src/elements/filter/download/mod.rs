//! Every element that downloads a GPU-resident frame back to a CPU-resident
//! one — `CudaDownload` under [`cuda`] and the D3D11/D3D12 downloads under `windows`,
//! the mirror of [`crate::elements::filter::upload`].

#[cfg(feature = "cuda")]
pub(crate) mod cuda;
#[cfg(all(target_os = "windows", any(feature = "d3d11", feature = "d3d12")))]
mod windows;

#[cfg(feature = "cuda")]
pub use cuda::{CudaDownload, CudaDownloadError};

#[cfg(all(target_os = "windows", any(feature = "d3d11", feature = "d3d12")))]
pub use windows::*;
