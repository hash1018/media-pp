//! Every element that downloads a GPU-resident frame back to a CPU-resident
//! one — `CudaDownload` under [`cuda`] and `D3d11Download` under `windows`,
//! the mirror of [`crate::elements::filter::upload`].

#[cfg(feature = "cuda")]
pub(crate) mod cuda;
#[cfg(all(target_os = "windows", feature = "d3d11"))]
mod windows;

#[cfg(feature = "cuda")]
pub use cuda::{CudaDownload, CudaDownloadError};

#[cfg(all(target_os = "windows", feature = "d3d11"))]
pub use windows::*;
