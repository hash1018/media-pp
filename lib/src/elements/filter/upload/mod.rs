//! Every element that uploads CPU-resident frames to a GPU-resident texture.
//! The current D3D11/D3D12 implementations are Windows-specific and live
//! under [`windows`].

#[cfg(feature = "cuda")]
pub(crate) mod cuda;
#[cfg(any(
    feature = "cuda",
    feature = "vulkan",
    all(target_os = "windows", feature = "d3d11")
))]
mod nv12;
#[cfg(feature = "vulkan")]
mod vulkan;
#[cfg(all(target_os = "windows", any(feature = "d3d11", feature = "d3d12")))]
mod windows;

#[cfg(feature = "cuda")]
pub use cuda::{CudaUpload, CudaUploadError};
#[cfg(feature = "vulkan")]
pub use vulkan::{VulkanUpload, VulkanUploadError};

#[cfg(all(target_os = "windows", any(feature = "d3d11", feature = "d3d12")))]
pub use windows::*;
