//! Every element that resizes or converts video frames. The
//! backend-independent software scaler stays here, while the
//! CUDA-resident one lives under [`cuda`], the Vulkan-resident one under
//! `vulkan`, and the D3D11-resident one under `windows`.

#[cfg(feature = "cuda")]
pub(crate) mod cuda;
mod sw_scaler;
#[cfg(feature = "vulkan")]
mod vulkan;
#[cfg(all(target_os = "windows", any(feature = "d3d11", feature = "d3d12")))]
mod windows;

#[cfg(feature = "cuda")]
pub use cuda::{CudaScaler, CudaScalerError, CudaScalerInterp};
pub use sw_scaler::{SwScaler, SwScalerError};
#[cfg(feature = "vulkan")]
pub use vulkan::{VulkanScaler, VulkanScalerError, VulkanScalerInterp};
#[cfg(all(target_os = "windows", any(feature = "d3d11", feature = "d3d12")))]
pub use windows::*;
