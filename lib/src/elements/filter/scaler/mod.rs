//! Every element that resizes or converts video frames. The
//! backend-independent software scaler stays here, while the
//! CUDA-resident one lives under [`cuda`] and the D3D11-resident one under
//! `windows`.

#[cfg(feature = "cuda")]
pub(crate) mod cuda;
mod sw_scaler;
#[cfg(all(target_os = "windows", any(feature = "d3d11", feature = "d3d12")))]
mod windows;

#[cfg(feature = "cuda")]
pub use cuda::{CudaScaler, CudaScalerError, CudaScalerInterp};
pub use sw_scaler::{SwScaler, SwScalerError};
#[cfg(all(target_os = "windows", any(feature = "d3d11", feature = "d3d12")))]
pub use windows::*;
