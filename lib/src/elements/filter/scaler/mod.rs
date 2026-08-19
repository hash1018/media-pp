//! Every element that resizes or converts video frames. The
//! backend-independent software scaler stays here, while the
//! CUDA-resident one lives under [`cuda`].

#[cfg(feature = "cuda")]
mod cuda;
mod sw_scaler;

#[cfg(feature = "cuda")]
pub use cuda::{CudaScaler, CudaScalerError, CudaScalerInterp};
pub use sw_scaler::{SwScaler, SwScalerError};
