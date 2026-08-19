//! The CUDA device context this crate's CUDA elements share. Unlike
//! [`crate::platform::windows`]/[`crate::platform::linux`], this module is
//! gated on its Cargo feature alone: CUDA is not a platform, and the same
//! elements build on Linux and Windows.

pub(crate) mod device;
pub(crate) mod format;

pub use device::{CudaDevice, CudaDeviceError};
pub use format::CudaFrameFormat;
