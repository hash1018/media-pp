//! Elements that change what a GPU-resident surface holds, as opposed to
//! where it lives ([`super::upload`], [`super::download`]) or how large it is
//! ([`super::scaler`]).

#[cfg(feature = "cuda")]
pub(crate) mod cuda;
#[cfg(feature = "vulkan")]
mod vulkan;

#[cfg(feature = "cuda")]
pub use cuda::{CudaConverter, CudaConverterError};
#[cfg(feature = "vulkan")]
pub use vulkan::{VulkanConverter, VulkanConverterError};
