//! Every element that turns `Packet`s into `Video`/`Audio` frames. The
//! backend-independent software decoder stays here, while Windows-specific
//! D3D11VA/D3D12VA implementations live under [`windows`]. Shared hardware-frame
//! ABI helpers live under `crate::platform::windows`.

#[cfg(feature = "cuda")]
mod cuda;
mod sw_decoder;
#[cfg(all(target_os = "windows", any(feature = "d3d11", feature = "d3d12")))]
mod windows;

#[cfg(feature = "cuda")]
pub use cuda::{CudaDecoder, CudaDecoderError};
pub use sw_decoder::{SwDecoder, SwDecoderError};
#[cfg(all(target_os = "windows", any(feature = "d3d11", feature = "d3d12")))]
pub use windows::*;
