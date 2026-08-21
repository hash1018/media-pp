//! Every element that turns `Packet`s into `Video`/`Audio` frames. The
//! backend-independent software decoder stays here, while Windows-specific
//! D3D11VA/D3D12VA implementations live under [`windows`].
//!
//! The D3D11VA module is re-exported `pub(crate)`, not private: its small
//! `pub(crate)` helpers
//! ([`d3d11va_decoder::wrap_d3d11_texture`]/
//! [`d3d11va_decoder::create_hw_device_ctx`])
//! are reused outside this module — by D3D11 upload, download, scaling,
//! rendering, and encoding elements, and by
//! [`crate::elements::DxgiCaptureSource`]'s GPU capture mode
//! (which shares the D3D11 frame representation without itself decoding
//! anything).

#[cfg(feature = "cuda")]
mod cuda;
mod sw_decoder;
#[cfg(all(target_os = "windows", any(feature = "d3d11", feature = "d3d12")))]
mod windows;

#[cfg(feature = "cuda")]
pub use cuda::{CudaDecoder, CudaDecoderError};
pub use sw_decoder::{SwDecoder, SwDecoderError};
#[cfg(all(target_os = "windows", feature = "d3d11"))]
pub(crate) use windows::d3d11va_decoder;
#[cfg(all(target_os = "windows", any(feature = "d3d11", feature = "d3d12")))]
pub use windows::*;
