//! HDR video brought into SDR on a device — see `core/tone_map.rs` for
//! the definition every backend evaluates. `CudaConverter` does it on CUDA.

mod windows;

#[cfg(feature = "d3d11")]
pub use windows::{D3d11ToneMap, D3d11ToneMapError};
#[cfg(feature = "d3d12")]
pub use windows::{D3d12ToneMap, D3d12ToneMapError};
