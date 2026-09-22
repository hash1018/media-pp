//! HDR video brought into SDR on a device — see `core/tone_map.rs` for
//! the definition every backend evaluates. `CudaConverter` does it on CUDA.

mod windows;

pub use windows::{D3d11ToneMap, D3d11ToneMapError};
