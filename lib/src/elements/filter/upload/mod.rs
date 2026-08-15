//! Every element that uploads CPU-resident frames to a GPU-resident texture.
//! The current D3D11/D3D12 implementations are Windows-specific and live
//! under [`windows`].

#[cfg(all(
    target_os = "windows",
    any(feature = "d3d11-renderer", feature = "d3d12-renderer")
))]
mod windows;

#[cfg(all(
    target_os = "windows",
    any(feature = "d3d11-renderer", feature = "d3d12-renderer")
))]
pub use windows::*;
