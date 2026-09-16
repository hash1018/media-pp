//! Sources that take a texture another device already holds. The current
//! D3D11 implementation is Windows-specific and lives under [`windows`].

#[cfg(all(target_os = "windows", feature = "d3d11"))]
mod windows;

#[cfg(all(target_os = "windows", feature = "d3d11"))]
pub use windows::*;
