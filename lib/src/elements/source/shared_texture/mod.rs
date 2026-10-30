//! Sources that take a picture another device or process already holds:
//! D3D11 shared textures on Windows, under [`windows`], `IOSurface`s
//! through Metal on macOS, under [`macos`], and DMA-BUFs imported into
//! CUDA on Linux, under [`linux`].

#[cfg(all(target_os = "windows", feature = "d3d11"))]
mod windows;

#[cfg(all(target_os = "windows", feature = "d3d11"))]
pub use windows::*;

#[cfg(all(target_os = "macos", feature = "metal"))]
mod macos;

#[cfg(all(target_os = "macos", feature = "metal"))]
pub use macos::*;

#[cfg(all(target_os = "linux", feature = "cuda"))]
mod linux;

#[cfg(all(target_os = "linux", feature = "cuda"))]
pub use linux::*;
