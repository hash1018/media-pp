//! Sources that take a texture another device, or another process, already
//! holds: D3D11 shared textures on Windows under [`windows`], and
//! `IOSurface`s through Metal on macOS under [`macos`].

#[cfg(all(target_os = "windows", feature = "d3d11"))]
mod windows;

#[cfg(all(target_os = "windows", feature = "d3d11"))]
pub use windows::*;

#[cfg(all(target_os = "macos", feature = "metal"))]
mod macos;

#[cfg(all(target_os = "macos", feature = "metal"))]
pub use macos::*;
