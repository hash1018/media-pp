//! Every element that downloads a GPU-resident texture back to a
//! CPU-resident frame — [`d3d11_download`], the mirror of
//! [`crate::elements::filter::upload`].

#[cfg(all(target_os = "windows", feature = "d3d11"))]
mod windows;

#[cfg(all(target_os = "windows", feature = "d3d11"))]
pub use windows::*;
