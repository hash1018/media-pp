//! Every element that keys a solid background color out of a video frame
//! into alpha. [`ChromaKeyMethod`]/[`ChromaKeyOptions`] are backend-
//! independent and shared; the CPU implementation lives in
//! [`sw_chroma_key`] and the D3D11-resident one under `windows`, mirroring
//! [`super::scaler`]'s layout.

mod options;
mod sw_chroma_key;
#[cfg(all(target_os = "windows", feature = "d3d11"))]
mod windows;

pub use options::{ChromaKeyMethod, ChromaKeyOptions};
pub use sw_chroma_key::{SwChromaKey, SwChromaKeyError};
#[cfg(all(target_os = "windows", feature = "d3d11"))]
pub use windows::{D3d11ChromaKey, D3d11ChromaKeyError};
