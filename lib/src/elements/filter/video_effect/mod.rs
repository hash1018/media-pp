//! Per-pixel effects on a BGRA picture: a colour correction and a luma key.
//!
//! One element per backend rather than one per effect. Every effect here
//! resolves to the same small block of numbers — a colour matrix, an
//! exponent, an opacity and a luma mask, see `options` — so one D3D11
//! shader, one CUDA kernel and one software loop run them all, and an effect
//! is added once, in `options`, rather than three times. [`VideoEffect`]
//! names which one an element runs, and its [`VideoEffectHandle`] can switch
//! it while it runs.
//!
//! The chroma key predates this and stays its own family: keying against a
//! colour is a distance, which the block above cannot express.
//!
//! The layout mirrors [`super::chroma_key`]: shared settings and handle
//! here, the CPU implementation in [`sw_video_effect`], the D3D11 one under
//! `windows`, the CUDA one under `cuda`.

#[cfg(feature = "cuda")]
mod cuda;
mod handle;
mod options;
mod sw_video_effect;
#[cfg(all(target_os = "windows", feature = "d3d11"))]
mod windows;

#[cfg(feature = "cuda")]
pub use cuda::{CudaVideoEffect, CudaVideoEffectError};
pub use handle::VideoEffectHandle;
pub use options::{ColorCorrection, LumaKey, VideoEffect};
pub use sw_video_effect::{SwVideoEffect, SwVideoEffectError};
#[cfg(all(target_os = "windows", feature = "d3d11"))]
pub use windows::{D3d11VideoEffect, D3d11VideoEffectError};
