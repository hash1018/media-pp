//! Terminal elements that submit media to platform renderers. Windows
//! implementations live under [`windows`], Linux ones under [`linux`]. [`submit_error`]'s `SubmitError`
//! remains here because it is backend-independent and shared by D3D11/D3D12
//! (and potentially other GPU renderers). It is
//! `media-pp`'s own type — always available regardless of which, if
//! either, renderer feature is on).

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(all(target_os = "linux", feature = "pipewire-audio-renderer"))]
mod linux;
mod submit_error;
#[cfg(all(
    target_os = "windows",
    any(feature = "d3d11", feature = "d3d12", feature = "wasapi-renderer")
))]
mod windows;

#[cfg(feature = "cuda")]
pub use cuda::*;
#[cfg(all(target_os = "linux", feature = "pipewire-audio-renderer"))]
pub use linux::*;
pub use submit_error::SubmitError;
#[cfg(all(
    target_os = "windows",
    any(feature = "d3d11", feature = "d3d12", feature = "wasapi-renderer")
))]
pub use windows::*;
