//! Terminal elements that submit media to platform renderers. Windows
//! implementations live under [`windows`]. [`submit_error`]'s `SubmitError`
//! remains here because it is backend-independent and shared by D3D11/D3D12
//! (and potentially other GPU renderers). It is
//! `media-pp`'s own type — always available regardless of which, if
//! either, renderer feature is on).

mod submit_error;
#[cfg(all(
    target_os = "windows",
    any(feature = "d3d11", feature = "d3d12", feature = "wasapi-renderer")
))]
mod windows;

pub use submit_error::SubmitError;
#[cfg(all(
    target_os = "windows",
    any(feature = "d3d11", feature = "d3d12", feature = "wasapi-renderer")
))]
pub use windows::*;
