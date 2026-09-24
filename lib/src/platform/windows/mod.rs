#[cfg(any(
    feature = "wasapi-capture",
    feature = "wasapi-renderer",
    feature = "mf-capture"
))]
pub(crate) mod com;
#[cfg(feature = "d3d11")]
pub(crate) mod d3d11;
#[cfg(feature = "d3d11")]
pub(crate) mod d3d11_full_frame;
#[cfg(feature = "d3d11")]
pub(crate) mod d3d11_gpu;
#[cfg(feature = "d3d11")]
pub(crate) mod d3d11va;
#[cfg(feature = "d3d12")]
pub(crate) mod d3d12_gpu;
#[cfg(feature = "d3d12")]
pub(crate) mod d3d12va;
#[cfg(any(feature = "d3d11", feature = "d3d12"))]
pub(crate) mod hlsl;
#[cfg(feature = "mf-capture")]
pub(crate) mod mf;
#[cfg(any(feature = "wasapi-capture", feature = "wasapi-renderer"))]
pub(crate) mod wasapi;
#[cfg(any(feature = "d3d11", feature = "d3d12"))]
pub(crate) mod window;
