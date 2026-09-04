#[cfg(any(
    feature = "wasapi-capture",
    feature = "wasapi-renderer",
    feature = "mf-capture"
))]
pub(crate) mod com;
#[cfg(feature = "d3d11")]
pub(crate) mod d3d11;
#[cfg(feature = "d3d11")]
pub(crate) mod d3d11va;
#[cfg(feature = "d3d12")]
pub(crate) mod d3d12va;
#[cfg(feature = "mf-capture")]
pub(crate) mod mf;
#[cfg(any(feature = "wasapi-capture", feature = "wasapi-renderer"))]
pub(crate) mod wasapi;
