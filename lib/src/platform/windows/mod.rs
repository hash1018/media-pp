#[cfg(feature = "d3d11")]
pub(crate) mod d3d11;
#[cfg(any(feature = "wasapi-capture", feature = "wasapi-renderer"))]
pub(crate) mod wasapi;
