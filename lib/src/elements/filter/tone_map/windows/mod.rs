#[cfg(feature = "d3d11")]
mod d3d11_tone_map;
#[cfg(feature = "d3d12")]
mod d3d12_tone_map;

#[cfg(feature = "d3d11")]
pub use d3d11_tone_map::{D3d11ToneMap, D3d11ToneMapError};
#[cfg(feature = "d3d12")]
pub use d3d12_tone_map::{D3d12ToneMap, D3d12ToneMapError};
