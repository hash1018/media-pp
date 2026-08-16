mod sw_encoder;
#[cfg(all(target_os = "windows", feature = "d3d11"))]
mod windows;

pub use sw_encoder::{SwEncoder, SwEncoderError, SwEncoderOptions, VideoCodec};
#[cfg(all(target_os = "windows", feature = "d3d11"))]
pub use windows::*;
