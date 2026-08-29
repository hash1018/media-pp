//! Every element that turns `Video`/`Audio` frames into `Packet`s, split
//! by media kind: [`video`] (software video encode via `libx264`/
//! `libx265`/`libopenh264`/`libkvazaar`/`libvpx`/`libaom-av1`/
//! `libsvtav1`) and [`audio`] (software audio encode via `aac`) — kept as
//! their own directory anyway, alongside
//! [`crate::elements::filter::decoder`]/[`crate::elements::filter::upload`],
//! so a hardware encoder added later has an obvious place to live rather
//! than prompting another reorg. [`video`] is where that landed:
//! `D3d11VideoEncoder` sits beside `SwEncoder` under its own `windows`
//! submodule, mirroring how the decoder directory is laid out.

mod audio;
mod video;

pub use audio::{AudioCodec, SwAudioEncoder, SwAudioEncoderError, SwAudioEncoderOptions};
#[cfg(feature = "cuda")]
pub use video::{CudaCodec, CudaEncoder, CudaEncoderError, CudaEncoderOptions};
#[cfg(all(target_os = "windows", feature = "d3d11"))]
pub use video::{
    D3d11VideoCodec, D3d11VideoEncoder, D3d11VideoEncoderError, D3d11VideoEncoderOptions,
    D3d11VideoInputFormat,
};
pub use video::{SwEncoder, SwEncoderError, SwEncoderOptions, VideoCodec};
