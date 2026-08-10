//! Every element that turns `Video`/`Audio` frames into `Packet`s, split
//! by media kind: [`video`] (software video encode via `libx264`/
//! `libx265`/`libopenh264`/`libkvazaar`/`libvpx`/`libaom-av1`/
//! `libsvtav1`) and [`audio`] (software audio encode via `aac`) — kept as
//! their own directory anyway, alongside
//! [`crate::elements::filter::decoder`]/[`crate::elements::filter::upload`],
//! so a hardware encoder added later has an obvious place to live rather
//! than prompting another reorg.

mod audio;
mod video;

pub use audio::{AudioCodec, SwAudioEncoder, SwAudioEncoderError, SwAudioEncoderOptions};
pub use video::{SwEncoder, SwEncoderError, SwEncoderOptions, VideoCodec};
