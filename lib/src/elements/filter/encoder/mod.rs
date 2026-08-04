//! Every element that turns `Video`/`Audio` frames into `Packet`s. Just
//! [`sw_encoder`] today (software encode via `libx264`/`libx265`/
//! `libopenh264`/`libkvazaar`/`libvpx`/`libaom-av1`/`libsvtav1`) — kept
//! as its own directory anyway, alongside
//! [`crate::elements::filter::decoder`]/[`crate::elements::filter::upload`],
//! so a hardware encoder added later has an obvious place to live rather
//! than prompting another reorg.

mod sw_encoder;

pub use sw_encoder::{SwEncoder, SwEncoderError, SwEncoderOptions, VideoCodec};
