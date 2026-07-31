use std::sync::Arc;

use ffmpeg_next as ffmpeg;

use crate::pool::UnboundObjectPoolRef;

/// The unit of data that flows between elements.
///
/// Compressed and uncompressed data are kept as distinct variants (rather
/// than a single opaque `Buffer` type like GStreamer) because ffmpeg-next
/// already gives us strongly-typed `Packet`/`Frame` types — collapsing them
/// into one type would just mean unwrapping again downstream.
///
/// Payloads are `Arc`-wrapped so `MediaBuffer` is cheaply `Clone` —
/// duplicating a buffer (e.g. [`crate::elements::Tee`] fanning packets out
/// to a decode branch and a remux branch) is a refcount bump, never a copy
/// of the encoded/decoded data.
///
/// `Video` specifically wraps an [`UnboundObjectPoolRef`], not a plain
/// `ffmpeg::frame::Video` — that's what lets whichever element produced it
/// (see [`crate::pool::UnboundObjectPool`], owned as that element's own
/// struct field) get the underlying buffer back automatically once every
/// `Arc` clone downstream has been dropped, instead of it just being freed.
#[derive(Clone)]
pub enum MediaBuffer {
    Packet(Arc<ffmpeg::Packet>),
    Video(Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>),
    Audio(Arc<ffmpeg::frame::Audio>),
    /// End of stream marker. Elements that hold resources (encoders with
    /// delayed frames, muxers, ...) should flush when they see this.
    Eos,
}

impl MediaBuffer {
    pub fn is_eos(&self) -> bool {
        matches!(self, MediaBuffer::Eos)
    }
}
