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

    /// Stable, human-readable variant name for diagnostics emitted when
    /// elements are wired to an incompatible media type.
    pub fn kind(&self) -> &'static str {
        match self {
            MediaBuffer::Packet(_) => "Packet",
            MediaBuffer::Video(_) => "Video",
            MediaBuffer::Audio(_) => "Audio",
            MediaBuffer::Eos => "Eos",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_reports_each_variant() {
        assert_eq!(
            MediaBuffer::Packet(Arc::new(ffmpeg::Packet::empty())).kind(),
            "Packet"
        );
        assert_eq!(
            MediaBuffer::Audio(Arc::new(ffmpeg::frame::Audio::empty())).kind(),
            "Audio"
        );
        assert_eq!(MediaBuffer::Eos.kind(), "Eos");
    }
}
