use ffmpeg_next as ffmpeg;

/// The unit of data that flows between elements.
///
/// Compressed and uncompressed data are kept as distinct variants (rather
/// than a single opaque `Buffer` type like GStreamer) because ffmpeg-next
/// already gives us strongly-typed `Packet`/`Frame` types — collapsing them
/// into one type would just mean unwrapping again downstream.
pub enum MediaBuffer {
    Packet(ffmpeg::Packet),
    Video(ffmpeg::frame::Video),
    Audio(ffmpeg::frame::Audio),
    /// End of stream marker. Elements that hold resources (encoders with
    /// delayed frames, muxers, ...) should flush when they see this.
    Eos,
}

impl MediaBuffer {
    pub fn is_eos(&self) -> bool {
        matches!(self, MediaBuffer::Eos)
    }
}
