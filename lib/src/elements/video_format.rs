use ffmpeg_next as ffmpeg;

/// A complete uncompressed-video geometry/timing description.
///
/// Unlike a `(width, height, time_base)` tuple, this can be passed directly
/// from a capture source such as [`crate::elements::DxgiCaptureSource`] to a
/// downstream [`crate::elements::SwScaler`]/[`crate::elements::SwEncoder`]/
/// [`crate::elements::Mp4Muxer`] without the caller re-threading three
/// separate values by hand — same role [`crate::elements::AudioFormat`]
/// plays for uncompressed audio. Kept directly under `elements` rather than
/// under any one of those (same reasoning as [`crate::elements::RtspTransport`]'s
/// own placement): it crosses source/filter/sink, not owned by any single one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoFormat {
    pub width: u32,
    pub height: u32,
    pub time_base: ffmpeg::Rational,
}
