use ffmpeg_next as ffmpeg;

/// A complete uncompressed-audio format description.
///
/// Unlike a `(sample_rate, channels)` tuple, this also carries the sample
/// representation, so it can be passed directly from a hardware endpoint
/// such as `WasapiCaptureSource`/`WasapiRenderer` to an
/// [`crate::elements::AudioResampler`]/[`crate::elements::SwAudioEncoder`]
/// without guessing either one. Kept directly under `elements` rather than
/// under any one of those (same reasoning as
/// [`crate::elements::RtspTransport`]/[`crate::elements::VideoFormat`]'s own
/// placement): it crosses source/filter/sink, not owned by any single one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioFormat {
    pub sample_format: ffmpeg::format::Sample,
    pub sample_rate: u32,
    pub channels: u16,
}

impl AudioFormat {
    pub fn new(sample_format: ffmpeg::format::Sample, sample_rate: u32, channels: u16) -> Self {
        Self {
            sample_format,
            sample_rate,
            channels,
        }
    }

    pub fn channels(self) -> u16 {
        self.channels
    }

    pub fn channel_layout(self) -> ffmpeg::ChannelLayout {
        ffmpeg::ChannelLayout::default(i32::from(self.channels))
    }
}
