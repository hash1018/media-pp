#[cfg(feature = "coreaudio-capture")]
mod coreaudio_capture_source;

#[cfg(feature = "coreaudio-capture")]
pub use coreaudio_capture_source::{
    CoreAudioCaptureOptions, CoreAudioCaptureSource, CoreAudioCaptureSourceError,
};
