#[cfg(feature = "avfoundation-capture")]
mod avfoundation_capture_source;
#[cfg(feature = "coreaudio-capture")]
mod coreaudio_capture_source;

#[cfg(feature = "avfoundation-capture")]
pub use avfoundation_capture_source::{
    AvFoundationCaptureOptions, AvFoundationCaptureSource, AvFoundationCaptureSourceError,
};
#[cfg(feature = "coreaudio-capture")]
pub use coreaudio_capture_source::{
    CoreAudioCaptureOptions, CoreAudioCaptureSource, CoreAudioCaptureSourceError,
};
