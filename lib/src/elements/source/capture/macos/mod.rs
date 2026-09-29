#[cfg(feature = "avfoundation-capture")]
mod avfoundation_capture_source;
#[cfg(feature = "coreaudio-capture")]
mod coreaudio_capture_source;
#[cfg(feature = "screencapturekit-capture")]
mod screencapturekit_source;

#[cfg(feature = "avfoundation-capture")]
pub use avfoundation_capture_source::{
    AvFoundationCaptureOptions, AvFoundationCaptureSource, AvFoundationCaptureSourceError,
};
#[cfg(feature = "coreaudio-capture")]
pub use coreaudio_capture_source::{
    CoreAudioCaptureOptions, CoreAudioCaptureSource, CoreAudioCaptureSourceError,
};
#[cfg(feature = "screencapturekit-capture")]
pub use screencapturekit_source::{
    ScreenCaptureKitOptions, ScreenCaptureKitSource, ScreenCaptureKitSourceError,
    ScreenCaptureKitTarget,
};
