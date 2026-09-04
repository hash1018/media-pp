#[cfg(feature = "pipewire-audio-capture")]
mod pipewire_audio_capture_source;
#[cfg(feature = "pipewire-screen-capture")]
mod pipewire_screen_capture_source;
#[cfg(feature = "v4l2-capture")]
mod v4l2_capture_source;

#[cfg(feature = "pipewire-audio-capture")]
pub use pipewire_audio_capture_source::{
    PipeWireAudioCaptureOptions, PipeWireAudioCaptureSource, PipeWireAudioCaptureSourceError,
};
#[cfg(feature = "pipewire-screen-capture")]
pub use pipewire_screen_capture_source::{
    CaptureSourceKind, PipeWireScreenCaptureOptions, PipeWireScreenCaptureSource,
    PipeWireScreenCaptureSourceError,
};

#[cfg(feature = "v4l2-capture")]
pub use v4l2_capture_source::{V4l2CaptureOptions, V4l2CaptureSource, V4l2CaptureSourceError};
