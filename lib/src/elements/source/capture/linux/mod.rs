#[cfg(feature = "pipewire-audio-capture")]
mod pipewire_audio_capture_source;
#[cfg(feature = "pipewire-screen-capture")]
mod pipewire_screen_capture_source;

#[cfg(feature = "pipewire-audio-capture")]
pub use pipewire_audio_capture_source::{
    PipeWireAudioCaptureOptions, PipeWireAudioCaptureSource, PipeWireAudioCaptureSourceError,
    PipeWireAudioDevice, PipeWireAudioDeviceKind,
};
#[cfg(feature = "pipewire-screen-capture")]
pub use pipewire_screen_capture_source::{
    CaptureSourceKind, PipeWireScreenCaptureOptions, PipeWireScreenCaptureSource,
    PipeWireScreenCaptureSourceError,
};
