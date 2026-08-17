#[cfg(feature = "pipewire-capture")]
mod pipewire_capture_source;

#[cfg(feature = "pipewire-capture")]
pub use pipewire_capture_source::{
    CaptureSourceKind, PipeWireCaptureOptions, PipeWireCaptureSource, PipeWireCaptureSourceError,
};
