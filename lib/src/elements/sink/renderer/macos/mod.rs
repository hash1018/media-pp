#[cfg(feature = "coreaudio-renderer")]
mod coreaudio_renderer;
#[cfg(feature = "coreaudio-renderer")]
mod sample_ring;

#[cfg(feature = "coreaudio-renderer")]
pub use coreaudio_renderer::{CoreAudioRenderer, CoreAudioRendererError, CoreAudioRendererOptions};
