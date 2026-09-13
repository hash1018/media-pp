//! Filters on decoded audio: resampling, gain, gating, compression,
//! limiting and noise suppression.
//!
//! Grouped by what they take rather than by what they do, and only the
//! filters — the audio encoder stays with the other encoders, the mixer and
//! the test tone with the other sources. What the dynamics filters share is
//! here too and nowhere else: `audio_f32` reads and writes a frame's
//! samples a channel at a time, and `tuning` is how a handle's new settings
//! reach an element that is reading them.

mod audio_compressor;
mod audio_f32;
mod audio_gate;
mod audio_limiter;
pub(crate) mod audio_resampler;
mod audio_volume;
#[cfg(feature = "rnnoise")]
mod noise_suppressor;
mod tuning;

pub use audio_compressor::{
    AudioCompressor, AudioCompressorError, AudioCompressorHandle, AudioCompressorOptions,
};
pub use audio_gate::{AudioGate, AudioGateError, AudioGateHandle, AudioGateOptions};
pub use audio_limiter::{AudioLimiter, AudioLimiterError, AudioLimiterHandle, AudioLimiterOptions};
pub use audio_resampler::{AudioResampler, AudioResamplerError};
pub use audio_volume::{AudioVolume, AudioVolumeError, AudioVolumeHandle, AudioVolumeOptions};
#[cfg(feature = "rnnoise")]
pub use noise_suppressor::{NOISE_SUPPRESSOR_SAMPLE_RATE, NoiseSuppressor, NoiseSuppressorError};
