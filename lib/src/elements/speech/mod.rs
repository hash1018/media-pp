//! Speech: hearing what is said in sound, and doing something with it.
//!
//! What every element here shares is [`Transcript`], the
//! [`Metadata`](crate::buffer::Metadata) a transcriber puts on the sound it
//! hands on and every later element reads, as a detector puts
//! [`Detections`](crate::elements::Detections) on a picture. A typical chain
//! is `decoder -> AudioResampler -> queue -> transcriber ->
//! SubtitleEncoder -> muxer`, each element handing on what it was given and
//! the next reading what the one before it heard.
//!
//! - `transcribe` — speech to timed text through whisper.cpp, with the
//!   `whisper` features: `SwWhisperTranscriber` on the CPU, and
//!   `CudaWhisperTranscriber`, `VulkanWhisperTranscriber` and
//!   `MetalWhisperTranscriber` running the same model on a GPU. Each holds
//!   the sound until the words in it are settled, and hands it on carrying
//!   the lines that begin in it.
//! - [`SubtitleEncoder`] — the lines as subtitle packets, for an MP4's text
//!   track or an `.srt` or `.vtt` file of their own.
//!
//! The data and the subtitle encoder need no feature of their own; only
//! transcribing brings a runtime with it.

mod meta;
mod subtitle_encoder;
#[cfg(feature = "whisper")]
mod transcribe;

pub use meta::{Transcript, Utterance, Word};
pub use subtitle_encoder::{SubtitleEncoder, SubtitleEncoderError};
#[cfg(feature = "whisper-cuda")]
pub use transcribe::CudaWhisperTranscriber;
#[cfg(all(target_os = "macos", feature = "whisper-metal"))]
pub use transcribe::MetalWhisperTranscriber;
#[cfg(feature = "whisper-vulkan")]
pub use transcribe::VulkanWhisperTranscriber;
#[cfg(feature = "whisper")]
pub use transcribe::{
    ChunkPolicy, SAMPLE_RATE as WHISPER_SAMPLE_RATE, SwWhisperTranscriber, TokenTiming,
    WhisperOptions, WhisperTranscriberError,
};
