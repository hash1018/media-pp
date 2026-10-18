//! Speech to timed text: transcribers, grouped by the runtime that runs
//! their model.
//!
//! - `whisper` — whisper.cpp, with the `whisper` features:
//!   `SwWhisperTranscriber` on the CPU, and `CudaWhisperTranscriber`,
//!   `VulkanWhisperTranscriber` and `MetalWhisperTranscriber` on a GPU.

mod whisper;

#[cfg(feature = "whisper-cuda")]
pub use whisper::CudaWhisperTranscriber;
#[cfg(all(target_os = "macos", feature = "whisper-metal"))]
pub use whisper::MetalWhisperTranscriber;
#[cfg(feature = "whisper-vulkan")]
pub use whisper::VulkanWhisperTranscriber;
pub use whisper::{
    ChunkPolicy, SAMPLE_RATE, SwWhisperTranscriber, TokenTiming, WhisperOptions,
    WhisperTranscriberError,
};
