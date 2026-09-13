//! Sources that make their own media: a test pattern and a test tone.
//!
//! Called `synthetic` rather than after the types it holds, because a
//! directory called `test` reads as test code, and these are public elements
//! a pipeline runs — the fixtures `test_support` builds are made with them.

mod audio;
mod video;

pub use audio::{TestAudioOptions, TestAudioSource, TestAudioSourceError};
pub use video::{TestVideoOptions, TestVideoSource, TestVideoSourceError};
