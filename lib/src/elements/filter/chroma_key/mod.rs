//! Every element that keys a solid background color out of a video frame
//! into alpha. The backend-independent software version lives here; a
//! GPU-resident one (D3D11/CUDA) is planned to join it under `windows`/
//! `cuda`, mirroring [`super::scaler`]'s layout.

mod sw_chroma_key;

pub use sw_chroma_key::{ChromaKeyMethod, ChromaKeyOptions, SwChromaKey, SwChromaKeyError};
