//! What a program would otherwise build out of elements itself: a whole
//! pipeline behind one type, for the common case — [`player`], a file played
//! in a window with its sound.
//!
//! A layer above [`crate::elements`], not part of it: nothing here goes into
//! a pipeline, each one *is* the pipeline, built and driven for its caller.
//! It is built on the public elements and pipeline, and a program that
//! outgrows one builds the same graph itself. Where it reaches for something
//! crate-private, that item says why.
//!
//! Private for the same reason as `core`: every module here is re-exported at
//! the crate root, so callers write `media_pp::player`.

#[cfg(any(
    all(
        target_os = "windows",
        any(feature = "d3d11", feature = "d3d12"),
        feature = "wasapi-renderer"
    ),
    all(
        target_os = "linux",
        feature = "vulkan",
        feature = "pipewire-audio-renderer"
    )
))]
pub mod player;
