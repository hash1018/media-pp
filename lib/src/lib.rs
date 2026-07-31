mod core;
pub mod elements;
pub mod error;

// Flat re-export: `core/` only exists to group these files on disk (see
// its module doc) — every external and internal caller keeps using
// `crate::pipeline`/`media_pp::pipeline` etc., never `crate::core::...`.
pub use core::{buffer, bus, clock, control, element, pad, pipeline, pool, queue};

pub use error::{Error, Result};

/// Must be called once before using any element that touches ffmpeg.
pub fn init() -> Result<()> {
    ffmpeg_next::init()?;
    Ok(())
}
