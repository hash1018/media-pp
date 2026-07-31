pub mod buffer;
pub mod bus;
pub mod clock;
pub mod control;
pub mod element;
pub mod elements;
pub mod error;
pub mod pad;
pub mod pipeline;
pub mod queue;

pub use error::{Error, Result};

/// Must be called once before using any element that touches ffmpeg.
pub fn init() -> Result<()> {
    ffmpeg_next::init()?;
    Ok(())
}
