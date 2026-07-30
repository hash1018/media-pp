pub mod buffer;
pub mod bus;
pub mod clock;
pub mod element;
pub mod elements;
pub mod error;
pub mod pad;
pub mod pipeline;
pub mod queue;

pub use error::{Error, Result};

/// Must be called once before using any element that touches ffmpeg. Not
/// tied to any one element, so failures just get wrapped in `Other`
/// rather than chaining through an `{Element}Error`.
pub fn init() -> Result<()> {
    ffmpeg_next::init().map_err(|e| Error::Other(e.to_string()))
}
