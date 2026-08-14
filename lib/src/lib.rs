mod core;
pub mod elements;
pub mod error;
mod platform;

// Flat re-export: `core/` only exists to group these files on disk (see
// its module doc) — every external and internal caller keeps using
// `crate::pipeline`/`media_pp::pipeline` etc., never `crate::core::...`.
pub use core::{
    buffer, bus, clock, color, control, driver, element, graph, log, pad, pipeline, pool, queue,
};

// Same flat-namespace reasoning as above, but crate-private: `schedule`/
// `time` are pacing/rescale internals `crate::elements` builds on, not
// exposed in any public element's own field/method signature — nothing
// downstream of this crate needs `PeriodicSchedule`/`ActiveTimeline`/
// `MediaTimestamp`/`TimeBase` itself. `pub(crate) use` keeps the same
// `crate::schedule`/`crate::time` paths working for every internal caller
// without also making them part of this crate's external API surface.
pub(crate) use core::{schedule, time};

pub use error::{Error, Result};

/// Must be called once before using any element that touches ffmpeg.
pub fn init() -> Result<()> {
    ffmpeg_next::init()?;
    Ok(())
}
