//! When things happen: the pipeline's clock and the playback clock beside
//! it, the rate a live source or compositor runs at, the schedule that
//! paces it, and the timestamp and time-base arithmetic under all of them.
//!
//! Grouped on disk only. Each module is re-exported at the crate root —
//! `crate::clock`, `crate::schedule` and the rest — exactly as it was
//! before the group existed; nothing refers to `crate::core::timing`.

pub mod clock;
pub mod playback_clock;
pub mod rate;
pub mod schedule;
pub mod time;
