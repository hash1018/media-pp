//! What a pipeline says about itself: the private file logger, the
//! identity every element's records carry, and the counters behind
//! `Pipeline::stats`.
//!
//! Grouped on disk only. Each module is re-exported at the crate root —
//! `crate::log`, `crate::pp_log`, `crate::stats` — exactly as it was before
//! the group existed; nothing refers to `crate::core::diagnostics`.

pub mod log;
pub mod pp_log;
pub mod stats;
