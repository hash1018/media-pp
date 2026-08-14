//! The pipeline framework itself — the graph primitives (`element`,
//! `pad`, `buffer`) and everything that drives them at runtime
//! (`pipeline`, `queue`, `control`, `clock`, `bus`, `pool`), plus small
//! shared value types with no pipeline behavior of their own (`color`,
//! `schedule`).
//! Kept as its own module purely to group these files on disk; this module itself is
//! private (see `lib.rs`) and every item here is re-exported flat at the
//! crate root, so nothing outside this file (internal or external to the
//! crate) refers to `crate::core::...` directly — `crate::pipeline`,
//! `media_pp::pipeline`, etc. keep working exactly as before.
//!
//! [`crate::elements`] (the built-in `Sink`/`Source`/`Filter`
//! implementations, e.g. `FileDemuxer`/`SwDecoder`/`RtspServer`) is
//! deliberately its own top-level module, not part of this one — those
//! are built *on* this framework, not part of it.

pub mod buffer;
pub mod bus;
pub mod clock;
pub mod color;
pub mod control;
pub mod driver;
pub mod element;
pub mod graph;
pub mod log;
pub mod pad;
pub mod pipeline;
pub mod pool;
pub mod queue;
pub mod schedule;
