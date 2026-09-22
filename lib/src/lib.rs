//! A small, GStreamer-flavored media pipeline library built on
//! [`ffmpeg-next`](https://docs.rs/ffmpeg-next).
//!
//! A pipeline has one or more [`SourceElement`](element::SourceElement)s,
//! each feeding a graph of [`Filter`](element::Filter)s that ends in a
//! [`Sink`](element::Sink):
//!
//! ```text
//! FileDemuxer -> SwDecoder -> Queue -> Pacer -> FrameCounter
//! ```
//!
//! Each source registered with a [`Pipeline`](pipeline::Pipeline) runs on its
//! own background thread. Within that source's graph,
//! [`Sink::consume`](element::Sink::consume) is otherwise a plain synchronous
//! call that returns [`Result`], so a stage's failure propagates straight back
//! up the call stack with `?`. A [`Queue`](queue::Queue) adds another explicit
//! thread boundary inside a branch: it owns a worker thread and a bounded
//! channel, which is also where error handling changes shape — past that
//! boundary a downstream failure can no longer be returned to the pusher, so
//! it is reported on the [`Bus`](bus::Bus) as
//! [`BusEvent::Error`](bus::BusEvent::Error) and the worker keeps going.
//!
//! ```no_run
//! use std::{sync::atomic::Ordering, time::Duration};
//!
//! use media_pp::{
//!     elements::{FrameCounter, TestVideoOptions, TestVideoSource},
//!     pipeline::Pipeline,
//! };
//!
//! # fn main() -> media_pp::Result<()> {
//! media_pp::init()?;
//!
//! let source = TestVideoSource::new("source", TestVideoOptions::default());
//! let (counter, frames) = FrameCounter::new("counter");
//!
//! let pipeline = Pipeline::new("demo", source, |source, ctx| {
//!     let branch = ctx.branch().to(counter)?;
//!     ctx.attach(source, 0, branch)?;
//!     Ok(())
//! })?;
//!
//! pipeline.run()?;
//! std::thread::sleep(Duration::from_millis(200));
//! pipeline.stop();
//!
//! println!("frames: {}", frames.load(Ordering::Relaxed));
//! # Ok(())
//! # }
//! ```
//!
//! # Where to start
//!
//! - [`elements`] is the inventory of built-in sources, filters, and sinks.
//!   Each type's own documentation states what buffers it accepts, what it
//!   owns, and how it behaves under error and runtime control.
//! - [`pipeline`] builds and runs a graph; [`element`] and [`pad`] are the
//!   traits and the one output port everything is wired through.
//! - [`buffer`] is what travels between elements, [`control`] is what
//!   Pause/Resume/Stop/Seek travel through, and [`bus`] is how an element
//!   reports something the caller could not have been handed directly.
//!
//! # Buffer and timeline contract
//!
//! [`MediaBuffer`](buffer::MediaBuffer) payloads are `Arc`-wrapped, so
//! fan-out clones a reference rather than the media itself. PTS, duration,
//! packet time bases, and video color information survive every stage that
//! does not deliberately create a new timeline.
//!
//! [`Eos`](buffer::MediaBuffer::Eos) is data, and it is forwarded like data:
//! stateful stages (encoders holding delayed frames, muxers, resamplers)
//! flush on it before passing it on. That is what separates the two ways a
//! pipeline ends — [`Pipeline::finish`](pipeline::Pipeline::finish) sends
//! ordered EOS from the source and drains everything behind it, while
//! [`Pipeline::stop`](pipeline::Pipeline::stop) abandons buffered work.
//!
//! # Changing a running pipeline
//!
//! A branch keeps the shape it was built with, and a few elements are where a
//! running graph changes anyway:
//!
//! - [`Tee`](elements::Tee) fans out, and its
//!   [`TeeHandle`](elements::TeeHandle) adds a branch while buffers flow,
//!   finishes one cleanly — a recording finalized while its preview keeps
//!   running — or abandons one.
//! - [`AudioMixer`](elements::AudioMixer) and the video compositors fan in,
//!   taking inputs and layers while they run. With
//!   [`VideoCompositorOptions::background_alpha`](elements::VideoCompositorOptions::background_alpha)
//!   one composition can be a layer of the next: an overlay, rather than a
//!   rectangle covering what is under it.
//! - [`Rack`](elements::Rack) replaces the filters in a stretch of chain
//!   between two buffers, so one can be added or removed without reopening
//!   the source.
//! - [`PipelineBridge`](elements::PipelineBridge) carries buffers from one
//!   pipeline into another, so a source that dies takes only its own pipeline
//!   with it.
//! - Compositor and capture frame rates, and the mixer's format, change while
//!   running. Both re-mean the timestamps that follow, so they are for a
//!   preview, not the middle of a recording.
//! - [`FileDemuxerHandle`](elements::FileDemuxerHandle) makes a file loop,
//!   carrying its timeline across each lap so pacing and muxing continue.
//!
//! # Seeking
//!
//! A source says whether it is live and whether it is seekable.
//! [`Pipeline::seek`](pipeline::Pipeline::seek) first asks every branch
//! whether it can follow — a recording muxer cannot — and changes nothing if
//! one refuses; then it runs Pause, Flush, Seek and Preroll, and restores the
//! state the caller had. [`SeekMode::Accurate`](pipeline::SeekMode::Accurate)
//! decodes forward to the exact target, and
//! [`SeekMode::Keyframe`](pipeline::SeekMode::Keyframe) shows the keyframe the
//! demuxer landed on.
//!
//! # Link contracts
//!
//! Building a branch refuses a connection that could never carry data —
//! encoded packets into something that takes frames, an audio stream into a
//! video decoder, a D3D11 texture into a CPU or CUDA filter — before anything
//! runs:
//!
//! ```text
//! decoder produces VideoFrame (System), which rec cannot accept
//! (it takes VideoPacket|AudioPacket)
//! ```
//!
//! It compares only what an element knows when it is constructed: the media
//! kind, and for a decoded frame the memory domain. It is not caps
//! negotiation — nothing is converted or renegotiated — and pixel format, size
//! and device are still checked against each real buffer. An element that
//! declares nothing always links. See [`contract`].
//!
//! # Watching it run
//!
//! [`Pipeline::stats`](pipeline::Pipeline::stats) reads what every element is
//! doing — buffers and packet bytes, time inside `consume`, how long it has
//! been idle, errors, a queue's fill and drops, a compositor's frames drawn
//! and missed — as running totals, so two readings give a rate.
//!
//! # Features and platforms
//!
//! The crate has no default features. Hardware backends (`d3d11`, `d3d12`,
//! `dxgi-capture`, `cuda`, `wasapi-*`, `pipewire-*`) and the optional `ort`
//! and `webrtc` integrations are each behind their own Cargo feature, and
//! backend-specific types carry the backend's prefix. [docs.rs] builds this
//! crate for Linux and therefore omits the Windows-only API; the complete
//! reference is published separately (see the repository README).
//!
//! # Logging
//!
//! Diagnostics never install a global `log` logger or `tracing` subscriber.
//! The file logger in [`log`] is private and opt-in through
//! [`log::init`], and the caller owns the returned guard for as long as
//! records must keep being written and flushed.
//!
//! [docs.rs]: https://docs.rs/media-pp

// docs.rs passes `--cfg docsrs` (see `package.metadata.docs.rs`), which labels
// every feature-gated item with the Cargo feature that enables it. Stable
// builds never see the `feature` attribute.
#![cfg_attr(docsrs, feature(doc_cfg))]

mod core;
pub mod elements;
pub mod error;
mod platform;
#[cfg(test)]
mod test_support;

// Flat re-export: `core/` only exists to group these files on disk (see
// its module doc) — every external and internal caller keeps using
// `crate::pipeline`/`media_pp::pipeline` etc., never `crate::core::...`.
pub use core::diagnostics::{log, pp_log, stats};
pub use core::timing::{clock, playback_clock, rate};
pub use core::{
    buffer, bus, color, contract, control, driver, element, graph, pad, pipeline, pool, queue,
    subtitle,
};

// Same flat-namespace reasoning as above, but crate-private: `schedule`/
// `time` are pacing/rescale internals `crate::elements` builds on, not
// exposed in any public element's own field/method signature — nothing
// downstream of this crate needs `PeriodicSchedule`/`ActiveTimeline`/
// `MediaTimestamp`/`TimeBase` itself. `pub(crate) use` keeps the same
// `crate::schedule`/`crate::time` paths working for every internal caller
// without also making them part of this crate's external API surface.
pub(crate) use core::repeat;
pub(crate) use core::timing::{schedule, time};

pub use error::{Error, Result};

/// The [`ffmpeg-next`](https://docs.rs/ffmpeg-next) this crate is built on.
///
/// Re-exported because it is part of this crate's API, not an implementation
/// detail behind it: [`MediaBuffer`](buffer::MediaBuffer) carries `ffmpeg`
/// packets and frames directly, an encoder's `parameters()`/`time_base` are
/// `ffmpeg` types, and [`Error::Ffmpeg`] wraps `ffmpeg`'s own error.
///
/// Use this rather than depending on `ffmpeg-next` separately. A separate
/// dependency has to resolve to the same version as this crate's — when it
/// does not, the two `ffmpeg-next`s are distinct crates to the compiler and
/// every one of the types above stops matching, with nothing in the error
/// pointing at the version as the cause.
pub use ffmpeg_next as ffmpeg;

/// Must be called once before using any element that touches ffmpeg.
pub fn init() -> Result<()> {
    ffmpeg_next::init()?;
    Ok(())
}
