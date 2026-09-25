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
//! use std::time::Duration;
//!
//! use media_pp::{
//!     elements::{FrameCounter, TestVideoOptions, TestVideoSource},
//!     pipeline::Pipeline,
//! };
//!
//! # fn main() -> media_pp::Result<()> {
//! let source = TestVideoSource::new("source", TestVideoOptions::default());
//! let (counter, frames) = FrameCounter::new("counter");
//!
//! let (pipeline, ()) = Pipeline::new("demo", source, |source, ctx| {
//!     let branch = ctx.branch().to(counter)?;
//!     ctx.attach(source, 0, branch)?;
//!     Ok(())
//! })?;
//!
//! pipeline.run()?;
//! std::thread::sleep(Duration::from_millis(200));
//! pipeline.stop();
//!
//! println!("frames: {}", frames.get());
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
//! Every element that completes EOS posts a
//! [`BusEvent::Eos`](bus::BusEvent::Eos), so the first one on the bus is only
//! the first thing to end; the pipeline posts
//! [`BusEvent::Finished`](bus::BusEvent::Finished) once every terminal sink
//! has, and that is when a stream played to its end can be stopped. For a
//! file it has to be: a file's source does not end at the end of the file
//! but waits there, since a seek could still take it back, so the bus stays
//! open and the pipeline running until it is stopped.
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
//! # Connecting elements
//!
//! A branch is built in the order buffers flow and attached to a source's
//! output, as the example above does:
//!
//! ```text
//! ctx.branch()                 start a branch
//!    .pipe(decoder)            a filter: consumes, and pushes on
//!    .queue("frames", 8)       a thread boundary with a bounded buffer
//!    .pipe(pacer)
//!    .to(renderer)?            the sink it ends in — checked here
//! ctx.attach(source, index, branch)?   onto the source's stream — checked again
//! ```
//!
//! ## What is caught before anything runs
//!
//! Building or attaching a branch refuses a link that could never carry
//! data, naming both sides and, where one element makes that crossing, what
//! to put between them:
//!
//! ```text
//! decoder produces VideoFrame (System), which rec cannot accept
//! (it takes VideoPacket|AudioPacket); encode it first: an encoder turns
//! frames into packets
//! ```
//!
//! That is what each element knows when it is constructed: packets or
//! frames and of which medium, which memory a frame lives in, and — where
//! construction settles it — its pixel layout, NV12, P010, BGRA or another.
//! Everything that only a frame can say is checked against each frame by
//! the element that reads it, and reported as an error on that frame: the
//! size and whether it is odd, a colour or HDR tag, which device a texture
//! belongs to, a format's finer points. An element that declares nothing
//! links to anything. None of it is caps negotiation: nothing is converted
//! or inserted for you. See [`contract`].
//!
//! ## Asking before linking
//!
//! The same rules answer while the elements are still in hand, so a caller
//! can decide whether something goes between them — which trying the link
//! cannot, since a refused branch drops what it was given:
//!
//! ```ignore
//! use media_pp::contract::check_elements;
//!
//! let fits = check_elements(&mut decoder, &renderer);
//! if fits.is_refused() {
//!     println!("{fits}"); // what does not fit, and what goes between
//!     // put the converter it names between them
//! }
//! ```
//!
//! [`contract::check_elements`] asks about a producer's first output;
//! [`contract::check_link`] takes a pad's [`pad::SrcPad::contract`] and a
//! sink's [`element::Sink::input_contract`] for a source with several. Both
//! answer [`LinkCheck::Fits`](contract::LinkCheck::Fits),
//! [`Refused`](contract::LinkCheck::Refused), or
//! [`Unknown`](contract::LinkCheck::Unknown) where one side says too little
//! to tell. They judge one link: past a `Queue` or a `Pacer`, which pass
//! frames on unchanged, ask the element before it.
//!
//! ## What goes between
//!
//! ```text
//! from                          to                      put between
//! encoded packets               frames                  SwDecoder, VideoDecodeBin, a hardware decoder
//! frames                        encoded packets         an encoder
//! system memory                 D3D11 / D3D12 / CUDA    D3d11Upload / D3d12Upload / CudaUpload
//! D3D11 / D3D12 / CUDA          system memory           D3d11Download / D3d12Download / CudaDownload
//! D3D11                         CUDA, or back           system memory: download, then upload
//! any layout, system memory     another                 SwScaler::to_format
//! NV12, P010 or BGRA, D3D11     NV12 or BGRA            D3d11Scaler::to_format with a D3d11ScalerFormat
//! PQ or HLG, D3D11              SDR BGRA                D3d11ToneMap
//! NV12, CUDA                    BGRA                    CudaConverter built for Bgra
//! BGRA, CUDA                    NV12                    CudaConverter built for Nv12
//! P010, CUDA                    NV12                    CudaScaler::to_format(.., Nv12)
//! ```
//!
//! None of these is told a size: an upload, a download, a converter and
//! [`SwScaler::to_format`](elements::SwScaler::to_format) are built for a
//! device and a layout, and take the size from the frames themselves, so
//! the element a refusal names can be put between the two without first
//! finding out how large the pictures are. A source that changes
//! resolution mid-stream is followed rather than refused. Where the size
//! is the point — a scaler asked for one through
//! [`SwScaler::new`](elements::SwScaler::new), a compositor's canvas, an
//! encoder's stream — it is still given, and such an element is what
//! absorbs a resolution change for whatever cannot take one.
//!
//! An upload takes only some layouts — `D3d11Upload` NV12 or BGRA,
//! `D3d12Upload` NV12 — so a software decoder's planar YUV goes through a
//! `SwScaler` first. [`elements::VideoDecodeBin`] makes the decode-side
//! choices itself: it decodes onto the device on whichever path can, and
//! says what it will put out through [`output_format`](elements::VideoDecodeBin::output_format).
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

mod app;
mod core;
pub mod elements;
pub mod error;
mod platform;
#[cfg(test)]
mod test_support;

// Flat re-export: `core/` and `app/` only exist to group these files on disk
// (see their module docs) — every external and internal caller keeps using
// `crate::pipeline`/`media_pp::pipeline` etc., never `crate::core::...`.
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
pub use app::player;
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
pub(crate) use core::frame_size;
pub(crate) use core::playback_state;
pub(crate) use core::repeat;
pub(crate) use core::timing::{schedule, time, timeline};
#[cfg(any(feature = "cuda", all(target_os = "windows", feature = "d3d11")))]
pub(crate) use core::tone_map;

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

/// Readies FFmpeg for this process, once, before anything here uses it.
///
/// What it does is register FFmpeg's error descriptions — without them an
/// `ffmpeg::Error` displays as an empty string, and a failure reads as
/// `ffmpeg error: ` with nothing after it — and its input devices, which an
/// element opening a camera through FFmpeg needs to find one. It used to be
/// the caller's to remember as `media_pp::init()`; forgetting it was silent,
/// so every element that reaches FFmpeg calls this on its way in instead —
/// through [`element::element_pp_log`], which every element builds its
/// identity with, and at the top of each constructor whose first FFmpeg call
/// comes before that.
pub(crate) fn ensure_ffmpeg() {
    static READY: std::sync::Once = std::sync::Once::new();
    READY.call_once(|| {
        // Registration only: nothing in it can fail on a supported FFmpeg,
        // and there is no caller here to hand an error to.
        let _ = ffmpeg_next::init();
    });
}
