use std::sync::Arc;

use ffmpeg_next as ffmpeg;
use rust_hlog::{HLog, herror};
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_hlog},
    error::Result,
    pad::SrcPad,
    pool::UnboundObjectPool,
};

/// How many output frames [`Scaler`] pre-allocates up front. Unlike
/// [`crate::elements::SwDecoder`]/[`crate::elements::D3d12vaDecoder`],
/// this doesn't have to start empty and grow — `dst_format`/`dst_width`/
/// `dst_height` are known at construction time, so the pool can be
/// correctly sized from the very first frame instead of paying for a
/// handful of allocations up front, amortized. Not exposed as a
/// constructor parameter (yet): this is a reasonable default for "a
/// `Queue` or two downstream," not a hard limit — the pool still grows
/// past this if more frames end up in flight at once.
const POOL_SIZE: usize = 4;

/// Errors specific to `Scaler`. Converts into the crate-wide `Error` via
/// `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum ScalerError {
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),

    #[error(
        "Scaler only converts/resizes decoded Video frames, got a {0}; \
         link it straight after a decoder, not a demuxer"
    )]
    UnsupportedBuffer(&'static str),
}

/// Converts/resizes decoded video frames — pixel format (e.g. the YUV a
/// decoder produces -> the RGB most inference models expect) and
/// resolution (source resolution -> a model's fixed input size) in one
/// pass via `libswscale`. A `Filter`: receives via `Sink`, pushes the
/// converted frame on through its own (single) src pad.
///
/// Typical placement: right before something with a fixed input
/// contract, e.g. an ONNX object-detection model — not a general-purpose
/// pipeline stage, so most chains won't need one at all.
#[rust_hlog::hlog]
pub struct Scaler {
    name: Arc<str>,
    dst_format: ffmpeg::format::Pixel,
    dst_width: u32,
    dst_height: u32,
    flags: ffmpeg::software::scaling::Flags,
    /// Built lazily from the *first* frame's own format/dimensions
    /// (rather than requiring the caller to pass them up front) and
    /// rebuilt in place — via `Context::cached`, cheaper than tearing
    /// down and reallocating from scratch — if a later frame's
    /// format/dimensions ever differ (e.g. mid-stream resolution
    /// change). `None` until the first frame arrives.
    context: Option<ffmpeg::software::scaling::Context>,
    /// Reused across every scaled frame instead of allocating a fresh one
    /// each time — see [`UnboundObjectPool`]'s docs. Pre-filled to
    /// `dst_format`/`dst_width`/`dst_height` in `new` (unlike a decoder's
    /// pool, the output shape here is known up front, not learned from
    /// the first frame).
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
    pad: SrcPad,
}

// SAFETY: `ffmpeg::software::scaling::Context` wraps a heap-allocated
// `SwsContext` with no thread affinity of its own — ffmpeg-next marks
// the analogous audio `resampling::Context` (`SwrContext`) and every
// codec type `Send` for the same reason, this one's just missing it.
// `&mut self` on every method that touches it (see `D3d12vaDecoder`'s
// `hw_device_ctx` for the same reasoning) already rules out concurrent
// access from multiple threads.
unsafe impl Send for Scaler {}

impl Scaler {
    /// `dst_format`/`dst_width`/`dst_height` describe what every output
    /// frame will be; the source side is learned automatically from
    /// whatever frames actually arrive (see `context`'s docs), so this
    /// doesn't need decoder parameters up front the way
    /// [`crate::elements::SwDecoder::new`] does.
    pub fn new(
        name: impl Into<String>,
        dst_format: ffmpeg::format::Pixel,
        dst_width: u32,
        dst_height: u32,
        flags: ffmpeg::software::scaling::Flags,
    ) -> Self {
        let name: Arc<str> = name.into().into();
        let hlog = element_hlog(ElementType::Scaler, &name, None);
        let pad = SrcPad::new(format!("{name}_src"));
        let pool = UnboundObjectPool::new(
            POOL_SIZE,
            move || ffmpeg::frame::Video::new(dst_format, dst_width, dst_height),
            |_| {},
        );
        Self {
            name,
            hlog,
            dst_format,
            dst_width,
            dst_height,
            flags,
            context: None,
            pool,
            pad,
        }
    }

    /// Whether `self.context` (if any) is already configured for `frame`'s
    /// own format/dimensions — if not, `consume` has to (re)build it
    /// before scaling can proceed.
    fn context_matches(&self, frame: &ffmpeg::frame::Video) -> bool {
        match &self.context {
            Some(context) => {
                let input = context.input();
                input.format == frame.format()
                    && input.width == frame.width()
                    && input.height == frame.height()
            }
            None => false,
        }
    }
}

impl Element for Scaler {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::Scaler
    }

    fn hlog(&self) -> &HLog {
        &self.hlog
    }

    fn hlog_mut(&mut self) -> &mut HLog {
        &mut self.hlog
    }
}

impl Source for Scaler {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for Scaler {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Video(frame) => {
                if !self.context_matches(&frame) {
                    match &mut self.context {
                        Some(context) => context.cached(
                            frame.format(),
                            frame.width(),
                            frame.height(),
                            self.dst_format,
                            self.dst_width,
                            self.dst_height,
                            self.flags,
                        ),
                        None => {
                            self.context = Some(
                                ffmpeg::software::scaling::Context::get(
                                    frame.format(),
                                    frame.width(),
                                    frame.height(),
                                    self.dst_format,
                                    self.dst_width,
                                    self.dst_height,
                                    self.flags,
                                )
                                .inspect_err(|error| {
                                    herror!(self, "failed to build scaling context: {error}")
                                })
                                .map_err(ScalerError::from)?,
                            );
                        }
                    }
                }

                // Already allocated to `dst_format`/`dst_width`/
                // `dst_height` (see `pool`'s docs), so `run` skips its own
                // allocation and scales straight into this buffer.
                let mut output = self.pool.get();
                self.context
                    .as_mut()
                    .expect("built or confirmed matching above")
                    .run(&frame, &mut output)
                    .inspect_err(|error| herror!(self, "scale failed: {error}"))
                    .map_err(ScalerError::from)?;
                // `run` only copies pixel data, not metadata — carry the
                // pts through by hand so downstream pacing/muxing still
                // sees the original timestamp.
                output.set_pts(frame.pts());

                self.pad.push(MediaBuffer::Video(Arc::new(output)))
            }
            MediaBuffer::Eos => self.pad.push(MediaBuffer::Eos),
            MediaBuffer::Packet(_) => {
                herror!(self, "unsupported buffer: Packet");
                Err(ScalerError::UnsupportedBuffer("Packet").into())
            }
            MediaBuffer::Audio(_) => {
                herror!(self, "unsupported buffer: Audio");
                Err(ScalerError::UnsupportedBuffer("Audio").into())
            }
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        // Nothing local to react to for any `ControlMsg`: unlike a
        // decoder, this has no reference-frame/reordering state to
        // flush on `Seek`, and nothing buffered to drop on `Stop` — a
        // pure per-frame spatial transform, so just forward.
        self.pad.control(msg)
    }
}
