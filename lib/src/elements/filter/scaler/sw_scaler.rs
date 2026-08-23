use std::sync::Arc;

use crate::pp_log::{PpLog, pp_debug, pp_error, pp_info};
use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    error::Result,
    pad::SrcPad,
    pool::UnboundObjectPool,
};

/// How many output frames [`SwScaler`] pre-allocates up front. Unlike
/// [`crate::elements::SwDecoder`]/[`crate::elements::D3d12Decoder`],
/// this doesn't have to start empty and grow — `dst_format`/`dst_width`/
/// `dst_height` are known at construction time, so the pool can be
/// correctly sized from the very first frame instead of paying for a
/// handful of allocations up front, amortized. Not exposed as a
/// constructor parameter (yet): this is a reasonable default for "a
/// `Queue` or two downstream," not a hard limit — the pool still grows
/// past this if more frames end up in flight at once.
const POOL_SIZE: usize = 4;

/// Errors specific to `SwScaler`. Converts into the crate-wide `Error` via
/// `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum SwScalerError {
    /// FFmpeg rejected creation or use of the scaling context.
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),

    /// The sink received a buffer other than decoded video or end-of-stream.
    #[error(
        "SwScaler only converts/resizes decoded Video frames, got a {0}; \
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
pub struct SwScaler {
    pp_log: PpLog,
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
// `&mut self` on every method that touches it (see `D3d12Decoder`'s
// `hw_device_ctx` for the same reasoning) already rules out concurrent
// access from multiple threads.
unsafe impl Send for SwScaler {}

impl SwScaler {
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
        let pp_log = element_pp_log(ElementType::SwScaler, &name, None);
        pp_info!(
            pp_log: &pp_log,
            "created: dst_format={dst_format:?}, dst={dst_width}x{dst_height}"
        );
        let pad = SrcPad::with_contract(
            format!("{name}_src"),
            OutputContract::Fixed(
                PortContract::of(MediaKind::VideoFrame).in_memory(MemoryDomain::System),
            ),
        );
        let pool = UnboundObjectPool::new(
            POOL_SIZE,
            move || ffmpeg::frame::Video::new(dst_format, dst_width, dst_height),
            |_| {},
        );
        Self {
            name,
            pp_log,
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

impl Element for SwScaler {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::SwScaler
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for SwScaler {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for SwScaler {
    /// swscale reads the planes on the CPU, so a device texture is unreachable memory here rather than merely the wrong format.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::of(MediaKind::VideoFrame).in_memory(MemoryDomain::System),
        )
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Video(frame) => {
                if !self.context_matches(&frame) {
                    match &mut self.context {
                        Some(context) => {
                            // A live source can renegotiate mid-stream — a
                            // captured window being resized, say. Absorbing that
                            // is exactly what keeps a fixed-geometry encoder
                            // downstream working, but it is also the kind of
                            // change worth seeing in a log when output suddenly
                            // looks stretched.
                            let previous = context.input();
                            pp_info!(
                                self,
                                "input changed: {}x{} {:?} -> {}x{} {:?}, rebuilding context",
                                previous.width,
                                previous.height,
                                previous.format,
                                frame.width(),
                                frame.height(),
                                frame.format()
                            );
                            context.cached(
                                frame.format(),
                                frame.width(),
                                frame.height(),
                                self.dst_format,
                                self.dst_width,
                                self.dst_height,
                                self.flags,
                            );
                        }
                        None => {
                            pp_debug!(
                                self,
                                "input is {}x{} {:?}, building context",
                                frame.width(),
                                frame.height(),
                                frame.format()
                            );
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
                                    pp_error!(self, "failed to build scaling context: {error}")
                                })
                                .map_err(SwScalerError::from)?,
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
                    .inspect_err(|error| pp_error!(self, "scale failed: {error}"))
                    .map_err(SwScalerError::from)?;
                // `run` only copies pixel data, not metadata — carry the
                // pts through by hand so downstream pacing/muxing still
                // sees the original timestamp.
                output.set_pts(frame.pts());

                self.pad.push(MediaBuffer::Video(Arc::new(output)))
            }
            MediaBuffer::Eos => self.pad.push(MediaBuffer::Eos),
            MediaBuffer::Packet(_) => {
                pp_error!(self, "unsupported buffer: Packet");
                Err(SwScalerError::UnsupportedBuffer("Packet").into())
            }
            MediaBuffer::Audio(_) => {
                pp_error!(self, "unsupported buffer: Audio");
                Err(SwScalerError::UnsupportedBuffer("Audio").into())
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

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    struct CapturingSink {
        pp_log: PpLog,
        received: Arc<Mutex<Vec<MediaBuffer>>>,
    }

    impl Element for CapturingSink {
        fn name(&self) -> Arc<str> {
            "capture".into()
        }

        fn element_type(&self) -> ElementType {
            ElementType::Other
        }

        fn pp_log(&self) -> &PpLog {
            &self.pp_log
        }

        fn pp_log_mut(&mut self) -> &mut PpLog {
            &mut self.pp_log
        }
    }

    impl Sink for CapturingSink {
        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            self.received.lock().unwrap().push(buf);
            Ok(())
        }

        fn control(&mut self, _msg: ControlMsg) -> Result<()> {
            Ok(())
        }
    }

    fn video_frame(
        format: ffmpeg::format::Pixel,
        width: u32,
        height: u32,
        pts: i64,
    ) -> MediaBuffer {
        let pool = UnboundObjectPool::new(
            0,
            move || ffmpeg::frame::Video::new(format, width, height),
            |_| {},
        );
        let mut frame = pool.get();
        frame.set_pts(Some(pts));
        MediaBuffer::Video(Arc::new(frame))
    }

    fn new_scaler(
        dst_format: ffmpeg::format::Pixel,
        dst_width: u32,
        dst_height: u32,
    ) -> (SwScaler, Arc<Mutex<Vec<MediaBuffer>>>) {
        let mut scaler = SwScaler::new(
            "scaler",
            dst_format,
            dst_width,
            dst_height,
            ffmpeg::software::scaling::Flags::BILINEAR,
        );
        let received = Arc::new(Mutex::new(Vec::new()));
        scaler.src_pads()[0].link(Box::new(CapturingSink {
            received: received.clone(),
            pp_log: element_pp_log(ElementType::Other, "capture", None),
        }));
        (scaler, received)
    }

    #[test]
    fn converts_pixel_format_and_size_while_preserving_pts() {
        let (mut scaler, received) = new_scaler(ffmpeg::format::Pixel::RGB24, 80, 60);
        scaler
            .consume(video_frame(ffmpeg::format::Pixel::YUV420P, 160, 120, 4242))
            .expect("scale must succeed");

        let received = received.lock().unwrap();
        assert_eq!(received.len(), 1);
        let MediaBuffer::Video(frame) = &received[0] else {
            panic!("expected a Video buffer");
        };
        assert_eq!(frame.format(), ffmpeg::format::Pixel::RGB24);
        assert_eq!(frame.width(), 80);
        assert_eq!(frame.height(), 60);
        assert_eq!(frame.pts(), Some(4242));
    }

    /// `context_matches` has to catch a mid-stream resolution change and
    /// rebuild, not silently keep scaling from a stale `sws_scale` context
    /// built for the previous frame's dimensions.
    #[test]
    fn rebuilds_its_scaling_context_when_input_dimensions_change_mid_stream() {
        let (mut scaler, received) = new_scaler(ffmpeg::format::Pixel::RGB24, 80, 60);
        scaler
            .consume(video_frame(ffmpeg::format::Pixel::YUV420P, 160, 120, 0))
            .expect("first frame must scale");
        scaler
            .consume(video_frame(ffmpeg::format::Pixel::YUV420P, 320, 240, 1))
            .expect("a differently-sized second frame must still scale, not reuse a stale context");

        let received = received.lock().unwrap();
        assert_eq!(received.len(), 2);
        for buf in received.iter() {
            let MediaBuffer::Video(frame) = buf else {
                panic!("expected a Video buffer");
            };
            assert_eq!(frame.width(), 80);
            assert_eq!(frame.height(), 60);
        }
    }

    #[test]
    fn eos_forwards_downstream() {
        let (mut scaler, received) = new_scaler(ffmpeg::format::Pixel::RGB24, 80, 60);
        scaler
            .consume(MediaBuffer::Eos)
            .expect("eos must forward cleanly");
        assert!(matches!(
            received.lock().unwrap().as_slice(),
            [MediaBuffer::Eos]
        ));
    }

    #[test]
    fn rejects_packet_and_audio_buffers_with_a_clean_error_instead_of_scaling_garbage() {
        let (mut scaler, _received) = new_scaler(ffmpeg::format::Pixel::RGB24, 80, 60);

        let packet = MediaBuffer::Packet(Arc::new(ffmpeg::Packet::empty()));
        assert!(
            scaler.consume(packet).is_err(),
            "Packet must be rejected, not silently accepted"
        );

        let audio = MediaBuffer::Audio(Arc::new(ffmpeg::frame::Audio::empty()));
        assert!(
            scaler.consume(audio).is_err(),
            "Audio must be rejected, not silently accepted"
        );
    }
}
