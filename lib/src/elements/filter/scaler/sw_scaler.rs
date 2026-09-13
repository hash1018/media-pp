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
    pool::{UnboundObjectPool, UnboundObjectPoolRef},
    repeat::{PerFrameTransform, RepeatedOutput},
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

    /// FFmpeg could not take a second reference to the scale already in
    /// hand, which is how an unchanged input picture is answered.
    #[error("failed to reference the previous scale (code {0})")]
    FrameRef(i32),

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
    /// The colour description `context` was last set up to read, so a
    /// frame that says something different is read by what it says — and
    /// one that says the same costs nothing. `None` whenever the context is
    /// (re)built, since building it forgets.
    colour: Option<Colour>,
    /// Reused across every scaled frame instead of allocating a fresh one
    /// each time — see [`UnboundObjectPool`]'s docs. Pre-filled to
    /// `dst_format`/`dst_width`/`dst_height` in `new` (unlike a decoder's
    /// pool, the output shape here is known up front, not learned from
    /// the first frame).
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
    /// The last scale and the picture it was made from, so a producer that
    /// re-emits an unchanged frame is answered with it instead of another
    /// pass over every pixel — see [`RepeatedOutput`].
    repeated: RepeatedOutput,
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
            OutputContract::Fixed(PortContract::frame(
                MediaKind::VideoFrame,
                MemoryDomain::System,
            )),
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
            colour: None,
            pool,
            repeated: RepeatedOutput::new(),
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
        InputContract::Fixed(PortContract::frame(
            MediaKind::VideoFrame,
            MemoryDomain::System,
        ))
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            // The same picture as last time scales to the frame already in
            // hand — see [`PerFrameTransform`], which is where that is
            // decided. `DxgiCaptureSource` under `CaptureMode::Cpu` re-emits
            // one on every tick of a still screen.
            MediaBuffer::Video(frame) => {
                let scaled = self.transform(&frame)?;
                self.pad.push(MediaBuffer::Video(scaled))
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
        // Nothing local to react to beyond the cached scale: unlike a
        // decoder, this has no reference-frame/reordering state to flush on
        // `Seek`, and nothing buffered to drop on `Stop` — a pure per-frame
        // spatial transform.
        if matches!(msg, ControlMsg::Flush | ControlMsg::Stop) {
            self.repeated.clear();
        }
        self.pad.control(msg)
    }
}

impl PerFrameTransform for SwScaler {
    fn repeated(&mut self) -> &mut RepeatedOutput {
        &mut self.repeated
    }

    fn frame_ref_failed(&self, code: i32) -> crate::error::Error {
        pp_error!(self, "av_frame_ref failed: {code}");
        SwScalerError::FrameRef(code).into()
    }

    fn produce(
        &mut self,
        frame: &Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        if !self.context_matches(frame) {
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
                    self.colour = None;
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

        let colour = Colour::of(frame);
        if self.colour != Some(colour) {
            self.read_as(colour, frame.format());
            self.colour = Some(colour);
        }

        // Already allocated to `dst_format`/`dst_width`/
        // `dst_height` (see `pool`'s docs), so `run` skips its own
        // allocation and scales straight into this buffer.
        let mut output = self.pool.get();
        self.context
            .as_mut()
            .expect("built or confirmed matching above")
            .run(frame, &mut output)
            .inspect_err(|error| pp_error!(self, "scale failed: {error}"))
            .map_err(SwScalerError::from)?;
        // `run` only copies pixel data, not metadata — carry the
        // pts through by hand so downstream pacing/muxing still
        // sees the original timestamp.
        output.set_pts(frame.pts());
        self.describe(frame, &mut output);
        Ok(output)
    }
}

impl SwScaler {
    /// Sets the context up to read frames that say `colour`.
    ///
    /// Only for a YUV source, which is the only kind a matrix and a range
    /// mean anything for. The destination keeps the source's own matrix and
    /// range when it is YUV too — a change of layout or size is not a change
    /// of colour, and rewriting BT.709 into swscale's default BT.601 there
    /// would put the wrong colours into everything downstream that believes
    /// the frame's description — and is full range when it is RGB, which
    /// has no other kind.
    ///
    /// A frame that says nothing is read as BT.601 limited, which is what
    /// swscale has always done with one, so an untagged source scales
    /// exactly as it did.
    fn read_as(&mut self, colour: Colour, input: ffmpeg::format::Pixel) {
        if is_rgb(input) {
            return;
        }
        let Some(context) = self.context.as_mut() else {
            return;
        };
        let full = colour.range == ffmpeg::color::Range::JPEG;
        let destination_full = is_rgb(self.dst_format) || full;
        // SAFETY: `context` is a live `SwsContext` this element owns, and the
        // table is one of swscale's own static ones, which outlives it.
        let code = unsafe {
            let table = ffmpeg::ffi::sws_getCoefficients(matrix(colour.space));
            ffmpeg::ffi::sws_setColorspaceDetails(
                context.as_mut_ptr(),
                table,
                i32::from(full),
                table,
                i32::from(destination_full),
                0,
                1 << 16,
                1 << 16,
            )
        };
        if code < 0 {
            // Scaling still works, with swscale's default reading — the
            // colours of a tagged source are then off, not the picture.
            pp_error!(self, "could not read frames as {colour:?} (code {code})");
        }
    }

    /// Says what `output` is, from what `input` was — see [`Self::read_as`]
    /// for why each case is what it is. A source this did not read by its
    /// description says nothing, as before, rather than something a pooled
    /// frame kept from an earlier one.
    fn describe(&self, input: &ffmpeg::frame::Video, output: &mut ffmpeg::frame::Video) {
        use ffmpeg::color::{Primaries, Range, Space, TransferCharacteristic};

        if is_rgb(input.format()) {
            output.set_color_space(Space::Unspecified);
            output.set_color_range(Range::Unspecified);
            output.set_color_primaries(Primaries::Unspecified);
            output.set_color_transfer_characteristic(TransferCharacteristic::Unspecified);
            return;
        }
        if is_rgb(self.dst_format) {
            output.set_color_space(Space::RGB);
            output.set_color_range(Range::JPEG);
        } else {
            output.set_color_space(input.color_space());
            output.set_color_range(input.color_range());
        }
        output.set_color_primaries(input.color_primaries());
        output.set_color_transfer_characteristic(input.color_transfer_characteristic());
    }
}

/// The part of a frame's description that decides how its YUV is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Colour {
    space: ffmpeg::color::Space,
    range: ffmpeg::color::Range,
}

impl Colour {
    fn of(frame: &ffmpeg::frame::Video) -> Self {
        Self {
            space: frame.color_space(),
            range: frame.color_range(),
        }
    }
}

/// swscale's name for the matrix a frame says it was made with.
fn matrix(space: ffmpeg::color::Space) -> std::ffi::c_int {
    use ffmpeg::color::Space;
    use ffmpeg::ffi::{SWS_CS_BT2020, SWS_CS_DEFAULT, SWS_CS_FCC, SWS_CS_ITU709, SWS_CS_SMPTE240M};
    match space {
        Space::BT709 => SWS_CS_ITU709,
        Space::FCC => SWS_CS_FCC,
        Space::SMPTE240M => SWS_CS_SMPTE240M,
        Space::BT2020NCL | Space::BT2020CL => SWS_CS_BT2020,
        // BT.601 under each of its names, and a frame that names nothing.
        _ => SWS_CS_DEFAULT,
    }
}

/// Whether `pixel` holds RGB rather than YUV — for which a matrix and a
/// range mean nothing.
fn is_rgb(pixel: ffmpeg::format::Pixel) -> bool {
    // SAFETY: a lookup in libavutil's static table of descriptors, which
    // answers null for a format it does not know.
    unsafe {
        let descriptor = ffmpeg::ffi::av_pix_fmt_desc_get(pixel.into());
        !descriptor.is_null()
            && (*descriptor).flags & (ffmpeg::ffi::AV_PIX_FMT_FLAG_RGB as u64) != 0
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

    /// Another `AVFrame` over the same picture, with its own timestamp —
    /// what a capture with nothing new to show hands over on every tick.
    fn repeat_of(buffer: &MediaBuffer, pts: i64) -> MediaBuffer {
        let MediaBuffer::Video(source) = buffer else {
            panic!("expected a Video buffer");
        };
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut slot = pool.get();
        // SAFETY: both are live `AVFrame`s and distinct — the slot is the
        // empty one just taken from the pool.
        unsafe {
            assert!(ffmpeg::ffi::av_frame_ref(slot.as_mut_ptr(), source.as_ptr()) >= 0);
        }
        slot.set_pts(Some(pts));
        MediaBuffer::Video(Arc::new(slot))
    }

    /// A capture of a still screen re-emits the picture it already has, and
    /// scaling it again would produce the frame already in hand — another
    /// `libswscale` pass over every pixel for nothing. The repeat carries
    /// its own timestamp; only the pixels are shared.
    #[test]
    fn a_repeated_input_is_scaled_once() {
        let (mut scaler, received) = new_scaler(ffmpeg::format::Pixel::RGB24, 80, 60);
        let source = video_frame(ffmpeg::format::Pixel::YUV420P, 160, 120, 100);
        let repeat = repeat_of(&source, 200);

        scaler.consume(source).expect("scale the first frame");
        scaler.consume(repeat).expect("scale the repeat");

        let received = received.lock().unwrap();
        assert_eq!(received.len(), 2, "every frame still produces one");
        let (MediaBuffer::Video(first), MediaBuffer::Video(repeated)) =
            (&received[0], &received[1])
        else {
            panic!("expected Video buffers");
        };
        assert_eq!(
            crate::buffer::picture_id(repeated),
            crate::buffer::picture_id(first),
            "an unchanged picture was scaled a second time"
        );
        assert_eq!(
            repeated.pts(),
            Some(200),
            "a repeat carries this frame's timestamp, not the one it points at"
        );
        assert_eq!(repeated.format(), ffmpeg::format::Pixel::RGB24);
    }

    /// A flat NV12 picture holding one colour, described as `space`.
    fn flat_nv12(
        (y, u, v): (u8, u8, u8),
        space: ffmpeg::color::Space,
        range: ffmpeg::color::Range,
    ) -> MediaBuffer {
        let pool = UnboundObjectPool::new(
            0,
            || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, 16, 16),
            |_| {},
        );
        let mut frame = pool.get();
        frame.data_mut(0).fill(y);
        for pair in frame.data_mut(1).as_chunks_mut::<2>().0 {
            pair[0] = u;
            pair[1] = v;
        }
        frame.set_color_space(space);
        frame.set_color_range(range);
        frame.set_pts(Some(0));
        MediaBuffer::Video(Arc::new(frame))
    }

    fn scaled_once(scaler_format: ffmpeg::format::Pixel, input: MediaBuffer) -> MediaBuffer {
        let (mut scaler, received) = new_scaler(scaler_format, 16, 16);
        scaler.consume(input).expect("scale must succeed");
        received.lock().unwrap().remove(0)
    }

    fn middle_rgb(buffer: &MediaBuffer) -> [u8; 3] {
        let MediaBuffer::Video(frame) = buffer else {
            panic!("expected a Video buffer");
        };
        let at = 8 * frame.stride(0) + 8 * 3;
        let data = frame.data(0);
        [data[at], data[at + 1], data[at + 2]]
    }

    fn near(actual: [u8; 3], expected: [u8; 3]) -> bool {
        actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| actual.abs_diff(expected) <= 3)
    }

    /// (230, 20, 20) as BT.709 limited range — what the CUDA compositor's
    /// canvas holds for it.
    const RED_709: (u8, u8, u8) = (72, 107, 220);

    /// The defect this was written against: the CUDA compositor's canvas
    /// is BT.709 and says so, and read without asking it came out of here
    /// as BT.601 — (230, 20, 20) as (211, 0, 22), in a screenshot and in
    /// every tool that trusted it.
    #[test]
    fn a_frame_that_says_bt709_is_read_as_bt709() {
        let rgb = scaled_once(
            ffmpeg::format::Pixel::RGB24,
            flat_nv12(
                RED_709,
                ffmpeg::color::Space::BT709,
                ffmpeg::color::Range::MPEG,
            ),
        );
        let got = middle_rgb(&rgb);
        assert!(near(got, [230, 21, 21]), "read as {got:?}");
    }

    /// A frame that says nothing is read the way swscale always read one,
    /// so nothing that fed this an untagged picture sees it change.
    #[test]
    fn a_frame_that_says_nothing_is_read_as_it_always_was() {
        let rgb = scaled_once(
            ffmpeg::format::Pixel::RGB24,
            flat_nv12(
                RED_709,
                ffmpeg::color::Space::Unspecified,
                ffmpeg::color::Range::Unspecified,
            ),
        );
        let got = middle_rgb(&rgb);
        assert!(near(got, [212, 0, 23]), "read as {got:?}");
    }

    /// What comes out says what it is: RGB, full range.
    #[test]
    fn an_rgb_result_says_it_is_rgb() {
        let MediaBuffer::Video(rgb) = scaled_once(
            ffmpeg::format::Pixel::RGB24,
            flat_nv12(
                RED_709,
                ffmpeg::color::Space::BT709,
                ffmpeg::color::Range::MPEG,
            ),
        ) else {
            panic!("expected a Video buffer");
        };
        assert_eq!(rgb.color_space(), ffmpeg::color::Space::RGB);
        assert_eq!(rgb.color_range(), ffmpeg::color::Range::JPEG);
    }

    /// A change of layout is not a change of colour. Converted into
    /// swscale's default instead, a BT.709 canvas going to a software
    /// encoder would arrive as BT.601 numbers still described as BT.709.
    #[test]
    fn yuv_to_yuv_keeps_both_the_colours_and_what_they_are() {
        let MediaBuffer::Video(planar) = scaled_once(
            ffmpeg::format::Pixel::YUV420P,
            flat_nv12(
                RED_709,
                ffmpeg::color::Space::BT709,
                ffmpeg::color::Range::MPEG,
            ),
        ) else {
            panic!("expected a Video buffer");
        };
        let (y, u, v) = RED_709;
        let middle = |plane: usize, stride_step: usize| {
            planar.data(plane)[stride_step * planar.stride(plane) + stride_step]
        };
        assert!(middle(0, 8).abs_diff(y) <= 1, "Y {}", middle(0, 8));
        assert!(middle(1, 4).abs_diff(u) <= 1, "U {}", middle(1, 4));
        assert!(middle(2, 4).abs_diff(v) <= 1, "V {}", middle(2, 4));
        assert_eq!(planar.color_space(), ffmpeg::color::Space::BT709);
        assert_eq!(planar.color_range(), ffmpeg::color::Range::MPEG);
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
