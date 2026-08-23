use std::sync::Arc;

use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_error, pp_info};

use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    elements::{CudaDriverError, CudaUploadError},
    error::Result,
    pad::SrcPad,
    platform::cuda::{
        CudaDevice, CudaFrameFormat,
        driver::{BgraSurface, CudaDriver, Nv12Surface},
        frame::create_hw_frames_ctx,
    },
    platform::ffmpeg::AvBufferRef,
    pool::UnboundObjectPool,
};

/// Errors specific to `CudaConverter`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum CudaConverterError {
    /// The input frame is not backed by CUDA hardware surfaces.
    #[error("CudaConverter converts CUDA frames, got {0:?}")]
    UnsupportedFormat(ffmpeg::format::Pixel),
    /// The sink received a buffer other than decoded video.

    #[error("CudaConverter only accepts Video buffers, got a {0}")]
    UnsupportedBuffer(&'static str),
    /// The frame does not retain the CUDA hardware frames context that owns it.

    #[error("the frame carries no CUDA frames context")]
    MissingFramesContext,
    /// The input frame belongs to another CUDA context.

    #[error("the frame belongs to a different CUDA device than this CudaConverter")]
    ForeignContext,
    /// The CUDA surface is not BGRA.

    #[error("CudaConverter converts BGRA surfaces, got {0:?}")]
    UnsupportedSurfaceFormat(ffmpeg::format::Pixel),
    /// Input dimensions differ from the converter's fixed dimensions.

    #[error(
        "frame is {actual_width}x{actual_height}, but this CudaConverter was built for \
         {expected_width}x{expected_height}"
    )]
    DimensionMismatch {
        /// Input frame width in pixels.
        actual_width: u32,
        /// Input frame height in pixels.
        actual_height: u32,
        /// Width configured for the converter.
        expected_width: u32,
        /// Height configured for the converter.
        expected_height: u32,
    },
    /// NV12 output was requested with odd dimensions.

    #[error("NV12 needs even dimensions, got {width}x{height}")]
    OddDimensions {
        /// Odd output width in pixels.
        width: u32,
        /// Odd output height in pixels.
        height: u32,
    },
    /// A CUDA frame contains no usable device-memory plane.

    #[error("a surface arrived with no device pointer")]
    MissingSurface,
    /// Allocating or initializing the CUDA output frame failed.

    #[error(transparent)]
    Frames(#[from] CudaUploadError),
    /// The CUDA color-conversion kernel failed.

    #[error(transparent)]
    Driver(#[from] CudaDriverError),
}

/// Converts CUDA-resident BGRA surfaces into CUDA-resident NV12 ones.
///
/// A `Filter`: receives via `Sink`, pushes the converted frame into its own
/// single src pad. PTS, duration, and color metadata are carried across with
/// `av_frame_copy_props`, so this creates no new timeline.
///
/// # What this is for
///
/// BGRA is what every screen capture in this crate produces and what NVENC
/// ingests directly, so a capture-to-recording pipeline needs no conversion
/// at all — see [`CudaFrameFormat`]. Everything else on the CUDA path works
/// in NV12: [`crate::elements::CudaVideoCompositor`] composites it,
/// [`crate::elements::CudaRenderer`] presents it, and
/// [`crate::elements::CudaDecoder`] produces it. Without this element a
/// GPU-resident capture can only be encoded, never overlaid or shown.
///
/// [`crate::elements::CudaScaler`] deliberately does not do this: it resizes
/// through `scale_cuda`, which answers a BGRA input with "Unsupported
/// conversion: bgra -> semiplanar8". The conversion here is this crate's own
/// kernel instead.
///
/// # Colour
///
/// Full-range RGB in, BT.709 limited-range Y'CbCr out — the same definition
/// [`crate::elements::CudaVideoCompositor`] fills backgrounds with, so a
/// converted capture and a composited background agree. Chroma is the
/// conversion of each 2x2 block's average colour, which is why the dimensions
/// must be even: an odd extent has no whole chroma sample to write.
///
/// The converted frame is tagged for what it now holds — `BT709` /
/// limited-range — rather than inheriting the source's RGB tags. That is the
/// one part of a frame's metadata this element deliberately replaces; PTS and
/// duration cross unchanged.
pub struct CudaConverter {
    pp_log: PpLog,
    name: Arc<str>,
    /// This element's own reference to the shared context, released in `Drop`.
    _hw_device_ctx: Arc<AvBufferRef>,
    /// The pool converted frames are allocated from.
    hw_frames_ctx: AvBufferRef,
    /// The device context incoming frames must belong to, compared by
    /// pointer — a surface from another device would be read against the
    /// wrong context.
    device_ctx: *mut ffi::AVHWDeviceContext,
    driver: CudaDriver,
    width: u32,
    height: u32,
    pad: SrcPad,
    /// Reuses only the small CPU-side `AVFrame` wrapper; the CUDA surface
    /// itself comes from `hw_frames_ctx`'s own pool. Same split as
    /// `CudaUpload`.
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
}

// SAFETY: the buffers are heap-allocated FFmpeg buffers with no thread
// affinity of their own, `device_ctx` only ever has its address compared, and
// `&mut self` on every method that touches them rules out concurrent access.
// Same reasoning as `CudaUpload`.
unsafe impl Send for CudaConverter {}

impl CudaConverter {
    /// `device` must be the same [`CudaDevice`] every other CUDA element in
    /// this pipeline was built from. This element takes its own FFmpeg
    /// reference, so `device` itself need not outlive the call.
    pub fn new(
        name: impl Into<String>,
        device: &CudaDevice,
        width: u32,
        height: u32,
    ) -> std::result::Result<Self, CudaConverterError> {
        if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Err(CudaConverterError::OddDimensions { width, height });
        }
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::CudaConverter, &name, None);

        let driver = CudaDriver::retain_primary()?;
        let hw_device_ctx = device.retain();
        let hw_frames_ctx =
            // SAFETY: `create_hw_frames_ctx`'s contract is a live device context, which
            // is what the owned `AvBufferRef` beside it is.
            unsafe { create_hw_frames_ctx(&hw_device_ctx, CudaFrameFormat::Nv12, width, height) }
                .map_err(CudaUploadError::from)?;
        // SAFETY: `hw_device_ctx` owns a live `AVBufferRef` for a CUDA device
        // context, whose `data` is that `AVHWDeviceContext` by FFmpeg's own
        // definition. Only the pointer's identity is kept, to compare against an
        // incoming frame's; the reference held alongside it is what keeps that
        // identity from being reused by a different context.
        let device_ctx = unsafe { (*hw_device_ctx.as_ptr()).data as *mut ffi::AVHWDeviceContext };

        let pad = SrcPad::with_contract(
            format!("{name}_src"),
            OutputContract::Fixed(
                PortContract::of(MediaKind::VideoFrame).in_memory(MemoryDomain::Cuda),
            ),
        );
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        pp_info!(pp_log: &pp_log, "opened: {width}x{height} BGRA -> NV12");
        Ok(Self {
            name,
            pp_log,
            _hw_device_ctx: hw_device_ctx,
            hw_frames_ctx,
            device_ctx,
            driver,
            width,
            height,
            pad,
            pool,
        })
    }

    /// Rejects anything this cannot convert before a device pointer is read
    /// out of it: the format, the frames context and its device, the surface
    /// layout, and the size this element was built for.
    fn validate(
        &self,
        frame: &ffmpeg::frame::Video,
    ) -> std::result::Result<(), CudaConverterError> {
        if frame.format() != ffmpeg::format::Pixel::CUDA {
            return Err(CudaConverterError::UnsupportedFormat(frame.format()));
        }
        if frame.width() != self.width || frame.height() != self.height {
            return Err(CudaConverterError::DimensionMismatch {
                actual_width: frame.width(),
                actual_height: frame.height(),
                expected_width: self.width,
                expected_height: self.height,
            });
        }
        // SAFETY: `frame` is a live `frame::Video` already confirmed to be
        // `Pixel::CUDA`, so `as_ptr` yields an initialized `AVFrame` and a hardware
        // frame's `hw_frames_ctx` is either null — rejected here — or an
        // `AVBufferRef` whose `data` is an `AVHWFramesContext`. Only pointer
        // identity is compared, never dereferenced past that.
        unsafe {
            let frames_ref = (*frame.as_ptr()).hw_frames_ctx;
            if frames_ref.is_null() {
                return Err(CudaConverterError::MissingFramesContext);
            }
            let frames_ctx = (*frames_ref).data as *const ffi::AVHWFramesContext;
            if !std::ptr::eq((*frames_ctx).device_ctx, self.device_ctx) {
                return Err(CudaConverterError::ForeignContext);
            }
            let sw_format = (*frames_ctx).sw_format;
            if CudaFrameFormat::from_sw_format(sw_format) != Some(CudaFrameFormat::Bgra) {
                return Err(CudaConverterError::UnsupportedSurfaceFormat(
                    ffmpeg::format::Pixel::from(sw_format),
                ));
            }
        }
        Ok(())
    }

    fn convert(&mut self, source: &ffmpeg::frame::Video) -> Result<()> {
        self.validate(source)
            .inspect_err(|error| pp_error!(self, "{error}"))?;

        let mut destination = self.pool.get();
        // SAFETY: `ptr` is the pooled wrapper's own `AVFrame`, and the unref before
        // the allocation is what hands its previous surface back — see the comment
        // beside it. The frames context is this element's own, held for its life.
        unsafe {
            let dst = destination.as_mut_ptr();
            // The pooled wrapper may still reference the previous frame's
            // surface; releasing it here is what returns that surface to the
            // frames pool rather than leaking it for the element's lifetime.
            ffi::av_frame_unref(dst);
            let code = ffi::av_hwframe_get_buffer(self.hw_frames_ctx.as_ptr(), dst, 0);
            if code < 0 {
                pp_error!(self, "av_hwframe_get_buffer failed: {code}");
                return Err(CudaConverterError::Frames(CudaUploadError::HwFrameGet(code)).into());
            }
        }

        (|| -> std::result::Result<(), CudaConverterError> {
            let source_surface =
                BgraSurface::from_frame(source).ok_or(CudaConverterError::MissingSurface)?;
            let destination_surface =
                Nv12Surface::from_frame(&destination).ok_or(CudaConverterError::MissingSurface)?;
            self.driver.bgra_to_nv12(
                source_surface,
                destination_surface,
                self.width,
                self.height,
            )?;
            // The kernels are issued on this driver's own context, while
            // whatever reads the result next — an encoder, a download —
            // issues its work through FFmpeg's stream. One synchronize per
            // frame is what makes the conversion visible to both, the same
            // point `CudaVideoCompositor` synchronizes at.
            self.driver.synchronize()?;
            Ok(())
        })()
        .inspect_err(|error| pp_error!(self, "{error}"))?;

        // SAFETY: both frames are live and distinct — `destination` came from the
        // pool, `source` is the caller's — so `av_frame_copy_props` reads one and
        // writes the other with no aliasing.
        unsafe {
            // PTS and duration are part of the buffer contract and would
            // otherwise be dropped here.
            ffi::av_frame_copy_props(destination.as_mut_ptr(), source.as_ptr());
        }
        // Colour is the one thing this does *not* carry across: the input is
        // full-range RGB and the output is BT.709 limited-range Y'CbCr, so
        // copying the source's tags would describe the result as something it
        // is not. Everything downstream reads these — an encoder tags its
        // stream from them, a scaler picks its matrix from them — and
        // libavfilter reports the mismatch as frame properties changing on
        // the fly.
        destination.set_color_space(ffmpeg::color::Space::BT709);
        destination.set_color_range(ffmpeg::color::Range::MPEG);
        self.pad.push(MediaBuffer::Video(Arc::new(destination)))
    }
}

impl Element for CudaConverter {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::CudaConverter
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for CudaConverter {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for CudaConverter {
    /// Converts pixel layout on the device; the layout itself is a runtime value, not part of this.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::of(MediaKind::VideoFrame).in_memory(MemoryDomain::Cuda))
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Video(frame) => self.convert(&frame),
            MediaBuffer::Eos => self.pad.push(MediaBuffer::Eos),
            other => Err(CudaConverterError::UnsupportedBuffer(other.kind()).into()),
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        // Nothing local to react to — a pure per-frame conversion, same
        // reasoning as `CudaUpload::control`.
        self.pad.control(msg)
    }
}

impl Drop for CudaConverter {
    fn drop(&mut self) {
        pp_info!(self, "dropped: releasing hw contexts");
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::elements::CudaUpload;
    use crate::test_support::try_cuda_device;

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

    fn capture(element: &mut dyn Source) -> Arc<Mutex<Vec<MediaBuffer>>> {
        let received = Arc::new(Mutex::new(Vec::new()));
        element.src_pads()[0].link(Box::new(CapturingSink {
            received: received.clone(),
            pp_log: element_pp_log(ElementType::Other, "capture", None),
        }));
        received
    }

    /// One CUDA-resident frame of `format`, by way of the upload element that
    /// allocates the same kind of frames context this converter validates
    /// against.
    fn cuda_frame(
        device: &CudaDevice,
        format: CudaFrameFormat,
        width: u32,
        height: u32,
        pts: i64,
    ) -> Option<MediaBuffer> {
        let Ok(mut upload) = CudaUpload::new("upload", device, format, width, height) else {
            eprintln!("skipping: this machine has no usable CUDA frames context");
            return None;
        };
        let uploaded = capture(&mut upload);
        let mut frame = ffmpeg::frame::Video::new(format.pixel(), width, height);
        frame.set_pts(Some(pts));
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut slot = pool.get();
        *slot = frame;
        upload
            .consume(MediaBuffer::Video(Arc::new(slot)))
            .expect("upload");
        Some(uploaded.lock().unwrap().remove(0))
    }

    /// Another reference to the same buffer, so one uploaded surface can be
    /// offered to two consumers.
    fn clone_buffer(buffer: &MediaBuffer) -> MediaBuffer {
        match buffer {
            MediaBuffer::Video(frame) => MediaBuffer::Video(frame.clone()),
            other => panic!("expected a Video buffer, got {}", other.kind()),
        }
    }

    fn converter(device: &CudaDevice, width: u32, height: u32) -> Option<CudaConverter> {
        match CudaConverter::new("convert", device, width, height) {
            Ok(converter) => Some(converter),
            Err(error) => {
                eprintln!("skipping: no usable CUDA conversion here ({error})");
                None
            }
        }
    }

    #[test]
    fn a_converted_frame_is_an_nv12_surface_carrying_the_input_timestamp() {
        let Some((device, _cuda_lock)) = try_cuda_device() else {
            return;
        };
        let Some(mut converter) = converter(&device, 64, 32) else {
            return;
        };
        let Some(source) = cuda_frame(&device, CudaFrameFormat::Bgra, 64, 32, 7) else {
            return;
        };
        let converted = capture(&mut converter);

        converter.consume(source).expect("convert");

        let received = converted.lock().unwrap();
        let MediaBuffer::Video(frame) = &received[0] else {
            panic!("expected a Video buffer");
        };
        assert_eq!(frame.format(), ffmpeg::format::Pixel::CUDA);
        assert_eq!((frame.width(), frame.height()), (64, 32));
        assert_eq!(frame.pts(), Some(7), "props are carried, not recreated");
        // SAFETY: the assertions above have established this is a live CUDA frame,
        // so its `hw_frames_ctx` is set and its `data` is an `AVHWFramesContext`.
        let sw_format = unsafe {
            let frames_ctx =
                (*(*frame.as_ptr()).hw_frames_ctx).data as *const ffi::AVHWFramesContext;
            (*frames_ctx).sw_format
        };
        assert_eq!(
            CudaFrameFormat::from_sw_format(sw_format),
            Some(CudaFrameFormat::Nv12)
        );
    }

    /// The conversion changes what the pixels *are*, so the tags that
    /// describe them have to change with it: an encoder tags its stream from
    /// these, and a scaler picks its matrix from them.
    #[test]
    fn a_converted_frame_is_tagged_for_the_colour_it_now_holds() {
        let Some((device, _cuda_lock)) = try_cuda_device() else {
            return;
        };
        let Some(mut converter) = converter(&device, 64, 32) else {
            return;
        };
        // What a capture hands over: full-range RGB.
        let Some(source) = cuda_frame(&device, CudaFrameFormat::Bgra, 64, 32, 0) else {
            return;
        };
        let MediaBuffer::Video(frame) = &source else {
            panic!("the upload produces a Video buffer");
        };
        let mut tagged = ffmpeg::frame::Video::empty();
        // SAFETY: `frame` is a live frame from the upload above, and the target is
        // the empty local on the line before, so the two cannot alias.
        unsafe {
            ffi::av_frame_ref(tagged.as_mut_ptr(), frame.as_ptr());
        }
        tagged.set_color_space(ffmpeg::color::Space::RGB);
        tagged.set_color_range(ffmpeg::color::Range::JPEG);
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut slot = pool.get();
        *slot = tagged;

        let converted = capture(&mut converter);
        converter
            .consume(MediaBuffer::Video(Arc::new(slot)))
            .expect("convert");

        let received = converted.lock().unwrap();
        let MediaBuffer::Video(out) = &received[0] else {
            panic!("expected a Video buffer");
        };
        assert_eq!(out.color_space(), ffmpeg::color::Space::BT709);
        assert_eq!(out.color_range(), ffmpeg::color::Range::MPEG);
    }

    /// NV12 in is not a conversion this performs, and reading it as BGRA
    /// would produce a picture rather than an error.
    #[test]
    fn an_nv12_surface_is_refused_rather_than_reinterpreted() {
        let Some((device, _cuda_lock)) = try_cuda_device() else {
            return;
        };
        let Some(mut converter) = converter(&device, 64, 32) else {
            return;
        };
        let Some(source) = cuda_frame(&device, CudaFrameFormat::Nv12, 64, 32, 0) else {
            return;
        };

        let error = converter.consume(source).expect_err("NV12 is refused");

        assert!(
            error
                .to_string()
                .contains("CudaConverter converts BGRA surfaces"),
            "got {error}"
        );
    }

    #[test]
    fn a_cpu_frame_is_refused() {
        let Some((device, _cuda_lock)) = try_cuda_device() else {
            return;
        };
        let Some(mut converter) = converter(&device, 64, 32) else {
            return;
        };
        let frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, 64, 32);
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut slot = pool.get();
        *slot = frame;

        let error = converter
            .consume(MediaBuffer::Video(Arc::new(slot)))
            .expect_err("a CPU frame is refused");

        assert!(
            error
                .to_string()
                .contains("CudaConverter converts CUDA frames"),
            "got {error}"
        );
    }

    #[test]
    fn a_differently_sized_frame_is_refused() {
        let Some((device, _cuda_lock)) = try_cuda_device() else {
            return;
        };
        let Some(mut converter) = converter(&device, 64, 32) else {
            return;
        };
        let Some(source) = cuda_frame(&device, CudaFrameFormat::Bgra, 32, 32, 0) else {
            return;
        };

        let error = converter
            .consume(source)
            .expect_err("a mismatch is refused");

        assert!(error.to_string().contains("32x32"), "got {error}");
    }

    /// NV12 chroma is 2x2 subsampled, so an odd extent has no whole chroma
    /// sample to write. Refused where the size is chosen rather than per
    /// frame.
    #[test]
    fn odd_dimensions_are_refused_at_construction() {
        let Some((device, _cuda_lock)) = try_cuda_device() else {
            return;
        };
        let error = CudaConverter::new("convert", &device, 65, 32)
            .err()
            .expect("odd width is refused");
        assert!(matches!(
            error,
            CudaConverterError::OddDimensions {
                width: 65,
                height: 32
            }
        ));
        assert!(matches!(
            CudaConverter::new("convert", &device, 64, 33).err(),
            Some(CudaConverterError::OddDimensions { .. })
        ));
    }

    /// The reason this element exists: a BGRA capture cannot reach an NV12
    /// consumer without it. The compositor validates an incoming surface's
    /// format, its frames context, and its device, so accepting the converted
    /// frame is that whole contract met — what a BGRA surface fails.
    #[test]
    fn a_converted_surface_is_one_the_compositor_accepts() {
        use crate::color::Color;
        use crate::elements::{
            CudaVideoCompositor, VideoCompositorOptions, VideoFit, VideoLayer, VideoRect,
        };

        let Some((device, _cuda_lock)) = try_cuda_device() else {
            return;
        };
        let Some(mut converter) = converter(&device, 64, 64) else {
            return;
        };
        let Ok((_compositor, handle)) = CudaVideoCompositor::new(
            "compositor",
            &device,
            VideoCompositorOptions {
                width: 128,
                height: 128,
                frame_rate: ffmpeg::Rational::new(30, 1),
                background: Color::BLACK,
            },
        ) else {
            eprintln!("skipping: this machine cannot open a CUDA compositor");
            return;
        };
        let mut layer = handle
            .add_source(
                "capture",
                VideoLayer {
                    fit: VideoFit::Stretch,
                    ..VideoLayer::new(VideoRect::new(0, 0, 64, 64))
                },
            )
            .expect("add the converted source");

        let Some(bgra) = cuda_frame(&device, CudaFrameFormat::Bgra, 64, 64, 0) else {
            return;
        };
        // The same surface before conversion is what the compositor is built
        // to refuse, so this is a real gate rather than a formality.
        assert!(
            layer.sink.consume(clone_buffer(&bgra)).is_err(),
            "a BGRA surface must not compose"
        );

        let converted = capture(&mut converter);
        converter.consume(bgra).expect("convert");
        let nv12 = converted.lock().unwrap().remove(0);

        layer
            .sink
            .consume(nv12)
            .expect("the compositor accepts a converted surface");
    }

    #[test]
    fn eos_is_forwarded() {
        let Some((device, _cuda_lock)) = try_cuda_device() else {
            return;
        };
        let Some(mut converter) = converter(&device, 64, 32) else {
            return;
        };
        let received = capture(&mut converter);

        converter.consume(MediaBuffer::Eos).expect("eos");

        assert!(received.lock().unwrap()[0].is_eos());
    }
}
