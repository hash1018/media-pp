use std::sync::Arc;

use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_error, pp_info};

use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Element, ElementType, Sink, element_pp_log},
    elements::sink::renderer::SubmitError,
    platform::{cuda::CudaDevice, ffmpeg::AvBufferRef},
};

/// What [`CudaRenderer`] needs from an actual windowing/graphics
/// implementation — the CUDA sibling of
/// [`crate::elements::D3d11FrameRenderer`].
///
/// # Why this trait carries device pointers rather than an image handle
///
/// A CUDA frame cannot be handed to a graphics API the way a
/// `Pixel::D3D11` texture can, because there is no shared device object:
/// NVDEC's output lives in CUDA memory that Vulkan/D3D cannot name, and
/// CUDA cannot export it either. The working direction is the reverse —
/// the implementation allocates its *own* image in its graphics API with
/// exportable memory, imports that once into CUDA
/// (`cuImportExternalMemory`), and copies device-to-device into it on
/// every submit. So what this trait can usefully pass is exactly what a
/// `cuMemcpy2D` needs: source pointers and pitches.
///
/// That copy is what makes this path different from the D3D11 stack's
/// genuinely zero-copy submits. It never touches the CPU, but it is a
/// copy, and it is unavoidable as long as the decoder and the presenter
/// are different APIs.
pub trait CudaFrameRenderer: Send {
    /// Presents one NV12 frame.
    ///
    /// # Safety
    /// `y` and `uv` must be valid CUDA device pointers into an NV12 surface
    /// allocated on the same CUDA context this renderer imported its own
    /// image into, readable for `height` rows of `y_pitch` bytes and
    /// `height / 2` rows of `uv_pitch` bytes respectively. They are only
    /// valid for the duration of the call — the frame they belong to may
    /// return to the decoder's pool as soon as it returns.
    unsafe fn submit_nv12(
        &self,
        y: *const u8,
        y_pitch: usize,
        uv: *const u8,
        uv_pitch: usize,
        width: u32,
        height: u32,
    ) -> std::result::Result<(), SubmitError>;

    /// Call when the target window resizes.
    fn resize(&self, width: u32, height: u32) -> std::result::Result<(), SubmitError>;
}

/// Errors specific to `CudaRenderer`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum CudaRendererError {
    #[error("CudaRenderer only accepts CUDA frames, got {0:?}")]
    UnsupportedFormat(ffmpeg::format::Pixel),

    /// The frame carries no hardware frames context, so there is nothing to
    /// check its CUDA context against — it did not come from a hardware
    /// decoder at all.
    #[error("CUDA frame has no hardware frames context")]
    MissingFramesContext,

    /// The frame was allocated against a different CUDA context than this
    /// renderer imported its target image into. Its device pointers are
    /// meaningless here, so this is caught up front rather than left to
    /// fail somewhere inside the driver.
    #[error("CUDA frame belongs to a different CUDA context than this renderer")]
    ForeignContext,

    #[error("CudaRenderer only presents NV12 CUDA frames, got {0:?}")]
    UnsupportedSoftwareFormat(ffmpeg::format::Pixel),

    #[error("frame has no CUDA device pointer")]
    MissingPlane,

    #[error("submit failed: {0:?}")]
    Submit(SubmitError),

    #[error("resize failed: {0:?}")]
    Resize(SubmitError),
}

/// Presents GPU-resident CUDA frames — the terminal for a
/// [`crate::elements::CudaDecoder`] branch, and the CUDA sibling of
/// [`crate::elements::D3d11Renderer`].
///
/// Named for the frame type it consumes, not for the graphics API that ends
/// up drawing it: this element never touches Vulkan or D3D, and which one
/// the [`CudaFrameRenderer`] implementation uses is invisible here. That
/// also keeps the name honest on both platforms — a CUDA frame is a CUDA
/// frame whether the presenter behind it is Vulkan on Linux or D3D11 on
/// Windows.
///
/// Every frame is validated against the [`CudaDevice`] this renderer was
/// built from before its pointers are handed out, the same guard
/// `D3d11Renderer` applies with its captured `ID3D11Device`.
pub struct CudaRenderer {
    pp_log: PpLog,
    name: Arc<str>,
    inner: Box<dyn CudaFrameRenderer>,
    /// Captured once at construction from the shared [`CudaDevice`]. Only
    /// ever compared, never dereferenced — this element holds its own
    /// reference to the context (below) so the pointer cannot go stale
    /// while it is alive.
    device_ctx: *const ffi::AVHWDeviceContext,
    /// This element's own reference to the shared context, released in
    /// `Drop`. Keeps `device_ctx` a valid identity for the renderer's whole
    /// life even if the caller drops its `CudaDevice` first.
    _hw_device_ctx: Arc<AvBufferRef>,
}

// SAFETY: `device_ctx`/`hw_device_ctx` are only compared and refcounted,
// never dereferenced for mutation, and `inner` is `Send` by its own bound.
// `&mut self` on `consume` rules out concurrent access.
unsafe impl Send for CudaRenderer {}

impl CudaRenderer {
    /// `device` must be the same [`CudaDevice`] the upstream
    /// [`crate::elements::CudaDecoder`] was built from, and the one
    /// `renderer` imported its target image into. `renderer` is the
    /// caller's own [`CudaFrameRenderer`], already pointed at a real
    /// window by the time it gets here.
    pub fn new(
        name: impl Into<String>,
        device: &CudaDevice,
        renderer: Box<dyn CudaFrameRenderer>,
    ) -> Self {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::CudaRenderer, &name, None);
        let hw_device_ctx = device.retain();
        // SAFETY: `hw_device_ctx` owns a live `AVBufferRef` for a CUDA device
        // context, whose `data` is that `AVHWDeviceContext` by FFmpeg's own
        // definition. Only the pointer's identity is kept, to compare against an
        // incoming frame's; the reference held alongside it is what keeps that
        // identity from being reused by a different context.
        let device_ctx = unsafe { (*hw_device_ctx.as_ptr()).data as *const ffi::AVHWDeviceContext };
        pp_info!(pp_log: &pp_log, "created");
        Self {
            name,
            pp_log,
            inner: renderer,
            device_ctx,
            _hw_device_ctx: hw_device_ctx,
        }
    }

    /// Call when the target window resizes.
    pub fn resize(&self, width: u32, height: u32) -> crate::error::Result<()> {
        self.inner
            .resize(width, height)
            .inspect_err(|error| pp_error!(self, "resize failed: {error:?}"))
            .map_err(CudaRendererError::Resize)?;
        pp_info!(self, "resized: {width}x{height}");
        Ok(())
    }

    fn submit(&self, frame: &ffmpeg::frame::Video) -> Result<(), CudaRendererError> {
        // SAFETY: `frame` is a live `frame::Video` already confirmed to be
        // `Pixel::CUDA`, so `as_ptr` yields an initialized `AVFrame` and a hardware
        // frame's `hw_frames_ctx` is either null — rejected here — or an
        // `AVBufferRef` whose `data` is an `AVHWFramesContext`. Only pointer
        // identity is compared, never dereferenced past that.
        let (y, y_pitch, uv, uv_pitch, width, height) = unsafe {
            let ptr = frame.as_ptr();

            let frames_ref = (*ptr).hw_frames_ctx;
            if frames_ref.is_null() {
                return Err(CudaRendererError::MissingFramesContext);
            }
            let frames_ctx = (*frames_ref).data as *const ffi::AVHWFramesContext;
            if !std::ptr::eq((*frames_ctx).device_ctx, self.device_ctx) {
                return Err(CudaRendererError::ForeignContext);
            }

            // The layout *inside* the CUDA surface. NVDEC produces NV12 for
            // 8-bit streams and P010 for 10-bit; only the former is wired up
            // here, so a 10-bit stream is rejected rather than presented as
            // garbage.
            let sw_format = ffmpeg::format::Pixel::from((*frames_ctx).sw_format);
            if sw_format != ffmpeg::format::Pixel::NV12 {
                return Err(CudaRendererError::UnsupportedSoftwareFormat(sw_format));
            }

            let y = (*ptr).data[0];
            let uv = (*ptr).data[1];
            if y.is_null() || uv.is_null() {
                return Err(CudaRendererError::MissingPlane);
            }
            (
                y as *const u8,
                (*ptr).linesize[0] as usize,
                uv as *const u8,
                (*ptr).linesize[1] as usize,
                (*ptr).width as u32,
                (*ptr).height as u32,
            )
        };

        // SAFETY: validated just above — CUDA frame, this renderer's own
        // context, NV12 layout, both planes present.
        unsafe {
            self.inner
                .submit_nv12(y, y_pitch, uv, uv_pitch, width, height)
        }
        .map_err(CudaRendererError::Submit)
    }
}

impl Element for CudaRenderer {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::CudaRenderer
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Sink for CudaRenderer {
    fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
        let MediaBuffer::Video(frame) = buf else {
            return Ok(());
        };

        if frame.format() != ffmpeg::format::Pixel::CUDA {
            let format = frame.format();
            pp_error!(self, "unsupported pixel format: {format:?}");
            return Err(CudaRendererError::UnsupportedFormat(format).into());
        }
        self.submit(&frame)
            .inspect_err(|error| pp_error!(self, "submit failed: {error}"))?;
        Ok(())
    }

    fn control(&mut self, _msg: ControlMsg) -> crate::error::Result<()> {
        // Terminal, nothing to flush or forward — same reasoning as
        // `D3d11Renderer::control`.
        Ok(())
    }
}

impl Drop for CudaRenderer {
    fn drop(&mut self) {
        pp_info!(self, "dropped: releasing hw_device_ctx");
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    };

    use super::*;
    use crate::{
        element::Source,
        elements::CudaDecoder,
        pool::UnboundObjectPool,
        test_support::{try_cuda_device, try_test_video},
    };

    /// Records what reached the graphics side without touching a GPU, so the
    /// element's own validation can be tested on its own.
    #[derive(Default)]
    struct RecordingRenderer {
        submits: AtomicUsize,
        last: Mutex<Option<(usize, usize, u32, u32)>>,
    }

    impl CudaFrameRenderer for RecordingRenderer {
        unsafe fn submit_nv12(
            &self,
            y: *const u8,
            y_pitch: usize,
            uv: *const u8,
            uv_pitch: usize,
            width: u32,
            height: u32,
        ) -> std::result::Result<(), SubmitError> {
            assert!(!y.is_null() && !uv.is_null());
            self.submits.fetch_add(1, Ordering::Relaxed);
            *self.last.lock().unwrap() = Some((y_pitch, uv_pitch, width, height));
            Ok(())
        }

        fn resize(&self, _width: u32, _height: u32) -> std::result::Result<(), SubmitError> {
            Ok(())
        }
    }

    /// A CPU frame must be refused outright — its `data[0]` is host memory,
    /// and handing that to a `cuMemcpy2D` as a device pointer is exactly the
    /// kind of failure this guard exists to make impossible.
    #[test]
    fn a_cpu_frame_is_rejected_as_a_typed_error() {
        let Some((device, _cuda_lock)) = try_cuda_device() else {
            return;
        };
        let mut renderer = CudaRenderer::new(
            "cuda-renderer",
            &device,
            Box::new(RecordingRenderer::default()),
        );

        let frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, 64, 64);
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut pooled = pool.get();
        *pooled = frame;
        let error = renderer
            .consume(MediaBuffer::Video(Arc::new(pooled)))
            .expect_err("a CPU frame must not be presented");
        assert!(
            error.to_string().contains("only accepts CUDA frames"),
            "expected UnsupportedFormat, got {error}"
        );
    }

    /// The single-shared-context invariant. A frame decoded against one CUDA
    /// context carries device pointers meaningless to another, so mixing two
    /// must fail here rather than inside the driver.
    #[test]
    fn a_frame_from_another_cuda_context_is_rejected() {
        let Some((decode_device, _cuda_lock)) = try_cuda_device() else {
            return;
        };
        // Directly, not `try_cuda_device` again: the lock it returns is
        // already held for this test and does not nest.
        let render_device = CudaDevice::new().expect("a second CUDA device");
        let Some(path) = try_test_video() else {
            return;
        };

        let frames = decode_some_frames(&decode_device, &path);
        let Some(frame) = frames.into_iter().next() else {
            eprintln!("skipping: NVDEC produced no frames for the configured fixture");
            return;
        };

        let mut renderer = CudaRenderer::new(
            "cuda-renderer",
            &render_device,
            Box::new(RecordingRenderer::default()),
        );
        let error = renderer
            .consume(frame)
            .expect_err("a frame from a foreign CUDA context must not be presented");
        assert!(
            error.to_string().contains("different CUDA context"),
            "expected ForeignContext, got {error}"
        );
    }

    /// The happy path: a real NVDEC frame reaches the graphics side with the
    /// dimensions and pitches it actually has.
    #[test]
    fn a_decoded_frame_reaches_the_graphics_side() {
        let Some((device, _cuda_lock)) = try_cuda_device() else {
            return;
        };
        let Some(path) = try_test_video() else {
            return;
        };

        let frames = decode_some_frames(&device, &path);
        if frames.is_empty() {
            eprintln!("skipping: NVDEC produced no frames for the configured fixture");
            return;
        }
        let expected = frames.len();

        let inner = Arc::new(RecordingRenderer::default());
        let mut renderer = CudaRenderer::new("cuda-renderer", &device, Box::new(inner.clone()));
        for frame in frames {
            renderer.consume(frame).expect("submit failed");
        }

        assert_eq!(inner.submits.load(Ordering::Relaxed), expected);
        let (y_pitch, uv_pitch, width, height) =
            inner.last.lock().unwrap().expect("nothing was submitted");
        assert!(width > 0 && height > 0, "frame had no dimensions");
        assert!(
            y_pitch >= width as usize,
            "y pitch {y_pitch} is narrower than width {width}"
        );
        assert!(
            uv_pitch >= width as usize,
            "uv pitch {uv_pitch} is narrower than width {width}"
        );
    }

    impl CudaFrameRenderer for Arc<RecordingRenderer> {
        unsafe fn submit_nv12(
            &self,
            y: *const u8,
            y_pitch: usize,
            uv: *const u8,
            uv_pitch: usize,
            width: u32,
            height: u32,
        ) -> std::result::Result<(), SubmitError> {
            // SAFETY: forwarding the caller's own obligation unchanged — this trait
            // method is `unsafe` for exactly the pointer validity that
            // `RecordingRenderer::submit_nv12` requires.
            unsafe { RecordingRenderer::submit_nv12(self, y, y_pitch, uv, uv_pitch, width, height) }
        }

        fn resize(&self, width: u32, height: u32) -> std::result::Result<(), SubmitError> {
            RecordingRenderer::resize(self, width, height)
        }
    }

    /// Decodes a handful of real frames, so the renderer tests assert against
    /// genuine NVDEC output rather than a hand-built `AVFrame`.
    fn decode_some_frames(device: &CudaDevice, path: &str) -> Vec<MediaBuffer> {
        let mut input = ffmpeg::format::input(path).expect("failed to open the test video");
        let stream = input
            .streams()
            .best(ffmpeg::media::Type::Video)
            .expect("the test video has no video stream");
        let stream_index = stream.index();
        let params = stream.parameters();

        let mut decoder =
            CudaDecoder::new("cuda-decoder", params, device, 4).expect("failed to open NVDEC");
        let received = Arc::new(Mutex::new(Vec::new()));
        decoder.src_pads()[0].link(Box::new(Collector {
            received: received.clone(),
            pp_log: element_pp_log(ElementType::Other, "collect", None),
        }));

        for (stream, packet) in input.packets() {
            if stream.index() != stream_index {
                continue;
            }
            decoder
                .consume(MediaBuffer::Packet(Arc::new(packet)))
                .expect("decode failed");
            if received.lock().unwrap().len() >= 3 {
                break;
            }
        }
        std::mem::take(&mut *received.lock().unwrap())
    }

    struct Collector {
        pp_log: PpLog,
        received: Arc<Mutex<Vec<MediaBuffer>>>,
    }

    impl Element for Collector {
        fn name(&self) -> Arc<str> {
            "collect".into()
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

    impl Sink for Collector {
        fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
            if matches!(buf, MediaBuffer::Video(_)) {
                self.received.lock().unwrap().push(buf);
            }
            Ok(())
        }
        fn control(&mut self, _msg: ControlMsg) -> crate::error::Result<()> {
            Ok(())
        }
    }
}
