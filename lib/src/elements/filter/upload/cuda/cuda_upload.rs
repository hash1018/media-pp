use std::sync::Arc;

use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_error, pp_info};

use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    error::Result,
    frame_size::ForSize,
    pad::SrcPad,
    platform::cuda::{
        CudaDevice, CudaFrameFormat,
        frame::{CudaFramesContextError, create_hw_frames_ctx},
    },
    platform::ffmpeg::AvBufferRef,
    pool::{UnboundObjectPool, UnboundObjectPoolRef},
    repeat::{PerFrameTransform, RepeatedOutput},
};

/// Errors specific to `CudaUpload`. Converts into the crate-wide `Error` via
/// `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum CudaUploadError {
    /// FFmpeg could not take a second reference to the upload already in
    /// hand, which is how an unchanged CPU picture is answered.
    #[error("failed to reference the previous upload (code {0})")]
    FrameRef(i32),

    /// The CPU frame format differs from the upload surface format.
    #[error("this CudaUpload uploads {expected:?} frames, got {actual:?}")]
    UnsupportedFormat {
        /// CPU pixel format configured for this uploader.
        expected: ffmpeg::format::Pixel,
        /// Pixel format carried by the input frame.
        actual: ffmpeg::format::Pixel,
    },
    /// The sink received a buffer other than decoded video or end-of-stream.

    #[error("CudaUpload only accepts Video buffers, got a {0}")]
    UnsupportedBuffer(&'static str),
    /// FFmpeg could not allocate the CUDA hardware frames context.

    #[error("failed to allocate the CUDA frames context")]
    HwFramesAlloc,
    /// FFmpeg could not initialize the fixed-size CUDA frame pool.

    #[error("failed to initialize the CUDA frames context (code {0}) for {1}x{2}")]
    HwFramesInit(i32, u32, u32),
    /// FFmpeg could not acquire a surface from the CUDA frame pool.

    #[error("failed to take a frame from the CUDA pool (code {0})")]
    HwFrameGet(i32),
    /// FFmpeg failed to transfer CPU pixels into the CUDA surface.

    #[error("CPU to CUDA transfer failed (code {0})")]
    Transfer(i32),
}

impl From<CudaFramesContextError> for CudaUploadError {
    fn from(error: CudaFramesContextError) -> Self {
        match error {
            CudaFramesContextError::Alloc => Self::HwFramesAlloc,
            CudaFramesContextError::Init {
                code,
                width,
                height,
            } => Self::HwFramesInit(code, width, height),
        }
    }
}

/// Uploads CPU-resident `Video` frames into CUDA-resident ones — the
/// CUDA sibling of `D3d11Upload`, and what lets a CPU
/// source reach [`crate::elements::CudaEncoder`] or
/// [`crate::elements::CudaRenderer`] at all.
/// [`crate::elements::CudaDownload`] is the mirror of this element.
///
/// A `Filter`: receives via `Sink`, pushes the uploaded frame into its own
/// single src pad. PTS, duration, and color metadata are carried across with
/// `av_frame_copy_props`, so this creates no new timeline.
///
/// # One format, chosen up front
///
/// `format` fixes what every surface this allocates holds, and every frame
/// `consume` receives must already be in the matching CPU layout — nothing
/// here converts. `Bgra` is the format a screen capture already produces and
/// NVENC ingests directly; `Nv12` is what a decoder produces and
/// [`crate::elements::CudaRenderer`] presents. See [`CudaFrameFormat`] on
/// why the choice has to be made here rather than converted later: no
/// element on the CUDA path can turn one into the other. Put a
/// [`crate::elements::SwScaler`] in front if the source produces neither.
///
/// # Why the frames context is built by hand here
///
/// Unlike the D3D11VA case (see `d3d11va_decoder`'s notes on the memory
/// corruption that came of hand-mirroring `AVD3D11VAFramesContext`), nothing
/// CUDA-specific is touched: `format`, `sw_format`, `width`, `height`, and
/// `initial_pool_size` are all plain fields of the type-agnostic
/// `AVHWFramesContext` that `ffmpeg-sys-next` binds directly, and FFmpeg's
/// own code fills in everything else during `av_hwframe_ctx_init`.
pub struct CudaUpload {
    pp_log: PpLog,
    name: Arc<str>,
    /// This element's own reference to the shared context, released in `Drop`.
    hw_device_ctx: Arc<AvBufferRef>,
    /// The pool uploaded frames are allocated from. A frames context's
    /// dimensions are fixed at `av_hwframe_ctx_init`, so it is made for the
    /// first frame's size and made again if a source changes resolution
    /// mid-stream — see `ForSize`.
    hw_frames_ctx: ForSize<AvBufferRef>,
    format: CudaFrameFormat,
    pad: SrcPad,
    /// Reuses only the small CPU-side `AVFrame` wrapper; the CUDA surface
    /// itself comes from `hw_frames_ctx`'s own pool. Same split as
    /// `D3d11Upload`.
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
    /// The last upload and the CPU picture it came from, so a producer that
    /// re-emits an unchanged picture is answered with the surface already on
    /// the GPU instead of another transfer across PCIe — see
    /// [`RepeatedOutput`].
    repeated: RepeatedOutput,
}

// SAFETY: both buffers are heap-allocated FFmpeg buffers with no thread
// affinity of their own, and `&mut self` on every method that touches them
// rules out concurrent access. Same reasoning as `CudaDecoder`.
unsafe impl Send for CudaUpload {}

impl CudaUpload {
    /// `device` must be the same [`CudaDevice`] every other CUDA element in
    /// this pipeline was built from. This element takes its own FFmpeg
    /// reference, so `device` itself need not outlive the call.
    pub fn new(
        name: impl Into<String>,
        device: &CudaDevice,
        format: CudaFrameFormat,
    ) -> std::result::Result<Self, CudaUploadError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::CudaUpload, &name, None);

        let hw_device_ctx = device.retain();

        let pad = SrcPad::with_contract(
            format!("{name}_src"),
            OutputContract::Fixed(
                PortContract::frame(MediaKind::VideoFrame, MemoryDomain::Cuda)
                    .with_layouts(format.layouts()),
            ),
        );
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        pp_info!(pp_log: &pp_log, "opened: {:?} -> CUDA", format.pixel());
        Ok(Self {
            name,
            pp_log,
            hw_device_ctx,
            hw_frames_ctx: ForSize::new(),
            format,
            pad,
            pool,
            repeated: RepeatedOutput::new(),
        })
    }

    fn upload(
        &mut self,
        source: &ffmpeg::frame::Video,
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        if source.format() != self.format.pixel() {
            pp_error!(self, "unsupported pixel format: {:?}", source.format());
            return Err(CudaUploadError::UnsupportedFormat {
                expected: self.format.pixel(),
                actual: source.format(),
            }
            .into());
        }
        let (device, format, pp_log) = (&self.hw_device_ctx, self.format, &self.pp_log);
        let frames_ctx = self
            .hw_frames_ctx
            .try_get(source.width(), source.height(), |width, height| {
                pp_info!(pp_log: pp_log, "allocating a {width}x{height} pool");
                // SAFETY: `create_hw_frames_ctx`'s contract is a live device
                // context, which is what the owned `AvBufferRef` is.
                unsafe { create_hw_frames_ctx(device, format, width, height) }
                    .map_err(CudaUploadError::from)
            })
            .inspect_err(|error| pp_error!(self, "{error}"))?
            .as_ptr();

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

            let code = ffi::av_hwframe_get_buffer(frames_ctx, dst, 0);
            if code < 0 {
                pp_error!(self, "av_hwframe_get_buffer failed: {code}");
                return Err(CudaUploadError::HwFrameGet(code).into());
            }
            let code = ffi::av_hwframe_transfer_data(dst, source.as_ptr(), 0);
            if code < 0 {
                pp_error!(self, "av_hwframe_transfer_data failed: {code}");
                return Err(CudaUploadError::Transfer(code).into());
            }
            // `av_hwframe_transfer_data` moves pixels only — PTS, duration,
            // and color metadata are part of the buffer contract and would
            // otherwise be dropped here.
            ffi::av_frame_copy_props(dst, source.as_ptr());
        }
        Ok(destination)
    }
}

impl PerFrameTransform for CudaUpload {
    fn repeated(&mut self) -> &mut RepeatedOutput {
        &mut self.repeated
    }

    fn frame_ref_failed(&self, code: i32) -> crate::error::Error {
        pp_error!(self, "av_frame_ref failed: {code}");
        CudaUploadError::FrameRef(code).into()
    }

    fn produce(
        &mut self,
        source: &Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        self.upload(source)
    }
}

impl Element for CudaUpload {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::CudaUpload
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for CudaUpload {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for CudaUpload {
    /// CPU-readable planes: uploading is what this does, so a frame already in device memory has no work here.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::System)
                .with_layouts(self.format.layouts()),
        )
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            // The same CPU buffer as last time is the same pixels as last
            // time, and uploading them again produces the surface already on
            // the GPU — see [`PerFrameTransform`].
            MediaBuffer::Video(frame) => {
                let uploaded = self.transform(&frame)?;
                self.pad.push(MediaBuffer::Video(uploaded))
            }
            MediaBuffer::Eos => self.pad.push(MediaBuffer::Eos),
            other => Err(CudaUploadError::UnsupportedBuffer(other.kind()).into()),
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        // Nothing local to react to beyond the cached upload — a pure
        // per-frame CPU->GPU transfer, same reasoning as
        // `D3d11Upload::control`.
        if matches!(msg, ControlMsg::Flush | ControlMsg::Stop) {
            self.repeated.clear();
        }
        self.pad.control(msg)
    }
}

impl Drop for CudaUpload {
    fn drop(&mut self) {
        pp_info!(self, "dropped: releasing hw contexts");
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::CapturingSink;
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::test_support::try_cuda_device;

    fn nv12_frame(width: u32, height: u32, pts: i64) -> MediaBuffer {
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, width, height);
        frame.set_pts(Some(pts));
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut pooled = pool.get();
        *pooled = frame;
        MediaBuffer::Video(Arc::new(pooled))
    }

    type UploadFixture = (
        CudaUpload,
        Arc<Mutex<Vec<MediaBuffer>>>,
        std::sync::MutexGuard<'static, ()>,
    );

    /// Carries the CUDA lock out with the element: dropping it here would
    /// unlock before the test has even started running (see
    /// [`try_cuda_device`]).
    fn new_upload() -> Option<UploadFixture> {
        let (device, cuda_lock) = try_cuda_device()?;
        let mut upload = CudaUpload::new("upload", &device, CudaFrameFormat::Nv12).ok()?;
        let received = Arc::new(Mutex::new(Vec::new()));
        upload.src_pads()[0].link(Box::new(CapturingSink {
            received: received.clone(),
            pp_log: element_pp_log(ElementType::Other, "capture", None),
        }));
        Some((upload, received, cuda_lock))
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
            assert!(ffi::av_frame_ref(slot.as_mut_ptr(), source.as_ptr()) >= 0);
        }
        slot.set_pts(Some(pts));
        MediaBuffer::Video(Arc::new(slot))
    }

    /// A capture of a still screen re-emits the picture it already has, and
    /// uploading it again would produce the surface already on the GPU. The
    /// repeat carries its own timestamp; only the surface is shared.
    #[test]
    fn a_repeated_input_is_uploaded_once() {
        let Some((mut upload, received, _cuda_lock)) = new_upload() else {
            return;
        };
        let source = nv12_frame(64, 64, 100);
        let repeat = repeat_of(&source, 200);
        upload.consume(source).expect("upload the first frame");
        upload.consume(repeat).expect("upload the repeat");

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
            "an unchanged picture was uploaded into a second surface"
        );
        assert_eq!(
            repeated.pts(),
            Some(200),
            "a repeat carries this frame's timestamp, not the one it points at"
        );
    }

    /// The contract: what comes out is GPU-resident and keeps its timestamp.
    #[test]
    fn uploads_nv12_into_cuda_frames_and_preserves_pts() {
        let Some((mut upload, received, _cuda_lock)) = new_upload() else {
            return;
        };
        upload.consume(nv12_frame(64, 64, 1234)).expect("upload");
        upload.consume(MediaBuffer::Eos).expect("eos");

        let received = received.lock().unwrap();
        let MediaBuffer::Video(frame) = &received[0] else {
            panic!("expected a Video buffer, got {}", received[0].kind());
        };
        assert_eq!(frame.format(), ffmpeg::format::Pixel::CUDA);
        assert_eq!(frame.pts(), Some(1234), "upload dropped the pts");
        assert!(
            received.last().is_some_and(MediaBuffer::is_eos),
            "Eos was not forwarded"
        );
    }

    /// The BGRA path a screen capture uses: what a capture source already
    /// produces becomes a CUDA surface that still holds BGRA, since nothing
    /// downstream on the CUDA path can convert RGB to YUV.
    #[test]
    fn uploads_bgra_into_bgra_surfaces() {
        let Some((device, _cuda_lock)) = try_cuda_device() else {
            return;
        };
        let (width, height) = (64u32, 64u32);
        let Ok(mut upload) = CudaUpload::new("upload", &device, CudaFrameFormat::Bgra) else {
            eprintln!("skipping: this machine has no usable CUDA frames context");
            return;
        };
        let received = Arc::new(Mutex::new(Vec::new()));
        upload.src_pads()[0].link(Box::new(CapturingSink {
            received: received.clone(),
            pp_log: element_pp_log(ElementType::Other, "capture", None),
        }));

        let mut bgra = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, width, height);
        bgra.set_pts(Some(7));
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut pooled = pool.get();
        *pooled = bgra;
        upload
            .consume(MediaBuffer::Video(Arc::new(pooled)))
            .expect("bgra upload");

        let received = received.lock().unwrap();
        let MediaBuffer::Video(frame) = &received[0] else {
            panic!("expected a Video buffer, got {}", received[0].kind());
        };
        assert_eq!(frame.format(), ffmpeg::format::Pixel::CUDA);
        assert_eq!(frame.pts(), Some(7));
        // SAFETY: the assertions above have established this is a live CUDA frame,
        // so its `hw_frames_ctx` is set and its `data` is an `AVHWFramesContext`.
        let sw_format = unsafe {
            let frames_ref = (*frame.as_ptr()).hw_frames_ctx;
            let frames_ctx = (*frames_ref).data as *const ffi::AVHWFramesContext;
            ffmpeg::format::Pixel::from((*frames_ctx).sw_format)
        };
        assert_eq!(
            sw_format,
            ffmpeg::format::Pixel::BGRA,
            "the surface does not hold BGRA"
        );
    }

    /// A source that changes resolution mid-stream was a per-frame error
    /// here while the surface pool was allocated before any frame arrived.
    #[test]
    fn a_source_that_changes_resolution_is_followed() {
        let Some((mut upload, received, _cuda_lock)) = new_upload() else {
            return;
        };

        upload
            .consume(nv12_frame(64, 64, 0))
            .expect("the first size");
        upload
            .consume(nv12_frame(32, 32, 1))
            .expect("and the next one");

        let received = received.lock().unwrap();
        let sizes: Vec<(u32, u32)> = received
            .iter()
            .map(|buffer| {
                let MediaBuffer::Video(frame) = buffer else {
                    panic!("expected a Video buffer");
                };
                (frame.width(), frame.height())
            })
            .collect();
        assert_eq!(
            sizes,
            [(64, 64), (32, 32)],
            "each surface is made for the frame it holds"
        );
    }

    /// A frame in a layout this uploader was not built for must be refused
    /// rather than reinterpreted.
    #[test]
    fn a_wrong_format_is_a_typed_error() {
        let Some((mut upload, _received, _cuda_lock)) = new_upload() else {
            return;
        };
        let mut rgb = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::RGB24, 64, 64);
        rgb.set_pts(Some(0));
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut pooled = pool.get();
        *pooled = rgb;
        let error = upload
            .consume(MediaBuffer::Video(Arc::new(pooled)))
            .expect_err("a frame in another layout must not upload");
        assert!(
            error.to_string().contains("uploads NV12 frames, got RGB24"),
            "expected UnsupportedFormat, got {error}"
        );
    }
}
