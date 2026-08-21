use std::sync::Arc;

use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_error, pp_info};

use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    error::Result,
    pad::SrcPad,
    platform::cuda::{
        CudaDevice, CudaFrameFormat,
        frame::{CudaFramesContextError, create_hw_frames_ctx},
    },
    platform::ffmpeg::AvBufferRef,
    pool::UnboundObjectPool,
};

/// Errors specific to `CudaUpload`. Converts into the crate-wide `Error` via
/// `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum CudaUploadError {
    #[error("this CudaUpload uploads {expected:?} frames, got {actual:?}")]
    UnsupportedFormat {
        expected: ffmpeg::format::Pixel,
        actual: ffmpeg::format::Pixel,
    },

    #[error("CudaUpload only accepts Video buffers, got a {0}")]
    UnsupportedBuffer(&'static str),

    #[error(
        "frame is {actual_width}x{actual_height}, but this CudaUpload was built for \
         {expected_width}x{expected_height}"
    )]
    DimensionMismatch {
        actual_width: u32,
        actual_height: u32,
        expected_width: u32,
        expected_height: u32,
    },

    #[error("failed to allocate the CUDA frames context")]
    HwFramesAlloc,

    #[error("failed to initialize the CUDA frames context (code {0}) for {1}x{2}")]
    HwFramesInit(i32, u32, u32),

    #[error("failed to take a frame from the CUDA pool (code {0})")]
    HwFrameGet(i32),

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
/// CUDA sibling of [`crate::elements::D3d11Upload`], and what lets a CPU
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
    _hw_device_ctx: Arc<AvBufferRef>,
    /// The pool uploaded frames are allocated from.
    hw_frames_ctx: AvBufferRef,
    format: CudaFrameFormat,
    width: u32,
    height: u32,
    pad: SrcPad,
    /// Reuses only the small CPU-side `AVFrame` wrapper; the CUDA surface
    /// itself comes from `hw_frames_ctx`'s own pool. Same split as
    /// `D3d11Upload`.
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
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
        width: u32,
        height: u32,
    ) -> std::result::Result<Self, CudaUploadError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::CudaUpload, &name, None);

        let hw_device_ctx = device.retain();
        let hw_frames_ctx = unsafe { create_hw_frames_ctx(&hw_device_ctx, format, width, height) }
            .map_err(CudaUploadError::from)?;

        let pad = SrcPad::new(format!("{name}_src"));
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        pp_info!(
            pp_log: &pp_log,
            "opened: {width}x{height} {:?} -> CUDA",
            format.pixel()
        );
        Ok(Self {
            name,
            pp_log,
            _hw_device_ctx: hw_device_ctx,
            hw_frames_ctx,
            format,
            width,
            height,
            pad,
            pool,
        })
    }

    fn upload(&mut self, source: &ffmpeg::frame::Video) -> Result<()> {
        if source.format() != self.format.pixel() {
            pp_error!(self, "unsupported pixel format: {:?}", source.format());
            return Err(CudaUploadError::UnsupportedFormat {
                expected: self.format.pixel(),
                actual: source.format(),
            }
            .into());
        }
        if source.width() != self.width || source.height() != self.height {
            let error = CudaUploadError::DimensionMismatch {
                actual_width: source.width(),
                actual_height: source.height(),
                expected_width: self.width,
                expected_height: self.height,
            };
            pp_error!(self, "{error}");
            return Err(error.into());
        }

        let mut destination = self.pool.get();
        unsafe {
            let dst = destination.as_mut_ptr();
            // The pooled wrapper may still reference the previous frame's
            // surface; releasing it here is what returns that surface to the
            // frames pool rather than leaking it for the element's lifetime.
            ffi::av_frame_unref(dst);

            let code = ffi::av_hwframe_get_buffer(self.hw_frames_ctx.as_ptr(), dst, 0);
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

        self.pad.push(MediaBuffer::Video(Arc::new(destination)))
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
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Video(frame) => self.upload(&frame),
            MediaBuffer::Eos => self.pad.push(MediaBuffer::Eos),
            other => Err(CudaUploadError::UnsupportedBuffer(other.kind()).into()),
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        // Nothing local to react to — a pure per-frame CPU->GPU transfer,
        // same reasoning as `D3d11Upload::control`.
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
    use std::sync::{Arc, Mutex};

    use super::*;
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
    fn new_upload(width: u32, height: u32) -> Option<UploadFixture> {
        let (device, cuda_lock) = try_cuda_device()?;
        let mut upload =
            CudaUpload::new("upload", &device, CudaFrameFormat::Nv12, width, height).ok()?;
        let received = Arc::new(Mutex::new(Vec::new()));
        upload.src_pads()[0].link(Box::new(CapturingSink {
            received: received.clone(),
            pp_log: element_pp_log(ElementType::Other, "capture", None),
        }));
        Some((upload, received, cuda_lock))
    }

    /// The contract: what comes out is GPU-resident and keeps its timestamp.
    #[test]
    fn uploads_nv12_into_cuda_frames_and_preserves_pts() {
        let Some((mut upload, received, _cuda_lock)) = new_upload(64, 64) else {
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
        let Ok(mut upload) =
            CudaUpload::new("upload", &device, CudaFrameFormat::Bgra, width, height)
        else {
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

    /// A mismatched frame must be refused rather than uploaded into a
    /// differently-sized surface.
    #[test]
    fn wrong_size_and_format_are_typed_errors() {
        let Some((mut upload, _received, _cuda_lock)) = new_upload(64, 64) else {
            return;
        };
        let error = upload
            .consume(nv12_frame(32, 32, 0))
            .expect_err("a differently-sized frame must not upload");
        assert!(
            error.to_string().contains("32x32"),
            "expected DimensionMismatch, got {error}"
        );

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
