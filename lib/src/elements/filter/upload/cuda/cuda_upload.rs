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
    platform::cuda::CudaDevice,
    pool::UnboundObjectPool,
};

/// Errors specific to `CudaUpload`. Converts into the crate-wide `Error` via
/// `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum CudaUploadError {
    #[error("CudaUpload only uploads NV12 frames, got {0:?}")]
    UnsupportedFormat(ffmpeg::format::Pixel),

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

/// Uploads CPU-resident NV12 `Video` frames into CUDA-resident ones — the
/// CUDA sibling of [`crate::elements::D3d11Upload`], and what lets a CPU
/// source reach [`crate::elements::CudaEncoder`] or
/// [`crate::elements::CudaRenderer`] at all.
/// [`crate::elements::CudaDownload`] is the mirror of this element.
///
/// A `Filter`: receives via `Sink`, pushes the uploaded frame into its own
/// single src pad. PTS, duration, and color metadata are carried across with
/// `av_frame_copy_props`, so this creates no new timeline.
///
/// # NV12 only
///
/// Everything on this crate's CUDA path speaks NV12 (NVDEC produces it,
/// [`crate::elements::CudaEncoder`] consumes it), so an upload that accepted
/// other layouts would only be able to hand them to elements that reject
/// them. Put a [`crate::elements::SwScaler`] in front to convert, exactly as
/// the D3D11 path does.
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
    hw_device_ctx: *mut ffi::AVBufferRef,
    /// The pool uploaded frames are allocated from.
    hw_frames_ctx: *mut ffi::AVBufferRef,
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
        width: u32,
        height: u32,
    ) -> std::result::Result<Self, CudaUploadError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::CudaUpload, &name, None);

        let hw_device_ctx = unsafe { ffi::av_buffer_ref(device.as_ptr()) };
        let hw_frames_ctx = match unsafe { create_hw_frames_ctx(hw_device_ctx, width, height) } {
            Ok(ctx) => ctx,
            Err(error) => {
                unsafe { free_buffer(hw_device_ctx) };
                return Err(error);
            }
        };

        let pad = SrcPad::new(format!("{name}_src"));
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        pp_info!(pp_log: &pp_log, "opened: {width}x{height} NV12 -> CUDA");
        Ok(Self {
            name,
            pp_log,
            hw_device_ctx,
            hw_frames_ctx,
            width,
            height,
            pad,
            pool,
        })
    }

    fn upload(&mut self, source: &ffmpeg::frame::Video) -> Result<()> {
        if source.format() != ffmpeg::format::Pixel::NV12 {
            pp_error!(self, "unsupported pixel format: {:?}", source.format());
            return Err(CudaUploadError::UnsupportedFormat(source.format()).into());
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

            let code = ffi::av_hwframe_get_buffer(self.hw_frames_ctx, dst, 0);
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
        unsafe {
            free_buffer(self.hw_frames_ctx);
            free_buffer(self.hw_device_ctx);
        }
    }
}

/// Builds the CUDA/NV12 pool uploaded frames are allocated from. Shared with
/// [`crate::elements::CudaEncoder`], which needs the identically-shaped
/// context for `AVCodecContext.hw_frames_ctx`.
pub(crate) unsafe fn create_hw_frames_ctx(
    hw_device_ctx: *mut ffi::AVBufferRef,
    width: u32,
    height: u32,
) -> std::result::Result<*mut ffi::AVBufferRef, CudaUploadError> {
    unsafe {
        let buf = ffi::av_hwframe_ctx_alloc(hw_device_ctx);
        if buf.is_null() {
            return Err(CudaUploadError::HwFramesAlloc);
        }

        let frames_ctx = (*buf).data as *mut ffi::AVHWFramesContext;
        (*frames_ctx).format = ffi::AVPixelFormat::AV_PIX_FMT_CUDA;
        (*frames_ctx).sw_format = ffi::AVPixelFormat::AV_PIX_FMT_NV12;
        (*frames_ctx).width = width as i32;
        (*frames_ctx).height = height as i32;
        // Left at libavutil's dynamic `AVBufferPool`, same choice as
        // `D3d11NvencEncoder`'s own frames context: a fixed pool would have
        // to be sized against the deepest downstream buffering, and unlike
        // NVDEC's decode surfaces (capped at 32 — see `CudaDecoder::new`)
        // nothing here requires one.
        (*frames_ctx).initial_pool_size = 0;

        let code = ffi::av_hwframe_ctx_init(buf);
        if code < 0 {
            free_buffer(buf);
            return Err(CudaUploadError::HwFramesInit(code, width, height));
        }
        Ok(buf)
    }
}

pub(crate) unsafe fn free_buffer(mut buf: *mut ffi::AVBufferRef) {
    unsafe { ffi::av_buffer_unref(&mut buf) };
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    fn try_cuda_device() -> Option<CudaDevice> {
        match CudaDevice::new() {
            Ok(device) => Some(device),
            Err(error) => {
                eprintln!("skipping: no usable CUDA device on this machine ({error})");
                None
            }
        }
    }

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

    fn new_upload(width: u32, height: u32) -> Option<(CudaUpload, Arc<Mutex<Vec<MediaBuffer>>>)> {
        let device = try_cuda_device()?;
        let mut upload = CudaUpload::new("upload", &device, width, height).ok()?;
        let received = Arc::new(Mutex::new(Vec::new()));
        upload.src_pads()[0].link(Box::new(CapturingSink {
            received: received.clone(),
            pp_log: element_pp_log(ElementType::Other, "capture", None),
        }));
        Some((upload, received))
    }

    /// The contract: what comes out is GPU-resident and keeps its timestamp.
    #[test]
    fn uploads_nv12_into_cuda_frames_and_preserves_pts() {
        let Some((mut upload, received)) = new_upload(64, 64) else {
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

    /// A mismatched frame must be refused rather than uploaded into a
    /// differently-sized surface.
    #[test]
    fn wrong_size_and_format_are_typed_errors() {
        let Some((mut upload, _received)) = new_upload(64, 64) else {
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
            .expect_err("a non-NV12 frame must not upload");
        assert!(
            error.to_string().contains("only uploads NV12"),
            "expected UnsupportedFormat, got {error}"
        );
    }
}
