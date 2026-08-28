//! FFmpeg CUDA hardware-frame context helpers shared by CUDA producers.

use ffmpeg_next::ffi;
use thiserror::Error as ThisError;

use super::CudaFrameFormat;
use crate::platform::ffmpeg::AvBufferRef;

#[derive(Debug, ThisError)]
pub(crate) enum CudaFramesContextError {
    #[error("failed to allocate the CUDA frames context")]
    Alloc,

    #[error("failed to initialize the CUDA frames context (code {code}) for {width}x{height}")]
    Init { code: i32, width: u32, height: u32 },
}

/// Builds the dynamic FFmpeg CUDA frame pool shared by upload, conversion,
/// compositing, capture, and encoding elements.
pub(crate) unsafe fn create_hw_frames_ctx(
    hw_device_ctx: &AvBufferRef,
    format: CudaFrameFormat,
    width: u32,
    height: u32,
) -> Result<AvBufferRef, CudaFramesContextError> {
    // SAFETY: this function's own contract is a live device context, which
    // `hw_device_ctx` is. The allocation is wrapped as an `AvBufferRef` before
    // anything can fail, so every path below either returns it or drops it.
    // `data` is an `AVHWFramesContext` by FFmpeg's own definition, and the
    // fields written here are the ones `av_hwframe_ctx_init` reads.
    unsafe {
        let buf = AvBufferRef::from_raw(ffi::av_hwframe_ctx_alloc(hw_device_ctx.as_ptr()))
            .ok_or(CudaFramesContextError::Alloc)?;

        let frames_ctx = (*buf.as_ptr()).data as *mut ffi::AVHWFramesContext;
        (*frames_ctx).format = ffi::AVPixelFormat::AV_PIX_FMT_CUDA;
        (*frames_ctx).sw_format = format.sw_format();
        (*frames_ctx).width = width as i32;
        (*frames_ctx).height = height as i32;
        // CUDA uses libavutil's growable AVBufferPool. Unlike NVDEC's fixed,
        // capped decode surfaces, no producer here requires a fixed pool.
        (*frames_ctx).initial_pool_size = 0;

        let code = ffi::av_hwframe_ctx_init(buf.as_ptr());
        if code < 0 {
            return Err(CudaFramesContextError::Init {
                code,
                width,
                height,
            });
        }
        Ok(buf)
    }
}
