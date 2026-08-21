//! FFmpeg CUDA hardware-frame context helpers shared by CUDA producers.

use ffmpeg_next::ffi;
use thiserror::Error as ThisError;

use super::CudaFrameFormat;

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
    hw_device_ctx: *mut ffi::AVBufferRef,
    format: CudaFrameFormat,
    width: u32,
    height: u32,
) -> Result<*mut ffi::AVBufferRef, CudaFramesContextError> {
    unsafe {
        let buf = ffi::av_hwframe_ctx_alloc(hw_device_ctx);
        if buf.is_null() {
            return Err(CudaFramesContextError::Alloc);
        }

        let frames_ctx = (*buf).data as *mut ffi::AVHWFramesContext;
        (*frames_ctx).format = ffi::AVPixelFormat::AV_PIX_FMT_CUDA;
        (*frames_ctx).sw_format = format.sw_format();
        (*frames_ctx).width = width as i32;
        (*frames_ctx).height = height as i32;
        // CUDA uses libavutil's growable AVBufferPool. Unlike NVDEC's fixed,
        // capped decode surfaces, no producer here requires a fixed pool.
        (*frames_ctx).initial_pool_size = 0;

        let code = ffi::av_hwframe_ctx_init(buf);
        if code < 0 {
            free_buffer(buf);
            return Err(CudaFramesContextError::Init {
                code,
                width,
                height,
            });
        }
        Ok(buf)
    }
}

pub(crate) unsafe fn free_buffer(mut buf: *mut ffi::AVBufferRef) {
    unsafe { ffi::av_buffer_unref(&mut buf) };
}
