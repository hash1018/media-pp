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

/// Which CUDA surface a frame's pixels live in.
///
/// Not the frame: a producer with nothing new to show re-emits a fresh
/// `AVFrame` referencing the same surface every tick — a screen capture of a
/// still desktop does exactly that — so comparing frames, or the `Arc`s
/// around them, says "changed" every time while the pixels have not moved.
/// The plane pointers do not.
///
/// Only sound as an identity while the frame that produced it is still
/// referenced: a surface that returns to its pool can be handed out again at
/// the same address. Every caller here holds that reference for as long as it
/// holds the identity.
pub(crate) fn surface_id(frame: &ffmpeg_next::frame::Video) -> (usize, usize) {
    // SAFETY: `as_ptr` is a live `AVFrame`. Only the values of the first two
    // plane pointers are read; nothing dereferences them, which for CUDA
    // device memory would not be valid from the host anyway.
    unsafe {
        let ptr = frame.as_ptr();
        ((*ptr).data[0] as usize, (*ptr).data[1] as usize)
    }
}
