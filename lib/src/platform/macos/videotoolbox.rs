//! [`VideoToolboxDevice`], and FFmpeg's VideoToolbox frames: making a pool
//! of them, and reading what one holds.
//!
//! A VideoToolbox frame is an `AVFrame` in `AV_PIX_FMT_VIDEOTOOLBOX` whose
//! `data[3]` is a `CVPixelBuffer`, backed by an IOSurface. Unlike a CUDA
//! allocation or a Vulkan image, a pixel buffer belongs to no context: any
//! VideoToolbox session, and later any Metal device, in the process can
//! read one. So nothing here refuses a frame for having been made on
//! another device — a download reads it through the frames context it came
//! with, whichever that is.

use std::sync::Arc;

use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;

use crate::{
    platform::ffmpeg::AvBufferRef,
    pool::{UnboundObjectPool, UnboundObjectPoolRef},
};

/// Why a [`VideoToolboxDevice`] could not be opened.
#[derive(Debug, ThisError)]
pub enum VideoToolboxDeviceError {
    /// FFmpeg could not open a VideoToolbox context: an FFmpeg built
    /// without VideoToolbox, most likely.
    #[error("FFmpeg could not open VideoToolbox: {0}")]
    Open(ffmpeg::Error),

    /// FFmpeg reported success without returning a context reference.
    #[error("FFmpeg opened VideoToolbox without returning a device context")]
    MissingContext,
}

/// The one VideoToolbox context a pipeline's VideoToolbox elements share —
/// the macOS counterpart of `CudaDevice` and `VulkanDevice`.
///
/// It is FFmpeg's `AV_HWDEVICE_TYPE_VIDEOTOOLBOX` context, which a decoder
/// decodes on and an upload makes its frames on. There is no GPU to choose
/// and no state to keep apart: the system's VideoToolbox serves every
/// session, and the frames made on one context are readable on any other,
/// which is why [`crate::elements::VideoToolboxDownload`] takes none.
/// Sharing one is what keeps a pipeline's shape the same as on the other
/// backends, where it matters.
///
/// Cloning is cheap and shares the one context, which stays open until the
/// last clone — and every frame made on it — is gone.
#[derive(Clone)]
pub struct VideoToolboxDevice {
    ctx: Arc<AvBufferRef>,
}

impl std::fmt::Debug for VideoToolboxDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VideoToolboxDevice").finish_non_exhaustive()
    }
}

impl VideoToolboxDevice {
    /// Opens a VideoToolbox context.
    pub fn new() -> Result<Self, VideoToolboxDeviceError> {
        let mut ctx: *mut ffi::AVBufferRef = std::ptr::null_mut();
        // SAFETY: `ctx` is a live local FFmpeg writes the allocated context
        // into; the two nulls are the documented "default device, no options"
        // form, and VideoToolbox takes no flags.
        let result = unsafe {
            ffi::av_hwdevice_ctx_create(
                &mut ctx,
                ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VIDEOTOOLBOX,
                std::ptr::null(),
                std::ptr::null_mut(),
                0,
            )
        };
        if result < 0 {
            return Err(VideoToolboxDeviceError::Open(ffmpeg::Error::from(result)));
        }
        // SAFETY: `av_hwdevice_ctx_create` left `ctx` owning one reference and
        // nothing else has taken it; a failure left it null and returned above.
        let ctx =
            unsafe { AvBufferRef::from_raw(ctx) }.ok_or(VideoToolboxDeviceError::MissingContext)?;
        Ok(Self { ctx: Arc::new(ctx) })
    }

    /// Another reference to FFmpeg's context, for an element to keep — and
    /// to make its frames with.
    pub(crate) fn retain(&self) -> Arc<AvBufferRef> {
        Arc::clone(&self.ctx)
    }
}

/// What VideoToolbox frames hold, where an element has to be told — as
/// [`crate::elements::EncodeInput::VideoToolbox`] and
/// [`crate::elements::VideoToolboxEncoderOptions::format`] are: the two
/// layouts everything that makes VideoToolbox frames here puts out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoToolboxFrameFormat {
    /// NV12: what `VideoToolboxDecoder` decodes to, and what the media
    /// engine's encoders take.
    Nv12,
    /// 8-bit BGRA, with alpha: what `VideoToolboxUpload` makes of a BGRA
    /// frame, and what `ScreenCaptureKitSource` captures.
    Bgra,
}

impl VideoToolboxFrameFormat {
    /// The layout as FFmpeg names it, where a frame holding it is `Some`.
    pub(crate) fn of(format: ffmpeg::format::Pixel) -> Option<Self> {
        match format {
            ffmpeg::format::Pixel::NV12 => Some(Self::Nv12),
            ffmpeg::format::Pixel::BGRA => Some(Self::Bgra),
            _ => None,
        }
    }

    /// The layout as FFmpeg names it.
    pub(crate) const fn pixel(self) -> ffmpeg::format::Pixel {
        match self {
            Self::Nv12 => ffmpeg::format::Pixel::NV12,
            Self::Bgra => ffmpeg::format::Pixel::BGRA,
        }
    }

    pub(crate) const fn layouts(self) -> crate::contract::PixelLayoutSet {
        match self {
            Self::Nv12 => crate::contract::PixelLayoutSet::NV12,
            Self::Bgra => crate::contract::PixelLayoutSet::BGRA,
        }
    }
}

#[derive(Debug, ThisError)]
pub(crate) enum VideoToolboxFramesContextError {
    #[error("failed to allocate the VideoToolbox frames context")]
    Alloc,

    #[error(
        "failed to initialize the VideoToolbox frames context (code {code}) for {width}x{height} {format:?}"
    )]
    Init {
        code: i32,
        width: u32,
        height: u32,
        format: ffmpeg::format::Pixel,
    },
}

/// A pool of VideoToolbox frames holding `sw_format`, `width` by `height`,
/// on the context `hw_device_ctx` is. The pixel buffers come from Core
/// Video's own pool, which grows as frames are held.
///
/// # Safety
///
/// `hw_device_ctx` must be a live VideoToolbox device context.
pub(crate) unsafe fn create_frames_ctx(
    hw_device_ctx: &AvBufferRef,
    sw_format: ffmpeg::format::Pixel,
    width: u32,
    height: u32,
) -> Result<AvBufferRef, VideoToolboxFramesContextError> {
    // SAFETY: this function's own contract is a live device context. The
    // allocation is wrapped as an `AvBufferRef` before anything can fail, so
    // every path below either returns it or drops it. `data` is an
    // `AVHWFramesContext` by FFmpeg's own definition, and the fields written
    // here are the ones `av_hwframe_ctx_init` reads; VideoToolbox has no
    // frames context of its own to fill in.
    unsafe {
        let buf = AvBufferRef::from_raw(ffi::av_hwframe_ctx_alloc(hw_device_ctx.as_ptr()))
            .ok_or(VideoToolboxFramesContextError::Alloc)?;
        let frames_ctx = (*buf.as_ptr()).data as *mut ffi::AVHWFramesContext;
        (*frames_ctx).format = ffi::AVPixelFormat::AV_PIX_FMT_VIDEOTOOLBOX;
        (*frames_ctx).sw_format = sw_format.into();
        (*frames_ctx).width = width as i32;
        (*frames_ctx).height = height as i32;
        let code = ffi::av_hwframe_ctx_init(buf.as_ptr());
        if code < 0 {
            return Err(VideoToolboxFramesContextError::Init {
                code,
                width,
                height,
                format: sw_format,
            });
        }
        Ok(buf)
    }
}

/// Errors from making or drawing from a [`VideoToolboxFramePool`].
#[derive(Debug, ThisError)]
pub enum VideoToolboxFramePoolError {
    /// FFmpeg could not allocate the frames context.
    #[error("failed to allocate the VideoToolbox frames context")]
    Alloc,

    /// FFmpeg would not make a frames context of this size.
    #[error(
        "failed to initialize the VideoToolbox frames context (code {code}) for {width}x{height}"
    )]
    Init {
        /// FFmpeg's error code.
        code: i32,
        /// The width asked for.
        width: u32,
        /// The height asked for.
        height: u32,
    },

    /// The pool could not hand out a pixel buffer.
    #[error("failed to take a frame from the VideoToolbox pool (code {0})")]
    Get(i32),
}

impl From<VideoToolboxFramesContextError> for VideoToolboxFramePoolError {
    fn from(error: VideoToolboxFramesContextError) -> Self {
        match error {
            VideoToolboxFramesContextError::Alloc => Self::Alloc,
            VideoToolboxFramesContextError::Init {
                code,
                width,
                height,
                ..
            } => Self::Init {
                code,
                width,
                height,
            },
        }
    }
}

/// VideoToolbox pictures of one format and size, for an element of your own
/// to write into and hand on — the macOS counterpart of `CudaFramePool`.
///
/// A picture from here is what the VideoToolbox and Metal elements of this
/// crate take from their own upload or decoder: a frame in FFmpeg's
/// VideoToolbox format, carrying the frames context that says what it
/// holds, its pixel buffer from Core Video's own pool and backed by an
/// `IOSurface`, so Metal makes textures over it without a copy
/// (`MetalSurfaceView`, with the `metal` feature). Each pixel buffer goes
/// back to the pool when the last reference to the picture downstream is
/// dropped, and the pool grows as many are held at once.
///
/// A picture comes out with undefined pixels and no timing: copy what it is
/// made from — `av_frame_copy_props` from the picture handed in — before
/// pushing it. Make a pool again for another size; each one keeps the
/// device alive.
pub struct VideoToolboxFramePool {
    format: VideoToolboxFrameFormat,
    width: u32,
    height: u32,
    frames_ctx: AvBufferRef,
    /// Reuses only the CPU-side `AVFrame` wrapper; each pixel buffer comes
    /// from `frames_ctx`.
    wrappers: UnboundObjectPool<ffmpeg::frame::Video>,
}

impl VideoToolboxFramePool {
    /// A pool of `format` pictures, `width` by `height`, on `device`.
    ///
    /// # Errors
    ///
    /// [`VideoToolboxFramePoolError`] where FFmpeg cannot make a frames
    /// context of this size.
    pub fn new(
        device: &VideoToolboxDevice,
        format: VideoToolboxFrameFormat,
        width: u32,
        height: u32,
    ) -> Result<Self, VideoToolboxFramePoolError> {
        // SAFETY: `create_frames_ctx`'s contract is a live device context,
        // which the device's own reference is; the frames context takes a
        // reference of its own to it.
        let frames_ctx =
            unsafe { create_frames_ctx(&device.retain(), format.pixel(), width, height) }?;
        Ok(Self {
            format,
            width,
            height,
            frames_ctx,
            wrappers: UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {}),
        })
    }

    /// A picture to write into.
    ///
    /// # Errors
    ///
    /// [`VideoToolboxFramePoolError::Get`] where Core Video has no pixel
    /// buffer to give.
    pub fn get(
        &self,
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>, VideoToolboxFramePoolError> {
        let mut frame = self.wrappers.get();
        // SAFETY: `frame` is the pooled wrapper's own `AVFrame`;
        // unreferencing it first hands any previous pixel buffer back to its
        // pool. The frames context is this pool's own, held for its life.
        unsafe {
            let frame = frame.as_mut_ptr();
            ffi::av_frame_unref(frame);
            let code = ffi::av_hwframe_get_buffer(self.frames_ctx.as_ptr(), frame, 0);
            if code < 0 {
                return Err(VideoToolboxFramePoolError::Get(code));
            }
        }
        Ok(frame)
    }

    /// The format its pictures are in.
    pub fn format(&self) -> VideoToolboxFrameFormat {
        self.format
    }

    /// Its pictures' width.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Its pictures' height.
    pub fn height(&self) -> u32 {
        self.height
    }
}

/// Why a frame is not one a VideoToolbox element can read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NotVideoToolbox {
    /// It is not a VideoToolbox frame at all.
    Format(ffmpeg::format::Pixel),
    /// It says it is one, but carries no frames context to say what it
    /// holds or how to read it.
    NoFramesContext,
}

/// What `frame` holds, where it is a VideoToolbox frame.
pub(crate) fn sw_format_of(
    frame: &ffmpeg::frame::Video,
) -> Result<ffmpeg::format::Pixel, NotVideoToolbox> {
    if frame.format() != ffmpeg::format::Pixel::VIDEOTOOLBOX {
        return Err(NotVideoToolbox::Format(frame.format()));
    }
    // SAFETY: a live frame in FFmpeg's hardware format carries the frames
    // context it came from, whose `data` is an `AVHWFramesContext`; a null
    // reference is refused rather than read.
    unsafe {
        let frames_ref = (*frame.as_ptr()).hw_frames_ctx;
        if frames_ref.is_null() {
            return Err(NotVideoToolbox::NoFramesContext);
        }
        let frames_ctx = (*frames_ref).data as *const ffi::AVHWFramesContext;
        Ok(ffmpeg::format::Pixel::from((*frames_ctx).sw_format))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A context opens, and clones share it.
    #[test]
    fn a_clone_is_the_same_context() {
        let Some(device) = crate::test_support::try_videotoolbox_device() else {
            return;
        };
        let clone = device.clone();
        drop(device);
        assert!(Arc::ptr_eq(&clone.retain(), &clone.clone().retain()));
    }

    /// A frame in system memory is not a VideoToolbox one, and is refused
    /// by what it is.
    #[test]
    fn a_system_frame_is_not_a_videotoolbox_one() {
        let frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, 16, 16);
        assert_eq!(
            sw_format_of(&frame),
            Err(NotVideoToolbox::Format(ffmpeg::format::Pixel::NV12))
        );
    }
}
