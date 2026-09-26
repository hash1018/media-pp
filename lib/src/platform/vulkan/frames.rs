//! FFmpeg's Vulkan frames: making a pool of them, and checking that a frame
//! is one.

use ash::vk;
use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;

use crate::platform::ffmpeg::AvBufferRef;

use super::ffi::AVVulkanFramesContext;

#[derive(Debug, ThisError)]
pub(crate) enum VulkanFramesContextError {
    #[error("failed to allocate the Vulkan frames context")]
    Alloc,

    #[error(
        "failed to initialize the Vulkan frames context (code {code}) for {width}x{height} {format:?}"
    )]
    Init {
        code: i32,
        width: u32,
        height: u32,
        format: ffmpeg::format::Pixel,
    },
}

/// A growable pool of Vulkan frames holding `sw_format`, `width` by
/// `height`, on the device `hw_device_ctx` is.
///
/// FFmpeg fills in the rest of the `AVVulkanFramesContext` itself during
/// `av_hwframe_ctx_init`: images that can be sampled, stored to, and copied
/// both ways, where the format allows — `usage`, where not empty, in place of
/// that. A multi-planar image, NV12's or P010's, is made so that each of its
/// planes can be viewed on its own, which is how this crate's kernels read
/// and write one.
///
/// # Safety
///
/// `hw_device_ctx` must be a live Vulkan device context.
pub(crate) unsafe fn create_frames_ctx(
    hw_device_ctx: &AvBufferRef,
    sw_format: ffmpeg::format::Pixel,
    width: u32,
    height: u32,
    usage: vk::ImageUsageFlags,
) -> Result<AvBufferRef, VulkanFramesContextError> {
    // SAFETY: this function's own contract is a live device context. The
    // allocation is wrapped as an `AvBufferRef` before anything can fail, so
    // every path below either returns it or drops it. `data` is an
    // `AVHWFramesContext` by FFmpeg's own definition, and the fields written
    // here are the ones `av_hwframe_ctx_init` reads.
    unsafe {
        let buf = AvBufferRef::from_raw(ffi::av_hwframe_ctx_alloc(hw_device_ctx.as_ptr()))
            .ok_or(VulkanFramesContextError::Alloc)?;
        let frames_ctx = (*buf.as_ptr()).data as *mut ffi::AVHWFramesContext;
        (*frames_ctx).format = ffi::AVPixelFormat::AV_PIX_FMT_VULKAN;
        (*frames_ctx).sw_format = sw_format.into();
        (*frames_ctx).width = width as i32;
        (*frames_ctx).height = height as i32;
        // Growable: no producer here needs the fixed pool a decoder does.
        (*frames_ctx).initial_pool_size = 0;
        let hwctx = (*frames_ctx).hwctx as *mut AVVulkanFramesContext;
        (*hwctx).usage = usage.as_raw() as _;
        if matches!(
            sw_format,
            ffmpeg::format::Pixel::NV12 | ffmpeg::format::Pixel::P010LE
        ) {
            // Unset, FFmpeg makes an image aliasable and nothing more.
            (*hwctx).img_flags = (vk::ImageCreateFlags::ALIAS
                | vk::ImageCreateFlags::MUTABLE_FORMAT
                | vk::ImageCreateFlags::EXTENDED_USAGE)
                .as_raw() as _;
        }
        let code = ffi::av_hwframe_ctx_init(buf.as_ptr());
        if code < 0 {
            return Err(VulkanFramesContextError::Init {
                code,
                width,
                height,
                format: sw_format,
            });
        }
        Ok(buf)
    }
}

/// Why a frame is not one a Vulkan element can use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NotOurs {
    /// It is not a Vulkan frame at all.
    NotVulkan(ffmpeg::format::Pixel),
    /// It is a Vulkan frame of another `VkDevice`.
    ForeignDevice,
}

/// What `frame` holds, where it is a Vulkan frame made on the device
/// `device_ctx` is: an image made on another `VkDevice` cannot be used on
/// this one.
///
/// `device_ctx` is only compared, never read.
pub(crate) fn sw_format_of(
    frame: &ffmpeg::frame::Video,
    device_ctx: *const ffi::AVHWDeviceContext,
) -> Result<ffmpeg::format::Pixel, NotOurs> {
    if frame.format() != ffmpeg::format::Pixel::VULKAN {
        return Err(NotOurs::NotVulkan(frame.format()));
    }
    // SAFETY: a live frame in FFmpeg's hardware format carries the frames
    // context it came from, whose `data` is an `AVHWFramesContext`; a null
    // reference is refused rather than read.
    unsafe {
        let frames_ref = (*frame.as_ptr()).hw_frames_ctx;
        if frames_ref.is_null() {
            return Err(NotOurs::ForeignDevice);
        }
        let frames_ctx = (*frames_ref).data as *const ffi::AVHWFramesContext;
        if !std::ptr::eq((*frames_ctx).device_ctx, device_ctx) {
            return Err(NotOurs::ForeignDevice);
        }
        Ok(ffmpeg::format::Pixel::from((*frames_ctx).sw_format))
    }
}
