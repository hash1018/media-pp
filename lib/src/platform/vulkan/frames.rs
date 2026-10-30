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
/// that. FFmpeg makes a multi-planar image, NV12's or P010's, that can
/// be sampled or stored to so that each of its planes can be viewed on its
/// own — which is how this crate's kernels read and write one.
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

/// `errno`'s out-of-memory, as FFmpeg reports an allocation that failed.
const ENOMEM: i32 = 12;

/// What holds a picture handed to one of FFmpeg's Vulkan encoders: its
/// buffer, and the frames context that buffer goes back to.
struct HeldPicture {
    // First, so dropped first: the picture goes back to its pool while the
    // context the pool belongs to is still there.
    buffer: AvBufferRef,
    _frames_ctx: AvBufferRef,
}

/// Lets go of a [`HeldPicture`]: the picture's buffer, then its frames
/// context.
///
/// # Safety
///
/// `opaque` is the `HeldPicture` [`holding_its_frames_ctx`] boxed, and
/// FFmpeg calls this once, when the last reference to the buffer goes.
unsafe extern "C" fn release_held(opaque: *mut std::ffi::c_void, _data: *mut u8) {
    // SAFETY: this function's own contract — the box made for this buffer,
    // given back exactly once.
    drop(unsafe { Box::from_raw(opaque.cast::<HeldPicture>()) });
}

/// Makes whatever holds `frame`'s picture hold its frames context too, so
/// the picture can go back to its pool however late it is let go of.
///
/// FFmpeg's Vulkan encoders keep only a picture's buffer once they have
/// submitted it (`ff_vk_exec_add_dep_frame` refs `buf[0]`, and
/// `vulkan_encode.c` frees the frame at once), until its packet comes out;
/// the frame's reference to its frames context goes with the frame. A
/// context let go of in between — a decoder's, replaced when the stream
/// changes size, or an upload's, dropped before the encoder drained — is
/// freed while one of its pictures is out, and that picture's return to the
/// pool reads it after it is gone. So `frame`'s `buf[0]` is swapped for a
/// buffer over the same picture that holds both, and lets go of them in
/// that order.
///
/// A frame without a buffer or a frames context is left as it is.
pub(crate) fn holding_its_frames_ctx(
    frame: &mut ffmpeg::frame::Video,
) -> Result<(), ffmpeg::Error> {
    // SAFETY: `frame` is a live frame this caller owns, so its `buf[0]` and
    // `hw_frames_ctx` are references it may take or give away; null ones are
    // left alone. `buf[0]`'s reference moves into the `HeldPicture` and,
    // where FFmpeg cannot make the buffer that owns it, back into the frame,
    // so it is held exactly once on every path. The new buffer describes the
    // same bytes as the one it replaces, and `release_held` is the matching
    // inverse of the box made for it.
    unsafe {
        let raw = frame.as_mut_ptr();
        let buffer = (*raw).buf[0];
        if buffer.is_null() || (*raw).hw_frames_ctx.is_null() {
            return Ok(());
        }
        let frames_ctx = AvBufferRef::from_raw(ffi::av_buffer_ref((*raw).hw_frames_ctx))
            .ok_or(ffmpeg::Error::Other { errno: ENOMEM })?;
        let (data, size) = ((*buffer).data, (*buffer).size);
        let held = Box::into_raw(Box::new(HeldPicture {
            buffer: AvBufferRef::from_raw(buffer).expect("checked non-null above"),
            _frames_ctx: frames_ctx,
        }));
        let wrapper = ffi::av_buffer_create(data, size, Some(release_held), held.cast(), 0);
        if wrapper.is_null() {
            let held = Box::from_raw(held);
            (*raw).buf[0] = held.buffer.into_raw();
            return Err(ffmpeg::Error::Other { errno: ENOMEM });
        }
        (*raw).buf[0] = wrapper;
    }
    Ok(())
}
