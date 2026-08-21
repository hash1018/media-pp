//! FFmpeg D3D12VA ABI and context helpers shared by D3D12 elements.
//!
//! Nothing pipeline-shaped lives here. The decoder, upload, scaler, and
//! renderer all use the same FFmpeg hardware-context and frame layout, so the
//! representation belongs to the Windows backend rather than to any one
//! element.

use std::ffi::c_void;

use ffmpeg_next::{self as ffmpeg, ffi};
use windows::{Win32::Graphics::Direct3D12::ID3D12Device, core::Interface};

use crate::platform::ffmpeg::AvBufferRef;

/// Mirrors of the D3D12VA-specific structs from FFmpeg's
/// `libavutil/hwcontext_d3d12va.h` (as of FFmpeg n8.0), hand-written because
/// `ffmpeg-sys-next` does not bind that header. COM pointers stay raw so the
/// layout depends only on FFmpeg's C ABI, not on `windows-rs` internals. A
/// future FFmpeg header change can make these mirrors stale without a compile
/// error, so this module must be checked when the minimum FFmpeg version moves.
#[repr(C)]
struct AVD3D12VADeviceContext {
    device: *mut c_void,
    video_device: *mut c_void,
    lock: Option<unsafe extern "C" fn(lock_ctx: *mut c_void)>,
    unlock: Option<unsafe extern "C" fn(lock_ctx: *mut c_void)>,
    lock_ctx: *mut c_void,
}

#[repr(C)]
struct AVD3D12VASyncContext {
    fence: *mut c_void,
    event: *mut c_void,
    fence_value: u64,
}

#[repr(C)]
struct AVD3D12VAFrame {
    texture: *mut c_void,
    sync_ctx: AVD3D12VASyncContext,
}

/// Creates an FFmpeg D3D12VA hardware device context with an independently
/// owned COM reference to `device`.
pub(crate) unsafe fn create_hw_device_ctx(device: &ID3D12Device) -> Result<AvBufferRef, i32> {
    unsafe {
        let buf = AvBufferRef::from_raw(ffi::av_hwdevice_ctx_alloc(
            ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_D3D12VA,
        ))
        .ok_or(-1)?;

        let hw_device_ctx = (*buf.as_ptr()).data as *mut ffi::AVHWDeviceContext;
        let d3d12_ctx = (*hw_device_ctx).hwctx as *mut AVD3D12VADeviceContext;
        // FFmpeg always releases a caller-supplied device when this context
        // is freed, so transfer an independently owned COM reference.
        (*d3d12_ctx).device = device.clone().into_raw();

        let result = ffi::av_hwdevice_ctx_init(buf.as_ptr());
        if result < 0 {
            return Err(result);
        }
        Ok(buf)
    }
}

/// Creates an NV12 D3D12VA frame pool tied to `hw_device_ctx`.
pub(crate) unsafe fn create_hw_frames_ctx(
    hw_device_ctx: &AvBufferRef,
    width: u32,
    height: u32,
    initial_pool_size: i32,
) -> Result<AvBufferRef, i32> {
    unsafe {
        let buf =
            AvBufferRef::from_raw(ffi::av_hwframe_ctx_alloc(hw_device_ctx.as_ptr())).ok_or(-1)?;
        let frames_ctx = (*buf.as_ptr()).data as *mut ffi::AVHWFramesContext;
        (*frames_ctx).format = ffi::AVPixelFormat::AV_PIX_FMT_D3D12;
        (*frames_ctx).sw_format = ffi::AVPixelFormat::AV_PIX_FMT_NV12;
        (*frames_ctx).width = width as i32;
        (*frames_ctx).height = height as i32;
        (*frames_ctx).initial_pool_size = initial_pool_size;
        let result = ffi::av_hwframe_ctx_init(buf.as_ptr());
        if result < 0 {
            return Err(result);
        }
        Ok(buf)
    }
}

/// Extracts borrowed `(texture, fence, fence_value)` pointers from an FFmpeg
/// D3D12VA frame. A caller retaining either COM object must clone it first.
pub(crate) fn d3d12va_texture(
    frame: &ffmpeg::frame::Video,
) -> Option<(*mut c_void, *mut c_void, u64)> {
    if frame.format() != ffmpeg::format::Pixel::D3D12 {
        return None;
    }
    unsafe {
        let data = (*frame.as_ptr()).data[0];
        if data.is_null() {
            return None;
        }
        let d3d12_frame = &*(data as *const AVD3D12VAFrame);
        Some((
            d3d12_frame.texture,
            d3d12_frame.sync_ctx.fence,
            d3d12_frame.sync_ctx.fence_value,
        ))
    }
}

/// Updates the fence value after a producer queues work and signals the frame
/// pool's own fence.
pub(crate) fn set_d3d12va_fence_value(frame: &mut ffmpeg::frame::Video, fence_value: u64) -> bool {
    if frame.format() != ffmpeg::format::Pixel::D3D12 {
        return false;
    }
    unsafe {
        let data = (*frame.as_mut_ptr()).data[0];
        if data.is_null() {
            return false;
        }
        (*(data as *mut AVD3D12VAFrame)).sync_ctx.fence_value = fence_value;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_a_d3d12_tagged_frame_without_its_abi_payload() {
        let mut frame = ffmpeg::frame::Video::empty();
        unsafe {
            (*frame.as_mut_ptr()).format = ffi::AVPixelFormat::AV_PIX_FMT_D3D12 as i32;
        }

        assert!(d3d12va_texture(&frame).is_none());
        assert!(!set_d3d12va_fence_value(&mut frame, 1));
    }
}
