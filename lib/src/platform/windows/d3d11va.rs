//! FFmpeg D3D11VA ABI and frame helpers shared by D3D11 elements.
//!
//! Decoder negotiation remains in the decoder element. This module owns only
//! the backend representation used identically by decode, upload, download,
//! scaling, compositing, chroma keying, rendering, capture, and encoding.

use std::ffi::c_void;

use ffmpeg_next::{self as ffmpeg, ffi};
use windows::{
    Win32::Graphics::Direct3D11::{ID3D11Device, ID3D11Texture2D},
    core::Interface,
};

use crate::platform::ffmpeg::AvBufferRef;

/// Mirrors `AVD3D11VADeviceContext` from FFmpeg n8.0. A future minimum
/// FFmpeg-version change must recheck this layout because the C header is not
/// included in `ffmpeg-sys-next`'s generated bindings.
#[repr(C)]
struct AVD3D11VADeviceContext {
    device: *mut c_void,
    device_context: *mut c_void,
    video_device: *mut c_void,
    video_context: *mut c_void,
    lock: Option<unsafe extern "C" fn(lock_ctx: *mut c_void)>,
    unlock: Option<unsafe extern "C" fn(lock_ctx: *mut c_void)>,
    lock_ctx: *mut c_void,
}

/// Mirrors the prefix of FFmpeg n8.0's `AVD3D11VAFramesContext` needed to
/// update `bind_flags` on a context FFmpeg has already initialized.
///
/// Never construct this struct or an `AVHWFramesContext` from the mirror. An
/// earlier implementation did so and caused memory corruption. The only valid
/// operation here is OR-ing flags into an existing FFmpeg-owned context before
/// `av_hwframe_ctx_init`.
#[repr(C)]
struct AVD3D11VAFramesContext {
    texture: *mut c_void,
    bind_flags: u32,
    misc_flags: u32,
    texture_infos: *mut c_void,
}

/// Creates an FFmpeg D3D11VA hardware device context with an independently
/// owned COM reference to `device`.
pub(crate) unsafe fn create_hw_device_ctx(device: &ID3D11Device) -> Result<AvBufferRef, i32> {
    // SAFETY: the caller supplies a live D3D11 device. The allocated FFmpeg
    // buffer is wrapped before any failure can occur; `IUnknown::clone` adds
    // the reference transferred into FFmpeg's D3D11VA device context.
    unsafe {
        let buf = AvBufferRef::from_raw(ffi::av_hwdevice_ctx_alloc(
            ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_D3D11VA,
        ))
        .ok_or(-1)?;

        let hw_device_ctx = (*buf.as_ptr()).data as *mut ffi::AVHWDeviceContext;
        let d3d11_ctx = (*hw_device_ctx).hwctx as *mut AVD3D11VADeviceContext;
        // FFmpeg unconditionally releases this interface when the context is
        // freed, so a borrowed `as_raw()` would consume the caller's ref.
        (*d3d11_ctx).device = device.clone().into_raw();

        let result = ffi::av_hwdevice_ctx_init(buf.as_ptr());
        if result < 0 {
            return Err(result);
        }
        Ok(buf)
    }
}

unsafe extern "C" fn release_d3d11_texture(_opaque: *mut c_void, data: *mut u8) {
    // SAFETY: `wrap_d3d11_texture` stores exactly one `ID3D11Texture2D`
    // reference as `data` and installs this callback to reclaim it once.
    unsafe {
        drop(ID3D11Texture2D::from_raw(data as *mut c_void));
    }
}

/// Wraps an owned texture reference as a `Pixel::D3D11` frame without
/// constructing a hand-mirrored FFmpeg hardware frames context.
///
/// A normal D3D11VA frame stores `ID3D11Texture2D*` directly in `data[0]` and
/// its texture-array slice in `data[1]`. `av_buffer_create` gives FFmpeg
/// reference-counted ownership of the COM reference and calls
/// `release_d3d11_texture` exactly once when the last frame reference drops.
pub(crate) fn wrap_d3d11_texture(
    texture: ID3D11Texture2D,
    width: u32,
    height: u32,
) -> ffmpeg::frame::Video {
    let mut frame = ffmpeg::frame::Video::empty();
    let raw = texture.into_raw();
    // SAFETY: `frame` owns a live, writable `AVFrame`; `raw` is the COM
    // reference transferred into the matching `av_buffer_create` callback,
    // and the array-slice encoding is stored as an integer and never dereferenced.
    unsafe {
        let ptr = frame.as_mut_ptr();
        (*ptr).format = ffi::AVPixelFormat::AV_PIX_FMT_D3D11 as i32;
        (*ptr).width = width as i32;
        (*ptr).height = height as i32;
        (*ptr).data[0] = raw as *mut u8;
        (*ptr).data[1] = std::ptr::null_mut();
        let buf = ffi::av_buffer_create(
            raw as *mut u8,
            std::mem::size_of::<*mut c_void>(),
            Some(release_d3d11_texture),
            std::ptr::null_mut(),
            0,
        );
        if buf.is_null() {
            (*ptr).data[0] = std::ptr::null_mut();
            drop(ID3D11Texture2D::from_raw(raw));
            panic!("av_buffer_create failed to allocate a D3D11 texture buffer wrapper");
        }
        (*ptr).buf[0] = buf;
    }
    frame
}

/// ORs resource bind flags into an existing FFmpeg-allocated D3D11VA frames
/// context before `av_hwframe_ctx_init`.
pub(crate) unsafe fn or_frames_bind_flags(frames_ctx: *mut ffi::AVHWFramesContext, flags: u32) {
    // SAFETY: the caller guarantees a live, not-yet-initialized D3D11VA frames
    // context, whose `hwctx` is FFmpeg's initialized `AVD3D11VAFramesContext`.
    unsafe {
        let d3d11_frames = (*frames_ctx).hwctx as *mut AVD3D11VAFramesContext;
        (*d3d11_frames).bind_flags |= flags;
    }
}

/// Extracts borrowed `(texture, array_index)` values from a D3D11VA frame.
/// A caller retaining the texture must clone its COM reference first.
pub(crate) fn d3d11va_texture(frame: &ffmpeg::frame::Video) -> Option<(*mut c_void, isize)> {
    if frame.format() != ffmpeg::format::Pixel::D3D11 {
        return None;
    }
    // SAFETY: `frame` is live and its fixed-size `data` array is initialized;
    // D3D11 frames encode the borrowed texture and slice in entries 0 and 1.
    unsafe {
        let ptr = frame.as_ptr();
        let texture = (*ptr).data[0] as *mut c_void;
        if texture.is_null() {
            return None;
        }
        Some((texture, (*ptr).data[1] as isize))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_a_d3d11_tagged_frame_without_a_texture() {
        let mut frame = ffmpeg::frame::Video::empty();
        // SAFETY: the test exclusively owns this live frame and writes only
        // its pixel-format discriminator.
        unsafe {
            (*frame.as_mut_ptr()).format = ffi::AVPixelFormat::AV_PIX_FMT_D3D11 as i32;
        }
        assert!(d3d11va_texture(&frame).is_none());
    }
}
