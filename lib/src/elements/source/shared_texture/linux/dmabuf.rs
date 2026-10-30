//! What a DMA-BUF another process hands over is, whichever GPU API it is
//! brought into.

use std::ffi::c_int;

/// What the fourth byte of each pixel in the pushed buffers is.
///
/// Fixed for a source, because it is a property of whoever produces the
/// pictures rather than of any one of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DmaBufAlpha {
    /// Nothing: every picture is opaque, as a screen capture is. The byte is
    /// read as `DRM_FORMAT_XRGB8888` and comes out as 255.
    Ignored,
    /// Alpha, with colour that has not been multiplied by it — what every
    /// other BGRA frame in this crate carries.
    Straight,
    /// Alpha, with colour already multiplied by it — what a compositing
    /// producer such as a browser engine hands over. The frames that come
    /// out have it divided back out, so they are straight like every other
    /// BGRA frame here: the CUDA and Vulkan compositors blend straight
    /// colour, and a filter keying on colour wants the colour itself.
    Premultiplied,
}

/// One picture another process has drawn, as it hands it over: a single
/// plane of packed BGRA or BGRx in a DMA-BUF.
///
/// Borrowed rather than owned — the file descriptor is the producer's, and
/// it only has to stay open until the push it is handed to returns:
/// `CudaDmaBufHandle::push`, or `VulkanDmaBufHandle::push`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DmaBuf {
    /// The DMA-BUF.
    pub fd: c_int,
    /// Where the plane starts in it, in bytes.
    pub offset: u32,
    /// Bytes per row, which is often more than four times the width.
    pub stride: u32,
    /// The DRM format modifier the buffer was allocated with —
    /// `DRM_FORMAT_MOD_LINEAR` (zero) for plain rows of pixels, and else a
    /// GPU's own tiled layout: NVIDIA's block-linear is what Chromium hands
    /// over, drawing off-screen on its Wayland platform.
    pub modifier: u64,
    /// The picture's size in pixels.
    pub width: u32,
    pub height: u32,
}

#[cfg(test)]
pub(super) mod tests {
    use std::ffi::{c_int, c_void};
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

    unsafe extern "C" {
        fn dlopen(file: *const std::ffi::c_char, mode: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, name: *const std::ffi::c_char) -> *mut c_void;
    }

    /// A buffer the way a GPU producer hands one over: allocated by the
    /// driver's GBM with `DRM_FORMAT_MOD_LINEAR` — on NVIDIA, exactly what a
    /// browser engine's are, down to the stride — and filled through the
    /// CPU mapping GBM offers, so what it holds is known. Returns the
    /// DMA-BUF and its stride, or `None` where GBM or a GPU is missing.
    ///
    /// Not `udmabuf`, which would need no GPU: on NVIDIA 595 its import
    /// read other processes' memory as often as the buffer's own.
    pub(in super::super) fn gbm_buffer_with(
        pixels: &[u8],
        width: u32,
        height: u32,
        modifier: u64,
    ) -> Option<(OwnedFd, u32)> {
        type Device = *mut c_void;
        type Bo = *mut c_void;
        const GBM_FORMAT_ARGB8888: u32 = 0x3432_5241;
        const GBM_BO_USE_SCANOUT: u32 = 1;
        const GBM_BO_TRANSFER_WRITE: u32 = 2;

        let node = match std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/dri/renderD128")
        {
            Ok(node) => node,
            Err(error) => {
                eprintln!("skipping: no render node ({error})");
                return None;
            }
        };
        // SAFETY: `dlopen`/`dlsym` get NUL-terminated names, and each symbol
        // is cast to its signature from `gbm.h`. The device and buffer are
        // used only while `node` is open, a linear buffer's mapping is written
        // within the `stride * height` bytes GBM maps — a tiled one is not
        // mapped at all — and the buffer's DMA-BUF is a new fd owned by the
        // returned `OwnedFd`. Neither the buffer nor the device
        // is destroyed: the process ends with the test.
        unsafe {
            let library = dlopen(c"libgbm.so.1".as_ptr(), 2);
            if library.is_null() {
                eprintln!("skipping: libgbm.so.1 is missing");
                return None;
            }
            let symbol = |name: &std::ffi::CStr| dlsym(library, name.as_ptr());
            let create_device: unsafe extern "C" fn(c_int) -> Device =
                std::mem::transmute(symbol(c"gbm_create_device"));
            let create_bo: unsafe extern "C" fn(Device, u32, u32, u32, *const u64, u32, u32) -> Bo =
                std::mem::transmute(symbol(c"gbm_bo_create_with_modifiers2"));
            let map: unsafe extern "C" fn(
                Bo,
                u32,
                u32,
                u32,
                u32,
                u32,
                *mut u32,
                *mut *mut c_void,
            ) -> *mut c_void = std::mem::transmute(symbol(c"gbm_bo_map"));
            let unmap: unsafe extern "C" fn(Bo, *mut c_void) =
                std::mem::transmute(symbol(c"gbm_bo_unmap"));
            let get_fd: unsafe extern "C" fn(Bo) -> c_int =
                std::mem::transmute(symbol(c"gbm_bo_get_fd"));
            let get_stride: unsafe extern "C" fn(Bo) -> u32 =
                std::mem::transmute(symbol(c"gbm_bo_get_stride"));

            let device = create_device(node.as_raw_fd());
            assert!(!device.is_null(), "gbm_create_device failed");
            let modifiers = [modifier];
            let bo = create_bo(
                device,
                width,
                height,
                GBM_FORMAT_ARGB8888,
                modifiers.as_ptr(),
                1,
                GBM_BO_USE_SCANOUT,
            );
            if bo.is_null() {
                eprintln!("skipping: this GPU's GBM allocates no buffer of modifier {modifier:#x}");
                return None;
            }
            if modifier != 0 {
                // A tiled layout the CPU cannot map: written through GL.
                let stride = get_stride(bo);
                let fd = get_fd(bo);
                assert!(fd >= 0, "gbm_bo_get_fd failed");
                let fd = OwnedFd::from_raw_fd(fd);
                fill_tiled(&fd, stride, modifier, width, height, pixels)?;
                std::mem::forget(node);
                return Some((fd, stride));
            }
            let (mut stride, mut data) = (0u32, std::ptr::null_mut());
            let mapped = map(
                bo,
                0,
                0,
                width,
                height,
                GBM_BO_TRANSFER_WRITE,
                &mut stride,
                &mut data,
            );
            assert!(!mapped.is_null(), "gbm_bo_map failed");
            let row = width as usize * 4;
            for y in 0..height as usize {
                std::ptr::copy_nonoverlapping(
                    pixels[y * row..].as_ptr(),
                    mapped.cast::<u8>().add(y * stride as usize),
                    row,
                );
            }
            unmap(bo, data);
            let fd = get_fd(bo);
            assert!(fd >= 0, "gbm_bo_get_fd failed");
            std::mem::forget(node);
            Some((OwnedFd::from_raw_fd(fd), stride))
        }
    }

    /// Writes `pixels` into the tiled buffer `fd` through GL, with the CUDA
    /// import's EGL context — `None`, after saying so, where it cannot.
    #[cfg(feature = "cuda")]
    fn fill_tiled(
        fd: &OwnedFd,
        stride: u32,
        modifier: u64,
        width: u32,
        height: u32,
        pixels: &[u8],
    ) -> Option<()> {
        use crate::platform::linux::dmabuf_cuda::{
            DRM_FORMAT_ARGB8888, DmaBufCudaImporter, DmaBufPlane,
        };
        let mut importer = match DmaBufCudaImporter::new() {
            Ok(importer) => importer,
            Err(error) => {
                eprintln!("skipping: no GL to fill a tiled buffer with ({error})");
                return None;
            }
        };
        importer
            .write(
                DmaBufPlane {
                    fd: fd.as_raw_fd(),
                    offset: 0,
                    stride: stride as i32,
                    modifier,
                    fourcc: DRM_FORMAT_ARGB8888,
                },
                width,
                height,
                pixels,
            )
            .expect("writing the tiled buffer through GL");
        Some(())
    }

    /// Without `cuda` there is no GL here to fill a tiled buffer with.
    #[cfg(not(feature = "cuda"))]
    fn fill_tiled(
        _fd: &OwnedFd,
        _stride: u32,
        _modifier: u64,
        _width: u32,
        _height: u32,
        _pixels: &[u8],
    ) -> Option<()> {
        eprintln!("skipping: filling a tiled buffer takes the `cuda` feature's GL");
        None
    }
}
