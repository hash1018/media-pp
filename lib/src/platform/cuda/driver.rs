//! The handful of CUDA driver API entry points this crate calls directly,
//! declared here rather than pulled in through a CUDA binding crate.
//!
//! Everything here is the *driver* API (`cu*`), which the NVIDIA driver
//! itself ships — no CUDA toolkit is needed to build or run, the same
//! property `render_common`'s own `cuda_ffi` relies on. The signatures come
//! from `cuda.h`; they are plain C functions with scalar/pointer arguments
//! rather than structs whose layout has to be mirrored, except
//! `CUDA_MEMCPY2D`, which is versioned by name (`cuMemcpy2D_v2`) and has been
//! stable since CUDA 4.
//!
//! # Why this exists at all
//!
//! [`crate::elements::CudaVideoCompositor`] has to place one surface inside
//! another at an arbitrary offset, and libavfilter offers no CUDA filter that
//! can: `overlay_cuda` cannot crop, so `VideoFit::Cover` is not expressible,
//! and it ignores runtime commands, so moving a layer would mean rebuilding a
//! filter graph. A 2D device-to-device copy does all of it directly.

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};

use thiserror::Error as ThisError;

use crate::color::Color;

pub(crate) type CUresult = c_int;
pub(crate) type CUdevice = c_int;
pub(crate) type CUcontext = *mut c_void;
pub(crate) type CUdeviceptr = u64;

const CUDA_SUCCESS: CUresult = 0;
/// `CU_MEMORYTYPE_DEVICE`.
const CU_MEMORYTYPE_DEVICE: c_uint = 2;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CudaMemcpy2D {
    src_x_in_bytes: usize,
    src_y: usize,
    src_memory_type: c_uint,
    src_host: *const c_void,
    src_device: CUdeviceptr,
    src_array: *mut c_void,
    src_pitch: usize,

    dst_x_in_bytes: usize,
    dst_y: usize,
    dst_memory_type: c_uint,
    dst_host: *mut c_void,
    dst_device: CUdeviceptr,
    dst_array: *mut c_void,
    dst_pitch: usize,

    width_in_bytes: usize,
    height: usize,
}

// SAFETY of the block: these are the driver's own C ABI declarations, and
// every call site below checks the returned `CUresult`. On Windows the
// driver's `nvcuda.dll` is linked by name — `raw-dylib` needs no import
// library, so no CUDA toolkit install is required there either.
#[cfg_attr(windows, link(name = "nvcuda", kind = "raw-dylib"))]
#[cfg_attr(not(windows), link(name = "cuda"))]
unsafe extern "C" {
    fn cuInit(flags: c_uint) -> CUresult;
    fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;
    /// Retains the device's *primary* context — the same one
    /// [`crate::elements::CudaDevice`] makes FFmpeg use, so the frames these
    /// calls touch are reachable without mirroring any FFmpeg struct.
    fn cuDevicePrimaryCtxRetain(ctx: *mut CUcontext, device: CUdevice) -> CUresult;
    fn cuDevicePrimaryCtxRelease_v2(device: CUdevice) -> CUresult;
    fn cuCtxPushCurrent_v2(ctx: CUcontext) -> CUresult;
    fn cuCtxPopCurrent_v2(ctx: *mut CUcontext) -> CUresult;
    fn cuMemcpy2D_v2(copy: *const CudaMemcpy2D) -> CUresult;
    fn cuMemsetD2D8_v2(
        dst: CUdeviceptr,
        dst_pitch: usize,
        value: u8,
        width: usize,
        height: usize,
    ) -> CUresult;
    fn cuMemsetD2D16_v2(
        dst: CUdeviceptr,
        dst_pitch: usize,
        value: u16,
        width: usize,
        height: usize,
    ) -> CUresult;
    fn cuGetErrorString(error: CUresult, str_: *mut *const c_char) -> CUresult;
}

/// Errors from the CUDA driver calls this crate makes directly.
#[derive(Debug, ThisError)]
pub enum CudaDriverError {
    #[error("{call} failed: {message}")]
    Call { call: &'static str, message: String },
}

fn check(call: &'static str, result: CUresult) -> Result<(), CudaDriverError> {
    if result == CUDA_SUCCESS {
        return Ok(());
    }
    let mut raw: *const c_char = std::ptr::null();
    let message = unsafe {
        if cuGetErrorString(result, &mut raw) == CUDA_SUCCESS && !raw.is_null() {
            CStr::from_ptr(raw).to_string_lossy().into_owned()
        } else {
            format!("CUDA error {result}")
        }
    };
    Err(CudaDriverError::Call { call, message })
}

/// One retained reference to the device's primary CUDA context, plus the 2D
/// memory operations this crate issues against it.
///
/// A CUDA context is per-thread state, so every operation here pushes and
/// pops it rather than assuming it is current — the same reason
/// `render_common`'s `with_context` exists. Owning this keeps the primary
/// context alive for as long as the element that holds it, which is what
/// makes the pointers inside a frame it composites remain valid.
pub(crate) struct CudaDriver {
    device: CUdevice,
    ctx: CUcontext,
}

// SAFETY: a `CUcontext` is not thread-affine — it is pushed onto whichever
// thread uses it, which is exactly what `with_context` does around every
// call. The element owning this drives it from its own single source thread.
unsafe impl Send for CudaDriver {}

impl CudaDriver {
    /// Retains the primary context of CUDA device 0 — deliberately the same
    /// device [`crate::elements::CudaDevice`] opens, and for the same reason
    /// it takes no ordinal: a composite mixing surfaces from two GPUs is not
    /// expressible here anyway.
    pub(crate) fn retain_primary() -> Result<Self, CudaDriverError> {
        unsafe {
            check("cuInit", cuInit(0))?;
            let mut device: CUdevice = 0;
            check("cuDeviceGet", cuDeviceGet(&mut device, 0))?;
            let mut ctx: CUcontext = std::ptr::null_mut();
            check(
                "cuDevicePrimaryCtxRetain",
                cuDevicePrimaryCtxRetain(&mut ctx, device),
            )?;
            Ok(Self { device, ctx })
        }
    }

    fn with_context<T>(
        &self,
        f: impl FnOnce() -> Result<T, CudaDriverError>,
    ) -> Result<T, CudaDriverError> {
        unsafe { check("cuCtxPushCurrent", cuCtxPushCurrent_v2(self.ctx))? };
        let value = f();
        let mut popped: CUcontext = std::ptr::null_mut();
        unsafe { check("cuCtxPopCurrent", cuCtxPopCurrent_v2(&mut popped))? };
        value
    }

    /// Fills an NV12 surface with one opaque color.
    ///
    /// Two operations rather than one because NV12 is planar: luma is a byte
    /// per pixel, chroma is a `(U, V)` byte pair per 2x2 block — which is
    /// exactly a 16-bit pattern, so `cuMemsetD2D16` writes it without a
    /// kernel of its own.
    pub(crate) fn fill_nv12(
        &self,
        surface: Nv12Surface,
        width: u32,
        height: u32,
        color: Color,
    ) -> Result<(), CudaDriverError> {
        let (y, u, v) = rgb_to_bt709_limited(color);
        // Little-endian: the low byte lands at the lower address, which in an
        // interleaved NV12 chroma plane is U.
        let chroma = u16::from(u) | (u16::from(v) << 8);
        self.with_context(|| unsafe {
            check(
                "cuMemsetD2D8",
                cuMemsetD2D8_v2(
                    surface.luma,
                    surface.luma_pitch,
                    y,
                    width as usize,
                    height as usize,
                ),
            )?;
            check(
                "cuMemsetD2D16",
                cuMemsetD2D16_v2(
                    surface.chroma,
                    surface.chroma_pitch,
                    chroma,
                    (width / 2) as usize,
                    (height / 2) as usize,
                ),
            )
        })
    }

    /// Copies a rectangle of one NV12 surface into another, device to device.
    pub(crate) fn blit_nv12(
        &self,
        source: Nv12Surface,
        destination: Nv12Surface,
        region: Nv12Region,
    ) -> Result<(), CudaDriverError> {
        let Nv12Region {
            source_x,
            source_y,
            destination_x,
            destination_y,
            width,
            height,
        } = region;
        debug_assert!(
            [
                source_x,
                source_y,
                destination_x,
                destination_y,
                width,
                height
            ]
            .iter()
            .all(|value| value.is_multiple_of(2)),
            "NV12 blits must be aligned to the 2x2 chroma grid"
        );
        if width == 0 || height == 0 {
            return Ok(());
        }
        self.with_context(|| unsafe {
            let luma = CudaMemcpy2D {
                src_memory_type: CU_MEMORYTYPE_DEVICE,
                src_device: source.luma,
                src_pitch: source.luma_pitch,
                src_x_in_bytes: source_x as usize,
                src_y: source_y as usize,
                dst_memory_type: CU_MEMORYTYPE_DEVICE,
                dst_device: destination.luma,
                dst_pitch: destination.luma_pitch,
                dst_x_in_bytes: destination_x as usize,
                dst_y: destination_y as usize,
                width_in_bytes: width as usize,
                height: height as usize,
                ..CudaMemcpy2D::default()
            };
            check("cuMemcpy2D", cuMemcpy2D_v2(&luma))?;

            // Half the resolution in both axes, but two bytes per sample, so
            // the byte width stays `width` while the row count halves.
            let chroma = CudaMemcpy2D {
                src_memory_type: CU_MEMORYTYPE_DEVICE,
                src_device: source.chroma,
                src_pitch: source.chroma_pitch,
                src_x_in_bytes: source_x as usize,
                src_y: (source_y / 2) as usize,
                dst_memory_type: CU_MEMORYTYPE_DEVICE,
                dst_device: destination.chroma,
                dst_pitch: destination.chroma_pitch,
                dst_x_in_bytes: destination_x as usize,
                dst_y: (destination_y / 2) as usize,
                width_in_bytes: width as usize,
                height: (height / 2) as usize,
                ..CudaMemcpy2D::default()
            };
            check("cuMemcpy2D", cuMemcpy2D_v2(&chroma))
        })
    }
}

impl Drop for CudaDriver {
    fn drop(&mut self) {
        // Nothing useful to do with a failure here, and the process is
        // usually on its way out; the retain count is what matters.
        unsafe { cuDevicePrimaryCtxRelease_v2(self.device) };
    }
}

/// Which rectangle [`CudaDriver::blit_nv12`] moves, in luma pixels. Every
/// field must be even — chroma is subsampled 2x2, so an odd offset or extent
/// has no corresponding chroma rectangle. Callers align before calling; see
/// [`crate::elements::CudaVideoCompositor`]'s own notes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Nv12Region {
    pub(crate) source_x: u32,
    pub(crate) source_y: u32,
    pub(crate) destination_x: u32,
    pub(crate) destination_y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

/// The device pointers and pitches of one NV12 CUDA surface — what an
/// `AVFrame` carries in `data[0..2]`/`linesize[0..2]`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Nv12Surface {
    pub(crate) luma: CUdeviceptr,
    pub(crate) luma_pitch: usize,
    pub(crate) chroma: CUdeviceptr,
    pub(crate) chroma_pitch: usize,
}

impl Nv12Surface {
    /// Reads the planes out of a CUDA-resident frame. The caller has already
    /// validated that this *is* one — format, frames context, and device —
    /// so the only thing left to reject is a frame with no pointers at all.
    pub(crate) fn from_frame(frame: &ffmpeg_next::frame::Video) -> Option<Self> {
        let (luma, chroma, luma_pitch, chroma_pitch) = unsafe {
            let ptr = frame.as_ptr();
            (
                (*ptr).data[0],
                (*ptr).data[1],
                (*ptr).linesize[0],
                (*ptr).linesize[1],
            )
        };
        if luma.is_null() || chroma.is_null() || luma_pitch <= 0 || chroma_pitch <= 0 {
            return None;
        }
        Some(Self {
            luma: luma as CUdeviceptr,
            luma_pitch: luma_pitch as usize,
            chroma: chroma as CUdeviceptr,
            chroma_pitch: chroma_pitch as usize,
        })
    }
}

/// BT.709 limited-range Y'CbCr, matching what NVDEC produces and what NVENC
/// expects for HD content — a background filled with anything else would not
/// match the layers composited on top of it.
fn rgb_to_bt709_limited(color: Color) -> (u8, u8, u8) {
    let r = f32::from(color.red);
    let g = f32::from(color.green);
    let b = f32::from(color.blue);
    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let u = (b - y) / 1.8556;
    let v = (r - y) / 1.5748;
    (
        (16.0 + y * 219.0 / 255.0).round().clamp(0.0, 255.0) as u8,
        (128.0 + u * 224.0 / 255.0).round().clamp(0.0, 255.0) as u8,
        (128.0 + v * 224.0 / 255.0).round().clamp(0.0, 255.0) as u8,
    )
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use ffmpeg_next::{self as ffmpeg};

    use super::*;
    use crate::{
        buffer::MediaBuffer,
        control::ControlMsg,
        element::{Element, ElementType, Sink, Source, element_pp_log},
        elements::{CudaDownload, CudaFrameFormat, CudaUpload},
        pool::UnboundObjectPool,
        pp_log::PpLog,
        test_support::try_cuda_device,
    };

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
        fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
            self.received.lock().unwrap().push(buf);
            Ok(())
        }
        fn control(&mut self, _msg: ControlMsg) -> crate::error::Result<()> {
            Ok(())
        }
    }

    fn capture(element: &mut dyn Source) -> Arc<Mutex<Vec<MediaBuffer>>> {
        let received = Arc::new(Mutex::new(Vec::new()));
        element.src_pads()[0].link(Box::new(CapturingSink {
            received: received.clone(),
            pp_log: element_pp_log(ElementType::Other, "capture", None),
        }));
        received
    }

    /// Uploads one NV12 frame whose luma is `luma` everywhere and hands back
    /// the CUDA-resident result, so a test has a real surface to operate on.
    fn cuda_surface(
        device: &crate::elements::CudaDevice,
        width: u32,
        height: u32,
        luma: u8,
    ) -> Option<MediaBuffer> {
        let Ok(mut upload) =
            CudaUpload::new("upload", device, CudaFrameFormat::Nv12, width, height)
        else {
            eprintln!("skipping: this machine has no usable CUDA frames context");
            return None;
        };
        let uploaded = capture(&mut upload);
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, width, height);
        let y_stride = frame.stride(0);
        frame.data_mut(0)[..y_stride * height as usize].fill(luma);
        let uv_stride = frame.stride(1);
        frame.data_mut(1)[..uv_stride * (height / 2) as usize].fill(128);
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut slot = pool.get();
        *slot = frame;
        upload
            .consume(MediaBuffer::Video(Arc::new(slot)))
            .expect("upload");
        Some(uploaded.lock().unwrap().remove(0))
    }

    fn download(
        device: &crate::elements::CudaDevice,
        frame: MediaBuffer,
        width: u32,
        height: u32,
    ) -> Arc<crate::pool::UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        let mut download =
            CudaDownload::new("download", device, CudaFrameFormat::Nv12, width, height);
        let received = capture(&mut download);
        download.consume(frame).expect("download");
        let buf = received.lock().unwrap().remove(0);
        match buf {
            MediaBuffer::Video(frame) => frame,
            other => panic!("expected a Video buffer, got {}", other.kind()),
        }
    }

    /// The driver layer's whole contract in one pass: a fill covers the
    /// surface, and a blit moves exactly the requested rectangle to exactly
    /// the requested place, leaving everything else as the fill left it.
    #[test]
    fn fill_then_blit_writes_the_expected_rectangles() {
        let Some((device, _cuda_lock)) = try_cuda_device() else {
            return;
        };
        let driver = match CudaDriver::retain_primary() {
            Ok(driver) => driver,
            Err(error) => {
                eprintln!("skipping: no usable CUDA driver context ({error})");
                return;
            }
        };
        let (width, height) = (64u32, 64u32);
        let Some(canvas) = cuda_surface(&device, width, height, 0) else {
            return;
        };
        let Some(layer) = cuda_surface(&device, 32, 32, 200) else {
            return;
        };

        let (MediaBuffer::Video(canvas_frame), MediaBuffer::Video(layer_frame)) = (&canvas, &layer)
        else {
            panic!("expected Video buffers");
        };
        let canvas_surface = Nv12Surface::from_frame(canvas_frame).expect("canvas planes");
        let layer_surface = Nv12Surface::from_frame(layer_frame).expect("layer planes");

        driver
            .fill_nv12(canvas_surface, width, height, Color::WHITE)
            .expect("fill");
        driver
            .blit_nv12(
                layer_surface,
                canvas_surface,
                Nv12Region {
                    source_x: 0,
                    source_y: 0,
                    destination_x: 16,
                    destination_y: 8,
                    width: 32,
                    height: 32,
                },
            )
            .expect("blit");

        let out = download(&device, canvas.clone(), width, height);
        let stride = out.stride(0);
        let at = |x: usize, y: usize| out.data(0)[y * stride + x];
        assert_eq!(at(0, 0), 235, "the fill did not cover the top-left corner");
        assert_eq!(at(63, 63), 235, "the fill did not cover the bottom-right");
        assert_eq!(at(16, 8), 200, "the blit missed its top-left corner");
        assert_eq!(at(47, 39), 200, "the blit missed its bottom-right corner");
        assert_eq!(at(15, 8), 235, "the blit wrote left of its rectangle");
        assert_eq!(at(48, 8), 235, "the blit wrote right of its rectangle");
        assert_eq!(at(16, 7), 235, "the blit wrote above its rectangle");
        assert_eq!(at(16, 40), 235, "the blit wrote below its rectangle");

        let uv_stride = out.stride(1);
        assert_eq!(
            out.data(1)[uv_stride * 20 + 4],
            128,
            "chroma did not survive the fill"
        );
    }

    /// The two colors every background test in this crate uses, checked
    /// against the BT.709 limited-range values they are defined to produce.
    #[test]
    fn black_and_white_map_to_limited_range_endpoints() {
        assert_eq!(rgb_to_bt709_limited(Color::BLACK), (16, 128, 128));
        let (y, u, v) = rgb_to_bt709_limited(Color::WHITE);
        assert_eq!(y, 235);
        assert!(
            u.abs_diff(128) <= 1 && v.abs_diff(128) <= 1,
            "white must be chroma-neutral, got ({u}, {v})"
        );
    }
}
