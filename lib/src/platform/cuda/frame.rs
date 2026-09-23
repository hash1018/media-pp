//! FFmpeg CUDA hardware-frame helpers shared by the CUDA elements: the
//! frames context a producer allocates from, and the questions every
//! consumer asks of a frame before reading a device pointer out of it.

use ffmpeg_next::{self as ffmpeg, ffi, format::Pixel};
use thiserror::Error as ThisError;

use super::CudaFrameFormat;
use crate::{element::ElementType, platform::ffmpeg::AvBufferRef};

/// The surface layouts a CUDA element reads, as its errors name them.
///
/// One value both decides what [`CudaFrameError::UnsupportedSurfaceFormat`]
/// refuses and says what was wanted instead, so the two cannot disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaSurfaces(&'static [Pixel]);

impl CudaSurfaces {
    pub(crate) const NV12: Self = Self(&[Pixel::NV12]);
    pub(crate) const BGRA: Self = Self(&[Pixel::BGRA]);
    pub(crate) const NV12_OR_BGRA: Self = Self(&[Pixel::NV12, Pixel::BGRA]);
    pub(crate) const NV12_OR_P010: Self = Self(&[Pixel::NV12, Pixel::P010LE]);
    /// What `scale_cuda` resizes: both [`CudaFrameFormat`]s, and the P010 a
    /// 10-bit decode arrives as.
    pub(crate) const SCALABLE: Self = Self(&[Pixel::NV12, Pixel::BGRA, Pixel::P010LE]);

    /// The one layout a [`CudaFrameFormat`] is.
    pub(crate) fn of(format: CudaFrameFormat) -> Self {
        match format {
            CudaFrameFormat::Nv12 => Self::NV12,
            CudaFrameFormat::Bgra => Self::BGRA,
        }
    }

    /// The layouts, as FFmpeg names them.
    pub fn layouts(self) -> &'static [Pixel] {
        self.0
    }

    fn contains(self, layout: Pixel) -> bool {
        self.0.contains(&layout)
    }
}

impl std::fmt::Display for CudaSurfaces {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (index, layout) in self.0.iter().enumerate() {
            if index > 0 {
                f.write_str(" or ")?;
            }
            write!(f, "{layout:?}")?;
        }
        Ok(())
    }
}

/// What is wrong with a frame a CUDA element was handed.
///
/// Every CUDA element asks the same four questions before it reads a device
/// pointer out of a frame — is it a CUDA frame at all, does it carry the
/// frames context that describes it, was it allocated against this
/// element's own device, and is its surface in a layout this element reads —
/// and asks them through one function, so a pointer is never read on the
/// strength of a copy that forgot one. Each element's own error carries this
/// through `#[from]`, and names the element in its message.
#[derive(Debug, ThisError)]
pub enum CudaFrameError {
    /// The frame is not in CUDA memory: a system-memory frame, or another
    /// backend's.
    #[error("{element:?} takes CUDA frames, got {actual:?}")]
    NotCuda {
        /// The element that refused it.
        element: ElementType,
        /// The frame's own pixel format.
        actual: Pixel,
    },

    /// A CUDA frame arrived without the frames context that describes it,
    /// so it did not come from a CUDA producer at all.
    #[error("{element:?} was handed a CUDA frame with no frames context")]
    MissingFramesContext {
        /// The element that refused it.
        element: ElementType,
    },

    /// The frame was allocated against a different CUDA device than this
    /// element's, so its device pointers mean nothing here.
    #[error("{element:?} was handed a surface from a different CUDA device than its own")]
    ForeignContext {
        /// The element that refused it.
        element: ElementType,
    },

    /// The surface is CUDA-resident but not in a layout this element reads.
    #[error("{element:?} reads {accepts} surfaces, got {actual:?}")]
    UnsupportedSurfaceFormat {
        /// The element that refused it.
        element: ElementType,
        /// What it reads.
        accepts: CudaSurfaces,
        /// What the surface holds.
        actual: Pixel,
    },
}

/// A frame [`validate`] passed: the frames context it came from, and the
/// layout its surface holds.
pub(crate) struct CudaSurface {
    /// The frame's own `hw_frames_ctx`, borrowed for as long as the frame
    /// is — what a filter graph has to be configured against.
    pub(crate) frames_ctx: *mut ffi::AVBufferRef,
    /// One of the layouts `validate` was told this element reads.
    pub(crate) layout: Pixel,
}

/// Asks the four questions [`CudaFrameError`] describes, in the order a
/// pointer has to be safe to read for the next one.
///
/// `device_ctx` is the element's own device context, compared by address
/// only.
pub(crate) fn validate(
    frame: &ffmpeg::frame::Video,
    element: ElementType,
    device_ctx: *const ffi::AVHWDeviceContext,
    accepts: CudaSurfaces,
) -> Result<CudaSurface, CudaFrameError> {
    if frame.format() != Pixel::CUDA {
        return Err(CudaFrameError::NotCuda {
            element,
            actual: frame.format(),
        });
    }
    // SAFETY: `frame` is a live `frame::Video` just confirmed to be
    // `Pixel::CUDA`, so `as_ptr` yields an initialized `AVFrame` and a
    // hardware frame's `hw_frames_ctx` is either null — refused here — or an
    // `AVBufferRef` whose `data` is an `AVHWFramesContext`, also checked.
    // Its `device_ctx` is only compared by address, never dereferenced.
    unsafe {
        let frames_ref = (*frame.as_ptr()).hw_frames_ctx;
        if frames_ref.is_null() || (*frames_ref).data.is_null() {
            return Err(CudaFrameError::MissingFramesContext { element });
        }
        let frames_ctx = (*frames_ref).data as *const ffi::AVHWFramesContext;
        if !std::ptr::eq((*frames_ctx).device_ctx, device_ctx) {
            return Err(CudaFrameError::ForeignContext { element });
        }
        let layout = Pixel::from((*frames_ctx).sw_format);
        if !accepts.contains(layout) {
            return Err(CudaFrameError::UnsupportedSurfaceFormat {
                element,
                accepts,
                actual: layout,
            });
        }
        Ok(CudaSurface {
            frames_ctx: frames_ref,
            layout,
        })
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    /// What an error says was wanted is the same value that decided the
    /// refusal, so it has to read as a list a person can act on.
    #[test]
    fn surfaces_read_as_the_layouts_they_accept() {
        assert_eq!(CudaSurfaces::NV12.to_string(), "NV12");
        assert_eq!(CudaSurfaces::NV12_OR_P010.to_string(), "NV12 or P010LE");
        assert_eq!(CudaSurfaces::of(CudaFrameFormat::Bgra), CudaSurfaces::BGRA);
    }

    /// The first question needs no device: a system-memory frame is refused
    /// before anything of it is read as a CUDA frame, and the refusal names
    /// the element that asked.
    #[test]
    fn a_system_memory_frame_is_refused_before_it_is_read() {
        let frame = ffmpeg::frame::Video::new(Pixel::NV12, 16, 16);
        let error = validate(
            &frame,
            ElementType::CudaRenderer,
            std::ptr::null(),
            CudaSurfaces::NV12,
        )
        .err()
        .expect("a CPU frame is not a CUDA frame");

        assert!(matches!(
            error,
            CudaFrameError::NotCuda {
                element: ElementType::CudaRenderer,
                actual: Pixel::NV12,
            }
        ));
        assert_eq!(
            error.to_string(),
            "CudaRenderer takes CUDA frames, got NV12"
        );
    }
}
