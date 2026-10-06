//! What an element of your own needs to work on this crate's CUDA pictures
//! with CUDA calls of its own, the pictures never leaving the GPU: where a
//! picture's planes are, and pictures to write into that the rest of the
//! pipeline takes as its own.

use std::marker::PhantomData;

use ffmpeg_next::{self as ffmpeg, ffi, format::Pixel};
use thiserror::Error as ThisError;

use super::{
    CudaDevice, CudaFrameError, CudaFrameFormat, CudaSurfaces,
    frame::{self, CudaFramesContextError, create_hw_frames_ctx},
};
use crate::{
    element::ElementType,
    platform::ffmpeg::AvBufferRef,
    pool::{UnboundObjectPool, UnboundObjectPoolRef},
};

/// Where a CUDA picture's pixels are: one [`CudaPlane`] for BGRA, two for
/// NV12 and P010 (luma, then interleaved chroma).
///
/// What a custom element reads a picture by — a detector or a model of its
/// own, a measurement — or, for a picture from its own [`CudaFramePool`],
/// writes one by. Made by [`CudaSurfaceView::new`], which asks of the
/// frame what every CUDA element of this crate asks before it reads a
/// device pointer: that it is a CUDA frame, from `device`, in one of the
/// layouts this crate makes. The view borrows the frame, so the planes it
/// names stay allocated while it is held; a pointer copied out of it is
/// valid only as long as the frame is.
///
/// # The contract with the rest of the pipeline
///
/// - **Context.** Every CUDA picture of a pipeline lives in the primary
///   context of [`CudaDevice::ordinal`]. Work on it in that context —
///   `cuDevicePrimaryCtxRetain`, or `cudarc`'s `CudaContext::new`.
/// - **A picture handed to you is ready for the legacy default stream.**
///   The CUDA elements of this crate wait for their own work on a picture
///   before handing it on, and what FFmpeg does for them — a decode, an
///   upload — is queued on the legacy default stream. Work on that stream,
///   or on a stream created without `CU_STREAM_NON_BLOCKING`, is ordered
///   after it; a non-blocking stream has to synchronize the context first.
/// - **Finish before you hand on.** Do the same: synchronize the stream
///   or context you worked on before pushing a picture you wrote, since
///   the element after yours reads it from a stream of its own.
/// - **Never write a picture you were handed.** It may be read on another
///   branch of a `Tee`, or be the one a source offers again. Write into a
///   frame from a [`CudaFramePool`] instead, copying what you keep.
#[derive(Debug)]
pub struct CudaSurfaceView<'a> {
    layout: Pixel,
    width: u32,
    height: u32,
    planes: [CudaPlane; 2],
    count: usize,
    frame: PhantomData<&'a ffmpeg::frame::Video>,
}

/// One plane of a CUDA picture, in device memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaPlane {
    /// The first byte of its first row, as a `CUdeviceptr`.
    pub pointer: u64,
    /// The bytes from the start of one row to the start of the next.
    pub pitch: usize,
    /// The bytes of each row that are picture: a NV12 chroma row's
    /// interleaved pairs, a P010 row's two bytes a sample.
    pub row_bytes: usize,
    /// The rows that are picture.
    pub rows: u32,
}

impl<'a> CudaSurfaceView<'a> {
    /// The planes of `frame`, a picture on `device`.
    ///
    /// # Errors
    ///
    /// [`CudaFrameError`] naming [`ElementType::Other`] where `frame` is
    /// not a CUDA frame, has no frames context, is from another device, is
    /// not NV12, P010 or BGRA, or carries no device pointer.
    pub fn new(
        device: &CudaDevice,
        frame: &'a ffmpeg::frame::Video,
    ) -> Result<Self, CudaFrameError> {
        let layout = frame::validate(
            frame,
            ElementType::Other,
            device.device_ctx(),
            CudaSurfaces::SCALABLE,
        )?
        .layout;
        let (width, height) = (frame.width(), frame.height());
        let half = |size: u32| size.div_ceil(2);
        // (bytes a sample, bytes a row of chroma pairs)
        let (luma, chroma) = match layout {
            Pixel::NV12 => (width as usize, half(width) as usize * 2),
            Pixel::P010LE => (width as usize * 2, half(width) as usize * 4),
            _ => (width as usize * 4, 0),
        };
        let count = if layout == Pixel::BGRA { 1 } else { 2 };
        let mut planes = [CudaPlane {
            pointer: 0,
            pitch: 0,
            row_bytes: luma,
            rows: height,
        }; 2];
        planes[1].row_bytes = chroma;
        planes[1].rows = half(height);
        for (index, plane) in planes.iter_mut().enumerate().take(count) {
            // SAFETY: `frame` is a live `frame::Video`, so `as_ptr` yields an
            // initialized `AVFrame`, whose `data` and `linesize` arrays have
            // these indices for every format; what the values are is checked
            // below rather than trusted.
            let (pointer, pitch) = unsafe {
                (
                    (*frame.as_ptr()).data[index],
                    (*frame.as_ptr()).linesize[index],
                )
            };
            if pointer.is_null() || pitch <= 0 || (pitch as usize) < plane.row_bytes {
                return Err(CudaFrameError::NoSurface {
                    element: ElementType::Other,
                });
            }
            plane.pointer = pointer as u64;
            plane.pitch = pitch as usize;
        }
        Ok(Self {
            layout,
            width,
            height,
            planes,
            count,
            frame: PhantomData,
        })
    }

    /// What the surface holds: NV12, P010LE or BGRA.
    pub fn layout(&self) -> Pixel {
        self.layout
    }

    /// The picture's width, in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// The picture's height, in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Its planes: luma then chroma, or BGRA's one.
    pub fn planes(&self) -> &[CudaPlane] {
        &self.planes[..self.count]
    }
}

/// Errors from making or drawing from a [`CudaFramePool`].
#[derive(Debug, ThisError)]
pub enum CudaFramePoolError {
    /// FFmpeg could not allocate the frames context.
    #[error("failed to allocate the CUDA frames context")]
    Alloc,

    /// FFmpeg would not make a frames context of this size.
    #[error("failed to initialize the CUDA frames context (code {code}) for {width}x{height}")]
    Init {
        /// FFmpeg's error code.
        code: i32,
        /// The width asked for.
        width: u32,
        /// The height asked for.
        height: u32,
    },

    /// The pool could not hand out a surface.
    #[error("failed to take a frame from the CUDA pool (code {0})")]
    Get(i32),
}

impl From<CudaFramesContextError> for CudaFramePoolError {
    fn from(error: CudaFramesContextError) -> Self {
        match error {
            CudaFramesContextError::Alloc => Self::Alloc,
            CudaFramesContextError::Init {
                code,
                width,
                height,
            } => Self::Init {
                code,
                width,
                height,
            },
        }
    }
}

/// CUDA pictures of one format and size on a [`CudaDevice`], for an element
/// of your own to write into and hand on.
///
/// A picture from here is one every CUDA element of the pipeline takes as
/// from its own device; one allocated through another FFmpeg device
/// context, even on the same GPU and context, is refused by them as
/// [`CudaFrameError::ForeignContext`]. Each surface goes back to the pool
/// when the last reference to the picture downstream is dropped, and the
/// pool grows as many are held at once.
///
/// A picture comes out blank, with no timing: copy what it is made from —
/// `av_frame_copy_props` from the picture handed in — before pushing it.
/// [`CudaSurfaceView::new`] gives its planes. Make a pool again for
/// another size; each one keeps the device alive.
pub struct CudaFramePool {
    format: CudaFrameFormat,
    width: u32,
    height: u32,
    frames_ctx: AvBufferRef,
    /// Reuses only the CPU-side `AVFrame` wrapper; each surface comes from
    /// `frames_ctx`.
    wrappers: UnboundObjectPool<ffmpeg::frame::Video>,
}

impl CudaFramePool {
    /// A pool of `format` pictures, `width` by `height`, on `device`.
    ///
    /// # Errors
    ///
    /// [`CudaFramePoolError`] where FFmpeg cannot make a frames context of
    /// this size.
    pub fn new(
        device: &CudaDevice,
        format: CudaFrameFormat,
        width: u32,
        height: u32,
    ) -> Result<Self, CudaFramePoolError> {
        // SAFETY: `create_hw_frames_ctx`'s contract is a live device context,
        // which the device's own reference is; the frames context takes a
        // reference of its own to it.
        let frames_ctx = unsafe { create_hw_frames_ctx(&device.retain(), format, width, height) }?;
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
    /// [`CudaFramePoolError::Get`] where the device has no memory for one.
    pub fn get(&self) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>, CudaFramePoolError> {
        let mut frame = self.wrappers.get();
        // SAFETY: `frame` is the pooled wrapper's own `AVFrame`;
        // unreferencing it first hands any previous surface back to its
        // pool. The frames context is this pool's own, held for its life.
        unsafe {
            let frame = frame.as_mut_ptr();
            ffi::av_frame_unref(frame);
            let code = ffi::av_hwframe_get_buffer(self.frames_ctx.as_ptr(), frame, 0);
            if code < 0 {
                return Err(CudaFramePoolError::Get(code));
            }
        }
        Ok(frame)
    }

    /// The format its pictures are in.
    pub fn format(&self) -> CudaFrameFormat {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn device() -> Option<CudaDevice> {
        match CudaDevice::new() {
            Ok(device) => Some(device),
            Err(error) => {
                eprintln!("skipping: no CUDA device ({error})");
                None
            }
        }
    }

    /// A pool's picture is one the view takes as from the device, with the
    /// planes its format has, each at least a row of picture wide.
    #[test]
    fn a_pooled_picture_is_seen_as_from_its_device() {
        let Some(device) = device() else { return };
        for (format, layout, planes) in [
            (CudaFrameFormat::Nv12, Pixel::NV12, 2),
            (CudaFrameFormat::Bgra, Pixel::BGRA, 1),
        ] {
            let pool = CudaFramePool::new(&device, format, 64, 36).expect("pool");
            let frame = pool.get().expect("frame");
            let view = CudaSurfaceView::new(&device, &frame).expect("view");
            assert_eq!(view.layout(), layout);
            assert_eq!((view.width(), view.height()), (64, 36));
            assert_eq!(view.planes().len(), planes);
            for plane in view.planes() {
                assert_ne!(plane.pointer, 0);
                assert!(plane.pitch >= plane.row_bytes);
            }
            if planes == 2 {
                assert_eq!(view.planes()[1].rows, 18);
                assert_eq!(view.planes()[1].row_bytes, 64);
            } else {
                assert_eq!(view.planes()[0].row_bytes, 256);
            }
        }
    }

    /// A picture from another FFmpeg device context is refused, though it
    /// is the same GPU's same primary context: what the rest of the
    /// pipeline would refuse too.
    #[test]
    fn a_picture_from_another_device_is_refused() {
        let Some(device) = device() else { return };
        let other = CudaDevice::new().expect("a second device context");
        let pool = CudaFramePool::new(&other, CudaFrameFormat::Nv12, 64, 36).expect("pool");
        let frame = pool.get().expect("frame");
        assert!(matches!(
            CudaSurfaceView::new(&device, &frame),
            Err(CudaFrameError::ForeignContext {
                element: ElementType::Other
            })
        ));
    }

    /// A system-memory picture is refused before anything of it is read as
    /// device memory.
    #[test]
    fn a_system_memory_picture_is_refused() {
        let Some(device) = device() else { return };
        let frame = ffmpeg::frame::Video::new(Pixel::NV12, 64, 36);
        assert!(matches!(
            CudaSurfaceView::new(&device, &frame),
            Err(CudaFrameError::NotCuda {
                actual: Pixel::NV12,
                ..
            })
        ));
    }
}
