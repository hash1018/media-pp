use std::sync::Arc;

use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;

use crate::platform::ffmpeg::AvBufferRef;

/// Errors from opening the CUDA device context.
#[derive(Debug, ThisError)]
pub enum CudaDeviceError {
    #[error("failed to open the CUDA device: {0}")]
    Open(ffmpeg::Error),

    #[error("FFmpeg opened CUDA without returning a device context")]
    MissingContext,
}

/// The one CUDA device context every CUDA element in a pipeline shares.
///
/// This is the exact analog of the single `ID3D11Device` the D3D11 stack
/// requires (see `D3d11Renderer`'s docs on why that stack
/// needs one shared device rather than merely one adapter): a CUDA frame
/// allocated against one context is not readable from another, so
/// [`crate::elements::CudaDecoder`] and [`crate::elements::CudaRenderer`]
/// must be constructed from the *same* `CudaDevice`. Create one per stack
/// and hand it to every CUDA element.
///
/// Each element takes its own FFmpeg reference at construction, so this
/// value only has to outlive the constructor calls — not the pipeline. The
/// underlying context stays alive as long as any element, or any frame
/// still in flight downstream, references it.
///
/// # Create it once, up front
///
/// Because this is the device's *primary* context (see below), constructing
/// and dropping a `CudaDevice` retains and releases process-wide driver
/// state. Doing that on one thread while another thread has NVDEC or NVENC
/// work in flight has been observed to segfault inside `libnvcuvid`, on a
/// thread the driver itself owns — nothing in this crate can catch or
/// recover that. Build one per process before the pipelines start, hand it
/// to every CUDA element, and keep it until they are gone.
///
/// Opens the default CUDA device. There is deliberately no ordinal
/// parameter: a `CudaRenderer`'s graphics device has to be on the same
/// physical GPU, and this crate has no way to verify that pairing, so
/// offering a choice here would only make a mismatch expressible without
/// making it detectable.
///
/// # Why the primary context
///
/// This opens the device's **primary** CUDA context rather than letting
/// FFmpeg create a private one, and that choice is what makes
/// [`crate::elements::CudaFrameRenderer`] implementable at all. Importing a
/// Vulkan/D3D image into CUDA means calling `cuImportExternalMemory` on the
/// *same* context the frames live on — and the only other way to reach
/// FFmpeg's private context is to hand-mirror `AVCUDADeviceContext`, which
/// `ffmpeg-sys-next` does not bind. Reconstructing FFmpeg hwcontext structs
/// by hand is exactly the practice that corrupted memory in this crate's
/// D3D11VA history (see `d3d11va_decoder`'s own notes). With the primary
/// context, an interop implementation calls `cuDevicePrimaryCtxRetain` and
/// provably gets the same `CUcontext` these frames were decoded on, with no
/// struct layout guessed anywhere.
pub struct CudaDevice {
    ctx: Arc<AvBufferRef>,
}

/// `AV_CUDA_USE_PRIMARY_CONTEXT` from `libavutil/hwcontext_cuda.h` — passed
/// as `av_hwdevice_ctx_create`'s `flags`, which is a plain documented `int`
/// parameter, not a struct layout this crate has to mirror.
const AV_CUDA_USE_PRIMARY_CONTEXT: i32 = 1 << 0;

impl CudaDevice {
    pub fn new() -> Result<Self, CudaDeviceError> {
        let mut ctx: *mut ffi::AVBufferRef = std::ptr::null_mut();
        // Unlike D3D11VA/D3D12VA, nothing has to be filled in by hand here:
        // there is no caller-provided device object to hand over, so
        // `av_hwdevice_ctx_create` both allocates and initializes the
        // context, and this crate never touches `AVCUDADeviceContext`'s
        // layout at all.
        // SAFETY: `ctx` is a live local FFmpeg writes the allocated context into,
        // and the two nulls are the documented "default device, no options" form.
        // The comment above records why this crate never touches the
        // `AVCUDADeviceContext` layout itself.
        let result = unsafe {
            ffi::av_hwdevice_ctx_create(
                &mut ctx,
                ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_CUDA,
                std::ptr::null(),
                std::ptr::null_mut(),
                AV_CUDA_USE_PRIMARY_CONTEXT,
            )
        };
        if result < 0 {
            return Err(CudaDeviceError::Open(ffmpeg::Error::from(result)));
        }
        // SAFETY: `av_hwdevice_ctx_create` left `ctx` owning one reference and
        // nothing else has taken it, which is what `from_raw` requires; a failure
        // left it null and returned above.
        let ctx = unsafe { AvBufferRef::from_raw(ctx) }.ok_or(CudaDeviceError::MissingContext)?;
        Ok(Self { ctx: Arc::new(ctx) })
    }

    pub(crate) fn retain(&self) -> Arc<AvBufferRef> {
        self.ctx.clone()
    }
}
