use ffmpeg_next::{self as ffmpeg, ffi};

/// What a CUDA surface on this crate's path actually holds — the
/// `sw_format` half of a hardware frame, as opposed to
/// [`ffmpeg::format::Pixel::CUDA`], which only says the pixels live in CUDA
/// memory at all.
///
/// Every CUDA element has to agree on this: [`crate::elements::CudaUpload`]
/// allocates surfaces in it, [`crate::elements::CudaEncoder`] configures
/// NVENC for it, and [`crate::elements::CudaScaler`] and
/// [`crate::elements::CudaDownload`] refuse a frame that carries something
/// else rather than reading it as if it were this.
///
/// # Why exactly these two
///
/// `Nv12` is what NVDEC produces and what
/// [`crate::elements::CudaRenderer`] presents. `Bgra` is what every screen
/// capture in this crate emits
/// (`PipeWireScreenCaptureSource`,
/// `DxgiCaptureSource`), and NVENC ingests it directly,
/// converting to YUV in hardware — so a capture-to-recording pipeline never
/// needs a colorspace conversion at all. That matters because the one thing
/// the GPU path *cannot* do is convert between them:
/// [`crate::elements::CudaScaler`]'s `scale_cuda` has no RGB-to-YUV kernel.
/// Pick the format the source already produces and keep it to the encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaFrameFormat {
    /// 8-bit 4:2:0, luma plane plus interleaved chroma.
    Nv12,
    /// 8-bit packed BGRA. The alpha byte is carried but ignored by NVENC.
    Bgra,
}

impl CudaFrameFormat {
    /// The `AVHWFramesContext.sw_format` a surface of this kind carries.
    pub(crate) fn sw_format(self) -> ffi::AVPixelFormat {
        match self {
            Self::Nv12 => ffi::AVPixelFormat::AV_PIX_FMT_NV12,
            Self::Bgra => ffi::AVPixelFormat::AV_PIX_FMT_BGRA,
        }
    }

    /// The CPU-side pixel format that uploads to, and downloads from, this
    /// kind of surface.
    pub(crate) fn pixel(self) -> ffmpeg::format::Pixel {
        match self {
            Self::Nv12 => ffmpeg::format::Pixel::NV12,
            Self::Bgra => ffmpeg::format::Pixel::BGRA,
        }
    }

    /// Which kind an existing surface holds, or `None` for a layout no
    /// element in this crate is wired up for (P010 from a 10-bit stream,
    /// say). Callers report the raw format in their own error, so this
    /// deliberately loses nothing a caller still needs.
    pub(crate) fn from_sw_format(format: ffi::AVPixelFormat) -> Option<Self> {
        match format {
            ffi::AVPixelFormat::AV_PIX_FMT_NV12 => Some(Self::Nv12),
            ffi::AVPixelFormat::AV_PIX_FMT_BGRA => Some(Self::Bgra),
            _ => None,
        }
    }
}
