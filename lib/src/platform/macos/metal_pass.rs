//! One Metal compute kernel over a VideoToolbox frame, into a new BGRA one of
//! the same size — what a per-pixel filter on Metal is: `MetalVideoEffect`,
//! `MetalChromaKey`, and `MetalConverter` from NV12 or HDR P010. The Metal counterpart
//! of `platform::vulkan::bgra_pass`.
//!
//! The kernel reads the source through textures made over its pixel
//! buffer's `IOSurface` and writes the output frame's the same way, nothing
//! copied: texture 0 is the output, `bgra8Unorm`, written; texture 1 the
//! source — for NV12 and P010 its luma, and texture 2 its chroma — and
//! buffer 0 the kernel's parameters, which say the picture's size.

use std::sync::Arc;

use ffmpeg_next::{self as ffmpeg, ffi};
use objc2_metal::{MTLPixelFormat, MTLTextureUsage};
use thiserror::Error as ThisError;

use super::{
    metal::{Kernel, MetalError, MetalGpu, Texture},
    pixel_buffer::PixelBuffer,
    videotoolbox::{NotVideoToolbox, create_frames_ctx, sw_format_of},
};
use crate::{
    elements::VideoToolboxDevice,
    frame_size::ForSize,
    platform::ffmpeg::AvBufferRef,
    pool::{UnboundObjectPool, UnboundObjectPoolRef},
};

/// Why a pass could not run over a frame.
#[derive(Debug, ThisError)]
pub(crate) enum MetalPassError {
    #[error("got a {0:?} frame; upload it first")]
    NotVideoToolbox(ffmpeg::format::Pixel),
    #[error("got a VideoToolbox frame with no frames context")]
    NoFramesContext,
    #[error("takes {expected:?} frames, got {got:?}")]
    WrongLayout {
        expected: ffmpeg::format::Pixel,
        got: ffmpeg::format::Pixel,
    },
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error("{0}")]
    Pool(String),
    #[error("failed to take a frame from the VideoToolbox pool (code {0})")]
    FrameGet(i32),
}

/// What a pass reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PassInput {
    Bgra,
    Nv12,
    /// Luma as `r16Unorm` and chroma as `rg16Unorm`, each sample as it is
    /// stored: ten bits at the top of two bytes.
    P010,
}

impl PassInput {
    fn pixel(self) -> ffmpeg::format::Pixel {
        match self {
            Self::Bgra => ffmpeg::format::Pixel::BGRA,
            Self::Nv12 => ffmpeg::format::Pixel::NV12,
            Self::P010 => ffmpeg::format::Pixel::P010LE,
        }
    }
}

/// The kernel, and everything one frame through it needs.
pub(crate) struct MetalPass {
    gpu: MetalGpu,
    hw_device_ctx: Arc<AvBufferRef>,
    input: PassInput,
    kernel: Kernel,
    /// The pool output frames come from, for the size of the frames
    /// arriving.
    frames: ForSize<AvBufferRef>,
    wrappers: UnboundObjectPool<ffmpeg::frame::Video>,
}

// SAFETY: the FFmpeg buffers have no thread affinity, and the Metal objects
// are thread-safe and touched only through `&mut self`.
unsafe impl Send for MetalPass {}

impl MetalPass {
    /// Kernel `entry` of `shader`, whose output frames are made on `device`.
    pub(crate) fn new(
        device: &VideoToolboxDevice,
        shader: &str,
        entry: &str,
        input: PassInput,
    ) -> Result<Self, MetalError> {
        let gpu = MetalGpu::new()?;
        let kernel = gpu
            .kernels(shader, &[entry])?
            .pop()
            .expect("one kernel for one name");
        Ok(Self {
            gpu,
            hw_device_ctx: device.retain(),
            input,
            kernel,
            frames: ForSize::new(),
            wrappers: UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {}),
        })
    }

    /// `source` through the kernel with `parameters` at buffer 0, into a new
    /// frame with `source`'s timing and colour.
    pub(crate) fn run(
        &mut self,
        source: &ffmpeg::frame::Video,
        parameters: &[u8],
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>, MetalPassError> {
        match sw_format_of(source) {
            Ok(got) if got == self.input.pixel() => {}
            Ok(got) => {
                return Err(MetalPassError::WrongLayout {
                    expected: self.input.pixel(),
                    got,
                });
            }
            Err(NotVideoToolbox::Format(format)) => {
                return Err(MetalPassError::NotVideoToolbox(format));
            }
            Err(NotVideoToolbox::NoFramesContext) => return Err(MetalPassError::NoFramesContext),
        }
        let source_buffer = PixelBuffer::of_frame(source)
            .ok_or(MetalPassError::NotVideoToolbox(source.format()))?;
        let (width, height) = (source.width(), source.height());
        let hw_device_ctx = &self.hw_device_ctx;
        let frames = self
            .frames
            .try_get(width, height, |width, height| {
                // SAFETY: a live VideoToolbox device context, this pass's own.
                unsafe {
                    create_frames_ctx(hw_device_ctx, ffmpeg::format::Pixel::BGRA, width, height)
                }
                .map_err(|error| MetalPassError::Pool(error.to_string()))
            })?
            .as_ptr();

        let mut output = self.wrappers.get();
        // SAFETY: the pooled wrapper's own `AVFrame`, its previous pixel
        // buffer handed back first; the frames context is this pass's own.
        unsafe {
            let dst = output.as_mut_ptr();
            ffi::av_frame_unref(dst);
            let code = ffi::av_hwframe_get_buffer(frames, dst, 0);
            if code < 0 {
                return Err(MetalPassError::FrameGet(code));
            }
        }
        let output_buffer = PixelBuffer::of_frame(&output).expect("a frame of this pass's pool");

        let read = MTLTextureUsage::ShaderRead;
        let mut textures: Vec<Texture> = vec![self.gpu.plane(
            &output_buffer,
            0,
            MTLPixelFormat::BGRA8Unorm,
            MTLTextureUsage::ShaderWrite,
        )?];
        match self.input {
            PassInput::Bgra => {
                textures.push(self.gpu.plane(
                    &source_buffer,
                    0,
                    MTLPixelFormat::BGRA8Unorm,
                    read,
                )?);
            }
            PassInput::Nv12 | PassInput::P010 => {
                let (luma, chroma) = if self.input == PassInput::Nv12 {
                    (MTLPixelFormat::R8Unorm, MTLPixelFormat::RG8Unorm)
                } else {
                    (MTLPixelFormat::R16Unorm, MTLPixelFormat::RG16Unorm)
                };
                textures.push(self.gpu.plane(&source_buffer, 0, luma, read)?);
                textures.push(self.gpu.plane(&source_buffer, 1, chroma, read)?);
            }
        }
        let bound: Vec<&Texture> = textures.iter().collect();
        let mut pass = self.gpu.pass()?;
        pass.dispatch(&self.kernel, &bound, Some(parameters), (width, height));
        pass.finish()?;
        // SAFETY: two distinct live frames; props are timing, colour and side
        // data, not buffers.
        unsafe {
            ffi::av_frame_copy_props(output.as_mut_ptr(), source.as_ptr());
        }
        Ok(output)
    }
}

/// The kernel's parameters, as the words it reads them in.
pub(crate) fn parameters(words: &[[u8; 4]]) -> Vec<u8> {
    words.iter().flatten().copied().collect()
}

/// `map` as `shaders/metal/tone_map.metal`'s `ToneMap` reads it, for a
/// picture of `size`: three rows, the transfer and the EETF's three
/// numbers, three rows of the gamut, then the size.
pub(crate) fn tone_map_parameters(map: &crate::tone_map::ToneMap, size: (u32, u32)) -> Vec<u8> {
    let mut bytes: Vec<u8> = map
        .rows
        .iter()
        .flatten()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    bytes.extend(map.transfer.to_ne_bytes());
    for value in [map.source_peak_pq, map.target_peak, map.knee] {
        bytes.extend(value.to_ne_bytes());
    }
    bytes.extend(
        map.gamut
            .iter()
            .flatten()
            .flat_map(|value| value.to_ne_bytes()),
    );
    for word in [size.0, size.1, 0, 0] {
        bytes.extend(word.to_ne_bytes());
    }
    bytes
}
