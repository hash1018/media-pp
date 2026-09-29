//! Metal, as `MetalVideoCompositor` uses it: the system's GPU and a command
//! queue on it, compute kernels compiled from Metal Shading Language at run
//! time — no shader toolchain at build time — and textures, made over the
//! `IOSurface` behind a VideoToolbox frame's pixel buffer, so a picture is
//! drawn from and into where it already is.
//!
//! A pixel buffer belongs to no Metal device: any device in the process can
//! make a texture over its surface, as any VideoToolbox session can read it.
//! So there is no device to share with the rest of a pipeline, and each
//! element makes its own; on Apple silicon they are the one GPU.

use std::{ffi::c_void, ptr::NonNull};

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBarrierScope, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLibrary, MTLOrigin, MTLPixelFormat, MTLRegion, MTLSize, MTLStorageMode, MTLTexture,
    MTLTextureDescriptor, MTLTextureUsage,
};
use thiserror::Error as ThisError;

use super::pixel_buffer::PixelBuffer;

/// A texture, as Metal hands one out.
pub(crate) type Texture = Retained<ProtocolObject<dyn MTLTexture>>;
/// A compiled compute kernel.
pub(crate) type Kernel = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// Why Metal would not do what a Metal element asked of it.
#[derive(Debug, ThisError)]
pub enum MetalError {
    /// There is no Metal device: a Mac too old for Metal, or a session with
    /// no GPU at all.
    #[error("no Metal device is available")]
    NoDevice,

    /// Metal would not make a command queue, command buffer or encoder.
    #[error("Metal would not make a {0}")]
    Unavailable(&'static str),

    /// A kernel's source would not compile, or a kernel of it not become a
    /// pipeline — a fault in this crate's own shader, reported with
    /// Metal's words for it.
    #[error("the Metal kernel {kernel} would not compile: {reason}")]
    Compile {
        /// The kernel's name.
        kernel: String,
        /// What Metal said.
        reason: String,
    },

    /// Metal would not make a texture.
    #[error("Metal would not make a {width}x{height} {format:?} texture")]
    Texture {
        /// The texture's width in pixels.
        width: u32,
        /// Its height in pixels.
        height: u32,
        /// Its format.
        format: MTLPixelFormat,
    },

    /// A pixel buffer is not backed by an `IOSurface`, so no texture can be
    /// made over it without a copy.
    #[error("the pixel buffer is not backed by an IOSurface")]
    NotIoSurface,

    /// The GPU did not finish what it was given.
    #[error("the GPU did not finish the command buffer: {0}")]
    Execution(String),
}

/// The GPU and a queue on it.
pub(crate) struct MetalGpu {
    pub(crate) device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

// SAFETY: Metal's device and command queue are thread-safe objects, which
// Apple documents as usable from any thread.
unsafe impl Send for MetalGpu {}
// SAFETY: as above.
unsafe impl Sync for MetalGpu {}

impl MetalGpu {
    /// The system's default GPU, and a command queue on it.
    pub(crate) fn new() -> Result<Self, MetalError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::NoDevice)?;
        let queue = device
            .newCommandQueue()
            .ok_or(MetalError::Unavailable("command queue"))?;
        Ok(Self { device, queue })
    }

    /// Compiles `source` and each of `kernels` in it into a pipeline, in the
    /// order they are named.
    pub(crate) fn kernels(
        &self,
        source: &str,
        kernels: &[&str],
    ) -> Result<Vec<Kernel>, MetalError> {
        let library = self
            .device
            .newLibraryWithSource_options_error(&NSString::from_str(source), None)
            .map_err(|error| MetalError::Compile {
                kernel: "(library)".into(),
                reason: error.localizedDescription().to_string(),
            })?;
        kernels
            .iter()
            .map(|&kernel| {
                let function = library
                    .newFunctionWithName(&NSString::from_str(kernel))
                    .ok_or_else(|| MetalError::Compile {
                        kernel: kernel.into(),
                        reason: "no such function in the library".into(),
                    })?;
                self.device
                    .newComputePipelineStateWithFunction_error(&function)
                    .map_err(|error| MetalError::Compile {
                        kernel: kernel.into(),
                        reason: error.localizedDescription().to_string(),
                    })
            })
            .collect()
    }

    /// A texture of its own, in GPU memory where `shared` is false and in
    /// memory the CPU writes too where it is true.
    pub(crate) fn texture(
        &self,
        format: MTLPixelFormat,
        width: u32,
        height: u32,
        usage: MTLTextureUsage,
        shared: bool,
    ) -> Result<Texture, MetalError> {
        let descriptor = descriptor(format, width, height, usage);
        descriptor.setStorageMode(if shared {
            MTLStorageMode::Shared
        } else {
            MTLStorageMode::Private
        });
        self.device
            .newTextureWithDescriptor(&descriptor)
            .ok_or(MetalError::Texture {
                width,
                height,
                format,
            })
    }

    /// A texture over plane `plane` of `buffer`'s surface, seen as `format`
    /// — the pixels themselves, nothing copied, read and written where they
    /// are.
    pub(crate) fn plane(
        &self,
        buffer: &PixelBuffer,
        plane: usize,
        format: MTLPixelFormat,
        usage: MTLTextureUsage,
    ) -> Result<Texture, MetalError> {
        let surface = buffer.io_surface().ok_or(MetalError::NotIoSurface)?;
        let (width, height) = buffer.plane_size(plane);
        let descriptor = descriptor(format, width, height, usage);
        // A plane the buffer has, whose size the descriptor is made of.
        self.device
            .newTextureWithDescriptor_iosurface_plane(&descriptor, &surface, plane)
            .ok_or(MetalError::Texture {
                width,
                height,
                format,
            })
    }

    /// A command buffer with one compute encoder open on it, for one
    /// frame's work.
    pub(crate) fn pass(&self) -> Result<Pass, MetalError> {
        let commands = self
            .queue
            .commandBuffer()
            .ok_or(MetalError::Unavailable("command buffer"))?;
        let encoder = commands
            .computeCommandEncoder()
            .ok_or(MetalError::Unavailable("compute encoder"))?;
        Ok(Pass { commands, encoder })
    }
}

fn descriptor(
    format: MTLPixelFormat,
    width: u32,
    height: u32,
    usage: MTLTextureUsage,
) -> Retained<MTLTextureDescriptor> {
    // SAFETY: a plain constructor; a size of zero is refused by the device
    // when the texture is made, not here.
    let descriptor = unsafe {
        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            format,
            width as usize,
            height as usize,
            false,
        )
    };
    descriptor.setUsage(usage);
    descriptor
}

/// One frame's work: dispatches encoded in turn, each seeing what the one
/// before it wrote, then run and waited for.
pub(crate) struct Pass {
    commands: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
}

impl Pass {
    /// Runs `kernel` over `threads`, eight by eight at a time, with
    /// `textures` bound from index 0 and `bytes` at buffer 0 — after
    /// everything already encoded has written its textures.
    ///
    /// Each texture must be alive until [`Self::finish`] returns, as Metal
    /// retains the objects a command buffer uses.
    pub(crate) fn dispatch(
        &mut self,
        kernel: &Kernel,
        textures: &[&Texture],
        bytes: Option<&[u8]>,
        threads: (u32, u32),
    ) {
        let encoder = &self.encoder;
        encoder.memoryBarrierWithScope(MTLBarrierScope::Textures);
        encoder.setComputePipelineState(kernel);
        for (index, texture) in textures.iter().enumerate() {
            // SAFETY: a live texture, at an index the kernel declares.
            unsafe { encoder.setTexture_atIndex(Some(texture), index) };
        }
        if let Some(bytes) = bytes {
            // SAFETY: Metal copies `bytes` before returning; they are live
            // for the call.
            unsafe {
                encoder.setBytes_length_atIndex(
                    NonNull::from(bytes).cast::<c_void>(),
                    bytes.len(),
                    0,
                )
            };
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: threads.0.div_ceil(8) as usize,
                height: threads.1.div_ceil(8) as usize,
                depth: 1,
            },
            MTLSize {
                width: 8,
                height: 8,
                depth: 1,
            },
        );
    }

    /// Runs what was encoded and waits for the GPU to finish it: once this
    /// returns, every texture written holds what the pass wrote.
    pub(crate) fn finish(self) -> Result<(), MetalError> {
        self.encoder.endEncoding();
        self.commands.commit();
        self.commands.waitUntilCompleted();
        if self.commands.status() == MTLCommandBufferStatus::Completed {
            return Ok(());
        }
        Err(MetalError::Execution(
            self.commands
                .error()
                .map(|error| error.localizedDescription().to_string())
                .unwrap_or_else(|| format!("status {:?}", self.commands.status())),
        ))
    }
}

/// Writes `pixels`, `width` bytes a row, into the whole of `texture`, a
/// one-byte-a-pixel texture of that size in memory the CPU writes.
pub(crate) fn fill_bytes(texture: &Texture, pixels: &[u8], width: u32, height: u32) {
    debug_assert_eq!(pixels.len(), width as usize * height as usize);
    // SAFETY: the texture is `width` by `height` of one byte a pixel, in
    // shared memory, and `pixels` holds exactly that many bytes, tightly
    // packed; no command buffer is using it while it is written.
    unsafe {
        texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
            MTLRegion {
                origin: MTLOrigin { x: 0, y: 0, z: 0 },
                size: MTLSize {
                    width: width as usize,
                    height: height as usize,
                    depth: 1,
                },
            },
            0,
            NonNull::from(pixels).cast::<c_void>(),
            width as usize,
        );
    }
}
