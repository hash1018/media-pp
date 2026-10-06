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
use objc2_io_surface::IOSurfaceRef;
use objc2_metal::{
    MTLBarrierScope, MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandBufferStatus,
    MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLOrigin,
    MTLPixelFormat, MTLRegion, MTLSize, MTLStorageMode, MTLTexture, MTLTextureDescriptor,
    MTLTextureUsage,
};
use thiserror::Error as ThisError;

use super::pixel_buffer::PixelBuffer;

/// A texture, as Metal hands one out.
pub(crate) type Texture = Retained<ProtocolObject<dyn MTLTexture>>;
/// A compiled compute kernel.
pub(crate) type Kernel = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
/// A buffer, as Metal hands one out.
pub(crate) type Buffer = Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>;

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

    /// [`Self::kernels`] for a fixed set of names, one kernel each, as an
    /// array to take apart.
    pub(crate) fn kernel_array<const N: usize>(
        &self,
        source: &str,
        names: [&str; N],
    ) -> Result<[Kernel; N], MetalError> {
        Self::array(self.kernels(source, &names)?)
    }

    /// [`Self::kernel_array`] compiled without fast math: every
    /// floating-point operation rounded as written, as the CPU rounds it,
    /// for kernels whose output is to be the CPU's byte for byte. Fast math
    /// is Metal's default, and lets the compiler fuse and reorder.
    pub(crate) fn precise_kernel_array<const N: usize>(
        &self,
        source: &str,
        names: [&str; N],
    ) -> Result<[Kernel; N], MetalError> {
        let options = MTLCompileOptions::new();
        // `mathMode`, which replaces it, is macOS 15 and later; this is
        // every version, and turns the same optimisations off.
        #[allow(deprecated)]
        options.setFastMathEnabled(false);
        Self::array(self.compile(source, &names, Some(&options))?)
    }

    fn array<const N: usize>(made: Vec<Kernel>) -> Result<[Kernel; N], MetalError> {
        made.try_into()
            .map_err(|made: Vec<Kernel>| MetalError::Compile {
                kernel: "(library)".into(),
                reason: format!("{N} kernels asked for, {} made", made.len()),
            })
    }

    /// Compiles `source` and each of `kernels` in it into a pipeline, in the
    /// order they are named.
    pub(crate) fn kernels(
        &self,
        source: &str,
        kernels: &[&str],
    ) -> Result<Vec<Kernel>, MetalError> {
        self.compile(source, kernels, None)
    }

    fn compile(
        &self,
        source: &str,
        kernels: &[&str],
        options: Option<&MTLCompileOptions>,
    ) -> Result<Vec<Kernel>, MetalError> {
        let library = self
            .device
            .newLibraryWithSource_options_error(&NSString::from_str(source), options)
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

    /// A buffer of `bytes` bytes in shared memory: what a kernel writes the
    /// CPU reads where it is, once the pass that wrote it has finished —
    /// on Apple silicon the one memory both use, so nothing is copied.
    pub(crate) fn shared_buffer(&self, bytes: usize) -> Result<Buffer, MetalError> {
        self.device
            .newBufferWithLength_options(bytes, objc2_metal::MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::Unavailable("buffer"))
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
        plane_on(&self.device, buffer, plane, format, usage)
    }

    /// A texture over the whole of `surface`, a surface of one plane, seen
    /// as `format` — its pixels where they are, as [`Self::plane`] sees a
    /// pixel buffer's.
    pub(crate) fn surface(
        &self,
        surface: &IOSurfaceRef,
        format: MTLPixelFormat,
        usage: MTLTextureUsage,
    ) -> Result<Texture, MetalError> {
        let (width, height) = (surface.width() as u32, surface.height() as u32);
        let descriptor = descriptor(format, width, height, usage);
        self.device
            .newTextureWithDescriptor_iosurface_plane(&descriptor, surface, 0)
            .ok_or(MetalError::Texture {
                width,
                height,
                format,
            })
    }

    /// Copies the whole of `from` into `to`, which are the same size and
    /// format, and waits for the GPU to have done it.
    pub(crate) fn copy(&self, from: &Texture, to: &Texture) -> Result<(), MetalError> {
        let commands = self
            .queue
            .commandBuffer()
            .ok_or(MetalError::Unavailable("command buffer"))?;
        let encoder = commands
            .blitCommandEncoder()
            .ok_or(MetalError::Unavailable("blit encoder"))?;
        // SAFETY: both textures are live, the caller's, until this returns,
        // which is after the GPU has finished with them.
        unsafe { encoder.copyFromTexture_toTexture(from, to) };
        encoder.endEncoding();
        run(&commands)
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

/// [`MetalGpu::plane`] on any device — one an application draws with.
pub(crate) fn plane_on(
    device: &ProtocolObject<dyn MTLDevice>,
    buffer: &PixelBuffer,
    plane: usize,
    format: MTLPixelFormat,
    usage: MTLTextureUsage,
) -> Result<Texture, MetalError> {
    let surface = buffer.io_surface().ok_or(MetalError::NotIoSurface)?;
    let (width, height) = buffer.plane_size(plane);
    let descriptor = descriptor(format, width, height, usage);
    // A plane the buffer has, whose size the descriptor is made of.
    device
        .newTextureWithDescriptor_iosurface_plane(&descriptor, &surface, plane)
        .ok_or(MetalError::Texture {
            width,
            height,
            format,
        })
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

    /// Binds `buffer` at buffer index `index` for the dispatches encoded
    /// after this — beside the parameters [`Self::dispatch`] binds at 0.
    ///
    /// The buffer must be alive until [`Self::finish`] returns, as for a
    /// texture.
    #[cfg(feature = "ort-coreml")]
    pub(crate) fn bind_buffer(&mut self, buffer: &Buffer, index: usize) {
        // SAFETY: a live buffer, at an index the kernel declares, read from
        // its start.
        unsafe {
            self.encoder
                .setBuffer_offset_atIndex(Some(buffer), 0, index)
        };
    }

    /// Runs `kernel` as `groups` threadgroups of `per_group` threads each,
    /// with `textures` bound from index 0, `bytes` at buffer 0 and
    /// `buffers`, each from its byte offset, from buffer 1 — after
    /// everything already encoded has written its textures and buffers. The
    /// kernel bounds-checks its own threads.
    ///
    /// Each texture and buffer must be alive until [`Self::finish`]
    /// returns.
    pub(crate) fn dispatch_groups(
        &mut self,
        kernel: &Kernel,
        textures: &[&Texture],
        buffers: &[(&Buffer, usize)],
        bytes: Option<&[u8]>,
        groups: (usize, usize, usize),
        per_group: (usize, usize, usize),
    ) {
        let encoder = &self.encoder;
        encoder.memoryBarrierWithScope(MTLBarrierScope::Textures | MTLBarrierScope::Buffers);
        encoder.setComputePipelineState(kernel);
        for (index, texture) in textures.iter().enumerate() {
            // SAFETY: a live texture, at an index the kernel declares.
            unsafe { encoder.setTexture_atIndex(Some(texture), index) };
        }
        for (index, (buffer, offset)) in buffers.iter().enumerate() {
            // SAFETY: a live buffer, at an index the kernel declares, read
            // from an offset the caller keeps within it.
            unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), *offset, index + 1) };
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
                width: groups.0,
                height: groups.1,
                depth: groups.2,
            },
            MTLSize {
                width: per_group.0,
                height: per_group.1,
                depth: per_group.2,
            },
        );
    }

    /// Puts `drawable` on the screen once what was encoded has drawn it.
    pub(crate) fn present(&self, drawable: &ProtocolObject<dyn objc2_metal::MTLDrawable>) {
        self.commands.presentDrawable(drawable);
    }

    /// Runs what was encoded and waits for the GPU to finish it: once this
    /// returns, every texture written holds what the pass wrote.
    pub(crate) fn finish(self) -> Result<(), MetalError> {
        self.encoder.endEncoding();
        run(&self.commands)
    }
}

/// Runs `commands` and waits for the GPU to finish them.
fn run(commands: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<(), MetalError> {
    commands.commit();
    commands.waitUntilCompleted();
    if commands.status() == MTLCommandBufferStatus::Completed {
        return Ok(());
    }
    Err(MetalError::Execution(
        commands
            .error()
            .map(|error| error.localizedDescription().to_string())
            .unwrap_or_else(|| format!("status {:?}", commands.status())),
    ))
}

/// Writes `pixels` into the whole of `texture`, `width` by `height` pixels
/// in memory the CPU writes, each row `bytes_per_row` apart in `pixels` —
/// of which the texture's own row size is read.
///
/// # Panics
///
/// If `pixels` is shorter than `height` rows of `bytes_per_row`, the last
/// one only as long as the texture's row: the caller has checked.
pub(crate) fn write_texture(
    texture: &Texture,
    pixels: &[u8],
    bytes_per_row: usize,
    width: u32,
    height: u32,
) {
    let texel = match texture.pixelFormat() {
        MTLPixelFormat::RG8Unorm => 2,
        MTLPixelFormat::BGRA8Unorm | MTLPixelFormat::RGBA8Unorm => 4,
        _ => 1,
    };
    let row = width as usize * texel;
    let needed = (height as usize).saturating_sub(1) * bytes_per_row + row;
    assert!(
        height == 0 || (bytes_per_row >= row && pixels.len() >= needed),
        "{} bytes for {height} rows of {row} bytes, {bytes_per_row} apart",
        pixels.len()
    );
    // SAFETY: the texture is `width` by `height` in shared memory, and
    // `pixels` holds that many rows `bytes_per_row` apart — checked above;
    // no command buffer is using it while it is written.
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
            bytes_per_row,
        );
    }
}
