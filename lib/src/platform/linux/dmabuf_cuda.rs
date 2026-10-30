//! Imports a DMA-BUF into a CUDA surface, for
//! [`crate::elements::PipeWireScreenCaptureSource`]'s GPU capture mode and
//! for [`crate::elements::CudaDmaBufSource`].
//!
//! # Why this goes through OpenGL
//!
//! The obvious path — `eglCreateImage(EGL_LINUX_DMA_BUF_EXT)` then
//! `cuGraphicsEGLRegisterImage` — **does not work on desktop NVIDIA**. The
//! image is created successfully and CUDA then refuses to register it with
//! `CUDA_ERROR_INVALID_VALUE`, for every combination of modifier/no-modifier
//! attributes, `EGL_IMAGE_PRESERVED_KHR`, register flags, and EGL display
//! (both the `EGL_PLATFORM_DEVICE_EXT` display whose `EGL_CUDA_DEVICE_NV`
//! matches the CUDA device, and the default one). Registering the *external*
//! GL texture the image is bound to fails the same way. Measured on driver
//! 595.84 with mutter's screencast node; CUDA's EGL interop is a Tegra-era
//! path that this configuration does not honour.
//!
//! What does work, and is what this module implements:
//!
//! ```text
//! dma-buf fd -> eglCreateImage -> glEGLImageTargetTexture2DOES -> FBO
//!            -> glReadPixels(GL_BGRA) -> pixel buffer object
//!            -> cuGraphicsGLRegisterBuffer/MapResources -> CUdeviceptr
//!            -> cuMemcpy2D -> the CUDA surface of an AV_PIX_FMT_CUDA frame
//! ```
//!
//! A buffer the driver can only import as `GL_TEXTURE_EXTERNAL_OES` — which
//! on NVIDIA is every `DRM_FORMAT_MOD_LINEAR` one — cannot be attached to a
//! framebuffer, so
//! [`DmaBufCudaImporter::draw_into`] draws it first: a full-screen triangle
//! sampling the external texture into a texture of its own, which is then
//! read back the same way. The same draw is where premultiplied colour is
//! divided back out.
//!
//! A plain GL texture as the copy destination (`glCopyTexSubImage2D` into
//! `glTexStorage2D(GL_RGBA8)`) registers with CUDA but the copy itself raises
//! `GL_INVALID_OPERATION`, so the pixel buffer is the destination that
//! actually carries pixels. Both copies stay on the GPU: nothing here maps
//! anything into system memory.
//!
//! # Thread affinity
//!
//! The EGL context is made current on the thread that constructs this and is
//! never moved, so this type is deliberately **not** `Send`: it lives on
//! `PipeWireScreenCaptureSource`'s PipeWire loop thread, which is the only
//! thread that touches a captured buffer, or on `CudaDmaBufSource`'s import
//! thread.

use std::ffi::{CString, c_char, c_int, c_uint, c_void};

use thiserror::Error as ThisError;

/// `DRM_FORMAT_XRGB8888` — what a screen capture is: the `BGRx`/`BGRA` SPA
/// formats this crate's capture negotiates, read with the fourth byte
/// ignored.
pub(crate) const DRM_FORMAT_XRGB8888: u32 = 0x3432_5258;

/// `DRM_FORMAT_ARGB8888` — the same bytes with the fourth one kept as alpha,
/// for a producer whose pictures have transparency in them.
pub(crate) const DRM_FORMAT_ARGB8888: u32 = 0x3432_5241;

/// What can go wrong bringing a captured DMA-BUF into CUDA memory, through
/// an EGL image on the NVIDIA driver's EGL device.
#[derive(Debug, ThisError)]
pub enum DmaBufCudaError {
    /// A library the import needs — named — would not load.
    #[error("failed to load {0}")]
    Library(&'static str),

    /// libEGL lacks the named entry point.
    #[error("{0} is missing from libEGL")]
    EglSymbol(&'static str),

    /// libGLESv2 lacks the named entry point.
    #[error("{0} is missing from libGLESv2")]
    GlSymbol(&'static str),

    /// libcuda lacks the named entry point.
    #[error("{0} is missing from libcuda")]
    CudaSymbol(&'static str),

    /// No EGL device is the GPU CUDA device 0 is: the driver is not
    /// NVIDIA's, or exposes no EGL device for it.
    #[error("no EGL device is backed by CUDA device 0")]
    NoCudaEglDevice,

    /// `eglInitialize` failed, with the EGL error code.
    #[error("eglInitialize failed (EGL error {0:#x})")]
    EglInit(c_int),

    /// The EGL device lacks the named extension.
    #[error("the EGL device is missing {0}")]
    EglExtension(&'static str),

    /// `eglCreateContext` failed, with the EGL error code.
    #[error("eglCreateContext failed (EGL error {0:#x})")]
    EglContext(c_int),

    /// `eglMakeCurrent` failed, with the EGL error code.
    #[error("eglMakeCurrent failed (EGL error {0:#x})")]
    EglMakeCurrent(c_int),

    /// The driver offers no DMA-BUF format modifier for XRGB8888, so no
    /// capture could be imported.
    #[error("the driver accepts no DMA-BUF modifier for XRGB8888")]
    NoModifiers,

    /// `eglCreateImage` refused the captured DMA-BUF, with the EGL error
    /// code.
    #[error("eglCreateImage failed for the captured DMA-BUF (EGL error {0:#x})")]
    CreateImage(c_int),

    /// The named GL call failed, with the GL error code.
    #[error("{0} failed (GL error {1:#x})")]
    Gl(&'static str, c_uint),

    /// The framebuffer over the imported image is incomplete, with GL's
    /// status.
    #[error("the DMA-BUF framebuffer is incomplete ({0:#x})")]
    FramebufferIncomplete(c_uint),

    /// The named CUDA call failed, with its `CUresult`.
    #[error("{0} failed (CUresult {1})")]
    Cuda(&'static str, c_int),

    #[error("the import shader did not build: {0}")]
    Shader(String),
}

/// One PipeWire DMA-BUF plane, as `process` reads it off the buffer.
///
/// Single-plane only: the negotiated format is packed BGRx/BGRA, which
/// mutter delivers as one plane. A multi-plane buffer would be a format this
/// element never asked for.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DmaBufPlane {
    pub(crate) fd: c_int,
    pub(crate) offset: u32,
    pub(crate) stride: i32,
    /// The modifier the stream fixated on, which the import has to be told —
    /// the same buffer is uninterpretable without it.
    pub(crate) modifier: u64,
    /// What the four bytes of a pixel are — [`DRM_FORMAT_XRGB8888`] or
    /// [`DRM_FORMAT_ARGB8888`].
    pub(crate) fourcc: u32,
}

/// The destination surface of an import: an `AV_PIX_FMT_CUDA` frame's
/// `data[0]`/`linesize[0]`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CudaBgraSurface {
    pub(crate) pixels: u64,
    pub(crate) pitch: usize,
}

impl CudaBgraSurface {
    /// Reads the surface out of a CUDA-resident BGRA frame. The caller has
    /// already established that this frame came from its own BGRA frames
    /// context, so the only thing left to reject is a frame with no pointer.
    pub(crate) fn from_frame(frame: &ffmpeg_next::frame::Video) -> Option<Self> {
        // SAFETY: `frame` is a live `frame::Video`, so `as_ptr` yields an
        // initialized `AVFrame`. `data` and `linesize` are plain arrays in it and
        // index 0 exists for every pixel format; whether the values are usable is
        // what the check below decides.
        let (pixels, pitch) = unsafe {
            let ptr = frame.as_ptr();
            ((*ptr).data[0], (*ptr).linesize[0])
        };
        (!pixels.is_null() && pitch > 0).then_some(Self {
            pixels: pixels as u64,
            pitch: pitch as usize,
        })
    }
}

/// Reads a captured DMA-BUF into a CUDA surface, through EGL and GL.
///
/// # An EGLImage per import, and never a cached one
///
/// The obvious optimization here is to cache the `EGLImage` per buffer: a
/// screencast stream cycles through a pool of a dozen or so DMA-BUFs, each
/// with a stable fd, so the same handful of images would serve for the whole
/// session and each frame would cost only a re-bind. This did exactly that,
/// and it was wrong.
///
/// Binding a cached image to the texture again does not reliably re-read the
/// buffer behind it. What `glReadPixels` then returns is what that buffer held
/// when the image was made — a picture from that buffer's last trip through
/// the pool, which is to say about a dozen frames old. It happens
/// occasionally and under load, so a recording carries the odd stale frame
/// and reads as judder rather than as anything broken.
///
/// Measured by burning a frame counter into the captured screen and reading
/// it back out of the imported surface, alternating between a cached image
/// and a fresh one every 300 imports so both meet the same load:
///
/// ```text
///           cached   fresh
/// run 1        5       0
/// run 2        5       0
/// run 3        4       0
/// run 4        1       0
/// run 5        2       0
/// run 6        5       0
/// ```
///
/// Zero, every run, on about seven thousand frames. Creating the image costs
/// nothing measurable — the fresh arm captured slightly *more* frames per
/// second than the cached one — because what it describes is a buffer that
/// already exists.
///
/// The cache also needed telling when the compositor reallocated its pool,
/// since an fd may come back describing different memory; that whole
/// apparatus went with it.
pub(crate) struct DmaBufCudaImporter {
    egl: Egl,
    gl: Gl,
    cuda: Cuda,
    display: EglDisplay,
    context: *mut c_void,
    /// The modifiers this driver will accept for XRGB8888, in the order EGL
    /// reported them — what the stream offers the compositor.
    #[cfg_attr(not(feature = "pipewire-screen-capture"), allow(dead_code))]
    modifiers: Vec<u64>,
    texture: c_uint,
    framebuffer: c_uint,
    /// The CUDA-registered pixel buffer the readback lands in, sized to the
    /// current capture. `None` until the first frame fixes a size.
    pixel_buffer: Option<PixelBuffer>,
    /// What [`DmaBufCudaImporter::draw_into`] draws with. `None` until it is
    /// first called, which a screen capture never does.
    drawing: Option<Drawing>,
}

/// The shader, and the two textures it draws from and into.
struct Drawing {
    program: c_uint,
    /// Where the `unpremultiply` uniform is.
    unpremultiply: c_int,
    /// The imported buffer, bound as `GL_TEXTURE_EXTERNAL_OES`.
    source: c_uint,
    /// What the draw lands in: `GL_RGBA8`, `width` x `height`. Immutable
    /// storage, so a new size is a new texture.
    target: c_uint,
    width: u32,
    height: u32,
}

/// A triangle that covers the whole target, from no vertex data at all.
const VERTEX_SHADER: &str = "#version 300 es
void main() {
    vec2 corner = vec2(float((gl_VertexID & 1) << 2), float((gl_VertexID & 2) << 1)) - 1.0;
    gl_Position = vec4(corner, 0.0, 1.0);
}
";

/// One texel of the buffer per pixel of the target, row 0 to row 0 — which,
/// read back from the bottom of the target up, is the buffer's own row order.
///
/// Written out as BGRA, so an `GL_RGBA` readback — the one form every GLES
/// implementation accepts — lands as the BGRA the CUDA frame holds.
///
/// Division rounds the way the exact integer form does: `round(c / a)` over
/// normalized values is `round(C * 255 / A)`, and a colour above its own
/// alpha — which premultiplied colour cannot be — saturates at 255.
const FRAGMENT_SHADER: &str = "#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
precision highp float;
uniform samplerExternalOES picture;
uniform bool unpremultiply;
out vec4 color;
void main() {
    vec4 texel = texture(picture, gl_FragCoord.xy / vec2(textureSize(picture, 0)));
    if (unpremultiply) {
        texel.rgb = texel.a > 0.0 ? texel.rgb / texel.a : vec3(0.0);
    }
    color = texel.bgra;
}
";

/// The readback destination: a GL buffer object registered with CUDA once,
/// then mapped per frame.
struct PixelBuffer {
    buffer: c_uint,
    resource: *mut c_void,
    width: u32,
    height: u32,
}

/// Completes the two fallible steps that turn a fresh GL name into an owned
/// interop resource. Until both succeed, `buffer` is temporary and `delete`
/// must reclaim it on the caller's behalf.
fn initialize_temporary_buffer<T, E>(
    buffer: c_uint,
    allocate: impl FnOnce() -> Result<(), E>,
    register: impl FnOnce(c_uint) -> Result<T, E>,
    delete: impl Fn(c_uint),
) -> Result<T, E> {
    if let Err(error) = allocate() {
        delete(buffer);
        return Err(error);
    }
    match register(buffer) {
        Ok(resource) => Ok(resource),
        Err(error) => {
            delete(buffer);
            Err(error)
        }
    }
}

impl DmaBufCudaImporter {
    /// Opens the EGL device that backs CUDA device 0, makes a surfaceless
    /// GLES context current on the calling thread, and queries the DMA-BUF
    /// modifiers the driver can import.
    ///
    /// Nothing is allocated for a capture yet: the size is only known once
    /// the stream has negotiated one, and it changes when a captured window
    /// is resized.
    pub(crate) fn new() -> Result<Self, DmaBufCudaError> {
        let egl = Egl::load()?;
        let gl = Gl::load(&egl)?;
        let cuda = Cuda::load()?;

        let display = egl.cuda_device_display()?;
        let extensions = egl.extensions(display);
        for required in [
            "EGL_EXT_image_dma_buf_import",
            "EGL_EXT_image_dma_buf_import_modifiers",
            "EGL_KHR_no_config_context",
            "EGL_KHR_surfaceless_context",
        ] {
            if !extensions.split(' ').any(|ext| ext == required) {
                return Err(DmaBufCudaError::EglExtension(match required {
                    "EGL_EXT_image_dma_buf_import" => "EGL_EXT_image_dma_buf_import",
                    "EGL_EXT_image_dma_buf_import_modifiers" => {
                        "EGL_EXT_image_dma_buf_import_modifiers"
                    }
                    "EGL_KHR_no_config_context" => "EGL_KHR_no_config_context",
                    _ => "EGL_KHR_surfaceless_context",
                }));
            }
        }

        let modifiers = egl.dma_buf_modifiers(display, DRM_FORMAT_XRGB8888);
        if modifiers.is_empty() {
            return Err(DmaBufCudaError::NoModifiers);
        }

        let context = egl.create_context(display)?;

        // After the context is current: both names are created in it.
        let texture = gl.gen_texture();
        let framebuffer = gl.gen_framebuffer();

        Ok(Self {
            egl,
            gl,
            cuda,
            display,
            context,
            modifiers,
            texture,
            framebuffer,
            pixel_buffer: None,
            drawing: None,
        })
    }

    /// The DMA-BUF modifiers to offer the compositor, most preferred first.
    ///
    /// An import of a buffer allocated with anything else would fail, so
    /// these are exactly what the stream may fixate on.
    #[cfg_attr(not(feature = "pipewire-screen-capture"), allow(dead_code))]
    pub(crate) fn modifiers(&self) -> &[u64] {
        &self.modifiers
    }

    /// Copies one captured DMA-BUF into `destination`, which must be a
    /// `width` x `height` CUDA BGRA surface.
    #[cfg_attr(not(feature = "pipewire-screen-capture"), allow(dead_code))]
    pub(crate) fn copy_into(
        &mut self,
        plane: DmaBufPlane,
        width: u32,
        height: u32,
        destination: CudaBgraSurface,
    ) -> Result<(), DmaBufCudaError> {
        let (buffer, resource) = self.ensure_pixel_buffer(width, height)?;
        // A new image every time, kept only for this one import — see
        // [`DmaBufCudaImporter`] on why these are not cached.
        let image = self
            .egl
            .create_dma_buf_image(self.display, plane, width, height)?;
        let result = self.read_through(image, buffer, resource, width, height, destination);
        self.egl.destroy_image(self.display, image);
        result
    }

    /// [`Self::copy_into`] for a buffer that has to be drawn to be read: one
    /// the driver imports only as an external texture, or one whose colour
    /// is premultiplied, which `unpremultiply` divides back out.
    ///
    /// Costs a draw more than `copy_into`, and works for any buffer that
    /// one does.
    pub(crate) fn draw_into(
        &mut self,
        plane: DmaBufPlane,
        width: u32,
        height: u32,
        destination: CudaBgraSurface,
        unpremultiply: bool,
    ) -> Result<(), DmaBufCudaError> {
        let (buffer, resource) = self.ensure_pixel_buffer(width, height)?;
        self.ensure_drawing(width, height)?;
        let image = self
            .egl
            .create_dma_buf_image(self.display, plane, width, height)?;
        let result = self
            .draw(image, width, height, unpremultiply)
            .and_then(|()| self.read_back(GL_RGBA, buffer, resource, width, height, destination));
        self.unbind();
        self.egl.destroy_image(self.display, image);
        result
    }

    /// Draws `image` into the drawing's target and leaves that attached to
    /// the framebuffer, for [`Self::read_back`].
    fn draw(
        &self,
        image: EglImage,
        width: u32,
        height: u32,
        unpremultiply: bool,
    ) -> Result<(), DmaBufCudaError> {
        let drawing = self.drawing.as_ref().expect("built by draw_into");
        self.gl
            .bind_texture(GL_TEXTURE_EXTERNAL_OES, drawing.source);
        self.gl.image_target_texture(GL_TEXTURE_EXTERNAL_OES, image);
        self.gl.check("glEGLImageTargetTexture2DOES")?;

        self.gl.bind_framebuffer(GL_FRAMEBUFFER, self.framebuffer);
        self.gl.framebuffer_texture_2d(
            GL_FRAMEBUFFER,
            GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D,
            drawing.target,
            0,
        );
        let status = self.gl.check_framebuffer_status(GL_FRAMEBUFFER);
        if status != GL_FRAMEBUFFER_COMPLETE {
            return Err(DmaBufCudaError::FramebufferIncomplete(status));
        }

        self.gl.use_program(drawing.program);
        self.gl
            .uniform_1i(drawing.unpremultiply, c_int::from(unpremultiply));
        self.gl.viewport(0, 0, width as c_int, height as c_int);
        self.gl.draw_arrays(GL_TRIANGLES, 0, 3);
        self.gl.check("glDrawArrays")
    }

    /// Builds the shader the first time, and the target texture whenever the
    /// size is not the one it has.
    fn ensure_drawing(&mut self, width: u32, height: u32) -> Result<(), DmaBufCudaError> {
        if self.drawing.is_none() {
            let program = self.gl.build_program(VERTEX_SHADER, FRAGMENT_SHADER)?;
            let unpremultiply = self.gl.uniform_location(program, c"unpremultiply");
            let source = self.gl.gen_texture();
            self.gl.bind_texture(GL_TEXTURE_EXTERNAL_OES, source);
            // One texel per pixel, so nothing is ever between two of them —
            // but the default is to blend, and exactly is cheaper to trust.
            for filter in [GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER] {
                self.gl
                    .tex_parameter(GL_TEXTURE_EXTERNAL_OES, filter, GL_NEAREST as c_int);
            }
            self.gl.bind_texture(GL_TEXTURE_EXTERNAL_OES, 0);
            // On by default in GLES, and a dithered picture is not the one
            // that was handed over.
            self.gl.disable(GL_DITHER);
            self.drawing = Some(Drawing {
                program,
                unpremultiply,
                source,
                target: 0,
                width: 0,
                height: 0,
            });
        }
        let drawing = self.drawing.as_mut().expect("built above");
        if drawing.target != 0 && drawing.width == width && drawing.height == height {
            return Ok(());
        }
        if drawing.target != 0 {
            self.gl.delete_texture(drawing.target);
            drawing.target = 0;
        }
        let target = self.gl.gen_texture();
        self.gl.bind_texture(GL_TEXTURE_2D, target);
        self.gl
            .tex_storage_2d(GL_TEXTURE_2D, 1, GL_RGBA8, width as c_int, height as c_int);
        self.gl.bind_texture(GL_TEXTURE_2D, 0);
        if let Err(error) = self.gl.check("glTexStorage2D") {
            self.gl.delete_texture(target);
            return Err(error);
        }
        drawing.target = target;
        drawing.width = width;
        drawing.height = height;
        Ok(())
    }

    /// Reads the desktop through `image` and lands it in `destination`.
    ///
    /// Split out so that [`DmaBufCudaImporter::copy_into`] can destroy the
    /// image on every path out of this, of which there are several.
    #[allow(clippy::too_many_arguments)]
    fn read_through(
        &mut self,
        image: EglImage,
        buffer: c_uint,
        resource: *mut c_void,
        width: u32,
        height: u32,
        destination: CudaBgraSurface,
    ) -> Result<(), DmaBufCudaError> {
        // One texture rather than one per buffer: it keeps the GL state this
        // has to restore to a single object.
        self.gl.bind_texture(GL_TEXTURE_2D, self.texture);
        self.gl.image_target_texture(GL_TEXTURE_2D, image);
        self.gl.check("glEGLImageTargetTexture2DOES")?;

        self.gl.bind_framebuffer(GL_FRAMEBUFFER, self.framebuffer);
        self.gl.framebuffer_texture_2d(
            GL_FRAMEBUFFER,
            GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D,
            self.texture,
            0,
        );
        let status = self.gl.check_framebuffer_status(GL_FRAMEBUFFER);
        if status != GL_FRAMEBUFFER_COMPLETE {
            self.unbind();
            return Err(DmaBufCudaError::FramebufferIncomplete(status));
        }
        let result = self.read_back(GL_BGRA_EXT, buffer, resource, width, height, destination);
        self.unbind();
        result
    }

    /// Reads the framebuffer's picture into `buffer`, as `format`, and copies
    /// that into `destination`. The caller unbinds.
    ///
    /// `format` is what gives BGRA bytes: `GL_BGRA_EXT` for an imported
    /// buffer, and `GL_RGBA` for a drawn one — a texture of GL's own, which
    /// reads as RGBA, and which the shader wrote in BGRA order.
    #[allow(clippy::too_many_arguments)]
    fn read_back(
        &self,
        format: c_uint,
        buffer: c_uint,
        resource: *mut c_void,
        width: u32,
        height: u32,
        destination: CudaBgraSurface,
    ) -> Result<(), DmaBufCudaError> {
        self.gl.bind_buffer(GL_PIXEL_PACK_BUFFER, buffer);
        // Reads bottom-up in window coordinates, which for an FBO-attached
        // texture is its first row first — the DMA-BUF's own row order, so
        // the readback lands top-down with no flip.
        self.gl.read_pixels(
            0,
            0,
            width as c_int,
            height as c_int,
            format,
            GL_UNSIGNED_BYTE,
            std::ptr::null_mut(),
        );
        let read_result = self.gl.check("glReadPixels");
        self.gl.bind_buffer(GL_PIXEL_PACK_BUFFER, 0);
        read_result?;

        // Mapping is the synchronization point: CUDA guarantees GL work
        // issued before it completes first, so no glFinish is needed.
        let source = self.cuda.map_buffer(resource);
        let result = source.and_then(|source| {
            self.cuda.copy_2d(
                source,
                width as usize * 4,
                destination.pixels,
                destination.pitch,
                width as usize * 4,
                height as usize,
            )
        });
        let unmapped = self.cuda.unmap(resource);
        result.and(unmapped)
    }

    /// Releases every GL binding this took, so the next frame starts from the
    /// same state regardless of which path returned.
    /// Writes `pixels` — `width` x `height` packed BGRA — into the buffer
    /// `plane` describes, through GL, and waits for the write to land: how a
    /// test fills a buffer in a layout the CPU cannot map.
    #[cfg(test)]
    pub(crate) fn write(
        &mut self,
        plane: DmaBufPlane,
        width: u32,
        height: u32,
        pixels: &[u8],
    ) -> Result<(), DmaBufCudaError> {
        assert_eq!(pixels.len(), width as usize * height as usize * 4);
        // As RGBA, the one upload format GLES takes into any four-byte
        // texture: the buffer's own BGRA, swapped into it.
        let pixels: Vec<u8> = pixels
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|&[blue, green, red, alpha]| [red, green, blue, alpha])
            .collect();
        let image = self
            .egl
            .create_dma_buf_image(self.display, plane, width, height)?;
        self.gl.bind_texture(GL_TEXTURE_2D, self.texture);
        self.gl.image_target_texture(GL_TEXTURE_2D, image);
        let result = self
            .gl
            .check("glEGLImageTargetTexture2DOES")
            .and_then(|()| {
                // SAFETY: `pixels` is `width * height` four-byte pixels, read
                // by the call and not kept; the texture bound is the image's.
                unsafe {
                    (self.gl.tex_sub_image_2d)(
                        GL_TEXTURE_2D,
                        0,
                        0,
                        0,
                        width as c_int,
                        height as c_int,
                        GL_RGBA,
                        GL_UNSIGNED_BYTE,
                        pixels.as_ptr().cast(),
                    );
                    (self.gl.finish)();
                }
                self.gl.check("glTexSubImage2D")
            });
        self.unbind();
        self.egl.destroy_image(self.display, image);
        result
    }

    fn unbind(&self) {
        self.gl.bind_framebuffer(GL_FRAMEBUFFER, 0);
        self.gl.bind_texture(GL_TEXTURE_2D, 0);
        if self.drawing.is_some() {
            self.gl.bind_texture(GL_TEXTURE_EXTERNAL_OES, 0);
            self.gl.use_program(0);
        }
    }

    /// Returns the GL name and CUDA resource of a pixel buffer sized for
    /// `width` x `height`, rebuilding it when the capture has been
    /// renegotiated to a new size.
    fn ensure_pixel_buffer(
        &mut self,
        width: u32,
        height: u32,
    ) -> Result<(c_uint, *mut c_void), DmaBufCudaError> {
        let matches = self
            .pixel_buffer
            .as_ref()
            .is_some_and(|pbo| pbo.width == width && pbo.height == height);
        if !matches {
            // Build and register the replacement before taking the old one.
            // Every failure path in `create_pixel_buffer` deletes its temporary
            // GL name, and the previously working registration remains owned
            // here until the replacement is complete.
            let replacement = self.create_pixel_buffer(width, height)?;
            if let Some(old) = self.pixel_buffer.replace(replacement) {
                let _ = self.cuda.unregister(old.resource);
                self.gl.delete_buffer(old.buffer);
            }
        }
        let pixel_buffer = self.pixel_buffer.as_ref().expect("built above");
        Ok((pixel_buffer.buffer, pixel_buffer.resource))
    }

    /// Allocates one replacement PBO without publishing it into `self` until
    /// both GL allocation and CUDA registration have succeeded.
    fn create_pixel_buffer(&self, width: u32, height: u32) -> Result<PixelBuffer, DmaBufCudaError> {
        let buffer = self.gl.gen_buffer();
        let resource = initialize_temporary_buffer(
            buffer,
            || {
                self.gl.bind_buffer(GL_PIXEL_PACK_BUFFER, buffer);
                self.gl.buffer_data(
                    GL_PIXEL_PACK_BUFFER,
                    (width as isize) * (height as isize) * 4,
                    std::ptr::null(),
                    GL_STREAM_READ,
                );
                self.gl.bind_buffer(GL_PIXEL_PACK_BUFFER, 0);
                self.gl.check("glBufferData")
            },
            |buffer| self.cuda.register_buffer(buffer),
            |buffer| self.gl.delete_buffer(buffer),
        )?;
        Ok(PixelBuffer {
            buffer,
            resource,
            width,
            height,
        })
    }
}

impl Drop for DmaBufCudaImporter {
    fn drop(&mut self) {
        if let Some(pbo) = self.pixel_buffer.take() {
            let _ = self.cuda.unregister(pbo.resource);
            self.gl.delete_buffer(pbo.buffer);
        }
        if let Some(drawing) = self.drawing.take() {
            self.gl.delete_program(drawing.program);
            self.gl.delete_texture(drawing.source);
            if drawing.target != 0 {
                self.gl.delete_texture(drawing.target);
            }
        }
        self.gl.delete_framebuffer(self.framebuffer);
        self.gl.delete_texture(self.texture);
        self.egl.release_context(self.display, self.context);
    }
}

// ---------------------------------------------------------------------------
// The three libraries this loads at runtime.
//
// `dlopen` rather than a link-time dependency, so a build of this crate needs
// no EGL/GLES/CUDA development packages and a machine without them still
// builds and runs everything except GPU capture, which fails at `open` with
// one of the errors above.
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn dlopen(file: *const c_char, mode: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
}

const RTLD_NOW: c_int = 2;

type EglDisplay = *mut c_void;
type EglImage = *mut c_void;
type EglDevice = *mut c_void;
/// `EGLAttrib` — pointer-sized, unlike the `EGLint` list the older
/// `eglCreateImageKHR` takes.
type EglAttrib = isize;

const EGL_NONE: EglAttrib = 0x3038;
const EGL_NONE_INT: c_int = 0x3038;
const EGL_EXTENSIONS: c_int = 0x3055;
const EGL_WIDTH: EglAttrib = 0x3057;
const EGL_HEIGHT: EglAttrib = 0x3056;
const EGL_LINUX_DRM_FOURCC_EXT: EglAttrib = 0x3271;
const EGL_DMA_BUF_PLANE0_FD_EXT: EglAttrib = 0x3272;
const EGL_DMA_BUF_PLANE0_OFFSET_EXT: EglAttrib = 0x3273;
const EGL_DMA_BUF_PLANE0_PITCH_EXT: EglAttrib = 0x3274;
const EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT: EglAttrib = 0x3443;
const EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT: EglAttrib = 0x3444;
const EGL_LINUX_DMA_BUF_EXT: c_uint = 0x3270;
const EGL_PLATFORM_DEVICE_EXT: c_uint = 0x313F;
const EGL_CUDA_DEVICE_NV: c_int = 0x323A;
const EGL_OPENGL_ES_API: c_uint = 0x30A0;
const EGL_CONTEXT_CLIENT_VERSION: c_int = 0x3098;

const GL_TEXTURE_2D: c_uint = 0x0DE1;
const GL_FRAMEBUFFER: c_uint = 0x8D40;
const GL_COLOR_ATTACHMENT0: c_uint = 0x8CE0;
const GL_FRAMEBUFFER_COMPLETE: c_uint = 0x8CD5;
const GL_PIXEL_PACK_BUFFER: c_uint = 0x88EB;
const GL_STREAM_READ: c_uint = 0x88E1;
const GL_BGRA_EXT: c_uint = 0x80E1;
const GL_UNSIGNED_BYTE: c_uint = 0x1401;
const GL_NO_ERROR: c_uint = 0;
const GL_RGBA: c_uint = 0x1908;
const GL_RGBA8: c_uint = 0x8058;
const GL_TEXTURE_EXTERNAL_OES: c_uint = 0x8D65;
const GL_TEXTURE_MIN_FILTER: c_uint = 0x2801;
const GL_TEXTURE_MAG_FILTER: c_uint = 0x2800;
const GL_NEAREST: c_uint = 0x2600;
const GL_TRIANGLES: c_uint = 0x0004;
const GL_DITHER: c_uint = 0x0BD0;
const GL_VERTEX_SHADER: c_uint = 0x8B31;
const GL_FRAGMENT_SHADER: c_uint = 0x8B30;
const GL_COMPILE_STATUS: c_uint = 0x8B81;
const GL_LINK_STATUS: c_uint = 0x8B82;

/// `CU_MEMORYTYPE_DEVICE`.
const CU_MEMORYTYPE_DEVICE: c_uint = 2;

/// SAFETY: every caller passes a pointer freshly resolved for exactly the
/// signature the field it is assigned to declares, taken from the EGL/GLES/CUDA
/// headers.
unsafe fn cast<T: Copy>(ptr: *mut c_void) -> T {
    debug_assert_eq!(
        std::mem::size_of::<T>(),
        std::mem::size_of::<*mut c_void>(),
        "function pointers are pointer-sized"
    );
    // SAFETY: the contract above makes `ptr` a function pointer of type `T`,
    // and the assert confirms `T` is pointer-sized, so this copies the whole
    // value rather than reading past `ptr`.
    unsafe { std::mem::transmute_copy(&ptr) }
}

fn open_library(name: &str) -> Option<*mut c_void> {
    let name = CString::new(name).ok()?;
    // SAFETY: `name` is a live `CString`, so the pointer is NUL-terminated for
    // the length of the call. `dlopen` returns either a handle or null, and the
    // caller keeps only the non-null case.
    let handle = unsafe { dlopen(name.as_ptr(), RTLD_NOW) };
    (!handle.is_null()).then_some(handle)
}

fn raw_symbol(lib: *mut c_void, name: &str) -> Option<*mut c_void> {
    let name = CString::new(name).ok()?;
    // SAFETY: `lib` is a handle from `open_library` and is still open —
    // nothing in this module ever calls `dlclose` — and `name` is a live
    // NUL-terminated `CString`.
    let symbol = unsafe { dlsym(lib, name.as_ptr()) };
    (!symbol.is_null()).then_some(symbol)
}

struct Egl {
    get_proc: unsafe extern "C" fn(*const c_char) -> *mut c_void,
    query_string: unsafe extern "C" fn(EglDisplay, c_int) -> *const c_char,
    get_error: unsafe extern "C" fn() -> c_int,
    initialize: unsafe extern "C" fn(EglDisplay, *mut c_int, *mut c_int) -> c_uint,
    bind_api: unsafe extern "C" fn(c_uint) -> c_uint,
    create_context:
        unsafe extern "C" fn(EglDisplay, *mut c_void, *mut c_void, *const c_int) -> *mut c_void,
    destroy_context: unsafe extern "C" fn(EglDisplay, *mut c_void) -> c_uint,
    make_current: unsafe extern "C" fn(EglDisplay, *mut c_void, *mut c_void, *mut c_void) -> c_uint,
    create_image: unsafe extern "C" fn(
        EglDisplay,
        *mut c_void,
        c_uint,
        *mut c_void,
        *const EglAttrib,
    ) -> EglImage,
    destroy_image: unsafe extern "C" fn(EglDisplay, EglImage) -> c_uint,
    query_devices: unsafe extern "C" fn(c_int, *mut EglDevice, *mut c_int) -> c_uint,
    query_device_attrib: unsafe extern "C" fn(EglDevice, c_int, *mut EglAttrib) -> c_uint,
    get_platform_display: unsafe extern "C" fn(c_uint, *mut c_void, *const EglAttrib) -> EglDisplay,
    query_dma_buf_modifiers:
        unsafe extern "C" fn(EglDisplay, c_int, c_int, *mut u64, *mut c_uint, *mut c_int) -> c_uint,
}

impl Egl {
    fn load() -> Result<Self, DmaBufCudaError> {
        let lib = open_library("libEGL.so.1").ok_or(DmaBufCudaError::Library("libEGL.so.1"))?;
        let get_proc_ptr = raw_symbol(lib, "eglGetProcAddress")
            .ok_or(DmaBufCudaError::EglSymbol("eglGetProcAddress"))?;
        // SAFETY: `get_proc_ptr` was resolved by the name `eglGetProcAddress`,
        // and the type it is cast to is that function's signature from
        // `EGL/egl.h`.
        let get_proc: unsafe extern "C" fn(*const c_char) -> *mut c_void =
            unsafe { cast(get_proc_ptr) };

        // Extension entry points are not in libEGL's dynamic symbol table;
        // only eglGetProcAddress resolves them. Core symbols come from either.
        let resolve = |name: &'static str| -> Result<*mut c_void, DmaBufCudaError> {
            if let Some(symbol) = raw_symbol(lib, name) {
                return Ok(symbol);
            }
            let c_name = CString::new(name).map_err(|_| DmaBufCudaError::EglSymbol(name))?;
            // SAFETY: `get_proc` is `eglGetProcAddress` and `c_name` outlives the
            // call. A name this EGL does not implement comes back null, which is
            // what the check below is for.
            let symbol = unsafe { get_proc(c_name.as_ptr()) };
            (!symbol.is_null())
                .then_some(symbol)
                .ok_or(DmaBufCudaError::EglSymbol(name))
        };

        // SAFETY: every `cast` receives the pointer `resolve` returned for the
        // symbol named on its own line, and the field being initialized declares
        // that symbol's signature from `EGL/egl.h` or `EGL/eglext.h`. Name and
        // type sit on one line so the pairing stays checkable by reading.
        Ok(unsafe {
            Self {
                get_proc,
                query_string: cast(resolve("eglQueryString")?),
                get_error: cast(resolve("eglGetError")?),
                initialize: cast(resolve("eglInitialize")?),
                bind_api: cast(resolve("eglBindAPI")?),
                create_context: cast(resolve("eglCreateContext")?),
                destroy_context: cast(resolve("eglDestroyContext")?),
                make_current: cast(resolve("eglMakeCurrent")?),
                create_image: cast(resolve("eglCreateImage")?),
                destroy_image: cast(resolve("eglDestroyImage")?),
                query_devices: cast(resolve("eglQueryDevicesEXT")?),
                query_device_attrib: cast(resolve("eglQueryDeviceAttribEXT")?),
                get_platform_display: cast(resolve("eglGetPlatformDisplayEXT")?),
                query_dma_buf_modifiers: cast(resolve("eglQueryDmaBufModifiersEXT")?),
            }
        })
    }

    /// The display for the EGL device whose `EGL_CUDA_DEVICE_NV` is 0 — the
    /// same device [`crate::elements::CudaDevice`] and
    /// `platform::cuda::driver` open, so an imported buffer and the frame it
    /// is copied into live on one GPU by construction rather than by
    /// agreement.
    fn cuda_device_display(&self) -> Result<EglDisplay, DmaBufCudaError> {
        // SAFETY: `query_devices` fills at most `devices.len()` entries and
        // reports how many through `count`, which is what bounds the slice below.
        // The remaining calls take out-params that are live locals, and a display
        // EGL declines to create arrives as null and is skipped rather than used.
        unsafe {
            let mut devices = [std::ptr::null_mut::<c_void>(); 16];
            let mut count = 0;
            if (self.query_devices)(devices.len() as c_int, devices.as_mut_ptr(), &mut count) == 0 {
                return Err(DmaBufCudaError::NoCudaEglDevice);
            }
            for &device in &devices[..count.max(0) as usize] {
                let mut ordinal: EglAttrib = -1;
                if (self.query_device_attrib)(device, EGL_CUDA_DEVICE_NV, &mut ordinal) == 0
                    || ordinal != 0
                {
                    continue;
                }
                let display =
                    (self.get_platform_display)(EGL_PLATFORM_DEVICE_EXT, device, std::ptr::null());
                if display.is_null() {
                    continue;
                }
                let (mut major, mut minor) = (0, 0);
                if (self.initialize)(display, &mut major, &mut minor) == 0 {
                    return Err(DmaBufCudaError::EglInit((self.get_error)()));
                }
                return Ok(display);
            }
            Err(DmaBufCudaError::NoCudaEglDevice)
        }
    }

    fn extensions(&self, display: EglDisplay) -> String {
        // SAFETY: `query_string` returns either null — handled — or a
        // NUL-terminated string owned by EGL and valid for the life of the
        // display, which `CStr::from_ptr` only borrows while copying it.
        unsafe {
            let ptr = (self.query_string)(display, EGL_EXTENSIONS);
            if ptr.is_null() {
                return String::new();
            }
            std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }

    /// Every modifier the driver can import for `fourcc`, minus the ones it
    /// marks `external_only`: those are only usable through
    /// `GL_TEXTURE_EXTERNAL_OES`, which cannot be attached to a framebuffer,
    /// and the readback here needs exactly that.
    fn dma_buf_modifiers(&self, display: EglDisplay, fourcc: u32) -> Vec<u64> {
        // SAFETY: the first call passes null out-pointers, which the extension
        // defines as "report the count only". The second passes vectors allocated
        // to exactly that count, and both are sized from the same value, so
        // neither can be written past its end.
        unsafe {
            let mut count = 0;
            if (self.query_dma_buf_modifiers)(
                display,
                fourcc as c_int,
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut count,
            ) == 0
                || count <= 0
            {
                return Vec::new();
            }
            let mut modifiers = vec![0u64; count as usize];
            let mut external_only = vec![0u32; count as usize];
            if (self.query_dma_buf_modifiers)(
                display,
                fourcc as c_int,
                count,
                modifiers.as_mut_ptr(),
                external_only.as_mut_ptr(),
                &mut count,
            ) == 0
            {
                return Vec::new();
            }
            modifiers
                .into_iter()
                .zip(external_only)
                .filter(|&(_, external)| external == 0)
                .map(|(modifier, _)| modifier)
                .collect()
        }
    }

    /// A surfaceless GLES context with no config — all this GL context ever
    /// does is own a texture, a framebuffer, and a pixel buffer, so there is
    /// no drawable to configure.
    fn create_context(&self, display: EglDisplay) -> Result<*mut c_void, DmaBufCudaError> {
        // SAFETY: `attribs` is terminated with `EGL_NONE_INT` and outlives the
        // call. Every failure path returns before the context is used, and the
        // one that fails after creating it destroys it first.
        unsafe {
            if (self.bind_api)(EGL_OPENGL_ES_API) == 0 {
                return Err(DmaBufCudaError::EglContext((self.get_error)()));
            }
            let attribs = [EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE_INT];
            let context = (self.create_context)(
                display,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                attribs.as_ptr(),
            );
            if context.is_null() {
                return Err(DmaBufCudaError::EglContext((self.get_error)()));
            }
            if (self.make_current)(display, std::ptr::null_mut(), std::ptr::null_mut(), context)
                == 0
            {
                (self.destroy_context)(display, context);
                return Err(DmaBufCudaError::EglMakeCurrent((self.get_error)()));
            }
            Ok(context)
        }
    }

    fn create_dma_buf_image(
        &self,
        display: EglDisplay,
        plane: DmaBufPlane,
        width: u32,
        height: u32,
    ) -> Result<EglImage, DmaBufCudaError> {
        let attribs: [EglAttrib; 17] = [
            EGL_WIDTH,
            width as EglAttrib,
            EGL_HEIGHT,
            height as EglAttrib,
            EGL_LINUX_DRM_FOURCC_EXT,
            plane.fourcc as EglAttrib,
            EGL_DMA_BUF_PLANE0_FD_EXT,
            plane.fd as EglAttrib,
            EGL_DMA_BUF_PLANE0_OFFSET_EXT,
            plane.offset as EglAttrib,
            EGL_DMA_BUF_PLANE0_PITCH_EXT,
            plane.stride as EglAttrib,
            EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT,
            (plane.modifier & 0xffff_ffff) as EglAttrib,
            EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT,
            (plane.modifier >> 32) as EglAttrib,
            EGL_NONE,
        ];
        // SAFETY: `attribs` is a live 17-element list terminated by `EGL_NONE`,
        // which is the layout `EGL_EXT_image_dma_buf_import` defines for a
        // single-plane buffer — the only kind this negotiates. `plane.fd` need
        // only be open for the duration of the call, as noted below.
        unsafe {
            // EGL dups the fd, so the PipeWire buffer keeps ownership of its
            // own and this image outlives any single `process` callback.
            let image = (self.create_image)(
                display,
                std::ptr::null_mut(),
                EGL_LINUX_DMA_BUF_EXT,
                std::ptr::null_mut(),
                attribs.as_ptr(),
            );
            if image.is_null() {
                return Err(DmaBufCudaError::CreateImage((self.get_error)()));
            }
            Ok(image)
        }
    }

    fn destroy_image(&self, display: EglDisplay, image: EglImage) {
        // SAFETY: `image` was created by this module on this `display`, and
        // `copy_into` — the only caller — destroys the one it just made, once.
        unsafe { (self.destroy_image)(display, image) };
    }

    fn release_context(&self, display: EglDisplay, context: *mut c_void) {
        // SAFETY: unbinding with null before destroying is what lets EGL release
        // the context here rather than deferring it to some later
        // `eglMakeCurrent`. `context` is this importer's own and `Drop` runs once.
        unsafe {
            (self.make_current)(
                display,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            (self.destroy_context)(display, context);
        }
    }
}

/// The GLES 3 entry points the readback needs, resolved once from
/// `libGLESv2.so.2`.
///
/// Every method below is one call through one of these pointers, so they
/// share a precondition: a current EGL context on the calling thread.
/// [`DmaBufCudaImporter`] makes one current in `new` and is not `Send`, so
/// no thread can reach these without it. Each method's own `SAFETY` note
/// records only what that call adds on top.
struct Gl {
    gen_textures: unsafe extern "C" fn(c_int, *mut c_uint),
    delete_textures: unsafe extern "C" fn(c_int, *const c_uint),
    bind_texture: unsafe extern "C" fn(c_uint, c_uint),
    gen_framebuffers: unsafe extern "C" fn(c_int, *mut c_uint),
    delete_framebuffers: unsafe extern "C" fn(c_int, *const c_uint),
    bind_framebuffer: unsafe extern "C" fn(c_uint, c_uint),
    framebuffer_texture_2d: unsafe extern "C" fn(c_uint, c_uint, c_uint, c_uint, c_int),
    check_framebuffer_status: unsafe extern "C" fn(c_uint) -> c_uint,
    gen_buffers: unsafe extern "C" fn(c_int, *mut c_uint),
    delete_buffers: unsafe extern "C" fn(c_int, *const c_uint),
    bind_buffer: unsafe extern "C" fn(c_uint, c_uint),
    buffer_data: unsafe extern "C" fn(c_uint, isize, *const c_void, c_uint),
    read_pixels: unsafe extern "C" fn(c_int, c_int, c_int, c_int, c_uint, c_uint, *mut c_void),
    get_error: unsafe extern "C" fn() -> c_uint,
    tex_storage_2d: unsafe extern "C" fn(c_uint, c_int, c_uint, c_int, c_int),
    tex_parameteri: unsafe extern "C" fn(c_uint, c_uint, c_int),
    disable: unsafe extern "C" fn(c_uint),
    viewport: unsafe extern "C" fn(c_int, c_int, c_int, c_int),
    draw_arrays: unsafe extern "C" fn(c_uint, c_int, c_int),
    create_shader: unsafe extern "C" fn(c_uint) -> c_uint,
    shader_source: unsafe extern "C" fn(c_uint, c_int, *const *const c_char, *const c_int),
    compile_shader: unsafe extern "C" fn(c_uint),
    get_shaderiv: unsafe extern "C" fn(c_uint, c_uint, *mut c_int),
    get_shader_info_log: unsafe extern "C" fn(c_uint, c_int, *mut c_int, *mut c_char),
    delete_shader: unsafe extern "C" fn(c_uint),
    create_program: unsafe extern "C" fn() -> c_uint,
    attach_shader: unsafe extern "C" fn(c_uint, c_uint),
    link_program: unsafe extern "C" fn(c_uint),
    get_programiv: unsafe extern "C" fn(c_uint, c_uint, *mut c_int),
    get_program_info_log: unsafe extern "C" fn(c_uint, c_int, *mut c_int, *mut c_char),
    delete_program: unsafe extern "C" fn(c_uint),
    use_program: unsafe extern "C" fn(c_uint),
    get_uniform_location: unsafe extern "C" fn(c_uint, *const c_char) -> c_int,
    uniform_1i: unsafe extern "C" fn(c_int, c_int),
    /// From `GL_OES_EGL_image`, resolved through `eglGetProcAddress` — this
    /// is the call that makes the imported DMA-BUF a GL texture.
    image_target_texture: unsafe extern "C" fn(c_uint, EglImage),
    /// What a test fills a buffer through that the CPU cannot map: one in
    /// a GPU's own tiled layout.
    #[cfg(test)]
    tex_sub_image_2d: unsafe extern "C" fn(
        c_uint,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_uint,
        c_uint,
        *const c_void,
    ),
    #[cfg(test)]
    finish: unsafe extern "C" fn(),
}

impl Gl {
    fn load(egl: &Egl) -> Result<Self, DmaBufCudaError> {
        let lib =
            open_library("libGLESv2.so.2").ok_or(DmaBufCudaError::Library("libGLESv2.so.2"))?;
        let resolve =
            |name: &'static str| raw_symbol(lib, name).ok_or(DmaBufCudaError::GlSymbol(name));
        let image_target = {
            let name = CString::new("glEGLImageTargetTexture2DOES")
                .map_err(|_| DmaBufCudaError::GlSymbol("glEGLImageTargetTexture2DOES"))?;
            // SAFETY: `glEGLImageTargetTexture2DOES` is an extension entry point, so
            // libGLESv2's dynamic symbol table does not carry it and only
            // `eglGetProcAddress` resolves it. `name` outlives the call and the
            // result is null-checked.
            let symbol = unsafe { (egl.get_proc)(name.as_ptr()) };
            (!symbol.is_null())
                .then_some(symbol)
                .ok_or(DmaBufCudaError::GlSymbol("glEGLImageTargetTexture2DOES"))?
        };

        // SAFETY: as in `Egl::load` — each `cast` gets the pointer resolved for
        // the symbol named on that line, typed as that GLES 3 entry point's
        // signature.
        Ok(unsafe {
            Self {
                gen_textures: cast(resolve("glGenTextures")?),
                delete_textures: cast(resolve("glDeleteTextures")?),
                bind_texture: cast(resolve("glBindTexture")?),
                gen_framebuffers: cast(resolve("glGenFramebuffers")?),
                delete_framebuffers: cast(resolve("glDeleteFramebuffers")?),
                bind_framebuffer: cast(resolve("glBindFramebuffer")?),
                framebuffer_texture_2d: cast(resolve("glFramebufferTexture2D")?),
                check_framebuffer_status: cast(resolve("glCheckFramebufferStatus")?),
                gen_buffers: cast(resolve("glGenBuffers")?),
                delete_buffers: cast(resolve("glDeleteBuffers")?),
                bind_buffer: cast(resolve("glBindBuffer")?),
                buffer_data: cast(resolve("glBufferData")?),
                read_pixels: cast(resolve("glReadPixels")?),
                get_error: cast(resolve("glGetError")?),
                tex_storage_2d: cast(resolve("glTexStorage2D")?),
                tex_parameteri: cast(resolve("glTexParameteri")?),
                disable: cast(resolve("glDisable")?),
                viewport: cast(resolve("glViewport")?),
                draw_arrays: cast(resolve("glDrawArrays")?),
                create_shader: cast(resolve("glCreateShader")?),
                shader_source: cast(resolve("glShaderSource")?),
                compile_shader: cast(resolve("glCompileShader")?),
                get_shaderiv: cast(resolve("glGetShaderiv")?),
                get_shader_info_log: cast(resolve("glGetShaderInfoLog")?),
                delete_shader: cast(resolve("glDeleteShader")?),
                create_program: cast(resolve("glCreateProgram")?),
                attach_shader: cast(resolve("glAttachShader")?),
                link_program: cast(resolve("glLinkProgram")?),
                get_programiv: cast(resolve("glGetProgramiv")?),
                get_program_info_log: cast(resolve("glGetProgramInfoLog")?),
                delete_program: cast(resolve("glDeleteProgram")?),
                use_program: cast(resolve("glUseProgram")?),
                get_uniform_location: cast(resolve("glGetUniformLocation")?),
                uniform_1i: cast(resolve("glUniform1i")?),
                image_target_texture: cast(image_target),
                #[cfg(test)]
                tex_sub_image_2d: cast(resolve("glTexSubImage2D")?),
                #[cfg(test)]
                finish: cast(resolve("glFinish")?),
            }
        })
    }

    /// Drains the GL error queue and reports the first error as `op`'s.
    ///
    /// GL reports asynchronously, so this is called after each step that can
    /// fail rather than once at the end, where an error could no longer be
    /// attributed.
    fn check(&self, op: &'static str) -> Result<(), DmaBufCudaError> {
        let mut first = GL_NO_ERROR;
        loop {
            // SAFETY: `glGetError` takes no arguments and needs only a current
            // context.
            let error = unsafe { (self.get_error)() };
            if error == GL_NO_ERROR {
                break;
            }
            if first == GL_NO_ERROR {
                first = error;
            }
        }
        (first == GL_NO_ERROR)
            .then_some(())
            .ok_or(DmaBufCudaError::Gl(op, first))
    }

    fn gen_texture(&self) -> c_uint {
        let mut name = 0;
        // SAFETY: `glGenTextures` writes as many names as it is asked for and
        // has no bound of its own, so the count is fixed at the one name
        // `name` has room for.
        unsafe { (self.gen_textures)(1, &mut name) };
        name
    }

    fn delete_texture(&self, texture: c_uint) {
        // SAFETY: reads one name, which is what `texture` is.
        unsafe { (self.delete_textures)(1, &texture) };
    }

    fn bind_texture(&self, target: c_uint, texture: c_uint) {
        // SAFETY: a target enum and a texture name, both plain integers. Values
        // GL does not recognize are a GL error, not undefined behaviour.
        unsafe { (self.bind_texture)(target, texture) };
    }

    fn gen_framebuffer(&self) -> c_uint {
        let mut name = 0;
        // SAFETY: as `gen_texture` — one name, one `c_uint` of room.
        unsafe { (self.gen_framebuffers)(1, &mut name) };
        name
    }

    fn delete_framebuffer(&self, framebuffer: c_uint) {
        // SAFETY: reads one name, which is what `framebuffer` is.
        unsafe { (self.delete_framebuffers)(1, &framebuffer) };
    }

    fn bind_framebuffer(&self, target: c_uint, framebuffer: c_uint) {
        // SAFETY: a target enum and a framebuffer name, both plain integers.
        unsafe { (self.bind_framebuffer)(target, framebuffer) };
    }

    fn framebuffer_texture_2d(
        &self,
        target: c_uint,
        attachment: c_uint,
        texture_target: c_uint,
        texture: c_uint,
        level: c_int,
    ) {
        // SAFETY: all five arguments are plain integers. A target, attachment, or
        // texture that does not fit together is reported by the
        // `check_framebuffer_status` call the caller makes next.
        unsafe {
            (self.framebuffer_texture_2d)(target, attachment, texture_target, texture, level)
        };
    }

    fn check_framebuffer_status(&self, target: c_uint) -> c_uint {
        // SAFETY: takes a target enum and returns a status.
        unsafe { (self.check_framebuffer_status)(target) }
    }

    fn gen_buffer(&self) -> c_uint {
        let mut name = 0;
        // SAFETY: as `gen_texture` — one name, one `c_uint` of room.
        unsafe { (self.gen_buffers)(1, &mut name) };
        name
    }

    fn delete_buffer(&self, buffer: c_uint) {
        // SAFETY: reads one name, which is what `buffer` is.
        unsafe { (self.delete_buffers)(1, &buffer) };
    }

    fn bind_buffer(&self, target: c_uint, buffer: c_uint) {
        // SAFETY: a target enum and a buffer name, both plain integers.
        unsafe { (self.bind_buffer)(target, buffer) };
    }

    fn buffer_data(&self, target: c_uint, size: isize, data: *const c_void, usage: c_uint) {
        // SAFETY: `ensure_pixel_buffer` passes a null `data` with a non-negative
        // `size`, which allocates without reading anything — the only form used
        // here. A non-null `data` would instead have to be readable for `size`
        // bytes.
        unsafe { (self.buffer_data)(target, size, data, usage) };
    }

    #[allow(clippy::too_many_arguments)]
    fn read_pixels(
        &self,
        x: c_int,
        y: c_int,
        width: c_int,
        height: c_int,
        format: c_uint,
        kind: c_uint,
        pixels: *mut c_void,
    ) {
        // SAFETY: `copy_into` calls this with a buffer bound to
        // `GL_PIXEL_PACK_BUFFER` and a null `pixels`, so the pointer is an offset
        // into that buffer rather than a host address, and
        // `ensure_pixel_buffer` sized it to the same `width * height * 4`. With
        // no pack buffer bound the pointer would instead have to address that
        // many writable host bytes.
        unsafe { (self.read_pixels)(x, y, width, height, format, kind, pixels) };
    }

    fn image_target_texture(&self, target: c_uint, image: EglImage) {
        // SAFETY: `image` is an EGLImage this module created, and both callers
        // destroy it only after the import this binds it for is done.
        unsafe { (self.image_target_texture)(target, image) };
    }

    fn tex_storage_2d(
        &self,
        target: c_uint,
        levels: c_int,
        format: c_uint,
        width: c_int,
        height: c_int,
    ) {
        // SAFETY: plain integers; allocates with nothing to read.
        unsafe { (self.tex_storage_2d)(target, levels, format, width, height) };
    }

    fn tex_parameter(&self, target: c_uint, name: c_uint, value: c_int) {
        // SAFETY: plain integers.
        unsafe { (self.tex_parameteri)(target, name, value) };
    }

    fn disable(&self, capability: c_uint) {
        // SAFETY: a plain integer.
        unsafe { (self.disable)(capability) };
    }

    fn viewport(&self, x: c_int, y: c_int, width: c_int, height: c_int) {
        // SAFETY: plain integers.
        unsafe { (self.viewport)(x, y, width, height) };
    }

    fn draw_arrays(&self, mode: c_uint, first: c_int, count: c_int) {
        // SAFETY: the program drawn with reads no vertex attributes — its
        // positions come from `gl_VertexID` — so there is no array for this to
        // read past.
        unsafe { (self.draw_arrays)(mode, first, count) };
    }

    fn use_program(&self, program: c_uint) {
        // SAFETY: a plain integer.
        unsafe { (self.use_program)(program) };
    }

    fn delete_program(&self, program: c_uint) {
        // SAFETY: a plain integer.
        unsafe { (self.delete_program)(program) };
    }

    fn uniform_location(&self, program: c_uint, name: &std::ffi::CStr) -> c_int {
        // SAFETY: `name` is NUL-terminated and outlives the call.
        unsafe { (self.get_uniform_location)(program, name.as_ptr()) }
    }

    fn uniform_1i(&self, location: c_int, value: c_int) {
        // SAFETY: plain integers; a location of -1 is ignored by GL.
        unsafe { (self.uniform_1i)(location, value) };
    }

    /// Compiles and links a program from two sources, or says what the
    /// driver said about them.
    fn build_program(&self, vertex: &str, fragment: &str) -> Result<c_uint, DmaBufCudaError> {
        let vertex = self.compile(GL_VERTEX_SHADER, vertex)?;
        let fragment = match self.compile(GL_FRAGMENT_SHADER, fragment) {
            Ok(fragment) => fragment,
            Err(error) => {
                // SAFETY: a shader name this just created.
                unsafe { (self.delete_shader)(vertex) };
                return Err(error);
            }
        };
        // SAFETY: `program` and both shaders are names created here, on this
        // context. The status out-param is a live local, and the log is read
        // by `info_log`, which bounds it. Shaders are deleted once attached
        // (GL keeps them alive for the program) and the program on failure.
        unsafe {
            let program = (self.create_program)();
            (self.attach_shader)(program, vertex);
            (self.attach_shader)(program, fragment);
            (self.link_program)(program);
            (self.delete_shader)(vertex);
            (self.delete_shader)(fragment);
            let mut linked = 0;
            (self.get_programiv)(program, GL_LINK_STATUS, &mut linked);
            if linked == 0 {
                let log = info_log(|size, written, log| {
                    (self.get_program_info_log)(program, size, written, log)
                });
                (self.delete_program)(program);
                return Err(DmaBufCudaError::Shader(log));
            }
            Ok(program)
        }
    }

    fn compile(&self, kind: c_uint, source: &str) -> Result<c_uint, DmaBufCudaError> {
        let length = c_int::try_from(source.len())
            .map_err(|_| DmaBufCudaError::Shader("a shader too long to pass".to_owned()))?;
        // SAFETY: `source` is passed with its length, so it needs no NUL, and
        // outlives the call; the one-element pointer and length arrays match the
        // count of 1. The status out-param is a live local.
        unsafe {
            let shader = (self.create_shader)(kind);
            (self.shader_source)(shader, 1, &source.as_ptr().cast(), &length);
            (self.compile_shader)(shader);
            let mut compiled = 0;
            (self.get_shaderiv)(shader, GL_COMPILE_STATUS, &mut compiled);
            if compiled == 0 {
                let log = info_log(|size, written, log| {
                    (self.get_shader_info_log)(shader, size, written, log)
                });
                (self.delete_shader)(shader);
                return Err(DmaBufCudaError::Shader(log));
            }
            Ok(shader)
        }
    }
}

/// Reads a shader or program log through `read`, which is handed the size of
/// a buffer, somewhere to say how much it wrote, and the buffer.
fn info_log(read: impl FnOnce(c_int, *mut c_int, *mut c_char)) -> String {
    let mut log = vec![0u8; 4096];
    let mut written = 0;
    read(log.len() as c_int, &mut written, log.as_mut_ptr().cast());
    log.truncate(written.clamp(0, log.len() as c_int) as usize);
    String::from_utf8_lossy(&log).trim().to_owned()
}

/// `CUDA_MEMCPY2D`, versioned by name (`cuMemcpy2D_v2`) and unchanged since
/// CUDA 10 — the same struct `platform::cuda::driver` mirrors, redeclared
/// here so this module needs nothing from it but the driver library itself.
#[repr(C)]
#[derive(Default)]
struct CuMemcpy2D {
    src_x_in_bytes: usize,
    src_y: usize,
    src_memory_type: c_uint,
    src_host: *const c_void,
    src_device: u64,
    src_array: *mut c_void,
    src_pitch: usize,
    dst_x_in_bytes: usize,
    dst_y: usize,
    dst_memory_type: c_uint,
    dst_host: *mut c_void,
    dst_device: u64,
    dst_array: *mut c_void,
    dst_pitch: usize,
    width_in_bytes: usize,
    height: usize,
}

struct Cuda {
    device: c_int,
    primary_ctx_release: unsafe extern "C" fn(c_int) -> c_int,
    register_buffer: unsafe extern "C" fn(*mut *mut c_void, c_uint, c_uint) -> c_int,
    map_resources: unsafe extern "C" fn(c_uint, *mut *mut c_void, *mut c_void) -> c_int,
    mapped_pointer: unsafe extern "C" fn(*mut u64, *mut usize, *mut c_void) -> c_int,
    unmap_resources: unsafe extern "C" fn(c_uint, *mut *mut c_void, *mut c_void) -> c_int,
    unregister_resource: unsafe extern "C" fn(*mut c_void) -> c_int,
    memcpy_2d: unsafe extern "C" fn(*const CuMemcpy2D) -> c_int,
}

impl Cuda {
    /// Loads the driver API and makes device 0's **primary** context current
    /// on this thread, for the whole life of the importer.
    ///
    /// Set once rather than pushed and popped per call the way
    /// `platform::cuda::driver::CudaDriver` does: that type is shared across
    /// threads, while this one never leaves the thread that built it. Being
    /// the primary context is what makes the pixel buffer registered here
    /// and the frames FFmpeg allocates live in the same context — see
    /// [`crate::elements::CudaDevice`]'s own notes on why it opens that one.
    fn load() -> Result<Self, DmaBufCudaError> {
        let lib = open_library("libcuda.so.1").ok_or(DmaBufCudaError::Library("libcuda.so.1"))?;
        let resolve =
            |name: &'static str| raw_symbol(lib, name).ok_or(DmaBufCudaError::CudaSymbol(name));

        // SAFETY: each `cast` gets the pointer resolved for the symbol on its own
        // line, typed as that entry point's signature from `cuda.h`. The calls run
        // in the order the driver API requires — `cuInit` before anything else,
        // the retained primary context made current before anything uses it — and
        // each result is checked before the next call relies on it.
        unsafe {
            let init: unsafe extern "C" fn(c_uint) -> c_int = cast(resolve("cuInit")?);
            let device_get: unsafe extern "C" fn(*mut c_int, c_int) -> c_int =
                cast(resolve("cuDeviceGet")?);
            let primary_ctx_retain: unsafe extern "C" fn(*mut *mut c_void, c_int) -> c_int =
                cast(resolve("cuDevicePrimaryCtxRetain")?);
            let set_current: unsafe extern "C" fn(*mut c_void) -> c_int =
                cast(resolve("cuCtxSetCurrent")?);

            check_cuda("cuInit", init(0))?;
            let mut device = 0;
            check_cuda("cuDeviceGet", device_get(&mut device, 0))?;
            let mut context = std::ptr::null_mut();
            check_cuda(
                "cuDevicePrimaryCtxRetain",
                primary_ctx_retain(&mut context, device),
            )?;
            check_cuda("cuCtxSetCurrent", set_current(context))?;

            Ok(Self {
                device,
                primary_ctx_release: cast(resolve("cuDevicePrimaryCtxRelease_v2")?),
                register_buffer: cast(resolve("cuGraphicsGLRegisterBuffer")?),
                map_resources: cast(resolve("cuGraphicsMapResources")?),
                mapped_pointer: cast(resolve("cuGraphicsResourceGetMappedPointer_v2")?),
                unmap_resources: cast(resolve("cuGraphicsUnmapResources")?),
                unregister_resource: cast(resolve("cuGraphicsUnregisterResource")?),
                memcpy_2d: cast(resolve("cuMemcpy2D_v2")?),
            })
        }
    }

    /// Registers the pixel buffer once. `CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY`
    /// (`0x01`): CUDA only ever reads what GL wrote into it.
    fn register_buffer(&self, buffer: c_uint) -> Result<*mut c_void, DmaBufCudaError> {
        let mut resource = std::ptr::null_mut();
        // SAFETY: `resource` is a live local out-param, and `buffer` names a GL
        // buffer `ensure_pixel_buffer` allocated on the context current for this
        // thread.
        unsafe {
            check_cuda(
                "cuGraphicsGLRegisterBuffer",
                (self.register_buffer)(&mut resource, buffer, 0x01),
            )?;
        }
        Ok(resource)
    }

    fn map_buffer(&self, resource: *mut c_void) -> Result<u64, DmaBufCudaError> {
        let mut resource = resource;
        // SAFETY: `resource` is a local copy, so the `&mut` these take cannot
        // alias the `PixelBuffer` field it came from; `pointer` and `size` are
        // live locals. Mapping on the null stream is what orders this after the
        // GL writes, as `copy_into` records.
        unsafe {
            check_cuda(
                "cuGraphicsMapResources",
                (self.map_resources)(1, &mut resource, std::ptr::null_mut()),
            )?;
            let (mut pointer, mut size) = (0u64, 0usize);
            check_cuda(
                "cuGraphicsResourceGetMappedPointer",
                (self.mapped_pointer)(&mut pointer, &mut size, resource),
            )?;
            Ok(pointer)
        }
    }

    fn unmap(&self, resource: *mut c_void) -> Result<(), DmaBufCudaError> {
        let mut resource = resource;
        // SAFETY: `resource` is registered, and again a local copy. Unmapping one
        // that is not mapped returns an error rather than being undefined, which
        // is what lets `copy_into` unmap unconditionally — covering the case
        // where `map_buffer` failed *after* the map itself succeeded.
        unsafe {
            check_cuda(
                "cuGraphicsUnmapResources",
                (self.unmap_resources)(1, &mut resource, std::ptr::null_mut()),
            )
        }
    }

    fn unregister(&self, resource: *mut c_void) -> Result<(), DmaBufCudaError> {
        // SAFETY: `resource` came from `register_buffer`, and both callers take
        // the `PixelBuffer` out of its `Option` first, so it is unregistered
        // once.
        unsafe {
            check_cuda(
                "cuGraphicsUnregisterResource",
                (self.unregister_resource)(resource),
            )
        }
    }

    fn copy_2d(
        &self,
        source: u64,
        source_pitch: usize,
        destination: u64,
        destination_pitch: usize,
        width_in_bytes: usize,
        height: usize,
    ) -> Result<(), DmaBufCudaError> {
        let copy = CuMemcpy2D {
            src_memory_type: CU_MEMORYTYPE_DEVICE,
            src_device: source,
            src_pitch: source_pitch,
            dst_memory_type: CU_MEMORYTYPE_DEVICE,
            dst_device: destination,
            dst_pitch: destination_pitch,
            width_in_bytes,
            height,
            ..Default::default()
        };
        // SAFETY: `copy` is a live `CuMemcpy2D` with both memory types set to
        // device, so CUDA reads `src_device`/`dst_device` and ignores the host and
        // array fields left null by `Default`. The extents come from the caller
        // and both pitches from the allocations themselves.
        unsafe { check_cuda("cuMemcpy2D", (self.memcpy_2d)(&copy)) }
    }
}

impl Drop for Cuda {
    fn drop(&mut self) {
        // SAFETY: balances the `cuDevicePrimaryCtxRetain` in `load`. `Cuda` is
        // owned by one importer and is not `Clone`, so this runs once.
        unsafe { (self.primary_ctx_release)(self.device) };
    }
}

fn check_cuda(op: &'static str, result: c_int) -> Result<(), DmaBufCudaError> {
    (result == 0)
        .then_some(())
        .ok_or(DmaBufCudaError::Cuda(op, result))
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    #[test]
    fn temporary_buffer_is_deleted_when_allocation_fails() {
        let deleted = Cell::new(None);
        let register_called = Cell::new(false);

        let result = initialize_temporary_buffer(
            7,
            || Err("allocation failed"),
            |_: u32| {
                register_called.set(true);
                Ok(())
            },
            |buffer| deleted.set(Some(buffer)),
        );

        assert_eq!(result, Err("allocation failed"));
        assert_eq!(deleted.get(), Some(7));
        assert!(!register_called.get());
    }

    #[test]
    fn temporary_buffer_is_deleted_when_registration_fails() {
        let deleted = Cell::new(None);

        let result: Result<(), _> = initialize_temporary_buffer(
            11,
            || Ok(()),
            |_| Err("registration failed"),
            |buffer| deleted.set(Some(buffer)),
        );

        assert_eq!(result, Err("registration failed"));
        assert_eq!(deleted.get(), Some(11));
    }

    #[test]
    fn registered_buffer_is_not_deleted_before_ownership_transfers() {
        let deleted = Cell::new(None);

        let result = initialize_temporary_buffer(
            13,
            || Ok::<_, &str>(()),
            |buffer| Ok(buffer + 1),
            |buffer| deleted.set(Some(buffer)),
        );

        assert_eq!(result, Ok(14));
        assert_eq!(deleted.get(), None);
    }

    /// Everything this module does before a buffer arrives: opening the EGL
    /// device that backs CUDA device 0, making a context current, and finding
    /// modifiers the driver will import. Those modifiers are what the capture
    /// offers the compositor, so an empty list is a capture that can never
    /// negotiate.
    ///
    /// A machine with no CUDA device skips, and so does one missing the EGL
    /// libraries entirely — but a CUDA-capable machine whose import path is
    /// broken fails, which is the regression this is here to catch.
    #[test]
    fn the_importer_reports_modifiers_it_can_import() {
        let Some((_device, _cuda_lock)) = crate::test_support::try_cuda_device() else {
            return;
        };
        let importer = match DmaBufCudaImporter::new() {
            Ok(importer) => importer,
            Err(error @ DmaBufCudaError::Library(_)) => {
                eprintln!("skipping: {error}");
                return;
            }
            Err(error) => panic!("DMA-BUF import is unusable on a CUDA machine: {error}"),
        };
        assert!(
            !importer.modifiers().is_empty(),
            "an importer with no modifiers could never negotiate a capture"
        );
    }
}
