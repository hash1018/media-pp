//! Imports a PipeWire DMA-BUF into a CUDA surface, for
//! [`crate::elements::PipeWireScreenCaptureSource`]'s GPU capture mode.
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
//! thread that touches a captured buffer.

use std::{
    collections::HashMap,
    ffi::{CString, c_char, c_int, c_uint, c_void},
};

use thiserror::Error as ThisError;

/// `DRM_FORMAT_XRGB8888` / `DRM_FORMAT_ARGB8888` — the two fourccs that
/// correspond to the `BGRx`/`BGRA` SPA video formats this crate's capture
/// negotiates. Byte order is the same for both; the alpha byte is ignored.
const DRM_FORMAT_XRGB8888: u32 = 0x3432_5258;

#[derive(Debug, ThisError)]
pub enum DmaBufCudaError {
    #[error("failed to load {0}")]
    Library(&'static str),

    #[error("{0} is missing from libEGL")]
    EglSymbol(&'static str),

    #[error("{0} is missing from libGLESv2")]
    GlSymbol(&'static str),

    #[error("{0} is missing from libcuda")]
    CudaSymbol(&'static str),

    #[error("no EGL device is backed by CUDA device 0")]
    NoCudaEglDevice,

    #[error("eglInitialize failed (EGL error {0:#x})")]
    EglInit(c_int),

    #[error("the EGL device is missing {0}")]
    EglExtension(&'static str),

    #[error("eglCreateContext failed (EGL error {0:#x})")]
    EglContext(c_int),

    #[error("eglMakeCurrent failed (EGL error {0:#x})")]
    EglMakeCurrent(c_int),

    #[error("the driver accepts no DMA-BUF modifier for XRGB8888")]
    NoModifiers,

    #[error("eglCreateImage failed for the captured DMA-BUF (EGL error {0:#x})")]
    CreateImage(c_int),

    #[error("{0} failed (GL error {1:#x})")]
    Gl(&'static str, c_uint),

    #[error("the DMA-BUF framebuffer is incomplete ({0:#x})")]
    FramebufferIncomplete(c_uint),

    #[error("{0} failed (CUresult {1})")]
    Cuda(&'static str, c_int),
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
        let (pixels, pitch) = unsafe {
            let ptr = frame.as_ptr();
            ((*ptr).data[0], (*ptr).linesize[0])
        };
        (!pixels.is_null() && pitch > 0).then(|| Self {
            pixels: pixels as u64,
            pitch: pitch as usize,
        })
    }
}

pub(crate) struct DmaBufCudaImporter {
    egl: Egl,
    gl: Gl,
    cuda: Cuda,
    display: EglDisplay,
    context: *mut c_void,
    /// The modifiers this driver will accept for XRGB8888, in the order EGL
    /// reported them — what the stream offers the compositor.
    modifiers: Vec<u64>,
    /// One EGLImage per PipeWire buffer, keyed by the buffer's fd. PipeWire
    /// recycles a small set of buffers, so this converges after the first
    /// few frames instead of rebuilding an image per capture.
    images: HashMap<c_int, EglImage>,
    texture: c_uint,
    framebuffer: c_uint,
    /// The CUDA-registered pixel buffer the readback lands in, sized to the
    /// current capture. `None` until the first frame fixes a size.
    pixel_buffer: Option<PixelBuffer>,
}

/// The readback destination: a GL buffer object registered with CUDA once,
/// then mapped per frame.
struct PixelBuffer {
    buffer: c_uint,
    resource: *mut c_void,
    width: u32,
    height: u32,
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

        let mut importer = Self {
            egl,
            gl,
            cuda,
            display,
            context,
            modifiers,
            images: HashMap::new(),
            texture: 0,
            framebuffer: 0,
            pixel_buffer: None,
        };
        importer.gl.gen_textures(1, &mut importer.texture);
        importer.gl.gen_framebuffers(1, &mut importer.framebuffer);
        Ok(importer)
    }

    /// The DMA-BUF modifiers to offer the compositor, most preferred first.
    ///
    /// An import of a buffer allocated with anything else would fail, so
    /// these are exactly what the stream may fixate on.
    pub(crate) fn modifiers(&self) -> &[u64] {
        &self.modifiers
    }

    /// Copies one captured DMA-BUF into `destination`, which must be a
    /// `width` x `height` CUDA BGRA surface.
    pub(crate) fn copy_into(
        &mut self,
        plane: DmaBufPlane,
        width: u32,
        height: u32,
        destination: CudaBgraSurface,
    ) -> Result<(), DmaBufCudaError> {
        let image = self.image_for(plane, width, height)?;

        // The texture is retargeted rather than kept per buffer: binding an
        // existing EGLImage is cheap, and one texture keeps the GL state this
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

        let (buffer, resource) = self.ensure_pixel_buffer(width, height)?;
        self.gl.bind_buffer(GL_PIXEL_PACK_BUFFER, buffer);
        // Reads bottom-up in window coordinates, which for an FBO-attached
        // texture is its first row first — the DMA-BUF's own row order, so
        // the readback lands top-down with no flip.
        self.gl.read_pixels(
            0,
            0,
            width as c_int,
            height as c_int,
            GL_BGRA_EXT,
            GL_UNSIGNED_BYTE,
            std::ptr::null_mut(),
        );
        let read_result = self.gl.check("glReadPixels");
        self.gl.bind_buffer(GL_PIXEL_PACK_BUFFER, 0);
        if let Err(error) = read_result {
            self.unbind();
            return Err(error);
        }

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
        self.unbind();
        result.and(unmapped)
    }

    /// Releases every GL binding this took, so the next frame starts from the
    /// same state regardless of which path returned.
    fn unbind(&self) {
        self.gl.bind_framebuffer(GL_FRAMEBUFFER, 0);
        self.gl.bind_texture(GL_TEXTURE_2D, 0);
    }

    fn image_for(
        &mut self,
        plane: DmaBufPlane,
        width: u32,
        height: u32,
    ) -> Result<EglImage, DmaBufCudaError> {
        if let Some(image) = self.images.get(&plane.fd) {
            return Ok(*image);
        }
        let image = self
            .egl
            .create_dma_buf_image(self.display, plane, width, height)?;
        self.images.insert(plane.fd, image);
        Ok(image)
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
            // A resize invalidates every cached image too: the compositor
            // reallocates its buffers, and an fd may be reused for a buffer
            // of the new size.
            self.release_images();
            if let Some(old) = self.pixel_buffer.take() {
                let _ = self.cuda.unregister(old.resource);
                self.gl.delete_buffers(1, &old.buffer);
            }
            let mut buffer = 0;
            self.gl.gen_buffers(1, &mut buffer);
            self.gl.bind_buffer(GL_PIXEL_PACK_BUFFER, buffer);
            self.gl.buffer_data(
                GL_PIXEL_PACK_BUFFER,
                (width as isize) * (height as isize) * 4,
                std::ptr::null(),
                GL_STREAM_READ,
            );
            self.gl.bind_buffer(GL_PIXEL_PACK_BUFFER, 0);
            self.gl.check("glBufferData")?;
            let resource = self.cuda.register_buffer(buffer)?;
            self.pixel_buffer = Some(PixelBuffer {
                buffer,
                resource,
                width,
                height,
            });
        }
        let pixel_buffer = self.pixel_buffer.as_ref().expect("built above");
        Ok((pixel_buffer.buffer, pixel_buffer.resource))
    }

    fn release_images(&mut self) {
        for (_, image) in self.images.drain() {
            self.egl.destroy_image(self.display, image);
        }
    }
}

impl Drop for DmaBufCudaImporter {
    fn drop(&mut self) {
        self.release_images();
        if let Some(pbo) = self.pixel_buffer.take() {
            let _ = self.cuda.unregister(pbo.resource);
            self.gl.delete_buffers(1, &pbo.buffer);
        }
        self.gl.delete_framebuffers(1, &self.framebuffer);
        self.gl.delete_textures(1, &self.texture);
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
    unsafe { std::mem::transmute_copy(&ptr) }
}

fn open_library(name: &str) -> Option<*mut c_void> {
    let name = CString::new(name).ok()?;
    let handle = unsafe { dlopen(name.as_ptr(), RTLD_NOW) };
    (!handle.is_null()).then_some(handle)
}

fn raw_symbol(lib: *mut c_void, name: &str) -> Option<*mut c_void> {
    let name = CString::new(name).ok()?;
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
        let get_proc: unsafe extern "C" fn(*const c_char) -> *mut c_void =
            unsafe { cast(get_proc_ptr) };

        // Extension entry points are not in libEGL's dynamic symbol table;
        // only eglGetProcAddress resolves them. Core symbols come from either.
        let resolve = |name: &'static str| -> Result<*mut c_void, DmaBufCudaError> {
            if let Some(symbol) = raw_symbol(lib, name) {
                return Ok(symbol);
            }
            let c_name = CString::new(name).map_err(|_| DmaBufCudaError::EglSymbol(name))?;
            let symbol = unsafe { get_proc(c_name.as_ptr()) };
            (!symbol.is_null())
                .then_some(symbol)
                .ok_or(DmaBufCudaError::EglSymbol(name))
        };

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
            DRM_FORMAT_XRGB8888 as EglAttrib,
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
        unsafe { (self.destroy_image)(display, image) };
    }

    fn release_context(&self, display: EglDisplay, context: *mut c_void) {
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
    /// From `GL_OES_EGL_image`, resolved through `eglGetProcAddress` — this
    /// is the call that makes the imported DMA-BUF a GL texture.
    image_target_texture: unsafe extern "C" fn(c_uint, EglImage),
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
            let symbol = unsafe { (egl.get_proc)(name.as_ptr()) };
            (!symbol.is_null())
                .then_some(symbol)
                .ok_or(DmaBufCudaError::GlSymbol("glEGLImageTargetTexture2DOES"))?
        };

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
                image_target_texture: cast(image_target),
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

    fn gen_textures(&self, count: c_int, out: &mut c_uint) {
        unsafe { (self.gen_textures)(count, out) };
    }

    fn delete_textures(&self, count: c_int, textures: &c_uint) {
        unsafe { (self.delete_textures)(count, textures) };
    }

    fn bind_texture(&self, target: c_uint, texture: c_uint) {
        unsafe { (self.bind_texture)(target, texture) };
    }

    fn gen_framebuffers(&self, count: c_int, out: &mut c_uint) {
        unsafe { (self.gen_framebuffers)(count, out) };
    }

    fn delete_framebuffers(&self, count: c_int, framebuffers: &c_uint) {
        unsafe { (self.delete_framebuffers)(count, framebuffers) };
    }

    fn bind_framebuffer(&self, target: c_uint, framebuffer: c_uint) {
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
        unsafe {
            (self.framebuffer_texture_2d)(target, attachment, texture_target, texture, level)
        };
    }

    fn check_framebuffer_status(&self, target: c_uint) -> c_uint {
        unsafe { (self.check_framebuffer_status)(target) }
    }

    fn gen_buffers(&self, count: c_int, out: &mut c_uint) {
        unsafe { (self.gen_buffers)(count, out) };
    }

    fn delete_buffers(&self, count: c_int, buffers: &c_uint) {
        unsafe { (self.delete_buffers)(count, buffers) };
    }

    fn bind_buffer(&self, target: c_uint, buffer: c_uint) {
        unsafe { (self.bind_buffer)(target, buffer) };
    }

    fn buffer_data(&self, target: c_uint, size: isize, data: *const c_void, usage: c_uint) {
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
        unsafe { (self.read_pixels)(x, y, width, height, format, kind, pixels) };
    }

    fn image_target_texture(&self, target: c_uint, image: EglImage) {
        unsafe { (self.image_target_texture)(target, image) };
    }
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
        unsafe {
            check_cuda(
                "cuGraphicsUnmapResources",
                (self.unmap_resources)(1, &mut resource, std::ptr::null_mut()),
            )
        }
    }

    fn unregister(&self, resource: *mut c_void) -> Result<(), DmaBufCudaError> {
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
        unsafe { check_cuda("cuMemcpy2D", (self.memcpy_2d)(&copy)) }
    }
}

impl Drop for Cuda {
    fn drop(&mut self) {
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
    use super::*;

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
