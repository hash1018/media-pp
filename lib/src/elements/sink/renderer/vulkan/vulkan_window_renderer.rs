//! A Vulkan renderer that draws into a window it is given — the part a
//! program otherwise writes for itself behind a `CudaFrameRenderer`.

#[cfg(feature = "cuda")]
use std::os::fd::{FromRawFd, OwnedFd};
use std::{
    any::Any,
    ffi::CStr,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use ash::vk;
use ffmpeg_next::{self as ffmpeg, format::Pixel};
use raw_window_handle::{
    HandleError, HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle,
};
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    contract::{
        InputContract, MediaKind, MediaKindSet, MemoryDomain, MemoryDomainSet, PixelLayout,
        PixelLayoutSet, PortContract,
    },
    control::ControlMsg,
    element::{Element, ElementType, Sink, element_pp_log},
    elements::VulkanGpu,
    error::Result,
    platform::linux::vulkan::VulkanShared,
    pp_log::{PpLog, pp_error, pp_info},
};

#[cfg(feature = "cuda")]
use crate::platform::cuda::driver::interop::{CudaInterop, DeviceRows, ImportedMemory};

const SHADER: &str = include_str!("../../../../shaders/vulkan/present.wgsl");

/// Why a [`VulkanWindowRenderer`] could not be set up, or could not draw a
/// frame.
#[derive(Debug, ThisError)]
pub enum VulkanWindowRendererError {
    /// The window has no handle to give, or no display to find it on.
    #[error("the window given has no handle to draw into: {0}")]
    NoHandle(#[from] HandleError),

    /// Vulkan would not make a surface for the window — one that is neither
    /// X11 nor Wayland, most likely.
    #[error("could not make a Vulkan surface for the window: {0}")]
    Surface(vk::Result),

    /// The device the renderer was given cannot present into this window.
    #[error("{0} cannot present into this window")]
    CannotPresent(String),

    /// A Vulkan call failed.
    #[error("{call} failed: {result}")]
    Call {
        /// The Vulkan function that failed.
        call: &'static str,
        /// What it answered.
        result: vk::Result,
    },

    /// The device has no memory of a kind the renderer needs.
    #[error("the device has no {0} memory")]
    NoMemoryType(&'static str),

    /// The presenting shader did not compile — a fault in this crate, not in
    /// the program using it.
    #[error("the presenting shader did not build: {0}")]
    Shader(String),

    /// A frame in a layout this renderer does not draw: not NV12, YUV420P
    /// or BGRA in system memory, or not NV12 or BGRA in CUDA memory.
    #[error("a {0:?} frame is not one this renderer draws")]
    UnsupportedFrame(Pixel),

    /// A CUDA frame reached a renderer whose [`VulkanGpu`] was not made for
    /// CUDA, so there is no GPU memory to copy it into.
    #[cfg(feature = "cuda")]
    #[error("CUDA frames need a VulkanGpu made with VulkanGpu::for_cuda")]
    NotForCuda,

    /// A CUDA frame from another device, or in a layout the renderer does not
    /// draw.
    #[cfg(feature = "cuda")]
    #[error(transparent)]
    CudaFrame(#[from] crate::platform::cuda::CudaFrameError),

    /// The copy out of CUDA failed.
    #[cfg(feature = "cuda")]
    #[error(transparent)]
    Cuda(#[from] crate::platform::cuda::CudaDriverError),
}

fn call(call: &'static str) -> impl FnOnce(vk::Result) -> VulkanWindowRendererError {
    move |result| VulkanWindowRendererError::Call { call, result }
}

/// The size a [`VulkanWindowRenderer`] draws at where the window cannot say
/// — a Wayland one.
///
/// An X11 window reports its size to Vulkan, and the renderer follows it on
/// its own, reading it before every frame. A Wayland window does not: on
/// Wayland the size of a window is its client's to choose, so the program
/// that owns the window tells the renderer through this, on every resize
/// its own event loop sees. Until it does, the renderer draws at the size of
/// the frames. Cheap to clone; every clone sets the same size.
#[derive(Debug, Clone, Default)]
pub struct WindowSize(Arc<AtomicU64>);

impl WindowSize {
    /// The window is now `width` x `height` pixels.
    pub fn set(&self, width: u32, height: u32) {
        self.0.store(
            (u64::from(width) << 32) | u64::from(height),
            Ordering::Relaxed,
        );
    }

    fn get(&self) -> Option<vk::Extent2D> {
        let packed = self.0.load(Ordering::Relaxed);
        (packed != 0).then_some(vk::Extent2D {
            width: (packed >> 32) as u32,
            height: packed as u32,
        })
    }
}

/// A terminal sink that shows video frames in a window the application
/// gives it — an X11 window or a Wayland one, such as a `winit` window.
///
/// # What it takes
///
/// Frames in system memory, on any GPU a [`VulkanGpu`] can open: NV12,
/// YUV420P (and YUVJ420P) or BGRA — what a hardware decode downloaded, a
/// software decode, and a screen capture give, each drawn as it comes with
/// no conversion in front. And CUDA frames as well, NV12 or BGRA, when that
/// [`VulkanGpu`] was made with `VulkanGpu::for_cuda` (with the `cuda`
/// feature): a CUDA frame is copied, device to device, into memory this
/// renderer's device allocated and CUDA imported, which only works within
/// one GPU.
///
/// That is also why it is named for what draws rather than for what it
/// takes, unlike `CudaRenderer`: a renderer of one memory
/// domain is best found by the frames it takes, and this one takes more than
/// one, the way a GStreamer `vulkansink` takes system memory and its own.
///
/// # What it draws
///
/// Each YUV frame is drawn by its own colour description: BT.709, BT.601 or
/// BT.2020 coefficients as the frame says, limited or full range as it says,
/// and where it says nothing, BT.709 for a picture 720 rows high or more and
/// BT.601 below that; a YUVJ420P frame is full range whatever it says. A
/// BGRA frame is drawn as it is, and its alpha ignored. The picture keeps
/// its aspect ratio inside the window,
/// with black bars as needed, and each frame is presented synchronized to
/// the display's refresh.
///
/// It draws; it does not pace. Put a
/// [`crate::elements::VideoSynchronizer`] or [`crate::elements::Pacer`] in
/// front for a picture shown at its own time rather than as fast as it is
/// decoded.
///
/// # Following the window's size
///
/// On X11 it reads the window's size before every frame, so nothing has to
/// tell it about a resize. On Wayland nothing can be read — see
/// [`WindowSize`], which the application sets from its own resize events.
pub struct VulkanWindowRenderer {
    name: Arc<str>,
    pp_log: PpLog,
    presenter: Presenter,
    size: WindowSize,
    domains: MemoryDomainSet,
    /// What the last frame was and what it was drawn into, for a log line
    /// whenever either changes rather than one per frame.
    drawing: Option<Drawing>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Drawing {
    frame: [u32; 2],
    layout: Layout,
    domain: MemoryDomain,
    window: vk::Extent2D,
}

impl VulkanWindowRenderer {
    /// Draws into `window`, which the application owns and runs the event
    /// loop of. The renderer keeps its `Arc`, so the window cannot be dropped
    /// while it is being drawn into.
    ///
    /// Frames it draws must come from `gpu`'s own CUDA device where they are
    /// CUDA frames; frames in system memory can come from anywhere.
    pub fn for_window<W>(
        name: impl Into<String>,
        gpu: &VulkanGpu,
        window: Arc<W>,
    ) -> std::result::Result<Self, VulkanWindowRendererError>
    where
        W: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static,
    {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::VulkanWindowRenderer, &name, None);
        let display = window.display_handle()?.as_raw();
        let handle = window.window_handle()?.as_raw();
        let size = WindowSize::default();
        let presenter = Presenter::new(gpu.shared(), display, handle, window, size.clone())?;
        #[cfg(feature = "cuda")]
        let domains = if gpu.shared().cuda.is_some() {
            MemoryDomainSet::from_slice(&[MemoryDomain::System, MemoryDomain::Cuda])
        } else {
            MemoryDomainSet::of(MemoryDomain::System)
        };
        #[cfg(not(feature = "cuda"))]
        let domains = MemoryDomainSet::of(MemoryDomain::System);
        pp_info!(
            pp_log: &pp_log,
            "created: drawing on {} into a {} window",
            gpu.name(),
            window_kind(handle)
        );
        Ok(Self {
            name,
            pp_log,
            presenter,
            size,
            domains,
            drawing: None,
        })
    }

    /// Where to tell it the window's size, for a Wayland window — see
    /// [`WindowSize`]. Taken before the renderer goes into its pipeline.
    pub fn window_size(&self) -> WindowSize {
        self.size.clone()
    }

    /// Draws one frame, and says what it was.
    fn draw(
        &mut self,
        frame: &ffmpeg::frame::Video,
    ) -> std::result::Result<(Layout, MemoryDomain), VulkanWindowRendererError> {
        let (width, height) = (frame.width(), frame.height());
        let range = match frame.format() {
            // The J is the range: FFmpeg's name for full-range 4:2:0, which
            // not every producer also says in the range field.
            Pixel::YUVJ420P => ffmpeg::color::Range::JPEG,
            _ => frame.color_range(),
        };
        let colour = Colour::of(frame.color_space(), range, height);
        match frame.format() {
            #[cfg(feature = "cuda")]
            Pixel::CUDA => {
                use crate::platform::cuda::{CudaSurfaces, frame::validate};

                let pairing = self
                    .presenter
                    .gpu
                    .cuda
                    .as_ref()
                    .ok_or(VulkanWindowRendererError::NotForCuda)?;
                let surface = validate(
                    frame,
                    ElementType::VulkanWindowRenderer,
                    pairing.device_ctx,
                    CudaSurfaces::NV12_OR_BGRA,
                )?;
                let layout = Layout::of(surface.layout)
                    .ok_or(VulkanWindowRendererError::UnsupportedFrame(surface.layout))?;
                let interop = Arc::clone(&pairing.interop);
                let mut planes = [(0u64, 0usize); 3];
                for (index, plane) in planes.iter_mut().enumerate().take(layout.plane_count()) {
                    // SAFETY: `validate` has established a live CUDA frame from
                    // the device this renderer's GPU was paired with, in a
                    // layout of `plane_count` planes, whose pointers are
                    // device pointers in that context.
                    *plane = unsafe {
                        let ptr = frame.as_ptr();
                        ((*ptr).data[index] as u64, (*ptr).linesize[index] as usize)
                    };
                    if plane.0 == 0 {
                        return Err(VulkanWindowRendererError::UnsupportedFrame(surface.layout));
                    }
                }
                let source = Source::Cuda {
                    planes,
                    interop: &interop,
                };
                self.presenter
                    .present(width, height, layout, colour, source)?;
                Ok((layout, MemoryDomain::Cuda))
            }
            format => {
                let layout = Layout::of(format)
                    .ok_or(VulkanWindowRendererError::UnsupportedFrame(format))?;
                if frame.planes() < layout.plane_count() {
                    return Err(VulkanWindowRendererError::UnsupportedFrame(format));
                }
                self.presenter
                    .present(width, height, layout, colour, Source::System(frame))?;
                Ok((layout, MemoryDomain::System))
            }
        }
    }
}

fn window_kind(handle: RawWindowHandle) -> &'static str {
    match handle {
        RawWindowHandle::Xlib(_) | RawWindowHandle::Xcb(_) => "X11",
        RawWindowHandle::Wayland(_) => "Wayland",
        _ => "foreign",
    }
}

impl Element for VulkanWindowRenderer {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::VulkanWindowRenderer
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Sink for VulkanWindowRenderer {
    /// NV12, YUV420P and BGRA, in system memory; in CUDA memory too where
    /// the GPU was made for it, where the frames are NV12 or BGRA.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::Frames(
            MediaKindSet::of(MediaKind::VideoFrame),
            self.domains,
            PixelLayoutSet::from_slice(&[
                PixelLayout::Nv12,
                PixelLayout::Yuv420p,
                PixelLayout::Bgra,
            ]),
        ))
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        let MediaBuffer::Video(frame) = buf else {
            return Ok(());
        };
        let (layout, domain) = self
            .draw(&frame)
            .inspect_err(|error| pp_error!(self, "draw failed: {error}"))?;
        let drawing = self.presenter.extent().map(|window| Drawing {
            frame: [frame.width(), frame.height()],
            layout,
            domain,
            window,
        });
        if let Some(now) = drawing
            && drawing != self.drawing
        {
            pp_info!(
                self,
                "drawing {}x{} {:?} {:?} frames into {}x{}",
                now.frame[0],
                now.frame[1],
                now.layout,
                now.domain,
                now.window.width,
                now.window.height
            );
            self.drawing = drawing;
        }
        Ok(())
    }

    fn control(&mut self, _msg: ControlMsg) -> Result<()> {
        // Terminal, with nothing to flush or forward — as `CudaRenderer`.
        Ok(())
    }
}

/// The colour conversion one frame is drawn with: the eight push constants
/// `present.wgsl`'s YUV shaders read.
///
/// Derived from the matrix's two luma weights, `kr` and `kb`, rather than
/// written out per standard, so there is one formula to trust. Limited range
/// scales luma by 255/219 and chroma by 255/224 after taking their offsets
/// away, which reproduces the familiar constants exactly: BT.709 red from Cr
/// is 1.5748 x 255/224 = 1.793, and BT.601's is 1.402 x 255/224 = 1.596.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
struct Colour {
    y_offset: f32,
    y_scale: f32,
    c_offset: f32,
    c_scale: f32,
    cr_to_r: f32,
    cb_to_g: f32,
    cr_to_g: f32,
    cb_to_b: f32,
}

impl Colour {
    fn of(space: ffmpeg::color::Space, range: ffmpeg::color::Range, height: u32) -> Self {
        use ffmpeg::color::{Range, Space};

        let (kr, kb) = match space {
            Space::BT709 => (0.2126, 0.0722),
            Space::BT470BG | Space::SMPTE170M | Space::SMPTE240M | Space::FCC => (0.299, 0.114),
            Space::BT2020NCL | Space::BT2020CL => (0.2627, 0.0593),
            // Nothing said: what a player assumes, by the picture's height.
            _ if height >= 720 => (0.2126, 0.0722),
            _ => (0.299, 0.114),
        };
        let kg = 1.0 - kr - kb;
        let (y_offset, y_scale, c_scale) = match range {
            Range::JPEG => (0.0, 1.0, 1.0),
            // MPEG, or unspecified: limited is what video almost always is.
            _ => (16.0 / 255.0, 255.0 / 219.0, 255.0 / 224.0),
        };
        Self {
            y_offset,
            y_scale,
            c_offset: 128.0 / 255.0,
            c_scale,
            cr_to_r: 2.0 * (1.0 - kr),
            cb_to_g: 2.0 * kb * (1.0 - kb) / kg,
            cr_to_g: 2.0 * kr * (1.0 - kr) / kg,
            cb_to_b: 2.0 * (1.0 - kb),
        }
    }

    fn bytes(&self) -> [u8; 32] {
        let values = [
            self.y_offset,
            self.y_scale,
            self.c_offset,
            self.c_scale,
            self.cr_to_r,
            self.cb_to_g,
            self.cr_to_g,
            self.cb_to_b,
        ];
        let mut bytes = [0u8; 32];
        for (chunk, value) in bytes.chunks_exact_mut(4).zip(values) {
            chunk.copy_from_slice(&value.to_ne_bytes());
        }
        bytes
    }
}

/// A frame layout this renderer draws, each with a fragment shader of its
/// own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layout {
    /// A luma plane and an interleaved chroma plane at half size.
    Nv12,
    /// Luma, Cb and Cr planes, the chroma ones at half size.
    Yuv420p,
    /// One plane of packed B, G, R and A bytes.
    Bgra,
}

impl Layout {
    const ALL: [Self; 3] = [Self::Nv12, Self::Yuv420p, Self::Bgra];

    fn of(format: Pixel) -> Option<Self> {
        match format {
            Pixel::NV12 => Some(Self::Nv12),
            Pixel::YUV420P | Pixel::YUVJ420P => Some(Self::Yuv420p),
            Pixel::BGRA => Some(Self::Bgra),
            _ => None,
        }
    }

    /// Which of the pipelines made from [`Self::ALL`] draws it.
    fn index(self) -> usize {
        match self {
            Self::Nv12 => 0,
            Self::Yuv420p => 1,
            Self::Bgra => 2,
        }
    }

    fn fragment_shader(self) -> &'static CStr {
        match self {
            Self::Nv12 => c"fs_nv12",
            Self::Yuv420p => c"fs_yuv420p",
            Self::Bgra => c"fs_bgra",
        }
    }

    fn plane_count(self) -> usize {
        match self {
            Self::Nv12 => 2,
            Self::Yuv420p => 3,
            Self::Bgra => 1,
        }
    }

    /// The planes of a `width` x `height` frame, as they are packed one
    /// after another in the staging buffer.
    ///
    /// Chroma is rounded up, as FFmpeg rounds it: an odd-sized frame's last
    /// column and row have a chroma sample of their own. Each plane starts
    /// on a 16-byte boundary, which is more than a buffer-to-image copy
    /// asks of any of these formats.
    fn planes(self, width: u32, height: u32) -> Vec<PlaneShape> {
        let (half_width, half_height) = (width.div_ceil(2), height.div_ceil(2));
        let shapes: &[(vk::Format, u32, u32, u32)] = match self {
            Self::Nv12 => &[
                (vk::Format::R8_UNORM, width, height, 1),
                (vk::Format::R8G8_UNORM, half_width, half_height, 2),
            ],
            Self::Yuv420p => &[
                (vk::Format::R8_UNORM, width, height, 1),
                (vk::Format::R8_UNORM, half_width, half_height, 1),
                (vk::Format::R8_UNORM, half_width, half_height, 1),
            ],
            Self::Bgra => &[(vk::Format::B8G8R8A8_UNORM, width, height, 4)],
        };
        let mut offset = 0u64;
        shapes
            .iter()
            .map(|&(format, width, height, bytes_per_pixel)| {
                let shape = PlaneShape {
                    format,
                    width,
                    height,
                    bytes_per_pixel,
                    offset,
                };
                offset = (offset + shape.bytes()).next_multiple_of(16);
                shape
            })
            .collect()
    }
}

/// One plane of a frame: the image it is sampled from, and where its rows
/// sit in the staging buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PlaneShape {
    format: vk::Format,
    width: u32,
    height: u32,
    bytes_per_pixel: u32,
    offset: u64,
}

impl PlaneShape {
    /// One row's bytes, packed: the staging buffer has no padding between
    /// rows.
    fn row_bytes(&self) -> usize {
        (self.width * self.bytes_per_pixel) as usize
    }

    fn bytes(&self) -> u64 {
        self.row_bytes() as u64 * u64::from(self.height)
    }
}

/// Where a frame's planes are.
enum Source<'a> {
    /// In system memory: copied into mapped memory on the CPU.
    System(&'a ffmpeg::frame::Video),
    /// In CUDA memory: copied device to device into memory CUDA imported.
    /// Each plane's device pointer and pitch, as many as the layout has.
    #[cfg(feature = "cuda")]
    Cuda {
        planes: [(u64, usize); 3],
        interop: &'a Arc<CudaInterop>,
    },
}

impl Source<'_> {
    fn domain(&self) -> MemoryDomain {
        match self {
            Self::System(_) => MemoryDomain::System,
            #[cfg(feature = "cuda")]
            Self::Cuda { .. } => MemoryDomain::Cuda,
        }
    }
}

/// The swapchain, the pipeline and the per-frame resources a window is drawn
/// with.
///
/// Everything a frame's memory domain decides happens before the staging
/// buffer: how its bytes get there. From the buffer on — the copy into the
/// sampled images, the draw, the present — nothing here knows or cares where
/// a frame came from; what the layout decides is how many images there are
/// and which shader samples them.
struct Presenter {
    gpu: Arc<VulkanShared>,
    surface_fn: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    swapchain_fn: ash::khr::swapchain::Device,
    format: vk::Format,
    /// Made at the first frame, at the size then known, and remade whenever
    /// the window's size is no longer its own.
    swapchain: Option<Swapchain>,
    render_pass: vk::RenderPass,
    descriptor_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    /// One per [`Layout`], in [`Layout::ALL`]'s order.
    pipelines: [vk::Pipeline; 3],
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    sampler: vk::Sampler,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    image_available: vk::Semaphore,
    in_flight: vk::Fence,
    /// Whether `in_flight` has been submitted since it was last waited for
    /// — a fence starts unsignalled, so a frame with nothing before it must
    /// not wait on it.
    submitted: bool,
    video: Option<VideoResources>,
    size: WindowSize,
    /// Keeps the window alive until the surface on it is destroyed, which is
    /// why it is the last field: fields drop after `Drop::drop` has run.
    _window: Arc<dyn Any + Send + Sync>,
}

// SAFETY: every Vulkan handle here is a plain value, and the one raw pointer
// — a system staging buffer's mapping — is only written through `&mut self`.
// The queue these submit on is reached only through `VulkanShared::queue`'s
// lock.
unsafe impl Send for Presenter {}

struct Swapchain {
    handle: vk::SwapchainKHR,
    extent: vk::Extent2D,
    views: Vec<vk::ImageView>,
    framebuffers: Vec<vk::Framebuffer>,
    /// One per image rather than one for all: a present may still be
    /// waiting on the one signalled for the image before, and signalling a
    /// semaphore a present is waiting on is undefined.
    render_finished: Vec<vk::Semaphore>,
}

/// The frame-sized resources: the images the shader samples, one per plane,
/// and the buffer a frame's bytes are put in on their way there.
struct VideoResources {
    width: u32,
    height: u32,
    layout: Layout,
    shapes: Vec<PlaneShape>,
    planes: Vec<Plane>,
    staging: Staging,
}

struct Plane {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
}

/// One frame, its planes packed as [`Layout::planes`] lays them out.
enum Staging {
    /// Mapped for the CPU, and left mapped.
    System {
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
        mapped: *mut u8,
    },
    /// Allocated exportable and imported into CUDA once.
    #[cfg(feature = "cuda")]
    Cuda {
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
        /// Dropped before `memory` is freed: the mapping refers to it.
        imported: Option<ImportedMemory>,
    },
}

impl Staging {
    fn domain(&self) -> MemoryDomain {
        match self {
            Self::System { .. } => MemoryDomain::System,
            #[cfg(feature = "cuda")]
            Self::Cuda { .. } => MemoryDomain::Cuda,
        }
    }

    fn buffer(&self) -> vk::Buffer {
        match self {
            Self::System { buffer, .. } => *buffer,
            #[cfg(feature = "cuda")]
            Self::Cuda { buffer, .. } => *buffer,
        }
    }
}

impl Presenter {
    fn new<W>(
        gpu: &Arc<VulkanShared>,
        display: RawDisplayHandle,
        handle: RawWindowHandle,
        window: Arc<W>,
        size: WindowSize,
    ) -> std::result::Result<Self, VulkanWindowRendererError>
    where
        W: Send + Sync + 'static,
    {
        let gpu = Arc::clone(gpu);
        // SAFETY: `display` and `handle` come from a window the caller passed
        // in, whose `Arc` is kept in the returned presenter until after the
        // surface is destroyed. The instance was made with every window-surface
        // extension this Vulkan has, which is what `create_surface` needs for
        // the kind of window this is.
        let surface =
            unsafe { ash_window::create_surface(&gpu.entry, &gpu.instance, display, handle, None) }
                .map_err(VulkanWindowRendererError::Surface)?;
        let surface_fn = ash::khr::surface::Instance::new(&gpu.entry, &gpu.instance);
        let destroy_surface = |error| {
            // SAFETY: nothing has been made from the surface yet.
            unsafe { surface_fn.destroy_surface(surface, None) };
            error
        };

        // SAFETY: a plain query of the device, the queue family and the surface
        // just made.
        let supported = unsafe {
            surface_fn.get_physical_device_surface_support(
                gpu.physical_device,
                gpu.queue_family,
                surface,
            )
        }
        .map_err(call("vkGetPhysicalDeviceSurfaceSupportKHR"))
        .map_err(destroy_surface)?;
        if !supported {
            return Err(destroy_surface(VulkanWindowRendererError::CannotPresent(
                gpu.name.clone(),
            )));
        }
        let format =
            pick_format(&surface_fn, gpu.physical_device, surface).map_err(destroy_surface)?;

        let device = &gpu.device;
        let render_pass = create_render_pass(device, format).map_err(destroy_surface)?;
        let pipeline = create_pipeline(device, render_pass);
        let (descriptor_layout, pipeline_layout, pipelines) = match pipeline {
            Ok(made) => made,
            Err(error) => {
                // SAFETY: made just above and used by nothing yet.
                unsafe { device.destroy_render_pass(render_pass, None) };
                return Err(destroy_surface(error));
            }
        };

        // From here the presenter owns what has been made, so a failure is
        // cleaned up by its `Drop`: every handle below starts null, which the
        // destroy calls ignore.
        let mut presenter = Self {
            gpu: Arc::clone(&gpu),
            surface_fn,
            surface,
            swapchain_fn: ash::khr::swapchain::Device::new(&gpu.instance, &gpu.device),
            format,
            swapchain: None,
            render_pass,
            descriptor_layout,
            pipeline_layout,
            pipelines,
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_set: vk::DescriptorSet::null(),
            sampler: vk::Sampler::null(),
            command_pool: vk::CommandPool::null(),
            command_buffer: vk::CommandBuffer::null(),
            image_available: vk::Semaphore::null(),
            in_flight: vk::Fence::null(),
            submitted: false,
            video: None,
            size,
            _window: window,
        };
        presenter.finish()?;
        Ok(presenter)
    }

    /// Makes the rest: the descriptor set, the sampler, the command buffer
    /// and the synchronization objects.
    fn finish(&mut self) -> std::result::Result<(), VulkanWindowRendererError> {
        let device = &self.gpu.device;
        let sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLER)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(3),
        ];
        // SAFETY: `sizes` outlives the call.
        self.descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&sizes)
                    .max_sets(1),
                None,
            )
        }
        .map_err(call("vkCreateDescriptorPool"))?;
        let layouts = [self.descriptor_layout];
        // SAFETY: the pool and layout are this presenter's own and live.
        self.descriptor_set = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.descriptor_pool)
                    .set_layouts(&layouts),
            )
        }
        .map_err(call("vkAllocateDescriptorSets"))?[0];
        // SAFETY: a plain object with no references.
        self.sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )
        }
        .map_err(call("vkCreateSampler"))?;
        // SAFETY: the queue family is the device's own.
        self.command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(self.gpu.queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .map_err(call("vkCreateCommandPool"))?;
        // SAFETY: the pool was made just above.
        self.command_buffer = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(call("vkAllocateCommandBuffers"))?[0];
        // SAFETY: plain objects with no references.
        self.image_available =
            unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
                .map_err(call("vkCreateSemaphore"))?;
        // SAFETY: as above.
        self.in_flight = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(call("vkCreateFence"))?;
        Ok(())
    }

    fn present(
        &mut self,
        width: u32,
        height: u32,
        layout: Layout,
        colour: Colour,
        source: Source<'_>,
    ) -> std::result::Result<(), VulkanWindowRendererError> {
        // The last frame's commands read the staging buffer this one is about
        // to overwrite.
        if self.submitted {
            // SAFETY: the fence is this presenter's own and was submitted.
            unsafe {
                self.gpu
                    .device
                    .wait_for_fences(&[self.in_flight], true, u64::MAX)
            }
            .map_err(call("vkWaitForFences"))?;
            self.submitted = false;
        }
        self.ensure_video(width, height, layout, source.domain())?;
        self.stage(source)?;
        self.draw(width, height, colour)
    }

    /// Puts one frame's planes into the staging buffer, packed.
    fn stage(&mut self, source: Source<'_>) -> std::result::Result<(), VulkanWindowRendererError> {
        let video = self.video.as_ref().expect("ensured before staging");
        match (source, &video.staging) {
            (Source::System(frame), Staging::System { mapped, .. }) => {
                for (index, shape) in video.shapes.iter().enumerate() {
                    let (data, pitch) = (frame.data(index), frame.stride(index));
                    let row_bytes = shape.row_bytes();
                    // SAFETY: the mapping holds every plane `ensure_video` made it
                    // for, this one at `offset` for `bytes`, and nothing else
                    // writes it while `&mut self` is held — the GPU's last read of
                    // it was waited for in `present`.
                    let packed = unsafe {
                        std::slice::from_raw_parts_mut(
                            mapped.add(shape.offset as usize),
                            shape.bytes() as usize,
                        )
                    };
                    for (row, out) in packed.chunks_exact_mut(row_bytes).enumerate() {
                        out.copy_from_slice(&data[row * pitch..row * pitch + row_bytes]);
                    }
                }
                Ok(())
            }
            #[cfg(feature = "cuda")]
            (Source::Cuda { planes, interop }, Staging::Cuda { imported, .. }) => {
                let dst = imported.as_ref().expect("imported with the buffer").ptr();
                let copies: Vec<DeviceRows> = video
                    .shapes
                    .iter()
                    .zip(planes)
                    .map(|(shape, (src, src_pitch))| DeviceRows {
                        src,
                        src_pitch,
                        width_bytes: shape.row_bytes(),
                        rows: shape.height as usize,
                        dst_offset: shape.offset,
                        dst_pitch: shape.row_bytes(),
                    })
                    .collect();
                // SAFETY: every source is a plane of a frame `draw` validated as
                // this layout on this GPU's CUDA device, as many rows of its own
                // pitch as the shape says; the destination is the mapping of a
                // buffer made to hold exactly these planes.
                unsafe { interop.copy_rows(dst, &copies)? };
                Ok(())
            }
            #[cfg(feature = "cuda")]
            _ => unreachable!("the staging buffer is made for the domain of the frame"),
        }
    }

    /// Draws what is in the staging buffer into the window, letterboxed, and
    /// presents it.
    fn draw(
        &mut self,
        width: u32,
        height: u32,
        colour: Colour,
    ) -> std::result::Result<(), VulkanWindowRendererError> {
        let target = self.target_extent(width, height)?;
        // Minimized: nothing to draw into, and a swapchain cannot be
        // zero-sized. The next frame will look again.
        if target.width == 0 || target.height == 0 {
            return Ok(());
        }
        if self
            .swapchain
            .as_ref()
            .is_none_or(|swapchain| swapchain.extent != target)
        {
            self.remake_swapchain(target)?;
        }
        let image = match self.acquire() {
            Ok(image) => image,
            Err(VulkanWindowRendererError::Call {
                result: vk::Result::ERROR_OUT_OF_DATE_KHR,
                ..
            }) => {
                self.remake_swapchain(target)?;
                self.acquire()?
            }
            Err(error) => return Err(error),
        };

        // SAFETY: the fence is this presenter's own and not in use: the last
        // submission on it was waited for in `present`.
        unsafe { self.gpu.device.reset_fences(&[self.in_flight]) }
            .map_err(call("vkResetFences"))?;
        self.record(image as usize, [width, height], colour)?;

        let swapchain = self.swapchain.as_ref().expect("made above");
        let wait = [self.image_available];
        let signal = [swapchain.render_finished[image as usize]];
        let stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let buffers = [self.command_buffer];
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&wait)
            .wait_dst_stage_mask(&stages)
            .command_buffers(&buffers)
            .signal_semaphores(&signal);
        let handles = [swapchain.handle];
        let indices = [image];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal)
            .swapchains(&handles)
            .image_indices(&indices);
        let presented = {
            // One lock for the submit and the present: both use the queue, and
            // another renderer on this GPU must not come between them.
            let queue = self.gpu.queue();
            // SAFETY: everything `submit` refers to lives until the call returns,
            // the command buffer was recorded just above, and the queue is held.
            unsafe {
                self.gpu
                    .device
                    .queue_submit(*queue, &[submit], self.in_flight)
            }
            .map_err(call("vkQueueSubmit"))?;
            self.submitted = true;
            // SAFETY: as above; the image was acquired from this swapchain.
            unsafe { self.swapchain_fn.queue_present(*queue, &present_info) }
        };
        match presented {
            Ok(false) => Ok(()),
            // Suboptimal or out of date: the next frame remakes it, at whatever
            // size the window is by then.
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.swapchain_stale();
                Ok(())
            }
            Err(result) => Err(VulkanWindowRendererError::Call {
                call: "vkQueuePresentKHR",
                result,
            }),
        }
    }

    /// The size of the swapchain drawn into last, if there is one.
    fn extent(&self) -> Option<vk::Extent2D> {
        self.swapchain
            .as_ref()
            .map(|swapchain| swapchain.extent)
            .filter(|extent| extent.width != 0)
    }

    fn acquire(&self) -> std::result::Result<u32, VulkanWindowRendererError> {
        let swapchain = self.swapchain.as_ref().expect("made before acquiring");
        // SAFETY: the swapchain and semaphore are this presenter's own; the
        // semaphore is unsignalled, its last wait having been submitted.
        unsafe {
            self.swapchain_fn.acquire_next_image(
                swapchain.handle,
                u64::MAX,
                self.image_available,
                vk::Fence::null(),
            )
        }
        .map(|(image, _suboptimal)| image)
        .map_err(call("vkAcquireNextImageKHR"))
    }

    /// Marks the swapchain as needing to be remade before the next frame.
    fn swapchain_stale(&mut self) {
        if let Some(swapchain) = self.swapchain.as_mut() {
            swapchain.extent = vk::Extent2D::default();
        }
    }

    /// The size to draw at: the window's own where it has one to report, the
    /// size the application last set where it does not, and otherwise the
    /// frame's.
    fn target_extent(
        &self,
        width: u32,
        height: u32,
    ) -> std::result::Result<vk::Extent2D, VulkanWindowRendererError> {
        // SAFETY: a plain query of the device and this presenter's surface.
        let caps = unsafe {
            self.surface_fn
                .get_physical_device_surface_capabilities(self.gpu.physical_device, self.surface)
        }
        .map_err(call("vkGetPhysicalDeviceSurfaceCapabilitiesKHR"))?;
        if caps.current_extent.width != u32::MAX {
            return Ok(caps.current_extent);
        }
        let wanted = self.size.get().unwrap_or(vk::Extent2D { width, height });
        Ok(vk::Extent2D {
            width: wanted.width.clamp(
                caps.min_image_extent.width,
                caps.max_image_extent.width.max(caps.min_image_extent.width),
            ),
            height: wanted.height.clamp(
                caps.min_image_extent.height,
                caps.max_image_extent
                    .height
                    .max(caps.min_image_extent.height),
            ),
        })
    }

    fn remake_swapchain(
        &mut self,
        extent: vk::Extent2D,
    ) -> std::result::Result<(), VulkanWindowRendererError> {
        // SAFETY: waits for everything submitted, so nothing still uses the
        // swapchain being replaced.
        unsafe { self.gpu.device.device_wait_idle() }.map_err(call("vkDeviceWaitIdle"))?;
        self.submitted = false;
        let old = self.swapchain.take();
        let made = create_swapchain(
            self,
            extent,
            old.as_ref()
                .map_or(vk::SwapchainKHR::null(), |old| old.handle),
        );
        if let Some(old) = old {
            self.destroy_swapchain(old);
        }
        self.swapchain = Some(made?);
        Ok(())
    }

    fn destroy_swapchain(&self, swapchain: Swapchain) {
        let device = &self.gpu.device;
        // SAFETY: the device is idle — every caller waited first — so nothing
        // uses these any more, and each is destroyed once.
        unsafe {
            for framebuffer in swapchain.framebuffers {
                device.destroy_framebuffer(framebuffer, None);
            }
            for view in swapchain.views {
                device.destroy_image_view(view, None);
            }
            for semaphore in swapchain.render_finished {
                device.destroy_semaphore(semaphore, None);
            }
            self.swapchain_fn.destroy_swapchain(swapchain.handle, None);
        }
    }

    /// (Re)makes the frame-sized resources when the frame's size, layout, or
    /// the domain it arrives in, is not what they were made for.
    fn ensure_video(
        &mut self,
        width: u32,
        height: u32,
        layout: Layout,
        domain: MemoryDomain,
    ) -> std::result::Result<(), VulkanWindowRendererError> {
        if self.video.as_ref().is_some_and(|video| {
            video.width == width
                && video.height == height
                && video.layout == layout
                && video.staging.domain() == domain
        }) {
            return Ok(());
        }
        // SAFETY: waits for everything submitted, so the resources being
        // replaced are no longer read.
        unsafe { self.gpu.device.device_wait_idle() }.map_err(call("vkDeviceWaitIdle"))?;
        self.submitted = false;
        if let Some(video) = self.video.take() {
            self.destroy_video(video);
        }

        let shapes = layout.planes(width, height);
        let mut planes = Vec::with_capacity(shapes.len());
        for shape in &shapes {
            match self.create_plane(shape.width, shape.height, shape.format) {
                Ok(plane) => planes.push(plane),
                Err(error) => {
                    planes
                        .into_iter()
                        .for_each(|plane| self.destroy_plane(plane));
                    return Err(error);
                }
            }
        }
        let last = shapes.last().expect("every layout has a plane");
        let bytes = last.offset + last.bytes();
        let staging = match domain {
            #[cfg(feature = "cuda")]
            MemoryDomain::Cuda => self.create_cuda_staging(bytes),
            _ => self.create_system_staging(bytes),
        };
        let staging = match staging {
            Ok(staging) => staging,
            Err(error) => {
                planes
                    .into_iter()
                    .for_each(|plane| self.destroy_plane(plane));
                return Err(error);
            }
        };

        // Bindings 1 to 3, one per plane; a layout with fewer points the rest
        // at its first, which its shader never samples but which keeps every
        // binding valid.
        let images: Vec<vk::DescriptorImageInfo> = (0..3)
            .map(|index| {
                vk::DescriptorImageInfo::default()
                    .image_view(planes.get(index).unwrap_or(&planes[0]).view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            })
            .collect();
        let sampler = [vk::DescriptorImageInfo::default().sampler(self.sampler)];
        let mut writes = vec![
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .image_info(&sampler),
        ];
        for (index, image) in images.iter().enumerate() {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_set)
                    .dst_binding(index as u32 + 1)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(std::slice::from_ref(image)),
            );
        }
        // SAFETY: the set is not in use — the device is idle — and every view
        // and the sampler it is pointed at are live.
        unsafe { self.gpu.device.update_descriptor_sets(&writes, &[]) };

        self.video = Some(VideoResources {
            width,
            height,
            layout,
            shapes,
            planes,
            staging,
        });
        Ok(())
    }

    fn create_plane(
        &self,
        width: u32,
        height: u32,
        format: vk::Format,
    ) -> std::result::Result<Plane, VulkanWindowRendererError> {
        let device = &self.gpu.device;
        // SAFETY: a plain image with no references.
        let image = unsafe {
            device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
        }
        .map_err(call("vkCreateImage"))?;
        let destroy_image = |error| {
            // SAFETY: made just above, bound to nothing that outlives it.
            unsafe { device.destroy_image(image, None) };
            error
        };
        // SAFETY: a plain query of the image just made.
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let index = self
            .memory_type(requirements, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .ok_or(VulkanWindowRendererError::NoMemoryType("device-local"))
            .map_err(destroy_image)?;
        // SAFETY: the size and type come from the image's own requirements.
        let memory = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(index),
                None,
            )
        }
        .map_err(call("vkAllocateMemory"))
        .map_err(destroy_image)?;
        let free = |error| {
            // SAFETY: made just above; nothing uses either yet.
            unsafe {
                device.destroy_image(image, None);
                device.free_memory(memory, None);
            }
            error
        };
        // SAFETY: the memory was allocated for this image's requirements.
        unsafe { device.bind_image_memory(image, memory, 0) }
            .map_err(call("vkBindImageMemory"))
            .map_err(free)?;
        // SAFETY: the image is bound and the view describes its one level and
        // layer in its own format.
        let view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1),
                    ),
                None,
            )
        }
        .map_err(call("vkCreateImageView"))
        .map_err(free)?;
        Ok(Plane {
            image,
            memory,
            view,
        })
    }

    fn destroy_plane(&self, plane: Plane) {
        let device = &self.gpu.device;
        // SAFETY: nothing reads the plane — every caller waited for the device
        // or has not submitted it — and each handle is destroyed once.
        unsafe {
            device.destroy_image_view(plane.view, None);
            device.destroy_image(plane.image, None);
            device.free_memory(plane.memory, None);
        }
    }

    fn create_system_staging(
        &self,
        bytes: u64,
    ) -> std::result::Result<Staging, VulkanWindowRendererError> {
        let device = &self.gpu.device;
        // SAFETY: a plain buffer with no references.
        let buffer = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(bytes)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }
        .map_err(call("vkCreateBuffer"))?;
        let destroy_buffer = |error| {
            // SAFETY: made just above, bound to nothing.
            unsafe { device.destroy_buffer(buffer, None) };
            error
        };
        // SAFETY: a plain query of the buffer just made.
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let index = self
            .memory_type(
                requirements,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .ok_or(VulkanWindowRendererError::NoMemoryType("host-visible"))
            .map_err(destroy_buffer)?;
        // SAFETY: the size and type come from the buffer's own requirements.
        let memory = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(index),
                None,
            )
        }
        .map_err(call("vkAllocateMemory"))
        .map_err(destroy_buffer)?;
        let free = |error| {
            // SAFETY: made just above; nothing uses either yet.
            unsafe {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            }
            error
        };
        // SAFETY: the memory was allocated for this buffer's requirements.
        unsafe { device.bind_buffer_memory(buffer, memory, 0) }
            .map_err(call("vkBindBufferMemory"))
            .map_err(free)?;
        // SAFETY: host-visible memory, mapped once over the buffer's bytes and
        // left mapped until it is freed.
        let mapped = unsafe { device.map_memory(memory, 0, bytes, vk::MemoryMapFlags::empty()) }
            .map_err(call("vkMapMemory"))
            .map_err(free)?;
        Ok(Staging::System {
            buffer,
            memory,
            mapped: mapped.cast(),
        })
    }

    /// Allocates the staging buffer exportable, and imports it into CUDA —
    /// the one direction the two will share memory in.
    #[cfg(feature = "cuda")]
    fn create_cuda_staging(
        &self,
        bytes: u64,
    ) -> std::result::Result<Staging, VulkanWindowRendererError> {
        let pairing = self
            .gpu
            .cuda
            .as_ref()
            .ok_or(VulkanWindowRendererError::NotForCuda)?;
        let device = &self.gpu.device;
        let mut external = vk::ExternalMemoryBufferCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        // SAFETY: `external` outlives the call.
        let buffer = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(bytes)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .push_next(&mut external),
                None,
            )
        }
        .map_err(call("vkCreateBuffer"))?;
        let destroy_buffer = |error| {
            // SAFETY: made just above, bound to nothing.
            unsafe { device.destroy_buffer(buffer, None) };
            error
        };
        // SAFETY: a plain query of the buffer just made.
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let index = self
            .memory_type(requirements, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .ok_or(VulkanWindowRendererError::NoMemoryType("device-local"))
            .map_err(destroy_buffer)?;
        let mut export = vk::ExportMemoryAllocateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        // Dedicated: CUDA imports the whole allocation, so it backs this
        // buffer and nothing else.
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
        // SAFETY: both chained structs outlive the call, and the size and type
        // come from the buffer's own requirements.
        let memory = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(index)
                    .push_next(&mut export)
                    .push_next(&mut dedicated),
                None,
            )
        }
        .map_err(call("vkAllocateMemory"))
        .map_err(destroy_buffer)?;
        let free = |error| {
            // SAFETY: made just above; nothing uses either yet.
            unsafe {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            }
            error
        };
        // SAFETY: the memory was allocated for this buffer's requirements.
        unsafe { device.bind_buffer_memory(buffer, memory, 0) }
            .map_err(call("vkBindBufferMemory"))
            .map_err(free)?;
        let external_fn =
            ash::khr::external_memory_fd::Device::new(&self.gpu.instance, &self.gpu.device);
        // SAFETY: the memory was allocated exportable as an opaque fd, which the
        // device enabled `VK_KHR_external_memory_fd` for when it was made for CUDA.
        let fd = unsafe {
            external_fn.get_memory_fd(
                &vk::MemoryGetFdInfoKHR::default()
                    .memory(memory)
                    .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD),
            )
        }
        .map_err(call("vkGetMemoryFdKHR"))
        .map_err(free)?;
        // SAFETY: `vkGetMemoryFdKHR` hands out a new descriptor the caller owns.
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        // The whole allocation, not the buffer's own size: a dedicated
        // allocation is imported as what it is.
        let imported = pairing
            .interop
            .import(fd, requirements.size)
            .map_err(|error| free(error.into()))?;
        Ok(Staging::Cuda {
            buffer,
            memory,
            imported: Some(imported),
        })
    }

    fn destroy_video(&self, video: VideoResources) {
        let VideoResources {
            planes, staging, ..
        } = video;
        planes
            .into_iter()
            .for_each(|plane| self.destroy_plane(plane));
        let device = &self.gpu.device;
        match staging {
            Staging::System { buffer, memory, .. } => {
                // SAFETY: the device is idle, and freeing the memory unmaps it.
                unsafe {
                    device.destroy_buffer(buffer, None);
                    device.free_memory(memory, None);
                }
            }
            #[cfg(feature = "cuda")]
            Staging::Cuda {
                buffer,
                memory,
                mut imported,
            } => {
                // CUDA's mapping first: it refers to the allocation below.
                drop(imported.take());
                // SAFETY: the device is idle and CUDA has let go of the memory.
                unsafe {
                    device.destroy_buffer(buffer, None);
                    device.free_memory(memory, None);
                }
            }
        }
    }

    fn memory_type(
        &self,
        requirements: vk::MemoryRequirements,
        flags: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        let properties = &self.gpu.memory_properties;
        (0..properties.memory_type_count).find(|&index| {
            requirements.memory_type_bits & (1 << index) != 0
                && properties.memory_types[index as usize]
                    .property_flags
                    .contains(flags)
        })
    }

    /// Records one frame: the staging buffer into the plane images, then the
    /// draw.
    fn record(
        &self,
        image: usize,
        frame: [u32; 2],
        colour: Colour,
    ) -> std::result::Result<(), VulkanWindowRendererError> {
        let device = &self.gpu.device;
        let cmd = self.command_buffer;
        let video = self.video.as_ref().expect("ensured before drawing");
        let swapchain = self.swapchain.as_ref().expect("made before drawing");
        // SAFETY: the command buffer is not in use — its last submission was
        // waited for in `present` — and every handle recorded into it is live
        // for as long as that submission runs, the device being waited for
        // before any of them is destroyed.
        unsafe {
            device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .map_err(call("vkResetCommandBuffer"))?;
            device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(call("vkBeginCommandBuffer"))?;

            for (plane, shape) in video.planes.iter().zip(&video.shapes) {
                transition(
                    device,
                    cmd,
                    plane.image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                );
                let region = vk::BufferImageCopy::default()
                    .buffer_offset(shape.offset)
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1),
                    )
                    .image_extent(vk::Extent3D {
                        width: shape.width,
                        height: shape.height,
                        depth: 1,
                    });
                device.cmd_copy_buffer_to_image(
                    cmd,
                    video.staging.buffer(),
                    plane.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                );
                transition(
                    device,
                    cmd,
                    plane.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                );
            }

            // Black everywhere first, which is what the bars are.
            let clear = [vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            }];
            let whole = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: swapchain.extent,
            };
            device.cmd_begin_render_pass(
                cmd,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(self.render_pass)
                    .framebuffer(swapchain.framebuffers[image])
                    .render_area(whole)
                    .clear_values(&clear),
                vk::SubpassContents::INLINE,
            );
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipelines[video.layout.index()],
            );
            device.cmd_set_viewport(cmd, 0, &[letterbox(frame, swapchain.extent)]);
            device.cmd_set_scissor(cmd, 0, &[whole]);
            device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                &colour.bytes(),
            );
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );
            device.cmd_draw(cmd, 3, 1, 0, 0);
            device.cmd_end_render_pass(cmd);
            device
                .end_command_buffer(cmd)
                .map_err(call("vkEndCommandBuffer"))?;
        }
        Ok(())
    }
}

impl Drop for Presenter {
    fn drop(&mut self) {
        // SAFETY: waits for everything this presenter submitted, so nothing it
        // destroys below is still in use. Each handle is destroyed once; the
        // ones never made are null, which every destroy call ignores.
        unsafe {
            let _ = self.gpu.device.device_wait_idle();
        }
        if let Some(video) = self.video.take() {
            self.destroy_video(video);
        }
        if let Some(swapchain) = self.swapchain.take() {
            self.destroy_swapchain(swapchain);
        }
        let device = &self.gpu.device;
        // SAFETY: as above.
        unsafe {
            device.destroy_fence(self.in_flight, None);
            device.destroy_semaphore(self.image_available, None);
            device.destroy_command_pool(self.command_pool, None);
            device.destroy_sampler(self.sampler, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            for pipeline in self.pipelines {
                device.destroy_pipeline(pipeline, None);
            }
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_layout, None);
            device.destroy_render_pass(self.render_pass, None);
            self.surface_fn.destroy_surface(self.surface, None);
        }
    }
}

/// The largest rectangle of the frame's aspect ratio that fits the window,
/// centred — the part of the window the picture is drawn in, the rest being
/// left black.
fn letterbox(frame: [u32; 2], window: vk::Extent2D) -> vk::Viewport {
    let (frame_width, frame_height) = (frame[0] as f32, frame[1] as f32);
    let (window_width, window_height) = (window.width as f32, window.height as f32);
    let scale = (window_width / frame_width).min(window_height / frame_height);
    let (width, height) = (frame_width * scale, frame_height * scale);
    vk::Viewport {
        x: (window_width - width) * 0.5,
        y: (window_height - height) * 0.5,
        width,
        height,
        min_depth: 0.0,
        max_depth: 1.0,
    }
}

fn transition(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    from: vk::ImageLayout,
    to: vk::ImageLayout,
) {
    let (src_access, dst_access, src_stage, dst_stage) = match from {
        vk::ImageLayout::UNDEFINED => (
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
        ),
        _ => (
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        ),
    };
    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(from)
        .new_layout(to)
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1),
        );
    // SAFETY: recorded into a command buffer in the recording state, for an
    // image of this device, as `record`'s own contract says.
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        )
    };
}

/// B8G8R8A8 UNORM where the surface offers it. UNORM rather than SRGB: the
/// shader's output is R'G'B' already, gamma and all, and an sRGB target
/// would encode it a second time.
fn pick_format(
    surface_fn: &ash::khr::surface::Instance,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
) -> std::result::Result<vk::Format, VulkanWindowRendererError> {
    // SAFETY: a plain query of the device and a live surface.
    let formats =
        unsafe { surface_fn.get_physical_device_surface_formats(physical_device, surface) }
            .map_err(call("vkGetPhysicalDeviceSurfaceFormatsKHR"))?;
    formats
        .iter()
        .find(|format| format.format == vk::Format::B8G8R8A8_UNORM)
        .or_else(|| {
            formats
                .iter()
                .find(|format| format.format == vk::Format::R8G8B8A8_UNORM)
        })
        .or_else(|| formats.first())
        .map(|format| format.format)
        .ok_or(VulkanWindowRendererError::CannotPresent(
            "a surface offering no formats".into(),
        ))
}

fn create_render_pass(
    device: &ash::Device,
    format: vk::Format,
) -> std::result::Result<vk::RenderPass, VulkanWindowRendererError> {
    let attachment = [vk::AttachmentDescription::default()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)];
    let color = [vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
    let subpass = [vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color)];
    let dependency = [vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];
    // SAFETY: every array the info points at outlives the call.
    unsafe {
        device.create_render_pass(
            &vk::RenderPassCreateInfo::default()
                .attachments(&attachment)
                .subpasses(&subpass)
                .dependencies(&dependency),
            None,
        )
    }
    .map_err(call("vkCreateRenderPass"))
}

fn create_swapchain(
    presenter: &Presenter,
    extent: vk::Extent2D,
    old: vk::SwapchainKHR,
) -> std::result::Result<Swapchain, VulkanWindowRendererError> {
    let device = &presenter.gpu.device;
    // SAFETY: a plain query of the device and the presenter's surface.
    let caps = unsafe {
        presenter
            .surface_fn
            .get_physical_device_surface_capabilities(
                presenter.gpu.physical_device,
                presenter.surface,
            )
    }
    .map_err(call("vkGetPhysicalDeviceSurfaceCapabilitiesKHR"))?;
    let count = (caps.min_image_count + 1).min(if caps.max_image_count == 0 {
        u32::MAX
    } else {
        caps.max_image_count
    });
    // SAFETY: the surface is live, `old` is either null or the swapchain being
    // replaced, and every value comes from the surface's own capabilities.
    let handle = unsafe {
        presenter.swapchain_fn.create_swapchain(
            &vk::SwapchainCreateInfoKHR::default()
                .surface(presenter.surface)
                .min_image_count(count)
                .image_format(presenter.format)
                .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(caps.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(vk::PresentModeKHR::FIFO)
                .clipped(true)
                .old_swapchain(old),
            None,
        )
    }
    .map_err(call("vkCreateSwapchainKHR"))?;

    let mut swapchain = Swapchain {
        handle,
        extent,
        views: Vec::new(),
        framebuffers: Vec::new(),
        render_finished: Vec::new(),
    };
    // SAFETY: the swapchain was made just above.
    let images = match unsafe { presenter.swapchain_fn.get_swapchain_images(handle) } {
        Ok(images) => images,
        Err(result) => {
            presenter.destroy_swapchain(swapchain);
            return Err(VulkanWindowRendererError::Call {
                call: "vkGetSwapchainImagesKHR",
                result,
            });
        }
    };
    for image in images {
        let made = (|| {
            // SAFETY: the image is one of this swapchain's, in its format.
            let view = unsafe {
                device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(presenter.format)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .level_count(1)
                                .layer_count(1),
                        ),
                    None,
                )
            }
            .map_err(call("vkCreateImageView"))?;
            swapchain.views.push(view);
            let attachments = [view];
            // SAFETY: the render pass and view are live, and the size is the
            // swapchain's own.
            let framebuffer = unsafe {
                device.create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(presenter.render_pass)
                        .attachments(&attachments)
                        .width(extent.width)
                        .height(extent.height)
                        .layers(1),
                    None,
                )
            }
            .map_err(call("vkCreateFramebuffer"))?;
            swapchain.framebuffers.push(framebuffer);
            // SAFETY: a plain object with no references.
            let semaphore =
                unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
                    .map_err(call("vkCreateSemaphore"))?;
            swapchain.render_finished.push(semaphore);
            Ok(())
        })();
        if let Err(error) = made {
            presenter.destroy_swapchain(swapchain);
            return Err(error);
        }
    }
    Ok(swapchain)
}

/// The pipelines, one per [`Layout`] in [`Layout::ALL`]'s order: made up
/// front rather than at the first frame of each, since there are three and
/// they differ only in their fragment shader.
fn create_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
) -> std::result::Result<
    (
        vk::DescriptorSetLayout,
        vk::PipelineLayout,
        [vk::Pipeline; 3],
    ),
    VulkanWindowRendererError,
> {
    let spirv = compile_shader(SHADER)?;
    // SAFETY: `spirv` is a validated module naga wrote, and outlives the call.
    let module = unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spirv), None)
    }
    .map_err(call("vkCreateShaderModule"))?;

    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(3)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ];
    let made = (|| {
        // SAFETY: `bindings` outlives the call.
        let descriptor_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }
        .map_err(call("vkCreateDescriptorSetLayout"))?;
        let layouts = [descriptor_layout];
        let constants = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<Colour>() as u32)];
        // SAFETY: both arrays outlive the call and the layout is live.
        let pipeline_layout = match unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&layouts)
                    .push_constant_ranges(&constants),
                None,
            )
        } {
            Ok(layout) => layout,
            Err(result) => {
                // SAFETY: made just above and used by nothing.
                unsafe { device.destroy_descriptor_set_layout(descriptor_layout, None) };
                return Err(VulkanWindowRendererError::Call {
                    call: "vkCreatePipelineLayout",
                    result,
                });
            }
        };

        let stages = Layout::ALL.map(|layout| {
            [
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(module)
                    .name(c"vs_main"),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(module)
                    .name(layout.fragment_shader()),
            ]
        });
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let blend_attachment = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachment);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let info = stages.each_ref().map(|stages| {
            vk::GraphicsPipelineCreateInfo::default()
                .stages(stages)
                .vertex_input_state(&vertex_input)
                .input_assembly_state(&assembly)
                .viewport_state(&viewport)
                .rasterization_state(&raster)
                .multisample_state(&multisample)
                .color_blend_state(&blend)
                .dynamic_state(&dynamic)
                .layout(pipeline_layout)
                .render_pass(render_pass)
                .subpass(0)
        });
        // SAFETY: every struct the infos point at outlives the call, and the
        // module, layout and render pass are live.
        match unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &info, None) } {
            Ok(pipelines) => Ok((
                descriptor_layout,
                pipeline_layout,
                [pipelines[0], pipelines[1], pipelines[2]],
            )),
            Err((made, result)) => {
                // SAFETY: made above and used by nothing; a pipeline that failed
                // is null, which destroying ignores.
                unsafe {
                    for pipeline in made {
                        device.destroy_pipeline(pipeline, None);
                    }
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_descriptor_set_layout(descriptor_layout, None);
                }
                Err(VulkanWindowRendererError::Call {
                    call: "vkCreateGraphicsPipelines",
                    result,
                })
            }
        }
    })();
    // SAFETY: the pipeline, if made, has its own copy of the module's code.
    unsafe { device.destroy_shader_module(module, None) };
    made
}

/// WGSL to SPIR-V, in-process — as the D3D11 renderer compiles its HLSL.
fn compile_shader(source: &str) -> std::result::Result<Vec<u32>, VulkanWindowRendererError> {
    let shader = |error: String| VulkanWindowRendererError::Shader(error);
    let module = naga::front::wgsl::parse_str(source).map_err(|error| shader(error.to_string()))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::IMMEDIATES,
    )
    .validate(&module)
    .map_err(|error| shader(error.to_string()))?;
    naga::back::spv::write_vec(&module, &info, &naga::back::spv::Options::default(), None)
        .map_err(|error| shader(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ffmpeg::color::{Range, Space};

    #[test]
    fn letterbox_keeps_the_aspect_ratio_and_centres_it() {
        let view = |frame: [u32; 2], width, height| {
            let view = letterbox(frame, vk::Extent2D { width, height });
            [view.x, view.y, view.width, view.height]
        };
        let near = |got: [f32; 4], expected: [f32; 4]| {
            assert!(
                got.iter()
                    .zip(expected)
                    .all(|(got, expected)| (got - expected).abs() < 0.01),
                "{got:?} is not {expected:?}"
            );
        };
        // 16:9 into a 4:3 window: bars above and below.
        near(view([1920, 1080], 800, 600), [0.0, 75.0, 800.0, 450.0]);
        // 4:3 into a 16:9 window: bars at the sides.
        near(view([640, 480], 1280, 720), [160.0, 0.0, 960.0, 720.0]);
        // The same ratio fills it.
        near(view([1920, 1080], 1280, 720), [0.0, 0.0, 1280.0, 720.0]);
    }

    /// The derived constants are the familiar ones, applied to limited-range
    /// chroma: BT.709 and BT.601 as the rest of the world writes them out.
    #[test]
    fn a_frame_is_converted_by_its_own_matrix() {
        let effective = |colour: Colour| {
            [
                colour.cr_to_r * colour.c_scale,
                colour.cb_to_g * colour.c_scale,
                colour.cr_to_g * colour.c_scale,
                colour.cb_to_b * colour.c_scale,
            ]
        };
        let near = |got: [f32; 4], expected: [f32; 4]| {
            assert!(
                got.iter()
                    .zip(expected)
                    .all(|(got, expected)| (got - expected).abs() < 0.002),
                "{got:?} is not {expected:?}"
            );
        };
        near(
            effective(Colour::of(Space::BT709, Range::MPEG, 1080)),
            [1.793, 0.213, 0.533, 2.112],
        );
        near(
            effective(Colour::of(Space::BT470BG, Range::MPEG, 1080)),
            [1.596, 0.392, 0.813, 2.017],
        );
        // Nothing said: 709 for HD, 601 below it.
        assert_eq!(
            Colour::of(Space::Unspecified, Range::MPEG, 1080),
            Colour::of(Space::BT709, Range::MPEG, 1080)
        );
        assert_eq!(
            Colour::of(Space::Unspecified, Range::MPEG, 480),
            Colour::of(Space::SMPTE170M, Range::MPEG, 480)
        );
        // Full range: nothing taken off luma, nothing scaled.
        let full = Colour::of(Space::BT709, Range::JPEG, 1080);
        assert_eq!((full.y_offset, full.y_scale, full.c_scale), (0.0, 1.0, 1.0));
    }

    #[test]
    fn the_presenting_shader_builds() {
        let spirv = compile_shader(SHADER).expect("the shader compiles");
        assert_eq!(spirv.first(), Some(&0x0723_0203), "SPIR-V's magic number");
        let module = naga::front::wgsl::parse_str(SHADER).expect("it parses");
        for layout in Layout::ALL {
            let name = layout.fragment_shader().to_str().expect("ASCII");
            assert!(
                module.entry_points.iter().any(|entry| entry.name == name),
                "no {name} in the shader"
            );
        }
    }

    /// The planes are where the copies look for them: each after the last,
    /// on a 16-byte boundary, chroma rounded up for an odd size.
    #[test]
    fn a_layout_packs_its_planes_one_after_another() {
        let extents = |layout: Layout, width, height| {
            layout
                .planes(width, height)
                .iter()
                .map(|plane| (plane.width, plane.height, plane.row_bytes(), plane.offset))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            extents(Layout::Nv12, 1280, 720),
            [(1280, 720, 1280, 0), (640, 360, 1280, 921_600)]
        );
        assert_eq!(
            extents(Layout::Yuv420p, 1280, 720),
            [
                (1280, 720, 1280, 0),
                (640, 360, 640, 921_600),
                (640, 360, 640, 1_152_000)
            ]
        );
        assert_eq!(extents(Layout::Bgra, 1280, 720), [(1280, 720, 5120, 0)]);
        // 7x5: a 35-byte luma plane, the next one aligned past it, and 4x3
        // chroma.
        assert_eq!(
            extents(Layout::Yuv420p, 7, 5),
            [(7, 5, 7, 0), (4, 3, 4, 48), (4, 3, 4, 64)]
        );
        for layout in Layout::ALL {
            assert_eq!(layout.planes(8, 8).len(), layout.plane_count());
            assert_eq!(Layout::ALL[layout.index()], layout);
        }
    }

    #[test]
    fn a_window_size_is_unset_until_set() {
        let size = WindowSize::default();
        assert_eq!(size.get(), None);
        size.clone().set(1280, 720);
        assert_eq!(
            size.get(),
            Some(vk::Extent2D {
                width: 1280,
                height: 720
            })
        );
    }
}
