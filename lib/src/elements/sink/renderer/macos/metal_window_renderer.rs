//! A Metal renderer that brings its own window, or draws into a view it is
//! given — the macOS counterpart of `D3d11WindowRenderer` and
//! `VulkanWindowRenderer`.

use std::{
    any::Any,
    sync::{Arc, Mutex},
    time::Duration,
};

use ffmpeg_next::{self as ffmpeg, format::Pixel};
use objc2::{MainThreadMarker, runtime::ProtocolObject};
use objc2_app_kit::NSView;
use objc2_core_foundation::CGSize;
use objc2_metal::{MTLDrawable, MTLPixelFormat, MTLTextureUsage};
use objc2_quartz_core::{CACurrentMediaTime, CAMetalDrawable, CAMetalLayer};
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    color::yuv_to_rgb_rows,
    contract::{
        InputContract, MediaKind, MediaKindSet, MemoryDomain, MemoryDomainSet, PixelLayout,
        PixelLayoutSet, PortContract,
    },
    element::{Element, ElementType, Sink, element_pp_log},
    elements::{
        MetalError, WindowControl, WindowEvents, WindowOptions,
        sink::renderer::presentation_delay::PresentationDelay,
    },
    error::Result,
    platform::macos::{
        metal::{Kernel, MetalGpu, Texture, write_texture},
        pixel_buffer::PixelBuffer,
        videotoolbox::{NotVideoToolbox, sw_format_of},
        window::{MetalLayer, OwnedWindow, WindowError},
    },
    pp_log::{PpLog, pp_error, pp_info},
};

/// The renderer's kernels, compiled from this one source.
const SHADER: &str = include_str!("../../../../shaders/metal/present.metal");

/// Why a [`MetalWindowRenderer`] could not be set up, or could not draw a
/// frame.
#[derive(Debug, ThisError)]
pub enum MetalWindowRendererError {
    /// [`WindowOptions`] asked for a window with no area.
    #[error("a window of {width}x{height} has nothing to draw in")]
    EmptyWindow {
        /// Width asked for.
        width: u32,
        /// Height asked for.
        height: u32,
    },
    /// The window given is not an AppKit view.
    #[error("the window given has no AppKit view to draw into")]
    NotAnAppKitWindow,
    /// The window could not be opened.
    #[error("could not open a window: {0}")]
    Window(#[from] WindowError),
    /// Metal refused something drawing needs.
    #[error(transparent)]
    Metal(#[from] MetalError),
    /// A frame of a layout this renderer does not draw: in system memory,
    /// NV12, YUV420P (or YUVJ420P) and BGRA are; as a VideoToolbox frame,
    /// NV12 and BGRA.
    #[error("MetalWindowRenderer draws NV12, YUV420P and BGRA, got {0:?}")]
    UnsupportedFormat(Pixel),
    /// A VideoToolbox frame that carries no frames context to say what it
    /// holds.
    #[error("MetalWindowRenderer was handed a VideoToolbox frame with no frames context")]
    NoFramesContext,
    /// A frame in system memory whose planes are shorter than its size
    /// says.
    #[error("a {width}x{height} {format:?} frame is missing part of its planes")]
    Truncated {
        /// The frame's layout.
        format: Pixel,
        /// Its width.
        width: u32,
        /// Its height.
        height: u32,
    },
}

/// A terminal sink that shows video frames in a window — one it opens for
/// itself, the way a GStreamer video sink does when nobody hands it one, or
/// an AppKit view the application gives it.
///
/// It takes two kinds of frame:
///
/// - A VideoToolbox frame, NV12 or BGRA — what `VideoToolboxDecoder`, a
///   camera or a screen capture, or a `MetalVideoCompositor` gives — drawn
///   from the `IOSurface` its pixel buffer is in, nothing copied.
/// - A frame in system memory, NV12, YUV420P (or YUVJ420P) or BGRA — what a
///   software decode, a CPU capture or an application's own frames give —
///   uploaded here into textures made for its layout and size, and remade
///   only when either changes.
///
/// A YUV frame is converted with its own colour description, BT.709, BT.601
/// or BT.2020 and limited or full range as it says, and where it says
/// nothing, BT.709 for a picture over 576 rows and BT.601 otherwise; a BGRA
/// frame is drawn as it is. The picture keeps its aspect ratio inside the
/// window, with black bars as needed, and follows the window's size, read
/// before every frame.
///
/// It draws; it does not pace. Put a [`crate::elements::VideoSynchronizer`]
/// or [`crate::elements::Pacer`] in front for a picture shown at its own
/// time. Each frame is presented at the display's refresh, and waited for
/// until the GPU has drawn it. What putting a picture on the screen takes is
/// told to the pipeline's playback clock, so a `VideoSynchronizer` in front
/// hands pictures over that much early: measured from the time Core
/// Animation says a picture was shown where it says one, and otherwise —
/// for every window the WindowServer composites, which is given none —
/// estimated as two refreshes of the display.
///
/// # The main thread
///
/// AppKit makes windows only on the process's main thread, and serves them
/// only from its event loop. A window of its own is made there, from
/// whichever thread opens it; a program with no event loop of its own runs
/// inside [`crate::elements::run_with_windows`]. Drawing happens on the
/// pipeline's thread.
///
/// Its GPU is its own: a pixel buffer belongs to no Metal device, so there
/// is none to share with the elements in front of it.
pub struct MetalWindowRenderer {
    name: Arc<str>,
    pp_log: PpLog,
    presenter: Presenter,
    /// `Some` for a window it opened itself.
    control: Option<WindowControl>,
}

impl MetalWindowRenderer {
    /// Opens a window of its own and draws into it.
    ///
    /// What the window reports — keys, clicks, resizing, the user closing
    /// it — comes out of the [`WindowEvents`] returned beside it; the
    /// renderer acts on none of it but resizing. Closing only hides the
    /// window, and the window is closed when the renderer is dropped.
    /// [`WindowOptions`]' size is in points, as AppKit sizes windows — twice
    /// as many pixels each way on a Retina display.
    pub fn open(
        name: impl Into<String>,
        options: WindowOptions,
    ) -> std::result::Result<(Self, WindowEvents), MetalWindowRendererError> {
        if options.width == 0 || options.height == 0 {
            return Err(MetalWindowRendererError::EmptyWindow {
                width: options.width,
                height: options.height,
            });
        }
        let (events_tx, events) = crossbeam_channel::unbounded();
        let window = OwnedWindow::open(&options, events_tx)?;
        let control = WindowControl {
            control: window.control(),
        };
        let layer = window.layer();
        let presenter = Presenter::new(layer, Keep::Owned(window))?;
        Ok((
            Self::around(name, presenter, Some(control)),
            WindowEvents { events },
        ))
    }

    /// Draws into `window`'s view, which the application owns and runs the
    /// event loop of — a `winit` window, or anything else with an AppKit
    /// handle. The view is given a Metal layer of this renderer's, on the
    /// main thread. The renderer keeps its `Arc`, so the window cannot be
    /// dropped while it is being drawn into. Keys and closing are the
    /// application's own events here; there is nothing for the renderer to
    /// report.
    pub fn for_window<W>(
        name: impl Into<String>,
        window: Arc<W>,
    ) -> std::result::Result<Self, MetalWindowRendererError>
    where
        W: HasWindowHandle + Send + Sync + 'static,
    {
        let view = match window.window_handle().map(|handle| handle.as_raw()) {
            Ok(RawWindowHandle::AppKit(handle)) => handle.ns_view.as_ptr() as usize,
            _ => return Err(MetalWindowRendererError::NotAnAppKitWindow),
        };
        let layer = attach_layer(view)?;
        let presenter = Presenter::new(Arc::new(layer), Keep::Given(window))?;
        Ok(Self::around(name, presenter, None))
    }

    /// What changes the window it opened — its title, whether it fills the
    /// screen — while it is drawn into; taken before the renderer goes into
    /// a pipeline. `None` for a window it was given, which is the
    /// application's to change. See [`WindowControl`].
    pub fn window_control(&self) -> Option<WindowControl> {
        self.control.clone()
    }

    fn around(
        name: impl Into<String>,
        presenter: Presenter,
        control: Option<WindowControl>,
    ) -> Self {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::MetalWindowRenderer, &name, None);
        pp_info!(pp_log: &pp_log, "created");
        Self {
            name,
            pp_log,
            presenter,
            control,
        }
    }
}

/// Gives the view at `view` — an `NSView` the application keeps alive — a
/// Metal layer of its own, on the main thread, and returns it.
fn attach_layer(view: usize) -> std::result::Result<MetalLayer, MetalWindowRendererError> {
    let attach = move |_mtm: MainThreadMarker| {
        // SAFETY: the application's live view, touched on the main thread.
        let view = unsafe { &*(view as *const NSView) };
        let layer = CAMetalLayer::new();
        if let Some(window) = view.window() {
            layer.setContentsScale(window.backingScaleFactor());
        }
        view.setLayer(Some(&layer));
        view.setWantsLayer(true);
        MetalLayer(layer)
    };
    if let Some(mtm) = MainThreadMarker::new() {
        return Ok(attach(mtm));
    }
    let (tx, rx) = std::sync::mpsc::channel();
    dispatch2::DispatchQueue::main().exec_async(move || {
        // SAFETY: a block on the main dispatch queue runs on the main thread.
        let _ = tx.send(attach(unsafe { MainThreadMarker::new_unchecked() }));
    });
    rx.recv_timeout(Duration::from_secs(5))
        .map_err(|_| MetalWindowRendererError::Window(WindowError::NoMainLoop))
}

impl Element for MetalWindowRenderer {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::MetalWindowRenderer
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }

    /// Takes a place in the pipeline's presentation delay, which a
    /// `VideoSynchronizer` in front hands pictures over early by.
    fn attach_context(&mut self, context: &Arc<crate::element::Context>) {
        self.presenter.delay.registration = Some(context.playback_clock.register_presenter());
    }
}

impl Sink for MetalWindowRenderer {
    /// A VideoToolbox frame, NV12 or BGRA; or a frame in system memory,
    /// NV12, YUV420P or BGRA.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::Frames(
            MediaKindSet::of(MediaKind::VideoFrame),
            MemoryDomainSet::from_slice(&[MemoryDomain::VideoToolbox, MemoryDomain::System]),
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
        self.presenter
            .draw(&frame)
            .inspect_err(|error| pp_error!(self, "draw failed: {error}"))?;
        if let Some((delay, source)) = self.presenter.delay.take_change() {
            let source = source.to_owned();
            pp_info!(
                self,
                "a picture takes {delay:.1?} to reach the screen, {source}"
            );
        }
        Ok(())
    }
}

/// What keeps the window alive for as long as the presenter draws into it.
enum Keep {
    /// Opened here, closed when this drops.
    Owned(#[allow(dead_code)] OwnedWindow),
    /// The application's, kept by its `Arc`.
    Given(#[allow(dead_code)] Arc<dyn Any + Send + Sync>),
}

/// The layouts a picture is drawn from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layout {
    Nv12,
    Yuv420p,
    Bgra,
}

impl Layout {
    /// The texture format and size of each plane of a `width` by `height`
    /// picture.
    fn planes(self, width: u32, height: u32) -> Vec<(MTLPixelFormat, u32, u32)> {
        let (half_width, half_height) = (width.div_ceil(2), height.div_ceil(2));
        match self {
            Self::Nv12 => vec![
                (MTLPixelFormat::R8Unorm, width, height),
                (MTLPixelFormat::RG8Unorm, half_width, half_height),
            ],
            Self::Yuv420p => vec![
                (MTLPixelFormat::R8Unorm, width, height),
                (MTLPixelFormat::R8Unorm, half_width, half_height),
                (MTLPixelFormat::R8Unorm, half_width, half_height),
            ],
            Self::Bgra => vec![(MTLPixelFormat::BGRA8Unorm, width, height)],
        }
    }
}

/// The textures a frame is drawn from, their layout, and their size — which
/// a pixel buffer's may exceed the picture's by.
struct Picture {
    layout: Layout,
    textures: Vec<Texture>,
    size: (u32, u32),
}

/// Textures a system-memory frame's planes are uploaded into, made for the
/// layout and size of the last one and remade when either changes.
struct SystemPlanes {
    layout: Layout,
    size: (u32, u32),
    textures: Vec<Texture>,
}

/// What one picture is drawn with — the shader's `Present`, at buffer 0.
#[derive(Clone, Copy, Default)]
struct Present {
    picture: [i32; 4],
    uv: [f32; 4],
    rows: [[f32; 4]; 3],
}

impl Present {
    fn bytes(&self) -> [u8; 80] {
        let mut bytes = [0u8; 80];
        let words = self.picture.iter().map(|value| value.to_ne_bytes()).chain(
            std::iter::once(&self.uv)
                .chain(&self.rows)
                .flatten()
                .map(|value| value.to_ne_bytes()),
        );
        for (chunk, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            *chunk = word;
        }
        bytes
    }
}

/// The layer, kernels and textures a [`MetalWindowRenderer`] draws with.
struct Presenter {
    gpu: MetalGpu,
    layer: Arc<MetalLayer>,
    nv12: Kernel,
    yuv420p: Kernel,
    bgra: Kernel,
    system: Option<SystemPlanes>,
    /// What a picture takes to reach the screen.
    delay: PresentationDelay,
    /// What the last measured picture took, once Core Animation has said —
    /// written by its presented handler, taken by the next draw.
    measured: Arc<Mutex<Option<Duration>>>,
    /// Whether anything has been published, measured or estimated.
    estimated: bool,
    /// Last, so the layer above is let go of before the window goes.
    _keep: Keep,
}

// SAFETY: Metal's device, queue, pipelines and textures, and a Metal layer,
// are thread-safe objects Apple documents as usable from any thread; the
// presenter touches them only through `&mut self`, on the one thread the
// pipeline runs its sink on.
unsafe impl Send for Presenter {}

impl Presenter {
    fn new(
        layer: Arc<MetalLayer>,
        keep: Keep,
    ) -> std::result::Result<Self, MetalWindowRendererError> {
        let gpu = MetalGpu::new()?;
        let mut kernels = gpu
            .kernels(SHADER, &["present_nv12", "present_yuv420p", "present_bgra"])?
            .into_iter();
        let mut next = || kernels.next().expect("one kernel for each name");
        let (nv12, yuv420p, bgra) = (next(), next(), next());
        let metal = &layer.0;
        metal.setDevice(Some(&gpu.device));
        metal.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        // Written by a compute kernel, which a drawable only allows when
        // it is not for rendering alone.
        metal.setFramebufferOnly(false);
        // What video's R'G'B' is — BT.709's primaries are sRGB's — so a wide
        // gamut display shows it as meant rather than stretched to its own,
        // as a swap chain's sRGB does on Windows.
        // SAFETY: a constant Core Graphics exports, read once it is loaded,
        // which linking it guarantees.
        let name = unsafe { objc2_core_graphics::kCGColorSpaceSRGB };
        let srgb = objc2_core_graphics::CGColorSpace::with_name(Some(name));
        metal.setColorspace(srgb.as_deref());
        Ok(Self {
            gpu,
            layer,
            nv12,
            yuv420p,
            bgra,
            system: None,
            delay: PresentationDelay::default(),
            measured: Arc::default(),
            estimated: false,
            _keep: keep,
        })
    }

    /// Draws `frame` into the next drawable, letterboxed, and presents it.
    fn draw(
        &mut self,
        frame: &ffmpeg::frame::Video,
    ) -> std::result::Result<(), MetalWindowRendererError> {
        if let Some(measured) = self.measured.lock().ok().and_then(|mut slot| slot.take()) {
            self.delay.record(measured);
            self.estimated = true;
        }
        // Until a picture's own presented time says otherwise — which a
        // composited window never does — two refreshes of the display: one
        // for the WindowServer to take the picture, one to show it.
        if !self.estimated && self.delay.registration.is_some() {
            let interval = refresh_interval();
            self.delay.estimate(interval * 2, interval);
            self.estimated = true;
        }
        let Picture {
            layout,
            textures,
            size: texture_size,
        } = self.pictures(frame)?;

        // The drawable follows the layer's size in pixels; a window with no
        // area — minimised — is not drawn into.
        let layer = &self.layer.0;
        let bounds = layer.bounds().size;
        let scale = layer.contentsScale();
        let (width, height) = (
            (bounds.width * scale).round() as u32,
            (bounds.height * scale).round() as u32,
        );
        if width == 0 || height == 0 {
            return Ok(());
        }
        let size = CGSize::new(f64::from(width), f64::from(height));
        if layer.drawableSize() != size {
            layer.setDrawableSize(size);
        }
        // Waits for one of the layer's drawables to be free, which is where
        // presenting at the display's refresh holds a fast producer back.
        let Some(drawable) = layer.nextDrawable() else {
            return Ok(());
        };
        let target = drawable.texture();

        let picture = letterbox(frame.width(), frame.height(), width, height);
        let rows = match layout {
            Layout::Bgra => [[0.0; 4]; 3],
            Layout::Nv12 | Layout::Yuv420p => {
                // FFmpeg's name for full-range 4:2:0, which not every
                // producer also says in the range field.
                let range = match frame.format() {
                    Pixel::YUVJ420P => ffmpeg::color::Range::JPEG,
                    _ => frame.color_range(),
                };
                yuv_to_rgb_rows(frame.color_space(), range, frame.height())
            }
        };
        let present = Present {
            picture,
            uv: [
                frame.width() as f32 / texture_size.0 as f32,
                frame.height() as f32 / texture_size.1 as f32,
                0.0,
                0.0,
            ],
            rows,
        };
        let kernel = match layout {
            Layout::Nv12 => &self.nv12,
            Layout::Yuv420p => &self.yuv420p,
            Layout::Bgra => &self.bgra,
        };
        let bound: Vec<&Texture> = std::iter::once(&target).chain(textures.iter()).collect();
        let mut pass = self.gpu.pass()?;
        pass.dispatch(kernel, &bound, Some(&present.bytes()), (width, height));
        let drawable: &ProtocolObject<dyn MTLDrawable> = ProtocolObject::from_ref(&*drawable);
        if self.delay.due() {
            self.measure(drawable);
        }
        pass.present(drawable);
        pass.finish()?;
        Ok(())
    }

    /// Asks Core Animation to say when `drawable` reaches the screen, and
    /// keeps how long that took from now for the next draw.
    fn measure(&self, drawable: &ProtocolObject<dyn MTLDrawable>) {
        let handed = CACurrentMediaTime();
        let measured = Arc::clone(&self.measured);
        let handler = block2::RcBlock::new(
            move |drawable: std::ptr::NonNull<ProtocolObject<dyn MTLDrawable>>| {
                // SAFETY: Metal hands over the drawable the handler was
                // added to, live for the call.
                let presented = unsafe { drawable.as_ref() }.presentedTime();
                // Zero where the drawable was not shown, and for every
                // drawable of a window the WindowServer composites — which
                // is then estimated instead; the handler itself is called as
                // the picture is handed to the WindowServer, not as it is
                // shown, so its own time says nothing.
                if presented > handed
                    && let Ok(mut slot) = measured.lock()
                {
                    *slot = Some(Duration::from_secs_f64(presented - handed));
                }
            },
        );
        // SAFETY: the block is copied by Metal and called once, on a thread
        // of its own, touching only what it owns.
        unsafe { drawable.addPresentedHandler(block2::RcBlock::as_ptr(&handler)) };
    }

    /// What `frame` is drawn from.
    fn pictures(
        &mut self,
        frame: &ffmpeg::frame::Video,
    ) -> std::result::Result<Picture, MetalWindowRendererError> {
        let read = MTLTextureUsage::ShaderRead;
        if frame.format() == Pixel::VIDEOTOOLBOX {
            let layout = match sw_format_of(frame) {
                Ok(Pixel::NV12) => Layout::Nv12,
                Ok(Pixel::BGRA) => Layout::Bgra,
                Ok(other) => return Err(MetalWindowRendererError::UnsupportedFormat(other)),
                Err(NotVideoToolbox::Format(format)) => {
                    return Err(MetalWindowRendererError::UnsupportedFormat(format));
                }
                Err(NotVideoToolbox::NoFramesContext) => {
                    return Err(MetalWindowRendererError::NoFramesContext);
                }
            };
            let buffer = PixelBuffer::of_frame(frame)
                .ok_or(MetalWindowRendererError::UnsupportedFormat(frame.format()))?;
            let size = buffer.size();
            let textures = match layout {
                Layout::Bgra => {
                    vec![
                        self.gpu
                            .plane(&buffer, 0, MTLPixelFormat::BGRA8Unorm, read)?,
                    ]
                }
                _ => vec![
                    self.gpu.plane(&buffer, 0, MTLPixelFormat::R8Unorm, read)?,
                    self.gpu.plane(&buffer, 1, MTLPixelFormat::RG8Unorm, read)?,
                ],
            };
            return Ok(Picture {
                layout,
                textures,
                size,
            });
        }

        let layout = match frame.format() {
            Pixel::NV12 => Layout::Nv12,
            Pixel::YUV420P | Pixel::YUVJ420P => Layout::Yuv420p,
            Pixel::BGRA => Layout::Bgra,
            other => return Err(MetalWindowRendererError::UnsupportedFormat(other)),
        };
        let size = (frame.width(), frame.height());
        let planes = layout.planes(size.0, size.1);
        if !self
            .system
            .as_ref()
            .is_some_and(|system| system.layout == layout && system.size == size)
        {
            let textures = planes
                .iter()
                .map(|&(format, width, height)| self.gpu.texture(format, width, height, read, true))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            self.system = Some(SystemPlanes {
                layout,
                size,
                textures,
            });
        }
        let system = self.system.as_ref().expect("made above");
        for (index, (&(format, width, height), texture)) in
            planes.iter().zip(&system.textures).enumerate()
        {
            let texel = match format {
                MTLPixelFormat::RG8Unorm => 2,
                MTLPixelFormat::BGRA8Unorm => 4,
                _ => 1,
            };
            let (data, stride) = (frame.data(index), frame.stride(index));
            let row = width as usize * texel;
            let needed = (height as usize).saturating_sub(1) * stride + row;
            if stride < row || data.len() < needed {
                return Err(MetalWindowRendererError::Truncated {
                    format: frame.format(),
                    width: size.0,
                    height: size.1,
                });
            }
            write_texture(texture, data, stride, width, height);
        }
        Ok(Picture {
            layout,
            textures: system.textures.clone(),
            size,
        })
    }
}

/// How often the main display refreshes: its mode's rate, or 60 Hz for one
/// that states none — a variable-rate (ProMotion) display, whose rate is at
/// most that of the content.
fn refresh_interval() -> Duration {
    use objc2_core_graphics::{CGDisplayCopyDisplayMode, CGDisplayMode, CGMainDisplayID};
    let hertz = CGDisplayCopyDisplayMode(CGMainDisplayID())
        .map(|mode| CGDisplayMode::refresh_rate(Some(&mode)))
        .filter(|&hertz| hertz > 0.0)
        .unwrap_or(60.0);
    Duration::from_secs_f64(1.0 / hertz)
}

/// The largest rectangle of the frame's aspect ratio that fits the
/// drawable, centred in it, as `[x, y, width, height]` in whole pixels.
fn letterbox(frame_width: u32, frame_height: u32, width: u32, height: u32) -> [i32; 4] {
    let scale =
        (width as f32 / frame_width.max(1) as f32).min(height as f32 / frame_height.max(1) as f32);
    let (picture_width, picture_height) = (
        (frame_width as f32 * scale).round().max(1.0),
        (frame_height as f32 * scale).round().max(1.0),
    );
    [
        ((width as f32 - picture_width) * 0.5).round() as i32,
        ((height as f32 - picture_height) * 0.5).round() as i32,
        picture_width as i32,
        picture_height as i32,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letterbox_keeps_the_aspect_ratio_and_centres_it() {
        // A wide picture in a square: full width, bars above and below.
        assert_eq!(letterbox(1920, 1080, 1000, 1000), [0, 219, 1000, 563]);
        // A tall picture in a wide drawable: full height, bars at the sides.
        assert_eq!(letterbox(720, 1280, 1000, 500), [360, 0, 281, 500]);
        // The same shape fills it.
        assert_eq!(letterbox(640, 360, 1280, 720), [0, 0, 1280, 720]);
    }

    /// The shader's `Present` is laid out as the bytes say: the rectangle,
    /// then the scale, then the three rows.
    #[test]
    fn present_is_laid_out_as_the_shader_reads_it() {
        let bytes = Present {
            picture: [1, 2, 3, 4],
            uv: [0.5, 0.25, 0.0, 0.0],
            rows: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
        }
        .bytes();
        assert_eq!(&bytes[..4], &1i32.to_ne_bytes());
        assert_eq!(&bytes[12..16], &4i32.to_ne_bytes());
        assert_eq!(&bytes[16..20], &0.5f32.to_ne_bytes());
        assert_eq!(&bytes[32..36], &1.0f32.to_ne_bytes());
        assert_eq!(&bytes[72..76], &1.0f32.to_ne_bytes());
    }
}
