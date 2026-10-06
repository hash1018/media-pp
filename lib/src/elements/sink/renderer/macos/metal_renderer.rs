use std::sync::Arc;

use ffmpeg_next::format::Pixel;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLDevice, MTLTexture, MTLTextureUsage};
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    color::ColorDescription,
    contract::{InputContract, MediaKind, MemoryDomain, PixelLayoutSet, PortContract},
    element::{Element, ElementType, Sink, element_pp_log},
    elements::SubmitError,
    error::Result,
    platform::macos::{
        metal::MetalError,
        surface::{MetalSurfaceError, textures},
    },
    pool::UnboundObjectPoolRef,
    pp_log::{PpLog, pp_error, pp_info},
    render::{SinkStage, sink_stage},
};

/// What [`MetalRenderer`] needs from an application's own Metal drawing —
/// the Metal sibling of `D3d11FrameRenderer` and `CudaFrameRenderer`.
///
/// It is handed each picture as Metal textures on its own device, made over
/// the `IOSurface` the VideoToolbox frame's pixel buffer is in: nothing is
/// copied, and the implementation samples them in whatever it draws — a
/// UI's own `CAMetalLayer`, an offscreen target, a texture another engine
/// composites.
///
/// A successful submit must install the frame as the current presentation
/// content or enqueue its presentation before returning. Pipeline preroll
/// treats that return as the terminal's presentation commitment; it does not
/// require the implementation to wait for physical scanout.
pub trait MetalFrameRenderer: Send {
    /// The device this implementation draws with. [`MetalRenderer`] reads
    /// it once, when it is made, and makes every frame's textures on it — a
    /// texture belongs to the device it was made on, even where the pixels
    /// under it belong to none.
    fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>>;

    /// Draws `frame`, or queues its drawing. See [`MetalFrame`] on how
    /// long its textures stay the picture.
    fn submit(&self, frame: MetalFrame) -> std::result::Result<(), SubmitError>;

    /// Updates the presentation target dimensions.
    fn resize(&self, width: u32, height: u32) -> std::result::Result<(), SubmitError>;
}

/// A Metal texture, as the implementation's device made it.
pub type MetalTexture = Retained<ProtocolObject<dyn MTLTexture>>;

/// The textures a [`MetalFrame`] is drawn from.
pub enum MetalFramePlanes {
    /// Packed BGRA, as a `bgra8Unorm` texture.
    Bgra(MetalTexture),
    /// NV12: its luma as an `r8Unorm` texture of the picture's size, and
    /// its chroma, Cb in red and Cr in green, as an `rg8Unorm` texture of
    /// half its width and height.
    Nv12 {
        /// The luma plane.
        luma: MetalTexture,
        /// The interleaved chroma plane.
        chroma: MetalTexture,
    },
}

/// One picture handed to a [`MetalFrameRenderer`].
///
/// It holds the frame it was made from, so while it is alive the pixel
/// buffer under its textures stays the frame's and is not handed out again
/// by the producer's pool. Keep it until the GPU has finished reading them
/// — a command buffer's completed handler is where to let go of it; drop
/// it sooner and a later picture can be written into what the GPU is still
/// reading.
pub struct MetalFrame {
    planes: MetalFramePlanes,
    width: u32,
    height: u32,
    color: ColorDescription,
    _frame: Arc<UnboundObjectPoolRef<ffmpeg_next::frame::Video>>,
}

// SAFETY: Metal textures are thread-safe objects, which Apple documents as
// usable from any thread, and the frame is held only to keep its buffer
// checked out.
unsafe impl Send for MetalFrame {}

impl MetalFrame {
    /// The textures to draw from.
    pub fn planes(&self) -> &MetalFramePlanes {
        &self.planes
    }

    /// The picture's width in pixels, from its top-left corner. The
    /// textures are the whole pixel buffer, which is at least this.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// The picture's height in pixels, from its top-left corner. The
    /// textures are the whole pixel buffer, which is at least this.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// What an NV12 frame says of its Y'CbCr, each part `Unspecified` where
    /// it says nothing; its
    /// [`yuv_to_rgb_rows`](crate::color::ColorDescription::yuv_to_rgb_rows)
    /// are the rows to convert it with, filling in what it leaves unsaid as
    /// this crate's own renderers do. A BGRA frame's is RGB.
    pub fn color(&self) -> ColorDescription {
        self.color
    }
}

/// Errors specific to `MetalRenderer`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum MetalRendererError {
    /// The caller-provided renderer rejected frame submission.
    #[error("failed to submit frame: {0:?}")]
    Submit(SubmitError),
    /// The caller-provided renderer rejected a size change.
    #[error("failed to resize: {0:?}")]
    Resize(SubmitError),
    /// A frame that is not a VideoToolbox one, or one holding a layout
    /// other than NV12 or BGRA.
    #[error("MetalRenderer draws VideoToolbox frames holding NV12 or BGRA, got {0:?}")]
    UnsupportedFormat(Pixel),
    /// A VideoToolbox frame that carries no frames context to say what it
    /// holds.
    #[error("MetalRenderer was handed a VideoToolbox frame with no frames context")]
    NoFramesContext,
    /// A frame whose pixel buffer is smaller than the frame says it is.
    #[error("a {width}x{height} frame is on a {buffer_width}x{buffer_height} pixel buffer")]
    SmallerBuffer {
        /// The frame's width.
        width: u32,
        /// The frame's height.
        height: u32,
        /// Its pixel buffer's width.
        buffer_width: u32,
        /// Its pixel buffer's height.
        buffer_height: u32,
    },
    /// Metal would not make a texture over the frame's surface.
    #[error(transparent)]
    Metal(#[from] MetalError),
}

/// Terminal sink that hands VideoToolbox frames to a caller-supplied
/// [`MetalFrameRenderer`] as textures on its device — the Metal sibling of
/// `D3d11Renderer` and `CudaRenderer`, for drawing into something that is
/// not a window of this crate's: a UI's own layer, an offscreen target.
/// [`crate::elements::MetalWindowRenderer`] is the one that draws into a
/// window itself.
///
/// It takes what the macOS stack makes — `VideoToolboxDecoder`, a camera or
/// a screen capture, `VideoToolboxUpload`, a `MetalVideoCompositor` — NV12
/// or BGRA, and makes each picture's textures over the `IOSurface` its
/// pixel buffer is in, on the implementation's device: zero-copy, and with
/// no device to share, since a pixel buffer belongs to none.
///
/// It draws nothing and does not pace. Put a
/// [`crate::elements::VideoSynchronizer`] or [`crate::elements::Pacer`] in
/// front for a picture shown at its own time.
pub struct MetalRenderer(SinkStage<Submitting>);

sink_stage!(MetalRenderer);

/// What a [`MetalRenderer`] does with each frame: makes its textures and
/// hands them over.
struct Submitting {
    pp_log: PpLog,
    name: Arc<str>,
    inner: Box<dyn MetalFrameRenderer>,
    /// Read once from `inner`, where every texture is made.
    device: Retained<ProtocolObject<dyn MTLDevice>>,
}

// SAFETY: `device` is a Metal device, which Apple documents as thread-safe,
// and `inner` is `Send` by its own bound.
unsafe impl Send for Submitting {}

impl MetalRenderer {
    /// `renderer` is whatever the caller's own [`MetalFrameRenderer`]
    /// implementation is — already set up to draw by the time it gets
    /// here.
    pub fn new(name: impl Into<String>, renderer: Box<dyn MetalFrameRenderer>) -> Self {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::MetalRenderer, &name, None);
        let device = renderer.device();
        pp_info!(pp_log: &pp_log, "created");
        Self(SinkStage::new(Submitting {
            pp_log,
            name,
            inner: renderer,
            device,
        }))
    }

    /// Call when the target resizes.
    pub fn resize(&self, width: u32, height: u32) -> Result<()> {
        let submitting = &self.0.inner;
        submitting
            .inner
            .resize(width, height)
            .inspect_err(|error| pp_error!(submitting, "resize failed: {error:?}"))
            .map_err(MetalRendererError::Resize)?;
        pp_info!(submitting, "resized: {width}x{height}");
        Ok(())
    }
}

impl Submitting {
    /// `frame`'s textures on the implementation's device.
    fn picture(
        &self,
        frame: &Arc<UnboundObjectPoolRef<ffmpeg_next::frame::Video>>,
    ) -> std::result::Result<MetalFrame, MetalRendererError> {
        let (_, planes, color) = textures(&self.device, frame, MTLTextureUsage::ShaderRead)
            .map_err(|error| match error {
                MetalSurfaceError::NotVideoToolbox(format)
                | MetalSurfaceError::UnsupportedLayout(format) => {
                    MetalRendererError::UnsupportedFormat(format)
                }
                MetalSurfaceError::NoPixelBuffer => {
                    MetalRendererError::UnsupportedFormat(Pixel::VIDEOTOOLBOX)
                }
                MetalSurfaceError::NoFramesContext => MetalRendererError::NoFramesContext,
                MetalSurfaceError::SmallerBuffer {
                    width,
                    height,
                    buffer_width,
                    buffer_height,
                } => MetalRendererError::SmallerBuffer {
                    width,
                    height,
                    buffer_width,
                    buffer_height,
                },
                MetalSurfaceError::Metal(error) => MetalRendererError::Metal(error),
            })?;
        let (width, height) = (frame.width(), frame.height());
        Ok(MetalFrame {
            planes,
            width,
            height,
            color,
            _frame: frame.clone(),
        })
    }
}

impl Element for Submitting {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::MetalRenderer
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Sink for Submitting {
    /// VideoToolbox frames holding NV12 or BGRA — anything else has no
    /// surface to make a texture over.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::VideoToolbox)
                .with_layouts(PixelLayoutSet::NV12_OR_BGRA),
        )
    }

    fn render(&mut self, buf: MediaBuffer) -> Result<()> {
        let MediaBuffer::Video(frame) = buf else {
            return Ok(());
        };
        let picture = self
            .picture(&frame)
            .inspect_err(|error| pp_error!(self, "frame refused: {error}"))?;
        self.inner
            .submit(picture)
            .inspect_err(|error| pp_error!(self, "submit failed: {error:?}"))
            .map_err(MetalRendererError::Submit)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use objc2_metal::{MTLCreateSystemDefaultDevice, MTLPixelFormat};

    use super::*;
    use crate::{
        element::RawSink,
        elements::VideoToolboxUpload,
        test_support::{capture, try_videotoolbox_device},
    };

    /// Keeps what reached the application side, textures and all.
    struct Recording {
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        frames: Arc<Mutex<Vec<MetalFrame>>>,
        sizes: Arc<Mutex<Vec<(u32, u32)>>>,
    }

    impl MetalFrameRenderer for Recording {
        fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>> {
            self.device.clone()
        }

        fn submit(&self, frame: MetalFrame) -> std::result::Result<(), SubmitError> {
            self.frames.lock().unwrap().push(frame);
            Ok(())
        }

        fn resize(&self, width: u32, height: u32) -> std::result::Result<(), SubmitError> {
            self.sizes.lock().unwrap().push((width, height));
            Ok(())
        }
    }

    type Recorded = (
        MetalRenderer,
        Arc<Mutex<Vec<MetalFrame>>>,
        Arc<Mutex<Vec<(u32, u32)>>>,
    );

    fn recording() -> Option<Recorded> {
        let Some(device) = MTLCreateSystemDefaultDevice() else {
            eprintln!("skipping: no Metal device");
            return None;
        };
        let frames = Arc::default();
        let sizes = Arc::default();
        let renderer = MetalRenderer::new(
            "metal-renderer",
            Box::new(Recording {
                device,
                frames: Arc::clone(&frames),
                sizes: Arc::clone(&sizes),
            }),
        );
        Some((renderer, frames, sizes))
    }

    fn uploaded(
        device: &crate::elements::VideoToolboxDevice,
        frame: ffmpeg_next::frame::Video,
    ) -> MediaBuffer {
        let mut upload = VideoToolboxUpload::new("upload", device);
        let out = capture(&mut upload);
        upload.consume(MediaBuffer::video(frame)).unwrap();
        out.lock().unwrap().remove(0)
    }

    /// An NV12 frame reaches the application as its two planes on the
    /// application's device, at their sizes, with the frame's colour — and
    /// the textures hold the frame's own samples.
    #[test]
    fn an_nv12_frame_arrives_as_its_planes_with_its_colour() {
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let Some((mut renderer, frames, _)) = recording() else {
            return;
        };
        let mut picture = ffmpeg_next::frame::Video::new(Pixel::NV12, 8, 4);
        ColorDescription::BT709_LIMITED.describe(&mut picture);
        let stride = picture.stride(0);
        for row in 0..4 {
            picture.data_mut(0)[row * stride..][..8].fill(100);
        }
        let stride = picture.stride(1);
        for row in 0..2 {
            for pair in picture.data_mut(1)[row * stride..][..8].chunks_mut(2) {
                pair.copy_from_slice(&[60, 200]);
            }
        }
        renderer.consume(uploaded(&device, picture)).unwrap();

        let frames = frames.lock().unwrap();
        let [frame] = &frames[..] else {
            panic!("one frame submitted, got {}", frames.len());
        };
        assert_eq!((frame.width(), frame.height()), (8, 4));
        assert_eq!(frame.color(), ColorDescription::BT709_LIMITED);
        let MetalFramePlanes::Nv12 { luma, chroma } = frame.planes() else {
            panic!("NV12 planes");
        };
        assert_eq!((luma.width(), luma.height()), (8, 4));
        assert_eq!(luma.pixelFormat(), MTLPixelFormat::R8Unorm);
        assert_eq!((chroma.width(), chroma.height()), (4, 2));
        assert_eq!(chroma.pixelFormat(), MTLPixelFormat::RG8Unorm);
        assert_eq!(read_texel(luma, 1), [100]);
        assert_eq!(read_texel(chroma, 2), [60, 200]);
    }

    /// A BGRA frame reaches it as one texture, said to be RGB.
    #[test]
    fn a_bgra_frame_arrives_as_one_texture() {
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let Some((mut renderer, frames, _)) = recording() else {
            return;
        };
        let mut picture = ffmpeg_next::frame::Video::new(Pixel::BGRA, 4, 4);
        let stride = picture.stride(0);
        for row in 0..4 {
            for pixel in picture.data_mut(0)[row * stride..][..16].chunks_mut(4) {
                pixel.copy_from_slice(&[10, 20, 30, 255]);
            }
        }
        renderer.consume(uploaded(&device, picture)).unwrap();

        let frames = frames.lock().unwrap();
        let MetalFramePlanes::Bgra(texture) = frames[0].planes() else {
            panic!("one BGRA texture");
        };
        assert_eq!(texture.pixelFormat(), MTLPixelFormat::BGRA8Unorm);
        assert_eq!(frames[0].color().space, ffmpeg_next::color::Space::RGB);
        assert_eq!(read_texel(texture, 4), [10, 20, 30, 255]);
    }

    /// A frame in system memory has no surface to make a texture over, so
    /// it is refused as a typed error, and nothing reaches the application.
    #[test]
    fn a_frame_in_system_memory_is_refused() {
        let Some((mut renderer, frames, _)) = recording() else {
            return;
        };
        let frame = ffmpeg_next::frame::Video::new(Pixel::NV12, 8, 8);
        let error = renderer
            .consume(MediaBuffer::video(frame))
            .expect_err("a frame in system memory must not be submitted");
        assert!(
            matches!(
                error,
                crate::error::Error::MetalRendererError(MetalRendererError::UnsupportedFormat(
                    Pixel::NV12
                ))
            ),
            "expected UnsupportedFormat, got {error:?}"
        );
        assert!(frames.lock().unwrap().is_empty());
    }

    /// A resize reaches the application.
    #[test]
    fn resize_reaches_the_application() {
        let Some((renderer, _, sizes)) = recording() else {
            return;
        };
        renderer.resize(640, 360).unwrap();
        assert_eq!(*sizes.lock().unwrap(), [(640, 360)]);
    }

    /// The first texel of `texture`, `bytes` of it, read back through a
    /// shared texture the GPU copies it into.
    fn read_texel(texture: &MetalTexture, bytes: usize) -> Vec<u8> {
        let gpu = crate::platform::macos::metal::MetalGpu::new().unwrap();
        let copy = gpu
            .texture(
                texture.pixelFormat(),
                texture.width() as u32,
                texture.height() as u32,
                MTLTextureUsage::ShaderRead,
                true,
            )
            .unwrap();
        gpu.copy(texture, &copy).unwrap();
        let mut out = vec![0u8; bytes];
        // SAFETY: `copy` is a shared texture the CPU reads, finished with by
        // the GPU; `out` is one texel of its format, for a region of one.
        unsafe {
            copy.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                std::ptr::NonNull::new(out.as_mut_ptr()).unwrap().cast(),
                bytes * copy.width(),
                objc2_metal::MTLRegion {
                    origin: objc2_metal::MTLOrigin { x: 0, y: 0, z: 0 },
                    size: objc2_metal::MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                },
                0,
            )
        };
        out
    }
}
