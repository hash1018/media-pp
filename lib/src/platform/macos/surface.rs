//! What an element of your own needs to work on this crate's VideoToolbox
//! pictures with Metal of its own, the pictures never leaving the GPU: the
//! textures a picture is, made over its pixel buffer. Pictures to write
//! into are [`super::videotoolbox::VideoToolboxFramePool`]'s.

use std::marker::PhantomData;

use ffmpeg_next::{self as ffmpeg, format::Pixel};
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice, MTLPixelFormat, MTLTextureUsage};
use thiserror::Error as ThisError;

use super::{
    metal::{MetalError, plane_on},
    pixel_buffer::PixelBuffer,
    videotoolbox::{NotVideoToolbox, VideoToolboxFrameFormat, sw_format_of},
};
use crate::{color::ColorDescription, elements::MetalFramePlanes};

/// Why a frame has no [`MetalSurfaceView`].
#[derive(Debug, ThisError)]
pub enum MetalSurfaceError {
    /// The frame is not a VideoToolbox one: what it is instead.
    #[error("not a VideoToolbox frame: {0:?}")]
    NotVideoToolbox(Pixel),

    /// A VideoToolbox frame that carries no frames context to say what it
    /// holds.
    #[error("a VideoToolbox frame with no frames context")]
    NoFramesContext,

    /// A VideoToolbox frame holding a layout other than NV12 or BGRA.
    #[error("a VideoToolbox frame holding {0:?}, not NV12 or BGRA")]
    UnsupportedLayout(Pixel),

    /// A VideoToolbox frame with no pixel buffer behind it.
    #[error("a VideoToolbox frame with no pixel buffer behind it")]
    NoPixelBuffer,

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

/// A VideoToolbox picture as Metal textures, made over the `IOSurface` its
/// pixel buffer is in — nothing is copied.
///
/// What a custom element reads a picture by — a model of its own, a
/// measurement — or, for a picture from its own
/// [`VideoToolboxFramePool`](crate::elements::VideoToolboxFramePool),
/// writes one by. Made by [`MetalSurfaceView::new`], which asks of the frame
/// what the Metal elements of this crate ask: that it is a VideoToolbox
/// frame holding NV12 or BGRA, on a pixel buffer at least its size. The
/// textures cover the whole pixel buffer, which may be larger than the
/// picture; [`Self::width`] and [`Self::height`] are the picture, from the
/// top-left corner.
///
/// # The contract with the rest of the pipeline
///
/// - **Any device.** A pixel buffer belongs to no Metal device, so the
///   textures are made on whichever one you draw with —
///   `MTLCreateSystemDefaultDevice` is the one this crate's elements use.
/// - **A picture handed to you is finished.** Every element of this crate
///   that writes a VideoToolbox picture waits for the GPU to finish before
///   handing it on, so a command buffer of yours reads what it wrote.
/// - **Finish before you hand on.** Do the same: wait for your command
///   buffer to complete (`waitUntilCompleted`, or push the picture from its
///   completed handler) before pushing a picture you wrote, since the
///   element after yours reads it on a queue of its own.
/// - **Hold the frame while the GPU reads it.** The view borrows the frame,
///   so its pixel buffer stays the picture while the view is held. A texture
///   cloned out of it stays valid Metal after that, but once the frame is
///   dropped its pixel buffer goes back to its pool, and a later picture
///   may be written into it.
/// - **Never write a picture you were handed.** It may be read on another
///   branch of a `Tee`, or be the one a source offers again. Write into a
///   frame from a `VideoToolboxFramePool` instead, copying what you keep.
pub struct MetalSurfaceView<'a> {
    format: VideoToolboxFrameFormat,
    width: u32,
    height: u32,
    color: ColorDescription,
    planes: MetalFramePlanes,
    frame: PhantomData<&'a ffmpeg::frame::Video>,
}

impl std::fmt::Debug for MetalSurfaceView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalSurfaceView")
            .field("format", &self.format)
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

impl<'a> MetalSurfaceView<'a> {
    /// The textures of `frame`, made on `device` to be read, written and
    /// rendered to.
    ///
    /// # Errors
    ///
    /// [`MetalSurfaceError`] where `frame` is not a VideoToolbox frame
    /// holding NV12 or BGRA on a pixel buffer at least its size, or Metal
    /// will not make a texture over it.
    pub fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        frame: &'a ffmpeg::frame::Video,
    ) -> Result<Self, MetalSurfaceError> {
        let usage = MTLTextureUsage::ShaderRead
            | MTLTextureUsage::ShaderWrite
            | MTLTextureUsage::RenderTarget;
        let (format, planes, color) = textures(device, frame, usage)?;
        Ok(Self {
            format,
            width: frame.width(),
            height: frame.height(),
            color,
            planes,
            frame: PhantomData,
        })
    }

    /// What the picture holds: NV12 or BGRA.
    pub fn format(&self) -> VideoToolboxFrameFormat {
        self.format
    }

    /// The picture's width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// The picture's height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// What the frame says of its colour, as [`crate::elements::MetalFrame::color`]
    /// gives it: a BGRA frame's is RGB.
    pub fn color(&self) -> ColorDescription {
        self.color
    }

    /// The textures: a `bgra8Unorm` one, or NV12's `r8Unorm` luma and its
    /// `rg8Unorm` chroma at half the width and height.
    pub fn planes(&self) -> &MetalFramePlanes {
        &self.planes
    }
}

/// `frame`'s textures on `device`, with `usage`, after asking of it what
/// every Metal element of this crate asks.
pub(crate) fn textures(
    device: &ProtocolObject<dyn MTLDevice>,
    frame: &ffmpeg::frame::Video,
    usage: MTLTextureUsage,
) -> Result<(VideoToolboxFrameFormat, MetalFramePlanes, ColorDescription), MetalSurfaceError> {
    let format = match sw_format_of(frame) {
        Ok(layout) => VideoToolboxFrameFormat::of(layout)
            .ok_or(MetalSurfaceError::UnsupportedLayout(layout))?,
        Err(NotVideoToolbox::Format(other)) => {
            return Err(MetalSurfaceError::NotVideoToolbox(other));
        }
        Err(NotVideoToolbox::NoFramesContext) => return Err(MetalSurfaceError::NoFramesContext),
    };
    let buffer = PixelBuffer::of_frame(frame).ok_or(MetalSurfaceError::NoPixelBuffer)?;
    let (width, height) = (frame.width(), frame.height());
    let (buffer_width, buffer_height) = buffer.size();
    if buffer_width < width || buffer_height < height {
        return Err(MetalSurfaceError::SmallerBuffer {
            width,
            height,
            buffer_width,
            buffer_height,
        });
    }
    Ok(match format {
        VideoToolboxFrameFormat::Bgra => (
            format,
            MetalFramePlanes::Bgra(plane_on(
                device,
                &buffer,
                0,
                MTLPixelFormat::BGRA8Unorm,
                usage,
            )?),
            ColorDescription {
                space: ffmpeg::color::Space::RGB,
                ..ColorDescription::of(frame)
            },
        ),
        VideoToolboxFrameFormat::Nv12 => (
            format,
            MetalFramePlanes::Nv12 {
                luma: plane_on(device, &buffer, 0, MTLPixelFormat::R8Unorm, usage)?,
                chroma: plane_on(device, &buffer, 1, MTLPixelFormat::RG8Unorm, usage)?,
            },
            ColorDescription::of(frame),
        ),
    })
}

#[cfg(test)]
mod tests {
    use objc2_metal::{MTLCreateSystemDefaultDevice, MTLTexture};

    use super::*;
    use crate::{elements::VideoToolboxFramePool, test_support::try_videotoolbox_device};

    /// A pool's picture is seen as the textures its format has, each the
    /// size of its plane.
    #[test]
    fn a_pooled_picture_is_seen_as_its_textures() {
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let Some(metal) = MTLCreateSystemDefaultDevice() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        for format in [VideoToolboxFrameFormat::Nv12, VideoToolboxFrameFormat::Bgra] {
            let pool = VideoToolboxFramePool::new(&device, format, 64, 36).expect("pool");
            let frame = pool.get().expect("frame");
            let view = MetalSurfaceView::new(&metal, &frame).expect("view");
            assert_eq!(view.format(), format);
            assert_eq!((view.width(), view.height()), (64, 36));
            match view.planes() {
                MetalFramePlanes::Nv12 { luma, chroma } => {
                    assert_eq!(format, VideoToolboxFrameFormat::Nv12);
                    assert!(luma.width() >= 64 && luma.height() >= 36);
                    assert_eq!(luma.pixelFormat(), MTLPixelFormat::R8Unorm);
                    assert!(chroma.width() >= 32 && chroma.height() >= 18);
                    assert_eq!(chroma.pixelFormat(), MTLPixelFormat::RG8Unorm);
                }
                MetalFramePlanes::Bgra(texture) => {
                    assert_eq!(format, VideoToolboxFrameFormat::Bgra);
                    assert!(texture.width() >= 64 && texture.height() >= 36);
                    assert_eq!(texture.pixelFormat(), MTLPixelFormat::BGRA8Unorm);
                }
            }
        }
    }

    /// A system-memory picture is refused before anything of it is read as
    /// a pixel buffer.
    #[test]
    fn a_system_memory_picture_is_refused() {
        let Some(metal) = MTLCreateSystemDefaultDevice() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let frame = ffmpeg::frame::Video::new(Pixel::NV12, 64, 36);
        assert!(matches!(
            MetalSurfaceView::new(&metal, &frame),
            Err(MetalSurfaceError::NotVideoToolbox(Pixel::NV12))
        ));
    }
}
