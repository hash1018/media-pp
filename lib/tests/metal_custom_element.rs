//! An element of an application's own, working on VideoToolbox pictures
//! with Metal of its own between this crate's Metal elements, the pictures
//! never leaving the GPU — built from the public API alone, as a crate
//! outside this one would build it.
//!
//! `Pictures -> VideoToolboxUpload -> Mark -> MetalDetectionOverlay ->
//! VideoToolboxDownload -> Collect`: `Mark` reads each uploaded picture
//! through a `MetalSurfaceView`, copies it into a picture of its own
//! `VideoToolboxFramePool` with a blit, puts a black square there, and says
//! where a box is with `Detections`; the overlay after it takes that
//! picture as one of its own and fills the box white. The test then reads
//! both back.

#![cfg(all(target_os = "macos", feature = "metal"))]

use std::{
    ffi::c_void,
    ptr::NonNull,
    sync::{Arc, Mutex},
};

use media_pp::{
    Error, Result,
    buffer::{MediaBuffer, set_time_base},
    bus::BusEvent,
    color::Color,
    contract::{
        InputContract, MediaKind, MemoryDomain, OutputContract, PixelLayoutSet, PortContract,
    },
    element::{Element, ElementType, Filter, Output, Produced, Sink, Source, Wait, element_pp_log},
    elements::{
        Detection, DetectionOverlayOptions, Detections, Hiding, MetalDetectionOverlay,
        MetalFramePlanes, MetalSurfaceView, MetalTexture, RedactStyle, Treatment,
        VideoToolboxDevice, VideoToolboxDownload, VideoToolboxFrameFormat, VideoToolboxFramePool,
        VideoToolboxUpload,
    },
    ffmpeg::{self, ffi, format::Pixel, frame::Video},
    pipeline::Pipeline,
    pp_log::PpLog,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLOrigin, MTLPixelFormat, MTLRegion,
    MTLSize, MTLTexture, MTLTextureDescriptor,
};

const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;
const PICTURES: i64 = 5;
/// The luma every picture is made with.
const GREY: u8 = 100;
/// Where `Mark` puts black, in pixels: x and y from, and the side.
const SQUARE: (usize, usize, usize) = (8, 8, 16);

fn identity(name: &str) -> (Arc<str>, PpLog) {
    (name.into(), element_pp_log(ElementType::Other, name, None))
}

macro_rules! element {
    ($type:ty) => {
        impl Element for $type {
            fn name(&self) -> Arc<str> {
                self.name.clone()
            }
            fn element_type(&self) -> ElementType {
                ElementType::Other
            }
            fn pp_log(&self) -> &PpLog {
                &self.pp_log
            }
            fn pp_log_mut(&mut self) -> &mut PpLog {
                &mut self.pp_log
            }
        }
    };
}

/// `PICTURES` grey NV12 pictures in system memory.
struct Pictures {
    name: Arc<str>,
    pp_log: PpLog,
    next: i64,
}
element!(Pictures);

impl Source for Pictures {
    /// Made as fast as they are asked for, as a file is read.
    fn is_live(&self) -> bool {
        false
    }

    fn produce(&mut self, _wait: &mut Wait<'_>) -> Result<Produced> {
        if self.next == PICTURES {
            return Ok(Produced::End);
        }
        let mut picture = Video::new(Pixel::NV12, WIDTH, HEIGHT);
        picture.data_mut(0).fill(GREY);
        picture.data_mut(1).fill(128);
        picture.set_pts(Some(self.next));
        set_time_base(&mut picture, ffmpeg::Rational(1, 30));
        self.next += 1;
        Ok(Produced::Buffer(MediaBuffer::video(picture)))
    }
}

/// The application's own element: a copy of each picture with a black
/// square on it, carrying a box for the overlay.
struct Mark {
    name: Arc<str>,
    pp_log: PpLog,
    device: VideoToolboxDevice,
    metal: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// The square, black in video range, copied onto each picture's luma.
    black: MetalTexture,
    pool: Option<VideoToolboxFramePool>,
}
element!(Mark);

// SAFETY: a Metal device, its queue and a texture are thread-safe objects,
// which Apple documents as usable from any thread.
unsafe impl Send for Mark {}

fn size(width: usize, height: usize) -> MTLSize {
    MTLSize {
        width,
        height,
        depth: 1,
    }
}

fn origin(x: usize, y: usize) -> MTLOrigin {
    MTLOrigin { x, y, z: 0 }
}

impl Mark {
    fn new(device: VideoToolboxDevice, metal: Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        let queue = metal.newCommandQueue().expect("a command queue");
        let side = SQUARE.2;
        // SAFETY: a plain constructor of a size Metal accepts.
        let descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat::R8Unorm,
                side,
                side,
                false,
            )
        };
        let black = metal
            .newTextureWithDescriptor(&descriptor)
            .expect("a texture");
        let pixels = vec![16u8; side * side];
        // SAFETY: the texture is `side` by `side` bytes in memory the CPU
        // writes, `pixels` that many rows of `side`, and no command buffer
        // uses it yet.
        unsafe {
            black.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                MTLRegion {
                    origin: origin(0, 0),
                    size: size(side, side),
                },
                0,
                NonNull::from(pixels.as_slice()).cast::<c_void>(),
                side,
            );
        }
        let (name, pp_log) = identity("mark");
        Self {
            name,
            pp_log,
            device,
            metal,
            queue,
            black,
            pool: None,
        }
    }

    fn mark(&mut self, picture: &Video) -> Result<MediaBuffer> {
        let from = MetalSurfaceView::new(&self.metal, picture)?;
        let MetalFramePlanes::Nv12 { luma, chroma } = from.planes() else {
            return Err(Error::Other(format!(
                "Mark takes NV12, got {:?}",
                from.format()
            )));
        };
        let (width, height) = (from.width(), from.height());
        if self.pool.as_ref().map(|pool| (pool.width(), pool.height())) != Some((width, height)) {
            self.pool = Some(VideoToolboxFramePool::new(
                &self.device,
                VideoToolboxFrameFormat::Nv12,
                width,
                height,
            )?);
        }
        let mut marked = self.pool.as_ref().expect("made above").get()?;
        let to = MetalSurfaceView::new(&self.metal, &marked)?;
        let MetalFramePlanes::Nv12 {
            luma: to_luma,
            chroma: to_chroma,
        } = to.planes()
        else {
            unreachable!("an NV12 pool's picture");
        };

        let commands = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Error::Other("no command buffer".into()))?;
        let blit = commands
            .blitCommandEncoder()
            .ok_or_else(|| Error::Other("no blit encoder".into()))?;
        let (width, height) = (width as usize, height as usize);
        let chroma_size = size(width.div_ceil(2), height.div_ceil(2));
        let (x, y, side) = SQUARE;
        // SAFETY: every texture is live until the command buffer has
        // completed, below, and each region lies inside both textures: the
        // picture's planes, which the pixel buffers are at least, and the
        // square inside the luma.
        unsafe {
            for (from, to, size) in [
                (luma, to_luma, size(width, height)),
                (chroma, to_chroma, chroma_size),
            ] {
                blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                    from, 0, 0, origin(0, 0), size, to, 0, 0, origin(0, 0),
                );
            }
            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                &self.black, 0, 0, origin(0, 0), size(side, side), to_luma, 0, 0, origin(x, y),
            );
        }
        blit.endEncoding();
        // What the next element reads, finished first.
        commands.commit();
        commands.waitUntilCompleted();
        if commands.status() != MTLCommandBufferStatus::Completed {
            return Err(Error::Other(format!(
                "the copy did not complete: {:?}",
                commands.status()
            )));
        }
        drop(to);
        // SAFETY: two live, distinct frames.
        unsafe { ffi::av_frame_copy_props(marked.as_mut_ptr(), picture.as_ptr()) };
        let found = Detections::new(
            self.name.clone(),
            Arc::from([Arc::from("thing")]),
            vec![Detection::new(0, 0.9, 0.5, 0.5, 0.25, 0.25)],
        );
        Ok(found.attach_to(MediaBuffer::Video(Arc::new(marked).into())))
    }
}

impl Filter for Mark {
    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        let MediaBuffer::Video(picture) = &buf else {
            return Err(Error::Other(format!(
                "Mark takes pictures, got {}",
                buf.kind()
            )));
        };
        out.push(self.mark(picture)?);
        Ok(())
    }

    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::VideoToolbox)
                .with_layouts(PixelLayoutSet::NV12),
        )
    }

    fn output_contract(&self) -> OutputContract {
        OutputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::VideoToolbox)
                .with_layouts(PixelLayoutSet::NV12),
        )
    }
}

/// Each picture's PTS and luma plane, as it came back.
type Seen = Arc<Mutex<Vec<(Option<i64>, Vec<u8>, usize)>>>;

struct Collect {
    name: Arc<str>,
    pp_log: PpLog,
    seen: Seen,
}
element!(Collect);

impl Sink for Collect {
    fn render(&mut self, buf: MediaBuffer) -> Result<()> {
        let MediaBuffer::Video(picture) = &buf else {
            return Ok(());
        };
        self.seen.lock().unwrap().push((
            picture.pts(),
            picture.data(0).to_vec(),
            picture.stride(0),
        ));
        Ok(())
    }
}

#[test]
fn an_application_element_works_on_videotoolbox_pictures_between_this_crates_own() {
    let device = match VideoToolboxDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping: no VideoToolbox ({error})");
            return;
        }
    };
    let Some(metal) = MTLCreateSystemDefaultDevice() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let overlay = MetalDetectionOverlay::new(
        "overlay",
        &device,
        DetectionOverlayOptions {
            others: Treatment {
                draw: None,
                hide: Some(Hiding {
                    margin: 0.0,
                    ..Hiding::new(RedactStyle::Fill(Color::WHITE))
                }),
                min_score: 0.0,
            },
            ..DetectionOverlayOptions::default()
        },
    )
    .expect("overlay");
    let mark = Mark::new(device.clone(), metal);
    let (name, pp_log) = identity("pictures");
    let pictures = Pictures {
        name,
        pp_log,
        next: 0,
    };
    let seen = Seen::default();
    let (name, pp_log) = identity("collect");
    let collect = Collect {
        name,
        pp_log,
        seen: seen.clone(),
    };

    let (pipeline, ()) = Pipeline::new("custom-metal", pictures, |pictures, ctx| {
        let branch = ctx
            .branch()
            .pipe(VideoToolboxUpload::new("upload", &device))
            .pipe(mark)
            .pipe(overlay)
            .pipe(VideoToolboxDownload::new("download"))
            .to(collect)?;
        ctx.attach(pictures, 0, branch)?;
        Ok(())
    })
    .expect("a pipeline whose links agree");
    pipeline.run().expect("run");
    for event in pipeline.bus().iter() {
        match event {
            BusEvent::Finished => break,
            BusEvent::Error { .. } => panic!("{event}"),
            _ => {}
        }
    }
    pipeline.stop();

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), PICTURES as usize, "every picture came back");
    let (x, y, side) = SQUARE;
    // The overlay's box: half-way across and down, a quarter of each side.
    let boxed = (WIDTH as usize / 2, HEIGHT as usize / 2);
    for (index, (pts, luma, stride)) in seen.iter().enumerate() {
        assert_eq!(*pts, Some(index as i64), "picture {index}'s time");
        let at = |x: usize, y: usize| luma[y * stride + x];
        assert_eq!(at(x, y), 16, "picture {index}: Mark's square");
        assert_eq!(
            at(x + side - 1, y + side - 1),
            16,
            "picture {index}: its far corner"
        );
        assert_eq!(
            at(x + side, y + side),
            GREY,
            "picture {index}: past the square"
        );
        assert_eq!(
            at(boxed.0, boxed.1),
            235,
            "picture {index}: the overlay's fill"
        );
        assert_eq!(
            at(WIDTH as usize - 1, 0),
            GREY,
            "picture {index}: untouched"
        );
    }
}
