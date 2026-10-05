use std::sync::Arc;

use ffmpeg_next::{self as ffmpeg, ffi};
use objc2_core_video::kCVPixelFormatType_32BGRA;
use objc2_io_surface::IOSurfaceRef;
use objc2_metal::{MTLPixelFormat, MTLTextureUsage};
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    contract::{MediaKind, MemoryDomain, OutputContract, PixelLayoutSet, PortContract},
    element::{Element, ElementType, SourceStage},
    elements::{AppSource, AppSourceHandle, VideoToolboxDevice, source::app_source::Receiving},
    platform::{
        ffmpeg::AvBufferRef,
        macos::{
            metal::{MetalError, MetalGpu},
            pixel_buffer::PixelBuffer,
            videotoolbox::create_frames_ctx,
        },
    },
    pool::UnboundObjectPool,
    pp_log::pp_info,
    produce::source_stage,
};

/// Errors specific to `MetalSharedTextureSource`. Converts into the
/// crate-wide `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum MetalSharedTextureSourceError {
    /// The surface is not the size this source was opened for. Nothing
    /// here resizes.
    #[error(
        "the IOSurface is {actual_width}x{actual_height}, but \
         MetalSharedTextureSource was opened for {expected_width}x{expected_height}"
    )]
    DimensionMismatch {
        /// Width of the surface pushed.
        actual_width: u32,
        /// Height of the surface pushed.
        actual_height: u32,
        /// Width this source copies into.
        expected_width: u32,
        /// Height this source copies into.
        expected_height: u32,
    },

    /// The surface holds pixels in a format this source does not copy —
    /// its four-character code. See [`MetalSharedTextureSource`]'s own
    /// docs.
    #[error("MetalSharedTextureSource only takes BGRA surfaces, got pixel format {0:#010x}")]
    UnsupportedFormat(u32),

    /// Metal would not make a texture over the surface or this source's
    /// own frame, or would not copy between them.
    #[error(transparent)]
    Metal(#[from] MetalError),

    /// FFmpeg could not set up the pool the frames come from.
    #[error("{0}")]
    Pool(String),

    /// FFmpeg could not take a frame from that pool.
    #[error("failed to take a frame from the VideoToolbox pool (code {0})")]
    FrameGet(i32),

    /// The source has ended — its `Pipeline` finished, or end-of-stream was
    /// already submitted — so nothing more can be pushed into it.
    #[error("MetalSharedTextureSource has already ended")]
    Closed,
}

/// Brings `IOSurface`s another part of the program — or another process —
/// draws into this pipeline, as VideoToolbox frames of its own. The macOS
/// sibling of `D3d11SharedTextureSource`.
///
/// A pushed source rather than a reading one: whoever produces the pictures
/// calls [`MetalSharedTextureHandle::push`] with a surface per picture, and
/// this emits one BGRA `Pixel::VIDEOTOOLBOX` frame downstream for each.
/// That is [`crate::elements::AppSource`]'s shape, which this is built on —
/// the difference is the copy, which is the whole point of the element.
///
/// # Why it copies
///
/// The surface belongs to the producer, and a producer reuses its surfaces:
/// a browser engine draws its next picture into one it handed over two
/// pictures ago. What a pipeline holds has to outlive the call — a
/// compositor keeps the last frame of every input for as long as that input
/// is quiet — so each push copies, with a Metal blit, into a pixel buffer of
/// this pipeline's own. The copy is GPU to GPU; no pixel crosses to the
/// CPU.
///
/// # What it does not do
///
/// Resize, convert, pace, or interpret. A surface of another size or format
/// is refused rather than adapted — chain a [`crate::elements::MetalScaler`]
/// after this for a size — and a producer that paints only when something
/// changed produces frames only then. That is usually enough, since a
/// compositor answers its own rate out of the last frame each input gave
/// it.
///
/// # Colour and alpha
///
/// The frames are said to be RGB at full range, as a screen's are, and hold
/// whatever the producer put in the surface, byte for byte. A browser
/// engine composites its page with the alpha already multiplied into the
/// colour, and a layer drawn from such a frame has to say so — see
/// [`VideoLayer::premultiplied_alpha`](crate::elements::VideoLayer::premultiplied_alpha).
pub struct MetalSharedTextureSource(SourceStage<Receiving>);

source_stage!(MetalSharedTextureSource);

/// Pushes surfaces into a [`MetalSharedTextureSource`].
///
/// Cheap to clone — two refcount bumps and an `Arc` — and every clone feeds
/// the same source. Push [`Self::finish`] when done, or drop every clone,
/// which ends the source the same way [`AppSourceHandle`]'s own drop does.
/// A clone keeps this source's Metal queue and frame pool alive, not its
/// pipeline.
///
/// # Push while the surface is still the picture
///
/// [`Self::push`] copies before it returns, so it must be called while the
/// producer still holds that picture — inside its paint callback, not
/// queued for later. That is the entire reason this work happens on the
/// caller's thread instead of the source's own.
#[derive(Clone)]
pub struct MetalSharedTextureHandle {
    pusher: AppSourceHandle,
    import: Arc<Import>,
}

/// What a push needs: a queue to copy on, the shape it copies into, and the
/// pool the outgoing frames come from.
struct Import {
    gpu: MetalGpu,
    width: u32,
    height: u32,
    /// BGRA VideoToolbox frames of `width` by `height`, their pixel buffers
    /// from Core Video's own pool.
    frames: AvBufferRef,
    /// The small `AVFrame` wrappers only, each handing its pixel buffer
    /// back when downstream lets go of it.
    wrappers: UnboundObjectPool<ffmpeg::frame::Video>,
}

impl MetalSharedTextureSource {
    /// `device` is the VideoToolbox context the frames are made on — the
    /// one the rest of the pipeline shares, although a pixel buffer is
    /// readable on any.
    ///
    /// `width`/`height` are what every pushed surface must be and what the
    /// frames coming out are. `capacity` bounds how many frames may sit
    /// unconsumed before [`MetalSharedTextureHandle::push`] blocks, exactly
    /// as [`AppSource::new`]'s own does.
    pub fn new(
        name: impl Into<String>,
        device: &VideoToolboxDevice,
        width: u32,
        height: u32,
        capacity: usize,
    ) -> std::result::Result<(Self, MetalSharedTextureHandle), MetalSharedTextureSourceError> {
        let gpu = MetalGpu::new()?;
        // SAFETY: a live VideoToolbox device context, held by `device`.
        let frames = unsafe {
            create_frames_ctx(&device.retain(), ffmpeg::format::Pixel::BGRA, width, height)
        }
        .map_err(|error| MetalSharedTextureSourceError::Pool(error.to_string()))?;
        let (source, pusher) = AppSource::receiving(
            name,
            capacity,
            ElementType::MetalSharedTextureSource,
            OutputContract::Fixed(
                PortContract::frame(MediaKind::VideoFrame, MemoryDomain::VideoToolbox)
                    .with_layouts(PixelLayoutSet::BGRA),
            ),
            // Unlike a plain `AppSource`, live: what it emits is whatever
            // the producer drew just now, and a paused pipeline cannot ask
            // it for a first picture.
            true,
        );
        pp_info!(pp_log: source.pp_log(), "opened: {width}x{height}");
        Ok((
            Self(source),
            MetalSharedTextureHandle {
                pusher,
                import: Arc::new(Import {
                    gpu,
                    width,
                    height,
                    frames,
                    wrappers: UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |frame| {
                        // SAFETY: the wrapper's own frame, which nothing
                        // downstream holds once it is back in the pool.
                        unsafe { ffi::av_frame_unref(frame.as_mut_ptr()) }
                    }),
                }),
            },
        ))
    }
}

impl MetalSharedTextureHandle {
    /// Copies `surface` into this pipeline and pushes it downstream as one
    /// frame, stamped `pts` (in the caller's own time base — `None` leaves
    /// it unstamped, which is what a
    /// [`crate::elements::VideoCompositorOptions`]-driven graph wants,
    /// since the compositor sets its own). The frame says no unit for it,
    /// so a [`crate::elements::Pacer`] refuses a stamped one: a picture
    /// pushed as it is drawn is already arriving at its own rate.
    ///
    /// `surface` is a BGRA `IOSurface` of the size this source was opened
    /// for, of one plane: what a browser engine's accelerated paint hands
    /// over, or what `IOSurfaceLookupFromMachPort` gives back for a surface
    /// another process sent. It is read before this returns; nothing keeps
    /// it afterwards.
    ///
    /// What the producer owes in return is a picture that is *finished*,
    /// not merely submitted: this copies on its own Metal queue, which
    /// cannot wait on the producer's, so the producer's drawing has to have
    /// completed — a command buffer waited for, or a callback whose own
    /// contract says the surface is ready to read. Nothing here can check
    /// it, so an unfinished picture shows up as a stale or partly drawn one
    /// rather than as an error.
    ///
    /// Blocks while the source's queue is full, as
    /// [`AppSourceHandle::push`] does. See this type's own docs on where
    /// this must be called from.
    pub fn push(
        &self,
        surface: &IOSurfaceRef,
        pts: Option<i64>,
    ) -> std::result::Result<(), MetalSharedTextureSourceError> {
        let frame = self.import.copy(surface, pts)?;
        self.pusher
            .push(frame)
            .map_err(|_| MetalSharedTextureSourceError::Closed)
    }

    /// [`Self::push`] for a producer that cannot wait — which a browser
    /// engine's paint callback cannot: it is that engine's own thread, and
    /// everything else it does happens there too.
    ///
    /// `Ok(false)` (not an error) means the source's queue was full and this
    /// picture was dropped; `Err` means the source itself has ended. A
    /// pipeline this source feeds can be *paused*, and a paused pipeline
    /// consumes nothing at all, so a blocking push would stop the producer
    /// for as long as the pause lasts.
    ///
    /// The copy still happens — whether there is room is only known once
    /// there is something to put there — so a source nobody is draining
    /// costs a copy per picture until the producer is told to stop drawing.
    pub fn try_push(
        &self,
        surface: &IOSurfaceRef,
        pts: Option<i64>,
    ) -> std::result::Result<bool, MetalSharedTextureSourceError> {
        let frame = self.import.copy(surface, pts)?;
        self.pusher
            .try_push(frame)
            .map_err(|_| MetalSharedTextureSourceError::Closed)
    }

    /// Ends the stream, as [`AppSourceHandle::finish`] does — or drop every
    /// clone of this, which does the same thing.
    pub fn finish(&self) -> std::result::Result<(), MetalSharedTextureSourceError> {
        self.pusher
            .finish()
            .map_err(|_| MetalSharedTextureSourceError::Closed)
    }
}

impl Import {
    /// Copies what `surface` holds into a frame of this pipeline's own.
    fn copy(
        &self,
        surface: &IOSurfaceRef,
        pts: Option<i64>,
    ) -> std::result::Result<MediaBuffer, MetalSharedTextureSourceError> {
        let (width, height) = (surface.width() as u32, surface.height() as u32);
        if (width, height) != (self.width, self.height) {
            return Err(MetalSharedTextureSourceError::DimensionMismatch {
                actual_width: width,
                actual_height: height,
                expected_width: self.width,
                expected_height: self.height,
            });
        }
        let format = surface.pixel_format();
        if format != kCVPixelFormatType_32BGRA || surface.plane_count() > 1 {
            return Err(MetalSharedTextureSourceError::UnsupportedFormat(format));
        }

        let mut output = self.wrappers.get();
        // SAFETY: the pooled wrapper's own `AVFrame`, emptied when it came
        // back to the pool; the frames context is this source's own.
        unsafe {
            let code = ffi::av_hwframe_get_buffer(self.frames.as_ptr(), output.as_mut_ptr(), 0);
            if code < 0 {
                return Err(MetalSharedTextureSourceError::FrameGet(code));
            }
        }
        let buffer = PixelBuffer::of_frame(&output).expect("a frame of this source's pool");
        let from = self.gpu.surface(
            surface,
            MTLPixelFormat::BGRA8Unorm,
            MTLTextureUsage::ShaderRead,
        )?;
        let to = self.gpu.plane(
            &buffer,
            0,
            MTLPixelFormat::BGRA8Unorm,
            MTLTextureUsage::ShaderWrite,
        )?;
        self.gpu.copy(&from, &to)?;

        output.set_pts(pts);
        output.set_color_space(ffmpeg::color::Space::RGB);
        output.set_color_range(ffmpeg::color::Range::JPEG);
        Ok(MediaBuffer::Video(Arc::new(output)))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use objc2_core_foundation::CFRetained;

    use super::*;
    use crate::{
        element::RawSink,
        elements::{VideoToolboxDownload, VideoToolboxUpload},
        pipeline::Pipeline,
        test_support::{capture, try_videotoolbox_device},
    };

    /// A surface as a producer would hand one over: `width` by `height` of
    /// `format`, every pixel of it `bgra` where it is BGRA, drawn and
    /// finished. Made through an upload, whose pixel buffer is on a
    /// surface of its own; the frame is returned too, since the surface is
    /// its.
    fn producer_surface(
        device: &VideoToolboxDevice,
        format: ffmpeg::format::Pixel,
        width: u32,
        height: u32,
        bgra: [u8; 4],
    ) -> (MediaBuffer, CFRetained<IOSurfaceRef>) {
        let mut frame = ffmpeg::frame::Video::new(format, width, height);
        if format == ffmpeg::format::Pixel::BGRA {
            let stride = frame.stride(0);
            for row in 0..height as usize {
                for pixel in frame.data_mut(0)[row * stride..][..width as usize * 4].chunks_mut(4) {
                    pixel.copy_from_slice(&bgra);
                }
            }
        }
        let mut upload = VideoToolboxUpload::new("producer", device);
        let uploaded = capture(&mut upload);
        upload.consume(MediaBuffer::video(frame)).unwrap();
        let uploaded = uploaded.lock().unwrap().remove(0);
        let MediaBuffer::Video(video) = &uploaded else {
            panic!("a picture");
        };
        let surface = PixelBuffer::of_frame(video)
            .and_then(|buffer| buffer.io_surface())
            .expect("an uploaded frame is on a surface");
        (uploaded, surface)
    }

    fn download(
        frame: MediaBuffer,
    ) -> Arc<crate::pool::UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        let mut download = VideoToolboxDownload::new("download");
        let received = capture(&mut download);
        download.consume(frame).expect("download");
        let MediaBuffer::Video(video) = received.lock().unwrap().remove(0) else {
            panic!("a picture");
        };
        video
    }

    /// The point of the element: what the producer drew is what a frame of
    /// this pipeline's own holds, on a surface that is not the producer's.
    #[test]
    fn a_surface_arrives_as_this_pipelines_own_frame() {
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let (_source, handle) = MetalSharedTextureSource::new("shared", &device, 8, 8, 4)
            .expect("MetalSharedTextureSource::new should succeed");
        let (_producer, surface) = producer_surface(
            &device,
            ffmpeg::format::Pixel::BGRA,
            8,
            8,
            [20, 30, 230, 255],
        );

        let frame = handle
            .import
            .copy(&surface, Some(7))
            .expect("importing the producer's surface must succeed");
        let MediaBuffer::Video(video) = &frame else {
            panic!("expected a Video buffer, got {}", frame.kind());
        };
        assert_eq!(video.format(), ffmpeg::format::Pixel::VIDEOTOOLBOX);
        assert_eq!((video.width(), video.height()), (8, 8));
        assert_eq!(video.pts(), Some(7), "the pushed timestamp is kept");
        assert_eq!(video.color_space(), ffmpeg::color::Space::RGB);
        let copied = PixelBuffer::of_frame(video)
            .and_then(|buffer| buffer.io_surface())
            .expect("the frame is on a surface");
        assert!(
            !std::ptr::eq::<IOSurfaceRef>(&*copied, &*surface),
            "the frame is on a surface of this source's own"
        );

        let downloaded = download(frame);
        assert_eq!(downloaded.format(), ffmpeg::format::Pixel::BGRA);
        let stride = downloaded.stride(0);
        for row in 0..8 {
            for pixel in downloaded.data(0)[row * stride..][..8 * 4].chunks(4) {
                assert_eq!(
                    pixel,
                    [20, 30, 230, 255],
                    "the frame holds what the producer drew"
                );
            }
        }
    }

    /// A producer that cannot wait is told there was no room rather than
    /// held until there is. Nothing drains this source — its pipeline is
    /// never started — which is the state a paused one puts a live producer
    /// in.
    #[test]
    fn a_full_queue_costs_a_picture_rather_than_the_producers_thread() {
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let (_source, handle) = MetalSharedTextureSource::new("shared", &device, 8, 8, 1)
            .expect("MetalSharedTextureSource::new should succeed");
        let (_producer, surface) =
            producer_surface(&device, ffmpeg::format::Pixel::BGRA, 8, 8, [0, 0, 0, 255]);
        assert!(
            handle
                .try_push(&surface, None)
                .expect("the source is running"),
            "the first picture takes the one place there is"
        );
        assert!(
            !handle
                .try_push(&surface, None)
                .expect("the source is running"),
            "the second finds it full and is dropped"
        );
    }

    /// Nothing here resizes or converts, so a surface of another shape or
    /// format is a caller mistake reported as one rather than something to
    /// adapt to.
    #[test]
    fn a_surface_of_another_size_or_format_is_refused() {
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let (_source, handle) = MetalSharedTextureSource::new("shared", &device, 8, 8, 4)
            .expect("MetalSharedTextureSource::new should succeed");

        let (_larger, surface) =
            producer_surface(&device, ffmpeg::format::Pixel::BGRA, 16, 16, [0, 0, 0, 255]);
        let error = handle
            .push(&surface, None)
            .expect_err("a surface of another size must be refused");
        assert!(
            matches!(
                error,
                MetalSharedTextureSourceError::DimensionMismatch {
                    actual_width: 16,
                    actual_height: 16,
                    expected_width: 8,
                    expected_height: 8,
                }
            ),
            "expected DimensionMismatch, got {error:?}"
        );

        let (_nv12, surface) =
            producer_surface(&device, ffmpeg::format::Pixel::NV12, 8, 8, [0, 0, 0, 0]);
        let error = handle
            .push(&surface, None)
            .expect_err("an NV12 surface must be refused");
        assert!(
            matches!(error, MetalSharedTextureSourceError::UnsupportedFormat(_)),
            "expected UnsupportedFormat, got {error:?}"
        );
    }

    /// Unlike the plain `AppSource` it is built as, it is live: what it
    /// emits is the producer's latest picture, which a paused pipeline
    /// cannot ask it for.
    #[test]
    fn it_is_live_where_a_plain_app_source_is_not() {
        use crate::element::RawSource;

        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let (source, _handle) =
            MetalSharedTextureSource::new("shared", &device, 8, 8, 4).expect("the source opens");
        assert!(source.is_live());
        let (plain, _handle) = AppSource::new("plain", 4);
        assert!(!plain.is_live());
    }

    /// The element as a pipeline sees it: pushes reach downstream, and
    /// `finish` ends the source.
    #[test]
    fn pushed_surfaces_reach_downstream_and_finish_ends_the_source() {
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let (source, handle) = MetalSharedTextureSource::new("shared", &device, 8, 8, 4)
            .expect("MetalSharedTextureSource::new should succeed");

        let frames = Arc::new(AtomicUsize::new(0));
        let counted = frames.clone();
        let (pipeline, ()) = Pipeline::new("shared-texture", source, move |source, ctx| {
            let branch = ctx.branch().to(crate::elements::AppSink::new(
                "count",
                move |_buf: MediaBuffer| {
                    counted.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                },
            ))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("the pipeline must wire up");
        pipeline.run().expect("the pipeline must start");

        let (_producer, surface) =
            producer_surface(&device, ffmpeg::format::Pixel::BGRA, 8, 8, [0, 0, 0, 255]);
        for pts in 0..3 {
            handle
                .push(&surface, Some(pts))
                .expect("pushing must succeed while the source runs");
        }
        handle.finish().expect("finishing must succeed");

        let events: Vec<_> = pipeline.bus().iter().collect();
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, crate::bus::BusEvent::Error { .. })),
            "unexpected error event(s): {events:?}"
        );
        assert_eq!(frames.load(Ordering::SeqCst), 3);
    }
}
