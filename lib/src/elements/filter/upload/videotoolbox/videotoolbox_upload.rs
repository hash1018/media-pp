use std::sync::Arc;

use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_error, pp_info};

use crate::{
    buffer::MediaBuffer,
    contract::{
        InputContract, MediaKind, MemoryDomain, OutputContract, PixelLayout, PixelLayoutSet,
        PortContract,
    },
    element::{Element, ElementType, Output, Transform, element_pp_log},
    elements::{VideoToolboxDevice, filter::upload::nv12},
    error::Result,
    platform::{
        ffmpeg::AvBufferRef,
        macos::videotoolbox::{VideoToolboxFramesContextError, create_frames_ctx},
    },
    pool::{UnboundObjectPool, UnboundObjectPoolRef},
    repeat::{PerFrameTransform, RepeatedOutput},
    transform::{TransformStage, transform_filter},
};

/// Errors specific to [`VideoToolboxUpload`]. Converts into the crate-wide
/// `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum VideoToolboxUploadError {
    /// FFmpeg could not take a second reference to the upload already in
    /// hand, which is how an unchanged CPU picture is answered.
    #[error("failed to reference the previous upload (code {0})")]
    FrameRef(i32),

    /// The frame is in a layout this does not upload.
    #[error("VideoToolboxUpload uploads NV12, P010, BGRA and YUV420P frames, got {0:?}")]
    UnsupportedFormat(ffmpeg::format::Pixel),

    /// The sink received a buffer other than decoded video or end-of-stream.
    #[error("VideoToolboxUpload only accepts Video buffers, got a {0}")]
    UnsupportedBuffer(&'static str),

    /// FFmpeg could not make a pool of VideoToolbox frames for the frame's
    /// size and layout.
    #[error("{0}")]
    Pool(String),

    /// FFmpeg could not take a frame from the pool.
    #[error("failed to take a frame from the VideoToolbox pool (code {0})")]
    FrameGet(i32),

    /// FFmpeg failed to copy the pixels into the pixel buffer.
    #[error("CPU to VideoToolbox transfer failed (code {0})")]
    Transfer(i32),

    /// A CPU frame plane is shorter than its stride and height require.
    #[error(
        "frame's plane holds {actual} bytes, too few for {height} rows of \
         stride {stride}; uploading it would read past the end of the buffer"
    )]
    PlaneTooSmall {
        /// Bytes actually available in the plane.
        actual: usize,
        /// Declared row stride in bytes.
        stride: usize,
        /// Number of rows that must be uploaded.
        height: u32,
    },
}

impl From<VideoToolboxFramesContextError> for VideoToolboxUploadError {
    fn from(error: VideoToolboxFramesContextError) -> Self {
        Self::Pool(error.to_string())
    }
}

/// The layouts that go up as they are.
const LAYOUTS: PixelLayoutSet =
    PixelLayoutSet::from_slice(&[PixelLayout::Nv12, PixelLayout::P010, PixelLayout::Bgra]);

/// Uploads `Video` frames in system memory into VideoToolbox frames — Core
/// Video pixel buffers — on a [`VideoToolboxDevice`], for a VideoToolbox
/// encoder to take. The macOS sibling of `VulkanUpload`;
/// [`crate::elements::VideoToolboxDownload`] is the mirror of this element.
///
/// A `Filter`: receives via `Sink`, pushes the uploaded frame into its own
/// single src pad. PTS, duration, and color metadata are carried across with
/// `av_frame_copy_props`, so this creates no new timeline.
///
/// NV12, P010 and BGRA go up as they are, each into pixel buffers of its own
/// layout, so this needs no format chosen up front: a frame's own decides.
/// YUV420P (and YUVJ420P, tagged full range) goes up as NV12, the same
/// samples with Cb and Cr interleaved on the CPU on the way. Put a
/// [`crate::elements::SwScaler`] in front of a source that produces none of
/// these.
///
/// The pool frames come from is made for the first frame's size and layout,
/// and made again when a source changes either mid-stream.
pub struct VideoToolboxUpload(TransformStage<Uploading>);

transform_filter!(VideoToolboxUpload);

/// What a [`VideoToolboxUpload`] does to each frame: all of its work, which the
/// framework makes the filter.
struct Uploading {
    pp_log: PpLog,
    name: Arc<str>,
    /// This element's own reference to the device, which frames are made on.
    hw_device_ctx: Arc<AvBufferRef>,
    /// The pool uploaded frames come from, and the size and layout it was
    /// made for.
    frames: Option<(u32, u32, ffmpeg::format::Pixel, AvBufferRef)>,
    /// Reuses only the small CPU-side `AVFrame` wrapper; the pixel buffer
    /// itself comes from `frames`' own pool.
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
    /// Where a YUV420P frame is made NV12 before it goes up, kept for the
    /// next one of the same size.
    staging: Option<ffmpeg::frame::Video>,
    /// The last upload and the CPU picture it came from — see
    /// [`RepeatedOutput`].
    repeated: RepeatedOutput,
}

// SAFETY: the buffers are heap-allocated FFmpeg buffers with no thread
// affinity of their own, and `&mut self` on every method that touches them
// rules out concurrent access. Same reasoning as `CudaUpload`.
unsafe impl Send for Uploading {}

impl VideoToolboxUpload {
    /// `device` is the [`VideoToolboxDevice`] the pipeline's other
    /// VideoToolbox elements share. This element takes its own reference, so
    /// `device` itself need not outlive the call.
    pub fn new(name: impl Into<String>, device: &VideoToolboxDevice) -> Self {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::VideoToolboxUpload, &name, None);
        pp_info!(pp_log: &pp_log, "opened: -> VideoToolbox");
        Self(TransformStage::new(Uploading {
            name,
            pp_log,
            hw_device_ctx: device.retain(),
            frames: None,
            pool: UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {}),
            staging: None,
            repeated: RepeatedOutput::new(),
        }))
    }
}

impl Uploading {
    /// The pool for `width` by `height` frames holding `format`, made
    /// where there is none yet or the one held is for another size or
    /// layout. The previous one is kept if making the new one fails.
    fn frames_for(
        &mut self,
        width: u32,
        height: u32,
        format: ffmpeg::format::Pixel,
    ) -> std::result::Result<*mut ffi::AVBufferRef, VideoToolboxUploadError> {
        let held = self
            .frames
            .as_ref()
            .is_some_and(|&(w, h, f, _)| (w, h, f) == (width, height, format));
        if !held {
            pp_info!(self, "allocating a {width}x{height} {format:?} pool");
            // SAFETY: `hw_device_ctx` is this element's own reference to a live
            // VideoToolbox device context.
            let frames = unsafe { create_frames_ctx(&self.hw_device_ctx, format, width, height) }?;
            self.frames = Some((width, height, format, frames));
        }
        Ok(self.frames.as_ref().expect("made above").3.as_ptr())
    }

    fn upload(
        &mut self,
        source: &ffmpeg::frame::Video,
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        let planar = nv12::is_planar_420(source.format());
        let layout = if planar {
            ffmpeg::format::Pixel::NV12
        } else {
            source.format()
        };
        if !matches!(
            layout,
            ffmpeg::format::Pixel::NV12
                | ffmpeg::format::Pixel::P010LE
                | ffmpeg::format::Pixel::BGRA
        ) {
            pp_error!(self, "unsupported pixel format: {:?}", source.format());
            return Err(VideoToolboxUploadError::UnsupportedFormat(source.format()).into());
        }
        let staged = if planar {
            let staged = nv12::staged(source, self.staging.take()).map_err(
                |nv12::PlaneTooSmall {
                     actual,
                     stride,
                     height,
                 }| VideoToolboxUploadError::PlaneTooSmall {
                    actual,
                    stride,
                    height,
                },
            );
            Some(staged.inspect_err(|error| pp_error!(self, "{error}"))?)
        } else {
            None
        };
        let pixels = staged.as_ref().unwrap_or(source);
        let frames_ctx = self
            .frames_for(source.width(), source.height(), layout)
            .inspect_err(|error| pp_error!(self, "{error}"))?;

        let mut destination = self.pool.get();
        // SAFETY: `dst` is the pooled wrapper's own `AVFrame`, and the unref
        // before the allocation is what hands its previous image back to its
        // pool. The frames context is this element's own, held for its life.
        unsafe {
            let dst = destination.as_mut_ptr();
            ffi::av_frame_unref(dst);
            let code = ffi::av_hwframe_get_buffer(frames_ctx, dst, 0);
            if code < 0 {
                pp_error!(self, "av_hwframe_get_buffer failed: {code}");
                return Err(VideoToolboxUploadError::FrameGet(code).into());
            }
            let code = ffi::av_hwframe_transfer_data(dst, pixels.as_ptr(), 0);
            if code < 0 {
                pp_error!(self, "av_hwframe_transfer_data failed: {code}");
                return Err(VideoToolboxUploadError::Transfer(code).into());
            }
            // Pixels only move with the transfer: PTS, duration and colour
            // are part of the buffer contract.
            ffi::av_frame_copy_props(dst, source.as_ptr());
        }
        // The J of YUVJ420P is its range, which on NV12 only the field can say.
        if source.format() == ffmpeg::format::Pixel::YUVJ420P {
            destination.set_color_range(ffmpeg::color::Range::JPEG);
        }
        self.staging = staged;
        Ok(destination)
    }
}

impl PerFrameTransform for Uploading {
    fn repeated(&mut self) -> &mut RepeatedOutput {
        &mut self.repeated
    }

    fn frame_ref_failed(&self, code: i32) -> crate::error::Error {
        pp_error!(self, "av_frame_ref failed: {code}");
        VideoToolboxUploadError::FrameRef(code).into()
    }

    fn produce(
        &mut self,
        source: &Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        self.upload(source)
    }
}

impl Element for Uploading {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::VideoToolboxUpload
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Transform for Uploading {
    /// CPU-readable planes, in a layout that goes up — see the type's docs.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::System).with_layouts(
                PixelLayoutSet::from_slice(&[
                    PixelLayout::Nv12,
                    PixelLayout::P010,
                    PixelLayout::Bgra,
                    PixelLayout::Yuv420p,
                ]),
            ),
        )
    }

    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        match buf {
            // The same CPU buffer as last time is the same pixels, and
            // uploading them again makes the pixel buffer already in hand.
            MediaBuffer::Video(frame) => {
                let uploaded = PerFrameTransform::transform(self, &frame)?;
                out.push(MediaBuffer::Video(uploaded));
                Ok(())
            }
            other => Err(VideoToolboxUploadError::UnsupportedBuffer(other.kind()).into()),
        }
    }

    fn reset(&mut self) {
        // A per-frame transfer: nothing held beyond the cached upload.
        self.repeated.clear();
    }

    fn output_contract(&self) -> OutputContract {
        OutputContract::SameLayout(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::VideoToolbox)
                .with_layouts(LAYOUTS),
        )
    }
}

impl Drop for Uploading {
    fn drop(&mut self) {
        pp_info!(self, "dropped: releasing hw contexts");
    }
}
