use std::sync::Arc;

use ffmpeg_next::{self as ffmpeg, ffi};
use objc2_metal::{MTLPixelFormat, MTLTextureUsage};
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_error, pp_info};

use crate::{
    buffer::MediaBuffer,
    contract::{
        InputContract, MediaKind, MemoryDomain, OutputContract, PixelLayoutSet, PortContract,
    },
    element::{Element, ElementType, Filter, Output, element_pp_log},
    elements::VideoToolboxDevice,
    error::Result,
    frame_size::ForSize,
    platform::{
        ffmpeg::AvBufferRef,
        macos::{
            metal::{Kernel, MetalError, MetalGpu, Texture},
            pixel_buffer::PixelBuffer,
            videotoolbox::{NotVideoToolbox, create_frames_ctx, sw_format_of},
        },
    },
    pool::{UnboundObjectPool, UnboundObjectPoolRef},
    repeat::{PerFrameTransform, RepeatedOutput},
    transform::{FilterStage, filter_stage},
};

const SHADER: &str = include_str!("../../../../shaders/metal/scale.metal");

/// Errors specific to [`MetalScaler`]. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum MetalScalerError {
    /// A size of zero was asked for.
    #[error("MetalScaler cannot scale to {width}x{height}")]
    EmptySize {
        /// The width asked for.
        width: u32,
        /// The height asked for.
        height: u32,
    },

    /// FFmpeg could not take a second reference to the frame already in
    /// hand, which is how an unchanged input is answered.
    #[error("failed to reference the previous frame (code {0})")]
    FrameRef(i32),

    /// The sink received something other than a decoded video frame.
    #[error("MetalScaler only accepts Video buffers, got a {0}")]
    UnsupportedBuffer(&'static str),

    /// The frame is not a VideoToolbox frame at all.
    #[error("MetalScaler got a {0:?} frame; upload it first")]
    NotVideoToolbox(ffmpeg::format::Pixel),

    /// The frame says it is a VideoToolbox frame but carries no frames
    /// context to say what it holds.
    #[error("MetalScaler got a VideoToolbox frame with no frames context")]
    NoFramesContext,

    /// The frame holds a layout this does not resize.
    #[error("MetalScaler resizes NV12 and BGRA frames, got {0:?}")]
    UnsupportedLayout(ffmpeg::format::Pixel),

    /// The pool output frames come from could not be made.
    #[error("{0}")]
    Pool(String),

    /// FFmpeg could not hand out an output frame.
    #[error("failed to take a frame from the VideoToolbox pool (code {0})")]
    FrameGet(i32),

    /// A Metal call this element made failed.
    #[error(transparent)]
    Metal(#[from] MetalError),
}

/// How [`MetalScaler`] weighs the source samples each output sample is made
/// from — the Metal counterpart of `VulkanScalerInterp`, with the same four
/// choices.
///
/// Whichever is chosen, shrinking spreads it over every source sample the
/// output one covers, so a picture made smaller is averaged rather than
/// sampled: `Nearest` shrinking is an average over boxes, and `Lanczos` the
/// sharpest of the four. Growing, each is what its name says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalScalerInterp {
    /// The nearest source sample; a box average, shrinking.
    Nearest,
    /// A straight line between the two nearest samples.
    Bilinear,
    /// Catmull-Rom cubic, over four samples.
    Bicubic,
    /// Lanczos with three lobes, over six samples: the sharpest, and what a
    /// picture with text in it wants made smaller.
    Lanczos,
}

impl MetalScalerInterp {
    fn kernel(self) -> u32 {
        match self {
            Self::Nearest => 0,
            Self::Bilinear => 1,
            Self::Bicubic => 2,
            Self::Lanczos => 3,
        }
    }
}

/// Resizes NV12 and BGRA VideoToolbox frames on the GPU with Metal, into
/// frames of the same layout at the size it was made for — the Metal
/// counterpart of `VulkanScaler` and `CudaScaler`, and what keeps
/// `VideoToolboxDecoder -> MetalScaler -> VideoToolboxEncoder`, or a camera
/// or screen into a smaller encode, on the GPU, where the alternative was
/// `VideoToolboxDownload -> SwScaler -> VideoToolboxUpload`.
///
/// Each plane is resampled on its own, across and then down, by the kernel
/// [`MetalScalerInterp`] chooses — see there for what shrinking does to it —
/// reading and writing the pixel buffers where they are. A frame that is
/// already the size asked for goes on as it came.
///
/// The colour description and the timing are carried through unchanged; so
/// is the layout, since this converts nothing.
pub struct MetalScaler(FilterStage<Scaling>);

filter_stage!(MetalScaler);

/// What a [`MetalScaler`] does to each frame: all of its work, which the
/// framework makes the filter.
struct Scaling {
    pp_log: PpLog,
    name: Arc<str>,
    gpu: MetalGpu,
    hw_device_ctx: Arc<AvBufferRef>,
    width: u32,
    height: u32,
    interp: MetalScalerInterp,
    across: Kernel,
    down: Kernel,
    /// Output pools, one per layout, made as the first frame of it arrives.
    nv12_frames: Option<AvBufferRef>,
    bgra_frames: Option<AvBufferRef>,
    /// What the across pass writes, one texture per plane, for the size of
    /// the frames arriving.
    between: ForSize<Vec<Texture>>,
    wrappers: UnboundObjectPool<ffmpeg::frame::Video>,
    repeated: RepeatedOutput,
}

// SAFETY: the FFmpeg buffers have no thread affinity, and the Metal objects
// are thread-safe and touched only through `&mut self`.
unsafe impl Send for Scaling {}

impl MetalScaler {
    /// Output frames are made on `device`; what it takes may come from any,
    /// since a pixel buffer belongs to none. Every frame comes out `width`
    /// by `height`.
    pub fn new(
        name: impl Into<String>,
        device: &VideoToolboxDevice,
        width: u32,
        height: u32,
        interp: MetalScalerInterp,
    ) -> std::result::Result<Self, MetalScalerError> {
        if width == 0 || height == 0 {
            return Err(MetalScalerError::EmptySize { width, height });
        }
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::MetalScaler, &name, None);
        let gpu = MetalGpu::new()?;
        let mut kernels = gpu.kernels(SHADER, &["across", "down"])?.into_iter();
        let (across, down) = (
            kernels.next().expect("one kernel for each name"),
            kernels.next().expect("one kernel for each name"),
        );
        pp_info!(pp_log: &pp_log, "opened: to {width}x{height}, {interp:?}");
        Ok(Self(FilterStage::new(Scaling {
            pp_log,
            name,
            gpu,
            hw_device_ctx: device.retain(),
            width,
            height,
            interp,
            across,
            down,
            nv12_frames: None,
            bgra_frames: None,
            between: ForSize::new(),
            wrappers: UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {}),
            repeated: RepeatedOutput::new(),
        })))
    }
}

impl Scaling {
    fn scale(
        &mut self,
        source: &ffmpeg::frame::Video,
    ) -> std::result::Result<UnboundObjectPoolRef<ffmpeg::frame::Video>, MetalScalerError> {
        let layout = match sw_format_of(source) {
            Ok(layout @ (ffmpeg::format::Pixel::NV12 | ffmpeg::format::Pixel::BGRA)) => layout,
            Ok(other) => return Err(MetalScalerError::UnsupportedLayout(other)),
            Err(NotVideoToolbox::Format(format)) => {
                return Err(MetalScalerError::NotVideoToolbox(format));
            }
            Err(NotVideoToolbox::NoFramesContext) => {
                return Err(MetalScalerError::NoFramesContext);
            }
        };
        let source_buffer = PixelBuffer::of_frame(source)
            .ok_or(MetalScalerError::NotVideoToolbox(source.format()))?;
        let nv12 = layout == ffmpeg::format::Pixel::NV12;
        let (width, height) = (source.width(), source.height());
        let (out_width, out_height) = (self.width, self.height);
        let half = |value: u32| value.div_ceil(2);

        // Each plane: its source and output size — NV12's chroma half of
        // either, rounded up as FFmpeg rounds it — and the format it is read
        // and written through.
        let planes: Vec<([u32; 2], [u32; 2], MTLPixelFormat)> = if nv12 {
            vec![
                (
                    [width, height],
                    [out_width, out_height],
                    MTLPixelFormat::R8Unorm,
                ),
                (
                    [half(width), half(height)],
                    [half(out_width), half(out_height)],
                    MTLPixelFormat::RG8Unorm,
                ),
            ]
        } else {
            vec![(
                [width, height],
                [out_width, out_height],
                MTLPixelFormat::BGRA8Unorm,
            )]
        };

        // Made before an output frame is taken, so a failure here leaves
        // nothing half done.
        let gpu = &self.gpu;
        let between = self
            .between
            .try_get(width, height, |_, _| {
                planes
                    .iter()
                    .map(|&([_, source_height], [output_width, _], _)| {
                        gpu.texture(
                            MTLPixelFormat::RGBA16Float,
                            output_width,
                            source_height,
                            MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite,
                            false,
                        )
                    })
                    .collect::<std::result::Result<Vec<_>, _>>()
            })?
            .clone();
        let pool = if nv12 {
            &mut self.nv12_frames
        } else {
            &mut self.bgra_frames
        };
        if pool.is_none() {
            // SAFETY: a live VideoToolbox device context, this element's own.
            let made =
                unsafe { create_frames_ctx(&self.hw_device_ctx, layout, out_width, out_height) }
                    .map_err(|error| MetalScalerError::Pool(error.to_string()))?;
            *pool = Some(made);
        }
        let frames = pool.as_ref().expect("made above").as_ptr();

        let mut output = self.wrappers.get();
        // SAFETY: the pooled wrapper's own `AVFrame`, its previous pixel
        // buffer handed back first; the frames context is this element's own.
        unsafe {
            let dst = output.as_mut_ptr();
            ffi::av_frame_unref(dst);
            let code = ffi::av_hwframe_get_buffer(frames, dst, 0);
            if code < 0 {
                return Err(MetalScalerError::FrameGet(code));
            }
        }
        let output_buffer = PixelBuffer::of_frame(&output).expect("a frame of this element's pool");

        let mut pass = self.gpu.pass()?;
        let mut held = Vec::with_capacity(planes.len() * 2);
        for (index, (&(source_size, output_size, format), between)) in
            planes.iter().zip(&between).enumerate()
        {
            let read =
                self.gpu
                    .plane(&source_buffer, index, format, MTLTextureUsage::ShaderRead)?;
            let written =
                self.gpu
                    .plane(&output_buffer, index, format, MTLTextureUsage::ShaderWrite)?;
            let parameters: Vec<u8> = [
                source_size[0],
                source_size[1],
                output_size[0],
                output_size[1],
                self.interp.kernel(),
                0,
                0,
                0,
            ]
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
            pass.dispatch(
                &self.across,
                &[between, &read],
                Some(&parameters),
                (output_size[0], source_size[1]),
            );
            pass.dispatch(
                &self.down,
                &[between, &written],
                Some(&parameters),
                (output_size[0], output_size[1]),
            );
            held.push(read);
            held.push(written);
        }
        pass.finish()?;
        drop(held);
        // SAFETY: two distinct live frames; props are timing, colour and side
        // data, not buffers. The source's crop is the source's: the output
        // is exactly the size it was made at.
        unsafe {
            let dst = output.as_mut_ptr();
            ffi::av_frame_copy_props(dst, source.as_ptr());
            (*dst).crop_top = 0;
            (*dst).crop_bottom = 0;
            (*dst).crop_left = 0;
            (*dst).crop_right = 0;
        }
        Ok(output)
    }
}

impl PerFrameTransform for Scaling {
    fn repeated(&mut self) -> &mut RepeatedOutput {
        &mut self.repeated
    }

    fn frame_ref_failed(&self, code: i32) -> crate::error::Error {
        pp_error!(self, "av_frame_ref failed: {code}");
        MetalScalerError::FrameRef(code).into()
    }

    fn produce(
        &mut self,
        source: &Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        self.scale(source)
            .inspect_err(|error| pp_error!(self, "{error}"))
            .map_err(Into::into)
    }
}

impl Element for Scaling {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::MetalScaler
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Filter for Scaling {
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::VideoToolbox)
                .with_layouts(PixelLayoutSet::NV12_OR_BGRA),
        )
    }

    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        match buf {
            // Already the size asked for: nothing to do to it.
            MediaBuffer::Video(frame)
                if frame.width() == self.width && frame.height() == self.height =>
            {
                out.push(MediaBuffer::Video(frame));
                Ok(())
            }
            MediaBuffer::Video(frame) => {
                let scaled = PerFrameTransform::transform(self, &frame)?;
                out.push(MediaBuffer::Video(scaled));
                Ok(())
            }
            other => {
                let kind = other.kind();
                pp_error!(self, "unsupported buffer: {kind}");
                Err(MetalScalerError::UnsupportedBuffer(kind).into())
            }
        }
    }

    fn reset(&mut self) {
        self.repeated.clear();
    }

    fn output_contract(&self) -> OutputContract {
        OutputContract::SameLayout(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::VideoToolbox)
                .with_layouts(PixelLayoutSet::NV12_OR_BGRA),
        )
    }
}

impl Drop for Scaling {
    fn drop(&mut self) {
        pp_info!(self, "dropped: releasing hw contexts");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::RawSink;
    use crate::{
        elements::{VideoToolboxDownload, VideoToolboxUpload},
        test_support::{capture, try_videotoolbox_device},
    };

    /// `frame` up to VideoToolbox, through `scaler`, and back.
    fn through(
        device: &VideoToolboxDevice,
        scaler: &mut MetalScaler,
        frame: ffmpeg::frame::Video,
    ) -> ffmpeg::frame::Video {
        let mut upload = VideoToolboxUpload::new("upload", device);
        let uploaded = capture(&mut upload);
        upload.consume(MediaBuffer::video(frame)).unwrap();
        let scaled = capture(scaler);
        scaler.consume(uploaded.lock().unwrap().remove(0)).unwrap();
        let mut download = VideoToolboxDownload::new("download");
        let back = capture(&mut download);
        download.consume(scaled.lock().unwrap().remove(0)).unwrap();
        let MediaBuffer::Video(out) = back.lock().unwrap().remove(0) else {
            panic!("a picture");
        };
        Arc::try_unwrap(out)
            .map(|pooled| (*pooled).clone())
            .unwrap_or_else(|shared| (**shared).clone())
    }

    /// An NV12 picture of `width` by `height`, each column's luma and each
    /// chroma column's Cb and Cr from the functions given.
    fn nv12(
        width: u32,
        height: u32,
        luma: impl Fn(usize, usize) -> u8,
        chroma: impl Fn(usize, usize) -> [u8; 2],
    ) -> ffmpeg::frame::Video {
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, width, height);
        let (luma_stride, chroma_stride) = (frame.stride(0), frame.stride(1));
        for y in 0..height as usize {
            for x in 0..width as usize {
                frame.data_mut(0)[y * luma_stride + x] = luma(x, y);
            }
        }
        for y in 0..height as usize / 2 {
            for x in 0..width as usize / 2 {
                frame.data_mut(1)[y * chroma_stride + x * 2..][..2].copy_from_slice(&chroma(x, y));
            }
        }
        frame
    }

    /// Made smaller, an NV12 picture keeps its colours where they are, and
    /// its timing and colour description.
    #[test]
    fn nv12_is_resized_and_keeps_its_colours() {
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        // BT.709 limited-range red on the left half, blue on the right.
        let (red, blue) = ((63, [102, 240]), (32, [240, 118]));
        let mut frame = nv12(
            64,
            32,
            |x, _| if x < 32 { red.0 } else { blue.0 },
            |x, _| if x < 16 { red.1 } else { blue.1 },
        );
        frame.set_pts(Some(21));
        frame.set_color_space(ffmpeg::color::Space::BT709);
        frame.set_color_range(ffmpeg::color::Range::MPEG);
        let mut scaler =
            MetalScaler::new("scale", &device, 32, 16, MetalScalerInterp::Lanczos).unwrap();
        let out = through(&device, &mut scaler, frame);

        assert_eq!((out.width(), out.height()), (32, 16));
        assert_eq!(out.format(), ffmpeg::format::Pixel::NV12);
        assert_eq!(out.pts(), Some(21));
        assert_eq!(out.color_space(), ffmpeg::color::Space::BT709);
        let (luma_stride, chroma_stride) = (out.stride(0), out.stride(1));
        for y in 0..16 {
            for (x, (luma, _)) in [(4, red), (27, blue)] {
                let got = out.data(0)[y * luma_stride + x];
                assert!(got.abs_diff(luma) <= 2, "luma at {x},{y}: {got} for {luma}");
            }
        }
        for y in 0..8 {
            for (x, (_, chroma)) in [(2, red), (13, blue)] {
                let got = &out.data(1)[y * chroma_stride + x * 2..][..2];
                for (got, wanted) in got.iter().zip(chroma) {
                    assert!(
                        got.abs_diff(wanted) <= 2,
                        "chroma at {x},{y}: {got} for {wanted}"
                    );
                }
            }
        }
    }

    /// Made larger, a BGRA picture keeps its channels in their order: red
    /// stays red and green green, and alpha goes through.
    #[test]
    fn bgra_is_resized_in_its_own_byte_order() {
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, 8, 4);
        let stride = frame.stride(0);
        for y in 0..4 {
            for x in 0..8 {
                let pixel = if x < 4 {
                    [0, 0, 255, 255]
                } else {
                    [0, 255, 0, 200]
                };
                frame.data_mut(0)[y * stride + x * 4..][..4].copy_from_slice(&pixel);
            }
        }
        let mut scaler =
            MetalScaler::new("scale", &device, 16, 8, MetalScalerInterp::Bilinear).unwrap();
        let out = through(&device, &mut scaler, frame);

        assert_eq!((out.width(), out.height()), (16, 8));
        assert_eq!(out.format(), ffmpeg::format::Pixel::BGRA);
        let stride = out.stride(0);
        for y in 0..8 {
            assert_eq!(
                &out.data(0)[y * stride..][..4],
                &[0, 0, 255, 255],
                "row {y}, left"
            );
            assert_eq!(
                &out.data(0)[y * stride + 15 * 4..][..4],
                &[0, 255, 0, 200],
                "row {y}, right"
            );
        }
    }

    /// Made a quarter of the size, a checkerboard of single pixels comes out
    /// the grey it averages to, whichever kernel: shrinking widens each over
    /// every source sample the output one covers.
    #[test]
    fn shrinking_averages_what_it_covers() {
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        for interp in [
            MetalScalerInterp::Nearest,
            MetalScalerInterp::Bilinear,
            MetalScalerInterp::Bicubic,
            MetalScalerInterp::Lanczos,
        ] {
            let frame = nv12(
                64,
                64,
                |x, y| if (x + y) % 2 == 0 { 16 } else { 235 },
                |_, _| [128, 128],
            );
            let mut scaler = MetalScaler::new("scale", &device, 16, 16, interp).unwrap();
            let out = through(&device, &mut scaler, frame);
            let stride = out.stride(0);
            for y in 0..16 {
                for x in 0..16 {
                    let got = out.data(0)[y * stride + x];
                    assert!(
                        got.abs_diff(126) <= 8,
                        "{interp:?} at {x},{y}: {got}, not the grey it averages to"
                    );
                }
            }
        }
    }

    /// A frame already the size asked for goes on as it came.
    #[test]
    fn a_frame_of_the_size_asked_for_goes_on_as_it_came() {
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let mut upload = VideoToolboxUpload::new("upload", &device);
        let uploaded = capture(&mut upload);
        upload
            .consume(MediaBuffer::video(nv12(
                32,
                16,
                |_, _| 90,
                |_, _| [128, 128],
            )))
            .unwrap();
        let MediaBuffer::Video(frame) = uploaded.lock().unwrap().remove(0) else {
            panic!("a picture");
        };
        let mut scaler =
            MetalScaler::new("scale", &device, 32, 16, MetalScalerInterp::Lanczos).unwrap();
        let scaled = capture(&mut scaler);
        scaler
            .consume(MediaBuffer::Video(Arc::clone(&frame)))
            .unwrap();
        let MediaBuffer::Video(out) = scaled.lock().unwrap().remove(0) else {
            panic!("a picture");
        };
        assert!(Arc::ptr_eq(&out, &frame), "the same frame, not a copy");
    }

    /// A frame in system memory is refused by name rather than read as
    /// something it is not.
    #[test]
    fn a_frame_not_on_the_gpu_is_refused() {
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let mut scaler =
            MetalScaler::new("scale", &device, 16, 8, MetalScalerInterp::Bilinear).unwrap();
        let error = scaler
            .consume(MediaBuffer::video(nv12(
                32,
                16,
                |_, _| 90,
                |_, _| [128, 128],
            )))
            .unwrap_err();
        assert!(
            matches!(
                error,
                crate::error::Error::MetalScalerError(MetalScalerError::NotVideoToolbox(
                    ffmpeg::format::Pixel::NV12
                ))
            ),
            "{error}"
        );
    }

    /// No size to scale to is refused when the scaler is made.
    #[test]
    fn an_empty_size_is_refused() {
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        assert!(matches!(
            MetalScaler::new("scale", &device, 0, 8, MetalScalerInterp::Bilinear),
            Err(MetalScalerError::EmptySize {
                width: 0,
                height: 8
            })
        ));
    }
}
