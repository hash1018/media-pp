use std::sync::Arc;

use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_error, pp_info};

use crate::{
    buffer::MediaBuffer,
    color::ColorDescription,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    element::{Element, ElementType, Filter, Output, element_pp_log},
    elements::VideoToolboxDevice,
    error::Result,
    platform::macos::{
        metal::MetalError,
        metal_pass::{MetalPass, MetalPassError, PassInput, parameters},
    },
    pool::UnboundObjectPoolRef,
    repeat::{PerFrameTransform, RepeatedOutput},
    transform::{FilterStage, filter_stage},
};

const SHADER: &str = include_str!("../../../../shaders/metal/convert.metal");

/// Errors specific to [`MetalConverter`]. Converts into the crate-wide
/// `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum MetalConverterError {
    /// FFmpeg could not take a second reference to the frame already in
    /// hand, which is how an unchanged input is answered.
    #[error("failed to reference the previous frame (code {0})")]
    FrameRef(i32),

    /// The sink received something other than a decoded video frame.
    #[error("MetalConverter only accepts Video buffers, got a {0}")]
    UnsupportedBuffer(&'static str),

    /// The frame is not an NV12 VideoToolbox frame.
    #[error("MetalConverter {0}")]
    Frame(String),

    /// A Metal call this element made failed.
    #[error(transparent)]
    Metal(#[from] MetalError),
}

impl From<MetalPassError> for MetalConverterError {
    fn from(error: MetalPassError) -> Self {
        match error {
            MetalPassError::Metal(error) => Self::Metal(error),
            other => Self::Frame(other.to_string()),
        }
    }
}

/// Turns NV12 VideoToolbox frames into BGRA ones, on the GPU with Metal —
/// what a BGRA-only element on Metal, a key or an effect, is put behind when
/// what arrives is a decoder's or a camera's NV12. The Metal counterpart of
/// `VulkanConverter`; the other direction is the NV12
/// [`MetalVideoCompositor`](crate::elements::MetalVideoCompositor)'s own.
///
/// Each frame is read by its own colour description — matrix and range,
/// BT.709 above 576 rows and BT.601 at or below where it names none — and
/// comes out full-range RGB, opaque, with its timing carried through and
/// tagged as what it now is.
pub struct MetalConverter(FilterStage<Converting>);

filter_stage!(MetalConverter);

/// What a [`MetalConverter`] does to each frame: all of its work, which the
/// framework makes the filter.
struct Converting {
    pp_log: PpLog,
    name: Arc<str>,
    pass: MetalPass,
    repeated: RepeatedOutput,
}

impl MetalConverter {
    /// Output frames are made on `device`; what it takes may come from any,
    /// since a pixel buffer belongs to none. Frames come out the size they
    /// went in.
    pub fn new(
        name: impl Into<String>,
        device: &VideoToolboxDevice,
    ) -> std::result::Result<Self, MetalConverterError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::MetalConverter, &name, None);
        let pass = MetalPass::new(device, SHADER, "convert", PassInput::Nv12)?;
        pp_info!(pp_log: &pp_log, "opened: NV12 -> BGRA");
        Ok(Self(FilterStage::new(Converting {
            name,
            pp_log,
            pass,
            repeated: RepeatedOutput::new(),
        })))
    }
}

impl Converting {
    fn convert(
        &mut self,
        source: &ffmpeg::frame::Video,
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        let rows = ColorDescription::of(source).yuv_to_rgb_rows(source.height());
        let mut words = vec![
            source.width().to_ne_bytes(),
            source.height().to_ne_bytes(),
            0u32.to_ne_bytes(),
            0u32.to_ne_bytes(),
        ];
        words.extend(rows.iter().flatten().map(|value| value.to_ne_bytes()));
        let mut output = self
            .pass
            .run(source, &parameters(&words))
            .map_err(MetalConverterError::from)
            .inspect_err(|error| pp_error!(self, "{error}"))?;
        // What it now is: full-range RGB, with the primaries it had.
        let described = ColorDescription::of(source);
        ColorDescription {
            space: ffmpeg::color::Space::RGB,
            range: ffmpeg::color::Range::JPEG,
            ..described
        }
        .describe(&mut output);
        Ok(output)
    }
}

impl PerFrameTransform for Converting {
    fn repeated(&mut self) -> &mut RepeatedOutput {
        &mut self.repeated
    }

    fn frame_ref_failed(&self, code: i32) -> crate::error::Error {
        pp_error!(self, "av_frame_ref failed: {code}");
        MetalConverterError::FrameRef(code).into()
    }

    fn produce(
        &mut self,
        source: &Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        self.convert(source)
    }
}

impl Element for Converting {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::MetalConverter
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Filter for Converting {
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::VideoToolbox)
                .with_layouts(crate::contract::PixelLayoutSet::NV12),
        )
    }

    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        match buf {
            MediaBuffer::Video(frame) => {
                let converted = PerFrameTransform::transform(self, &frame)?;
                out.push(MediaBuffer::Video(converted));
                Ok(())
            }
            other => {
                let kind = other.kind();
                pp_error!(self, "unsupported buffer: {kind}");
                Err(MetalConverterError::UnsupportedBuffer(kind).into())
            }
        }
    }

    fn reset(&mut self) {
        self.repeated.clear();
    }

    fn output_contract(&self) -> OutputContract {
        OutputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::VideoToolbox)
                .with_layouts(crate::contract::PixelLayoutSet::BGRA),
        )
    }
}

impl Drop for Converting {
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

    /// BT.709 limited-range red, grey and blue in NV12 come out as that red,
    /// grey and blue in BGRA — each colour a 2x2 block, as NV12's chroma is
    /// — with the timestamp carried and the frame tagged RGB.
    #[test]
    fn nv12_comes_out_as_its_rgb() {
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        // Y, Cb, Cr of each block, and the RGB it is.
        let blocks: [([u8; 3], [u8; 3]); 3] = [
            ([63, 102, 240], [255, 0, 0]),
            ([126, 128, 128], [128, 128, 128]),
            ([32, 240, 118], [0, 0, 255]),
        ];
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, 6, 2);
        frame.set_pts(Some(11));
        frame.set_color_space(ffmpeg::color::Space::BT709);
        frame.set_color_range(ffmpeg::color::Range::MPEG);
        let luma_stride = frame.stride(0);
        for (block, ([y, cb, cr], _)) in blocks.iter().enumerate() {
            for row in 0..2 {
                frame.data_mut(0)[row * luma_stride + block * 2..][..2].fill(*y);
            }
            frame.data_mut(1)[block * 2..block * 2 + 2].copy_from_slice(&[*cb, *cr]);
        }
        let mut upload = VideoToolboxUpload::new("upload", &device);
        let uploaded = capture(&mut upload);
        upload.consume(MediaBuffer::video(frame)).unwrap();

        let mut converter = MetalConverter::new("convert", &device).unwrap();
        let converted = capture(&mut converter);
        converter
            .consume(uploaded.lock().unwrap().remove(0))
            .unwrap();
        let mut download = VideoToolboxDownload::new("download");
        let back = capture(&mut download);
        download
            .consume(converted.lock().unwrap().remove(0))
            .unwrap();
        let MediaBuffer::Video(out) = back.lock().unwrap().remove(0) else {
            panic!("a picture");
        };
        assert_eq!(out.format(), ffmpeg::format::Pixel::BGRA);
        assert_eq!(out.pts(), Some(11));
        assert_eq!(out.color_space(), ffmpeg::color::Space::RGB);
        for (block, (_, [r, g, b])) in blocks.iter().enumerate() {
            let at = block * 2 * 4;
            let [got_b, got_g, got_r, a] = out.data(0)[at..at + 4] else {
                unreachable!()
            };
            assert_eq!(a, 255);
            for (got, wanted) in [(got_r, r), (got_g, g), (got_b, b)] {
                assert!(
                    got.abs_diff(*wanted) <= 3,
                    "block {block}: {:?} for {:?}",
                    [got_r, got_g, got_b],
                    [r, g, b]
                );
            }
        }
    }
}
