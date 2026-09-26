use std::sync::Arc;

use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_error, pp_info};

use crate::{
    buffer::MediaBuffer,
    color::ColorDescription,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    elements::VulkanDevice,
    error::Result,
    pad::SrcPad,
    platform::vulkan::{
        bgra_pass::{BgraPass, BgraPassError, PassInput, immediates},
        gpu::VulkanError,
    },
    pool::UnboundObjectPoolRef,
    repeat::{PerFrameTransform, RepeatedOutput},
};

const SHADER: &str = include_str!("../../../../shaders/vulkan/convert.wgsl");

/// Errors specific to [`VulkanConverter`]. Converts into the crate-wide
/// `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum VulkanConverterError {
    /// FFmpeg could not take a second reference to the frame already in
    /// hand, which is how an unchanged input is answered.
    #[error("failed to reference the previous frame (code {0})")]
    FrameRef(i32),

    /// The sink received something other than a decoded video frame.
    #[error("VulkanConverter only accepts Video buffers, got a {0}")]
    UnsupportedBuffer(&'static str),

    /// The frame is not an NV12 Vulkan frame of this element's own device.
    #[error("VulkanConverter {0}")]
    Frame(String),

    /// A Vulkan call this element made failed.
    #[error(transparent)]
    Vulkan(#[from] VulkanError),
}

impl From<BgraPassError> for VulkanConverterError {
    fn from(error: BgraPassError) -> Self {
        match error {
            BgraPassError::Vulkan(error) => Self::Vulkan(error),
            other => Self::Frame(other.to_string()),
        }
    }
}

/// Turns NV12 Vulkan frames into BGRA ones, on the GPU — what a BGRA-only
/// element on Vulkan, a key or an effect, is put behind when what arrives is
/// a decoder's or a camera's NV12. The Vulkan counterpart of
/// `CudaConverter` built for BGRA; the other direction is the NV12
/// [`VulkanVideoCompositor`](crate::elements::VulkanVideoCompositor)'s own.
///
/// Each frame is read by its own colour description — matrix and range,
/// BT.709 above 576 rows and BT.601 at or below where it names none — and
/// comes out full-range RGB, opaque, with its timing carried through and
/// tagged as what it now is.
pub struct VulkanConverter {
    pp_log: PpLog,
    name: Arc<str>,
    pass: BgraPass,
    repeated: RepeatedOutput,
    pad: SrcPad,
}

impl VulkanConverter {
    /// `device` must be the same [`VulkanDevice`] every other Vulkan element
    /// in this pipeline was built from. Frames come out the size they went
    /// in.
    pub fn new(
        name: impl Into<String>,
        device: &VulkanDevice,
    ) -> std::result::Result<Self, VulkanConverterError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::VulkanConverter, &name, None);
        let pass = BgraPass::new(device, SHADER, c"main", 64, PassInput::Nv12)?;
        let pad = SrcPad::with_contract(
            format!("{name}_src"),
            OutputContract::Fixed(
                PortContract::frame(MediaKind::VideoFrame, MemoryDomain::Vulkan)
                    .with_layouts(crate::contract::PixelLayoutSet::BGRA),
            ),
        );
        pp_info!(pp_log: &pp_log, "opened: NV12 -> BGRA on {}", device.name());
        Ok(Self {
            name,
            pp_log,
            pass,
            repeated: RepeatedOutput::new(),
            pad,
        })
    }

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
            .run(source, &immediates(&words))
            .map_err(VulkanConverterError::from)
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

impl PerFrameTransform for VulkanConverter {
    fn repeated(&mut self) -> &mut RepeatedOutput {
        &mut self.repeated
    }

    fn frame_ref_failed(&self, code: i32) -> crate::error::Error {
        pp_error!(self, "av_frame_ref failed: {code}");
        VulkanConverterError::FrameRef(code).into()
    }

    fn produce(
        &mut self,
        source: &Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        self.convert(source)
    }
}

impl Element for VulkanConverter {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::VulkanConverter
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for VulkanConverter {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for VulkanConverter {
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::Vulkan)
                .with_layouts(crate::contract::PixelLayoutSet::NV12),
        )
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Video(frame) => {
                let converted = self.transform(&frame)?;
                self.pad.push(MediaBuffer::Video(converted))
            }
            MediaBuffer::Eos => self.pad.push(MediaBuffer::Eos),
            other => {
                let kind = other.kind();
                pp_error!(self, "unsupported buffer: {kind}");
                Err(VulkanConverterError::UnsupportedBuffer(kind).into())
            }
        }
    }

    fn control(&mut self, msg: &ControlMsg) -> Result<()> {
        if matches!(msg, ControlMsg::Flush | ControlMsg::Stop) {
            self.repeated.clear();
        }
        Ok(())
    }
}

impl Drop for VulkanConverter {
    fn drop(&mut self) {
        pp_info!(self, "dropped: releasing hw contexts");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        elements::{VulkanDownload, VulkanUpload},
        test_support::{capture, try_vulkan_device},
    };

    /// BT.709 limited-range red, grey and blue in NV12 come out as that red,
    /// grey and blue in BGRA — each colour a 2x2 block, as NV12's chroma is
    /// — with the timestamp carried and the frame tagged RGB.
    #[test]
    fn nv12_comes_out_as_its_rgb() {
        let Some(device) = try_vulkan_device() else {
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
        let mut upload = VulkanUpload::new("upload", &device);
        let uploaded = capture(&mut upload);
        upload.consume(MediaBuffer::video(frame)).unwrap();

        let mut converter = VulkanConverter::new("convert", &device).unwrap();
        let converted = capture(&mut converter);
        converter
            .consume(uploaded.lock().unwrap().remove(0))
            .unwrap();
        let mut download = VulkanDownload::new("download", &device);
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
