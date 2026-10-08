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
        metal_pass::{MetalPass, MetalPassError, PassInput, parameters, tone_map_parameters},
        videotoolbox::sw_format_of,
    },
    pool::UnboundObjectPoolRef,
    repeat::{PerFrameTransform, RepeatedOutput},
    transform::{FilterStage, filter_stage},
};

const SHADER: &str = include_str!("../../../../shaders/metal/convert.metal");
const TONE_MAP: &str = include_str!("../../../../shaders/metal/tone_map.metal");

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

    /// The frame is not an NV12 or P010 VideoToolbox frame.
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

/// Turns NV12 and P010 VideoToolbox frames into BGRA ones, on the GPU with
/// Metal — what a BGRA-only element on Metal, a key or an effect, is put
/// behind when what arrives is a decoder's or a camera's NV12, and what
/// brings a 10-bit decode to something `MetalRenderer` shows. The Metal counterpart of
/// `VulkanConverter`; the other direction is the NV12
/// [`MetalVideoCompositor`](crate::elements::MetalVideoCompositor)'s own.
///
/// Each frame is read by its own colour description — matrix and range,
/// BT.709 above 576 rows and BT.601 at or below where it names none — and
/// comes out full-range RGB, opaque, with its timing carried through and
/// tagged as what it now is.
///
/// P010 — what `VideoToolboxDecoder` makes of 10-bit video — is read the
/// same way, its ten bits taken at the top of each sample's two bytes. One
/// tagged PQ or HLG is brought to SDR BT.709 on the way instead, by
/// `core/tone_map.rs`'s definition, the one `CudaConverter` and
/// `D3d11ToneMap` draw with, and tagged BT.709 for it.
pub struct MetalConverter(FilterStage<Converting>);

filter_stage!(MetalConverter);

/// What a [`MetalConverter`] does to each frame: all of its work, which the
/// framework makes the filter.
struct Converting {
    pp_log: PpLog,
    name: Arc<str>,
    pass: MetalPass,
    /// The SDR P010 pass, made at the first such frame.
    wide: Option<MetalPass>,
    /// The HDR P010 pass, made at the first such frame.
    tone_map: Option<MetalPass>,
    /// What output frames are made on, for the passes made later.
    device: VideoToolboxDevice,
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
        pp_info!(pp_log: &pp_log, "opened: NV12 or P010 -> BGRA");
        Ok(Self(FilterStage::new(Converting {
            name,
            pp_log,
            pass,
            wide: None,
            tone_map: None,
            device: device.clone(),
            repeated: RepeatedOutput::new(),
        })))
    }
}

impl Converting {
    fn convert(
        &mut self,
        source: &ffmpeg::frame::Video,
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        let wide = sw_format_of(source) == Ok(ffmpeg::format::Pixel::P010LE);
        if wide && let Some(map) = crate::tone_map::ToneMap::of_frame(source) {
            return self.tone_map(source, &map);
        }
        let rows = ColorDescription::of(source).yuv_to_rgb_rows(source.height());
        let mut words = vec![
            source.width().to_ne_bytes(),
            source.height().to_ne_bytes(),
            0u32.to_ne_bytes(),
            0u32.to_ne_bytes(),
        ];
        words.extend(rows.iter().flatten().map(|value| value.to_ne_bytes()));
        // P010's samples are read as sixteen-bit fractions, which the rows
        // for eight take to within 0.4% — as the tone map reads them.
        let pass = if wide {
            later(
                &mut self.wide,
                &self.device,
                SHADER,
                "convert",
                PassInput::P010,
            )
            .inspect_err(|error| pp_error!(self, "{error}"))?
        } else {
            &mut self.pass
        };
        let mut output = pass
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

    /// An HDR P010 `source` brought to SDR BGRA as `map` says.
    fn tone_map(
        &mut self,
        source: &ffmpeg::frame::Video,
        map: &crate::tone_map::ToneMap,
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        let pass = later(
            &mut self.tone_map,
            &self.device,
            TONE_MAP,
            "tone_map",
            PassInput::P010,
        )
        .inspect_err(|error| pp_error!(self, "{error}"))?;
        let parameters = tone_map_parameters(map, (source.width(), source.height()));
        let mut output = pass
            .run(source, &parameters)
            .map_err(MetalConverterError::from)
            .inspect_err(|error| pp_error!(self, "{error}"))?;
        // What it now is: full-range RGB, SDR BT.709.
        ColorDescription {
            space: ffmpeg::color::Space::RGB,
            range: ffmpeg::color::Range::JPEG,
            ..ColorDescription::BT709_LIMITED
        }
        .describe(&mut output);
        Ok(output)
    }
}

/// The pass in `slot`, made on `device` the first time it is asked for.
fn later<'a>(
    slot: &'a mut Option<MetalPass>,
    device: &VideoToolboxDevice,
    shader: &str,
    entry: &str,
    input: PassInput,
) -> std::result::Result<&'a mut MetalPass, MetalConverterError> {
    match slot {
        Some(pass) => Ok(pass),
        empty => Ok(empty.insert(MetalPass::new(device, shader, entry, input)?)),
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
                .with_layouts(crate::contract::PixelLayoutSet::YUV420),
        )
    }

    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        match buf {
            MediaBuffer::Video(frame) => {
                let converted = PerFrameTransform::transform(self, &frame)?;
                out.push(MediaBuffer::Video(converted.into()));
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

    /// A P010 picture comes out BGRA with its timestamp carried: an HDR
    /// one, HLG or PQ, by `core/tone_map.rs`'s definition, pixel for pixel
    /// to within the rounding of a GPU's powers, and tagged BT.709; an SDR
    /// one as its own rows make it, with the primaries it had.
    #[test]
    fn p010_comes_out_as_its_rgb_tone_mapped_where_hdr() {
        use crate::tone_map::ToneMap;
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let (width, height) = (64usize, 32usize);
        // Ten-bit luma across the range, two chroma pairs off grey.
        let luma = |x: usize, y: usize| (64 + (x * 13 + y * 7) % 877) as u16;
        let chroma = |x: usize, y: usize| -> (u16, u16) {
            match (x + y) % 3 {
                0 => (512, 512),
                1 => (400, 620),
                _ => (600, 450),
            }
        };
        let mut converter = MetalConverter::new("convert", &device).unwrap();
        let converted = capture(&mut converter);
        for transfer in [
            ffmpeg::color::TransferCharacteristic::ARIB_STD_B67,
            ffmpeg::color::TransferCharacteristic::SMPTE2084,
            ffmpeg::color::TransferCharacteristic::BT709,
        ] {
            let mut frame = ffmpeg::frame::Video::new(
                ffmpeg::format::Pixel::P010LE,
                width as u32,
                height as u32,
            );
            frame.set_pts(Some(7));
            let (stride, cstride) = (frame.stride(0), frame.stride(1));
            for y in 0..height {
                for x in 0..width {
                    frame.data_mut(0)[y * stride + x * 2..][..2]
                        .copy_from_slice(&(luma(x, y) << 6).to_le_bytes());
                }
            }
            for y in 0..height / 2 {
                for x in 0..width / 2 {
                    let (cb, cr) = chroma(x, y);
                    let at = y * cstride + x * 4;
                    frame.data_mut(1)[at..at + 2].copy_from_slice(&(cb << 6).to_le_bytes());
                    frame.data_mut(1)[at + 2..at + 4].copy_from_slice(&(cr << 6).to_le_bytes());
                }
            }
            ColorDescription {
                space: ffmpeg::color::Space::BT2020NCL,
                range: ffmpeg::color::Range::MPEG,
                primaries: ffmpeg::color::Primaries::BT2020,
                transfer,
            }
            .describe(&mut frame);
            let mut upload = VideoToolboxUpload::new("upload", &device);
            let uploaded = capture(&mut upload);
            upload.consume(MediaBuffer::video(frame.clone())).unwrap();
            let input = uploaded.lock().unwrap().remove(0);

            let map = ToneMap::of_frame(&frame);
            let rows = ColorDescription::of(&frame).yuv_to_rgb_rows(height as u32);
            converter.consume(input).unwrap();
            let mut download = VideoToolboxDownload::new("download");
            let back = capture(&mut download);
            download
                .consume(converted.lock().unwrap().remove(0))
                .unwrap();
            let MediaBuffer::Video(out) = back.lock().unwrap().remove(0) else {
                panic!("a picture");
            };
            assert_eq!(out.format(), ffmpeg::format::Pixel::BGRA);
            assert_eq!(out.pts(), Some(7));
            assert_eq!(out.color_space(), ffmpeg::color::Space::RGB);
            let primaries = if map.is_some() {
                ffmpeg::color::Primaries::BT709
            } else {
                ffmpeg::color::Primaries::BT2020
            };
            assert_eq!(out.color_primaries(), primaries);
            assert_eq!(
                out.color_transfer_characteristic(),
                ffmpeg::color::TransferCharacteristic::BT709
            );
            let mut worst = 0;
            for y in 0..height {
                for x in 0..width {
                    let (cb, cr) = chroma(x / 2, y / 2);
                    let unit = |code: u16| f32::from(code << 6) / 65535.0;
                    let ycc = [unit(luma(x, y)), unit(cb), unit(cr)];
                    let want = match &map {
                        Some(map) => map.apply(ycc[0], ycc[1], ycc[2]),
                        None => rows.map(|[y, cb, cr, offset]| {
                            let value = y * ycc[0] + cb * ycc[1] + cr * ycc[2] + offset;
                            (value.clamp(0.0, 1.0) * 255.0).round() as u8
                        }),
                    };
                    let got = &out.data(0)[y * out.stride(0) + x * 4..][..4];
                    assert_eq!(got[3], 255);
                    for (got, want) in [got[2], got[1], got[0]].iter().zip(want) {
                        worst = worst.max(got.abs_diff(want));
                    }
                }
            }
            assert!(worst <= 2, "{transfer:?}: off by {worst} of 255");
        }
    }
}
