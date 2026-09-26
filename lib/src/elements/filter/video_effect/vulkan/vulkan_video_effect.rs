use std::sync::Arc;

use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_debug, pp_error, pp_info};

use super::super::handle::{VideoEffectControl, VideoEffectHandle};
use super::super::options::{EffectParams, VideoEffect};
use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    elements::VulkanDevice,
    error::Result,
    pad::SrcPad,
    platform::vulkan::{
        bgra_pass::{BgraPass, BgraPassError, immediates},
        gpu::VulkanError,
    },
    pool::UnboundObjectPoolRef,
    repeat::{PerFrameTransform, RepeatedOutput},
};

const SHADER: &str = include_str!("../../../../shaders/vulkan/video_effect.wgsl");

/// Errors specific to [`VulkanVideoEffect`]. Converts into the crate-wide
/// `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum VulkanVideoEffectError {
    /// FFmpeg could not take a second reference to the frame already in
    /// hand, which is how an unchanged input is answered.
    #[error("failed to reference the previous frame (code {0})")]
    FrameRef(i32),

    /// The sink received something other than a decoded video frame.
    #[error("VulkanVideoEffect only accepts Video buffers, got a {0}")]
    UnsupportedBuffer(&'static str),

    /// The frame is not a BGRA Vulkan frame of this element's own device.
    #[error("VulkanVideoEffect {0}")]
    Frame(String),

    /// A Vulkan call this element made failed.
    #[error(transparent)]
    Vulkan(#[from] VulkanError),
}

impl From<BgraPassError> for VulkanVideoEffectError {
    fn from(error: BgraPassError) -> Self {
        match error {
            BgraPassError::Vulkan(error) => Self::Vulkan(error),
            other => Self::Frame(other.to_string()),
        }
    }
}

/// Applies a [`VideoEffect`] — a colour correction or a luma key — to a
/// BGRA Vulkan frame, on the GPU: the Vulkan member of the family whose
/// software member is [`SwVideoEffect`](crate::elements::SwVideoEffect),
/// evaluating the same resolved numbers per pixel as every other.
///
/// BGRA in and out, at whatever size the frames arrive in, on the
/// [`VulkanDevice`] every other Vulkan element in the pipeline uses, with
/// PTS, duration and colour tags carried through. An effect that changes
/// nothing, and an element that is turned off, hand each frame straight
/// through — the same picture, not a copy of it.
pub struct VulkanVideoEffect {
    pp_log: PpLog,
    name: Arc<str>,
    pass: BgraPass,
    /// The last output and the picture it was made from — see
    /// [`RepeatedOutput`]. Cleared when the effect changes.
    repeated: RepeatedOutput,
    /// The effect in force and what it resolves to, refreshed from `control`
    /// once per frame.
    effect: VideoEffect,
    params: EffectParams,
    control: Arc<VideoEffectControl>,
    enabled: bool,
    pad: SrcPad,
}

impl VulkanVideoEffect {
    /// `device` must be the same [`VulkanDevice`] every other Vulkan element
    /// in this pipeline was built from. Output dimensions are the input's:
    /// an effect is per pixel.
    pub fn new(
        name: impl Into<String>,
        device: &VulkanDevice,
        effect: VideoEffect,
    ) -> std::result::Result<(Self, VideoEffectHandle), VulkanVideoEffectError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::VulkanVideoEffect, &name, None);
        let pass = BgraPass::new(device, SHADER, c"main", 96)?;
        let pad = SrcPad::with_contract(
            format!("{name}_src"),
            OutputContract::Fixed(
                PortContract::frame(MediaKind::VideoFrame, MemoryDomain::Vulkan)
                    .with_layouts(crate::contract::PixelLayoutSet::BGRA),
            ),
        );
        pp_info!(pp_log: &pp_log, "opened: BGRA, {} {effect:?}", effect.name());
        let control = Arc::new(VideoEffectControl::new(effect));
        let handle = VideoEffectHandle::new(control.clone());
        Ok((
            Self {
                name,
                pp_log,
                pass,
                repeated: RepeatedOutput::new(),
                effect,
                params: effect.params(),
                control,
                enabled: true,
                pad,
            },
            handle,
        ))
    }

    /// Picks up whatever the handle has been set to, once per frame.
    fn refresh(&mut self) {
        let effect = self.control.get();
        if effect != self.effect {
            pp_debug!(self, "retuned: {} {effect:?}", effect.name());
            self.effect = effect;
            self.params = effect.params();
            self.repeated.clear();
        }
        let enabled = self.control.enabled();
        if enabled != self.enabled {
            pp_debug!(self, "effect {}", if enabled { "on" } else { "off" });
            self.enabled = enabled;
            self.repeated.clear();
        }
    }

    fn run(
        &mut self,
        source: &ffmpeg::frame::Video,
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        let params = &self.params;
        let float = |value: f32| value.to_ne_bytes();
        let mut words = vec![
            source.width().to_ne_bytes(),
            source.height().to_ne_bytes(),
            0u32.to_ne_bytes(),
            0u32.to_ne_bytes(),
        ];
        words.extend(params.rows.iter().flatten().map(|&value| float(value)));
        words.extend(
            [
                params.exponent,
                params.opacity,
                params.luma_low,
                params.luma_low_inv,
                params.luma_high,
                params.luma_high_inv,
                0.0,
                0.0,
            ]
            .map(float),
        );
        self.pass
            .run(source, &immediates(&words))
            .map_err(VulkanVideoEffectError::from)
            .inspect_err(|error| pp_error!(self, "{error}"))
            .map_err(Into::into)
    }
}

impl PerFrameTransform for VulkanVideoEffect {
    fn repeated(&mut self) -> &mut RepeatedOutput {
        &mut self.repeated
    }

    fn frame_ref_failed(&self, code: i32) -> crate::error::Error {
        pp_error!(self, "av_frame_ref failed: {code}");
        VulkanVideoEffectError::FrameRef(code).into()
    }

    fn produce(
        &mut self,
        source: &Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        self.run(source)
    }
}

impl Element for VulkanVideoEffect {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::VulkanVideoEffect
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for VulkanVideoEffect {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for VulkanVideoEffect {
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::Vulkan)
                .with_layouts(crate::contract::PixelLayoutSet::BGRA),
        )
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Video(frame) => {
                self.refresh();
                if !self.enabled || self.params.is_identity() {
                    // Straight through: the same picture, not a copy of it.
                    return self.pad.push(MediaBuffer::Video(frame));
                }
                let output = self.transform(&frame)?;
                self.pad.push(MediaBuffer::Video(output))
            }
            MediaBuffer::Eos => self.pad.push(MediaBuffer::Eos),
            other => {
                let kind = other.kind();
                pp_error!(self, "unsupported buffer: {kind}");
                Err(VulkanVideoEffectError::UnsupportedBuffer(kind).into())
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

impl Drop for VulkanVideoEffect {
    fn drop(&mut self) {
        pp_info!(self, "dropped: releasing hw contexts");
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::options::apply;
    use super::*;
    use crate::{
        elements::{ColorCorrection, LumaKey, VulkanDownload, VulkanUpload},
        test_support::{capture, try_vulkan_device},
    };

    /// One BGRA Vulkan frame whose pixel at `x` is `pixels[x]`.
    fn row(device: &VulkanDevice, pixels: &[[u8; 4]]) -> MediaBuffer {
        let mut upload = VulkanUpload::new("upload", device);
        let uploaded = capture(&mut upload);
        let mut frame =
            ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, pixels.len() as u32, 1);
        frame.set_pts(Some(7));
        for (x, pixel) in pixels.iter().enumerate() {
            frame.data_mut(0)[x * 4..x * 4 + 4].copy_from_slice(pixel);
        }
        upload.consume(MediaBuffer::video(frame)).unwrap();
        uploaded.lock().unwrap().remove(0)
    }

    fn read_back(device: &VulkanDevice, buffer: MediaBuffer, width: usize) -> Vec<[u8; 4]> {
        let mut download = VulkanDownload::new("download", device);
        let received = capture(&mut download);
        download.consume(buffer).unwrap();
        let MediaBuffer::Video(frame) = received.lock().unwrap().remove(0) else {
            panic!("a picture");
        };
        (0..width)
            .map(|x| frame.data(0)[x * 4..x * 4 + 4].try_into().unwrap())
            .collect()
    }

    const PIXELS: [[u8; 4]; 6] = [
        [0, 0, 0, 255],
        [255, 255, 255, 255],
        [30, 140, 220, 255],
        [200, 60, 10, 160],
        [100, 110, 120, 255],
        [1, 254, 77, 200],
    ];

    /// Every effect agrees with the software element's own arithmetic to
    /// within one step a channel — the exponent is the GPU's `pow` — and the
    /// timestamp is carried through.
    #[test]
    fn every_effect_agrees_with_the_software_one() {
        let Some(device) = try_vulkan_device() else {
            return;
        };
        let effects = [
            VideoEffect::ColorCorrection(ColorCorrection {
                brightness: 0.1,
                contrast: 1.4,
                saturation: 0.6,
                hue_degrees: 35.0,
                gamma: 1.6,
                opacity: 0.8,
            }),
            VideoEffect::ColorCorrection(ColorCorrection {
                saturation: 0.0,
                ..ColorCorrection::default()
            }),
            VideoEffect::LumaKey(LumaKey {
                min: 0.25,
                min_smoothing: 0.1,
                max: 0.75,
                max_smoothing: 0.1,
            }),
        ];
        for effect in effects {
            let (mut element, _handle) = VulkanVideoEffect::new("effect", &device, effect).unwrap();
            let out = capture(&mut element);
            element.consume(row(&device, &PIXELS)).unwrap();
            let buffer = out.lock().unwrap().remove(0);
            let MediaBuffer::Video(frame) = &buffer else {
                panic!("a picture");
            };
            assert_eq!(frame.pts(), Some(7));
            let got = read_back(&device, buffer, PIXELS.len());
            for (pixel, got) in PIXELS.iter().zip(got) {
                let wanted = apply(&effect.params(), *pixel);
                for (channel, (got, wanted)) in got.iter().zip(wanted).enumerate() {
                    assert!(
                        got.abs_diff(wanted) <= 1,
                        "{effect:?}, {pixel:?} channel {channel}: got {got}, wanted {wanted}"
                    );
                }
            }
        }
    }

    /// An effect that changes nothing, and one turned off, hand the picture
    /// through as it is.
    #[test]
    fn a_neutral_or_disabled_effect_passes_the_picture_through() {
        let Some(device) = try_vulkan_device() else {
            return;
        };
        let (mut element, handle) = VulkanVideoEffect::new(
            "effect",
            &device,
            VideoEffect::ColorCorrection(ColorCorrection::default()),
        )
        .unwrap();
        let out = capture(&mut element);
        let input = row(&device, &PIXELS);
        let MediaBuffer::Video(sent) = &input else {
            unreachable!()
        };
        let sent = Arc::clone(sent);
        element.consume(input).unwrap();
        handle.set_effect(VideoEffect::ColorCorrection(ColorCorrection {
            saturation: 0.0,
            ..ColorCorrection::default()
        }));
        handle.set_enabled(false);
        element
            .consume(MediaBuffer::Video(Arc::clone(&sent)))
            .unwrap();
        let out = out.lock().unwrap();
        for buffer in out.iter() {
            let MediaBuffer::Video(frame) = buffer else {
                panic!("a picture");
            };
            assert!(Arc::ptr_eq(frame, &sent), "the same picture");
        }
    }
}
