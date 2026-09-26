use std::sync::Arc;

use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_debug, pp_error, pp_info};

use super::super::handle::{ChromaKeyControl, ChromaKeyHandle};
use super::super::options::{ChromaKeyOptions, feather_band};
use crate::{
    buffer::MediaBuffer,
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

const SHADER: &str = include_str!("../../../../shaders/vulkan/chroma_key.wgsl");

/// Errors specific to [`VulkanChromaKey`]. Converts into the crate-wide
/// `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum VulkanChromaKeyError {
    /// FFmpeg could not take a second reference to the keyed frame already
    /// in hand, which is how an unchanged input is answered.
    #[error("failed to reference the previous keyed frame (code {0})")]
    FrameRef(i32),

    /// The sink received something other than a decoded video frame.
    #[error("VulkanChromaKey only accepts Video buffers, got a {0}")]
    UnsupportedBuffer(&'static str),

    /// The frame is not a BGRA Vulkan frame of this element's own device.
    #[error("VulkanChromaKey {0}")]
    Frame(String),

    /// A Vulkan call this element made failed.
    #[error(transparent)]
    Vulkan(#[from] VulkanError),
}

impl From<BgraPassError> for VulkanChromaKeyError {
    fn from(error: BgraPassError) -> Self {
        match error {
            BgraPassError::Vulkan(error) => Self::Vulkan(error),
            other => Self::Frame(other.to_string()),
        }
    }
}

/// Keys a solid background colour out of a BGRA Vulkan frame into alpha, on
/// the GPU — the Vulkan member of the family whose software member is
/// [`SwChromaKey`](crate::elements::SwChromaKey), computing the same
/// normalized distance and the same feather band.
///
/// BGRA in, BGRA out, at whatever size the frames arrive in: only alpha is
/// written, multiplied by the key; the colour, PTS, duration and colour tags
/// pass through. Tuned while it runs through the [`ChromaKeyHandle`]
/// [`Self::new`] returns; a disabled one hands every frame straight through.
pub struct VulkanChromaKey {
    pp_log: PpLog,
    name: Arc<str>,
    pass: BgraPass,
    /// The last keyed frame and the picture it was made from — see
    /// [`RepeatedOutput`]. Forgotten when the settings change.
    repeated: RepeatedOutput,
    /// What this is keying by right now, refreshed from `control` once per
    /// frame.
    options: ChromaKeyOptions,
    control: Arc<ChromaKeyControl>,
    enabled: bool,
    pad: SrcPad,
}

impl VulkanChromaKey {
    /// `device` must be the same [`VulkanDevice`] every other Vulkan element
    /// in this pipeline was built from. Keying is per pixel, so every frame
    /// comes out the size it went in.
    pub fn new(
        name: impl Into<String>,
        device: &VulkanDevice,
        options: ChromaKeyOptions,
    ) -> std::result::Result<(Self, ChromaKeyHandle), VulkanChromaKeyError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::VulkanChromaKey, &name, None);
        let pass = BgraPass::new(device, SHADER, c"main", 48, PassInput::Bgra)?;
        let pad = SrcPad::with_contract(
            format!("{name}_src"),
            OutputContract::Fixed(
                PortContract::frame(MediaKind::VideoFrame, MemoryDomain::Vulkan)
                    .with_layouts(crate::contract::PixelLayoutSet::BGRA),
            ),
        );
        pp_info!(
            pp_log: &pp_log,
            "opened: BGRA, key_color={:?}, threshold={}, smoothing={}",
            options.method.key_color(),
            options.threshold,
            options.smoothing
        );
        let control = Arc::new(ChromaKeyControl::new(options));
        let handle = ChromaKeyHandle::new(control.clone());
        Ok((
            Self {
                name,
                pp_log,
                pass,
                repeated: RepeatedOutput::new(),
                options,
                control,
                enabled: true,
                pad,
            },
            handle,
        ))
    }

    /// Picks up whatever the handle has been set to, once per frame.
    fn refresh(&mut self) {
        let options = self.control.get();
        if options != self.options {
            pp_debug!(
                self,
                "retuned: key_color={:?}, threshold={}, smoothing={}",
                options.method.key_color(),
                options.threshold,
                options.smoothing
            );
            self.options = options;
            self.repeated.clear();
        }
        let enabled = self.control.enabled();
        if enabled != self.enabled {
            pp_debug!(self, "keying {}", if enabled { "on" } else { "off" });
            self.enabled = enabled;
            self.repeated.clear();
        }
    }

    fn key(
        &mut self,
        source: &ffmpeg::frame::Video,
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        let colour = self.options.method.key_color();
        let (band_low, inv_band_width) =
            feather_band(self.options.threshold, self.options.smoothing);
        let channel = |value: u8| (f32::from(value) / 255.0).to_ne_bytes();
        let words = [
            source.width().to_ne_bytes(),
            source.height().to_ne_bytes(),
            0u32.to_ne_bytes(),
            0u32.to_ne_bytes(),
            channel(colour.red),
            channel(colour.green),
            channel(colour.blue),
            band_low.to_ne_bytes(),
            inv_band_width.to_ne_bytes(),
            0f32.to_ne_bytes(),
            0f32.to_ne_bytes(),
            0f32.to_ne_bytes(),
        ];
        self.pass
            .run(source, &immediates(&words))
            .map_err(VulkanChromaKeyError::from)
            .inspect_err(|error| pp_error!(self, "{error}"))
            .map_err(Into::into)
    }
}

impl PerFrameTransform for VulkanChromaKey {
    fn repeated(&mut self) -> &mut RepeatedOutput {
        &mut self.repeated
    }

    fn frame_ref_failed(&self, code: i32) -> crate::error::Error {
        pp_error!(self, "av_frame_ref failed: {code}");
        VulkanChromaKeyError::FrameRef(code).into()
    }

    fn produce(
        &mut self,
        source: &Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        self.key(source)
    }
}

impl Element for VulkanChromaKey {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::VulkanChromaKey
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for VulkanChromaKey {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for VulkanChromaKey {
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
                if !self.enabled {
                    return self.pad.push(MediaBuffer::Video(frame));
                }
                let keyed = self.transform(&frame)?;
                self.pad.push(MediaBuffer::Video(keyed))
            }
            MediaBuffer::Eos => self.pad.push(MediaBuffer::Eos),
            other => {
                let kind = other.kind();
                pp_error!(self, "unsupported buffer: {kind}");
                Err(VulkanChromaKeyError::UnsupportedBuffer(kind).into())
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

impl Drop for VulkanChromaKey {
    fn drop(&mut self) {
        pp_info!(self, "dropped: releasing hw contexts");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        elements::{ChromaKeyMethod, VulkanDownload, VulkanUpload},
        test_support::{capture, try_vulkan_device},
    };

    fn row(device: &VulkanDevice, pixels: &[[u8; 4]]) -> MediaBuffer {
        let mut upload = VulkanUpload::new("upload", device);
        let uploaded = capture(&mut upload);
        let mut frame =
            ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, pixels.len() as u32, 1);
        frame.set_pts(Some(3));
        for (x, pixel) in pixels.iter().enumerate() {
            frame.data_mut(0)[x * 4..x * 4 + 4].copy_from_slice(pixel);
        }
        upload.consume(MediaBuffer::video(frame)).unwrap();
        uploaded.lock().unwrap().remove(0)
    }

    /// Green goes, what is far from it stays, the colour is untouched and
    /// alpha an earlier key cut is kept; turned off, the picture passes as
    /// it is.
    #[test]
    fn green_is_keyed_out_and_the_rest_kept() {
        let Some(device) = try_vulkan_device() else {
            return;
        };
        let (mut key, handle) = VulkanChromaKey::new(
            "key",
            &device,
            ChromaKeyOptions {
                method: ChromaKeyMethod::Green,
                threshold: 0.3,
                smoothing: 0.0,
            },
        )
        .unwrap();
        let out = capture(&mut key);
        // BGRA: green, red, half-transparent blue.
        let pixels = [[0, 255, 0, 255], [0, 0, 255, 255], [255, 0, 0, 128]];
        key.consume(row(&device, &pixels)).unwrap();
        let keyed = out.lock().unwrap().remove(0);

        let mut download = VulkanDownload::new("download", &device);
        let back = capture(&mut download);
        download.consume(keyed).unwrap();
        let MediaBuffer::Video(frame) = back.lock().unwrap().remove(0) else {
            panic!("a picture");
        };
        assert_eq!(frame.pts(), Some(3));
        let at = |x: usize| -> [u8; 4] { frame.data(0)[x * 4..x * 4 + 4].try_into().unwrap() };
        assert_eq!(at(0), [0, 255, 0, 0], "green keyed out, colour kept");
        assert_eq!(at(1), [0, 0, 255, 255], "red kept");
        assert_eq!(at(2), [255, 0, 0, 128], "an earlier cut kept");

        handle.set_enabled(false);
        let input = row(&device, &pixels);
        let MediaBuffer::Video(sent) = &input else {
            unreachable!()
        };
        let sent = Arc::clone(sent);
        key.consume(input).unwrap();
        let MediaBuffer::Video(passed) = out.lock().unwrap().remove(0) else {
            panic!("a picture");
        };
        assert!(Arc::ptr_eq(&passed, &sent), "off, the same picture");
    }
}
