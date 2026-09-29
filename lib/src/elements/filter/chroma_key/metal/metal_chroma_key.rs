use std::sync::Arc;

use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_debug, pp_error, pp_info};

use super::super::handle::{ChromaKeyControl, ChromaKeyHandle};
use super::super::options::{ChromaKeyOptions, feather_band};
use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    element::{Element, ElementType, Output, Transform, element_pp_log},
    elements::VideoToolboxDevice,
    error::Result,
    platform::macos::{
        metal::MetalError,
        metal_pass::{MetalPass, MetalPassError, PassInput, parameters},
    },
    pool::UnboundObjectPoolRef,
    repeat::{PerFrameTransform, RepeatedOutput},
    transform::{TransformStage, transform_filter},
};

const SHADER: &str = include_str!("../../../../shaders/metal/chroma_key.metal");

/// Errors specific to [`MetalChromaKey`]. Converts into the crate-wide
/// `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum MetalChromaKeyError {
    /// FFmpeg could not take a second reference to the keyed frame already
    /// in hand, which is how an unchanged input is answered.
    #[error("failed to reference the previous keyed frame (code {0})")]
    FrameRef(i32),

    /// The sink received something other than a decoded video frame.
    #[error("MetalChromaKey only accepts Video buffers, got a {0}")]
    UnsupportedBuffer(&'static str),

    /// The frame is not a BGRA VideoToolbox frame.
    #[error("MetalChromaKey {0}")]
    Frame(String),

    /// A Metal call this element made failed.
    #[error(transparent)]
    Metal(#[from] MetalError),
}

impl From<MetalPassError> for MetalChromaKeyError {
    fn from(error: MetalPassError) -> Self {
        match error {
            MetalPassError::Metal(error) => Self::Metal(error),
            other => Self::Frame(other.to_string()),
        }
    }
}

/// Keys a solid background colour out of a BGRA VideoToolbox frame into
/// alpha, on the GPU with Metal — the Metal member of the family whose
/// software member is
/// [`SwChromaKey`](crate::elements::SwChromaKey), computing the same
/// normalized distance and the same feather band.
///
/// BGRA in, BGRA out, at whatever size the frames arrive in: only alpha is
/// written, multiplied by the key; the colour, PTS, duration and colour tags
/// pass through. Tuned while it runs through the [`ChromaKeyHandle`]
/// [`Self::new`] returns; a disabled one hands every frame straight through.
pub struct MetalChromaKey(TransformStage<Keying>);

transform_filter!(MetalChromaKey);

/// What a [`MetalChromaKey`] does to each frame: all of its work, which the
/// framework makes the filter.
struct Keying {
    pp_log: PpLog,
    name: Arc<str>,
    pass: MetalPass,
    /// The last keyed frame and the picture it was made from — see
    /// [`RepeatedOutput`]. Forgotten when the settings change.
    repeated: RepeatedOutput,
    /// What this is keying by right now, refreshed from `control` once per
    /// frame.
    options: ChromaKeyOptions,
    control: Arc<ChromaKeyControl>,
    enabled: bool,
}

impl MetalChromaKey {
    /// Output frames are made on `device`; what it takes may come from any,
    /// since a pixel buffer belongs to none. Keying is per pixel, so every
    /// frame comes out the size it went in.
    pub fn new(
        name: impl Into<String>,
        device: &VideoToolboxDevice,
        options: ChromaKeyOptions,
    ) -> std::result::Result<(Self, ChromaKeyHandle), MetalChromaKeyError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::MetalChromaKey, &name, None);
        let pass = MetalPass::new(device, SHADER, "key", PassInput::Bgra)?;
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
            Self(TransformStage::new(Keying {
                name,
                pp_log,
                pass,
                repeated: RepeatedOutput::new(),
                options,
                control,
                enabled: true,
            })),
            handle,
        ))
    }
}

impl Keying {
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
            .run(source, &parameters(&words))
            .map_err(MetalChromaKeyError::from)
            .inspect_err(|error| pp_error!(self, "{error}"))
            .map_err(Into::into)
    }
}

impl PerFrameTransform for Keying {
    fn repeated(&mut self) -> &mut RepeatedOutput {
        &mut self.repeated
    }

    fn frame_ref_failed(&self, code: i32) -> crate::error::Error {
        pp_error!(self, "av_frame_ref failed: {code}");
        MetalChromaKeyError::FrameRef(code).into()
    }

    fn produce(
        &mut self,
        source: &Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        self.key(source)
    }
}

impl Element for Keying {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::MetalChromaKey
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Transform for Keying {
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::VideoToolbox)
                .with_layouts(crate::contract::PixelLayoutSet::BGRA),
        )
    }

    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        match buf {
            MediaBuffer::Video(frame) => {
                self.refresh();
                if !self.enabled {
                    out.push(MediaBuffer::Video(frame));
                    return Ok(());
                }
                let keyed = PerFrameTransform::transform(self, &frame)?;
                out.push(MediaBuffer::Video(keyed));
                Ok(())
            }
            other => {
                let kind = other.kind();
                pp_error!(self, "unsupported buffer: {kind}");
                Err(MetalChromaKeyError::UnsupportedBuffer(kind).into())
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

impl Drop for Keying {
    fn drop(&mut self) {
        pp_info!(self, "dropped: releasing hw contexts");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::Sink;
    use crate::{
        elements::{ChromaKeyMethod, VideoToolboxDownload, VideoToolboxUpload},
        test_support::{capture, try_videotoolbox_device},
    };

    fn row(device: &VideoToolboxDevice, pixels: &[[u8; 4]]) -> MediaBuffer {
        let mut upload = VideoToolboxUpload::new("upload", device);
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
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let (mut key, handle) = MetalChromaKey::new(
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

        let mut download = VideoToolboxDownload::new("download");
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
