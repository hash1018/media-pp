//! [`V4l2VirtualCamera`]: a pipeline's pictures as a camera other
//! applications open, through a v4l2loopback device.

use std::sync::Arc;

use crate::ffmpeg;
use crate::platform::linux::v4l2::{
    LoopbackError, LoopbackOutput, V4l2Device, list_loopback_devices,
};
use crate::pp_log::{PpLog, pp_info};
use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, PortContract},
    element::{Element, ElementType, Sink, element_pp_log},
    elements::filter::scaler::{is_rgb, matrix},
    error::Result,
    render::{SinkStage, sink_stage},
};

/// Shows what reaches it as a camera, through a [v4l2loopback] device:
/// every application that reads cameras through V4L2 — a browser, Zoom,
/// Discord, OBS Studio — lists it under the device's `card_label`.
///
/// It takes decoded video in system memory, any pixel format and size, and
/// writes each picture as I420 at the size it was made with, converted to
/// BT.709 limited range. Pictures arriving faster than the rate it was made
/// with are written as they come; the device hands readers the newest.
///
/// The camera exists while the element holds the device open: from
/// [`V4l2VirtualCamera::new`] until the element is dropped. A device loaded
/// with `exclusive_caps=1` — what every browser needs to list it — offers
/// itself as a camera only during that time.
///
/// # Requirements
///
/// The v4l2loopback module, loaded — for one,
/// `modprobe v4l2loopback exclusive_caps=1 card_label="My Camera"`, which
/// takes root. [`V4l2VirtualCamera::list_devices`] lists the devices there
/// are to write to, and answers none where the module is not loaded.
///
/// [v4l2loopback]: https://github.com/v4l2loopback/v4l2loopback
pub struct V4l2VirtualCamera(SinkStage<Writing>);

sink_stage!(V4l2VirtualCamera);

/// Why a [`V4l2VirtualCamera`] could not be made or could not take a buffer.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum V4l2VirtualCameraError {
    /// The device could not be opened — it is not there, or this user may
    /// not write to it.
    #[error("could not open {device}: {source}")]
    Open {
        /// The device node.
        device: String,
        /// What opening it answered.
        source: std::io::Error,
    },
    /// The device is a video node, but not a v4l2loopback one.
    #[error("{device} is not a v4l2loopback device")]
    NotLoopback {
        /// The device node.
        device: String,
    },
    /// Another program already writes to the device.
    #[error("{device} is already shown by another program")]
    InUse {
        /// The device node.
        device: String,
    },
    /// The device would not take I420 pictures of this size.
    #[error("{device} refused {width}x{height} I420 pictures: {source}")]
    Format {
        /// The device node.
        device: String,
        /// The size asked for.
        width: u32,
        /// The size asked for.
        height: u32,
        /// What the device answered.
        source: std::io::Error,
    },
    /// I420 halves the picture both ways for its colour, so its size must
    /// be even.
    #[error("a camera picture's width and height must be even, not {width}x{height}")]
    OddSize {
        /// The size asked for.
        width: u32,
        /// The size asked for.
        height: u32,
    },
    /// Writing a picture to the device failed.
    #[error("could not write a picture to the camera: {0}")]
    Write(std::io::Error),
    /// Not a video frame in system memory.
    #[error("V4l2VirtualCamera takes video frames in system memory, got {0}")]
    UnsupportedBuffer(&'static str),
    /// Converting a picture to the camera's size and format failed.
    #[error("could not convert a picture for the camera: {0}")]
    Convert(#[from] ffmpeg::Error),
}

/// What a [`V4l2VirtualCamera`] does with each picture.
struct Writing {
    name: Arc<str>,
    pp_log: PpLog,
    output: LoopbackOutput,
    size: (u32, u32),
    conversion: Option<Conversion>,
    /// One picture laid out as the device takes it, reused for each.
    packed: Vec<u8>,
}

/// One picture shape's conversion to the camera's, and the frame it writes.
struct Conversion {
    from: (
        ffmpeg::format::Pixel,
        u32,
        u32,
        ffmpeg::color::Space,
        ffmpeg::color::Range,
    ),
    context: ffmpeg::software::scaling::Context,
    frame: ffmpeg::frame::Video,
}

// SAFETY: the scaling context and frame are heap allocations owned solely
// by this conversion, used only by the one thread rendering at a time,
// exactly as `SwScaler` holds its own.
unsafe impl Send for Conversion {}

impl V4l2VirtualCamera {
    /// Opens `device` — a v4l2loopback node, from
    /// [`Self::list_devices`] — and makes it a camera of `width` by `height`
    /// pictures, `frame_rate` a second, for as long as the element lives.
    pub fn new(
        name: impl Into<String>,
        device: &str,
        width: u32,
        height: u32,
        frame_rate: ffmpeg::Rational,
    ) -> Result<Self> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::V4l2VirtualCamera, &name, None);
        if width == 0 || height == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Err(V4l2VirtualCameraError::OddSize { width, height }.into());
        }
        let output = LoopbackOutput::open(device, width, height, frame_rate).map_err(|error| {
            let device = device.to_owned();
            match error {
                LoopbackError::Open(source) => V4l2VirtualCameraError::Open { device, source },
                LoopbackError::NotLoopback => V4l2VirtualCameraError::NotLoopback { device },
                LoopbackError::InUse => V4l2VirtualCameraError::InUse { device },
                LoopbackError::Format(source) => V4l2VirtualCameraError::Format {
                    device,
                    width,
                    height,
                    source,
                },
            }
        })?;
        pp_info!(
            pp_log: &pp_log,
            "created: {device} shows {width}x{height} I420 at {}/{} fps",
            frame_rate.numerator(),
            frame_rate.denominator()
        );
        Ok(Self::writing(name, pp_log, output, (width, height)))
    }

    /// Every v4l2loopback device there is to write to now, in node order —
    /// none where the module is not loaded. One another program already
    /// writes to is left out where the module was loaded with
    /// `exclusive_caps=1`.
    pub fn list_devices() -> std::io::Result<Vec<V4l2Device>> {
        list_loopback_devices()
    }

    fn writing(name: Arc<str>, pp_log: PpLog, output: LoopbackOutput, size: (u32, u32)) -> Self {
        let packed = vec![0; output.frame_bytes()];
        Self(SinkStage::new(Writing {
            name,
            pp_log,
            output,
            size,
            conversion: None,
            packed,
        }))
    }

    /// The element writing into `file` rather than a device, for a test to
    /// read back what a reader would.
    #[cfg(test)]
    fn for_file(file: std::fs::File, width: u32, height: u32) -> Self {
        let name: Arc<str> = "camera".into();
        let pp_log = element_pp_log(ElementType::V4l2VirtualCamera, &name, None);
        Self::writing(
            name,
            pp_log,
            LoopbackOutput::for_file(file, width, height),
            (width, height),
        )
    }
}

impl Writing {
    /// Converts `frame` to I420 at the camera's size, reusing the conversion
    /// while the pictures keep their shape.
    fn convert(&mut self, frame: &ffmpeg::frame::Video) -> Result<&ffmpeg::frame::Video> {
        let from = (
            frame.format(),
            frame.width(),
            frame.height(),
            frame.color_space(),
            frame.color_range(),
        );
        if self
            .conversion
            .as_ref()
            .is_none_or(|conversion| conversion.from != from)
        {
            self.conversion = Some(conversion(from, self.size)?);
        }
        let conversion = self.conversion.as_mut().expect("made above");
        conversion
            .context
            .run(frame, &mut conversion.frame)
            .map_err(V4l2VirtualCameraError::Convert)?;
        Ok(&conversion.frame)
    }
}

/// A conversion of pictures shaped `from` to BT.709 limited-range I420 at
/// `to` — what the device's format says its pictures are.
fn conversion(
    from: (
        ffmpeg::format::Pixel,
        u32,
        u32,
        ffmpeg::color::Space,
        ffmpeg::color::Range,
    ),
    to: (u32, u32),
) -> std::result::Result<Conversion, V4l2VirtualCameraError> {
    let (format, width, height, space, range) = from;
    let mut context = ffmpeg::software::scaling::Context::get(
        format,
        width,
        height,
        ffmpeg::format::Pixel::YUV420P,
        to.0,
        to.1,
        ffmpeg::software::scaling::Flags::BILINEAR,
    )?;
    // SAFETY: `context` is a live `SwsContext` this conversion owns, and the
    // tables are swscale's own static ones.
    unsafe {
        let source = ffmpeg::ffi::sws_getCoefficients(matrix(space));
        let destination = ffmpeg::ffi::sws_getCoefficients(ffmpeg::ffi::SWS_CS_ITU709);
        let source_full = is_rgb(format) || range == ffmpeg::color::Range::JPEG;
        ffmpeg::ffi::sws_setColorspaceDetails(
            context.as_mut_ptr(),
            source,
            i32::from(source_full),
            destination,
            0,
            0,
            1 << 16,
            1 << 16,
        );
    }
    Ok(Conversion {
        from,
        context,
        frame: ffmpeg::frame::Video::new(ffmpeg::format::Pixel::YUV420P, to.0, to.1),
    })
}

/// Whether `format` is a hardware frame's, whose pixels are not in it.
fn is_hardware(format: ffmpeg::format::Pixel) -> bool {
    // SAFETY: a lookup in libavutil's static table of descriptors.
    unsafe {
        let descriptor = ffmpeg::ffi::av_pix_fmt_desc_get(format.into());
        !descriptor.is_null()
            && (*descriptor).flags & (ffmpeg::ffi::AV_PIX_FMT_FLAG_HWACCEL as u64) != 0
    }
}

/// Packs `frame`'s three planes, each `linesize` apart, into `into`: the
/// luma plane, then the two quarter-size chroma planes, rows end to end.
fn pack_i420(frame: &ffmpeg::frame::Video, into: &mut [u8]) {
    let (width, height) = (frame.width() as usize, frame.height() as usize);
    let (luma, chroma) = into.split_at_mut(width * height);
    let (blue, red) = chroma.split_at_mut((width / 2) * (height / 2));
    for (plane, out, row_bytes) in [(0, luma, width), (1, blue, width / 2), (2, red, width / 2)] {
        let stride = frame.stride(plane);
        let data = frame.data(plane);
        for (row, out) in out.chunks_exact_mut(row_bytes).enumerate() {
            out.copy_from_slice(&data[row * stride..row * stride + row_bytes]);
        }
    }
}

impl Element for Writing {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::V4l2VirtualCamera
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Sink for Writing {
    /// Decoded video in system memory, any pixel layout: each picture is
    /// converted to the camera's.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::frame(
            MediaKind::VideoFrame,
            MemoryDomain::System,
        ))
    }

    fn render(&mut self, buf: MediaBuffer) -> Result<()> {
        let MediaBuffer::Video(frame) = &buf else {
            return Err(V4l2VirtualCameraError::UnsupportedBuffer(buf.kind()).into());
        };
        if is_hardware(frame.format()) {
            return Err(V4l2VirtualCameraError::UnsupportedBuffer("a hardware video frame").into());
        }
        self.convert(frame)?;
        let converted = &self.conversion.as_ref().expect("converted above").frame;
        pack_i420(converted, &mut self.packed);
        self.output
            .write_frame(&self.packed)
            .map_err(V4l2VirtualCameraError::Write)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Seek};

    use super::*;
    use crate::element::RawSink;

    /// A flat picture in `format`, `width` by `height`.
    fn picture(format: ffmpeg::format::Pixel, width: u32, height: u32, fill: u8) -> MediaBuffer {
        let mut frame = ffmpeg::frame::Video::new(format, width, height);
        for plane in 0..frame.planes() {
            frame.data_mut(plane).fill(fill);
        }
        MediaBuffer::video(frame)
    }

    /// What the element wrote into `file`, from the start.
    fn written(file: &mut std::fs::File) -> Vec<u8> {
        let mut bytes = Vec::new();
        file.rewind().expect("rewind");
        file.read_to_end(&mut bytes).expect("read back");
        bytes
    }

    /// A file of this test's own, already unlinked, so nothing is left
    /// behind whatever the test does.
    fn scratch_file() -> std::fs::File {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "media-pp-v4l2-virtual-camera-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .expect("a file to write into");
        std::fs::remove_file(&path).expect("unlinked");
        file
    }

    #[test]
    fn each_picture_is_written_whole_as_i420_at_the_cameras_size() {
        let mut file = scratch_file();
        let mut element = V4l2VirtualCamera::for_file(file.try_clone().expect("clone"), 640, 360);
        for _ in 0..2 {
            element
                .consume(picture(ffmpeg::format::Pixel::NV12, 1920, 1080, 128))
                .expect("render");
        }
        let bytes = written(&mut file);
        assert_eq!(bytes.len(), 2 * 640 * 360 * 3 / 2, "two whole pictures");
        // Within swscale's rounding of a 1920 to 640 shrink.
        assert!(
            bytes.iter().all(|&byte| byte.abs_diff(128) <= 2),
            "a flat grey stays flat grey"
        );
    }

    #[test]
    fn a_bgra_picture_is_converted_to_bt709_limited_range() {
        let mut file = scratch_file();
        let mut element = V4l2VirtualCamera::for_file(file.try_clone().expect("clone"), 320, 240);
        element
            .consume(picture(ffmpeg::format::Pixel::BGRA, 320, 240, 255))
            .expect("render");
        let bytes = written(&mut file);
        let luma = 320 * 240;
        assert!(
            bytes[..luma].iter().all(|&byte| byte == 235),
            "white is BT.709 limited white"
        );
        assert!(
            bytes[luma..].iter().all(|&byte| byte.abs_diff(128) <= 1),
            "and neutral"
        );
    }

    #[test]
    fn a_picture_changing_shape_is_still_written_at_the_cameras_size() {
        let mut file = scratch_file();
        let mut element = V4l2VirtualCamera::for_file(file.try_clone().expect("clone"), 320, 240);
        element
            .consume(picture(ffmpeg::format::Pixel::YUV420P, 640, 480, 16))
            .expect("render");
        element
            .consume(picture(ffmpeg::format::Pixel::RGB24, 1280, 720, 0))
            .expect("render");
        assert_eq!(written(&mut file).len(), 2 * 320 * 240 * 3 / 2);
    }

    #[test]
    fn a_sound_frame_is_refused_by_name() {
        let mut element = V4l2VirtualCamera::for_file(scratch_file(), 320, 240);
        let sound = ffmpeg::frame::Audio::new(
            ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed),
            16,
            ffmpeg::ChannelLayout::STEREO,
        );
        let error = element
            .consume(MediaBuffer::Audio(Arc::new(sound)))
            .expect_err("sound is not a picture");
        assert!(error.to_string().contains("system memory"), "{error}");
    }

    #[test]
    fn an_odd_size_is_refused_before_any_device_is_opened() {
        let error = V4l2VirtualCamera::new(
            "camera",
            "/dev/null",
            641,
            360,
            ffmpeg::Rational::new(30, 1),
        )
        .err()
        .expect("I420 needs an even size");
        assert!(error.to_string().contains("even"), "{error}");
    }

    #[test]
    fn a_node_that_is_not_a_loopback_device_is_refused() {
        let error = V4l2VirtualCamera::new(
            "camera",
            "/dev/null",
            640,
            360,
            ffmpeg::Rational::new(30, 1),
        )
        .err()
        .expect("/dev/null is no camera");
        // `/dev/null` opens, and then cannot say what it is.
        assert!(error.to_string().contains("/dev/null"), "{error}");
    }

    /// The real thing, where this machine has a loopback device free: the
    /// device takes the format, and a picture written is one a reader of
    /// the node could open. Skipped, saying so, where there is none.
    #[test]
    fn a_loopback_device_takes_the_format_and_a_picture() {
        let Some(device) = V4l2VirtualCamera::list_devices()
            .expect("/dev is readable")
            .into_iter()
            .next()
        else {
            eprintln!("skipping: this machine has no v4l2loopback device free");
            return;
        };
        let mut element =
            V4l2VirtualCamera::new("camera", &device.id, 640, 360, ffmpeg::Rational::new(30, 1))
                .expect("a free loopback device takes I420");
        element
            .consume(picture(ffmpeg::format::Pixel::NV12, 1280, 720, 128))
            .expect("a picture is written");
    }
}
