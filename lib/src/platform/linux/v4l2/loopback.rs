//! The writing end of a v4l2loopback device.
//!
//! [v4l2loopback] is a kernel module that makes video nodes nobody's
//! hardware stands behind: what one program writes into the node, every
//! other program reads out of it as a camera. Writing takes three steps,
//! all here: say the picture's format on the node's *output* side
//! (`VIDIOC_S_FMT`), say its rate (`VIDIOC_S_PARM`), and `write` each
//! picture whole. FFmpeg's `v4l2` output muxer does the same but only for
//! a fixed stream it encodes itself, so the node is written directly — with
//! the request numbers made as [`super::ioctl`] explains.
//!
//! The node keeps the format for as long as the writer holds it open; once
//! the writer closes it, a module loaded with `exclusive_caps=1` stops
//! offering a camera at all, which is what every application's camera list
//! then shows.
//!
//! [v4l2loopback]: https://github.com/v4l2loopback/v4l2loopback

use std::io::Write;

use ffmpeg_next as ffmpeg;

use super::V4l2Device;
use super::ioctl::{READ_WRITE, call, query_capability, request, video_nodes};

/// The driver name every v4l2loopback node reports.
const LOOPBACK_DRIVER: &str = "v4l2 loopback";

/// Every loopback node a program can write into now, in node order.
///
/// A node is left out once something else writes to it, where the module
/// was loaded with `exclusive_caps=1`: such a node offers its writing end
/// only until a writer has set its format.
pub(crate) fn list_loopback_devices() -> std::io::Result<Vec<V4l2Device>> {
    Ok(video_nodes()?
        .into_iter()
        .filter_map(|path| {
            let file = std::fs::File::open(&path).ok()?;
            let capability = query_capability(&file).ok()?;
            let loopback = capability.driver().as_deref() == Some(LOOPBACK_DRIVER);
            (loopback && capability.node_caps() & CAP_VIDEO_OUTPUT != 0).then(|| V4l2Device {
                name: capability
                    .card()
                    .unwrap_or_else(|| path.display().to_string()),
                id: path.display().to_string(),
            })
        })
        .collect())
}

/// Why a loopback node could not be opened for writing.
#[derive(Debug)]
pub(crate) enum LoopbackError {
    /// The node could not be opened, or did not answer what it is.
    Open(std::io::Error),
    /// The node is not a v4l2loopback node.
    NotLoopback,
    /// Another program already writes to it.
    InUse,
    /// The node refused the picture's format.
    Format(std::io::Error),
}

/// A loopback node opened for writing, its format set: I420 — V4L2's
/// `YU12` — at one size, BT.709 limited range.
///
/// I420 because it is what every application reading a camera through
/// V4L2 takes — browsers, conferencing clients, OBS Studio's own virtual
/// camera writes it — at half the bytes of a packed 4:2:2 picture.
pub(crate) struct LoopbackOutput {
    file: std::fs::File,
    frame_bytes: usize,
}

impl LoopbackOutput {
    /// Opens `device` and sets its writing end to I420 at `width` by
    /// `height`, `frame_rate` pictures a second.
    ///
    /// `width` and `height` must be even, which I420's halved chroma needs;
    /// the caller checks.
    pub(crate) fn open(
        device: &str,
        width: u32,
        height: u32,
        frame_rate: ffmpeg::Rational,
    ) -> Result<Self, LoopbackError> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(device)
            .map_err(LoopbackError::Open)?;
        let capability = query_capability(&file).map_err(LoopbackError::Open)?;
        if capability.driver().as_deref() != Some(LOOPBACK_DRIVER) {
            return Err(LoopbackError::NotLoopback);
        }
        if capability.node_caps() & CAP_VIDEO_OUTPUT == 0 {
            return Err(LoopbackError::InUse);
        }

        let frame_bytes = i420_bytes(width, height);
        let mut format = Format {
            kind: BUF_TYPE_VIDEO_OUTPUT,
            pix: PixFormat {
                width,
                height,
                pixel_format: u32::from_le_bytes(*b"YU12"),
                field: FIELD_NONE,
                bytes_per_line: width,
                size_image: frame_bytes as u32,
                colorspace: COLORSPACE_REC709,
                ycbcr_enc: YCBCR_ENC_709,
                quantization: QUANTIZATION_LIM_RANGE,
                xfer_func: XFER_FUNC_709,
                ..PixFormat::default()
            },
            ..Format::default()
        };
        // SAFETY: as `query_capability`, for this ioctl's own struct.
        call(&file, VIDIOC_S_FMT, &mut format).map_err(LoopbackError::Format)?;
        if format.pix.width != width
            || format.pix.height != height
            || format.pix.pixel_format != u32::from_le_bytes(*b"YU12")
        {
            return Err(LoopbackError::Format(std::io::Error::other(format!(
                "the device answered {}x{} rather than {width}x{height} I420",
                format.pix.width, format.pix.height
            ))));
        }

        // The rate the readers are told, which a reader that asks before
        // opening — a browser does — offers as the camera's. A module that
        // does not take it still shows each picture as it is written, so a
        // refusal is not a failure.
        let mut parameters = StreamParm {
            kind: BUF_TYPE_VIDEO_OUTPUT,
            output: OutputParm {
                time_per_frame: [
                    frame_rate.denominator().max(1) as u32,
                    frame_rate.numerator().max(1) as u32,
                ],
                ..OutputParm::default()
            },
            ..StreamParm::default()
        };
        // SAFETY: as `query_capability`, for this ioctl's own struct.
        let _ = call(&file, VIDIOC_S_PARM, &mut parameters);

        Ok(Self { file, frame_bytes })
    }

    /// Writes into any file rather than a device, for a test to read back.
    #[cfg(test)]
    pub(crate) fn for_file(file: std::fs::File, width: u32, height: u32) -> Self {
        Self {
            file,
            frame_bytes: i420_bytes(width, height),
        }
    }

    /// How many bytes one picture is.
    pub(crate) fn frame_bytes(&self) -> usize {
        self.frame_bytes
    }

    /// Writes one whole picture, `frame_bytes` long. A loopback node takes a
    /// picture in one `write` or not at all.
    pub(crate) fn write_frame(&mut self, frame: &[u8]) -> std::io::Result<()> {
        debug_assert_eq!(frame.len(), self.frame_bytes);
        let written = self.file.write(frame)?;
        if written != frame.len() {
            return Err(std::io::Error::other(format!(
                "the device took {written} of a picture's {} bytes",
                frame.len()
            )));
        }
        Ok(())
    }
}

/// How many bytes an I420 picture of this size is: a full-size luma plane
/// and two quarter-size chroma planes.
pub(crate) fn i420_bytes(width: u32, height: u32) -> usize {
    let (width, height) = (width as usize, height as usize);
    width * height + 2 * (width / 2) * (height / 2)
}

const VIDIOC_S_FMT: libc::c_ulong = request(READ_WRITE, 5, size_of::<Format>());
const VIDIOC_S_PARM: libc::c_ulong = request(READ_WRITE, 22, size_of::<StreamParm>());

const BUF_TYPE_VIDEO_OUTPUT: u32 = 2;
/// The node takes video written to it: a loopback device's writing end.
const CAP_VIDEO_OUTPUT: u32 = 0x0000_0002;
const FIELD_NONE: u32 = 1;
const COLORSPACE_REC709: u32 = 3;
const YCBCR_ENC_709: u32 = 2;
const QUANTIZATION_LIM_RANGE: u32 = 2;
const XFER_FUNC_709: u32 = 1;

/// `struct v4l2_format`, with only its `pix` member modelled.
///
/// The union after `type` holds pointers in one of its members
/// (`v4l2_window`), so on a 64-bit kernel it is aligned to eight bytes and
/// starts four bytes after `type`; on a 32-bit one it starts straight after.
/// The union is 200 bytes either way.
#[repr(C)]
#[derive(Default)]
struct Format {
    kind: u32,
    #[cfg(target_pointer_width = "64")]
    union_alignment: u32,
    pix: PixFormat,
    rest_of_union: RestOfUnion,
}

/// What follows `v4l2_pix_format` in `v4l2_format`'s 200-byte union.
#[repr(C)]
struct RestOfUnion([u8; 200 - size_of::<PixFormat>()]);

impl Default for RestOfUnion {
    fn default() -> Self {
        Self([0; 200 - size_of::<PixFormat>()])
    }
}

/// `struct v4l2_pix_format`.
#[repr(C)]
#[derive(Default)]
struct PixFormat {
    width: u32,
    height: u32,
    pixel_format: u32,
    field: u32,
    bytes_per_line: u32,
    size_image: u32,
    colorspace: u32,
    private: u32,
    flags: u32,
    ycbcr_enc: u32,
    quantization: u32,
    xfer_func: u32,
}

/// `struct v4l2_streamparm`, with only its output member modelled. No
/// member of its union holds a pointer, so it follows `type` directly.
#[repr(C)]
#[derive(Default)]
struct StreamParm {
    kind: u32,
    output: OutputParm,
    rest_of_union: RestOfParm,
}

/// What follows `v4l2_outputparm` in `v4l2_streamparm`'s 200-byte union.
#[repr(C)]
struct RestOfParm([u8; 200 - size_of::<OutputParm>()]);

impl Default for RestOfParm {
    fn default() -> Self {
        Self([0; 200 - size_of::<OutputParm>()])
    }
}

/// `struct v4l2_outputparm`. `time_per_frame` is a `v4l2_fract`: seconds
/// per picture, numerator then denominator.
#[repr(C)]
#[derive(Default)]
struct OutputParm {
    capability: u32,
    output_mode: u32,
    time_per_frame: [u32; 2],
    extended_mode: u32,
    write_buffers: u32,
    reserved: [u32; 4],
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The numbers computed here against the ones the kernel's headers
    /// publish for this machine's word size — see [`super::super::ioctl`]
    /// for why that checks every struct's layout at once.
    #[test]
    fn the_request_numbers_match_the_kernels() {
        #[cfg(target_pointer_width = "64")]
        assert_eq!(VIDIOC_S_FMT, 0xC0D0_5605);
        #[cfg(target_pointer_width = "32")]
        assert_eq!(VIDIOC_S_FMT, 0xC0CC_5605);
        assert_eq!(VIDIOC_S_PARM, 0xC0CC_5616);
    }

    #[test]
    fn an_i420_picture_is_one_and_a_half_bytes_a_pixel() {
        assert_eq!(i420_bytes(1920, 1080), 1920 * 1080 * 3 / 2);
        assert_eq!(i420_bytes(640, 360), 640 * 360 * 3 / 2);
    }

    /// Whatever is listed is a loopback node that says what it is called;
    /// a machine without the module lists nothing, which is also right.
    #[test]
    fn every_loopback_device_listed_names_itself() {
        for device in list_loopback_devices().expect("/dev is readable") {
            assert!(device.id.starts_with("/dev/video"), "{device:?}");
            assert!(!device.name.is_empty(), "{device:?}");
        }
    }
}
