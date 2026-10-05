//! Camera enumeration through V4L2's own ioctls.
//!
//! Only the read-only half of the interface is here: which devices exist,
//! what each is called, and which modes it offers. Capture itself goes
//! through FFmpeg's `video4linux2` demuxer, which already owns the buffer
//! negotiation, the mmap ring and the dequeue loop — see
//! [`crate::elements::V4l2CaptureSource`]. Asking a device what it offers is
//! the one thing that demuxer has no call for, so it is asked directly —
//! with the request numbers made as [`super::ioctl`] explains.

use std::os::fd::OwnedFd;

use ffmpeg_next as ffmpeg;

use super::V4l2Device;
use super::ioctl::{READ_WRITE, call, query_capability, request, video_nodes};

/// One picture shape a camera offers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct V4l2CaptureFormat {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Frames per second, as the fraction the device reports rather than a
    /// rounded number: 30000/1001 is not 30, and a camera that offers both
    /// has two modes rather than one.
    pub frame_rate: ffmpeg::Rational,
}

/// Everything that answers to a video capture node, in node order.
///
/// A camera commonly has more than one node — a second for metadata, a third
/// for its still-image path — and only the ones that say they capture video
/// are offered.
pub fn list_devices() -> std::io::Result<Vec<V4l2Device>> {
    Ok(video_nodes()?
        .into_iter()
        .filter_map(|path| {
            let file = std::fs::File::open(&path).ok()?;
            let capability = query_capability(&file).ok()?;
            (capability.node_caps() & CAP_VIDEO_CAPTURE != 0).then(|| V4l2Device {
                name: capability
                    .card()
                    .unwrap_or_else(|| path.display().to_string()),
                id: path.display().to_string(),
            })
        })
        .collect())
}

/// Which of the camera's pixel formats offers this mode, as the name
/// FFmpeg's `video4linux2` demuxer knows it by.
///
/// Necessary because one geometry is not one mode: a camera commonly offers
/// 640x480 both raw and compressed, and 1280x720 only compressed. Asking the
/// demuxer for a size without saying which format carries it lets it pick the
/// other one, and the driver then answers with a size nobody asked for — which
/// is what an unnamed format looked like in practice: 320x240 requested,
/// 480x320 delivered.
///
/// Raw is preferred where both carry the mode, because a raw frame skips a
/// JPEG decode per picture. `None` where nothing offers it, and where the
/// format is one this does not have a name for — in both cases the caller
/// leaves the choice to the demuxer, which is what it did before.
pub fn format_name_for(
    device: &str,
    width: u32,
    height: u32,
    frame_rate: ffmpeg::Rational,
) -> Option<&'static str> {
    let file = std::fs::File::open(device).ok()?;
    let mut compressed = None;
    for pixel_format in pixel_formats(&file) {
        let offered =
            frame_sizes(&file, pixel_format)
                .into_iter()
                .any(|(offered_width, offered_height)| {
                    offered_width == width && offered_height == height
                })
                && frame_rates(&file, pixel_format, width, height).contains(&frame_rate);
        if !offered {
            continue;
        }
        match demuxer_name(pixel_format) {
            // Raw, and the first one wins: nothing to decode.
            Some(name) if !is_compressed(pixel_format) => return Some(name),
            Some(name) => compressed = compressed.or(Some(name)),
            None => {}
        }
    }
    compressed
}

/// What the demuxer calls a V4L2 pixel format.
///
/// The ones a USB camera actually offers. A format with no name here is one
/// the caller leaves to the demuxer rather than one it refuses: an unusual
/// camera should still open.
fn demuxer_name(pixel_format: u32) -> Option<&'static str> {
    Some(match &pixel_format.to_le_bytes() {
        b"YUYV" => "yuyv422",
        b"UYVY" => "uyvy422",
        b"NV12" => "nv12",
        b"YU12" => "yuv420p",
        b"RGB3" => "rgb24",
        b"BGR3" => "bgr24",
        b"MJPG" => "mjpeg",
        _ => return None,
    })
}

/// Whether a frame in this format has to be decoded before it is a picture.
fn is_compressed(pixel_format: u32) -> bool {
    matches!(&pixel_format.to_le_bytes(), b"MJPG" | b"JPEG" | b"H264")
}

/// Every discrete mode a camera offers, best first.
///
/// Best is largest, then fastest: a picker's first row is the one most people
/// want, and a camera lists its modes in whatever order its firmware happens
/// to.
///
/// Continuous and stepwise ranges are left out. A camera that describes its
/// sizes as a range rather than a list has no modes to choose between, and
/// inventing some would offer shapes it was never asked about.
pub fn list_formats(device: &str) -> std::io::Result<Vec<V4l2CaptureFormat>> {
    let file = std::fs::File::open(device)?;
    let mut formats = Vec::new();
    for pixel_format in pixel_formats(&file) {
        for (width, height) in frame_sizes(&file, pixel_format) {
            for framerate in frame_rates(&file, pixel_format, width, height) {
                let mode = V4l2CaptureFormat {
                    width,
                    height,
                    frame_rate: framerate,
                };
                // Two pixel formats commonly offer the same shape — a raw one
                // and a compressed one — and which of them carries it is the
                // demuxer's business rather than the picker's.
                if !formats.contains(&mode) {
                    formats.push(mode);
                }
            }
        }
    }
    formats.sort_by(|left, right| {
        let area = |mode: &V4l2CaptureFormat| u64::from(mode.width) * u64::from(mode.height);
        let fps = |mode: &V4l2CaptureFormat| {
            f64::from(mode.frame_rate.numerator()) / f64::from(mode.frame_rate.denominator().max(1))
        };
        area(right)
            .cmp(&area(left))
            .then_with(|| fps(right).total_cmp(&fps(left)))
    });
    Ok(formats)
}

fn pixel_formats(file: &std::fs::File) -> Vec<u32> {
    (0u32..)
        .map_while(|index| {
            let mut description = FmtDesc {
                index,
                kind: BUF_TYPE_VIDEO_CAPTURE,
                ..FmtDesc::default()
            };
            // SAFETY: as `query_capability`, for this ioctl's own struct.
            call(file, VIDIOC_ENUM_FMT, &mut description)
                .ok()
                .map(|()| description.pixel_format)
        })
        .collect()
}

fn frame_sizes(file: &std::fs::File, pixel_format: u32) -> Vec<(u32, u32)> {
    (0u32..)
        .map_while(|index| {
            let mut sizes = FrameSizeEnum {
                index,
                pixel_format,
                ..FrameSizeEnum::default()
            };
            // SAFETY: as `query_capability`, for this ioctl's own struct.
            call(file, VIDIOC_ENUM_FRAMESIZES, &mut sizes).ok()?;
            Some(sizes)
        })
        .filter(|sizes| sizes.kind == FRMSIZE_TYPE_DISCRETE)
        .map(|sizes| (sizes.width, sizes.height))
        .collect()
}

fn frame_rates(
    file: &std::fs::File,
    pixel_format: u32,
    width: u32,
    height: u32,
) -> Vec<ffmpeg::Rational> {
    (0u32..)
        .map_while(|index| {
            let mut intervals = FrameIntervalEnum {
                index,
                pixel_format,
                width,
                height,
                ..FrameIntervalEnum::default()
            };
            // SAFETY: as `query_capability`, for this ioctl's own struct.
            call(file, VIDIOC_ENUM_FRAMEINTERVALS, &mut intervals).ok()?;
            Some(intervals)
        })
        .filter(|intervals| intervals.kind == FRMIVAL_TYPE_DISCRETE)
        // An *interval* is seconds per frame, so the rate is its inverse.
        .filter(|intervals| intervals.numerator > 0)
        .map(|intervals| {
            ffmpeg::Rational::new(intervals.denominator as i32, intervals.numerator as i32)
        })
        .collect()
}

const VIDIOC_ENUM_FMT: libc::c_ulong = request(READ_WRITE, 2, size_of::<FmtDesc>());
const VIDIOC_ENUM_FRAMESIZES: libc::c_ulong = request(READ_WRITE, 74, size_of::<FrameSizeEnum>());
const VIDIOC_ENUM_FRAMEINTERVALS: libc::c_ulong =
    request(READ_WRITE, 75, size_of::<FrameIntervalEnum>());

const BUF_TYPE_VIDEO_CAPTURE: u32 = 1;
/// The node captures video.
const CAP_VIDEO_CAPTURE: u32 = 0x0000_0001;
const FRMSIZE_TYPE_DISCRETE: u32 = 1;
const FRMIVAL_TYPE_DISCRETE: u32 = 1;

#[repr(C)]
#[derive(Default)]
struct FmtDesc {
    index: u32,
    kind: u32,
    flags: u32,
    description: [u8; 32],
    pixel_format: u32,
    mbus_code: u32,
    reserved: [u32; 3],
}

#[repr(C)]
#[derive(Default)]
struct FrameSizeEnum {
    index: u32,
    pixel_format: u32,
    kind: u32,
    width: u32,
    height: u32,
    /// The stepwise form of the union above is four more words wide than the
    /// discrete one this reads. Kept as padding rather than modelled: a
    /// stepwise camera is filtered out by `kind`, and the size still has to
    /// be right or the request number would be.
    stepwise_tail: [u32; 4],
    reserved: [u32; 2],
}

#[repr(C)]
#[derive(Default)]
struct FrameIntervalEnum {
    index: u32,
    pixel_format: u32,
    width: u32,
    height: u32,
    kind: u32,
    numerator: u32,
    denominator: u32,
    /// As `FrameSizeEnum::stepwise_tail` — the stepwise form is wider, and
    /// only its size matters here.
    stepwise_tail: [u32; 4],
    reserved: [u32; 2],
}

/// Kept so a caller can hold a node open while it asks several things of it.
/// Unused today; `list_formats` opens per call, which is what a picker does.
#[allow(dead_code)]
type Node = OwnedFd;

#[cfg(test)]
mod tests {
    use super::*;

    /// The numbers this module computes against the ones the kernel's own
    /// headers publish. They are the whole safety argument for every ioctl
    /// here: the size a request carries is baked into it, so a struct whose
    /// layout drifted would be a driver writing to a length this side did not
    /// allocate. Comparing the derived numbers to the published ones checks
    /// every field of every struct at once.
    #[test]
    fn the_request_numbers_match_the_kernels() {
        assert_eq!(VIDIOC_ENUM_FMT, 0xC040_5602);
        assert_eq!(VIDIOC_ENUM_FRAMESIZES, 0xC02C_564A);
        assert_eq!(VIDIOC_ENUM_FRAMEINTERVALS, 0xC034_564B);
    }

    /// A machine with a camera answers with one; a machine without answers
    /// with none. Both are correct, so what this checks is that whatever
    /// comes back describes itself — a device with no node or no name would
    /// be one a picker cannot show.
    #[test]
    fn every_camera_offered_names_itself() {
        let devices = list_devices().expect("/dev is readable");
        for device in &devices {
            assert!(device.id.starts_with("/dev/video"), "{device:?}");
            assert!(!device.name.is_empty(), "{device:?}");
        }
        let Some(first) = devices.first() else {
            eprintln!("skipping: this machine has no camera");
            return;
        };
        for mode in list_formats(&first.id).expect("a camera that is there") {
            assert!(mode.width > 0 && mode.height > 0, "{mode:?}");
            assert!(mode.frame_rate.numerator() > 0, "{mode:?}");
        }
    }
}
