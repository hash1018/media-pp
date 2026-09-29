//! What `AvFoundationCaptureSource` asks of AVFoundation before it captures:
//! the cameras there are, the modes each offers, putting one in a mode, and
//! whether this program may use them at all.

use std::{sync::mpsc, time::Duration};

use ffmpeg_next as ffmpeg;
use objc2::{rc::Retained, runtime::Bool};
use objc2_av_foundation::{
    AVAuthorizationStatus, AVCaptureDevice, AVCaptureDeviceDiscoverySession, AVCaptureDeviceFormat,
    AVCaptureDevicePosition, AVCaptureDeviceTypeBuiltInWideAngleCamera,
    AVCaptureDeviceTypeContinuityCamera, AVCaptureDeviceTypeDeskViewCamera,
    AVCaptureDeviceTypeExternal, AVMediaType, AVMediaTypeVideo,
};
use objc2_core_media::{CMTime, CMVideoFormatDescriptionGetDimensions};
use objc2_foundation::{NSArray, NSString};

/// One camera, as [`crate::elements::AvFoundationCaptureSource::list_devices`]
/// lists it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AvFoundationDevice {
    /// AVFoundation's `uniqueID`, which stays the same for one camera across
    /// restarts and replugging, and is what opening one looks it up by.
    pub id: String,
    /// The name the system shows for it.
    pub name: String,
    /// Whether this was the system's default camera when it was listed.
    pub is_default: bool,
}

/// One picture shape a camera offers, as a caller would show it in a picker.
///
/// Deliberately not the subtype the camera would deliver it in — what this
/// crate asks for is NV12 whatever the mode, which Core Video converts to
/// where the camera speaks something else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AvFoundationCaptureFormat {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Frames per second, as the camera states it. `30000/1001` is a real
    /// answer and is not the same mode as `30/1`.
    pub frame_rate: ffmpeg::Rational,
}

/// Whether this program may use the camera, as far as the system has said.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Authorization {
    /// It may.
    Authorized,
    /// It may not: the user said no, or a policy does.
    Denied,
    /// Nobody has been asked yet.
    NotDetermined,
}

fn video() -> &'static AVMediaType {
    // SAFETY: a constant AVFoundation exports, read once it is loaded, which
    // linking it guarantees.
    unsafe { AVMediaTypeVideo }.expect("AVFoundation exports AVMediaTypeVideo")
}

/// What the system says of this program and the camera, without asking the
/// user anything — which is safe even for a program that could not ask.
pub(crate) fn authorization() -> Authorization {
    // SAFETY: a class method taking a media type constant.
    let status = unsafe { AVCaptureDevice::authorizationStatusForMediaType(video()) };
    match status {
        AVAuthorizationStatus::Authorized => Authorization::Authorized,
        AVAuthorizationStatus::NotDetermined => Authorization::NotDetermined,
        _ => Authorization::Denied,
    }
}

/// Asks the user whether this program may use the camera, and waits for
/// the answer however long it takes; `true` where it may.
///
/// macOS ends a program that asks without an `NSCameraUsageDescription` in
/// its `Info.plist` — the application's, for one bundled, and the
/// terminal's or whatever started it, for one that is not.
pub(crate) fn request_access() -> bool {
    let (tx, rx) = mpsc::channel();
    let answer = block2::RcBlock::new(move |granted: Bool| {
        let _ = tx.send(granted.as_bool());
    });
    // SAFETY: the media type is AVFoundation's own constant, and the block
    // outlives the call — AVFoundation copies it and calls it once.
    unsafe { AVCaptureDevice::requestAccessForMediaType_completionHandler(video(), &answer) };
    loop {
        match rx.recv_timeout(Duration::from_secs(1)) {
            Ok(granted) => return granted,
            // The dialog is up; nobody has answered it.
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => return false,
        }
    }
}

/// Every camera there is: built in, plugged in, an iPhone as a Continuity
/// Camera, and a Desk View.
fn devices() -> Retained<NSArray<AVCaptureDevice>> {
    // SAFETY: the device types are AVFoundation's own constants, and the
    // discovery session is a plain query.
    unsafe {
        let types = NSArray::from_slice(&[
            AVCaptureDeviceTypeBuiltInWideAngleCamera,
            AVCaptureDeviceTypeExternal,
            AVCaptureDeviceTypeContinuityCamera,
            AVCaptureDeviceTypeDeskViewCamera,
        ]);
        AVCaptureDeviceDiscoverySession::discoverySessionWithDeviceTypes_mediaType_position(
            &types,
            Some(video()),
            AVCaptureDevicePosition::Unspecified,
        )
        .devices()
    }
}

/// Every camera, the system's default marked. Listing asks nothing of the
/// user.
pub(crate) fn list_devices() -> Vec<AvFoundationDevice> {
    // SAFETY: plain queries of AVFoundation's own objects.
    unsafe {
        let default = AVCaptureDevice::defaultDeviceWithMediaType(video())
            .map(|device| device.uniqueID().to_string());
        devices()
            .iter()
            .map(|device| {
                let id = device.uniqueID().to_string();
                AvFoundationDevice {
                    is_default: default.as_deref() == Some(id.as_str()),
                    name: device.localizedName().to_string(),
                    id,
                }
            })
            .collect()
    }
}

/// The camera `id` names, where it is still there.
pub(crate) fn device(id: &str) -> Option<Retained<AVCaptureDevice>> {
    // SAFETY: a class method looking up by the string it is handed.
    unsafe { AVCaptureDevice::deviceWithUniqueID(&NSString::from_str(id)) }
}

/// A frame rate as the frame duration AVFoundation states it by: `1/30`
/// seconds is `30/1`.
fn rate_of(duration: CMTime) -> Option<ffmpeg::Rational> {
    (duration.value > 0 && duration.timescale > 0).then(|| {
        let value = i32::try_from(duration.value).unwrap_or(i32::MAX);
        ffmpeg::Rational::new(duration.timescale, value).reduce()
    })
}

/// Every mode `device` offers, largest first and fastest first within a
/// size, each once — a format a camera lists in several subtypes, or with
/// several rates, is as many modes as it has sizes and rates.
pub(crate) fn list_formats(device: &AVCaptureDevice) -> Vec<AvFoundationCaptureFormat> {
    let mut modes = Vec::new();
    // SAFETY: plain queries of the device's own format objects.
    unsafe {
        for format in device.formats().iter() {
            let (width, height) = dimensions(&format);
            for range in format.videoSupportedFrameRateRanges().iter() {
                if let Some(frame_rate) = rate_of(range.minFrameDuration()) {
                    let mode = AvFoundationCaptureFormat {
                        width,
                        height,
                        frame_rate,
                    };
                    if !modes.contains(&mode) {
                        modes.push(mode);
                    }
                }
            }
        }
    }
    modes.sort_by(|a, b| {
        let area =
            |mode: &AvFoundationCaptureFormat| u64::from(mode.width) * u64::from(mode.height);
        area(b)
            .cmp(&area(a))
            .then(f64::from(b.frame_rate).total_cmp(&f64::from(a.frame_rate)))
    });
    modes
}

fn dimensions(format: &AVCaptureDeviceFormat) -> (u32, u32) {
    // SAFETY: a plain query of a live format description.
    let dimensions = unsafe { CMVideoFormatDescriptionGetDimensions(&format.formatDescription()) };
    (
        u32::try_from(dimensions.width).unwrap_or(0),
        u32::try_from(dimensions.height).unwrap_or(0),
    )
}

/// The format `device` is in now, at the highest rate it offers — the
/// rate a session left to itself runs it at in good light.
pub(crate) fn active_format(device: &AVCaptureDevice) -> AvFoundationCaptureFormat {
    // SAFETY: plain queries of the device's own state.
    let format = unsafe { device.activeFormat() };
    let (width, height) = dimensions(&format);
    // SAFETY: as above.
    let frame_rate = unsafe { format.videoSupportedFrameRateRanges() }
        .iter()
        // SAFETY: as above.
        .filter_map(|range| rate_of(unsafe { range.minFrameDuration() }))
        .max_by(|a, b| f64::from(*a).total_cmp(&f64::from(*b)))
        .unwrap_or(ffmpeg::Rational::new(30, 1));
    AvFoundationCaptureFormat {
        width,
        height,
        frame_rate,
    }
}

/// Why a camera could not be put in the mode asked for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SetFormatError {
    /// It offers no such mode.
    NotOffered,
    /// It would not be configured — another program holds it.
    Locked(String),
}

/// A camera held in the mode [`hold_format`] put it in, until this drops.
///
/// On macOS a capture session applies its preset's format to its camera
/// whenever it starts running — `AVCaptureSessionPresetInputPriority`,
/// which on iOS would leave it alone, is refused there. What a session
/// leaves alone is a camera locked for configuration, so the camera is put
/// in its mode and held locked across each `startRunning`.
pub(crate) struct FormatHold<'a>(&'a AVCaptureDevice);

impl Drop for FormatHold<'_> {
    fn drop(&mut self) {
        // SAFETY: the lock `hold_format` took, released once.
        unsafe { self.0.unlockForConfiguration() };
    }
}

/// Puts `device` in `mode` — the format of that size with that rate among
/// its ranges, with the frame duration fixed to it — and keeps it locked
/// for configuration until the returned hold drops.
pub(crate) fn hold_format(
    device: &AVCaptureDevice,
    mode: AvFoundationCaptureFormat,
) -> Result<FormatHold<'_>, SetFormatError> {
    // SAFETY: plain queries of the device's own format objects.
    let found = unsafe {
        device.formats().iter().find_map(|format| {
            if dimensions(&format) != (mode.width, mode.height) {
                return None;
            }
            format
                .videoSupportedFrameRateRanges()
                .iter()
                .map(|range| range.minFrameDuration())
                .find(|&duration| rate_of(duration) == Some(mode.frame_rate))
                .map(|duration| (format, duration))
        })
    };
    let (format, duration) = found.ok_or(SetFormatError::NotOffered)?;
    // SAFETY: the configuration lock is taken before the changes it guards,
    // and released by the hold; the format is one of the device's own, and
    // the duration one of its ranges'.
    unsafe {
        device
            .lockForConfiguration()
            .map_err(|error| SetFormatError::Locked(error.localizedDescription().to_string()))?;
        let hold = FormatHold(device);
        device.setActiveFormat(&format);
        device.setActiveVideoMinFrameDuration(duration);
        device.setActiveVideoMaxFrameDuration(duration);
        Ok(hold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whatever cameras the machine has, each is listed once with a name
    /// and an id it can be found by again, at most one is the default, and
    /// its modes come largest first. Listing asks the user nothing.
    #[test]
    fn listed_cameras_can_be_found_again_with_their_modes() {
        let cameras = list_devices();
        if cameras.is_empty() {
            eprintln!("skipping: this machine has no camera");
            return;
        }
        assert!(cameras.iter().filter(|camera| camera.is_default).count() <= 1);
        for camera in &cameras {
            assert!(
                !camera.name.is_empty() && !camera.id.is_empty(),
                "{camera:?}"
            );
            let device = device(&camera.id).expect("a listed camera is there by its id");
            let modes = list_formats(&device);
            assert!(!modes.is_empty(), "{camera:?} offers a mode");
            let areas: Vec<u64> = modes
                .iter()
                .map(|mode| u64::from(mode.width) * u64::from(mode.height))
                .collect();
            assert!(areas.windows(2).all(|pair| pair[0] >= pair[1]), "{modes:?}");
        }
    }

    /// A frame duration is the rate it is the inverse of, NTSC's included.
    #[test]
    fn a_frame_duration_is_its_rate() {
        let duration = |value, timescale| CMTime {
            value,
            timescale,
            flags: objc2_core_media::CMTimeFlags(1),
            epoch: 0,
        };
        assert_eq!(rate_of(duration(1, 30)), Some(ffmpeg::Rational::new(30, 1)));
        assert_eq!(
            rate_of(duration(1001, 30000)),
            Some(ffmpeg::Rational::new(30000, 1001))
        );
        assert_eq!(rate_of(duration(0, 30)), None);
    }
}
