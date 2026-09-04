//! Video capture device enumeration, and the Media Foundation platform
//! lifetime every call into it needs.

use std::ffi::c_void;
use std::ptr;

use ffmpeg_next as ffmpeg;
use windows::{
    Win32::{
        Media::MediaFoundation::{
            IMFActivate, IMFAttributes, IMFMediaSource, IMFMediaType, IMFSourceReader,
            MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
            MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID,
            MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, MF_E_NO_MORE_TYPES,
            MF_MT_FRAME_RATE, MF_MT_FRAME_SIZE, MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING,
            MF_SOURCE_READER_FIRST_VIDEO_STREAM, MF_VERSION, MFCreateAttributes,
            MFCreateDeviceSource, MFCreateSourceReaderFromMediaSource, MFEnumDeviceSources,
            MFSTARTUP_NOSOCKET, MFShutdown, MFStartup,
        },
        System::Com::CoTaskMemFree,
    },
    core::{GUID, HSTRING, PWSTR},
};

use super::super::com::ComApartment;

/// Balances one successful `MFStartup` on this process.
///
/// Media Foundation counts its own startups, so holding one of these across
/// each call is correct and cheap: the platform stays up while any guard is
/// alive, and the last `Drop` is what actually shuts it down.
pub(crate) struct MfRuntime;

impl MfRuntime {
    pub(crate) fn new() -> windows::core::Result<Self> {
        // SAFETY: `MF_VERSION` is the header's own version constant and
        // `MFSTARTUP_NOSOCKET` a documented flag; the successful startup is
        // balanced in `Drop`.
        unsafe { MFStartup(MF_VERSION, MFSTARTUP_NOSOCKET) }?;
        Ok(Self)
    }
}

impl Drop for MfRuntime {
    fn drop(&mut self) {
        // SAFETY: this instance records a successful `MFStartup`, so this
        // call balances it exactly once.
        let _ = unsafe { MFShutdown() };
    }
}

/// One video capture device Media Foundation currently offers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MfDevice {
    /// The device's symbolic link — the only identity that survives a
    /// restart, and what opening one is resolved against. Enumeration order
    /// is not identity: unplugging one camera renumbers the rest.
    pub id: String,
    /// Human-readable device name, falling back to `id` when the device
    /// exposes no friendly name.
    pub name: String,
}

/// One picture shape a camera offers, as a caller would show it in a picker.
///
/// Deliberately not the subtype the camera would deliver it in.
/// [`MfCaptureSource`] always asks for NV12 and lets Media Foundation put
/// whatever decoder or colour converter that needs in front of it, so
/// whether a mode is natively MJPEG, YUY2 or NV12 is this crate's business
/// rather than a choice a caller could make usefully.
///
/// [`MfCaptureSource`]: crate::elements::MfCaptureSource
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MfCaptureFormat {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Frames per second, as the camera states it. `30000/1001` is a real
    /// answer and is not the same mode as `30/1`.
    pub framerate: ffmpeg::Rational,
}

/// Reads a `CoTaskMemAlloc`-ed string attribute, freeing it either way.
///
/// `None` for an attribute the device does not carry, which is an ordinary
/// answer rather than a failure — a camera with no friendly name is still a
/// camera.
fn allocated_string(attributes: &IMFAttributes, key: &GUID) -> Option<String> {
    let mut value = PWSTR::null();
    let mut length = 0u32;
    // SAFETY: both outputs are live locals. On success Media Foundation hands
    // back a NUL-terminated `CoTaskMemAlloc` string this frees below; on
    // failure it writes neither.
    unsafe { attributes.GetAllocatedString(key, &mut value, &mut length) }.ok()?;
    // SAFETY: `value` is the NUL-terminated string just returned and stays
    // valid until it is freed on the next line.
    let text = unsafe { value.to_string() }.ok();
    // SAFETY: balances the allocation `GetAllocatedString` made, exactly once.
    unsafe { CoTaskMemFree(Some(value.as_ptr() as *const c_void)) };
    text
}

/// The two 32-bit halves Media Foundation packs into one 64-bit attribute.
///
/// `MFGetAttributeSize` and `MFGetAttributeRatio` are inline helpers in
/// `mfapi.h` rather than exported functions, so there is no binding to call;
/// this is what they do.
fn packed_pair(media_type: &IMFMediaType, key: &GUID) -> windows::core::Result<(u32, u32)> {
    // SAFETY: `media_type` is live and `key` a documented UINT64 attribute.
    let packed = unsafe { media_type.GetUINT64(key) }?;
    Ok(((packed >> 32) as u32, packed as u32))
}

pub(crate) fn frame_size(media_type: &IMFMediaType) -> windows::core::Result<(u32, u32)> {
    packed_pair(media_type, &MF_MT_FRAME_SIZE)
}

pub(crate) fn frame_rate(media_type: &IMFMediaType) -> windows::core::Result<ffmpeg::Rational> {
    let (numerator, denominator) = packed_pair(media_type, &MF_MT_FRAME_RATE)?;
    Ok(ffmpeg::Rational::new(numerator as i32, denominator as i32))
}

/// Attributes naming one video capture device, or every one of them when
/// `symbolic_link` is `None`.
fn vidcap_attributes(symbolic_link: Option<&str>) -> windows::core::Result<IMFAttributes> {
    let mut attributes = None;
    // SAFETY: the out parameter is a live local. Two entries is a sizing
    // hint rather than a bound, so it cannot be wrong.
    unsafe { MFCreateAttributes(&mut attributes, 2) }?;
    let attributes = attributes.expect("MFCreateAttributes yields an object whenever it succeeds");
    // SAFETY: `attributes` is the store just created, and both keys are
    // documented for it.
    unsafe {
        attributes.SetGUID(
            &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
            &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID,
        )?;
        if let Some(link) = symbolic_link {
            attributes.SetString(
                &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK,
                &HSTRING::from(link),
            )?;
        }
    }
    Ok(attributes)
}

/// Every video capture device currently attached.
///
/// A device with no symbolic link is skipped rather than reported: it is
/// nothing a later open could resolve.
pub(crate) fn list_devices() -> windows::core::Result<Vec<MfDevice>> {
    let _apartment = ComApartment::new()?;
    let _runtime = MfRuntime::new()?;
    let attributes = vidcap_attributes(None)?;

    let mut activates: *mut Option<IMFActivate> = ptr::null_mut();
    let mut count = 0u32;
    // SAFETY: both outputs are live locals. On success Media Foundation hands
    // back a `CoTaskMemAlloc`-ed array of `count` references this takes
    // ownership of and frees below.
    unsafe { MFEnumDeviceSources(&attributes, &mut activates, &mut count) }?;

    let mut devices = Vec::with_capacity(count as usize);
    for index in 0..count as usize {
        // SAFETY: `index` is bounded by the count just returned for this
        // array, and `take` moves the reference out so its own `Drop`
        // releases it exactly once.
        let Some(activate) = (unsafe { (*activates.add(index)).take() }) else {
            continue;
        };
        let Some(id) = allocated_string(
            &activate,
            &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK,
        ) else {
            continue;
        };
        let name = allocated_string(&activate, &MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME)
            // Trimmed because a picker shows this. USB descriptors are
            // fixed-width fields, and cameras routinely pad their name out to
            // the end of one — this machine has one that answers
            // `"    HCAM01L    "`.
            .map(|name| name.trim().to_owned())
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| id.clone());
        devices.push(MfDevice { id, name });
    }
    // SAFETY: balances the array allocation `MFEnumDeviceSources` made. Every
    // reference it held was moved out and released above.
    unsafe { CoTaskMemFree(Some(activates as *const c_void)) };

    Ok(devices)
}

/// Opens the device `symbolic_link` names.
///
/// The caller owns the returned source and must `Shutdown` it; dropping the
/// last reference without that leaves the camera held.
pub(crate) fn open_device_source(symbolic_link: &str) -> windows::core::Result<IMFMediaSource> {
    let attributes = vidcap_attributes(Some(symbolic_link))?;
    // SAFETY: `attributes` names exactly one video capture device; the
    // returned source owns its COM reference.
    unsafe { MFCreateDeviceSource(&attributes) }
}

/// Every picture shape one device offers, in the order it offers them.
///
/// Deduplicated, because a camera that can deliver 1280x720 at 30 fps as both
/// MJPEG and YUY2 states that twice and it is one choice as far as a caller
/// is concerned. The first entry is the device's own preference, which is
/// what asking for no particular format takes.
pub(crate) fn list_formats(
    reader: &IMFSourceReader,
) -> windows::core::Result<Vec<MfCaptureFormat>> {
    let stream = MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32;
    let mut formats: Vec<MfCaptureFormat> = Vec::new();
    for index in 0.. {
        // SAFETY: `reader` is live. Walking the index until the documented
        // `MF_E_NO_MORE_TYPES` is how this list is enumerated.
        let media_type = match unsafe { reader.GetNativeMediaType(stream, index) } {
            Ok(media_type) => media_type,
            Err(error) if error.code() == MF_E_NO_MORE_TYPES => break,
            Err(error) => return Err(error),
        };
        let (Ok((width, height)), Ok(framerate)) =
            (frame_size(&media_type), frame_rate(&media_type))
        else {
            // A mode that states neither a size nor a rate is nothing a
            // caller could pick, and skipping it leaves the rest usable.
            continue;
        };
        let format = MfCaptureFormat {
            width,
            height,
            framerate,
        };
        if !formats.contains(&format) {
            formats.push(format);
        }
    }
    Ok(formats)
}

/// A source reader over `source`, with Media Foundation's own converters
/// allowed in front of it.
///
/// `MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING` is what lets one
/// `SetCurrentMediaType(NV12)` work against a camera that only speaks MJPEG
/// or YUY2: Media Foundation inserts the decoder and colour converter itself.
/// Without it this element would have to carry one of each.
pub(crate) fn open_reader(source: &IMFMediaSource) -> windows::core::Result<IMFSourceReader> {
    let mut attributes = None;
    // SAFETY: the out parameter is a live local.
    unsafe { MFCreateAttributes(&mut attributes, 1) }?;
    let attributes = attributes.expect("MFCreateAttributes yields an object whenever it succeeds");
    // SAFETY: `attributes` is the store just created and the key is a
    // documented UINT32 reader attribute.
    unsafe { attributes.SetUINT32(&MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, 1) }?;
    // SAFETY: `source` is a live media source and `attributes` a store built
    // for this call; the returned reader owns its COM reference.
    unsafe { MFCreateSourceReaderFromMediaSource(source, &attributes) }
}
