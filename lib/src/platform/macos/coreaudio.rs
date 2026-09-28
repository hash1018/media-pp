//! What `CoreAudioRenderer` asks of Core Audio's hardware layer: the output
//! devices there are, the format each plays, and how long a sample takes to
//! be heard once handed to it.
//!
//! Every query is `AudioObjectGetPropertyData` on an `AudioObjectID`, with a
//! selector and a scope. A device that disappears between being listed and
//! being asked answers with `kAudioHardwareBadObjectError`, which reaches the
//! caller as an [`OsStatusError`] like any other failure.

use std::{
    ffi::c_void,
    mem::{MaybeUninit, offset_of, size_of},
    ptr::{self, NonNull},
};

use objc2_core_audio::{
    AudioObjectGetPropertyData, AudioObjectGetPropertyDataSize, AudioObjectID,
    AudioObjectPropertyAddress, AudioObjectPropertyScope, AudioObjectPropertySelector,
    kAudioDevicePropertyDeviceUID, kAudioDevicePropertyLatency,
    kAudioDevicePropertyNominalSampleRate, kAudioDevicePropertyStreamConfiguration,
    kAudioDevicePropertyStreams, kAudioHardwareBadPropertySizeError,
    kAudioHardwarePropertyDefaultOutputDevice, kAudioHardwarePropertyDevices,
    kAudioObjectPropertyElementMain, kAudioObjectPropertyName, kAudioObjectPropertyScopeGlobal,
    kAudioObjectPropertyScopeOutput, kAudioObjectSystemObject, kAudioStreamPropertyLatency,
};
use objc2_core_audio_types::{AudioBuffer, AudioBufferList};
use objc2_core_foundation::{CFRetained, CFString};

/// One device Core Audio can play to, as `CoreAudioRenderer::list_devices`
/// lists it and `CoreAudioRendererOptions::device` takes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreAudioDevice {
    /// The HAL's `AudioObjectID`. Good only while the device stays attached
    /// and the system keeps running: a device plugged in again, or the same
    /// one after a restart, can have another. Persist [`Self::uid`] to find a
    /// choice again.
    pub id: u32,
    /// `kAudioDevicePropertyDeviceUID`, which stays the same for one physical
    /// device across restarts.
    pub uid: String,
    /// The name the system shows for the device.
    pub name: String,
    /// Whether this was the system's default output when it was listed.
    pub is_default: bool,
}

/// A Core Audio call that did not return `noErr`, and which one it was.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OsStatusError {
    pub(crate) operation: &'static str,
    pub(crate) status: i32,
}

/// `Ok` for `noErr`, and `operation`'s failure otherwise.
pub(crate) fn check(status: i32, operation: &'static str) -> Result<(), OsStatusError> {
    if status == 0 {
        Ok(())
    } else {
        Err(OsStatusError { operation, status })
    }
}

/// Every device with at least one output channel, the system's default
/// marked. A device that fails to answer while being listed is left out
/// rather than failing the whole list: it is most likely one unplugged
/// between the two queries.
pub(crate) fn list_output_devices() -> Result<Vec<CoreAudioDevice>, OsStatusError> {
    let default = default_output_device().ok();
    let ids: Vec<AudioObjectID> = get_array(
        kAudioObjectSystemObject as AudioObjectID,
        kAudioHardwarePropertyDevices,
        kAudioObjectPropertyScopeGlobal,
        "list the audio devices",
    )?;
    Ok(ids
        .into_iter()
        .filter(|&id| output_channels(id).is_ok_and(|channels| channels > 0))
        .filter_map(|id| {
            let uid = get_string(id, kAudioDevicePropertyDeviceUID, "read a device's UID").ok()?;
            let name = get_string(id, kAudioObjectPropertyName, "read a device's name")
                .unwrap_or_else(|_| uid.clone());
            Some(CoreAudioDevice {
                id,
                uid,
                name,
                is_default: Some(id) == default,
            })
        })
        .collect())
}

/// The system's default output device.
pub(crate) fn default_output_device() -> Result<AudioObjectID, OsStatusError> {
    get(
        kAudioObjectSystemObject as AudioObjectID,
        kAudioHardwarePropertyDefaultOutputDevice,
        kAudioObjectPropertyScopeGlobal,
        "read the default output device",
    )
}

/// How many channels `device` plays, across all its output streams.
pub(crate) fn output_channels(device: AudioObjectID) -> Result<u32, OsStatusError> {
    const OPERATION: &str = "read a device's output channels";
    let address = address(
        kAudioDevicePropertyStreamConfiguration,
        kAudioObjectPropertyScopeOutput,
    );
    let size = data_size(device, &address, OPERATION)?;
    // An `AudioBufferList` of as many buffers as the device has streams,
    // laid out as C lays it out. `u64`s so that its pointers are aligned.
    let mut list = vec![0u64; (size as usize).div_ceil(size_of::<u64>())];
    let mut filled = size;
    // SAFETY: `list` is writable for `size` bytes and 8-byte aligned, which is
    // what an `AudioBufferList` needs; the HAL writes at most `filled` bytes.
    let status = unsafe {
        AudioObjectGetPropertyData(
            device,
            NonNull::from(&address),
            0,
            ptr::null(),
            NonNull::from(&mut filled),
            NonNull::new_unchecked(list.as_mut_ptr().cast::<c_void>()),
        )
    };
    check(status, OPERATION)?;
    let filled = filled as usize;
    if filled < offset_of!(AudioBufferList, mBuffers) {
        return Ok(0);
    }
    let base = list.as_ptr().cast::<u8>();
    // SAFETY: at least the list's count lies inside what the HAL filled,
    // checked above, and `base` is aligned for it.
    let buffers = unsafe { base.cast::<AudioBufferList>().read() }.mNumberBuffers as usize;
    let mut channels = 0u32;
    for index in 0..buffers {
        let offset = offset_of!(AudioBufferList, mBuffers) + index * size_of::<AudioBuffer>();
        if offset + size_of::<AudioBuffer>() > filled {
            break;
        }
        // SAFETY: this buffer lies wholly inside what the HAL filled, checked
        // just above, and follows the list's own alignment.
        let buffer = unsafe { base.add(offset).cast::<AudioBuffer>().read() };
        channels = channels.saturating_add(buffer.mNumberChannels);
    }
    Ok(channels)
}

/// The rate `device` runs at now.
pub(crate) fn nominal_sample_rate(device: AudioObjectID) -> Result<f64, OsStatusError> {
    get(
        device,
        kAudioDevicePropertyNominalSampleRate,
        kAudioObjectPropertyScopeGlobal,
        "read a device's sample rate",
    )
}

/// How many frames after the HAL hands `device` a sample it is heard: the
/// device's own latency and that of its first output stream. Neither is
/// something every device reports, so one it does not is counted as none.
pub(crate) fn output_latency_frames(device: AudioObjectID) -> u32 {
    let device_latency: u32 = get(
        device,
        kAudioDevicePropertyLatency,
        kAudioObjectPropertyScopeOutput,
        "read a device's latency",
    )
    .unwrap_or(0);
    let stream_latency = get_array::<AudioObjectID>(
        device,
        kAudioDevicePropertyStreams,
        kAudioObjectPropertyScopeOutput,
        "list a device's output streams",
    )
    .ok()
    .and_then(|streams| streams.first().copied())
    .and_then(|stream| {
        get::<u32>(
            stream,
            kAudioStreamPropertyLatency,
            kAudioObjectPropertyScopeGlobal,
            "read a stream's latency",
        )
        .ok()
    })
    .unwrap_or(0);
    device_latency.saturating_add(stream_latency)
}

fn address(
    selector: AudioObjectPropertySelector,
    scope: AudioObjectPropertyScope,
) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress {
        mSelector: selector,
        mScope: scope,
        mElement: kAudioObjectPropertyElementMain,
    }
}

/// A value any bit pattern of its size is valid for, which is all a
/// property read promises of what it writes.
trait Plain: Copy {}
impl Plain for u32 {}
impl Plain for f64 {}

fn data_size(
    object: AudioObjectID,
    address: &AudioObjectPropertyAddress,
    operation: &'static str,
) -> Result<u32, OsStatusError> {
    let mut size = 0u32;
    // SAFETY: `address` and `size` are live for the call, and no qualifier is
    // passed.
    let status = unsafe {
        AudioObjectGetPropertyDataSize(
            object,
            NonNull::from(address),
            0,
            ptr::null(),
            NonNull::from(&mut size),
        )
    };
    check(status, operation)?;
    Ok(size)
}

/// One fixed-size property.
fn get<T: Plain>(
    object: AudioObjectID,
    selector: AudioObjectPropertySelector,
    scope: AudioObjectPropertyScope,
    operation: &'static str,
) -> Result<T, OsStatusError> {
    let address = address(selector, scope);
    let mut value = MaybeUninit::<T>::uninit();
    let mut size = size_of::<T>() as u32;
    // SAFETY: `value` is writable for `size` bytes, which the HAL writes no
    // more than, and `address` and `size` are live for the call.
    let status = unsafe {
        AudioObjectGetPropertyData(
            object,
            NonNull::from(&address),
            0,
            ptr::null(),
            NonNull::from(&mut size),
            NonNull::new_unchecked(value.as_mut_ptr().cast::<c_void>()),
        )
    };
    check(status, operation)?;
    if size as usize != size_of::<T>() {
        return Err(OsStatusError {
            operation,
            status: kAudioHardwareBadPropertySizeError,
        });
    }
    // SAFETY: the HAL filled all of `T`, checked just above, and any bit
    // pattern is a valid `T` (`Plain`).
    Ok(unsafe { value.assume_init() })
}

/// A property that is an array of one fixed-size value.
fn get_array<T: Plain + Default>(
    object: AudioObjectID,
    selector: AudioObjectPropertySelector,
    scope: AudioObjectPropertyScope,
    operation: &'static str,
) -> Result<Vec<T>, OsStatusError> {
    let address = address(selector, scope);
    let size = data_size(object, &address, operation)?;
    let mut values = vec![T::default(); size as usize / size_of::<T>()];
    let mut filled = (values.len() * size_of::<T>()) as u32;
    if filled == 0 {
        return Ok(values);
    }
    // SAFETY: `values` is writable for `filled` bytes, which the HAL writes no
    // more than, and `address` and `filled` are live for the call.
    let status = unsafe {
        AudioObjectGetPropertyData(
            object,
            NonNull::from(&address),
            0,
            ptr::null(),
            NonNull::from(&mut filled),
            NonNull::new_unchecked(values.as_mut_ptr().cast::<c_void>()),
        )
    };
    check(status, operation)?;
    // The list can have shrunk between the two calls — a device unplugged.
    values.truncate(filled as usize / size_of::<T>());
    Ok(values)
}

/// A property that is a `CFString` the caller owns.
fn get_string(
    object: AudioObjectID,
    selector: AudioObjectPropertySelector,
    operation: &'static str,
) -> Result<String, OsStatusError> {
    let address = address(selector, kAudioObjectPropertyScopeGlobal);
    let mut string: *const CFString = ptr::null();
    let mut size = size_of::<*const CFString>() as u32;
    // SAFETY: `string` is writable for one pointer, which is what these
    // selectors write, and `address` and `size` are live for the call.
    let status = unsafe {
        AudioObjectGetPropertyData(
            object,
            NonNull::from(&address),
            0,
            ptr::null(),
            NonNull::from(&mut size),
            NonNull::from(&mut string).cast::<c_void>(),
        )
    };
    check(status, operation)?;
    let string = NonNull::new(string.cast_mut()).ok_or(OsStatusError {
        operation,
        status: kAudioHardwareBadPropertySizeError,
    })?;
    // SAFETY: the HAL hands over a `CFString` it retained for the caller —
    // the "copy" rule these properties document — so this takes ownership
    // of exactly that one reference.
    let string = unsafe { CFRetained::from_raw(string) };
    Ok(string.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whatever the machine has, what is listed can be asked again: every
    /// device plays at a rate and has channels, and at most one is the
    /// default. A machine with no output lists nothing, which says nothing.
    #[test]
    fn listed_devices_answer_for_their_format() {
        let devices = list_output_devices().expect("the HAL lists its devices");
        if devices.is_empty() {
            eprintln!("skipping: no Core Audio output device");
            return;
        }
        for device in &devices {
            assert!(!device.uid.is_empty(), "{device:?} has a UID");
            assert!(output_channels(device.id).unwrap() > 0, "{device:?}");
            assert!(nominal_sample_rate(device.id).unwrap() > 0.0, "{device:?}");
        }
        assert!(devices.iter().filter(|device| device.is_default).count() <= 1);
    }

    /// A device that is not there is a typed failure naming what was asked,
    /// not a panic or a zero.
    #[test]
    fn a_device_that_is_not_there_is_an_error() {
        let error = nominal_sample_rate(u32::MAX).unwrap_err();
        assert_eq!(error.operation, "read a device's sample rate");
        assert_ne!(error.status, 0);
    }
}
