//! What the Core Audio elements ask of Core Audio: the devices there are to
//! play to and record from, the format each runs in, how long a sample
//! takes to be heard — and the AUHAL unit both drive a device through.
//!
//! Every device query is `AudioObjectGetPropertyData` on an `AudioObjectID`,
//! with a selector and a scope. A device that disappears between being
//! listed and being asked answers with `kAudioHardwareBadObjectError`, which
//! reaches the caller as an [`OsStatusError`] like any other failure.

use std::{
    ffi::c_void,
    mem::{MaybeUninit, offset_of, size_of},
    ptr::{self, NonNull},
    sync::Arc,
};

use objc2_audio_toolbox::{
    AudioComponentDescription, AudioComponentFindNext, AudioComponentInstanceDispose,
    AudioComponentInstanceNew, AudioOutputUnitStart, AudioOutputUnitStop, AudioUnit,
    AudioUnitGetProperty, AudioUnitInitialize, AudioUnitSetProperty, AudioUnitUninitialize,
    kAudioUnitManufacturer_Apple, kAudioUnitSubType_HALOutput, kAudioUnitType_Output,
};
use objc2_core_audio::{
    AudioObjectGetPropertyData, AudioObjectGetPropertyDataSize, AudioObjectID,
    AudioObjectPropertyAddress, AudioObjectPropertyScope, AudioObjectPropertySelector,
    kAudioDevicePropertyDeviceUID, kAudioDevicePropertyNominalSampleRate,
    kAudioDevicePropertyStreamConfiguration, kAudioHardwareBadPropertySizeError,
    kAudioHardwarePropertyDevices, kAudioObjectPropertyElementMain, kAudioObjectPropertyName,
    kAudioObjectPropertyScopeGlobal, kAudioObjectSystemObject,
};
#[cfg(feature = "coreaudio-capture")]
use objc2_core_audio::{
    kAudioDevicePropertyDeviceIsAlive, kAudioHardwarePropertyDefaultInputDevice,
    kAudioObjectPropertyScopeInput,
};
#[cfg(feature = "coreaudio-renderer")]
use objc2_core_audio::{
    kAudioDevicePropertyLatency, kAudioDevicePropertyStreams,
    kAudioHardwarePropertyDefaultOutputDevice, kAudioObjectPropertyScopeOutput,
    kAudioStreamPropertyLatency,
};
use objc2_core_audio_types::AudioBuffer;
use objc2_core_foundation::{CFRetained, CFString};

/// One audio device, as `CoreAudioRenderer::list_devices` lists those it can
/// play to and `CoreAudioCaptureSource::list_devices` those it can record
/// from. A device that does both — a headset — is in both lists, as the
/// same device.
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
    /// Whether this was the system's default device for the list's
    /// direction — output or input — when it was listed.
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

/// An `OSStatus` that spells four printable characters — most of Core
/// Audio's do, `'!obj'` — as those characters too, for an error message.
pub(crate) fn four_char_code(status: &i32) -> String {
    let bytes = status.to_be_bytes();
    if bytes
        .iter()
        .all(|byte| byte.is_ascii_graphic() || *byte == b' ')
    {
        format!(" ('{}')", String::from_utf8_lossy(&bytes))
    } else {
        String::new()
    }
}

/// Every device with at least one output channel, the system's default
/// output marked.
#[cfg(feature = "coreaudio-renderer")]
pub(crate) fn list_output_devices() -> Result<Vec<CoreAudioDevice>, OsStatusError> {
    list_devices(
        kAudioObjectPropertyScopeOutput,
        default_output_device().ok(),
    )
}

/// Every device with at least one input channel, the system's default
/// input marked.
#[cfg(feature = "coreaudio-capture")]
pub(crate) fn list_input_devices() -> Result<Vec<CoreAudioDevice>, OsStatusError> {
    list_devices(kAudioObjectPropertyScopeInput, default_input_device().ok())
}

/// Every device with channels in `scope`. One that fails to answer while
/// being listed is left out rather than failing the whole list: it is most
/// likely one unplugged between the two queries.
fn list_devices(
    scope: AudioObjectPropertyScope,
    default: Option<AudioObjectID>,
) -> Result<Vec<CoreAudioDevice>, OsStatusError> {
    let ids: Vec<AudioObjectID> = get_array(
        kAudioObjectSystemObject as AudioObjectID,
        kAudioHardwarePropertyDevices,
        kAudioObjectPropertyScopeGlobal,
        "list the audio devices",
    )?;
    Ok(ids
        .into_iter()
        .filter(|&id| channels(id, scope).is_ok_and(|channels| channels > 0))
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
#[cfg(feature = "coreaudio-renderer")]
pub(crate) fn default_output_device() -> Result<AudioObjectID, OsStatusError> {
    get(
        kAudioObjectSystemObject as AudioObjectID,
        kAudioHardwarePropertyDefaultOutputDevice,
        kAudioObjectPropertyScopeGlobal,
        "read the default output device",
    )
}

/// The system's default input device.
#[cfg(feature = "coreaudio-capture")]
pub(crate) fn default_input_device() -> Result<AudioObjectID, OsStatusError> {
    get(
        kAudioObjectSystemObject as AudioObjectID,
        kAudioHardwarePropertyDefaultInputDevice,
        kAudioObjectPropertyScopeGlobal,
        "read the default input device",
    )
}

/// How many channels `device` plays, across all its output streams.
#[cfg(feature = "coreaudio-renderer")]
pub(crate) fn output_channels(device: AudioObjectID) -> Result<u32, OsStatusError> {
    channels(device, kAudioObjectPropertyScopeOutput)
}

/// How many channels `device` records, across all its input streams.
#[cfg(feature = "coreaudio-capture")]
pub(crate) fn input_channels(device: AudioObjectID) -> Result<u32, OsStatusError> {
    channels(device, kAudioObjectPropertyScopeInput)
}

/// Whether `device` is still there to be used — false once it is unplugged.
#[cfg(feature = "coreaudio-capture")]
pub(crate) fn is_alive(device: AudioObjectID) -> bool {
    get::<u32>(
        device,
        kAudioDevicePropertyDeviceIsAlive,
        kAudioObjectPropertyScopeGlobal,
        "ask whether a device is alive",
    )
    .is_ok_and(|alive| alive != 0)
}

/// How many channels `device`'s streams in `scope` have, all together.
fn channels(device: AudioObjectID, scope: AudioObjectPropertyScope) -> Result<u32, OsStatusError> {
    const OPERATION: &str = "read a device's channels";
    let address = address(kAudioDevicePropertyStreamConfiguration, scope);
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
    let buffers_at = offset_of!(objc2_core_audio_types::AudioBufferList, mBuffers);
    if filled < buffers_at {
        return Ok(0);
    }
    let base = list.as_ptr().cast::<u8>();
    // SAFETY: the count, the list's first field, lies inside what the HAL
    // filled, checked above, and `base` is aligned for it.
    let buffers = unsafe { base.cast::<u32>().read() } as usize;
    let mut channels = 0u32;
    for index in 0..buffers {
        let offset = buffers_at + index * size_of::<AudioBuffer>();
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
#[cfg(feature = "coreaudio-renderer")]
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
pub(crate) trait Plain: Copy {}
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

/// Apple's AUHAL output unit, for one hardware device: what plays to it and
/// what records from it. Made stopped and uninitialized; the caller sets it
/// up with [`Self::set`] and [`Self::initialize`]s it.
///
/// Dropping it stops, uninitializes and disposes of it, after which Core
/// Audio calls none of its callbacks again — so what they read is dropped
/// after it: see [`Refcon`].
pub(crate) struct HalUnit {
    unit: AudioUnit,
    initialized: bool,
}

// SAFETY: an AudioUnit may be called from any thread; Core Audio serializes
// its property and transport calls itself. Every call here takes `&mut self`
// or reads a property, so this side never overlaps them either.
unsafe impl Send for HalUnit {}

impl HalUnit {
    /// A new AUHAL unit, or `None` where the system has none.
    pub(crate) fn new() -> Result<Option<Self>, OsStatusError> {
        let description = AudioComponentDescription {
            componentType: kAudioUnitType_Output,
            componentSubType: kAudioUnitSubType_HALOutput,
            componentManufacturer: kAudioUnitManufacturer_Apple,
            componentFlags: 0,
            componentFlagsMask: 0,
        };
        // SAFETY: `description` is live for the call; a null start searches
        // from the first component.
        let component =
            unsafe { AudioComponentFindNext(ptr::null_mut(), NonNull::from(&description)) };
        if component.is_null() {
            return Ok(None);
        }
        let mut unit: AudioUnit = ptr::null_mut();
        // SAFETY: `component` was just found and `unit` is a live out-param.
        let status = unsafe { AudioComponentInstanceNew(component, NonNull::from(&mut unit)) };
        check(status, "create an AUHAL unit")?;
        Ok(Some(Self {
            unit,
            initialized: false,
        }))
    }

    /// The unit itself, for a callback to render from.
    #[cfg(feature = "coreaudio-capture")]
    pub(crate) fn raw(&self) -> AudioUnit {
        self.unit
    }

    /// Sets `property` in `scope` on `element` to `value`.
    pub(crate) fn set<T>(
        &mut self,
        property: u32,
        scope: u32,
        element: u32,
        value: &T,
        operation: &'static str,
    ) -> Result<(), OsStatusError> {
        // SAFETY: the unit is live and `value` is readable for its size for
        // the call, which copies it.
        let status = unsafe {
            AudioUnitSetProperty(
                self.unit,
                property,
                scope,
                element,
                ptr::from_ref(value).cast::<c_void>(),
                size_of::<T>() as u32,
            )
        };
        check(status, operation)
    }

    /// `property` in `scope` on `element`, where it is a plain value.
    pub(crate) fn get<T: Plain + Default>(
        &self,
        property: u32,
        scope: u32,
        element: u32,
        operation: &'static str,
    ) -> Result<T, OsStatusError> {
        let mut value = T::default();
        let mut size = size_of::<T>() as u32;
        // SAFETY: the unit is live and `value` is writable for `size` bytes.
        let status = unsafe {
            AudioUnitGetProperty(
                self.unit,
                property,
                scope,
                element,
                NonNull::from(&mut value).cast::<c_void>(),
                NonNull::from(&mut size),
            )
        };
        check(status, operation)?;
        Ok(value)
    }

    pub(crate) fn initialize(&mut self) -> Result<(), OsStatusError> {
        // SAFETY: the unit is live; the caller has configured it.
        let status = unsafe { AudioUnitInitialize(self.unit) };
        check(status, "initialize the AUHAL unit")?;
        self.initialized = true;
        Ok(())
    }

    pub(crate) fn start(&mut self) -> Result<(), OsStatusError> {
        // SAFETY: the unit is live and initialized.
        check(
            unsafe { AudioOutputUnitStart(self.unit) },
            "start the device",
        )
    }

    /// Stops the device. Returns once the unit no longer calls back.
    pub(crate) fn stop(&mut self) -> Result<(), OsStatusError> {
        // SAFETY: the unit is live; stopping a stopped unit does nothing.
        check(unsafe { AudioOutputUnitStop(self.unit) }, "stop the device")
    }
}

impl Drop for HalUnit {
    fn drop(&mut self) {
        // SAFETY: the unit is live; each call is best-effort teardown in the
        // order Core Audio documents — stop, uninitialize, dispose — after
        // which nothing calls its callbacks again.
        unsafe {
            AudioOutputUnitStop(self.unit);
            if self.initialized {
                AudioUnitUninitialize(self.unit);
            }
            AudioComponentInstanceDispose(self.unit);
        }
    }
}

/// An `Arc` handed to Core Audio as a callback's context: one strong
/// reference, taken back when this drops.
///
/// Declared after the [`HalUnit`] whose callback reads it, in the struct
/// holding both, so the unit is disposed of — and the callback done with
/// for good — before this lets go.
pub(crate) struct Refcon<T>(NonNull<T>);

// SAFETY: this is one reference of an `Arc<T>`, which is `Send` for a `T`
// that is `Send + Sync`.
unsafe impl<T: Send + Sync> Send for Refcon<T> {}

impl<T> Refcon<T> {
    pub(crate) fn new(value: Arc<T>) -> Self {
        // SAFETY: `Arc::into_raw` never returns null.
        Self(unsafe { NonNull::new_unchecked(Arc::into_raw(value).cast_mut()) })
    }

    /// What Core Audio is given, and hands the callback back.
    pub(crate) fn as_ptr(&self) -> *mut c_void {
        self.0.as_ptr().cast::<c_void>()
    }
}

impl<T> Drop for Refcon<T> {
    fn drop(&mut self) {
        // SAFETY: the pointer came from `Arc::into_raw` in `new`, and is
        // given back exactly once.
        drop(unsafe { Arc::from_raw(self.0.as_ptr()) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whatever the machine has, what is listed can be asked again: every
    /// device plays at a rate and has channels, and at most one is the
    /// default. A machine with no output lists nothing, which says nothing.
    /// Whatever the machine has, what is listed can be asked again: every
    /// device runs at a rate and has channels in the list's direction, and
    /// at most one is the default. A machine with none lists nothing, which
    /// says nothing.
    fn listed_devices_answer_for_their_format(
        devices: Vec<CoreAudioDevice>,
        scope: AudioObjectPropertyScope,
    ) {
        if devices.is_empty() {
            eprintln!("skipping: no Core Audio device in this direction");
            return;
        }
        for device in &devices {
            assert!(!device.uid.is_empty(), "{device:?} has a UID");
            assert!(channels(device.id, scope).unwrap() > 0, "{device:?}");
            assert!(nominal_sample_rate(device.id).unwrap() > 0.0, "{device:?}");
        }
        assert!(devices.iter().filter(|device| device.is_default).count() <= 1);
    }

    #[cfg(feature = "coreaudio-renderer")]
    #[test]
    fn listed_outputs_answer_for_their_format() {
        listed_devices_answer_for_their_format(
            list_output_devices().expect("the HAL lists its devices"),
            kAudioObjectPropertyScopeOutput,
        );
    }

    #[cfg(feature = "coreaudio-capture")]
    #[test]
    fn listed_inputs_answer_for_their_format() {
        listed_devices_answer_for_their_format(
            list_input_devices().expect("the HAL lists its devices"),
            kAudioObjectPropertyScopeInput,
        );
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
