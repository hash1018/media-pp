//! A Core Audio process tap — what the system, or some of its processes,
//! plays — and the private aggregate device that makes it something to
//! record from, the way a microphone is.
//!
//! Taps arrived in macOS 14.2. The two functions that make and unmake one
//! are looked up when first needed rather than linked, so a program built
//! with this still starts on an older system and hears
//! [`TapError::Unsupported`] only if it asks for a tap.

use std::{
    ffi::{CStr, c_char, c_void},
    ptr::NonNull,
    sync::{
        OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

use objc2::{AnyThread, rc::Retained};
use objc2_core_audio::{
    AudioHardwareCreateAggregateDevice, AudioHardwareDestroyAggregateDevice, AudioObjectID,
    CATapDescription, kAudioAggregateDeviceIsPrivateKey, kAudioAggregateDeviceIsStackedKey,
    kAudioAggregateDeviceMainSubDeviceKey, kAudioAggregateDeviceNameKey,
    kAudioAggregateDeviceSubDeviceListKey, kAudioAggregateDeviceTapAutoStartKey,
    kAudioAggregateDeviceTapListKey, kAudioAggregateDeviceUIDKey, kAudioDevicePropertyDeviceUID,
    kAudioSubDeviceUIDKey, kAudioSubTapDriftCompensationKey, kAudioSubTapUIDKey,
};
use objc2_core_foundation::CFDictionary;
use objc2_foundation::{NSArray, NSDictionary, NSNumber, NSObject, NSString};

use super::{OsStatusError, check, default_output_device, get_string, process_object};

/// Why a tap could not be made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TapError {
    /// The system has no process taps: older than macOS 14.2.
    Unsupported,
    /// None of the processes asked for is a Core Audio client — gone, or
    /// never having played or recorded anything.
    NoSuchProcess(u32),
    /// A Core Audio call failed.
    Status(OsStatusError),
}

impl From<OsStatusError> for TapError {
    fn from(error: OsStatusError) -> Self {
        Self::Status(error)
    }
}

type CreateTap = unsafe extern "C-unwind" fn(*const CATapDescription, *mut AudioObjectID) -> i32;
type DestroyTap = unsafe extern "C-unwind" fn(AudioObjectID) -> i32;

#[derive(Clone, Copy)]
struct TapFunctions {
    create: CreateTap,
    destroy: DestroyTap,
}

unsafe extern "C" {
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

/// dlfcn.h's `RTLD_DEFAULT` on macOS: every image loaded — CoreAudio among
/// them, which this crate links.
const RTLD_DEFAULT: *mut c_void = -2isize as *mut c_void;

/// `AudioHardwareCreateProcessTap` and `AudioHardwareDestroyProcessTap`,
/// where the system has them.
fn tap_functions() -> Option<TapFunctions> {
    static FUNCTIONS: OnceLock<Option<TapFunctions>> = OnceLock::new();
    *FUNCTIONS.get_or_init(|| {
        let find = |name: &CStr| {
            // SAFETY: `name` is a nul-terminated symbol name, and
            // `RTLD_DEFAULT` searches every loaded image.
            NonNull::new(unsafe { dlsym(RTLD_DEFAULT, name.as_ptr()) })
        };
        let create = find(c"AudioHardwareCreateProcessTap")?;
        let destroy = find(c"AudioHardwareDestroyProcessTap")?;
        // SAFETY: both are CoreAudio's own functions of these signatures,
        // as AudioHardware.h declares them.
        Some(unsafe {
            TapFunctions {
                create: std::mem::transmute::<NonNull<c_void>, CreateTap>(create),
                destroy: std::mem::transmute::<NonNull<c_void>, DestroyTap>(destroy),
            }
        })
    })
}

/// A tap and the private aggregate device it records through. Both are
/// this process's alone, and both go when this drops — or when the process
/// ends, whichever is first.
pub(crate) struct ProcessTap {
    aggregate: AudioObjectID,
    tap: AudioObjectID,
    destroy: DestroyTap,
}

impl ProcessTap {
    /// What every process plays to the output device `uid`, in that
    /// device's own channels.
    pub(crate) fn device(uid: &str) -> Result<Self, TapError> {
        let functions = tap_functions().ok_or(TapError::Unsupported)?;
        // SAFETY: an empty exclusion list and a device UID are what this
        // initializer takes; the class exists where the functions do.
        let description = unsafe {
            CATapDescription::initExcludingProcesses_andDeviceUID_withStream(
                CATapDescription::alloc(),
                &NSArray::new(),
                &NSString::from_str(uid),
                0,
            )
        };
        Self::open(functions, &description, uid)
    }

    /// What `pids` play, wherever they play it, mixed to stereo — clocked by
    /// the default output, which is where a process plays unless it chose
    /// otherwise.
    pub(crate) fn processes(pids: &[u32]) -> Result<Self, TapError> {
        let functions = tap_functions().ok_or(TapError::Unsupported)?;
        let objects: Vec<Retained<NSNumber>> = pids
            .iter()
            .filter_map(|&pid| process_object(pid))
            .map(NSNumber::new_u32)
            .collect();
        if objects.is_empty() {
            return Err(TapError::NoSuchProcess(pids.first().copied().unwrap_or(0)));
        }
        let clock = default_output_device()?;
        let clock = get_string(
            clock,
            kAudioDevicePropertyDeviceUID,
            "read the default output's UID",
        )?;
        // SAFETY: a list of process objects is what this initializer takes;
        // the class exists where the functions do.
        let description = unsafe {
            CATapDescription::initStereoMixdownOfProcesses(
                CATapDescription::alloc(),
                &NSArray::from_retained_slice(&objects),
            )
        };
        Self::open(functions, &description, &clock)
    }

    /// Makes the tap `description` says and an aggregate device of it,
    /// clocked by the output device `clock_uid`.
    fn open(
        functions: TapFunctions,
        description: &CATapDescription,
        clock_uid: &str,
    ) -> Result<Self, TapError> {
        static SERIAL: AtomicU64 = AtomicU64::new(0);
        let name = NSString::from_str("media-pp capture");
        // SAFETY: plain property setters on a description this owns.
        unsafe {
            description.setPrivate(true);
            description.setName(&name);
        }
        let mut tap: AudioObjectID = 0;
        // SAFETY: `description` is a live CATapDescription and `tap` a live
        // out-param.
        let status = unsafe { (functions.create)(description, &mut tap) };
        check(status, "create a process tap")?;
        // From here on the tap is this one's to destroy, whatever fails.
        let mut made = Self {
            aggregate: 0,
            tap,
            destroy: functions.destroy,
        };

        let key = |key: &CStr| NSString::from_str(&key.to_string_lossy());
        let object = |value: Retained<NSString>| Retained::into_super(value);
        let flag =
            |value: bool| Retained::into_super(Retained::into_super(NSNumber::new_bool(value)));
        // SAFETY: a plain getter on a description this owns.
        let tap_uid = unsafe { description.UUID() }.UUIDString();
        let tap_entry: Retained<NSDictionary<NSString, NSObject>> =
            NSDictionary::from_retained_objects(
                &[
                    &*key(kAudioSubTapUIDKey),
                    &*key(kAudioSubTapDriftCompensationKey),
                ],
                &[object(tap_uid), flag(true)],
            );
        let clock_entry: Retained<NSDictionary<NSString, NSObject>> =
            NSDictionary::from_retained_objects(
                &[&*key(kAudioSubDeviceUIDKey)],
                &[object(NSString::from_str(clock_uid))],
            );
        let uid = format!(
            "media-pp.capture.{}.{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        );
        let aggregate: Retained<NSDictionary<NSString, NSObject>> =
            NSDictionary::from_retained_objects(
                &[
                    &*key(kAudioAggregateDeviceNameKey),
                    &*key(kAudioAggregateDeviceUIDKey),
                    &*key(kAudioAggregateDeviceMainSubDeviceKey),
                    &*key(kAudioAggregateDeviceIsPrivateKey),
                    &*key(kAudioAggregateDeviceIsStackedKey),
                    &*key(kAudioAggregateDeviceTapAutoStartKey),
                    &*key(kAudioAggregateDeviceSubDeviceListKey),
                    &*key(kAudioAggregateDeviceTapListKey),
                ],
                &[
                    object(name),
                    object(NSString::from_str(&uid)),
                    object(NSString::from_str(clock_uid)),
                    flag(true),
                    flag(false),
                    flag(true),
                    Retained::into_super(NSArray::from_retained_slice(&[clock_entry])),
                    Retained::into_super(NSArray::from_retained_slice(&[tap_entry])),
                ],
            );
        // SAFETY: an NSDictionary is toll-free bridged to the CFDictionary
        // this takes, and it outlives the call.
        let aggregate = unsafe { &*Retained::as_ptr(&aggregate).cast::<CFDictionary>() };
        let mut device: AudioObjectID = 0;
        // SAFETY: the description is a live dictionary and `device` a live
        // out-param.
        let status =
            unsafe { AudioHardwareCreateAggregateDevice(aggregate, NonNull::from(&mut device)) };
        check(status, "create the tap's aggregate device")?;
        made.aggregate = device;
        Ok(made)
    }

    /// The aggregate device to record the tap from.
    pub(crate) fn device_id(&self) -> AudioObjectID {
        self.aggregate
    }
}

impl Drop for ProcessTap {
    fn drop(&mut self) {
        // SAFETY: both were made by `open` and are destroyed once, the
        // aggregate that records the tap first. What records the aggregate
        // is gone by now: its owner drops it before this.
        unsafe {
            if self.aggregate != 0 {
                AudioHardwareDestroyAggregateDevice(self.aggregate);
            }
            (self.destroy)(self.tap);
        }
    }
}
