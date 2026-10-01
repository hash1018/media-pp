//! `media-pp-vcam`: the camera side of media-pp's Windows virtual camera.
//!
//! A COM in-process server holding one Media Foundation media source. A
//! [`MfVirtualCamera`] in an application registers a camera with
//! `MFCreateVirtualCamera` naming this server's CLSID, and Windows' Frame
//! Server loads this DLL into its own service process to serve every
//! application that opens the camera. The pictures come from the producing
//! application through a section of shared memory — see the protocol module,
//! the one file both sides compile.
//!
//! The DLL has to be registered once, from an elevated prompt, where Frame
//! Server can read it — a service account cannot read most of a user's
//! profile, so not from inside one:
//!
//! ```text
//! copy media_pp_vcam.dll "C:\Program Files\media-pp\"
//! regsvr32 "C:\Program Files\media-pp\media_pp_vcam.dll"
//! ```
//!
//! Windows 11 (build 22000) or newer.
//!
//! [`MfVirtualCamera`]: https://docs.rs/media-pp

#![cfg(windows)]

#[path = "../../lib/src/elements/sink/virtual_camera/protocol.rs"]
#[allow(dead_code)]
mod protocol;

mod activator;
mod attributes;
mod section;
mod source;
mod stream;

#[cfg(test)]
mod tests;

use std::ffi::c_void;
use std::sync::atomic::{AtomicPtr, Ordering};

use windows::Win32::{
    Foundation::{
        CLASS_E_CLASSNOTAVAILABLE, CLASS_E_NOAGGREGATION, E_POINTER, ERROR_SUCCESS, HINSTANCE,
        HMODULE, S_FALSE, S_OK,
    },
    System::{
        Com::{IClassFactory, IClassFactory_Impl},
        LibraryLoader::GetModuleFileNameW,
        Registry::{
            HKEY, HKEY_LOCAL_MACHINE, KEY_WRITE, REG_OPTION_NON_VOLATILE, REG_SZ, RegCloseKey,
            RegCreateKeyExW, RegDeleteTreeW, RegSetValueExW,
        },
        SystemServices::DLL_PROCESS_ATTACH,
    },
};
use windows_core::{
    BOOL, ComObject, GUID, HRESULT, HSTRING, IUnknown, Interface, Ref, Result, implement,
};

/// A COM object Media Foundation documents as free-threaded, carried to
/// another thread. The `windows` crate cannot know which interfaces are,
/// so it marks none of them `Send`.
pub(crate) struct Agile<T>(pub(crate) T);

// SAFETY: only ever wraps the event queue, sample allocator, samples and
// request tokens, which Media Foundation creates free-threaded.
unsafe impl<T> Send for Agile<T> {}
// SAFETY: as above.
unsafe impl<T> Sync for Agile<T> {}

/// This DLL's module handle, for finding its own path to register.
static MODULE: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());

const CLSID: GUID = GUID::from_u128(protocol::CLSID_U128);

#[unsafe(no_mangle)]
extern "system" fn DllMain(module: HINSTANCE, reason: u32, _reserved: *mut c_void) -> BOOL {
    if reason == DLL_PROCESS_ATTACH {
        MODULE.store(module.0, Ordering::Release);
    }
    true.into()
}

/// # Safety
///
/// Called by COM with valid pointers, as `DllGetClassObject` is.
#[unsafe(no_mangle)]
unsafe extern "system" fn DllGetClassObject(
    class: *const GUID,
    iid: *const GUID,
    object: *mut *mut c_void,
) -> HRESULT {
    if class.is_null() || iid.is_null() || object.is_null() {
        return E_POINTER;
    }
    // SAFETY: checked non-null above.
    unsafe { *object = std::ptr::null_mut() };
    // SAFETY: as above.
    if unsafe { *class } != CLSID {
        return CLASS_E_CLASSNOTAVAILABLE;
    }
    let factory: IClassFactory = ComObject::new(Factory).to_interface();
    // SAFETY: the caller's `iid` and `object`, passed to the factory's own
    // `QueryInterface`.
    unsafe { factory.query(iid, object) }
}

/// Never unloaded: Frame Server holds the source for as long as a camera is
/// open, and a stale delivery thread in an unloaded DLL would crash it.
#[unsafe(no_mangle)]
extern "system" fn DllCanUnloadNow() -> HRESULT {
    S_FALSE
}

#[unsafe(no_mangle)]
extern "system" fn DllRegisterServer() -> HRESULT {
    match register() {
        Ok(()) => S_OK,
        Err(error) => error.code(),
    }
}

#[unsafe(no_mangle)]
extern "system" fn DllUnregisterServer() -> HRESULT {
    // SAFETY: deletes this server's own key and nothing else.
    let result = unsafe { RegDeleteTreeW(HKEY_LOCAL_MACHINE, &HSTRING::from(class_key())) };
    if result == ERROR_SUCCESS || result.0 == 2 {
        S_OK
    } else {
        result.to_hresult()
    }
}

fn class_key() -> String {
    format!("Software\\Classes\\CLSID\\{}", protocol::CLSID)
}

/// Writes `HKLM\Software\Classes\CLSID\{…}\InprocServer32`: this DLL's path,
/// and `Both` as its threading model, since the source is free-threaded.
fn register() -> Result<()> {
    let mut path = vec![0u16; 1024];
    let module = HMODULE(MODULE.load(Ordering::Acquire));
    // SAFETY: `path` is a buffer of the length passed.
    let length = unsafe { GetModuleFileNameW(Some(module), &mut path) } as usize;
    if length == 0 || length >= path.len() {
        return Err(windows_core::Error::from_thread());
    }
    path.truncate(length);
    let mut key = HKEY::default();
    // SAFETY: creates or opens one key under HKLM, closed below.
    unsafe {
        RegCreateKeyExW(
            HKEY_LOCAL_MACHINE,
            &HSTRING::from(format!("{}\\InprocServer32", class_key())),
            None,
            None,
            REG_OPTION_NON_VOLATILE,
            KEY_WRITE,
            None,
            &mut key,
            None,
        )
        .ok()?;
    }
    let result = set_string(key, None, &path).and_then(|()| {
        let model: Vec<u16> = "Both".encode_utf16().collect();
        set_string(key, Some("ThreadingModel"), &model)
    });
    // SAFETY: the key opened above.
    let _ = unsafe { RegCloseKey(key) };
    result
}

/// Sets a `REG_SZ` value — `name`, or the key's default — to `value`.
fn set_string(key: HKEY, name: Option<&str>, value: &[u16]) -> Result<()> {
    let mut data: Vec<u8> = value.iter().flat_map(|unit| unit.to_le_bytes()).collect();
    data.extend_from_slice(&[0, 0]);
    let name = name.map(HSTRING::from);
    // SAFETY: `data` is a null-terminated UTF-16 string as bytes.
    unsafe {
        RegSetValueExW(
            key,
            name.as_ref().map_or(windows_core::PCWSTR::null(), |name| {
                windows_core::PCWSTR(name.as_ptr())
            }),
            None,
            REG_SZ,
            Some(&data),
        )
        .ok()
    }
}

/// Makes an [`activator::Activator`] for each `CreateInstance`.
#[implement(IClassFactory)]
struct Factory;

impl IClassFactory_Impl for Factory_Impl {
    fn CreateInstance(
        &self,
        outer: Ref<IUnknown>,
        iid: *const GUID,
        object: *mut *mut c_void,
    ) -> Result<()> {
        if object.is_null() {
            return Err(E_POINTER.into());
        }
        // SAFETY: checked non-null above.
        unsafe { *object = std::ptr::null_mut() };
        if !outer.is_null() {
            return Err(CLASS_E_NOAGGREGATION.into());
        }
        let activator: IUnknown =
            ComObject::new(activator::Activator::new(protocol::SECTION_NAME)?).to_interface();
        // SAFETY: the caller's `iid` and `object`, passed to the activator's
        // own `QueryInterface`.
        unsafe { activator.query(iid, object).ok() }
    }

    fn LockServer(&self, _lock: BOOL) -> Result<()> {
        Ok(())
    }
}
