//! WASAPI endpoint enumeration and COM apartment management.

use windows::{
    Win32::{
        Devices::FunctionDiscovery::PKEY_Device_FriendlyName,
        Foundation::RPC_E_CHANGED_MODE,
        Media::Audio::{
            DEVICE_STATE_ACTIVE, IMMDevice, IMMDeviceEnumerator, MMDeviceEnumerator, eCapture,
            eConsole, eRender,
        },
        System::{
            Com::{
                CLSCTX_ALL, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx, CoUninitialize,
                STGM_READ,
                StructuredStorage::{PROPVARIANT, PropVariantClear},
            },
            Variant::VT_LPWSTR,
        },
        UI::Shell::PropertiesSystem::IPropertyStore,
    },
    core::HSTRING,
};

/// Which direction a WASAPI endpoint flows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WasapiDeviceKind {
    /// Speakers, headphones, HDMI audio, or another playback endpoint.
    Render,
    /// A microphone or another recording endpoint.
    Capture,
}

/// One active WASAPI endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WasapiDevice {
    /// Opaque `IMMDevice::GetId` value.
    pub id: String,
    /// Human-readable endpoint name, falling back to `id` when Windows
    /// does not expose a friendly name.
    pub name: String,
    pub kind: WasapiDeviceKind,
    /// Whether this was the default console endpoint for `kind` when it
    /// was enumerated.
    pub is_default: bool,
}

/// Balances one successful `CoInitializeEx` on the current thread.
pub(crate) struct ComApartment {
    uninitialize: bool,
}

impl ComApartment {
    pub(crate) fn new() -> windows::core::Result<Self> {
        // SAFETY: initializes COM for the current thread with no reserved
        // pointer; the successful initialization is balanced in `Drop`.
        let result = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
        if result == RPC_E_CHANGED_MODE {
            // The caller already initialized this thread as STA. COM is
            // available and WASAPI works there; only the apartment model
            // cannot be changed, and this call must not be balanced.
            return Ok(Self {
                uninitialize: false,
            });
        }
        result.ok()?;
        Ok(Self { uninitialize: true })
    }
}

impl Drop for ComApartment {
    fn drop(&mut self) {
        if self.uninitialize {
            // SAFETY: this instance records a successful `CoInitializeEx` on
            // this same thread, so this call balances it exactly once.
            unsafe { CoUninitialize() };
        }
    }
}

pub(crate) fn list_devices(
    kind_filter: Option<WasapiDeviceKind>,
) -> windows::core::Result<Vec<WasapiDevice>> {
    let _apartment = ComApartment::new()?;
    // SAFETY: COM is initialized on this thread and the registered class is
    // requested as its documented `IMMDeviceEnumerator` interface.
    let enumerator: IMMDeviceEnumerator =
        unsafe { CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)? };

    let kinds: &[(windows::Win32::Media::Audio::EDataFlow, WasapiDeviceKind)] = match kind_filter {
        Some(WasapiDeviceKind::Render) => &[(eRender, WasapiDeviceKind::Render)],
        Some(WasapiDeviceKind::Capture) => &[(eCapture, WasapiDeviceKind::Capture)],
        None => &[
            (eRender, WasapiDeviceKind::Render),
            (eCapture, WasapiDeviceKind::Capture),
        ],
    };

    let mut devices = Vec::new();
    for &(dataflow, kind) in kinds {
        // SAFETY: the enumerator is live and both enums are valid WASAPI
        // values; the returned interface and allocated ID own their lifetimes.
        let default_id = unsafe { enumerator.GetDefaultAudioEndpoint(dataflow, eConsole) }
            .ok()
            // SAFETY: `device` is the live endpoint returned above; `GetId`
            // returns a COM-allocated NUL-terminated string wrapper.
            .and_then(|device| unsafe { device.GetId() }.ok())
            // SAFETY: the returned `PWSTR` is NUL-terminated and remains valid
            // for this conversion while its wrapper is alive.
            .and_then(|id| unsafe { id.to_string() }.ok());

        // SAFETY: the enumerator is live and the flags/enums are documented
        // values; the returned collection owns its COM reference.
        let collection = unsafe { enumerator.EnumAudioEndpoints(dataflow, DEVICE_STATE_ACTIVE)? };
        // SAFETY: `collection` is live and `GetCount` has no pointer inputs.
        let count = unsafe { collection.GetCount()? };
        for index in 0..count {
            // SAFETY: `index` is bounded by the count obtained from this same
            // live collection.
            let device = unsafe { collection.Item(index)? };
            // SAFETY: `device` is a live endpoint and `GetId` returns its
            // allocated, NUL-terminated identifier.
            let Some(id) = unsafe { device.GetId() }
                .ok()
                // SAFETY: the identifier wrapper remains alive for the string
                // conversion and guarantees a NUL terminator.
                .and_then(|id| unsafe { id.to_string() }.ok())
            else {
                continue;
            };
            let name = device_friendly_name(&device).unwrap_or_else(|| id.clone());
            let is_default = default_id.as_deref() == Some(id.as_str());
            devices.push(WasapiDevice {
                id,
                name,
                kind,
                is_default,
            });
        }
    }
    Ok(devices)
}

pub(crate) fn open_device(id: &str) -> windows::core::Result<IMMDevice> {
    // SAFETY: callers establish a COM apartment; the registered class is
    // requested as its documented `IMMDeviceEnumerator` interface.
    let enumerator: IMMDeviceEnumerator =
        unsafe { CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)? };
    let id = HSTRING::from(id);
    // SAFETY: `id` is a live Windows string for the duration of the call and
    // `enumerator` is a live COM interface.
    unsafe { enumerator.GetDevice(&id) }
}

fn device_friendly_name(device: &IMMDevice) -> Option<String> {
    // SAFETY: `device` and the returned property store are live COM
    // interfaces. `variant` is initialized by `GetValue` and cleared exactly
    // once after its borrowed string value is copied.
    unsafe {
        let store: IPropertyStore = device.OpenPropertyStore(STGM_READ).ok()?;
        let mut variant: PROPVARIANT = store.GetValue(&PKEY_Device_FriendlyName).ok()?;
        let name = property_variant_to_string(&variant);
        let _ = PropVariantClear(&mut variant);
        name
    }
}

fn property_variant_to_string(variant: &PROPVARIANT) -> Option<String> {
    // SAFETY: the caller passes an initialized `PROPVARIANT`; after checking
    // `VT_LPWSTR`, reading the matching union member yields its NUL-terminated
    // string pointer, which is borrowed only for this conversion.
    unsafe {
        if variant.Anonymous.Anonymous.vt != VT_LPWSTR {
            return None;
        }
        variant
            .Anonymous
            .Anonymous
            .Anonymous
            .pwszVal
            .to_string()
            .ok()
    }
}
