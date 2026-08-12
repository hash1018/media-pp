//! Shared Windows Audio Session API infrastructure.

mod device;
mod format;

pub use device::{WasapiDevice, WasapiDeviceKind};

pub(crate) use device::{ComApartment, list_devices, open_device};
pub(crate) use format::resolve_mix_format;
