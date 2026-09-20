//! Shared Windows Audio Session API infrastructure.

mod device;
mod format;
// Only a capture activates a process loopback or lists what could be one; a
// renderer builds on this module too, and would carry both as dead code.
#[cfg(feature = "wasapi-capture")]
mod process;
#[cfg(feature = "wasapi-capture")]
mod session;

pub use device::{WasapiDevice, WasapiDeviceKind};
#[cfg(feature = "wasapi-capture")]
pub use session::WasapiProcess;

pub(crate) use super::com::ComApartment;
pub(crate) use device::{list_devices, open_device};
pub(crate) use format::resolve_mix_format;
#[cfg(feature = "wasapi-capture")]
pub(crate) use process::activate as activate_process_loopback;
#[cfg(feature = "wasapi-capture")]
pub(crate) use session::list_processes;
