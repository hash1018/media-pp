//! Shared Media Foundation infrastructure.

mod device;

pub use device::{MfCaptureFormat, MfDevice};

pub(crate) use device::{
    MfRuntime, frame_rate, frame_size, list_devices, list_formats, open_device_source, open_reader,
};
