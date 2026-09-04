//! What V4L2 is asked directly, which is only what FFmpeg's own demuxer
//! cannot answer — see [`device`].

mod device;

pub use device::{V4l2CaptureFormat, V4l2Device, format_name_for, list_devices, list_formats};
