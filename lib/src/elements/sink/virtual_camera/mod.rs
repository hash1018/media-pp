//! A pipeline's pictures as a camera other applications can open.
//!
//! [`MfVirtualCamera`] on Windows; what it and the camera it registers
//! share is `protocol`, the one file both compile. [`V4l2VirtualCamera`] on
//! Linux, writing into a v4l2loopback device.

#[cfg(all(target_os = "windows", feature = "mf-virtual-camera"))]
mod mf_virtual_camera;
#[cfg(all(target_os = "windows", feature = "mf-virtual-camera"))]
// Shared with the camera DLL, which uses the half this side does not.
#[allow(dead_code)]
pub(crate) mod protocol;

#[cfg(all(target_os = "linux", feature = "v4l2-virtual-camera"))]
mod v4l2_virtual_camera;

#[cfg(all(target_os = "windows", feature = "mf-virtual-camera"))]
pub use mf_virtual_camera::{MfVirtualCamera, MfVirtualCameraError};
#[cfg(all(target_os = "linux", feature = "v4l2-virtual-camera"))]
pub use v4l2_virtual_camera::{V4l2VirtualCamera, V4l2VirtualCameraError};
