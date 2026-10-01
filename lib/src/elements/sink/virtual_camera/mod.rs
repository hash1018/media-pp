//! A pipeline's pictures as a camera other applications can open.
//!
//! [`MfVirtualCamera`] on Windows; what it and the camera it registers
//! share is `protocol`, the one file both compile.

#[cfg(all(target_os = "windows", feature = "mf-virtual-camera"))]
mod mf_virtual_camera;
#[cfg(all(target_os = "windows", feature = "mf-virtual-camera"))]
// Shared with the camera DLL, which uses the half this side does not.
#[allow(dead_code)]
pub(crate) mod protocol;

#[cfg(all(target_os = "windows", feature = "mf-virtual-camera"))]
pub use mf_virtual_camera::{MfVirtualCamera, MfVirtualCameraError};
