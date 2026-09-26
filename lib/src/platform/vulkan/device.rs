//! [`VulkanDevice`]: the one Vulkan device every Vulkan element in a
//! pipeline shares.

use std::{
    ffi::{CStr, CString},
    sync::Arc,
};

use ash::vk;
use ffmpeg_next::{self as ffmpeg, ffi as av};
use thiserror::Error as ThisError;

use crate::platform::ffmpeg::AvBufferRef;

/// Why a [`VulkanDevice`] could not be opened.
#[derive(Debug, ThisError)]
pub enum VulkanDeviceError {
    /// There is no Vulkan loader to open.
    #[error("could not load Vulkan: {0}")]
    Loader(String),

    /// A Vulkan call failed.
    #[error("{call} failed: {result}")]
    Call {
        /// The Vulkan function that failed.
        call: &'static str,
        /// What it answered.
        result: vk::Result,
    },

    /// No GPU has a Vulkan driver.
    #[error("no Vulkan device")]
    NoDevice,

    /// FFmpeg could not open the device it was asked for: among other
    /// reasons, because the device lacks an extension FFmpeg needs.
    #[error("FFmpeg could not open the Vulkan device {device}: {error}")]
    Open {
        /// The device asked for.
        device: String,
        /// What FFmpeg answered.
        error: ffmpeg::Error,
    },

    /// FFmpeg reported success without returning a context reference.
    #[error("FFmpeg opened Vulkan without returning a device context")]
    MissingContext,
}

/// The one Vulkan device every Vulkan element in a pipeline shares.
///
/// The Vulkan counterpart of `CudaDevice`: an image made
/// on one `VkDevice` cannot be used on another, so every element that makes
/// or takes Vulkan frames must be built from the same `VulkanDevice`.
///
/// FFmpeg makes the device, with whatever queues and extensions its Vulkan
/// decoders, encoders and frames need, and owns it. Nothing here creates or
/// destroys the `VkInstance` or `VkDevice`.
///
/// Opens a discrete GPU where there is one — an integrated GPU listed first
/// is the ordinary laptop, and the slower of the two — else the first there
/// is.
///
/// Cloning is cheap and shares the one device, which stays open until the
/// last clone — and every frame made on it — is gone.
#[derive(Clone)]
pub struct VulkanDevice {
    shared: Arc<DeviceShared>,
}

impl std::fmt::Debug for VulkanDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanDevice")
            .field("device", &self.shared.name)
            .finish_non_exhaustive()
    }
}

/// What every clone of a [`VulkanDevice`] refers to.
pub(crate) struct DeviceShared {
    /// The `AVHWDeviceContext` FFmpeg made, which owns the device.
    ctx: Arc<AvBufferRef>,
    /// The device's own name, for a log line.
    pub(crate) name: String,
}

// SAFETY: `ctx` is an FFmpeg reference to a device context that FFmpeg
// itself hands between threads; nothing here is mutated after
// construction.
unsafe impl Send for DeviceShared {}
// SAFETY: as above.
unsafe impl Sync for DeviceShared {}

impl VulkanDevice {
    /// Opens a Vulkan device — see the type's docs for which.
    pub fn new() -> Result<Self, VulkanDeviceError> {
        let (index, name) = choose_device()?;
        let selector = CString::new(index.to_string()).expect("a number has no NUL");
        let mut ctx: *mut av::AVBufferRef = std::ptr::null_mut();
        // SAFETY: `ctx` is a live local FFmpeg writes the allocated context
        // into; `selector` outlives the call and is the documented device
        // string, an index into the devices Vulkan lists; there are no
        // options and no flags.
        let result = unsafe {
            av::av_hwdevice_ctx_create(
                &mut ctx,
                av::AVHWDeviceType::AV_HWDEVICE_TYPE_VULKAN,
                selector.as_ptr(),
                std::ptr::null_mut(),
                0,
            )
        };
        if result < 0 {
            return Err(VulkanDeviceError::Open {
                device: name,
                error: ffmpeg::Error::from(result),
            });
        }
        // SAFETY: `av_hwdevice_ctx_create` left `ctx` owning one reference and
        // nothing else has taken it; a failure left it null and returned above.
        let ctx = unsafe { AvBufferRef::from_raw(ctx) }.ok_or(VulkanDeviceError::MissingContext)?;

        Ok(Self {
            shared: Arc::new(DeviceShared {
                ctx: Arc::new(ctx),
                name,
            }),
        })
    }

    /// The device's name, as the driver gives it.
    pub fn name(&self) -> &str {
        &self.shared.name
    }

    /// Another reference to FFmpeg's device context, for an element to keep
    /// — and to make its frames with.
    pub(crate) fn retain(&self) -> Arc<AvBufferRef> {
        Arc::clone(&self.shared.ctx)
    }

    /// FFmpeg's context for the device, to compare a frame's against — see
    /// `frames::sw_format_of`. Kept alive by this value's reference.
    pub(crate) fn device_ctx(&self) -> *const av::AVHWDeviceContext {
        // SAFETY: a device context's `data` is its `AVHWDeviceContext`; only
        // the pointer is taken, never read through here.
        unsafe { (*self.shared.ctx.as_ptr()).data as *const av::AVHWDeviceContext }
    }
}

/// Which device to open — see [`VulkanDevice`] — as its index in the order
/// Vulkan lists devices, which is the order FFmpeg picks from, and its name.
fn choose_device() -> Result<(usize, String), VulkanDeviceError> {
    // SAFETY: loads the Vulkan loader and resolves its entry points; the
    // `Entry` keeps it loaded for as long as the instance below lives.
    let entry = unsafe { ash::Entry::load() }
        .map_err(|error| VulkanDeviceError::Loader(error.to_string()))?;
    let app = vk::ApplicationInfo::default()
        .application_name(c"media-pp")
        .api_version(vk::API_VERSION_1_1);
    let info = vk::InstanceCreateInfo::default().application_info(&app);
    // SAFETY: `info` and everything it points at outlive the call.
    let instance = unsafe { entry.create_instance(&info, None) }.map_err(|result| {
        VulkanDeviceError::Call {
            call: "vkCreateInstance",
            result,
        }
    })?;
    // SAFETY: plain queries of this instance, destroyed once they are done
    // and nothing made from it survives.
    unsafe {
        let chosen = instance
            .enumerate_physical_devices()
            .map_err(|result| VulkanDeviceError::Call {
                call: "vkEnumeratePhysicalDevices",
                result,
            })
            .and_then(|devices| {
                let properties: Vec<_> = devices
                    .iter()
                    .map(|&device| instance.get_physical_device_properties(device))
                    .collect();
                properties
                    .iter()
                    .position(|p| p.device_type == vk::PhysicalDeviceType::DISCRETE_GPU)
                    .or((!properties.is_empty()).then_some(0))
                    .map(|index| {
                        let name = properties[index]
                            .device_name_as_c_str()
                            .map(CStr::to_string_lossy)
                            .map_or_else(
                                |_| "an unnamed device".to_owned(),
                                |name| name.into_owned(),
                            );
                        (index, name)
                    })
                    .ok_or(VulkanDeviceError::NoDevice)
            });
        instance.destroy_instance(None);
        chosen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Clones share the one device, which outlives the value it was made
    /// from.
    #[test]
    fn a_clone_is_the_same_device() {
        let Some(device) = crate::test_support::try_vulkan_device() else {
            return;
        };
        let clone = device.clone();
        drop(device);
        assert!(Arc::ptr_eq(&clone.retain(), &clone.clone().retain()));
        assert!(!clone.name().is_empty());
    }
}
