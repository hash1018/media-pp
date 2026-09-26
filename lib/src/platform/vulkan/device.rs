//! [`VulkanDevice`]: the one Vulkan device every Vulkan element in a
//! pipeline shares.

use std::{
    ffi::{CStr, CString},
    sync::Arc,
};

use ash::vk::{self, Handle};
use ffmpeg_next::{self as ffmpeg, ffi as av};
use thiserror::Error as ThisError;

use super::ffi::AVVulkanDeviceContext;
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

    /// FFmpeg made the device without a queue that computes, which this
    /// crate's own Vulkan work is submitted to.
    #[error("the Vulkan device {device} has no compute queue")]
    NoComputeQueue {
        /// The device that was opened.
        device: String,
    },
}

/// The one Vulkan device every Vulkan element in a pipeline shares.
///
/// The Vulkan counterpart of `CudaDevice`: an image made
/// on one `VkDevice` cannot be used on another, so every element that makes
/// or takes Vulkan frames must be built from the same `VulkanDevice`.
///
/// FFmpeg makes the device, with whatever queues and extensions its Vulkan
/// decoders, encoders and frames need, and owns it; this crate's own Vulkan
/// work runs on it too, submitted to a queue FFmpeg made, under FFmpeg's own
/// lock for that queue. Nothing here creates or destroys the `VkInstance` or
/// `VkDevice`.
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
    pub(crate) device: ash::Device,
    pub(crate) memory_properties: vk::PhysicalDeviceMemoryProperties,
    /// The queue family this crate submits its own work to: one FFmpeg
    /// made, that computes.
    pub(crate) compute_family: u32,
    /// The device's own name, for a log line.
    pub(crate) name: String,
}

// SAFETY: `ctx` is an FFmpeg reference to a device context that FFmpeg
// itself hands between threads, and the ash tables are plain function
// pointers and handles; nothing here is mutated after construction.
unsafe impl Send for DeviceShared {}
// SAFETY: as above. Queue access, which Vulkan requires to be externally
// synchronized, goes through FFmpeg's own lock — see `DeviceShared::queue`.
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
        // SAFETY: a Vulkan device context's `data` is its `AVHWDeviceContext`,
        // whose `hwctx` is the `AVVulkanDeviceContext` FFmpeg filled in as it
        // made the device, and the reference just taken keeps both alive.
        let hwctx = unsafe {
            let device_ctx = (*ctx.as_ptr()).data as *const av::AVHWDeviceContext;
            &*((*device_ctx).hwctx as *const AVVulkanDeviceContext)
        };
        let get_instance_proc_addr = hwctx
            .get_proc_addr
            .ok_or(VulkanDeviceError::MissingContext)?;
        // SAFETY: both are `PFN_vkGetInstanceProcAddr`, the one generated from
        // the header FFmpeg was built with and the other ash's; `VKAPI_PTR`,
        // the calling convention the header declares it with, is what
        // `extern "system"` names.
        let get_instance_proc_addr = unsafe {
            std::mem::transmute::<
                unsafe extern "C" fn(
                    super::ffi::VkInstance,
                    *const std::ffi::c_char,
                ) -> super::ffi::PFN_vkVoidFunction,
                vk::PFN_vkGetInstanceProcAddr,
            >(get_instance_proc_addr)
        };
        let static_fn = ash::StaticFn {
            get_instance_proc_addr,
        };
        // SAFETY: the function comes from the loader FFmpeg opened, which
        // stays open while its device context does — as long as this lives.
        let entry = unsafe { ash::Entry::from_static_fn(static_fn) };
        let instance_handle = vk::Instance::from_raw(hwctx.inst as u64);
        // SAFETY: `inst` is the live instance FFmpeg made through that loader.
        let instance = unsafe { ash::Instance::load(entry.static_fn(), instance_handle) };
        let physical_device = vk::PhysicalDevice::from_raw(hwctx.phys_dev as u64);
        // SAFETY: `act_dev` is the live device FFmpeg made on `phys_dev`.
        let device = unsafe {
            ash::Device::load(
                instance.fp_v1_0(),
                vk::Device::from_raw(hwctx.act_dev as u64),
            )
        };
        // SAFETY: a plain query of the physical device FFmpeg opened.
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let families = usize::try_from(hwctx.nb_qf)
            .unwrap_or(0)
            .min(hwctx.qf.len());
        let compute_family = hwctx.qf[..families]
            .iter()
            .find(|family| {
                family.num > 0
                    && vk::QueueFlags::from_raw(family.flags as u32)
                        .contains(vk::QueueFlags::COMPUTE)
            })
            .and_then(|family| u32::try_from(family.idx).ok())
            .ok_or_else(|| VulkanDeviceError::NoComputeQueue {
                device: name.clone(),
            })?;

        Ok(Self {
            shared: Arc::new(DeviceShared {
                ctx: Arc::new(ctx),
                device,
                memory_properties,
                compute_family,
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

    pub(crate) fn shared(&self) -> &Arc<DeviceShared> {
        &self.shared
    }

    /// FFmpeg's context for the device, to compare a frame's against — see
    /// `frames::sw_format_of`. Kept alive by this value's reference.
    pub(crate) fn device_ctx(&self) -> *const av::AVHWDeviceContext {
        // SAFETY: a device context's `data` is its `AVHWDeviceContext`; only
        // the pointer is taken, never read through here.
        unsafe { (*self.shared.ctx.as_ptr()).data as *const av::AVHWDeviceContext }
    }
}

impl DeviceShared {
    /// Runs `submit` with the first queue of the compute family, holding
    /// FFmpeg's lock for it throughout: FFmpeg submits its own decoding and
    /// encoding to the queues it made, from its own threads, and Vulkan
    /// requires every submission to a queue to be externally synchronized.
    pub(crate) fn queue<R>(&self, submit: impl FnOnce(vk::Queue) -> R) -> R {
        // SAFETY: as in `VulkanDevice::new`; the reference `ctx` keeps the
        // contexts alive.
        let (device_ctx, hwctx) = unsafe {
            let device_ctx = (*self.ctx.as_ptr()).data as *mut av::AVHWDeviceContext;
            (
                device_ctx,
                &*((*device_ctx).hwctx as *const AVVulkanDeviceContext),
            )
        };
        let (lock, unlock) = (hwctx.lock_queue, hwctx.unlock_queue);
        // SAFETY: FFmpeg sets both locking functions as it makes a device (a
        // caller-made one may leave them null, which is why they are optional
        // here); they take the device context they belong to and a queue it
        // made, which the compute family's first queue is.
        unsafe {
            if let Some(lock) = lock {
                lock(device_ctx.cast(), self.compute_family, 0);
            }
        }
        // SAFETY: the family is one FFmpeg made at least one queue of.
        let queue = unsafe { self.device.get_device_queue(self.compute_family, 0) };
        let result = submit(queue);
        // SAFETY: as for `lock`, releasing what was taken above.
        unsafe {
            if let Some(unlock) = unlock {
                unlock(device_ctx.cast(), self.compute_family, 0);
            }
        }
        result
    }

    /// The first memory type among those `bits` allows that has every one of
    /// `properties`.
    pub(crate) fn memory_type(
        &self,
        bits: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        let count = self.memory_properties.memory_type_count as usize;
        self.memory_properties.memory_types[..count]
            .iter()
            .enumerate()
            .find(|(index, kind)| {
                bits & (1 << index) != 0 && kind.property_flags.contains(properties)
            })
            .map(|(index, _)| index as u32)
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
