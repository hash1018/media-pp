use std::ffi::{CStr, c_char};

use ash::vk;
use raw_window_handle::RawDisplayHandle;

use super::cuda_ffi::{
    CUDA_SUCCESS, CUcontext, CUdevice, cuDeviceGet, cuDeviceGetUuid, cuDevicePrimaryCtxRelease_v2,
    cuDevicePrimaryCtxRetain, cuInit, cuda_error,
};

/// The process-wide Vulkan device this stack renders with, paired with the
/// CUDA context it is allowed to import memory into — the Linux sibling of
/// [`crate::D3d11GpuContext`]. Create one per stack and share it across
/// every window.
///
/// # The pairing is verified, not assumed
///
/// Importing Vulkan memory into CUDA only works when both refer to the same
/// physical GPU. Rather than taking the first Vulkan device and hoping, this
/// picks the one whose `VkPhysicalDeviceIDProperties::deviceUUID` matches
/// the CUDA device's own UUID. On a laptop with an integrated GPU listed
/// first, "just take device 0" would otherwise produce an import that fails
/// far from its cause — or, worse, succeeds against the wrong memory.
///
/// The CUDA context here is the device's *primary* context, which is what
/// `media_pp::elements::CudaDevice` also makes FFmpeg use. That is what
/// makes this pairing possible without mirroring any FFmpeg struct.
pub struct VulkanGpuContext {
    pub(super) entry: ash::Entry,
    pub(super) instance: ash::Instance,
    pub(super) physical_device: vk::PhysicalDevice,
    pub(super) device: ash::Device,
    pub(super) queue: vk::Queue,
    pub(super) queue_family: u32,
    pub(super) memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub(super) cuda_ctx: CUcontext,
    cuda_device: CUdevice,
}

impl VulkanGpuContext {
    /// `display` comes from the windowing library (winit's
    /// `HasDisplayHandle`), and only decides which surface extension is
    /// enabled — Wayland or X11.
    pub fn new(display: RawDisplayHandle) -> Result<Self, String> {
        let (cuda_device, cuda_ctx, cuda_uuid) = init_cuda()?;

        let entry = unsafe { ash::Entry::load() }
            .map_err(|e| format!("failed to load the Vulkan loader: {e}"))?;

        let app_name = c"media-pp render_common";
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            // 1.1 is where `VkPhysicalDeviceIDProperties` and the external
            // memory core structs live; nothing here needs more.
            .api_version(vk::API_VERSION_1_1);

        let mut extensions = ash_window::enumerate_required_extensions(display)
            .map_err(|e| format!("this display is not supported by Vulkan: {e}"))?
            .to_vec();
        extensions.push(vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_NAME.as_ptr());

        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extensions);
        let instance = unsafe { entry.create_instance(&instance_info, None) }
            .map_err(|e| format!("vkCreateInstance failed: {e}"))?;

        let physical_device = match find_device_with_uuid(&instance, cuda_uuid) {
            Some(device) => device,
            None => {
                unsafe { instance.destroy_instance(None) };
                release_cuda(cuda_device);
                return Err(
                    "no Vulkan device matches the CUDA device's UUID — the CUDA and \
                     graphics stacks are on different GPUs"
                        .into(),
                );
            }
        };

        let queue_family = match find_graphics_queue(&instance, physical_device) {
            Some(index) => index,
            None => {
                unsafe { instance.destroy_instance(None) };
                release_cuda(cuda_device);
                return Err("the matching Vulkan device has no graphics queue".into());
            }
        };

        let device_extensions = [
            vk::KHR_SWAPCHAIN_NAME.as_ptr(),
            vk::KHR_EXTERNAL_MEMORY_FD_NAME.as_ptr(),
        ];
        let priorities = [1.0f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&device_extensions);
        let device = match unsafe { instance.create_device(physical_device, &device_info, None) } {
            Ok(device) => device,
            Err(error) => {
                unsafe { instance.destroy_instance(None) };
                release_cuda(cuda_device);
                return Err(format!("vkCreateDevice failed: {error}"));
            }
        };

        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        Ok(Self {
            entry,
            instance,
            physical_device,
            device,
            queue,
            queue_family,
            memory_properties,
            cuda_ctx,
            cuda_device,
        })
    }
}

impl Drop for VulkanGpuContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
        release_cuda(self.cuda_device);
    }
}

/// Brings up CUDA and retains the primary context, returning the device's
/// UUID so the Vulkan side can be matched against it.
fn init_cuda() -> Result<(CUdevice, CUcontext, [u8; 16]), String> {
    unsafe {
        let result = cuInit(0);
        if result != CUDA_SUCCESS {
            return Err(cuda_error("cuInit", result));
        }
        let mut device: CUdevice = 0;
        let result = cuDeviceGet(&mut device, 0);
        if result != CUDA_SUCCESS {
            return Err(cuda_error("cuDeviceGet", result));
        }
        let mut uuid = [0u8; 16];
        let result = cuDeviceGetUuid(&mut uuid, device);
        if result != CUDA_SUCCESS {
            return Err(cuda_error("cuDeviceGetUuid", result));
        }
        let mut ctx: CUcontext = std::ptr::null_mut();
        let result = cuDevicePrimaryCtxRetain(&mut ctx, device);
        if result != CUDA_SUCCESS {
            return Err(cuda_error("cuDevicePrimaryCtxRetain", result));
        }
        Ok((device, ctx, uuid))
    }
}

fn release_cuda(device: CUdevice) {
    unsafe { cuDevicePrimaryCtxRelease_v2(device) };
}

fn find_device_with_uuid(instance: &ash::Instance, uuid: [u8; 16]) -> Option<vk::PhysicalDevice> {
    let devices = unsafe { instance.enumerate_physical_devices() }.ok()?;
    devices.into_iter().find(|&device| {
        let mut id_properties = vk::PhysicalDeviceIDProperties::default();
        let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut id_properties);
        unsafe { instance.get_physical_device_properties2(device, &mut properties) };
        id_properties.device_uuid == uuid
    })
}

fn find_graphics_queue(instance: &ash::Instance, device: vk::PhysicalDevice) -> Option<u32> {
    let families = unsafe { instance.get_physical_device_queue_family_properties(device) };
    families
        .iter()
        .position(|family| family.queue_flags.contains(vk::QueueFlags::GRAPHICS))
        .map(|index| index as u32)
}

/// Names a Vulkan device for a log line.
pub(super) fn device_name(instance: &ash::Instance, device: vk::PhysicalDevice) -> String {
    let properties = unsafe { instance.get_physical_device_properties(device) };
    let name: &[c_char] = &properties.device_name;
    unsafe { CStr::from_ptr(name.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}
