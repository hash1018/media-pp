//! The one Vulkan device a program's `VulkanWindowRenderer`s draw with, and
//! the one lock around its queue — the Linux sibling of `D3d11Gpu`.

use std::{
    ffi::{CStr, c_char},
    sync::{Arc, Mutex, MutexGuard},
};

use ash::vk;
use thiserror::Error as ThisError;

/// Why a [`VulkanGpu`] could not be set up.
#[derive(Debug, ThisError)]
pub enum VulkanGpuError {
    /// There is no Vulkan loader (`libvulkan.so.1`) to open.
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

    /// This Vulkan cannot draw into a window of any kind: it has no
    /// `VK_KHR_surface`, or no X11 or Wayland surface to go with it.
    #[error("this Vulkan cannot present into a window")]
    NoSurfaceSupport,

    /// No device has a queue that draws.
    #[error("no Vulkan device can draw")]
    NoDevice,

    /// The device lacks an extension presenting needs.
    #[error("the Vulkan device {device} lacks {extension}")]
    MissingExtension {
        /// The device that was chosen.
        device: String,
        /// What it lacks.
        extension: &'static str,
    },

    /// No Vulkan device is the GPU CUDA decodes on, so there is no device a
    /// CUDA frame could be copied into.
    #[cfg(feature = "cuda")]
    #[error("no Vulkan device is the GPU the CUDA frames are on")]
    NoCudaDevice,

    /// CUDA would not say which GPU it is on.
    #[cfg(feature = "cuda")]
    #[error(transparent)]
    Cuda(#[from] crate::platform::cuda::CudaDriverError),
}

/// The Vulkan device a program's window renderers share, and the one lock
/// around its queue.
///
/// Every submission to a Vulkan queue, and every present, must be externally
/// synchronized: two renderers submitting on one queue from their own
/// threads at once is undefined. So the queue is only reached through one
/// lock, held for a renderer's whole submit-present sequence — the
/// reason `D3d11Gpu` hands out its context behind one `Arc<Mutex<_>>`.
///
/// Two ways to make one, which decide what frames a renderer on it takes:
///
/// - [`Self::new`] — any GPU that draws, integrated or not. Frames in system
///   memory only.
/// - `for_cuda`, with the `cuda` feature — the GPU a `CudaDevice` decodes
///   on, and only that one. Frames in system memory and CUDA frames both:
///   a CUDA frame is copied into memory this device allocates, and that is
///   only possible within one GPU. The pairing is checked by UUID rather than
///   assumed from the order devices are listed in, which on a laptop with an
///   integrated GPU first would pick the wrong one.
///
/// Cloning is cheap and every clone is the same device and the same lock.
#[derive(Clone)]
pub struct VulkanGpu {
    shared: Arc<VulkanShared>,
}

impl std::fmt::Debug for VulkanGpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanGpu")
            .field("device", &self.shared.name)
            .finish_non_exhaustive()
    }
}

/// What every clone of a [`VulkanGpu`] refers to.
pub(crate) struct VulkanShared {
    pub(crate) entry: ash::Entry,
    pub(crate) instance: ash::Instance,
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) device: ash::Device,
    queue: Mutex<vk::Queue>,
    pub(crate) queue_family: u32,
    pub(crate) memory_properties: vk::PhysicalDeviceMemoryProperties,
    /// The device's own name, for a log line.
    pub(crate) name: String,
    /// The CUDA context frames are copied out of — set by
    /// [`VulkanGpu::for_cuda`] only.
    #[cfg(feature = "cuda")]
    pub(crate) cuda: Option<CudaPairing>,
}

/// What a device made for CUDA keeps of it.
#[cfg(feature = "cuda")]
pub(crate) struct CudaPairing {
    /// Copies into memory this device allocated.
    pub(crate) interop: Arc<crate::platform::cuda::driver::interop::CudaInterop>,
    /// The device context a frame must come from — compared, never
    /// dereferenced, as `CudaRenderer` does.
    pub(crate) device_ctx: *const ffmpeg_next::ffi::AVHWDeviceContext,
    /// The reference that keeps `device_ctx` a valid identity for as long as
    /// this lives.
    _hw_device_ctx: Arc<crate::platform::ffmpeg::AvBufferRef>,
}

// SAFETY: `device_ctx` is only ever compared, never dereferenced, and the
// reference beside it keeps the context it names alive; `interop` is Send and
// Sync on its own.
#[cfg(feature = "cuda")]
unsafe impl Send for CudaPairing {}

// SAFETY: as above — nothing here is mutated after construction.
#[cfg(feature = "cuda")]
unsafe impl Sync for CudaPairing {}

impl VulkanShared {
    /// The queue, for one submit-present sequence. Held for all of it.
    pub(crate) fn queue(&self) -> MutexGuard<'_, vk::Queue> {
        self.queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl Drop for VulkanShared {
    fn drop(&mut self) {
        // SAFETY: nothing else refers to the device or instance once the last
        // clone is gone — every renderer holds a clone — so waiting for the
        // device and destroying both, device first, is the end of their use.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// What a device is made to do, and so which extensions it needs.
enum Purpose {
    /// Draw frames that arrive in system memory.
    System,
    /// Draw those, and frames copied in from CUDA on the GPU with this UUID.
    #[cfg(feature = "cuda")]
    Cuda([u8; 16]),
}

impl VulkanGpu {
    /// A device on any GPU that draws, for frames in system memory — a
    /// discrete GPU where there is one.
    pub fn new() -> Result<Self, VulkanGpuError> {
        let shared = create(Purpose::System)?;
        Ok(Self {
            shared: Arc::new(shared),
        })
    }

    /// A device on the GPU `cuda` decodes on, for CUDA frames as well as
    /// frames in system memory.
    ///
    /// Like `CudaDevice` itself, make it once, before the pipelines start:
    /// it retains the device's primary CUDA context, and doing that while
    /// another thread has NVDEC work in flight has crashed inside the
    /// driver.
    #[cfg(feature = "cuda")]
    pub fn for_cuda(cuda: &crate::elements::CudaDevice) -> Result<Self, VulkanGpuError> {
        use crate::platform::cuda::driver::interop::CudaInterop;

        let interop = Arc::new(CudaInterop::retain_primary()?);
        let uuid = interop.uuid()?;
        let mut shared = create(Purpose::Cuda(uuid))?;
        let hw_device_ctx = cuda.retain();
        // SAFETY: `hw_device_ctx` owns a live `AVBufferRef` for a CUDA device
        // context, whose `data` is that `AVHWDeviceContext`. Only the pointer's
        // identity is kept, and the reference beside it keeps it from being
        // reused by another context.
        let device_ctx =
            unsafe { (*hw_device_ctx.as_ptr()).data as *const ffmpeg_next::ffi::AVHWDeviceContext };
        shared.cuda = Some(CudaPairing {
            interop,
            device_ctx,
            _hw_device_ctx: hw_device_ctx,
        });
        Ok(Self {
            shared: Arc::new(shared),
        })
    }

    /// The device's name, as the driver gives it.
    pub fn name(&self) -> &str {
        &self.shared.name
    }

    pub(crate) fn shared(&self) -> &Arc<VulkanShared> {
        &self.shared
    }
}

/// The surface extensions this crate can present through, and one of them
/// must be there: `ash-window` makes a surface for whichever the window is.
const WINDOW_SURFACES: [&CStr; 3] = [
    ash::khr::xlib_surface::NAME,
    ash::khr::xcb_surface::NAME,
    ash::khr::wayland_surface::NAME,
];

fn call(call: &'static str) -> impl FnOnce(vk::Result) -> VulkanGpuError {
    move |result| VulkanGpuError::Call { call, result }
}

fn create(purpose: Purpose) -> Result<VulkanShared, VulkanGpuError> {
    // SAFETY: loads `libvulkan.so.1` and resolves its entry points; the
    // `Entry` it returns keeps the library loaded for as long as it lives,
    // and every clone of the instance below is outlived by it.
    let entry =
        unsafe { ash::Entry::load() }.map_err(|error| VulkanGpuError::Loader(error.to_string()))?;

    // Every window-surface extension this Vulkan has, not the one for a
    // window this does not have yet: a device made here draws into X11
    // windows and Wayland ones alike.
    // SAFETY: a plain query of the loader, with no layer named.
    let available = unsafe { entry.enumerate_instance_extension_properties(None) }
        .map_err(call("vkEnumerateInstanceExtensionProperties"))?;
    let has = |name: &CStr| {
        available
            .iter()
            .any(|extension| extension.extension_name_as_c_str() == Ok(name))
    };
    if !has(ash::khr::surface::NAME) {
        return Err(VulkanGpuError::NoSurfaceSupport);
    }
    let mut extensions: Vec<*const c_char> = vec![ash::khr::surface::NAME.as_ptr()];
    extensions.extend(
        WINDOW_SURFACES
            .iter()
            .filter(|name| has(name))
            .map(|name| name.as_ptr()),
    );
    if extensions.len() == 1 {
        return Err(VulkanGpuError::NoSurfaceSupport);
    }

    let app = vk::ApplicationInfo::default()
        .application_name(c"media-pp")
        // 1.1: where `VkPhysicalDeviceIDProperties` and external memory are
        // core. Nothing here needs more.
        .api_version(vk::API_VERSION_1_1);
    let info = vk::InstanceCreateInfo::default()
        .application_info(&app)
        .enabled_extension_names(&extensions);
    // SAFETY: `info` and everything it points at outlive the call.
    let instance =
        unsafe { entry.create_instance(&info, None) }.map_err(call("vkCreateInstance"))?;

    match open_device(&instance, &purpose) {
        Ok((physical_device, device, queue_family, name)) => {
            // SAFETY: the queue family was chosen from this device's own list
            // and one queue was asked of it.
            let queue = unsafe { device.get_device_queue(queue_family, 0) };
            // SAFETY: a plain query of a device of this instance.
            let memory_properties =
                unsafe { instance.get_physical_device_memory_properties(physical_device) };
            Ok(VulkanShared {
                entry,
                instance,
                physical_device,
                device,
                queue: Mutex::new(queue),
                queue_family,
                memory_properties,
                name,
                #[cfg(feature = "cuda")]
                cuda: None,
            })
        }
        Err(error) => {
            // SAFETY: nothing was made from the instance that outlives this.
            unsafe { instance.destroy_instance(None) };
            Err(error)
        }
    }
}

fn open_device(
    instance: &ash::Instance,
    purpose: &Purpose,
) -> Result<(vk::PhysicalDevice, ash::Device, u32, String), VulkanGpuError> {
    // SAFETY: a plain query of this instance.
    let devices = unsafe { instance.enumerate_physical_devices() }
        .map_err(call("vkEnumeratePhysicalDevices"))?;
    let drawing: Vec<(vk::PhysicalDevice, u32)> = devices
        .into_iter()
        .filter_map(|device| graphics_queue(instance, device).map(|family| (device, family)))
        .collect();

    let (physical_device, queue_family) = match purpose {
        // A discrete GPU where there is one: an integrated one listed first
        // is the ordinary laptop, and the slower of the two.
        Purpose::System => drawing
            .iter()
            .copied()
            .find(|&(device, _)| {
                // SAFETY: a plain query of a device of this instance.
                let properties = unsafe { instance.get_physical_device_properties(device) };
                properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
            })
            .or_else(|| drawing.first().copied())
            .ok_or(VulkanGpuError::NoDevice)?,
        #[cfg(feature = "cuda")]
        Purpose::Cuda(uuid) => drawing
            .iter()
            .copied()
            .find(|&(device, _)| {
                let mut id = vk::PhysicalDeviceIDProperties::default();
                let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
                // SAFETY: `properties` chains a live `id`, which the call fills.
                unsafe { instance.get_physical_device_properties2(device, &mut properties) };
                id.device_uuid == *uuid
            })
            .ok_or(VulkanGpuError::NoCudaDevice)?,
    };
    let name = device_name(instance, physical_device);

    #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
    let mut needed: Vec<&'static CStr> = vec![ash::khr::swapchain::NAME];
    #[cfg(feature = "cuda")]
    if matches!(purpose, Purpose::Cuda(_)) {
        needed.push(ash::khr::external_memory_fd::NAME);
    }
    // SAFETY: a plain query of a device of this instance.
    let available = unsafe { instance.enumerate_device_extension_properties(physical_device) }
        .map_err(call("vkEnumerateDeviceExtensionProperties"))?;
    for extension in &needed {
        if !available
            .iter()
            .any(|have| have.extension_name_as_c_str() == Ok(*extension))
        {
            return Err(VulkanGpuError::MissingExtension {
                device: name,
                extension: extension.to_str().unwrap_or("an extension"),
            });
        }
    }
    let extensions: Vec<*const c_char> = needed.iter().map(|name| name.as_ptr()).collect();
    let priorities = [1.0f32];
    let queues = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family)
        .queue_priorities(&priorities)];
    let info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queues)
        .enabled_extension_names(&extensions);
    // SAFETY: `info` and everything it points at outlive the call, and the
    // queue family and extensions were taken from this device's own lists.
    let device = unsafe { instance.create_device(physical_device, &info, None) }
        .map_err(call("vkCreateDevice"))?;
    Ok((physical_device, device, queue_family, name))
}

/// The first queue family on `device` that draws.
fn graphics_queue(instance: &ash::Instance, device: vk::PhysicalDevice) -> Option<u32> {
    // SAFETY: a plain query of a device of this instance.
    let families = unsafe { instance.get_physical_device_queue_family_properties(device) };
    families
        .iter()
        .position(|family| family.queue_flags.contains(vk::QueueFlags::GRAPHICS))
        .map(|index| index as u32)
}

fn device_name(instance: &ash::Instance, device: vk::PhysicalDevice) -> String {
    // SAFETY: a plain query of a device of this instance.
    let properties = unsafe { instance.get_physical_device_properties(device) };
    properties
        .device_name_as_c_str()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|_| "an unnamed device".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A machine with no Vulkan at all skips rather than fails: nothing here
    /// can be checked without one.
    fn skip(error: &VulkanGpuError) -> bool {
        matches!(
            error,
            VulkanGpuError::Loader(_) | VulkanGpuError::NoSurfaceSupport | VulkanGpuError::NoDevice
        )
    }

    #[test]
    fn a_device_for_system_frames_opens_wherever_vulkan_draws() {
        match VulkanGpu::new() {
            Ok(gpu) => assert!(!gpu.name().is_empty()),
            Err(error) if skip(&error) => eprintln!("skipping: {error}"),
            Err(error) => panic!("Vulkan is here but no device opened: {error}"),
        }
    }

    /// The pairing is by UUID, so what matters is that it finds one — on a
    /// machine with CUDA, the GPU CUDA is on always has a Vulkan device.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_device_for_cuda_is_the_gpu_cuda_is_on() {
        let Some((cuda, _lock)) = crate::test_support::try_cuda_device() else {
            return;
        };
        match VulkanGpu::for_cuda(&cuda) {
            Ok(gpu) => {
                assert!(gpu.shared().cuda.is_some(), "the CUDA pairing is kept");
            }
            Err(error) if skip(&error) => eprintln!("skipping: {error}"),
            Err(error) => panic!("CUDA and Vulkan are both here but did not pair: {error}"),
        }
    }
}
