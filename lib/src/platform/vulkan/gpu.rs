//! What this crate's own Vulkan work is made of, on a [`VulkanDevice`]'s
//! device: compute kernels compiled from WGSL, images and buffers of its own,
//! and one recording submitted and waited for at a time.
//!
//! Every object here holds the device it was made on and destroys itself on
//! drop; none outlives the work it was used for, since a [`Recording`] is
//! waited for before it returns.
//!
//! [`VulkanDevice`]: crate::elements::VulkanDevice

use std::{ffi::CStr, sync::Arc};

use ash::vk;
use thiserror::Error as ThisError;

use super::device::DeviceShared;

/// A Vulkan operation of this crate's own that failed.
#[derive(Debug, ThisError)]
pub enum VulkanError {
    /// A Vulkan call failed.
    #[error("{call} failed: {result}")]
    Call {
        /// The Vulkan function that failed.
        call: &'static str,
        /// What it answered.
        result: vk::Result,
    },

    /// A shader this crate ships did not compile — a defect in this crate,
    /// reported rather than panicked on.
    #[error("a shader did not compile: {0}")]
    Shader(String),

    /// The device has no memory of a kind an allocation needs.
    #[error("the device has no {0} memory for this")]
    NoMemoryType(&'static str),
}

pub(crate) fn call(call: &'static str) -> impl FnOnce(vk::Result) -> VulkanError {
    move |result| VulkanError::Call { call, result }
}

/// `entry` of the WGSL `source`, as SPIR-V holding that entry point alone —
/// so a kernel's descriptor layout is exactly the bindings it uses.
pub(crate) fn compile(source: &str, entry: &str) -> Result<Vec<u32>, VulkanError> {
    let shader = |error: String| VulkanError::Shader(format!("{entry}: {error}"));
    let module = naga::front::wgsl::parse_str(source)
        .map_err(|error| shader(error.emit_to_string(source)))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::IMMEDIATES,
    )
    .validate(&module)
    .map_err(|error| shader(error.emit_to_string(source)))?;
    let pipeline = naga::back::spv::PipelineOptions {
        shader_stage: naga::ShaderStage::Compute,
        entry_point: entry.to_owned(),
    };
    naga::back::spv::write_vec(
        &module,
        &info,
        &naga::back::spv::Options::default(),
        Some(&pipeline),
    )
    .map_err(|error| shader(error.to_string()))
}

/// One compute entry point, ready to dispatch: its pipeline, and the layout
/// of the one descriptor set it reads.
pub(crate) struct Kernel {
    shared: Arc<DeviceShared>,
    pub(crate) set_layout: vk::DescriptorSetLayout,
    pub(crate) pipeline_layout: vk::PipelineLayout,
    pub(crate) pipeline: vk::Pipeline,
}

impl Kernel {
    /// Compiles `entry` of `source` and makes its pipeline: `bindings` are the
    /// descriptors of set 0 it reads, by binding number, and `immediates` the
    /// bytes of push constants it takes.
    pub(crate) fn new(
        shared: &Arc<DeviceShared>,
        source: &str,
        entry: &CStr,
        bindings: &[(u32, vk::DescriptorType)],
        immediates: u32,
    ) -> Result<Self, VulkanError> {
        let spirv = compile(source, entry.to_str().unwrap_or("main"))?;
        let device = &shared.device;
        let layout_bindings: Vec<_> = bindings
            .iter()
            .map(|&(binding, kind)| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding)
                    .descriptor_type(kind)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let mut kernel = Self {
            shared: Arc::clone(shared),
            set_layout: vk::DescriptorSetLayout::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
        };
        // SAFETY: every create info and what it points at outlives its call;
        // whatever is made is stored in `kernel` at once, whose drop destroys
        // it on any later failure.
        unsafe {
            kernel.set_layout = device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&layout_bindings),
                    None,
                )
                .map_err(call("vkCreateDescriptorSetLayout"))?;
            let ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .size(immediates)];
            let set_layouts = [kernel.set_layout];
            let mut layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
            if immediates > 0 {
                layout_info = layout_info.push_constant_ranges(&ranges);
            }
            kernel.pipeline_layout = device
                .create_pipeline_layout(&layout_info, None)
                .map_err(call("vkCreatePipelineLayout"))?;
            let module = device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spirv), None)
                .map_err(call("vkCreateShaderModule"))?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(entry);
            let made = device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(kernel.pipeline_layout)],
                None,
            );
            device.destroy_shader_module(module, None);
            kernel.pipeline = made
                .map_err(|(_, result)| VulkanError::Call {
                    call: "vkCreateComputePipelines",
                    result,
                })?
                .remove(0);
        }
        Ok(kernel)
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        let device = &self.shared.device;
        // SAFETY: made on this device, and nothing recorded with them is still
        // running: every recording is waited for before it returns. Null
        // handles, left by a failed construction, are ignored by Vulkan.
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

/// An image of this crate's own, in device memory: a canvas, a mask.
pub(crate) struct Image {
    shared: Arc<DeviceShared>,
    pub(crate) image: vk::Image,
    memory: vk::DeviceMemory,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

impl Image {
    pub(crate) fn new(
        shared: &Arc<DeviceShared>,
        format: vk::Format,
        width: u32,
        height: u32,
        usage: vk::ImageUsageFlags,
    ) -> Result<Self, VulkanError> {
        let device = &shared.device;
        let mut made = Self {
            shared: Arc::clone(shared),
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            width,
            height,
        };
        // SAFETY: the create infos outlive their calls, and each handle is
        // stored in `made` as soon as it exists, whose drop frees it on any
        // later failure.
        unsafe {
            made.image = device
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(format)
                        .extent(vk::Extent3D {
                            width,
                            height,
                            depth: 1,
                        })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(usage)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .initial_layout(vk::ImageLayout::UNDEFINED),
                    None,
                )
                .map_err(call("vkCreateImage"))?;
            let requirements = device.get_image_memory_requirements(made.image);
            let kind = shared
                .memory_type(
                    requirements.memory_type_bits,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )
                .ok_or(VulkanError::NoMemoryType("device-local"))?;
            made.memory = device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(requirements.size)
                        .memory_type_index(kind),
                    None,
                )
                .map_err(call("vkAllocateMemory"))?;
            device
                .bind_image_memory(made.image, made.memory, 0)
                .map_err(call("vkBindImageMemory"))?;
        }
        Ok(made)
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        let device = &self.shared.device;
        // SAFETY: made on this device, and no recording using it is still
        // running — see the module docs.
        unsafe {
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}

/// A view of an image, destroyed with this value — made for one recording,
/// or kept with the image it views.
pub(crate) struct View {
    shared: Arc<DeviceShared>,
    pub(crate) view: vk::ImageView,
}

impl View {
    /// A 2D view of the first layer of `image` — of the plane `aspect` names,
    /// for a multi-planar one — read as `format`.
    pub(crate) fn new(
        shared: &Arc<DeviceShared>,
        image: vk::Image,
        format: vk::Format,
        aspect: vk::ImageAspectFlags,
    ) -> Result<Self, VulkanError> {
        // SAFETY: `image` is live on this device for as long as the view is
        // used, which the caller keeps; the create info outlives the call.
        let view = unsafe {
            shared.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(aspect)
                            .level_count(1)
                            .layer_count(1),
                    ),
                None,
            )
        }
        .map_err(call("vkCreateImageView"))?;
        Ok(Self {
            shared: Arc::clone(shared),
            view,
        })
    }
}

impl Drop for View {
    fn drop(&mut self) {
        // SAFETY: made on this device, and no recording using it is still
        // running — see the module docs.
        unsafe { self.shared.device.destroy_image_view(self.view, None) };
    }
}

/// A buffer in host-visible, coherent memory, mapped for as long as it
/// lives: what bytes from the CPU are copied into an image from.
pub(crate) struct HostBuffer {
    shared: Arc<DeviceShared>,
    pub(crate) buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    pub(crate) size: usize,
}

// SAFETY: the mapping is this value's own and only written through
// `&mut self`.
unsafe impl Send for HostBuffer {}

impl HostBuffer {
    pub(crate) fn new(shared: &Arc<DeviceShared>, size: usize) -> Result<Self, VulkanError> {
        let device = &shared.device;
        let mut made = Self {
            shared: Arc::clone(shared),
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            mapped: std::ptr::null_mut(),
            size,
        };
        // SAFETY: as in `Image::new`; the mapping covers the whole
        // allocation, which outlives it.
        unsafe {
            made.buffer = device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(size.max(1) as u64)
                        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .map_err(call("vkCreateBuffer"))?;
            let requirements = device.get_buffer_memory_requirements(made.buffer);
            let kind = shared
                .memory_type(
                    requirements.memory_type_bits,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
                .ok_or(VulkanError::NoMemoryType("host-visible"))?;
            made.memory = device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(requirements.size)
                        .memory_type_index(kind),
                    None,
                )
                .map_err(call("vkAllocateMemory"))?;
            device
                .bind_buffer_memory(made.buffer, made.memory, 0)
                .map_err(call("vkBindBufferMemory"))?;
            made.mapped = device
                .map_memory(made.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .map_err(call("vkMapMemory"))?
                .cast();
        }
        Ok(made)
    }

    /// The mapped bytes, to write what the GPU will copy.
    pub(crate) fn bytes(&mut self) -> &mut [u8] {
        // SAFETY: the mapping is `size` bytes of this buffer's own memory,
        // live as long as `self`, and `&mut self` makes this the one borrow.
        unsafe { std::slice::from_raw_parts_mut(self.mapped, self.size) }
    }
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        let device = &self.shared.device;
        // SAFETY: made on this device, and no recording using it is still
        // running — see the module docs. Freeing the memory unmaps it.
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

/// A command buffer, the fence it is waited for with, and the descriptor
/// sets it binds — reused recording after recording, one at a time.
pub(crate) struct Recording {
    shared: Arc<DeviceShared>,
    pool: vk::CommandPool,
    pub(crate) commands: vk::CommandBuffer,
    fence: vk::Fence,
    descriptors: vk::DescriptorPool,
}

/// Descriptor sets a recording can bind, and how many of each descriptor
/// they can hold between them.
const MAX_SETS: u32 = 256;

impl Recording {
    pub(crate) fn new(shared: &Arc<DeviceShared>) -> Result<Self, VulkanError> {
        let device = &shared.device;
        let mut made = Self {
            shared: Arc::clone(shared),
            pool: vk::CommandPool::null(),
            commands: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            descriptors: vk::DescriptorPool::null(),
        };
        // SAFETY: as in `Image::new`.
        unsafe {
            made.pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(shared.compute_family)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .map_err(call("vkCreateCommandPool"))?;
            made.commands = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(made.pool)
                        .command_buffer_count(1),
                )
                .map_err(call("vkAllocateCommandBuffers"))?[0];
            made.fence = device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(call("vkCreateFence"))?;
            let sizes = [
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(MAX_SETS * 2),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::SAMPLED_IMAGE)
                    .descriptor_count(MAX_SETS * 2),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::SAMPLER)
                    .descriptor_count(MAX_SETS),
            ];
            made.descriptors = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(MAX_SETS)
                        .pool_sizes(&sizes),
                    None,
                )
                .map_err(call("vkCreateDescriptorPool"))?;
        }
        Ok(made)
    }

    /// Starts a new recording, forgetting every descriptor set the last one
    /// bound.
    pub(crate) fn begin(&mut self) -> Result<(), VulkanError> {
        let device = &self.shared.device;
        // SAFETY: the last recording was waited for before it returned, so
        // nothing in the command buffer or the sets is still in use.
        unsafe {
            device
                .reset_descriptor_pool(self.descriptors, vk::DescriptorPoolResetFlags::empty())
                .map_err(call("vkResetDescriptorPool"))?;
            device
                .reset_command_buffer(self.commands, vk::CommandBufferResetFlags::empty())
                .map_err(call("vkResetCommandBuffer"))?;
            device
                .begin_command_buffer(
                    self.commands,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(call("vkBeginCommandBuffer"))
        }
    }

    /// A descriptor set of `layout`, for this recording only.
    pub(crate) fn set(
        &mut self,
        layout: vk::DescriptorSetLayout,
    ) -> Result<vk::DescriptorSet, VulkanError> {
        let layouts = [layout];
        // SAFETY: the pool and the layout are live on this device.
        unsafe {
            self.shared.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.descriptors)
                    .set_layouts(&layouts),
            )
        }
        .map(|mut sets| sets.remove(0))
        .map_err(call("vkAllocateDescriptorSets"))
    }

    /// Ends the recording and submits it, waiting for `waits` and
    /// signalling `signals`. [`Self::wait`] waits for it to finish.
    pub(crate) fn submit(
        &mut self,
        waits: &[vk::SemaphoreSubmitInfo<'_>],
        signals: &[vk::SemaphoreSubmitInfo<'_>],
    ) -> Result<(), VulkanError> {
        let device = &self.shared.device;
        // SAFETY: the command buffer is in the recording state, begun by
        // `begin`; the submission's arrays outlive the call, which is made
        // under the queue's lock; the fence is unsignalled — reset by `wait`
        // after the last submission, or never used.
        unsafe {
            device
                .end_command_buffer(self.commands)
                .map_err(call("vkEndCommandBuffer"))?;
            let buffers = [vk::CommandBufferSubmitInfo::default().command_buffer(self.commands)];
            let submit = [vk::SubmitInfo2::default()
                .wait_semaphore_infos(waits)
                .command_buffer_infos(&buffers)
                .signal_semaphore_infos(signals)];
            self.shared
                .queue(|queue| device.queue_submit2(queue, &submit, self.fence))
                .map_err(call("vkQueueSubmit2"))
        }
    }

    /// Waits for what [`Self::submit`] submitted to finish.
    pub(crate) fn wait(&mut self) -> Result<(), VulkanError> {
        let device = &self.shared.device;
        // SAFETY: the fence is this recording's own, signalled by the last
        // submission; it is reset for the next one either way.
        unsafe {
            let waited = device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(call("vkWaitForFences"));
            device
                .reset_fences(&[self.fence])
                .map_err(call("vkResetFences"))?;
            waited
        }
    }
}

impl Drop for Recording {
    fn drop(&mut self) {
        let device = &self.shared.device;
        // SAFETY: made on this device, and the last submission was waited for.
        unsafe {
            device.destroy_descriptor_pool(self.descriptors, None);
            device.destroy_fence(self.fence, None);
            device.destroy_command_pool(self.pool, None);
        }
    }
}

/// A sampler that reads between pixels linearly and clamps at the edges.
pub(crate) struct Sampler {
    shared: Arc<DeviceShared>,
    pub(crate) sampler: vk::Sampler,
}

impl Sampler {
    pub(crate) fn linear(shared: &Arc<DeviceShared>) -> Result<Self, VulkanError> {
        // SAFETY: the create info outlives the call.
        let sampler = unsafe {
            shared.device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )
        }
        .map_err(call("vkCreateSampler"))?;
        Ok(Self {
            shared: Arc::clone(shared),
            sampler,
        })
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        // SAFETY: made on this device, and nothing using it is still running.
        unsafe { self.shared.device.destroy_sampler(self.sampler, None) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every kernel the compositor dispatches compiles on its own, which needs
    /// no GPU: a shader this crate ships that did not would fail every
    /// compositor at construction.
    #[test]
    fn the_compositor_kernels_compile() {
        let source = include_str!("../../shaders/vulkan/composite.wgsl");
        for entry in ["fill", "layer_nv12", "layer_bgra", "text", "to_nv12"] {
            let spirv = compile(source, entry).unwrap_or_else(|error| panic!("{error}"));
            assert!(!spirv.is_empty(), "{entry}");
        }
    }
}
