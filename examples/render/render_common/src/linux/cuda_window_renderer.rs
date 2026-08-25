use std::sync::Mutex;

use ash::vk;
use media_pp::elements::{CudaFrameRenderer, SubmitError};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use super::cuda_ffi::{
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD, CU_MEMORYTYPE_DEVICE, CUDA_SUCCESS, CUcontext,
    CUdeviceptr, CUexternalMemory, CudaExternalMemoryBufferDesc, CudaExternalMemoryHandle,
    CudaExternalMemoryHandleDesc, CudaMemcpy2D, cuCtxSynchronize, cuDestroyExternalMemory,
    cuExternalMemoryGetMappedBuffer, cuImportExternalMemory, cuMemFree_v2, cuMemcpy2D_v2,
    cuda_error, with_context,
};
use super::vulkan_context::{VulkanGpuContext, device_name};

/// A fullscreen triangle plus a BT.709 limited-range NV12 → RGB conversion.
/// Written in WGSL and compiled with `naga` at construction, so building this
/// example needs no shader toolchain (see this crate's `Cargo.toml`).
const SHADER_WGSL: &str = r#"
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VsOut {
    // One oversized triangle covering the viewport — no vertex buffer.
    let x = f32((index << 1u) & 2u);
    let y = f32(index & 2u);
    var out: VsOut;
    // `v` is flipped against the clip-space y. WGSL's clip space points +y
    // up, but this pipeline draws through an ordinary positive-height Vulkan
    // viewport, whose framebuffer y grows downward; wgpu papers over that with
    // an inverted viewport, and nothing does so here. Without the flip the
    // picture renders upside down.
    out.uv = vec2<f32>(x, 1.0 - y);
    out.pos = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

@group(0) @binding(0) var video_sampler: sampler;
@group(0) @binding(1) var luma: texture_2d<f32>;
@group(0) @binding(2) var chroma: texture_2d<f32>;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let y = textureSample(luma, video_sampler, in.uv).r;
    let uv = textureSample(chroma, video_sampler, in.uv).rg - vec2<f32>(0.5, 0.5);
    // Limited-range Y' (16..235) expanded, then BT.709 coefficients.
    let luma_scaled = (y - 0.0625) * 1.164;
    let r = luma_scaled + 1.793 * uv.y;
    let g = luma_scaled - 0.213 * uv.x - 0.533 * uv.y;
    let b = luma_scaled + 2.112 * uv.x;
    return vec4<f32>(r, g, b, 1.0);
}
"#;

/// Presents NV12 CUDA frames through a Vulkan swapchain.
///
/// # How a frame gets from CUDA to the screen
///
/// Vulkan owns the memory, because that is the only direction that works
/// (see `media_pp::elements::CudaFrameRenderer`'s docs): a `VkBuffer` is
/// allocated with exportable memory, its fd is imported once into CUDA, and
/// every submit is
///
/// 1. `cuMemcpy2D` — decoder surface → that shared buffer, device-to-device,
///    de-pitching each plane to a tightly packed layout;
/// 2. `vkCmdCopyBufferToImage` — shared buffer → two sampled images (R8 luma,
///    R8G8 chroma);
/// 3. one fullscreen draw that samples both and converts to RGB.
///
/// Step 1 never touches the CPU, and steps 2–3 never leave the GPU either.
/// The unavoidable cost is that single device-to-device copy.
///
/// # Synchronization
///
/// The shared buffer is written by CUDA and read by Vulkan, and neither
/// stack can see the other's work. So each submit waits for the previous
/// frame's Vulkan work on a fence *before* the CUDA copy overwrites the
/// buffer, and calls `cuCtxSynchronize` *after* it so Vulkan cannot read a
/// copy still in flight. That is heavier than a shared semaphore would be,
/// but it is correct with no extension beyond external memory, and one
/// frame's latency at video rates is not the bottleneck here.
pub struct CudaWindowRenderer {
    inner: Mutex<Inner>,
}

// SAFETY: every field is either an ash handle (Send+Sync by ash's own
// definition), a plain integer, or the CUDA context/pointers, which are
// only ever used under `inner`'s lock and always with the context explicitly
// pushed on the calling thread (see `cuda_ffi::with_context`).
unsafe impl Send for CudaWindowRenderer {}

struct Inner {
    entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    queue_family: u32,
    memory_properties: vk::PhysicalDeviceMemoryProperties,

    surface_fn: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    swapchain_fn: ash::khr::swapchain::Device,
    swapchain: Swapchain,

    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    sampler: vk::Sampler,

    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,
    in_flight: vk::Fence,
    /// Whether `in_flight` has ever been submitted — a fence starts
    /// unsignalled, so the first frame must not wait on it.
    submitted: bool,

    /// The video-sized resources, rebuilt when the frame size changes.
    video: Option<VideoResources>,

    cuda_ctx: CUcontext,
}

struct Swapchain {
    handle: vk::SwapchainKHR,
    format: vk::Format,
    extent: vk::Extent2D,
    views: Vec<vk::ImageView>,
    framebuffers: Vec<vk::Framebuffer>,
}

struct VideoResources {
    width: u32,
    height: u32,
    luma: Plane,
    chroma: Plane,
    /// Vulkan-owned, CUDA-visible staging memory holding one tightly packed
    /// NV12 frame: `width * height` luma bytes followed by
    /// `width * height / 2` interleaved chroma bytes.
    buffer: vk::Buffer,
    buffer_memory: vk::DeviceMemory,
    external_memory: CUexternalMemory,
    device_ptr: CUdeviceptr,
}

struct Plane {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
}

impl CudaWindowRenderer {
    pub fn new(
        gpu: &VulkanGpuContext,
        display: RawDisplayHandle,
        window: RawWindowHandle,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        let entry = gpu.entry.clone();
        let instance = gpu.instance.clone();
        let device = gpu.device.clone();

        let surface = unsafe {
            ash_window::create_surface(&entry, &instance, display, window, None)
                .map_err(|e| format!("failed to create a Vulkan surface: {e}"))?
        };
        let surface_fn = ash::khr::surface::Instance::new(&entry, &instance);

        let supported = unsafe {
            surface_fn.get_physical_device_surface_support(
                gpu.physical_device,
                gpu.queue_family,
                surface,
            )
        }
        .map_err(|e| format!("surface support query failed: {e}"))?;
        if !supported {
            return Err("the graphics queue cannot present to this surface".into());
        }

        let swapchain_fn = ash::khr::swapchain::Device::new(&instance, &device);
        let format = pick_format(&surface_fn, gpu.physical_device, surface)?;
        let render_pass = create_render_pass(&device, format)?;
        let swapchain = create_swapchain(
            &surface_fn,
            &swapchain_fn,
            &device,
            gpu.physical_device,
            surface,
            format,
            render_pass,
            vk::Extent2D { width, height },
            vk::SwapchainKHR::null(),
        )?;

        let (descriptor_layout, pipeline_layout, pipeline) = create_pipeline(&device, render_pass)?;
        let (descriptor_pool, descriptor_set) = create_descriptor_set(&device, descriptor_layout)?;
        let sampler = create_sampler(&device)?;

        let command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(gpu.queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .map_err(|e| format!("vkCreateCommandPool failed: {e}"))?;
        let command_buffer = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(|e| format!("vkAllocateCommandBuffers failed: {e}"))?[0];

        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let image_available = unsafe { device.create_semaphore(&semaphore_info, None) }
            .map_err(|e| format!("vkCreateSemaphore failed: {e}"))?;
        let render_finished = unsafe { device.create_semaphore(&semaphore_info, None) }
            .map_err(|e| format!("vkCreateSemaphore failed: {e}"))?;
        let in_flight = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(|e| format!("vkCreateFence failed: {e}"))?;

        println!(
            "vulkan: {} ({}x{}, {:?})",
            device_name(&instance, gpu.physical_device),
            swapchain.extent.width,
            swapchain.extent.height,
            format
        );

        Ok(Self {
            inner: Mutex::new(Inner {
                entry,
                instance,
                physical_device: gpu.physical_device,
                device,
                queue: gpu.queue,
                queue_family: gpu.queue_family,
                memory_properties: gpu.memory_properties,
                surface_fn,
                surface,
                swapchain_fn,
                swapchain,
                render_pass,
                pipeline_layout,
                pipeline,
                descriptor_layout,
                descriptor_pool,
                descriptor_set,
                sampler,
                command_pool,
                command_buffer,
                image_available,
                render_finished,
                in_flight,
                submitted: false,
                video: None,
                cuda_ctx: gpu.cuda_ctx,
            }),
        })
    }
}

impl CudaFrameRenderer for CudaWindowRenderer {
    unsafe fn submit_nv12(
        &self,
        y: *const u8,
        y_pitch: usize,
        uv: *const u8,
        uv_pitch: usize,
        width: u32,
        height: u32,
    ) -> Result<(), SubmitError> {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        inner
            .present(y, y_pitch, uv, uv_pitch, width, height)
            .map_err(|error| {
                eprintln!("[cuda-renderer] {error}");
                SubmitError::RenderFailed
            })
    }

    fn resize(&self, width: u32, height: u32) -> Result<(), SubmitError> {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        inner
            .recreate_swapchain(vk::Extent2D { width, height })
            .map_err(|error| {
                eprintln!("[cuda-renderer] {error}");
                SubmitError::RenderFailed
            })
    }
}

impl Inner {
    fn present(
        &mut self,
        y: *const u8,
        y_pitch: usize,
        uv: *const u8,
        uv_pitch: usize,
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        if width == 0 || height == 0 {
            return Err("frame has zero extent".into());
        }
        // The previous frame's Vulkan work reads the same staging buffer the
        // CUDA copy below overwrites — wait for it first.
        if self.submitted {
            unsafe {
                self.device
                    .wait_for_fences(&[self.in_flight], true, u64::MAX)
            }
            .map_err(|e| format!("vkWaitForFences failed: {e}"))?;
        }

        self.ensure_video_resources(width, height)?;
        self.copy_from_cuda(y, y_pitch, uv, uv_pitch, width, height)?;
        self.draw()
    }

    /// (Re)builds the video-sized images and the shared staging buffer.
    /// Frame size only changes when the stream does, so this is not a hot
    /// path — but it must tear down the CUDA import before the Vulkan memory
    /// it refers to goes away.
    fn ensure_video_resources(&mut self, width: u32, height: u32) -> Result<(), String> {
        if self
            .video
            .as_ref()
            .is_some_and(|v| v.width == width && v.height == height)
        {
            return Ok(());
        }
        unsafe { self.device.device_wait_idle() }
            .map_err(|e| format!("vkDeviceWaitIdle failed: {e}"))?;
        if let Some(video) = self.video.take() {
            self.destroy_video(video);
        }

        let luma = self.create_plane(width, height, vk::Format::R8_UNORM)?;
        let chroma = self.create_plane(width / 2, height / 2, vk::Format::R8G8_UNORM)?;

        let luma_bytes = (width * height) as u64;
        let size = luma_bytes + luma_bytes / 2;
        let (buffer, buffer_memory, fd) = self.create_exportable_buffer(size)?;
        let (external_memory, device_ptr) = import_into_cuda(self.cuda_ctx, fd, size)?;

        // Point the descriptor set at the new views.
        let image_info = [
            vk::DescriptorImageInfo::default()
                .image_view(luma.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            vk::DescriptorImageInfo::default()
                .image_view(chroma.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        ];
        let sampler_info = [vk::DescriptorImageInfo::default().sampler(self.sampler)];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .image_info(&sampler_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(&image_info[0..1]),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(&image_info[1..2]),
        ];
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };

        self.video = Some(VideoResources {
            width,
            height,
            luma,
            chroma,
            buffer,
            buffer_memory,
            external_memory,
            device_ptr,
        });
        Ok(())
    }

    /// Device-to-device copy of both planes into the shared buffer,
    /// de-pitching them to the tightly packed layout the Vulkan copy expects.
    fn copy_from_cuda(
        &mut self,
        y: *const u8,
        y_pitch: usize,
        uv: *const u8,
        uv_pitch: usize,
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        let video = self.video.as_ref().expect("resources were just ensured");
        let luma_bytes = (width * height) as u64;
        let ctx = self.cuda_ctx;
        let dst = video.device_ptr;

        with_context(ctx, || {
            let luma_copy = CudaMemcpy2D {
                src_memory_type: CU_MEMORYTYPE_DEVICE,
                src_device: y as CUdeviceptr,
                src_pitch: y_pitch,
                dst_memory_type: CU_MEMORYTYPE_DEVICE,
                dst_device: dst,
                dst_pitch: width as usize,
                width_in_bytes: width as usize,
                height: height as usize,
                ..Default::default()
            };
            let result = unsafe { cuMemcpy2D_v2(&luma_copy) };
            if result != CUDA_SUCCESS {
                return Err(cuda_error("cuMemcpy2D (luma)", result));
            }

            let chroma_copy = CudaMemcpy2D {
                src_memory_type: CU_MEMORYTYPE_DEVICE,
                src_device: uv as CUdeviceptr,
                src_pitch: uv_pitch,
                dst_memory_type: CU_MEMORYTYPE_DEVICE,
                dst_device: dst + luma_bytes,
                dst_pitch: width as usize,
                width_in_bytes: width as usize,
                height: (height / 2) as usize,
                ..Default::default()
            };
            let result = unsafe { cuMemcpy2D_v2(&chroma_copy) };
            if result != CUDA_SUCCESS {
                return Err(cuda_error("cuMemcpy2D (chroma)", result));
            }

            // Vulkan cannot see CUDA's stream, so the copy has to be known
            // complete before the command buffer below reads the buffer.
            let result = unsafe { cuCtxSynchronize() };
            if result != CUDA_SUCCESS {
                return Err(cuda_error("cuCtxSynchronize", result));
            }
            Ok(())
        })??;
        Ok(())
    }

    fn draw(&mut self) -> Result<(), String> {
        let (image_index, suboptimal) = unsafe {
            self.swapchain_fn.acquire_next_image(
                self.swapchain.handle,
                u64::MAX,
                self.image_available,
                vk::Fence::null(),
            )
        }
        .map_err(|e| format!("vkAcquireNextImageKHR failed: {e}"))?;

        unsafe { self.device.reset_fences(&[self.in_flight]) }
            .map_err(|e| format!("vkResetFences failed: {e}"))?;
        self.record(image_index as usize)?;

        let wait = [self.image_available];
        let signal = [self.render_finished];
        let stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let buffers = [self.command_buffer];
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&wait)
            .wait_dst_stage_mask(&stages)
            .command_buffers(&buffers)
            .signal_semaphores(&signal);
        unsafe {
            self.device
                .queue_submit(self.queue, &[submit], self.in_flight)
        }
        .map_err(|e| format!("vkQueueSubmit failed: {e}"))?;
        self.submitted = true;

        let swapchains = [self.swapchain.handle];
        let indices = [image_index];
        let present = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal)
            .swapchains(&swapchains)
            .image_indices(&indices);
        // Must precede successful return: terminal preroll completion means
        // this renderer has committed the frame to presentation.
        let out_of_date = match unsafe { self.swapchain_fn.queue_present(self.queue, &present) } {
            Ok(suboptimal_present) => suboptimal || suboptimal_present,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => true,
            Err(error) => return Err(format!("vkQueuePresentKHR failed: {error}")),
        };
        if out_of_date {
            let extent = self.swapchain.extent;
            self.recreate_swapchain(extent)?;
        }
        Ok(())
    }

    fn record(&mut self, image_index: usize) -> Result<(), String> {
        let device = &self.device;
        let cmd = self.command_buffer;
        let video = self.video.as_ref().expect("resources were ensured");

        unsafe {
            device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .map_err(|e| format!("vkResetCommandBuffer failed: {e}"))?;
            device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|e| format!("vkBeginCommandBuffer failed: {e}"))?;

            for (plane, width, height, offset) in [
                (&video.luma, video.width, video.height, 0u64),
                (
                    &video.chroma,
                    video.width / 2,
                    video.height / 2,
                    (video.width * video.height) as u64,
                ),
            ] {
                transition(
                    device,
                    cmd,
                    plane.image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                );
                let region = vk::BufferImageCopy::default()
                    .buffer_offset(offset)
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1),
                    )
                    .image_extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    });
                device.cmd_copy_buffer_to_image(
                    cmd,
                    video.buffer,
                    plane.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                );
                transition(
                    device,
                    cmd,
                    plane.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                );
            }

            let clear = [vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            }];
            device.cmd_begin_render_pass(
                cmd,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(self.render_pass)
                    .framebuffer(self.swapchain.framebuffers[image_index])
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: self.swapchain.extent,
                    })
                    .clear_values(&clear),
                vk::SubpassContents::INLINE,
            );
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            device.cmd_set_viewport(
                cmd,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: self.swapchain.extent.width as f32,
                    height: self.swapchain.extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            device.cmd_set_scissor(
                cmd,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: self.swapchain.extent,
                }],
            );
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );
            device.cmd_draw(cmd, 3, 1, 0, 0);
            device.cmd_end_render_pass(cmd);
            device
                .end_command_buffer(cmd)
                .map_err(|e| format!("vkEndCommandBuffer failed: {e}"))?;
        }
        Ok(())
    }

    fn recreate_swapchain(&mut self, extent: vk::Extent2D) -> Result<(), String> {
        unsafe { self.device.device_wait_idle() }
            .map_err(|e| format!("vkDeviceWaitIdle failed: {e}"))?;
        let old = std::mem::replace(
            &mut self.swapchain,
            Swapchain {
                handle: vk::SwapchainKHR::null(),
                format: vk::Format::UNDEFINED,
                extent,
                views: Vec::new(),
                framebuffers: Vec::new(),
            },
        );
        let format = old.format;
        unsafe {
            for framebuffer in &old.framebuffers {
                self.device.destroy_framebuffer(*framebuffer, None);
            }
            for view in &old.views {
                self.device.destroy_image_view(*view, None);
            }
        }
        self.swapchain = create_swapchain(
            &self.surface_fn,
            &self.swapchain_fn,
            &self.device,
            self.physical_device,
            self.surface,
            format,
            self.render_pass,
            extent,
            old.handle,
        )?;
        unsafe { self.swapchain_fn.destroy_swapchain(old.handle, None) };
        self.submitted = false;
        Ok(())
    }

    fn create_plane(&self, width: u32, height: u32, format: vk::Format) -> Result<Plane, String> {
        let image = unsafe {
            self.device.create_image(
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
                    .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
        }
        .map_err(|e| format!("vkCreateImage failed: {e}"))?;

        let requirements = unsafe { self.device.get_image_memory_requirements(image) };
        let index = self
            .memory_type_index(requirements, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .ok_or("no device-local memory type for a video plane")?;
        let memory = unsafe {
            self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(index),
                None,
            )
        }
        .map_err(|e| format!("vkAllocateMemory failed: {e}"))?;
        unsafe { self.device.bind_image_memory(image, memory, 0) }
            .map_err(|e| format!("vkBindImageMemory failed: {e}"))?;

        let view = unsafe {
            self.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1),
                    ),
                None,
            )
        }
        .map_err(|e| format!("vkCreateImageView failed: {e}"))?;

        Ok(Plane {
            image,
            memory,
            view,
        })
    }

    /// Allocates the staging buffer with `VkExportMemoryAllocateInfo` and
    /// hands back a POSIX fd for it — the one thing CUDA can import.
    fn create_exportable_buffer(
        &self,
        size: u64,
    ) -> Result<(vk::Buffer, vk::DeviceMemory, i32), String> {
        let mut external = vk::ExternalMemoryBufferCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        let buffer = unsafe {
            self.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .push_next(&mut external),
                None,
            )
        }
        .map_err(|e| format!("vkCreateBuffer failed: {e}"))?;

        let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let index = self
            .memory_type_index(requirements, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .ok_or("no device-local memory type for the shared buffer")?;

        let mut export = vk::ExportMemoryAllocateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        // Dedicated allocation: CUDA imports the whole allocation, so it must
        // back exactly this buffer and nothing else.
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
        let memory = unsafe {
            self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(index)
                    .push_next(&mut export)
                    .push_next(&mut dedicated),
                None,
            )
        }
        .map_err(|e| format!("vkAllocateMemory (exportable) failed: {e}"))?;
        unsafe { self.device.bind_buffer_memory(buffer, memory, 0) }
            .map_err(|e| format!("vkBindBufferMemory failed: {e}"))?;

        let external_fn = ash::khr::external_memory_fd::Device::new(&self.instance, &self.device);
        let fd = unsafe {
            external_fn.get_memory_fd(
                &vk::MemoryGetFdInfoKHR::default()
                    .memory(memory)
                    .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD),
            )
        }
        .map_err(|e| format!("vkGetMemoryFdKHR failed: {e}"))?;

        Ok((buffer, memory, fd))
    }

    fn memory_type_index(
        &self,
        requirements: vk::MemoryRequirements,
        flags: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        (0..self.memory_properties.memory_type_count).find(|&i| {
            requirements.memory_type_bits & (1 << i) != 0
                && self.memory_properties.memory_types[i as usize]
                    .property_flags
                    .contains(flags)
        })
    }

    fn destroy_video(&self, video: VideoResources) {
        // CUDA first: its mapping refers to the Vulkan allocation below.
        let _ = with_context(self.cuda_ctx, || unsafe {
            cuMemFree_v2(video.device_ptr);
            cuDestroyExternalMemory(video.external_memory);
        });
        unsafe {
            self.device.destroy_buffer(video.buffer, None);
            self.device.free_memory(video.buffer_memory, None);
            for plane in [&video.luma, &video.chroma] {
                self.device.destroy_image_view(plane.view, None);
                self.device.destroy_image(plane.image, None);
                self.device.free_memory(plane.memory, None);
            }
        }
    }
}

impl Drop for CudaWindowRenderer {
    fn drop(&mut self) {
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        unsafe {
            let _ = inner.device.device_wait_idle();
            if let Some(video) = inner.video.as_ref() {
                // Same order as `destroy_video`, but the field cannot be
                // moved out of a borrowed `Inner` here.
                let _ = with_context(inner.cuda_ctx, || {
                    cuMemFree_v2(video.device_ptr);
                    cuDestroyExternalMemory(video.external_memory);
                });
                inner.device.destroy_buffer(video.buffer, None);
                inner.device.free_memory(video.buffer_memory, None);
                for plane in [&video.luma, &video.chroma] {
                    inner.device.destroy_image_view(plane.view, None);
                    inner.device.destroy_image(plane.image, None);
                    inner.device.free_memory(plane.memory, None);
                }
            }
            inner.device.destroy_fence(inner.in_flight, None);
            inner.device.destroy_semaphore(inner.render_finished, None);
            inner.device.destroy_semaphore(inner.image_available, None);
            inner.device.destroy_command_pool(inner.command_pool, None);
            inner.device.destroy_sampler(inner.sampler, None);
            inner
                .device
                .destroy_descriptor_pool(inner.descriptor_pool, None);
            inner
                .device
                .destroy_descriptor_set_layout(inner.descriptor_layout, None);
            inner.device.destroy_pipeline(inner.pipeline, None);
            inner
                .device
                .destroy_pipeline_layout(inner.pipeline_layout, None);
            for framebuffer in &inner.swapchain.framebuffers {
                inner.device.destroy_framebuffer(*framebuffer, None);
            }
            for view in &inner.swapchain.views {
                inner.device.destroy_image_view(*view, None);
            }
            inner
                .swapchain_fn
                .destroy_swapchain(inner.swapchain.handle, None);
            inner.device.destroy_render_pass(inner.render_pass, None);
            inner.surface_fn.destroy_surface(inner.surface, None);
        }
        let _ = &inner.entry;
        let _ = inner.queue_family;
    }
}

fn import_into_cuda(
    ctx: CUcontext,
    fd: i32,
    size: u64,
) -> Result<(CUexternalMemory, CUdeviceptr), String> {
    with_context(ctx, || {
        let desc = CudaExternalMemoryHandleDesc {
            type_: CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD,
            handle: CudaExternalMemoryHandle { fd },
            size,
            // `CUDA_EXTERNAL_MEMORY_DEDICATED` — matches the
            // `VkMemoryDedicatedAllocateInfo` used on the Vulkan side.
            flags: 1,
            reserved: [0; 16],
        };
        let mut external: CUexternalMemory = std::ptr::null_mut();
        // CUDA takes ownership of `fd` on success and closes it itself.
        let result = unsafe { cuImportExternalMemory(&mut external, &desc) };
        if result != CUDA_SUCCESS {
            return Err(cuda_error("cuImportExternalMemory", result));
        }

        let buffer_desc = CudaExternalMemoryBufferDesc {
            offset: 0,
            size,
            ..Default::default()
        };
        let mut device_ptr: CUdeviceptr = 0;
        let result =
            unsafe { cuExternalMemoryGetMappedBuffer(&mut device_ptr, external, &buffer_desc) };
        if result != CUDA_SUCCESS {
            unsafe { cuDestroyExternalMemory(external) };
            return Err(cuda_error("cuExternalMemoryGetMappedBuffer", result));
        }
        Ok((external, device_ptr))
    })?
}

fn transition(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    from: vk::ImageLayout,
    to: vk::ImageLayout,
) {
    let (src_access, dst_access, src_stage, dst_stage) = match (from, to) {
        (vk::ImageLayout::UNDEFINED, _) => (
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
        ),
        _ => (
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        ),
    };
    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(from)
        .new_layout(to)
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1),
        );
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        )
    };
}

fn pick_format(
    surface_fn: &ash::khr::surface::Instance,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
) -> Result<vk::Format, String> {
    let formats =
        unsafe { surface_fn.get_physical_device_surface_formats(physical_device, surface) }
            .map_err(|e| format!("surface format query failed: {e}"))?;
    formats
        .iter()
        .find(|format| format.format == vk::Format::B8G8R8A8_UNORM)
        .or_else(|| formats.first())
        .map(|format| format.format)
        .ok_or_else(|| "the surface offers no formats".into())
}

fn create_render_pass(device: &ash::Device, format: vk::Format) -> Result<vk::RenderPass, String> {
    let attachment = [vk::AttachmentDescription::default()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)];
    let color = [vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
    let subpass = [vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color)];
    let dependency = [vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];
    unsafe {
        device.create_render_pass(
            &vk::RenderPassCreateInfo::default()
                .attachments(&attachment)
                .subpasses(&subpass)
                .dependencies(&dependency),
            None,
        )
    }
    .map_err(|e| format!("vkCreateRenderPass failed: {e}"))
}

#[allow(clippy::too_many_arguments)]
fn create_swapchain(
    surface_fn: &ash::khr::surface::Instance,
    swapchain_fn: &ash::khr::swapchain::Device,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    format: vk::Format,
    render_pass: vk::RenderPass,
    requested: vk::Extent2D,
    old: vk::SwapchainKHR,
) -> Result<Swapchain, String> {
    let caps =
        unsafe { surface_fn.get_physical_device_surface_capabilities(physical_device, surface) }
            .map_err(|e| format!("surface capability query failed: {e}"))?;
    let extent = if caps.current_extent.width == u32::MAX {
        requested
    } else {
        caps.current_extent
    };
    let count = (caps.min_image_count + 1).min(if caps.max_image_count == 0 {
        u32::MAX
    } else {
        caps.max_image_count
    });

    let handle = unsafe {
        swapchain_fn.create_swapchain(
            &vk::SwapchainCreateInfoKHR::default()
                .surface(surface)
                .min_image_count(count)
                .image_format(format)
                .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(caps.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(vk::PresentModeKHR::FIFO)
                .clipped(true)
                .old_swapchain(old),
            None,
        )
    }
    .map_err(|e| format!("vkCreateSwapchainKHR failed: {e}"))?;

    let images = unsafe { swapchain_fn.get_swapchain_images(handle) }
        .map_err(|e| format!("vkGetSwapchainImagesKHR failed: {e}"))?;
    let mut views = Vec::with_capacity(images.len());
    let mut framebuffers = Vec::with_capacity(images.len());
    for image in images {
        let view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1),
                    ),
                None,
            )
        }
        .map_err(|e| format!("vkCreateImageView failed: {e}"))?;
        let attachments = [view];
        let framebuffer = unsafe {
            device.create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(extent.width)
                    .height(extent.height)
                    .layers(1),
                None,
            )
        }
        .map_err(|e| format!("vkCreateFramebuffer failed: {e}"))?;
        views.push(view);
        framebuffers.push(framebuffer);
    }

    Ok(Swapchain {
        handle,
        format,
        extent,
        views,
        framebuffers,
    })
}

fn create_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
) -> Result<(vk::DescriptorSetLayout, vk::PipelineLayout, vk::Pipeline), String> {
    let spirv = compile_shader()?;
    let module = unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spirv), None)
    }
    .map_err(|e| format!("vkCreateShaderModule failed: {e}"))?;

    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ];
    let descriptor_layout = unsafe {
        device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )
    }
    .map_err(|e| format!("vkCreateDescriptorSetLayout failed: {e}"))?;

    let layouts = [descriptor_layout];
    let pipeline_layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts),
            None,
        )
    }
    .map_err(|e| format!("vkCreatePipelineLayout failed: {e}"))?;

    let vs_name = c"vs_main";
    let fs_name = c"fs_main";
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(module)
            .name(vs_name),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(module)
            .name(fs_name),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let blend_attachment = [vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachment);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let info = [vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&assembly)
        .viewport_state(&viewport)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0)];
    let pipeline =
        unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &info, None) }
            .map_err(|(_, error)| format!("vkCreateGraphicsPipelines failed: {error}"))?[0];

    unsafe { device.destroy_shader_module(module, None) };
    Ok((descriptor_layout, pipeline_layout, pipeline))
}

fn create_descriptor_set(
    device: &ash::Device,
    layout: vk::DescriptorSetLayout,
) -> Result<(vk::DescriptorPool, vk::DescriptorSet), String> {
    let sizes = [
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::SAMPLER)
            .descriptor_count(1),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(2),
    ];
    let pool = unsafe {
        device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&sizes)
                .max_sets(1),
            None,
        )
    }
    .map_err(|e| format!("vkCreateDescriptorPool failed: {e}"))?;
    let layouts = [layout];
    let set = unsafe {
        device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&layouts),
        )
    }
    .map_err(|e| format!("vkAllocateDescriptorSets failed: {e}"))?[0];
    Ok((pool, set))
}

fn create_sampler(device: &ash::Device) -> Result<vk::Sampler, String> {
    unsafe {
        device.create_sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
            None,
        )
    }
    .map_err(|e| format!("vkCreateSampler failed: {e}"))
}

/// WGSL → SPIR-V, in-process.
fn compile_shader() -> Result<Vec<u32>, String> {
    let module = naga::front::wgsl::parse_str(SHADER_WGSL)
        .map_err(|e| format!("the built-in shader failed to parse: {e}"))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .map_err(|e| format!("the built-in shader failed to validate: {e}"))?;
    let options = naga::back::spv::Options::default();
    naga::back::spv::write_vec(&module, &info, &options, None)
        .map_err(|e| format!("SPIR-V generation failed: {e}"))
}
