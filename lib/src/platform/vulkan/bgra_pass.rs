//! One compute kernel over a Vulkan frame, into a new BGRA one of the same
//! size — what a per-pixel filter on Vulkan is: `VulkanVideoEffect`,
//! `VulkanChromaKey`, and `VulkanConverter` from NV12.
//!
//! The kernel reads the source texel by texel and writes an RGBA image of
//! this pass's own in a BGRA frame's byte order, which is copied into the
//! output frame: an image a kernel can store to in every Vulkan, where a
//! B8G8R8A8 storage image is not one every driver offers. The kernel's
//! module declares binding 0 as that image, `rgba8unorm`, write-only,
//! binding 1 as the source — for NV12 its luma, and binding 2 its chroma —
//! and its push constants begin with the picture's size.

use std::{ffi::CStr, sync::Arc};

use ash::vk;
use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;

use super::{
    device::DeviceShared,
    frame_access::{Claim, abandon, claim, images_of},
    frames::{NotOurs, create_frames_ctx, sw_format_of},
    gpu::{Image, Kernel, Recording, View, VulkanError, plane_views},
};
use crate::{
    elements::VulkanDevice,
    frame_size::ForSize,
    platform::ffmpeg::AvBufferRef,
    pool::{UnboundObjectPool, UnboundObjectPoolRef},
};

/// Why a pass could not run over a frame.
#[derive(Debug, ThisError)]
pub(crate) enum BgraPassError {
    #[error("got a {0:?} frame; upload it first")]
    NotVulkan(ffmpeg::format::Pixel),
    #[error("a frame from another Vulkan device")]
    ForeignDevice,
    #[error("takes {expected:?} frames, got {got:?}")]
    WrongLayout {
        expected: ffmpeg::format::Pixel,
        got: ffmpeg::format::Pixel,
    },
    #[error(transparent)]
    Vulkan(#[from] VulkanError),
    #[error("{0}")]
    Pool(String),
    #[error("failed to take a frame from the Vulkan pool (code {0})")]
    FrameGet(i32),
}

/// What a pass reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PassInput {
    Bgra,
    Nv12,
}

impl PassInput {
    fn pixel(self) -> ffmpeg::format::Pixel {
        match self {
            Self::Bgra => ffmpeg::format::Pixel::BGRA,
            Self::Nv12 => ffmpeg::format::Pixel::NV12,
        }
    }
}

/// The kernel, and everything one frame through it needs.
pub(crate) struct BgraPass {
    gpu: Arc<DeviceShared>,
    hw_device_ctx: Arc<AvBufferRef>,
    device_ctx: *const ffi::AVHWDeviceContext,
    input: PassInput,
    kernel: Kernel,
    recording: Recording,
    /// The pool output frames come from, for the size of the frames
    /// arriving.
    frames: ForSize<AvBufferRef>,
    /// What the kernel writes, for the same size.
    scratch: ForSize<(Image, View)>,
    wrappers: UnboundObjectPool<ffmpeg::frame::Video>,
}

// SAFETY: the FFmpeg buffers have no thread affinity, `device_ctx` is only
// compared, and the Vulkan objects are touched only through `&mut self`;
// queue access goes through FFmpeg's lock.
unsafe impl Send for BgraPass {}

impl BgraPass {
    /// `entry` of `shader`, taking `immediates` bytes of push constants.
    pub(crate) fn new(
        device: &VulkanDevice,
        shader: &str,
        entry: &CStr,
        immediates: u32,
        input: PassInput,
    ) -> Result<Self, VulkanError> {
        let gpu = Arc::clone(device.shared());
        let kernel = Kernel::new(
            &gpu,
            shader,
            entry,
            match input {
                PassInput::Bgra => &[
                    (0, vk::DescriptorType::STORAGE_IMAGE),
                    (1, vk::DescriptorType::SAMPLED_IMAGE),
                ][..],
                PassInput::Nv12 => &[
                    (0, vk::DescriptorType::STORAGE_IMAGE),
                    (1, vk::DescriptorType::SAMPLED_IMAGE),
                    (2, vk::DescriptorType::SAMPLED_IMAGE),
                ][..],
            },
            immediates,
        )?;
        let recording = Recording::new(&gpu)?;
        Ok(Self {
            gpu,
            hw_device_ctx: device.retain(),
            input,
            device_ctx: device.device_ctx(),
            kernel,
            recording,
            frames: ForSize::new(),
            scratch: ForSize::new(),
            wrappers: UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {}),
        })
    }

    /// `source` through the kernel with `immediates` for its push constants,
    /// into a new frame with `source`'s timing and colour.
    pub(crate) fn run(
        &mut self,
        source: &ffmpeg::frame::Video,
        immediates: &[u8],
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>, BgraPassError> {
        match sw_format_of(source, self.device_ctx) {
            Ok(got) if got == self.input.pixel() => {}
            Ok(got) => {
                return Err(BgraPassError::WrongLayout {
                    expected: self.input.pixel(),
                    got,
                });
            }
            Err(NotOurs::NotVulkan(format)) => return Err(BgraPassError::NotVulkan(format)),
            Err(NotOurs::ForeignDevice) => return Err(BgraPassError::ForeignDevice),
        }
        let (width, height) = (source.width(), source.height());
        let (gpu, hw_device_ctx) = (&self.gpu, &self.hw_device_ctx);
        let frames = self
            .frames
            .try_get(width, height, |width, height| {
                // SAFETY: a live Vulkan device context, this pass's own.
                unsafe {
                    create_frames_ctx(
                        hw_device_ctx,
                        ffmpeg::format::Pixel::BGRA,
                        width,
                        height,
                        vk::ImageUsageFlags::TRANSFER_DST
                            | vk::ImageUsageFlags::TRANSFER_SRC
                            | vk::ImageUsageFlags::SAMPLED,
                    )
                }
                .map_err(|error| BgraPassError::Pool(error.to_string()))
            })?
            .as_ptr();
        let (scratch, scratch_view) = {
            let (image, view) = self.scratch.try_get(width, height, |width, height| {
                let image = Image::new(
                    gpu,
                    vk::Format::R8G8B8A8_UNORM,
                    width,
                    height,
                    vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
                )?;
                let view = View::new(
                    gpu,
                    image.image,
                    vk::Format::R8G8B8A8_UNORM,
                    vk::ImageAspectFlags::COLOR,
                )?;
                Ok::<_, BgraPassError>((image, view))
            })?;
            (image.image, view.view)
        };

        let mut output = self.wrappers.get();
        // SAFETY: the pooled wrapper's own `AVFrame`, its previous image
        // handed back first; the frames context is this pass's own.
        unsafe {
            let dst = output.as_mut_ptr();
            ffi::av_frame_unref(dst);
            let code = ffi::av_hwframe_get_buffer(frames, dst, 0);
            if code < 0 {
                return Err(BgraPassError::FrameGet(code));
            }
        }
        // SAFETY: both validated or made above as live Vulkan frames of this
        // device.
        let (source_images, output_image) =
            unsafe { (images_of(source).images, images_of(&output).images[0].0) };
        let source_views = plane_views(&self.gpu, &source_images, self.input == PassInput::Nv12)?;

        let mut claims = Claim::default();
        // SAFETY: two distinct live frames of this device — the source the
        // caller's, the output this pass's own.
        unsafe {
            claims.extend(claim(
                source,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ));
            claims.extend(claim(
                &output,
                vk::PipelineStageFlags2::COPY,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            ));
        }
        let recorded = self.record(
            &claims,
            scratch,
            scratch_view,
            &source_views
                .iter()
                .map(|view| view.view)
                .collect::<Vec<_>>(),
            output_image,
            [width, height],
            immediates,
        );
        let submitted =
            recorded.and_then(|()| self.recording.submit(&claims.waits, &claims.signals));
        if let Err(error) = submitted {
            abandon(&self.gpu.device, &claims);
            return Err(error.into());
        }
        let waited = self.recording.wait();
        drop(source_views);
        waited?;
        // SAFETY: two distinct live frames; props are timing, colour and side
        // data, not buffers.
        unsafe {
            ffi::av_frame_copy_props(output.as_mut_ptr(), source.as_ptr());
        }
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn record(
        &mut self,
        claims: &Claim,
        scratch: vk::Image,
        scratch_view: vk::ImageView,
        source_views: &[vk::ImageView],
        output_image: vk::Image,
        [width, height]: [u32; 2],
        immediates: &[u8],
    ) -> Result<(), VulkanError> {
        self.recording.begin()?;
        let device = &self.gpu.device;
        let commands = self.recording.commands;
        let whole = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        let mut barriers = claims.barriers.clone();
        barriers.push(
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(scratch)
                .subresource_range(whole),
        );
        let set = self.recording.set(self.kernel.set_layout)?;
        let written = [vk::DescriptorImageInfo::default()
            .image_view(scratch_view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let read: Vec<[vk::DescriptorImageInfo; 1]> = source_views
            .iter()
            .map(|&view| {
                [vk::DescriptorImageInfo::default()
                    .image_view(view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]
            })
            .collect();
        let mut writes = vec![
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&written),
        ];
        for (binding, info) in (1..).zip(&read) {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(binding)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(info),
            );
        }
        let copied = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(scratch)
            .subresource_range(whole)];
        let layers = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .layer_count(1);
        // SAFETY: the command buffer is recording; every image and view is
        // live until the recording has been waited for; the barriers' layouts
        // are the ones the images are in; the copy is between two images of
        // `width` by `height` whose formats are the same size, the scratch
        // holding the output's byte order.
        unsafe {
            device.cmd_pipeline_barrier2(
                commands,
                &vk::DependencyInfo::default().image_memory_barriers(&barriers),
            );
            device.update_descriptor_sets(&writes, &[]);
            device.cmd_bind_pipeline(
                commands,
                vk::PipelineBindPoint::COMPUTE,
                self.kernel.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                commands,
                vk::PipelineBindPoint::COMPUTE,
                self.kernel.pipeline_layout,
                0,
                &[set],
                &[],
            );
            device.cmd_push_constants(
                commands,
                self.kernel.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                immediates,
            );
            device.cmd_dispatch(commands, width.div_ceil(8), height.div_ceil(8), 1);
            device.cmd_pipeline_barrier2(
                commands,
                &vk::DependencyInfo::default().image_memory_barriers(&copied),
            );
            device.cmd_copy_image(
                commands,
                scratch,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                output_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::ImageCopy::default()
                    .src_subresource(layers)
                    .dst_subresource(layers)
                    .extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    })],
            );
        }
        Ok(())
    }
}

/// `words`, each four bytes in native order, as push constants.
pub(crate) fn immediates(words: &[[u8; 4]]) -> Vec<u8> {
    words.iter().flatten().copied().collect()
}
