use std::sync::Arc;

use ash::vk;
use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_error, pp_info};

use crate::{
    buffer::MediaBuffer,
    contract::{
        InputContract, MediaKind, MemoryDomain, OutputContract, PixelLayoutSet, PortContract,
    },
    element::{Element, ElementType, Output, Transform, element_pp_log},
    elements::VulkanDevice,
    error::Result,
    frame_size::ForSize,
    platform::{
        ffmpeg::AvBufferRef,
        vulkan::{
            device::DeviceShared,
            frame_access::{Claim, abandon, claim, images_of},
            frames::{NotOurs, create_frames_ctx, sw_format_of},
            gpu::{Image, Kernel, Recording, View, VulkanError, plane_views},
        },
    },
    pool::{UnboundObjectPool, UnboundObjectPoolRef},
    repeat::{PerFrameTransform, RepeatedOutput},
    transform::{TransformStage, transform_filter},
};

const SHADER: &str = include_str!("../../../../shaders/vulkan/scale.wgsl");

/// The push constants every kernel of `scale.wgsl` takes: the two plane
/// sizes and the kernel.
const IMMEDIATES: u32 = 32;

/// Errors specific to [`VulkanScaler`]. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum VulkanScalerError {
    /// A size of zero was asked for.
    #[error("VulkanScaler cannot scale to {width}x{height}")]
    EmptySize {
        /// The width asked for.
        width: u32,
        /// The height asked for.
        height: u32,
    },

    /// FFmpeg could not take a second reference to the frame already in
    /// hand, which is how an unchanged input is answered.
    #[error("failed to reference the previous frame (code {0})")]
    FrameRef(i32),

    /// The sink received something other than a decoded video frame.
    #[error("VulkanScaler only accepts Video and Eos buffers, got a {0}")]
    UnsupportedBuffer(&'static str),

    /// The frame is not a Vulkan frame at all.
    #[error("VulkanScaler got a {0:?} frame; upload it first")]
    NotVulkan(ffmpeg::format::Pixel),

    /// The frame is a Vulkan frame of another device.
    #[error("VulkanScaler got a frame from another Vulkan device")]
    ForeignDevice,

    /// The frame holds a layout this does not resize.
    #[error("VulkanScaler resizes NV12 and BGRA frames, got {0:?}")]
    UnsupportedLayout(ffmpeg::format::Pixel),

    /// The pool output frames come from could not be made.
    #[error("{0}")]
    Pool(String),

    /// FFmpeg could not hand out an output frame.
    #[error("failed to take a frame from the Vulkan pool (code {0})")]
    FrameGet(i32),

    /// A Vulkan call this element made failed.
    #[error(transparent)]
    Vulkan(#[from] VulkanError),
}

/// How [`VulkanScaler`] weighs the source samples each output sample is
/// made from — the Vulkan counterpart of
/// `CudaScalerInterp`, with the same four choices.
///
/// Whichever is chosen, shrinking spreads it over every source sample the
/// output one covers, so a picture made smaller is averaged rather than
/// sampled: `Nearest` shrinking is an average over boxes, and `Lanczos` the
/// sharpest of the four. Growing, each is what its name says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VulkanScalerInterp {
    /// The nearest source sample; a box average, shrinking.
    Nearest,
    /// A straight line between the two nearest samples.
    Bilinear,
    /// Catmull-Rom cubic, over four samples.
    Bicubic,
    /// Lanczos with three lobes, over six samples: the sharpest, and what
    /// a picture with text in it wants made smaller.
    Lanczos,
}

impl VulkanScalerInterp {
    fn kernel(self) -> u32 {
        match self {
            Self::Nearest => 0,
            Self::Bilinear => 1,
            Self::Bicubic => 2,
            Self::Lanczos => 3,
        }
    }
}

/// Resizes NV12 and BGRA Vulkan frames on the GPU, into frames of the same
/// layout at the size it was made for — the Vulkan counterpart of
/// `CudaScaler`, and what keeps
/// `VulkanDecoder -> VulkanScaler -> VulkanEncoder`, or a compositor into a
/// smaller encode, on the device, where the alternative was
/// `VulkanDownload -> SwScaler -> VulkanUpload`.
///
/// Its kernels are this crate's own rather than FFmpeg's `scale_vulkan`,
/// which only a build of FFmpeg with a GLSL compiler linked in has: they are
/// compiled from WGSL at construction, as every Vulkan element here is.
/// Each plane is resampled on its own, across and then down, by the kernel
/// [`VulkanScalerInterp`] chooses — see there for what shrinking does to it.
/// A frame that is already the size asked for goes on as it came.
///
/// The colour description and the timing are carried through unchanged; so
/// is the layout, since this converts nothing.
pub struct VulkanScaler(TransformStage<Scaling>);

transform_filter!(VulkanScaler);

/// What a [`VulkanScaler`] does to each frame: all of its work, which the
/// framework makes the filter.
struct Scaling {
    pp_log: PpLog,
    name: Arc<str>,
    gpu: Arc<DeviceShared>,
    hw_device_ctx: Arc<AvBufferRef>,
    /// Only compared, to refuse another device's frames.
    device_ctx: *const ffi::AVHWDeviceContext,
    width: u32,
    height: u32,
    interp: VulkanScalerInterp,
    across: Kernel,
    down_r: Kernel,
    down_rg: Kernel,
    down_bgra: Kernel,
    recording: Recording,
    /// Output pools, one per layout, made as the first frame of it arrives.
    nv12_frames: Option<AvBufferRef>,
    bgra_frames: Option<AvBufferRef>,
    /// What the across pass writes, one image per plane, for the size of
    /// the frames arriving.
    between: ForSize<Vec<(Image, View)>>,
    /// What a BGRA output is written into before it is copied — see
    /// `platform::vulkan::bgra_pass` for why.
    scratch: Option<(Image, View)>,
    wrappers: UnboundObjectPool<ffmpeg::frame::Video>,
    repeated: RepeatedOutput,
}

// SAFETY: the FFmpeg buffers have no thread affinity, `device_ctx` is only
// compared, and the Vulkan objects are touched only through `&mut self`;
// queue access goes through FFmpeg's lock.
unsafe impl Send for Scaling {}

/// One plane of the work: its source and output size, the view it is read
/// through, where the across pass leaves it, and the output view or image
/// the down pass writes.
struct Plane {
    source: [u32; 2],
    output: [u32; 2],
    read: vk::ImageView,
    between: (vk::Image, vk::ImageView),
    written: vk::ImageView,
    down: DownPass,
}

#[derive(Clone, Copy)]
enum DownPass {
    R,
    Rg,
    Bgra,
}

impl VulkanScaler {
    /// `device` must be the same [`VulkanDevice`] every other Vulkan element
    /// in this pipeline was built from. Every frame comes out `width` by
    /// `height`.
    pub fn new(
        name: impl Into<String>,
        device: &VulkanDevice,
        width: u32,
        height: u32,
        interp: VulkanScalerInterp,
    ) -> std::result::Result<Self, VulkanScalerError> {
        if width == 0 || height == 0 {
            return Err(VulkanScalerError::EmptySize { width, height });
        }
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::VulkanScaler, &name, None);
        let gpu = Arc::clone(device.shared());
        let kernel = |entry: &std::ffi::CStr, bindings: &[(u32, vk::DescriptorType)]| {
            Kernel::new(&gpu, SHADER, entry, bindings, IMMEDIATES)
        };
        use vk::DescriptorType as D;
        let across = kernel(c"across", &[(0, D::STORAGE_IMAGE), (1, D::SAMPLED_IMAGE)])?;
        let down = [(2, D::STORAGE_IMAGE)];
        let down_r = kernel(c"down_r", &[down[0], (3, D::STORAGE_IMAGE)])?;
        let down_rg = kernel(c"down_rg", &[down[0], (4, D::STORAGE_IMAGE)])?;
        let down_bgra = kernel(c"down_bgra", &[down[0], (5, D::STORAGE_IMAGE)])?;
        let recording = Recording::new(&gpu)?;
        pp_info!(
            pp_log: &pp_log,
            "opened: to {width}x{height}, {interp:?}, on {}",
            device.name()
        );
        Ok(Self(TransformStage::new(Scaling {
            pp_log,
            name,
            gpu,
            hw_device_ctx: device.retain(),
            device_ctx: device.device_ctx(),
            width,
            height,
            interp,
            across,
            down_r,
            down_rg,
            down_bgra,
            recording,
            nv12_frames: None,
            bgra_frames: None,
            between: ForSize::new(),
            scratch: None,
            wrappers: UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {}),
            repeated: RepeatedOutput::new(),
        })))
    }
}

impl Scaling {
    fn scale(
        &mut self,
        source: &ffmpeg::frame::Video,
    ) -> std::result::Result<UnboundObjectPoolRef<ffmpeg::frame::Video>, VulkanScalerError> {
        let layout = match sw_format_of(source, self.device_ctx) {
            Ok(layout @ (ffmpeg::format::Pixel::NV12 | ffmpeg::format::Pixel::BGRA)) => layout,
            Ok(other) => return Err(VulkanScalerError::UnsupportedLayout(other)),
            Err(NotOurs::NotVulkan(format)) => return Err(VulkanScalerError::NotVulkan(format)),
            Err(NotOurs::ForeignDevice) => return Err(VulkanScalerError::ForeignDevice),
        };
        let nv12 = layout == ffmpeg::format::Pixel::NV12;
        let (width, height) = (source.width(), source.height());
        let (out_width, out_height) = (self.width, self.height);
        let half = |value: u32| value.div_ceil(2);

        // What each plane is, source and output. NV12's chroma is half the
        // size either way, rounded up as FFmpeg rounds it.
        let sizes: Vec<([u32; 2], [u32; 2])> = if nv12 {
            vec![
                ([width, height], [out_width, out_height]),
                (
                    [half(width), half(height)],
                    [half(out_width), half(out_height)],
                ),
            ]
        } else {
            vec![([width, height], [out_width, out_height])]
        };

        // Made before any frame is claimed, so a failure here leaves every
        // frame as it was.
        let gpu = Arc::clone(&self.gpu);
        let between = self.between.try_get(width, height, |_, _| {
            sizes
                .iter()
                .map(|&([_, source_height], [output_width, _])| {
                    let image = Image::new(
                        &gpu,
                        vk::Format::R16G16B16A16_SFLOAT,
                        output_width,
                        source_height,
                        vk::ImageUsageFlags::STORAGE,
                    )?;
                    let view = View::new(
                        &gpu,
                        image.image,
                        vk::Format::R16G16B16A16_SFLOAT,
                        vk::ImageAspectFlags::COLOR,
                    )?;
                    Ok::<_, VulkanError>((image, view))
                })
                .collect::<std::result::Result<Vec<_>, _>>()
        })?;
        let between: Vec<(vk::Image, vk::ImageView)> = between
            .iter()
            .map(|(image, view)| (image.image, view.view))
            .collect();
        if !nv12 && self.scratch.is_none() {
            let image = Image::new(
                &gpu,
                vk::Format::R8G8B8A8_UNORM,
                out_width,
                out_height,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            )?;
            let view = View::new(
                &gpu,
                image.image,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageAspectFlags::COLOR,
            )?;
            self.scratch = Some((image, view));
        }
        let frames = {
            let (pool, usage) = if nv12 {
                // Written plane by plane: FFmpeg's own choice of usage,
                // which is every one the format allows, storage among them,
                // on images whose planes can each be viewed on their own.
                (&mut self.nv12_frames, vk::ImageUsageFlags::empty())
            } else {
                (
                    &mut self.bgra_frames,
                    vk::ImageUsageFlags::TRANSFER_DST
                        | vk::ImageUsageFlags::TRANSFER_SRC
                        | vk::ImageUsageFlags::SAMPLED,
                )
            };
            if pool.is_none() {
                // SAFETY: a live Vulkan device context, this element's own.
                let made = unsafe {
                    create_frames_ctx(&self.hw_device_ctx, layout, out_width, out_height, usage)
                }
                .map_err(|error| VulkanScalerError::Pool(error.to_string()))?;
                *pool = Some(made);
            }
            pool.as_ref().expect("made above").as_ptr()
        };

        let mut output = self.wrappers.get();
        // SAFETY: the pooled wrapper's own `AVFrame`, its previous image
        // handed back first; the frames context is this element's own.
        unsafe {
            let dst = output.as_mut_ptr();
            ffi::av_frame_unref(dst);
            let code = ffi::av_hwframe_get_buffer(frames, dst, 0);
            if code < 0 {
                return Err(VulkanScalerError::FrameGet(code));
            }
        }
        // SAFETY: both validated or made above as live Vulkan frames of this
        // device.
        let (source_images, output_images) =
            unsafe { (images_of(source).images, images_of(&output).images) };
        let source_views = plane_views(&self.gpu, &source_images, nv12)?;
        let output_views = if nv12 {
            plane_views(&self.gpu, &output_images, true)?
        } else {
            Vec::new()
        };
        let planes: Vec<Plane> = sizes
            .iter()
            .enumerate()
            .map(|(index, &(source_size, output_size))| Plane {
                source: source_size,
                output: output_size,
                read: source_views[index].view,
                between: between[index],
                written: match output_views.get(index) {
                    Some(view) => view.view,
                    None => self.scratch.as_ref().expect("made above").1.view,
                },
                down: match (nv12, index) {
                    (true, 0) => DownPass::R,
                    (true, _) => DownPass::Rg,
                    (false, _) => DownPass::Bgra,
                },
            })
            .collect();

        let mut claims = Claim::default();
        // SAFETY: two distinct live frames of this device — the source the
        // caller's, the output this element's own.
        unsafe {
            claims.extend(claim(
                source,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ));
            claims.extend(if nv12 {
                claim(
                    &output,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    vk::ImageLayout::GENERAL,
                )
            } else {
                claim(
                    &output,
                    vk::PipelineStageFlags2::COPY,
                    vk::AccessFlags2::TRANSFER_WRITE,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                )
            });
        }
        let copy = (!nv12).then(|| {
            (
                self.scratch.as_ref().expect("made above").0.image,
                output_images[0].0,
            )
        });
        let recorded = self.record(&claims, &planes, copy);
        let submitted =
            recorded.and_then(|()| self.recording.submit(&claims.waits, &claims.signals));
        if let Err(error) = submitted {
            abandon(&self.gpu.device, &claims);
            return Err(error.into());
        }
        let waited = self.recording.wait();
        drop(source_views);
        drop(output_views);
        waited?;
        // SAFETY: two distinct live frames; props are timing, colour and side
        // data, not buffers. The source's crop is the source's: the output
        // is exactly the size it was made at.
        unsafe {
            let dst = output.as_mut_ptr();
            ffi::av_frame_copy_props(dst, source.as_ptr());
            (*dst).crop_top = 0;
            (*dst).crop_bottom = 0;
            (*dst).crop_left = 0;
            (*dst).crop_right = 0;
        }
        Ok(output)
    }

    fn record(
        &mut self,
        claims: &Claim,
        planes: &[Plane],
        copy: Option<(vk::Image, vk::Image)>,
    ) -> std::result::Result<(), VulkanError> {
        self.recording.begin()?;
        let device = &self.gpu.device;
        let commands = self.recording.commands;
        let whole = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        let fresh = |image: vk::Image, access: vk::AccessFlags2| {
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(access)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(whole)
        };
        let mut barriers = claims.barriers.clone();
        for plane in planes {
            barriers.push(fresh(
                plane.between.0,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
            ));
        }
        if let Some((scratch, _)) = copy {
            barriers.push(fresh(scratch, vk::AccessFlags2::SHADER_STORAGE_WRITE));
        }
        // What the across pass wrote, for the down pass to read.
        let across_written: Vec<_> = planes
            .iter()
            .map(|plane| {
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(plane.between.0)
                    .subresource_range(whole)
            })
            .collect();
        let kernel = self.interp.kernel();
        let immediates = |plane: &Plane| {
            [
                plane.source[0],
                plane.source[1],
                plane.output[0],
                plane.output[1],
                kernel,
                0,
                0,
                0,
            ]
            .iter()
            .flat_map(|word| word.to_ne_bytes())
            .collect::<Vec<u8>>()
        };
        let image_info = |view: vk::ImageView, layout: vk::ImageLayout| {
            [vk::DescriptorImageInfo::default()
                .image_view(view)
                .image_layout(layout)]
        };

        // SAFETY: the command buffer is recording; every image and view is
        // live until the recording has been waited for, and each barrier's
        // old layout is the one its image is in.
        unsafe {
            device.cmd_pipeline_barrier2(
                commands,
                &vk::DependencyInfo::default().image_memory_barriers(&barriers),
            );
        }
        for plane in planes {
            let set = self.recording.set(self.across.set_layout)?;
            let written = image_info(plane.between.1, vk::ImageLayout::GENERAL);
            let read = image_info(plane.read, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&written),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&read),
            ];
            // SAFETY: as above; the set is this recording's, of the
            // kernel's layout.
            unsafe {
                device.update_descriptor_sets(&writes, &[]);
                self.dispatch(
                    &self.across,
                    set,
                    &immediates(plane),
                    [plane.output[0], plane.source[1]],
                );
            }
        }
        // SAFETY: as above.
        unsafe {
            device.cmd_pipeline_barrier2(
                commands,
                &vk::DependencyInfo::default().image_memory_barriers(&across_written),
            );
        }
        for plane in planes {
            let (kernel, binding) = match plane.down {
                DownPass::R => (&self.down_r, 3),
                DownPass::Rg => (&self.down_rg, 4),
                DownPass::Bgra => (&self.down_bgra, 5),
            };
            let set = self.recording.set(kernel.set_layout)?;
            let read = image_info(plane.between.1, vk::ImageLayout::GENERAL);
            let written = image_info(plane.written, vk::ImageLayout::GENERAL);
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&read),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&written),
            ];
            // SAFETY: as above.
            unsafe {
                device.update_descriptor_sets(&writes, &[]);
                self.dispatch(kernel, set, &immediates(plane), plane.output);
            }
        }
        if let Some((scratch, output)) = copy {
            let [width, height] = planes[0].output;
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
            // SAFETY: the scratch is read once the down pass has written it,
            // into the output image the claim put in the layout a copy
            // writes; both are the output's size, and their formats are the
            // same size — the scratch holds a BGRA frame's byte order.
            unsafe {
                device.cmd_pipeline_barrier2(
                    commands,
                    &vk::DependencyInfo::default().image_memory_barriers(&copied),
                );
                device.cmd_copy_image(
                    commands,
                    scratch,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    output,
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
        }
        Ok(())
    }

    /// Binds `kernel` with `set` and `immediates`, over `size` samples.
    ///
    /// # Safety
    ///
    /// The recording must be recording, and `set` hold live views of the
    /// kernel's layout, in the layouts its descriptors say.
    unsafe fn dispatch(
        &self,
        kernel: &Kernel,
        set: vk::DescriptorSet,
        immediates: &[u8],
        [width, height]: [u32; 2],
    ) {
        let (device, commands) = (&self.gpu.device, self.recording.commands);
        // SAFETY: the caller's promise.
        unsafe {
            device.cmd_bind_pipeline(commands, vk::PipelineBindPoint::COMPUTE, kernel.pipeline);
            device.cmd_bind_descriptor_sets(
                commands,
                vk::PipelineBindPoint::COMPUTE,
                kernel.pipeline_layout,
                0,
                &[set],
                &[],
            );
            device.cmd_push_constants(
                commands,
                kernel.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                immediates,
            );
            device.cmd_dispatch(commands, width.div_ceil(8), height.div_ceil(8), 1);
        }
    }
}

impl PerFrameTransform for Scaling {
    fn repeated(&mut self) -> &mut RepeatedOutput {
        &mut self.repeated
    }

    fn frame_ref_failed(&self, code: i32) -> crate::error::Error {
        pp_error!(self, "av_frame_ref failed: {code}");
        VulkanScalerError::FrameRef(code).into()
    }

    fn produce(
        &mut self,
        source: &Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        self.scale(source)
            .inspect_err(|error| pp_error!(self, "{error}"))
            .map_err(Into::into)
    }
}

impl Element for Scaling {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::VulkanScaler
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Transform for Scaling {
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::Vulkan)
                .with_layouts(PixelLayoutSet::NV12_OR_BGRA),
        )
    }

    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        match buf {
            // Already the size asked for: nothing to do to it.
            MediaBuffer::Video(frame)
                if frame.width() == self.width && frame.height() == self.height =>
            {
                out.push(MediaBuffer::Video(frame));
                Ok(())
            }
            MediaBuffer::Video(frame) => {
                let scaled = PerFrameTransform::transform(self, &frame)?;
                out.push(MediaBuffer::Video(scaled));
                Ok(())
            }
            // The stage's, never handed here.
            MediaBuffer::Eos => Ok(()),
            other => {
                let kind = other.kind();
                pp_error!(self, "unsupported buffer: {kind}");
                Err(VulkanScalerError::UnsupportedBuffer(kind).into())
            }
        }
    }

    fn reset(&mut self) {
        self.repeated.clear();
    }

    fn output_contract(&self) -> OutputContract {
        OutputContract::SameLayout(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::Vulkan)
                .with_layouts(PixelLayoutSet::NV12_OR_BGRA),
        )
    }
}

impl Drop for Scaling {
    fn drop(&mut self) {
        pp_info!(self, "dropped: releasing hw contexts");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::Sink;
    use crate::{
        elements::{VulkanDownload, VulkanUpload},
        platform::vulkan::gpu::compile,
        test_support::{capture, try_vulkan_device},
    };

    /// Every kernel compiles on its own, which needs no GPU: one that did
    /// not would fail every scaler at construction.
    #[test]
    fn every_kernel_compiles() {
        for entry in ["across", "down_r", "down_rg", "down_bgra"] {
            let spirv = compile(SHADER, entry).unwrap_or_else(|error| panic!("{error}"));
            assert!(!spirv.is_empty(), "{entry}");
        }
    }

    /// `frame` up to the device, through `scaler`, and back.
    fn through(
        device: &VulkanDevice,
        scaler: &mut VulkanScaler,
        frame: ffmpeg::frame::Video,
    ) -> ffmpeg::frame::Video {
        let mut upload = VulkanUpload::new("upload", device);
        let uploaded = capture(&mut upload);
        upload.consume(MediaBuffer::video(frame)).unwrap();
        let scaled = capture(scaler);
        scaler.consume(uploaded.lock().unwrap().remove(0)).unwrap();
        let mut download = VulkanDownload::new("download", device);
        let back = capture(&mut download);
        download.consume(scaled.lock().unwrap().remove(0)).unwrap();
        let MediaBuffer::Video(out) = back.lock().unwrap().remove(0) else {
            panic!("a picture");
        };
        Arc::try_unwrap(out)
            .map(|pooled| (*pooled).clone())
            .unwrap_or_else(|shared| (**shared).clone())
    }

    /// An NV12 picture of `width` by `height`, each column's luma and each
    /// chroma column's Cb and Cr from the functions given.
    fn nv12(
        width: u32,
        height: u32,
        luma: impl Fn(usize, usize) -> u8,
        chroma: impl Fn(usize, usize) -> [u8; 2],
    ) -> ffmpeg::frame::Video {
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, width, height);
        let (luma_stride, chroma_stride) = (frame.stride(0), frame.stride(1));
        for y in 0..height as usize {
            for x in 0..width as usize {
                frame.data_mut(0)[y * luma_stride + x] = luma(x, y);
            }
        }
        for y in 0..height as usize / 2 {
            for x in 0..width as usize / 2 {
                frame.data_mut(1)[y * chroma_stride + x * 2..][..2].copy_from_slice(&chroma(x, y));
            }
        }
        frame
    }

    /// Made smaller, an NV12 picture keeps its colours where they are, and
    /// its timing and colour description.
    #[test]
    fn nv12_is_resized_and_keeps_its_colours() {
        let Some(device) = try_vulkan_device() else {
            return;
        };
        // BT.709 limited-range red on the left half, blue on the right.
        let (red, blue) = ((63, [102, 240]), (32, [240, 118]));
        let mut frame = nv12(
            64,
            32,
            |x, _| if x < 32 { red.0 } else { blue.0 },
            |x, _| if x < 16 { red.1 } else { blue.1 },
        );
        frame.set_pts(Some(21));
        frame.set_color_space(ffmpeg::color::Space::BT709);
        frame.set_color_range(ffmpeg::color::Range::MPEG);
        let mut scaler =
            VulkanScaler::new("scale", &device, 32, 16, VulkanScalerInterp::Lanczos).unwrap();
        let out = through(&device, &mut scaler, frame);

        assert_eq!((out.width(), out.height()), (32, 16));
        assert_eq!(out.format(), ffmpeg::format::Pixel::NV12);
        assert_eq!(out.pts(), Some(21));
        assert_eq!(out.color_space(), ffmpeg::color::Space::BT709);
        let (luma_stride, chroma_stride) = (out.stride(0), out.stride(1));
        for y in 0..16 {
            for (x, (luma, _)) in [(4, red), (27, blue)] {
                let got = out.data(0)[y * luma_stride + x];
                assert!(got.abs_diff(luma) <= 2, "luma at {x},{y}: {got} for {luma}");
            }
        }
        for y in 0..8 {
            for (x, (_, chroma)) in [(2, red), (13, blue)] {
                let got = &out.data(1)[y * chroma_stride + x * 2..][..2];
                for (got, wanted) in got.iter().zip(chroma) {
                    assert!(
                        got.abs_diff(wanted) <= 2,
                        "chroma at {x},{y}: {got} for {wanted}"
                    );
                }
            }
        }
    }

    /// Made larger, a BGRA picture keeps its channels in their order:
    /// red stays red and green green, so nothing swapped a channel on the
    /// way through the scratch image.
    #[test]
    fn bgra_is_resized_in_its_own_byte_order() {
        let Some(device) = try_vulkan_device() else {
            return;
        };
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, 8, 4);
        let stride = frame.stride(0);
        for y in 0..4 {
            for x in 0..8 {
                let pixel = if x < 4 {
                    [0, 0, 255, 255]
                } else {
                    [0, 255, 0, 200]
                };
                frame.data_mut(0)[y * stride + x * 4..][..4].copy_from_slice(&pixel);
            }
        }
        let mut scaler =
            VulkanScaler::new("scale", &device, 16, 8, VulkanScalerInterp::Bilinear).unwrap();
        let out = through(&device, &mut scaler, frame);

        assert_eq!((out.width(), out.height()), (16, 8));
        assert_eq!(out.format(), ffmpeg::format::Pixel::BGRA);
        let stride = out.stride(0);
        for y in 0..8 {
            assert_eq!(
                &out.data(0)[y * stride..][..4],
                &[0, 0, 255, 255],
                "row {y}, left"
            );
            assert_eq!(
                &out.data(0)[y * stride + 15 * 4..][..4],
                &[0, 255, 0, 200],
                "row {y}, right"
            );
        }
    }

    /// Made a quarter of the size, a checkerboard of single pixels comes out
    /// the grey it averages to, whichever kernel: shrinking widens each over
    /// every source sample the output one covers. Sampled rather than
    /// averaged it would come out as whichever pixels it landed on — black,
    /// white, or stripes.
    #[test]
    fn shrinking_averages_what_it_covers() {
        let Some(device) = try_vulkan_device() else {
            return;
        };
        for interp in [
            VulkanScalerInterp::Nearest,
            VulkanScalerInterp::Bilinear,
            VulkanScalerInterp::Bicubic,
            VulkanScalerInterp::Lanczos,
        ] {
            let frame = nv12(
                64,
                64,
                |x, y| if (x + y) % 2 == 0 { 16 } else { 235 },
                |_, _| [128, 128],
            );
            let mut scaler = VulkanScaler::new("scale", &device, 16, 16, interp).unwrap();
            let out = through(&device, &mut scaler, frame);
            let stride = out.stride(0);
            for y in 0..16 {
                for x in 0..16 {
                    let got = out.data(0)[y * stride + x];
                    assert!(
                        got.abs_diff(126) <= 8,
                        "{interp:?} at {x},{y}: {got}, not the grey it averages to"
                    );
                }
            }
        }
    }

    /// A frame already the size asked for goes on as it came.
    #[test]
    fn a_frame_of_the_size_asked_for_goes_on_as_it_came() {
        let Some(device) = try_vulkan_device() else {
            return;
        };
        let mut upload = VulkanUpload::new("upload", &device);
        let uploaded = capture(&mut upload);
        upload
            .consume(MediaBuffer::video(nv12(
                32,
                16,
                |_, _| 90,
                |_, _| [128, 128],
            )))
            .unwrap();
        let MediaBuffer::Video(frame) = uploaded.lock().unwrap().remove(0) else {
            panic!("a picture");
        };
        let mut scaler =
            VulkanScaler::new("scale", &device, 32, 16, VulkanScalerInterp::Lanczos).unwrap();
        let scaled = capture(&mut scaler);
        scaler
            .consume(MediaBuffer::Video(Arc::clone(&frame)))
            .unwrap();
        let MediaBuffer::Video(out) = scaled.lock().unwrap().remove(0) else {
            panic!("a picture");
        };
        assert!(Arc::ptr_eq(&out, &frame), "the same frame, not a copy");
    }

    /// A frame in system memory is refused by name rather than read as
    /// something it is not.
    #[test]
    fn a_frame_not_on_the_device_is_refused() {
        let Some(device) = try_vulkan_device() else {
            return;
        };
        let mut scaler =
            VulkanScaler::new("scale", &device, 16, 8, VulkanScalerInterp::Bilinear).unwrap();
        let error = scaler
            .consume(MediaBuffer::video(nv12(
                32,
                16,
                |_, _| 90,
                |_, _| [128, 128],
            )))
            .unwrap_err();
        assert!(
            matches!(
                error,
                crate::error::Error::VulkanScalerError(VulkanScalerError::NotVulkan(
                    ffmpeg::format::Pixel::NV12
                ))
            ),
            "{error}"
        );
    }

    /// No size to scale to is refused when the scaler is made.
    #[test]
    fn an_empty_size_is_refused() {
        let Some(device) = try_vulkan_device() else {
            return;
        };
        assert!(matches!(
            VulkanScaler::new("scale", &device, 0, 8, VulkanScalerInterp::Bilinear),
            Err(VulkanScalerError::EmptySize {
                width: 0,
                height: 8
            })
        ));
    }
}
