//! Using FFmpeg's Vulkan frames in this crate's own work, on FFmpeg's terms.
//!
//! An `AVVkFrame` says, for each of its images, the layout it is in, the
//! access that last wrote it, and a timeline semaphore with the value at
//! which it is ready. Whoever uses one — FFmpeg's decoder and encoder, or
//! this crate — takes FFmpeg's lock on the frame, waits on each semaphore at
//! its value, signals it at the next, and leaves the new layout and access
//! behind for the next user (`libavutil/hwcontext_vulkan.h`). That is the
//! only synchronization between this crate's queue and FFmpeg's: nothing
//! here assumes an image is ready, or in any particular layout.
//!
//! A recording here uses several frames, and FFmpeg's decoder locks a
//! picture together with the pictures it refers to; holding several locks
//! at once, in an order of this crate's own, could deadlock against it. So
//! each frame is [`claim`]ed on its own: locked, read, told what the
//! recording will leave it as, and unlocked again, before the recording is
//! submitted. The next user then waits for a value this recording has not
//! signalled yet, which timeline semaphores allow — its work runs once this
//! recording's has.

use ash::vk::{self, Handle};
use ffmpeg_next::{self as ffmpeg, ffi as av};

use super::ffi::{AVVkFrame, AVVulkanFramesContext};

/// The images of a frame, and the formats FFmpeg made them in — one image
/// for a multi-planar format, else one per plane.
pub(crate) struct FrameImages {
    pub(crate) images: Vec<(vk::Image, vk::Format)>,
    /// The size the images were made at, which a decoder's may exceed the
    /// frame's own by — rounded up to its coding block.
    pub(crate) width: u32,
    pub(crate) height: u32,
}

/// `frame`'s images.
///
/// # Safety
///
/// `frame` must be a live `AV_PIX_FMT_VULKAN` frame with a frames context.
pub(crate) unsafe fn images_of(frame: &ffmpeg::frame::Video) -> FrameImages {
    // SAFETY: the caller's promise: `data[0]` of a Vulkan frame is its
    // `AVVkFrame`, and its frames context's `hwctx` the
    // `AVVulkanFramesContext` FFmpeg filled in, `format` included.
    unsafe {
        let vkf = (*frame.as_ptr()).data[0] as *const AVVkFrame;
        let frames_ctx = (*(*frame.as_ptr()).hw_frames_ctx).data as *const av::AVHWFramesContext;
        let hwctx = (*frames_ctx).hwctx as *const AVVulkanFramesContext;
        let images = (0..(*vkf).img.len())
            .map(|index| {
                (
                    vk::Image::from_raw((*vkf).img[index] as u64),
                    vk::Format::from_raw((*hwctx).format[index] as _),
                )
            })
            .take_while(|(image, _)| *image != vk::Image::null())
            .collect();
        FrameImages {
            images,
            width: (*frames_ctx).width as u32,
            height: (*frames_ctx).height as u32,
        }
    }
}

/// What a recording has to do about one frame it uses: the barriers that
/// bring its images to where the recording uses them, and the semaphore
/// values the submission waits for and signals.
#[derive(Default)]
pub(crate) struct Claim {
    pub(crate) barriers: Vec<vk::ImageMemoryBarrier2<'static>>,
    pub(crate) waits: Vec<vk::SemaphoreSubmitInfo<'static>>,
    pub(crate) signals: Vec<vk::SemaphoreSubmitInfo<'static>>,
}

impl Claim {
    /// Adds `other`'s, for one submission that uses both.
    pub(crate) fn extend(&mut self, other: Claim) {
        self.barriers.extend(other.barriers);
        self.waits.extend(other.waits);
        self.signals.extend(other.signals);
    }
}

/// Claims `frame` for a recording that uses it at `stage` with `access`
/// and leaves it in `layout` — see the module docs. From here the frame
/// says it is what the recording will make it, so the recording has to be
/// submitted, or [`abandon`]ed.
///
/// # Safety
///
/// `frame` must be a live `AV_PIX_FMT_VULKAN` frame with a frames context,
/// made on the device the recording is, and not locked by this thread.
pub(crate) unsafe fn claim(
    frame: &ffmpeg::frame::Video,
    stage: vk::PipelineStageFlags2,
    access: vk::AccessFlags2,
    layout: vk::ImageLayout,
) -> Claim {
    // SAFETY: the caller's promise, as in `images_of`; `lock_frame` and
    // `unlock_frame` are FFmpeg's own, set by `av_hwframe_ctx_init` where the
    // frames context's maker set none, and taken and released in pairs.
    unsafe {
        let vkf = (*frame.as_ptr()).data[0] as *mut AVVkFrame;
        let frames_ctx = (*(*frame.as_ptr()).hw_frames_ctx).data as *mut av::AVHWFramesContext;
        let hwctx = (*frames_ctx).hwctx as *const AVVulkanFramesContext;
        if let Some(lock) = (*hwctx).lock_frame {
            lock(frames_ctx.cast(), vkf);
        }
        let mut claim = Claim::default();
        let vkf_ref = &mut *vkf;
        for index in 0..vkf_ref.img.len() {
            if vkf_ref.img[index].is_null() {
                break;
            }
            let semaphore = vk::Semaphore::from_raw(vkf_ref.sem[index] as u64);
            let value = vkf_ref.sem_value[index];
            claim.barriers.push(
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .src_access_mask(vk::AccessFlags2::from_raw(vkf_ref.access[index] as u64))
                    .dst_stage_mask(stage)
                    .dst_access_mask(access)
                    .old_layout(vk::ImageLayout::from_raw(vkf_ref.layout[index] as _))
                    .new_layout(layout)
                    .src_queue_family_index(vkf_ref.queue_family[index])
                    .dst_queue_family_index(vkf_ref.queue_family[index])
                    .image(vk::Image::from_raw(vkf_ref.img[index] as u64))
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(vk::REMAINING_ARRAY_LAYERS),
                    ),
            );
            claim.waits.push(
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(semaphore)
                    .value(value)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
            );
            claim.signals.push(
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(semaphore)
                    .value(value + 1)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
            );
            vkf_ref.sem_value[index] = value + 1;
            vkf_ref.layout[index] = layout.as_raw() as _;
            vkf_ref.access[index] = access.as_raw() as _;
        }
        if let Some(unlock) = (*hwctx).unlock_frame {
            unlock(frames_ctx.cast(), vkf);
        }
        claim
    }
}

/// Signals, from the host, what a recording that was never submitted would
/// have: whoever uses its frames next waits for those values, and would
/// otherwise wait for ever. What the recording would have done to them is
/// not done — the one frame a failed submission leaves wrong rather than
/// hung.
///
/// Each value it waited for is waited for here first: a timeline only moves
/// forwards, and the work that signals that one may still be running.
pub(crate) fn abandon(device: &ash::Device, claim: &Claim) {
    for (wait, signal) in claim.waits.iter().zip(&claim.signals) {
        let semaphores = [wait.semaphore];
        let values = [wait.value];
        // SAFETY: the semaphore is a live timeline semaphore of a frame on
        // this device; its value is waited for, then the next one signalled,
        // which nothing else will signal now.
        let _ = unsafe {
            device.wait_semaphores(
                &vk::SemaphoreWaitInfo::default()
                    .semaphores(&semaphores)
                    .values(&values),
                u64::MAX,
            )
        };
        // SAFETY: as above.
        let _ = unsafe {
            device.signal_semaphore(
                &vk::SemaphoreSignalInfo::default()
                    .semaphore(signal.semaphore)
                    .value(signal.value),
            )
        };
    }
}
