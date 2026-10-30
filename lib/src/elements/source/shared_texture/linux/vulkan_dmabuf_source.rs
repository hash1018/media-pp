use std::{
    os::fd::BorrowedFd,
    sync::{Arc, Mutex},
    time::Duration,
};

use ash::vk;
use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use super::dmabuf::{DmaBuf, DmaBufAlpha};
use crate::{
    buffer::MediaBuffer,
    contract::{MediaKind, MemoryDomain, OutputContract, PixelLayoutSet, PortContract},
    element::{Element, ElementType, SourceStage},
    elements::{AppSource, AppSourceHandle, VulkanDevice, source::app_source::Receiving},
    platform::vulkan::{
        bgra_pass::{BgraPass, PassInput, immediates},
        device::DeviceShared,
        gpu::{DmaBufImage, DmaBufPlane, View, VulkanError},
    },
    pp_log::pp_info,
    produce::source_stage,
};

const SHADER: &str = include_str!("../../../../shaders/vulkan/unpremultiply.wgsl");

/// How long a push waits for the producer to have finished drawing into a
/// buffer before reading it anyway. A frame's worth is ample — a producer
/// hands a buffer over once it has drawn it, so the wait is normally none —
/// and a picture read early is a picture, where a producer's thread held
/// for ever is not.
const DRAWN_WITHIN: Duration = Duration::from_millis(100);

/// Errors specific to `VulkanDmaBufSource`. Converts into the crate-wide
/// `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum VulkanDmaBufSourceError {
    /// The kernel that reads a buffer could not be made, or did not run.
    #[error(transparent)]
    Vulkan(#[from] VulkanError),

    /// The frames the copies are made into could not be had.
    #[error("the Vulkan pool could not be used: {0}")]
    Pool(String),

    /// The device cannot import a DMA-BUF: FFmpeg made it without
    /// `VK_EXT_external_memory_dma_buf`, `VK_KHR_external_memory_fd` or
    /// `VK_EXT_image_drm_format_modifier`, which it turns on where the
    /// driver has them.
    #[error("this Vulkan device cannot import a DMA-BUF")]
    NoImport,

    /// A picture of another size than the source was made for.
    #[error(
        "a {width}x{height} DMA-BUF was pushed into a {expected_width}x{expected_height} source"
    )]
    WrongSize {
        width: u32,
        height: u32,
        expected_width: u32,
        expected_height: u32,
    },

    /// The source has ended — its `Pipeline` finished, or end-of-stream was
    /// already submitted — so nothing more can be pushed into it.
    #[error("VulkanDmaBufSource has already ended")]
    Closed,
}

/// Brings another process's DMA-BUFs into this pipeline, as BGRA Vulkan
/// frames on this pipeline's own device.
///
/// What `CudaDmaBufSource` is to a CUDA pipeline, in the same shape: a
/// pushed source, built on [`AppSource`], whose producer calls
/// [`VulkanDmaBufHandle::push`] with a buffer per picture. It copies for the
/// same reason that one does — the producer reuses its buffers — and the
/// copy never leaves the GPU.
///
/// # How the copy is made
///
/// The buffer's memory is imported, not copied, as an image of this
/// device, laid out as its DRM format modifier says — linear, or a GPU's
/// own tiles — and a compute kernel reads it into a frame of this source's
/// own, dividing out premultiplied colour where [`DmaBufAlpha`] says so.
/// Before that the buffer is polled, which waits for whatever the producer
/// was still drawing into it, and the copy is waited for before the push
/// returns, so the producer may draw into it again at once.
///
/// It needs a device with `VK_EXT_external_memory_dma_buf`,
/// `VK_KHR_external_memory_fd` and `VK_EXT_image_drm_format_modifier`,
/// which FFmpeg turns on where the driver has them; without them
/// [`VulkanDmaBufSource::new`] fails with
/// [`VulkanDmaBufSourceError::NoImport`]. Not FFmpeg's own mapping from a
/// `DRM_PRIME` frame, which a build made against kernel headers older than
/// 6.0 does without waiting for the producer, saying so at every picture.
///
/// # What it does not do
///
/// Resize, convert or pace, as `CudaDmaBufSource` does not. A buffer of
/// another size is refused.
pub struct VulkanDmaBufSource(SourceStage<Receiving>);

source_stage!(VulkanDmaBufSource);

/// Pushes another process's DMA-BUFs into a [`VulkanDmaBufSource`].
///
/// Cheap to clone, and every clone feeds the same source. Push
/// [`Self::finish`] when done, or drop every clone.
///
/// # Push while the buffer is still the producer's picture
///
/// [`Self::push`] copies before it returns, so it must be called while the
/// producer still holds that picture — inside its paint callback, not
/// queued for later.
#[derive(Clone)]
pub struct VulkanDmaBufHandle {
    pusher: AppSourceHandle,
    import: Arc<Mutex<Import>>,
}

/// What a push needs: the kernel, and the device a buffer is imported on.
struct Import {
    pass: BgraPass,
    gpu: Arc<DeviceShared>,
    width: u32,
    height: u32,
    alpha: DmaBufAlpha,
}

// SAFETY: `BgraPass` is `Send`, and the device's own tables are shared
// between threads already; both are only used under the handle's lock.
unsafe impl Send for Import {}

impl VulkanDmaBufSource {
    /// `device` must be the [`VulkanDevice`] every other Vulkan element in
    /// this pipeline was built from. `width`/`height` are what every pushed
    /// buffer must be and what the frames coming out are. `capacity` bounds
    /// how many frames may sit unconsumed, as [`AppSource::new`]'s does.
    pub fn new(
        name: impl Into<String>,
        device: &VulkanDevice,
        width: u32,
        height: u32,
        alpha: DmaBufAlpha,
        capacity: usize,
    ) -> std::result::Result<(Self, VulkanDmaBufHandle), VulkanDmaBufSourceError> {
        if device.shared().dma_buf.is_none() {
            return Err(VulkanDmaBufSourceError::NoImport);
        }
        let pass = BgraPass::new(device, SHADER, c"main", 16, PassInput::Bgra)?;
        let (source, pusher) = AppSource::receiving(
            name,
            capacity,
            ElementType::VulkanDmaBufSource,
            OutputContract::Fixed(
                PortContract::frame(MediaKind::VideoFrame, MemoryDomain::Vulkan)
                    .with_layouts(PixelLayoutSet::BGRA),
            ),
            // Live, as `CudaDmaBufSource` is: what it emits is whatever the
            // producer drew just now, and a paused pipeline cannot ask it
            // for a first picture.
            true,
        );
        pp_info!(pp_log: source.pp_log(), "opened: {width}x{height}, alpha {alpha:?}");
        Ok((
            Self(source),
            VulkanDmaBufHandle {
                pusher,
                import: Arc::new(Mutex::new(Import {
                    pass,
                    gpu: Arc::clone(device.shared()),
                    width,
                    height,
                    alpha,
                })),
            },
        ))
    }
}

impl VulkanDmaBufHandle {
    /// Copies `buffer` into this pipeline and pushes it downstream as one
    /// frame, stamped `pts` (`None` leaves it unstamped, which is what a
    /// compositor-driven graph wants).
    ///
    /// Blocks while the source's queue is full, as
    /// [`AppSourceHandle::push`] does.
    pub fn push(
        &self,
        buffer: &DmaBuf,
        pts: Option<i64>,
    ) -> std::result::Result<(), VulkanDmaBufSourceError> {
        let frame = self.copy(buffer, pts)?;
        self.pusher
            .push(frame)
            .map_err(|_| VulkanDmaBufSourceError::Closed)
    }

    /// [`Self::push`] for a producer that cannot wait — a browser engine's
    /// paint callback, which is that engine's one thread. `Ok(false)` means
    /// the queue was full and this picture was dropped; `Err` means the copy
    /// failed or the source has ended. The copy still happens.
    pub fn try_push(
        &self,
        buffer: &DmaBuf,
        pts: Option<i64>,
    ) -> std::result::Result<bool, VulkanDmaBufSourceError> {
        let frame = self.copy(buffer, pts)?;
        self.pusher
            .try_push(frame)
            .map_err(|_| VulkanDmaBufSourceError::Closed)
    }

    /// Ends the stream, as [`AppSourceHandle::finish`] does.
    pub fn finish(&self) -> std::result::Result<(), VulkanDmaBufSourceError> {
        self.pusher
            .finish()
            .map_err(|_| VulkanDmaBufSourceError::Closed)
    }

    fn copy(
        &self,
        buffer: &DmaBuf,
        pts: Option<i64>,
    ) -> std::result::Result<MediaBuffer, VulkanDmaBufSourceError> {
        self.import
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .copy(buffer, pts)
    }
}

impl Import {
    fn copy(
        &mut self,
        buffer: &DmaBuf,
        pts: Option<i64>,
    ) -> std::result::Result<MediaBuffer, VulkanDmaBufSourceError> {
        if (buffer.width, buffer.height) != (self.width, self.height) {
            return Err(VulkanDmaBufSourceError::WrongSize {
                width: buffer.width,
                height: buffer.height,
                expected_width: self.width,
                expected_height: self.height,
            });
        }
        // SAFETY: the producer's descriptor, open for the length of the push
        // this is called within, which is as long as it is used here.
        let fd = unsafe { BorrowedFd::borrow_raw(buffer.fd) };
        drawn(fd);
        let size = [self.width, self.height];
        // Both alphas the same four bytes: what the fourth means is the
        // kernel's to say.
        let format = vk::Format::B8G8R8A8_UNORM;
        let image = DmaBufImage::import(
            &self.gpu,
            DmaBufPlane {
                fd,
                offset: buffer.offset,
                stride: buffer.stride,
                modifier: buffer.modifier,
            },
            format,
            size,
            vk::ImageUsageFlags::SAMPLED,
        )?
        .ok_or(VulkanDmaBufSourceError::NoImport)?;
        let view = View::new(&self.gpu, image.image, format, vk::ImageAspectFlags::COLOR)?;
        let mode: u32 = match self.alpha {
            DmaBufAlpha::Straight => 0,
            DmaBufAlpha::Premultiplied => 1,
            DmaBufAlpha::Ignored => 2,
        };
        let words = [
            self.width.to_ne_bytes(),
            self.height.to_ne_bytes(),
            mode.to_ne_bytes(),
            0u32.to_ne_bytes(),
        ];
        let mut output = self
            .pass
            .run_imported(image.image, view.view, size, &immediates(&words))
            .map_err(|error| VulkanDmaBufSourceError::Pool(error.to_string()))?;
        // The copy has been waited for: the buffer is the producer's again.
        drop((view, image));
        output.set_pts(pts);
        output.set_color_space(ffmpeg::color::Space::RGB);
        output.set_color_range(ffmpeg::color::Range::JPEG);
        Ok(MediaBuffer::Video(Arc::new(output).into()))
    }
}

/// Waits, up to [`DRAWN_WITHIN`], for what is still drawing into the
/// DMA-BUF `fd` to finish: a DMA-BUF polls readable once every write fenced
/// on it has. A driver that fences nothing answers at once.
fn drawn(fd: BorrowedFd<'_>) {
    use std::os::fd::AsRawFd;
    let mut poll = libc::pollfd {
        fd: fd.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    // SAFETY: one live `pollfd`, for a descriptor open for the call.
    unsafe {
        libc::poll(&mut poll, 1, DRAWN_WITHIN.as_millis() as libc::c_int);
    }
}

#[cfg(test)]
mod tests {
    use std::os::fd::AsRawFd;

    use super::*;
    use crate::{
        element::RawSink,
        elements::VulkanDownload,
        test_support::{capture, try_vulkan_device},
    };

    /// A source on `device`, or `None` — with the reason — where the device
    /// cannot import a DMA-BUF, as a software Vulkan cannot.
    fn opened(
        device: &VulkanDevice,
        width: u32,
        height: u32,
        alpha: DmaBufAlpha,
    ) -> Option<(VulkanDmaBufSource, VulkanDmaBufHandle)> {
        match VulkanDmaBufSource::new("dmabuf", device, width, height, alpha, 1) {
            Ok(opened) => Some(opened),
            Err(VulkanDmaBufSourceError::NoImport) => {
                eprintln!("skipping: this Vulkan device cannot import a DMA-BUF");
                None
            }
            Err(error) => panic!("the source opens: {error}"),
        }
    }

    /// Pushes `pixels` — `width` x `height` packed BGRA, in a GBM buffer of
    /// `modifier` — through the source's copy, and brings the frame back to
    /// the CPU as rows of BGRA. `None` means skip.
    fn round_trip(
        pixels: &[u8],
        width: u32,
        height: u32,
        alpha: DmaBufAlpha,
        modifier: u64,
    ) -> Option<Vec<u8>> {
        let device = try_vulkan_device()?;
        let (buffer, stride) =
            super::super::dmabuf::tests::gbm_buffer_with(pixels, width, height, modifier)?;
        let (_source, handle) = opened(&device, width, height, alpha)?;
        let dmabuf = DmaBuf {
            fd: buffer.as_raw_fd(),
            offset: 0,
            stride,
            modifier,
            width,
            height,
        };
        let frame = handle.copy(&dmabuf, Some(5)).expect("the copy");

        let mut download = VulkanDownload::new("download", &device);
        let received = capture(&mut download);
        download.consume(frame).expect("download");
        let received = received.lock().unwrap();
        let MediaBuffer::Video(frame) = &received[0] else {
            panic!("expected a Video buffer, got {}", received[0].kind());
        };
        assert_eq!(frame.pts(), Some(5), "the copy dropped the pts");
        assert_eq!(frame.format(), ffmpeg::format::Pixel::BGRA);
        let stride = frame.stride(0);
        Some(
            (0..height as usize)
                .flat_map(|row| {
                    frame.data(0)[row * stride..row * stride + width as usize * 4].to_vec()
                })
                .collect(),
        )
    }

    /// A picture whose every pixel differs from its neighbours both ways, so
    /// that a row or a channel out of place is seen.
    fn pattern(width: u32, height: u32) -> Vec<u8> {
        (0..height)
            .flat_map(|row| {
                (0..width).flat_map(move |column| {
                    [
                        column as u8 * 3,
                        row as u8 * 7,
                        200 - column as u8,
                        255 - row as u8,
                    ]
                })
            })
            .collect()
    }

    #[test]
    fn a_straight_buffer_arrives_unchanged() {
        let pixels = pattern(64, 32);
        let Some(out) = round_trip(&pixels, 64, 32, DmaBufAlpha::Straight, 0) else {
            return;
        };
        assert_eq!(out, pixels);
    }

    /// Chromium's off-screen pictures on NVIDIA, in its own tiles.
    #[test]
    fn a_block_linear_buffer_arrives_unchanged() {
        const NVIDIA_BLOCK_LINEAR: u64 = 0x0300_0000_0060_6014;
        let pixels = pattern(64, 32);
        let Some(out) = round_trip(&pixels, 64, 32, DmaBufAlpha::Straight, NVIDIA_BLOCK_LINEAR)
        else {
            return;
        };
        assert_eq!(out, pixels);
    }

    /// Premultiplied colour is divided back out, rounding to nearest, and a
    /// fully transparent pixel stays black rather than dividing by zero.
    #[test]
    fn a_premultiplied_buffer_arrives_straight() {
        #[rustfmt::skip]
        let pixels = [
            100, 50, 25, 128,
            255, 128, 0, 255,
            0, 0, 0, 0,
            64, 64, 64, 64,
        ];
        let Some(out) = round_trip(&pixels, 4, 1, DmaBufAlpha::Premultiplied, 0) else {
            return;
        };
        #[rustfmt::skip]
        let expected = [
            199, 100, 50, 128,
            255, 128, 0, 255,
            0, 0, 0, 0,
            255, 255, 255, 64,
        ];
        assert_eq!(out, expected);
    }

    /// A byte the producer says means nothing comes out opaque.
    #[test]
    fn an_ignored_alpha_arrives_opaque() {
        let pixels = [10, 20, 30, 0, 40, 50, 60, 77];
        let Some(out) = round_trip(&pixels, 2, 1, DmaBufAlpha::Ignored, 0) else {
            return;
        };
        assert_eq!(out, [10, 20, 30, 255, 40, 50, 60, 255]);
    }

    /// Nothing here resizes, so a buffer of another size is a caller's
    /// mistake, reported as one before anything is mapped.
    #[test]
    fn a_buffer_of_another_size_is_refused() {
        let Some(device) = try_vulkan_device() else {
            return;
        };
        let Some((_source, handle)) = opened(&device, 8, 8, DmaBufAlpha::Straight) else {
            return;
        };
        let error = handle
            .push(
                &DmaBuf {
                    fd: -1,
                    offset: 0,
                    stride: 64,
                    modifier: 0,
                    width: 16,
                    height: 16,
                },
                None,
            )
            .expect_err("a buffer of another size must be refused");
        assert!(
            matches!(
                error,
                VulkanDmaBufSourceError::WrongSize {
                    width: 16,
                    height: 16,
                    expected_width: 8,
                    expected_height: 8,
                }
            ),
            "{error:?}"
        );
    }
}
