use std::sync::Arc;

use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_error, pp_info};

use crate::{
    buffer::MediaBuffer,
    contract::{
        InputContract, MediaKind, MemoryDomain, OutputContract, PixelLayout, PixelLayoutSet,
        PortContract,
    },
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    elements::VulkanDevice,
    error::Result,
    pad::SrcPad,
    platform::{
        ffmpeg::AvBufferRef,
        vulkan::frames::{NotOurs, sw_format_of},
    },
    pool::{UnboundObjectPool, UnboundObjectPoolRef},
    repeat::{PerFrameTransform, RepeatedOutput},
};

/// Errors specific to [`VulkanDownload`]. Converts into the crate-wide
/// `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum VulkanDownloadError {
    /// FFmpeg could not take a second reference to the download already in
    /// hand, which is how an unchanged image is answered.
    #[error("failed to reference the previous download (code {0})")]
    FrameRef(i32),

    /// The sink received a buffer other than decoded video or end-of-stream.
    #[error("VulkanDownload only accepts Video and Eos buffers, got a {0}")]
    UnsupportedBuffer(&'static str),

    /// The frame is not a Vulkan frame.
    #[error("VulkanDownload reads Vulkan frames, got a {0:?} frame")]
    NotVulkan(ffmpeg::format::Pixel),

    /// The frame was made on another `VkDevice` than this element's.
    #[error("VulkanDownload was handed a frame from another Vulkan device")]
    ForeignDevice,

    /// FFmpeg failed to copy the image's pixels into system memory.
    #[error("Vulkan to CPU transfer failed (code {0})")]
    Transfer(i32),
}

impl From<NotOurs> for VulkanDownloadError {
    fn from(error: NotOurs) -> Self {
        match error {
            NotOurs::NotVulkan(format) => Self::NotVulkan(format),
            NotOurs::ForeignDevice => Self::ForeignDevice,
        }
    }
}

/// What the frames this reads back may hold.
const LAYOUTS: PixelLayoutSet =
    PixelLayoutSet::from_slice(&[PixelLayout::Nv12, PixelLayout::P010, PixelLayout::Bgra]);

/// Reads Vulkan frames back into system memory, in the layout they hold —
/// the mirror of [`crate::elements::VulkanUpload`], and what lets a Vulkan
/// frame reach anything that reads pixel bytes.
///
/// A `Filter`: receives via `Sink`, pushes the downloaded frame into its own
/// single src pad. PTS, duration, and color metadata are carried across with
/// `av_frame_copy_props`, so this creates no new timeline.
///
/// Every frame is a copy from the GPU's memory, over PCIe on a discrete GPU:
/// put this where the pipeline genuinely has to leave Vulkan.
pub struct VulkanDownload {
    pp_log: PpLog,
    name: Arc<str>,
    /// Held for `device_ctx`'s sake: the pointer stays a valid identity only
    /// as long as this reference does.
    _hw_device_ctx: Arc<AvBufferRef>,
    /// This element's device, to compare an incoming frame's against. Only
    /// ever compared.
    device_ctx: *const ffi::AVHWDeviceContext,
    pad: SrcPad,
    /// CPU frames the transfer writes into, and the size and layout they are
    /// made for, made again when the frames arriving change either.
    pool: Option<(
        u32,
        u32,
        ffmpeg::format::Pixel,
        UnboundObjectPool<ffmpeg::frame::Video>,
    )>,
    /// The last download and the image it came from — see
    /// [`RepeatedOutput`].
    repeated: RepeatedOutput,
}

// SAFETY: `hw_device_ctx` is a heap-allocated FFmpeg buffer with no thread
// affinity of its own, `device_ctx` is only ever compared, and `&mut self`
// on every method that touches them rules out concurrent access.
unsafe impl Send for VulkanDownload {}

impl VulkanDownload {
    /// `device` must be the [`VulkanDevice`] the frames were made on — a
    /// frame from another is refused. This element takes its own reference,
    /// so `device` itself need not outlive the call.
    pub fn new(name: impl Into<String>, device: &VulkanDevice) -> Self {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::VulkanDownload, &name, None);
        let pad = SrcPad::with_contract(
            format!("{name}_src"),
            OutputContract::SameLayout(
                PortContract::frame(MediaKind::VideoFrame, MemoryDomain::System)
                    .with_layouts(LAYOUTS),
            ),
        );
        pp_info!(pp_log: &pp_log, "opened: Vulkan on {} ->", device.name());
        Self {
            name,
            pp_log,
            _hw_device_ctx: device.retain(),
            device_ctx: device.device_ctx(),
            pad,
            pool: None,
            repeated: RepeatedOutput::new(),
        }
    }

    fn download(
        &mut self,
        source: &ffmpeg::frame::Video,
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        let format = sw_format_of(source, self.device_ctx)
            .map_err(VulkanDownloadError::from)
            .inspect_err(|error| pp_error!(self, "{error}"))?;
        let (width, height) = (source.width(), source.height());
        if !self
            .pool
            .as_ref()
            .is_some_and(|&(w, h, f, _)| (w, h, f) == (width, height, format))
        {
            pp_info!(self, "reading back {width}x{height} {format:?}");
            let pool = UnboundObjectPool::new(
                0,
                move || ffmpeg::frame::Video::new(format, width, height),
                |_| {},
            );
            self.pool = Some((width, height, format, pool));
        }
        let mut destination = self.pool.as_ref().expect("made above").3.get();
        // SAFETY: `dst` is the pooled frame's own `AVFrame`, allocated for the
        // image's layout and size and kept across calls; `source` is a live
        // Vulkan frame on this element's device, checked just above.
        unsafe {
            let dst = destination.as_mut_ptr();
            let code = ffi::av_hwframe_transfer_data(dst, source.as_ptr(), 0);
            if code < 0 {
                pp_error!(self, "av_hwframe_transfer_data failed: {code}");
                return Err(VulkanDownloadError::Transfer(code).into());
            }
            // Pixels only move with the transfer: PTS, duration and colour
            // are part of the buffer contract.
            ffi::av_frame_copy_props(dst, source.as_ptr());
        }
        Ok(destination)
    }
}

impl PerFrameTransform for VulkanDownload {
    fn repeated(&mut self) -> &mut RepeatedOutput {
        &mut self.repeated
    }

    fn frame_ref_failed(&self, code: i32) -> crate::error::Error {
        pp_error!(self, "av_frame_ref failed: {code}");
        VulkanDownloadError::FrameRef(code).into()
    }

    fn produce(
        &mut self,
        source: &Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        self.download(source)
    }
}

impl Element for VulkanDownload {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::VulkanDownload
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for VulkanDownload {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for VulkanDownload {
    /// Only device memory has anything to bring back.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::Vulkan).with_layouts(LAYOUTS),
        )
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            // The same image as last time is the same pixels, and reading it
            // back again makes the frame already in hand.
            MediaBuffer::Video(frame) => {
                let downloaded = self.transform(&frame)?;
                self.pad.push(MediaBuffer::Video(downloaded))
            }
            MediaBuffer::Eos => self.pad.push(MediaBuffer::Eos),
            other => Err(VulkanDownloadError::UnsupportedBuffer(other.kind()).into()),
        }
    }

    fn control(&mut self, msg: &ControlMsg) -> Result<()> {
        // A per-frame transfer: nothing held beyond the cached download.
        if matches!(msg, ControlMsg::Flush | ControlMsg::Stop) {
            self.repeated.clear();
        }
        Ok(())
    }
}

impl Drop for VulkanDownload {
    fn drop(&mut self) {
        pp_info!(self, "dropped: releasing hw_device_ctx");
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::{
        elements::VulkanUpload,
        test_support::{CapturingSink, try_vulkan_device},
    };

    type Received = Arc<Mutex<Vec<MediaBuffer>>>;

    fn capture(source: &mut dyn Source) -> Received {
        let received = Arc::new(Mutex::new(Vec::new()));
        source.src_pads()[0].link(Box::new(CapturingSink {
            received: received.clone(),
            pp_log: element_pp_log(ElementType::Other, "capture", None),
        }));
        received
    }

    /// The frames `upload` pushed, each handed to `download`, and what that
    /// pushed.
    fn round_trip(device: &VulkanDevice, frames: Vec<ffmpeg::frame::Video>) -> Vec<MediaBuffer> {
        let mut upload = VulkanUpload::new("upload", device);
        let uploaded = capture(&mut upload);
        for frame in frames {
            upload
                .consume(MediaBuffer::video(frame))
                .expect("upload a frame");
        }
        upload.consume(MediaBuffer::Eos).expect("eos");
        let mut download = VulkanDownload::new("download", device);
        let downloaded = capture(&mut download);
        for buffer in uploaded.lock().unwrap().drain(..) {
            if let MediaBuffer::Video(frame) = &buffer {
                assert_eq!(frame.format(), ffmpeg::format::Pixel::VULKAN);
            }
            download.consume(buffer).expect("download a frame");
        }
        std::mem::take(&mut *downloaded.lock().unwrap())
    }

    fn frame(
        format: ffmpeg::format::Pixel,
        width: u32,
        height: u32,
        pts: i64,
    ) -> ffmpeg::frame::Video {
        let mut frame = ffmpeg::frame::Video::new(format, width, height);
        for plane in 0..frame.planes() {
            for (index, byte) in frame.data_mut(plane).iter_mut().enumerate() {
                *byte = (index * 7 + plane * 31) as u8;
            }
        }
        frame.set_pts(Some(pts));
        frame
    }

    /// Up and back again, NV12, BGRA and P010 come back as they went, with
    /// their timestamps, and the end of the stream is passed on.
    #[test]
    fn every_layout_comes_back_as_it_went_up() {
        let Some(device) = try_vulkan_device() else {
            return;
        };
        let formats = [
            ffmpeg::format::Pixel::NV12,
            ffmpeg::format::Pixel::BGRA,
            ffmpeg::format::Pixel::P010LE,
        ];
        let sent: Vec<_> = formats
            .iter()
            .enumerate()
            .map(|(pts, &format)| frame(format, 64, 32, pts as i64))
            .collect();
        let back = round_trip(&device, sent.clone());
        assert_eq!(back.len(), formats.len() + 1, "{} buffers", back.len());
        assert!(
            back.last().is_some_and(MediaBuffer::is_eos),
            "Eos passed on"
        );
        for (sent, back) in sent.iter().zip(&back) {
            let MediaBuffer::Video(back) = back else {
                panic!("expected a Video buffer, got {}", back.kind());
            };
            assert_eq!(back.format(), sent.format());
            assert_eq!(back.pts(), sent.pts());
            // Every plane of these three is as many bytes a row as the
            // first: NV12's chroma interleaves two half-width planes.
            let bytes = sent.width() as usize
                * match sent.format() {
                    ffmpeg::format::Pixel::BGRA => 4,
                    ffmpeg::format::Pixel::P010LE => 2,
                    _ => 1,
                };
            for plane in 0..sent.planes() {
                for row in 0..sent.plane_height(plane) as usize {
                    let wanted = &sent.data(plane)[row * sent.stride(plane)..][..bytes];
                    let got = &back.data(plane)[row * back.stride(plane)..][..bytes];
                    assert_eq!(wanted, got, "{:?} plane {plane} row {row}", sent.format());
                }
            }
        }
    }

    /// A software decode's YUVJ420P goes up as NV12 at full range, each
    /// chroma sample where NV12 has it.
    #[test]
    fn yuv420p_goes_up_as_nv12() {
        let Some(device) = try_vulkan_device() else {
            return;
        };
        let mut planar = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::YUVJ420P, 64, 32);
        for (plane, value) in [(0, 50u8), (1, 100), (2, 200)] {
            planar.data_mut(plane).fill(value);
        }
        planar.set_pts(Some(9));
        let back = round_trip(&device, vec![planar]);
        let MediaBuffer::Video(back) = &back[0] else {
            panic!("expected a Video buffer");
        };
        assert_eq!(back.format(), ffmpeg::format::Pixel::NV12);
        assert_eq!(back.color_range(), ffmpeg::color::Range::JPEG);
        assert_eq!(back.data(0)[0], 50, "luma");
        assert_eq!(&back.data(1)[..4], [100, 200, 100, 200], "Cb then Cr");
    }

    /// A frame in a layout that does not go up, a frame that is not a
    /// Vulkan one, and one from another device are each refused with the
    /// reason.
    #[test]
    fn what_cannot_be_moved_is_refused_by_name() {
        let Some(device) = try_vulkan_device() else {
            return;
        };
        let mut upload = VulkanUpload::new("upload", &device);
        let _ = capture(&mut upload);
        let error = upload
            .consume(MediaBuffer::video(frame(
                ffmpeg::format::Pixel::RGB24,
                16,
                16,
                0,
            )))
            .expect_err("RGB24 does not go up");
        assert!(error.to_string().contains("got RGB24"), "{error}");

        let mut download = VulkanDownload::new("download", &device);
        let _ = capture(&mut download);
        let error = download
            .consume(MediaBuffer::video(frame(
                ffmpeg::format::Pixel::NV12,
                16,
                16,
                0,
            )))
            .expect_err("a system frame has nothing to read back");
        assert!(error.to_string().contains("got a NV12 frame"), "{error}");

        let other = VulkanDevice::new().expect("a second device where there is one");
        let mut foreign = VulkanUpload::new("foreign", &other);
        let uploaded = capture(&mut foreign);
        foreign
            .consume(MediaBuffer::video(frame(
                ffmpeg::format::Pixel::NV12,
                16,
                16,
                0,
            )))
            .expect("upload on the other device");
        let buffer = uploaded.lock().unwrap().remove(0);
        let error = download
            .consume(buffer)
            .expect_err("a frame from another device");
        assert!(
            error.to_string().contains("another Vulkan device"),
            "{error}"
        );
    }
}
