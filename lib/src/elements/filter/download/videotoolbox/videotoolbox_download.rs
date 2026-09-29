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
    element::{Element, ElementType, Output, Transform, element_pp_log},
    error::Result,
    platform::macos::videotoolbox::{NotVideoToolbox, sw_format_of},
    pool::{UnboundObjectPool, UnboundObjectPoolRef},
    repeat::{PerFrameTransform, RepeatedOutput},
    transform::{TransformStage, transform_filter},
};

/// Errors specific to [`VideoToolboxDownload`]. Converts into the
/// crate-wide `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum VideoToolboxDownloadError {
    /// FFmpeg could not take a second reference to the download already in
    /// hand, which is how an unchanged image is answered.
    #[error("failed to reference the previous download (code {0})")]
    FrameRef(i32),

    /// The sink received a buffer other than decoded video or end-of-stream.
    #[error("VideoToolboxDownload only accepts Video buffers, got a {0}")]
    UnsupportedBuffer(&'static str),

    /// The frame is not a VideoToolbox frame.
    #[error("VideoToolboxDownload reads VideoToolbox frames, got a {0:?} frame")]
    NotVideoToolbox(ffmpeg::format::Pixel),

    /// The frame says it is a VideoToolbox frame but carries no frames
    /// context to read it through.
    #[error("VideoToolboxDownload was handed a VideoToolbox frame with no frames context")]
    NoFramesContext,

    /// FFmpeg failed to copy the pixel buffer into system memory.
    #[error("VideoToolbox to CPU transfer failed (code {0})")]
    Transfer(i32),
}

impl From<NotVideoToolbox> for VideoToolboxDownloadError {
    fn from(error: NotVideoToolbox) -> Self {
        match error {
            NotVideoToolbox::Format(format) => Self::NotVideoToolbox(format),
            NotVideoToolbox::NoFramesContext => Self::NoFramesContext,
        }
    }
}

/// What the frames this reads back may hold.
const LAYOUTS: PixelLayoutSet =
    PixelLayoutSet::from_slice(&[PixelLayout::Nv12, PixelLayout::P010, PixelLayout::Bgra]);

/// Reads VideoToolbox frames back into system memory, in the layout they
/// hold — the mirror of [`crate::elements::VideoToolboxUpload`], and what
/// lets a VideoToolbox frame reach anything that reads pixel bytes.
///
/// A `Filter`: receives via `Sink`, pushes the downloaded frame into its own
/// single src pad. PTS, duration, and color metadata are carried across with
/// `av_frame_copy_props`, so this creates no new timeline.
///
/// It takes no device: a pixel buffer is readable wherever in the process it
/// was made, through the frames context each frame carries.
pub struct VideoToolboxDownload(TransformStage<Downloading>);

transform_filter!(VideoToolboxDownload);

/// What a [`VideoToolboxDownload`] does to each frame: all of its work, which the
/// framework makes the filter.
struct Downloading {
    pp_log: PpLog,
    name: Arc<str>,
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

impl VideoToolboxDownload {
    /// A download of whatever VideoToolbox frames it is handed.
    pub fn new(name: impl Into<String>) -> Self {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::VideoToolboxDownload, &name, None);
        pp_info!(pp_log: &pp_log, "opened: VideoToolbox ->");
        Self(TransformStage::new(Downloading {
            name,
            pp_log,
            pool: None,
            repeated: RepeatedOutput::new(),
        }))
    }
}

impl Downloading {
    fn download(
        &mut self,
        source: &ffmpeg::frame::Video,
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        let format = sw_format_of(source)
            .map_err(VideoToolboxDownloadError::from)
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
        // pixel buffer's layout and size and kept across calls; `source` is a
        // live VideoToolbox frame with a frames context, checked just above.
        unsafe {
            let dst = destination.as_mut_ptr();
            let code = ffi::av_hwframe_transfer_data(dst, source.as_ptr(), 0);
            if code < 0 {
                pp_error!(self, "av_hwframe_transfer_data failed: {code}");
                return Err(VideoToolboxDownloadError::Transfer(code).into());
            }
            // Pixels only move with the transfer: PTS, duration and colour
            // are part of the buffer contract.
            ffi::av_frame_copy_props(dst, source.as_ptr());
        }
        Ok(destination)
    }
}

impl PerFrameTransform for Downloading {
    fn repeated(&mut self) -> &mut RepeatedOutput {
        &mut self.repeated
    }

    fn frame_ref_failed(&self, code: i32) -> crate::error::Error {
        pp_error!(self, "av_frame_ref failed: {code}");
        VideoToolboxDownloadError::FrameRef(code).into()
    }

    fn produce(
        &mut self,
        source: &Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        self.download(source)
    }
}

impl Element for Downloading {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::VideoToolboxDownload
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Transform for Downloading {
    /// Only device memory has anything to bring back.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::VideoToolbox)
                .with_layouts(LAYOUTS),
        )
    }

    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        match buf {
            // The same pixel buffer as last time is the same pixels, and
            // reading it back again makes the frame already in hand.
            MediaBuffer::Video(frame) => {
                let downloaded = PerFrameTransform::transform(self, &frame)?;
                out.push(MediaBuffer::Video(downloaded));
                Ok(())
            }
            other => Err(VideoToolboxDownloadError::UnsupportedBuffer(other.kind()).into()),
        }
    }

    fn reset(&mut self) {
        // A per-frame transfer: nothing held beyond the cached download.
        self.repeated.clear();
    }

    fn output_contract(&self) -> OutputContract {
        OutputContract::SameLayout(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::System).with_layouts(LAYOUTS),
        )
    }
}

impl Drop for Downloading {
    fn drop(&mut self) {
        pp_info!(self, "dropped");
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::element::{Sink, Source};
    use crate::{
        elements::{VideoToolboxDevice, VideoToolboxUpload},
        test_support::{CapturingSink, try_videotoolbox_device},
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
    fn round_trip(
        device: &VideoToolboxDevice,
        frames: Vec<ffmpeg::frame::Video>,
    ) -> Vec<MediaBuffer> {
        let mut upload = VideoToolboxUpload::new("upload", device);
        let uploaded = capture(&mut upload);
        for frame in frames {
            upload
                .consume(MediaBuffer::video(frame))
                .expect("upload a frame");
        }
        crate::stream::deliver(&mut upload, &crate::stream::StreamEvent::Eos).expect("eos");
        let mut download = VideoToolboxDownload::new("download");
        let downloaded = capture(&mut download);
        for buffer in uploaded.lock().unwrap().drain(..) {
            if let MediaBuffer::Video(frame) = &buffer {
                assert_eq!(frame.format(), ffmpeg::format::Pixel::VIDEOTOOLBOX);
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
    /// their timestamps.
    #[test]
    fn every_layout_comes_back_as_it_went_up() {
        let Some(device) = try_videotoolbox_device() else {
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
        assert_eq!(back.len(), formats.len(), "{} buffers", back.len());
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
        let Some(device) = try_videotoolbox_device() else {
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

    /// A frame in a layout that does not go up, and a frame that is not a
    /// VideoToolbox one, are each refused with the reason.
    #[test]
    fn what_cannot_be_moved_is_refused_by_name() {
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let mut upload = VideoToolboxUpload::new("upload", &device);
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

        let mut download = VideoToolboxDownload::new("download");
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
    }
}
