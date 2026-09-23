use std::sync::Arc;

use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_error, pp_info};
use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    error::Result,
    frame_size::ForSize,
    pad::SrcPad,
    pool::{UnboundObjectPool, UnboundObjectPoolRef},
    repeat::{PerFrameTransform, RepeatedOutput},
};

/// Errors specific to [`D3d12Download`].
#[derive(Debug, ThisError)]
pub enum D3d12DownloadError {
    /// FFmpeg could not take a second reference to the download already in
    /// hand, which is how an unchanged texture is answered.
    #[error("failed to reference the previous download (code {0})")]
    FrameRef(i32),

    /// The input frame is not backed by a D3D12 texture.
    #[error("D3d12Download only accepts Pixel::D3D12 frames, got {0:?}")]
    UnsupportedFormat(ffmpeg::format::Pixel),
    /// The sink received a buffer other than decoded video or end-of-stream.

    #[error("D3d12Download only accepts Video and Eos buffers, got a {0}")]
    UnsupportedBuffer(&'static str),
    /// The frame does not retain the D3D12 hardware frames context that owns it.

    #[error("D3D12 frame has no valid hardware frames context")]
    MissingFramesContext,
    /// The D3D12 hardware surface is not NV12.

    #[error("D3d12Download only reads NV12 D3D12 surfaces, got {0:?}")]
    UnsupportedSurfaceFormat(ffmpeg::format::Pixel),
    /// FFmpeg failed to transfer D3D12 pixels into a CPU frame.

    #[error("failed to download D3D12 frame (code {0})")]
    TransferData(i32),
    /// FFmpeg could not copy timing and color metadata to the CPU frame.

    #[error("failed to copy downloaded frame metadata (code {0})")]
    CopyProperties(i32),
}

/// Downloads GPU-resident `Pixel::D3D12` video frames to CPU-resident
/// `Pixel::NV12` frames.
///
/// This is the exit from a D3D12VA pipeline for CPU-only stages such as
/// [`crate::elements::SwScaler`], [`crate::elements::SwEncoder`],
/// `OrtDetector`, and [`crate::elements::AppSink`]:
/// `D3d12Decoder -> D3d12Download -> SwScaler -> ...`.
///
/// A D3D12 device is deliberately not a constructor argument. The source
/// frame's own `AVHWFramesContext` owns the device and synchronization state,
/// and `av_hwframe_transfer_data` waits on its D3D12VA fence before copying.
/// Supplying a second device here would create an invariant this operation
/// neither needs nor uses.
///
/// Only NV12-backed D3D12 frames are accepted, matching
/// [`crate::elements::D3d12Decoder`] and [`crate::elements::D3d12Upload`].
/// PTS, duration, and color metadata are copied without creating a new
/// timeline. The size of the frames comes from the frames themselves.
pub struct D3d12Download {
    pp_log: PpLog,
    name: Arc<str>,
    pad: SrcPad,
    /// CPU NV12 frames the transfer writes into, made for the size of the
    /// frames actually arriving and made again when that changes — see
    /// `ForSize`.
    pool: ForSize<UnboundObjectPool<ffmpeg::frame::Video>>,
    /// The last download and the texture it came from, so a producer that
    /// re-emits an unchanged picture is answered with the bytes already in
    /// system memory — see [`RepeatedOutput`].
    repeated: RepeatedOutput,
}

impl D3d12Download {
    /// Creates a downloader for NV12 D3D12 frames.
    pub fn new(name: impl Into<String>) -> Self {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::D3d12Download, &name, None);
        let pad = SrcPad::with_contract(
            format!("{name}_src"),
            OutputContract::Fixed(
                PortContract::frame(MediaKind::VideoFrame, MemoryDomain::System)
                    .with_layouts(crate::contract::PixelLayoutSet::NV12),
            ),
        );
        pp_info!(pp_log: &pp_log, "opened: D3D12 -> NV12");
        Self {
            pp_log,
            name,
            pad,
            pool: ForSize::new(),
            repeated: RepeatedOutput::new(),
        }
    }

    fn download(
        &mut self,
        source: &ffmpeg::frame::Video,
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        if source.format() != ffmpeg::format::Pixel::D3D12 {
            pp_error!(self, "unsupported pixel format: {:?}", source.format());
            return Err(D3d12DownloadError::UnsupportedFormat(source.format()).into());
        }
        // SAFETY: `source` is a live D3D12 hardware frame. Its hardware-frame
        // reference is null-checked before dereferencing, and only the device
        // context identity is read while the frame keeps both contexts alive.
        unsafe {
            let frames_ref = (*source.as_ptr()).hw_frames_ctx;
            if frames_ref.is_null() || (*frames_ref).data.is_null() {
                pp_error!(self, "frame has no valid hardware frames context");
                return Err(D3d12DownloadError::MissingFramesContext.into());
            }
            let frames_ctx = (*frames_ref).data as *const ffi::AVHWFramesContext;
            let sw_format = ffmpeg::format::Pixel::from((*frames_ctx).sw_format);
            if sw_format != ffmpeg::format::Pixel::NV12 {
                let error = D3d12DownloadError::UnsupportedSurfaceFormat(sw_format);
                pp_error!(self, "{error}");
                return Err(error.into());
            }
        }

        let pp_log = &self.pp_log;
        let mut destination = self
            .pool
            .get(source.width(), source.height(), |width, height| {
                pp_info!(pp_log: pp_log, "reading back {width}x{height}");
                UnboundObjectPool::new(
                    0,
                    move || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, width, height),
                    |_| {},
                )
            })
            .get();
        // SAFETY: `destination` is this element's writable pooled frame and
        // `source` is the validated live hardware frame. FFmpeg initializes
        // the destination pixels, after which copying properties is valid.
        unsafe {
            let dst = destination.as_mut_ptr();
            let ret = ffi::av_hwframe_transfer_data(dst, source.as_ptr(), 0);
            if ret < 0 {
                pp_error!(self, "av_hwframe_transfer_data failed: {ret}");
                return Err(D3d12DownloadError::TransferData(ret).into());
            }
            let ret = ffi::av_frame_copy_props(dst, source.as_ptr());
            if ret < 0 {
                pp_error!(self, "av_frame_copy_props failed: {ret}");
                return Err(D3d12DownloadError::CopyProperties(ret).into());
            }
        }

        Ok(destination)
    }
}

impl PerFrameTransform for D3d12Download {
    fn repeated(&mut self) -> &mut RepeatedOutput {
        &mut self.repeated
    }

    fn frame_ref_failed(&self, code: i32) -> crate::error::Error {
        pp_error!(self, "av_frame_ref failed: {code}");
        D3d12DownloadError::FrameRef(code).into()
    }

    fn produce(
        &mut self,
        source: &Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        self.download(source)
    }
}

impl Element for D3d12Download {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::D3d12Download
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for D3d12Download {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for D3d12Download {
    /// The mirror of D3d12Upload: only a device resource has anything to bring back.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::D3d12)
                .with_layouts(crate::contract::PixelLayoutSet::NV12),
        )
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            // The same texture as last time is the same pixels as last time,
            // and reading them back again produces the frame already in hand
            // — see [`PerFrameTransform`].
            MediaBuffer::Video(frame) => {
                let downloaded = self.transform(&frame)?;
                self.pad.push(MediaBuffer::Video(downloaded))
            }
            MediaBuffer::Eos => self.pad.push(MediaBuffer::Eos),
            other => {
                let error = D3d12DownloadError::UnsupportedBuffer(other.kind());
                pp_error!(self, "{error}");
                Err(error.into())
            }
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        // Nothing local to react to beyond the cached download.
        if matches!(msg, ControlMsg::Flush | ControlMsg::Stop) {
            self.repeated.clear();
        }
        self.pad.control(msg)
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::CapturingSink;
    use std::sync::Mutex;

    use super::*;
    use crate::elements::D3d12Upload;
    use crate::test_support::try_d3d12_device as try_device;

    #[test]
    fn upload_download_round_trip_preserves_pixels_and_metadata() {
        let Some(device) = try_device() else {
            return;
        };
        let (width, height) = (16u32, 16u32);
        let Ok(mut upload) = D3d12Upload::new("upload", &device) else {
            eprintln!("skipping: FFmpeg could not create a D3D12VA frames context");
            return;
        };
        let mut download = D3d12Download::new("download");
        let received = Arc::new(Mutex::new(Vec::new()));
        download.src_pads()[0].link(Box::new(CapturingSink {
            pp_log: element_pp_log(ElementType::Other, "capture", None),
            received: received.clone(),
        }));
        upload.src_pads()[0].link(Box::new(download));

        let pool = UnboundObjectPool::new(
            0,
            move || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, width, height),
            |_| {},
        );
        let mut source = pool.get();
        for row in 0..height as usize {
            let stride = source.stride(0);
            let data = source.data_mut(0);
            for column in 0..width as usize {
                data[row * stride + column] = (row * width as usize + column) as u8;
            }
        }
        for row in 0..height as usize / 2 {
            let stride = source.stride(1);
            let data = source.data_mut(1);
            for column in (0..width as usize).step_by(2) {
                data[row * stride + column] = 73;
                data[row * stride + column + 1] = 181;
            }
        }
        source.set_pts(Some(42));
        source.set_color_space(ffmpeg::color::Space::BT709);
        source.set_color_range(ffmpeg::color::Range::MPEG);
        // SAFETY: the test uniquely owns the live frame and mutates its plain
        // `duration` metadata field before publishing it.
        unsafe { (*source.as_mut_ptr()).duration = 3 };

        upload
            .consume(MediaBuffer::Video(Arc::new(source)))
            .expect("D3D12 upload/download should succeed");

        let received = received.lock().unwrap();
        let MediaBuffer::Video(frame) = &received[0] else {
            panic!("expected a Video buffer");
        };
        assert_eq!(frame.format(), ffmpeg::format::Pixel::NV12);
        assert_eq!(frame.pts(), Some(42));
        assert_eq!(frame.color_space(), ffmpeg::color::Space::BT709);
        assert_eq!(frame.color_range(), ffmpeg::color::Range::MPEG);
        // SAFETY: `frame` is live for this read of its plain metadata field.
        assert_eq!(unsafe { (*frame.as_ptr()).duration }, 3);

        for row in 0..height as usize {
            let stride = frame.stride(0);
            for column in 0..width as usize {
                assert_eq!(
                    frame.data(0)[row * stride + column],
                    (row * width as usize + column) as u8
                );
            }
        }
        for row in 0..height as usize / 2 {
            let stride = frame.stride(1);
            for column in (0..width as usize).step_by(2) {
                assert_eq!(&frame.data(1)[row * stride + column..][..2], &[73, 181]);
            }
        }
    }

    /// A source that changes resolution mid-stream was a per-frame error on
    /// both sides of this round trip while the frames context and the CPU
    /// frames were allocated before any frame arrived.
    #[test]
    fn a_source_that_changes_resolution_is_followed() {
        let Some(device) = try_device() else {
            return;
        };
        let Ok(mut upload) = D3d12Upload::new("upload", &device) else {
            eprintln!("skipping: FFmpeg could not create a D3D12VA frames context");
            return;
        };
        let mut download = D3d12Download::new("download");
        let received = Arc::new(Mutex::new(Vec::new()));
        download.src_pads()[0].link(Box::new(CapturingSink {
            pp_log: element_pp_log(ElementType::Other, "capture", None),
            received: received.clone(),
        }));
        upload.src_pads()[0].link(Box::new(download));

        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        for (width, height) in [(16u32, 16u32), (8, 8)] {
            let mut source = pool.get();
            *source = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, width, height);
            upload
                .consume(MediaBuffer::Video(Arc::new(source)))
                .expect("a D3D12 round trip at this size");
        }

        let received = received.lock().unwrap();
        let sizes: Vec<(u32, u32)> = received
            .iter()
            .map(|buffer| {
                let MediaBuffer::Video(frame) = buffer else {
                    panic!("expected a Video buffer");
                };
                (frame.width(), frame.height())
            })
            .collect();
        assert_eq!(
            sizes,
            [(16, 16), (8, 8)],
            "each frame crosses at its own size"
        );
    }

    #[test]
    fn rejects_cpu_frames_and_non_video_buffers() {
        let mut download = D3d12Download::new("download");
        let pool = UnboundObjectPool::new(
            0,
            || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, 16, 16),
            |_| {},
        );
        let error = download
            .consume(MediaBuffer::Video(Arc::new(pool.get())))
            .expect_err("a CPU frame must be rejected");
        assert!(error.to_string().contains("only accepts Pixel::D3D12"));

        let error = download
            .consume(MediaBuffer::Packet(Arc::new(ffmpeg::Packet::empty())))
            .expect_err("a packet must be rejected");
        assert!(error.to_string().contains("Video and Eos"));
    }
}
