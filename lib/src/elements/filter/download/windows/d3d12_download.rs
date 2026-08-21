use std::sync::Arc;

use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_error, pp_info};
use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    error::Result,
    pad::SrcPad,
    pool::UnboundObjectPool,
};

/// Errors specific to [`D3d12Download`].
#[derive(Debug, ThisError)]
pub enum D3d12DownloadError {
    #[error("D3d12Download only accepts Pixel::D3D12 frames, got {0:?}")]
    UnsupportedFormat(ffmpeg::format::Pixel),

    #[error("D3d12Download only accepts Video and Eos buffers, got a {0}")]
    UnsupportedBuffer(&'static str),

    #[error(
        "frame is {actual_width}x{actual_height}, but D3d12Download was opened for \
         {expected_width}x{expected_height}"
    )]
    DimensionMismatch {
        actual_width: u32,
        actual_height: u32,
        expected_width: u32,
        expected_height: u32,
    },

    #[error("D3D12 frame has no valid hardware frames context")]
    MissingFramesContext,

    #[error("D3d12Download only reads NV12 D3D12 surfaces, got {0:?}")]
    UnsupportedSurfaceFormat(ffmpeg::format::Pixel),

    #[error("failed to download D3D12 frame (code {0})")]
    TransferData(i32),

    #[error("failed to copy downloaded frame metadata (code {0})")]
    CopyProperties(i32),
}

/// Downloads GPU-resident `Pixel::D3D12` video frames to CPU-resident
/// `Pixel::NV12` frames.
///
/// This is the exit from a D3D12VA pipeline for CPU-only stages such as
/// [`crate::elements::SwScaler`], [`crate::elements::SwEncoder`],
/// [`crate::elements::OrtDetector`], and [`crate::elements::AppSink`]:
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
/// timeline. `width` and `height` are fixed for this element's lifetime.
pub struct D3d12Download {
    pp_log: PpLog,
    name: Arc<str>,
    width: u32,
    height: u32,
    pad: SrcPad,
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
}

impl D3d12Download {
    pub fn new(name: impl Into<String>, width: u32, height: u32) -> Self {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::D3d12Download, &name, None);
        let pad = SrcPad::new(format!("{name}_src"));
        let pool = UnboundObjectPool::new(
            0,
            move || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, width, height),
            |_| {},
        );
        pp_info!(pp_log: &pp_log, "opened: {width}x{height} D3D12 -> NV12");
        Self {
            pp_log,
            name,
            width,
            height,
            pad,
            pool,
        }
    }

    fn download(&mut self, source: &ffmpeg::frame::Video) -> Result<()> {
        if source.format() != ffmpeg::format::Pixel::D3D12 {
            pp_error!(self, "unsupported pixel format: {:?}", source.format());
            return Err(D3d12DownloadError::UnsupportedFormat(source.format()).into());
        }
        if source.width() != self.width || source.height() != self.height {
            let error = D3d12DownloadError::DimensionMismatch {
                actual_width: source.width(),
                actual_height: source.height(),
                expected_width: self.width,
                expected_height: self.height,
            };
            pp_error!(self, "{error}");
            return Err(error.into());
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

        let mut destination = self.pool.get();
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

        self.pad.push(MediaBuffer::Video(Arc::new(destination)))
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
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Video(frame) => self.download(&frame),
            MediaBuffer::Eos => self.pad.push(MediaBuffer::Eos),
            other => {
                let error = D3d12DownloadError::UnsupportedBuffer(other.kind());
                pp_error!(self, "{error}");
                Err(error.into())
            }
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        self.pad.control(msg)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::elements::D3d12Upload;
    use crate::test_support::try_d3d12_device as try_device;

    struct CapturingSink {
        pp_log: PpLog,
        received: Arc<Mutex<Vec<MediaBuffer>>>,
    }

    impl Element for CapturingSink {
        fn name(&self) -> Arc<str> {
            "capture".into()
        }

        fn element_type(&self) -> ElementType {
            ElementType::Other
        }

        fn pp_log(&self) -> &PpLog {
            &self.pp_log
        }

        fn pp_log_mut(&mut self) -> &mut PpLog {
            &mut self.pp_log
        }
    }

    impl Sink for CapturingSink {
        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            self.received.lock().unwrap().push(buf);
            Ok(())
        }

        fn control(&mut self, _msg: ControlMsg) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn upload_download_round_trip_preserves_pixels_and_metadata() {
        let Some(device) = try_device() else {
            return;
        };
        let (width, height) = (16u32, 16u32);
        let Ok(mut upload) = D3d12Upload::new("upload", &device, width, height) else {
            eprintln!("skipping: FFmpeg could not create a D3D12VA frames context");
            return;
        };
        let mut download = D3d12Download::new("download", width, height);
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

    #[test]
    fn rejects_cpu_frames_and_non_video_buffers() {
        let mut download = D3d12Download::new("download", 16, 16);
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
