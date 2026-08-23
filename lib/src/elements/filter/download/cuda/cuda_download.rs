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
    pad::SrcPad,
    platform::{
        cuda::{CudaDevice, CudaFrameFormat},
        ffmpeg::AvBufferRef,
    },
    pool::UnboundObjectPool,
};

/// Errors specific to `CudaDownload`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum CudaDownloadError {
    /// The input frame is not backed by CUDA hardware surfaces.
    #[error("CudaDownload only downloads CUDA frames, got {0:?}")]
    UnsupportedFormat(ffmpeg::format::Pixel),
    /// The sink received a buffer other than decoded video or end-of-stream.

    #[error("CudaDownload only accepts Video and Eos buffers, got a {0}")]
    UnsupportedBuffer(&'static str),
    /// Input dimensions differ from the fixed download dimensions.

    #[error(
        "frame is {actual_width}x{actual_height}, but this CudaDownload was built for \
         {expected_width}x{expected_height}"
    )]
    DimensionMismatch {
        /// Input frame width in pixels.
        actual_width: u32,
        /// Input frame height in pixels.
        actual_height: u32,
        /// Width configured for this downloader.
        expected_width: u32,
        /// Height configured for this downloader.
        expected_height: u32,
    },
    /// The CUDA surface layout differs from the expected CPU output format.

    /// The frame carries no hardware frames context, so it did not come from
    /// a CUDA producer at all.
    #[error("CUDA frame has no hardware frames context")]
    MissingFramesContext,

    /// The frame was allocated against a different CUDA context than this
    /// element's. Its device pointers mean nothing here.
    #[error("CUDA frame belongs to a different CUDA context than this CudaDownload")]
    ForeignContext,

    /// The CUDA surface layout differs from the expected CPU output format.
    #[error("this CudaDownload was built for {expected:?} surfaces, got {actual:?}")]
    UnsupportedSurfaceFormat {
        /// Surface format expected by this downloader.
        expected: ffmpeg::format::Pixel,
        /// Surface format carried by the input frame.
        actual: ffmpeg::format::Pixel,
    },
    /// FFmpeg failed to transfer CUDA pixels into a CPU frame.

    #[error("CUDA to CPU transfer failed (code {0})")]
    Transfer(i32),
}

/// Downloads GPU-resident `Pixel::CUDA` `Video` frames to CPU-resident ones
/// in the matching layout — the mirror of [`crate::elements::CudaUpload`]
/// and the CUDA sibling of `D3d11Download`.
///
/// This is what makes a CUDA frame reach anything other than
/// [`crate::elements::CudaEncoder`] or [`crate::elements::CudaRenderer`]:
/// [`crate::elements::SwScaler`], [`crate::elements::SwEncoder`],
/// `OrtDetector`, [`crate::elements::SwVideoCompositor`],
/// and [`crate::elements::AppSink`] all read pixel bytes, which a CUDA frame
/// does not expose. So NVDEC decode plus CPU-side work — inference on a
/// hardware-decoded stream, for instance — goes
/// `CudaDecoder -> CudaDownload -> SwScaler -> ...`.
///
/// A `Filter`: receives via `Sink`, pushes the downloaded frame into its own
/// single src pad. PTS, duration, and color metadata are carried across with
/// `av_frame_copy_props`, so this creates no new timeline.
///
/// # NV12 only
///
/// Everything on this crate's CUDA path speaks NV12 — see
/// [`crate::elements::CudaUpload`]'s own note — so a surface in any other
/// layout is refused here with a typed error rather than handed to
/// `av_hwframe_transfer_data` to fail as an opaque code. Chain a
/// [`crate::elements::SwScaler`] after this to convert to whatever pixel
/// format the downstream stage actually needs.
///
/// # Cost
///
/// Every frame is a device-to-host copy over PCIe. Downloading a stream only
/// to re-upload it is strictly worse than staying on the GPU; put this where
/// the pipeline genuinely has to leave CUDA.
pub struct CudaDownload {
    pp_log: PpLog,
    name: Arc<str>,
    /// This element's own reference to the shared context, released in
    /// `Drop`. Held for `device_ctx`'s sake: the pointer below stays valid
    /// only as long as this reference does.
    _hw_device_ctx: Arc<AvBufferRef>,
    /// Captured at construction so an incoming frame can be checked against
    /// this element's own CUDA context. Only ever compared, same as
    /// [`crate::elements::CudaEncoder`]'s.
    device_ctx: *const ffi::AVHWDeviceContext,
    format: CudaFrameFormat,
    width: u32,
    height: u32,
    pad: SrcPad,
    /// CPU NV12 frames the transfer writes into. Unlike `CudaUpload`'s pool
    /// this recycles the pixel allocation itself, since the destination of a
    /// download is ordinary host memory rather than a surface from a frames
    /// context.
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
}

// SAFETY: `hw_device_ctx` is a heap-allocated FFmpeg buffer with no thread
// affinity of its own, `device_ctx` is only ever compared, and `&mut self`
// on every method that touches them rules out concurrent access. Same
// reasoning as `CudaUpload`.
unsafe impl Send for CudaDownload {}

impl CudaDownload {
    /// `device` must be the same [`CudaDevice`] the upstream CUDA elements
    /// were built from — a frame allocated against another context is
    /// rejected rather than read from meaningless pointers. This element
    /// takes its own FFmpeg reference, so `device` itself need not outlive
    /// the call.
    ///
    /// `width`/`height` are fixed for this element's lifetime; every frame
    /// `consume` receives must match exactly, same convention as
    /// `CudaUpload`/`D3d11Download`.
    pub fn new(
        name: impl Into<String>,
        device: &CudaDevice,
        format: CudaFrameFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::CudaDownload, &name, None);

        let hw_device_ctx = device.retain();
        // SAFETY: `hw_device_ctx` owns a live `AVBufferRef` for a CUDA device
        // context, whose `data` is that `AVHWDeviceContext` by FFmpeg's own
        // definition. Only the pointer's identity is kept, to compare against an
        // incoming frame's; the reference held alongside it is what keeps that
        // identity from being reused by a different context.
        let device_ctx = unsafe { (*hw_device_ctx.as_ptr()).data as *const ffi::AVHWDeviceContext };

        let pad = SrcPad::with_contract(
            format!("{name}_src"),
            OutputContract::Fixed(
                PortContract::of(MediaKind::Video).in_memory(MemoryDomain::System),
            ),
        );
        let pixel = format.pixel();
        let pool = UnboundObjectPool::new(
            0,
            move || ffmpeg::frame::Video::new(pixel, width, height),
            |_| {},
        );
        pp_info!(pp_log: &pp_log, "opened: {width}x{height} CUDA -> {pixel:?}");
        Self {
            name,
            pp_log,
            _hw_device_ctx: hw_device_ctx,
            device_ctx,
            format,
            width,
            height,
            pad,
            pool,
        }
    }

    fn download(&mut self, source: &ffmpeg::frame::Video) -> Result<()> {
        if source.format() != ffmpeg::format::Pixel::CUDA {
            pp_error!(self, "unsupported pixel format: {:?}", source.format());
            return Err(CudaDownloadError::UnsupportedFormat(source.format()).into());
        }
        if source.width() != self.width || source.height() != self.height {
            let error = CudaDownloadError::DimensionMismatch {
                actual_width: source.width(),
                actual_height: source.height(),
                expected_width: self.width,
                expected_height: self.height,
            };
            pp_error!(self, "{error}");
            return Err(error.into());
        }
        // SAFETY: `frame` is a live `frame::Video` already confirmed to be
        // `Pixel::CUDA`, so `as_ptr` yields an initialized `AVFrame` and a hardware
        // frame's `hw_frames_ctx` is either null — rejected here — or an
        // `AVBufferRef` whose `data` is an `AVHWFramesContext`. Only pointer
        // identity is compared, never dereferenced past that.
        unsafe {
            let frames_ref = (*source.as_ptr()).hw_frames_ctx;
            if frames_ref.is_null() {
                pp_error!(self, "frame has no hardware frames context");
                return Err(CudaDownloadError::MissingFramesContext.into());
            }
            let frames_ctx = (*frames_ref).data as *const ffi::AVHWFramesContext;
            if !std::ptr::eq((*frames_ctx).device_ctx, self.device_ctx) {
                pp_error!(self, "frame belongs to a different CUDA context");
                return Err(CudaDownloadError::ForeignContext.into());
            }
            let sw_format = ffmpeg::format::Pixel::from((*frames_ctx).sw_format);
            if sw_format != self.format.pixel() {
                let error = CudaDownloadError::UnsupportedSurfaceFormat {
                    expected: self.format.pixel(),
                    actual: sw_format,
                };
                pp_error!(self, "{error}");
                return Err(error.into());
            }
        }

        let mut destination = self.pool.get();
        // SAFETY: `dst` is the pooled frame's own `AVFrame`, allocated for this
        // element's format and size and kept across calls — see the comment beside
        // it — and `source` is the caller's live CUDA frame, validated just above.
        unsafe {
            let dst = destination.as_mut_ptr();
            // The pooled frame keeps its own NV12 allocation across calls, so
            // this transfers into existing planes rather than letting
            // `av_hwframe_transfer_data` allocate a fresh one per frame.
            let code = ffi::av_hwframe_transfer_data(dst, source.as_ptr(), 0);
            if code < 0 {
                pp_error!(self, "av_hwframe_transfer_data failed: {code}");
                return Err(CudaDownloadError::Transfer(code).into());
            }
            // `av_hwframe_transfer_data` moves pixels only — PTS, duration,
            // and color metadata are part of the buffer contract and would
            // otherwise be dropped here.
            ffi::av_frame_copy_props(dst, source.as_ptr());
        }

        self.pad.push(MediaBuffer::Video(Arc::new(destination)))
    }
}

impl Element for CudaDownload {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::CudaDownload
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for CudaDownload {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for CudaDownload {
    /// The mirror of CudaUpload: only device memory has anything to bring back.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::of(MediaKind::Video).in_memory(MemoryDomain::Cuda))
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Video(frame) => self.download(&frame),
            MediaBuffer::Eos => self.pad.push(MediaBuffer::Eos),
            other => Err(CudaDownloadError::UnsupportedBuffer(other.kind()).into()),
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        // Nothing local to react to — a pure per-frame GPU->CPU transfer,
        // same reasoning as `CudaUpload::control`.
        self.pad.control(msg)
    }
}

impl Drop for CudaDownload {
    fn drop(&mut self) {
        pp_info!(self, "dropped: releasing hw_device_ctx");
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::{
        elements::{CudaDecoder, CudaUpload},
        test_support::{try_cuda_device, try_test_video},
    };

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

    fn capture(element: &mut dyn Source) -> Arc<Mutex<Vec<MediaBuffer>>> {
        let received = Arc::new(Mutex::new(Vec::new()));
        element.src_pads()[0].link(Box::new(CapturingSink {
            received: received.clone(),
            pp_log: element_pp_log(ElementType::Other, "capture", None),
        }));
        received
    }

    fn pooled(frame: ffmpeg::frame::Video) -> MediaBuffer {
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut slot = pool.get();
        *slot = frame;
        MediaBuffer::Video(Arc::new(slot))
    }

    /// A recognizable NV12 frame: a horizontal luma ramp with a fixed chroma
    /// plane, so a round trip that silently transferred the wrong plane or
    /// the wrong rows shows up as a mismatch rather than as plausible noise.
    fn nv12_pattern(width: u32, height: u32, pts: i64) -> ffmpeg::frame::Video {
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, width, height);
        frame.set_pts(Some(pts));
        let y_stride = frame.stride(0);
        let y = frame.data_mut(0);
        for row in 0..height as usize {
            for column in 0..width as usize {
                y[row * y_stride + column] = (row * 16 + column) as u8;
            }
        }
        let uv_stride = frame.stride(1);
        let uv = frame.data_mut(1);
        for row in 0..(height as usize / 2) {
            for column in 0..width as usize {
                uv[row * uv_stride + column] = (128 + column) as u8;
            }
        }
        frame
    }

    /// The contract: pixels that went up come back down unchanged, on the CPU,
    /// with their timestamp intact.
    #[test]
    fn a_cuda_frame_round_trips_back_to_cpu_nv12_pixels() {
        let Some((device, _cuda_lock)) = try_cuda_device() else {
            return;
        };
        let (width, height) = (64u32, 64u32);
        let Ok(mut upload) =
            CudaUpload::new("upload", &device, CudaFrameFormat::Nv12, width, height)
        else {
            eprintln!("skipping: this machine has no usable CUDA frames context");
            return;
        };
        let mut download =
            CudaDownload::new("download", &device, CudaFrameFormat::Nv12, width, height);
        let received = capture(&mut download);
        upload.src_pads()[0].link(Box::new(download));

        // Deterministic, so the same call reproduces exactly what was sent.
        let source = nv12_pattern(width, height, 4321);
        upload
            .consume(pooled(nv12_pattern(width, height, 4321)))
            .expect("upload then download");
        upload.consume(MediaBuffer::Eos).expect("eos");

        let received = received.lock().unwrap();
        let MediaBuffer::Video(frame) = &received[0] else {
            panic!("expected a Video buffer, got {}", received[0].kind());
        };
        assert_eq!(frame.format(), ffmpeg::format::Pixel::NV12);
        assert_eq!(frame.width(), width);
        assert_eq!(frame.height(), height);
        assert_eq!(frame.pts(), Some(4321), "download dropped the pts");
        for plane in 0..2 {
            let rows = if plane == 0 { height } else { height / 2 };
            for row in 0..rows as usize {
                let expected_stride = source.stride(plane);
                let actual_stride = frame.stride(plane);
                let expected = &source.data(plane)
                    [row * expected_stride..row * expected_stride + width as usize];
                let actual =
                    &frame.data(plane)[row * actual_stride..row * actual_stride + width as usize];
                assert_eq!(actual, expected, "plane {plane} row {row} mismatch");
            }
        }
        assert!(
            received.last().is_some_and(MediaBuffer::is_eos),
            "Eos was not forwarded"
        );
    }

    /// The same round trip in BGRA, the layout a screen capture produces.
    /// Nothing on the CUDA path converts between the two, so each format has
    /// to survive on its own.
    #[test]
    fn a_bgra_frame_round_trips_back_to_cpu_bgra_pixels() {
        let Some((device, _cuda_lock)) = try_cuda_device() else {
            return;
        };
        let (width, height) = (32u32, 32u32);
        let Ok(mut upload) =
            CudaUpload::new("upload", &device, CudaFrameFormat::Bgra, width, height)
        else {
            eprintln!("skipping: this machine has no usable CUDA frames context");
            return;
        };
        let mut download =
            CudaDownload::new("download", &device, CudaFrameFormat::Bgra, width, height);
        let received = capture(&mut download);
        upload.src_pads()[0].link(Box::new(download));

        let mut source = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, width, height);
        source.set_pts(Some(11));
        let stride = source.stride(0);
        let pixels = source.data_mut(0);
        for row in 0..height as usize {
            for column in 0..width as usize {
                let at = row * stride + column * 4;
                pixels[at] = column as u8;
                pixels[at + 1] = row as u8;
                pixels[at + 2] = (column + row) as u8;
                pixels[at + 3] = 255;
            }
        }
        let expected: Vec<u8> = (0..height as usize)
            .flat_map(|row| {
                source.data(0)[row * stride..row * stride + width as usize * 4].to_vec()
            })
            .collect();
        upload
            .consume(pooled(source))
            .expect("upload then download");

        let received = received.lock().unwrap();
        let MediaBuffer::Video(frame) = &received[0] else {
            panic!("expected a Video buffer, got {}", received[0].kind());
        };
        assert_eq!(frame.format(), ffmpeg::format::Pixel::BGRA);
        assert_eq!(frame.pts(), Some(11), "download dropped the pts");
        let out_stride = frame.stride(0);
        for row in 0..height as usize {
            let actual = &frame.data(0)[row * out_stride..row * out_stride + width as usize * 4];
            let expected = &expected[row * width as usize * 4..(row + 1) * width as usize * 4];
            assert_eq!(actual, expected, "row {row} mismatch");
        }
    }

    /// A surface holding something other than what this element was built
    /// for must be refused, not transferred into a differently-shaped CPU
    /// frame.
    #[test]
    fn a_surface_in_another_layout_is_a_typed_error() {
        let Some((device, _cuda_lock)) = try_cuda_device() else {
            return;
        };
        let Ok(mut upload) = CudaUpload::new("upload", &device, CudaFrameFormat::Bgra, 32, 32)
        else {
            eprintln!("skipping: this machine has no usable CUDA frames context");
            return;
        };
        let uploaded = capture(&mut upload);
        let bgra = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, 32, 32);
        upload.consume(pooled(bgra)).expect("upload");
        let frame = uploaded.lock().unwrap().remove(0);

        let mut download = CudaDownload::new("download", &device, CudaFrameFormat::Nv12, 32, 32);
        let _received = capture(&mut download);
        let error = download
            .consume(frame)
            .expect_err("a BGRA surface must not download as NV12");
        assert!(
            error.to_string().contains("built for NV12"),
            "expected UnsupportedSurfaceFormat, got {error}"
        );
    }

    /// The point of the element: real NVDEC output becomes readable CPU
    /// pixels, which is what lets a hardware-decoded stream reach a `SwScaler`,
    /// `SwEncoder`, or `OrtDetector` at all.
    #[test]
    fn decoded_nvdec_frames_become_readable_cpu_frames() {
        let Some((device, _cuda_lock)) = try_cuda_device() else {
            return;
        };
        let Some(path) = try_test_video() else {
            return;
        };
        let frames = decode_some_frames(&device, &path);
        let Some(first) = frames.first() else {
            eprintln!("skipping: NVDEC produced no frames for the configured fixture");
            return;
        };
        let MediaBuffer::Video(first) = first else {
            panic!("the decoder emitted a {}", first.kind());
        };
        let (width, height) = (first.width(), first.height());
        let expected_pts: Vec<_> = frames
            .iter()
            .map(|buf| match buf {
                MediaBuffer::Video(frame) => frame.pts(),
                other => panic!("the decoder emitted a {}", other.kind()),
            })
            .collect();

        let mut download =
            CudaDownload::new("download", &device, CudaFrameFormat::Nv12, width, height);
        let received = capture(&mut download);
        for frame in frames {
            download.consume(frame).expect("download failed");
        }

        let received = received.lock().unwrap();
        assert_eq!(received.len(), expected_pts.len());
        for (buf, pts) in received.iter().zip(expected_pts) {
            let MediaBuffer::Video(frame) = buf else {
                panic!("expected a Video buffer, got {}", buf.kind());
            };
            assert_eq!(frame.format(), ffmpeg::format::Pixel::NV12);
            assert_eq!(frame.width(), width);
            assert_eq!(frame.height(), height);
            assert_eq!(frame.pts(), pts, "download dropped the pts");
            assert!(
                frame.stride(0) >= width as usize,
                "luma stride {} is narrower than width {width}",
                frame.stride(0)
            );
            assert!(
                frame.data(0).iter().any(|&byte| byte != 0),
                "the downloaded luma plane is entirely zero"
            );
        }
    }

    /// A CPU frame carries no device pointers, and one from another context
    /// carries pointers that mean nothing here — both must be refused rather
    /// than read.
    #[test]
    fn a_cpu_frame_and_a_foreign_context_frame_are_typed_errors() {
        let Some((device, _cuda_lock)) = try_cuda_device() else {
            return;
        };
        let mut download = CudaDownload::new("download", &device, CudaFrameFormat::Nv12, 64, 64);
        let _received = capture(&mut download);

        let error = download
            .consume(pooled(nv12_pattern(64, 64, 0)))
            .expect_err("a CPU frame must not be downloaded");
        assert!(
            error.to_string().contains("only downloads CUDA frames"),
            "expected UnsupportedFormat, got {error}"
        );

        // Directly, not `try_cuda_device` again: the lock it returns is
        // already held for this test and does not nest.
        let other_device = CudaDevice::new().expect("a second CUDA device");
        let Ok(mut upload) =
            CudaUpload::new("upload", &other_device, CudaFrameFormat::Nv12, 64, 64)
        else {
            eprintln!("skipping: this machine has no usable CUDA frames context");
            return;
        };
        let uploaded = capture(&mut upload);
        upload
            .consume(pooled(nv12_pattern(64, 64, 0)))
            .expect("upload");
        let foreign = uploaded.lock().unwrap().remove(0);
        let error = download
            .consume(foreign)
            .expect_err("a frame from a foreign CUDA context must not be downloaded");
        assert!(
            error.to_string().contains("different CUDA context"),
            "expected ForeignContext, got {error}"
        );
    }

    /// A mismatched frame must be refused rather than transferred into a
    /// differently-sized CPU frame.
    #[test]
    fn a_wrong_size_frame_is_a_typed_error() {
        let Some((device, _cuda_lock)) = try_cuda_device() else {
            return;
        };
        let Ok(mut upload) = CudaUpload::new("upload", &device, CudaFrameFormat::Nv12, 32, 32)
        else {
            eprintln!("skipping: this machine has no usable CUDA frames context");
            return;
        };
        let uploaded = capture(&mut upload);
        upload
            .consume(pooled(nv12_pattern(32, 32, 0)))
            .expect("upload");
        let small = uploaded.lock().unwrap().remove(0);

        let mut download = CudaDownload::new("download", &device, CudaFrameFormat::Nv12, 64, 64);
        let _received = capture(&mut download);
        let error = download
            .consume(small)
            .expect_err("a differently-sized frame must not download");
        assert!(
            error.to_string().contains("32x32"),
            "expected DimensionMismatch, got {error}"
        );
    }

    /// Decodes a handful of real frames, so the test above asserts against
    /// genuine NVDEC output rather than a hand-built `AVFrame` — same helper
    /// shape as `cuda_renderer`'s own tests.
    fn decode_some_frames(device: &CudaDevice, path: &str) -> Vec<MediaBuffer> {
        let mut input = ffmpeg::format::input(path).expect("failed to open the test video");
        let stream = input
            .streams()
            .best(ffmpeg::media::Type::Video)
            .expect("the test video has no video stream");
        let stream_index = stream.index();
        let params = stream.parameters();

        let mut decoder =
            CudaDecoder::new("cuda-decoder", params, device, 4).expect("failed to open NVDEC");
        let received = capture(&mut decoder);

        for (stream, packet) in input.packets() {
            if stream.index() != stream_index {
                continue;
            }
            decoder
                .consume(MediaBuffer::Packet(Arc::new(packet)))
                .expect("decode failed");
            if received.lock().unwrap().len() >= 3 {
                break;
            }
        }
        std::mem::take(&mut *received.lock().unwrap())
    }
}
