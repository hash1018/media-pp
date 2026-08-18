use std::sync::Arc;

use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_error, pp_info};

use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    elements::filter::is_codec_drain_boundary,
    pad::SrcPad,
    platform::cuda::CudaDevice,
    pool::UnboundObjectPool,
};

/// Errors specific to `CudaDecoder`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum CudaDecoderError {
    #[error("unsupported media type: {0:?} (NVDEC decode is video-only)")]
    UnsupportedMediaType(ffmpeg::media::Type),

    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),

    #[error("failed to reference the CUDA device context")]
    HwDeviceRef,

    #[error(
        "decoder did not select the CUDA pixel format — hardware decode \
         unavailable for this stream/GPU/driver"
    )]
    HwAccelUnavailable,
}

/// Decodes one video stream's `Packet`s into GPU-resident `Video` frames on
/// NVDEC, tagged [`ffmpeg::format::Pixel::CUDA`]. A `Filter`, same shape as
/// [`crate::elements::SwDecoder`]/[`crate::elements::D3d11Decoder`].
///
/// Frames this produces are still plain `MediaBuffer::Video` — `Pacer`,
/// `Tee`, `Queue`, and `FrameCounter` only touch `.pts()` or match the enum
/// variant, so they work unmodified. What *cannot* read them is anything
/// that reaches for pixel bytes: `Scaler` and `SwEncoder` see no CPU planes
/// on a CUDA frame. [`crate::elements::CudaRenderer`] is the terminal built
/// for them.
///
/// Named for the frame type it produces, not for NVDEC, matching
/// `D3d11Decoder`'s own naming: CUDA frames are NVIDIA-only by
/// construction, so there is no sibling decoder a vendor name would
/// disambiguate it from.
pub struct CudaDecoder {
    pp_log: PpLog,
    name: Arc<str>,
    decoder: ffmpeg::decoder::Video,
    hw_device_ctx: *mut ffi::AVBufferRef,
    pad: SrcPad,
    /// Reused across every decoded frame — the GPU surface itself is already
    /// pooled by FFmpeg's own hw frames context, so this only recycles the
    /// small CPU-side `AVFrame` wrapper. Same reasoning as `D3d11Decoder`.
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
}

// SAFETY: `hw_device_ctx` is a heap-allocated FFmpeg buffer with no thread
// affinity of its own — FFmpeg's CUDA hwcontext pushes and pops the CUDA
// context around every call that needs it. `decoder`'s own `Send` covers
// the rest, and `&mut self` on every method that touches either rules out
// concurrent access. Same reasoning as `D3d11Decoder`.
unsafe impl Send for CudaDecoder {}

impl CudaDecoder {
    /// `device` must be the same [`CudaDevice`] every other CUDA element in
    /// this pipeline was built from — a frame allocated against one CUDA
    /// context is not readable from another. This decoder takes its own
    /// FFmpeg reference, so `device` itself need not outlive the call.
    ///
    /// `extra_hw_frames` sets `AVCodecContext.extra_hw_frames`: how many
    /// decode surfaces to allocate beyond what the codec's reference-frame
    /// count requires. NVDEC's surface pool is fixed-size like D3D11VA's
    /// (unlike the growable pools used elsewhere in this crate), so every
    /// decoded frame still alive downstream — sitting in a
    /// [`crate::queue::Queue`], held by a slow renderer — occupies a slot,
    /// and running out fails decode rather than blocking it. Pass at least
    /// the depth of the deepest downstream buffering.
    ///
    /// Unlike D3D11VA, though, NVDEC also imposes a hard **upper** bound: the
    /// pool may hold at most 32 surfaces in total, counting the codec's own
    /// reference frames. Asking for more does not degrade — `cuvidCreateDecoder`
    /// fails outright and the stream does not decode at all, reported as
    /// "Using more than 32 (N) decode surfaces might cause nvdec to fail".
    /// So a downstream queue cannot simply be made as deep as convenient
    /// here; it has to fit the budget alongside the reference frames.
    pub fn new(
        name: impl Into<String>,
        params: ffmpeg::codec::Parameters,
        device: &CudaDevice,
        extra_hw_frames: i32,
    ) -> Result<Self, CudaDecoderError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::CudaDecoder, &name, None);

        let mut context = ffmpeg::codec::context::Context::from_parameters(params)?;
        if context.medium() != ffmpeg::media::Type::Video {
            return Err(CudaDecoderError::UnsupportedMediaType(context.medium()));
        }

        // Taken before anything can fail below, and released by `Drop` once
        // the decoder exists. Nothing between here and the `Ok` needs an
        // explicit unref: `hw_device_ctx` below is a second, independent
        // reference owned by the codec context.
        let hw_device_ctx = unsafe { ffi::av_buffer_ref(device.as_ptr()) };
        if hw_device_ctx.is_null() {
            return Err(CudaDecoderError::HwDeviceRef);
        }
        unsafe {
            let ctx_ptr = context.as_mut_ptr();
            (*ctx_ptr).hw_device_ctx = ffi::av_buffer_ref(hw_device_ctx);
            (*ctx_ptr).get_format = Some(get_format);
            (*ctx_ptr).extra_hw_frames = extra_hw_frames;
        }

        let decoder = match context.decoder().video() {
            Ok(decoder) => decoder,
            Err(error) => {
                unsafe { free_buffer(hw_device_ctx) };
                return Err(error.into());
            }
        };

        let pad = SrcPad::new(format!("{name}_src"));
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        pp_info!(pp_log: &pp_log, "opened: codec={:?}", decoder.id());
        Ok(Self {
            name,
            pp_log,
            decoder,
            hw_device_ctx,
            pad,
            pool,
        })
    }

    fn drain(&mut self) -> crate::error::Result<()> {
        let mut frame = self.pool.get();
        loop {
            match self.decoder.receive_frame(&mut frame) {
                Ok(()) => {
                    if frame.format() != ffmpeg::format::Pixel::CUDA {
                        pp_error!(self, "decoder did not select the CUDA pixel format");
                        return Err(CudaDecoderError::HwAccelUnavailable.into());
                    }
                    self.pad.push(MediaBuffer::Video(Arc::new(frame)))?;
                    frame = self.pool.get();
                }
                Err(error) if is_codec_drain_boundary(&error) => break,
                Err(error) => return Err(CudaDecoderError::from(error).into()),
            }
        }
        Ok(())
    }
}

impl Element for CudaDecoder {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::CudaDecoder
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for CudaDecoder {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for CudaDecoder {
    fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
        match buf {
            MediaBuffer::Packet(packet) => {
                self.decoder
                    .send_packet(&*packet)
                    .inspect_err(|error| pp_error!(self, "send_packet failed: {error}"))
                    .map_err(CudaDecoderError::from)?;
                self.drain()
            }
            MediaBuffer::Eos => {
                self.decoder
                    .send_eof()
                    .inspect_err(|error| pp_error!(self, "send_eof failed: {error}"))
                    .map_err(CudaDecoderError::from)?;
                self.drain()?;
                self.pad.push(MediaBuffer::Eos)
            }
            other => {
                let _ = other;
                Ok(())
            }
        }
    }

    fn control(&mut self, msg: ControlMsg) -> crate::error::Result<()> {
        // Same reasoning as `D3d11Decoder::control`: nothing to do on `Stop`
        // (the hw device reference is released in `Drop`), flush
        // reference-frame state on `Seek`.
        if let ControlMsg::Seek(_) = msg {
            self.decoder.flush();
        }
        self.pad.control(msg)
    }
}

impl Drop for CudaDecoder {
    fn drop(&mut self) {
        pp_info!(self, "dropped: releasing hw_device_ctx");
        unsafe { free_buffer(self.hw_device_ctx) };
    }
}

unsafe fn free_buffer(mut buf: *mut ffi::AVBufferRef) {
    unsafe { ffi::av_buffer_unref(&mut buf) };
}

/// Picks `AV_PIX_FMT_CUDA` out of whatever libavcodec offers. Unlike the
/// D3D11VA sibling there is nothing to configure on the frames context —
/// this crate never customizes NVDEC's surface allocation, so libavcodec
/// sets the whole thing up itself.
unsafe extern "C" fn get_format(
    _ctx: *mut ffi::AVCodecContext,
    mut fmt: *const ffi::AVPixelFormat,
) -> ffi::AVPixelFormat {
    unsafe {
        while *fmt != ffi::AVPixelFormat::AV_PIX_FMT_NONE {
            if *fmt == ffi::AVPixelFormat::AV_PIX_FMT_CUDA {
                return ffi::AVPixelFormat::AV_PIX_FMT_CUDA;
            }
            fmt = fmt.add(1);
        }
        ffi::AVPixelFormat::AV_PIX_FMT_NONE
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::{error::Result, test_support::try_test_video};

    /// Skips with a reason when this machine has no usable CUDA device,
    /// the same way a D3D11 test skips without a device — a build with the
    /// `cuda` feature still has to run on CI boxes with no NVIDIA GPU.
    fn try_cuda_device() -> Option<CudaDevice> {
        match CudaDevice::new() {
            Ok(device) => Some(device),
            Err(error) => {
                eprintln!("skipping: no usable CUDA device on this machine ({error})");
                None
            }
        }
    }

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

    /// The whole point of this element: frames come out GPU-resident, and
    /// the timeline survives. Asserted against whatever fixture is
    /// configured — nothing here depends on its codec, size, or duration.
    #[test]
    fn decodes_into_cuda_frames_and_forwards_eos() {
        let Some(device) = try_cuda_device() else {
            return;
        };
        let Some(path) = try_test_video() else {
            return;
        };

        let mut input = ffmpeg::format::input(&path).expect("failed to open the test video");
        let stream = input
            .streams()
            .best(ffmpeg::media::Type::Video)
            .expect("the test video has no video stream");
        let stream_index = stream.index();
        let params = stream.parameters();

        let mut decoder = CudaDecoder::new("cuda-decoder", params, &device, 4)
            .expect("failed to open the CUDA decoder");
        let received = Arc::new(Mutex::new(Vec::new()));
        decoder.src_pads()[0].link(Box::new(CapturingSink {
            received: received.clone(),
            pp_log: element_pp_log(ElementType::Other, "capture", None),
        }));

        for (stream, packet) in input.packets() {
            if stream.index() != stream_index {
                continue;
            }
            decoder
                .consume(MediaBuffer::Packet(Arc::new(packet)))
                .expect("decode failed");
            if received.lock().unwrap().len() >= 5 {
                break;
            }
        }
        decoder.consume(MediaBuffer::Eos).expect("eos failed");

        let received = received.lock().unwrap();
        let frames: Vec<_> = received
            .iter()
            .filter_map(|buf| match buf {
                MediaBuffer::Video(frame) => Some(frame),
                _ => None,
            })
            .collect();
        assert!(
            !frames.is_empty(),
            "NVDEC produced no frames for the configured fixture"
        );
        for frame in &frames {
            assert_eq!(
                frame.format(),
                ffmpeg::format::Pixel::CUDA,
                "a decoded frame was not GPU-resident"
            );
            assert!(frame.pts().is_some(), "a decoded frame lost its pts");
        }
        assert!(
            received.last().is_some_and(MediaBuffer::is_eos),
            "Eos was not forwarded after draining"
        );
    }

    /// The decoder must reject a non-video stream with its own typed error
    /// rather than failing somewhere inside libavcodec later.
    #[test]
    fn audio_parameters_are_rejected_as_a_typed_error() {
        let Some(device) = try_cuda_device() else {
            return;
        };
        let mut params = ffmpeg::codec::Parameters::new();
        unsafe {
            (*params.as_mut_ptr()).codec_type = ffmpeg::media::Type::Audio.into();
        }

        let error = CudaDecoder::new("cuda-decoder", params, &device, 0)
            .err()
            .expect("audio parameters must not open a video decoder");
        assert!(
            matches!(
                error,
                CudaDecoderError::UnsupportedMediaType(ffmpeg::media::Type::Audio)
            ),
            "expected UnsupportedMediaType, got {error:?}"
        );
    }
}
