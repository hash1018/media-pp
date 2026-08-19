use std::sync::Arc;

use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_error, pp_info};

use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    elements::filter::is_codec_drain_boundary,
    elements::filter::upload::cuda_upload::{create_hw_frames_ctx, free_buffer},
    error::Result,
    pad::SrcPad,
    platform::cuda::CudaDevice,
};

/// Which NVENC codec [`CudaEncoder`] drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaCodec {
    /// `h264_nvenc` — H.264/AVC.
    H264,
    /// `hevc_nvenc` — H.265/HEVC.
    H265,
}

impl CudaCodec {
    fn encoder_name(self) -> &'static str {
        match self {
            Self::H264 => "h264_nvenc",
            Self::H265 => "hevc_nvenc",
        }
    }
}

/// Construction-time options for [`CudaEncoder`].
///
/// Carries no input-format choice, unlike
/// [`crate::elements::D3d11NvencEncoderOptions`]: every producer of a CUDA
/// frame in this crate ([`crate::elements::CudaDecoder`],
/// [`crate::elements::CudaUpload`]) emits NV12, so a second variant would
/// only be expressible, never reachable.
#[derive(Debug, Clone, Copy)]
pub struct CudaEncoderOptions {
    pub codec: CudaCodec,
    pub width: u32,
    pub height: u32,
    /// Must match the `pts` unit of whatever frames this receives.
    pub time_base: ffmpeg::Rational,
    /// The nominal rate NVENC uses for rate control and writes into the
    /// bitstream — not required to match the real interval between `consume`
    /// calls. See [`crate::elements::SwEncoderOptions::frame_rate`].
    pub frame_rate: ffmpeg::Rational,
    pub bit_rate: usize,
    /// Frames between keyframes (`AVCodecContext.gop_size`). Always set
    /// explicitly, for the join-latency and segmenting reasons
    /// [`crate::elements::SwEncoderOptions::gop_size`] documents.
    pub gop_size: u32,
}

/// Errors specific to `CudaEncoder`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum CudaEncoderError {
    #[error("encoder `{0}` not found in this ffmpeg build")]
    CodecNotFound(&'static str),

    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),

    #[error("CudaEncoder only encodes CUDA frames, got {0:?}")]
    UnsupportedFormat(ffmpeg::format::Pixel),

    #[error("CudaEncoder only accepts Video and Eos buffers, got a {0}")]
    UnsupportedBuffer(&'static str),

    #[error(
        "frame is {actual_width}x{actual_height}, but this CudaEncoder was built for \
         {expected_width}x{expected_height}"
    )]
    DimensionMismatch {
        actual_width: u32,
        actual_height: u32,
        expected_width: u32,
        expected_height: u32,
    },

    /// The frame carries no hardware frames context, so it did not come from
    /// a CUDA producer at all.
    #[error("CUDA frame has no hardware frames context")]
    MissingFramesContext,

    /// The frame was allocated against a different CUDA context than this
    /// encoder's. Its device pointers mean nothing here.
    #[error("CUDA frame belongs to a different CUDA context than this encoder")]
    ForeignContext,

    #[error("failed to build the CUDA frames context: {0}")]
    HwFrames(String),
}

/// Encodes GPU-resident `Pixel::CUDA` `Video` frames into `Packet`s on the
/// GPU's dedicated NVENC block — the CUDA counterpart to
/// [`crate::elements::SwEncoder`] and the sibling of
/// [`crate::elements::D3d11NvencEncoder`]. A `Filter`: receives via `Sink`,
/// pushes what it produces into its own single src pad.
///
/// Fed by [`crate::elements::CudaDecoder`] this is a transcode that never
/// brings a pixel to the CPU; fed by [`crate::elements::CudaUpload`] it is
/// the hardware replacement for a `SwScaler`-plus-`SwEncoder` recording tail.
///
/// Named for the frame type it consumes rather than for NVENC, matching
/// `CudaDecoder`/`CudaRenderer`: a CUDA frame is NVIDIA-only by
/// construction, so there is no sibling encoder a vendor name would
/// disambiguate this from.
///
/// # Packet timing
///
/// One frame can turn into zero or one packets per `send_frame` — NVENC's
/// lookahead and B-frame reordering delay some packets until later frames
/// arrive, or until `Eos` flushes what is left. `consume` drains
/// `receive_packet` in a loop after every `send_frame`/`send_eof`, the same
/// shape as `SwEncoder`'s own drain loop, and stamps each packet's
/// `time_base` and nominal duration since `avcodec_receive_packet` sets
/// neither.
pub struct CudaEncoder {
    pp_log: PpLog,
    name: Arc<str>,
    encoder: ffmpeg::encoder::Video,
    hw_device_ctx: *mut ffi::AVBufferRef,
    hw_frames_ctx: *mut ffi::AVBufferRef,
    /// Captured at construction so an incoming frame can be checked against
    /// this encoder's own CUDA context. Only ever compared.
    device_ctx: *const ffi::AVHWDeviceContext,
    width: u32,
    height: u32,
    /// Nominal frame duration in `time_base` ticks. NVENC leaves
    /// `AVPacket::duration` at zero; muxers such as
    /// [`crate::elements::HlsMuxer`] need it for precise segment durations.
    packet_duration: i64,
    time_base: ffmpeg::Rational,
    pad: SrcPad,
}

// SAFETY: both buffers are heap-allocated FFmpeg buffers with no thread
// affinity, `device_ctx` is only ever compared, and `encoder`'s own `Send`
// covers the codec context. `&mut self` on every method that touches them
// rules out concurrent access — same reasoning as `CudaDecoder`.
unsafe impl Send for CudaEncoder {}

fn nominal_packet_duration(time_base: ffmpeg::Rational, frame_rate: ffmpeg::Rational) -> i64 {
    if frame_rate.numerator() <= 0 || time_base.numerator() <= 0 {
        return 0;
    }
    let ticks = f64::from(time_base.denominator()) * f64::from(frame_rate.denominator())
        / (f64::from(time_base.numerator()) * f64::from(frame_rate.numerator()));
    ticks.round() as i64
}

impl CudaEncoder {
    /// `device` must be the same [`CudaDevice`] the upstream CUDA elements
    /// were built from — a frame allocated against another context is
    /// rejected rather than encoded from meaningless pointers.
    ///
    /// Opens the encoder eagerly, so a missing `h264_nvenc`/`hevc_nvenc`, a
    /// driver too old for the linked ffmpeg's NVENC API version, or a
    /// resolution this GPU's encode block rejects all surface here as a
    /// typed error rather than at the first frame.
    pub fn new(
        name: impl Into<String>,
        device: &CudaDevice,
        options: CudaEncoderOptions,
    ) -> std::result::Result<Self, CudaEncoderError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::CudaEncoder, &name, None);

        let encoder_name = options.codec.encoder_name();
        let codec = ffmpeg::encoder::find_by_name(encoder_name)
            .ok_or(CudaEncoderError::CodecNotFound(encoder_name))?;

        let hw_device_ctx = unsafe { ffi::av_buffer_ref(device.as_ptr()) };
        let hw_frames_ctx =
            match unsafe { create_hw_frames_ctx(hw_device_ctx, options.width, options.height) } {
                Ok(ctx) => ctx,
                Err(error) => {
                    unsafe { free_buffer(hw_device_ctx) };
                    return Err(CudaEncoderError::HwFrames(error.to_string()));
                }
            };
        let device_ctx = unsafe { (*hw_device_ctx).data as *const ffi::AVHWDeviceContext };

        let opened = (|| -> std::result::Result<ffmpeg::encoder::Video, ffmpeg::Error> {
            let context = ffmpeg::codec::context::Context::new_with_codec(codec);
            let mut video = context.encoder().video()?;
            video.set_width(options.width);
            video.set_height(options.height);
            video.set_format(ffmpeg::format::Pixel::CUDA);
            video.set_time_base(options.time_base);
            video.set_frame_rate(Some(options.frame_rate));
            video.set_bit_rate(options.bit_rate);
            video.set_gop(options.gop_size);
            unsafe {
                let ptr = video.as_mut_ptr();
                // NVENC needs both before `avcodec_open2`: the device to
                // reach the encode block at all, and the frames context to
                // learn the surface layout it will be handed.
                (*ptr).hw_device_ctx = ffi::av_buffer_ref(hw_device_ctx);
                (*ptr).hw_frames_ctx = ffi::av_buffer_ref(hw_frames_ctx);
            }
            video.open_as(codec)
        })();

        let encoder = match opened {
            Ok(encoder) => encoder,
            Err(error) => {
                unsafe {
                    free_buffer(hw_frames_ctx);
                    free_buffer(hw_device_ctx);
                }
                return Err(error.into());
            }
        };

        let pad = SrcPad::new(format!("{name}_src"));
        pp_info!(
            pp_log: &pp_log,
            "opened: {} {}x{}, {} bps, gop={}",
            encoder_name,
            options.width,
            options.height,
            options.bit_rate,
            options.gop_size
        );
        Ok(Self {
            name,
            pp_log,
            encoder,
            hw_device_ctx,
            hw_frames_ctx,
            device_ctx,
            width: options.width,
            height: options.height,
            packet_duration: nominal_packet_duration(options.time_base, options.frame_rate),
            time_base: options.time_base,
            pad,
        })
    }

    /// The encoded stream's parameters, for
    /// [`crate::elements::Mp4Muxer::add_stream`] — same accessor
    /// `SwEncoder`/`D3d11NvencEncoder` expose for the same reason.
    pub fn parameters(&self) -> ffmpeg::codec::Parameters {
        ffmpeg::codec::Parameters::from(&self.encoder)
    }

    fn encode(&mut self, frame: &ffmpeg::frame::Video) -> Result<()> {
        if frame.format() != ffmpeg::format::Pixel::CUDA {
            pp_error!(self, "unsupported pixel format: {:?}", frame.format());
            return Err(CudaEncoderError::UnsupportedFormat(frame.format()).into());
        }
        if frame.width() != self.width || frame.height() != self.height {
            let error = CudaEncoderError::DimensionMismatch {
                actual_width: frame.width(),
                actual_height: frame.height(),
                expected_width: self.width,
                expected_height: self.height,
            };
            pp_error!(self, "{error}");
            return Err(error.into());
        }
        unsafe {
            let frames_ref = (*frame.as_ptr()).hw_frames_ctx;
            if frames_ref.is_null() {
                pp_error!(self, "frame has no hardware frames context");
                return Err(CudaEncoderError::MissingFramesContext.into());
            }
            let frames_ctx = (*frames_ref).data as *const ffi::AVHWFramesContext;
            if !std::ptr::eq((*frames_ctx).device_ctx, self.device_ctx) {
                pp_error!(self, "frame belongs to a different CUDA context");
                return Err(CudaEncoderError::ForeignContext.into());
            }
        }

        self.encoder
            .send_frame(frame)
            .inspect_err(|error| pp_error!(self, "send_frame failed: {error}"))
            .map_err(CudaEncoderError::from)?;
        self.drain()
    }

    fn drain(&mut self) -> Result<()> {
        let mut packet = ffmpeg::Packet::empty();
        loop {
            match self.encoder.receive_packet(&mut packet) {
                Ok(()) => {
                    packet.set_time_base(self.time_base);
                    if packet.duration() == 0 && self.packet_duration > 0 {
                        packet.set_duration(self.packet_duration);
                    }
                    self.pad.push(MediaBuffer::Packet(Arc::new(packet)))?;
                    packet = ffmpeg::Packet::empty();
                }
                Err(error) if is_codec_drain_boundary(&error) => break,
                Err(error) => return Err(CudaEncoderError::from(error).into()),
            }
        }
        Ok(())
    }
}

impl Element for CudaEncoder {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::CudaEncoder
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for CudaEncoder {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for CudaEncoder {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Video(frame) => self.encode(&frame),
            MediaBuffer::Eos => {
                self.encoder
                    .send_eof()
                    .inspect_err(|error| pp_error!(self, "send_eof failed: {error}"))
                    .map_err(CudaEncoderError::from)?;
                self.drain()?;
                self.pad.push(MediaBuffer::Eos)
            }
            other => Err(CudaEncoderError::UnsupportedBuffer(other.kind()).into()),
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        // Same deliberate choice as `SwEncoder`/`D3d11NvencEncoder`: Seek is
        // forwarded without flushing, since NVENC can still emit packets
        // originating before the seek from later `send_frame` calls. A caller
        // needing a hard encoded-stream discontinuity rebuilds the encoder.
        self.pad.control(msg)
    }
}

impl Drop for CudaEncoder {
    fn drop(&mut self) {
        pp_info!(self, "dropped: releasing hw contexts");
        unsafe {
            free_buffer(self.hw_frames_ctx);
            free_buffer(self.hw_device_ctx);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::{
        elements::{CudaDecoder, CudaUpload, FileDemuxer, PacketCounter},
        pipeline::Pipeline,
        pool::UnboundObjectPool,
        test_support::try_test_video,
    };

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

    fn options(width: u32, height: u32) -> CudaEncoderOptions {
        CudaEncoderOptions {
            codec: CudaCodec::H264,
            width,
            height,
            time_base: ffmpeg::Rational::new(1, 30),
            frame_rate: ffmpeg::Rational::new(30, 1),
            bit_rate: 2_000_000,
            gop_size: 30,
        }
    }

    /// A moving pattern, so the encoder has real content to compress rather
    /// than a constant field that any broken path would still "encode".
    fn nv12_frame(width: u32, height: u32, index: i64) -> ffmpeg::frame::Video {
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, width, height);
        let stride = frame.stride(0);
        let luma = frame.data_mut(0);
        for y in 0..height as usize {
            for x in 0..width as usize {
                luma[y * stride + x] = ((x as i64 + index * 4) % 256) as u8;
            }
        }
        frame.set_pts(Some(index));
        frame
    }

    /// The contract: CUDA frames in, real H.264 packets out, drained on Eos.
    #[test]
    fn encodes_cuda_frames_into_packets_and_drains_on_eos() {
        let Some(device) = try_cuda_device() else {
            return;
        };
        let (width, height) = (320u32, 240u32);
        let mut upload = match CudaUpload::new("upload", &device, width, height) {
            Ok(upload) => upload,
            Err(error) => {
                eprintln!("skipping: CUDA upload unavailable ({error})");
                return;
            }
        };
        let mut encoder = match CudaEncoder::new("encoder", &device, options(width, height)) {
            Ok(encoder) => encoder,
            Err(error) => {
                eprintln!("skipping: NVENC unavailable on this machine ({error})");
                return;
            }
        };

        let received = Arc::new(Mutex::new(Vec::new()));
        encoder.src_pads()[0].link(Box::new(CapturingSink {
            received: received.clone(),
            pp_log: element_pp_log(ElementType::Other, "capture", None),
        }));

        // Upload straight into the encoder, so what is asserted is the real
        // CUDA path and not a hand-built frame the encoder might reject.
        let uploaded = Arc::new(Mutex::new(Vec::new()));
        upload.src_pads()[0].link(Box::new(CapturingSink {
            received: uploaded.clone(),
            pp_log: element_pp_log(ElementType::Other, "uploaded", None),
        }));

        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        for index in 0..30 {
            let mut pooled = pool.get();
            *pooled = nv12_frame(width, height, index);
            upload
                .consume(MediaBuffer::Video(Arc::new(pooled)))
                .expect("upload failed");
            let frame = uploaded.lock().unwrap().pop().expect("nothing uploaded");
            encoder.consume(frame).expect("encode failed");
        }
        encoder.consume(MediaBuffer::Eos).expect("eos failed");

        let received = received.lock().unwrap();
        let packets: Vec<_> = received
            .iter()
            .filter_map(|buf| match buf {
                MediaBuffer::Packet(packet) => Some(packet),
                _ => None,
            })
            .collect();
        assert!(!packets.is_empty(), "NVENC produced no packets");
        assert!(
            packets.iter().any(|packet| packet.size() > 0),
            "every packet was empty"
        );
        for packet in &packets {
            assert_eq!(
                packet.time_base(),
                ffmpeg::Rational::new(1, 30),
                "packet lost its time base"
            );
            assert!(packet.duration() > 0, "packet has no duration");
        }
        assert!(
            received.last().is_some_and(MediaBuffer::is_eos),
            "Eos was not forwarded after draining"
        );
    }

    /// The headline claim of this element: `CudaDecoder`'s output goes
    /// straight in, with no upload and no download between them, even though
    /// the decoder allocates from its own frames pool. What makes that work
    /// is the shared CUDA *context* — NVENC registers the decoder's surfaces
    /// rather than demanding frames from the encoder's own pool. If that ever
    /// stops holding, this fails with `ForeignContext` or a send error rather
    /// than silently falling back to a copy.
    #[test]
    fn decoded_frames_encode_without_an_upload_step() {
        let Some(device) = try_cuda_device() else {
            return;
        };
        let Some(path) = try_test_video() else {
            return;
        };

        let (source, streams) = FileDemuxer::open("demux", &path).expect("failed to open");
        let video = streams
            .iter()
            .find(|stream| stream.kind == ffmpeg::media::Type::Video)
            .expect("the test video has no video stream");
        let index = video.index;
        let params = source.stream_parameters(index).expect("video stream gone");
        let time_base = source.stream_time_base(index).expect("video stream gone");
        let (width, height) = unsafe {
            let ptr = params.as_ptr();
            ((*ptr).width as u32, (*ptr).height as u32)
        };

        let encoder = match CudaEncoder::new(
            "encoder",
            &device,
            CudaEncoderOptions {
                width,
                height,
                time_base,
                ..options(width, height)
            },
        ) {
            Ok(encoder) => encoder,
            Err(error) => {
                eprintln!("skipping: NVENC unavailable on this machine ({error})");
                return;
            }
        };
        let decoder = match CudaDecoder::new("decoder", params, &device, 8) {
            Ok(decoder) => decoder,
            Err(error) => {
                eprintln!("skipping: NVDEC unavailable on this machine ({error})");
                return;
            }
        };

        let (counter, encoded) = PacketCounter::new("count");
        let pipeline = Pipeline::new("cuda-transcode", source, |source, ctx| {
            let branch = ctx
                .branch()
                .pipe(decoder)
                .queue("frames", 8)
                .pipe(encoder)
                .to(Box::new(counter))?;
            ctx.attach(source, index, branch)?;
            Ok(())
        })
        .expect("failed to build the pipeline");

        pipeline.run();
        std::thread::sleep(std::time::Duration::from_secs(2));
        pipeline.stop();

        let errors: Vec<_> = pipeline
            .bus()
            .iter()
            .filter_map(|event| match event {
                crate::bus::BusEvent::Error { name, error, .. } => {
                    Some(format!("[{name}] {error}"))
                }
                _ => None,
            })
            .collect();
        assert!(errors.is_empty(), "transcode reported errors: {errors:?}");
        assert!(
            encoded.load(std::sync::atomic::Ordering::Relaxed) > 0,
            "no packets came out of the decode-to-encode chain"
        );
    }

    /// A CPU frame carries no device pointer NVENC could read; it must be
    /// refused with this element's own error.
    #[test]
    fn a_cpu_frame_is_rejected_as_a_typed_error() {
        let Some(device) = try_cuda_device() else {
            return;
        };
        let mut encoder = match CudaEncoder::new("encoder", &device, options(320, 240)) {
            Ok(encoder) => encoder,
            Err(error) => {
                eprintln!("skipping: NVENC unavailable on this machine ({error})");
                return;
            }
        };
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut pooled = pool.get();
        *pooled = nv12_frame(320, 240, 0);
        let error = encoder
            .consume(MediaBuffer::Video(Arc::new(pooled)))
            .expect_err("a CPU frame must not be encoded");
        assert!(
            error.to_string().contains("only encodes CUDA frames"),
            "expected UnsupportedFormat, got {error}"
        );
    }
}
