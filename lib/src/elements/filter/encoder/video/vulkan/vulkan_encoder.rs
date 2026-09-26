use std::sync::Arc;

use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_error, pp_info};

use crate::color::ColorDescription;
use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    elements::{VulkanDevice, filter::is_codec_drain_boundary},
    error::Result,
    pad::SrcPad,
    platform::{
        ffmpeg::AvBufferRef,
        vulkan::frames::{NotOurs, create_frames_ctx, sw_format_of},
    },
};

// The video-encoder helpers every backend shares, a module up from this one.
use super::super as shared;

/// Which of FFmpeg's Vulkan Video encoders [`VulkanEncoder`] drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VulkanCodec {
    /// `h264_vulkan` — H.264/AVC.
    H264,
    /// `hevc_vulkan` — H.265/HEVC.
    H265,
    /// `av1_vulkan` — AV1, on a GPU that encodes it.
    Av1,
}

impl VulkanCodec {
    fn encoder_name(self) -> &'static str {
        match self {
            Self::H264 => "h264_vulkan",
            Self::H265 => "hevc_vulkan",
            Self::Av1 => "av1_vulkan",
        }
    }
}

/// Construction-time options for [`VulkanEncoder`] — the same knobs, with
/// the same meanings, as `CudaEncoderOptions`.
#[derive(Debug, Clone, Copy)]
pub struct VulkanEncoderOptions {
    /// The bitstream to encode.
    pub codec: VulkanCodec,
    /// Encoded frame width in pixels.
    pub width: u32,
    /// Encoded frame height in pixels.
    pub height: u32,
    /// The nominal rate the encoder uses for rate control and writes into
    /// the bitstream — see [`crate::elements::SwEncoderOptions::frame_rate`].
    pub frame_rate: ffmpeg::Rational,
    /// Target encoded bit rate, in bits per second.
    pub bit_rate: usize,
    /// Frames between keyframes (`AVCodecContext.gop_size`), always set —
    /// see [`crate::elements::SwEncoderOptions::gop_size`].
    pub gop_size: u32,
    /// How many consecutive B-frames the encoder may insert, or `None` to
    /// leave its own default — see
    /// [`crate::elements::SwEncoderOptions::max_b_frames`].
    pub max_b_frames: Option<u32>,
}

/// Errors specific to [`VulkanEncoder`]. Converts into the crate-wide
/// `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum VulkanEncoderError {
    /// The encoder is not in this FFmpeg build.
    #[error("encoder `{0}` not found in this ffmpeg build")]
    CodecNotFound(&'static str),

    /// FFmpeg rejected the encoder or a frame — among other reasons, a GPU
    /// without Vulkan Video encoding, or without this codec, fails to open.
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),

    /// The sink received a buffer other than decoded video or end-of-stream.
    #[error("VulkanEncoder only accepts Video and Eos buffers, got a {0}")]
    UnsupportedBuffer(&'static str),

    /// The frame is not a Vulkan frame.
    #[error("VulkanEncoder encodes Vulkan frames, got a {0:?} frame; upload it first")]
    NotVulkan(ffmpeg::format::Pixel),

    /// The frame was made on another `VkDevice` than the encoder's.
    #[error("VulkanEncoder was handed a frame from another Vulkan device")]
    ForeignDevice,

    /// The frame holds a layout other than NV12.
    #[error("VulkanEncoder encodes NV12 frames, got {0:?}")]
    UnsupportedLayout(ffmpeg::format::Pixel),

    /// A frame arrived with a `pts` but no unit to read it in — see
    /// [`crate::elements::SwEncoderError::NoTimeBase`].
    #[error(
        "a Video frame arrived with a pts but no time base to read it in; the element \
         that made it has to set one (media_pp::buffer::set_time_base)"
    )]
    NoTimeBase,

    /// Input dimensions differ from the fixed encoder dimensions.
    #[error(
        "frame is {actual_width}x{actual_height}, but this VulkanEncoder was built for \
         {expected_width}x{expected_height}"
    )]
    DimensionMismatch {
        /// Input frame width in pixels.
        actual_width: u32,
        /// Input frame height in pixels.
        actual_height: u32,
        /// Width configured for the encoder.
        expected_width: u32,
        /// Height configured for the encoder.
        expected_height: u32,
    },

    /// FFmpeg could not make the frames context the encoder is opened with.
    #[error("failed to build the Vulkan frames context: {0}")]
    HwFrames(String),

    /// FFmpeg could not take another reference to the device or its frames.
    #[error("failed to reference the Vulkan device context")]
    HwDeviceRef,
}

/// Encodes NV12 Vulkan frames into `Packet`s with Vulkan Video, on a
/// [`VulkanDevice`] — on any GPU whose driver encodes through Vulkan. The
/// Vulkan counterpart of `CudaEncoder`, and a `Filter` as it is.
///
/// Fed by [`crate::elements::VulkanDecoder`] this is a transcode that never
/// brings a pixel to the CPU; fed by an NV12
/// [`crate::elements::VulkanVideoCompositor`] it records a composition as
/// it is made; fed by [`crate::elements::VulkanUpload`] it replaces a
/// software encoder.
///
/// # Packet timing
///
/// As `CudaEncoder`'s: packets are drained after every frame and at `Eos`,
/// and each is stamped with this encoder's [`Self::time_base`] and a
/// nominal duration.
pub struct VulkanEncoder {
    pp_log: PpLog,
    name: Arc<str>,
    encoder: ffmpeg::encoder::Video,
    _hw_device_ctx: Arc<AvBufferRef>,
    _hw_frames_ctx: AvBufferRef,
    /// This encoder's device, to compare an incoming frame's against. Only
    /// ever compared.
    device_ctx: *const ffi::AVHWDeviceContext,
    width: u32,
    height: u32,
    /// Nominal frame duration in `time_base` ticks, which the encoder
    /// leaves at zero.
    packet_duration: i64,
    pad: SrcPad,
}

// SAFETY: both buffers are heap-allocated FFmpeg buffers with no thread
// affinity, `device_ctx` is only ever compared, and `encoder`'s own `Send`
// covers the codec context. `&mut self` on every method that touches them
// rules out concurrent access — same reasoning as `CudaEncoder`.
unsafe impl Send for VulkanEncoder {}

fn nominal_packet_duration(time_base: ffmpeg::Rational, frame_rate: ffmpeg::Rational) -> i64 {
    if frame_rate.numerator() <= 0 || time_base.numerator() <= 0 {
        return 0;
    }
    let ticks = f64::from(time_base.denominator()) * f64::from(frame_rate.denominator())
        / (f64::from(time_base.numerator()) * f64::from(frame_rate.numerator()));
    ticks.round() as i64
}

impl VulkanEncoder {
    /// `device` must be the same [`VulkanDevice`] the frames are made on — a
    /// frame from another is refused.
    ///
    /// Opens the encoder eagerly, so a missing encoder, a GPU without Vulkan
    /// Video encoding or without this codec, or a size it refuses all surface
    /// here as a typed error rather than at the first frame.
    pub fn new(
        name: impl Into<String>,
        device: &VulkanDevice,
        options: VulkanEncoderOptions,
    ) -> std::result::Result<Self, VulkanEncoderError> {
        crate::ensure_ffmpeg();
        Self::open(name, device, options, None)
    }

    /// As [`Self::new`], and the stream says `color` is what it holds —
    /// written into its headers, where every player finds it; nothing is
    /// converted. An NV12 [`crate::elements::VulkanVideoCompositor`]'s canvas
    /// is [`ColorDescription::BT709_LIMITED`].
    pub fn with_color(
        name: impl Into<String>,
        device: &VulkanDevice,
        options: VulkanEncoderOptions,
        color: ColorDescription,
    ) -> std::result::Result<Self, VulkanEncoderError> {
        crate::ensure_ffmpeg();
        Self::open(name, device, options, Some(color))
    }

    fn open(
        name: impl Into<String>,
        device: &VulkanDevice,
        options: VulkanEncoderOptions,
        color: Option<ColorDescription>,
    ) -> std::result::Result<Self, VulkanEncoderError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::VulkanEncoder, &name, None);

        let encoder_name = options.codec.encoder_name();
        let codec = ffmpeg::encoder::find_by_name(encoder_name)
            .ok_or(VulkanEncoderError::CodecNotFound(encoder_name))?;

        let hw_device_ctx = device.retain();
        // SAFETY: `create_frames_ctx`'s contract is a live device context,
        // which the reference just taken is. Its usage is FFmpeg's own
        // choice, which is what an encoder reads from.
        let hw_frames_ctx = unsafe {
            create_frames_ctx(
                &hw_device_ctx,
                ffmpeg::format::Pixel::NV12,
                options.width,
                options.height,
                ash::vk::ImageUsageFlags::empty(),
            )
        }
        .map_err(|error| VulkanEncoderError::HwFrames(error.to_string()))?;
        let codec_device_ctx = hw_device_ctx
            .try_clone()
            .ok_or(VulkanEncoderError::HwDeviceRef)?;
        let codec_frames_ctx = hw_frames_ctx
            .try_clone()
            .ok_or(VulkanEncoderError::HwDeviceRef)?;

        let opened = (|| -> std::result::Result<ffmpeg::encoder::Video, ffmpeg::Error> {
            let mut ctx = ffmpeg::codec::context::Context::new_with_codec(codec);
            // Codec headers into `extradata` for the container to write, not
            // only in-band — see `SwEncoder::new`.
            ctx.set_flags(ffmpeg::codec::Flags::GLOBAL_HEADER);
            let mut video = ctx.encoder().video()?;
            video.set_width(options.width);
            video.set_height(options.height);
            video.set_format(ffmpeg::format::Pixel::VULKAN);
            video.set_time_base(shared::TIME_BASE);
            video.set_frame_rate(Some(options.frame_rate));
            video.set_bit_rate(options.bit_rate);
            video.set_gop(options.gop_size);
            if let Some(frames) = options.max_b_frames {
                video.set_max_b_frames(frames as usize);
            }
            // SAFETY: the encoder's own context, not yet opened, which is when
            // both have to be set; both references are transferred with
            // `into_raw`, so the codec frees them.
            unsafe {
                let ptr = video.as_mut_ptr();
                (*ptr).hw_device_ctx = codec_device_ctx.into_raw();
                (*ptr).hw_frames_ctx = codec_frames_ctx.into_raw();
                if let Some(color) = color {
                    color.tell(ptr);
                }
            }
            video.open_as(codec)
        })();
        let encoder = opened?;

        let pad = SrcPad::with_contract(
            format!("{name}_src"),
            OutputContract::Fixed(PortContract::packet(MediaKind::VideoPacket)),
        );
        pp_info!(
            pp_log: &pp_log,
            "opened: {} {}x{} on {}, {} bps, gop={}",
            encoder_name,
            options.width,
            options.height,
            device.name(),
            options.bit_rate,
            options.gop_size
        );
        Ok(Self {
            name,
            pp_log,
            encoder,
            _hw_device_ctx: hw_device_ctx,
            _hw_frames_ctx: hw_frames_ctx,
            device_ctx: device.device_ctx(),
            width: options.width,
            height: options.height,
            packet_duration: nominal_packet_duration(shared::TIME_BASE, options.frame_rate),
            pad,
        })
    }

    /// The encoded stream's parameters, for
    /// [`crate::elements::FileMuxer::add_stream`].
    pub fn parameters(&self) -> ffmpeg::codec::Parameters {
        ffmpeg::codec::Parameters::from(&self.encoder)
    }

    /// The unit each packet's `pts`, `dts` and duration are counted in — the
    /// encoder's own, into which each frame's timestamp is converted.
    pub fn time_base(&self) -> ffmpeg::Rational {
        shared::TIME_BASE
    }

    fn encode(&mut self, frame: &ffmpeg::frame::Video) -> Result<()> {
        match sw_format_of(frame, self.device_ctx) {
            Ok(ffmpeg::format::Pixel::NV12) => {}
            Ok(other) => return Err(self.refused(VulkanEncoderError::UnsupportedLayout(other))),
            Err(NotOurs::NotVulkan(format)) => {
                return Err(self.refused(VulkanEncoderError::NotVulkan(format)));
            }
            Err(NotOurs::ForeignDevice) => {
                return Err(self.refused(VulkanEncoderError::ForeignDevice));
            }
        }
        if frame.width() != self.width || frame.height() != self.height {
            return Err(self.refused(VulkanEncoderError::DimensionMismatch {
                actual_width: frame.width(),
                actual_height: frame.height(),
                expected_width: self.width,
                expected_height: self.height,
            }));
        }
        let pts = shared::pts_in(frame, shared::TIME_BASE)
            .map_err(|shared::NoTimeBase| VulkanEncoderError::NoTimeBase)?;
        let frame =
            shared::restamped(frame, pts, shared::TIME_BASE).map_err(VulkanEncoderError::from)?;
        self.encoder
            .send_frame(&frame)
            .inspect_err(|error| pp_error!(self, "send_frame failed: {error}"))
            .map_err(VulkanEncoderError::from)?;
        self.drain()
    }

    fn refused(&self, error: VulkanEncoderError) -> crate::error::Error {
        pp_error!(self, "{error}");
        error.into()
    }

    fn drain(&mut self) -> Result<()> {
        let mut packet = ffmpeg::Packet::empty();
        loop {
            match self.encoder.receive_packet(&mut packet) {
                Ok(()) => {
                    packet.set_time_base(shared::TIME_BASE);
                    if packet.duration() == 0 && self.packet_duration > 0 {
                        packet.set_duration(self.packet_duration);
                    }
                    self.pad.push(MediaBuffer::Packet(Arc::new(packet)))?;
                    packet = ffmpeg::Packet::empty();
                }
                Err(error) if is_codec_drain_boundary(&error) => break,
                Err(error) => return Err(VulkanEncoderError::from(error).into()),
            }
        }
        Ok(())
    }
}

/// The track this encoder's packets make: its [`VulkanEncoder::parameters`],
/// timed in its [`VulkanEncoder::time_base`].
impl From<&VulkanEncoder> for crate::elements::TrackFormat {
    fn from(encoder: &VulkanEncoder) -> Self {
        Self::new(encoder.parameters(), encoder.time_base())
    }
}

impl Element for VulkanEncoder {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::VulkanEncoder
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for VulkanEncoder {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for VulkanEncoder {
    /// Vulkan Video reads device memory; a system-memory frame needs a
    /// VulkanUpload first.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::Vulkan)
                .with_layouts(crate::contract::PixelLayoutSet::NV12),
        )
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Video(frame) => self.encode(&frame),
            MediaBuffer::Eos => {
                self.encoder
                    .send_eof()
                    .inspect_err(|error| pp_error!(self, "send_eof failed: {error}"))
                    .map_err(VulkanEncoderError::from)?;
                self.drain()?;
                self.pad.push(MediaBuffer::Eos)
            }
            other => Err(VulkanEncoderError::UnsupportedBuffer(other.kind()).into()),
        }
    }

    fn control(&mut self, _msg: &ControlMsg) -> Result<()> {
        // Nothing to reset, as for every encoder here: a caller needing a
        // hard encoded-stream discontinuity rebuilds the encoder.
        Ok(())
    }
}

impl Drop for VulkanEncoder {
    fn drop(&mut self) {
        pp_info!(self, "dropped: releasing hw contexts");
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::{
        elements::{SwDecoder, VulkanUpload},
        test_support::{CapturingSink, try_vulkan_device},
    };

    fn options(codec: VulkanCodec, width: u32, height: u32) -> VulkanEncoderOptions {
        VulkanEncoderOptions {
            codec,
            width,
            height,
            frame_rate: ffmpeg::Rational::new(30, 1),
            bit_rate: 4_000_000,
            gop_size: 30,
            max_b_frames: None,
        }
    }

    type Received = Arc<Mutex<Vec<MediaBuffer>>>;

    fn capture(source: &mut dyn Source) -> Received {
        let received = Arc::new(Mutex::new(Vec::new()));
        source.src_pads()[0].link(Box::new(CapturingSink {
            received: received.clone(),
            pp_log: element_pp_log(ElementType::Other, "capture", None),
        }));
        received
    }

    /// An encoder of `codec`, or `None` — with the reason — where this GPU
    /// or this FFmpeg has none.
    fn try_encoder(
        device: &VulkanDevice,
        codec: VulkanCodec,
        width: u32,
        height: u32,
    ) -> Option<VulkanEncoder> {
        match VulkanEncoder::new("encoder", device, options(codec, width, height)) {
            Ok(encoder) => Some(encoder),
            Err(error) => {
                eprintln!("skipping: no Vulkan {codec:?} encoder here ({error})");
                None
            }
        }
    }

    /// A moving luma ramp at `index`, so the encoder has real content and a
    /// picture that decodes to something checkable.
    fn ramp(width: u32, height: u32, index: i64) -> ffmpeg::frame::Video {
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, width, height);
        let stride = frame.stride(0);
        for y in 0..height as usize {
            for x in 0..width as usize {
                frame.data_mut(0)[y * stride + x] = (16 + (x as i64 + index * 4) % 200) as u8;
            }
        }
        frame.data_mut(1).fill(128);
        frame.set_pts(Some(index));
        crate::buffer::set_time_base(&mut frame, ffmpeg::Rational::new(1, 30));
        frame
    }

    /// Every packet of `packets`, decoded in software.
    fn decode(
        parameters: ffmpeg::codec::Parameters,
        packets: &[MediaBuffer],
    ) -> Vec<Arc<crate::pool::UnboundObjectPoolRef<ffmpeg::frame::Video>>> {
        let mut decoder = SwDecoder::new("check", parameters).expect("a software decoder");
        let decoded = capture(&mut decoder);
        for packet in packets {
            decoder.consume(packet.clone()).expect("decodes");
        }
        let decoded = decoded.lock().unwrap();
        decoded
            .iter()
            .filter_map(|buffer| match buffer {
                MediaBuffer::Video(frame) => Some(frame.clone()),
                _ => None,
            })
            .collect()
    }

    /// Uploaded frames encode into packets that decode back to the picture
    /// that was sent, every one, each packet timed in the encoder's base;
    /// the end of the stream drains what is left and is passed on.
    #[test]
    fn frames_encode_into_packets_that_decode_back_to_them() {
        let Some(device) = try_vulkan_device() else {
            return;
        };
        let _session = crate::test_support::encoder_session();
        let (width, height) = (320u32, 240u32);
        let Some(mut encoder) = try_encoder(&device, VulkanCodec::H264, width, height) else {
            return;
        };
        let packets = capture(&mut encoder);
        let mut upload = VulkanUpload::new("upload", &device);
        let uploaded = capture(&mut upload);
        let frames = 30;
        for index in 0..frames {
            upload
                .consume(MediaBuffer::video(ramp(width, height, index)))
                .unwrap();
            let frame = uploaded.lock().unwrap().remove(0);
            encoder.consume(frame).expect("encode");
        }
        encoder.consume(MediaBuffer::Eos).expect("eos");

        let packets = std::mem::take(&mut *packets.lock().unwrap());
        assert!(
            packets.last().is_some_and(MediaBuffer::is_eos),
            "Eos passed on"
        );
        for packet in &packets {
            if let MediaBuffer::Packet(packet) = packet {
                assert_eq!(packet.time_base(), encoder.time_base());
                assert!(packet.duration() > 0);
            }
        }
        let decoded = decode(encoder.parameters(), &packets);
        assert_eq!(decoded.len(), frames as usize, "every frame came back");
        let last = decoded.last().unwrap();
        let sent = ramp(width, height, frames - 1);
        let mean = (0..height as usize)
            .flat_map(|y| (0..width as usize).map(move |x| (x, y)))
            .map(|(x, y)| {
                let got = last.data(0)[y * last.stride(0) + x];
                let wanted = sent.data(0)[y * sent.stride(0) + x];
                f64::from(got.abs_diff(wanted))
            })
            .sum::<f64>()
            / f64::from(width * height);
        assert!(
            mean < 3.0,
            "the last picture decodes to what was sent: {mean}"
        );
    }

    /// A frame in system memory and one of another device are refused by
    /// name, before anything is encoded.
    #[test]
    fn a_cpu_frame_and_a_foreign_device_frame_are_refused() {
        let Some(device) = try_vulkan_device() else {
            return;
        };
        let _session = crate::test_support::encoder_session();
        let Some(mut encoder) = try_encoder(&device, VulkanCodec::H264, 320, 240) else {
            return;
        };
        let _ = capture(&mut encoder);
        let error = encoder
            .consume(MediaBuffer::video(ramp(320, 240, 0)))
            .unwrap_err();
        assert!(error.to_string().contains("upload it first"), "{error}");

        let other = VulkanDevice::new().expect("a second device");
        let mut upload = VulkanUpload::new("upload", &other);
        let uploaded = capture(&mut upload);
        upload
            .consume(MediaBuffer::video(ramp(320, 240, 0)))
            .unwrap();
        let error = encoder
            .consume(uploaded.lock().unwrap().remove(0))
            .unwrap_err();
        assert!(
            error.to_string().contains("another Vulkan device"),
            "{error}"
        );
    }

    /// H.265 and AV1 open where the GPU encodes them, and say why not where
    /// it does not.
    #[test]
    fn the_other_codecs_open_or_say_why_not() {
        let Some(device) = try_vulkan_device() else {
            return;
        };
        let _session = crate::test_support::encoder_session();
        for codec in [VulkanCodec::H265, VulkanCodec::Av1] {
            match VulkanEncoder::new("encoder", &device, options(codec, 320, 240)) {
                Ok(encoder) => assert_ne!(encoder.parameters().id(), ffmpeg::codec::Id::None),
                Err(error) => eprintln!("{codec:?} not here: {error}"),
            }
        }
    }
}
