use std::sync::Arc;

use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_error, pp_info};

use crate::color::ColorDescription;
use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    element::{Element, ElementType, Output, Transform, element_pp_log},
    elements::{VideoToolboxDevice, filter::is_codec_drain_boundary},
    error::Result,
    platform::{
        ffmpeg::AvBufferRef,
        macos::videotoolbox::{NotVideoToolbox, create_frames_ctx, sw_format_of},
    },
    transform::{TransformStage, transform_filter},
};

// The video-encoder helpers every backend shares, a module up from this one.
use super::super as shared;

/// Which of FFmpeg's VideoToolbox encoders [`VideoToolboxEncoder`] drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoToolboxCodec {
    /// `h264_videotoolbox` — H.264/AVC.
    H264,
    /// `hevc_videotoolbox` — H.265/HEVC.
    H265,
}

impl VideoToolboxCodec {
    fn encoder_name(self) -> &'static str {
        match self {
            Self::H264 => "h264_videotoolbox",
            Self::H265 => "hevc_videotoolbox",
        }
    }
}

/// Construction-time options for [`VideoToolboxEncoder`] — the same knobs,
/// with the same meanings, as `CudaEncoderOptions`.
#[derive(Debug, Clone, Copy)]
pub struct VideoToolboxEncoderOptions {
    /// The bitstream to encode.
    pub codec: VideoToolboxCodec,
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

/// Errors specific to [`VideoToolboxEncoder`]. Converts into the crate-wide
/// `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum VideoToolboxEncoderError {
    /// The encoder is not in this FFmpeg build.
    #[error("encoder `{0}` not found in this ffmpeg build")]
    CodecNotFound(&'static str),

    /// FFmpeg rejected the encoder or a frame — among other reasons, a size
    /// or setting this Mac's encoder will not take fails to open.
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),

    /// The sink received a buffer other than decoded video or end-of-stream.
    #[error("VideoToolboxEncoder only accepts Video buffers, got a {0}")]
    UnsupportedBuffer(&'static str),

    /// The frame is not a VideoToolbox frame.
    #[error("VideoToolboxEncoder encodes VideoToolbox frames, got a {0:?} frame; upload it first")]
    NotVideoToolbox(ffmpeg::format::Pixel),

    /// The frame says it is a VideoToolbox frame but carries no frames
    /// context to say what it holds.
    #[error("VideoToolboxEncoder was handed a VideoToolbox frame with no frames context")]
    NoFramesContext,

    /// The frame holds a layout other than NV12.
    #[error("VideoToolboxEncoder encodes NV12 frames, got {0:?}")]
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
        "frame is {actual_width}x{actual_height}, but this VideoToolboxEncoder was built for \
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
    #[error("failed to build the VideoToolbox frames context: {0}")]
    HwFrames(String),

    /// FFmpeg could not take another reference to the frames context.
    #[error("failed to reference the VideoToolbox frames context")]
    HwFramesRef,
}

/// Encodes NV12 VideoToolbox frames into `Packet`s with the Mac's hardware
/// encoder, through FFmpeg's VideoToolbox encoders — the macOS counterpart
/// of `CudaEncoder` and `VideoToolboxEncoder`, and a `Filter` as they are.
///
/// Fed by [`crate::elements::VideoToolboxDecoder`] this is a transcode that
/// never brings a pixel to the CPU; fed by
/// [`crate::elements::VideoToolboxUpload`] it replaces a software encoder.
///
/// It reads each frame's own pixel buffer, which belongs to no context, so
/// frames from any VideoToolbox element in the process encode; only their
/// layout and size are checked.
///
/// # Packet timing
///
/// As `CudaEncoder`'s: packets are drained after every frame and at `Eos`,
/// and each is stamped with this encoder's [`Self::time_base`] and a
/// nominal duration.
pub struct VideoToolboxEncoder(TransformStage<Encoding>);

transform_filter!(VideoToolboxEncoder);

/// What a [`VideoToolboxEncoder`] does to each buffer: all of its work, which the
/// framework makes the filter.
struct Encoding {
    pp_log: PpLog,
    name: Arc<str>,
    encoder: ffmpeg::encoder::Video,
    /// What the encoder was opened with: it reads the layout of its input
    /// from it.
    _hw_frames_ctx: AvBufferRef,
    width: u32,
    height: u32,
    /// Nominal frame duration in `time_base` ticks, which the encoder
    /// leaves at zero.
    packet_duration: i64,
}

// SAFETY: the frames context is a heap-allocated FFmpeg buffer with no thread
// affinity, and `encoder`'s own `Send` covers the codec context; VideoToolbox
// sessions may be driven from any thread. `&mut self` on every method that
// touches them rules out concurrent access — same reasoning as
// `CudaEncoder`.
unsafe impl Send for Encoding {}

fn nominal_packet_duration(time_base: ffmpeg::Rational, frame_rate: ffmpeg::Rational) -> i64 {
    if frame_rate.numerator() <= 0 || time_base.numerator() <= 0 {
        return 0;
    }
    let ticks = f64::from(time_base.denominator()) * f64::from(frame_rate.denominator())
        / (f64::from(time_base.numerator()) * f64::from(frame_rate.numerator()));
    ticks.round() as i64
}

impl VideoToolboxEncoder {
    /// `device` is the [`VideoToolboxDevice`] the pipeline's other
    /// VideoToolbox elements share; the encoder's frames context is made on
    /// it.
    ///
    /// Opens the encoder eagerly, so a missing encoder, or a size or setting
    /// this Mac refuses, surfaces here as a typed error rather than at the
    /// first frame.
    pub fn new(
        name: impl Into<String>,
        device: &VideoToolboxDevice,
        options: VideoToolboxEncoderOptions,
    ) -> std::result::Result<Self, VideoToolboxEncoderError> {
        crate::ensure_ffmpeg();
        Self::open(name, device, options, None)
    }

    /// As [`Self::new`], and the stream says `color` is what it holds —
    /// written into its headers, where every player finds it; nothing is
    /// converted.
    pub fn with_color(
        name: impl Into<String>,
        device: &VideoToolboxDevice,
        options: VideoToolboxEncoderOptions,
        color: ColorDescription,
    ) -> std::result::Result<Self, VideoToolboxEncoderError> {
        crate::ensure_ffmpeg();
        Self::open(name, device, options, Some(color))
    }

    fn open(
        name: impl Into<String>,
        device: &VideoToolboxDevice,
        options: VideoToolboxEncoderOptions,
        color: Option<ColorDescription>,
    ) -> std::result::Result<Self, VideoToolboxEncoderError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::VideoToolboxEncoder, &name, None);

        let encoder_name = options.codec.encoder_name();
        let codec = ffmpeg::encoder::find_by_name(encoder_name)
            .ok_or(VideoToolboxEncoderError::CodecNotFound(encoder_name))?;

        // The encoder reads the layout of the pixel buffers it is handed from
        // the frames context it is opened with — NV12 — and the pixel buffers
        // themselves from each frame.
        // SAFETY: `create_frames_ctx`'s contract is a live device context,
        // which the device's own reference is for the length of the call.
        let hw_frames_ctx = unsafe {
            create_frames_ctx(
                &device.retain(),
                ffmpeg::format::Pixel::NV12,
                options.width,
                options.height,
            )
        }
        .map_err(|error| VideoToolboxEncoderError::HwFrames(error.to_string()))?;
        let codec_frames_ctx = hw_frames_ctx
            .try_clone()
            .ok_or(VideoToolboxEncoderError::HwFramesRef)?;

        let opened = (|| -> std::result::Result<ffmpeg::encoder::Video, ffmpeg::Error> {
            let mut ctx = ffmpeg::codec::context::Context::new_with_codec(codec);
            // Codec headers into `extradata` for the container to write, not
            // only in-band — see `SwEncoder::new`.
            ctx.set_flags(ffmpeg::codec::Flags::GLOBAL_HEADER);
            let mut video = ctx.encoder().video()?;
            video.set_width(options.width);
            video.set_height(options.height);
            video.set_format(ffmpeg::format::Pixel::VIDEOTOOLBOX);
            video.set_time_base(shared::TIME_BASE);
            video.set_frame_rate(Some(options.frame_rate));
            video.set_bit_rate(options.bit_rate);
            video.set_gop(options.gop_size);
            if let Some(frames) = options.max_b_frames {
                video.set_max_b_frames(frames as usize);
            }
            // SAFETY: the encoder's own context, not yet opened, which is when
            // it has to be set; the reference is transferred with `into_raw`,
            // so the codec frees it.
            unsafe {
                let ptr = video.as_mut_ptr();
                (*ptr).hw_frames_ctx = codec_frames_ctx.into_raw();
                if let Some(color) = color {
                    color.tell(ptr);
                }
            }
            video.open_as(codec)
        })();
        let encoder = opened?;

        pp_info!(
            pp_log: &pp_log,
            "opened: {} {}x{}, {} bps, gop={}",
            encoder_name,
            options.width,
            options.height,
            options.bit_rate,
            options.gop_size
        );
        Ok(Self(TransformStage::new(Encoding {
            name,
            pp_log,
            encoder,
            _hw_frames_ctx: hw_frames_ctx,
            width: options.width,
            height: options.height,
            packet_duration: nominal_packet_duration(shared::TIME_BASE, options.frame_rate),
        })))
    }

    /// The encoded stream's parameters, for
    /// [`crate::elements::FileMuxer::add_stream`].
    pub fn parameters(&self) -> ffmpeg::codec::Parameters {
        self.0.inner().parameters()
    }

    /// The unit each packet's `pts`, `dts` and duration are counted in — the
    /// encoder's own, into which each frame's timestamp is converted.
    pub fn time_base(&self) -> ffmpeg::Rational {
        self.0.inner().time_base()
    }
}

impl Encoding {
    fn parameters(&self) -> ffmpeg::codec::Parameters {
        ffmpeg::codec::Parameters::from(&self.encoder)
    }

    fn time_base(&self) -> ffmpeg::Rational {
        shared::TIME_BASE
    }

    fn encode(&mut self, frame: &ffmpeg::frame::Video, out: &mut Output) -> Result<()> {
        match sw_format_of(frame) {
            Ok(ffmpeg::format::Pixel::NV12) => {}
            Ok(other) => {
                return Err(self.refused(VideoToolboxEncoderError::UnsupportedLayout(other)));
            }
            Err(NotVideoToolbox::Format(format)) => {
                return Err(self.refused(VideoToolboxEncoderError::NotVideoToolbox(format)));
            }
            Err(NotVideoToolbox::NoFramesContext) => {
                return Err(self.refused(VideoToolboxEncoderError::NoFramesContext));
            }
        }
        if frame.width() != self.width || frame.height() != self.height {
            return Err(self.refused(VideoToolboxEncoderError::DimensionMismatch {
                actual_width: frame.width(),
                actual_height: frame.height(),
                expected_width: self.width,
                expected_height: self.height,
            }));
        }
        let pts = shared::pts_in(frame, shared::TIME_BASE)
            .map_err(|shared::NoTimeBase| VideoToolboxEncoderError::NoTimeBase)?;
        let frame = shared::restamped(frame, pts, shared::TIME_BASE)
            .map_err(VideoToolboxEncoderError::from)?;
        self.encoder
            .send_frame(&frame)
            .inspect_err(|error| pp_error!(self, "send_frame failed: {error}"))
            .map_err(VideoToolboxEncoderError::from)?;
        self.receive_packets(out)
    }

    fn refused(&self, error: VideoToolboxEncoderError) -> crate::error::Error {
        pp_error!(self, "{error}");
        error.into()
    }

    fn receive_packets(&mut self, out: &mut Output) -> Result<()> {
        let mut packet = ffmpeg::Packet::empty();
        loop {
            match self.encoder.receive_packet(&mut packet) {
                Ok(()) => {
                    packet.set_time_base(shared::TIME_BASE);
                    if packet.duration() == 0 && self.packet_duration > 0 {
                        packet.set_duration(self.packet_duration);
                    }
                    out.push(MediaBuffer::Packet(Arc::new(packet)));
                    packet = ffmpeg::Packet::empty();
                }
                Err(error) if is_codec_drain_boundary(&error) => break,
                Err(error) => return Err(VideoToolboxEncoderError::from(error).into()),
            }
        }
        Ok(())
    }
}

/// The track this encoder's packets make: its
/// [`VideoToolboxEncoder::parameters`], timed in its
/// [`VideoToolboxEncoder::time_base`].
impl From<&VideoToolboxEncoder> for crate::elements::TrackFormat {
    fn from(encoder: &VideoToolboxEncoder) -> Self {
        Self::new(encoder.parameters(), encoder.time_base())
    }
}

impl Element for Encoding {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::VideoToolboxEncoder
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Transform for Encoding {
    fn output_contract(&self) -> OutputContract {
        OutputContract::Fixed(PortContract::packet(MediaKind::VideoPacket))
    }

    /// VideoToolbox reads pixel buffers; a system-memory frame needs a
    /// VideoToolboxUpload first.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::VideoToolbox)
                .with_layouts(crate::contract::PixelLayoutSet::NV12),
        )
    }

    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        match buf {
            MediaBuffer::Video(frame) => self.encode(&frame, out),
            other => Err(VideoToolboxEncoderError::UnsupportedBuffer(other.kind()).into()),
        }
    }

    /// Sends the end of the stream in and hands on what comes out.
    fn drain(&mut self, out: &mut Output) -> Result<()> {
        self.encoder
            .send_eof()
            .inspect_err(|error| pp_error!(self, "send_eof failed: {error}"))
            .map_err(VideoToolboxEncoderError::from)?;
        self.receive_packets(out)
    }

    fn reset(&mut self) {
        // Nothing to reset, as for every encoder here: a caller needing a
        // hard encoded-stream discontinuity rebuilds the encoder.
    }
}

impl Drop for Encoding {
    fn drop(&mut self) {
        pp_info!(self, "dropped: releasing the frames context");
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::element::{Sink, Source};
    use crate::{
        elements::{SwDecoder, VideoToolboxDecoder, VideoToolboxUpload},
        test_support::{CapturingSink, try_videotoolbox_device},
    };

    fn options(codec: VideoToolboxCodec, width: u32, height: u32) -> VideoToolboxEncoderOptions {
        VideoToolboxEncoderOptions {
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
        device: &VideoToolboxDevice,
        codec: VideoToolboxCodec,
        width: u32,
        height: u32,
    ) -> Option<VideoToolboxEncoder> {
        match VideoToolboxEncoder::new("encoder", device, options(codec, width, height)) {
            Ok(encoder) => Some(encoder),
            Err(error) => {
                eprintln!("skipping: no VideoToolbox {codec:?} encoder here ({error})");
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
        decoder
            .stream_event(&crate::stream::StreamEvent::Eos)
            .expect("drains");
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
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let _session = crate::test_support::encoder_session();
        let (width, height) = (320u32, 240u32);
        let Some(mut encoder) = try_encoder(&device, VideoToolboxCodec::H264, width, height) else {
            return;
        };
        let packets = capture(&mut encoder);
        let mut upload = VideoToolboxUpload::new("upload", &device);
        let uploaded = capture(&mut upload);
        let frames = 30;
        for index in 0..frames {
            upload
                .consume(MediaBuffer::video(ramp(width, height, index)))
                .unwrap();
            let frame = uploaded.lock().unwrap().remove(0);
            encoder.consume(frame).expect("encode");
        }
        crate::stream::deliver(&mut encoder, &crate::stream::StreamEvent::Eos).expect("eos");

        let packets = std::mem::take(&mut *packets.lock().unwrap());
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

    /// A frame in system memory is refused by name, before anything is
    /// encoded; so is a VideoToolbox frame of another layout.
    #[test]
    fn a_cpu_frame_and_another_layout_are_refused() {
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let _session = crate::test_support::encoder_session();
        let Some(mut encoder) = try_encoder(&device, VideoToolboxCodec::H264, 320, 240) else {
            return;
        };
        let _ = capture(&mut encoder);
        let error = encoder
            .consume(MediaBuffer::video(ramp(320, 240, 0)))
            .unwrap_err();
        assert!(error.to_string().contains("upload it first"), "{error}");

        let mut upload = VideoToolboxUpload::new("upload", &device);
        let uploaded = capture(&mut upload);
        let mut bgra = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, 320, 240);
        bgra.set_pts(Some(0));
        crate::buffer::set_time_base(&mut bgra, ffmpeg::Rational::new(1, 30));
        upload.consume(MediaBuffer::video(bgra)).unwrap();
        let error = encoder
            .consume(uploaded.lock().unwrap().remove(0))
            .unwrap_err();
        assert!(error.to_string().contains("got BGRA"), "{error}");
    }

    /// A stream decoded by VideoToolbox encodes straight from its frames,
    /// never read back: every picture of it comes out as a packet, and the
    /// packets decode.
    #[test]
    fn decoded_frames_encode_without_leaving_videotoolbox() {
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let _session = crate::test_support::encoder_session();
        let Some(path) = crate::test_support::try_test_video() else {
            return;
        };
        let mut input = ffmpeg::format::input(&path).expect("open the test video");
        let stream = input
            .streams()
            .best(ffmpeg::media::Type::Video)
            .expect("the test video has a picture");
        let (index, params) = (stream.index(), stream.parameters());
        let time_base = stream.time_base();
        let (width, height) = {
            let context = ffmpeg::codec::context::Context::from_parameters(params.clone())
                .expect("parameters");
            let video = context.decoder().video().expect("a picture");
            (video.width(), video.height())
        };
        let mut decoder = VideoToolboxDecoder::new("decoder", params, &device).expect("a decoder");
        let decoded = capture(&mut decoder);
        let Some(mut encoder) = try_encoder(&device, VideoToolboxCodec::H264, width, height) else {
            return;
        };
        let packets = capture(&mut encoder);
        let mut sent = 0;
        for (stream, mut packet) in input.packets() {
            if stream.index() != index {
                continue;
            }
            packet.set_time_base(time_base);
            decoder
                .consume(MediaBuffer::Packet(Arc::new(packet)))
                .expect("decode");
            sent += 1;
            for frame in decoded.lock().unwrap().drain(..) {
                if let MediaBuffer::Video(picture) = &frame {
                    assert_eq!(picture.format(), ffmpeg::format::Pixel::VIDEOTOOLBOX);
                }
                encoder.consume(frame).expect("encode a decoded frame");
            }
        }
        crate::stream::deliver(&mut decoder, &crate::stream::StreamEvent::Eos).expect("eos");
        for frame in decoded.lock().unwrap().drain(..) {
            encoder.consume(frame).expect("encode a decoded frame");
        }
        crate::stream::deliver(&mut encoder, &crate::stream::StreamEvent::Eos).expect("eos");
        let packets = std::mem::take(&mut *packets.lock().unwrap());
        let decoded_back = decode(encoder.parameters(), &packets);
        assert_eq!(decoded_back.len(), sent, "every picture was encoded");
    }

    /// H.265 opens too, on every Mac with a media engine.
    #[test]
    fn h265_opens() {
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let _session = crate::test_support::encoder_session();
        match VideoToolboxEncoder::new(
            "encoder",
            &device,
            options(VideoToolboxCodec::H265, 320, 240),
        ) {
            Ok(encoder) => assert_eq!(encoder.parameters().id(), ffmpeg::codec::Id::HEVC),
            Err(error) => eprintln!("skipping: H.265 not here: {error}"),
        }
    }
}
