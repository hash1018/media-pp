use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;

use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_error, pp_info, pp_warn};

use crate::color::ColorDescription;
use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    element::{Element, ElementType, Filter, Output, element_pp_log},
    elements::{VideoToolboxDevice, VideoToolboxFrameFormat, filter::is_codec_drain_boundary},
    error::Result,
    platform::{
        ffmpeg::AvBufferRef,
        macos::videotoolbox::{NotVideoToolbox, create_frames_ctx, sw_format_of},
    },
    transform::{FilterStage, filter_stage},
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
    /// What the frames it is handed hold: P010 as HEVC Main 10, which
    /// H.264 refuses. BGRA is converted to YUV by
    /// VideoToolbox itself, on the media engine, always with BT.709's
    /// matrix at limited range — which is what the stream then says it
    /// holds; see [`VideoToolboxEncoder::with_color`].
    pub format: VideoToolboxFrameFormat,
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
    /// [`crate::elements::SwEncoderOptions::max_b_frames`]. With any, the
    /// packets' decode timestamps are this encoder's own, as VideoToolbox
    /// reorders H.264 deeper than FFmpeg dates it.
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

    /// The frame holds a layout other than the one the encoder was opened
    /// for.
    #[error("VideoToolboxEncoder was opened for {expected:?} frames, got {got:?}")]
    UnsupportedLayout {
        /// What [`VideoToolboxEncoderOptions::format`] says.
        expected: ffmpeg::format::Pixel,
        /// What the frame holds.
        got: ffmpeg::format::Pixel,
    },

    /// P010 for H.264, which has no ten bits: HEVC encodes it, as Main 10.
    #[error("VideoToolboxEncoder encodes P010 as HEVC Main 10 only, not H.264")]
    TenBitsNeedHevc,

    /// [`VideoToolboxEncoder::with_color`] was asked to describe BGRA
    /// input, whose YUV VideoToolbox makes itself — always BT.709 at limited
    /// range, whatever the stream says.
    #[error(
        "VideoToolboxEncoder converts BGRA with BT.709 at limited range and says so; \
         another colour description would misdescribe the stream"
    )]
    ColorOfBgra,

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

/// Encodes NV12, P010 or BGRA VideoToolbox frames into `Packet`s with the Mac's
/// hardware encoder, through FFmpeg's VideoToolbox encoders — the macOS counterpart
/// of `CudaEncoder` and `VideoToolboxEncoder`, and a `RawFilter` as they are.
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
pub struct VideoToolboxEncoder(FilterStage<Encoding>);

filter_stage!(VideoToolboxEncoder);

/// What a [`VideoToolboxEncoder`] does to each buffer: all of its work, which the
/// framework makes the filter.
struct Encoding {
    pp_log: PpLog,
    name: Arc<str>,
    encoder: ffmpeg::encoder::Video,
    /// What the encoder was opened with: it reads the layout of its input
    /// from it.
    _hw_frames_ctx: AvBufferRef,
    /// What its input holds.
    format: VideoToolboxFrameFormat,
    width: u32,
    height: u32,
    /// Nominal frame duration in `time_base` ticks, which the encoder
    /// leaves at zero.
    packet_duration: i64,
    /// Where B-frames are allowed, the decode timestamps this writes in
    /// place of FFmpeg's: see [`Dating`].
    dating: Option<Dating>,
}

/// How far VideoToolbox may move a picture from where it was handed in,
/// which the decode timestamps [`Dating`] writes allow for. Its H.264 and
/// HEVC sessions were seen to move a picture two places at most, whatever
/// the B-frames allowed; room is left above that.
const REORDER: usize = 4;

/// Decode timestamps for packets that come out reordered. FFmpeg's
/// VideoToolbox encoders date each packet one picture behind its
/// presentation for H.264 — two for HEVC — but VideoToolbox's H.264 reorders
/// two deep even where one B-frame is allowed, and a packet then decodes
/// after it is shown (`pts < dts`), which a muxer refuses.
///
/// So the `n`th packet is dated the `n - REORDER`th earliest presentation
/// time of those come out so far — at or before its own wherever no picture
/// moves more than [`REORDER`] places, and later than the one before it —
/// and the first [`REORDER`] packets a picture apart before the first's.
#[derive(Debug, Default)]
struct Dating {
    /// The presentation times come out and not yet given as a decode time.
    waiting: BinaryHeap<Reverse<i64>>,
    /// How many packets have been dated.
    dated: usize,
    /// The first packet's presentation time: a key picture, shown first.
    first: Option<i64>,
}

impl Dating {
    /// The decode time of the next packet, presented at `pts`, a picture
    /// `duration` long.
    fn next(&mut self, pts: i64, duration: i64) -> i64 {
        self.waiting.push(Reverse(pts));
        let first = *self.first.get_or_insert(pts);
        let n = self.dated;
        self.dated += 1;
        if n < REORDER {
            first - (REORDER - n) as i64 * duration.max(1)
        } else {
            let Reverse(earliest) = self.waiting.pop().expect("REORDER + 1 waiting");
            earliest
        }
    }
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
    ///
    /// NV12 and P010 only: BGRA is converted by VideoToolbox with BT.709 at
    /// limited range whatever the stream is told, so [`Self::new`] says that
    /// of it and this refuses it with
    /// [`VideoToolboxEncoderError::ColorOfBgra`].
    ///
    /// An HDR stream — HLG or PQ in P010 — takes pictures whose pixel
    /// buffers say the same of themselves, as a
    /// [`VideoToolboxDecoder`](crate::elements::VideoToolboxDecoder)'s and an
    /// upload of a frame so tagged do: VideoToolbox fails a picture that
    /// says nothing of its colour under an HLG stream.
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
        // Measured: VideoToolbox makes YUV of RGB with BT.709's matrix at
        // limited range, and goes on doing so whatever matrix or range the
        // session is told — only the headers follow what it is told.
        let color = match (options.format, color) {
            (VideoToolboxFrameFormat::P010, _) if options.codec == VideoToolboxCodec::H264 => {
                return Err(VideoToolboxEncoderError::TenBitsNeedHevc);
            }
            (VideoToolboxFrameFormat::Nv12 | VideoToolboxFrameFormat::P010, color) => color,
            (VideoToolboxFrameFormat::Bgra, None) => Some(ColorDescription::BT709_LIMITED),
            (VideoToolboxFrameFormat::Bgra, Some(_)) => {
                return Err(VideoToolboxEncoderError::ColorOfBgra);
            }
        };

        let encoder_name = options.codec.encoder_name();
        let codec = ffmpeg::encoder::find_by_name(encoder_name)
            .ok_or(VideoToolboxEncoderError::CodecNotFound(encoder_name))?;

        // The encoder reads the layout of the pixel buffers it is handed from
        // the frames context it is opened with, and the pixel buffers
        // themselves from each frame.
        // SAFETY: `create_frames_ctx`'s contract is a live device context,
        // which the device's own reference is for the length of the call.
        let hw_frames_ctx = unsafe {
            create_frames_ctx(
                &device.retain(),
                options.format.pixel(),
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
                // Ten bits are HEVC Main 10, which the encoder does not pick
                // for itself from the frames it is told of.
                if options.format == VideoToolboxFrameFormat::P010 {
                    (*ptr).profile = ffmpeg::ffi::AV_PROFILE_HEVC_MAIN_10;
                }
                if let Some(color) = color {
                    color.tell(ptr);
                }
            }
            video.open_as(codec)
        })();
        let encoder = opened?;

        pp_info!(
            pp_log: &pp_log,
            "opened: {} {}x{} from {:?}, {} bps, gop={}",
            encoder_name,
            options.width,
            options.height,
            options.format,
            options.bit_rate,
            options.gop_size
        );
        Ok(Self(FilterStage::new(Encoding {
            name,
            pp_log,
            encoder,
            _hw_frames_ctx: hw_frames_ctx,
            format: options.format,
            width: options.width,
            height: options.height,
            packet_duration: nominal_packet_duration(shared::TIME_BASE, options.frame_rate),
            dating: options
                .max_b_frames
                .is_some_and(|frames| frames > 0)
                .then(Dating::default),
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
            Ok(got) if got == self.format.pixel() => {}
            Ok(got) => {
                return Err(self.refused(VideoToolboxEncoderError::UnsupportedLayout {
                    expected: self.format.pixel(),
                    got,
                }));
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
                    if let (Some(dating), Some(pts)) = (&mut self.dating, packet.pts()) {
                        let dts = dating.next(pts, self.packet_duration);
                        if dts > pts {
                            pp_warn!(
                                self,
                                "a picture moved more than {REORDER} places: dated {dts} after its {pts}"
                            );
                        }
                        packet.set_dts(Some(dts));
                    }
                    out.push(MediaBuffer::Packet(Arc::new(packet).into()));
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

impl Filter for Encoding {
    fn output_contract(&self) -> OutputContract {
        OutputContract::Fixed(PortContract::packet(MediaKind::VideoPacket))
    }

    /// VideoToolbox reads pixel buffers; a system-memory frame needs a
    /// VideoToolboxUpload first.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::VideoToolbox)
                .with_layouts(self.format.layouts()),
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
    use crate::element::{RawSink, SrcPads};
    use crate::{
        elements::{SwDecoder, VideoToolboxDecoder, VideoToolboxUpload},
        test_support::{CapturingSink, try_videotoolbox_device},
    };

    fn options(codec: VideoToolboxCodec, width: u32, height: u32) -> VideoToolboxEncoderOptions {
        VideoToolboxEncoderOptions {
            codec,
            format: VideoToolboxFrameFormat::Nv12,
            width,
            height,
            frame_rate: ffmpeg::Rational::new(30, 1),
            bit_rate: 4_000_000,
            gop_size: 30,
            max_b_frames: None,
        }
    }

    type Received = Arc<Mutex<Vec<MediaBuffer>>>;

    fn capture(source: &mut dyn SrcPads) -> Received {
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
                MediaBuffer::Video(frame) => Some(frame.payload().clone()),
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

    /// BGRA is made YUV on the media engine with BT.709 at limited range,
    /// and the stream says so: red decodes to BT.709's red, tagged BT.709.
    /// Describing it as anything else is refused before an encoder opens.
    #[test]
    fn bgra_encodes_as_bt709_at_limited_range() {
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let _session = crate::test_support::encoder_session();
        let (width, height) = (320u32, 240u32);
        let bgra = VideoToolboxEncoderOptions {
            format: VideoToolboxFrameFormat::Bgra,
            ..options(VideoToolboxCodec::H264, width, height)
        };
        assert!(matches!(
            VideoToolboxEncoder::with_color(
                "encoder",
                &device,
                bgra,
                ColorDescription::BT709_LIMITED
            ),
            Err(VideoToolboxEncoderError::ColorOfBgra)
        ));
        let mut encoder = match VideoToolboxEncoder::new("encoder", &device, bgra) {
            Ok(encoder) => encoder,
            Err(error) => {
                eprintln!("skipping: no VideoToolbox H.264 encoder here ({error})");
                return;
            }
        };
        let packets = capture(&mut encoder);
        let mut upload = VideoToolboxUpload::new("upload", &device);
        let uploaded = capture(&mut upload);
        let frames = 10;
        for index in 0..frames {
            let mut red = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, width, height);
            for pixel in red.data_mut(0).chunks_mut(4) {
                pixel.copy_from_slice(&[0, 0, 255, 255]);
            }
            red.set_pts(Some(index));
            crate::buffer::set_time_base(&mut red, ffmpeg::Rational::new(1, 30));
            upload.consume(MediaBuffer::video(red)).unwrap();
            let frame = uploaded.lock().unwrap().remove(0);
            encoder.consume(frame).expect("BGRA encodes");
        }
        crate::stream::deliver(&mut encoder, &crate::stream::StreamEvent::Eos).expect("eos");

        let packets = std::mem::take(&mut *packets.lock().unwrap());
        let decoded = decode(encoder.parameters(), &packets);
        assert_eq!(decoded.len(), frames as usize, "every frame came back");
        let last = decoded.last().unwrap();
        assert_eq!(last.color_space(), ffmpeg::color::Space::BT709);
        assert_eq!(last.color_range(), ffmpeg::color::Range::MPEG);
        // BT.709 red at limited range: 16 + 219 * 0.2126, and Cr at its top.
        let (y, u, v) = (last.data(0)[0], last.data(1)[0], last.data(2)[0]);
        assert!(
            y.abs_diff(63) <= 3 && u.abs_diff(102) <= 3 && v.abs_diff(240) <= 3,
            "{y} {u} {v}"
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
                .consume(MediaBuffer::Packet(Arc::new(packet).into()))
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

    /// P010 encodes as HEVC Main 10, its colour as the stream is told —
    /// HLG in BT.2020 — and decodes back to ten bits of what was sent; H.264
    /// is refused it before an encoder opens.
    #[test]
    fn p010_encodes_as_hevc_main_10() {
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let _session = crate::test_support::encoder_session();
        let (width, height) = (320u32, 240u32);
        let ten_bits = VideoToolboxEncoderOptions {
            format: VideoToolboxFrameFormat::P010,
            ..options(VideoToolboxCodec::H264, width, height)
        };
        assert!(matches!(
            VideoToolboxEncoder::new("encoder", &device, ten_bits),
            Err(VideoToolboxEncoderError::TenBitsNeedHevc)
        ));
        let hlg = ColorDescription {
            space: ffmpeg::color::Space::BT2020NCL,
            range: ffmpeg::color::Range::MPEG,
            primaries: ffmpeg::color::Primaries::BT2020,
            transfer: ffmpeg::color::TransferCharacteristic::ARIB_STD_B67,
        };
        let mut encoder = match VideoToolboxEncoder::with_color(
            "encoder",
            &device,
            VideoToolboxEncoderOptions {
                codec: VideoToolboxCodec::H265,
                ..ten_bits
            },
            hlg,
        ) {
            Ok(encoder) => encoder,
            Err(error) => {
                eprintln!("skipping: no HEVC Main 10 encoder here ({error})");
                return;
            }
        };
        // A ramp of ten-bit luma, in each sample's top ten bits.
        let level = |x: usize, index: i64| (64 + (x as i64 * 3 + index * 8) % 800) as u16;
        let picture = |index: i64| {
            let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::P010LE, width, height);
            let stride = frame.stride(0);
            for y in 0..height as usize {
                for x in 0..width as usize {
                    let at = y * stride + x * 2;
                    frame.data_mut(0)[at..at + 2]
                        .copy_from_slice(&(level(x, index) << 6).to_le_bytes());
                }
            }
            for pair in frame.data_mut(1).as_chunks_mut::<2>().0 {
                pair.copy_from_slice(&(512u16 << 6).to_le_bytes());
            }
            frame.set_pts(Some(index));
            crate::buffer::set_time_base(&mut frame, ffmpeg::Rational::new(1, 30));
            // Uploaded, its pixel buffer says so, as a decoder's does.
            hlg.describe(&mut frame);
            frame
        };
        let packets = capture(&mut encoder);
        let mut upload = VideoToolboxUpload::new("upload", &device);
        let uploaded = capture(&mut upload);
        let frames = 10;
        for index in 0..frames {
            upload.consume(MediaBuffer::video(picture(index))).unwrap();
            let frame = uploaded.lock().unwrap().remove(0);
            encoder.consume(frame).expect("encode");
        }
        crate::stream::deliver(&mut encoder, &crate::stream::StreamEvent::Eos).expect("eos");

        let parameters = encoder.parameters();
        // SAFETY: a live parameters struct this test owns.
        let (profile, transfer, primaries) = unsafe {
            let raw = parameters.as_ptr();
            ((*raw).profile, (*raw).color_trc, (*raw).color_primaries)
        };
        assert_eq!(parameters.id(), ffmpeg::codec::Id::HEVC);
        assert_eq!(profile, 2, "HEVC Main 10");
        assert_eq!(
            transfer,
            ffmpeg::ffi::AVColorTransferCharacteristic::AVCOL_TRC_ARIB_STD_B67
        );
        assert_eq!(primaries, ffmpeg::ffi::AVColorPrimaries::AVCOL_PRI_BT2020);
        let packets = std::mem::take(&mut *packets.lock().unwrap());
        let decoded = decode(parameters, &packets);
        assert_eq!(decoded.len(), frames as usize, "every frame came back");
        let last = decoded.last().unwrap();
        assert_eq!(last.format(), ffmpeg::format::Pixel::YUV420P10LE);
        let mean = (0..height as usize)
            .flat_map(|y| (0..width as usize).map(move |x| (x, y)))
            .map(|(x, y)| {
                let at = y * last.stride(0) + x * 2;
                let got = u16::from_le_bytes([last.data(0)[at], last.data(0)[at + 1]]);
                f64::from(got.abs_diff(level(x, frames - 1)))
            })
            .sum::<f64>()
            / f64::from(width * height);
        assert!(
            mean < 8.0,
            "the last picture decodes to its ten bits: {mean} of 1023 off"
        );
    }

    /// A picture an element made in a pool of its own says its colour in
    /// its frame's fields alone, which an HLG encoder fails; described, its
    /// pixel buffer says so too and it encodes. A picture of another pool's
    /// is not described.
    #[test]
    fn a_pool_picture_described_encodes_under_hlg() {
        use crate::elements::{VideoToolboxFramePool, VideoToolboxFramePoolError};
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let _session = crate::test_support::encoder_session();
        let (width, height) = (320u32, 240u32);
        let hlg = ColorDescription {
            space: ffmpeg::color::Space::BT2020NCL,
            range: ffmpeg::color::Range::MPEG,
            primaries: ffmpeg::color::Primaries::BT2020,
            transfer: ffmpeg::color::TransferCharacteristic::ARIB_STD_B67,
        };
        let options = VideoToolboxEncoderOptions {
            format: VideoToolboxFrameFormat::P010,
            ..options(VideoToolboxCodec::H265, width, height)
        };
        let pool =
            VideoToolboxFramePool::new(&device, VideoToolboxFrameFormat::P010, width, height)
                .expect("pool");
        let picture = |index: i64, described: bool| {
            let mut picture = pool.get().expect("a picture");
            picture.set_pts(Some(index));
            crate::buffer::set_time_base(&mut picture, ffmpeg::Rational::new(1, 30));
            hlg.describe(&mut picture);
            if described {
                pool.describe(&picture).expect("described");
            }
            MediaBuffer::Video(Arc::new(picture).into())
        };
        let encoder =
            |name: &str| match VideoToolboxEncoder::with_color(name, &device, options, hlg) {
                Ok(encoder) => Some(encoder),
                Err(error) => {
                    eprintln!("skipping: no HEVC Main 10 encoder here ({error})");
                    None
                }
            };
        let Some(mut bare) = encoder("bare") else {
            return;
        };
        assert!(
            bare.consume(picture(0, false)).is_err(),
            "a pixel buffer that says nothing is refused under HLG"
        );
        let Some(mut said) = encoder("said") else {
            return;
        };
        let packets = capture(&mut said);
        for index in 0..5 {
            said.consume(picture(index, true)).expect("encodes");
        }
        crate::stream::deliver(&mut said, &crate::stream::StreamEvent::Eos).expect("eos");
        assert!(!packets.lock().unwrap().is_empty(), "packets came out");

        let other =
            VideoToolboxFramePool::new(&device, VideoToolboxFrameFormat::P010, width, height)
                .expect("pool");
        let foreign = other.get().expect("a picture");
        assert!(matches!(
            pool.describe(&foreign),
            Err(VideoToolboxFramePoolError::ForeignPicture)
        ));
    }

    /// With B-frames allowed, the packets go into an MP4, which refuses one
    /// decoded after it is shown, and every picture comes back out of it in
    /// order — H.264 and HEVC alike, though VideoToolbox reorders H.264
    /// deeper than FFmpeg dates it.
    #[test]
    fn b_frames_are_dated_as_a_muxer_takes_them() {
        use crate::elements::FileMuxer;
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let _session = crate::test_support::encoder_session();
        let (width, height) = (320u32, 240u32);
        for codec in [VideoToolboxCodec::H264, VideoToolboxCodec::H265] {
            for b_frames in [1, 2] {
                let Ok(mut encoder) = VideoToolboxEncoder::new(
                    "encoder",
                    &device,
                    VideoToolboxEncoderOptions {
                        max_b_frames: Some(b_frames),
                        ..options(codec, width, height)
                    },
                ) else {
                    eprintln!("skipping: no {codec:?} encoder here");
                    continue;
                };
                let path = std::env::temp_dir()
                    .join(format!("media-pp-vt-b-frames-{codec:?}-{b_frames}.mp4"));
                let mut muxer = FileMuxer::create(&path).expect("create");
                let track = muxer.add_stream("video", &encoder).expect("add");
                let sink = muxer.open().expect("header").take(track).expect("track");
                encoder.src_pads()[0].link(sink.into_raw());
                let mut upload = VideoToolboxUpload::new("upload", &device);
                let uploaded = capture(&mut upload);
                let frames = 30;
                for index in 0..frames {
                    upload
                        .consume(MediaBuffer::video(ramp(width, height, index)))
                        .unwrap();
                    let frame = uploaded.lock().unwrap().remove(0);
                    encoder.consume(frame).expect("encoded and written");
                }
                crate::stream::deliver(&mut encoder, &crate::stream::StreamEvent::Eos)
                    .expect("the file finished");
                drop(encoder);

                let mut input = ffmpeg::format::input(&path).expect("reopen");
                let parameters = input
                    .streams()
                    .best(ffmpeg::media::Type::Video)
                    .expect("video")
                    .parameters();
                let mut decoder = ffmpeg::codec::context::Context::from_parameters(parameters)
                    .and_then(|context| context.decoder().video())
                    .expect("a decoder");
                let (mut times, mut shown) = (Vec::new(), Vec::new());
                let mut picture = ffmpeg::frame::Video::empty();
                for (_, packet) in input.packets() {
                    times.push((packet.pts(), packet.dts()));
                    decoder.send_packet(&packet).expect("decodes");
                    while decoder.receive_frame(&mut picture).is_ok() {
                        shown.push(picture.pts());
                    }
                }
                decoder.send_eof().expect("drains");
                while decoder.receive_frame(&mut picture).is_ok() {
                    shown.push(picture.pts());
                }
                assert!(
                    times.iter().any(|(pts, dts)| pts != dts),
                    "{codec:?} b={b_frames}: reordered at all: {times:?}"
                );
                assert_eq!(
                    shown.len(),
                    frames as usize,
                    "{codec:?} b={b_frames}: {shown:?}"
                );
                assert!(
                    shown.windows(2).all(|pair| pair[0] < pair[1]),
                    "{codec:?} b={b_frames}: {shown:?}"
                );
                let _ = std::fs::remove_file(&path);
            }
        }
    }
}
