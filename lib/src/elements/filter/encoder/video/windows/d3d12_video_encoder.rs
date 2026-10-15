//! [`D3d12VideoEncoder`]: D3D12 pictures encoded on any vendor's encode
//! block through D3D12 Video.

use std::sync::Arc;

use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;
use windows::{
    Win32::Graphics::{
        Direct3D12::{ID3D12Device, ID3D12Resource},
        Dxgi::Common::DXGI_FORMAT_NV12,
    },
    core::Interface,
};

use crate::pp_log::{PpLog, pp_error, pp_info};
use crate::{
    buffer::MediaBuffer,
    color::ColorDescription,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    element::{Element, ElementType, Filter, Output, element_pp_log},
    elements::filter::is_codec_drain_boundary,
    error::Result,
    platform::{
        ffmpeg::AvBufferRef,
        windows::d3d12_gpu::D3d12Gpu,
        windows::d3d12va::{create_hw_device_ctx, create_hw_frames_ctx_of, d3d12va_texture},
    },
    transform::{FilterStage, filter_stage},
};

// The video-encoder helpers every backend shares, a module up from this one.
use super::super as shared;

/// Which of FFmpeg's D3D12 Video encoders [`D3d12VideoEncoder`] drives.
///
/// Each has to be in the linked FFmpeg, and the GPU's driver has to encode
/// the codec through D3D12 Video; where either is missing,
/// [`D3d12VideoEncoder::new`] fails with a typed error. FFmpeg 8.0 has
/// `hevc_d3d12va` alone: `h264_d3d12va` came after it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum D3d12VideoCodec {
    /// `h264_d3d12va` — H.264/AVC.
    H264,
    /// `hevc_d3d12va` — H.265/HEVC.
    H265,
}

impl D3d12VideoCodec {
    fn encoder_name(self) -> &'static str {
        match self {
            Self::H264 => "h264_d3d12va",
            Self::H265 => "hevc_d3d12va",
        }
    }
}

/// Construction-time options for [`D3d12VideoEncoder`] — the same knobs,
/// with the same meanings, as `VulkanEncoderOptions`.
#[derive(Debug, Clone, Copy)]
pub struct D3d12VideoEncoderOptions {
    /// The bitstream to encode.
    pub codec: D3d12VideoCodec,
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
    /// How many consecutive B-frames the encoder may insert — see
    /// [`crate::elements::SwEncoderOptions::max_b_frames`] — and `None` for
    /// none. Not the encoder's own default, as elsewhere: FFmpeg's D3D12
    /// encoder inserts B-frames by default, and NVIDIA's D3D12 encode
    /// fails at the first frame given them (measured on an RTX 3050, with
    /// FFmpeg's own command line as with this element).
    pub max_b_frames: Option<u32>,
}

/// Errors specific to [`D3d12VideoEncoder`]. Converts into the crate-wide
/// `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum D3d12VideoEncoderError {
    /// The encoder is not in this FFmpeg build.
    #[error("encoder `{0}` not found in this ffmpeg build")]
    CodecNotFound(&'static str),
    /// FFmpeg could not take the device.
    #[error("failed to create the D3D12VA device context (code {0})")]
    HwDeviceInit(i32),
    /// FFmpeg could not make the frames context the encoder is opened with.
    #[error("failed to create the D3D12VA frames context (code {0})")]
    HwFrames(i32),
    /// FFmpeg rejected the encoder or a frame — among other reasons, a GPU
    /// that does not encode this codec through D3D12 Video fails to open.
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),
    /// The sink received a buffer other than decoded video.
    #[error("D3d12VideoEncoder only accepts Video buffers, got a {0}")]
    UnsupportedBuffer(&'static str),
    /// The frame is not a D3D12 one, or holds a layout other than NV12.
    #[error("D3d12VideoEncoder encodes NV12 D3D12 frames, got {0}")]
    UnsupportedPicture(String),
    /// The frame was made on another device than the encoder's.
    #[error("D3d12VideoEncoder was handed a frame of another D3D12 device")]
    DeviceMismatch,
    /// Input dimensions differ from the fixed encoder dimensions.
    #[error(
        "frame is {actual_width}x{actual_height}, but this D3d12VideoEncoder was built for \
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
    /// A frame arrived with a `pts` but no unit to read it in — see
    /// [`crate::elements::SwEncoderError::NoTimeBase`].
    #[error(
        "a Video frame arrived with a pts but no time base to read it in; the element \
         that made it has to set one (media_pp::buffer::set_time_base)"
    )]
    NoTimeBase,
    /// D3D12 failed asking a frame's texture about itself.
    #[error(transparent)]
    D3d12(#[from] windows::core::Error),
}

/// Encodes NV12 D3D12 frames into `Packet`s with D3D12 Video, on the
/// [`D3d12Gpu`] the frames are made on — on any vendor's GPU whose driver
/// encodes through D3D12. The D3D12 counterpart of `VulkanEncoder`, and a
/// a filter as it is.
///
/// Fed by [`crate::elements::D3d12Decoder`] this is a transcode that never
/// brings a pixel to the CPU; fed by a
/// [`D3d12DetectionOverlay`](crate::elements::D3d12DetectionOverlay) or a
/// [`D3d12ToneMap`](crate::elements::D3d12ToneMap) it records what they
/// make; fed by [`crate::elements::D3d12Upload`] it replaces a software
/// encoder.
///
/// Each frame is handed to the encoder as it is — a decoder's, an
/// upload's, an overlay's, each from a pool of its own, which FFmpeg's D3D12
/// encoder takes on the same device, waiting on the fence that says the
/// frame is written.
///
/// # Packet timing
///
/// As `VulkanEncoder`'s: packets are drained after every frame and at
/// `Eos`, and each is stamped with this encoder's [`Self::time_base`] and a
/// nominal duration.
pub struct D3d12VideoEncoder(FilterStage<Encoding>);

filter_stage!(D3d12VideoEncoder);

/// What a [`D3d12VideoEncoder`] does to each buffer: all of its work, which
/// the framework makes the filter.
struct Encoding {
    pp_log: PpLog,
    name: Arc<str>,
    encoder: ffmpeg::encoder::Video,
    _hw_device_ctx: AvBufferRef,
    _hw_frames_ctx: AvBufferRef,
    /// The GPU's device, to compare an incoming frame's against.
    device: ID3D12Device,
    width: u32,
    height: u32,
    /// Nominal frame duration in `time_base` ticks, which the encoder
    /// leaves at zero.
    packet_duration: i64,
}

// SAFETY: both buffers are heap-allocated FFmpeg buffers with no thread
// affinity, the device is free-threaded and only compared, and `encoder`'s
// own `Send` covers the codec context. `&mut self` on every method that
// touches them rules out concurrent access — as `VulkanEncoder`.
unsafe impl Send for Encoding {}

fn nominal_packet_duration(time_base: ffmpeg::Rational, frame_rate: ffmpeg::Rational) -> i64 {
    if frame_rate.numerator() <= 0 || time_base.numerator() <= 0 {
        return 0;
    }
    let ticks = f64::from(time_base.denominator()) * f64::from(frame_rate.denominator())
        / (f64::from(time_base.numerator()) * f64::from(frame_rate.numerator()));
    ticks.round() as i64
}

impl D3d12VideoEncoder {
    /// `gpu` must be the [`D3d12Gpu`] the frames are made on — a frame of
    /// another device is refused.
    ///
    /// Opens the encoder eagerly, so a missing encoder, a GPU that does not
    /// encode this codec through D3D12 Video, or a size it refuses all
    /// surface here as a typed error rather than at the first frame.
    pub fn new(
        name: impl Into<String>,
        gpu: &D3d12Gpu,
        options: D3d12VideoEncoderOptions,
    ) -> std::result::Result<Self, D3d12VideoEncoderError> {
        crate::ensure_ffmpeg();
        Self::open(name, gpu, options, None)
    }

    /// As [`Self::new`], and the stream says `color` is what it holds —
    /// written into its headers, where every player finds it; nothing is
    /// converted. A [`D3d12ToneMap`](crate::elements::D3d12ToneMap)'s output
    /// is [`ColorDescription::BT709_LIMITED`].
    pub fn with_color(
        name: impl Into<String>,
        gpu: &D3d12Gpu,
        options: D3d12VideoEncoderOptions,
        color: ColorDescription,
    ) -> std::result::Result<Self, D3d12VideoEncoderError> {
        crate::ensure_ffmpeg();
        Self::open(name, gpu, options, Some(color))
    }

    fn open(
        name: impl Into<String>,
        gpu: &D3d12Gpu,
        options: D3d12VideoEncoderOptions,
        color: Option<ColorDescription>,
    ) -> std::result::Result<Self, D3d12VideoEncoderError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::D3d12VideoEncoder, &name, None);
        let encoder_name = options.codec.encoder_name();
        let codec = ffmpeg::encoder::find_by_name(encoder_name)
            .ok_or(D3d12VideoEncoderError::CodecNotFound(encoder_name))?;

        // SAFETY: a live device, whose context owns its own reference.
        let hw_device_ctx = unsafe { create_hw_device_ctx(gpu.device()) }
            .map_err(D3d12VideoEncoderError::HwDeviceInit)?;
        // SAFETY: the device context just made, live.
        let hw_frames_ctx = unsafe {
            create_hw_frames_ctx_of(
                &hw_device_ctx,
                ffi::AVPixelFormat::AV_PIX_FMT_NV12,
                options.width,
                options.height,
                0,
            )
        }
        .map_err(D3d12VideoEncoderError::HwFrames)?;
        let codec_device_ctx = hw_device_ctx
            .try_clone()
            .ok_or(D3d12VideoEncoderError::HwDeviceInit(-1))?;
        let codec_frames_ctx = hw_frames_ctx
            .try_clone()
            .ok_or(D3d12VideoEncoderError::HwFrames(-1))?;

        let opened = (|| -> std::result::Result<ffmpeg::encoder::Video, ffmpeg::Error> {
            let mut ctx = ffmpeg::codec::context::Context::new_with_codec(codec);
            // Codec headers into `extradata` for the container to write, not
            // only in-band — see `SwEncoder::new`.
            ctx.set_flags(ffmpeg::codec::Flags::GLOBAL_HEADER);
            let mut video = ctx.encoder().video()?;
            video.set_width(options.width);
            video.set_height(options.height);
            video.set_format(ffmpeg::format::Pixel::D3D12);
            video.set_time_base(shared::TIME_BASE);
            video.set_frame_rate(Some(options.frame_rate));
            video.set_bit_rate(options.bit_rate);
            video.set_gop(options.gop_size);
            video.set_max_b_frames(options.max_b_frames.unwrap_or(0) as usize);
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

        pp_info!(
            pp_log: &pp_log,
            "opened: {} {}x{}, {} bps, gop={}",
            encoder_name,
            options.width,
            options.height,
            options.bit_rate,
            options.gop_size
        );
        Ok(Self(FilterStage::new(Encoding {
            name,
            pp_log,
            encoder,
            _hw_device_ctx: hw_device_ctx,
            _hw_frames_ctx: hw_frames_ctx,
            device: gpu.device().clone(),
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

    /// Why `frame` is not one this encoder takes, or `None` where it is.
    fn check(&self, frame: &ffmpeg::frame::Video) -> Option<D3d12VideoEncoderError> {
        let Some((texture, _, _)) =
            d3d12va_texture(frame).filter(|(texture, _, _)| !texture.is_null())
        else {
            return Some(D3d12VideoEncoderError::UnsupportedPicture(format!(
                "{:?}",
                frame.format()
            )));
        };
        // SAFETY: the live frame owns this non-null texture; the borrowed
        // wrapper is only asked about it.
        let texture = unsafe { ID3D12Resource::from_raw_borrowed(&texture) }?;
        let mut made_on: Option<ID3D12Device> = None;
        // SAFETY: a live resource and a correctly typed out-parameter.
        if let Err(error) = unsafe { texture.GetDevice(&mut made_on) } {
            return Some(error.into());
        }
        if made_on.is_none_or(|made_on| made_on.as_raw() != self.device.as_raw()) {
            return Some(D3d12VideoEncoderError::DeviceMismatch);
        }
        // SAFETY: a live resource's plain description, returned by value.
        let format = unsafe { texture.GetDesc() }.Format;
        if format != DXGI_FORMAT_NV12 {
            return Some(D3d12VideoEncoderError::UnsupportedPicture(format!(
                "a texture of DXGI format {}",
                format.0
            )));
        }
        if frame.width() != self.width || frame.height() != self.height {
            return Some(D3d12VideoEncoderError::DimensionMismatch {
                actual_width: frame.width(),
                actual_height: frame.height(),
                expected_width: self.width,
                expected_height: self.height,
            });
        }
        None
    }

    fn encode(&mut self, frame: &ffmpeg::frame::Video, out: &mut Output) -> Result<()> {
        if let Some(error) = self.check(frame) {
            pp_error!(self, "{error}");
            return Err(error.into());
        }
        let pts = shared::pts_in(frame, shared::TIME_BASE)
            .map_err(|shared::NoTimeBase| D3d12VideoEncoderError::NoTimeBase)?;
        let frame = shared::restamped(frame, pts, shared::TIME_BASE)
            .map_err(D3d12VideoEncoderError::from)?;
        self.encoder
            .send_frame(&frame)
            .inspect_err(|error| pp_error!(self, "send_frame failed: {error}"))
            .map_err(D3d12VideoEncoderError::from)?;
        self.receive_packets(out)
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
                    out.push(MediaBuffer::Packet(Arc::new(packet).into()));
                    packet = ffmpeg::Packet::empty();
                }
                Err(error) if is_codec_drain_boundary(&error) => break,
                Err(error) => return Err(D3d12VideoEncoderError::from(error).into()),
            }
        }
        Ok(())
    }
}

/// The track this encoder's packets make: its
/// [`D3d12VideoEncoder::parameters`], timed in its
/// [`D3d12VideoEncoder::time_base`]. FFmpeg 8.0's `hevc_d3d12va` hands its
/// parameter sets only in band, but they never change, so an MP4 still
/// writes the HEVC as `hvc1`.
impl From<&D3d12VideoEncoder> for crate::elements::TrackFormat {
    fn from(encoder: &D3d12VideoEncoder) -> Self {
        Self::new(encoder.parameters(), encoder.time_base()).with_fixed_parameter_sets()
    }
}

impl Element for Encoding {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::D3d12VideoEncoder
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

    /// D3D12 Video reads the texture; a system-memory frame needs a
    /// D3d12Upload first.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::D3d12)
                .with_layouts(crate::contract::PixelLayoutSet::NV12),
        )
    }

    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        match buf {
            MediaBuffer::Video(frame) => self.encode(&frame, out),
            other => Err(D3d12VideoEncoderError::UnsupportedBuffer(other.kind()).into()),
        }
    }

    /// Sends the end of the stream in and hands on what comes out.
    fn drain(&mut self, out: &mut Output) -> Result<()> {
        self.encoder
            .send_eof()
            .inspect_err(|error| pp_error!(self, "send_eof failed: {error}"))
            .map_err(D3d12VideoEncoderError::from)?;
        self.receive_packets(out)
    }

    fn reset(&mut self) {
        // Nothing to reset, as for every encoder here: a caller needing a
        // hard encoded-stream discontinuity rebuilds the encoder.
    }
}

impl Drop for Encoding {
    fn drop(&mut self) {
        pp_info!(self, "dropped: releasing hw contexts");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{RawSink, SrcPads};
    use crate::elements::{D3d12Upload, SwDecoder};
    use crate::test_support::{capture, try_d3d12_gpu};

    const SIZE: (u32, u32) = (320, 240);

    fn options(codec: D3d12VideoCodec) -> D3d12VideoEncoderOptions {
        D3d12VideoEncoderOptions {
            codec,
            width: SIZE.0,
            height: SIZE.1,
            frame_rate: ffmpeg::Rational::new(30, 1),
            bit_rate: 2_000_000,
            gop_size: 15,
            max_b_frames: None,
        }
    }

    /// The encoder for `codec` on `gpu`, or `None`, saying why, where this
    /// FFmpeg or this GPU does not encode it through D3D12.
    fn opened(gpu: &D3d12Gpu, codec: D3d12VideoCodec) -> Option<D3d12VideoEncoder> {
        match D3d12VideoEncoder::new("encoder", gpu, options(codec)) {
            Ok(encoder) => Some(encoder),
            Err(error) => {
                eprintln!("skipping {codec:?}: {error}");
                None
            }
        }
    }

    /// Picture `index` of a moving pattern, NV12, its time in 1/30 s.
    fn picture(index: i64) -> ffmpeg::frame::Video {
        let (width, height) = (SIZE.0 as usize, SIZE.1 as usize);
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, SIZE.0, SIZE.1);
        let stride = frame.stride(0);
        for y in 0..height {
            for x in 0..width {
                frame.data_mut(0)[y * stride + x] = (32 + (x + y + 4 * index as usize) % 192) as u8;
            }
        }
        let stride = frame.stride(1);
        for y in 0..height / 2 {
            for x in 0..width / 2 {
                frame.data_mut(1)[y * stride + 2 * x] = (96 + x % 64) as u8;
                frame.data_mut(1)[y * stride + 2 * x + 1] = (160 - y % 64) as u8;
            }
        }
        frame.set_pts(Some(index));
        crate::buffer::set_time_base(&mut frame, ffmpeg::Rational::new(1, 30));
        frame
    }

    /// Thirty uploaded pictures — from the upload's own frames, not the
    /// encoder's — come out as thirty packets, the first a keyframe, which a
    /// software decoder turns back into thirty pictures close to what went
    /// in, in order.
    #[test]
    fn uploaded_pictures_are_encoded_and_decode_back() {
        let _session = crate::test_support::encoder_session();
        let Some(gpu) = try_d3d12_gpu() else {
            return;
        };
        let Some(mut encoder) = opened(&gpu, D3d12VideoCodec::H265) else {
            return;
        };
        let parameters = encoder.parameters();
        assert_eq!(parameters.id(), ffmpeg::codec::Id::HEVC);
        let packets = capture(&mut encoder);
        let Ok(mut upload) = D3d12Upload::new("upload", &gpu) else {
            eprintln!("skipping: FFmpeg could not create a D3D12VA frames context");
            return;
        };
        let uploaded = capture(&mut upload);
        for index in 0..30 {
            upload
                .consume(MediaBuffer::video(picture(index)))
                .expect("uploads");
            let buf = uploaded.lock().unwrap().remove(0);
            encoder.consume(buf).expect("encodes");
        }
        crate::stream::deliver(&mut encoder, &crate::stream::StreamEvent::Eos).expect("drains");
        let packets: Vec<MediaBuffer> = std::mem::take(&mut *packets.lock().unwrap());
        assert_eq!(packets.len(), 30, "a packet a picture");
        let MediaBuffer::Packet(first) = &packets[0] else {
            panic!("packets");
        };
        assert!(first.is_key(), "the first is a keyframe");

        let mut decoder = SwDecoder::new("decoder", parameters).expect("opens");
        let decoded = capture(&mut decoder);
        for packet in packets {
            decoder.consume(packet).expect("decodes");
        }
        crate::stream::deliver(&mut decoder, &crate::stream::StreamEvent::Eos).expect("drains");
        let decoded = decoded.lock().unwrap();
        assert_eq!(decoded.len(), 30, "every picture decodes back");
        for (index, buf) in decoded.iter().enumerate() {
            let MediaBuffer::Video(frame) = buf else {
                panic!("pictures");
            };
            assert_eq!((frame.width(), frame.height()), SIZE);
            let sent = picture(index as i64);
            let (mut apart, mut count) = (0u64, 0u64);
            for y in 0..SIZE.1 as usize {
                for x in 0..SIZE.0 as usize {
                    let a = frame.data(0)[y * frame.stride(0) + x];
                    let b = sent.data(0)[y * sent.stride(0) + x];
                    apart += u64::from(a.abs_diff(b));
                    count += 1;
                }
            }
            let mean = apart as f64 / count as f64;
            assert!(mean < 6.0, "picture {index}: luma {mean} apart on average");
        }
    }

    /// HEVC from D3D12 goes into an MP4 as `hvc1`, which an iPhone plays,
    /// not FFmpeg's `hev1`, which it refuses — though `hevc_d3d12va` makes
    /// no extradata, the MP4 muxer takes the parameter sets from the first
    /// packet into the sample entry. The same packets described as a stream
    /// copied from elsewhere, whose sets could change, stay `hev1`.
    #[test]
    fn hevc_goes_into_an_mp4_as_hvc1() {
        use crate::elements::{FileMuxer, Mp4Layout, TrackFormat};
        let _session = crate::test_support::encoder_session();
        let Some(gpu) = try_d3d12_gpu() else {
            return;
        };
        for (fixed, tag) in [(true, *b"hvc1"), (false, *b"hev1")] {
            let Some(mut encoder) = opened(&gpu, D3d12VideoCodec::H265) else {
                return;
            };
            let Ok(mut upload) = D3d12Upload::new("upload", &gpu) else {
                eprintln!("skipping: FFmpeg could not create a D3D12VA frames context");
                return;
            };
            let path = std::env::temp_dir().join(format!(
                "media-pp-d3d12-hevc-tag-{}-{fixed}.mp4",
                std::process::id()
            ));
            let format = if fixed {
                TrackFormat::from(&encoder)
            } else {
                TrackFormat::new(encoder.parameters(), encoder.time_base())
            };
            let mut muxer = FileMuxer::create(&path, Mp4Layout::Standard).expect("create");
            let track = muxer.add_stream("video", format).expect("add");
            let sink = muxer.open().expect("header").take(track).expect("track");
            encoder.src_pads()[0].link(sink.into_raw());
            let uploaded = capture(&mut upload);
            for index in 0..10 {
                upload
                    .consume(MediaBuffer::video(picture(index)))
                    .expect("uploads");
                let buf = uploaded.lock().unwrap().remove(0);
                encoder.consume(buf).expect("encodes");
            }
            crate::stream::deliver(&mut encoder, &crate::stream::StreamEvent::Eos).expect("drains");
            drop(encoder);
            let mut input = ffmpeg::format::input(&path).expect("reopen");
            let stream = input
                .streams()
                .best(ffmpeg::media::Type::Video)
                .expect("video");
            let index = stream.index();
            // SAFETY: plain fields of the stream's own parameters.
            let (written, extradata) = unsafe {
                let raw = stream.parameters().as_ptr();
                ((*raw).codec_tag, (*raw).extradata_size)
            };
            let mut decoder = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
                .expect("context")
                .decoder()
                .video()
                .expect("decoder");
            let mut pictures = 0;
            let mut frame = ffmpeg::frame::Video::empty();
            for (stream, packet) in input.packets() {
                if stream.index() == index {
                    decoder.send_packet(&packet).expect("decodes");
                    while decoder.receive_frame(&mut frame).is_ok() {
                        pictures += 1;
                    }
                }
            }
            decoder.send_eof().expect("drains");
            while decoder.receive_frame(&mut frame).is_ok() {
                pictures += 1;
            }
            drop(input);
            let _ = std::fs::remove_file(&path);
            assert_eq!(written.to_le_bytes(), tag, "fixed={fixed}");
            assert!(
                extradata > 0,
                "fixed={fixed}: the sample entry holds the sets"
            );
            assert_eq!(pictures, 10, "fixed={fixed}: every picture decodes");
        }
    }
    /// A picture in system memory and one of another size are refused,
    /// saying so, and the encoder takes the next.
    #[test]
    fn what_it_cannot_encode_is_refused() {
        let _session = crate::test_support::encoder_session();
        let Some(gpu) = try_d3d12_gpu() else {
            return;
        };
        let Some(mut encoder) = opened(&gpu, D3d12VideoCodec::H265) else {
            return;
        };
        let refused = encoder.consume(MediaBuffer::video(picture(0)));
        assert!(
            matches!(
                refused,
                Err(crate::error::Error::D3d12VideoEncoderError(
                    D3d12VideoEncoderError::UnsupportedPicture(_)
                ))
            ),
            "{refused:?}"
        );
        let Ok(mut upload) = D3d12Upload::new("upload", &gpu) else {
            return;
        };
        let uploaded = capture(&mut upload);
        let mut small = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, 64, 48);
        small.data_mut(0).fill(100);
        small.data_mut(1).fill(128);
        upload.consume(MediaBuffer::video(small)).expect("uploads");
        let small = uploaded.lock().unwrap().remove(0);
        let refused = encoder.consume(small);
        assert!(
            matches!(
                refused,
                Err(crate::error::Error::D3d12VideoEncoderError(
                    D3d12VideoEncoderError::DimensionMismatch { .. }
                ))
            ),
            "{refused:?}"
        );
        upload
            .consume(MediaBuffer::video(picture(0)))
            .expect("uploads");
        let next = uploaded.lock().unwrap().remove(0);
        encoder.consume(next).expect("encodes the next");
    }

    /// H.264 opens where this FFmpeg has `h264_d3d12va`, and is refused as
    /// missing where it has not — never anything else.
    #[test]
    fn h264_opens_or_is_missing() {
        let _session = crate::test_support::encoder_session();
        let Some(gpu) = try_d3d12_gpu() else {
            return;
        };
        let has = ffmpeg::encoder::find_by_name("h264_d3d12va").is_some();
        match D3d12VideoEncoder::new("encoder", &gpu, options(D3d12VideoCodec::H264)) {
            Err(D3d12VideoEncoderError::CodecNotFound(name)) => {
                assert!(!has, "{name} is in this FFmpeg");
            }
            Ok(encoder) => {
                assert_eq!(encoder.parameters().id(), ffmpeg::codec::Id::H264);
            }
            Err(error) => eprintln!("skipping: this GPU does not encode H.264: {error}"),
        }
    }

    /// A 1080p stream decoded on the GPU — its surfaces taller than its
    /// pictures — is encoded picture for picture, and decodes back to what
    /// it held.
    #[test]
    fn decoded_pictures_are_encoded() {
        use crate::elements::{D3d12Decoder, D3d12DecoderError};
        let _session = crate::test_support::encoder_session();
        let Some(gpu) = try_d3d12_gpu() else {
            return;
        };
        let Some((parameters, packets)) = crate::test_support::try_encoded_packets(
            "libopenh264",
            ffmpeg::format::Pixel::YUV420P,
            (1920, 1080),
            100,
        ) else {
            return;
        };
        let mut decoder = match D3d12Decoder::new("decoder", parameters, &gpu) {
            Ok(decoder) => decoder,
            Err(D3d12DecoderError::HwDeviceInit(code)) => {
                eprintln!("skipping: this device does not do video decoding (code {code})");
                return;
            }
            Err(error) => panic!("the stream did not open: {error}"),
        };
        let Ok(mut encoder) = D3d12VideoEncoder::new(
            "encoder",
            &gpu,
            D3d12VideoEncoderOptions {
                width: 1920,
                height: 1080,
                ..options(D3d12VideoCodec::H265)
            },
        ) else {
            eprintln!("skipping: this GPU does not encode HEVC through D3D12");
            return;
        };
        let encoded_parameters = encoder.parameters();
        let encoded = capture(&mut encoder);
        let decoded = capture(&mut decoder);
        let sent = packets.len();
        for packet in packets {
            let mut packet = packet;
            packet.set_time_base(ffmpeg::Rational::new(1, 30));
            decoder
                .consume(MediaBuffer::Packet(Arc::new(packet).into()))
                .expect("decodes");
            for buf in std::mem::take(&mut *decoded.lock().unwrap()) {
                encoder.consume(buf).expect("encodes");
            }
        }
        crate::stream::deliver(&mut decoder, &crate::stream::StreamEvent::Eos).expect("drains");
        for buf in std::mem::take(&mut *decoded.lock().unwrap()) {
            encoder.consume(buf).expect("encodes");
        }
        crate::stream::deliver(&mut encoder, &crate::stream::StreamEvent::Eos).expect("drains");
        let encoded = std::mem::take(&mut *encoded.lock().unwrap());
        assert_eq!(encoded.len(), sent, "a packet a picture");

        let mut back = SwDecoder::new("back", encoded_parameters).expect("opens");
        let pictures = capture(&mut back);
        for packet in encoded {
            back.consume(packet).expect("decodes back");
        }
        crate::stream::deliver(&mut back, &crate::stream::StreamEvent::Eos).expect("drains");
        let pictures = pictures.lock().unwrap();
        assert_eq!(pictures.len(), sent);
        for buf in pictures.iter() {
            let MediaBuffer::Video(frame) = buf else {
                panic!("pictures");
            };
            assert_eq!((frame.width(), frame.height()), (1920, 1080));
            let luma = frame.data(0)[540 * frame.stride(0) + 960];
            assert!(luma.abs_diff(100) <= 3, "luma {luma}, not 100");
        }
    }
}
