use std::sync::Arc;

use crate::pp_log::{PpLog, pp_error, pp_info};
use ffmpeg_next as ffmpeg;
use ffmpeg_next::Rescale;
use thiserror::Error as ThisError;

use crate::color::ColorDescription;
use crate::{
    buffer::MediaBuffer,
    contract::{
        InputContract, MediaKind, MemoryDomain, OutputContract, PixelLayout, PixelLayoutSet,
        PortContract,
    },
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    error::Result,
    pad::SrcPad,
};

use crate::elements::filter::is_codec_drain_boundary;

/// Errors specific to `SwEncoder`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum SwEncoderError {
    /// The requested encoder is unavailable in the linked FFmpeg build.
    #[error(
        "encoder {0:?} not found — this ffmpeg build wasn't compiled with it \
         (see VideoCodec's own docs: GPL-licensed ones need --enable-gpl; \
         run `ffmpeg -encoders` to see what's actually available)"
    )]
    CodecNotFound(String),

    /// The sink received a buffer other than decoded video or end-of-stream.
    #[error("SwEncoder only accepts Video or Eos buffers, got {0}")]
    UnsupportedBuffer(&'static str),

    /// A frame arrived with a `pts` but no unit to read it in, so there is
    /// no telling where on the encoder's own timeline it belongs.
    ///
    /// Every element in this crate that makes a frame says what unit its
    /// `pts` is in — see [`crate::buffer::time_base`] — so this is a frame
    /// made elsewhere. Stamp it with [`crate::buffer::set_time_base`].
    #[error(
        "a Video frame arrived with a pts but no time base to read it in; the element \
         that made it has to set one (media_pp::buffer::set_time_base)"
    )]
    NoTimeBase,

    /// The codec publishes the pixel formats it encodes and
    /// [`SwEncoderOptions::pixel_format`] is not one of them — `libopenh264`
    /// takes only `YUV420P`, for example. Caught here because
    /// `avcodec_open2` would otherwise report it as a bare `Invalid argument`.
    #[error("{0:?} cannot encode {1:?} frames — only {2:?}")]
    UnsupportedPixelFormat(String, ffmpeg::format::Pixel, Vec<ffmpeg::format::Pixel>),

    /// A frame is not in the format or size this encoder was opened for.
    /// The encoder reads every frame as what it was opened with, so one in
    /// another layout — a BGRA capture, an NV12 frame into a `YUV420P`
    /// encoder — would be encoded as garbage rather than fail. Convert it
    /// first, with a [`SwScaler`](crate::elements::SwScaler).
    #[error(
        "SwEncoder was opened for {expected_width}x{expected_height} {expected_format:?} frames, \
         got {width}x{height} {format:?}; convert it first (e.g. with a SwScaler)"
    )]
    FrameMismatch {
        /// The pixel format the encoder was opened for.
        expected_format: ffmpeg::format::Pixel,
        /// The width the encoder was opened for.
        expected_width: u32,
        /// The height the encoder was opened for.
        expected_height: u32,
        /// The frame's pixel format.
        format: ffmpeg::format::Pixel,
        /// The frame's width.
        width: u32,
        /// The frame's height.
        height: u32,
    },

    /// FFmpeg rejected encoder creation or frame/packet processing.
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),
}

/// Which software encoder to open. Whatever's picked, [`SwEncoder::new`]
/// fails with [`SwEncoderError::CodecNotFound`] (not a panic) if the
/// linked ffmpeg build doesn't actually have it — this crate never
/// vendors any of these, it's whatever the local ffmpeg install was built
/// with (check with `ffmpeg -encoders`).
///
/// GPL-licensed encoders need an ffmpeg build compiled with
/// `--enable-gpl` (separate from — and unlike — FFmpeg's own native
/// `h264`/`hevc` **decoders**, which `SwDecoder` already uses and which
/// carry no such requirement; GPL only enters the picture on the encode
/// side, through these specific libraries): [`VideoCodec::H264`] and
/// [`VideoCodec::H265`] are GPL. Every other variant here is a
/// permissively-licensed (BSD/similar) alternative that needs no special
/// build flag beyond being enabled at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoCodec {
    /// `libx264` — H.264. GPL.
    H264,
    /// `libopenh264` — H.264, Cisco's BSD-2-Clause encoder (Cisco covers
    /// H.264 patent royalties for binary redistributions of it). The
    /// non-GPL alternative to [`VideoCodec::H264`].
    OpenH264,
    /// `libx265` — H.265/HEVC. GPL.
    H265,
    /// `libkvazaar` — H.265/HEVC, BSD-2-Clause. The non-GPL alternative
    /// to [`VideoCodec::H265`].
    Kvazaar,
    /// `libvpx` — VP8, BSD-3-Clause.
    Vp8,
    /// `libvpx-vp9` — VP9, BSD-3-Clause (same `libvpx` project as
    /// [`VideoCodec::Vp8`], different encoder name).
    Vp9,
    /// `libaom-av1` — AV1's reference encoder, BSD-2-Clause. Most
    /// broadly compatible AV1 output, but much slower than
    /// [`VideoCodec::Svtav1`] — prefer that one unless you specifically
    /// need `libaom`'s own encoding behavior.
    Av1,
    /// `libsvtav1` — AV1 via Intel's SVT-AV1, BSD-2-Clause-Patent. Far
    /// faster than [`VideoCodec::Av1`] (`libaom-av1`) at a given quality
    /// target — the practical default for real-time AV1 encoding.
    Svtav1,
    /// `mpeg4` — MPEG-4 Part 2, and the only encoder here that is FFmpeg's
    /// own rather than a library it was linked against.
    ///
    /// Which is the whole reason it is offered. Every variant above needs
    /// its library to have been enabled at build time, and an FFmpeg with
    /// none of them cannot encode video at all through this element. This
    /// one is in every build there is.
    ///
    /// It is also the only always-present encoder that will emit B-frames
    /// (see [`SwEncoderOptions::max_b_frames`]) — `libopenh264` emits none
    /// whatever it is asked for, and the H.264 encoders that would are the
    /// GPL ones. That makes it what a test reaches for when what it is
    /// testing is reordering.
    ///
    /// Not a codec to choose for output anyone has to play: it predates
    /// H.264 and compresses far worse at the same quality.
    Mpeg4,
}

impl VideoCodec {
    /// The unit this codec's packets are counted in — see
    /// [`SwEncoder::time_base`].
    fn time_base(self) -> ffmpeg::Rational {
        match self {
            // The MPEG-4 Part 2 bitstream stores the time base's denominator
            // in 16 bits, and FFmpeg refuses anything past 65535 rather than
            // truncate it. 60000 still counts 59.94 fps in whole ticks.
            VideoCodec::Mpeg4 => ffmpeg::Rational(1, 60_000),
            _ => super::TIME_BASE,
        }
    }

    fn encoder_name(self) -> &'static str {
        match self {
            VideoCodec::H264 => "libx264",
            VideoCodec::OpenH264 => "libopenh264",
            VideoCodec::H265 => "libx265",
            VideoCodec::Kvazaar => "libkvazaar",
            VideoCodec::Vp8 => "libvpx",
            VideoCodec::Vp9 => "libvpx-vp9",
            VideoCodec::Av1 => "libaom-av1",
            VideoCodec::Svtav1 => "libsvtav1",
            VideoCodec::Mpeg4 => "mpeg4",
        }
    }
}

/// Construction-time options for [`SwEncoder::new`]. `width`/`height` must
/// already be known rather than inferred from the first frame, since
/// `avcodec_open2` needs them set before this can be opened at all — and a
/// muxer needs the parameters that come out of it before the first frame.
/// The unit of the frames' timestamps is not asked for: each frame carries
/// its own, and the encoder converts it into [`SwEncoder::time_base`].
#[derive(Debug, Clone, Copy)]
pub struct SwEncoderOptions {
    /// Compressed video codec to open.
    pub codec: VideoCodec,
    /// Encoded frame width in pixels.
    pub width: u32,
    /// Encoded frame height in pixels.
    pub height: u32,
    /// The pixel format every frame given to the encoder is in, and the one
    /// it is opened for. `YUV420P` is what every codec here encodes; others
    /// — `NV12`, 10-bit `YUV420P10LE`, 4:4:4 — are taken by the codecs that
    /// publish them, which [`SwEncoder::new`] checks, failing with
    /// [`SwEncoderError::UnsupportedPixelFormat`] for one the codec lacks.
    ///
    /// Frames are not converted: one in another format or size is refused
    /// with [`SwEncoderError::FrameMismatch`], since the encoder would
    /// otherwise read its planes as this layout and encode garbage. Put a
    /// [`SwScaler`](crate::elements::SwScaler) to this format in front.
    pub pixel_format: ffmpeg::format::Pixel,
    /// The nominal rate the encoder uses for internal rate-control
    /// (targeting `bit_rate` per frame) and the frame-rate metadata it
    /// writes into the bitstream — *not* required to match the real
    /// interval between `send_frame` calls. For a source with its own
    /// genuinely fixed rate (e.g. [`crate::elements::TestVideoSource`]),
    /// that's `TestVideoOptions::frame_rate` itself, and for a screen
    /// capture its own `frame_rate` (`DxgiCaptureOptions::frame_rate` and
    /// the like), since a capture emits at that fixed rate. Encoded packets
    /// carry each frame's own `pts` either way, so muxing stays correct even
    /// where this nominal rate and the true one differ.
    pub frame_rate: ffmpeg::Rational,
    /// Target encoded bit rate, in bits per second.
    pub bit_rate: usize,
    /// How many frames between keyframes — `AVCodecContext.gop_size`
    /// directly (not a duration; multiply by `frame_rate` yourself, e.g.
    /// `frame_rate * 2` for "roughly every 2 seconds"). Not every codec's
    /// own default is a periodic interval at all — `libopenh264` was
    /// found, building [`crate::elements::SegmentedFileMuxer`], to rely on
    /// scene-change detection alone and go an *entire* recording without
    /// a second keyframe against smoothly-changing content — so this is
    /// always set explicitly rather than left to whatever a given codec
    /// happens to default to. Matters beyond segmenting a recording, too:
    /// [`crate::elements::RtspMuxer`]/`WebRtcTrackSink`
    /// viewers/peers joining mid-stream can't decode anything until the
    /// next keyframe, so an unbounded interval is a real join-latency
    /// problem, not just a segmenting one.
    pub gop_size: u32,
    /// How many consecutive B-frames the encoder may insert, or `None` to
    /// leave the codec's own default alone — which is what every caller
    /// wanting nothing to do with this should pass, since the defaults
    /// differ (`libx264` picks 3, `libopenh264` emits none at all whatever
    /// this says).
    ///
    /// A B-frame is coded from frames on both sides of it, so the encoder
    /// hands packets over in a different order than it was given them and
    /// their `dts` stops equalling their `pts`. That is the reason to reach
    /// for this deliberately rather than to compress a little better:
    /// reordering is a path a muxer, an RTP payloader and a seek all have to
    /// get right, and a pipeline whose every packet arrives in presentation
    /// order never exercises it.
    pub max_b_frames: Option<u32>,
}

/// Encodes `Video` frames in [`SwEncoderOptions::pixel_format`] (`YUV420P`
/// for most) into `Packet`s via a software encoder (see [`VideoCodec`]) — the mirror image of
/// [`crate::elements::SwDecoder`]'s decode direction. A `Filter`: receives
/// via `Sink`, pushes what it produces into its own (single) src pad.
///
/// One frame can turn into zero or one packets per `send_frame` (B-frame
/// reordering delays some frames' packets until later ones arrive, or
/// until `Eos` flushes whatever's left) — `consume` drains `receive_packet`
/// in a loop after every `send_frame`/`send_eof`, same shape as
/// `SwDecoder`'s own `receive_frame` drain loop.
pub struct SwEncoder {
    pp_log: PpLog,
    name: Arc<str>,
    encoder: ffmpeg::encoder::Video,
    /// Nominal frame duration in `encoder.time_base()` ticks. Some codecs
    /// (notably `libopenh264`) leave `AVPacket::duration` at zero; muxers
    /// such as HLS need it for precise segment durations.
    packet_duration: i64,
    /// The unit each produced packet's `pts` is expressed in — the codec's
    /// own, see [`SwEncoder::time_base`]. Stamped onto
    /// every packet in `drain` since `avcodec_receive_packet` itself never
    /// sets `AVPacket.time_base` (only the encoder context's own time base,
    /// via `set_time_base` below); without it, a packet's own
    /// `Packet::time_base()` reads back FFmpeg's `0/1` "unset" sentinel —
    /// wrong for anything (e.g. `WebRtcPeer::write_track`) that derives
    /// real time from a packet's own declared time_base rather than
    /// external knowledge of what this encoder was built with.
    time_base: ffmpeg::Rational,
    /// What every frame must be: the format and size the codec was opened
    /// for. See [`SwEncoderError::FrameMismatch`].
    pixel_format: ffmpeg::format::Pixel,
    width: u32,
    height: u32,
    pad: SrcPad,
}

/// `YUVJ*` is the deprecated full-range spelling of the same planes; a frame
/// in one is laid out exactly as the other, so either fits an encoder opened
/// for the pair. Range travels separately, in the frame's colour metadata.
fn same_layout(a: ffmpeg::format::Pixel, b: ffmpeg::format::Pixel) -> bool {
    use ffmpeg::format::Pixel;
    let plain = |pixel| match pixel {
        Pixel::YUVJ420P => Pixel::YUV420P,
        Pixel::YUVJ422P => Pixel::YUV422P,
        Pixel::YUVJ444P => Pixel::YUV444P,
        other => other,
    };
    plain(a) == plain(b)
}

fn nominal_packet_duration(time_base: ffmpeg::Rational, frame_rate: ffmpeg::Rational) -> i64 {
    if frame_rate.numerator() <= 0 || time_base.numerator() <= 0 || time_base.denominator() <= 0 {
        return 0;
    }
    1i64.rescale(
        ffmpeg::Rational::new(frame_rate.denominator(), frame_rate.numerator()),
        time_base,
    )
    .max(1)
}

impl SwEncoder {
    /// Opens the requested software video encoder with the supplied output definition.
    pub fn new(name: impl Into<String>, options: SwEncoderOptions) -> Result<Self> {
        crate::ensure_ffmpeg();
        Self::open(name, options, None)
    }

    /// As [`Self::new`], and the stream says `color` is what it holds —
    /// see [`ColorDescription`] for why that matters, and `CudaEncoder`'s
    /// `with_color` for the hardware twin. Nothing is converted: the frames must
    /// already be what `color` says, as a
    /// [`SwScaler`](crate::elements::SwScaler) keeps them when it changes
    /// only their layout.
    pub fn with_color(
        name: impl Into<String>,
        options: SwEncoderOptions,
        color: ColorDescription,
    ) -> Result<Self> {
        crate::ensure_ffmpeg();
        Self::open(name, options, Some(color))
    }

    fn open(
        name: impl Into<String>,
        options: SwEncoderOptions,
        color: Option<ColorDescription>,
    ) -> Result<Self> {
        let encoder_name = options.codec.encoder_name();
        let codec = ffmpeg::encoder::find_by_name(encoder_name)
            .ok_or_else(|| SwEncoderError::CodecNotFound(encoder_name.into()))?;

        // Asked of the codec rather than hardcoded per `VideoCodec`, as
        // `SwAudioEncoder` asks for sample rates. A codec that publishes no
        // list is left alone.
        {
            let capabilities = codec.video().map_err(SwEncoderError::from)?;
            if let Some(formats) = capabilities.formats() {
                let supported: Vec<ffmpeg::format::Pixel> = formats.collect();
                if !supported
                    .iter()
                    .any(|&format| same_layout(format, options.pixel_format))
                {
                    return Err(SwEncoderError::UnsupportedPixelFormat(
                        encoder_name.into(),
                        options.pixel_format,
                        supported,
                    )
                    .into());
                }
            }
        }

        let mut context = ffmpeg::codec::context::Context::new_with_codec(codec);
        // The codec's own headers — SPS/PPS and the like — go into
        // `extradata` for the container to write, rather than only in-band.
        // MP4 does not need that (`avcC` is built in the trailer out of the
        // packets themselves), but Matroska writes `CodecPrivate` before the
        // first frame and fails outright without it, and RTSP needs it to
        // describe the stream in its SDP. Muxers that want the headers
        // in-band as well — MPEG-TS — reinsert them from here themselves.
        context.set_flags(ffmpeg::codec::Flags::GLOBAL_HEADER);
        let time_base = options.codec.time_base();
        context.set_time_base(time_base);

        let mut video = context.encoder().video().map_err(SwEncoderError::from)?;
        video.set_width(options.width);
        video.set_height(options.height);
        video.set_format(options.pixel_format);
        video.set_time_base(time_base);
        video.set_frame_rate(Some(options.frame_rate));
        video.set_bit_rate(options.bit_rate);
        video.set_gop(options.gop_size);
        if let Some(frames) = options.max_b_frames {
            video.set_max_b_frames(frames as usize);
        }
        if let Some(color) = color {
            // SAFETY: the encoder's own context, not yet opened.
            unsafe { color.tell(video.as_mut_ptr()) };
        }

        let encoder = video.open_as(codec).map_err(SwEncoderError::from)?;
        let packet_duration = nominal_packet_duration(time_base, options.frame_rate);

        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::SwEncoder, &name, None);
        let pad = SrcPad::with_contract(
            format!("{name}_src"),
            OutputContract::Fixed(PortContract::packet(MediaKind::VideoPacket)),
        );
        pp_info!(
            pp_log: &pp_log,
            "opened: codec={encoder_name}, {}x{}, bit_rate={}",
            options.width,
            options.height,
            options.bit_rate
        );
        Ok(Self {
            name,
            pp_log,
            encoder,
            packet_duration,
            time_base,
            pixel_format: options.pixel_format,
            width: options.width,
            height: options.height,
            pad,
        })
    }

    /// This encoder's own codec parameters — what you need to construct
    /// a matching [`crate::elements::SwDecoder`] to decode the `Packet`s
    /// this produces, or what a [`crate::elements::FileMuxer`] track needs
    /// (same pattern [`crate::elements::SwAudioEncoder::parameters`]
    /// documents for audio), when there's no container/demuxer in the loop
    /// to get them from otherwise (e.g. encoding straight into a `Tee`/RTSP
    /// sink, or decoding straight back out for a round-trip smoke test).
    pub fn parameters(&self) -> ffmpeg::codec::Parameters {
        ffmpeg::codec::Parameters::from(&self.encoder)
    }

    /// The unit each packet's `pts`, `dts` and duration are counted in —
    /// what a muxer's `add_stream` is to be told. The encoder's own, chosen
    /// for its codec, and not the unit of the frames it is fed: those it
    /// reads off each frame and converts.
    pub fn time_base(&self) -> ffmpeg::Rational {
        self.time_base
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
                Err(error) => return Err(SwEncoderError::from(error).into()),
            }
        }
        Ok(())
    }
}

/// The track this encoder's packets make: its [`SwEncoder::parameters`], timed
/// in its [`SwEncoder::time_base`].
impl From<&SwEncoder> for crate::elements::TrackFormat {
    fn from(encoder: &SwEncoder) -> Self {
        Self::new(encoder.parameters(), encoder.time_base())
    }
}

impl Element for SwEncoder {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::SwEncoder
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for SwEncoder {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for SwEncoder {
    /// System memory specifically: this encoder reads the frame's planes
    /// on the CPU, so a D3D11 or CUDA frame is not merely the wrong
    /// format here, it is unreachable memory.
    fn input_contract(&self) -> InputContract {
        // The one layout it was opened for: a wiring that can only deliver
        // another — a BGRA capture straight in — is refused before the
        // pipeline starts, rather than encoded as garbage.
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::System)
                .with_layouts(PixelLayoutSet::of(PixelLayout::of(self.pixel_format))),
        )
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Video(frame) => {
                if !same_layout(frame.format(), self.pixel_format)
                    || frame.width() != self.width
                    || frame.height() != self.height
                {
                    return Err(SwEncoderError::FrameMismatch {
                        expected_format: self.pixel_format,
                        expected_width: self.width,
                        expected_height: self.height,
                        format: frame.format(),
                        width: frame.width(),
                        height: frame.height(),
                    }
                    .into());
                }
                let pts = super::pts_in(&frame, self.time_base)
                    .map_err(|super::NoTimeBase| SwEncoderError::NoTimeBase)?;
                let frame =
                    super::restamped(&frame, pts, self.time_base).map_err(SwEncoderError::from)?;
                self.encoder
                    .send_frame(&frame)
                    .inspect_err(|error| pp_error!(self, "send_frame failed: {error}"))
                    .map_err(SwEncoderError::from)?;
                self.drain()
            }
            MediaBuffer::Eos => {
                self.encoder
                    .send_eof()
                    .inspect_err(|error| pp_error!(self, "send_eof failed: {error}"))
                    .map_err(SwEncoderError::from)?;
                self.drain()?;
                self.pad.push(MediaBuffer::Eos)
            }
            other => Err(SwEncoderError::UnsupportedBuffer(other.kind()).into()),
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        // Current behavior deliberately forwards Seek without flushing
        // the encoder. Encoders may retain delayed/reordered frames (see
        // the type docs), so packets originating before the seek can still
        // be emitted by later `send_frame` calls. Callers that require a
        // hard encoded-stream discontinuity must rebuild the encoder; this
        // implementation does not promise that boundary. `Stop` abandons
        // the codec context without flushing it.
        self.pad.control(msg)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex as StdMutex;

    use super::*;

    #[test]
    fn nominal_frame_duration_is_expressed_in_encoder_time_base_ticks() {
        assert_eq!(
            nominal_packet_duration(ffmpeg::Rational::new(1, 30), ffmpeg::Rational::new(30, 1),),
            1
        );
        assert_eq!(
            nominal_packet_duration(
                ffmpeg::Rational::new(1, 90_000),
                ffmpeg::Rational::new(30_000, 1001),
            ),
            3003
        );
        assert_eq!(
            nominal_packet_duration(ffmpeg::Rational::new(1, 30), ffmpeg::Rational::new(0, 1),),
            0
        );
    }

    /// `SwEncoder::new` should fail cleanly (not panic) when the linked
    /// ffmpeg build wasn't compiled with the requested encoder — the
    /// GPL-only `libx264`/`libx265` in particular aren't guaranteed to be
    /// present (this crate never vendors them; it's whatever the local
    /// ffmpeg install was built with). Real regression coverage on
    /// whichever build runs this test, not a mock.
    #[test]
    fn codec_not_found_is_a_clean_error_not_a_panic() {
        for codec in [
            VideoCodec::H264,
            VideoCodec::OpenH264,
            VideoCodec::H265,
            VideoCodec::Kvazaar,
            VideoCodec::Vp8,
            VideoCodec::Vp9,
            VideoCodec::Av1,
            VideoCodec::Svtav1,
        ] {
            let result = SwEncoder::new(
                "encoder",
                SwEncoderOptions {
                    codec,
                    width: 640,
                    height: 480,
                    pixel_format: ffmpeg::format::Pixel::YUV420P,
                    frame_rate: ffmpeg::Rational::new(30, 1),
                    bit_rate: 1_000_000,
                    gop_size: 60, // ~2s @ 30fps
                    max_b_frames: None,
                },
            );
            // Whether it succeeds or fails depends on how this machine's
            // ffmpeg was built — either is fine, just never a panic. If it
            // fails, it must be *this* error, not some other ffmpeg
            // failure mode.
            if let Err(error) = result {
                assert!(
                    matches!(
                        error,
                        crate::error::Error::SwEncoderError(SwEncoderError::CodecNotFound(_))
                    ),
                    "expected CodecNotFound, got {error:?}"
                );
            }
        }
    }

    struct CapturingSink {
        pp_log: PpLog,
        packets: Arc<StdMutex<Vec<Arc<ffmpeg::Packet>>>>,
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
            if let MediaBuffer::Packet(packet) = buf {
                self.packets.lock().unwrap().push(packet);
            }
            Ok(())
        }

        fn control(&mut self, _msg: ControlMsg) -> Result<()> {
            Ok(())
        }
    }

    /// Regression test: `avcodec_receive_packet` never sets a packet's own
    /// `AVPacket.time_base` — only `drain` stamping it explicitly (added
    /// alongside this test) keeps `Packet::time_base()` from reading back
    /// FFmpeg's `0/1` "unset" sentinel. That silently broke
    /// `WebRtcPeer::write_track`, which derives real RTP time from each
    /// packet's own declared time_base (`0/1`'s numerator of `0` fails its
    /// validation, so every packet was dropped) — see
    /// `webrtc_av_loopback`'s own regression run.
    #[test]
    fn produced_packets_carry_the_encoders_time_base() {
        let options = SwEncoderOptions {
            codec: VideoCodec::OpenH264,
            width: 64,
            height: 64,
            pixel_format: ffmpeg::format::Pixel::YUV420P,
            frame_rate: ffmpeg::Rational::new(30, 1),
            bit_rate: 200_000,
            gop_size: 30,
            max_b_frames: None,
        };
        let Ok(mut encoder) = SwEncoder::new("encoder", options) else {
            return; // openh264 unavailable on this build, see codec_not_found_is_a_clean_error_not_a_panic
        };
        let packets = Arc::new(StdMutex::new(Vec::new()));
        encoder.src_pads()[0].link(Box::new(CapturingSink {
            packets: packets.clone(),
            pp_log: element_pp_log(ElementType::Other, "capture", None),
        }));

        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::YUV420P, 64, 64);
        frame.set_pts(Some(0));
        crate::buffer::set_time_base(&mut frame, ffmpeg::Rational::new(1, 30));
        for plane in 0..frame.planes() {
            frame.data_mut(plane).fill(128);
        }
        encoder.consume(MediaBuffer::video(frame)).unwrap();
        encoder.consume(MediaBuffer::Eos).unwrap();

        let packets = packets.lock().unwrap();
        assert!(!packets.is_empty(), "expected at least one packet");
        for packet in packets.iter() {
            assert_eq!(
                packet.time_base(),
                encoder.time_base(),
                "packet time_base must be the encoder's own, \
                 not FFmpeg's 0/1 unset sentinel"
            );
        }
    }

    /// What reaches an encoder's pad.
    type Captured = Arc<StdMutex<Vec<Arc<ffmpeg::Packet>>>>;

    /// An OpenH264 encoder and what reaches its pad, or `None` where this
    /// FFmpeg has no OpenH264.
    fn open_h264() -> Option<(SwEncoder, Captured)> {
        let options = SwEncoderOptions {
            codec: VideoCodec::OpenH264,
            width: 64,
            height: 64,
            pixel_format: ffmpeg::format::Pixel::YUV420P,
            frame_rate: ffmpeg::Rational::new(30, 1),
            bit_rate: 200_000,
            gop_size: 30,
            max_b_frames: None,
        };
        let mut encoder = SwEncoder::new("encoder", options).ok()?;
        let packets = Arc::new(StdMutex::new(Vec::new()));
        encoder.src_pads()[0].link(Box::new(CapturingSink {
            packets: packets.clone(),
            pp_log: element_pp_log(ElementType::Other, "capture", None),
        }));
        Some((encoder, packets))
    }

    /// A grey frame due `pts` in `unit`, or with no unit at all.
    fn grey(pts: i64, unit: Option<ffmpeg::Rational>) -> MediaBuffer {
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::YUV420P, 64, 64);
        frame.set_pts(Some(pts));
        if let Some(unit) = unit {
            crate::buffer::set_time_base(&mut frame, unit);
        }
        for plane in 0..frame.planes() {
            frame.data_mut(plane).fill(128);
        }
        MediaBuffer::video(frame)
    }

    /// The frames are not asked to be in the encoder's unit: each is read in
    /// its own and placed on the encoder's timeline. A tenth of a second in
    /// thirtieths and in milliseconds is the same place.
    #[test]
    fn a_frame_is_placed_in_the_unit_it_carries() {
        let Some((mut encoder, packets)) = open_h264() else {
            return;
        };
        encoder
            .consume(grey(3, Some(ffmpeg::Rational::new(1, 30))))
            .unwrap();
        encoder
            .consume(grey(200, Some(ffmpeg::Rational::new(1, 1000))))
            .unwrap();
        encoder.consume(MediaBuffer::Eos).unwrap();

        let tick = |seconds: f64| (seconds * f64::from(encoder.time_base().denominator())) as i64;
        let pts: Vec<_> = packets.lock().unwrap().iter().map(|p| p.pts()).collect();
        assert_eq!(pts, vec![Some(tick(0.1)), Some(tick(0.2))]);
    }

    /// A frame that does not say what unit its `pts` is in is refused by
    /// name rather than encoded at a guessed time, and nothing comes out.
    #[test]
    fn a_frame_with_no_time_base_is_refused_rather_than_guessed_at() {
        let Some((mut encoder, packets)) = open_h264() else {
            return;
        };
        let error = encoder.consume(grey(0, None)).unwrap_err();
        assert!(matches!(
            error,
            crate::Error::SwEncoderError(SwEncoderError::NoTimeBase)
        ));
        encoder.consume(MediaBuffer::Eos).unwrap();
        assert!(packets.lock().unwrap().is_empty());
    }

    /// MPEG-4 Part 2 cannot count in 90 kHz — its bitstream has 16 bits for
    /// the denominator — so it gets a unit it can, rather than failing to
    /// open.
    #[test]
    fn mpeg4_opens_with_a_unit_its_bitstream_can_hold() {
        let options = SwEncoderOptions {
            codec: VideoCodec::Mpeg4,
            width: 64,
            height: 64,
            pixel_format: ffmpeg::format::Pixel::YUV420P,
            frame_rate: ffmpeg::Rational::new(30, 1),
            bit_rate: 200_000,
            gop_size: 30,
            max_b_frames: None,
        };
        let encoder = match SwEncoder::new("encoder", options) {
            Ok(encoder) => encoder,
            Err(crate::Error::SwEncoderError(SwEncoderError::CodecNotFound(_))) => return,
            Err(error) => panic!("mpeg4 failed to open: {error}"),
        };
        assert!(encoder.time_base().denominator() <= 65_535);
    }

    /// What the muxer builds the stream from carries the colour the encoder
    /// was told — and one told nothing says nothing, as before.
    #[test]
    fn a_stream_opened_with_a_colour_says_it_and_one_without_does_not() {
        let options = SwEncoderOptions {
            codec: VideoCodec::OpenH264,
            width: 64,
            height: 64,
            pixel_format: ffmpeg::format::Pixel::YUV420P,
            frame_rate: ffmpeg::Rational::new(30, 1),
            bit_rate: 200_000,
            gop_size: 30,
            max_b_frames: None,
        };
        let Ok(told) = SwEncoder::with_color("told", options, ColorDescription::BT709_LIMITED)
        else {
            return; // openh264 unavailable on this build
        };
        let untold = SwEncoder::new("untold", options).expect("opened once already");

        let described = |encoder: &SwEncoder| {
            let parameters = encoder.parameters();
            // SAFETY: a live `AVCodecParameters` owned by `parameters`.
            unsafe {
                let raw = &*parameters.as_ptr();
                (
                    raw.color_space,
                    raw.color_range,
                    raw.color_primaries,
                    raw.color_trc,
                )
            }
        };
        assert_eq!(
            described(&told),
            (
                ffmpeg::ffi::AVColorSpace::AVCOL_SPC_BT709,
                ffmpeg::ffi::AVColorRange::AVCOL_RANGE_MPEG,
                ffmpeg::ffi::AVColorPrimaries::AVCOL_PRI_BT709,
                ffmpeg::ffi::AVColorTransferCharacteristic::AVCOL_TRC_BT709,
            )
        );
        assert_eq!(
            described(&untold).0,
            ffmpeg::ffi::AVColorSpace::AVCOL_SPC_UNSPECIFIED
        );
    }

    /// A frame of `format` and size, due at 0 in thirtieths.
    fn picture(format: ffmpeg::format::Pixel, width: u32, height: u32) -> MediaBuffer {
        let mut frame = ffmpeg::frame::Video::new(format, width, height);
        frame.set_pts(Some(0));
        crate::buffer::set_time_base(&mut frame, ffmpeg::Rational::new(1, 30));
        for plane in 0..frame.planes() {
            frame.data_mut(plane).fill(128);
        }
        MediaBuffer::video(frame)
    }

    /// A frame the encoder was not opened for — a BGRA capture straight in,
    /// NV12, the wrong size — used to be read as if it were, and encoded as
    /// a green picture with no error anywhere. It is refused, naming what
    /// it is, and the encoder goes on taking the frames it was opened for.
    #[test]
    fn a_frame_in_another_format_or_size_is_refused_and_the_encoder_goes_on() {
        use ffmpeg::format::Pixel;
        let Some((mut encoder, packets)) = open_h264() else {
            return;
        };
        for (format, width, height) in [
            (Pixel::BGRA, 64, 64),
            (Pixel::NV12, 64, 64),
            (Pixel::YUV420P, 32, 64),
        ] {
            let error = encoder
                .consume(picture(format, width, height))
                .expect_err("a frame the encoder was not opened for is refused");
            assert!(
                matches!(
                    error,
                    crate::Error::SwEncoderError(SwEncoderError::FrameMismatch {
                        expected_format: Pixel::YUV420P,
                        expected_width: 64,
                        expected_height: 64,
                        format: refused,
                        width: refused_width,
                        ..
                    }) if refused == format && refused_width == width
                ),
                "{error}"
            );
        }
        assert!(packets.lock().unwrap().is_empty(), "nothing was encoded");

        // `YUVJ420P` is the same planes, spelled for full range.
        for format in [Pixel::YUV420P, Pixel::YUVJ420P] {
            encoder
                .consume(picture(format, 64, 64))
                .expect("a frame in the opened layout encodes");
        }
        encoder.consume(MediaBuffer::Eos).unwrap();
        assert!(!packets.lock().unwrap().is_empty());
    }

    /// The same, before the pipeline starts: a wiring that can only deliver
    /// another layout does not link, and one that says nothing about layout
    /// still does, leaving it to the check above.
    #[test]
    fn only_the_layout_it_was_opened_for_links() {
        let Some((encoder, _)) = open_h264() else {
            return;
        };
        let InputContract::Fixed(accepted) = encoder.input_contract() else {
            panic!("an encoder states what it takes");
        };
        let frames = || PortContract::frame(MediaKind::VideoFrame, MemoryDomain::System);
        let yuv420p = PixelLayoutSet::of(PixelLayout::of(ffmpeg::format::Pixel::YUV420P));
        assert!(!accepted.accepts(&frames().with_layouts(PixelLayoutSet::BGRA)));
        assert!(!accepted.accepts(&frames().with_layouts(PixelLayoutSet::NV12)));
        assert!(accepted.accepts(&frames().with_layouts(yuv420p)));
        assert!(accepted.accepts(&frames()));
    }

    /// A codec that publishes the formats it encodes refuses one it lacks
    /// when it is built, naming the ones it takes — rather than failing
    /// inside `avcodec_open2` with a bare `Invalid argument`.
    #[test]
    fn a_pixel_format_the_codec_lacks_is_refused_when_it_is_built() {
        let result = SwEncoder::new(
            "encoder",
            SwEncoderOptions {
                codec: VideoCodec::OpenH264,
                width: 64,
                height: 64,
                pixel_format: ffmpeg::format::Pixel::BGRA,
                frame_rate: ffmpeg::Rational::new(30, 1),
                bit_rate: 200_000,
                gop_size: 30,
                max_b_frames: None,
            },
        );
        match result {
            Err(crate::Error::SwEncoderError(SwEncoderError::CodecNotFound(_))) => {}
            Err(crate::Error::SwEncoderError(SwEncoderError::UnsupportedPixelFormat(
                _,
                ffmpeg::format::Pixel::BGRA,
                supported,
            ))) => assert!(
                supported.contains(&ffmpeg::format::Pixel::YUV420P),
                "the alternatives name YUV420P: {supported:?}"
            ),
            Err(other) => panic!("expected UnsupportedPixelFormat, got {other}"),
            Ok(_) => panic!("libopenh264 does not encode BGRA"),
        }
    }

    /// A codec that takes NV12 is opened for it and encodes it as it is,
    /// with no conversion in front — where this FFmpeg has one.
    #[test]
    fn a_codec_that_takes_nv12_encodes_it_directly() {
        use ffmpeg::format::Pixel;
        let Some(codec) = [VideoCodec::H264, VideoCodec::H265]
            .into_iter()
            .find(|codec| {
                ffmpeg::encoder::find_by_name(codec.encoder_name())
                    .and_then(|found| found.video().ok()?.formats())
                    .is_some_and(|mut formats| formats.any(|format| format == Pixel::NV12))
            })
        else {
            eprintln!("skipping: no encoder here takes NV12");
            return;
        };
        let mut encoder = SwEncoder::new(
            "encoder",
            SwEncoderOptions {
                codec,
                width: 64,
                height: 64,
                pixel_format: Pixel::NV12,
                frame_rate: ffmpeg::Rational::new(30, 1),
                bit_rate: 200_000,
                gop_size: 30,
                max_b_frames: None,
            },
        )
        .expect("an encoder that lists NV12 opens for it");
        let packets = Arc::new(StdMutex::new(Vec::new()));
        encoder.src_pads()[0].link(Box::new(CapturingSink {
            packets: packets.clone(),
            pp_log: element_pp_log(ElementType::Other, "capture", None),
        }));
        encoder
            .consume(picture(Pixel::NV12, 64, 64))
            .expect("NV12 encodes");
        encoder.consume(MediaBuffer::Eos).unwrap();
        assert!(!packets.lock().unwrap().is_empty());
    }
}
