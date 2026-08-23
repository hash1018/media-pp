use std::sync::Arc;

use crate::pp_log::{PpLog, pp_error, pp_info};
use ffmpeg_next as ffmpeg;
use ffmpeg_next::Rescale;
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
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
}

impl VideoCodec {
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
        }
    }
}

/// Construction-time options for [`SwEncoder::new`]. `width`/`height`/
/// `time_base` must already be known — same convention as
/// [`crate::elements::SwScaler`]/[`crate::elements::Pacer`] — rather than
/// inferred from the first frame, since `avcodec_open2` needs them set
/// before this can be opened at all.
#[derive(Debug, Clone, Copy)]
pub struct SwEncoderOptions {
    /// Compressed video codec to open.
    pub codec: VideoCodec,
    /// Encoded frame width in pixels.
    pub width: u32,
    /// Encoded frame height in pixels.
    pub height: u32,
    /// Must match the `pts` unit of whatever frames this receives — e.g.
    /// [`crate::elements::TestVideoSource::time_base`] or a demuxed
    /// stream's own `time_base` if this is a transcode pipeline.
    /// Deliberately *not* used to derive [`SwEncoderOptions::frame_rate`]
    /// (an earlier version of this did exactly that) — the two aren't the
    /// same thing. A pts tick unit fine enough for accurate timestamps
    /// (say, microseconds) doesn't mean a million frames actually happen
    /// per second, which `1 / time_base` would wrongly claim.
    pub time_base: ffmpeg::Rational,
    /// The nominal rate the encoder uses for internal rate-control
    /// (targeting `bit_rate` per frame) and the frame-rate metadata it
    /// writes into the bitstream — *not* required to match the real
    /// interval between `send_frame` calls. For a source with its own
    /// genuinely fixed rate (e.g. [`crate::elements::TestVideoSource`]),
    /// that's `TestVideoOptions::framerate` itself. For an irregular/VFR
    /// source (e.g. `DxgiCaptureSource`, whose frames
    /// arrive only on real desktop changes, capped but not paced to a
    /// fixed cadence), use its configured cap
    /// (`DxgiCaptureOptions::max_fps`) as the closest meaningful nominal
    /// rate — actual encoded packets still carry each frame's real `pts`
    /// either way, so muxing stays correct regardless of how well this
    /// nominal rate matches the true one.
    pub frame_rate: ffmpeg::Rational,
    /// Target encoded bit rate, in bits per second.
    pub bit_rate: usize,
    /// How many frames between keyframes — `AVCodecContext.gop_size`
    /// directly (not a duration; multiply by `frame_rate` yourself, e.g.
    /// `frame_rate * 2` for "roughly every 2 seconds"). Not every codec's
    /// own default is a periodic interval at all — `libopenh264` was
    /// found, building [`crate::elements::SegmentedMp4Muxer`], to rely on
    /// scene-change detection alone and go an *entire* recording without
    /// a second keyframe against smoothly-changing content — so this is
    /// always set explicitly rather than left to whatever a given codec
    /// happens to default to. Matters beyond segmenting a recording, too:
    /// [`crate::elements::RtspSink`]/`WebRtcTrackSink`
    /// viewers/peers joining mid-stream can't decode anything until the
    /// next keyframe, so an unbounded interval is a real join-latency
    /// problem, not just a segmenting one.
    pub gop_size: u32,
}

/// Encodes `Pixel::YUV420P` `Video` frames into `Packet`s via a software
/// encoder (see [`VideoCodec`]) — the mirror image of
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
    /// The unit each produced packet's `pts` is expressed in — same value
    /// `SwEncoderOptions::time_base` was constructed with. Stamped onto
    /// every packet in `drain` since `avcodec_receive_packet` itself never
    /// sets `AVPacket.time_base` (only the encoder context's own time base,
    /// via `set_time_base` below); without it, a packet's own
    /// `Packet::time_base()` reads back FFmpeg's `0/1` "unset" sentinel —
    /// wrong for anything (e.g. `WebRtcPeer::write_track`) that derives
    /// real time from a packet's own declared time_base rather than
    /// external knowledge of what this encoder was built with.
    time_base: ffmpeg::Rational,
    pad: SrcPad,
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
        let encoder_name = options.codec.encoder_name();
        let codec = ffmpeg::encoder::find_by_name(encoder_name)
            .ok_or_else(|| SwEncoderError::CodecNotFound(encoder_name.into()))?;

        let mut context = ffmpeg::codec::context::Context::new_with_codec(codec);
        context.set_time_base(options.time_base);

        let mut video = context.encoder().video().map_err(SwEncoderError::from)?;
        video.set_width(options.width);
        video.set_height(options.height);
        video.set_format(ffmpeg::format::Pixel::YUV420P);
        video.set_time_base(options.time_base);
        video.set_frame_rate(Some(options.frame_rate));
        video.set_bit_rate(options.bit_rate);
        video.set_gop(options.gop_size);

        let encoder = video.open_as(codec).map_err(SwEncoderError::from)?;
        let packet_duration = nominal_packet_duration(options.time_base, options.frame_rate);

        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::SwEncoder, &name, None);
        let pad = SrcPad::with_contract(
            format!("{name}_src"),
            OutputContract::Fixed(PortContract::of(MediaKind::VideoPacket)),
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
            time_base: options.time_base,
            pad,
        })
    }

    /// This encoder's own codec parameters — what you need to construct
    /// a matching [`crate::elements::SwDecoder`] to decode the `Packet`s
    /// this produces, or what a [`crate::elements::Mp4Muxer`] track needs
    /// (same pattern [`crate::elements::SwAudioEncoder::parameters`]
    /// documents for audio), when there's no container/demuxer in the loop
    /// to get them from otherwise (e.g. encoding straight into a `Tee`/RTSP
    /// sink, or decoding straight back out for a round-trip smoke test).
    pub fn parameters(&self) -> ffmpeg::codec::Parameters {
        ffmpeg::codec::Parameters::from(&self.encoder)
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
        InputContract::Fixed(
            PortContract::of(MediaKind::VideoFrame).in_memory(MemoryDomain::System),
        )
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Video(frame) => {
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
    use crate::pool::UnboundObjectPool;

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
                    time_base: ffmpeg::Rational::new(1, 30),
                    frame_rate: ffmpeg::Rational::new(30, 1),
                    bit_rate: 1_000_000,
                    gop_size: 60, // ~2s @ 30fps
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

    fn pooled_video(frame: ffmpeg::frame::Video) -> MediaBuffer {
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut pooled = pool.get();
        *pooled = frame;
        MediaBuffer::Video(Arc::new(pooled))
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
    fn produced_packets_carry_the_configured_time_base() {
        let options = SwEncoderOptions {
            codec: VideoCodec::OpenH264,
            width: 64,
            height: 64,
            time_base: ffmpeg::Rational::new(1, 30),
            frame_rate: ffmpeg::Rational::new(30, 1),
            bit_rate: 200_000,
            gop_size: 30,
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
        for plane in 0..frame.planes() {
            frame.data_mut(plane).fill(128);
        }
        encoder.consume(pooled_video(frame)).unwrap();
        encoder.consume(MediaBuffer::Eos).unwrap();

        let packets = packets.lock().unwrap();
        assert!(!packets.is_empty(), "expected at least one packet");
        for packet in packets.iter() {
            assert_eq!(
                packet.time_base(),
                ffmpeg::Rational::new(1, 30),
                "packet time_base must match the encoder's configured time_base, \
                 not FFmpeg's 0/1 unset sentinel"
            );
        }
    }
}
