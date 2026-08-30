use std::{collections::VecDeque, sync::Arc};

use crate::pp_log::{PpLog, pp_error, pp_info};
use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    error::Result,
    pad::SrcPad,
};

use crate::elements::AudioFormat;
use crate::elements::filter::audio_resampler::AudioFrameResampler;
use crate::elements::filter::is_codec_drain_boundary;

/// Errors specific to `SwAudioEncoder`. Converts into the crate-wide
/// `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum SwAudioEncoderError {
    /// The requested encoder is unavailable in the linked FFmpeg build.
    #[error(
        "encoder {0:?} not found — this ffmpeg build wasn't compiled with it \
         (run `ffmpeg -encoders` to see what's actually available)"
    )]
    CodecNotFound(String),

    /// The sink received a buffer other than decoded audio or end-of-stream.
    #[error("SwAudioEncoder only accepts Audio or Eos buffers, got {0}")]
    UnsupportedBuffer(&'static str),

    /// [`SwAudioEncoder`] only ever resamples to `Sample::F32` (packed or
    /// planar, whichever the codec itself reports) — see its own docs on
    /// why. This is what a codec whose *only* supported formats are
    /// something else entirely (integer PCM, `f64`, ...) gets instead of
    /// a silently-wrong guess. Neither codec this crate opens today
    /// (`aac`, `libopus`) hits this.
    #[error("{0:?} doesn't support any Sample::F32 format — only {1:?} (unsupported)")]
    NoF32Format(String, Vec<ffmpeg::format::Sample>),

    /// The codec publishes a fixed list of sample rates and
    /// [`SwAudioEncoderOptions::sample_rate`] is not on it — `libopus`
    /// takes only 48/24/16/12/8 kHz, for example. Caught here because
    /// `avcodec_open2` reports it as a bare `Invalid argument` that names
    /// neither the rate nor the alternatives.
    #[error("{0:?} does not support {1} Hz — only {2:?}")]
    UnsupportedSampleRate(String, u32, Vec<u32>),

    /// FFmpeg rejected encoder, resampler, frame, or packet processing.
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),
}

/// Which software audio encoder to open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioCodec {
    /// FFmpeg's own native `aac` encoder — built in rather than an
    /// external library, so unlike [`AudioCodec::Opus`] it needs nothing
    /// added to the ffmpeg build (and no `--enable-gpl`, unlike
    /// [`crate::elements::VideoCodec::H264`]/
    /// [`crate::elements::VideoCodec::H265`]). The standard choice for
    /// MP4 audio.
    Aac,
    /// `libopus` — Opus (RFC 6716), BSD-3-Clause. A general-purpose
    /// codec covering speech and music alike, whose very short frames are
    /// why interactive systems standardized on it; WebRTC requires it,
    /// and Ogg, WebM/Matroska, and MP4 all carry it.
    ///
    /// Unlike [`AudioCodec::Aac`] this is an external library, so a
    /// stripped ffmpeg build may not have it; that surfaces as
    /// [`SwAudioEncoderError::CodecNotFound`]. It accepts only
    /// 48/24/16/12/8 kHz — see
    /// [`SwAudioEncoderError::UnsupportedSampleRate`].
    Opus,
}

impl AudioCodec {
    fn encoder_name(self) -> &'static str {
        match self {
            AudioCodec::Aac => "aac",
            AudioCodec::Opus => "libopus",
        }
    }
}

/// Construction-time options for [`SwAudioEncoder::new`]. `sample_rate`/
/// `channels`/`time_base` must already be known — same convention as
/// [`crate::elements::SwEncoder`]'s own `width`/`height`/`time_base` —
/// rather than inferred from the first frame, since `avcodec_open2` needs
/// them set before this can be opened at all. What's actually *fed* to
/// this element doesn't have to match `sample_rate`/`channels` exactly —
/// see [`SwAudioEncoder`]'s own docs on why.
#[derive(Debug, Clone, Copy)]
pub struct SwAudioEncoderOptions {
    /// Compressed audio codec to open.
    pub codec: AudioCodec,
    /// Encoder output sample rate, in hertz. A codec that accepts only
    /// certain rates rejects the rest at construction with
    /// [`SwAudioEncoderError::UnsupportedSampleRate`] —
    /// [`AudioCodec::Opus`] takes 48/24/16/12/8 kHz and nothing else,
    /// where [`AudioCodec::Aac`] takes any.
    pub sample_rate: u32,
    /// Encoder output channel count.
    pub channels: u16,
    /// Must match the `pts` unit of whatever frames this receives — e.g.
    /// [`crate::elements::TestAudioSource::time_base`]/
    /// [`crate::elements::AudioMixer::time_base`].
    pub time_base: ffmpeg::Rational,
    /// Target encoded bit rate, in bits per second.
    pub bit_rate: usize,
}

/// Encodes `Audio` frames into `Packet`s via a software encoder (see
/// [`AudioCodec`]) — the audio sibling of
/// [`crate::elements::SwEncoder`]. A `Filter`: receives via `Sink`,
/// pushes what it produces into its own (single) src pad.
///
/// Unlike `SwEncoder` (which requires exactly `Pixel::YUV420P` in and
/// errors on anything else), this resamples whatever it's fed —
/// building the resampler lazily from the first frame's own
/// `format`/`channel_layout`/`rate`, same pattern
/// [`crate::elements::AudioMixer`]'s own `InputBuffer::push` uses — to
/// whatever `sample_rate`/`channels` [`SwAudioEncoderOptions`] asked for,
/// in whichever `Sample::F32` layout (packed or planar) the opened codec
/// actually reports (queried via the codec's own capabilities, not
/// assumed — `aac`'s native encoder wants planar, but nothing here
/// hardcodes that). This is what lets a
/// [`crate::elements::TestAudioSource`]/[`crate::elements::AudioMixer`]
/// (both fixed `Sample::F32(Packed)`) feed this directly with no
/// separate resampler element in between, the same way `SwEncoder`
/// leaning on a fixed `Pixel::YUV420P` input still needs a
/// [`crate::elements::SwScaler`] in front of it for anything else.
///
/// Most audio codecs (`aac` included) require a *fixed* number of samples
/// per `send_frame` call (`encoder.frame_size()`) — not whatever size
/// happened to arrive. Resampled samples are buffered per-channel
/// (`pending`) and only handed to the encoder once a full
/// `frame_size`-sample chunk is ready; a codec that reports `frame_size
/// == 0` (accepts any size) skips the buffering and encodes each
/// resampled chunk as-is. `Eos` flushes both the resampler's own internal
/// delay line (a few samples of filter latency — see
/// [`ffmpeg::software::resampling::Context::flush`]) and whatever's left
/// in `pending` (as one final, possibly short, frame — allowed for the
/// last frame only) before flushing the encoder itself.
pub struct SwAudioEncoder {
    pp_log: PpLog,
    name: Arc<str>,
    encoder: ffmpeg::encoder::audio::Encoder,
    pad: SrcPad,
    target_format: ffmpeg::format::Sample,
    target_layout: ffmpeg::ChannelLayout,
    sample_rate: u32,
    /// `0` means the codec accepts any chunk size — see this struct's
    /// own docs.
    frame_size: usize,
    /// Shared resampling engine also used by `AudioResampler`; it builds
    /// lazily from the first input frame and flushes/rebuilds if the input
    /// definition changes.
    resampler: AudioFrameResampler,
    /// One queue per channel, holding already-resampled (`target_format`'s
    /// sample type, always `f32` — see [`SwAudioEncoderError::NoF32Format`])
    /// samples waiting for a full `frame_size`-sample chunk. Channel-major
    /// regardless of whether `target_format` itself is packed or planar —
    /// [`SwAudioEncoder::absorb_resampled`]/[`SwAudioEncoder::drain_pending`]
    /// are the only two places that ever reshape into/out of
    /// `target_format`'s actual on-the-wire layout.
    pending: Vec<VecDeque<f32>>,
    /// Cumulative sample count across every frame handed to the encoder —
    /// this element's `pts` unit (see [`SwAudioEncoder::time_base`]).
    samples_emitted: i64,
}

// SAFETY: see `AudioMixer`'s own `unsafe impl Send` docs — same
// reasoning, `target_layout` here is always `ChannelLayout::default`'s
// plain native layout (built in `SwAudioEncoder::new`), never a custom
// one with a real pointer in it.
unsafe impl Send for SwAudioEncoder {}

impl SwAudioEncoder {
    /// Opens the requested audio encoder and configures its output definition.
    pub fn new(name: impl Into<String>, options: SwAudioEncoderOptions) -> Result<Self> {
        let encoder_name = options.codec.encoder_name();
        let codec = ffmpeg::encoder::find_by_name(encoder_name)
            .ok_or_else(|| SwAudioEncoderError::CodecNotFound(encoder_name.into()))?;

        // Whichever `Sample::F32` layout the codec itself reports —
        // preferring planar (`aac`'s own native format) if both are
        // available, but not assuming either — see this struct's own
        // docs on why.
        let target_format = {
            let capabilities = codec.audio().map_err(SwAudioEncoderError::from)?;
            let available: Vec<ffmpeg::format::Sample> =
                capabilities.formats().into_iter().flatten().collect();
            available
                .iter()
                .copied()
                .find(|f| {
                    matches!(
                        f,
                        ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Planar)
                    )
                })
                .or_else(|| {
                    available
                        .iter()
                        .copied()
                        .find(|f| matches!(f, ffmpeg::format::Sample::F32(_)))
                })
                .ok_or_else(|| {
                    SwAudioEncoderError::NoF32Format(encoder_name.into(), available.clone())
                })?
        };

        // Asked of the codec rather than hardcoded per `AudioCodec`, the
        // same way the format above is. A codec that publishes no list
        // (`aac`) takes any rate and is left alone; one that does
        // (`libopus`) would otherwise fail inside `avcodec_open2` with a
        // bare `Invalid argument`.
        {
            let capabilities = codec.audio().map_err(SwAudioEncoderError::from)?;
            if let Some(rates) = capabilities.rates() {
                let supported: Vec<u32> = rates.map(|rate| rate as u32).collect();
                if !supported.contains(&options.sample_rate) {
                    return Err(SwAudioEncoderError::UnsupportedSampleRate(
                        encoder_name.into(),
                        options.sample_rate,
                        supported,
                    )
                    .into());
                }
            }
        }
        let target_layout = ffmpeg::ChannelLayout::default(options.channels as i32);

        let mut context = ffmpeg::codec::context::Context::new_with_codec(codec);
        // Codec headers into `extradata` for the container to write, not only
        // in-band — see `SwEncoder::new`, which says why in full.
        context.set_flags(ffmpeg::codec::Flags::GLOBAL_HEADER);
        context.set_time_base(options.time_base);

        let mut audio = context
            .encoder()
            .audio()
            .map_err(SwAudioEncoderError::from)?;
        audio.set_rate(options.sample_rate as i32);
        audio.set_channel_layout(target_layout);
        audio.set_format(target_format);
        audio.set_time_base(options.time_base);
        audio.set_bit_rate(options.bit_rate);

        let encoder = audio.open_as(codec).map_err(SwAudioEncoderError::from)?;
        let frame_size = encoder.frame_size() as usize;

        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::SwAudioEncoder, &name, None);
        let pad = SrcPad::with_contract(
            format!("{name}_src"),
            OutputContract::Fixed(PortContract::packet(MediaKind::AudioPacket)),
        );
        pp_info!(
            pp_log: &pp_log,
            "opened: codec={encoder_name}, {}Hz, {} channel(s), format={:?}, frame_size={}, bit_rate={}",
            options.sample_rate,
            options.channels,
            target_format,
            frame_size,
            options.bit_rate
        );
        Ok(Self {
            name,
            pp_log,
            encoder,
            pad,
            target_format,
            target_layout,
            sample_rate: options.sample_rate,
            frame_size,
            resampler: AudioFrameResampler::new(AudioFormat {
                sample_format: target_format,
                sample_rate: options.sample_rate,
                channels: options.channels,
            }),
            pending: (0..options.channels).map(|_| VecDeque::new()).collect(),
            samples_emitted: 0,
        })
    }

    /// This encoder's own codec parameters — what a
    /// [`crate::elements::FileMuxer`] track needs, same pattern
    /// [`crate::elements::SwEncoder::parameters`] documents for video.
    pub fn parameters(&self) -> ffmpeg::codec::Parameters {
        ffmpeg::codec::Parameters::from(&self.encoder)
    }

    /// The unit each produced packet's `pts` is expressed in.
    pub fn time_base(&self) -> ffmpeg::Rational {
        ffmpeg::Rational::new(1, self.sample_rate as i32)
    }

    /// Resamples `frame` through the same input-change-aware engine used by
    /// `AudioResampler` and appends every produced frame onto `pending`,
    /// channel-major.
    fn resample(&mut self, frame: &ffmpeg::frame::Audio) -> std::result::Result<(), ffmpeg::Error> {
        for output in self.resampler.run(frame)? {
            absorb_resampled(&output, self.target_format, &mut self.pending);
        }
        Ok(())
    }

    /// Encodes every full `frame_size`-sample chunk currently sitting in
    /// `pending` (all of it, if the codec accepts any size —
    /// `frame_size == 0`). Leaves a short remainder in `pending` for the
    /// next call, except when `flush_all` (only `Eos` sets this) forces
    /// out whatever's left as one final, possibly-short frame — valid for
    /// the last frame a codec ever sees, not for any frame before it.
    fn drain_pending(&mut self, flush_all: bool) -> Result<()> {
        loop {
            let available = self.pending.iter().map(VecDeque::len).min().unwrap_or(0);
            let take = if self.frame_size == 0 {
                available
            } else if available >= self.frame_size {
                self.frame_size
            } else if flush_all {
                available
            } else {
                0
            };
            if take == 0 {
                return Ok(());
            }

            let mut frame = ffmpeg::frame::Audio::new(self.target_format, take, self.target_layout);
            frame.set_rate(self.sample_rate);
            write_channels(&mut frame, self.target_format, &mut self.pending, take);
            frame.set_pts(Some(self.samples_emitted));
            self.samples_emitted += take as i64;

            self.encoder
                .send_frame(&frame)
                .inspect_err(|error| pp_error!(self, "send_frame failed: {error}"))
                .map_err(SwAudioEncoderError::from)?;
            self.drain_packets()?;

            if flush_all && take < self.frame_size.max(1) {
                return Ok(()); // that was the short final chunk — nothing more to take
            }
        }
    }

    fn drain_packets(&mut self) -> Result<()> {
        let mut packet = ffmpeg::Packet::empty();
        loop {
            match self.encoder.receive_packet(&mut packet) {
                Ok(()) => {
                    // `avcodec_receive_packet` never sets `AVPacket.time_base`
                    // itself — without this, `packet.time_base()` reads back
                    // FFmpeg's `0/1` "unset" sentinel, wrong for anything
                    // (e.g. `WebRtcPeer::write_track`) deriving real time
                    // from a packet's own declared time_base. `pts` above is
                    // already expressed in this same `1/sample_rate` unit.
                    packet.set_time_base(self.time_base());
                    self.pad.push(MediaBuffer::Packet(Arc::new(packet)))?;
                    packet = ffmpeg::Packet::empty();
                }
                Err(error) if is_codec_drain_boundary(&error) => break,
                Err(error) => return Err(SwAudioEncoderError::from(error).into()),
            }
        }
        Ok(())
    }

    /// Drains the resampler's own internal delay line (a few samples of
    /// filter latency, present even after every real input frame has
    /// already been resampled) into `pending` — see
    /// [`ffmpeg::software::resampling::Context::flush`]'s own docs.
    /// A no-op if the resampler was never built (nothing was ever fed).
    fn flush_resampler(&mut self) -> std::result::Result<(), ffmpeg::Error> {
        for output in self.resampler.flush()? {
            absorb_resampled(&output, self.target_format, &mut self.pending);
        }
        Ok(())
    }
}

/// Copies `output`'s samples onto the end of `pending` (one queue per
/// channel), reading `output` according to `format`'s actual packed/planar
/// layout — the only place resampler output ever gets de-interleaved.
///
/// Two *different* footguns in `ffmpeg_next`'s `frame::Audio` accessors,
/// one per layout, are why this isn't just one uniform read:
///
/// - **Planar**: must use `plane::<f32>(channel)`, not `data(channel)`.
///   `data()`'s length comes from `AVFrame.linesize[channel]` — but
///   FFmpeg's own convention for planar *audio* only ever fills in
///   `linesize[0]` (every plane is the same size, so it doesn't bother
///   repeating it); `linesize[1..]` stay `0`, so `data(1)`, `data(2)`, ...
///   always report a zero-length slice regardless of the real buffer.
///   `plane::<f32>()` sidesteps this — its length comes from
///   `self.samples()` instead, correct for planar regardless of channel
///   index.
/// - **Packed, multi-channel**: the opposite problem — must use
///   `data(0)` (raw bytes), not `plane::<f32>(0)`. `plane::<T>()` always
///   returns exactly `self.samples()` elements of type `T`, correct for
///   planar (one channel per plane) or packed **mono**, but for packed
///   multi-channel the real buffer holds `samples() * channels`
///   interleaved scalars — `plane::<f32>(0)` on packed stereo (or wider)
///   silently reads only the first `samples()` of them.
///
/// (Packed mono happens to work with either accessor; not treated as a
/// third case since packed multi-channel's `data()` path already
/// generalizes to it — `channels == 1` makes the two identical.)
fn absorb_resampled(
    output: &ffmpeg::frame::Audio,
    format: ffmpeg::format::Sample,
    pending: &mut [VecDeque<f32>],
) {
    let samples = output.samples();
    if samples == 0 {
        return;
    }
    if format.is_planar() {
        for (channel, queue) in pending.iter_mut().enumerate() {
            queue.extend(output.plane::<f32>(channel).iter().copied());
        }
    } else {
        let channels = pending.len();
        let bytes = &output.data(0)[..samples * channels * 4];
        for (index, &sample) in bytes_as_f32(bytes).iter().enumerate() {
            pending[index % channels].push_back(sample);
        }
    }
}

/// Pops exactly `take` samples off the front of every channel in `pending`
/// into `frame`, writing according to `format`'s actual packed/planar
/// layout — the only place `pending` ever gets re-interleaved. Same two
/// accessors, same reasons, as [`absorb_resampled`]'s own docs — just the
/// `_mut`/write-direction versions (`plane_mut::<f32>()` for planar,
/// `data_mut(0)` raw bytes for packed multi-channel).
fn write_channels(
    frame: &mut ffmpeg::frame::Audio,
    format: ffmpeg::format::Sample,
    pending: &mut [VecDeque<f32>],
    take: usize,
) {
    if format.is_planar() {
        for (channel, queue) in pending.iter_mut().enumerate() {
            let plane = frame.plane_mut::<f32>(channel);
            for (slot, sample) in plane.iter_mut().zip(queue.drain(0..take)) {
                *slot = sample;
            }
        }
    } else {
        let channels = pending.len();
        let bytes = &mut frame.data_mut(0)[..take * channels * 4];
        let floats = bytes_as_f32_mut(bytes);
        for row in 0..take {
            for (channel, queue) in pending.iter_mut().enumerate() {
                floats[row * channels + channel] = queue
                    .pop_front()
                    .expect("every channel's queue was already confirmed to have >= take samples");
            }
        }
    }
}

/// Reinterprets `bytes` (a tight, `f32`-aligned slice out of an
/// `ffmpeg::frame::Audio`'s own `Sample::F32` plane) as `f32`s.
///
/// # Safety requirement met by every caller
/// `bytes` must come from a `Sample::F32` plane (`data`/`data_mut`),
/// sliced to a whole number of `f32`s — both `absorb_resampled` and
/// `write_channels` only ever pass byte ranges sized as an exact multiple
/// of 4. FFmpeg allocates audio buffers with SIMD-friendly (well beyond
/// 4-byte) alignment, so the cast itself is always valid.
fn bytes_as_f32(bytes: &[u8]) -> &[f32] {
    // SAFETY: the alignment and length reasoning is in this function's own doc
    // comment above: FFmpeg's audio buffers are aligned well past 4, and the
    // length is rounded down to whole `f32`s.
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4) }
}

fn bytes_as_f32_mut(bytes: &mut [u8]) -> &mut [f32] {
    // SAFETY: as `bytes_as_f32`, and `&mut [u8]` is exclusive so the `&mut
    // [f32]` handed back cannot alias anything.
    unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut f32, bytes.len() / 4) }
}

impl Element for SwAudioEncoder {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::SwAudioEncoder
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for SwAudioEncoder {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for SwAudioEncoder {
    /// The audio mirror of SwEncoder: decoded samples in, encoded packets out.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::frame(
            MediaKind::AudioFrame,
            MemoryDomain::System,
        ))
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Audio(frame) => {
                self.resample(&frame)
                    .inspect_err(|error| pp_error!(self, "resample failed: {error}"))
                    .map_err(SwAudioEncoderError::from)?;
                self.drain_pending(false)
            }
            MediaBuffer::Eos => {
                self.flush_resampler()
                    .inspect_err(|error| pp_error!(self, "resampler flush failed: {error}"))
                    .map_err(SwAudioEncoderError::from)?;
                self.drain_pending(true)?;
                self.encoder
                    .send_eof()
                    .inspect_err(|error| pp_error!(self, "send_eof failed: {error}"))
                    .map_err(SwAudioEncoderError::from)?;
                self.drain_packets()?;
                self.pad.push(MediaBuffer::Eos)
            }
            other => Err(SwAudioEncoderError::UnsupportedBuffer(other.kind()).into()),
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        // Same reasoning as `SwEncoder::control`: `Seek` is forwarded
        // without flushing (delayed/reordered packets from before the
        // seek can still surface later), `Stop` abandons without
        // flushing.
        self.pad.control(msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `SwAudioEncoder::new` should fail cleanly (not panic) when the
    /// linked ffmpeg build wasn't compiled with `aac` — vanishingly
    /// unlikely (it's ffmpeg's own native encoder, not an optional
    /// external library), but the contract should hold regardless of
    /// which build runs this test, same reasoning as `SwEncoder`'s own
    /// `codec_not_found_is_a_clean_error_not_a_panic`.
    #[test]
    fn codec_not_found_is_a_clean_error_not_a_panic() {
        let result = SwAudioEncoder::new(
            "encoder",
            SwAudioEncoderOptions {
                codec: AudioCodec::Aac,
                sample_rate: 48000,
                channels: 2,
                time_base: ffmpeg::Rational::new(1, 48000),
                bit_rate: 128_000,
            },
        );
        if let Err(error) = result {
            assert!(
                matches!(
                    error,
                    crate::error::Error::SwAudioEncoderError(SwAudioEncoderError::CodecNotFound(_))
                ),
                "expected CodecNotFound, got {error:?}"
            );
        }
    }

    struct RecordingSink {
        pp_log: PpLog,
        packets: Arc<std::sync::Mutex<usize>>,
        eos: Arc<std::sync::atomic::AtomicBool>,
    }

    impl Element for RecordingSink {
        fn name(&self) -> Arc<str> {
            "recorder".into()
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

    impl Sink for RecordingSink {
        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            match buf {
                MediaBuffer::Packet(_) => {
                    *self.packets.lock().unwrap() += 1;
                }
                MediaBuffer::Eos => {
                    self.eos.store(true, std::sync::atomic::Ordering::SeqCst);
                }
                _ => {}
            }
            Ok(())
        }
        fn control(&mut self, _msg: ControlMsg) -> Result<()> {
            Ok(())
        }
    }

    fn constant_stereo_frame(value: f32, samples: usize, rate: u32) -> ffmpeg::frame::Audio {
        let mut frame = ffmpeg::frame::Audio::new(
            ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed),
            samples,
            ffmpeg::ChannelLayout::default(2),
        );
        frame.set_rate(rate);
        let interleaved = vec![value; samples * 2];
        // SAFETY: viewing an `f32` slice as bytes, which is always aligned and
        // exactly `size_of_val` long.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                interleaved.as_ptr() as *const u8,
                std::mem::size_of_val(&*interleaved),
            )
        };
        frame.data_mut(0)[..bytes.len()].copy_from_slice(bytes);
        frame
    }

    /// Feeds ~400ms of 20ms `Sample::F32(Packed)` stereo chunks (the same
    /// shape `TestAudioSource`/`AudioMixer` actually produce) through a
    /// real `aac` encoder and checks packets come out, `Eos` cascades, and
    /// nothing panics along the way — the resampling/`frame_size` chunking
    /// logic is real new logic, not just a thin wrapper over
    /// `avcodec_open2` the way `codec_not_found_is_a_clean_error_not_a_panic`
    /// alone would exercise.
    #[test]
    fn encodes_generated_tone_into_packets_and_flushes_on_eos() {
        let Ok(mut encoder) = SwAudioEncoder::new(
            "encoder",
            SwAudioEncoderOptions {
                codec: AudioCodec::Aac,
                sample_rate: 48000,
                channels: 2,
                time_base: ffmpeg::Rational::new(1, 48000),
                bit_rate: 128_000,
            },
        ) else {
            eprintln!("skipping: aac not available in this ffmpeg build");
            return;
        };

        let packets = Arc::new(std::sync::Mutex::new(0usize));
        let eos = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let sink = RecordingSink {
            packets: packets.clone(),
            eos: eos.clone(),
            pp_log: element_pp_log(ElementType::Other, "recorder", None),
        };
        encoder.src_pads()[0].link(Box::new(sink));

        for _ in 0..20 {
            encoder
                .consume(MediaBuffer::Audio(Arc::new(constant_stereo_frame(
                    0.1, 960, 48000,
                ))))
                .expect("consume should succeed");
        }
        encoder
            .consume(MediaBuffer::Eos)
            .expect("consume(Eos) should succeed");

        assert!(
            *packets.lock().unwrap() > 0,
            "expected at least one encoded packet"
        );
        assert!(
            eos.load(std::sync::atomic::Ordering::SeqCst),
            "expected Eos to cascade"
        );
    }

    struct CapturingSink {
        pp_log: PpLog,
        packets: Arc<std::sync::Mutex<Vec<Arc<ffmpeg::Packet>>>>,
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
    /// `AVPacket.time_base` — only `drain_packets` stamping it explicitly
    /// (added alongside this test) keeps `Packet::time_base()` from
    /// reading back FFmpeg's `0/1` "unset" sentinel. That silently broke
    /// `WebRtcPeer::write_track`, which derives real RTP time from each
    /// packet's own declared time_base (`0/1`'s numerator of `0` fails its
    /// validation, so every packet was dropped) — see
    /// `webrtc_av_loopback`'s own regression run.
    #[test]
    fn produced_packets_carry_the_configured_time_base() {
        let Ok(mut encoder) = SwAudioEncoder::new(
            "encoder",
            SwAudioEncoderOptions {
                codec: AudioCodec::Aac,
                sample_rate: 48000,
                channels: 2,
                time_base: ffmpeg::Rational::new(1, 48000),
                bit_rate: 128_000,
            },
        ) else {
            eprintln!("skipping: aac not available in this ffmpeg build");
            return;
        };

        let packets = Arc::new(std::sync::Mutex::new(Vec::new()));
        encoder.src_pads()[0].link(Box::new(CapturingSink {
            packets: packets.clone(),
            pp_log: element_pp_log(ElementType::Other, "capture", None),
        }));

        for _ in 0..20 {
            encoder
                .consume(MediaBuffer::Audio(Arc::new(constant_stereo_frame(
                    0.1, 960, 48000,
                ))))
                .expect("consume should succeed");
        }
        encoder
            .consume(MediaBuffer::Eos)
            .expect("consume(Eos) should succeed");

        let packets = packets.lock().unwrap();
        assert!(!packets.is_empty(), "expected at least one packet");
        for packet in packets.iter() {
            assert_eq!(
                packet.time_base(),
                ffmpeg::Rational::new(1, 48000),
                "packet time_base must match SwAudioEncoder::time_base(), \
                 not FFmpeg's 0/1 unset sentinel"
            );
        }
    }

    fn opus_options(sample_rate: u32) -> SwAudioEncoderOptions {
        SwAudioEncoderOptions {
            codec: AudioCodec::Opus,
            sample_rate,
            channels: 2,
            time_base: ffmpeg::Rational::new(1, sample_rate as i32),
            bit_rate: 96_000,
        }
    }

    /// Opus exists here because WebRTC negotiates nothing else for audio,
    /// so what matters is that it actually produces packets and drains on
    /// `Eos` — the same contract `aac` has. `libopus` is an external
    /// library rather than one of ffmpeg's own, so a build without it
    /// skips instead of failing, the way a missing device does.
    #[test]
    fn opus_encodes_into_packets_and_flushes_on_eos() {
        let Ok(mut encoder) = SwAudioEncoder::new("encoder", opus_options(48000)) else {
            eprintln!("skipping: libopus not available in this ffmpeg build");
            return;
        };

        let packets = Arc::new(std::sync::Mutex::new(0usize));
        let eos = Arc::new(std::sync::atomic::AtomicBool::new(false));
        encoder.src_pads()[0].link(Box::new(RecordingSink {
            packets: packets.clone(),
            eos: eos.clone(),
            pp_log: element_pp_log(ElementType::Other, "recorder", None),
        }));

        for _ in 0..20 {
            encoder
                .consume(MediaBuffer::Audio(Arc::new(constant_stereo_frame(
                    0.1, 960, 48000,
                ))))
                .expect("consume should succeed");
        }
        encoder
            .consume(MediaBuffer::Eos)
            .expect("consume(Eos) should succeed");

        assert!(
            *packets.lock().unwrap() > 0,
            "expected at least one encoded packet"
        );
        assert!(
            eos.load(std::sync::atomic::Ordering::SeqCst),
            "expected Eos to cascade"
        );
    }

    /// Opus publishes a fixed rate list, and 44100 is deliberately not on
    /// it. The point of the check is the message: `avcodec_open2` rejects
    /// this too, but as a bare `Invalid argument` naming neither the rate
    /// nor what would have worked.
    #[test]
    fn opus_rejects_a_sample_rate_it_cannot_encode() {
        if ffmpeg::encoder::find_by_name("libopus").is_none() {
            eprintln!("skipping: libopus not available in this ffmpeg build");
            return;
        }

        let error = SwAudioEncoder::new("encoder", opus_options(44100))
            .err()
            .expect("44100 Hz must not open an Opus encoder");
        let crate::error::Error::SwAudioEncoderError(SwAudioEncoderError::UnsupportedSampleRate(
            _,
            rate,
            supported,
        )) = error
        else {
            panic!("expected UnsupportedSampleRate, got {error}");
        };

        assert_eq!(rate, 44100);
        assert!(
            supported.contains(&48000),
            "the reported alternatives should include Opus's own 48 kHz, got {supported:?}"
        );
    }

    /// The rate check must stay scoped to codecs that publish a list.
    /// `aac` publishes none and takes 44100 happily, so a check that
    /// rejected an empty list would break the codec this crate already
    /// shipped.
    #[test]
    fn aac_still_accepts_a_rate_no_codec_list_covers() {
        let result = SwAudioEncoder::new(
            "encoder",
            SwAudioEncoderOptions {
                codec: AudioCodec::Aac,
                sample_rate: 44100,
                channels: 2,
                time_base: ffmpeg::Rational::new(1, 44100),
                bit_rate: 128_000,
            },
        );

        assert!(
            result.is_ok(),
            "aac publishes no rate list, so no rate should be rejected here"
        );
    }
}
