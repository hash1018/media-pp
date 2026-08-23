use std::sync::Arc;

use crate::pp_log::{PpLog, pp_error, pp_info};
use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    elements::AudioFormat,
    error::Result,
    pad::SrcPad,
    time::{InvalidTimeBase, MediaTimestamp, TimeBase},
};

/// The reusable `libswresample` state shared by [`AudioResampler`] and
/// [`crate::elements::SwAudioEncoder`]. It owns the otherwise easy-to-get-
/// subtly-wrong parts of resampling: rebuilding when the input definition
/// changes, draining the old context first, sizing output for upsampling,
/// and flushing delayed samples at EOS.
pub(crate) struct AudioFrameResampler {
    target: AudioFormat,
    context: Option<ffmpeg::software::resampling::Context>,
}

impl AudioFrameResampler {
    pub(crate) fn new(target: AudioFormat) -> Self {
        Self {
            target,
            context: None,
        }
    }

    fn context_matches(&self, frame: &ffmpeg::frame::Audio) -> bool {
        let Some(context) = &self.context else {
            return false;
        };
        let input = context.input();
        input.format == frame.format()
            && input.channel_layout == frame.channel_layout()
            && input.rate == frame.rate()
    }

    pub(crate) fn run(
        &mut self,
        frame: &ffmpeg::frame::Audio,
    ) -> std::result::Result<Vec<ffmpeg::frame::Audio>, ffmpeg::Error> {
        let mut output_frames = Vec::new();
        if !self.context_matches(frame) {
            output_frames.extend(self.flush()?);
            self.context = Some(ffmpeg::software::resampling::Context::get(
                frame.format(),
                frame.channel_layout(),
                frame.rate(),
                self.target.sample_format,
                self.target.channel_layout(),
                self.target.sample_rate,
            )?);
        }

        let context = self
            .context
            .as_mut()
            .expect("a resampling context was built above");
        let input_rate = frame.rate().max(1) as usize;
        let scaled_samples = frame
            .samples()
            .saturating_mul(self.target.sample_rate as usize)
            .div_ceil(input_rate);
        let delayed_samples = context
            .delay()
            .map(|delay| delay.output.max(0) as usize)
            .unwrap_or(0);
        let capacity = scaled_samples
            .saturating_add(delayed_samples)
            .saturating_add(32)
            .max(1);
        let mut output = ffmpeg::frame::Audio::new(
            self.target.sample_format,
            capacity,
            self.target.channel_layout(),
        );
        context.run(frame, &mut output)?;
        if output.samples() > 0 {
            output_frames.push(output);
        }
        Ok(output_frames)
    }

    pub(crate) fn flush(
        &mut self,
    ) -> std::result::Result<Vec<ffmpeg::frame::Audio>, ffmpeg::Error> {
        let Some(mut context) = self.context.take() else {
            return Ok(Vec::new());
        };
        let mut frames = Vec::new();
        loop {
            let capacity = context
                .delay()
                .map(|delay| delay.output.max(0) as usize)
                .unwrap_or(0)
                .max(1024);
            let mut output = ffmpeg::frame::Audio::new(
                self.target.sample_format,
                capacity,
                self.target.channel_layout(),
            );
            let delay = context.flush(&mut output)?;
            if output.samples() > 0 {
                frames.push(output);
            }
            if delay.is_none() {
                return Ok(frames);
            }
        }
    }

    pub(crate) fn reset(&mut self) {
        self.context = None;
    }
}

/// Errors specific to [`AudioResampler`].
#[derive(Debug, ThisError)]
pub enum AudioResamplerError {
    /// FFmpeg rejected creation or use of the resampling context.
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),

    /// The sink received a buffer other than decoded audio or end-of-stream.
    #[error(
        "AudioResampler only converts decoded Audio frames, got a {0}; link it after an audio decoder or source"
    )]
    UnsupportedBuffer(&'static str),

    /// The supplied input timestamp unit has a non-positive component.
    #[error(
        "invalid input time base {numerator}/{denominator}: both numerator and denominator must be positive"
    )]
    InvalidTimeBase {
        /// Invalid rational numerator.
        numerator: i32,
        /// Invalid rational denominator.
        denominator: i32,
    },
}

/// Converts decoded audio to one fixed [`AudioFormat`] via
/// `libswresample`. The input definition is learned from each frame and the
/// conversion context is rebuilt if it changes mid-stream.
///
/// Output timestamps use `1 / target.sample_rate` units and remain
/// contiguous across resampler buffering. The first output is anchored to
/// the first input frame's PTS, rescaled from the explicitly supplied
/// input time base. A decoded frame does not carry that unit itself, so
/// callers must pass the originating stream/source time base.
pub struct AudioResampler {
    pp_log: PpLog,
    name: Arc<str>,
    target: AudioFormat,
    input_time_base: TimeBase,
    resampler: AudioFrameResampler,
    next_pts: Option<i64>,
    pad: SrcPad,
}

impl AudioResampler {
    /// Creates a resampler with fixed output format and explicit input timestamp units.
    pub fn new(
        name: impl Into<String>,
        target: AudioFormat,
        input_time_base: ffmpeg::Rational,
    ) -> std::result::Result<Self, AudioResamplerError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::AudioResampler, &name, None);
        let pad = SrcPad::with_contract(
            format!("{name}_src"),
            OutputContract::Fixed(
                PortContract::of(MediaKind::Audio).in_memory(MemoryDomain::System),
            ),
        );
        pp_info!(
            pp_log: &pp_log,
            "created: {}Hz, {} channel(s), format={:?}",
            target.sample_rate,
            target.channels(),
            target.sample_format
        );
        let input_time_base = TimeBase::try_new(input_time_base).map_err(
            |InvalidTimeBase {
                 numerator,
                 denominator,
             }| AudioResamplerError::InvalidTimeBase {
                numerator,
                denominator,
            },
        )?;
        Ok(Self {
            name,
            pp_log,
            target,
            input_time_base,
            resampler: AudioFrameResampler::new(target),
            next_pts: None,
            pad,
        })
    }

    /// Returns the fixed output audio format.
    pub fn format(&self) -> AudioFormat {
        self.target
    }

    /// Returns the output PTS time base, `1 / sample_rate`.
    pub fn time_base(&self) -> ffmpeg::Rational {
        ffmpeg::Rational::new(1, self.target.sample_rate as i32)
    }

    fn anchor_pts(&mut self, input: &ffmpeg::frame::Audio) {
        if self.next_pts.is_some() {
            return;
        }
        let pts = input
            .pts()
            .map(|pts| {
                MediaTimestamp::new_unchecked(pts, self.input_time_base).rescale(
                    TimeBase::new_unchecked(ffmpeg::Rational::new(
                        1,
                        self.target.sample_rate as i32,
                    )),
                )
            })
            .unwrap_or(0);
        self.next_pts = Some(pts);
    }

    fn push_frames(&mut self, frames: Vec<ffmpeg::frame::Audio>) -> Result<()> {
        for mut frame in frames {
            let pts = self.next_pts.get_or_insert(0);
            frame.set_rate(self.target.sample_rate);
            frame.set_pts(Some(*pts));
            *pts += frame.samples() as i64;
            self.pad.push(MediaBuffer::Audio(Arc::new(frame)))?;
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.resampler.reset();
        self.next_pts = None;
    }
}

impl Element for AudioResampler {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::AudioResampler
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for AudioResampler {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for AudioResampler {
    /// Converts one audio layout to another; the rate and format it changes are runtime values, not part of this.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::of(MediaKind::Audio).in_memory(MemoryDomain::System))
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Audio(frame) => {
                self.anchor_pts(&frame);
                let frames = self
                    .resampler
                    .run(&frame)
                    .inspect_err(|error| pp_error!(self, "resample failed: {error}"))
                    .map_err(AudioResamplerError::from)?;
                self.push_frames(frames)
            }
            MediaBuffer::Eos => {
                let frames = self
                    .resampler
                    .flush()
                    .inspect_err(|error| pp_error!(self, "resampler flush failed: {error}"))
                    .map_err(AudioResamplerError::from)?;
                self.push_frames(frames)?;
                self.pad.push(MediaBuffer::Eos)
            }
            MediaBuffer::Packet(_) => Err(AudioResamplerError::UnsupportedBuffer("Packet").into()),
            MediaBuffer::Video(_) => Err(AudioResamplerError::UnsupportedBuffer("Video").into()),
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        if matches!(msg, ControlMsg::Seek(_) | ControlMsg::Stop) {
            self.reset();
        }
        self.pad.control(msg)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use ffmpeg::format::sample::Type;

    use super::*;

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

    fn f32_packed_frame(rate: u32, channels: u16, samples: usize, pts: i64) -> MediaBuffer {
        let format = ffmpeg::format::Sample::F32(Type::Packed);
        let layout = ffmpeg::ChannelLayout::default(i32::from(channels));
        let mut frame = ffmpeg::frame::Audio::new(format, samples, layout);
        frame.set_rate(rate);
        frame.set_pts(Some(pts));
        frame.data_mut(0).fill(0);
        MediaBuffer::Audio(Arc::new(frame))
    }

    fn new_resampler(target: AudioFormat) -> (AudioResampler, Arc<Mutex<Vec<MediaBuffer>>>) {
        let mut resampler =
            AudioResampler::new("resampler", target, ffmpeg::Rational::new(1, 48_000)).unwrap();
        let received = Arc::new(Mutex::new(Vec::new()));
        resampler.src_pads()[0].link(Box::new(CapturingSink {
            received: received.clone(),
            pp_log: element_pp_log(ElementType::Other, "capture", None),
        }));
        (resampler, received)
    }

    #[test]
    fn converts_rate_channels_and_timestamps_then_flushes_eos() {
        let target = AudioFormat::new(ffmpeg::format::Sample::F32(Type::Packed), 24_000, 1);
        let (mut resampler, received) = new_resampler(target);
        resampler
            .consume(f32_packed_frame(48_000, 2, 960, 48_000))
            .unwrap();
        resampler.consume(MediaBuffer::Eos).unwrap();

        let received = received.lock().unwrap();
        let audio: Vec<_> = received
            .iter()
            .filter_map(|buffer| match buffer {
                MediaBuffer::Audio(frame) => Some(frame),
                _ => None,
            })
            .collect();
        assert!(!audio.is_empty());
        assert_eq!(audio[0].pts(), Some(24_000));
        assert!(
            audio
                .iter()
                .all(|frame| frame.format() == target.sample_format)
        );
        assert!(audio.iter().all(|frame| frame.rate() == 24_000));
        assert!(audio.iter().all(|frame| frame.channels() == 1));
        for pair in audio.windows(2) {
            assert_eq!(
                pair[1].pts(),
                pair[0].pts().map(|pts| pts + pair[0].samples() as i64)
            );
        }
        assert!(matches!(received.last(), Some(MediaBuffer::Eos)));
    }

    #[test]
    fn rebuilds_when_the_input_definition_changes() {
        let target = AudioFormat::new(ffmpeg::format::Sample::F32(Type::Packed), 48_000, 2);
        let (mut resampler, received) = new_resampler(target);
        resampler
            .consume(f32_packed_frame(48_000, 2, 480, 0))
            .unwrap();
        resampler
            .consume(f32_packed_frame(44_100, 1, 441, 441))
            .unwrap();
        resampler.consume(MediaBuffer::Eos).unwrap();

        assert!(received.lock().unwrap().iter().any(|buffer| {
            matches!(buffer, MediaBuffer::Audio(frame) if frame.rate() == 48_000 && frame.channels() == 2)
        }));
    }

    #[test]
    fn rejects_non_audio_buffers() {
        let target = AudioFormat::new(ffmpeg::format::Sample::F32(Type::Packed), 48_000, 2);
        let (mut resampler, _) = new_resampler(target);
        let error = resampler
            .consume(MediaBuffer::Packet(Arc::new(ffmpeg::Packet::empty())))
            .unwrap_err();
        assert!(matches!(
            error,
            crate::Error::AudioResamplerError(AudioResamplerError::UnsupportedBuffer("Packet"))
        ));
    }
}
