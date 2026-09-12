use std::{sync::Arc, time::Duration};

use crate::pp_log::{PpLog, pp_info};
use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use super::audio_f32::{self, Unreadable, db_to_amplitude};
use super::tuning::Tuning;
use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    error::Result,
    pad::SrcPad,
};

/// How close to the gain it is releasing toward counts as there: -120 dB.
const SETTLED: f32 = 1e-6;

/// Settings for [`AudioLimiter`], and what [`AudioLimiterHandle::set_options`]
/// replaces them with — the two a streaming application's limiter asks for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioLimiterOptions {
    /// The level, in dBFS, nothing coming out goes over.
    pub threshold_db: f32,
    /// How quickly it lets go once the level falls back under.
    pub release: Duration,
}

impl Default for AudioLimiterOptions {
    fn default() -> Self {
        Self {
            threshold_db: -6.0,
            release: Duration::from_millis(60),
        }
    }
}

impl AudioLimiterOptions {
    fn validate(self) -> std::result::Result<Self, AudioLimiterError> {
        if !self.threshold_db.is_finite() {
            return Err(AudioLimiterError::InvalidThreshold(self.threshold_db));
        }
        Ok(self)
    }
}

/// Errors specific to [`AudioLimiter`].
#[derive(Debug, ThisError, PartialEq)]
pub enum AudioLimiterError {
    /// The threshold is NaN or infinite.
    #[error("a limiter threshold must be a finite level in dBFS, got {0}")]
    InvalidThreshold(f32),

    /// The input frame does not describe any audio channels.
    #[error("audio frame has no channel layout")]
    MissingChannels,

    /// The input is not `f32`. An [`AudioResampler`](crate::elements::AudioResampler)
    /// in front converts it.
    #[error(
        "AudioLimiter takes f32 audio, packed or planar, got {0:?}; put an AudioResampler in front"
    )]
    UnsupportedSampleFormat(ffmpeg::format::Sample),

    /// A packed frame's plane is shorter than its declared samples need.
    #[error(
        "audio plane is too small for its declared format: need {required} bytes, got {actual}"
    )]
    InvalidPlaneSize {
        /// Bytes the declared samples need.
        required: usize,
        /// Bytes the plane has.
        actual: usize,
    },

    /// The sink received a buffer other than decoded audio or end-of-stream.
    #[error("AudioLimiter only processes decoded Audio frames, got a {0}")]
    UnsupportedBuffer(&'static str),
}

impl From<Unreadable> for AudioLimiterError {
    fn from(unreadable: Unreadable) -> Self {
        match unreadable {
            Unreadable::Format(format) => Self::UnsupportedSampleFormat(format),
            Unreadable::NoChannels => Self::MissingChannels,
            Unreadable::TooShort { required, actual } => {
                Self::InvalidPlaneSize { required, actual }
            }
        }
    }
}

/// Thread-safe runtime control for an [`AudioLimiter`], with the ownership
/// of [`AudioGateHandle`](crate::elements::AudioGateHandle).
#[derive(Debug, Clone)]
pub struct AudioLimiterHandle {
    control: Arc<Tuning<AudioLimiterOptions>>,
}

impl AudioLimiterHandle {
    /// Replaces both settings at once, from the next frame on. Refused, with
    /// the running settings left alone, if they are invalid.
    pub fn set_options(
        &self,
        options: AudioLimiterOptions,
    ) -> std::result::Result<(), AudioLimiterError> {
        self.control.set(options.validate()?);
        Ok(())
    }

    /// The settings the limiter is running with, or will from its next
    /// frame.
    pub fn options(&self) -> AudioLimiterOptions {
        self.control.get()
    }
}

/// A limiter: nothing that comes out is louder than a level — the last
/// thing on a channel, so a shout or a knock cannot clip the mix, the way a
/// streaming application's limiter is used.
///
/// Takes decoded `f32` audio, packed or planar, at any rate and channel
/// count, and passes it on with its format, rate, layout, sample count and
/// timestamps unchanged; it only scales samples, and adds no delay. One gain
/// for every channel, from the loudest, as the other dynamics filters.
///
/// # A ceiling, not a tendency
///
/// It does not look ahead, so it cannot ease into a peak; it reacts in the
/// sample the peak arrives, turning the gain down to exactly what brings it
/// to the threshold. That is what makes the threshold a guarantee rather
/// than a target: no sample comes out over it. The cost is the distortion
/// of an instant change on the loudest peaks, which is what a limiter at
/// the end of a chain is there to trade for not clipping. Once the level
/// falls back, the gain recovers toward one over the release.
///
/// # Runtime control and errors
///
/// [`AudioLimiterHandle::set_options`] replaces the settings from the next
/// frame. A frame that is not `f32` is an error for that frame and changes
/// nothing. `Flush` and `Stop` let go at once; `Eos` passes straight
/// through.
pub struct AudioLimiter {
    pp_log: PpLog,
    name: Arc<str>,
    control: Arc<Tuning<AudioLimiterOptions>>,
    applied: Option<(u64, u32)>,
    /// The threshold as an amplitude, and how much of the way back to the
    /// wanted gain is kept each sample while releasing.
    ceiling: f32,
    release: f32,
    /// How far under one the gain is.
    reduction: f32,
    pad: SrcPad,
}

impl AudioLimiter {
    /// Creates a limiter with [`AudioLimiterOptions::default`].
    pub fn new(name: impl Into<String>) -> (Self, AudioLimiterHandle) {
        Self::with_options(name, AudioLimiterOptions::default())
            .expect("the default AudioLimiterOptions are valid")
    }

    /// Creates a limiter and a thread-safe runtime control handle.
    pub fn with_options(
        name: impl Into<String>,
        options: AudioLimiterOptions,
    ) -> std::result::Result<(Self, AudioLimiterHandle), AudioLimiterError> {
        let options = options.validate()?;
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::AudioLimiter, &name, None);
        let control = Tuning::new(options);
        pp_info!(pp_log: &pp_log, "created: {options:?}");
        Ok((
            Self {
                name: name.clone(),
                pp_log,
                control: control.clone(),
                applied: None,
                // Both replaced before the first sample is looked at.
                ceiling: 1.0,
                release: 0.0,
                reduction: 0.0,
                pad: SrcPad::with_contract(
                    format!("{name}_src"),
                    OutputContract::Fixed(PortContract::frame(
                        MediaKind::AudioFrame,
                        MemoryDomain::System,
                    )),
                ),
            },
            AudioLimiterHandle { control },
        ))
    }

    fn follow_settings(&mut self, sample_rate: u32) {
        if let Some(options) = self.control.fresh(&mut self.applied, sample_rate) {
            self.ceiling = db_to_amplitude(options.threshold_db);
            let samples = options.release.as_secs_f32() * sample_rate as f32;
            self.release = if samples < 1.0 {
                0.0
            } else {
                (-1.0 / samples).exp()
            };
        }
    }

    /// The gain for one sample, given the loudest channel's level at it.
    fn gain(&mut self, peak: f32) -> f32 {
        // What would bring this sample to the ceiling, and no lower.
        let wanted = if peak > self.ceiling {
            self.ceiling / peak
        } else {
            1.0
        };
        if 1.0 - wanted > self.reduction {
            // At once, which is the whole of the guarantee: `peak * wanted`
            // is exactly the ceiling.
            self.reduction = 1.0 - wanted;
            return wanted;
        }
        // Toward it gradually. Never past it, since each step moves only part
        // of the way — so `peak * gain` stays at or under the ceiling here
        // too. Kept as how far under one the gain is rather than as the gain,
        // because an `f32` just under one stops moving once a step is under
        // half its spacing: the gain would stall a hair short of one, and
        // every later frame be copied and rescaled for nothing.
        let wanted_reduction = 1.0 - wanted;
        let eased = wanted_reduction + self.release * (self.reduction - wanted_reduction);
        self.reduction = if eased - wanted_reduction < SETTLED {
            wanted_reduction
        } else {
            eased
        };
        1.0 - self.reduction
    }

    fn process(
        &mut self,
        frame: Arc<ffmpeg::frame::Audio>,
    ) -> std::result::Result<Arc<ffmpeg::frame::Audio>, AudioLimiterError> {
        self.follow_settings(frame.rate());
        Ok(audio_f32::scale_linked(frame, |peak| self.gain(peak))?)
    }
}

impl Element for AudioLimiter {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::AudioLimiter
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for AudioLimiter {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for AudioLimiter {
    /// Scales samples in place; nothing else has samples to scale.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::frame(
            MediaKind::AudioFrame,
            MemoryDomain::System,
        ))
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Audio(frame) => {
                let frame = self.process(frame)?;
                self.pad.push(MediaBuffer::Audio(frame))
            }
            MediaBuffer::Eos => self.pad.push(MediaBuffer::Eos),
            MediaBuffer::Packet(_) => Err(AudioLimiterError::UnsupportedBuffer("Packet").into()),
            MediaBuffer::Video(_) => Err(AudioLimiterError::UnsupportedBuffer("Video").into()),
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        if matches!(msg, ControlMsg::Flush | ControlMsg::Stop) {
            self.reduction = 0.0;
        }
        self.pad.control(msg)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex as StdMutex;

    use ffmpeg::format::sample::Type;

    use super::*;
    use crate::elements::filter::audio_f32::tests::frame;

    const RATE: u32 = 48_000;

    struct CapturingSink {
        pp_log: PpLog,
        received: Arc<StdMutex<Vec<MediaBuffer>>>,
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

    fn limiter(
        options: AudioLimiterOptions,
    ) -> (
        AudioLimiter,
        AudioLimiterHandle,
        Arc<StdMutex<Vec<MediaBuffer>>>,
    ) {
        let (mut limiter, handle) = AudioLimiter::with_options("limit", options).unwrap();
        let received = Arc::new(StdMutex::new(Vec::new()));
        limiter.src_pads()[0].link(Box::new(CapturingSink {
            pp_log: element_pp_log(ElementType::Other, "capture", None),
            received: received.clone(),
        }));
        (limiter, handle, received)
    }

    fn run(
        limiter: &mut AudioLimiter,
        received: &Arc<StdMutex<Vec<MediaBuffer>>>,
        channels: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        received.lock().unwrap().clear();
        let samples = channels[0].len();
        for start in (0..samples).step_by(480) {
            let end = (start + 480).min(samples);
            let chunk: Vec<Vec<f32>> = channels
                .iter()
                .map(|channel| channel[start..end].to_vec())
                .collect();
            limiter
                .consume(MediaBuffer::Audio(frame(&chunk, RATE, Type::Planar)))
                .unwrap();
        }
        let mut out = vec![Vec::new(); channels.len()];
        for buffer in received.lock().unwrap().iter() {
            if let MediaBuffer::Audio(frame) = buffer {
                for (all, part) in out.iter_mut().zip(audio_f32::read(frame).unwrap()) {
                    all.extend(part);
                }
            }
        }
        out
    }

    /// A deterministic loud noise, far over any threshold and over full
    /// scale too — the worst a limiter is asked to hold.
    fn noise(amplitude: f32, samples: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..samples)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f32 / u32::MAX as f32 * 2.0 - 1.0) * amplitude
            })
            .collect()
    }

    /// The guarantee: whatever goes in, not one sample comes out over the
    /// threshold, on either channel — tried against noise at twice full
    /// scale with the fastest release, where it has the least time to be
    /// right.
    #[test]
    fn nothing_comes_out_over_the_threshold() {
        let (mut limiter, _, received) = limiter(AudioLimiterOptions {
            threshold_db: -6.0,
            release: Duration::ZERO,
        });
        let ceiling = db_to_amplitude(-6.0);
        let input = [noise(2.0, RATE as usize, 7), noise(1.5, RATE as usize, 11)];
        let out = run(&mut limiter, &received, &input);
        let loudest = out
            .iter()
            .flatten()
            .fold(0.0_f32, |peak, sample| peak.max(sample.abs()));
        assert!(
            loudest <= ceiling * (1.0 + 1e-6),
            "{loudest} over {ceiling}"
        );
        assert!(
            loudest > ceiling * 0.99,
            "and held at it, not under: {loudest}"
        );
    }

    /// Under the threshold it does nothing at all — once it has let go of
    /// the last peak, which takes the release.
    #[test]
    fn under_the_threshold_it_lets_go_over_the_release_and_then_does_nothing() {
        let (mut limiter, _, received) = limiter(AudioLimiterOptions {
            threshold_db: -6.0,
            release: Duration::from_millis(20),
        });
        run(&mut limiter, &received, &[vec![1.0; 480]]);
        // Twenty time constants: from half, -120 dB takes thirteen.
        let quiet = vec![0.1; RATE as usize * 2 / 5];
        let out = run(&mut limiter, &received, std::slice::from_ref(&quiet)).remove(0);
        assert!(
            out[0] < 0.06,
            "still held down right after the peak: {}",
            out[0]
        );
        let one_constant = RATE as usize / 50;
        assert!(
            (out[one_constant] - 0.1 * (1.0 - 0.5 / std::f32::consts::E)).abs() < 1e-3,
            "1 - 1/e of the way back after one time constant: {}",
            out[one_constant]
        );
        assert_eq!(out[out.len() - 1], 0.1, "and all of it by the end");
        assert_eq!(
            run(&mut limiter, &received, std::slice::from_ref(&quiet)).remove(0),
            quiet,
            "and from then on passes it untouched"
        );
    }

    #[test]
    fn a_frame_that_is_not_f32_is_an_error_and_changes_nothing() {
        let (mut limiter, _, received) = limiter(AudioLimiterOptions::default());
        let mut wrong = ffmpeg::frame::Audio::new(
            ffmpeg::format::Sample::I16(Type::Packed),
            480,
            ffmpeg::ChannelLayout::MONO,
        );
        wrong.set_rate(RATE);
        let error = limiter
            .consume(MediaBuffer::Audio(Arc::new(wrong)))
            .unwrap_err();
        assert!(matches!(
            error,
            crate::Error::AudioLimiterError(AudioLimiterError::UnsupportedSampleFormat(_))
        ));
        assert!(received.lock().unwrap().is_empty());
        let quiet = vec![0.1; 480];
        assert_eq!(
            run(&mut limiter, &received, std::slice::from_ref(&quiet)).remove(0),
            quiet,
            "and the next good frame passes as if it had not been"
        );
    }

    #[test]
    fn a_threshold_that_is_not_a_level_is_refused_and_changes_nothing() {
        let (_, handle, _) = limiter(AudioLimiterOptions::default());
        let running = handle.options();
        assert_eq!(
            handle
                .set_options(AudioLimiterOptions {
                    threshold_db: f32::NAN,
                    ..running
                })
                .map_err(|error| matches!(error, AudioLimiterError::InvalidThreshold(_))),
            Err(true)
        );
        assert_eq!(handle.options(), running);
    }
}
