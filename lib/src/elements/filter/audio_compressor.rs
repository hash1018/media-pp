use std::{sync::Arc, time::Duration};

use crate::pp_log::{PpLog, pp_info};
use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use super::audio_f32::{self, Unreadable, amplitude_to_db, db_to_amplitude};
use super::tuning::Tuning;
use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    error::Result,
    pad::SrcPad,
};

/// Settings for [`AudioCompressor`], and what
/// [`AudioCompressorHandle::set_options`] replaces them with.
///
/// The five a streaming application's compressor asks for, with their
/// meaning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioCompressorOptions {
    /// The level, in dBFS, above which the compressor turns sound down.
    pub threshold_db: f32,
    /// How much it turns down what is above the threshold: at 4, a level
    /// 12 dB over comes out 3 dB over. One is no compression at all; the
    /// larger it gets, the nearer the compressor comes to a limiter.
    pub ratio: f32,
    /// How quickly it reacts to a level rising over the threshold.
    pub attack: Duration,
    /// How quickly it lets go once the level falls again.
    pub release: Duration,
    /// Gain applied after compression, in dB — what a compressed voice is
    /// usually brought back up with, since compressing takes its peaks down.
    pub output_gain_db: f32,
}

impl Default for AudioCompressorOptions {
    /// A streaming application's compressor starts from these, so they are
    /// what someone who has used one expects.
    fn default() -> Self {
        Self {
            threshold_db: -18.0,
            ratio: 10.0,
            attack: Duration::from_millis(6),
            release: Duration::from_millis(60),
            output_gain_db: 0.0,
        }
    }
}

impl AudioCompressorOptions {
    fn validate(self) -> std::result::Result<Self, AudioCompressorError> {
        if !self.threshold_db.is_finite() {
            return Err(AudioCompressorError::InvalidThreshold(self.threshold_db));
        }
        if !self.ratio.is_finite() || self.ratio < 1.0 {
            return Err(AudioCompressorError::InvalidRatio(self.ratio));
        }
        if !self.output_gain_db.is_finite() {
            return Err(AudioCompressorError::InvalidOutputGain(self.output_gain_db));
        }
        Ok(self)
    }
}

/// Errors specific to [`AudioCompressor`].
#[derive(Debug, ThisError, PartialEq)]
pub enum AudioCompressorError {
    /// The threshold is NaN or infinite.
    #[error("a compressor threshold must be a finite level in dBFS, got {0}")]
    InvalidThreshold(f32),

    /// The ratio is below one — which would be an expander — or not finite.
    #[error("a compressor ratio must be a finite number of at least 1, got {0}")]
    InvalidRatio(f32),

    /// The output gain is NaN or infinite.
    #[error("a compressor output gain must be a finite number of dB, got {0}")]
    InvalidOutputGain(f32),

    /// The input frame does not describe any audio channels.
    #[error("audio frame has no channel layout")]
    MissingChannels,

    /// The input is not `f32`. An [`AudioResampler`](crate::elements::AudioResampler)
    /// in front converts it.
    #[error(
        "AudioCompressor takes f32 audio, packed or planar, got {0:?}; put an AudioResampler in front"
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
    #[error("AudioCompressor only processes decoded Audio frames, got a {0}")]
    UnsupportedBuffer(&'static str),
}

impl From<Unreadable> for AudioCompressorError {
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

/// Thread-safe runtime control for an [`AudioCompressor`] — what
/// [`AudioGateHandle`](crate::elements::AudioGateHandle) is for a gate, and
/// with the same ownership: cheap to clone, keeping only the settings alive.
#[derive(Debug, Clone)]
pub struct AudioCompressorHandle {
    control: Arc<Tuning<AudioCompressorOptions>>,
}

impl AudioCompressorHandle {
    /// Replaces every setting at once, from the next frame on. Refused, with
    /// the running settings left alone, if they are invalid.
    pub fn set_options(
        &self,
        options: AudioCompressorOptions,
    ) -> std::result::Result<(), AudioCompressorError> {
        self.control.set(options.validate()?);
        Ok(())
    }

    /// The settings the compressor is running with, or will from its next
    /// frame.
    pub fn options(&self) -> AudioCompressorOptions {
        self.control.get()
    }
}

/// The settings as the per-sample loop uses them, for one sample rate.
#[derive(Debug, Clone, Copy)]
struct Rates {
    threshold_db: f32,
    /// The share of what is over the threshold that is taken off it.
    slope: f32,
    output_gain_db: f32,
    /// How much of the level follower is kept each sample, rising and
    /// falling — a one-pole smoothing whose time constant is the attack or
    /// the release.
    attack: f32,
    release: f32,
}

impl Rates {
    fn new(options: AudioCompressorOptions, sample_rate: u32) -> Self {
        let rate = sample_rate as f32;
        let keep = |span: Duration| {
            let samples = span.as_secs_f32() * rate;
            if samples < 1.0 {
                0.0
            } else {
                (-1.0 / samples).exp()
            }
        };
        Self {
            threshold_db: options.threshold_db,
            slope: 1.0 - 1.0 / options.ratio,
            output_gain_db: options.output_gain_db,
            attack: keep(options.attack),
            release: keep(options.release),
        }
    }
}

/// A compressor: turns down what is louder than a level, by a ratio, so a
/// voice's loud and quiet parts sit closer together — the way a streaming
/// application's compressor does.
///
/// Takes decoded `f32` audio, packed or planar, at any rate and channel
/// count, and passes it on with its format, rate, layout, sample count and
/// timestamps unchanged; it only scales samples, and adds no delay. Like
/// [`AudioGate`](crate::elements::AudioGate), one gain for every channel,
/// from the loudest.
///
/// # How it decides
///
/// A level follower tracks the peaks, rising toward a louder one over the
/// attack and falling back over the release. Where it is over the threshold
/// the gain comes down by `1 - 1 / ratio` of the excess — a hard knee — and
/// the output gain is applied to everything, compressed or not.
///
/// # Runtime control and errors
///
/// [`AudioCompressorHandle::set_options`] replaces the settings from the
/// next frame. A frame that is not `f32` is an error for that frame and
/// changes nothing. `Flush` and `Stop` let go of whatever it was holding
/// down; `Eos` passes straight through.
pub struct AudioCompressor {
    pp_log: PpLog,
    name: Arc<str>,
    control: Arc<Tuning<AudioCompressorOptions>>,
    applied: Option<(u64, u32)>,
    rates: Rates,
    /// The level follower, as an amplitude.
    level: f32,
    pad: SrcPad,
}

impl AudioCompressor {
    /// Creates a compressor with [`AudioCompressorOptions::default`].
    pub fn new(name: impl Into<String>) -> (Self, AudioCompressorHandle) {
        Self::with_options(name, AudioCompressorOptions::default())
            .expect("the default AudioCompressorOptions are valid")
    }

    /// Creates a compressor and a thread-safe runtime control handle.
    pub fn with_options(
        name: impl Into<String>,
        options: AudioCompressorOptions,
    ) -> std::result::Result<(Self, AudioCompressorHandle), AudioCompressorError> {
        let options = options.validate()?;
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::AudioCompressor, &name, None);
        let control = Tuning::new(options);
        pp_info!(pp_log: &pp_log, "created: {options:?}");
        Ok((
            Self {
                name: name.clone(),
                pp_log,
                control: control.clone(),
                applied: None,
                // Replaced before the first sample is looked at.
                rates: Rates::new(options, 48_000),
                level: 0.0,
                pad: SrcPad::with_contract(
                    format!("{name}_src"),
                    OutputContract::Fixed(PortContract::frame(
                        MediaKind::AudioFrame,
                        MemoryDomain::System,
                    )),
                ),
            },
            AudioCompressorHandle { control },
        ))
    }

    fn follow_settings(&mut self, sample_rate: u32) {
        if let Some(options) = self.control.fresh(&mut self.applied, sample_rate) {
            self.rates = Rates::new(options, sample_rate);
        }
    }

    /// The gain for one sample, given the loudest channel's level at it.
    fn gain(&mut self, peak: f32) -> f32 {
        let rates = self.rates;
        let keep = if peak > self.level {
            rates.attack
        } else {
            rates.release
        };
        self.level = keep * self.level + (1.0 - keep) * peak;
        let over = (amplitude_to_db(self.level) - rates.threshold_db).max(0.0);
        db_to_amplitude(rates.output_gain_db - over * rates.slope)
    }

    fn process(
        &mut self,
        frame: Arc<ffmpeg::frame::Audio>,
    ) -> std::result::Result<Arc<ffmpeg::frame::Audio>, AudioCompressorError> {
        self.follow_settings(frame.rate());
        Ok(audio_f32::scale_linked(frame, |peak| self.gain(peak))?)
    }
}

impl Element for AudioCompressor {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::AudioCompressor
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for AudioCompressor {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for AudioCompressor {
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
            MediaBuffer::Packet(_) => Err(AudioCompressorError::UnsupportedBuffer("Packet").into()),
            MediaBuffer::Video(_) => Err(AudioCompressorError::UnsupportedBuffer("Video").into()),
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        if matches!(msg, ControlMsg::Flush | ControlMsg::Stop) {
            self.level = 0.0;
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

    fn compressor(
        options: AudioCompressorOptions,
    ) -> (
        AudioCompressor,
        AudioCompressorHandle,
        Arc<StdMutex<Vec<MediaBuffer>>>,
    ) {
        let (mut compressor, handle) = AudioCompressor::with_options("comp", options).unwrap();
        let received = Arc::new(StdMutex::new(Vec::new()));
        compressor.src_pads()[0].link(Box::new(CapturingSink {
            pp_log: element_pp_log(ElementType::Other, "capture", None),
            received: received.clone(),
        }));
        (compressor, handle, received)
    }

    /// `seconds` of a 200 Hz tone whose peaks are at `db` dBFS.
    fn tone(db: f32, seconds: f32) -> Vec<f32> {
        let amplitude = db_to_amplitude(db);
        (0..(seconds * RATE as f32) as usize)
            .map(|index| {
                amplitude * (index as f32 * 200.0 * std::f32::consts::TAU / RATE as f32).sin()
            })
            .collect()
    }

    /// Feeds `signal` in 10 ms frames, one channel, and answers what came out.
    fn run(
        compressor: &mut AudioCompressor,
        received: &Arc<StdMutex<Vec<MediaBuffer>>>,
        signal: &[f32],
    ) -> Vec<f32> {
        received.lock().unwrap().clear();
        for chunk in signal.chunks(480) {
            compressor
                .consume(MediaBuffer::Audio(frame(
                    &[chunk.to_vec()],
                    RATE,
                    Type::Packed,
                )))
                .unwrap();
        }
        received
            .lock()
            .unwrap()
            .iter()
            .flat_map(|buffer| match buffer {
                MediaBuffer::Audio(frame) => audio_f32::read(frame).unwrap().remove(0),
                _ => Vec::new(),
            })
            .collect()
    }

    fn peak_db(samples: &[f32]) -> f32 {
        amplitude_to_db(
            samples
                .iter()
                .fold(0.0, |peak, sample| peak.max(sample.abs())),
        )
    }

    /// The static curve: a tone 12 dB over a -18 dB threshold at 4:1 comes
    /// out 3 dB over it, once the follower has settled — and a tone under
    /// the threshold is not touched at all.
    ///
    /// With no attack, so the follower sits on the tone's peaks and the
    /// arithmetic is the ratio's alone; a slower one rides below them by
    /// however much the waveform lets it, which is the attack's business and
    /// tested on its own below.
    #[test]
    fn a_level_over_the_threshold_comes_out_over_it_by_the_ratio() {
        let options = AudioCompressorOptions {
            threshold_db: -18.0,
            ratio: 4.0,
            attack: Duration::ZERO,
            ..AudioCompressorOptions::default()
        };
        let (mut loud, _, received) = compressor(options);
        let out = run(&mut loud, &received, &tone(-6.0, 1.0));
        let settled = peak_db(&out[RATE as usize / 2..]);
        assert!(
            (settled - -15.0).abs() < 1.0,
            "-6 dB at 4:1 over -18 should come out near -15, got {settled:.2}"
        );

        let (mut soft, _, received) = compressor(options);
        let quiet = tone(-30.0, 0.5);
        assert_eq!(run(&mut soft, &received, &quiet), quiet, "untouched");
    }

    /// Reacting takes the attack; letting go takes the release — the gain
    /// crosses most of the way within a few time constants, and not in one
    /// sample.
    #[test]
    fn it_turns_down_over_the_attack_and_lets_go_over_the_release() {
        let (mut compressor, _, _) = compressor(AudioCompressorOptions {
            threshold_db: -20.0,
            ratio: 20.0,
            attack: Duration::from_millis(10),
            release: Duration::from_millis(100),
            output_gain_db: 0.0,
        });
        compressor.follow_settings(RATE);
        let loud = 0.5;
        let attacking: Vec<f32> = (0..RATE / 10).map(|_| compressor.gain(loud)).collect();
        assert!(attacking[0] > 0.9, "not all at once: {}", attacking[0]);
        // Five time constants in, the follower has all but reached the
        // level, so the gain has all but reached the static curve's.
        let settled = db_to_amplitude(-(amplitude_to_db(loud) + 20.0) * (1.0 - 1.0 / 20.0));
        assert!(
            (attacking[2_400] - settled).abs() < 0.02,
            "{}",
            attacking[2_400]
        );

        let releasing: Vec<f32> = (0..RATE / 2).map(|_| compressor.gain(0.0)).collect();
        assert!(
            releasing[480] < 0.5,
            "still holding down 10 ms into the release"
        );
        assert!(
            releasing[RATE as usize / 2 - 1] > 0.99,
            "let go after five time constants"
        );
    }

    /// The output gain applies to everything, compressed or not.
    #[test]
    fn the_output_gain_brings_everything_up_by_the_same() {
        let (mut compressor, _, received) = compressor(AudioCompressorOptions {
            output_gain_db: 6.0,
            ..AudioCompressorOptions::default()
        });
        let quiet = tone(-40.0, 0.1);
        let out = run(&mut compressor, &received, &quiet);
        assert!((peak_db(&out) - peak_db(&quiet) - 6.0).abs() < 0.01);
    }

    /// Settings replaced while running apply from the next frame, and ones
    /// that make no sense are refused without touching the ones running.
    #[test]
    fn new_settings_apply_from_the_next_frame_and_bad_ones_change_nothing() {
        let (mut compressor, handle, received) = compressor(AudioCompressorOptions::default());
        let quiet = tone(-30.0, 0.1);
        assert_eq!(run(&mut compressor, &received, &quiet), quiet);

        handle
            .set_options(AudioCompressorOptions {
                threshold_db: -50.0,
                ..AudioCompressorOptions::default()
            })
            .unwrap();
        let now = run(&mut compressor, &received, &quiet);
        assert!(peak_db(&now) < peak_db(&quiet) - 3.0, "turned down");

        let running = handle.options();
        assert_eq!(
            handle.set_options(AudioCompressorOptions {
                ratio: 0.5,
                ..running
            }),
            Err(AudioCompressorError::InvalidRatio(0.5))
        );
        assert_eq!(handle.options(), running);
    }

    #[test]
    fn a_frame_that_is_not_f32_is_an_error_and_changes_nothing() {
        let (mut compressor, _, received) = compressor(AudioCompressorOptions::default());
        let mut wrong = ffmpeg::frame::Audio::new(
            ffmpeg::format::Sample::I16(Type::Packed),
            480,
            ffmpeg::ChannelLayout::MONO,
        );
        wrong.set_rate(RATE);
        let error = compressor
            .consume(MediaBuffer::Audio(Arc::new(wrong)))
            .unwrap_err();
        assert!(matches!(
            error,
            crate::Error::AudioCompressorError(AudioCompressorError::UnsupportedSampleFormat(_))
        ));
        assert!(received.lock().unwrap().is_empty());
    }
}
