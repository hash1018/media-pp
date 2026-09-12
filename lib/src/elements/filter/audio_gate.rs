use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use crate::pp_log::{PpLog, pp_info};
use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use super::audio_f32::{self, Unreadable};
use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    error::Result,
    pad::SrcPad,
};

/// How quickly the level the gate listens to falls once the sound stops.
///
/// The level follows each peak up at once and sinks exponentially after it.
/// This is how far it is allowed to sink between two peaks of one sound:
/// a voice's fundamental lies above 80 Hz, so its peaks are under 6 ms
/// apart, and at this rate the level drops less than 3 dB in that time —
/// well inside the few decibels between a gate's two thresholds. Much
/// faster and the level would dip under the close threshold within every
/// cycle of a low voice; much slower and the gate would stay open through
/// the start of every pause.
const LEVEL_DECAY: Duration = Duration::from_millis(20);

/// Settings for [`AudioGate`], and what [`AudioGateHandle::set_options`]
/// replaces them with.
///
/// The same five a streaming application's noise gate asks for, and with
/// the same meaning: two thresholds rather than one, so a level hovering
/// around the line does not open and close the gate on every syllable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioGateOptions {
    /// The level, in dBFS, at or above which a closed gate opens.
    pub open_threshold_db: f32,
    /// The level, in dBFS, below which an open gate starts to close. At or
    /// below `open_threshold_db`; the space between the two is what keeps
    /// the gate from chattering.
    pub close_threshold_db: f32,
    /// How long the gate takes to open fully once the level has reached the
    /// open threshold.
    pub attack: Duration,
    /// How long the gate stays open after the level falls below the close
    /// threshold, before it starts to close — the length of a pause between
    /// words that should not be cut out.
    pub hold: Duration,
    /// How long the gate takes to close fully once the hold has run out.
    pub release: Duration,
}

impl Default for AudioGateOptions {
    /// Values that suit a close speaking microphone, and the ones people
    /// know: a streaming application's noise gate starts from the same.
    fn default() -> Self {
        Self {
            open_threshold_db: -26.0,
            close_threshold_db: -32.0,
            attack: Duration::from_millis(25),
            hold: Duration::from_millis(200),
            release: Duration::from_millis(150),
        }
    }
}

impl AudioGateOptions {
    fn validate(self) -> std::result::Result<Self, AudioGateError> {
        for threshold in [self.open_threshold_db, self.close_threshold_db] {
            if !threshold.is_finite() {
                return Err(AudioGateError::InvalidThreshold(threshold));
            }
        }
        if self.close_threshold_db > self.open_threshold_db {
            return Err(AudioGateError::ThresholdsOutOfOrder {
                open: self.open_threshold_db,
                close: self.close_threshold_db,
            });
        }
        Ok(self)
    }
}

/// Errors specific to [`AudioGate`].
#[derive(Debug, ThisError, PartialEq)]
pub enum AudioGateError {
    /// A threshold is NaN or infinite.
    #[error("a gate threshold must be a finite level in dBFS, got {0}")]
    InvalidThreshold(f32),

    /// The close threshold is above the open one, which would leave the gate
    /// nowhere to rest between them.
    #[error("the close threshold ({close} dB) must not be above the open one ({open} dB)")]
    ThresholdsOutOfOrder {
        /// The open threshold asked for.
        open: f32,
        /// The close threshold asked for.
        close: f32,
    },

    /// The input frame does not describe any audio channels.
    #[error("audio frame has no channel layout")]
    MissingChannels,

    /// The input is not `f32`. An [`AudioResampler`](crate::elements::AudioResampler)
    /// in front converts it.
    #[error(
        "AudioGate takes f32 audio, packed or planar, got {0:?}; put an AudioResampler in front"
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
    #[error("AudioGate only processes decoded Audio frames, got a {0}")]
    UnsupportedBuffer(&'static str),
}

impl From<Unreadable> for AudioGateError {
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

/// The settings, and a count of how often they were replaced — which is
/// all the element reads per frame, so the lock is taken only when the
/// count has moved.
#[derive(Debug)]
struct GateControl {
    options: Mutex<AudioGateOptions>,
    revision: AtomicU64,
}

impl GateControl {
    fn options(&self) -> AudioGateOptions {
        *self
            .options
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// Thread-safe runtime control for an [`AudioGate`].
///
/// Cheap to clone. Retaining it keeps only the settings alive, not the
/// element, its pad or the pipeline; setting options after the element has
/// gone does nothing and harms nothing.
#[derive(Debug, Clone)]
pub struct AudioGateHandle {
    control: Arc<GateControl>,
}

impl AudioGateHandle {
    /// Replaces every setting at once, from the next frame on — one call
    /// rather than one per field, so the gate never sees a new open
    /// threshold beside the old close one.
    ///
    /// Refused, with the running settings left alone, if they are invalid.
    pub fn set_options(
        &self,
        options: AudioGateOptions,
    ) -> std::result::Result<(), AudioGateError> {
        let options = options.validate()?;
        *self
            .control
            .options
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = options;
        self.control.revision.fetch_add(1, Ordering::Release);
        Ok(())
    }

    /// The settings the gate is running with, or will from its next frame.
    pub fn options(&self) -> AudioGateOptions {
        self.control.options()
    }
}

/// Whether the gate is letting sound through.
#[derive(Debug, Clone, Copy, PartialEq)]
enum State {
    /// Open, or opening over the attack.
    Open,
    /// Below the close threshold, and staying open for this many more
    /// samples in case it was only a pause.
    Holding(u64),
    /// Closing over the release, or closed.
    Closed,
}

/// The settings as the per-sample loop uses them, for one sample rate.
#[derive(Debug, Clone, Copy)]
struct Rates {
    open: f32,
    close: f32,
    attack_step: f32,
    release_step: f32,
    hold: u64,
    decay: f32,
}

impl Rates {
    fn new(options: AudioGateOptions, sample_rate: u32) -> Self {
        let rate = sample_rate as f32;
        // How far the gain moves per sample to cross its whole range in
        // `span`; a span of zero is an immediate change.
        let step = |span: Duration| {
            let samples = span.as_secs_f32() * rate;
            if samples < 1.0 { 1.0 } else { 1.0 / samples }
        };
        Self {
            open: db_to_amplitude(options.open_threshold_db),
            close: db_to_amplitude(options.close_threshold_db),
            attack_step: step(options.attack),
            release_step: step(options.release),
            hold: (options.hold.as_secs_f64() * f64::from(sample_rate)).round() as u64,
            decay: (-1.0 / (LEVEL_DECAY.as_secs_f32() * rate)).exp(),
        }
    }
}

fn db_to_amplitude(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

/// A noise gate: passes sound at or above a level and silences what is
/// below it, the way a streaming application's noise gate does.
///
/// Takes decoded `f32` audio, packed or planar, at any rate and channel
/// count, and passes it on with its format, rate, layout, sample count and
/// timestamps unchanged — it only scales samples, and adds no delay. One
/// gate serves every channel of a frame together: the level it listens to
/// is the loudest channel's, and the gain it applies is the same on all of
/// them, so a stereo image is never pulled to one side.
///
/// # How it decides
///
/// The level follows each peak up at once and sinks after it, by a factor
/// of e every 20 ms — slow enough not to dip between the peaks of a low
/// voice, quick enough to fall away at the start of a pause. At or above
/// the open threshold the gate opens, rising
/// to full over the attack. Once the level falls below the close threshold
/// it holds for the hold time, and only then closes over the release; the
/// level returning to the open threshold at any point before it has
/// finished closing opens it again. Between the two thresholds nothing
/// changes — an open gate stays open and a closed one closed.
///
/// A new gate starts closed, so whatever it hears first fades in over the
/// attack rather than beginning with a click.
///
/// # Runtime control and errors
///
/// [`AudioGateHandle::set_options`] replaces the settings from the next
/// frame. A frame that is not `f32` is an error for that frame and changes
/// nothing — put an [`AudioResampler`](crate::elements::AudioResampler) in
/// front. `Flush` and `Stop` return it to closed, since whatever follows is
/// a different stretch of sound; `Eos` passes straight through, as nothing
/// is held back.
pub struct AudioGate {
    pp_log: PpLog,
    name: Arc<str>,
    control: Arc<GateControl>,
    /// The settings revision `rates` was worked out from, and the rate.
    applied: Option<(u64, u32)>,
    rates: Rates,
    state: State,
    level: f32,
    gain: f32,
    pad: SrcPad,
}

impl AudioGate {
    /// Creates a gate with [`AudioGateOptions::default`].
    pub fn new(name: impl Into<String>) -> (Self, AudioGateHandle) {
        Self::with_options(name, AudioGateOptions::default())
            .expect("the default AudioGateOptions are valid")
    }

    /// Creates a gate and a thread-safe runtime control handle.
    pub fn with_options(
        name: impl Into<String>,
        options: AudioGateOptions,
    ) -> std::result::Result<(Self, AudioGateHandle), AudioGateError> {
        let options = options.validate()?;
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::AudioGate, &name, None);
        let control = Arc::new(GateControl {
            options: Mutex::new(options),
            revision: AtomicU64::new(0),
        });
        pp_info!(pp_log: &pp_log, "created: {options:?}");
        Ok((
            Self {
                name: name.clone(),
                pp_log,
                control: control.clone(),
                applied: None,
                // Replaced before the first sample is looked at — see
                // `follow_settings`.
                rates: Rates::new(options, 48_000),
                state: State::Closed,
                level: 0.0,
                gain: 0.0,
                pad: SrcPad::with_contract(
                    format!("{name}_src"),
                    OutputContract::Fixed(PortContract::frame(
                        MediaKind::AudioFrame,
                        MemoryDomain::System,
                    )),
                ),
            },
            AudioGateHandle { control },
        ))
    }

    /// Works the settings out again when they, or the frame's rate, have
    /// changed since the last frame.
    fn follow_settings(&mut self, sample_rate: u32) {
        let revision = self.control.revision.load(Ordering::Acquire);
        if self.applied == Some((revision, sample_rate)) {
            return;
        }
        self.rates = Rates::new(self.control.options(), sample_rate);
        self.applied = Some((revision, sample_rate));
    }

    /// The gain for each sample, given the loudest channel's level at it.
    fn gains(&mut self, peaks: impl Iterator<Item = f32>) -> Vec<f32> {
        let rates = self.rates;
        peaks
            .map(|peak| {
                self.level = peak.max(self.level * rates.decay);
                self.state = if self.level >= rates.open {
                    State::Open
                } else if self.level < rates.close {
                    match self.state {
                        State::Open => State::Holding(rates.hold),
                        State::Holding(0) => State::Closed,
                        State::Holding(left) => State::Holding(left - 1),
                        State::Closed => State::Closed,
                    }
                } else {
                    // Between the thresholds: an open gate stays open and a
                    // closed one closed. A hold already running runs on.
                    match self.state {
                        State::Holding(0) => State::Closed,
                        State::Holding(left) => State::Holding(left - 1),
                        other => other,
                    }
                };
                match self.state {
                    State::Open => self.gain = (self.gain + rates.attack_step).min(1.0),
                    State::Holding(_) => {}
                    State::Closed => self.gain = (self.gain - rates.release_step).max(0.0),
                }
                self.gain
            })
            .collect()
    }

    fn process(
        &mut self,
        frame: Arc<ffmpeg::frame::Audio>,
    ) -> std::result::Result<Arc<ffmpeg::frame::Audio>, AudioGateError> {
        let mut channels = audio_f32::read(&frame)?;
        self.follow_settings(frame.rate());
        let samples = frame.samples();
        let gains = self.gains((0..samples).map(|index| {
            channels
                .iter()
                .map(|channel| channel[index].abs())
                .fold(0.0, f32::max)
        }));
        if gains.iter().all(|gain| *gain == 1.0) {
            return Ok(frame);
        }
        for channel in &mut channels {
            for (sample, gain) in channel.iter_mut().zip(&gains) {
                *sample *= gain;
            }
        }
        // Copied only when another branch still holds it — see `AudioVolume`.
        let mut frame = Arc::try_unwrap(frame).unwrap_or_else(|shared| shared.as_ref().clone());
        audio_f32::write(&mut frame, &channels);
        Ok(Arc::new(frame))
    }

    fn close(&mut self) {
        self.state = State::Closed;
        self.level = 0.0;
        self.gain = 0.0;
    }
}

impl Element for AudioGate {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::AudioGate
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for AudioGate {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for AudioGate {
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
            MediaBuffer::Packet(_) => Err(AudioGateError::UnsupportedBuffer("Packet").into()),
            MediaBuffer::Video(_) => Err(AudioGateError::UnsupportedBuffer("Video").into()),
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        if matches!(msg, ControlMsg::Flush | ControlMsg::Stop) {
            self.close();
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

    fn gate(
        options: AudioGateOptions,
    ) -> (AudioGate, AudioGateHandle, Arc<StdMutex<Vec<MediaBuffer>>>) {
        let (mut gate, handle) = AudioGate::with_options("gate", options).unwrap();
        let received = Arc::new(StdMutex::new(Vec::new()));
        gate.src_pads()[0].link(Box::new(CapturingSink {
            pp_log: element_pp_log(ElementType::Other, "capture", None),
            received: received.clone(),
        }));
        (gate, handle, received)
    }

    /// `seconds` of a 200 Hz tone at `db` dBFS — a low voice's pitch, whose
    /// peaks are as far apart as anything the gate is meant to hold open.
    fn tone(db: f32, seconds: f32) -> Vec<f32> {
        let amplitude = db_to_amplitude(db);
        (0..(seconds * RATE as f32) as usize)
            .map(|index| {
                amplitude * (index as f32 * 200.0 * std::f32::consts::TAU / RATE as f32).sin()
            })
            .collect()
    }

    /// Feeds `signal` in 10 ms frames, the size a capture delivers, and
    /// answers what came out, one channel.
    fn run(
        gate: &mut AudioGate,
        received: &Arc<StdMutex<Vec<MediaBuffer>>>,
        signal: &[f32],
    ) -> Vec<f32> {
        received.lock().unwrap().clear();
        for chunk in signal.chunks(480) {
            gate.consume(MediaBuffer::Audio(frame(
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

    fn peak(samples: &[f32]) -> f32 {
        samples
            .iter()
            .fold(0.0, |peak, sample| peak.max(sample.abs()))
    }

    fn no_timing() -> AudioGateOptions {
        AudioGateOptions {
            attack: Duration::ZERO,
            hold: Duration::ZERO,
            release: Duration::ZERO,
            ..AudioGateOptions::default()
        }
    }

    /// The point of a gate: speech through untouched, room noise silenced.
    #[test]
    fn a_level_above_the_open_threshold_passes_and_one_below_the_close_threshold_does_not() {
        let (mut gate, _, received) = gate(no_timing());
        let loud = run(&mut gate, &received, &tone(-12.0, 0.2));
        let quiet = run(&mut gate, &received, &tone(-40.0, 0.2));

        let input = tone(-12.0, 0.2);
        // Past the first cycle, where the level is still climbing to the
        // tone's own peak.
        assert_eq!(&loud[480..], &input[480..], "passed unchanged");
        // Past the ~46 ms the loud tone's level takes to sink below the
        // close threshold.
        assert_eq!(peak(&quiet[RATE as usize / 10..]), 0.0, "silenced");
    }

    /// Between the thresholds nothing moves: a gate that is open stays open
    /// through a level that would never have opened it.
    #[test]
    fn between_the_thresholds_an_open_gate_stays_open_and_a_closed_one_closed() {
        let (mut gate, _, received) = gate(no_timing());
        let between = tone(-29.0, 0.2);

        let from_closed = run(&mut gate, &received, &between);
        assert_eq!(peak(&from_closed), 0.0, "never opened");

        run(&mut gate, &received, &tone(-12.0, 0.1));
        let from_open = run(&mut gate, &received, &between);
        assert_eq!(from_open, between, "never closed");
    }

    /// A pause shorter than the hold is not cut out, and one longer is —
    /// after the hold, not at its start.
    #[test]
    fn a_pause_is_held_open_for_the_hold_and_then_closed() {
        let (mut gate, _, received) = gate(AudioGateOptions {
            hold: Duration::from_millis(100),
            ..no_timing()
        });
        run(&mut gate, &received, &tone(-12.0, 0.1));
        let pause = run(&mut gate, &received, &tone(-40.0, 0.3));

        // The level takes a few milliseconds to sink below the close
        // threshold, and the hold starts from there.
        let held = &pause[..(RATE as usize / 10)];
        let after = &pause[(RATE as usize / 5)..];
        assert!(peak(held) > 0.0, "still open inside the hold");
        assert_eq!(peak(after), 0.0, "closed once it ran out");
    }

    /// Opening and closing are ramps of the configured length, not steps —
    /// a step in the middle of a waveform is a click. And closing starts
    /// where the level, sinking at its decay, crosses the close threshold.
    #[test]
    fn the_gate_opens_over_the_attack_and_closes_over_the_release() {
        let (mut gate, _, _) = gate(AudioGateOptions {
            attack: Duration::from_millis(10),
            release: Duration::from_millis(20),
            ..no_timing()
        });
        gate.follow_settings(RATE);

        // 10 ms is 480 samples: halfway through, halfway open.
        let opening = gate.gains(std::iter::repeat_n(0.5, 960));
        assert!((opening[239] - 0.5).abs() < 0.01, "{}", opening[239]);
        assert_eq!(opening[490], 1.0, "fully open after it");

        // Silence after a level of 0.5: it sinks by e every 20 ms and is
        // under -32 dBFS after 20 ms × ln(0.5 / 0.0251) ≈ 2871 samples.
        // With no hold the gate closes on the next, and 20 ms of release is
        // 960 samples from there to nothing.
        let closing = gate.gains(std::iter::repeat_n(0.0, RATE as usize / 5));
        let closes = closing.iter().position(|gain| *gain < 1.0).unwrap();
        let closed = closing.iter().position(|gain| *gain == 0.0).unwrap();
        assert!((2865..=2880).contains(&closes), "began closing at {closes}");
        assert!(
            (955..=962).contains(&(closed - closes)),
            "took {} samples to close",
            closed - closes
        );
    }

    /// Settings replaced while running take effect on the next frame, and
    /// settings that make no sense are refused without touching the ones
    /// running.
    #[test]
    fn new_settings_apply_from_the_next_frame_and_bad_ones_change_nothing() {
        let (mut gate, handle, received) = gate(no_timing());
        let quiet = tone(-40.0, 0.1);
        assert_eq!(peak(&run(&mut gate, &received, &quiet)), 0.0);

        handle
            .set_options(AudioGateOptions {
                open_threshold_db: -50.0,
                close_threshold_db: -60.0,
                ..no_timing()
            })
            .unwrap();
        let now = run(&mut gate, &received, &quiet);
        assert_eq!(&now[480..], &quiet[480..], "opened by the lower threshold");

        let running = handle.options();
        assert_eq!(
            handle.set_options(AudioGateOptions {
                open_threshold_db: -40.0,
                close_threshold_db: -30.0,
                ..no_timing()
            }),
            Err(AudioGateError::ThresholdsOutOfOrder {
                open: -40.0,
                close: -30.0
            })
        );
        assert_eq!(handle.options(), running);
    }

    /// One gain for every channel, taken from the loudest: a gate that
    /// listened to each channel alone would pull a stereo image to one side
    /// whenever one side was quieter.
    #[test]
    fn every_channel_is_gated_together_by_the_loudest_of_them() {
        let (mut gate, _, received) = gate(no_timing());
        let loud = tone(-12.0, 0.1);
        let quiet = tone(-40.0, 0.1);
        for packing in [Type::Packed, Type::Planar] {
            received.lock().unwrap().clear();
            gate.consume(MediaBuffer::Audio(frame(
                &[loud.clone(), quiet.clone()],
                RATE,
                packing,
            )))
            .unwrap();
            let received = received.lock().unwrap();
            let MediaBuffer::Audio(out) = &received[0] else {
                panic!("audio")
            };
            let channels = audio_f32::read(out).unwrap();
            assert_eq!(&channels[1][480..], &quiet[480..], "{packing:?}");
            assert_eq!(out.pts(), Some(123), "timestamps pass through");
        }
    }

    /// A `Flush` is a different stretch of sound: the gate starts it closed
    /// rather than carrying an open state across the seek.
    #[test]
    fn a_flush_closes_the_gate() {
        let (mut gate, _, received) = gate(AudioGateOptions {
            attack: Duration::from_millis(10),
            ..no_timing()
        });
        run(&mut gate, &received, &tone(-12.0, 0.1));
        gate.control(ControlMsg::Flush).unwrap();
        let after = run(&mut gate, &received, &vec![0.5; 480]);
        assert!(after[0] < 0.01, "fades in again: {}", after[0]);
    }

    #[test]
    fn a_frame_that_is_not_f32_is_an_error_and_changes_nothing() {
        let (mut gate, _, received) = gate(no_timing());
        let mut wrong = ffmpeg::frame::Audio::new(
            ffmpeg::format::Sample::I16(Type::Packed),
            480,
            ffmpeg::ChannelLayout::MONO,
        );
        wrong.set_rate(RATE);
        let error = gate
            .consume(MediaBuffer::Audio(Arc::new(wrong)))
            .unwrap_err();
        assert!(matches!(
            error,
            crate::Error::AudioGateError(AudioGateError::UnsupportedSampleFormat(_))
        ));
        assert!(received.lock().unwrap().is_empty());

        let error = gate
            .consume(MediaBuffer::Packet(Arc::new(ffmpeg::Packet::empty())))
            .unwrap_err();
        assert!(matches!(
            error,
            crate::Error::AudioGateError(AudioGateError::UnsupportedBuffer("Packet"))
        ));
    }
}
