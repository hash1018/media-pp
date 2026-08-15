use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU32, Ordering},
    },
    time::Duration,
};

use crate::clog::{CLog, cinfo};
use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_clog},
    error::Result,
    pad::SrcPad,
};

const NANOS_PER_SECOND: u128 = 1_000_000_000;

/// Construction-time settings for [`AudioVolume`].
#[derive(Debug, Clone, Copy)]
pub struct AudioVolumeOptions {
    /// Initial linear gain. `1.0` is unchanged, `0.5` is half amplitude,
    /// and `0.0` is silence.
    pub gain: f32,
    /// Whether output starts muted. The configured `gain` is retained and
    /// restored when the element is unmuted.
    pub muted: bool,
    /// Time used to move from the current gain to a new runtime target.
    /// A short ramp avoids the discontinuity heard as a click when gain or
    /// mute changes in the middle of a waveform. Set to [`Duration::ZERO`]
    /// for an immediate change.
    pub ramp_duration: Duration,
}

impl Default for AudioVolumeOptions {
    fn default() -> Self {
        Self {
            gain: 1.0,
            muted: false,
            ramp_duration: Duration::from_millis(10),
        }
    }
}

/// Errors specific to [`AudioVolume`].
#[derive(Debug, ThisError, PartialEq)]
pub enum AudioVolumeError {
    #[error("gain must be finite and non-negative, got {0}")]
    InvalidGain(f32),

    #[error("gain in dB must be finite or negative infinity, got {0}")]
    InvalidGainDb(f32),

    #[error("audio frame has no channel layout")]
    MissingChannels,

    #[error("unsupported audio sample format: {0:?}")]
    UnsupportedSampleFormat(ffmpeg::format::Sample),

    #[error(
        "audio plane {plane} is too small for its declared format: need {required} bytes, got {actual}"
    )]
    InvalidPlaneSize {
        plane: usize,
        required: usize,
        actual: usize,
    },

    #[error(
        "AudioVolume only processes decoded Audio frames, got a {0}; link it after an audio decoder or source"
    )]
    UnsupportedBuffer(&'static str),
}

#[derive(Debug)]
struct VolumeControl {
    gain_bits: AtomicU32,
    muted: AtomicBool,
}

impl VolumeControl {
    fn new(gain: f32, muted: bool) -> Self {
        Self {
            gain_bits: AtomicU32::new(gain.to_bits()),
            muted: AtomicBool::new(muted),
        }
    }

    fn gain(&self) -> f32 {
        f32::from_bits(self.gain_bits.load(Ordering::Acquire))
    }

    fn effective_gain(&self) -> f32 {
        if self.muted.load(Ordering::Acquire) {
            0.0
        } else {
            self.gain()
        }
    }
}

/// Thread-safe runtime control for an [`AudioVolume`].
///
/// Retaining this handle only keeps the small atomic control state alive;
/// it does not retain the element, its pad, or the pipeline graph.
#[derive(Debug, Clone)]
pub struct AudioVolumeHandle {
    control: Arc<VolumeControl>,
}

impl AudioVolumeHandle {
    /// Sets a linear gain. The element ramps to the new value using its
    /// configured [`AudioVolumeOptions::ramp_duration`].
    pub fn set_gain(&self, gain: f32) -> std::result::Result<(), AudioVolumeError> {
        validate_gain(gain)?;
        self.control
            .gain_bits
            .store(gain.to_bits(), Ordering::Release);
        Ok(())
    }

    /// Sets gain in decibels. `0 dB` is unity, `-6 dB` is roughly half
    /// amplitude, and negative infinity is silence.
    pub fn set_gain_db(&self, gain_db: f32) -> std::result::Result<(), AudioVolumeError> {
        if gain_db == f32::NEG_INFINITY {
            return self.set_gain(0.0);
        }
        if !gain_db.is_finite() {
            return Err(AudioVolumeError::InvalidGainDb(gain_db));
        }
        let gain = 10.0_f32.powf(gain_db / 20.0);
        self.set_gain(gain)
            .map_err(|_| AudioVolumeError::InvalidGainDb(gain_db))
    }

    /// Enables or disables mute without discarding the configured gain.
    pub fn set_muted(&self, muted: bool) {
        self.control.muted.store(muted, Ordering::Release);
    }

    pub fn gain(&self) -> f32 {
        self.control.gain()
    }

    pub fn gain_db(&self) -> f32 {
        let gain = self.gain();
        if gain == 0.0 {
            f32::NEG_INFINITY
        } else {
            20.0 * gain.log10()
        }
    }

    pub fn is_muted(&self) -> bool {
        self.control.muted.load(Ordering::Acquire)
    }
}

/// Applies a runtime-adjustable master gain to decoded audio.
///
/// The input's sample format, rate, channels, sample count, and timestamps
/// are preserved. Integer formats saturate when amplified; floating-point
/// formats retain headroom. Both packed and planar FFmpeg PCM formats are
/// supported. Put an [`crate::elements::AudioResampler`] before this filter
/// when a downstream element also requires a specific audio format.
///
/// Runtime changes made through [`AudioVolumeHandle`] are linearly ramped
/// per audio sample. This changes parameters only and never changes graph
/// topology.
pub struct AudioVolume {
    clog: CLog,
    name: Arc<str>,
    control: Arc<VolumeControl>,
    ramp_duration: Duration,
    current_gain: f32,
    ramp_target: f32,
    ramp_step: f32,
    ramp_remaining: usize,
    gain_envelope: Vec<f32>,
    pad: SrcPad,
}

impl AudioVolume {
    /// Creates a unity-gain, unmuted filter with a 10 ms smoothing ramp.
    pub fn new(name: impl Into<String>) -> (Self, AudioVolumeHandle) {
        Self::with_options(name, AudioVolumeOptions::default())
            .expect("the default AudioVolumeOptions are valid")
    }

    pub fn with_options(
        name: impl Into<String>,
        options: AudioVolumeOptions,
    ) -> std::result::Result<(Self, AudioVolumeHandle), AudioVolumeError> {
        validate_gain(options.gain)?;
        let name: Arc<str> = name.into().into();
        let clog = element_clog(ElementType::AudioVolume, &name, None);
        let control = Arc::new(VolumeControl::new(options.gain, options.muted));
        let initial_gain = control.effective_gain();
        let handle = AudioVolumeHandle {
            control: control.clone(),
        };
        cinfo!(
            clog: &clog,
            "created: gain={}, muted={}, ramp={:?}",
            options.gain,
            options.muted,
            options.ramp_duration
        );
        Ok((
            Self {
                name: name.clone(),
                clog,
                control,
                ramp_duration: options.ramp_duration,
                current_gain: initial_gain,
                ramp_target: initial_gain,
                ramp_step: 0.0,
                ramp_remaining: 0,
                gain_envelope: Vec::new(),
                pad: SrcPad::new(format!("{name}_src")),
            },
            handle,
        ))
    }

    fn ramp_samples(&self, sample_rate: u32) -> usize {
        self.ramp_duration
            .as_nanos()
            .saturating_mul(u128::from(sample_rate))
            .div_ceil(NANOS_PER_SECOND)
            .min(usize::MAX as u128) as usize
    }

    fn prepare_gain_envelope(&mut self, sample_rate: u32, samples: usize) {
        let target = self.control.effective_gain();
        if target.to_bits() != self.ramp_target.to_bits() {
            self.ramp_target = target;
            self.ramp_remaining = self.ramp_samples(sample_rate);
            if self.ramp_remaining == 0 {
                self.current_gain = target;
                self.ramp_step = 0.0;
            } else {
                self.ramp_step = (target - self.current_gain) / self.ramp_remaining as f32;
            }
        }

        self.gain_envelope.clear();
        self.gain_envelope.reserve(samples);
        for _ in 0..samples {
            if self.ramp_remaining > 0 {
                self.current_gain += self.ramp_step;
                self.ramp_remaining -= 1;
                if self.ramp_remaining == 0 {
                    self.current_gain = self.ramp_target;
                }
            }
            self.gain_envelope.push(self.current_gain);
        }
    }

    fn snap_to_target(&mut self) {
        let target = self.control.effective_gain();
        self.current_gain = target;
        self.ramp_target = target;
        self.ramp_step = 0.0;
        self.ramp_remaining = 0;
        self.gain_envelope.clear();
    }

    fn process_audio(
        &mut self,
        frame: Arc<ffmpeg::frame::Audio>,
    ) -> std::result::Result<Arc<ffmpeg::frame::Audio>, AudioVolumeError> {
        let format = frame.format();
        if format == ffmpeg::format::Sample::None {
            return Err(AudioVolumeError::UnsupportedSampleFormat(format));
        }
        let channels = usize::from(frame.channels());
        if channels == 0 {
            return Err(AudioVolumeError::MissingChannels);
        }

        self.prepare_gain_envelope(frame.rate(), frame.samples());
        if self.gain_envelope.iter().all(|gain| *gain == 1.0) {
            return Ok(frame);
        }

        // `Audio::clone` allocates a new FFmpeg frame and copies its data.
        // That keeps a Tee sibling's Arc-backed frame untouched while the
        // unique-owner path avoids an unnecessary copy.
        let mut frame = match Arc::try_unwrap(frame) {
            Ok(frame) => frame,
            Err(frame) => frame.as_ref().clone(),
        };
        apply_gain(&mut frame, &self.gain_envelope)?;
        Ok(Arc::new(frame))
    }
}

impl Element for AudioVolume {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::AudioVolume
    }

    fn clog(&self) -> &CLog {
        &self.clog
    }

    fn clog_mut(&mut self) -> &mut CLog {
        &mut self.clog
    }
}

impl Source for AudioVolume {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for AudioVolume {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Audio(frame) => {
                let frame = self.process_audio(frame)?;
                self.pad.push(MediaBuffer::Audio(frame))
            }
            MediaBuffer::Eos => self.pad.push(MediaBuffer::Eos),
            MediaBuffer::Packet(_) => Err(AudioVolumeError::UnsupportedBuffer("Packet").into()),
            MediaBuffer::Video(_) => Err(AudioVolumeError::UnsupportedBuffer("Video").into()),
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        if matches!(msg, ControlMsg::Seek(_) | ControlMsg::Stop) {
            self.snap_to_target();
        }
        self.pad.control(msg)
    }
}

fn validate_gain(gain: f32) -> std::result::Result<(), AudioVolumeError> {
    if gain.is_finite() && gain >= 0.0 {
        Ok(())
    } else {
        Err(AudioVolumeError::InvalidGain(gain))
    }
}

fn apply_gain(
    frame: &mut ffmpeg::frame::Audio,
    gains: &[f32],
) -> std::result::Result<(), AudioVolumeError> {
    let format = frame.format();
    let channels = usize::from(frame.channels());
    let (planes, channels_per_plane) = if format.is_planar() {
        (channels, 1)
    } else {
        (1, channels)
    };
    let scalar_count = gains.len().saturating_mul(channels_per_plane);
    let required = scalar_count.saturating_mul(format.bytes());

    for plane in 0..planes {
        // `frame.data_mut(plane)`'s length comes from `AVFrame.linesize[plane]`,
        // but FFmpeg's convention for planar *audio* only ever fills in
        // `linesize[0]` (every plane is the same size, so it isn't repeated
        // elsewhere) — `linesize[1..]` stay `0`, so `data_mut` on any channel
        // but the first always reports a zero-length slice regardless of the
        // real buffer. Same footgun documented and worked around in
        // `encoder/audio/encoder.rs`'s `absorb_resampled`; read the real
        // per-plane byte range directly from the frame instead.
        if plane >= frame.planes() {
            return Err(AudioVolumeError::InvalidPlaneSize {
                plane,
                required,
                actual: 0,
            });
        }
        // Safety: `plane < frame.planes()`, and `required` is exactly
        // `gains.len() * channels_per_plane * format.bytes()`, which the
        // caller (`process_audio`) guarantees equals this plane's real
        // allocated size (`gains.len() == frame.samples()`).
        let data =
            unsafe { std::slice::from_raw_parts_mut((*frame.as_mut_ptr()).data[plane], required) };
        match format {
            ffmpeg::format::Sample::U8(_) => scale_u8(data, gains, channels_per_plane),
            ffmpeg::format::Sample::I16(_) => scale_i16(data, gains, channels_per_plane),
            ffmpeg::format::Sample::I32(_) => scale_i32(data, gains, channels_per_plane),
            ffmpeg::format::Sample::I64(_) => scale_i64(data, gains, channels_per_plane),
            ffmpeg::format::Sample::F32(_) => scale_f32(data, gains, channels_per_plane),
            ffmpeg::format::Sample::F64(_) => scale_f64(data, gains, channels_per_plane),
            ffmpeg::format::Sample::None => {
                return Err(AudioVolumeError::UnsupportedSampleFormat(format));
            }
        }
    }
    Ok(())
}

fn scale_u8(data: &mut [u8], gains: &[f32], channels: usize) {
    for (index, sample) in data.iter_mut().enumerate() {
        let centered = f32::from(*sample) - 128.0;
        *sample = (centered * gains[index / channels] + 128.0)
            .round()
            .clamp(0.0, 255.0) as u8;
    }
}

macro_rules! scale_integer {
    ($name:ident, $sample_type:ty, $bytes:literal) => {
        fn $name(data: &mut [u8], gains: &[f32], channels: usize) {
            for (index, sample) in data.chunks_exact_mut($bytes).enumerate() {
                let value = <$sample_type>::from_ne_bytes(sample.try_into().expect("chunk size"));
                let scaled = ((value as f64) * f64::from(gains[index / channels]))
                    .round()
                    .clamp(<$sample_type>::MIN as f64, <$sample_type>::MAX as f64)
                    as $sample_type;
                sample.copy_from_slice(&scaled.to_ne_bytes());
            }
        }
    };
}

scale_integer!(scale_i16, i16, 2);
scale_integer!(scale_i32, i32, 4);
scale_integer!(scale_i64, i64, 8);

fn scale_f32(data: &mut [u8], gains: &[f32], channels: usize) {
    for (index, sample) in data.chunks_exact_mut(4).enumerate() {
        let value = f32::from_ne_bytes(sample.try_into().expect("chunk size"));
        sample.copy_from_slice(&(value * gains[index / channels]).to_ne_bytes());
    }
}

fn scale_f64(data: &mut [u8], gains: &[f32], channels: usize) {
    for (index, sample) in data.chunks_exact_mut(8).enumerate() {
        let value = f64::from_ne_bytes(sample.try_into().expect("chunk size"));
        sample.copy_from_slice(&(value * f64::from(gains[index / channels])).to_ne_bytes());
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use ffmpeg::format::sample::Type;

    use super::*;

    struct CapturingSink {
        clog: CLog,
        received: Arc<Mutex<Vec<MediaBuffer>>>,
    }

    impl Element for CapturingSink {
        fn name(&self) -> Arc<str> {
            "capture".into()
        }

        fn element_type(&self) -> ElementType {
            ElementType::Other
        }

        fn clog(&self) -> &CLog {
            &self.clog
        }

        fn clog_mut(&mut self) -> &mut CLog {
            &mut self.clog
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

    fn new_volume(
        options: AudioVolumeOptions,
    ) -> (AudioVolume, AudioVolumeHandle, Arc<Mutex<Vec<MediaBuffer>>>) {
        let (mut volume, handle) = AudioVolume::with_options("volume", options).unwrap();
        let received = Arc::new(Mutex::new(Vec::new()));
        volume.src_pads()[0].link(Box::new(CapturingSink {
            received: received.clone(),
            clog: element_clog(ElementType::Other, "capture", None),
        }));
        (volume, handle, received)
    }

    fn f32_packed_frame(values: &[f32], rate: u32, channels: u16) -> Arc<ffmpeg::frame::Audio> {
        assert_eq!(values.len() % usize::from(channels), 0);
        let samples = values.len() / usize::from(channels);
        let mut frame = ffmpeg::frame::Audio::new(
            ffmpeg::format::Sample::F32(Type::Packed),
            samples,
            ffmpeg::ChannelLayout::default(i32::from(channels)),
        );
        frame.set_rate(rate);
        frame.set_pts(Some(123));
        let bytes = unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        };
        frame.data_mut(0)[..bytes.len()].copy_from_slice(bytes);
        Arc::new(frame)
    }

    /// One `&[f32]` per channel, written as a genuinely planar frame (each
    /// channel its own plane) — unlike `f32_packed_frame`, which only ever
    /// exercises plane 0.
    fn f32_planar_frame(channels_data: &[&[f32]], rate: u32) -> Arc<ffmpeg::frame::Audio> {
        let channels = channels_data.len();
        let samples = channels_data[0].len();
        assert!(channels_data.iter().all(|c| c.len() == samples));
        let mut frame = ffmpeg::frame::Audio::new(
            ffmpeg::format::Sample::F32(Type::Planar),
            samples,
            ffmpeg::ChannelLayout::default(channels as i32),
        );
        frame.set_rate(rate);
        frame.set_pts(Some(123));
        for (index, values) in channels_data.iter().enumerate() {
            frame.plane_mut::<f32>(index).copy_from_slice(values);
        }
        Arc::new(frame)
    }

    fn captured_f32(received: &Arc<Mutex<Vec<MediaBuffer>>>, index: usize) -> Vec<f32> {
        let received = received.lock().unwrap();
        let MediaBuffer::Audio(frame) = &received[index] else {
            panic!("expected audio")
        };
        let count = frame.samples() * usize::from(frame.channels());
        frame.data(0)[..count * 4]
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn mute_and_unmute_ramp_over_the_configured_number_of_samples() {
        let options = AudioVolumeOptions {
            ramp_duration: Duration::from_millis(10),
            ..AudioVolumeOptions::default()
        };
        let (mut volume, handle, received) = new_volume(options);
        handle.set_muted(true);
        volume
            .consume(MediaBuffer::Audio(f32_packed_frame(&[1.0; 10], 1_000, 1)))
            .unwrap();
        handle.set_muted(false);
        volume
            .consume(MediaBuffer::Audio(f32_packed_frame(&[1.0; 10], 1_000, 1)))
            .unwrap();

        let fade_out = captured_f32(&received, 0);
        let fade_in = captured_f32(&received, 1);
        for (index, sample) in fade_out.iter().enumerate() {
            let expected = 0.9 - index as f32 * 0.1;
            assert!((sample - expected).abs() < 1e-5, "{sample} != {expected}");
        }
        for (index, sample) in fade_in.iter().enumerate() {
            let expected = 0.1 + index as f32 * 0.1;
            assert!((sample - expected).abs() < 1e-5, "{sample} != {expected}");
        }
    }

    #[test]
    fn gain_db_changes_runtime_amplitude_without_changing_frame_metadata() {
        let options = AudioVolumeOptions {
            ramp_duration: Duration::ZERO,
            ..AudioVolumeOptions::default()
        };
        let (mut volume, handle, received) = new_volume(options);
        handle.set_gain_db(-6.0).unwrap();
        volume
            .consume(MediaBuffer::Audio(f32_packed_frame(
                &[1.0, -1.0, 0.5, -0.5],
                48_000,
                2,
            )))
            .unwrap();

        let gain = 10.0_f32.powf(-6.0 / 20.0);
        let output = captured_f32(&received, 0);
        for (actual, input) in output.iter().zip([1.0, -1.0, 0.5, -0.5]) {
            assert!((actual - input * gain).abs() < 1e-6);
        }
        let received = received.lock().unwrap();
        let MediaBuffer::Audio(frame) = &received[0] else {
            panic!("expected audio")
        };
        assert_eq!(frame.rate(), 48_000);
        assert_eq!(frame.channels(), 2);
        assert_eq!(frame.samples(), 2);
        assert_eq!(frame.pts(), Some(123));
    }

    /// Regression test for the planar-audio footgun documented in
    /// `apply_gain`: reading a channel via `data_mut(plane)` instead of the
    /// frame's real per-plane byte range made every channel but the first
    /// report a zero-length buffer, so any non-unity gain on planar audio
    /// with 2+ channels used to fail every frame instead of scaling it.
    #[test]
    fn gain_scales_every_channel_of_a_planar_frame() {
        let options = AudioVolumeOptions {
            gain: 0.5,
            ramp_duration: Duration::ZERO,
            ..AudioVolumeOptions::default()
        };
        let (mut volume, _handle, received) = new_volume(options);
        volume
            .consume(MediaBuffer::Audio(f32_planar_frame(
                &[&[1.0, -1.0], &[0.5, -0.5]],
                48_000,
            )))
            .unwrap();

        let received = received.lock().unwrap();
        let MediaBuffer::Audio(frame) = &received[0] else {
            panic!("expected audio")
        };
        assert_eq!(frame.plane::<f32>(0).to_vec(), vec![0.5, -0.5]);
        assert_eq!(frame.plane::<f32>(1).to_vec(), vec![0.25, -0.25]);
    }

    #[test]
    fn a_shared_input_is_copied_before_another_tee_branch_can_be_modified() {
        let options = AudioVolumeOptions {
            gain: 0.5,
            ramp_duration: Duration::ZERO,
            ..AudioVolumeOptions::default()
        };
        let (mut volume, _, received) = new_volume(options);
        let original = f32_packed_frame(&[1.0], 48_000, 1);
        volume
            .consume(MediaBuffer::Audio(original.clone()))
            .unwrap();

        assert_eq!(original.plane::<f32>(0)[0], 1.0);
        assert_eq!(captured_f32(&received, 0), vec![0.5]);
        let received = received.lock().unwrap();
        let MediaBuffer::Audio(output) = &received[0] else {
            panic!("expected audio")
        };
        assert!(!Arc::ptr_eq(&original, output));
    }

    #[test]
    fn integer_amplification_saturates_instead_of_wrapping() {
        let options = AudioVolumeOptions {
            gain: 2.0,
            ramp_duration: Duration::ZERO,
            ..AudioVolumeOptions::default()
        };
        let (mut volume, _, received) = new_volume(options);
        let mut frame = ffmpeg::frame::Audio::new(
            ffmpeg::format::Sample::I16(Type::Packed),
            2,
            ffmpeg::ChannelLayout::MONO,
        );
        frame.set_rate(48_000);
        for (destination, value) in frame.data_mut(0)[..4]
            .chunks_exact_mut(2)
            .zip([20_000_i16, -20_000_i16])
        {
            destination.copy_from_slice(&value.to_ne_bytes());
        }
        volume.consume(MediaBuffer::Audio(Arc::new(frame))).unwrap();

        let received = received.lock().unwrap();
        let MediaBuffer::Audio(frame) = &received[0] else {
            panic!("expected audio")
        };
        let output: Vec<_> = frame.data(0)[..4]
            .chunks_exact(2)
            .map(|bytes| i16::from_ne_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(output, vec![i16::MAX, i16::MIN]);
    }

    #[test]
    fn rejects_invalid_controls_and_non_audio_buffers() {
        let (_, handle) = AudioVolume::new("volume");
        assert_eq!(
            handle.set_gain(-1.0),
            Err(AudioVolumeError::InvalidGain(-1.0))
        );
        assert!(matches!(
            handle.set_gain_db(f32::NAN),
            Err(AudioVolumeError::InvalidGainDb(value)) if value.is_nan()
        ));

        let (mut volume, _) = AudioVolume::new("volume");
        let error = volume
            .consume(MediaBuffer::Packet(Arc::new(ffmpeg::Packet::empty())))
            .unwrap_err();
        assert!(matches!(
            error,
            crate::Error::AudioVolumeError(AudioVolumeError::UnsupportedBuffer("Packet"))
        ));
    }
}
