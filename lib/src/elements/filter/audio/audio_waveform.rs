//! [`AudioWaveform`] — sound drawn as its waveform, a picture at a steady
//! rate: what a player shows for a file with no picture of its own.

use std::{collections::VecDeque, sync::Arc};

use ffmpeg_next::{self as ffmpeg, Rescale};
use thiserror::Error as ThisError;

use super::audio_f32::{self, Unreadable};
use crate::{
    buffer::{MediaBuffer, set_time_base, time_base},
    color::Color,
    contract::{
        InputContract, MediaKind, MemoryDomain, OutputContract, PixelLayoutSet, PortContract,
    },
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    error::Result,
    pad::SrcPad,
    pool::UnboundObjectPool,
    pp_log::{PpLog, pp_info},
};

/// Why an [`AudioWaveform`] could not be made or could not draw.
#[derive(Debug, ThisError)]
pub enum AudioWaveformError {
    /// A picture with no area, or a rate that is not a positive fraction.
    #[error("a waveform of {width}x{height} at {frame_rate} cannot be drawn")]
    InvalidOptions {
        /// Width asked for.
        width: u32,
        /// Height asked for.
        height: u32,
        /// Rate asked for.
        frame_rate: ffmpeg::Rational,
    },
    /// The sound is not `f32`, packed or planar — put an
    /// [`AudioResampler`](crate::elements::AudioResampler) in front.
    #[error("AudioWaveform draws f32 sound, got {0:?}")]
    UnsupportedSampleFormat(ffmpeg::format::Sample),
    /// A frame with no channel layout, or one shorter than its samples.
    #[error("the sound frame cannot be read: {0}")]
    Unreadable(String),
    /// Something other than decoded sound or the end of it.
    #[error("AudioWaveform draws sound, got a {0}")]
    UnsupportedBuffer(&'static str),
}

impl From<Unreadable> for AudioWaveformError {
    fn from(error: Unreadable) -> Self {
        match error {
            Unreadable::Format(format) => Self::UnsupportedSampleFormat(format),
            other => Self::Unreadable(format!("{other:?}")),
        }
    }
}

/// How an [`AudioWaveform`] draws.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioWaveformOptions {
    /// Picture width in pixels.
    pub width: u32,
    /// Picture height in pixels.
    pub height: u32,
    /// How many pictures a second of sound makes.
    pub frame_rate: ffmpeg::Rational,
    /// The line's colour; the rest is black.
    pub color: Color,
}

impl Default for AudioWaveformOptions {
    /// 640x360 at 30 a second, in green.
    fn default() -> Self {
        Self {
            width: 640,
            height: 360,
            frame_rate: ffmpeg::Rational::new(30, 1),
            color: Color {
                red: 80,
                green: 220,
                blue: 120,
            },
        }
    }
}

/// Draws the sound it is given as a waveform — an oscilloscope's trace of
/// the last picture's worth of it, its channels averaged — into BGRA
/// pictures at a steady rate: the GStreamer `wavescope` of this crate, and
/// what a player shows for a file with no picture.
///
/// Takes decoded `f32` sound, packed or planar — an
/// [`AudioResampler`](crate::elements::AudioResampler) in front makes
/// anything else that. Puts out system-memory BGRA of the size asked for,
/// black with the trace on it.
///
/// # Timing
///
/// Each picture is stamped with the moment of the sound it ends at, in the
/// sound's own time base, so a
/// [`VideoSynchronizer`](crate::elements::VideoSynchronizer) after it shows
/// each as that sound is heard. Pictures are made as the sound arrives, which
/// is ahead of its playing by whatever is queued in front of the speakers —
/// the synchronizer holds them until then. Sound without timestamps is timed
/// from zero by its sample count.
///
/// A `Flush` — a seek — forgets what it had, and the next sound it is given
/// starts the pictures again from there. `Eos` is passed on as it comes.
pub struct AudioWaveform {
    pp_log: PpLog,
    name: Arc<str>,
    pad: SrcPad,
    options: AudioWaveformOptions,
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
    /// The most recent sound, channels averaged, at most a picture's worth
    /// and the sound still to reach the next picture's moment.
    samples: VecDeque<f32>,
    /// Where in the sound's time base the next picture ends, the sound's
    /// rate and unit, and the sample count `samples` ends at.
    clock: Option<Clock>,
}

/// Where the pictures are on the sound's timeline.
struct Clock {
    time_base: ffmpeg::Rational,
    sample_rate: u32,
    /// The sound's timestamp, in samples, of the sample after the last one
    /// in `samples`.
    end: i64,
    /// The sample at which the next picture ends — fractional, so a rate
    /// that does not divide the sound's still keeps its average.
    next: f64,
}

impl AudioWaveform {
    /// A waveform drawn as `options` say, refusing a picture with no area or
    /// a rate that is not a positive fraction.
    pub fn new(
        name: impl Into<String>,
        options: AudioWaveformOptions,
    ) -> std::result::Result<Self, AudioWaveformError> {
        let rate = options.frame_rate;
        if options.width == 0
            || options.height == 0
            || rate.numerator() <= 0
            || rate.denominator() <= 0
        {
            return Err(AudioWaveformError::InvalidOptions {
                width: options.width,
                height: options.height,
                frame_rate: rate,
            });
        }
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::AudioWaveform, &name, None);
        let (width, height) = (options.width, options.height);
        pp_info!(pp_log: &pp_log, "created: {width}x{height} at {rate}");
        Ok(Self {
            pad: SrcPad::with_contract(
                format!("{name}_src"),
                OutputContract::Fixed(
                    PortContract::frame(MediaKind::VideoFrame, MemoryDomain::System)
                        .with_layouts(PixelLayoutSet::BGRA),
                ),
            ),
            pp_log,
            name,
            options,
            pool: UnboundObjectPool::new(
                0,
                move || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, width, height),
                |_| {},
            ),
            samples: VecDeque::new(),
            clock: None,
        })
    }

    /// Samples a picture spans at `sample_rate`.
    fn span(&self, sample_rate: u32) -> f64 {
        let rate = self.options.frame_rate;
        f64::from(sample_rate) * f64::from(rate.denominator()) / f64::from(rate.numerator())
    }

    /// Takes in `frame`'s sound and pushes every picture it completes.
    fn take(&mut self, frame: &ffmpeg::frame::Audio) -> Result<()> {
        let channels = audio_f32::read(frame).map_err(AudioWaveformError::from)?;
        let sample_rate = frame.rate();
        let samples = frame.samples();
        let tb = time_base(frame).unwrap_or(ffmpeg::Rational::new(1, sample_rate as i32));
        let per_sample = ffmpeg::Rational::new(1, sample_rate as i32);
        let start = frame.pts().map(|pts| pts.rescale(tb, per_sample));
        let span = self.span(sample_rate);

        // Sound that does not follow on from what came before — the first,
        // a jump in its timestamps, another rate — starts the pictures again.
        let follows = self.clock.as_ref().is_some_and(|clock| {
            clock.sample_rate == sample_rate
                && start.is_none_or(|start| (start - clock.end).abs() <= 1)
        });
        if !follows {
            self.samples.clear();
            let start = start.unwrap_or(0);
            self.clock = Some(Clock {
                time_base: tb,
                sample_rate,
                end: start,
                next: start as f64 + span,
            });
        }
        let scale = 1.0 / channels.len() as f32;
        self.samples.extend(
            (0..samples)
                .map(|index| channels.iter().map(|channel| channel[index]).sum::<f32>() * scale),
        );
        let Some(clock) = self.clock.as_mut() else {
            return Ok(());
        };
        clock.end += samples as i64;
        let (end, time_base, mut next) = (clock.end, clock.time_base, clock.next);

        // Where sample `at` of the sound's timeline is in `samples`, which
        // ends at `end`; clamped to what it holds.
        let held = self.samples.len() as i64;
        let index = |at: i64| (held - (end - at)).clamp(0, held) as usize;
        let mut spans = Vec::new();
        while next <= end as f64 {
            let at = next.round() as i64;
            spans.push((at, index(at - span.round() as i64), index(at)));
            next += span;
        }
        // Nothing before the next picture's span is drawn again.
        let stale = index((next - span).floor() as i64);
        if let Some(clock) = self.clock.as_mut() {
            clock.next = next;
        }

        let pictures: Vec<_> = spans
            .into_iter()
            .map(|(at, first, last)| self.draw(first, last, at, time_base, sample_rate))
            .collect();
        self.samples.drain(..stale);
        for picture in pictures {
            self.pad.push(picture)?;
        }
        Ok(())
    }

    /// Draws `samples[first..last]` as one picture ending at sample `at`.
    fn draw(
        &self,
        first: usize,
        last: usize,
        at: i64,
        time_base: ffmpeg::Rational,
        sample_rate: u32,
    ) -> MediaBuffer {
        let (width, height) = (self.options.width as usize, self.options.height as usize);
        let mut frame = self.pool.get();
        let stride = frame.stride(0);
        let data = frame.data_mut(0);
        for row in 0..height {
            for pixel in data[row * stride..][..width * 4].as_chunks_mut::<4>().0 {
                *pixel = [0, 0, 0, 255];
            }
        }
        let color = self.options.color;
        let ink = [color.blue, color.green, color.red, 255];
        let middle = (height as f32 - 1.0) / 2.0;
        let row_of = |sample: f32| (middle - sample.clamp(-1.0, 1.0) * middle).round() as usize;
        let span = last.saturating_sub(first);
        let mut previous = None;
        for x in 0..width {
            let y = if span == 0 {
                row_of(0.0)
            } else {
                row_of(self.samples[first + x * span / width])
            };
            // A line, not dots: each column is drawn from the row the one
            // before it ended at, so a steep stretch stays joined up.
            let (top, bottom) = match previous {
                Some(previous) if previous < y => (previous, y),
                Some(previous) => (y, previous),
                None => (y, y),
            };
            for row in top..=bottom.min(height - 1) {
                data[row * stride + x * 4..][..4].copy_from_slice(&ink);
            }
            previous = Some(y);
        }
        let per_sample = ffmpeg::Rational::new(1, sample_rate as i32);
        frame.set_pts(Some(at.rescale(per_sample, time_base)));
        set_time_base(&mut frame, time_base);
        MediaBuffer::Video(Arc::new(frame))
    }
}

impl Element for AudioWaveform {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::AudioWaveform
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for AudioWaveform {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for AudioWaveform {
    /// Decoded sound in system memory.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::frame(
            MediaKind::AudioFrame,
            MemoryDomain::System,
        ))
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Audio(frame) => self.take(&frame),
            MediaBuffer::Eos => self.pad.push(MediaBuffer::Eos),
            MediaBuffer::Packet(_) => Err(AudioWaveformError::UnsupportedBuffer("Packet").into()),
            MediaBuffer::Video(_) => Err(AudioWaveformError::UnsupportedBuffer("Video").into()),
        }
    }

    fn control(&mut self, msg: &ControlMsg) -> Result<()> {
        if matches!(msg, ControlMsg::Flush | ControlMsg::Stop) {
            self.samples.clear();
            self.clock = None;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::capture;

    const RATE: u32 = 48_000;
    const CHUNK: usize = 1024;

    /// `seconds` of a `frequency` sine at `amplitude`, stereo `f32`, in
    /// frames of `CHUNK` stamped from `start` samples on.
    fn sound(seconds: f64, frequency: f32, amplitude: f32, start: i64) -> Vec<MediaBuffer> {
        let total = (seconds * f64::from(RATE)) as usize;
        (0..total)
            .step_by(CHUNK)
            .map(|offset| {
                let samples = CHUNK.min(total - offset);
                let mut frame = ffmpeg::frame::Audio::new(
                    ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed),
                    samples,
                    ffmpeg::ChannelLayout::STEREO,
                );
                frame.set_rate(RATE);
                let data = &mut frame.data_mut(0)[..samples * 8];
                for (index, bytes) in data.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    let sample = (offset + index / 2) as f32;
                    let value = amplitude
                        * (2.0 * std::f32::consts::PI * frequency * sample / RATE as f32).sin();
                    *bytes = value.to_ne_bytes();
                }
                frame.set_pts(Some(start + offset as i64));
                set_time_base(&mut frame, ffmpeg::Rational::new(1, RATE as i32));
                MediaBuffer::Audio(Arc::new(frame))
            })
            .collect()
    }

    fn waveform() -> (AudioWaveform, Arc<std::sync::Mutex<Vec<MediaBuffer>>>) {
        let mut waveform = AudioWaveform::new("waveform", AudioWaveformOptions::default()).unwrap();
        let received = capture(&mut waveform);
        (waveform, received)
    }

    fn pictures(
        received: &std::sync::Mutex<Vec<MediaBuffer>>,
    ) -> Vec<Arc<crate::pool::UnboundObjectPoolRef<ffmpeg::frame::Video>>> {
        received
            .lock()
            .unwrap()
            .iter()
            .filter_map(|buffer| match buffer {
                MediaBuffer::Video(frame) => Some(frame.clone()),
                _ => None,
            })
            .collect()
    }

    /// The rows in column `x` the trace was drawn on.
    fn inked_rows(picture: &ffmpeg::frame::Video, x: usize) -> Vec<usize> {
        let stride = picture.stride(0);
        (0..picture.height() as usize)
            .filter(|row| picture.data(0)[row * stride + x * 4..][..3] != [0, 0, 0])
            .collect()
    }

    /// A second of sound makes a second of pictures, each stamped with the
    /// moment of the sound it ends at, in the sound's own unit.
    #[test]
    fn a_second_of_sound_makes_a_second_of_pictures() {
        let (mut waveform, received) = waveform();
        for frame in sound(1.0, 440.0, 0.5, 0) {
            waveform.consume(frame).unwrap();
        }
        let pictures = pictures(&received);
        assert_eq!(pictures.len(), 30);
        for (index, picture) in pictures.iter().enumerate() {
            assert_eq!(picture.format(), ffmpeg::format::Pixel::BGRA);
            assert_eq!((picture.width(), picture.height()), (640, 360));
            assert_eq!(picture.pts(), Some((index as i64 + 1) * 1600));
            assert_eq!(
                time_base(picture),
                Some(ffmpeg::Rational::new(1, RATE as i32))
            );
        }
    }

    /// Silence is a flat line across the middle, and nothing else.
    #[test]
    fn silence_is_a_flat_line_across_the_middle() {
        let (mut waveform, received) = waveform();
        for frame in sound(0.1, 440.0, 0.0, 0) {
            waveform.consume(frame).unwrap();
        }
        let picture = pictures(&received).pop().expect("a picture");
        for x in [0, 320, 639] {
            assert_eq!(inked_rows(&picture, x), vec![180], "column {x}");
        }
    }

    /// A sine at half scale reaches a quarter of the way down from the top
    /// and up from the bottom, and no further.
    #[test]
    fn a_sine_reaches_as_far_as_its_amplitude() {
        let (mut waveform, received) = waveform();
        for frame in sound(0.2, 440.0, 0.5, 0) {
            waveform.consume(frame).unwrap();
        }
        let picture = pictures(&received).pop().expect("a picture");
        let inked: Vec<usize> = (0..640).flat_map(|x| inked_rows(&picture, x)).collect();
        let (top, bottom) = (*inked.iter().min().unwrap(), *inked.iter().max().unwrap());
        assert!((88..=92).contains(&top), "the trace's top is at row {top}");
        assert!((267..=271).contains(&bottom), "its bottom at row {bottom}");
    }

    /// After a seek the pictures start again from where the sound lands.
    #[test]
    fn a_seek_starts_the_pictures_again_where_the_sound_lands() {
        let (mut waveform, received) = waveform();
        for frame in sound(0.2, 440.0, 0.5, 0) {
            waveform.consume(frame).unwrap();
        }
        waveform.control(&ControlMsg::Flush).unwrap();
        received.lock().unwrap().clear();
        for frame in sound(0.2, 440.0, 0.5, 96_000) {
            waveform.consume(frame).unwrap();
        }
        let first = pictures(&received).first().cloned().expect("a picture");
        assert_eq!(first.pts(), Some(96_000 + 1600));
    }

    /// Sound that is not `f32` is refused, and so is a picture with no area.
    #[test]
    fn what_it_cannot_draw_is_refused() {
        let (mut waveform, _received) = waveform();
        let mut frame = ffmpeg::frame::Audio::new(
            ffmpeg::format::Sample::I16(ffmpeg::format::sample::Type::Packed),
            CHUNK,
            ffmpeg::ChannelLayout::STEREO,
        );
        frame.set_rate(RATE);
        assert!(matches!(
            waveform.consume(MediaBuffer::Audio(Arc::new(frame))),
            Err(crate::Error::AudioWaveformError(
                AudioWaveformError::UnsupportedSampleFormat(_)
            ))
        ));
        assert!(matches!(
            AudioWaveform::new(
                "empty",
                AudioWaveformOptions {
                    width: 0,
                    ..AudioWaveformOptions::default()
                }
            ),
            Err(AudioWaveformError::InvalidOptions { .. })
        ));
    }
}
