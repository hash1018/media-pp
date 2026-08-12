use std::{
    f64::consts::TAU,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use ffmpeg_next as ffmpeg;
use rust_hlog::hinfo;
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    control::{ControlReceiver, drain_control},
    element::{Element, ElementType, Source, SourceElement, element_hlog},
    error::Result,
    pad::SrcPad,
};

/// How often [`TestAudioSource::run`] wakes up to top up however many
/// samples wall-clock time now owes — same role/value as
/// [`crate::elements::AudioCaptureSource`]'s own `POLL_INTERVAL`/
/// [`crate::elements::AudioMixer`]'s `TICK_INTERVAL`.
const TICK_INTERVAL: Duration = Duration::from_millis(20);

/// Errors specific to `TestAudioSource`. Converts into the crate-wide
/// `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum TestAudioSourceError {
    #[error("TestAudioSource doesn't support seeking a generated stream")]
    SeekUnsupported,
}

/// Construction-time options for [`TestAudioSource::new`].
#[derive(Debug, Clone, Copy)]
pub struct TestAudioOptions {
    pub sample_rate: u32,
    pub channels: u16,
    /// The generated sine tone's frequency, in Hz. `440.0` (concert pitch
    /// A) by default — audible, easy to recognize on a scope or by ear;
    /// nothing else is special about the exact value.
    pub frequency: f64,
}

impl Default for TestAudioOptions {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            channels: 2,
            frequency: 440.0,
        }
    }
}

/// Generates a synthetic sine-wave tone — GStreamer's `audiotestsrc`
/// equivalent. No real capture device involved: [`TestAudioSource::run`]
/// fabricates however many samples wall-clock time now owes on a
/// drift-free absolute schedule (`expected = elapsed * sample_rate`,
/// `needed = expected - samples_emitted` — the same shape
/// [`crate::elements::AudioCaptureSource::fill_silence_gap`]/
/// [`crate::elements::AudioMixer::mix_tick`] both use, not a fixed
/// per-tick sample count, which would drift the same way a fixed-duration
/// `thread::sleep`-only schedule would), stamps it with an increasing
/// `pts` (one sample per tick of [`TestAudioSource::time_base`]'s units),
/// and pushes it straight downstream — useful for exercising
/// `AudioMixer`/an encoder/a muxer without a real microphone.
///
/// Always emits `Sample::F32(Packed)` — the same fixed internal format
/// `AudioMixer` mixes in, so this can feed a `MixerHandle` input directly
/// with nothing to resample (though `MixerInputSink` resamples regardless
/// if fed something else instead, so this isn't load-bearing).
///
/// Runs until `Stop` — never reaches `Eos` on its own, same as every other
/// live source in this crate (no sample-count limit is exposed,
/// deliberately, mirroring a live capture source more than a file).
#[rust_hlog::hlog]
pub struct TestAudioSource {
    name: Arc<str>,
    pad: SrcPad,
    sample_rate: u32,
    channels: u16,
    format: ffmpeg::format::Sample,
    channel_layout: ffmpeg::ChannelLayout,
    frequency: f64,
    /// Cumulative sample count across every emitted frame — this
    /// element's `pts` unit (see [`TestAudioSource::time_base`]) *and* the
    /// sine wave's own running phase ([`TestAudioSource::generate_frame`]
    /// divides this by `sample_rate` for `t`), so the waveform stays
    /// phase-continuous across frame boundaries instead of restarting
    /// from zero every tick.
    samples_emitted: i64,
}

// SAFETY: see `AudioMixer`'s own `unsafe impl Send` docs — same
// reasoning, `channel_layout` here is always `ChannelLayout::default`'s
// plain native layout.
unsafe impl Send for TestAudioSource {}

impl TestAudioSource {
    pub fn new(name: impl Into<String>, options: TestAudioOptions) -> Self {
        let name: Arc<str> = name.into().into();
        let hlog = element_hlog(ElementType::TestAudioSource, &name, None);
        hinfo!(
            hlog: &hlog,
            "created: {}Hz, {} channel(s), {}Hz tone",
            options.sample_rate,
            options.channels,
            options.frequency
        );
        let pad = SrcPad::new(format!("{name}_src"));
        Self {
            name,
            hlog,
            pad,
            sample_rate: options.sample_rate,
            channels: options.channels,
            format: ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed),
            channel_layout: ffmpeg::ChannelLayout::default(options.channels as i32),
            frequency: options.frequency,
            samples_emitted: 0,
        }
    }

    /// The unit each emitted frame's `pts` is expressed in.
    pub fn time_base(&self) -> ffmpeg::Rational {
        ffmpeg::Rational::new(1, self.sample_rate as i32)
    }

    /// Fabricates the next `needed`-sample frame: the same sine tone on
    /// every channel, phase-continuous with whatever's already been
    /// emitted (see [`TestAudioSource::samples_emitted`]'s own docs).
    fn generate_frame(&mut self, needed: usize) -> ffmpeg::frame::Audio {
        let channels = self.channels as usize;
        let mut interleaved = vec![0f32; needed * channels];
        for (index, chunk) in interleaved.chunks_mut(channels).enumerate() {
            let t = (self.samples_emitted + index as i64) as f64 / self.sample_rate as f64;
            let sample = (t * self.frequency * TAU).sin() as f32;
            chunk.fill(sample);
        }

        let mut frame = ffmpeg::frame::Audio::new(self.format, needed, self.channel_layout);
        frame.set_rate(self.sample_rate);
        let bytes = unsafe {
            std::slice::from_raw_parts(
                interleaved.as_ptr() as *const u8,
                std::mem::size_of_val(&*interleaved),
            )
        };
        // Same tight-length write `AudioMixer::mix_tick`/
        // `AudioCaptureSource::build_frame` both use — `data_mut(0)`'s own
        // length is FFmpeg's own padded linesize, not necessarily exactly
        // `bytes.len()`.
        frame.data_mut(0)[..bytes.len()].copy_from_slice(bytes);
        frame.set_pts(Some(self.samples_emitted));
        self.samples_emitted += needed as i64;
        frame
    }
}

impl Element for TestAudioSource {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::TestAudioSource
    }

    fn hlog(&self) -> &rust_hlog::HLog {
        &self.hlog
    }

    fn hlog_mut(&mut self) -> &mut rust_hlog::HLog {
        &mut self.hlog
    }
}

impl Source for TestAudioSource {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl SourceElement for TestAudioSource {
    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        hinfo!(self, "run: starting");
        let start = Instant::now();
        loop {
            if drain_control(control, self, bus)? {
                hinfo!(self, "run: stopped");
                return Ok(());
            }
            thread::sleep(TICK_INTERVAL);

            let expected = (start.elapsed().as_secs_f64() * self.sample_rate as f64) as i64;
            let needed = (expected - self.samples_emitted).max(0) as usize;
            if needed == 0 {
                continue;
            }
            let frame = self.generate_frame(needed);
            // A downstream failure drops just this one frame — same
            // "report, don't die" contract every other source in this
            // crate gives its own push.
            if let Err(error) = self.pad.push(MediaBuffer::Audio(Arc::new(frame))) {
                bus.post(
                    &self.hlog,
                    BusEvent::Error {
                        element_type: ElementType::TestAudioSource,
                        name: self.name.clone(),
                        error,
                    },
                );
            }
        }
    }

    fn seek(&mut self, _target: Duration) -> Result<Duration> {
        Err(TestAudioSourceError::SeekUnsupported.into())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use rust_hlog::HLog;

    use super::*;
    use crate::{control::ControlMsg, element::Sink, pipeline::Pipeline};

    /// Captures every frame's `(format, rate, channels, pts, first_sample)`
    /// it sees, in order.
    #[rust_hlog::hlog]
    struct RecordingSink {
        #[allow(clippy::type_complexity)]
        seen: Arc<Mutex<Vec<(ffmpeg::format::Sample, u32, u16, Option<i64>, f32)>>>,
    }

    impl Element for RecordingSink {
        fn name(&self) -> Arc<str> {
            "recorder".into()
        }
        fn element_type(&self) -> ElementType {
            ElementType::Other
        }
        fn hlog(&self) -> &HLog {
            &self.hlog
        }
        fn hlog_mut(&mut self) -> &mut HLog {
            &mut self.hlog
        }
    }

    impl Sink for RecordingSink {
        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            if let MediaBuffer::Audio(frame) = buf
                && frame.samples() > 0
            {
                self.seen.lock().unwrap().push((
                    frame.format(),
                    frame.rate(),
                    frame.channel_layout().channels() as u16,
                    frame.pts(),
                    frame.plane::<f32>(0)[0],
                ));
            }
            Ok(())
        }
        fn control(&mut self, _msg: ControlMsg) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn generates_f32_frames_with_increasing_pts_and_a_bounded_tone() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = RecordingSink {
            seen: seen.clone(),
            hlog: element_hlog(ElementType::Other, "recorder", None),
        };
        let source = TestAudioSource::new(
            "test-audio",
            TestAudioOptions {
                sample_rate: 48000,
                channels: 2,
                frequency: 440.0,
            },
        );

        let pipeline = Pipeline::new("test", source, |source, ctx| {
            let branch = ctx.branch().to(Box::new(sink))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("test pipeline wiring must succeed");

        pipeline.run();
        // Long enough to observe several ticks at the 20ms `TICK_INTERVAL`.
        std::thread::sleep(Duration::from_millis(200));
        pipeline.stop();
        pipeline.bus().log_events();

        let frames = seen.lock().unwrap();
        assert!(!frames.is_empty(), "expected at least one generated frame");
        for &(format, rate, channels, _, sample) in frames.iter() {
            assert_eq!(
                format,
                ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed)
            );
            assert_eq!((rate, channels), (48000, 2));
            assert!(
                (-1.0..=1.0).contains(&sample),
                "expected a bounded sine sample, got {sample}"
            );
        }
        for window in frames.windows(2) {
            assert!(
                window[1].3 > window[0].3,
                "expected pts to strictly increase frame over frame, got {:?} then {:?}",
                window[0].3,
                window[1].3
            );
        }
    }

    #[test]
    fn seek_is_explicitly_unsupported() {
        let mut source = TestAudioSource::new("test-audio", TestAudioOptions::default());
        assert!(source.seek(Duration::from_secs(1)).is_err());
    }
}
