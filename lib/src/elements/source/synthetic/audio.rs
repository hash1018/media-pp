use std::{
    f64::consts::TAU,
    sync::Arc,
    time::{Duration, Instant},
};

use crate::pp_log::{PpLog, pp_info};
use ffmpeg_next as ffmpeg;

use crate::{
    buffer::MediaBuffer,
    contract::{MediaKind, MemoryDomain, OutputContract, PortContract},
    element::{Element, ElementType, Produced, Source, SourceStage, Wait, element_pp_log},
    error::Result,
    produce::source_stage,
};

/// How often a [`TestAudioSource`] wakes up to top up however many
/// samples wall-clock time now owes — same role/value as
/// [`crate::elements::WasapiCaptureSource`]'s own `POLL_INTERVAL`/
/// [`crate::elements::AudioMixer`]'s `TICK_INTERVAL`.
const TICK_INTERVAL: Duration = Duration::from_millis(20);

/// Construction-time options for [`TestAudioSource::new`].
#[derive(Debug, Clone, Copy)]
pub struct TestAudioOptions {
    /// Sample rate of the generated audio, in hertz.
    pub sample_rate: u32,
    /// Channel count of the generated audio.
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
/// equivalent. No real capture device involved: each tick it
/// fabricates however many samples wall-clock time now owes on a
/// drift-free absolute schedule (`expected = elapsed * sample_rate`,
/// `needed = expected - samples_emitted` — the same shape
/// `WasapiCaptureSource`'s `silence_owed` and `AudioMixer`'s `mix_tick` both
/// use, not a fixed
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
pub struct TestAudioSource(SourceStage<Generating>);

source_stage!(TestAudioSource);

/// What a [`TestAudioSource`] makes, a tick's worth of samples when asked:
/// all of its work, which the framework makes the source.
struct Generating {
    pp_log: PpLog,
    name: Arc<str>,
    sample_rate: u32,
    channels: u16,
    format: ffmpeg::format::Sample,
    channel_layout: ffmpeg::ChannelLayout,
    frequency: f64,
    /// Cumulative sample count across every emitted frame — this
    /// element's `pts` unit (see [`TestAudioSource::time_base`]) *and* the
    /// sine wave's own running phase (`generate_frame` divides this by
    /// `sample_rate` for `t`), so the waveform stays phase-continuous across
    /// frame boundaries instead of restarting from zero every tick.
    samples_emitted: i64,
    /// When the first samples were made, on the clock a pause does not move
    /// — what every later tick counts the samples it owes from.
    started: Option<Instant>,
}

// SAFETY: see `AudioMixer`'s own `unsafe impl Send` docs — same
// reasoning, `channel_layout` here is always `ChannelLayout::default`'s
// plain native layout.
unsafe impl Send for Generating {}

impl TestAudioSource {
    /// Creates an unbounded synthetic sine-wave source with the requested output definition.
    pub fn new(name: impl Into<String>, options: TestAudioOptions) -> Self {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::TestAudioSource, &name, None);
        pp_info!(
            pp_log: &pp_log,
            "created: {}Hz, {} channel(s), {}Hz tone",
            options.sample_rate,
            options.channels,
            options.frequency
        );
        Self(SourceStage::new(Generating {
            name,
            pp_log,
            sample_rate: options.sample_rate,
            channels: options.channels,
            format: ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed),
            channel_layout: ffmpeg::ChannelLayout::default(options.channels as i32),
            frequency: options.frequency,
            samples_emitted: 0,
            started: None,
        }))
    }

    /// The unit each emitted frame's `pts` is expressed in.
    pub fn time_base(&self) -> ffmpeg::Rational {
        self.0.inner().time_base()
    }

    /// The next `count` samples at once, not when they are owed — for a
    /// test fixture written as fast as it encodes, stamped as a paced run
    /// stamps them.
    #[cfg(test)]
    pub(crate) fn next_samples(&mut self, count: usize) -> MediaBuffer {
        MediaBuffer::Audio(Arc::new(self.0.inner_mut().generate_frame(count)))
    }
}

impl Generating {
    fn time_base(&self) -> ffmpeg::Rational {
        ffmpeg::Rational::new(1, self.sample_rate as i32)
    }

    /// Fabricates the next `needed`-sample frame: the same sine tone on
    /// every channel, phase-continuous with whatever's already been
    /// emitted (see `samples_emitted`).
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
        // SAFETY: viewing an `f32` slice as bytes, which is always aligned and
        // exactly `size_of_val` long. What the destination can take is the separate
        // bound the comment below describes.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                interleaved.as_ptr() as *const u8,
                std::mem::size_of_val(&*interleaved),
            )
        };
        // Same tight-length write `AudioMixer::mix_tick`/
        // `WasapiCaptureSource::build_frame` both use — `data_mut(0)`'s own
        // length is FFmpeg's own padded linesize, not necessarily exactly
        // `bytes.len()`.
        frame.data_mut(0)[..bytes.len()].copy_from_slice(bytes);
        frame.set_pts(Some(self.samples_emitted));
        crate::buffer::set_time_base(&mut frame, self.time_base());
        self.samples_emitted += needed as i64;
        frame
    }
}

impl Element for Generating {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::TestAudioSource
    }

    fn pp_log(&self) -> &crate::pp_log::PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut crate::pp_log::PpLog {
        &mut self.pp_log
    }
}

impl Source for Generating {
    fn is_live(&self) -> bool {
        true
    }

    fn output_contract(&self) -> OutputContract {
        OutputContract::Fixed(PortContract::frame(
            MediaKind::AudioFrame,
            MemoryDomain::System,
        ))
    }

    /// What the time since the first samples owes, a tick at a time — on
    /// the clock a pause does not move, so playing on after one is not a
    /// burst of what it would have owed.
    fn produce(&mut self, wait: &mut Wait<'_>) -> Result<Produced> {
        let started = *self.started.get_or_insert_with(|| wait.now());
        if !wait.until(wait.now() + TICK_INTERVAL) {
            return Ok(Produced::Nothing);
        }
        let elapsed = wait.now().saturating_duration_since(started);
        let expected = (elapsed.as_secs_f64() * self.sample_rate as f64) as i64;
        let needed = (expected - self.samples_emitted).max(0) as usize;
        if needed == 0 {
            return Ok(Produced::Nothing);
        }
        Ok(Produced::Buffer(MediaBuffer::Audio(Arc::new(
            self.generate_frame(needed),
        ))))
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Mutex, thread};

    use crate::pp_log::PpLog;

    use super::*;
    use crate::{
        element::{RawSink, RawSource},
        pipeline::Pipeline,
    };

    /// Captures every frame's `(format, rate, channels, pts, first_sample)`
    /// it sees, in order.
    struct RecordingSink {
        pp_log: PpLog,
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
        fn pp_log(&self) -> &PpLog {
            &self.pp_log
        }
        fn pp_log_mut(&mut self) -> &mut PpLog {
            &mut self.pp_log
        }
    }

    impl RawSink for RecordingSink {
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
    }

    #[test]
    fn generates_f32_frames_with_increasing_pts_and_a_bounded_tone() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = RecordingSink {
            seen: seen.clone(),
            pp_log: element_pp_log(ElementType::Other, "recorder", None),
        };
        let source = TestAudioSource::new(
            "test-audio",
            TestAudioOptions {
                sample_rate: 48000,
                channels: 2,
                frequency: 440.0,
            },
        );

        let (pipeline, ()) = Pipeline::new("test", source, |source, ctx| {
            let branch = ctx.branch().to(sink)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("test pipeline wiring must succeed");

        pipeline.run().unwrap();
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
    fn a_generated_stream_cannot_be_sought() {
        let mut source = TestAudioSource::new("test-audio", TestAudioOptions::default());
        assert!(source.as_seekable().is_none());
    }

    /// Regression test for the pause/resume timing bug: `start.elapsed()`
    /// keeps advancing while [`Pipeline::pause`] blocks this source's own
    /// loop inside `drain_control`. Without subtracting the accumulated
    /// `ControlOutcome::paused_for` back out, `Resume` would find the
    /// whole pause suddenly counted as owed samples and emit one wildly
    /// oversized frame to cover it, instead of resuming its steady
    /// per-tick sample count — each frame's `pts` is a running sample
    /// count, so a healthy run never has two consecutive frames whose
    /// `pts` gap is anywhere near a whole pause's worth of samples.
    #[test]
    fn resuming_after_a_pause_does_not_dump_a_burst_of_samples() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = RecordingSink {
            seen: seen.clone(),
            pp_log: element_pp_log(ElementType::Other, "recorder", None),
        };
        let source = TestAudioSource::new(
            "test-audio",
            TestAudioOptions {
                sample_rate: 48000,
                channels: 2,
                frequency: 440.0,
            },
        );

        let (pipeline, ()) = Pipeline::new("pause-resume-test", source, |source, ctx| {
            let branch = ctx.branch().to(sink)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("test pipeline wiring must succeed");

        pipeline.run().unwrap();
        thread::sleep(Duration::from_millis(60));
        pipeline.pause();
        thread::sleep(Duration::from_millis(400));
        pipeline.resume();
        thread::sleep(Duration::from_millis(100));
        pipeline.stop();
        pipeline.bus().log_events();

        let frames = seen.lock().unwrap();
        let pts: Vec<i64> = frames.iter().filter_map(|&(_, _, _, pts, _)| pts).collect();
        assert!(
            pts.len() >= 2,
            "expected multiple frames spanning the pause/resume, got {}",
            pts.len()
        );
        for window in pts.windows(2) {
            let gap = window[1] - window[0];
            // A healthy tick's worth of samples at 48kHz/20ms is ~960; a
            // 400ms pause treated as owed catch-up would show up as a
            // ~19200-sample gap. 12000 (250ms) sits comfortably between
            // the two.
            assert!(
                gap < 12_000,
                "expected steady per-tick sample counts across resume, not a single burst \
                 frame covering the whole pause: consecutive pts gap was {gap} samples \
                 ({:.0}ms) — full pts sequence: {pts:?}",
                gap as f64 / 48.0
            );
        }
    }
}
