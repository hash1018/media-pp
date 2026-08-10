use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex, Weak},
    thread,
    time::{Duration, Instant},
};

use ffmpeg_next as ffmpeg;
use rust_hlog::{HLog, herror, hinfo};
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    control::{ControlMsg, ControlReceiver, drain_control},
    element::{Element, ElementType, Sink, Source, SourceElement, element_hlog},
    error::Result,
    pad::SrcPad,
};

/// How often [`AudioMixer::run`] mixes and emits a combined frame — same
/// role as [`crate::elements::DxgiScreenSource`]'s own `POLL_GRANULARITY`/
/// `crate::elements::AudioCaptureSource`'s `POLL_INTERVAL`: bounds `Stop`
/// latency and sets the mixer's own output granularity.
const TICK_INTERVAL: Duration = Duration::from_millis(20);

/// Errors specific to `AudioMixer`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum AudioMixerError {
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg_next::Error),

    #[error("AudioMixer doesn't support seeking a live mix")]
    SeekUnsupported,
}

/// Construction-time options for [`AudioMixer::new`] — the mixer's fixed
/// *output* format. Every input is resampled to match this on the way in
/// (see [`InputBuffer::push`]); the mixer never adapts to whatever an
/// input happens to produce.
#[derive(Debug, Clone, Copy)]
pub struct AudioMixerOptions {
    pub sample_rate: u32,
    pub channels: u16,
}

/// One input's own resampler and accumulated (already-resampled,
/// interleaved `f32`) samples, waiting to be drained by the next
/// [`AudioMixer::mix_tick`]. The resampler is built lazily from the first
/// frame this input ever sees (its `format`/`channel_layout`/`rate`
/// self-describe — no need for [`MixerHandle::add_source`] to be told
/// this upfront).
struct InputBuffer {
    resampler: Option<ffmpeg::software::resampling::Context>,
    samples: VecDeque<f32>,
    /// Set once this input's `Eos` arrives — [`AudioMixer::mix_tick`]
    /// drops the input entirely once it's both `eos` and fully drained,
    /// same as a `Tee` branch dropping out once removed. Unlike a fixed
    /// two-track muxer, `AudioMixer` has no fixed input count to wait on:
    /// one input reaching `Eos` just means the mix continues without it.
    eos: bool,
}

impl InputBuffer {
    fn push(
        &mut self,
        frame: &ffmpeg::frame::Audio,
        target_format: ffmpeg::format::Sample,
        target_layout: ffmpeg::ChannelLayout,
        target_rate: u32,
    ) -> std::result::Result<(), AudioMixerError> {
        let resampler = match &mut self.resampler {
            Some(resampler) => resampler,
            None => {
                let resampler = ffmpeg::software::resampling::Context::get(
                    frame.format(),
                    frame.channel_layout(),
                    frame.rate(),
                    target_format,
                    target_layout,
                    target_rate,
                )?;
                self.resampler.insert(resampler)
            }
        };
        let mut output = ffmpeg::frame::Audio::empty();
        resampler.run(frame, &mut output)?;
        self.samples.extend(output.plane::<f32>(0).iter().copied());
        Ok(())
    }
}

/// Shared state between [`AudioMixer`] and every [`MixerHandle`]/
/// [`MixerInputSink`] derived from it — just the input map, behind one
/// lock (same granularity [`crate::elements::Tee`]'s own `TeeShared::pads`
/// uses: one lock for the whole collection, not one per entry, since a mix
/// tick already needs to visit every input together anyway).
struct MixerShared {
    inputs: Mutex<HashMap<Arc<str>, InputBuffer>>,
}

/// A cheaply-cloneable handle for adding or removing an [`AudioMixer`]'s
/// input sources while the pipeline is running — the mirror image of
/// [`crate::elements::TeeHandle`]: `Tee` lets you attach/detach *outputs*
/// from another thread; this lets you attach/detach *inputs*. Keeps only a
/// [`Weak`] reference for the same reason `TeeHandle` does: retaining a
/// handle after the mixer's own pipeline finishes must not keep its
/// internal state alive forever, and every operation becomes a harmless
/// no-op once the mixer is gone.
#[derive(Clone)]
pub struct MixerHandle {
    shared: Weak<MixerShared>,
    sample_rate: u32,
    format: ffmpeg::format::Sample,
    channel_layout: ffmpeg::ChannelLayout,
}

impl MixerHandle {
    /// Registers a new input under `name` and returns a [`Sink`] to link
    /// an upstream source's pad to (e.g.
    /// `capture_source.src_pads()[0].link(handle.add_source("mic").unwrap())`
    /// inside that source's own `Pipeline::new` `wire` closure — a
    /// *different* pipeline/thread than this mixer's own, which is exactly
    /// the point). `None` once the mixer itself is gone. Calling this
    /// again with a name already in use replaces that input outright
    /// (whatever it had buffered is dropped) rather than erroring — same
    /// "just do what was asked" spirit as `HashMap::insert`.
    ///
    /// **Known gap**: the returned [`MixerInputSink`] isn't registered in
    /// any [`crate::element::ElementRegistry`] (it has no access to the
    /// *other* pipeline's own one, and that registry's `register` is
    /// crate-private besides), so it won't show up in that source's
    /// [`crate::pipeline::Pipeline::topology`]. Nothing today depends on
    /// cross-pipeline topology, so this is left unsolved rather than
    /// guessed at.
    pub fn add_source(&self, name: impl Into<String>) -> Option<Box<dyn Sink>> {
        let shared = self.shared.upgrade()?;
        let name: Arc<str> = name.into().into();
        shared.inputs.lock().unwrap().insert(
            name.clone(),
            InputBuffer {
                resampler: None,
                samples: VecDeque::new(),
                eos: false,
            },
        );
        Some(Box::new(MixerInputSink {
            name: name.clone(),
            hlog: element_hlog(ElementType::AudioMixer, &name, None),
            shared: self.shared.clone(),
            target_format: self.format,
            target_layout: self.channel_layout,
            target_rate: self.sample_rate,
        }))
    }

    /// Drops `name`'s input immediately, discarding whatever it had
    /// buffered — a no-op if `name` isn't currently registered, or the
    /// mixer is gone.
    pub fn remove_source(&self, name: &str) {
        if let Some(shared) = self.shared.upgrade() {
            shared.inputs.lock().unwrap().remove(name);
        }
    }

    pub fn source_count(&self) -> usize {
        self.shared
            .upgrade()
            .map(|shared| shared.inputs.lock().unwrap().len())
            .unwrap_or(0)
    }
}

/// One [`AudioMixer`] input, returned by [`MixerHandle::add_source`].
/// Resamples every incoming frame to the mixer's fixed output format and
/// appends it to this input's own buffer — the actual summing happens
/// later, on [`AudioMixer::run`]'s own thread, not here. `consume` runs on
/// whatever thread is driving the *upstream* source this got linked to
/// (a different pipeline's own thread, in the normal case), so every
/// access to the shared input map goes through `MixerShared`'s lock.
#[rust_hlog::hlog]
pub struct MixerInputSink {
    name: Arc<str>,
    shared: Weak<MixerShared>,
    target_format: ffmpeg::format::Sample,
    target_layout: ffmpeg::ChannelLayout,
    target_rate: u32,
}

// SAFETY: `ffmpeg::ChannelLayout` wraps `AVChannelLayout`, which carries a
// non-`Send` custom-layout pointer only for `AV_CHANNEL_ORDER_CUSTOM`
// layouts. Every `ChannelLayout` here comes from `ChannelLayout::default`
// (see `AudioMixer::new`) — a plain native layout, that pointer always
// null — so there's nothing thread-unsafe actually being sent.
unsafe impl Send for MixerInputSink {}

impl Element for MixerInputSink {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::AudioMixer
    }

    fn hlog(&self) -> &HLog {
        &self.hlog
    }

    fn hlog_mut(&mut self) -> &mut HLog {
        &mut self.hlog
    }
}

impl Sink for MixerInputSink {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        let Some(shared) = self.shared.upgrade() else {
            return Ok(()); // mixer's own pipeline already ended — nothing to feed
        };
        match buf {
            MediaBuffer::Audio(frame) => {
                let mut inputs = shared.inputs.lock().unwrap();
                if let Some(input) = inputs.get_mut(&self.name) {
                    input.push(
                        &frame,
                        self.target_format,
                        self.target_layout,
                        self.target_rate,
                    )?;
                }
                // Absent means `remove_source` raced ahead of this frame —
                // dropping it is correct, there's nowhere left to put it.
            }
            MediaBuffer::Eos => {
                if let Some(input) = shared.inputs.lock().unwrap().get_mut(&self.name) {
                    input.eos = true;
                }
            }
            _ => herror!(self, "unsupported buffer: expected Audio or Eos"),
        }
        Ok(())
    }

    /// No downstream of its own to cascade to — this is a leaf input slot,
    /// not a passthrough. `Stop` removes this input immediately, same as
    /// [`MixerHandle::remove_source`]: `Stop` means abandon now, not drain
    /// to a natural `Eos` (see `ControlMsg::Stop`'s own docs), and for a
    /// live capture source — [`crate::elements::AudioCaptureSource`]
    /// included — `Stop` is the *only* shutdown signal that ever arrives;
    /// it never reaches `Eos` on its own. Relying on `Eos` alone to clean
    /// up (as an earlier version of this did) left a stale entry in
    /// `shared.inputs` forever whenever a caller stopped its capture
    /// pipeline normally instead of remembering to call
    /// `MixerHandle::remove_source` by hand. `Pause`/`Resume`/`Seek` need
    /// no handling here — this input has no thread or queue of its own to
    /// freeze/resume, and a live capture source doesn't seek.
    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        if msg == ControlMsg::Stop
            && let Some(shared) = self.shared.upgrade()
        {
            shared.inputs.lock().unwrap().remove(&self.name);
        }
        Ok(())
    }
}

/// Sums an arbitrary, dynamically-changing number of audio sources into
/// one output stream — the structural mirror of [`crate::elements::Tee`]:
/// `Tee` is one input fanned out to a dynamic set of outputs behind a
/// lock; `AudioMixer` is a dynamic set of inputs (added/removed via
/// [`MixerHandle`], from whatever thread each one's own source pipeline
/// runs on) summed into one output. Unlike `Tee`, which is a passive
/// [`Sink`] driven entirely by whatever calls `consume`, `AudioMixer` has
/// to drive itself: it's a [`SourceElement`] with its own `run` thread,
/// ticking every [`TICK_INTERVAL`] to sum however many samples each
/// currently-attached input has ready — because mixing has to keep
/// producing *something* on a steady clock even when some (or all) inputs
/// have gone quiet, the same reason
/// [`crate::elements::AudioCaptureSource`] synthesizes silence for gaps
/// rather than just emitting nothing.
///
/// Every input is resampled to this mixer's own fixed
/// `sample_rate`/`channels` (always `Sample::F32(Packed)` internally —
/// float headroom during summation, same reason real mixing consoles
/// work in float even when everything else is integer PCM) — an input
/// short on samples for a given tick contributes silence for the
/// shortfall rather than blocking the whole mix. Samples are summed and
/// **hard-clipped** to `[-1.0, 1.0]`, not averaged: two or three sources
/// is the expected case, where clipping is rare, and averaging would
/// quietly lower the whole mix's volume every time a source count
/// changes — a caller who wants headroom can lower an individual input's
/// gain before it ever reaches the mixer (not implemented — nothing needs
/// it yet).
///
/// `pts` is a plain, always-continuous sample count (see
/// [`AudioMixer::time_base`]), advancing in lockstep with wall-clock time
/// regardless of which/how many inputs are actually contributing at any
/// moment.
///
/// Runs until `Stop` — never reaches `Eos` on its own, same as every
/// other live source in this crate; an individual input reaching `Eos` or
/// being removed just drops out of future ticks, it doesn't end the mix.
#[rust_hlog::hlog]
pub struct AudioMixer {
    name: Arc<str>,
    shared: Arc<MixerShared>,
    pad: SrcPad,
    sample_rate: u32,
    format: ffmpeg::format::Sample,
    channel_layout: ffmpeg::ChannelLayout,
    channels: u16,
    /// Cumulative sample count across every emitted frame — see
    /// [`AudioMixer::time_base`].
    samples_emitted: i64,
}

// SAFETY: see `MixerInputSink`'s own `unsafe impl Send` docs — same
// reasoning, `channel_layout` here is always `ChannelLayout::default`'s
// plain native layout.
unsafe impl Send for AudioMixer {}

impl AudioMixer {
    /// Starts with no inputs — add some via the returned [`MixerHandle`]
    /// before (or any time after) wiring `AudioMixer` into a
    /// [`crate::pipeline::Pipeline`] (`Pipeline::new` registers it as that
    /// pipeline's own source automatically, same as any other
    /// [`SourceElement`] — no [`crate::element::Context`] needed here,
    /// unlike [`crate::elements::Tee::new`], since `AudioMixer` has no
    /// chains of its own for a handle to build).
    pub fn new(name: impl Into<String>, options: AudioMixerOptions) -> (Self, MixerHandle) {
        let name: Arc<str> = name.into().into();
        let hlog = element_hlog(ElementType::AudioMixer, &name, None);
        hinfo!(
            hlog: &hlog,
            "created: {}Hz, {} channel(s)",
            options.sample_rate,
            options.channels
        );
        let format = ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed);
        let channel_layout = ffmpeg::ChannelLayout::default(options.channels as i32);
        let shared = Arc::new(MixerShared {
            inputs: Mutex::new(HashMap::new()),
        });
        let pad = SrcPad::new(format!("{name}_src"));
        (
            Self {
                name: name.clone(),
                hlog,
                shared: shared.clone(),
                pad,
                sample_rate: options.sample_rate,
                format,
                channel_layout,
                channels: options.channels,
                samples_emitted: 0,
            },
            MixerHandle {
                shared: Arc::downgrade(&shared),
                sample_rate: options.sample_rate,
                format,
                channel_layout,
            },
        )
    }

    /// The unit each emitted frame's `pts` is expressed in.
    pub fn time_base(&self) -> ffmpeg::Rational {
        ffmpeg::Rational::new(1, self.sample_rate as i32)
    }

    /// Sums however many samples are needed to keep `samples_emitted` in
    /// lockstep with `start.elapsed()` (a no-op if nothing's owed yet —
    /// same wall-clock-deficit shape as
    /// [`crate::elements::AudioCaptureSource::fill_silence_gap`], just
    /// summing real contributions from every input instead of emitting
    /// pure silence). Drops any input that's both `eos` and fully drained
    /// — it contributed its last real samples on a previous tick and has
    /// nothing left to give.
    fn mix_tick(&mut self, start: Instant, bus: &Bus) {
        let channels = self.channels as usize;
        let expected = (start.elapsed().as_secs_f64() * self.sample_rate as f64) as i64;
        let needed = (expected - self.samples_emitted).max(0) as usize;
        if needed == 0 {
            return;
        }
        let mut mixed = vec![0f32; needed * channels];
        {
            let mut inputs = self.shared.inputs.lock().unwrap();
            inputs.retain(|_, input| !(input.eos && input.samples.is_empty()));
            for input in inputs.values_mut() {
                let take = mixed.len().min(input.samples.len());
                for (slot, sample) in mixed.iter_mut().zip(input.samples.iter()) {
                    *slot += *sample;
                }
                input.samples.drain(0..take);
            }
        }
        for sample in &mut mixed {
            *sample = sample.clamp(-1.0, 1.0);
        }

        let mut frame = ffmpeg::frame::Audio::new(self.format, needed, self.channel_layout);
        frame.set_rate(self.sample_rate);
        let bytes = unsafe {
            std::slice::from_raw_parts(mixed.as_ptr() as *const u8, std::mem::size_of_val(&*mixed))
        };
        // `frame.data_mut(0)`'s length is FFmpeg's own padded linesize,
        // not necessarily `mixed.len() * 4` exactly — only ever write that
        // tight amount (same bound `frame.plane::<T>()` itself reads via
        // `samples()`), never assume the destination's full length
        // matches `bytes` (see `AudioCaptureSource::build_frame`'s own
        // identical fix).
        frame.data_mut(0)[..bytes.len()].copy_from_slice(bytes);
        frame.set_pts(Some(self.samples_emitted));
        self.samples_emitted += needed as i64;

        if let Err(error) = self.pad.push(MediaBuffer::Audio(Arc::new(frame))) {
            bus.post(
                &self.hlog,
                BusEvent::Error {
                    element_type: ElementType::AudioMixer,
                    name: self.name.clone(),
                    error,
                },
            );
        }
    }
}

impl Element for AudioMixer {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::AudioMixer
    }

    fn hlog(&self) -> &HLog {
        &self.hlog
    }

    fn hlog_mut(&mut self) -> &mut HLog {
        &mut self.hlog
    }
}

impl Source for AudioMixer {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl SourceElement for AudioMixer {
    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        hinfo!(self, "run: starting");
        let start = Instant::now();
        loop {
            if drain_control(control, self, bus)? {
                hinfo!(self, "run: stopped");
                return Ok(());
            }
            thread::sleep(TICK_INTERVAL);
            self.mix_tick(start, bus);
        }
    }

    fn seek(&mut self, _target: Duration) -> Result<Duration> {
        Err(AudioMixerError::SeekUnsupported.into())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Mutex as StdMutex,
        atomic::{AtomicBool, Ordering},
    };

    use rust_hlog::HLog;

    use super::*;
    use crate::pipeline::{ChainBuilder, Pipeline};

    fn constant_frame(value: f32, samples: usize, rate: u32) -> ffmpeg::frame::Audio {
        let mut frame = ffmpeg::frame::Audio::new(
            ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed),
            samples,
            ffmpeg::ChannelLayout::default(1),
        );
        frame.set_rate(rate);
        frame.plane_mut::<f32>(0).fill(value);
        frame
    }

    #[rust_hlog::hlog]
    struct RecordingSink {
        seen: Arc<StdMutex<Vec<f32>>>,
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
                self.seen.lock().unwrap().push(frame.plane::<f32>(0)[0]);
            }
            Ok(())
        }
        fn control(&mut self, _msg: ControlMsg) -> Result<()> {
            Ok(())
        }
    }

    /// Two inputs, each pushing a constant `0.6` from their own thread
    /// (standing in for two independent capture pipelines), should sum to
    /// `1.2` and get hard-clipped to `1.0` — verifies resampling-on-first-
    /// frame, cross-thread `consume`, summation, and clipping all work
    /// together, not just in isolation.
    #[test]
    fn mixes_two_sources_and_hard_clips() {
        let (mixer, handle) = AudioMixer::new(
            "mixer",
            AudioMixerOptions {
                sample_rate: 48000,
                channels: 1,
            },
        );
        let seen = Arc::new(StdMutex::new(Vec::new()));
        let sink = RecordingSink {
            seen: seen.clone(),
            hlog: element_hlog(ElementType::Other, "recorder", None),
        };

        let pipeline = Pipeline::new("mixer-test", mixer, |source, ctx| {
            let branch = ChainBuilder::new(ctx.clone()).build(Box::new(sink));
            source.src_pads()[0].link(branch);
        });
        pipeline.run();

        let mut input_a = handle.add_source("a").expect("mixer still alive");
        let mut input_b = handle.add_source("b").expect("mixer still alive");
        assert_eq!(handle.source_count(), 2);

        let stop = Arc::new(AtomicBool::new(false));
        let feeder_stop = stop.clone();
        let feeder = std::thread::spawn(move || {
            while !feeder_stop.load(Ordering::Relaxed) {
                let _ = input_a.consume(MediaBuffer::Audio(Arc::new(constant_frame(
                    0.6, 480, 48000,
                ))));
                let _ = input_b.consume(MediaBuffer::Audio(Arc::new(constant_frame(
                    0.6, 480, 48000,
                ))));
                std::thread::sleep(Duration::from_millis(10));
            }
        });

        std::thread::sleep(Duration::from_millis(300));
        stop.store(true, Ordering::Relaxed);
        feeder.join().unwrap();
        pipeline.stop();
        pipeline.bus().log_events();

        let seen = seen.lock().unwrap();
        assert!(
            seen.len() > 5,
            "expected several mixed frames, got {seen:?}"
        );
        // Skip the first few ticks (the feeder thread may not have caught
        // up yet) and the last couple (ticks after the feeder stopped but
        // before `pipeline.stop()` landed correctly drain to silence) —
        // check the steady state in between is clipped to 1.0.
        let steady = &seen[3..seen.len() - 2];
        for &value in steady {
            assert!(
                (value - 1.0).abs() < 0.01,
                "expected hard-clipped ~1.0, got {value} in {seen:?}"
            );
        }
    }

    #[test]
    fn removed_source_stops_contributing() {
        let (mixer, handle) = AudioMixer::new(
            "mixer",
            AudioMixerOptions {
                sample_rate: 48000,
                channels: 1,
            },
        );
        let seen = Arc::new(StdMutex::new(Vec::new()));
        let sink = RecordingSink {
            seen: seen.clone(),
            hlog: element_hlog(ElementType::Other, "recorder", None),
        };
        let pipeline = Pipeline::new("mixer-test-2", mixer, |source, ctx| {
            let branch = ChainBuilder::new(ctx.clone()).build(Box::new(sink));
            source.src_pads()[0].link(branch);
        });
        pipeline.run();

        let mut input_a = handle.add_source("a").unwrap();
        input_a
            .consume(MediaBuffer::Audio(Arc::new(constant_frame(
                0.5, 480, 48000,
            ))))
            .unwrap();
        std::thread::sleep(Duration::from_millis(100));
        handle.remove_source("a");
        assert_eq!(handle.source_count(), 0);
        seen.lock().unwrap().clear();

        std::thread::sleep(Duration::from_millis(100));
        pipeline.stop();
        pipeline.bus().log_events();

        assert!(
            seen.lock().unwrap().iter().all(|&v| v == 0.0),
            "removed source must not keep contributing: {:?}",
            *seen.lock().unwrap()
        );
    }

    /// Regression test: a capture pipeline ending via `Stop` — the only
    /// shutdown signal a live source like `AudioCaptureSource` ever sends,
    /// since it never reaches `Eos` on its own — used to leave a stale
    /// entry in the mixer's input map forever, because only `Eos` cleared
    /// it. `Sink::control` is what a `Queue`/`Pipeline` actually calls on
    /// `Stop` (mirrored by hand here, since this input isn't wired into a
    /// real second `Pipeline` in this test), not `consume`.
    #[test]
    fn stopped_source_is_removed_without_an_explicit_remove_source_call() {
        let (mixer, handle) = AudioMixer::new(
            "mixer",
            AudioMixerOptions {
                sample_rate: 48000,
                channels: 1,
            },
        );
        let seen = Arc::new(StdMutex::new(Vec::new()));
        let sink = RecordingSink {
            seen: seen.clone(),
            hlog: element_hlog(ElementType::Other, "recorder", None),
        };
        let pipeline = Pipeline::new("mixer-test-3", mixer, |source, ctx| {
            let branch = ChainBuilder::new(ctx.clone()).build(Box::new(sink));
            source.src_pads()[0].link(branch);
        });
        pipeline.run();

        let mut input_a = handle.add_source("a").unwrap();
        input_a
            .consume(MediaBuffer::Audio(Arc::new(constant_frame(
                0.5, 480, 48000,
            ))))
            .unwrap();
        assert_eq!(handle.source_count(), 1);

        // What a `Queue`/`Pipeline` actually calls on this input's own
        // `Sink` when its upstream capture pipeline is stopped — never
        // `consume(Eos)`, since `AudioCaptureSource` doesn't send one.
        input_a.control(ControlMsg::Stop).unwrap();

        assert_eq!(
            handle.source_count(),
            0,
            "Stop should remove the input immediately, same as remove_source"
        );

        pipeline.stop();
        pipeline.bus().log_events();
    }
}
