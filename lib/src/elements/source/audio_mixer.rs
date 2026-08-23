use std::{
    collections::{HashMap, VecDeque},
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicU64, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use crate::pp_log::{PpLog, pp_error, pp_info};
use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    control::{ControlMsg, ControlReceiver, drain_control},
    element::{Element, ElementType, Sink, Source, SourceElement, element_pp_log},
    error::Result,
    pad::SrcPad,
    schedule::ActiveTimeline,
};

/// How often [`AudioMixer::run`] mixes and emits a combined frame — same
/// role as [`crate::elements::DxgiCaptureSource`]'s own `POLL_GRANULARITY`/
/// `crate::elements::WasapiCaptureSource`'s `POLL_INTERVAL`: bounds `Stop`
/// latency and sets the mixer's own output granularity.
const TICK_INTERVAL: Duration = Duration::from_millis(20);

/// Errors specific to `AudioMixer`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum AudioMixerError {
    /// FFmpeg rejected resampler creation or audio conversion.
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg_next::Error),

    /// Seeking was requested on a live mixer with no stored timeline.
    #[error("AudioMixer doesn't support seeking a live mix")]
    SeekUnsupported,

    /// An input sink received a buffer other than decoded audio or end-of-stream.
    #[error("AudioMixer inputs only accept Audio or Eos buffers, got {0}")]
    UnsupportedBuffer(&'static str),
}

/// Construction-time options for [`AudioMixer::new`] — the mixer's fixed
/// *output* format. Every input is resampled to match this on the way in
/// (see `InputBuffer::push`); the mixer never adapts to whatever an
/// input happens to produce.
#[derive(Debug, Clone, Copy)]
pub struct AudioMixerOptions {
    /// Sample rate of every mixed output frame, in hertz.
    pub sample_rate: u32,
    /// Channel count of every mixed output frame.
    pub channels: u16,
}

/// One input's own resampler and accumulated (already-resampled,
/// interleaved `f32`) samples, waiting to be drained by the next
/// [`AudioMixer::mix_tick`]. The resampler is built lazily from the first
/// frame this input ever sees (its `format`/`channel_layout`/`rate`
/// self-describe — no need for [`MixerHandle::add_source`] to be told
/// this upfront).
struct InputBuffer {
    /// Identity of this particular registration. The name can be reused,
    /// but an older sink must not be allowed to touch its replacement.
    id: u64,
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
        // Raw bytes, not `plane::<f32>(0)`: `ffmpeg_next`'s `plane::<T>()`
        // always returns exactly `output.samples()` elements of type `T`,
        // which for **packed multi-channel** data (this mixer's own fixed
        // `Sample::F32(Packed)` target — see `AudioMixer::new`) covers only
        // the first `samples()` of the real `samples() * channels`
        // interleaved scalars actually in the buffer, silently dropping
        // every channel past the first once `target_layout` has more than
        // one. Same fix, and the same root cause, as
        // `crate::elements::SwAudioEncoder`'s own `absorb_resampled`
        // (found while building that element — this call predates it).
        let samples = output.samples();
        let channels = target_layout.channels() as usize;
        let bytes = &output.data(0)[..samples * channels * 4];
        let interleaved =
            // SAFETY: `bytes` is a prefix of an FFmpeg audio plane, which is aligned
            // well past 4 and whose length here is an exact multiple of four bytes —
            // `samples * channels * 4`.
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4) };
        self.samples.extend(interleaved.iter().copied());
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
    /// Issues a distinct identity for every `add_source` call, including
    /// replacements registered under an existing name.
    next_input_id: AtomicU64,
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
    /// Registers a new input under `name` and returns a [`Sink`] to use as
    /// a detached branch terminal. Build and attach it inside that source's
    /// own `Pipeline::new` wiring closure — a
    /// *different* pipeline/thread than this mixer's own, which is exactly
    /// the point). `None` once the mixer itself is gone. Calling this
    /// again with a name already in use replaces that input outright
    /// (whatever it had buffered is dropped) rather than erroring — same
    /// "just do what was asked" spirit as `HashMap::insert`. A sink from
    /// the previous registration then becomes inert: its data, `Eos`, and
    /// `Stop` cannot affect the replacement sharing its name.
    ///
    /// The input endpoint appears in the upstream source pipeline's graph
    /// when attached through [`crate::element::Context::attach`]. The graph
    /// intentionally does not invent a cross-pipeline edge to the mixer.
    pub fn add_source(&self, name: impl Into<String>) -> Option<Box<dyn Sink>> {
        let shared = self.shared.upgrade()?;
        let name: Arc<str> = name.into().into();
        let id = shared.next_input_id.fetch_add(1, Ordering::Relaxed);
        shared.inputs.lock().unwrap().insert(
            name.clone(),
            InputBuffer {
                id,
                resampler: None,
                samples: VecDeque::new(),
                eos: false,
            },
        );
        Some(Box::new(MixerInputSink {
            name: name.clone(),
            id,
            pp_log: element_pp_log(ElementType::AudioMixer, &name, None),
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

    /// Returns the number of inputs currently registered with the live mixer.
    ///
    /// Returns zero after the mixer has been dropped.
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
pub struct MixerInputSink {
    pp_log: PpLog,
    name: Arc<str>,
    /// Identity returned by the corresponding `add_source` call. Compared
    /// with the map entry before every mutation so a stale sink cannot
    /// write to or remove a same-name replacement.
    id: u64,
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

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Sink for MixerInputSink {
    /// Every input is summed sample by sample, so each carries decoded
    /// audio just as the mixed output does.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::of(MediaKind::AudioFrame).in_memory(MemoryDomain::System),
        )
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        let Some(shared) = self.shared.upgrade() else {
            return Ok(()); // mixer's own pipeline already ended — nothing to feed
        };
        match buf {
            MediaBuffer::Audio(frame) => {
                let mut inputs = shared.inputs.lock().unwrap();
                if let Some(input) = inputs.get_mut(&self.name)
                    && input.id == self.id
                {
                    input.push(
                        &frame,
                        self.target_format,
                        self.target_layout,
                        self.target_rate,
                    )?;
                }
                // Absent means `remove_source` raced ahead of this frame;
                // an ID mismatch means this name was replaced. Dropping
                // the frame is correct in both cases.
            }
            MediaBuffer::Eos => {
                let mut inputs = shared.inputs.lock().unwrap();
                if let Some(input) = inputs.get_mut(&self.name)
                    && input.id == self.id
                {
                    input.eos = true;
                }
            }
            other => {
                pp_error!(self, "unsupported buffer: expected Audio or Eos");
                return Err(AudioMixerError::UnsupportedBuffer(other.kind()).into());
            }
        }
        Ok(())
    }

    /// No downstream of its own to cascade to — this is a leaf input slot,
    /// not a passthrough. `Stop` removes this input immediately, same as
    /// [`MixerHandle::remove_source`]: `Stop` means abandon now, not drain
    /// to a natural `Eos` (see `ControlMsg::Stop`'s own docs), and for a
    /// live capture source — `WasapiCaptureSource`
    /// included — `Stop` is the *only* shutdown signal that ever arrives;
    /// it never reaches `Eos` on its own. Relying on `Eos` alone to clean
    /// up (as an earlier version of this did) left a stale entry in
    /// `shared.inputs` forever whenever a caller stopped its capture
    /// pipeline normally instead of remembering to call
    /// `MixerHandle::remove_source` by hand. `Pause`/`Resume`/`Seek` need
    /// no handling here — this input has no thread or queue of its own to
    /// freeze/resume, and a live capture source doesn't seek. Removal is
    /// conditional on the registration ID: a late `Stop` from a replaced
    /// sink must not remove the newer input using the same name.
    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        if msg == ControlMsg::Stop
            && let Some(shared) = self.shared.upgrade()
        {
            let mut inputs = shared.inputs.lock().unwrap();
            if inputs
                .get(&self.name)
                .is_some_and(|input| input.id == self.id)
            {
                inputs.remove(&self.name);
            }
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
/// ticking every `TICK_INTERVAL` to sum however many samples each
/// currently-attached input has ready — because mixing has to keep
/// producing *something* on a steady clock even when some (or all) inputs
/// have gone quiet, the same reason
/// `WasapiCaptureSource` synthesizes silence for gaps
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
pub struct AudioMixer {
    pp_log: PpLog,
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
    /// unlike [`crate::elements::TeeBuilder::new`], since `AudioMixer` has no
    /// chains of its own for a handle to build).
    pub fn new(name: impl Into<String>, options: AudioMixerOptions) -> (Self, MixerHandle) {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::AudioMixer, &name, None);
        pp_info!(
            pp_log: &pp_log,
            "created: {}Hz, {} channel(s)",
            options.sample_rate,
            options.channels
        );
        let format = ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed);
        let channel_layout = ffmpeg::ChannelLayout::default(options.channels as i32);
        let shared = Arc::new(MixerShared {
            inputs: Mutex::new(HashMap::new()),
            next_input_id: AtomicU64::new(0),
        });
        let pad = SrcPad::with_contract(
            format!("{name}_src"),
            OutputContract::Fixed(
                PortContract::of(MediaKind::AudioFrame).in_memory(MemoryDomain::System),
            ),
        );
        (
            Self {
                name: name.clone(),
                pp_log,
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
    /// lockstep with `elapsed` (a no-op if nothing's owed yet — same
    /// wall-clock-deficit shape as
    /// [`crate::elements::WasapiCaptureSource::fill_silence_gap`], just
    /// summing real contributions from every input instead of emitting
    /// pure silence). `elapsed` already excludes time spent frozen inside
    /// `Pause` (see [`crate::schedule::ActiveTimeline`]) so a `Pause`/
    /// `Resume` pair doesn't get summed as a burst of owed samples the
    /// moment playback resumes. Drops any input that's both `eos` and
    /// fully drained — it contributed its last real samples on a previous
    /// tick and has nothing left to give.
    fn mix_tick(&mut self, elapsed: Duration, bus: &Bus) {
        let channels = self.channels as usize;
        let expected = (elapsed.as_secs_f64() * self.sample_rate as f64) as i64;
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
        // SAFETY: viewing an `f32` slice as bytes, which is always aligned and
        // exactly `size_of_val` long. The read is only as wide as `mixed` itself;
        // what the *destination* can take is the separate bound the comment below
        // describes.
        let bytes = unsafe {
            std::slice::from_raw_parts(mixed.as_ptr() as *const u8, std::mem::size_of_val(&*mixed))
        };
        // `frame.data_mut(0)`'s length is FFmpeg's own padded linesize,
        // not necessarily `mixed.len() * 4` exactly — only ever write that
        // tight amount (same bound `frame.plane::<T>()` itself reads via
        // `samples()`), never assume the destination's full length
        // matches `bytes` (see `WasapiCaptureSource::build_frame`'s own
        // identical fix).
        frame.data_mut(0)[..bytes.len()].copy_from_slice(bytes);
        frame.set_pts(Some(self.samples_emitted));
        self.samples_emitted += needed as i64;

        if let Err(error) = self.pad.push(MediaBuffer::Audio(Arc::new(frame))) {
            bus.post(
                &self.pp_log,
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

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for AudioMixer {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl SourceElement for AudioMixer {
    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        pp_info!(self, "started");
        let mut timeline = ActiveTimeline::new(Instant::now());
        loop {
            let outcome = drain_control(control, self, bus)?;
            if outcome.stopped {
                pp_info!(self, "stopped");
                return Ok(());
            }
            timeline.account_pause(outcome.paused_for);
            thread::sleep(TICK_INTERVAL);
            self.mix_tick(timeline.elapsed(Instant::now()), bus);
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

    use crate::pp_log::PpLog;

    use super::*;
    use crate::pipeline::Pipeline;

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

    struct RecordingSink {
        pp_log: PpLog,
        seen: Arc<StdMutex<Vec<f32>>>,
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

    fn constant_stereo_frame(
        left: f32,
        right: f32,
        samples: usize,
        rate: u32,
    ) -> ffmpeg::frame::Audio {
        let mut frame = ffmpeg::frame::Audio::new(
            ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed),
            samples,
            ffmpeg::ChannelLayout::default(2),
        );
        frame.set_rate(rate);
        let bytes = frame.data_mut(0);
        let floats =
            // SAFETY: `bytes` is this frame's own plane, which FFmpeg aligns well past
            // 4, and `samples * 2` f32s is what the frame was allocated for.
            unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut f32, samples * 2) };
        for pair in floats.chunks_mut(2) {
            pair[0] = left;
            pair[1] = right;
        }
        frame
    }

    struct StereoRecordingSink {
        pp_log: PpLog,
        seen: Arc<StdMutex<Vec<(f32, f32)>>>,
    }

    impl Element for StereoRecordingSink {
        fn name(&self) -> Arc<str> {
            "stereo-recorder".into()
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

    impl Sink for StereoRecordingSink {
        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            if let MediaBuffer::Audio(frame) = buf
                && frame.samples() > 0
            {
                // Raw bytes, not `plane::<f32>(0)`, for the same reason
                // `InputBuffer::push` above does: `AudioMixer`'s output is
                // packed multi-channel, and `plane::<T>()` only ever
                // returns `samples()` elements regardless of channel
                // count — reading channel 1 through it would silently
                // read the wrong offset (still inside channel 0's data),
                // not the second channel.
                let samples = frame.samples();
                let bytes = &frame.data(0)[..samples * 2 * 4];
                // SAFETY: `bytes` is a prefix of the frame's plane, aligned by FFmpeg and
                // cut to exactly `samples * 2 * 4` bytes just above.
                let floats = unsafe {
                    std::slice::from_raw_parts(bytes.as_ptr() as *const f32, samples * 2)
                };
                self.seen.lock().unwrap().push((floats[0], floats[1]));
            }
            Ok(())
        }
        fn control(&mut self, _msg: ControlMsg) -> Result<()> {
            Ok(())
        }
    }

    /// Regression test for the packed-multichannel `InputBuffer::push` bug
    /// (see the comment there): two stereo inputs, each with distinct,
    /// asymmetric L/R values, should sum per-channel without the channels
    /// bleeding into each other or silently dropping to zero. Before the
    /// fix, `plane::<f32>(0)` under-read the resampled buffer (only
    /// `samples()` interleaved scalars instead of `samples() * channels`),
    /// which desynced every input's channel alignment.
    #[test]
    fn mixes_stereo_sources_without_channel_corruption() {
        let (mixer, handle) = AudioMixer::new(
            "mixer",
            AudioMixerOptions {
                sample_rate: 48000,
                channels: 2,
            },
        );
        let seen = Arc::new(StdMutex::new(Vec::new()));
        let sink = StereoRecordingSink {
            seen: seen.clone(),
            pp_log: element_pp_log(ElementType::Other, "stereo-recorder", None),
        };

        let pipeline = Pipeline::new("mixer-stereo-test", mixer, |source, ctx| {
            let branch = ctx.branch().to(Box::new(sink))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("test pipeline wiring must succeed");
        pipeline.run().unwrap();

        let mut input_a = handle.add_source("a").expect("mixer still alive");
        let mut input_b = handle.add_source("b").expect("mixer still alive");

        let stop = Arc::new(AtomicBool::new(false));
        let feeder_stop = stop.clone();
        let feeder = std::thread::spawn(move || {
            while !feeder_stop.load(Ordering::Relaxed) {
                let _ = input_a.consume(MediaBuffer::Audio(Arc::new(constant_stereo_frame(
                    0.2, -0.1, 480, 48000,
                ))));
                let _ = input_b.consume(MediaBuffer::Audio(Arc::new(constant_stereo_frame(
                    0.1, -0.2, 480, 48000,
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
        let steady = &seen[3..seen.len() - 2];
        for &(left, right) in steady {
            assert!(
                (left - 0.3).abs() < 0.01,
                "expected left channel ~0.3, got {left} in {seen:?}"
            );
            assert!(
                (right - -0.3).abs() < 0.01,
                "expected right channel ~-0.3, got {right} in {seen:?}"
            );
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
            pp_log: element_pp_log(ElementType::Other, "recorder", None),
        };

        let pipeline = Pipeline::new("mixer-test", mixer, |source, ctx| {
            let branch = ctx.branch().to(Box::new(sink))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("test pipeline wiring must succeed");
        pipeline.run().unwrap();

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
            pp_log: element_pp_log(ElementType::Other, "recorder", None),
        };
        let pipeline = Pipeline::new("mixer-test-2", mixer, |source, ctx| {
            let branch = ctx.branch().to(Box::new(sink))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("test pipeline wiring must succeed");
        pipeline.run().unwrap();

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
    /// shutdown signal a live source like `WasapiCaptureSource` ever sends,
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
            pp_log: element_pp_log(ElementType::Other, "recorder", None),
        };
        let pipeline = Pipeline::new("mixer-test-3", mixer, |source, ctx| {
            let branch = ctx.branch().to(Box::new(sink))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("test pipeline wiring must succeed");
        pipeline.run().unwrap();

        let mut input_a = handle.add_source("a").unwrap();
        input_a
            .consume(MediaBuffer::Audio(Arc::new(constant_frame(
                0.5, 480, 48000,
            ))))
            .unwrap();
        assert_eq!(handle.source_count(), 1);

        // What a `Queue`/`Pipeline` actually calls on this input's own
        // `Sink` when its upstream capture pipeline is stopped — never
        // `consume(Eos)`, since `WasapiCaptureSource` doesn't send one.
        input_a.control(ControlMsg::Stop).unwrap();

        assert_eq!(
            handle.source_count(),
            0,
            "Stop should remove the input immediately, same as remove_source"
        );

        pipeline.stop();
        pipeline.bus().log_events();
    }

    /// Re-registering a name replaces its input buffer, but callers may
    /// still hold the sink returned for the old registration. Every late
    /// operation through that stale sink must be inert rather than being
    /// redirected to (or deleting) the replacement merely because the map
    /// key is the same.
    #[test]
    fn replacing_an_input_by_name_invalidates_the_stale_sink() {
        let (_mixer, handle) = AudioMixer::new(
            "mixer",
            AudioMixerOptions {
                sample_rate: 48000,
                channels: 1,
            },
        );
        let mut stale = handle.add_source("mic").expect("mixer still alive");
        let mut current = handle.add_source("mic").expect("mixer still alive");
        assert_eq!(handle.source_count(), 1);

        stale
            .consume(MediaBuffer::Audio(Arc::new(constant_frame(
                0.75, 480, 48000,
            ))))
            .unwrap();
        stale.consume(MediaBuffer::Eos).unwrap();
        stale.control(ControlMsg::Stop).unwrap();

        assert_eq!(
            handle.source_count(),
            1,
            "a stale sink's Stop must not remove its replacement"
        );
        let shared = handle.shared.upgrade().expect("mixer still alive");
        {
            let inputs = shared.inputs.lock().unwrap();
            let input = inputs.get("mic").expect("replacement remains registered");
            assert!(
                input.resampler.is_none() && input.samples.is_empty(),
                "stale audio must not enter the replacement buffer"
            );
            assert!(!input.eos, "stale Eos must not mark the replacement ended");
        }

        // The current sink still owns the registration and therefore
        // remains fully functional.
        current
            .consume(MediaBuffer::Audio(Arc::new(constant_frame(
                0.25, 480, 48000,
            ))))
            .unwrap();
        current.consume(MediaBuffer::Eos).unwrap();
        {
            let inputs = shared.inputs.lock().unwrap();
            let input = inputs.get("mic").expect("replacement remains registered");
            assert!(input.resampler.is_some(), "current audio was not accepted");
            assert!(input.eos, "current Eos was not accepted");
        }

        current.control(ControlMsg::Stop).unwrap();
        assert_eq!(handle.source_count(), 0);
    }

    /// A misrouted `Packet`/`Video` buffer used to be silently logged and
    /// dropped — no `BusEvent::Error`, no way for a misconfigured pipeline
    /// to ever find out. Matches the typed-error pattern every other
    /// `Sink` in this codebase already uses for a wrong `MediaBuffer`
    /// variant (e.g. `Mp4MuxerStreamSink`).
    #[test]
    fn rejects_buffers_that_are_neither_audio_nor_eos() {
        let (mixer, handle) = AudioMixer::new(
            "mixer",
            AudioMixerOptions {
                sample_rate: 48000,
                channels: 2,
            },
        );
        let mut input = handle.add_source("a").expect("mixer still alive");

        let error = input
            .consume(MediaBuffer::Packet(Arc::new(ffmpeg::Packet::empty())))
            .expect_err("a Packet buffer must be rejected, not silently dropped");
        assert!(
            matches!(
                error,
                crate::error::Error::AudioMixerError(AudioMixerError::UnsupportedBuffer("Packet"))
            ),
            "unexpected error: {error:?}"
        );

        drop(mixer);
    }
}
