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
    elements::AudioFormat,
    elements::filter::audio_resampler::AudioFrameResampler,
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

/// The format every input is resampled to and every output frame carries.
///
/// One value rather than three fields, packed into a word, so an input
/// resampling on its own thread can never catch a sample rate from one
/// setting and a channel count from another. `format` and `channel_layout`
/// are derived from it rather than stored: this mixer works in interleaved
/// `f32` and the layout is whatever is default for the channel count, so a
/// stored copy of either could only ever disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MixFormat {
    pub sample_rate: u32,
    pub channels: u16,
}

impl MixFormat {
    /// Interleaved `f32` — see [`AudioMixer`] on why the mix is summed in
    /// one fixed sample format rather than whatever arrived.
    fn sample_format(self) -> ffmpeg::format::Sample {
        ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed)
    }

    fn channel_layout(self) -> ffmpeg::ChannelLayout {
        ffmpeg::ChannelLayout::default(self.channels as i32)
    }

    fn pack(self) -> u64 {
        ((self.sample_rate as u64) << 16) | self.channels as u64
    }

    fn unpack(packed: u64) -> Self {
        Self {
            sample_rate: (packed >> 16) as u32,
            channels: packed as u16,
        }
    }
}

/// Where the sample deficit is measured from.
///
/// Not `Duration::ZERO` and zero samples, because the mix format can change
/// while this runs and `elapsed × sample_rate` only means anything against
/// the rate that produced the count it is compared to — see
/// [`AudioMixer::mix_tick`].
#[derive(Debug, Clone, Copy)]
struct MixAnchor {
    format: MixFormat,
    elapsed: Duration,
    samples: i64,
}

/// The mix format, shared between the mixer's own tick and every input
/// resampling into it from another thread.
#[derive(Debug)]
struct SharedMixFormat(AtomicU64);

impl SharedMixFormat {
    fn new(format: MixFormat) -> Self {
        Self(AtomicU64::new(format.pack()))
    }

    fn get(&self) -> MixFormat {
        MixFormat::unpack(self.0.load(Ordering::Relaxed))
    }

    /// `false` for a format nothing could be resampled to, which leaves the
    /// running one alone rather than a mixer summing into nothing.
    fn set(&self, format: MixFormat) -> bool {
        if format.sample_rate == 0 || format.channels == 0 {
            return false;
        }
        self.0.store(format.pack(), Ordering::Relaxed);
        true
    }
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
    /// Built lazily from the first frame, and rebuilt when the mix format
    /// moves under it. Kept beside it rather than walked and invalidated
    /// from outside: an input that notices for itself needs no lock held
    /// across somebody else's state, and an input registered after a change
    /// is correct without being told. The *arriving* format moving is the
    /// resampler's own business — [`AudioFrameResampler`] rebuilds for that,
    /// draining what the old context still held first.
    resampler: Option<(MixFormat, AudioFrameResampler)>,
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
        to: MixFormat,
    ) -> std::result::Result<(), AudioMixerError> {
        // Rebuilt when the mix format has moved. A resampler is fixed at
        // both ends when it is created, so one built for the old mix format
        // would keep producing it — silently, and into a mix that no longer
        // wants it.
        if self
            .resampler
            .as_ref()
            .is_none_or(|(built, _)| *built != to)
        {
            self.resampler = Some((
                to,
                AudioFrameResampler::new(AudioFormat::new(
                    to.sample_format(),
                    to.sample_rate,
                    to.channels,
                )),
            ));
        }
        // Through the shared engine rather than a `Context` of this
        // element's own, for the sizing above all. Handed an unallocated
        // output frame, `Context::run` gives it room for exactly as many
        // samples as went in — which is short by the rate ratio whenever an
        // input is *slower* than the mix, so 44.1kHz media into a 48kHz mix
        // left 8% of every frame behind in libswresample's own delay. It is
        // not dropped, which would merely be a click: it comes back on the
        // next call, which is short by the same 8% again, so the input feeds
        // the mix at 92% of real time. The mixer fills the shortfall of each
        // tick with silence — a chop at the tick rate — and what does arrive
        // falls further behind its own picture every second.
        let resampled = {
            let (_, resampler) = self
                .resampler
                .as_mut()
                .expect("just built if it was missing or stale");
            resampler.run(frame)?
        };
        for output in &resampled {
            // Raw bytes, not `plane::<f32>(0)`: `ffmpeg_next`'s `plane::<T>()`
            // always returns exactly `output.samples()` elements of type `T`,
            // which for **packed multi-channel** data (this mixer's own fixed
            // `Sample::F32(Packed)` target — see `AudioMixer::new`) covers only
            // the first `samples()` of the real `samples() * channels`
            // interleaved scalars actually in the buffer, silently dropping
            // every channel past the first once the mix layout has more than
            // one. Same fix, and the same root cause, as
            // `crate::elements::SwAudioEncoder`'s own `absorb_resampled`
            // (found while building that element — this call predates it).
            let samples = output.samples();
            let channels = to.channels as usize;
            let bytes = &output.data(0)[..samples * channels * 4];
            let interleaved =
                // SAFETY: `bytes` is a prefix of an FFmpeg audio plane, which is aligned
                // well past 4 and whose length here is an exact multiple of four bytes —
                // `samples * channels * 4`.
                unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4) };
            self.samples.extend(interleaved.iter().copied());
        }
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
    /// What every input resamples to and every output frame carries. Here
    /// rather than on the mixer, because the inputs reading it are on other
    /// threads and this is the only thing they share with it.
    format: SharedMixFormat,
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
        }))
    }

    /// The format the mix is summed into and emitted at.
    ///
    /// `None` once the mixer is gone.
    pub fn mix_format(&self) -> Option<MixFormat> {
        Some(self.shared.upgrade()?.format.get())
    }

    /// Changes it, from the next tick.
    ///
    /// Returns `false` for a format nothing could be resampled to, and for a
    /// mixer that has already been dropped.
    ///
    /// # What moves with it
    ///
    /// Every input rebuilds its own resampler when it next pushes — each
    /// remembers what its own was built for, so none has to be found and
    /// invalidated from here, and an input registered after this call is
    /// correct without being told.
    ///
    /// [`AudioMixer::time_base`] is `1/sample_rate`, and the output `pts` is
    /// a running sample count in those units. So changing the rate re-means
    /// every timestamp after it, while the ones already downstream were
    /// stamped under the old one. Nothing here can repair that: a muxer
    /// holding a time base from `avformat_write_header` will not be told, and
    /// an encoder was opened for a channel count.
    ///
    /// So this is safe exactly while nothing downstream is reading timestamps
    /// — a level meter, an idle mixer — and it is the caller's to know. In
    /// practice: change it between recordings, not during one. The mixer
    /// itself keeps running either way, which is the point: its `pts` stays
    /// continuous, and a rate change is not a reason to restart the one
    /// element every audio source in the application is registered with.
    pub fn set_mix_format(&self, format: MixFormat) -> bool {
        self.shared
            .upgrade()
            .is_some_and(|shared| shared.format.set(format))
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
        InputContract::Fixed(PortContract::frame(
            MediaKind::AudioFrame,
            MemoryDomain::System,
        ))
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
                    // Read per buffer rather than copied in at `add_source`: this
                    // sink lives on the input's own thread, and a mix format
                    // changed while it is running has to reach it there.
                    input.push(&frame, shared.format.get())?;
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
/// Every input is resampled to this mixer's own `sample_rate`/`channels`,
/// which [`MixerHandle::set_mix_format`] can change while it runs — each
/// input notices for itself and rebuilds its own resampler. The sample
/// format is fixed at `Sample::F32(Packed)`: float headroom during
/// summation, the same reason real mixing consoles work in float even when
/// everything else is integer PCM. An input
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
    /// Cumulative sample count across every emitted frame — see
    /// [`AudioMixer::time_base`].
    samples_emitted: i64,
    /// What the deficit is measured from — see [`MixAnchor`].
    anchor: MixAnchor,
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

        let format = MixFormat {
            sample_rate: options.sample_rate,
            channels: options.channels,
        };
        let shared = Arc::new(MixerShared {
            inputs: Mutex::new(HashMap::new()),
            format: SharedMixFormat::new(format),
            next_input_id: AtomicU64::new(0),
        });
        let pad = SrcPad::with_contract(
            format!("{name}_src"),
            OutputContract::Fixed(PortContract::frame(
                MediaKind::AudioFrame,
                MemoryDomain::System,
            )),
        );
        (
            Self {
                name: name.clone(),
                pp_log,
                shared: shared.clone(),
                pad,
                samples_emitted: 0,
                anchor: MixAnchor {
                    format,
                    elapsed: Duration::ZERO,
                    samples: 0,
                },
            },
            MixerHandle {
                shared: Arc::downgrade(&shared),
            },
        )
    }

    /// The unit each emitted frame's `pts` is expressed in.
    pub fn time_base(&self) -> ffmpeg::Rational {
        ffmpeg::Rational::new(1, self.shared.format.get().sample_rate as i32)
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
        let format = self.shared.format.get();
        // Re-anchored when the format moves. The deficit below is
        // `elapsed × sample_rate` against a running count, and those two are
        // only comparable while the rate that produced them is the same one:
        // measured straight, a drop from 48 kHz to 44.1 makes `expected` fall
        // *below* what has already been emitted, and the mixer goes silent
        // for the minute it takes the new rate to catch up. So the count is
        // kept — `pts` must stay continuous — and only the deficit starts
        // again, from here.
        if format != self.anchor.format {
            pp_info!(
                self,
                "mix format is now {}Hz, {} channel(s)",
                format.sample_rate,
                format.channels
            );
            self.anchor = MixAnchor {
                format,
                elapsed,
                samples: self.samples_emitted,
            };
        }
        let channels = format.channels as usize;
        let since = elapsed.saturating_sub(self.anchor.elapsed);
        let expected =
            self.anchor.samples + (since.as_secs_f64() * format.sample_rate as f64) as i64;
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

        let mut frame =
            ffmpeg::frame::Audio::new(format.sample_format(), needed, format.channel_layout());
        frame.set_rate(format.sample_rate);
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
    fn is_live(&self) -> bool {
        true
    }

    fn is_seekable(&self) -> bool {
        false
    }

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

    /// A file's 44.1kHz sound into a 48kHz mix — the ordinary case, since
    /// most media is 44.1kHz and a capture device is usually 48kHz.
    ///
    /// What the mixer takes per tick is fixed by the wall clock, so an input
    /// resampled *short* is not merely quieter: the shortfall is filled with
    /// silence at the tick rate, and what does arrive falls further behind
    /// its own picture every second. Handed an unallocated output frame,
    /// libswresample sizes it for as many samples as went in, which is 8%
    /// short at this ratio.
    #[test]
    fn a_slower_input_arrives_at_the_mix_s_own_rate() {
        const FRAMES: usize = 43;
        const PER_FRAME: usize = 1024;
        let to = MixFormat {
            sample_rate: 48_000,
            channels: 1,
        };
        let mut input = InputBuffer {
            id: 1,
            resampler: None,
            samples: VecDeque::new(),
            eos: false,
        };

        for _ in 0..FRAMES {
            input
                .push(&constant_frame(0.5, PER_FRAME, 44_100), to)
                .expect("push");
        }

        let arrived = input.samples.len();
        let expected = FRAMES * PER_FRAME * 48_000 / 44_100;
        assert!(
            arrived * 100 >= expected * 99,
            "a second of 44.1kHz sound is still a second of mix: \
             {arrived} samples arrived, {expected} expected"
        );
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

    /// Records the *shape* of every frame rather than a sample of it — what
    /// a format change is visible in.
    struct ShapeSink {
        pp_log: PpLog,
        seen: Arc<StdMutex<Vec<(u32, u16)>>>,
    }

    impl Element for ShapeSink {
        fn name(&self) -> Arc<str> {
            "shapes".into()
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

    impl Sink for ShapeSink {
        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            if let MediaBuffer::Audio(frame) = buf
                && frame.samples() > 0
            {
                self.seen
                    .lock()
                    .unwrap()
                    .push((frame.rate(), frame.channel_layout().channels() as u16));
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

    /// The mixer must keep emitting with nothing feeding it at all.
    ///
    /// It is what a recording attached later is made of, and what the
    /// application's own mixer dock is built on: a file whose audio track
    /// stopped because the last source was removed would be a worse answer
    /// than one carrying silence. The compositor's own version of this — an
    /// empty Scene still composites black — had to be fixed once, so this
    /// says so for the mixer before anybody has to find out.
    ///
    /// `removed_source_stops_contributing` above cannot see it: it asserts
    /// every sample that arrives is zero, which an empty vector satisfies.
    #[test]
    fn a_mixer_with_no_sources_still_emits_silence() {
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
        let pipeline = Pipeline::new("mixer-test-idle", mixer, |source, ctx| {
            let branch = ctx.branch().to(Box::new(sink))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("test pipeline wiring must succeed");
        pipeline.run().unwrap();

        // Nothing is ever added: no `add_source`, no buffer, no removal.
        std::thread::sleep(Duration::from_millis(200));
        pipeline.stop();
        pipeline.bus().log_events();

        let seen = seen.lock().unwrap();
        assert_eq!(handle.source_count(), 0, "this test adds no sources");
        assert!(
            !seen.is_empty(),
            "the mixer must go on emitting with nothing feeding it"
        );
        assert!(
            seen.iter().all(|&sample| sample == 0.0),
            "what it emits with no sources must be silence: {seen:?}"
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
    /// variant (e.g. `FileMuxerStreamSink`).
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

    /// A rate change reaches the output frames, and the mix does not stop
    /// while the new rate "catches up" with the samples already emitted.
    ///
    /// That stall is the whole reason `MixAnchor` exists: the deficit is
    /// `elapsed × sample_rate` against a running count, and measured straight
    /// across a change to a lower rate it goes negative for as long as it
    /// takes the new rate to reach the old count — a minute of silence for a
    /// setting somebody just applied.
    #[test]
    fn changing_the_mix_format_keeps_the_mix_going() {
        let (mixer, handle) = AudioMixer::new(
            "mixer",
            AudioMixerOptions {
                sample_rate: 48000,
                channels: 2,
            },
        );
        let seen: Arc<StdMutex<Vec<(u32, u16)>>> = Arc::new(StdMutex::new(Vec::new()));
        let sink = ShapeSink {
            seen: seen.clone(),
            pp_log: element_pp_log(ElementType::Other, "shapes", None),
        };
        let pipeline = Pipeline::new("mixer-test-rate", mixer, |source, ctx| {
            let branch = ctx.branch().to(Box::new(sink))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("test pipeline wiring must succeed");
        pipeline.run().unwrap();

        std::thread::sleep(Duration::from_millis(150));
        assert_eq!(
            handle.mix_format(),
            Some(MixFormat {
                sample_rate: 48000,
                channels: 2
            })
        );
        // Down, which is the direction that stalls without the anchor, and
        // to mono so the frame's own shape has to move as well.
        assert!(handle.set_mix_format(MixFormat {
            sample_rate: 16000,
            channels: 1,
        }));
        let before = seen.lock().unwrap().len();
        std::thread::sleep(Duration::from_millis(250));
        pipeline.stop();
        pipeline.bus().log_events();

        let seen = seen.lock().unwrap();
        assert!(
            seen.len() > before,
            "the mix stopped after the format changed: {before} frames before, {} after",
            seen.len()
        );
        assert_eq!(
            seen.last().copied(),
            Some((16000, 1)),
            "the new rate and channel count have to reach the output"
        );
    }

    /// A format nothing could be resampled to is refused, and refusing leaves
    /// the running one alone rather than a mixer summing into nothing.
    #[test]
    fn an_impossible_mix_format_is_refused_and_changes_nothing() {
        let (_mixer, handle) = AudioMixer::new(
            "mixer",
            AudioMixerOptions {
                sample_rate: 48000,
                channels: 2,
            },
        );
        for refused in [
            MixFormat {
                sample_rate: 0,
                channels: 2,
            },
            MixFormat {
                sample_rate: 48000,
                channels: 0,
            },
        ] {
            assert!(!handle.set_mix_format(refused), "{refused:?} was accepted");
            assert_eq!(
                handle.mix_format(),
                Some(MixFormat {
                    sample_rate: 48000,
                    channels: 2
                })
            );
        }
    }

    /// And the handle answers rather than taking effect once the mixer is
    /// gone, like every other method on it.
    #[test]
    fn the_mix_format_setter_reports_a_mixer_that_is_gone() {
        let (mixer, handle) = AudioMixer::new(
            "mixer",
            AudioMixerOptions {
                sample_rate: 48000,
                channels: 2,
            },
        );
        drop(mixer);

        assert!(!handle.set_mix_format(MixFormat {
            sample_rate: 16000,
            channels: 1,
        }));
        assert_eq!(handle.mix_format(), None);
    }

    /// A file whose sound is not at the mix's rate, played into the mix.
    ///
    /// The ordinary case, not a corner one: most media is 44.1kHz and most
    /// capture devices are 48kHz, so an application playing a file alongside
    /// a device has a resampler in the path whether it asked for one or not.
    ///
    /// What broke there was invisible to a test of the parts. Every element
    /// reported success, the mix kept coming out at its own rate, and the
    /// resampler was quietly handing back 8% less audio than went in — which
    /// this mixer, whose contract is to keep producing on a wall clock, made
    /// up with silence. So these measure the mix itself.
    mod against_a_file {
        use std::sync::Mutex as StdMutex;

        use super::*;
        use crate::{
            elements::{AppSink, FileDemuxer, Pacer, SwDecoder, TestAudioOptions, TestAudioSource},
            pipeline::Pipeline,
            test_support,
        };

        const MIX_RATE: u32 = 48_000;
        const FILE_RATE: u32 = 44_100;

        /// How long the mix is watched. Long enough for a shortfall of a few
        /// percent per tick to be unmistakable, short enough to stay a test.
        const WATCH: Duration = Duration::from_secs(3);

        /// How much of a mix carrying a continuous tone may be silence, in
        /// parts per thousand.
        ///
        /// The tone never lands exactly on 0.0 for a run of samples, so what
        /// this counts is the mixer's own padding: what it puts in when an
        /// input is short for a tick. A resampler handing back less than it
        /// was given pads *every* tick — 80‰ at the ratio used here. What is
        /// left when nothing is wrong is the occasional scheduling hiccup, a
        /// sub-millisecond gap every few seconds, measured at well under one
        /// part in a thousand.
        const SILENT_PER_MILLE: usize = 10;

        /// Runs shorter than this are a waveform touching zero rather than a
        /// gap in it.
        const HOLE: usize = 4;

        #[derive(Default, Clone)]
        struct MixShape {
            samples: usize,
            silent: usize,
            holes: usize,
            longest_hole: usize,
            current_run: usize,
        }

        impl MixShape {
            fn absorb(&mut self, frame: &ffmpeg::frame::Audio) {
                let bytes = frame.samples() * frame.channels() as usize * 4;
                let data = &frame.data(0)[..bytes.min(frame.data(0).len())];
                for chunk in data.chunks_exact(4) {
                    let sample = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    self.samples += 1;
                    if sample == 0.0 {
                        self.silent += 1;
                        self.current_run += 1;
                    } else {
                        if self.current_run >= HOLE {
                            self.holes += 1;
                            self.longest_hole = self.longest_hole.max(self.current_run);
                        }
                        self.current_run = 0;
                    }
                }
            }

            fn silent_per_mille(&self) -> usize {
                if self.samples == 0 {
                    return 0;
                }
                self.silent * 1000 / self.samples
            }

            fn report(&self) -> String {
                format!(
                    "{} of {} samples silent ({}per mille), in {} hole(s), longest {}",
                    self.silent,
                    self.samples,
                    self.silent_per_mille(),
                    self.holes,
                    self.longest_hole
                )
            }
        }

        /// A mixer running on its own pipeline, with everything it emits
        /// measured.
        fn watched_mix(rate: u32) -> (Arc<Pipeline>, MixerHandle, Arc<StdMutex<MixShape>>) {
            let shape = Arc::new(StdMutex::new(MixShape::default()));
            let listener = AppSink::new("mix-listener", {
                let shape = Arc::clone(&shape);
                move |buffer| {
                    if let MediaBuffer::Audio(frame) = &buffer {
                        shape.lock().expect("mix shape poisoned").absorb(frame);
                    }
                    Ok(())
                }
            });
            let (mixer, handle) = AudioMixer::new(
                "mixer",
                AudioMixerOptions {
                    sample_rate: rate,
                    channels: 2,
                },
            );
            let pipeline = Pipeline::new("mix", mixer, move |source, context| {
                let branch = context.branch().to(Box::new(listener))?;
                context.attach(source, 0, branch)?;
                Ok(())
            })
            .expect("wire the mix");
            pipeline.run().expect("run the mix");
            (pipeline, handle, shape)
        }

        /// The fixture's sound, decoded, paced and pushed into the mix on a
        /// pipeline of its own — which is how an application does it, the
        /// mixer on its own thread with sources coming and going around it.
        ///
        /// The `Pacer` is not decoration. Without one the demuxer reads the
        /// file as fast as it decodes and the mixer's input never runs dry,
        /// so an input handing back less audio than it was given only fills a
        /// queue more slowly and nothing downstream can tell. Paced, the file
        /// arrives at the rate it claims and a shortfall is a tick the mixer
        /// has to finish with silence — which is the whole of what this
        /// measures.
        fn play_into(path: &str, mixer: &MixerHandle) -> Arc<Pipeline> {
            let (demuxer, streams) = FileDemuxer::open("fixture", path).expect("open the fixture");
            let audio = streams
                .iter()
                .find(|stream| stream.kind == ffmpeg::media::Type::Audio)
                .expect("the fixture has sound")
                .index;
            let parameters = demuxer
                .stream_parameters(audio)
                .expect("the audio stream describes itself");
            let time_base = demuxer
                .stream_time_base(audio)
                .expect("the audio stream has a unit");
            let decoder = SwDecoder::new("fixture-decoder", parameters).expect("open the decoder");
            let sink = mixer
                .add_source("fixture".to_owned())
                .expect("the mixer is running");

            let pipeline = Pipeline::new("playback", demuxer, move |source, context| {
                let branch = context
                    .branch()
                    .pipe(decoder)
                    .queue("audio", 32)
                    .pipe(Pacer::new("fixture-pacer", time_base)?)
                    .to(sink)?;
                context.attach(source, audio, branch)?;
                Ok(())
            })
            .expect("wire the playback pipeline");
            pipeline.run().expect("play the fixture");
            pipeline
        }

        /// The fixture has to be what these tests assume, or what they measure
        /// is something else. Cheap, and it fails at the generator rather than
        /// three assertions later.
        #[test]
        fn the_fixture_carries_sound_at_a_rate_the_mix_does_not_run_at() {
            let fixture = test_support::synthesize("mixed-rate", 2.0, FILE_RATE);
            let input =
                ffmpeg::format::input(&fixture.path).expect("the fixture is a readable container");
            let audio = input
                .streams()
                .find(|stream| stream.parameters().medium() == ffmpeg::media::Type::Audio)
                .expect("the fixture has an audio stream");
            let decoder = ffmpeg::codec::context::Context::from_parameters(audio.parameters())
                .expect("the audio stream describes itself")
                .decoder()
                .audio()
                .expect("it is audio");

            assert_eq!(
                decoder.rate(),
                fixture.audio_rate,
                "the fixture must carry the rate it was asked for"
            );
            assert_ne!(FILE_RATE, MIX_RATE, "otherwise nothing is resampled");
            assert_eq!(decoder.channels(), fixture.channels);
            assert!(
                input
                    .streams()
                    .any(|stream| stream.parameters().medium() == ffmpeg::media::Type::Video),
                "a Source that occupies a rectangle needs its picture too"
            );
        }

        /// A 44.1kHz file into a 48kHz mix has to fill every tick of it.
        ///
        /// Not "arrive" as in the mix keeps producing — it does that with or
        /// without an input, filling whatever an input is short by with
        /// silence, which is exactly what made this so quiet. What is
        /// measured is the silence.
        #[test]
        fn a_file_below_the_mix_rate_fills_every_tick_of_it() {
            let fixture = test_support::synthesize("mixed-rate-mix", 6.0, FILE_RATE);
            let (mix, handle, shape) = watched_mix(MIX_RATE);
            let playback = play_into(&fixture.path.to_string_lossy(), &handle);

            // What a mix emits before its input has said anything is silence,
            // correctly, so the measurement starts after the first samples.
            thread::sleep(Duration::from_millis(500));
            *shape.lock().expect("mix shape poisoned") = MixShape::default();
            thread::sleep(WATCH);
            let measured = shape.lock().expect("mix shape poisoned").clone();
            playback.stop();
            mix.stop();

            assert!(
                measured.samples > 0,
                "the mix produced nothing to look at in {WATCH:?}"
            );
            assert!(
                measured.silent_per_mille() <= SILENT_PER_MILLE,
                "a {FILE_RATE}Hz file did not fill a {MIX_RATE}Hz mix: {}",
                measured.report()
            );
        }

        /// The same mixer fed at its own rate, so a failure above is read as
        /// what it is. Nothing is resampled here, and a hole would mean
        /// something other than the rate conversion.
        #[test]
        fn a_source_at_the_mix_rate_fills_it_too() {
            let (mix, handle, shape) = watched_mix(MIX_RATE);
            let tone = TestAudioSource::new(
                "tone",
                TestAudioOptions {
                    sample_rate: MIX_RATE,
                    channels: 2,
                    frequency: 440.0,
                },
            );
            let sink = handle
                .add_source("tone".to_owned())
                .expect("the mixer is running");
            let source = Pipeline::new("tone", tone, move |source, context| {
                let branch = context.branch().to(sink)?;
                context.attach(source, 0, branch)?;
                Ok(())
            })
            .expect("wire the tone");
            source.run().expect("run the tone");

            thread::sleep(Duration::from_millis(500));
            *shape.lock().expect("mix shape poisoned") = MixShape::default();
            thread::sleep(WATCH);
            let measured = shape.lock().expect("mix shape poisoned").clone();
            source.stop();
            mix.stop();

            assert!(
                measured.silent_per_mille() <= SILENT_PER_MILLE,
                "a source at the mix's own rate did not fill it: {}",
                measured.report()
            );
        }
    }
}
