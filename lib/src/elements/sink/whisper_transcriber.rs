//! Speech to timed text, through whisper.cpp.

use std::sync::Arc;

use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

use crate::pp_log::{PpLog, pp_debug, pp_info, pp_trace};
use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, PortContract},
    control::ControlMsg,
    element::{Element, ElementType, Sink, element_pp_log},
    error::Result,
};

/// The rate Whisper's own front end works at. Nothing else is accepted, and
/// nothing here resamples: put an
/// [`AudioResampler`](crate::elements::AudioResampler) in front instead,
/// the way the ONNX detector has a scaler in front to reach the size its
/// own model wants.
pub const SAMPLE_RATE: u32 = 16_000;

/// Errors produced by [`WhisperTranscriber`].
#[derive(Debug, ThisError)]
pub enum WhisperTranscriberError {
    /// The model file could not be read, or is not a whisper.cpp model.
    #[error("could not load the whisper model at {path}: {source}")]
    ModelLoad {
        /// What was asked for.
        path: String,
        /// whisper.cpp's own complaint.
        source: whisper_rs::WhisperError,
    },
    /// whisper.cpp refused to create the per-run state.
    #[error("could not create whisper state: {0}")]
    State(whisper_rs::WhisperError),
    /// Inference failed on a chunk.
    #[error("whisper failed to transcribe a chunk: {0}")]
    Inference(whisper_rs::WhisperError),
    /// Audio arrived in a shape this cannot read.
    #[error(
        "WhisperTranscriber needs {SAMPLE_RATE}Hz mono f32 audio, got {rate}Hz \
         {channels} channel(s) as {format:?}; put an AudioResampler in front"
    )]
    UnsupportedAudio {
        /// The rate that arrived.
        rate: u32,
        /// The channel count that arrived.
        channels: u16,
        /// The sample format that arrived.
        format: ffmpeg::format::Sample,
    },
    /// A buffer that is not audio reached this sink.
    #[error("WhisperTranscriber only accepts audio frames, got {0:?}")]
    UnsupportedBuffer(&'static str),
}

/// One stretch of speech, and when it was said.
///
/// The times are milliseconds from the first sample this transcriber was
/// given, which is its own timeline and not the source's. A caller that
/// needs the source's adds whatever offset it knows — for a file read from
/// the beginning that offset is zero, which is why the example does not
/// mention it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    /// When the speech starts, in milliseconds.
    pub start_ms: i64,
    /// When it ends, in milliseconds.
    pub end_ms: i64,
    /// What was said, trimmed. Never empty — a segment with nothing in it
    /// is not reported.
    pub text: String,
}

/// How much audio to gather before running inference, and how much of the
/// result to trust.
///
/// # Why a chunk at all
///
/// Whisper's encoder takes exactly 30 seconds of audio: the positional
/// embeddings are sized for it, so a shorter input is padded and a longer
/// one cannot be given at all. Transcribing a stream is therefore a loop,
/// and these two numbers are what the loop is made of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkPolicy {
    /// How much new audio to gather before each inference, in milliseconds.
    ///
    /// This is the floor on how late a line can be: nothing is transcribed
    /// until this much has arrived. Larger chunks give the model more to
    /// work with and cost more delay.
    pub chunk_ms: u32,
    /// How much of the newest audio to distrust, in milliseconds.
    ///
    /// The model's output near the end of what it was given is unstable —
    /// a word half-heard is guessed at, and the guess changes once the rest
    /// of it arrives. Rather than emit that and correct it later, this much
    /// of the tail is left alone and picked up on the next round, when it
    /// is no longer at the edge.
    ///
    /// The cost is that a line waits this much longer. What it buys is that
    /// a line, once reported, never changes — so a caller never has to
    /// withdraw text it has already written to a file or drawn on a screen.
    pub live_edge_ms: u32,
}

impl Default for ChunkPolicy {
    /// Four seconds of audio, holding back the last one.
    ///
    /// The delay a caller sees is the sum of the two, so this is text about
    /// five seconds behind the speech — before whatever inference itself
    /// costs. For a recording that is nothing, because a subtitle's place
    /// in the file is its timestamp and not its arrival. For captions on a
    /// screen it is very visible, and the numbers are here to be changed.
    fn default() -> Self {
        Self {
            chunk_ms: 4_000,
            live_edge_ms: 1_000,
        }
    }
}

impl ChunkPolicy {
    fn chunk_samples(&self) -> usize {
        (self.chunk_ms as usize * SAMPLE_RATE as usize) / 1000
    }

    fn live_edge_samples(&self) -> usize {
        (self.live_edge_ms as usize * SAMPLE_RATE as usize) / 1000
    }
}

/// What the [`Chunker`] wants run, and which of the answer to keep.
#[derive(Debug, PartialEq)]
struct Inference {
    /// Where `audio` starts, in milliseconds from the first sample ever
    /// given to the chunker. Whisper reports its segments relative to the
    /// audio it was handed, so this is what turns those into stream time.
    offset_ms: i64,
    /// Everything to feed the model: the chunk to transcribe, with the one
    /// before it in front as context.
    audio: Vec<f32>,
    /// Segments starting before this are already reported; ignore them.
    accept_from_ms: i64,
    /// Segments starting at or after this are too near the edge to trust;
    /// the next round will cover them.
    accept_until_ms: i64,
}

/// Decides what to transcribe and what of the result to believe.
///
/// Separated from the model so that the awkward part — the arithmetic of
/// overlapping windows — can be tested without one. Every decision this
/// makes is a function of sample counts, and none of it needs whisper.cpp
/// to be present, let alone a multi-gigabyte model file.
///
/// # The overlap
///
/// Each inference is given the previous chunk as well as the current one.
/// A word spanning the boundary is otherwise seen half by one round and
/// half by the next, and comes out mangled or twice. Feeding the context
/// costs a second pass over audio already transcribed — roughly doubling
/// the work — and buys words that survive the seam.
struct Chunker {
    policy: ChunkPolicy,
    /// New audio, not yet transcribed.
    pending: Vec<f32>,
    /// The chunk before `pending`, kept to feed as context.
    context: Vec<f32>,
    /// Where `context` starts, in samples from the first ever pushed.
    context_start: u64,
    /// Everything before this has been reported, in samples.
    accepted_until: u64,
}

impl Chunker {
    fn new(policy: ChunkPolicy) -> Self {
        Self {
            policy,
            pending: Vec::new(),
            context: Vec::new(),
            context_start: 0,
            accepted_until: 0,
        }
    }

    fn samples_to_ms(samples: u64) -> i64 {
        (samples as i64 * 1000) / SAMPLE_RATE as i64
    }

    /// Takes more audio, and answers the inference it now wants run.
    ///
    /// `None` while there is not yet a full chunk. One call yields at most
    /// one inference, so a caller handing over more than a chunk at a time
    /// should keep calling until it answers `None`.
    fn push(&mut self, samples: &[f32]) -> Option<Inference> {
        self.pending.extend_from_slice(samples);
        self.take_chunk()
    }

    fn take_chunk(&mut self) -> Option<Inference> {
        let chunk = self.policy.chunk_samples();
        if self.pending.len() < chunk {
            return None;
        }
        let rest = self.pending.split_off(chunk);
        let current = std::mem::replace(&mut self.pending, rest);

        // The edge is where the audio runs out, and the last stretch before
        // it is what the model is least sure of.
        let current_start = self.context_start + self.context.len() as u64;
        let current_end = current_start + current.len() as u64;
        let accept_until = current_end.saturating_sub(self.policy.live_edge_samples() as u64);

        let inference = self.inference(&current, accept_until);

        self.context_start = current_start;
        self.context = current;
        self.accepted_until = accept_until.max(self.accepted_until);
        Some(inference)
    }

    /// The last inference, once no more audio is coming.
    ///
    /// Whatever is left is transcribed however short it is, and the live
    /// edge is not held back: there is no later round to pick it up, and
    /// nothing more will arrive to change what the model hears. `None` when
    /// everything has already been reported.
    fn flush(&mut self) -> Option<Inference> {
        let current = std::mem::take(&mut self.pending);
        let current_start = self.context_start + self.context.len() as u64;
        let current_end = current_start + current.len() as u64;
        if current_end <= self.accepted_until {
            return None;
        }
        let inference = self.inference(&current, current_end);
        self.context_start = current_start;
        self.context = current;
        self.accepted_until = current_end;
        Some(inference)
    }

    /// Builds the request: context then the chunk, and the window of the
    /// answer that is neither already said nor too new to trust.
    fn inference(&self, current: &[f32], accept_until: u64) -> Inference {
        let mut audio = Vec::with_capacity(self.context.len() + current.len());
        audio.extend_from_slice(&self.context);
        audio.extend_from_slice(current);
        Inference {
            offset_ms: Self::samples_to_ms(self.context_start),
            audio,
            accept_from_ms: Self::samples_to_ms(self.accepted_until),
            accept_until_ms: Self::samples_to_ms(accept_until),
        }
    }

    /// Forgets everything, for a [`ControlMsg::Flush`] or a seek.
    ///
    /// The timeline is not reset: positions stay where they were, because
    /// what the caller does with a seek is its own business and inventing a
    /// new zero here would only disagree with it.
    fn reset(&mut self) {
        self.pending.clear();
        self.context.clear();
    }
}

/// Turns speech into timed text, one stretch at a time.
///
/// A terminal sink, like the ONNX detector and for the same reason: what
/// comes out is not media. The callback receives
/// [`Segment`]s as they are settled, and what to do with them — write an
/// SRT, build [`crate::subtitle`] packets for a muxer, draw them on a
/// preview — is the caller's.
///
/// # What it takes
///
/// [`SAMPLE_RATE`] mono `f32`, and nothing else. An
/// [`AudioResampler`](crate::elements::AudioResampler) in front is how
/// audio gets into that shape; this refuses anything else rather than
/// resampling badly on its own.
///
/// # Threading and pace
///
/// `consume` is synchronous and slow: it returns when inference does. Put a
/// [`Queue`](crate::queue::Queue) in front, which is this crate's thread
/// boundary, so the branch producing the audio is not held up by the model.
///
/// A model slower than real time cannot be rescued by any of this. What
/// happens then is that the queue fills and drops audio, and the text has
/// gaps — which is the honest outcome, and better than an unbounded backlog
/// that turns into unbounded memory.
///
/// # EOS
///
/// [`MediaBuffer::Eos`] transcribes whatever is left, including the stretch
/// normally held back at the live edge — nothing more is coming, so nothing
/// can change what the model hears. A caller that stops the pipeline with
/// [`ControlMsg::Stop`] instead abandons that audio, which is what `Stop`
/// means everywhere else in this crate.
pub struct WhisperTranscriber<F> {
    pp_log: PpLog,
    name: Arc<str>,
    /// Made once and reused for every chunk.
    ///
    /// Creating one allocates the model's working buffers and, on a GPU
    /// backend, initialises the backend itself — which is not a per-chunk
    /// cost anybody wants to pay. It holds its own reference to the model,
    /// so the context it came from does not have to be kept beside it.
    state: WhisperState,
    chunker: Chunker,
    on_segment: F,
    /// Reported once, because it is the same answer every time and a line
    /// per chunk would bury the log.
    reported_zero_length: bool,
}

impl<F> WhisperTranscriber<F>
where
    F: FnMut(&Segment) -> Result<()> + Send + 'static,
{
    /// Loads a whisper.cpp model and prepares to transcribe.
    ///
    /// # This blocks, and on a cold machine it blocks for a long time
    ///
    /// Reading the model is seconds. What can be far longer is the first
    /// call on a GPU backend: the graphics driver compiles the compute
    /// shaders and caches them on disk, which has been measured at over a
    /// minute the first time and under two seconds every time after — the
    /// cache outlives the process. An application that calls this when a
    /// user presses a button will look hung; call it when there is
    /// somewhere to say so.
    ///
    /// Which backend runs is settled at build time by this crate's
    /// features, not here: `whisper` alone is CPU, `whisper-vulkan` uses
    /// the GPU through Vulkan. Whether the GPU was actually found is worth
    /// checking in the log, because whisper.cpp falls back to the CPU
    /// without raising anything.
    pub fn new(
        name: impl Into<String>,
        model_path: impl AsRef<std::path::Path>,
        policy: ChunkPolicy,
        on_segment: F,
    ) -> std::result::Result<Self, WhisperTranscriberError> {
        let path = model_path.as_ref().display().to_string();
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::WhisperTranscriber, &name, None);

        let context = WhisperContext::new_with_params(&path, WhisperContextParameters::default())
            .map_err(|source| WhisperTranscriberError::ModelLoad {
            path: path.clone(),
            source,
        })?;

        pp_info!(
            pp_log: &pp_log,
            "model loaded: path={path}, chunk={}ms, live_edge={}ms",
            policy.chunk_ms,
            policy.live_edge_ms
        );

        let state = context
            .create_state()
            .map_err(WhisperTranscriberError::State)?;

        Ok(Self {
            pp_log,
            name,
            state,
            chunker: Chunker::new(policy),
            on_segment,
            reported_zero_length: false,
        })
    }

    /// Reads one audio frame's samples, refusing anything not in the one
    /// shape the model's front end reads.
    fn samples_of(
        frame: &ffmpeg::frame::Audio,
    ) -> std::result::Result<&[f32], WhisperTranscriberError> {
        let rate = frame.rate();
        let channels = frame.channels();
        let format = frame.format();
        if rate != SAMPLE_RATE
            || channels != 1
            || format != ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed)
        {
            return Err(WhisperTranscriberError::UnsupportedAudio {
                rate,
                channels,
                format,
            });
        }
        Ok(frame.plane::<f32>(0))
    }

    /// Runs one inference and reports the segments inside its window.
    fn transcribe(&mut self, inference: Inference) -> Result<()> {
        if inference.audio.is_empty() {
            return Ok(());
        }
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_print_progress(false);
        params.set_print_special(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        self.state
            .full(params, &inference.audio)
            .map_err(WhisperTranscriberError::Inference)?;

        let count = self
            .state
            .full_n_segments()
            .map_err(WhisperTranscriberError::Inference)?;
        for index in 0..count {
            let Ok(text) = self.state.full_get_segment_text(index) else {
                continue;
            };
            let text = text.trim().to_owned();
            if text.is_empty() {
                continue;
            }
            // whisper.cpp reports in hundredths of a second, from the start
            // of the audio it was handed — which began at `offset_ms`.
            let start =
                self.state.full_get_segment_t0(index).unwrap_or(0) * 10 + inference.offset_ms;
            let end = self.state.full_get_segment_t1(index).unwrap_or(0) * 10 + inference.offset_ms;
            if start < inference.accept_from_ms || start >= inference.accept_until_ms {
                pp_trace!(
                    self,
                    "event=segment phase=skipped start={start}ms outcome=outside-window"
                );
                continue;
            }
            if end <= start && !self.reported_zero_length {
                self.reported_zero_length = true;
                pp_debug!(
                    self,
                    "a segment arrived with no duration; keeping it anyway"
                );
            }
            let segment = Segment {
                start_ms: start,
                end_ms: end,
                text,
            };
            (self.on_segment)(&segment)?;
        }
        Ok(())
    }
}

impl<F> Element for WhisperTranscriber<F>
where
    F: FnMut(&Segment) -> Result<()> + Send + 'static,
{
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::WhisperTranscriber
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl<F> Sink for WhisperTranscriber<F>
where
    F: FnMut(&Segment) -> Result<()> + Send + 'static,
{
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::frame(
            MediaKind::AudioFrame,
            MemoryDomain::System,
        ))
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Audio(frame) => {
                let samples = Self::samples_of(&frame)?;
                // One push can complete more than one chunk, when whoever
                // is upstream hands over a long frame.
                let mut inference = self.chunker.push(samples);
                while let Some(next) = inference {
                    self.transcribe(next)?;
                    inference = self.chunker.take_chunk();
                }
                Ok(())
            }
            MediaBuffer::Eos => {
                pp_trace!(self, "event=eos phase=received");
                if let Some(inference) = self.chunker.flush() {
                    self.transcribe(inference)?;
                }
                pp_trace!(self, "event=eos phase=completed outcome=ok");
                Ok(())
            }
            other => Err(WhisperTranscriberError::UnsupportedBuffer(other.kind()).into()),
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        pp_trace!(self, "event=control control={msg:?} phase=received");
        match msg {
            // All three abandon the audio being held: a flush because what
            // follows is from elsewhere on the timeline, a seek for the
            // same reason, and a stop because `Stop` means abandon rather
            // than finish — that is what separates it from `Eos`, which
            // transcribes the tail.
            ControlMsg::Flush | ControlMsg::Seek(_) | ControlMsg::Stop => self.chunker.reset(),
            // Nothing is scheduled here. Audio arrives or it does not, and
            // a gap in it is silence the model is welcome to hear.
            ControlMsg::Pause | ControlMsg::Resume => {}
            // Repositioning costs this nothing — the buffered audio is
            // dropped and the next chunk fills from wherever the source
            // lands — so there is no reason to refuse one.
            ControlMsg::CheckSeek(_) => {}
            // A decoder's business: preroll waits for terminals to report a
            // first sample of the new timeline, and the gate that does the
            // reporting lives upstream of every terminal.
            ControlMsg::Preroll(_) => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const POLICY: ChunkPolicy = ChunkPolicy {
        chunk_ms: 4_000,
        live_edge_ms: 1_000,
    };

    fn samples(ms: usize) -> Vec<f32> {
        vec![0.0; ms * SAMPLE_RATE as usize / 1000]
    }

    /// Nothing is transcribed until a whole chunk has arrived, because a
    /// shorter one is padded to thirty seconds and the model was not
    /// trained to hear the padding as silence at the end of a sentence.
    #[test]
    fn a_partial_chunk_is_not_transcribed() {
        let mut chunker = Chunker::new(POLICY);
        assert_eq!(chunker.push(&samples(3_999)), None);
        assert!(chunker.push(&samples(1)).is_some(), "and now it is full");
    }

    /// The window that is believed starts where the last one stopped and
    /// ends short of the edge, so every millisecond is reported once.
    #[test]
    fn the_believed_window_stops_short_of_the_edge_and_resumes_there() {
        let mut chunker = Chunker::new(POLICY);

        let first = chunker.push(&samples(4_000)).expect("a full chunk");
        assert_eq!(first.offset_ms, 0);
        assert_eq!(first.accept_from_ms, 0);
        assert_eq!(
            first.accept_until_ms, 3_000,
            "the last second is not trusted"
        );

        let second = chunker.push(&samples(4_000)).expect("another");
        assert_eq!(
            second.accept_from_ms, 3_000,
            "the second held back before is picked up here"
        );
        assert_eq!(second.accept_until_ms, 7_000);
    }

    /// A word across the seam is heard whole because the round that reports
    /// it was given the audio before it too.
    #[test]
    fn each_inference_carries_the_previous_chunk_as_context() {
        let mut chunker = Chunker::new(POLICY);

        let first = chunker.push(&samples(4_000)).expect("a full chunk");
        assert_eq!(
            first.audio.len(),
            samples(4_000).len(),
            "nothing precedes the first"
        );

        let second = chunker.push(&samples(4_000)).expect("another");
        assert_eq!(
            second.audio.len(),
            samples(8_000).len(),
            "four seconds of context and four of new audio"
        );
        assert_eq!(second.offset_ms, 0, "and it starts where the context does");
    }

    /// A caller handing over more than a chunk at once must not have the
    /// surplus swallowed.
    #[test]
    fn a_long_push_yields_one_inference_per_chunk() {
        let mut chunker = Chunker::new(POLICY);
        assert!(chunker.push(&samples(12_000)).is_some());
        assert!(chunker.take_chunk().is_some());
        assert!(chunker.take_chunk().is_some());
        assert_eq!(chunker.take_chunk(), None, "three chunks and no more");
    }

    /// At the end there is no later round, so the tail is transcribed
    /// however short it is and the live edge is not held back.
    #[test]
    fn the_tail_is_transcribed_at_eos_without_holding_the_edge_back() {
        let mut chunker = Chunker::new(POLICY);
        chunker.push(&samples(4_000)).expect("a full chunk");
        chunker.push(&samples(500));

        let last = chunker.flush().expect("the tail is worth hearing");
        assert_eq!(
            last.accept_from_ms, 3_000,
            "starting where the chunk stopped"
        );
        assert_eq!(last.accept_until_ms, 4_500, "and running to the very end");
        assert_eq!(
            last.audio.len(),
            samples(4_500).len(),
            "the whole chunk before it, as context, then the tail"
        );
    }

    /// Flushing twice must not transcribe the tail again and report every
    /// line of it a second time.
    #[test]
    fn a_second_flush_has_nothing_left_to_say() {
        let mut chunker = Chunker::new(POLICY);
        chunker.push(&samples(4_000));
        chunker.push(&samples(500));
        assert!(chunker.flush().is_some());
        assert_eq!(chunker.flush(), None);
    }

    /// A stream that ends on an exact chunk boundary has already had every
    /// millisecond transcribed except the second held at the edge.
    #[test]
    fn a_stream_ending_on_a_boundary_still_reports_its_held_back_edge() {
        let mut chunker = Chunker::new(POLICY);
        chunker.push(&samples(4_000)).expect("a full chunk");

        let last = chunker.flush().expect("the held-back second is still owed");
        assert_eq!(last.accept_from_ms, 3_000);
        assert_eq!(last.accept_until_ms, 4_000);
    }
}
