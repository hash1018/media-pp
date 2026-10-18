//! whisper.cpp's transcribers, one per backend, and the options and loop
//! they share.
//!
//! Every backend runs the same model through the same loop — [`engine`]'s
//! chunking, and [`hold`]'s holding of the audio until its words are
//! settled. What differs is only where whisper.cpp runs the model, settled
//! when the model is loaded: on the CPU, or on one GPU of one backend.

use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

use crate::pp_log::{PpLog, pp_debug, pp_error, pp_info, pp_trace, pp_warn};
use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    element::{Attached, ElementType, Identity},
    error::Result,
    transform::{Filter, Output},
};

#[cfg(feature = "whisper-cuda")]
mod cuda;
mod engine;
#[cfg(any(
    feature = "whisper-cuda",
    feature = "whisper-vulkan",
    all(target_os = "macos", feature = "whisper-metal")
))]
mod gpu;
mod hold;
#[cfg(all(target_os = "macos", feature = "whisper-metal"))]
mod metal;
mod sw_whisper_transcriber;
#[cfg(feature = "whisper-vulkan")]
mod vulkan;

#[cfg(feature = "whisper-cuda")]
pub use cuda::CudaWhisperTranscriber;
#[cfg(all(target_os = "macos", feature = "whisper-metal"))]
pub use metal::MetalWhisperTranscriber;
pub use sw_whisper_transcriber::SwWhisperTranscriber;
#[cfg(feature = "whisper-vulkan")]
pub use vulkan::VulkanWhisperTranscriber;

use engine::{Chunker, Heard, Inference};
use hold::Held;

use crate::elements::speech::meta::Utterance;

/// The rate Whisper's own front end works at. Nothing else is accepted, and
/// nothing here resamples: put an
/// [`AudioResampler`](crate::elements::AudioResampler) in front instead,
/// the way a detector has a scaler in front to reach the size its own model
/// wants.
pub const SAMPLE_RATE: u32 = 16_000;

/// Why a whisper transcriber could not load its model or transcribe.
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
        "a whisper transcriber needs {SAMPLE_RATE}Hz mono f32 audio, got {rate}Hz \
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
    /// A buffer that is not audio reached the transcriber.
    #[error("a whisper transcriber only accepts audio frames, got {0}")]
    UnsupportedBuffer(&'static str),
    /// A language whisper.cpp does not know.
    #[error("whisper does not know the language {0:?}; use a code such as \"en\" or \"ko\"")]
    UnknownLanguage(String),
    /// A prompt whisper.cpp cannot be handed: one with a NUL in it.
    #[error("a whisper prompt cannot contain a NUL character")]
    InvalidPrompt,
    /// The backend offers fewer GPUs than the one asked for.
    #[error("no {backend} GPU number {index}: whisper.cpp found {found} {backend} GPU(s)")]
    NoGpu {
        /// The backend asked for — `CUDA`, `Vulkan`, `Metal`.
        backend: &'static str,
        /// Which of its GPUs, from zero.
        index: usize,
        /// How many of its GPUs whisper.cpp found.
        found: usize,
    },
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
    /// How each word's time is worked out — which is what decides where a
    /// line is cut between one round and the next.
    pub token_timing: TokenTiming,
}

impl Default for ChunkPolicy {
    /// Four seconds of audio, holding back the last one, with estimated
    /// token times.
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
            token_timing: TokenTiming::Estimated,
        }
    }
}

/// How the time of each token is worked out.
///
/// Every round hears the round before it again as context, and what it
/// reports is cut from what it heard by those times: the words inside its
/// window are its own, the ones before were said already. So the times
/// decide whether a word at the seam is reported once, twice or not at all.
/// Measured on a minute of dense Korean speech with `large-v3-turbo`:
///
/// | | real time | at the seams |
/// |---|---|---|
/// | [`Estimated`](Self::Estimated) | 6.4x | pieces said twice — "13시간 / 시간 정도" |
/// | [`Aligned`](Self::Aligned) | 6.1x | one piece said twice in the minute |
///
/// The first aligned run on a GPU also compiles the alignment's own
/// shaders, which took as long again as the inference here — a cost the
/// driver's cache pays once per machine, not once per run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenTiming {
    /// whisper.cpp's own estimate, from the timestamps it predicts beside
    /// the text. Costs nothing, and is rough enough that the same word
    /// heard in two rounds gets two different times.
    Estimated,
    /// Aligned against the audio by dynamic time warping over the model's
    /// alignment heads. A few percent more inference time, and a line cut
    /// where the speech actually is.
    ///
    /// Which heads to read depends on the model, and is worked out from the
    /// model file's own dimensions. For a model this does not recognise it
    /// falls back to [`Estimated`](Self::Estimated) and says so in the log,
    /// rather than aligning against the wrong heads.
    Aligned,
}

impl ChunkPolicy {
    fn chunk_samples(&self) -> usize {
        (self.chunk_ms as usize * SAMPLE_RATE as usize) / 1000
    }

    fn live_edge_samples(&self) -> usize {
        (self.live_edge_ms as usize * SAMPLE_RATE as usize) / 1000
    }
}

/// What a whisper transcriber is to hear, and how — the same for every
/// backend.
///
/// [`Default`] detects the language, transcribes rather than translates,
/// gives the model no prompt, and gathers audio as [`ChunkPolicy::default`]
/// says.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WhisperOptions {
    /// How much audio to gather for each inference, and how much of the
    /// answer to trust.
    pub chunk: ChunkPolicy,
    /// The language spoken, as whisper.cpp names it — a code such as `"en"`
    /// or `"ko"`, or the English name, `"korean"` — or `None` to detect it.
    ///
    /// Detection is the default and the reason there is one: whisper.cpp's
    /// own is English, and handed Korean speech under that it does not
    /// fail — it writes an English paraphrase of it, confidently and at full
    /// speed. Saying it is still better where it is known: detecting takes
    /// the few seconds of one chunk to go on, costs about 60 % more time,
    /// and a chunk of music, laughter or two words can be heard as something
    /// else, which puts a line in the wrong language in the middle of the
    /// right ones.
    pub language: Option<String>,
    /// Writes the lines in English, whatever is spoken — whisper's own
    /// translation, not a second model.
    pub translate: bool,
    /// Text the model reads as what came before the audio: the names and
    /// terms it is to spell as given. A word the model has never seen —
    /// a company's name, a place — is otherwise heard as the nearest one it
    /// knows. Keep it to a sentence or two of the vocabulary; it is given
    /// before every chunk.
    pub prompt: Option<String>,
    /// How many CPU threads whisper.cpp runs on, or `None` for its own
    /// choice. On a GPU backend the CPU only feeds the GPU, and this matters
    /// little.
    pub threads: Option<u16>,
}

/// Where a transcriber's model runs, settled as it is loaded.
pub(super) enum Compute {
    /// On the CPU.
    Cpu,
    /// On the `index`-th GPU of `backend`, as its registry names it.
    #[cfg(any(
        feature = "whisper-cuda",
        feature = "whisper-vulkan",
        all(target_os = "macos", feature = "whisper-metal")
    ))]
    Gpu {
        /// `CUDA`, `Vulkan`, `Metal`.
        backend: &'static str,
        /// Which of its GPUs, from zero.
        index: usize,
    },
}

/// What every whisper transcriber does with the audio it is handed:
/// transcribes it a chunk at a time, and hands each buffer on once the
/// speech in it is settled, carrying the lines that begin in it.
pub(super) struct Transcribing {
    pp_log: PpLog,
    /// Made once and reused for every chunk.
    ///
    /// Creating one allocates the model's working buffers and, on a GPU
    /// backend, initialises the backend itself — which is not a per-chunk
    /// cost anybody wants to pay. It holds its own reference to the model,
    /// so the context it came from does not have to be kept beside it.
    state: WhisperState,
    chunker: Chunker,
    held: Held,
    /// Where the next sample is expected on the audio's timeline, so a
    /// frame stamping nothing is placed where the last one ended.
    next_sample: Option<i64>,
    /// whisper.cpp's own code for the language, or `None` to detect it.
    /// Its own `&'static str` rather than the caller's string, which is
    /// what lets each chunk's parameters borrow it without anything to keep
    /// alive.
    language: Option<&'static str>,
    translate: bool,
    prompt: Option<String>,
    threads: Option<u16>,
    /// The model's end-of-text token. Every token numbered below it is
    /// text; everything from it up is a timestamp or a control token.
    end_of_text: whisper_rs::WhisperTokenId,
    /// Whether the context was loaded with alignment heads, so each token
    /// carries an aligned time — see [`TokenTiming::Aligned`].
    aligned: bool,
    /// Reported once, because it is the same answer every time and a line
    /// per chunk would bury the log.
    reported_zero_length: bool,
}

impl Transcribing {
    /// Loads a whisper.cpp model to run where `compute` says, and the state
    /// every chunk runs in.
    ///
    /// Blocks while the model is read and, on a GPU backend, while the
    /// backend starts — over a minute the first time a driver compiles
    /// whisper.cpp's shaders, a second or two every time after.
    pub(super) fn load(
        element_type: ElementType,
        name: impl Into<String>,
        model_path: &std::path::Path,
        options: WhisperOptions,
        compute: Compute,
    ) -> std::result::Result<(Identity, Self), WhisperTranscriberError> {
        let path = model_path.display().to_string();
        let identity = Identity::new(element_type, name);
        let pp_log = identity.pp_log().clone();

        // Checked before anything is loaded: a wrong language or prompt is
        // said before the seconds a model takes to read.
        let language = match &options.language {
            Some(language) => Some(engine::language_code(language)?),
            None => None,
        };
        if options
            .prompt
            .as_deref()
            .is_some_and(|prompt| prompt.contains('\0'))
        {
            return Err(WhisperTranscriberError::InvalidPrompt);
        }

        // Settled before loading, because whisper.cpp builds the alignment
        // into the context it loads rather than into each run.
        let heads = match options.chunk.token_timing {
            TokenTiming::Estimated => None,
            TokenTiming::Aligned => {
                let heads = engine::model_shape(model_path).and_then(engine::alignment_heads);
                if heads.is_none() {
                    pp_warn!(
                        pp_log: &pp_log,
                        "no alignment heads are known for {path}; token times will be estimated"
                    );
                }
                heads
            }
        };
        let mut params = WhisperContextParameters::default();
        let on = match compute {
            Compute::Cpu => {
                params.use_gpu(false);
                "CPU".to_owned()
            }
            #[cfg(any(
                feature = "whisper-cuda",
                feature = "whisper-vulkan",
                all(target_os = "macos", feature = "whisper-metal")
            ))]
            Compute::Gpu { backend, index } => {
                let device = gpu::device(backend, index)?;
                params.use_gpu(true);
                params.gpu_device(device);
                format!("{backend} GPU {index}")
            }
        };
        if let Some(preset) = heads.clone() {
            params.dtw_parameters(whisper_rs::DtwParameters {
                mode: whisper_rs::DtwMode::ModelPreset {
                    model_preset: preset,
                },
                ..Default::default()
            });
        }
        let context = WhisperContext::new_with_params(&path, params).map_err(|source| {
            WhisperTranscriberError::ModelLoad {
                path: path.clone(),
                source,
            }
        })?;

        pp_info!(
            pp_log: &pp_log,
            "model loaded: path={path}, on {on}, chunk={}ms, live_edge={}ms, token_timing={}, \
             language={}, translate={}, prompt={}",
            options.chunk.chunk_ms,
            options.chunk.live_edge_ms,
            match &heads {
                Some(preset) => format!("aligned({preset:?})"),
                None => "estimated".to_owned(),
            },
            language.unwrap_or("detected"),
            options.translate,
            if options.prompt.is_some() { "given" } else { "none" },
        );

        let state = context
            .create_state()
            .map_err(WhisperTranscriberError::State)?;
        let end_of_text = context.token_eot();

        Ok((
            identity,
            Self {
                pp_log,
                state,
                chunker: Chunker::new(options.chunk),
                held: Held::default(),
                next_sample: None,
                language,
                translate: options.translate,
                prompt: options.prompt,
                threads: options.threads,
                end_of_text,
                aligned: heads.is_some(),
                reported_zero_length: false,
            },
        ))
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

    /// Runs one inference and answers the lines inside its window.
    fn transcribe(&mut self, inference: Inference) -> Result<Vec<Utterance>> {
        if inference.audio.is_empty() {
            return Ok(Vec::new());
        }
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_print_progress(false);
        params.set_print_special(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        // `None` is detection, which has to be asked for: left unset,
        // whisper.cpp assumes English — see `WhisperOptions::language`.
        params.set_language(self.language);
        params.set_translate(self.translate);
        if let Some(prompt) = &self.prompt {
            params.set_initial_prompt(prompt);
        }
        if let Some(threads) = self.threads {
            params.set_n_threads(i32::from(threads));
        }
        // What lets a segment be cut where this round's window is rather
        // than kept or dropped whole — see `Chunker::place`.
        params.set_token_timestamps(true);

        self.state
            .full(params, &inference.audio)
            .map_err(WhisperTranscriberError::Inference)?;

        let mut lines = Vec::new();
        for index in 0..self.state.full_n_segments() {
            let Some(segment) = self.state.get_segment(index) else {
                continue;
            };
            let tokens = self.heard(&segment, inference.offset_ms);
            let Some(line) = self.chunker.place(&inference, &tokens) else {
                pp_trace!(
                    self,
                    "event=segment phase=skipped tokens={} \
                     outcome=outside-window-or-already-reported",
                    tokens.len()
                );
                continue;
            };
            if line.end_ms <= line.start_ms && !self.reported_zero_length {
                self.reported_zero_length = true;
                pp_debug!(
                    self,
                    "a segment arrived with no duration; keeping it anyway"
                );
            }
            lines.push(line);
        }
        Ok(lines)
    }

    /// The text tokens of one segment, placed on the stream's timeline.
    ///
    /// whisper.cpp reports in hundredths of a second from the start of the
    /// audio it was handed — which began at `offset_ms`.
    fn heard(&self, segment: &whisper_rs::WhisperSegment<'_>, offset_ms: i64) -> Vec<Heard> {
        (0..segment.n_tokens())
            .filter_map(|index| segment.get_token(index))
            .filter(|token| token.token_id() < self.end_of_text)
            .filter_map(|token| {
                let data = token.token_data();
                // An aligned time is one moment per token rather than a span,
                // and is the one worth cutting by. whisper.cpp leaves it at
                // -1 where it computed none.
                let start = if self.aligned && data.t_dtw >= 0 {
                    data.t_dtw
                } else {
                    data.t0
                };
                Some(Heard {
                    bytes: token.to_bytes().ok()?.to_vec(),
                    start_ms: start * 10 + offset_ms,
                    end_ms: data.t1.max(start) * 10 + offset_ms,
                })
            })
            .collect()
    }

    fn samples_to_ms(samples: i64) -> i64 {
        (samples * 1000) / i64::from(SAMPLE_RATE)
    }
}

impl Filter for Transcribing {
    /// Decoded sound in system memory, at [`SAMPLE_RATE`] mono `f32`.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::frame(
            MediaKind::AudioFrame,
            MemoryDomain::System,
        ))
    }

    /// The sound it was handed, later.
    fn output_contract(&self) -> OutputContract {
        OutputContract::Passthrough
    }

    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        let MediaBuffer::Audio(frame) = &buf else {
            let kind = buf.kind();
            pp_error!(self, "unsupported buffer: {kind}");
            return Err(WhisperTranscriberError::UnsupportedBuffer(kind).into());
        };
        let samples = Self::samples_of(frame).inspect_err(|error| pp_error!(self, "{error}"))?;
        // The frame's own timestamps place it, in a sample index — the time
        // base the resampler in front stamps at this rate.
        let start = frame.pts().or(self.next_sample).unwrap_or(0);
        let end = start + samples.len() as i64;
        self.next_sample = Some(end);

        // One push can complete more than one chunk, when whoever is
        // upstream hands over a long frame.
        let mut inference = self.chunker.push(frame.pts(), samples);
        self.held
            .hold(buf, Self::samples_to_ms(start), Self::samples_to_ms(end));
        while let Some(next) = inference {
            let lines = self.transcribe(next)?;
            self.held.heard(lines);
            inference = self.chunker.take_chunk();
        }
        for settled in self.held.release(self.chunker.settled_until_ms()) {
            out.push(settled);
        }
        Ok(())
    }

    /// Transcribes the tail held back at the live edge — nothing more is
    /// coming, so nothing can change what the model hears — and hands on
    /// everything held.
    fn drain(&mut self, out: &mut Output) -> Result<()> {
        pp_trace!(self, "event=eos phase=received");
        if let Some(inference) = self.chunker.flush() {
            let lines = self.transcribe(inference)?;
            self.held.heard(lines);
        }
        let (going, stray) = self.held.release_all();
        if stray > 0 {
            pp_warn!(
                self,
                "{stray} line(s) heard with no audio left to carry them"
            );
        }
        for buf in going {
            out.push(buf);
        }
        pp_trace!(self, "event=eos phase=completed outcome=ok");
        Ok(())
    }

    /// Lets go of the audio being held and what was heard in it: for a seek
    /// because what follows is from elsewhere on the timeline, and for a
    /// stop because `Stop` means abandon rather than finish — which is what
    /// separates it from the end of the stream, whose tail `drain`
    /// transcribes. A pause needs nothing: audio arrives or it does not.
    fn reset(&mut self) {
        self.chunker.reset();
        self.held.clear();
    }

    fn attached(&mut self, cx: &Attached<'_>) {
        self.pp_log = cx.pp_log().clone();
    }
}
