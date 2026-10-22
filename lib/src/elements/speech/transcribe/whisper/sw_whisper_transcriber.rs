//! [`SwWhisperTranscriber`]: speech to timed text on the CPU.

use crate::{
    element::ElementType,
    transform::{FilterStage, filter_stage},
};

use super::{Compute, Transcribing, WhisperOptions, WhisperTranscriberError};

/// Turns speech into timed text with a whisper.cpp model, on the CPU.
///
/// It takes decoded sound at [`WHISPER_SAMPLE_RATE`](super::SAMPLE_RATE),
/// mono, 32-bit float — an [`AudioResampler`](crate::elements::AudioResampler)
/// in front gets it there — behind a [`Queue`](crate::queue::Queue), since
/// inference is slow. It hands the same sound on, each buffer once the speech
/// in it is settled, the one each line begins in carrying a
/// [`Transcript`](crate::elements::Transcript). A
/// [`SubtitleEncoder`](crate::elements::SubtitleEncoder) after it makes the
/// lines a subtitle track. On the CPU only the small models keep up with
/// speech as it is said; `CudaWhisperTranscriber`, `VulkanWhisperTranscriber`
/// and `MetalWhisperTranscriber` run the same model on a GPU. Needs the
/// `whisper` feature.
///
/// # Example
///
/// Print what is said in a file:
///
/// ```no_run
/// # fn main() -> media_pp::Result<()> {
/// use media_pp::bus::BusEvent;
/// use media_pp::elements::{
///     AppSink, AudioFormat, AudioResampler, FileDemuxer, SwDecoder, SwWhisperTranscriber,
///     Transcript, WHISPER_SAMPLE_RATE, WhisperOptions,
/// };
/// use media_pp::{ffmpeg, pipeline::Pipeline};
///
/// let (source, _) = FileDemuxer::open("demux", "talk.mp4")?;
/// let audio = source.best(ffmpeg::media::Type::Audio)?;
/// let decoder = SwDecoder::new("decoder", audio.parameters().clone())?;
/// let mono_f32 = ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed);
/// let resampler = AudioResampler::new("resampler", AudioFormat::new(mono_f32, WHISPER_SAMPLE_RATE, 1));
/// let options = WhisperOptions {
///     language: Some("ko".into()),
///     ..WhisperOptions::default()
/// };
/// let transcriber = SwWhisperTranscriber::new("whisper", "ggml-base.bin", options)?;
/// let lines = AppSink::new("lines", |buf| {
///     if let Some(transcript) = buf.metadata().and_then(|m| m.get::<Transcript>()) {
///         for line in &transcript.utterances {
///             println!("{:>8} ms  {}", line.start_ms, line.text);
///         }
///     }
///     Ok(())
/// });
///
/// let (pipeline, ()) = Pipeline::new("transcribe", source, |source, ctx| {
///     let branch = ctx
///         .branch()
///         .pipe(decoder)
///         .pipe(resampler)
///         .queue("speech", 64)
///         .pipe(transcriber)
///         .to(lines)?;
///     ctx.attach(source, audio.index(), branch)?;
///     Ok(())
/// })?;
/// pipeline.run()?;
/// for event in pipeline.bus().iter() {
///     if matches!(event, BusEvent::Finished) {
///         pipeline.stop();
///     }
/// }
/// # Ok(())
/// # }
/// ```
///
/// # How it works
///
/// Whisper's encoder hears thirty seconds at a time, so a stream is
/// transcribed in rounds — every [`ChunkPolicy::chunk_ms`](super::ChunkPolicy::chunk_ms),
/// four seconds by default, the audio not yet settled is heard again with
/// what has arrived since. A word is reported once two rounds in a row
/// agree on the words up to it, by their text rather than by the times the
/// model gives them, which move between rounds: so a word is said once,
/// and a line, once reported, never changes. What is reported stays in the
/// audio heard, as context, until the audio is long, and is then cut away
/// and given to the model as its prompt instead.
///
/// The sound is held until the speech in it is settled — one to two rounds
/// behind it — so that the line can ride on the sound it was said in. For a file that costs nothing: the
/// file is written by timestamp, not by when a buffer arrives. For sound
/// that is also being played, split it with a [`Tee`](crate::elements::Tee)
/// in front of this and play the other branch.
///
/// # What it takes
///
/// [`WHISPER_SAMPLE_RATE`](super::SAMPLE_RATE) mono `f32`, and nothing else;
/// it refuses anything else rather than resampling badly on its own. The
/// frames' timestamps are read, as a sample index — the time base that
/// resampler stamps at this rate — and they are what places each line, so
/// audio that arrives with a gap in it leaves a gap rather than pulling
/// everything after it earlier.
///
/// # Threading and pace
///
/// Each push returns when inference does. Put a [`Queue`](crate::queue::Queue)
/// in front, this crate's thread boundary, so the branch producing the audio
/// is not held up by the model. A model slower than real time cannot be
/// rescued by any of this: the queue fills and drops audio, and the text has
/// gaps — the honest outcome, and better than an unbounded backlog.
///
/// # The end, a seek and a stop
///
/// The end of the stream hears what is left once more and reports it
/// without a round to agree with, and hands on everything held. A seek lets go
/// of the held audio, which is from before it; a stop abandons it, as a stop
/// does everywhere in this crate.
pub struct SwWhisperTranscriber(FilterStage<Transcribing>);

filter_stage!(SwWhisperTranscriber);

impl SwWhisperTranscriber {
    /// Loads a whisper.cpp model to run on the CPU.
    ///
    /// # This blocks
    ///
    /// Reading the model takes seconds.
    ///
    /// # Errors
    ///
    /// A model that cannot be loaded, or a language or prompt in `options`
    /// whisper.cpp cannot take.
    pub fn new(
        name: impl Into<String>,
        model_path: impl AsRef<std::path::Path>,
        options: WhisperOptions,
    ) -> std::result::Result<Self, WhisperTranscriberError> {
        let (identity, transcribing) = Transcribing::load(
            ElementType::SwWhisperTranscriber,
            name,
            model_path.as_ref(),
            options,
            Compute::Cpu,
        )?;
        Ok(Self(FilterStage::new(identity, transcribing)))
    }
}
