//! [`CudaWhisperTranscriber`]: speech to timed text on an NVIDIA GPU.

use crate::{
    element::ElementType,
    transform::{FilterStage, filter_stage},
};

use super::super::{Compute, Transcribing, WhisperOptions, WhisperTranscriberError};

/// Turns speech into timed text with a whisper.cpp model, on an NVIDIA GPU with CUDA.
///
/// It does what [`SwWhisperTranscriber`](crate::elements::SwWhisperTranscriber)
/// does — takes decoded sound at
/// [`WHISPER_SAMPLE_RATE`](crate::elements::WHISPER_SAMPLE_RATE), mono, 32-bit float, behind a
/// [`Queue`](crate::queue::Queue), and hands the same sound on carrying a
/// [`Transcript`](crate::elements::Transcript) where each line begins — with
/// the model on an NVIDIA GPU. The sound stays in system memory: only the
/// model's work runs on the GPU, so nothing else in the pipeline need be on
/// the same device. Choose it on a machine with an NVIDIA card where the CUDA libraries are already there — beside `CudaDecoder` or the CUDA detectors; `VulkanWhisperTranscriber` runs on the same card, and on any other, with nothing to install. Needs the `whisper-cuda` feature, and to build, the CUDA toolkit.
///
/// # Example
///
/// The large model, fast enough on an NVIDIA GPU to keep ahead of speech:
///
/// ```no_run
/// # fn main() -> media_pp::Result<()> {
/// use media_pp::elements::{
///     AudioFormat, AudioResampler, FileDemuxer, FileMuxer, Mp4Layout, SubtitleEncoder,
///     SwDecoder, CudaWhisperTranscriber, WHISPER_SAMPLE_RATE, WhisperOptions,
/// };
/// use media_pp::{bus::BusEvent, ffmpeg, pipeline::Pipeline, subtitle};
///
/// let (source, _) = FileDemuxer::open("demux", "talk.mp4")?;
/// let audio = source.best(ffmpeg::media::Type::Audio)?;
/// let decoder = SwDecoder::new("decoder", audio.parameters.clone())?;
/// let mono_f32 = ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed);
/// let resampler = AudioResampler::new("resampler", AudioFormat::new(mono_f32, WHISPER_SAMPLE_RATE, 1));
/// // The first GPU CUDA offers.
/// let transcriber =
///     CudaWhisperTranscriber::new("whisper", "ggml-large-v3-turbo.bin", WhisperOptions::default(), 0)?;
///
/// // The lines as an .srt file of their own.
/// let encoder = SubtitleEncoder::new("srt", subtitle::Codec::SubRip);
/// let mut muxer = FileMuxer::create("talk.srt", Mp4Layout::Standard)?;
/// let track = muxer.add_stream("text", &encoder)?;
/// let sink = muxer.open()?.take(track)?;
///
/// let (pipeline, ()) = Pipeline::new("subtitles", source, |source, ctx| {
///     let branch = ctx
///         .branch()
///         .pipe(decoder)
///         .pipe(resampler)
///         .queue("speech", 64)
///         .pipe(transcriber)
///         .pipe(encoder)
///         .to(sink)?;
///     ctx.attach(source, audio.index, branch)?;
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
/// The loop is [`SwWhisperTranscriber`](crate::elements::SwWhisperTranscriber)'s
/// — the same rounds, the same agreement between them, the same holding of
/// the sound until its words are settled; see there. What differs is where whisper.cpp runs
/// the model, which is settled as it is loaded. Built with CUDA, whisper.cpp links CUDA's runtime and cuBLAS: a program built with it does not start without them, as it does not without FFmpeg.
///
/// whisper.cpp numbers its GPUs across every backend it was built with, so
/// a build with both `whisper-cuda` and `whisper-vulkan` sees one NVIDIA card
/// twice. The number given here counts CUDA's own GPUs alone, and a
/// GPU that is not there is an error rather than whisper.cpp's quiet fall
/// back to the CPU.
pub struct CudaWhisperTranscriber(FilterStage<Transcribing>);

filter_stage!(CudaWhisperTranscriber);

impl CudaWhisperTranscriber {
    /// Loads a whisper.cpp model to run on the `gpu`-th GPU CUDA
    /// offers, counted from zero.
    ///
    /// # This blocks, and on a cold machine it blocks for a long time
    ///
    /// Reading the model takes seconds. The first run also starts CUDA on the card, which takes a second or two.
    ///
    /// # Errors
    ///
    /// A model that cannot be loaded, a language or prompt in `options`
    /// whisper.cpp cannot take, or no such CUDA GPU.
    pub fn new(
        name: impl Into<String>,
        model_path: impl AsRef<std::path::Path>,
        options: WhisperOptions,
        gpu: usize,
    ) -> std::result::Result<Self, WhisperTranscriberError> {
        let (identity, transcribing) = Transcribing::load(
            ElementType::CudaWhisperTranscriber,
            name,
            model_path.as_ref(),
            options,
            Compute::Gpu {
                backend: "CUDA",
                index: gpu,
            },
        )?;
        Ok(Self(FilterStage::new(identity, transcribing)))
    }
}
