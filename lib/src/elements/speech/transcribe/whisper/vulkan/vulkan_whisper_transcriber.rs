//! [`VulkanWhisperTranscriber`]: speech to timed text on the GPU through Vulkan.

use crate::{
    element::ElementType,
    transform::{FilterStage, filter_stage},
};

use super::super::{Compute, Transcribing, WhisperOptions, WhisperTranscriberError};

/// Turns speech into timed text with a whisper.cpp model, on any GPU through Vulkan.
///
/// It does what [`SwWhisperTranscriber`](crate::elements::SwWhisperTranscriber)
/// does — takes decoded sound at
/// [`WHISPER_SAMPLE_RATE`](crate::elements::WHISPER_SAMPLE_RATE), mono, 32-bit float, behind a
/// [`Queue`](crate::queue::Queue), and hands the same sound on carrying a
/// [`Transcript`](crate::elements::Transcript) where each line begins — with
/// the model on the GPU through Vulkan. The sound stays in system memory: only the
/// model's work runs on the GPU, so nothing else in the pipeline need be on
/// the same device. Choose it for a GPU whatever its maker — NVIDIA, AMD or Intel — with nothing to install beyond the driver; on an RTX 3050 the large turbo model ran six times faster than speech, chunk by chunk. Needs the `whisper-vulkan` feature, and to build, the Vulkan SDK.
///
/// # Example
///
/// The large model, fast enough on the GPU through Vulkan to keep ahead of speech:
///
/// ```no_run
/// # fn main() -> media_pp::Result<()> {
/// use media_pp::elements::{
///     AudioFormat, AudioResampler, FileDemuxer, FileMuxer, Mp4Layout, SubtitleEncoder,
///     SwDecoder, VulkanWhisperTranscriber, WHISPER_SAMPLE_RATE, WhisperOptions,
/// };
/// use media_pp::{bus::BusEvent, ffmpeg, pipeline::Pipeline, subtitle};
///
/// let (source, _) = FileDemuxer::open("demux", "talk.mp4")?;
/// let audio = source.best(ffmpeg::media::Type::Audio)?;
/// let decoder = SwDecoder::new("decoder", audio.parameters().clone())?;
/// let mono_f32 = ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed);
/// let resampler = AudioResampler::new("resampler", AudioFormat::new(mono_f32, WHISPER_SAMPLE_RATE, 1));
/// // The first GPU Vulkan offers.
/// let transcriber =
///     VulkanWhisperTranscriber::new("whisper", "ggml-large-v3-turbo.bin", WhisperOptions::default(), 0)?;
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
/// The loop is [`SwWhisperTranscriber`](crate::elements::SwWhisperTranscriber)'s
/// — the same rounds, the same agreement between them, the same holding of
/// the sound until its words are settled; see there. What differs is where whisper.cpp runs
/// the model, which is settled as it is loaded. Vulkan needs nothing at run time beyond the graphics driver.
///
/// whisper.cpp numbers its GPUs across every backend it was built with, so
/// a build with both `whisper-cuda` and `whisper-vulkan` sees one NVIDIA card
/// twice. The number given here counts Vulkan's own GPUs alone, and a
/// GPU that is not there is an error rather than whisper.cpp's quiet fall
/// back to the CPU.
pub struct VulkanWhisperTranscriber(FilterStage<Transcribing>);

filter_stage!(VulkanWhisperTranscriber);

impl VulkanWhisperTranscriber {
    /// Loads a whisper.cpp model to run on the `gpu`-th GPU Vulkan
    /// offers, counted from zero.
    ///
    /// # This blocks, and on a cold machine it blocks for a long time
    ///
    /// Reading the model takes seconds. What can be far longer is the first run on a machine: the driver compiles whisper.cpp's shaders and caches them on disk, which took over a minute the first time and under two seconds every time after — the cache outlives the process. An application that calls this when a user presses a button will look hung; call it where there is somewhere to say so.
    ///
    /// # Errors
    ///
    /// A model that cannot be loaded, a language or prompt in `options`
    /// whisper.cpp cannot take, or no such Vulkan GPU.
    pub fn new(
        name: impl Into<String>,
        model_path: impl AsRef<std::path::Path>,
        options: WhisperOptions,
        gpu: usize,
    ) -> std::result::Result<Self, WhisperTranscriberError> {
        let (identity, transcribing) = Transcribing::load(
            ElementType::VulkanWhisperTranscriber,
            name,
            model_path.as_ref(),
            options,
            Compute::Gpu {
                backend: "Vulkan",
                index: gpu,
            },
        )?;
        Ok(Self(FilterStage::new(identity, transcribing)))
    }
}
