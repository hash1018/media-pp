//! Transcribes a file's speech and writes a copy of it carrying the result
//! as a subtitle track — video, audio and text in one `.mp4`.
//!
//! ```text
//!                    ┌─ video packets ───────────────────────────────────┐
//!                    │                                                   │
//! FileDemuxer ───────┤              ┌─ audio packets ────────────────────┤─ FileMuxer
//!                    │              │                                    │
//!                    └─ audio ─ Tee ┤                                    │
//!                                   └─ SwDecoder ─ AudioResampler ─ Queue│
//!                                        ─ XxxWhisperTranscriber ─ Tee ──┤
//!                                            ├─ SubtitleEncoder (mov_text)┘
//!                                            ├─ SubtitleEncoder (--sidecar) ─ FileMuxer
//!                                            └─ AppSink (prints each line)
//! ```
//!
//! The picture and the sound are copied through as packets — nothing is
//! decoded for them and nothing re-encoded. Only the audio is decoded, and
//! only to be resampled into the one shape Whisper reads: 16 kHz mono f32.
//!
//! The transcriber hands that sound on, each buffer carrying the lines that
//! begin in it as a `Transcript`; a `SubtitleEncoder` makes those lines
//! subtitle packets, and an `AppSink` prints them.
//!
//! # Why the text arrives late, and why that is fine
//!
//! The transcriber hears the sound again every few seconds, reports a word
//! once two rounds in a row agree on it, and holds the sound until its words
//! are settled. So a line for the fifth second is handed over while the
//! tenth is being heard — and here, at the end of a
//! file, the last lines arrive after every video and audio packet has
//! already been muxed. That is not a problem to work around: a packet's
//! place in an MP4 is its timestamp, not its arrival, and the index is
//! written at the end.
//!
//! # Running it
//!
//! ```text
//! cargo run -p transcribe --release -- model.bin input.mp4 [output.mp4]
//! cargo run -p transcribe --release --features vulkan -- --gpu vulkan --language ko model.bin input.mp4
//! ```
//!
//! `--gpu cuda`, `--gpu vulkan` or `--gpu metal` runs the model on that
//! backend's first GPU, with the matching feature built in; without it, the
//! CPU. The language is detected unless `--language` names it, which is
//! slower and can be fooled by a stretch of music — see the README.
//! `--align` times each word against the audio rather than taking
//! whisper.cpp's estimate. `--prompt` gives the model the names and terms
//! to spell as given. `--translate` writes the lines in English.
//! `--sidecar out.srt` (or `.vtt`) also writes the lines to a subtitle file
//! of their own, through a second `FileMuxer`.
//!
//! The model is a whisper.cpp GGML file — `ggml-base.bin` and friends, from
//! `huggingface.co/ggerganov/whisper.cpp`. Nothing that size belongs in a
//! repository, so this takes a path rather than shipping one.
//!
//! `--release` matters more than usual: a debug build of the inference is
//! slower than real time even on a GPU.

fn main() -> Result<(), media_pp::Report> {
    Ok(example::run()?)
}

mod example {
    use media_pp::bus::BusEvent;
    use media_pp::element::BoxFilter;
    use media_pp::elements::{
        AppSink, AudioFormat, AudioResampler, ChunkPolicy, FileDemuxer, FileMuxer, Mp4Layout,
        SubtitleEncoder, SwDecoder, SwWhisperTranscriber, TokenTiming, Transcript,
        WHISPER_SAMPLE_RATE, WhisperOptions,
    };
    use media_pp::{ffmpeg, ffmpeg::media, pipeline::Pipeline, subtitle};

    /// Decoded audio waiting to be transcribed, in buffers.
    ///
    /// Deeper than a display queue would be, because the thing behind it is
    /// slow and bursty: one buffer in ten completes a chunk and pays for a
    /// whole inference while the rest cost nothing. Too shallow and the
    /// demuxer stalls on every tenth buffer; too deep and a model slower
    /// than real time hides how far behind it is.
    const AUDIO_QUEUE_DEPTH: usize = 64;

    pub(super) fn run() -> media_pp::Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        // Options anywhere, and the rest by position.
        let mut options = WhisperOptions::default();
        let mut gpu = None;
        let mut sidecar = None;
        let mut positional = Vec::new();
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--gpu" => gpu = args.next(),
                "--language" => options.language = args.next(),
                "--prompt" => options.prompt = args.next(),
                "--translate" => options.translate = true,
                "--align" => {
                    options.chunk = ChunkPolicy {
                        token_timing: TokenTiming::Aligned,
                        ..ChunkPolicy::default()
                    }
                }
                "--sidecar" => sidecar = args.next(),
                _ => positional.push(arg),
            }
        }
        let mut positional = positional.into_iter();
        let (Some(model_path), Some(input_path)) = (positional.next(), positional.next()) else {
            eprintln!(
                "usage: transcribe [--gpu cuda|vulkan|metal] [--language <code>] [--align] \
                 [--prompt <text>] [--translate] [--sidecar <out.srt|out.vtt>] \
                 <model.bin> <input.mp4> [output.mp4]"
            );
            eprintln!("  the model is a whisper.cpp GGML file, e.g. ggml-base.bin");
            eprintln!("  --gpu runs it on that backend's first GPU, built with its feature");
            eprintln!("  the language is detected unless given, e.g. --language ko");
            eprintln!("  --align times each word against the audio, for cleaner seams");
            eprintln!("  --prompt gives the names and terms to spell as given");
            eprintln!("  --translate writes the lines in English");
            eprintln!("  --sidecar also writes the lines to a subtitle file of their own");
            std::process::exit(1);
        };
        // Settled before anything is opened, so a wrong extension is said
        // before a model has been loaded for nothing.
        let sidecar = sidecar.map(|path| {
            let codec = match std::path::Path::new(&path)
                .extension()
                .and_then(|extension| extension.to_str())
            {
                Some("srt") => subtitle::Codec::SubRip,
                Some("vtt") => subtitle::Codec::WebVtt,
                _ => {
                    eprintln!("--sidecar writes .srt or .vtt, and {path} is neither");
                    std::process::exit(1);
                }
            };
            (path, codec)
        });
        let output_path = positional
            .next()
            .unwrap_or_else(|| "transcribed.mp4".into());

        let (source, _) = FileDemuxer::open("demux", &input_path)?;
        let video = source.best(media::Type::Video).ok();
        let Some(audio) = source.best(media::Type::Audio).ok() else {
            eprintln!("{input_path} has no audio stream, so there is nothing to transcribe");
            std::process::exit(1);
        };

        // Said before rather than after, because on a GPU backend this is
        // where the wait is: the first run on a machine has the driver
        // compile whisper.cpp's shaders, which took over a minute here and
        // under two seconds every run after — that cache outlives the
        // process. Without a line first it looks like a hang.
        println!(
            "loading {model_path} on {} ...",
            gpu.as_deref().unwrap_or("the CPU")
        );
        let transcriber = transcriber(gpu.as_deref(), &model_path, options)?;

        let encoder = SubtitleEncoder::new("mov-text", subtitle::Codec::MovText);
        // Every track is described before the header is written, which is
        // why the text track is registered now — see `FileMuxer::open`.
        let mut muxer = FileMuxer::create(&output_path, Mp4Layout::Standard)?;
        let video_track = match &video {
            Some(video) => Some((video.index(), muxer.add_stream("video", video)?)),
            None => None,
        };
        let audio_track = muxer.add_stream("audio", &audio)?;
        let text_track = muxer.add_stream("text", &encoder)?;
        let mut sinks = muxer.open()?;
        let video_out = match video_track {
            Some((index, track)) => Some((index, sinks.take(track)?)),
            None => None,
        };
        let audio_sink = sinks.take(audio_track)?;
        let text_sink = sinks.take(text_track)?;

        // A file of its own is a muxer of its own: FFmpeg picks the SubRip
        // or WebVTT writer from the extension, and either takes exactly the
        // one text track.
        let sidecar = match sidecar {
            Some((path, codec)) => {
                let encoder = SubtitleEncoder::new("sidecar", codec);
                let mut muxer = FileMuxer::create(&path, Mp4Layout::Standard)?;
                let track = muxer.add_stream("sidecar", &encoder)?;
                Some((encoder, muxer.open()?.take(track)?))
            }
            None => None,
        };

        let lines = AppSink::new("lines", |buf| {
            if let Some(transcript) = buf.metadata().and_then(|m| m.get::<Transcript>()) {
                for line in &transcript.utterances {
                    println!(
                        "[{:>8.2} -> {:>8.2}] {}",
                        line.start_ms as f32 / 1000.0,
                        line.end_ms as f32 / 1000.0,
                        line.text
                    );
                }
            }
            Ok(())
        });

        let decoder = SwDecoder::new("audio-decode", audio.parameters().clone())?;
        let resampler = AudioResampler::new(
            "to-whisper",
            AudioFormat::new(
                ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed),
                WHISPER_SAMPLE_RATE,
                1,
            ),
        );

        let (pipeline, ()) = Pipeline::new("transcribe", source, move |source, ctx| {
            if let Some((index, sink)) = video_out {
                let branch = ctx.branch().to(sink)?;
                ctx.attach(source, index, branch)?;
            }

            // What was heard goes three ways: into the copy's text track,
            // into the sidecar when there is one, and onto the screen.
            let mut heard = ctx
                .tee("heard")
                .branch(ctx.branch().pipe(encoder).to(text_sink)?)
                .branch(ctx.branch().to(lines)?);
            if let Some((encoder, sink)) = sidecar {
                heard = heard.branch(ctx.branch().pipe(encoder).to(sink)?);
            }
            let listen = ctx
                .branch()
                .pipe(decoder)
                .pipe(resampler)
                .queue("speech", AUDIO_QUEUE_DEPTH)
                .pipe(transcriber)
                .to_branch(heard.build()?)?;

            // The audio is wanted twice: once to copy into the file, and
            // once to listen to.
            let copy = ctx.branch().to(audio_sink)?;
            let tee = ctx.tee("audio").branch(copy).branch(listen).build()?;
            ctx.attach(source, audio.index(), tee)?;
            Ok(())
        })?;

        println!("transcribing {input_path} -> {output_path} ...");
        pipeline.run()?;
        for event in pipeline.bus().iter() {
            match &event {
                BusEvent::Error { name, error, .. } => eprintln!("[{name}] error: {error}"),
                BusEvent::Dropped { name, .. } => {
                    eprintln!("[{name}] dropped audio — the model is behind real time")
                }
                // Every track has its end, the transcriber's tail included,
                // and the muxers their trailers.
                BusEvent::Finished => pipeline.stop(),
                _ => {}
            }
        }
        println!("wrote {output_path}");
        Ok(())
    }

    /// The transcriber `gpu` names, on that backend's first GPU, or the CPU.
    fn transcriber(
        gpu: Option<&str>,
        model_path: &str,
        options: WhisperOptions,
    ) -> media_pp::Result<BoxFilter> {
        let name = "transcribe";
        Ok(match gpu {
            None | Some("cpu") => {
                BoxFilter::new(SwWhisperTranscriber::new(name, model_path, options)?)
            }
            #[cfg(feature = "cuda")]
            Some("cuda") => BoxFilter::new(media_pp::elements::CudaWhisperTranscriber::new(
                name, model_path, options, 0,
            )?),
            #[cfg(feature = "vulkan")]
            Some("vulkan") => BoxFilter::new(media_pp::elements::VulkanWhisperTranscriber::new(
                name, model_path, options, 0,
            )?),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Some("metal") => BoxFilter::new(media_pp::elements::MetalWhisperTranscriber::new(
                name, model_path, options, 0,
            )?),
            Some(other) => {
                eprintln!(
                    "--gpu {other} is not built in: build with --features {other} \
                     (cuda, vulkan, or metal on macOS)"
                );
                std::process::exit(1);
            }
        })
    }
}
