//! Transcribes a file's speech and writes a copy of it carrying the result
//! as a subtitle track — video, audio and text in one `.mp4`.
//!
//! ```text
//!                    ┌─ video packets ─────────────────────────────┐
//!                    │                                             │
//! FileDemuxer ───────┤              ┌─ audio packets ──────────────┤─ FileMuxer
//!                    │              │                              │
//!                    └─ audio ─ Tee ┤                              │
//!                                   └─ SwDecoder ─ AudioResampler ─┤
//!                                        ─ Queue ─ WhisperTranscriber
//!                                                        │         │
//!                                                     segments ────┘
//! ```
//!
//! The picture and the sound are copied through as packets — nothing is
//! decoded for them and nothing re-encoded. Only the audio is decoded, and
//! only to be resampled into the one shape Whisper reads: 16 kHz mono f32.
//!
//! # Why the text arrives late, and why that is fine
//!
//! `WhisperTranscriber` gathers a few seconds before it transcribes any of
//! them, then holds back the newest second as too uncertain to report. So a
//! line for the fifth second is handed over while the tenth is being
//! written — and here, at the end of a file, the last lines arrive after
//! every video and audio packet has already been muxed.
//!
//! That is not a problem to work around: a packet's place in an MP4 is its
//! timestamp, not its arrival, and the index is written at the end. The
//! `Queue` before the transcriber is what keeps the wait off the demuxer's
//! thread.
//!
//! # Running it
//!
//! ```text
//! cargo run -p transcribe --release -- model.bin input.mp4 [output.mp4]
//! cargo run -p transcribe --release --features gpu -- --language ko model.bin input.mp4
//! ```
//!
//! The language is detected unless `--language` names it, which is slower
//! and can be fooled by a stretch of music — see the README.
//!
//! The model is a whisper.cpp GGML file — `ggml-base.bin` and friends, from
//! `huggingface.co/ggerganov/whisper.cpp`. Nothing that size belongs in a
//! repository, so this takes a path rather than shipping one.
//!
//! `--release` matters more than usual: a debug build of the inference is
//! slower than real time even on a GPU.

fn main() -> impl std::process::Termination {
    example::run()
}

mod example {
    use std::sync::{Arc, Mutex};

    use media_pp::elements::{
        AudioFormat, AudioResampler, ChunkPolicy, FileDemuxer, FileMuxer, SwDecoder, TeeBuilder,
        TokenTiming, WHISPER_SAMPLE_RATE, WhisperTranscriber,
    };
    use media_pp::ffmpeg;
    use media_pp::{bus::BusEvent, ffmpeg::media, pipeline::Pipeline, subtitle};

    /// The text track's own time base, chosen rather than inherited:
    /// milliseconds are what `Segment` reports in, so this makes the
    /// conversion into packet timestamps the identity.
    const TEXT_TIME_BASE: ffmpeg::Rational = ffmpeg::Rational(1, 1000);

    /// Decoded audio waiting to be transcribed, in buffers.
    ///
    /// Deeper than a display queue would be, because the thing behind it is
    /// slow and bursty: one buffer in ten completes a chunk and pays for a
    /// whole inference while the rest cost nothing. Too shallow and the
    /// demuxer stalls on every tenth buffer; too deep and a model slower
    /// than real time hides how far behind it is.
    const AUDIO_QUEUE_DEPTH: usize = 64;

    pub(super) fn run() -> media_pp::Result<()> {
        media_pp::init()?;
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        // Options anywhere, and the rest by position.
        let mut language = None;
        let mut token_timing = TokenTiming::Estimated;
        let mut positional = Vec::new();
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--language" => language = args.next(),
                "--align" => token_timing = TokenTiming::Aligned,
                _ => positional.push(arg),
            }
        }
        let mut positional = positional.into_iter();
        let (Some(model_path), Some(input_path)) = (positional.next(), positional.next()) else {
            eprintln!(
                "usage: transcribe [--language <code>] [--align] <model.bin> <input.mp4> [output.mp4]"
            );
            eprintln!("  the model is a whisper.cpp GGML file, e.g. ggml-base.bin");
            eprintln!("  the language is detected unless given, e.g. --language ko");
            eprintln!("  --align times each word against the audio, for cleaner seams");
            std::process::exit(1);
        };
        let output_path = positional
            .next()
            .unwrap_or_else(|| "transcribed.mp4".into());

        let (source, streams) = FileDemuxer::open("demux", &input_path)?;
        let video = streams
            .iter()
            .find(|stream| stream.kind == media::Type::Video);
        let Some(audio) = streams
            .iter()
            .find(|stream| stream.kind == media::Type::Audio)
        else {
            eprintln!("{input_path} has no audio stream, so there is nothing to transcribe");
            std::process::exit(1);
        };
        let audio_index = audio.index;
        let audio_params = source
            .stream_parameters(audio_index)
            .expect("the stream was just listed");
        let audio_time_base = source
            .stream_time_base(audio_index)
            .expect("the stream was just listed");

        // Every track is described before the header is written, which is
        // why the text track is registered now and not when the first line
        // of it exists — see `FileMuxer::open`.
        let mut muxer = FileMuxer::create(&output_path)?;
        let video_index = match video {
            Some(video) => {
                muxer.add_stream(
                    "video",
                    source
                        .stream_parameters(video.index)
                        .expect("the stream was just listed"),
                    source
                        .stream_time_base(video.index)
                        .expect("the stream was just listed"),
                )?;
                Some(video.index)
            }
            None => None,
        };
        muxer.add_stream("audio", audio_params.clone(), audio_time_base)?;
        muxer.add_stream("text", subtitle::mov_text_parameters(), TEXT_TIME_BASE)?;

        let mut sinks = muxer.open()?;
        let text_sink = sinks.pop().expect("the text track was registered last");
        let audio_sink = sinks.pop().expect("audio was registered before it");
        let video_sink = sinks.pop();

        // Shared because two things end up writing to it from different
        // moments: the transcriber's callback, for as long as there is
        // speech, and this thread once, to close the track. Nothing else
        // can send that `Eos` — the transcriber is a terminal sink and the
        // callback never learns the audio ran out.
        let text_sink = Arc::new(Mutex::new(text_sink));
        let text_for_segments = Arc::clone(&text_sink);

        // Said before rather than after, because on a GPU backend this is
        // where the wait is: the first run on a machine has the driver
        // compile whisper.cpp's shaders, which took over a minute here and
        // under two seconds every run after — that cache outlives the
        // process. Without a line first it looks like a hang.
        println!("loading {model_path} ({token_timing:?} token times) ...");
        let mut transcriber = WhisperTranscriber::new(
            "transcribe",
            &model_path,
            ChunkPolicy {
                token_timing,
                ..ChunkPolicy::default()
            },
            move |segment| {
                println!(
                    "[{:>8.2} -> {:>8.2}] {}",
                    segment.start_ms as f32 / 1000.0,
                    segment.end_ms as f32 / 1000.0,
                    segment.text
                );
                // A line with no length would be on screen for no time at
                // all, and stretching it to one millisecond would overlap
                // whatever starts where it does — which a text track cannot
                // hold. Printed, and left out of the file.
                let duration = segment.end_ms - segment.start_ms;
                if duration <= 0 {
                    return Ok(());
                }
                let packet = subtitle::packet(&segment.text, segment.start_ms, duration);
                text_for_segments
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .consume(packet)
            },
        )?;
        if let Some(language) = &language {
            transcriber = transcriber.with_language(language)?;
        }

        let decoder = SwDecoder::new("audio-decode", audio_params)?;
        let resampler = AudioResampler::new(
            "to-whisper",
            AudioFormat::new(
                ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed),
                WHISPER_SAMPLE_RATE,
                1,
            ),
            audio_time_base,
        )?;

        // Which terminals have to finish before the text track can be
        // closed. By name rather than by counting, because a `Tee` reports
        // its own `Eos` too and counting would take that for a track's and
        // stop early — cutting off exactly the tail the transcriber flushes
        // last.
        //
        // The transcriber is in here for that reason: it is the only thing
        // that knows whether more lines are coming.
        let mut waiting: Vec<&str> = vec!["audio", "transcribe"];
        if video_index.is_some() {
            waiting.push("video");
        }

        let pipeline = Pipeline::new("transcribe", source, move |source, ctx| {
            if let (Some(index), Some(sink)) = (video_index, video_sink) {
                let branch = ctx.branch().to(sink)?;
                ctx.attach(source, index, branch)?;
            }

            // The audio is wanted twice: once to copy into the file, and
            // once to listen to.
            let copy_branch = ctx.branch().to(audio_sink)?;
            let listen_branch = ctx
                .branch()
                .pipe(decoder)
                .pipe(resampler)
                .queue("speech", AUDIO_QUEUE_DEPTH)
                .to(Box::new(transcriber))?;
            let tee = TeeBuilder::new("audio-tee", ctx.clone())
                .branch(copy_branch)
                .branch(listen_branch)
                .build()?;
            ctx.attach(source, audio_index, tee)?;
            Ok(())
        })?;

        println!("transcribing {input_path} -> {output_path} ...");
        pipeline.run()?;

        for event in pipeline.bus().iter() {
            match &event {
                BusEvent::Eos { name, .. } => {
                    println!("[{name}] eos");
                    waiting.retain(|awaited| *awaited != &**name);
                }
                BusEvent::Error { name, error, .. } => eprintln!("[{name}] error: {error}"),
                BusEvent::Dropped { name, .. } => {
                    eprintln!("[{name}] dropped audio — the model is behind real time")
                }
                _ => {}
            }
            if waiting.is_empty() || matches!(event, BusEvent::Error { .. }) {
                break;
            }
        }

        // The transcriber has flushed its tail by now, so every line that
        // is ever coming has been written. Closing the track is what lets
        // the muxer write its trailer, and without it the file has no index
        // and will not play.
        text_sink
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .consume(media_pp::buffer::MediaBuffer::Eos)?;

        println!("wrote {output_path}");
        Ok(())
    }
}
