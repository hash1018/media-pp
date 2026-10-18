//! How fast and how well a transcriber hears, one configuration at a time,
//! so that configurations can be compared — the harness
//! `docs/benchmarks/speech` is measured with. Each run builds the pipeline
//! it is asked for, plays the clip through it, writes what was heard where
//! `--transcript` says, and ends with one `RESULT` line of `key=value` pairs
//! that a script can collect.
//!
//!     FileDemuxer -> SwDecoder -> AudioResampler -> [Pacer] -> Queue
//!         -> XxxWhisperTranscriber -> AppSink
//!
//! Without `--realtime` the clip goes through as fast as the model hears
//! it, and `speed` is how many seconds of sound a second of that took —
//! 2.0 is twice real time. With it, a `Pacer` hands the sound over as it
//! would arrive live, and `lag` is how long after a line's last word was
//! said the line came out: what a viewer of live captions waits.
//!
//! `cpu` is the process's CPU time while the clip played, in cores: 1.0 is
//! one core kept busy. `ready` is the seconds the model took to load.
//!
//!     cargo run --release -p speech_bench --features cuda,vulkan -- clip.wav \
//!         --model ggml-base.bin [--backend cpu|cuda|vulkan|metal] [--language ko] \
//!         [--align] [--chunk-ms N] [--threads N] [--realtime] \
//!         [--transcript heard.txt] [--label NAME]
//!
//! Each GPU backend is there only when built with its feature.

mod common;

fn main() -> Result<(), media_pp::Report> {
    Ok(bench::run()?)
}

mod bench {
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    use media_pp::buffer::MediaBuffer;
    use media_pp::bus::BusEvent;
    use media_pp::element::{BoxFilter, Filter, Output};
    use media_pp::elements::{
        AppSink, AudioFormat, AudioResampler, ChunkPolicy, FileDemuxer, Pacer, SwDecoder,
        SwWhisperTranscriber, TokenTiming, Transcript, WHISPER_SAMPLE_RATE, WhisperOptions,
    };
    use media_pp::{ffmpeg, ffmpeg::media, pipeline::Pipeline};

    use crate::common::{cpu_seconds, parse, registry_name, write_transcript};

    /// Sound waiting for the transcriber, in buffers: deep enough that the
    /// demuxer never waits on an inference, as in the `transcribe` example.
    const QUEUE_DEPTH: usize = 64;

    /// What the sink saw: the lines, each with how late it came out when
    /// paced, and how much sound went by.
    #[derive(Default)]
    struct Heard {
        lines: Vec<String>,
        lags: Vec<f64>,
        samples: u64,
    }

    /// When the first sound left the pacer, which is when its timeline's
    /// zero was being played.
    type Origin = Arc<Mutex<Option<(Instant, f64)>>>;

    /// Notes when the paced sound began, and hands every buffer on.
    struct Stamp(Origin);

    impl Filter for Stamp {
        fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> media_pp::Result<()> {
            let mut origin = self.0.lock().unwrap();
            if origin.is_none() {
                let seconds = match (buf.pts(), buf.time_base()) {
                    (Some(pts), Some(unit)) => pts as f64 * f64::from(unit),
                    _ => 0.0,
                };
                *origin = Some((Instant::now(), seconds));
            }
            drop(origin);
            out.push(buf);
            Ok(())
        }
    }

    pub(super) fn run() -> media_pp::Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let mut options = WhisperOptions::default();
        let mut chunk = ChunkPolicy::default();
        let (mut model, mut backend, mut transcript, mut label) =
            (None, "cpu".to_owned(), None, None);
        let mut realtime = false;
        let mut positional = Vec::new();
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model" => model = args.next(),
                "--backend" => backend = args.next().unwrap_or_default(),
                "--language" => options.language = args.next(),
                "--align" => chunk.token_timing = TokenTiming::Aligned,
                "--chunk-ms" => chunk.chunk_ms = parse("--chunk-ms", args.next()),
                "--threads" => options.threads = Some(parse("--threads", args.next())),
                "--realtime" => realtime = true,
                "--transcript" => transcript = args.next(),
                "--label" => label = args.next(),
                _ => positional.push(arg),
            }
        }
        options.chunk = chunk;
        let (Some(clip), Some(model)) = (positional.first().cloned(), model) else {
            eprintln!(
                "usage: speech_bench <clip> --model <ggml.bin> [--backend cpu|cuda|vulkan|metal] \
                 [--language <code>] [--align] [--chunk-ms N] [--threads N] \
                 [--realtime] [--transcript <out.txt>] [--label <name>]"
            );
            std::process::exit(2);
        };
        let Some(gpu) = registry_name(&backend) else {
            eprintln!("--backend {backend} is not built in");
            std::process::exit(2);
        };

        let (source, _) = FileDemuxer::open("demux", &clip)?;
        let audio = source.best(media::Type::Audio)?;
        let decoder = SwDecoder::new("decode", audio.parameters.clone())?;
        let resampler = AudioResampler::new(
            "to-whisper",
            AudioFormat::new(
                ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed),
                WHISPER_SAMPLE_RATE,
                1,
            ),
        );

        let loading = Instant::now();
        let transcriber = transcriber(gpu, &model, options)?;
        let ready = loading.elapsed().as_secs_f64();

        let heard = Arc::new(Mutex::new(Heard::default()));
        let origin = Origin::default();
        let sink = {
            let (heard, origin) = (heard.clone(), origin.clone());
            AppSink::new("heard", move |buf| {
                let now = Instant::now();
                let mut heard = heard.lock().unwrap();
                if let MediaBuffer::Audio(frame) = &buf {
                    heard.samples += frame.samples() as u64;
                }
                if let Some(said) = buf.metadata().and_then(|m| m.get::<Transcript>()) {
                    let origin = *origin.lock().unwrap();
                    for line in &said.utterances {
                        heard.lines.push(line.text.trim().to_owned());
                        if let Some((at, zero)) = origin {
                            let ended = line.end_ms as f64 / 1000.0 - zero;
                            heard
                                .lags
                                .push(now.duration_since(at).as_secs_f64() - ended);
                        }
                    }
                }
                Ok(())
            })
        };

        let (pipeline, ()) = Pipeline::new("speech-bench", source, move |source, ctx| {
            let mut chain = ctx.branch().pipe(decoder).pipe(resampler);
            if realtime {
                chain = chain.pipe(Pacer::new("live")).pipe(Stamp(origin.clone()));
            }
            let branch = chain
                .queue("speech", QUEUE_DEPTH)
                .pipe(transcriber)
                .to(sink)?;
            ctx.attach(source, audio.index, branch)?;
            Ok(())
        })?;

        let started = (Instant::now(), cpu_seconds());
        pipeline.run()?;
        for event in pipeline.bus().iter() {
            match &event {
                BusEvent::Error { name, error, .. } => eprintln!("[{name}] error: {error}"),
                BusEvent::Finished => pipeline.stop(),
                _ => {}
            }
        }
        let wall = started.0.elapsed().as_secs_f64();
        let cpu = match (started.1, cpu_seconds()) {
            (Some(before), Some(after)) => format!("{:.2}", (after - before) / wall),
            _ => "-".into(),
        };

        let heard = heard.lock().unwrap();
        write_transcript(transcript.as_deref(), &heard.lines);
        let seconds = heard.samples as f64 / f64::from(WHISPER_SAMPLE_RATE);
        let lag = if realtime && !heard.lags.is_empty() {
            let mut lags = heard.lags.clone();
            lags.sort_by(f64::total_cmp);
            let mean = lags.iter().sum::<f64>() / lags.len() as f64;
            let p90 = lags[(lags.len() * 9).div_ceil(10) - 1];
            format!(
                " lag={mean:.2} lag_p90={p90:.2} lag_max={:.2}",
                lags[lags.len() - 1]
            )
        } else {
            String::new()
        };
        let model_name = std::path::Path::new(&model)
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned())
            .unwrap_or_default();
        println!(
            "RESULT label={} backend={backend} model={model_name} audio={seconds:.1} \
             wall={wall:.2} speed={:.2} cpu={cpu} ready={ready:.2} lines={}{lag}",
            label.as_deref().unwrap_or(&backend),
            seconds / wall,
            heard.lines.len(),
        );
        Ok(())
    }

    /// The transcriber for `gpu`'s backend, on its first GPU, or the CPU.
    fn transcriber(
        gpu: Option<&str>,
        model: &str,
        options: WhisperOptions,
    ) -> media_pp::Result<BoxFilter> {
        let name = "transcribe";
        Ok(match gpu {
            None => BoxFilter::new(SwWhisperTranscriber::new(name, model, options)?),
            #[cfg(feature = "cuda")]
            Some("CUDA") => BoxFilter::new(media_pp::elements::CudaWhisperTranscriber::new(
                name, model, options, 0,
            )?),
            #[cfg(feature = "vulkan")]
            Some("Vulkan") => BoxFilter::new(media_pp::elements::VulkanWhisperTranscriber::new(
                name, model, options, 0,
            )?),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Some("Metal") => BoxFilter::new(media_pp::elements::MetalWhisperTranscriber::new(
                name, model, options, 0,
            )?),
            Some(other) => unreachable!("{other} is not a backend registry_name gives"),
        })
    }
}
