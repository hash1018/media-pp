//! The model alone: the whole clip handed to whisper.cpp in one call, as
//! its own command-line tool would hand it — the ceiling the streaming
//! transcribers are held against, in speed and in what is heard.
//!
//! A transcriber hears a live stream a few seconds at a time, each round
//! running the model's 30-second window over what it has, so it pays for
//! far more audio than it hears. Here whisper.cpp walks the clip itself, a
//! window at a time with no overlap, and nothing waits on what has not
//! arrived yet. The clip is decoded first and only the inference is timed.
//!
//!     cargo run --release -p speech_bench --bin model_only --features cuda,vulkan -- \
//!         clip.wav --model ggml-base.bin [--backend cpu|cuda|vulkan|metal] \
//!         [--language ko] [--threads N] [--transcript heard.txt] [--label NAME]

#[path = "../common.rs"]
mod common;

use std::ffi::CStr;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use media_pp::buffer::MediaBuffer;
use media_pp::bus::BusEvent;
use media_pp::elements::{AppSink, AudioFormat, AudioResampler, FileDemuxer, SwDecoder};
use media_pp::{ffmpeg, ffmpeg::media, pipeline::Pipeline};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use common::{cpu_seconds, parse, registry_name, write_transcript};

/// What Whisper reads: 16 kHz.
const SAMPLE_RATE: u32 = 16_000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (mut model, mut backend, mut language, mut threads) =
        (None, "cpu".to_owned(), None, None::<i32>);
    let (mut transcript, mut label) = (None, None);
    let mut positional = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model = args.next(),
            "--backend" => backend = args.next().unwrap_or_default(),
            "--language" => language = args.next(),
            "--threads" => threads = Some(parse("--threads", args.next())),
            "--transcript" => transcript = args.next(),
            "--label" => label = args.next(),
            _ => positional.push(arg),
        }
    }
    let (Some(clip), Some(model)) = (positional.first().cloned(), model) else {
        eprintln!(
            "usage: model_only <clip> --model <ggml.bin> [--backend cpu|cuda|vulkan|metal] \
             [--language <code>] [--threads N] [--transcript <out.txt>] [--label <name>]"
        );
        std::process::exit(2);
    };
    let Some(gpu) = registry_name(&backend) else {
        eprintln!("--backend {backend} is not built in");
        std::process::exit(2);
    };

    let audio = decode(&clip)?;

    let loading = Instant::now();
    let mut params = WhisperContextParameters::default();
    match gpu {
        None => {
            params.use_gpu(false);
        }
        Some(name) => {
            params.use_gpu(true);
            params.gpu_device(gpu_device(name)?);
        }
    }
    let context = WhisperContext::new_with_params(&model, params)?;
    let mut state = context.create_state()?;
    let ready = loading.elapsed().as_secs_f64();

    // As the transcribers ask, less what only streaming needs.
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_print_progress(false);
    params.set_print_special(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_language(Some(language.as_deref().unwrap_or("auto")));
    if let Some(threads) = threads {
        params.set_n_threads(threads);
    }

    let started = (Instant::now(), cpu_seconds());
    state.full(params, &audio)?;
    let wall = started.0.elapsed().as_secs_f64();
    let cpu = match (started.1, cpu_seconds()) {
        (Some(before), Some(after)) => format!("{:.2}", (after - before) / wall),
        _ => "-".into(),
    };

    let lines: Vec<String> = (0..state.full_n_segments())
        .filter_map(|index| state.get_segment(index))
        .filter_map(|segment| {
            segment
                .to_str_lossy()
                .ok()
                .map(|text| text.trim().to_owned())
        })
        .collect();
    write_transcript(transcript.as_deref(), &lines);

    let seconds = audio.len() as f64 / f64::from(SAMPLE_RATE);
    let model_name = std::path::Path::new(&model)
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_default();
    println!(
        "RESULT label={} backend={backend} model={model_name} audio={seconds:.1} \
         wall={wall:.2} speed={:.2} cpu={cpu} ready={ready:.2} lines={}",
        label.as_deref().unwrap_or(&backend),
        seconds / wall,
        lines.len(),
    );
    Ok(())
}

/// The clip's best audio stream, as 16 kHz mono f32.
fn decode(clip: &str) -> media_pp::Result<Vec<f32>> {
    let _log_guard = media_pp::log::init("model_only", "logs", media_pp::log::Level::Trace, 7)?;
    let (source, _) = FileDemuxer::open("demux", clip)?;
    let audio = source.best(media::Type::Audio)?;
    let decoder = SwDecoder::new("decode", audio.parameters.clone())?;
    let resampler = AudioResampler::new(
        "to-whisper",
        AudioFormat::new(
            ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed),
            SAMPLE_RATE,
            1,
        ),
    );
    let samples = Arc::new(Mutex::new(Vec::new()));
    let sink = {
        let samples = samples.clone();
        AppSink::new("samples", move |buf| {
            if let MediaBuffer::Audio(frame) = &buf {
                samples
                    .lock()
                    .unwrap()
                    .extend_from_slice(&frame.plane::<f32>(0)[..frame.samples()]);
            }
            Ok(())
        })
    };
    let (pipeline, ()) = Pipeline::new("decode", source, move |source, ctx| {
        let branch = ctx.branch().pipe(decoder).pipe(resampler).to(sink)?;
        ctx.attach(source, audio.index, branch)?;
        Ok(())
    })?;
    pipeline.run()?;
    for event in pipeline.bus().iter() {
        match &event {
            BusEvent::Error { name, error, .. } => eprintln!("[{name}] error: {error}"),
            BusEvent::Finished => pipeline.stop(),
            _ => {}
        }
    }
    drop(pipeline);
    Ok(std::mem::take(&mut *samples.lock().unwrap()))
}

/// whisper.cpp's number for the first GPU of the backend registered as
/// `name`, as the transcribers find theirs: built with CUDA and Vulkan, one
/// card is two GPUs to it.
fn gpu_device(name: &str) -> Result<i32, String> {
    use whisper_rs::whisper_rs_sys as sys;
    let mut gpus = 0;
    // SAFETY: the registry is ggml's own, built on first use and never torn
    // down; the devices and names it hands out live as long as the process.
    unsafe {
        for at in 0..sys::ggml_backend_dev_count() {
            let device = sys::ggml_backend_dev_get(at);
            let kind = sys::ggml_backend_dev_type(device);
            if kind != sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU
                && kind != sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU
            {
                continue;
            }
            let registry = sys::ggml_backend_dev_backend_reg(device);
            let registered = CStr::from_ptr(sys::ggml_backend_reg_name(registry));
            if registered.to_bytes().eq_ignore_ascii_case(name.as_bytes()) {
                return Ok(gpus);
            }
            gpus += 1;
        }
    }
    Err(format!("no {name} GPU found"))
}
