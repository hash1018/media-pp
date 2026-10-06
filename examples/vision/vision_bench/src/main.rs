//! How fast a vision pipeline runs, one configuration at a time, so that
//! configurations can be compared — the harness `docs/benchmarks/vision`
//! is measured with. Each run builds the pipeline it is asked for, plays the
//! file through it as fast as it goes, and ends with one `RESULT` line of
//! `key=value` pairs that a script can collect.
//!
//!     FileDemuxer -> decoder -> Queue -> [detector] -> [ObjectTracker]
//!         -> [classifier] -> [overlay] -> [Queue -> encoder] -> AppSink
//!
//! `--streams N` plays N copies of the file. With `--batch B` they meet in a
//! `StreamMux` and one detector runs up to B pictures at once — DeepStream's
//! `nvstreammux -> nvinfer -> nvstreamdemux` — the tracker and classifier
//! after it, and the overlay and encoder on each stream's own branch after
//! the demux. With `--batch 0`, the default, each stream is a pipeline of
//! its own with a detector of its own.
//!
//! The first `--warmup` pictures (all streams together) are left out of the
//! rate, so a TensorRT engine's first runs and the decoder's start do not
//! count. `cpu` is the process's CPU time over the same span, in cores: 1.0
//! is one core kept busy.
//!
//!     cargo run --release -p vision_bench -- video.mp4 --model yolo11n.onnx \
//!         [--backend cpu|cuda|tensorrt|metal] [--fp32] [--compute-units all|gpu|ane] \
//!         [--streams N] [--batch B] \
//!         [--interval N] [--track motion|visual] [--classifier model.onnx] \
//!         [--reclassify N] [--min-detection-score S] [--overlay] [--encode] \
//!         [--pictures N] [--warmup N] [--label NAME] [--engine-cache-root DIR]
//!
//! Without `--model`, only decoding is measured. `--track visual` follows
//! CUDA pictures on the GPU when built with `--features gpu-dcf`, and copies
//! each object down to follow it on the CPU without. `cuda` and `tensorrt`
//! are Linux and Windows. `metal` is macOS: VideoToolbox decoding and
//! encoding, the model through Core ML — on the units `--compute-units`
//! lets it use, anywhere by default — and the tracker following
//! VideoToolbox pictures on the GPU.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};

use media_pp::{
    Result,
    buffer::MediaBuffer,
    bus::BusEvent,
    elements::{
        AppSink, BoxStyle, DetectionOverlayOptions, Detections, FileDemuxer, InputScale,
        LabelStyle, ObjectTracker, OrtClassifierOptions, OrtDetectorOptions, RenderMode, StreamMux,
        StreamMuxOptions, SwDecoder, SwDetectionOverlay, SwOrtClassifier, SwOrtDetector,
        TrackerOptions, Treatment,
    },
    ffmpeg::{Rational, codec::Parameters, media},
    pipeline::{ChainBuilder, DetachedBranch, Pipeline},
};

#[cfg(any(target_os = "linux", target_os = "windows"))]
use media_pp::elements::{
    CudaCodec, CudaDecoder, CudaDetectionOverlay, CudaDevice, CudaEncoder, CudaEncoderOptions,
    CudaFrameFormat, CudaOrtClassifier, CudaOrtDetector, CudaOrtDetectorOptions, UseTensorRtPolicy,
};

#[cfg(target_os = "macos")]
use media_pp::elements::{
    CoreMlComputeUnits, MetalDetectionOverlay, MetalOrtClassifier, MetalOrtDetector,
    MetalOrtDetectorOptions, VideoToolboxCodec, VideoToolboxDecoder, VideoToolboxDevice,
    VideoToolboxEncoder, VideoToolboxEncoderOptions, VideoToolboxFrameFormat,
};

/// Where the labels' font is looked for; without one, boxes alone.
const FONTS: [&str; 5] = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
];

fn main() -> impl std::process::Termination {
    run()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Backend {
    Cpu,
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    Cuda,
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    TensorRt,
    #[cfg(target_os = "macos")]
    Metal,
}

impl Backend {
    fn name(self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            Backend::Cuda => "cuda",
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            Backend::TensorRt => "tensorrt",
            #[cfg(target_os = "macos")]
            Backend::Metal => "metal",
        }
    }
}

/// Where Core ML may run the model, as `--compute-units` names it.
#[cfg(target_os = "macos")]
fn compute_units(name: &str) -> Option<CoreMlComputeUnits> {
    match name {
        "all" => Some(CoreMlComputeUnits::All),
        "gpu" => Some(CoreMlComputeUnits::CpuAndGpu),
        "ane" => Some(CoreMlComputeUnits::CpuAndNeuralEngine),
        _ => None,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Track {
    Off,
    Motion,
    Visual,
}

struct Args {
    video: String,
    model: Option<String>,
    backend: Backend,
    fp32: bool,
    /// `--compute-units`, as given: Core ML's, `metal` alone.
    units: String,
    streams: usize,
    batch: usize,
    interval: u32,
    track: Track,
    classifier: Option<String>,
    reclassify: u32,
    min_detection_score: f32,
    overlay: bool,
    encode: bool,
    pictures: usize,
    warmup: usize,
    label: String,
    engine_root: Option<std::path::PathBuf>,
}

fn args() -> Args {
    let usage = |why: &str| -> ! {
        eprintln!(
            "{why}\nusage: vision_bench <video> [--model model.onnx] [--backend cpu|cuda|tensorrt] \
             [--fp32] [--compute-units all|gpu|ane] [--streams N] [--batch B] [--interval N] \
             [--track motion|visual] \
             [--classifier model.onnx] [--reclassify N] [--min-detection-score S] \
             [--overlay] [--encode] [--pictures N] [--warmup N] [--label NAME] \
             [--engine-cache-root DIR]"
        );
        std::process::exit(2);
    };
    let number = |value: Option<String>| -> usize {
        value
            .and_then(|n| n.parse().ok())
            .unwrap_or_else(|| usage("a number was expected"))
    };
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    let mut backend = Backend::TensorRt;
    #[cfg(target_os = "macos")]
    let mut backend = Backend::Metal;
    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    let mut backend = Backend::Cpu;
    let mut parsed = Args {
        video: String::new(),
        model: None,
        backend,
        fp32: false,
        units: String::from("all"),
        streams: 1,
        batch: 0,
        interval: 0,
        track: Track::Off,
        classifier: None,
        reclassify: OrtClassifierOptions::default().reclassify,
        min_detection_score: OrtClassifierOptions::default().min_detection_score,
        overlay: false,
        encode: false,
        pictures: usize::MAX,
        warmup: 60,
        label: String::from("-"),
        engine_root: None,
    };
    let mut positional = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => parsed.model = args.next(),
            "--backend" => {
                backend = match args.next().as_deref() {
                    Some("cpu") => Backend::Cpu,
                    #[cfg(any(target_os = "linux", target_os = "windows"))]
                    Some("cuda") => Backend::Cuda,
                    #[cfg(any(target_os = "linux", target_os = "windows"))]
                    Some("tensorrt") => Backend::TensorRt,
                    #[cfg(target_os = "macos")]
                    Some("metal") => Backend::Metal,
                    _ => usage("no such backend here"),
                };
                parsed.backend = backend;
            }
            "--fp32" => parsed.fp32 = true,
            "--compute-units" => {
                parsed.units = args
                    .next()
                    .unwrap_or_else(|| usage("--compute-units all|gpu|ane"))
            }
            "--streams" => parsed.streams = number(args.next()).max(1),
            "--batch" => parsed.batch = number(args.next()),
            "--interval" => parsed.interval = number(args.next()) as u32,
            "--track" => {
                parsed.track = match args.next().as_deref() {
                    Some("motion") => Track::Motion,
                    Some("visual") => Track::Visual,
                    _ => usage("--track is motion or visual"),
                }
            }
            "--classifier" => parsed.classifier = args.next(),
            "--reclassify" => parsed.reclassify = number(args.next()) as u32,
            "--min-detection-score" => {
                parsed.min_detection_score = args
                    .next()
                    .and_then(|n| n.parse().ok())
                    .unwrap_or_else(|| usage("a score was expected"))
            }
            "--overlay" => parsed.overlay = true,
            "--encode" => parsed.encode = true,
            "--pictures" => parsed.pictures = number(args.next()),
            "--warmup" => parsed.warmup = number(args.next()),
            "--label" => parsed.label = args.next().unwrap_or_else(|| usage("--label NAME")),
            "--engine-cache-root" => {
                parsed.engine_root =
                    Some(args.next().unwrap_or_else(|| usage("a directory")).into())
            }
            _ => positional.push(arg),
        }
    }
    let [video] = <[String; 1]>::try_from(positional).unwrap_or_else(|_| usage("one video"));
    parsed.video = video;
    if parsed.model.is_none()
        && (parsed.batch > 0
            || parsed.track != Track::Off
            || parsed.classifier.is_some()
            || parsed.overlay)
    {
        usage("batching, tracking, classifying and drawing need --model");
    }
    if parsed.classifier.is_some() && parsed.track == Track::Off {
        usage("--classifier remembers by track: it needs --track");
    }
    if parsed.backend == Backend::Cpu && (parsed.batch > 0 || parsed.encode) {
        usage("the cpu backend neither batches nor encodes");
    }
    #[cfg(target_os = "macos")]
    if compute_units(&parsed.units).is_none() {
        usage("--compute-units is all, gpu or ane");
    }
    parsed
}

/// What the sinks counted, and when the warm-up ended.
struct Meter {
    pictures: AtomicUsize,
    objects: AtomicUsize,
    warmup: usize,
    warm: Mutex<Option<(Instant, Option<f64>)>>,
}

impl Meter {
    /// Ends `branch` in a sink that counts what reaches it.
    fn count(self: &Arc<Self>, branch: ChainBuilder, name: String) -> Result<DetachedBranch> {
        let meter = Arc::clone(self);
        branch.to(AppSink::new(name, move |buf| {
            if !matches!(buf, MediaBuffer::Video(_) | MediaBuffer::Packet(_)) {
                return Ok(());
            }
            let n = meter.pictures.fetch_add(1, Ordering::Relaxed) + 1;
            if n == meter.warmup {
                *meter.warm.lock().unwrap() = Some((Instant::now(), cpu_seconds()));
            }
            if let Some(found) = buf.metadata().and_then(|m| m.get::<Detections>()) {
                meter
                    .objects
                    .fetch_add(found.items.len(), Ordering::Relaxed);
            }
            Ok(())
        }))
    }
}

/// The CPU time this process has spent, user and system together.
#[cfg(unix)]
fn cpu_seconds() -> Option<f64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage writes a whole rusage into the pointer it is given,
    // which points at one, and is read only when it says it succeeded.
    let usage = unsafe {
        if libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) != 0 {
            return None;
        }
        usage.assume_init()
    };
    let seconds = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 / 1e6;
    Some(seconds(usage.ru_utime) + seconds(usage.ru_stime))
}

/// The CPU time this process has spent, user and kernel together.
#[cfg(windows)]
fn cpu_seconds() -> Option<f64> {
    use windows::Win32::{
        Foundation::FILETIME,
        System::Threading::{GetCurrentProcess, GetProcessTimes},
    };
    let (mut created, mut exited, mut kernel, mut user) = (
        FILETIME::default(),
        FILETIME::default(),
        FILETIME::default(),
        FILETIME::default(),
    );
    // SAFETY: the pseudo-handle of the current process needs no closing, and
    // each pointer is to a FILETIME of this frame, read only once the call
    // has said it succeeded.
    unsafe {
        GetProcessTimes(
            GetCurrentProcess(),
            &mut created,
            &mut exited,
            &mut kernel,
            &mut user,
        )
        .ok()?;
    }
    // Hundreds of nanoseconds.
    let seconds = |t: FILETIME| {
        ((u64::from(t.dwHighDateTime) << 32) | u64::from(t.dwLowDateTime)) as f64 / 1e7
    };
    Some(seconds(kernel) + seconds(user))
}

#[cfg(not(any(unix, windows)))]
fn cpu_seconds() -> Option<f64> {
    None
}

/// The GPU the GPU backends share.
#[cfg(any(target_os = "linux", target_os = "windows"))]
type Device = Option<CudaDevice>;
#[cfg(target_os = "macos")]
type Device = Option<VideoToolboxDevice>;
#[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
type Device = Option<()>;

/// What one stream is, for its decoder and encoder.
struct Stream {
    index: usize,
    params: Parameters,
    size: (u32, u32),
    rate: Rational,
}

fn decoder(branch: ChainBuilder, args: &Args, device: &Device, s: &Stream) -> Result<ChainBuilder> {
    let name = format!("decoder {}", s.index);
    let branch = match args.backend {
        Backend::Cpu => branch.pipe(SwDecoder::new(name, s.params.clone())?),
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        Backend::Cuda | Backend::TensorRt => {
            let device = device.as_ref().expect("a GPU backend has a device");
            branch.pipe(CudaDecoder::new(name, s.params.clone(), device, 16)?)
        }
        #[cfg(target_os = "macos")]
        Backend::Metal => {
            let device = device.as_ref().expect("a GPU backend has a device");
            branch.pipe(VideoToolboxDecoder::new(name, s.params.clone(), device)?)
        }
    };
    let _ = device;
    Ok(branch.queue(format!("pictures {}", s.index), 8))
}

fn detector(
    branch: ChainBuilder,
    args: &Args,
    device: &Device,
    name: String,
) -> Result<ChainBuilder> {
    let Some(model) = &args.model else {
        return Ok(branch);
    };
    let options = OrtDetectorOptions {
        // The tracker matches unconfident detections too.
        conf_threshold: if args.track == Track::Off {
            OrtDetectorOptions::default().conf_threshold
        } else {
            TrackerOptions::default().low_score
        },
        interval: args.interval,
        ..OrtDetectorOptions::default()
    };
    let _ = device;
    Ok(match args.backend {
        Backend::Cpu => branch.pipe(SwOrtDetector::new(name, model, options)?),
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        backend @ (Backend::Cuda | Backend::TensorRt) => {
            let device = device.as_ref().expect("a GPU backend has a device");
            branch.pipe(CudaOrtDetector::new(
                name,
                device,
                model,
                CudaOrtDetectorOptions {
                    detector: options,
                    max_batch: args.batch.max(1),
                    tensorrt: if backend == Backend::TensorRt {
                        UseTensorRtPolicy::Required
                    } else {
                        UseTensorRtPolicy::Off
                    },
                    fp16: !args.fp32,
                    engine_cache: args
                        .engine_root
                        .as_ref()
                        .map(|root| engine_cache(root, model, !args.fp32, args.batch.max(1))),
                },
            )?)
        }
        #[cfg(target_os = "macos")]
        Backend::Metal => branch.pipe(MetalOrtDetector::new(
            name,
            model,
            MetalOrtDetectorOptions {
                detector: options,
                max_batch: args.batch.max(1),
                compute_units: compute_units(&args.units).unwrap_or_default(),
            },
        )?),
    })
}

/// Where TensorRT keeps the engine for this model, precision and batch.
/// ONNX Runtime names an engine after the model's graph alone, so engines
/// for two batches of one model would take each other's place in one
/// directory, and each run would build its own again.
#[cfg(any(target_os = "linux", target_os = "windows"))]
fn engine_cache(
    root: &std::path::Path,
    model: &str,
    fp16: bool,
    batch: usize,
) -> std::path::PathBuf {
    let stem = std::path::Path::new(model)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    root.join(format!(
        "{stem}-{}-b{batch}",
        if fp16 { "fp16" } else { "fp32" }
    ))
}

/// The tracker and the classifier after it, as asked for.
fn analysis(
    branch: ChainBuilder,
    args: &Args,
    device: &Device,
    index: String,
) -> Result<ChainBuilder> {
    let mut branch = branch;
    if args.track != Track::Off {
        branch = branch.pipe(ObjectTracker::new(
            format!("tracker {index}"),
            TrackerOptions {
                visual: args.track == Track::Visual,
                ..TrackerOptions::default()
            },
        ));
    }
    if let Some(model) = &args.classifier {
        // Every answer is kept, however unsure: what is measured is the
        // classifying, not whether this model knows these objects — an
        // ImageNet model on people is unsure of all of them, and an answer
        // not kept is asked for again on the next picture.
        let options = OrtClassifierOptions {
            input: InputScale::ImageNet,
            min_score: 0.0,
            reclassify: args.reclassify,
            min_detection_score: args.min_detection_score,
            ..OrtClassifierOptions::default()
        };
        let name = format!("classifier {index}");
        branch = match args.backend {
            Backend::Cpu => branch.pipe(SwOrtClassifier::new(name, model, options)?),
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            Backend::Cuda | Backend::TensorRt => {
                let device = device.as_ref().expect("a GPU backend has a device");
                branch.pipe(CudaOrtClassifier::new(name, device, model, options)?)
            }
            #[cfg(target_os = "macos")]
            Backend::Metal => branch.pipe(MetalOrtClassifier::new(name, model, options)?),
        };
    }
    let _ = device;
    Ok(branch)
}

/// The overlay and the encoder, as asked for.
fn output(branch: ChainBuilder, args: &Args, device: &Device, s: &Stream) -> Result<ChainBuilder> {
    let mut branch = branch;
    if args.overlay {
        let font = FONTS.iter().find_map(|path| std::fs::read(path).ok());
        let options = DetectionOverlayOptions {
            others: Treatment::boxes(BoxStyle {
                line_width: 4,
                label: font.is_some().then(|| LabelStyle::new(22.0)),
                ..BoxStyle::default()
            }),
            font,
            ..DetectionOverlayOptions::default()
        };
        let name = format!("overlay {}", s.index);
        branch = match args.backend {
            Backend::Cpu => branch.pipe(SwDetectionOverlay::new(name, options)?),
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            Backend::Cuda | Backend::TensorRt => {
                let device = device.as_ref().expect("a GPU backend has a device");
                branch.pipe(CudaDetectionOverlay::new(name, device, options)?)
            }
            #[cfg(target_os = "macos")]
            Backend::Metal => {
                let device = device.as_ref().expect("a GPU backend has a device");
                branch.pipe(MetalDetectionOverlay::new(name, device, options)?)
            }
        };
    }
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    if args.encode {
        let device = device.as_ref().expect("a GPU backend has a device");
        let encoder = CudaEncoder::new(
            format!("encoder {}", s.index),
            device,
            CudaEncoderOptions {
                codec: CudaCodec::H264,
                input_format: CudaFrameFormat::Nv12,
                width: s.size.0,
                height: s.size.1,
                frame_rate: s.rate,
                bit_rate: 8_000_000,
                gop_size: 60,
                max_b_frames: None,
            },
        )?;
        branch = branch.queue(format!("drawn {}", s.index), 8).pipe(encoder);
    }
    #[cfg(target_os = "macos")]
    if args.encode {
        let device = device.as_ref().expect("a GPU backend has a device");
        let encoder = VideoToolboxEncoder::new(
            format!("encoder {}", s.index),
            device,
            VideoToolboxEncoderOptions {
                codec: VideoToolboxCodec::H264,
                format: VideoToolboxFrameFormat::Nv12,
                width: s.size.0,
                height: s.size.1,
                frame_rate: s.rate,
                bit_rate: 8_000_000,
                gop_size: 60,
                max_b_frames: None,
            },
        )?;
        branch = branch.queue(format!("drawn {}", s.index), 8).pipe(encoder);
    }
    let _ = (device, s.size, s.rate);
    Ok(branch)
}

fn run() -> Result<()> {
    let _log_guard = media_pp::log::init(
        env!("CARGO_PKG_NAME"),
        "logs",
        media_pp::log::Level::Warn,
        7,
    )?;
    let args = args();
    let meter = Arc::new(Meter {
        pictures: AtomicUsize::new(0),
        objects: AtomicUsize::new(0),
        warmup: args.warmup,
        warm: Mutex::new(None),
    });

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    let device: Device = match args.backend {
        Backend::Cpu => None,
        Backend::Cuda | Backend::TensorRt => Some(CudaDevice::new()?),
    };
    #[cfg(target_os = "macos")]
    let device: Device = match args.backend {
        Backend::Cpu => None,
        Backend::Metal => Some(VideoToolboxDevice::new()?),
    };
    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    let device: Device = None;

    let started = Instant::now();
    let mut pipelines = Vec::new();
    // Where the run is over: the mux's pipeline once every stream has gone
    // through it, or every stream's own.
    let mut finishing = Vec::new();

    let mut sources = Vec::new();
    for index in 0..args.streams {
        let (source, _) = FileDemuxer::open(format!("file {index}"), &args.video)?;
        let stream = source.best(media::Type::Video)?;
        let info = Stream {
            index,
            params: stream.parameters.clone(),
            size: stream.size().expect("a video stream says its size"),
            rate: stream.frame_rate.unwrap_or(Rational::new(30, 1)),
        };
        sources.push((source, stream.index, info));
    }

    if args.batch == 0 {
        for (source, stream_index, s) in sources {
            let (pipeline, ()) =
                Pipeline::new(format!("stream {}", s.index), source, |source, ctx| {
                    let branch = decoder(ctx.branch(), &args, &device, &s)?;
                    let branch = detector(branch, &args, &device, format!("detector {}", s.index))?;
                    let branch = analysis(branch, &args, &device, s.index.to_string())?;
                    let branch = output(branch, &args, &device, &s)?;
                    let branch = meter.count(branch, format!("count {}", s.index))?;
                    ctx.attach(source, stream_index, branch)?;
                    Ok(())
                })?;
            finishing.push(Arc::clone(&pipeline));
            pipelines.push(pipeline);
        }
    } else {
        let (mux, handle) = StreamMux::new(
            "mux",
            StreamMuxOptions {
                mode: RenderMode::Offline { end: None },
                max_batch: args.batch,
                ..StreamMuxOptions::default()
            },
        )?;
        let (batched, demux) = Pipeline::new("batched", mux, |source, ctx| {
            let (demuxed, demux) = handle.demux(ctx, "demux")?;
            let branch = detector(ctx.branch(), &args, &device, String::from("detector"))?;
            let branch = analysis(branch, &args, &device, String::from("batched"))?;
            ctx.attach(source, 0, branch.to_branch(demuxed)?)?;
            Ok(demux)
        })?;
        for (source, stream_index, s) in sources {
            let (input, id) = handle.add_source(format!("stream {}", s.index))?;
            let branch = output(demux.branch()?, &args, &device, &s)?;
            demux.attach(id, meter.count(branch, format!("count {}", s.index))?)?;
            let (feed, ()) =
                Pipeline::new(format!("stream {}", s.index), source, |source, ctx| {
                    let branch = decoder(ctx.branch(), &args, &device, &s)?.to(input)?;
                    ctx.attach(source, stream_index, branch)?;
                    Ok(())
                })?;
            pipelines.push(feed);
        }
        finishing.push(Arc::clone(&batched));
        pipelines.insert(0, batched);
    }
    let ready = started.elapsed();

    let running = Instant::now();
    for pipeline in &pipelines {
        pipeline.run()?;
    }
    let enough = args.pictures.saturating_mul(args.streams);
    let mut finished = 0;
    let mut failed = false;
    'watch: loop {
        if meter.pictures.load(Ordering::Relaxed) >= enough {
            break;
        }
        for pipeline in &pipelines {
            while let Some(event) = pipeline.bus().try_recv() {
                match event {
                    BusEvent::Finished if finishing.iter().any(|p| Arc::ptr_eq(p, pipeline)) => {
                        finished += 1;
                        if finished == finishing.len() {
                            break 'watch;
                        }
                    }
                    BusEvent::Error { .. } => {
                        eprintln!("{event}");
                        failed = true;
                        break 'watch;
                    }
                    _ => {}
                }
            }
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    let (end, cpu_end) = (Instant::now(), cpu_seconds());
    for pipeline in &pipelines {
        pipeline.stop();
    }
    if failed {
        std::process::exit(1);
    }

    let n = meter.pictures.load(Ordering::Relaxed);
    let warm = *meter.warm.lock().unwrap();
    let (counted, seconds, cpu) = match warm {
        Some((at, cpu_at)) if n > args.warmup => {
            let seconds = (end - at).as_secs_f64();
            let cpu = cpu_at.zip(cpu_end).map(|(a, b)| (b - a) / seconds);
            (n - args.warmup, seconds, cpu)
        }
        // Too short to leave the warm-up out.
        _ => (n, (end - running).as_secs_f64(), None),
    };
    let model = args.model.as_deref().map_or("-", |path| {
        std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(path)
    });
    let track = match args.track {
        Track::Off => "off",
        Track::Motion => "motion",
        // Metal follows VideoToolbox pictures on the GPU, with no build of
        // its own to ask for it.
        Track::Visual if cfg!(target_os = "macos") && args.backend != Backend::Cpu => "visual-gpu",
        Track::Visual if cfg!(feature = "gpu-dcf") && args.backend != Backend::Cpu => "visual-gpu",
        Track::Visual => "visual-cpu",
    };
    let video = std::path::Path::new(&args.video)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(&args.video);
    println!(
        "RESULT label={} decode={} backend={} precision={} model={model} video={video} \
         streams={} batch={} interval={} track={track} classify={} overlay={} encode={} \
         pictures={counted} seconds={seconds:.3} fps={:.1} cpu={} objects={:.2} found={} \
         ready={:.1}",
        args.label,
        match args.backend {
            Backend::Cpu => "sw",
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            _ => "nvdec",
            #[cfg(target_os = "macos")]
            Backend::Metal => "videotoolbox",
        },
        if args.model.is_some() {
            args.backend.name()
        } else {
            "-"
        },
        match args.backend {
            _ if args.model.is_none() => "-",
            Backend::Cpu => "fp32",
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            _ if args.fp32 => "fp32",
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            Backend::Cuda => "fp32",
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            Backend::TensorRt => "fp16",
            // Core ML picks its own precision; what it is told is where
            // it may run.
            #[cfg(target_os = "macos")]
            Backend::Metal => args.units.as_str(),
        },
        args.streams,
        args.batch,
        args.interval,
        args.classifier.as_ref().map_or_else(
            || String::from("off"),
            |_| format!("every-{}", args.reclassify)
        ),
        args.overlay,
        args.encode,
        counted as f64 / seconds,
        cpu.map_or_else(|| String::from("-"), |c| format!("{c:.2}")),
        meter.objects.load(Ordering::Relaxed) as f64 / n.max(1) as f64,
        meter.objects.load(Ordering::Relaxed),
        ready.as_secs_f64(),
    );
    Ok(())
}
