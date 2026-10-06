//! FileDemuxer -> CudaDecoder -> Queue -> CudaOrtDetector -> AppSink: finds
//! objects in a file's pictures without them leaving the GPU — NVDEC
//! decodes, a kernel fits each picture into the model's input in device
//! memory, TensorRT runs the model — and prints what each picture carries
//! on, then how fast it went.
//!
//! With `--out`, a Tee after the detector also draws what was found onto
//! each picture and records it, still on the GPU:
//! `Tee -> CudaDetectionOverlay -> Queue -> CudaEncoder -> FileMuxer`. The
//! overlay draws on copies, so the branch printing beside it sees the
//! pictures as the detector handed them on. `--hide person=blur` hides a
//! class there instead of boxing it — `mosaic`, `blur` or `fill`, a mosaic
//! where none is said — and may be given once for each class.
//!
//! A phone's portrait recording, stored on its side, is looked at and
//! labelled the right way up, and recorded saying it is turned, as the
//! file does.
//!
//! The model is an Ultralytics YOLO ONNX export — YOLOv8 and YOLO11, or
//! YOLOv10 and YOLO26. The first run builds a TensorRT engine for the model
//! and this GPU, which takes minutes; later runs load it from the cache in
//! under a second. Built with `ort-tensorrt`, which links CUDA 13, cuDNN 9
//! and TensorRT 10 into it: on Linux building and running need them where
//! the linker and the loader find them — `LD_LIBRARY_PATH` pointing at
//! them; on Windows building needs none of them, and running their DLLs on
//! `PATH`.
//!
//!     cargo run --release -p cuda_detect -- path/to/model.onnx path/to/video.mp4 \
//!         [--out boxes.mp4 [--hide CLASS[=mosaic|blur|fill]]...] [--pictures N]

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn main() {
    eprintln!(
        "{} example only supports Linux and Windows",
        env!("CARGO_PKG_NAME")
    );
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn main() -> impl std::process::Termination {
    example::run()
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
mod example {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use std::time::Instant;

    use media_pp::{
        Result,
        buffer::MediaBuffer,
        bus::BusEvent,
        color::Color,
        elements::{
            AppSink, BoxStyle, COCO_CLASS_LABELS, ClassRule, CudaCodec, CudaDecoder,
            CudaDetectionOverlay, CudaDevice, CudaEncoder, CudaEncoderOptions, CudaFrameFormat,
            CudaOrtDetector, CudaOrtDetectorOptions, DetectionOverlayOptions, Detections,
            FileDemuxer, FileMuxer, LabelStyle, RedactStyle, TrackFormat, Treatment,
        },
        ffmpeg::{Rational, media},
        pipeline::Pipeline,
    };

    /// Where the labels' font is looked for; without one, boxes alone.
    const FONTS: [&str; 4] = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ];

    struct Args {
        model: String,
        video: String,
        out: Option<String>,
        hide: Vec<ClassRule>,
        pictures: usize,
    }

    /// `CLASS[=mosaic|blur|fill]` from `--hide`: that class's detections
    /// hidden that way — a mosaic where no way is said — and drawn no box.
    fn hide_rule(spec: &str) -> Option<ClassRule> {
        let (class, style) = spec.split_once('=').unwrap_or((spec, "mosaic"));
        let style = match style {
            "mosaic" => RedactStyle::mosaic(),
            "blur" => RedactStyle::blur(),
            "fill" => RedactStyle::Fill(Color::BLACK),
            _ => return None,
        };
        (!class.is_empty()).then(|| ClassRule::new(class, Treatment::hidden(style)))
    }

    fn args() -> Args {
        let usage = || -> ! {
            eprintln!(
                "usage: cuda_detect <model.onnx> <video.mp4> \
                 [--out boxes.mp4 [--hide CLASS[=mosaic|blur|fill]]...] [--pictures N]"
            );
            std::process::exit(1);
        };
        let mut positional = Vec::new();
        let (mut out, mut pictures) = (None, usize::MAX);
        let mut hide = Vec::new();
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--out" => out = Some(args.next().unwrap_or_else(|| usage())),
                "--hide" => hide.push(
                    args.next()
                        .as_deref()
                        .and_then(hide_rule)
                        .unwrap_or_else(|| usage()),
                ),
                "--pictures" => {
                    pictures = args
                        .next()
                        .and_then(|n| n.parse().ok())
                        .unwrap_or_else(|| usage())
                }
                _ => positional.push(arg),
            }
        }
        let [model, video] = <[String; 2]>::try_from(positional).unwrap_or_else(|_| usage());
        if out.is_none() && !hide.is_empty() {
            usage();
        }
        Args {
            model,
            video,
            out,
            hide,
            pictures,
        }
    }

    pub(super) fn run() -> Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Info,
            7,
        )?;
        let Args {
            model,
            video,
            out,
            hide,
            pictures: limit,
        } = args();

        let device = CudaDevice::new()?;
        let started = Instant::now();
        let detector = CudaOrtDetector::new(
            "detector",
            &device,
            &model,
            CudaOrtDetectorOptions::default(),
        )?;
        println!("detector ready in {:.1?}", started.elapsed());

        let (source, _) = FileDemuxer::open("demux", &video)?;
        let stream = source.best(media::Type::Video)?;
        // A phone's portrait recording is stored on its side and says so;
        // what is recorded says so too, so it is shown the same way up.
        let orientation = stream.orientation()?;
        let seen = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&seen);
        let sink = AppSink::new("print", move |buf| {
            let n = counted.fetch_add(1, Ordering::Relaxed) + 1;
            let MediaBuffer::Video(frame) = &buf else {
                return Ok(());
            };
            let Some(found) = buf
                .metadata()
                .and_then(|metadata| metadata.get::<Detections>())
            else {
                return Ok(());
            };
            if n % 30 == 1 {
                let names: Vec<String> = found
                    .items
                    .iter()
                    .map(|item| {
                        let label = found
                            .label(item)
                            .or_else(|| COCO_CLASS_LABELS.get(item.class_id).copied())
                            .unwrap_or("?");
                        format!("{label} {:.0}%", item.score * 100.0)
                    })
                    .collect();
                println!(
                    "picture {n} ({}x{}, pts {:?}): {}",
                    frame.width(),
                    frame.height(),
                    frame.pts(),
                    names.join(", ")
                );
            }
            Ok(())
        });

        // The recording, if asked for: an overlay, and NVENC at the file's
        // own size and rate.
        let recording = match &out {
            Some(path) => {
                let font = FONTS.iter().find_map(|path| std::fs::read(path).ok());
                if font.is_none() {
                    println!("no font found: boxes without labels");
                }
                let overlay = CudaDetectionOverlay::new(
                    "overlay",
                    &device,
                    DetectionOverlayOptions {
                        others: Treatment::boxes(BoxStyle {
                            line_width: 4,
                            label: font.is_some().then(|| LabelStyle::new(22.0)),
                            ..BoxStyle::default()
                        }),
                        rules: hide,
                        font,
                        ..DetectionOverlayOptions::default()
                    },
                )?;
                let (width, height) = stream.size().expect("a video stream says its size");
                let encoder = CudaEncoder::new(
                    "encoder",
                    &device,
                    CudaEncoderOptions {
                        codec: CudaCodec::H264,
                        input_format: CudaFrameFormat::Nv12,
                        width,
                        height,
                        frame_rate: stream.frame_rate.unwrap_or(Rational::new(30, 1)),
                        bit_rate: 8_000_000,
                        gop_size: 60,
                        max_b_frames: None,
                    },
                )?;
                let mut muxer = FileMuxer::create(path)?;
                let track = muxer.add_stream(
                    "video",
                    TrackFormat::from(&encoder).with_orientation(orientation),
                )?;
                let muxer_sink = muxer.open()?.take(track)?;
                Some((overlay, encoder, muxer_sink))
            }
            None => None,
        };

        let params = stream.parameters.clone();
        let (pipeline, ()) = Pipeline::new("cuda-detect", source, |source, ctx| {
            let decoder = CudaDecoder::new("decoder", params, &device, 16)?;
            let detecting = ctx
                .branch()
                .pipe(decoder)
                .queue("pictures", 8)
                .pipe(detector);
            let branch = match recording {
                None => detecting.to(sink)?,
                Some((overlay, encoder, muxer_sink)) => {
                    let print = ctx.branch().to(sink)?;
                    let record = ctx
                        .branch()
                        .pipe(overlay)
                        .queue("drawn", 8)
                        .pipe(encoder)
                        .to(muxer_sink)?;
                    let tee = ctx.tee("tee").branch(print).branch(record).build()?;
                    detecting.to_branch(tee)?
                }
            };
            ctx.attach(source, stream.index, branch)?;
            Ok(())
        })?;

        let running = Instant::now();
        pipeline.run()?;
        for event in pipeline.bus().iter() {
            if seen.load(Ordering::Relaxed) >= limit {
                pipeline.stop();
                break;
            }
            match event {
                BusEvent::Finished => {
                    pipeline.stop();
                    break;
                }
                BusEvent::Error { .. } => {
                    eprintln!("{event}");
                    pipeline.stop();
                    break;
                }
                _ => {}
            }
        }
        let n = seen.load(Ordering::Relaxed);
        let elapsed = running.elapsed();
        println!(
            "{n} pictures in {elapsed:.1?}: {:.1} pictures a second, decode and detection together",
            n as f64 / elapsed.as_secs_f64()
        );
        if let Some(path) = out {
            println!("recorded with boxes: {path}");
        }
        Ok(())
    }
}
