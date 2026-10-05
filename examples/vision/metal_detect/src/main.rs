//! FileDemuxer -> VideoToolboxDecoder -> Queue -> MetalOrtDetector ->
//! AppSink: finds objects in a file's pictures with the pictures fitted on
//! the GPU — VideoToolbox decodes, a Metal kernel fits each picture into the
//! model's input where it is, Core ML runs the model on the GPU or the
//! Neural Engine — and prints what each picture carries on, then how fast it
//! went. The macOS counterpart of `cuda_detect`.
//!
//! With `--out`, a Tee after the detector also draws what was found onto
//! each picture and records it, still on the GPU:
//! `Tee -> MetalDetectionOverlay -> Queue -> VideoToolboxEncoder ->
//! FileMuxer`. The overlay draws on copies, so the branch printing beside it
//! sees the pictures as the detector handed them on.
//!
//! The model is an Ultralytics YOLO ONNX export — YOLOv8 and YOLO11, or
//! YOLOv10 and YOLO26 — of the stock weights: the boxes are named with
//! COCO's 80 classes, whatever the model says. Built with `ort-coreml`, on an Apple silicon Mac. The
//! file's video has to be one VideoToolbox decodes to NV12 — 8-bit H.264 or
//! HEVC, say: a 10-bit one is refused as the pipeline is wired, the decoder's
//! P010 being a layout the detector does not take. `--pictures` stops it
//! after about that many.
//!
//!     cargo run --release -p metal_detect -- path/to/model.onnx path/to/video.mp4 \
//!         [--out boxes.mp4] [--pictures N]

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("{} example only supports macOS", env!("CARGO_PKG_NAME"));
}

#[cfg(target_os = "macos")]
fn main() -> impl std::process::Termination {
    example::run()
}

#[cfg(target_os = "macos")]
mod example {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
        mpsc::RecvTimeoutError,
    };
    use std::time::{Duration, Instant};

    use media_pp::{
        Result,
        buffer::MediaBuffer,
        bus::BusEvent,
        elements::{
            AppSink, COCO_CLASS_LABELS, DetectionOverlayOptions, Detections, FileDemuxer,
            FileMuxer, LabelStyle, MetalDetectionOverlay, MetalOrtDetector, OrtDetectorOptions,
            VideoToolboxCodec, VideoToolboxDecoder, VideoToolboxDevice, VideoToolboxEncoder,
            VideoToolboxEncoderOptions, VideoToolboxFrameFormat,
        },
        ffmpeg::{Rational, media},
        pipeline::Pipeline,
    };

    /// Where the labels' font is looked for; without one, boxes alone.
    const FONTS: [&str; 2] = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ];

    struct Args {
        model: String,
        video: String,
        out: Option<String>,
        pictures: usize,
    }

    fn args() -> Args {
        let usage = || -> ! {
            eprintln!(
                "usage: metal_detect <model.onnx> <video.mp4> [--out boxes.mp4] [--pictures N]"
            );
            std::process::exit(1);
        };
        let mut positional = Vec::new();
        let (mut out, mut pictures) = (None, usize::MAX);
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--out" => out = Some(args.next().unwrap_or_else(|| usage())),
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
        Args {
            model,
            video,
            out,
            pictures,
        }
    }

    pub(super) fn run() -> Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;
        let Args {
            model,
            video,
            out,
            pictures: limit,
        } = args();

        let device = VideoToolboxDevice::new()?;
        let started = Instant::now();
        // COCO's class names, which stock Ultralytics weights are trained
        // on, given rather than read from the model: an export that lost
        // them would otherwise put numbers on the boxes.
        let detector = MetalOrtDetector::new(
            "detector",
            &model,
            OrtDetectorOptions {
                labels: Some(COCO_CLASS_LABELS.map(String::from).to_vec()),
                ..OrtDetectorOptions::default()
            },
        )?;
        println!("detector ready in {:.1?}", started.elapsed());

        let (source, _) = FileDemuxer::open("demux", &video)?;
        let stream = source.best(media::Type::Video)?;
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
                        let label = found.label(item).unwrap_or("?");
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

        // The recording, if asked for: an overlay, and the media engine's
        // H.264 at the file's own size and rate.
        let recording = match &out {
            Some(path) => {
                let font = FONTS.iter().find_map(|path| std::fs::read(path).ok());
                if font.is_none() {
                    println!("no font found: boxes without labels");
                }
                let overlay = MetalDetectionOverlay::new(
                    "overlay",
                    &device,
                    DetectionOverlayOptions {
                        line_width: 4,
                        labels: font.map(|font| LabelStyle {
                            size: 22.0,
                            ..LabelStyle::new(font)
                        }),
                        ..DetectionOverlayOptions::default()
                    },
                )?;
                let (width, height) = stream.size().expect("a video stream says its size");
                let encoder = VideoToolboxEncoder::new(
                    "encoder",
                    &device,
                    VideoToolboxEncoderOptions {
                        codec: VideoToolboxCodec::H264,
                        format: VideoToolboxFrameFormat::Nv12,
                        width,
                        height,
                        frame_rate: stream.frame_rate.unwrap_or(Rational::new(30, 1)),
                        bit_rate: 8_000_000,
                        gop_size: 60,
                        max_b_frames: None,
                    },
                )?;
                let mut muxer = FileMuxer::create(path)?;
                let track = muxer.add_stream("video", &encoder)?;
                let muxer_sink = muxer.open()?.take(track)?;
                Some((overlay, encoder, muxer_sink))
            }
            None => None,
        };

        let params = stream.parameters.clone();
        let (pipeline, ()) = Pipeline::new("metal-detect", source, |source, ctx| {
            let decoder = VideoToolboxDecoder::new("decoder", params, &device)?;
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
        // Woken every 50 ms as well as by an event, since a run that goes
        // well posts none until it finishes, and `--pictures` is counted
        // here.
        loop {
            if seen.load(Ordering::Relaxed) >= limit {
                pipeline.stop();
                break;
            }
            match pipeline.bus().recv_timeout(Duration::from_millis(50)) {
                Ok(BusEvent::Finished) | Err(RecvTimeoutError::Disconnected) => {
                    pipeline.stop();
                    break;
                }
                Ok(event @ BusEvent::Error { .. }) => {
                    eprintln!("{event}");
                    pipeline.stop();
                    break;
                }
                Ok(_) | Err(RecvTimeoutError::Timeout) => {}
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
