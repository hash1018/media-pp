//! FileDemuxer -> VideoToolboxDecoder -> Queue -> MetalOrtDetector ->
//! AppSink: finds objects in a file's pictures with the pictures fitted on
//! the GPU — VideoToolbox decodes, a Metal kernel fits each picture into the
//! model's input where it is, Core ML runs the model on the GPU or the
//! Neural Engine — and prints what each picture carries on, then how fast it
//! went. The macOS counterpart of `cuda_detect`.
//!
//! With `--track`, an `ObjectTracker` after the detector numbers each
//! object, the same on every picture it is followed through, and with
//! `--interval N` the detector lets N pictures by between two it looks at
//! while the tracker puts where it expects each object on them —
//! DeepStream's `interval`, as `cuda_track` measures it. `--confirm 1`
//! numbers a new object when it is first seen rather than on its second
//! sighting, which a long interval needs. `--visual` follows each object by
//! how it looks as well as by its motion — a correlation filter on the
//! pixels around it, read from the VideoToolbox picture where it is
//! (`TrackerOptions::visual`). Each implies `--track`.
//!
//! `--line X1,Y1,X2,Y2`, in fractions of the picture and as many times as
//! there are lines, puts an `ObjectAnalytics` after the tracker that counts
//! the objects crossing each — forward being from its left to its right as
//! seen from its start, so a line drawn left to right counts what moves down
//! — and prints each crossing as it happens and the totals at the end. It
//! implies `--track`, since a crossing is an object followed across.
//!
//! With `--out`, a Tee at the end also draws what was found onto each
//! picture and records it, still on the GPU:
//! `Tee -> MetalDetectionOverlay -> Queue -> VideoToolboxEncoder ->
//! FileMuxer`, each object tracked in a colour of its own and labelled with
//! its number. The overlay draws on copies, so the branch printing beside it
//! sees the pictures as they were handed on.
//!
//! The model is an Ultralytics YOLO ONNX export — YOLOv8 and YOLO11, or
//! YOLOv10 and YOLO26 — of the stock weights: the boxes are named with
//! COCO's 80 classes, whatever the model says. Built with `ort-coreml`, on
//! an Apple silicon Mac. The file's video has to be one VideoToolbox decodes
//! to NV12 — 8-bit H.264 or HEVC, say: a 10-bit one is refused as the
//! pipeline is wired, the decoder's P010 being a layout the detector does
//! not take. `--pictures` stops it after about that many.
//!
//!     cargo run --release -p metal_detect -- path/to/model.onnx path/to/video.mp4 \
//!         [--track] [--interval N] [--confirm N] [--visual] [--line X1,Y1,X2,Y2]... \
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
    use std::collections::HashSet;
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
        mpsc::RecvTimeoutError,
    };
    use std::time::{Duration, Instant};

    use media_pp::{
        Result,
        buffer::MediaBuffer,
        bus::BusEvent,
        elements::{
            Analytics, AnalyticsOptions, AppSink, BoxColors, COCO_CLASS_LABELS,
            DetectionOverlayOptions, Detections, FileDemuxer, FileMuxer, LabelStyle, Line,
            LineCount, MetalDetectionOverlay, MetalOrtDetector, ObjectAnalytics, ObjectTracker,
            OrtDetectorOptions, TrackerOptions, VideoToolboxCodec, VideoToolboxDecoder,
            VideoToolboxDevice, VideoToolboxEncoder, VideoToolboxEncoderOptions,
            VideoToolboxFrameFormat,
        },
        ffmpeg::{Rational, media},
        pipeline::Pipeline,
    };

    /// Where the labels' font is looked for; without one, boxes alone.
    const FONTS: [&str; 2] = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ];

    /// What is printed and drawn: Ultralytics' own threshold. A tracker is
    /// handed less confident detections too, to keep following an object
    /// partly hidden, but they are not worth showing.
    const SHOWN: f32 = 0.25;

    struct Args {
        model: String,
        video: String,
        track: bool,
        interval: u32,
        confirm: u32,
        visual: bool,
        lines: Vec<Line>,
        out: Option<String>,
        pictures: usize,
    }

    fn args() -> Args {
        let usage = || -> ! {
            eprintln!(
                "usage: metal_detect <model.onnx> <video.mp4> [--track] [--interval N] \
                 [--confirm N] [--visual] [--line X1,Y1,X2,Y2]... [--out boxes.mp4] \
                 [--pictures N]"
            );
            std::process::exit(1);
        };
        let mut positional = Vec::new();
        let (mut track, mut interval, mut out, mut pictures) = (false, 0, None, usize::MAX);
        let mut confirm = TrackerOptions::default().confirm_after;
        let mut lines = Vec::new();
        let mut visual = false;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--track" => track = true,
                "--visual" => {
                    track = true;
                    visual = true;
                }
                "--interval" => {
                    track = true;
                    interval = args
                        .next()
                        .and_then(|n| n.parse().ok())
                        .unwrap_or_else(|| usage())
                }
                "--confirm" => {
                    track = true;
                    confirm = args
                        .next()
                        .and_then(|n| n.parse().ok())
                        .unwrap_or_else(|| usage())
                }
                "--line" => {
                    track = true;
                    let ends: Vec<f32> = args
                        .next()
                        .unwrap_or_else(|| usage())
                        .split(',')
                        .map(|n| n.parse().unwrap_or_else(|_| usage()))
                        .collect();
                    let [x1, y1, x2, y2] = <[f32; 4]>::try_from(ends).unwrap_or_else(|_| usage());
                    lines.push(Line::new(
                        format!("line {}", lines.len() + 1),
                        (x1, y1),
                        (x2, y2),
                    ));
                }
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
            track,
            interval,
            confirm,
            visual,
            lines,
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
            track,
            interval,
            confirm,
            visual,
            lines,
            out,
            pictures: limit,
        } = args();
        let analytics = (!lines.is_empty())
            .then(|| {
                ObjectAnalytics::new(
                    "analytics",
                    AnalyticsOptions {
                        lines,
                        ..AnalyticsOptions::default()
                    },
                )
            })
            .transpose()?;

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
                // The tracker matches unconfident detections too, so it is
                // handed them.
                conf_threshold: if track {
                    TrackerOptions::default().low_score
                } else {
                    SHOWN
                },
                interval,
                ..OrtDetectorOptions::default()
            },
        )?;
        println!("detector ready in {:.1?}", started.elapsed());

        let (source, _) = FileDemuxer::open("demux", &video)?;
        let stream = source.best(media::Type::Video)?;
        let seen = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&seen);
        let followed: Arc<Mutex<HashSet<u64>>> = Arc::default();
        let numbered = Arc::clone(&followed);
        let totals: Arc<Mutex<Vec<LineCount>>> = Arc::default();
        let counting = Arc::clone(&totals);
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
            numbered
                .lock()
                .unwrap()
                .extend(found.items.iter().filter_map(|item| item.track_id));
            if let Some(analytics) = buf
                .metadata()
                .and_then(|metadata| metadata.get::<Analytics>())
            {
                for line in &analytics.lines {
                    for crossing in &line.crossed {
                        let label = found.labels.get(crossing.class_id).map_or("?", |l| &**l);
                        println!(
                            "picture {n}: {label} #{} crossed {} {}",
                            crossing.track_id,
                            line.name,
                            if crossing.forward {
                                "forward"
                            } else {
                                "backward"
                            }
                        );
                    }
                }
                *counting.lock().unwrap() = analytics.lines.clone();
            }
            if n % 30 == 1 {
                let names: Vec<String> = found
                    .items
                    .iter()
                    .filter(|item| item.score >= SHOWN)
                    .map(|item| {
                        let label = found.label(item).unwrap_or("?");
                        match item.track_id {
                            Some(id) => format!("{label} #{id} {:.0}%", item.score * 100.0),
                            None => format!("{label} {:.0}%", item.score * 100.0),
                        }
                    })
                    .collect();
                println!(
                    "picture {n} ({}x{}, pts {:?}){}: {}",
                    frame.width(),
                    frame.height(),
                    frame.pts(),
                    if found.predicted { ", expected" } else { "" },
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
                        min_score: SHOWN,
                        colors: if track {
                            BoxColors::ByTrack
                        } else {
                            BoxColors::ByClass
                        },
                        labels: font.map(|font| LabelStyle {
                            size: 22.0,
                            ..LabelStyle::new(font)
                        }),
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
            let mut detecting = ctx
                .branch()
                .pipe(decoder)
                .queue("pictures", 8)
                .pipe(detector);
            if track {
                detecting = detecting.pipe(ObjectTracker::new(
                    "tracker",
                    TrackerOptions {
                        confirm_after: confirm,
                        visual,
                        ..TrackerOptions::default()
                    },
                ));
            }
            if let Some(analytics) = analytics {
                detecting = detecting.pipe(analytics);
            }
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
        if track {
            println!(
                "{} objects followed, interval {interval}",
                followed.lock().unwrap().len()
            );
        }
        for line in totals.lock().unwrap().iter() {
            println!(
                "{}: {} forward, {} backward",
                line.name, line.forward, line.backward
            );
        }
        if let Some(path) = out {
            println!("recorded with boxes: {path}");
        }
        Ok(())
    }
}
