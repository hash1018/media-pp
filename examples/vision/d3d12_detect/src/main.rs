//! FileDemuxer -> D3d12Decoder -> Queue -> D3d12OrtDetector -> ObjectTracker
//! -> AppSink: objects found and followed from picture to picture on any
//! vendor's GPU, through D3D12 and ONNX Runtime's DirectML, without the
//! pictures leaving the GPU. Each object keeps its number; the detector looks
//! at one picture in `--interval N` + 1 and the tracker puts where it expects
//! each object on the rest. `--visual` follows each object by how it looks
//! as well as by its motion, on the GPU.
//!
//! `--out tracked.mp4` records it as well: a `Tee` after the tracker, and
//! `-> D3d12DetectionOverlay -> Queue -> D3d12VideoEncoder -> FileMuxer`,
//! each object in a colour of its own and labelled with its number, encoded
//! to HEVC on the GPU. `--hide person=blur` hides a class there instead —
//! `mosaic`, `blur` or `fill`, a mosaic where none is said, the ellipse
//! inside each box with `,ellipse` after it — and may be given once for each
//! class.
//!
//! The model must be exported at ONNX opset 20 or lower: DirectML hands the
//! operators of a later one to the CPU, many times slower, and the detector
//! warns of it in the log.
//!
//!     cargo run --release -p d3d12_detect -- model.onnx video.mp4 [--interval N] [--visual] \
//!         [--out tracked.mp4 [--hide CLASS[=mosaic|blur|fill][,ellipse]]...]

#[cfg(not(target_os = "windows"))]
fn main() {
    eprintln!("{} example only supports Windows", env!("CARGO_PKG_NAME"));
}

#[cfg(target_os = "windows")]
fn main() -> impl std::process::Termination {
    example::run()
}

#[cfg(target_os = "windows")]
mod example {
    use std::collections::HashSet;
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    use media_pp::{
        Result,
        bus::BusEvent,
        color::Color,
        elements::{
            AppSink, BoxColors, BoxStyle, COCO_CLASS_LABELS, ClassRule, D3d12Decoder,
            D3d12DetectionOverlay, D3d12Gpu, D3d12OrtDetector, D3d12OrtDetectorOptions,
            D3d12VideoCodec, D3d12VideoEncoder, D3d12VideoEncoderOptions, DetectionOverlayOptions,
            Detections, FileDemuxer, FileMuxer, HideShape, Hiding, LabelStyle, Mp4Layout,
            ObjectTracker, OrtDetectorOptions, RedactStyle, TrackFormat, TrackerOptions, Treatment,
        },
        ffmpeg::{Rational, media},
        pipeline::Pipeline,
    };

    /// The labels' font; without it, boxes alone.
    const FONT: &str = "C:/Windows/Fonts/arial.ttf";

    struct Args {
        model: String,
        video: String,
        interval: u32,
        visual: bool,
        out: Option<String>,
        hide: Vec<ClassRule>,
    }

    /// `CLASS[=mosaic|blur|fill][,ellipse]` from `--hide`: that class's
    /// detections hidden that way — a mosaic where no way is said, the whole
    /// box unless `,ellipse` — and drawn no box.
    fn hide_rule(spec: &str) -> Option<ClassRule> {
        let (spec, shape) = match spec.strip_suffix(",ellipse") {
            Some(spec) => (spec, HideShape::Ellipse),
            None => (spec, HideShape::Rectangle),
        };
        let (class, style) = spec.split_once('=').unwrap_or((spec, "mosaic"));
        let style = match style {
            "mosaic" => RedactStyle::mosaic(),
            "blur" => RedactStyle::blur(),
            "fill" => RedactStyle::Fill(Color::BLACK),
            _ => return None,
        };
        let hiding = Hiding {
            shape,
            ..Hiding::new(style)
        };
        (!class.is_empty()).then(|| {
            ClassRule::new(
                class,
                Treatment {
                    draw: None,
                    hide: Some(hiding),
                    ..Treatment::none()
                },
            )
        })
    }

    fn args() -> Args {
        let usage = || -> ! {
            eprintln!(
                "usage: d3d12_detect <model.onnx> <video.mp4> [--interval N] [--visual]\n\
                 \x20        [--out tracked.mp4 [--hide CLASS[=mosaic|blur|fill][,ellipse]]...]"
            );
            std::process::exit(1);
        };
        let mut positional = Vec::new();
        let (mut interval, mut visual, mut out, mut hide) = (0, false, None, Vec::new());
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--interval" => {
                    interval = args
                        .next()
                        .and_then(|n| n.parse().ok())
                        .unwrap_or_else(|| usage())
                }
                "--visual" => visual = true,
                "--out" => out = Some(args.next().unwrap_or_else(|| usage())),
                "--hide" => hide.push(
                    args.next()
                        .as_deref()
                        .and_then(hide_rule)
                        .unwrap_or_else(|| usage()),
                ),
                _ if arg.starts_with("--") => usage(),
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
            interval,
            visual,
            out,
            hide,
        }
    }

    pub(super) fn run() -> Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;
        let args = args();
        // One device for every D3D12 element: the decoder's pictures are
        // what the detector, the overlay and the encoder read.
        let gpu = D3d12Gpu::new()?;

        let (source, _) = FileDemuxer::open("demux", &args.video)?;
        let stream = source.best(media::Type::Video)?;
        // A phone's portrait recording is stored on its side and says so;
        // what is recorded says so too, so it is shown the same way up.
        let orientation = stream.orientation()?;

        let detector = D3d12OrtDetector::new(
            "detector",
            &gpu,
            &args.model,
            D3d12OrtDetectorOptions {
                detector: OrtDetectorOptions {
                    // Stock Ultralytics weights are COCO's, and an export
                    // that lost its names would otherwise put numbers on
                    // the boxes.
                    labels: Some(COCO_CLASS_LABELS.map(String::from).to_vec()),
                    // The tracker matches unconfident detections too.
                    conf_threshold: TrackerOptions::default().low_score,
                    interval: args.interval,
                    ..OrtDetectorOptions::default()
                },
                ..D3d12OrtDetectorOptions::default()
            },
        )?;
        let tracker = ObjectTracker::new(
            "tracker",
            TrackerOptions {
                visual: args.visual,
                ..TrackerOptions::default()
            },
        );

        // Every picture's objects, printed once a second of the video and
        // counted at the end.
        let followed: Arc<Mutex<HashSet<u64>>> = Arc::default();
        let pictures = Arc::new(Mutex::new(0usize));
        let (kept_ids, kept_count) = (Arc::clone(&followed), Arc::clone(&pictures));
        let frame_rate = stream.frame_rate.unwrap_or(Rational::new(30, 1));
        let per_second = (f64::from(frame_rate.numerator()) / f64::from(frame_rate.denominator()))
            .round()
            .max(1.0) as usize;
        let sink = AppSink::new("report", move |buf| {
            let found = buf.metadata().and_then(|m| m.get::<Detections>());
            let mut count = kept_count.lock().unwrap();
            if let Some(found) = found {
                kept_ids
                    .lock()
                    .unwrap()
                    .extend(found.items.iter().filter_map(|item| item.track_id));
                if *count % per_second == 0 {
                    let named: Vec<String> = found
                        .items
                        .iter()
                        .filter_map(|item| {
                            let id = item.track_id?;
                            Some(format!("{} #{id}", found.label(item).unwrap_or("?")))
                        })
                        .collect();
                    println!(
                        "picture {:5}{}: {}",
                        *count,
                        if found.predicted { " (filled in)" } else { "" },
                        named.join(", ")
                    );
                }
            }
            *count += 1;
            Ok(())
        });

        let recording = match &args.out {
            Some(path) => {
                let font = std::fs::read(FONT).ok();
                let overlay = D3d12DetectionOverlay::new(
                    "overlay",
                    &gpu,
                    DetectionOverlayOptions {
                        others: Treatment {
                            min_score: 0.25,
                            ..Treatment::boxes(BoxStyle {
                                line_width: 4,
                                color: BoxColors::ByTrack,
                                label: font.is_some().then(|| LabelStyle::new(22.0)),
                            })
                        },
                        rules: args.hide.clone(),
                        font,
                        ..DetectionOverlayOptions::default()
                    },
                )?;
                let (width, height) = stream.size().expect("a video stream says its size");
                let options = D3d12VideoEncoderOptions {
                    // What FFmpeg 8.0 encodes through D3D12.
                    codec: D3d12VideoCodec::H265,
                    width,
                    height,
                    frame_rate,
                    bit_rate: 8_000_000,
                    gop_size: 60,
                    max_b_frames: None,
                };
                // The file's own colours, written into the recording's
                // headers; nothing is converted.
                let encoder = D3d12VideoEncoder::for_stream("encoder", &gpu, options, &stream)?;
                let mut muxer = FileMuxer::create(path, Mp4Layout::Standard)?;
                let track = muxer.add_stream(
                    "video",
                    TrackFormat::from(&encoder).with_orientation(orientation),
                )?;
                Some((overlay, encoder, muxer.open()?.take(track)?))
            }
            None => None,
        };

        let params = stream.parameters.clone();
        let started = Instant::now();
        let (pipeline, ()) = Pipeline::new("d3d12-detect", source, |source, ctx| {
            let decoder = D3d12Decoder::new("decoder", params, &gpu)?;
            let branch = ctx
                .branch()
                .pipe(decoder)
                .queue("pictures", 8)
                .pipe(detector)
                .pipe(tracker);
            let branch = match recording {
                None => branch.to(sink)?,
                Some((overlay, encoder, muxer_sink)) => {
                    let report = ctx.branch().to(sink)?;
                    let record = ctx
                        .branch()
                        .pipe(overlay)
                        .queue("drawn", 8)
                        .pipe(encoder)
                        .to(muxer_sink)?;
                    let tee = ctx.tee("tee").branch(report).branch(record).build()?;
                    branch.to_branch(tee)?
                }
            };
            ctx.attach(source, stream.index, branch)?;
            Ok(())
        })?;
        pipeline.run()?;
        // A file's source waits at its end in case of a seek: the pipeline
        // has played to its end once every sink has everything.
        for event in pipeline.bus().iter() {
            match event {
                BusEvent::Finished => break,
                BusEvent::Error { .. } => {
                    eprintln!("{event}");
                    break;
                }
                _ => {}
            }
        }
        pipeline.stop();

        let pictures = *pictures.lock().unwrap();
        let elapsed = started.elapsed();
        println!(
            "{pictures} pictures in {elapsed:.1?} ({:.0} a second), interval {}: {} objects followed",
            pictures as f64 / elapsed.as_secs_f64(),
            args.interval,
            followed.lock().unwrap().len()
        );
        if let Some(out) = &args.out {
            println!("recorded: {out}");
        }
        Ok(())
    }
}
