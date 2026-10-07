//! FileDemuxer -> CudaDecoder -> Queue -> CudaOrtDetector -> ObjectTracker
//! -> …: objects found on the GPU and followed from picture to picture,
//! each keeping its number, with the detector looking at only some of the
//! pictures and the tracker filling in the rest — DeepStream's `interval`.
//!
//! `--out tracked.mp4` records it: `-> CudaDetectionOverlay -> Queue ->
//! CudaEncoder -> FileMuxer`, each object in a colour of its own and
//! labelled with its number. `--hide person=blur` hides a class there
//! instead — `mosaic`, `blur` or `fill`, a mosaic where none is said, the
//! ellipse inside each box with `,ellipse` after it — following each
//! object from picture to picture as the tracker does, and may be given
//! once for each class.
//!
//! A phone's portrait recording, stored on its side, is looked at and
//! labelled the right way up, and recorded saying it is turned, as the
//! file does.
//!
//! `--eval 1,2,4,9` measures how good the filled-in pictures are. It runs
//! the file once with the detector on every picture, which is the
//! reference, and then once per interval with the tracker; on each picture
//! the detector let by, it compares the boxes the tracker expected with the
//! reference's confident ones. Beside it, the same for holding the last
//! detected boxes still, which is what filling in without a motion model
//! would give.
//!
//!     cargo run --release -p cuda_track -- model.onnx video.mp4 [--interval N] \
//!         [--out tracked.mp4 [--hide CLASS[=mosaic|blur|fill][,ellipse]]...]
//!     cargo run --release -p cuda_track -- model.onnx video.mp4 --eval 1,2,4,9

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
    use std::collections::HashSet;
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    use media_pp::{
        Result,
        buffer::MediaBuffer,
        bus::BusEvent,
        color::Color,
        elements::{
            AppSink, BoxColors, BoxStyle, COCO_CLASS_LABELS, ClassRule, CudaCodec, CudaDecoder,
            CudaDetectionOverlay, CudaDevice, CudaEncoder, CudaEncoderOptions, CudaFrameFormat,
            CudaOrtClassifier, CudaOrtDetector, CudaOrtDetectorOptions, Detection,
            DetectionOverlayOptions, Detections, FileDemuxer, FileMuxer, HideShape, Hiding,
            InputScale, LabelStyle, ObjectTracker, OrtClassifierOptions, OrtDetectorOptions,
            RedactStyle, TrackFormat, TrackerOptions, Treatment,
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

    /// A reference detection counts from this score: what a viewer would
    /// call an object.
    const CONFIDENT: f32 = 0.5;

    struct Args {
        model: String,
        video: String,
        interval: u32,
        confirm: u32,
        visual: bool,
        classifier: Option<String>,
        classifier_labels: Option<String>,
        classify: Option<Vec<usize>>,
        out: Option<String>,
        hide: Vec<ClassRule>,
        eval: Option<Vec<u32>>,
    }

    /// `CLASS[=mosaic|blur|fill][,ellipse]` from `--hide`: that class's
    /// detections hidden that way — a mosaic where no way is said, the whole
    /// box unless `,ellipse` — and drawn no box.
    fn hide_rule(spec: &str) -> Option<ClassRule> {
        // `,ellipse` last hides the ellipse inside the box, not the box.
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
                "usage: cuda_track <model.onnx> <video.mp4> [--interval N] [--confirm N] [--visual]\n\
                 \x20        [--classifier imagenet.onnx [--classifier-labels classes.txt] [--classify 2,5,7]]\n\
                 \x20        [--out tracked.mp4 [--hide CLASS[=mosaic|blur|fill][,ellipse]]...]\n\
                 \x20      cuda_track <model.onnx> <video.mp4> --eval 1,2,4,9 [--confirm N] [--visual]"
            );
            std::process::exit(1);
        };
        let mut positional = Vec::new();
        let (mut interval, mut out, mut eval) = (0, None, None);
        let mut confirm = TrackerOptions::default().confirm_after;
        let mut visual = false;
        let mut hide = Vec::new();
        let (mut classifier, mut classifier_labels, mut classify) = (None, None, None);
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--interval" => {
                    interval = args
                        .next()
                        .and_then(|n| n.parse().ok())
                        .unwrap_or_else(|| usage())
                }
                "--confirm" => {
                    confirm = args
                        .next()
                        .and_then(|n| n.parse().ok())
                        .unwrap_or_else(|| usage())
                }
                "--visual" => visual = true,
                "--classifier" => classifier = Some(args.next().unwrap_or_else(|| usage())),
                "--classifier-labels" => {
                    classifier_labels = Some(args.next().unwrap_or_else(|| usage()))
                }
                "--classify" => {
                    classify = Some(
                        args.next()
                            .unwrap_or_else(|| usage())
                            .split(',')
                            .map(|n| n.parse().unwrap_or_else(|_| usage()))
                            .collect(),
                    )
                }
                "--out" => out = Some(args.next().unwrap_or_else(|| usage())),
                "--hide" => hide.push(
                    args.next()
                        .as_deref()
                        .and_then(hide_rule)
                        .unwrap_or_else(|| usage()),
                ),
                "--eval" => {
                    eval = Some(
                        args.next()
                            .unwrap_or_else(|| usage())
                            .split(',')
                            .map(|n| n.parse().unwrap_or_else(|_| usage()))
                            .collect(),
                    )
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
            interval,
            confirm,
            visual,
            classifier,
            classifier_labels,
            classify,
            out,
            hide,
            eval,
        }
    }

    pub(super) fn run() -> Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Info,
            7,
        )?;
        let args = args();
        let device = CudaDevice::new()?;
        match &args.eval {
            Some(intervals) => evaluate(&args, &device, intervals),
            None => {
                let started = Instant::now();
                let seen = pass(&args, &device, args.interval, true, args.out.as_deref())?;
                let ids: HashSet<u64> = seen
                    .iter()
                    .flat_map(|(_, found)| found.iter().flat_map(|f| &f.items))
                    .filter_map(|item| item.track_id)
                    .collect();
                println!(
                    "{} pictures in {:.1?}, interval {}: {} objects followed",
                    seen.len(),
                    started.elapsed(),
                    args.interval,
                    ids.len()
                );
                if let Some(out) = &args.out {
                    println!("recorded: {out}");
                }
                Ok(())
            }
        }
    }

    /// Every picture's timestamp and what it carried at the end of the
    /// branch.
    type Seen = Vec<(i64, Option<Detections>)>;

    /// Runs the file through the detector looking at every `interval + 1`th
    /// picture, then the tracker if `track`, then the recording if `out`.
    fn pass(
        args: &Args,
        device: &CudaDevice,
        interval: u32,
        track: bool,
        out: Option<&str>,
    ) -> Result<Seen> {
        let detector = CudaOrtDetector::new(
            "detector",
            device,
            &args.model,
            CudaOrtDetectorOptions {
                detector: OrtDetectorOptions {
                    // Stock Ultralytics weights are COCO's, and an export
                    // that lost its names would otherwise put numbers on the
                    // boxes.
                    labels: Some(COCO_CLASS_LABELS.map(String::from).to_vec()),
                    // The tracker matches unconfident detections too, so it
                    // is handed them; the reference keeps Ultralytics' own.
                    conf_threshold: if track {
                        TrackerOptions::default().low_score
                    } else {
                        0.25
                    },
                    interval,
                    ..OrtDetectorOptions::default()
                },
                ..CudaOrtDetectorOptions::default()
            },
        )?;
        let (source, _) = FileDemuxer::open("demux", &args.video)?;
        let stream = source.best(media::Type::Video)?;
        // A phone's portrait recording is stored on its side and says so;
        // what is recorded says so too, so it is shown the same way up.
        let orientation = stream.orientation()?;

        let seen: Arc<Mutex<Seen>> = Arc::default();
        let kept = Arc::clone(&seen);
        let sink = AppSink::new("keep", move |buf| {
            if let MediaBuffer::Video(frame) = &buf {
                let found = buf.metadata().and_then(|m| m.get::<Detections>()).cloned();
                kept.lock()
                    .unwrap()
                    .push((frame.pts().unwrap_or_default(), found));
            }
            Ok(())
        });

        let recording = match out {
            Some(path) => {
                let overlay = CudaDetectionOverlay::new("overlay", device, {
                    let font = FONTS.iter().find_map(|path| std::fs::read(path).ok());
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
                    }
                })?;
                let (width, height) = stream.size().expect("a video stream says its size");
                let encoder = CudaEncoder::new(
                    "encoder",
                    device,
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
                Some((overlay, encoder, muxer.open()?.take(track)?))
            }
            None => None,
        };

        // A second model on what was found, if asked for: an ImageNet
        // classifier, as the public ones are, on the classes asked for.
        let classifier = match (&args.classifier, track) {
            (Some(path), true) => Some(CudaOrtClassifier::new(
                "classifier",
                device,
                path,
                OrtClassifierOptions {
                    labels: args
                        .classifier_labels
                        .as_ref()
                        .and_then(|path| std::fs::read_to_string(path).ok())
                        .map(|text| text.lines().map(str::to_owned).collect()),
                    classes: args.classify.clone(),
                    input: InputScale::ImageNet,
                    min_score: 0.3,
                    ..OrtClassifierOptions::default()
                },
            )?),
            _ => None,
        };

        let params = stream.parameters.clone();
        let (pipeline, ()) = Pipeline::new("cuda-track", source, |source, ctx| {
            let decoder = CudaDecoder::new("decoder", params, device, 16)?;
            let mut branch = ctx
                .branch()
                .pipe(decoder)
                .queue("pictures", 8)
                .pipe(detector);
            if track {
                branch = branch.pipe(ObjectTracker::new(
                    "tracker",
                    TrackerOptions {
                        confirm_after: args.confirm,
                        visual: args.visual,
                        ..TrackerOptions::default()
                    },
                ));
            }
            if let Some(classifier) = classifier {
                branch = branch.pipe(classifier);
            }
            let branch = match recording {
                None => branch.to(sink)?,
                Some((overlay, encoder, muxer_sink)) => {
                    let keep = ctx.branch().to(sink)?;
                    let record = ctx
                        .branch()
                        .pipe(overlay)
                        .queue("drawn", 8)
                        .pipe(encoder)
                        .to(muxer_sink)?;
                    let tee = ctx.tee("tee").branch(keep).branch(record).build()?;
                    branch.to_branch(tee)?
                }
            };
            ctx.attach(source, stream.index, branch)?;
            Ok(())
        })?;
        pipeline.run()?;
        for event in pipeline.bus().iter() {
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
        let mut seen = std::mem::take(&mut *seen.lock().unwrap());
        seen.sort_by_key(|(pts, _)| *pts);
        Ok(seen)
    }

    fn iou(a: &Detection, b: &Detection) -> f32 {
        let w = (a.x + a.width).min(b.x + b.width) - a.x.max(b.x);
        let h = (a.y + a.height).min(b.y + b.height) - a.y.max(b.y);
        let inter = w.max(0.0) * h.max(0.0);
        let union = a.width * a.height + b.width * b.height - inter;
        if union <= 0.0 { 0.0 } else { inter / union }
    }

    /// How well `boxes` cover `reference`'s confident detections: each one's
    /// best overlap with a box of its class, summed, and how many of them
    /// overlap one by half or more.
    fn cover(reference: &[Detection], boxes: &[Detection]) -> (f32, usize, usize) {
        let mut total = 0.0;
        let mut hit = 0;
        let mut count = 0;
        for r in reference.iter().filter(|r| r.score >= CONFIDENT) {
            let best = boxes
                .iter()
                .filter(|b| b.class_id == r.class_id)
                .map(|b| iou(r, b))
                .fold(0.0, f32::max);
            total += best;
            hit += usize::from(best >= 0.5);
            count += 1;
        }
        (total, hit, count)
    }

    fn evaluate(args: &Args, device: &CudaDevice, intervals: &[u32]) -> Result<()> {
        let started = Instant::now();
        let reference = pass(args, device, 0, false, None)?;
        println!(
            "reference: {} pictures, every one detected, in {:.1?}",
            reference.len(),
            started.elapsed()
        );
        let by_pts: std::collections::HashMap<i64, &Detections> = reference
            .iter()
            .filter_map(|(pts, found)| Some((*pts, found.as_ref()?)))
            .collect();

        println!();
        println!(
            "interval | pictures filled | tracker: mean IoU, hit@0.5 | held still: mean IoU, hit@0.5 | objects"
        );
        for &interval in intervals {
            let started = Instant::now();
            let seen = pass(args, device, interval, true, None)?;
            let (mut tracked, mut held) = ((0.0, 0, 0), (0.0, 0, 0));
            let mut filled = 0;
            let mut last: Vec<Detection> = Vec::new();
            let mut ids = HashSet::new();
            for (pts, found) in &seen {
                let Some(found) = found else {
                    continue;
                };
                ids.extend(found.items.iter().filter_map(|item| item.track_id));
                if !found.predicted {
                    // What holding still would show until the next look:
                    // the objects the tracker had confirmed.
                    last = found
                        .items
                        .iter()
                        .filter(|item| item.track_id.is_some())
                        .cloned()
                        .collect();
                    continue;
                }
                let Some(reference) = by_pts.get(pts) else {
                    continue;
                };
                filled += 1;
                for (sum, boxes) in [(&mut tracked, &found.items), (&mut held, &last)] {
                    let (total, hit, count) = cover(&reference.items, boxes);
                    sum.0 += total;
                    sum.1 += hit;
                    sum.2 += count;
                }
            }
            let rate = |(total, hit, count): (f32, usize, usize)| {
                let count = count.max(1) as f32;
                format!("{:.3}, {:5.1}%", total / count, 100.0 * hit as f32 / count)
            };
            println!(
                "{interval:8} | {filled:15} | {:>26} | {:>29} | {:7}   ({:.1?})",
                rate(tracked),
                rate(held),
                ids.len(),
                started.elapsed()
            );
        }
        Ok(())
    }
}
