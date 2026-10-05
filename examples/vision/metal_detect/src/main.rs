//! FileDemuxer -> VideoToolboxDecoder -> Queue -> MetalOrtDetector ->
//! AppSink: finds objects in a file's pictures with the pictures fitted on
//! the GPU — VideoToolbox decodes, a Metal kernel fits each picture into the
//! model's input where it is, Core ML runs the model on the GPU or the
//! Neural Engine — and prints what each picture carries on, then how fast it
//! went. The macOS counterpart of `cuda_detect`.
//!
//! The model is an Ultralytics YOLO ONNX export — YOLOv8 and YOLO11, or
//! YOLOv10 and YOLO26. Built with `ort-coreml`, on an Apple silicon Mac. The
//! file's video has to be one VideoToolbox decodes to NV12 — 8-bit H.264 or
//! HEVC, say: a 10-bit one is refused as the pipeline is wired, the decoder's
//! P010 being a layout the detector does not take. `pictures` stops it after
//! about that many.
//!
//!     cargo run --release -p metal_detect -- path/to/model.onnx path/to/video.mp4 [pictures]

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
            AppSink, COCO_CLASS_LABELS, Detections, FileDemuxer, MetalOrtDetector,
            OrtDetectorOptions, VideoToolboxDecoder, VideoToolboxDevice,
        },
        ffmpeg::media,
        pipeline::Pipeline,
    };

    pub(super) fn run() -> Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;
        let mut args = std::env::args().skip(1);
        let (Some(model), Some(video)) = (args.next(), args.next()) else {
            eprintln!("usage: metal_detect <model.onnx> <video.mp4> [pictures]");
            std::process::exit(1);
        };
        let limit: usize = args
            .next()
            .and_then(|n| n.parse().ok())
            .unwrap_or(usize::MAX);

        let device = VideoToolboxDevice::new()?;
        let started = Instant::now();
        let detector = MetalOrtDetector::new("detector", &model, OrtDetectorOptions::default())?;
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

        let params = stream.parameters.clone();
        let (pipeline, ()) = Pipeline::new("metal-detect", source, |source, ctx| {
            let decoder = VideoToolboxDecoder::new("decoder", params, &device)?;
            let branch = ctx
                .branch()
                .pipe(decoder)
                .queue("pictures", 8)
                .pipe(detector)
                .to(sink)?;
            ctx.attach(source, stream.index, branch)?;
            Ok(())
        })?;

        let running = Instant::now();
        pipeline.run()?;
        // Woken every 50 ms as well as by an event, since a run that goes
        // well posts none until it finishes, and `pictures` is counted here.
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
        Ok(())
    }
}
