//! FileDemuxer -> CudaDecoder -> Queue -> CudaOrtDetector -> AppSink: finds
//! objects in a file's pictures without them leaving the GPU — NVDEC
//! decodes, a kernel fits each picture into the model's input in device
//! memory, TensorRT runs the model — and prints what each picture carries
//! on, then how fast it went.
//!
//! The model is an Ultralytics YOLO ONNX export — YOLOv8 and YOLO11, or
//! YOLOv10 and YOLO26. The first run builds a TensorRT engine for the model
//! and this GPU, which takes minutes; later runs load it from the cache in
//! under a second. Needs CUDA 13, cuDNN 9 and TensorRT 10 where the loader
//! finds them — `LD_LIBRARY_PATH` pointing at them.
//!
//!     cargo run --release -p cuda_detect -- path/to/model.onnx path/to/video.mp4 [pictures]

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("{} example only supports Linux", env!("CARGO_PKG_NAME"));
}

#[cfg(target_os = "linux")]
fn main() -> impl std::process::Termination {
    example::run()
}

#[cfg(target_os = "linux")]
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
        elements::{
            AppSink, COCO_CLASS_LABELS, CudaDecoder, CudaDevice, CudaOrtDetector,
            CudaOrtDetectorOptions, Detections, FileDemuxer,
        },
        ffmpeg::media,
        pipeline::Pipeline,
    };

    pub(super) fn run() -> Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Info,
            7,
        )?;
        let mut args = std::env::args().skip(1);
        let (Some(model), Some(video)) = (args.next(), args.next()) else {
            eprintln!("usage: cuda_detect <model.onnx> <video.mp4> [pictures]");
            std::process::exit(1);
        };
        let limit: usize = args
            .next()
            .and_then(|n| n.parse().ok())
            .unwrap_or(usize::MAX);

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
        let (pipeline, ()) = Pipeline::new("cuda-detect", source, |source, ctx| {
            let decoder = CudaDecoder::new("decoder", params, &device, 16)?;
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
        Ok(())
    }
}
