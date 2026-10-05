//! Several files through one detector, a batch at a time — DeepStream's
//! `nvstreammux -> nvinfer -> nvstreamdemux`. Each file is a pipeline of its
//! own that decodes on NVDEC and ends in an input of a `StreamMux`:
//!
//!     FileDemuxer -> CudaDecoder -> Queue -> StreamMux input      (one per stream)
//!
//! and the mux's pipeline gathers a picture of each stream into a batch,
//! runs the batch through TensorRT at once, and splits the streams out
//! again, each to a branch of its own that counts its pictures and what was
//! found in them:
//!
//!     StreamMux -> CudaOrtDetector -> demux ─┬→ AppSink   (stream 0)
//!                                            └→ AppSink   (stream 1) …
//!
//! The mux is offline, so nothing is dropped: a batch goes out once every
//! stream that has not ended has a picture, and the files are decoded as fast
//! as the detector takes them. At the end it prints each stream's count and
//! how many pictures a second went through altogether; `--batch 1` runs the
//! same streams one picture at a time, to compare.
//!
//! Given one file, `--streams N` runs N copies of it. The model is an
//! Ultralytics YOLO ONNX export whose batch is left open (`dynamic=True`) —
//! one made for a single picture runs one at a time, with a warning. The
//! first run builds a TensorRT engine for the model, this GPU and the batch,
//! which takes minutes; later runs load it from the cache. Built with
//! `ort-tensorrt`, whose libraries it needs as `cuda_detect` does: on Linux
//! where the linker and the loader find them, on Windows their DLLs on
//! `PATH`.
//!
//!     cargo run --release -p cuda_multi_detect -- model.onnx video.mp4 [video.mp4 …] \
//!         [--streams N] [--batch B] [--pictures N]

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
    use std::time::{Duration, Instant};

    use media_pp::{
        Result,
        bus::BusEvent,
        elements::{
            AppSink, CudaDecoder, CudaDevice, CudaOrtDetector, CudaOrtDetectorOptions, Detections,
            FileDemuxer, RenderMode, StreamMux, StreamMuxOptions,
        },
        ffmpeg::media,
        pipeline::Pipeline,
    };

    struct Args {
        model: String,
        videos: Vec<String>,
        batch: usize,
        pictures: usize,
    }

    fn args() -> Args {
        let usage = || -> ! {
            eprintln!(
                "usage: cuda_multi_detect <model.onnx> <video.mp4> [video.mp4 ...] \
                 [--streams N] [--batch B] [--pictures N]"
            );
            std::process::exit(1);
        };
        let number = |value: Option<String>| -> usize {
            value
                .and_then(|n| n.parse().ok())
                .filter(|n| *n > 0)
                .unwrap_or_else(|| usage())
        };
        let mut positional = Vec::new();
        let (mut streams, mut batch, mut pictures) = (None, None, usize::MAX);
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--streams" => streams = Some(number(args.next())),
                "--batch" => batch = Some(number(args.next())),
                "--pictures" => pictures = number(args.next()),
                _ => positional.push(arg),
            }
        }
        if positional.len() < 2 {
            usage();
        }
        let model = positional.remove(0);
        let videos = match streams {
            Some(n) if positional.len() == 1 => vec![positional[0].clone(); n],
            Some(n) if n != positional.len() => usage(),
            _ => positional,
        };
        // A picture of each stream in one batch, up to what a small model
        // gains from: past 8 the gain is a few percent.
        let batch = batch.unwrap_or(videos.len().min(8));
        Args {
            model,
            videos,
            batch,
            pictures,
        }
    }

    /// What one stream's branch counted.
    #[derive(Default)]
    struct Counts {
        pictures: AtomicUsize,
        objects: AtomicUsize,
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
            videos,
            batch,
            pictures: limit,
        } = args();

        let device = CudaDevice::new()?;
        let started = Instant::now();
        let detector = CudaOrtDetector::new(
            "detector",
            &device,
            &model,
            CudaOrtDetectorOptions {
                max_batch: batch,
                ..CudaOrtDetectorOptions::default()
            },
        )?;
        println!("detector ready in {:.1?}", started.elapsed());

        let (mux, handle) = StreamMux::new(
            "mux",
            StreamMuxOptions {
                mode: RenderMode::Offline { end: None },
                max_batch: batch,
                ..StreamMuxOptions::default()
            },
        )?;
        let (detecting, demux) = Pipeline::new("detect", mux, |source, ctx| {
            let (demuxed, demux) = handle.demux(ctx, "demux")?;
            let branch = ctx.branch().pipe(detector).to_branch(demuxed)?;
            ctx.attach(source, 0, branch)?;
            Ok(demux)
        })?;

        // A pipeline for each stream, decoding into the mux, and a branch
        // after the demux counting what comes out for it.
        let mut feeds = Vec::new();
        let mut counts = Vec::new();
        for (index, video) in videos.iter().enumerate() {
            let (input, stream_id) = handle.add_source(format!("stream {index}"))?;
            let counted = Arc::new(Counts::default());
            let count = Arc::clone(&counted);
            let sink = AppSink::new(format!("count {index}"), move |buf| {
                count.pictures.fetch_add(1, Ordering::Relaxed);
                if let Some(found) = buf
                    .metadata()
                    .and_then(|metadata| metadata.get::<Detections>())
                {
                    count
                        .objects
                        .fetch_add(found.items.len(), Ordering::Relaxed);
                }
                Ok(())
            });
            demux.attach(stream_id, demux.branch()?.to(sink)?)?;
            counts.push(counted);

            let (source, _) = FileDemuxer::open(format!("file {index}"), video)?;
            let stream = source.best(media::Type::Video)?;
            let params = stream.parameters.clone();
            let device = &device;
            let (feed, ()) = Pipeline::new(format!("stream {index}"), source, |source, ctx| {
                let decoder = CudaDecoder::new(format!("decoder {index}"), params, device, 16)?;
                let branch = ctx
                    .branch()
                    .pipe(decoder)
                    .queue(format!("pictures {index}"), 4)
                    .to(input)?;
                ctx.attach(source, stream.index, branch)?;
                Ok(())
            })?;
            feeds.push(feed);
        }
        println!(
            "{} streams, batches of up to {batch}, ready in {:.1?}",
            videos.len(),
            started.elapsed()
        );

        let running = Instant::now();
        detecting.run()?;
        for feed in &feeds {
            feed.run()?;
        }
        let total = || -> usize {
            counts
                .iter()
                .map(|count| count.pictures.load(Ordering::Relaxed))
                .sum()
        };
        let enough = limit.saturating_mul(videos.len());
        'watch: loop {
            if total() >= enough {
                break;
            }
            for pipeline in feeds.iter().chain([&detecting]) {
                while let Some(event) = pipeline.bus().try_recv() {
                    match event {
                        // Every stream has ended and been handed on.
                        BusEvent::Finished if Arc::ptr_eq(pipeline, &detecting) => break 'watch,
                        BusEvent::Error { .. } => {
                            eprintln!("{event}");
                            break 'watch;
                        }
                        _ => {}
                    }
                }
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        let elapsed = running.elapsed();
        for feed in &feeds {
            feed.stop();
        }
        detecting.stop();

        for (index, count) in counts.iter().enumerate() {
            println!(
                "stream {index}: {} pictures, {} objects found",
                count.pictures.load(Ordering::Relaxed),
                count.objects.load(Ordering::Relaxed)
            );
        }
        let n = total();
        println!(
            "{n} pictures in {elapsed:.1?}: {:.1} pictures a second, decode and detection together",
            n as f64 / elapsed.as_secs_f64()
        );
        Ok(())
    }
}
