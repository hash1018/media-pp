//! FileDemuxer -> SwDecoder -> SwCutDetector -> AppSink: finds where one
//! shot of an edited video ends and the next begins, and prints each cut —
//! the picture the new shot begins on, its time and how far it was from the
//! picture before — then how many there were and how fast it went.
//!
//! With `--cuda`, on Linux and Windows, the pictures are decoded on an
//! NVIDIA GPU and the cuts found there, the pictures never leaving it:
//! `FileDemuxer -> CudaDecoder -> Queue -> CudaCutDetector -> AppSink`. Both
//! find the same cuts in the same file, since both make the same thumbnail
//! of each picture.
//!
//! With `--metal`, on macOS, VideoToolbox decodes and the cuts are found on
//! the GPU by Metal, likewise the same cuts: `FileDemuxer ->
//! VideoToolboxDecoder -> Queue -> MetalCutDetector -> AppSink`.
//!
//!     cargo run --release -p cut_detect -- path/to/video.mp4 [--cuda | --metal]

fn main() -> impl std::process::Termination {
    example::run()
}

mod example {
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    };
    use std::time::Instant;

    use media_pp::{
        Result,
        bus::BusEvent,
        elements::{AppSink, CutDetectorOptions, FileDemuxer, SceneCut, SwCutDetector, SwDecoder},
        ffmpeg::media,
        pipeline::Pipeline,
    };

    struct Args {
        video: String,
        cuda: bool,
        metal: bool,
    }

    fn args() -> Args {
        let usage = || -> ! {
            eprintln!("usage: cut_detect <video.mp4> [--cuda | --metal]");
            std::process::exit(1);
        };
        let mut video = None;
        let (mut cuda, mut metal) = (false, false);
        for arg in std::env::args().skip(1) {
            match arg.as_str() {
                "--cuda" => cuda = true,
                "--metal" => metal = true,
                _ if video.is_none() && !arg.starts_with("--") => video = Some(arg),
                _ => usage(),
            }
        }
        if cuda && !cfg!(any(target_os = "linux", target_os = "windows")) {
            eprintln!("--cuda takes an NVIDIA GPU, on Linux or Windows");
            std::process::exit(1);
        }
        if metal && !cfg!(target_os = "macos") {
            eprintln!("--metal takes a Mac");
            std::process::exit(1);
        }
        if cuda && metal {
            usage();
        }
        Args {
            video: video.unwrap_or_else(|| usage()),
            cuda,
            metal,
        }
    }

    pub(super) fn run() -> Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;
        let Args { video, cuda, metal } = args();

        let (source, _) = FileDemuxer::open("demux", &video)?;
        let stream = source.best(media::Type::Video)?;
        let params = stream.parameters.clone();
        let index = stream.index;

        let seen = Arc::new(AtomicUsize::new(0));
        let cuts = Arc::new(Mutex::new(Vec::new()));
        let (counted, found) = (Arc::clone(&seen), Arc::clone(&cuts));
        let sink = AppSink::new("print", move |buf| {
            let picture = counted.fetch_add(1, Ordering::Relaxed);
            if let Some(cut) = buf
                .metadata()
                .and_then(|metadata| metadata.get::<SceneCut>())
            {
                let at = match &buf {
                    media_pp::buffer::MediaBuffer::Video(frame) => frame
                        .pts()
                        .zip(media_pp::buffer::time_base(frame))
                        .map(|(pts, unit)| pts as f64 * f64::from(unit)),
                    _ => None,
                };
                println!(
                    "cut at picture {picture} ({}), score {:.1}",
                    at.map_or("no time".into(), |at| format!("{at:.3} s")),
                    cut.score
                );
                found.lock().unwrap().push(picture);
            }
            Ok(())
        });

        let options = CutDetectorOptions::default();
        let (pipeline, ()) = Pipeline::new("cut-detect", source, |source, ctx| {
            let branch = if cuda {
                #[cfg(any(target_os = "linux", target_os = "windows"))]
                {
                    use media_pp::elements::{CudaCutDetector, CudaDecoder, CudaDevice};
                    let device = CudaDevice::new()?;
                    // The queue's eight, and the pictures the detector holds
                    // for those after them.
                    let budget = 8 + options.lookahead as i32 + 1;
                    ctx.branch()
                        .pipe(CudaDecoder::new("decoder", params, &device, budget)?)
                        .queue("pictures", 8)
                        .pipe(CudaCutDetector::new("cuts", &device, options)?)
                        .to(sink)?
                }
                #[cfg(not(any(target_os = "linux", target_os = "windows")))]
                unreachable!("refused in args()")
            } else if metal {
                #[cfg(target_os = "macos")]
                {
                    use media_pp::elements::{
                        MetalCutDetector, VideoToolboxDecoder, VideoToolboxDevice,
                    };
                    let device = VideoToolboxDevice::new()?;
                    ctx.branch()
                        .pipe(VideoToolboxDecoder::new("decoder", params, &device)?)
                        .queue("pictures", 8)
                        .pipe(MetalCutDetector::new("cuts", options)?)
                        .to(sink)?
                }
                #[cfg(not(target_os = "macos"))]
                unreachable!("refused in args()")
            } else {
                ctx.branch()
                    .pipe(SwDecoder::new("decoder", params)?)
                    .pipe(SwCutDetector::new("cuts", options)?)
                    .to(sink)?
            };
            ctx.attach(source, index, branch)?;
            Ok(())
        })?;

        let started = Instant::now();
        pipeline.run()?;
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
        let elapsed = started.elapsed();
        let pictures = seen.load(Ordering::Relaxed);
        println!(
            "{} cuts in {pictures} pictures, {elapsed:.1?}: {:.0} pictures a second",
            cuts.lock().unwrap().len(),
            pictures as f64 / elapsed.as_secs_f64()
        );
        Ok(())
    }
}
