//! Demux -> SwDecoder -> Pacer -> FrameCounter: proves `Pacer` paces decoded
//! frames out at real playback speed (via PTS + `Clock`) instead of as
//! fast as decode can produce them. Compare against `decode`, which runs
//! the same chain without a `Pacer` and finishes as fast as possible.
//!
//!     cargo run -p pace -- path/to/video.mp4

fn main() -> impl std::process::Termination {
    example::run()
}

mod example {
    use std::{sync::atomic::Ordering, time::Instant};

    use media_pp::ffmpeg::media;
    use media_pp::{
        bus::BusEvent,
        elements::{FileDemuxer, FrameCounter, Pacer, SwDecoder},
        pipeline::Pipeline,
    };

    pub(super) fn run() -> media_pp::Result<()> {
        media_pp::init()?;
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let Some(path) = std::env::args().nth(1) else {
            eprintln!("usage: pace <video.mp4>");
            std::process::exit(1);
        };

        let (source, streams) = FileDemuxer::open("demux", &path)?;

        println!("stream count: {}", streams.len());
        for s in &streams {
            println!("  [{}] {:?}", s.index, s.kind);
        }

        let video = source.best(media::Type::Video)?;
        let params = video.parameters.clone();

        let (counter, frame_count) = FrameCounter::new("counter");

        let pipeline = Pipeline::new("pace", source, |source, ctx| {
            let decoder = SwDecoder::new("decoder", params)?;
            let pacer = Pacer::new("pacer");
            let branch = ctx
                .branch()
                .pipe(decoder) // same thread as the demux — cheap enough not to need a queue
                .queue("frames", 32) // pacer sleeps on its own thread; let decode run ahead into this
                .pipe(pacer)
                .to(counter)?;
            ctx.attach(source, video.index, branch)?;
            Ok(())
        })?;

        let start = Instant::now();
        pipeline.run()?;

        // Watch for `Finished`/`Error` and `stop()` on either — errors no longer
        // end the pipeline on their own, so this is what makes the loop
        // below actually finish instead of running forever after a failure.
        for event in pipeline.bus().iter() {
            println!("{event}");
            if matches!(event, BusEvent::Finished | BusEvent::Error { .. }) {
                pipeline.stop();
            }
        }

        println!("decoded frames: {}", frame_count.load(Ordering::Relaxed));
        println!("wall time: {:.2}s", start.elapsed().as_secs_f64());
        Ok(())
    }
}
