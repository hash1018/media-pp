//! Demux -> SwDecoder -> FrameCounter: proves `SwDecoder` (a `Filter`,
//! both `Source` and `Sink`) actually decodes packets into frames, not
//! just that it compiles.
//!
//!     cargo run -p decode -- path/to/video.mp4

fn main() -> impl std::process::Termination {
    example::run()
}

mod example {
    use std::sync::atomic::Ordering;

    use media_pp::ffmpeg::media;
    use media_pp::{
        bus::BusEvent,
        elements::{FileDemuxer, FrameCounter, SwDecoder},
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
            eprintln!("usage: decode <video.mp4>");
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

        let pipeline = Pipeline::new("decode", source, |source, ctx| {
            let decoder = SwDecoder::new("decoder", params).expect("failed to open decoder");
            let branch = ctx
                .branch()
                .pipe(decoder) // same thread as the demux — cheap enough not to need a queue
                .to(counter)?;
            ctx.attach(source, video.index, branch)?;
            Ok(())
        })?;

        pipeline.run()?;

        // Same output `log_events()` would print, but also calls `stop()` on
        // `Finished`/`Error` — errors no longer end the pipeline on their own (see
        // `BusEvent`'s docs), so watching for one here is what makes this
        // still exit instead of running forever after a failure. `Finished`
        // rather than `Eos`: every element that ends posts an `Eos`, and
        // `Finished` is the pipeline saying they all have.
        for event in pipeline.bus().iter() {
            match &event {
                BusEvent::Eos { name, .. } => println!("[{name}] eos"),
                BusEvent::Error { name, error, .. } => eprintln!("[{name}] error: {error}"),
                BusEvent::Dropped { name, .. } => {
                    eprintln!("[{name}] dropped a buffer (queue full)")
                }
                BusEvent::Seeked {
                    name,
                    requested,
                    landed,
                    ..
                } => println!("[{name}] seeked: requested {requested:.2?}, landed {landed:.2?}"),
                // `BusEvent` is `#[non_exhaustive]`; this example only acts
                // on the events above.
                _ => {}
            }
            if matches!(event, BusEvent::Finished | BusEvent::Error { .. }) {
                pipeline.stop();
            }
        }

        println!("decoded frames: {}", frame_count.load(Ordering::Relaxed));
        Ok(())
    }
}
