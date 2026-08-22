use std::{sync::atomic::Ordering, time::Instant};

use media_pp::ffmpeg::media;
use media_pp::{
    Error,
    bus::BusEvent,
    elements::{FileDemuxer, FrameCounter, Pacer, SwDecoder},
    pipeline::Pipeline,
};

/// Demux -> SwDecoder -> Pacer -> FrameCounter: proves `Pacer` paces decoded
/// frames out at real playback speed (via PTS + `Clock`) instead of as
/// fast as decode can produce them. Compare against `decode`, which runs
/// the same chain without a `Pacer` and finishes as fast as possible.
///
///     cargo run -p pace -- path/to/video.mp4
fn main() -> media_pp::Result<()> {
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

    let video = streams
        .iter()
        .find(|s| s.kind == media::Type::Video)
        .ok_or_else(|| Error::Other("no video stream in file".into()))?;
    let params = source
        .stream_parameters(video.index)
        .ok_or_else(|| Error::Other("stream disappeared".into()))?;
    let time_base = source
        .stream_time_base(video.index)
        .ok_or_else(|| Error::Other("stream disappeared".into()))?;

    let (counter, frame_count) = FrameCounter::new("counter");

    let pipeline = Pipeline::new("pace", source, |source, ctx| {
        let decoder = SwDecoder::new("decoder", params).expect("failed to open decoder");
        let pacer = Pacer::new("pacer", time_base, ctx.clock.clone())?;
        let branch = ctx
            .branch()
            .pipe(decoder) // same thread as the demux — cheap enough not to need a queue
            .queue("frames", 32) // pacer sleeps on its own thread; let decode run ahead into this
            .pipe(pacer)
            .to(Box::new(counter))?;
        ctx.attach(source, video.index, branch)?;
        Ok(())
    })?;

    let start = Instant::now();
    pipeline.run()?;

    // Watch for `Eos`/`Error` and `stop()` on either — errors no longer
    // end the pipeline on their own, so this is what makes the loop
    // below actually finish instead of running forever after a failure.
    for event in pipeline.bus().iter() {
        match &event {
            BusEvent::Eos { name, .. } => println!("[{name}] eos"),
            BusEvent::Error { name, error, .. } => eprintln!("[{name}] error: {error}"),
            BusEvent::Dropped { name, .. } => eprintln!("[{name}] dropped a buffer (queue full)"),
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
        if matches!(event, BusEvent::Eos { .. } | BusEvent::Error { .. }) {
            pipeline.stop();
        }
    }

    println!("decoded frames: {}", frame_count.load(Ordering::Relaxed));
    println!("wall time: {:.2}s", start.elapsed().as_secs_f64());
    Ok(())
}
