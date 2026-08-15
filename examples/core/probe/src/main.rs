use std::sync::atomic::Ordering;

use ffmpeg_next::media;
use media_pp::{
    Error,
    bus::BusEvent,
    elements::{FileDemuxer, PacketCounter},
    pipeline::Pipeline,
};

/// Smoke test for the architecture: open a file, inspect its streams,
/// *then* decide how to wire the pipeline. Demuxes on the source thread,
/// hops across an explicit `Queue` thread boundary, and counts packets on
/// the queue's worker thread.
///
///     cargo run -p probe -- path/to/video.mp4
fn main() -> media_pp::Result<()> {
    media_pp::init()?;
    let _log_guard = media_pp::log::init(
        env!("CARGO_PKG_NAME"),
        "logs",
        media_pp::log::Level::Trace,
        7,
    )?;

    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: probe <video.mp4>");
        std::process::exit(1);
    };

    let (source, streams) = FileDemuxer::open("demux", &path)?;

    println!("stream count: {}", streams.len());
    for s in &streams {
        println!("  [{}] {:?}", s.index, s.kind);
    }

    // Now that we know what's in the file, decide what to demux by
    // linking the matching src pad — nothing else gets pulled off the
    // wire (unlinked pads just drop their packets).
    let video = streams
        .iter()
        .find(|s| s.kind == media::Type::Video)
        .ok_or_else(|| Error::Other("no video stream in file".into()))?;

    let (counter, count) = PacketCounter::new("counter");

    let pipeline = Pipeline::new("probe", source, |source, ctx| {
        let branch = ctx
            .branch()
            .queue("q1", 32) // thread boundary: demux thread -> counter thread
            .to(Box::new(counter))?;
        ctx.attach(source, video.index, branch)?;
        Ok(())
    })?;

    pipeline.run();

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
        }
        if matches!(event, BusEvent::Eos { .. } | BusEvent::Error { .. }) {
            pipeline.stop();
        }
    }

    println!("packet count: {}", count.load(Ordering::Relaxed));
    Ok(())
}
