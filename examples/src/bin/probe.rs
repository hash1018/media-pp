use std::sync::atomic::Ordering;

use ffmpeg_next::media;
use media_pp::{
    Error,
    bus::BusEvent,
    element::Source,
    elements::{FileDemuxSource, PacketCounter},
    pipeline::{ChainBuilder, Pipeline},
};

/// Smoke test for the architecture: open a file, inspect its streams,
/// *then* decide how to wire the pipeline. Demuxes on the source thread,
/// hops across an explicit `Queue` thread boundary, and counts packets on
/// the queue's worker thread.
///
///     cargo run -p media-pp-examples --bin probe -- path/to/video.mp4
fn main() -> media_pp::Result<()> {
    media_pp::init()?;

    let path = std::env::args().nth(1).expect("usage: probe <path>");

    let (source, streams) = FileDemuxSource::open("demux", &path)?;

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

    let mut pipeline = Pipeline::new(source, |source, bus| {
        let branch = ChainBuilder::new(bus.clone())
            .queue("q1", 32) // thread boundary: demux thread -> counter thread
            .build(Box::new(counter));
        source.src_pads()[video.index].link(branch);
    });

    // run() blocks until the source hits EOS and every queue worker
    // thread downstream has drained and joined, so it's safe to read the
    // bus and the counter right after.
    pipeline.run()?;
    for event in pipeline.bus().iter() {
        match event {
            BusEvent::Error { element, message } => eprintln!("[{element}] error: {message}"),
            BusEvent::Eos { element } => println!("[{element}] eos"),
            BusEvent::Dropped { element } => eprintln!("[{element}] dropped a buffer (queue full)"),
        }
    }

    println!("packet count: {}", count.load(Ordering::Relaxed));
    Ok(())
}
