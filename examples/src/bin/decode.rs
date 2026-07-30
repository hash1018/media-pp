use std::sync::atomic::Ordering;

use ffmpeg_next::media;
use media_pp::{
    Error,
    bus::BusEvent,
    element::Source,
    elements::{Decoder, FileDemuxSource, FrameCounter},
    pipeline::{ChainBuilder, Pipeline},
};

/// Demux -> Decoder -> FrameCounter: proves `Decoder` (a `Filter`, both
/// `Source` and `Sink`) actually decodes packets into frames, not just
/// that it compiles.
///
///     cargo run -p media-pp-examples --bin decode -- path/to/video.mp4
fn main() -> media_pp::Result<()> {
    media_pp::init()?;

    let path = std::env::args().nth(1).expect("usage: decode <path>");

    let (source, streams) = FileDemuxSource::open("demux", &path)?;

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

    let (counter, frame_count) = FrameCounter::new("counter");

    let mut pipeline = Pipeline::new(source, |source, bus| {
        let decoder = Decoder::new("decoder", params).expect("failed to open decoder");
        let branch = ChainBuilder::new(bus.clone())
            .pipe(decoder) // same thread as the demux — cheap enough not to need a queue
            .build(Box::new(counter));
        source.src_pads()[video.index].link(branch);
    });

    pipeline.run()?;
    for event in pipeline.bus().iter() {
        match event {
            BusEvent::Error { element, message } => eprintln!("[{element}] error: {message}"),
            BusEvent::Eos { element } => println!("[{element}] eos"),
            BusEvent::Dropped { element } => eprintln!("[{element}] dropped a buffer (queue full)"),
        }
    }

    println!("decoded frames: {}", frame_count.load(Ordering::Relaxed));
    Ok(())
}
