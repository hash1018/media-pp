use std::sync::atomic::Ordering;

use ffmpeg_next::media;
use media_pp::{
    Error,
    bus::BusEvent,
    element::Source,
    elements::{FileDemuxer, FrameCounter, SwDecoder},
    pipeline::{ChainBuilder, Pipeline},
};

/// Demux -> SwDecoder -> FrameCounter: proves `SwDecoder` (a `Filter`,
/// both `Source` and `Sink`) actually decodes packets into frames, not
/// just that it compiles.
///
///     cargo run -p decode -- path/to/video.mp4
fn main() -> media_pp::Result<()> {
    media_pp::init()?;

    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "test-video/h265.mp4".into());

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

    let (counter, frame_count) = FrameCounter::new("counter");

    let pipeline = Pipeline::new("decode", source, |source, bus, _clock, id, registry| {
        let decoder = SwDecoder::new("decoder", params).expect("failed to open decoder");
        let branch = ChainBuilder::new(bus.clone(), id, registry.clone())
            .pipe(decoder) // same thread as the demux — cheap enough not to need a queue
            .build(Box::new(counter));
        source.src_pads()[video.index].link(branch);
    });

    pipeline.run();

    // Same output `log_events()` would print, but also calls `stop()` on
    // `Eos`/`Error` — errors no longer end the pipeline on their own (see
    // `BusEvent`'s docs), so watching for one here is what makes this
    // still exit instead of running forever after a failure. Single
    // stream, so `Eos` calling `stop()` is a harmless no-op (everything's
    // already finished by then) — a multi-stream pipeline would need to
    // wait for every branch's `Eos` before stopping.
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

    println!("decoded frames: {}", frame_count.load(Ordering::Relaxed));
    Ok(())
}
