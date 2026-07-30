use std::sync::atomic::Ordering;

use ffmpeg_next::media;
use media_pp::{
    Error,
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

    let mut pipeline = Pipeline::new(source, |source, bus| {
        let decoder = SwDecoder::new("decoder", params).expect("failed to open decoder");
        let branch = ChainBuilder::new(bus.clone())
            .pipe(decoder) // same thread as the demux — cheap enough not to need a queue
            .build(Box::new(counter));
        source.src_pads()[video.index].link(branch);
    });

    pipeline.run()?;
    pipeline.bus().log_events();

    println!("decoded frames: {}", frame_count.load(Ordering::Relaxed));
    Ok(())
}
