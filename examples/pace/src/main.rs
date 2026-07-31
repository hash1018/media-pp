use std::{sync::atomic::Ordering, time::Instant};

use ffmpeg_next::media;
use media_pp::{
    Error,
    element::Source,
    elements::{FileDemuxer, FrameCounter, Pacer, SwDecoder},
    pipeline::{ChainBuilder, Pipeline},
};

/// Demux -> SwDecoder -> Pacer -> FrameCounter: proves `Pacer` paces decoded
/// frames out at real playback speed (via PTS + `Clock`) instead of as
/// fast as decode can produce them. Compare against `decode`, which runs
/// the same chain without a `Pacer` and finishes as fast as possible.
///
///     cargo run -p pace -- path/to/video.mp4
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
    let time_base = source
        .stream_time_base(video.index)
        .ok_or_else(|| Error::Other("stream disappeared".into()))?;

    let (counter, frame_count) = FrameCounter::new("counter");

    let pipeline = Pipeline::new(source, |source, bus, clock| {
        let decoder = SwDecoder::new("decoder", params).expect("failed to open decoder");
        let pacer = Pacer::new("pacer", time_base, clock.clone());
        let branch = ChainBuilder::new(bus.clone())
            .pipe(decoder) // same thread as the demux — cheap enough not to need a queue
            .queue("frames", 32) // pacer sleeps on its own thread; let decode run ahead into this
            .pipe(pacer)
            .build(Box::new(counter));
        source.src_pads()[video.index].link(branch);
    });

    let start = Instant::now();
    pipeline.run();
    pipeline.bus().log_events(); // blocks until playback actually finishes

    println!("decoded frames: {}", frame_count.load(Ordering::Relaxed));
    println!("wall time: {:.2}s", start.elapsed().as_secs_f64());
    Ok(())
}
