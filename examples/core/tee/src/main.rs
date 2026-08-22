use std::sync::atomic::Ordering;

use media_pp::ffmpeg::media;
use media_pp::{
    Error,
    elements::{FileDemuxer, FrameCounter, PacketCounter, SwDecoder, TeeBuilder},
    pipeline::Pipeline,
};

/// Demux -> Tee, fanning the same packets out to two independent
/// branches:
///   - SwDecoder -> FrameCounter: decodes and counts frames
///   - PacketCounter: counts the raw (still-encoded) packets
///
/// Proves `Tee` delivers every packet to both branches — same source
/// data, two unrelated consumers.
///
///     cargo run -p tee -- path/to/video.mp4
fn main() -> media_pp::Result<()> {
    media_pp::init()?;
    let _log_guard = media_pp::log::init(
        env!("CARGO_PKG_NAME"),
        "logs",
        media_pp::log::Level::Trace,
        7,
    )?;

    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: tee <video.mp4>");
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

    let (frame_counter, frame_count) = FrameCounter::new("frame-counter");
    let (packet_counter, packet_count) = PacketCounter::new("packet-counter");

    let pipeline = Pipeline::new("tee", source, |source, ctx| {
        let decoder = SwDecoder::new("decoder", params).expect("failed to open decoder");
        let decode_branch = ctx.branch().pipe(decoder).to(Box::new(frame_counter))?;
        let packet_branch = ctx.branch().to(Box::new(packet_counter))?;

        let tee_branch = TeeBuilder::new("tee", ctx.clone())
            .branch(decode_branch)
            .branch(packet_branch)
            .build()?;
        ctx.attach(source, video.index, tee_branch)?;
        Ok(())
    })?;

    pipeline.run()?;
    pipeline.bus().log_events();

    println!("decoded frames: {}", frame_count.load(Ordering::Relaxed));
    println!("raw packets: {}", packet_count.load(Ordering::Relaxed));
    Ok(())
}
