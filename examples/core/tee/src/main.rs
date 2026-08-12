use std::sync::atomic::Ordering;

use ffmpeg_next::media;
use media_pp::{
    Error,
    elements::{FileDemuxer, FrameCounter, PacketCounter, SwDecoder, Tee},
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

    let (frame_counter, frame_count) = FrameCounter::new("frame-counter");
    let (packet_counter, packet_count) = PacketCounter::new("packet-counter");

    let pipeline = Pipeline::new("tee", source, |source, ctx| {
        let decoder = SwDecoder::new("decoder", params).expect("failed to open decoder");
        let decode_branch = ctx.branch().pipe(decoder).to(Box::new(frame_counter))?;
        let packet_branch = ctx.branch().to(Box::new(packet_counter))?;

        let (tee, tee_handle) = Tee::new("tee", ctx.clone());
        let tee_branch = ctx.branch().to(Box::new(tee))?;
        ctx.attach(source, video.index, tee_branch)?;
        tee_handle.attach(decode_branch)?;
        tee_handle.attach(packet_branch)?;
        Ok(())
    })?;

    pipeline.run();
    pipeline.bus().log_events();

    println!("decoded frames: {}", frame_count.load(Ordering::Relaxed));
    println!("raw packets: {}", packet_count.load(Ordering::Relaxed));
    Ok(())
}
