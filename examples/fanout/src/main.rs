use std::sync::atomic::Ordering;

use ffmpeg_next::media;
use media_pp::{
    element::Source,
    elements::{FileDemuxer, PacketCounter},
    pipeline::{ChainBuilder, Pipeline},
};

/// Demonstrates fan-out: open a file, inspect its streams, then link
/// video and audio to separate branches (each behind its own `Queue`
/// thread boundary) — just two of the demuxer's src pads, no separate
/// "Tee" element involved.
///
///     cargo run -p fanout -- path/to/video_and_audio.mp4
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

    let video = streams.iter().find(|s| s.kind == media::Type::Video);
    let audio = streams.iter().find(|s| s.kind == media::Type::Audio);

    let (video_counter, video_count) = PacketCounter::new("video-counter");
    let (audio_counter, audio_count) = PacketCounter::new("audio-counter");

    let pipeline = Pipeline::new(source, |source, bus, _clock| {
        if let Some(v) = video {
            let branch = ChainBuilder::new(bus.clone())
                .queue("video-q", 32) // its own thread, separate from audio
                .build(Box::new(video_counter));
            source.src_pads()[v.index].link(branch);
        }
        if let Some(a) = audio {
            let branch = ChainBuilder::new(bus.clone())
                .queue("audio-q", 32)
                .build(Box::new(audio_counter));
            source.src_pads()[a.index].link(branch);
        }
        // Any other stream's pad is simply left unlinked.
    });

    pipeline.run(); // starts the source on a background thread, returns right away
    // Blocks until the demuxer hits EOS and both branch queues have
    // drained and joined (i.e. every `Bus` handle in the pipeline dropped).
    pipeline.bus().log_events();

    println!("video packets: {}", video_count.load(Ordering::Relaxed));
    println!("audio packets: {}", audio_count.load(Ordering::Relaxed));
    Ok(())
}
