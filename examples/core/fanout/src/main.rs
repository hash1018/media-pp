use std::sync::atomic::Ordering;

use ffmpeg_next::media;
use media_pp::{
    elements::{FileDemuxer, PacketCounter},
    pipeline::Pipeline,
};

/// Demonstrates fan-out: open a file, inspect its streams, then link
/// video and audio to separate branches (each behind its own `Queue`
/// thread boundary) — just two of the demuxer's src pads, no separate
/// "Tee" element involved.
///
///     cargo run -p fanout -- path/to/video_and_audio.mp4
fn main() -> media_pp::Result<()> {
    media_pp::init()?;
    let _log_guard = media_pp::log::init(
        env!("CARGO_PKG_NAME"),
        "logs",
        media_pp::log::Level::Trace,
        7,
    )?;

    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: fanout <video.mp4>");
        std::process::exit(1);
    };

    let (source, streams) = FileDemuxer::open("demux", &path)?;

    println!("stream count: {}", streams.len());
    for s in &streams {
        println!("  [{}] {:?}", s.index, s.kind);
    }

    let video = streams.iter().find(|s| s.kind == media::Type::Video);
    let audio = streams.iter().find(|s| s.kind == media::Type::Audio);

    let (video_counter, video_count) = PacketCounter::new("video-counter");
    let (audio_counter, audio_count) = PacketCounter::new("audio-counter");

    let pipeline = Pipeline::new("fanout", source, |source, ctx| {
        if let Some(v) = video {
            let branch = ctx
                .branch()
                .queue("video-q", 32) // its own thread, separate from audio
                .to(Box::new(video_counter))?;
            ctx.attach(source, v.index, branch)?;
        }
        if let Some(a) = audio {
            let branch = ctx
                .branch()
                .queue("audio-q", 32)
                .to(Box::new(audio_counter))?;
            ctx.attach(source, a.index, branch)?;
        }
        // Any other stream's pad is simply left unlinked.
        Ok(())
    })?;

    pipeline.run()?; // starts the source on a background thread, returns right away
    // Blocks until the demuxer hits EOS and both branch queues have
    // drained and joined (i.e. every `Bus` handle in the pipeline dropped).
    pipeline.bus().log_events();

    println!("video packets: {}", video_count.load(Ordering::Relaxed));
    println!("audio packets: {}", audio_count.load(Ordering::Relaxed));
    Ok(())
}
