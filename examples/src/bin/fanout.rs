use std::sync::atomic::Ordering;

use ffmpeg_next::media;
use media_pp::{
    bus::BusEvent,
    element::Source,
    elements::{FileDemuxSource, PacketCounter},
    pipeline::{ChainBuilder, Pipeline},
};

/// Demonstrates fan-out: open a file, inspect its streams, then link
/// video and audio to separate branches (each behind its own `Queue`
/// thread boundary) — just two of the demuxer's src pads, no separate
/// "Tee" element involved.
///
///     cargo run -p media-pp-examples --bin fanout -- path/to/video_and_audio.mp4
fn main() -> media_pp::Result<()> {
    media_pp::init()?;

    let path = std::env::args().nth(1).expect("usage: fanout <path>");

    let (source, streams) = FileDemuxSource::open("demux", &path)?;

    println!("stream count: {}", streams.len());
    for s in &streams {
        println!("  [{}] {:?}", s.index, s.kind);
    }

    let video = streams.iter().find(|s| s.kind == media::Type::Video);
    let audio = streams.iter().find(|s| s.kind == media::Type::Audio);

    let (video_counter, video_count) = PacketCounter::new("video-counter");
    let (audio_counter, audio_count) = PacketCounter::new("audio-counter");

    let mut pipeline = Pipeline::new(source, |source, bus| {
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

    // Blocks until the demuxer hits EOS and both branch queues have
    // drained and joined.
    pipeline.run()?;
    for event in pipeline.bus().iter() {
        match event {
            BusEvent::Error { element, message } => eprintln!("[{element}] error: {message}"),
            BusEvent::Eos { element } => println!("[{element}] eos"),
            BusEvent::Dropped { element } => eprintln!("[{element}] dropped a buffer (queue full)"),
        }
    }

    println!("video packets: {}", video_count.load(Ordering::Relaxed));
    println!("audio packets: {}", audio_count.load(Ordering::Relaxed));
    Ok(())
}
