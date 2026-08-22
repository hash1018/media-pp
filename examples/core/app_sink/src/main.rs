use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use media_pp::ffmpeg::media;
use media_pp::{
    Error,
    buffer::MediaBuffer,
    bus::BusEvent,
    elements::{AppSink, FileDemuxer, SwDecoder},
    pipeline::Pipeline,
};

/// Demux -> SwDecoder -> AppSink: same shape as `decode`, but the
/// terminal sink is a plain closure instead of a bespoke `FrameCounter`
/// — proves `AppSink` lets a caller consume frames without writing a
/// dedicated `Element`/`Sink` impl at all (the GStreamer `appsink`
/// equivalent).
///
///     cargo run -p app_sink -- path/to/video.mp4
fn main() -> media_pp::Result<()> {
    media_pp::init()?;
    let _log_guard = media_pp::log::init(
        env!("CARGO_PKG_NAME"),
        "logs",
        media_pp::log::Level::Trace,
        7,
    )?;

    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: app_sink <video.mp4>");
        std::process::exit(1);
    };

    let (source, streams) = FileDemuxer::open("demux", &path)?;
    let video = streams
        .iter()
        .find(|s| s.kind == media::Type::Video)
        .ok_or_else(|| Error::Other("no video stream in file".into()))?;
    let params = source
        .stream_parameters(video.index)
        .ok_or_else(|| Error::Other("stream disappeared".into()))?;

    let count = Arc::new(AtomicUsize::new(0));
    let sink = {
        let count = count.clone();
        AppSink::new("counter", move |buf: MediaBuffer| {
            if matches!(buf, MediaBuffer::Video(_)) {
                count.fetch_add(1, Ordering::Relaxed);
            }
            Ok(())
        })
    };

    let pipeline = Pipeline::new("app-sink", source, |source, ctx| {
        let decoder = SwDecoder::new("decoder", params).expect("failed to open decoder");
        let branch = ctx
            .branch()
            .pipe(decoder) // same thread as the demux — cheap enough not to need a queue
            .to(Box::new(sink))?;
        ctx.attach(source, video.index, branch)?;
        Ok(())
    })?;

    pipeline.run()?;

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
            // `BusEvent` is `#[non_exhaustive]`; this example only acts
            // on the events above.
            _ => {}
        }
        if matches!(event, BusEvent::Eos { .. } | BusEvent::Error { .. }) {
            pipeline.stop();
        }
    }

    println!("decoded frames: {}", count.load(Ordering::Relaxed));
    Ok(())
}
