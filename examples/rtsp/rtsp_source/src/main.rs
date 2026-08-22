use std::{sync::atomic::Ordering, thread, time::Duration, time::Instant};

use ffmpeg_next::media;
use media_pp::{
    Error,
    bus::BusEvent,
    elements::{PacketCounter, RtspOptions, RtspSource},
    pipeline::Pipeline,
};

/// Connects to a live RTSP stream and counts video packets for a few
/// seconds, then stops — proves `RtspSource` actually connects,
/// negotiates a transport, and demuxes real packets from a live camera,
/// not just that it compiles.
///
///     cargo run -p rtsp_source -- rtsp://host:port/path
fn main() -> media_pp::Result<()> {
    media_pp::init()?;
    let _log_guard = media_pp::log::init(
        env!("CARGO_PKG_NAME"),
        "logs",
        media_pp::log::Level::Trace,
        7,
    )?;

    let url = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "rtsp://192.168.0.180:8554/v11_00?4".into());

    println!("connecting to {url} ...");
    let (source, streams) = RtspSource::open("rtsp-src", &url, RtspOptions::default())
        .map_err(|e| Error::Other(format!("failed to open RTSP source: {e}")))?;

    println!("stream count: {}", streams.len());
    for s in &streams {
        println!("  [{}] {:?}", s.index, s.kind);
    }

    let video = streams
        .iter()
        .find(|s| s.kind == media::Type::Video)
        .ok_or_else(|| Error::Other("no video stream advertised".into()))?;
    let index = video.index;

    let (counter, count) = PacketCounter::new("counter");

    let pipeline = Pipeline::new("rtsp-source", source, |source, ctx| {
        let branch = ctx.branch().queue("q", 32).to(Box::new(counter))?;
        ctx.attach(source, index, branch)?;
        Ok(())
    })?;

    pipeline.run()?;

    // A live camera never reaches Eos on its own — run for a fixed
    // window, then stop and report what came in. A real connection
    // failure ends things early too: `RtspSource` doesn't retry
    // internally (see its own doc comment), so a dropped connection
    // shows up here as a `BusEvent::Error`, not a silent hang.
    let deadline = Duration::from_secs(10);
    let start = Instant::now();
    loop {
        if let Some(event) = pipeline.bus().try_recv() {
            match &event {
                BusEvent::Eos { name, .. } => println!("[{name}] eos"),
                BusEvent::Error { name, error, .. } => eprintln!("[{name}] error: {error}"),
                BusEvent::Dropped { name, .. } => {
                    eprintln!("[{name}] dropped a buffer (queue full)")
                }
                // `BusEvent` is `#[non_exhaustive]`; this example only acts
                // on the events above.
                _ => {}
            }
            if matches!(event, BusEvent::Eos { .. } | BusEvent::Error { .. }) {
                break;
            }
        }
        if start.elapsed() >= deadline {
            println!("time limit reached, stopping");
            pipeline.stop();
            break;
        }
        thread::sleep(Duration::from_millis(50));
    }
    // Drains whatever's left (including the Eos `stop()` triggers) and
    // blocks until every `Bus` handle has dropped.
    pipeline.bus().log_events();

    println!("video packets received: {}", count.load(Ordering::Relaxed));
    Ok(())
}
