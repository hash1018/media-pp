use ffmpeg_next::media;
use media_pp::{
    Error,
    bus::BusEvent,
    elements::{FileDemuxer, Pacer, PortPolicy, PublishTransport, RtspServer, ViewerTransport},
    pipeline::Pipeline,
};

/// Demux -> Queue -> Pacer -> RtspServer: remuxes a file's video packets
/// (no re-encoding) and serves them, paced to real playback speed, as a
/// live RTSP stream. `RtspServer` spawns `mediamtx` itself (vendored by the
/// `rtsp-server` feature, see `lib/build.rs`) — nothing else needs to be
/// started first.
///
/// Uses `PortPolicy::Any` with a fixed range rather than assuming 8554 is
/// free, so more than one of these can run at once (each picks its own
/// free port within the range, tracked across instances in this process).
/// Watch the printed URL for the port it actually chose.
///
///     cargo run -p rtsp_serve -- path/to/video.mp4 rtsp://127.0.0.1/stream
///     ffplay rtsp://127.0.0.1:8554/stream                    # in another terminal
fn main() -> media_pp::Result<()> {
    media_pp::init()?;

    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "test-video/h265.mp4".into());
    let url = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "rtsp://127.0.0.1/stream".into());
    let port_policy = PortPolicy::Any { range: 8554..=8654 };

    let (source, streams) = FileDemuxer::open("demux", &path)?;
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

    println!("starting mediamtx and publishing to {url} ...");
    let mut actual_url = String::new();

    let pipeline = Pipeline::new("rtsp-serve", source, |source, ctx| {
        // Spawns mediamtx, waits for it to come up, then publishes to it —
        // by the time this returns, `server.url()` is live for viewers.
        let server = RtspServer::new(
            "rtsp",
            &url,
            port_policy,
            ViewerTransport::default(),
            PublishTransport::default(),
            params,
            time_base,
        )
        .expect("failed to start RTSP server");
        actual_url = server.url().to_string();
        let pacer = Pacer::new("pacer", time_base, ctx.clock.clone());
        let branch = ctx
            .branch()
            .queue("packets", 32) // pacer sleeps on its own thread; let demux run ahead into this
            .pipe(pacer)
            .to(Box::new(server))?;
        ctx.attach(source, video.index, branch)?;
        Ok(())
    })?;

    println!("ready — connect with `ffplay {actual_url}`");
    // `run()` starts publishing on a background thread and returns right
    // away — any failure shows up as a `BusEvent::Error` here instead of
    // through a returned `Result`.
    pipeline.run();

    // Errors no longer end the pipeline on their own (see `BusEvent`'s
    // docs) — watch for one here and `stop()`, or this would just keep
    // trying to publish into a broken `mediamtx`/connection forever
    // instead of exiting. Single video stream, so `Eos` calling `stop()`
    // is a harmless no-op too.
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
    Ok(())
}
