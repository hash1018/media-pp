fn main() -> impl std::process::Termination {
    example::run()
}

mod example {
    use ffmpeg_next::media;
    use media_pp::{
        Error,
        bus::BusEvent,
        elements::{FileDemuxer, Pacer, RtspSink, RtspTransport},
        pipeline::Pipeline,
    };

    /// Demux -> Queue -> Pacer -> RtspSink: remuxes a file's video packets
    /// (no re-encoding) and publishes them at real playback speed to an
    /// already-running RTSP server.
    ///
    ///     cargo run -p rtsp_serve -- path/to/video.mp4 rtsp://127.0.0.1:8554/stream
    ///     ffplay rtsp://127.0.0.1:8554/stream                    # in another terminal
    pub(super) fn run() -> media_pp::Result<()> {
        media_pp::init()?;
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let path = std::env::args()
            .nth(1)
            .unwrap_or_else(|| "test-video/h265.mp4".into());
        let url = std::env::args()
            .nth(2)
            .unwrap_or_else(|| "rtsp://127.0.0.1:8554/stream".into());

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

        println!("publishing to {url} (the RTSP server must already be running) ...");

        let pipeline = Pipeline::new("rtsp-publish", source, |source, ctx| {
            let sink = RtspSink::open("rtsp", url.clone(), RtspTransport::Tcp, params, time_base)?;
            let pacer = Pacer::new("pacer", time_base, ctx.clock.clone())?;
            let branch = ctx
                .branch()
                .queue("packets", 32) // pacer sleeps on its own thread; let demux run ahead into this
                .pipe(pacer)
                .to(Box::new(sink))?;
            ctx.attach(source, video.index, branch)?;
            Ok(())
        })?;

        println!("publishing — connect a viewer to `{url}`");
        // `run()` starts publishing on a background thread and returns right
        // away — any failure shows up as a `BusEvent::Error` here instead of
        // through a returned `Result`.
        pipeline.run();

        // Errors no longer end the pipeline on their own (see `BusEvent`'s
        // docs) — watch for one here and `stop()`, or this would just keep
        // trying to publish into a broken server connection forever
        // instead of exiting. Single video stream, so `Eos` calling `stop()`
        // is a harmless no-op too.
        for event in pipeline.bus().iter() {
            match &event {
                BusEvent::Eos { name, .. } => println!("[{name}] eos"),
                BusEvent::Error { name, error, .. } => eprintln!("[{name}] error: {error}"),
                BusEvent::Dropped { name, .. } => {
                    eprintln!("[{name}] dropped a buffer (queue full)")
                }
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
}
