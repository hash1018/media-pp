//! Demux -> Queue -> Pacer -> RtspMuxer: remuxes a file's packets (no
//! re-encoding) and publishes them at real playback speed to an
//! already-running RTSP server.
//!
//! Video and audio go out as two tracks of one RTSP session when the file
//! has both, which is what `RtspMuxer` is for; a file with no audio track
//! publishes video alone.
//!
//!     cargo run -p rtsp_serve -- path/to/video.mp4 rtsp://127.0.0.1:8554/stream
//!     ffplay rtsp://127.0.0.1:8554/stream                    # in another terminal

fn main() -> impl std::process::Termination {
    example::run()
}

mod example {
    use media_pp::ffmpeg::media;
    use media_pp::{
        Error,
        bus::BusEvent,
        element::ElementType,
        elements::{FileDemuxer, Pacer, RtspMuxer, RtspTransport},
        pipeline::Pipeline,
    };

    pub(super) fn run() -> media_pp::Result<()> {
        media_pp::init()?;
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let Some(path) = std::env::args().nth(1) else {
            eprintln!("usage: rtsp_serve <video.mp4> [rtsp://host:port/path]");
            std::process::exit(1);
        };
        let url = std::env::args()
            .nth(2)
            .unwrap_or_else(|| "rtsp://127.0.0.1:8554/stream".into());

        let (source, streams) = FileDemuxer::open("demux", &path)?;
        let video = streams
            .iter()
            .find(|s| s.kind == media::Type::Video)
            .ok_or_else(|| Error::Other("no video stream in file".into()))?;
        let video_index = video.index;
        let video_params = source
            .stream_parameters(video_index)
            .ok_or_else(|| Error::Other("stream disappeared".into()))?;
        let video_time_base = source
            .stream_time_base(video_index)
            .ok_or_else(|| Error::Other("stream disappeared".into()))?;

        // Optional on purpose: this example took video only before
        // `RtspMuxer` could carry two tracks, and a file with no audio must
        // still work exactly as it did.
        let audio_track = match streams.iter().find(|s| s.kind == media::Type::Audio) {
            Some(audio) => {
                let params = source
                    .stream_parameters(audio.index)
                    .ok_or_else(|| Error::Other("stream disappeared".into()))?;
                let time_base = source
                    .stream_time_base(audio.index)
                    .ok_or_else(|| Error::Other("stream disappeared".into()))?;
                Some((audio.index, params, time_base))
            }
            None => None,
        };
        // How many `Eos` events to wait for below — one per published track.
        let tracks = 1 + usize::from(audio_track.is_some());

        println!("publishing to {url} (the RTSP server must already be running) ...");

        let pipeline = Pipeline::new("rtsp-publish", source, |source, ctx| {
            // Every track must be registered before `open`, which is what
            // announces them all in one SDP.
            let mut muxer = RtspMuxer::create(&url, RtspTransport::Tcp)?;
            muxer.add_stream("video", video_params, video_time_base)?;
            let audio = match audio_track {
                Some((index, params, time_base)) => {
                    muxer.add_stream("audio", params, time_base)?;
                    Some((index, time_base))
                }
                None => None,
            };
            let mut sinks = muxer.open()?;
            // Popped in reverse registration order.
            let audio_sink = audio.map(|_| sinks.pop().expect("audio was registered second"));
            let video_sink = sinks.pop().expect("video was registered first");

            let branch = ctx
                .branch()
                .queue("video-packets", 32) // pacer sleeps on its own thread; let demux run ahead into this
                .pipe(Pacer::new("video-pacer", video_time_base)?)
                .to(video_sink)?;
            ctx.attach(source, video_index, branch)?;

            // Its own Pacer: the two tracks carry different time bases, and
            // each paces its own packets against the wall clock.
            if let (Some((index, time_base)), Some(audio_sink)) = (audio, audio_sink) {
                let branch = ctx
                    .branch()
                    .queue("audio-packets", 32)
                    .pipe(Pacer::new("audio-pacer", time_base)?)
                    .to(audio_sink)?;
                ctx.attach(source, index, branch)?;
            }
            Ok(())
        })?;

        println!("publishing {tracks} track(s) — connect a viewer to `{url}`");
        // `run()` starts publishing on a background thread and returns right
        // away — any failure shows up as a `BusEvent::Error` here instead of
        // through a returned `Result`.
        pipeline.run()?;

        // Errors no longer end the pipeline on their own (see `BusEvent`'s
        // docs) — watch for one here and `stop()`, or this would just keep
        // trying to publish into a broken server connection forever instead
        // of exiting.
        //
        // Natural completion waits for *every* track, not the first: the
        // session's trailer is only written once all of them report `Eos`,
        // so stopping on whichever finishes first would cut the other off.
        let mut finished = 0;
        for event in pipeline.bus().iter() {
            match &event {
                BusEvent::Eos {
                    name,
                    element_type: ElementType::RtspMuxer,
                } => {
                    finished += 1;
                    println!("[{name}] eos ({finished}/{tracks})");
                }
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
                // `BusEvent` is `#[non_exhaustive]`; this example only acts
                // on the events above.
                _ => {}
            }
            if finished == tracks || matches!(event, BusEvent::Error { .. }) {
                pipeline.stop();
            }
        }
        Ok(())
    }
}
